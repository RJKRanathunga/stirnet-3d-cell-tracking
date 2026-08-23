from __future__ import annotations

"""
STIR-Net Investigation 08 — BioHub X/Y flip consistency diagnosis.

This is a controlled test for the systematic "only one side of each cell is
predicted" failure observed on BioHub.

The experiment keeps EVERYTHING fixed:
    same BioHub frame
    same production preprocessing
    same source labels
    same production 32x192x192 crop
    same checkpoint
    same spacing / dref

and changes only the spatial orientation of the FIVE scalar model channels.

Runs:
    A. original input
    B. X-flipped input -> prediction is flipped back to original coordinates
    C. Y-flipped input -> prediction is flipped back to original coordinates

Interpretation:
    * If A and X-flip-back differ strongly, the network/inference path is not
      reflection-consistent along X.
    * If the observed right-vs-left source-cell coverage bias reverses sign
      after X-flip-back, the failure is orientation-dependent.
    * If A and X-flip-back are nearly identical, the apparent half-cell failure
      is probably driven by image/source content rather than a simple X-axis
      directional preference.
    * Y provides the corresponding control for the other in-plane axis.

This experiment does NOT distinguish by itself between a learned directional
bias and a deterministic orientation-sensitive implementation detail. It tells
us whether the failure follows orientation. That is the decisive first control.

Default command:
    python investigations/stirnet/data/08_biohub_flip_consistency_diagnosis.py --napari

The script intentionally imports Investigation 06 so source preparation and
crop construction are identical to the BioHub inference that exposed the issue.
"""

import argparse
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage as ndi


DEFAULT_SAMPLE = "44b6_0113de3b"
DEFAULT_CHECKPOINT = (
    "runs/stirnet/training/01_nis3d_spatial_training/recovery/"
    "nis3d_zebrafish_spatial_v1/checkpoint_step_000600.pt"
)


# ======================================================================================
# Repository / Investigation 06
# ======================================================================================


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "learned").is_dir() and (candidate / "src").is_dir():
            return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "learned").is_dir() and (cwd / "src").is_dir():
        return cwd
    raise RuntimeError("Could not resolve cell-tracking repository root.")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_investigation06():
    path = (
        ROOT
        / "investigations"
        / "stirnet"
        / "data"
        / "06_biohub_spatial_checkpoint_inference.py"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Investigation 08 reuses the exact BioHub source path from 06, "
            f"but this file is missing:\n  {path}"
        )

    name = "_stirnet_investigation06_flip_control"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


I06 = _load_investigation06()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


# ======================================================================================
# Serialization
# ======================================================================================


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        x = float(value)
        return x if math.isfinite(x) else str(x)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


# ======================================================================================
# Model forward
# ======================================================================================


SCALAR_HEADS = (
    "foreground",
    "surface",
    "separator",
    "seed",
    "sdf",
)


def _cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(tensor).detach().float().cpu()


def _undo_spatial_flip(tensor: torch.Tensor, flip_axis: str | None) -> torch.Tensor:
    if flip_axis is None:
        return tensor
    if flip_axis == "x":
        return torch.flip(tensor, dims=(-1,))
    if flip_axis == "y":
        return torch.flip(tensor, dims=(-2,))
    raise ValueError(f"Unsupported flip axis: {flip_axis}")


def _run_variant(
    *,
    name: str,
    base_batch: dict,
    flip_axis: str | None,
    model,
    train_cfg,
    device: torch.device,
) -> dict[str, Any]:
    """
    Flip ALL five scalar spatial input channels together, run the same spatial
    forward, then transform scalar outputs back into original crop coordinates.
    """
    from learned.stirnet.training.trainer import (
        model_forward_from_batch,
        move_batch_to_device,
    )

    batch = {
        key: (value.clone() if torch.is_tensor(value) else value)
        for key, value in base_batch.items()
    }

    if flip_axis == "x":
        batch["spatial_inputs"] = torch.flip(
            batch["spatial_inputs"],
            dims=(-1,),
        )
    elif flip_axis == "y":
        batch["spatial_inputs"] = torch.flip(
            batch["spatial_inputs"],
            dims=(-2,),
        )
    elif flip_axis is not None:
        raise ValueError(f"Unknown flip axis: {flip_axis}")

    batch = move_batch_to_device(batch, device)
    amp_context, amp_name = I06.E04._amp_context(
        device,
        train_cfg.amp_dtype,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    with torch.inference_mode():
        with amp_context:
            output = model_forward_from_batch(
                model,
                batch,
                use_temporal=False,
                execution_stage="spatial",
                apply_existence_filter=False,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - started
    peak_vram_gib = (
        float(torch.cuda.max_memory_allocated(device) / 2**30)
        if device.type == "cuda"
        else 0.0
    )

    probabilities = output.geometry.probabilities()

    scalar = {
        "foreground": _undo_spatial_flip(
            _cpu_float(probabilities["foreground"][0, 0]),
            flip_axis,
        ),
        "surface": _undo_spatial_flip(
            _cpu_float(probabilities["surface"][0, 0]),
            flip_axis,
        ),
        "separator": _undo_spatial_flip(
            _cpu_float(probabilities["separator"][0, 0]),
            flip_axis,
        ),
        "seed": _undo_spatial_flip(
            _cpu_float(probabilities["seed"][0, 0]),
            flip_axis,
        ),
        "sdf": _undo_spatial_flip(
            _cpu_float(output.geometry.sdf[0, 0]),
            flip_axis,
        ),
    }

    # Labels are saved only for visual inspection. Label IDs are not expected
    # to correspond across transformed runs.
    supervoxels = _undo_spatial_flip(
        output.rag.supervoxel_labels[0].detach().cpu(),
        flip_axis,
    )
    partition = _undo_spatial_flip(
        output.spatial_partition.labels[0].detach().cpu(),
        flip_axis,
    )

    edge_prob = (
        output.rag.spatial_edge_logits
        .detach()
        .float()
        .sigmoid()
        .cpu()
    )

    result = {
        "name": name,
        "flip_axis": flip_axis,
        "amp_dtype": amp_name,
        "elapsed_seconds": float(elapsed),
        "peak_vram_gib": peak_vram_gib,
        "scalar": scalar,
        "supervoxels": supervoxels,
        "partition": partition,
        "edge_probability": edge_prob,
        "supervoxel_count": int(supervoxels.max()) if supervoxels.numel() else 0,
        "partition_count": int(partition.max()) if partition.numel() else 0,
        "edge_count": int(edge_prob.numel()),
    }

    del output, batch
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ======================================================================================
# Equivariance metrics
# ======================================================================================


def _dice_binary(a: torch.Tensor, b: torch.Tensor, threshold: float) -> float:
    aa = a >= threshold
    bb = b >= threshold
    intersection = int((aa & bb).sum())
    denom = int(aa.sum()) + int(bb.sum())
    return 1.0 if denom == 0 else 2.0 * intersection / denom


def _scalar_consistency(
    reference: torch.Tensor,
    transformed_back: torch.Tensor,
    *,
    threshold: float | None,
    source_mask: torch.Tensor,
) -> dict[str, float]:
    a = reference.float()
    b = transformed_back.float()
    diff = (a - b).abs()

    result = {
        "mae_all": float(diff.mean()),
        "rmse_all": float(torch.sqrt(((a - b) ** 2).mean())),
        "max_abs_error": float(diff.max()),
        "mae_inside_source": (
            float(diff[source_mask].mean())
            if bool(source_mask.any())
            else float("nan")
        ),
        "mae_outside_source": (
            float(diff[~source_mask].mean())
            if bool((~source_mask).any())
            else float("nan")
        ),
    }
    if threshold is not None:
        result["binary_dice"] = _dice_binary(a, b, threshold)
    return result


# ======================================================================================
# Source-cell side coverage
# ======================================================================================


def _boundary_ids(labels: np.ndarray) -> set[int]:
    values = np.concatenate(
        [
            labels[0].ravel(),
            labels[-1].ravel(),
            labels[:, 0, :].ravel(),
            labels[:, -1, :].ravel(),
            labels[:, :, 0].ravel(),
            labels[:, :, -1].ravel(),
        ]
    )
    return {int(v) for v in np.unique(values) if int(v) > 0}


def _complete_source_ids(
    labels: np.ndarray,
    *,
    min_voxels: int,
) -> list[int]:
    boundary = _boundary_ids(labels)
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    return [
        int(label_id)
        for label_id, count in zip(ids.tolist(), counts.tolist())
        if int(label_id) not in boundary and int(count) >= int(min_voxels)
    ]


def _cell_side_coverage(
    pred_foreground: torch.Tensor,
    source_labels: np.ndarray,
    *,
    foreground_threshold: float,
    min_voxels: int,
) -> dict[str, Any]:
    """
    Measure how much predicted foreground survives in each half of each COMPLETE
    source object. A positive x bias means right-half coverage > left-half
    coverage. A negative x bias means left-half coverage > right-half coverage.
    """
    pred = (
        torch.as_tensor(pred_foreground)
        .detach()
        .float()
        .cpu()
        .numpy()
        >= float(foreground_threshold)
    )

    ids = _complete_source_ids(
        source_labels,
        min_voxels=min_voxels,
    )

    x_left: list[float] = []
    x_right: list[float] = []
    y_low: list[float] = []
    y_high: list[float] = []
    overall: list[float] = []

    for label_id in ids:
        coords = np.argwhere(source_labels == int(label_id))
        if coords.shape[0] < min_voxels:
            continue

        values = pred[
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
        ]
        overall.append(float(values.mean()))

        x_mid = float(np.median(coords[:, 2]))
        left_sel = coords[:, 2] <= x_mid
        right_sel = coords[:, 2] > x_mid
        if left_sel.any() and right_sel.any():
            x_left.append(float(values[left_sel].mean()))
            x_right.append(float(values[right_sel].mean()))

        y_mid = float(np.median(coords[:, 1]))
        low_sel = coords[:, 1] <= y_mid
        high_sel = coords[:, 1] > y_mid
        if low_sel.any() and high_sel.any():
            y_low.append(float(values[low_sel].mean()))
            y_high.append(float(values[high_sel].mean()))

    def mean(values):
        return float(np.mean(values)) if values else float("nan")

    x_left_mean = mean(x_left)
    x_right_mean = mean(x_right)
    y_low_mean = mean(y_low)
    y_high_mean = mean(y_high)

    return {
        "complete_source_cell_count": len(ids),
        "mean_source_cell_foreground_coverage": mean(overall),
        "x_left_coverage": x_left_mean,
        "x_right_coverage": x_right_mean,
        "x_right_minus_left": (
            x_right_mean - x_left_mean
            if math.isfinite(x_right_mean) and math.isfinite(x_left_mean)
            else float("nan")
        ),
        "y_low_coverage": y_low_mean,
        "y_high_coverage": y_high_mean,
        "y_high_minus_low": (
            y_high_mean - y_low_mean
            if math.isfinite(y_high_mean) and math.isfinite(y_low_mean)
            else float("nan")
        ),
    }


def _foreground_behavior(
    pred_foreground: torch.Tensor,
    source_labels: np.ndarray,
    *,
    foreground_threshold: float,
    min_cell_voxels: int,
) -> dict[str, Any]:
    pred = (
        torch.as_tensor(pred_foreground)
        .detach()
        .float()
        .cpu()
        .numpy()
        >= float(foreground_threshold)
    )
    source = source_labels > 0

    source_recall = (
        float(pred[source].mean())
        if source.any()
        else float("nan")
    )
    bg_fp_rate = (
        float(pred[~source].mean())
        if (~source).any()
        else float("nan")
    )
    predicted_count = int(pred.sum())
    outside_pred = int((pred & ~source).sum())

    result = {
        "predicted_foreground_fraction": float(pred.mean()),
        "source_foreground_preservation_recall": source_recall,
        "background_false_positive_rate": bg_fp_rate,
        "fraction_of_predicted_fg_outside_source": (
            outside_pred / predicted_count
            if predicted_count
            else float("nan")
        ),
    }
    result.update(
        _cell_side_coverage(
            pred_foreground,
            source_labels,
            foreground_threshold=foreground_threshold,
            min_voxels=min_cell_voxels,
        )
    )
    return result


# ======================================================================================
# Heuristic interpretation
# ======================================================================================


def _sign_reversal(a: float, b: float, minimum_abs: float) -> bool:
    return (
        math.isfinite(a)
        and math.isfinite(b)
        and abs(a) >= minimum_abs
        and abs(b) >= minimum_abs
        and a * b < 0
    )


def _diagnostic_clues(
    *,
    original_behavior: dict[str, Any],
    x_behavior: dict[str, Any],
    y_behavior: dict[str, Any],
    x_consistency: dict[str, Any],
    y_consistency: dict[str, Any],
) -> list[str]:
    clues: list[str] = []

    x_dice = x_consistency["foreground"]["binary_dice"]
    y_dice = y_consistency["foreground"]["binary_dice"]

    original_x = float(original_behavior["x_right_minus_left"])
    x_back_x = float(x_behavior["x_right_minus_left"])
    original_y = float(original_behavior["y_high_minus_low"])
    y_back_y = float(y_behavior["y_high_minus_low"])

    if _sign_reversal(original_x, x_back_x, 0.08):
        clues.append(
            "X directional effect: source-cell left/right foreground bias reverses "
            "after X-flip-back."
        )
    if _sign_reversal(original_y, y_back_y, 0.08):
        clues.append(
            "Y directional effect: source-cell low/high foreground bias reverses "
            "after Y-flip-back."
        )

    if x_dice < 0.90:
        clues.append(
            f"Strong X reflection inconsistency: foreground flip-consistency Dice={x_dice:.3f}."
        )
    elif x_dice > 0.97:
        clues.append(
            f"X reflection is highly consistent: foreground flip-consistency Dice={x_dice:.3f}."
        )

    if y_dice < 0.90:
        clues.append(
            f"Strong Y reflection inconsistency: foreground flip-consistency Dice={y_dice:.3f}."
        )
    elif y_dice > 0.97:
        clues.append(
            f"Y reflection is highly consistent: foreground flip-consistency Dice={y_dice:.3f}."
        )

    if not clues:
        clues.append(
            "No conservative heuristic fired; inspect the Napari difference layers "
            "and the numeric side-coverage biases."
        )

    clues.append(
        "Flip inconsistency proves orientation sensitivity, but by itself does not "
        "separate learned directional bias from an orientation-sensitive implementation detail."
    )
    return clues


# ======================================================================================
# Output / Napari
# ======================================================================================


def _save_npz(
    path: Path,
    *,
    spatial_inputs: np.ndarray,
    source_labels: np.ndarray,
    crop,
    spacing,
    dref_um,
    original,
    xflip,
    yflip,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {
        "normalized_raw": spatial_inputs[0].astype(np.float16),
        "source_foreground_prior": spatial_inputs[1].astype(np.float16),
        "source_edt_prior": spatial_inputs[2].astype(np.float16),
        "source_boundary_prior": spatial_inputs[3].astype(np.float16),
        "source_marker_prior": spatial_inputs[4].astype(np.float16),
        "source_labels": source_labels.astype(np.int32),
        "crop_bounds_zyx": np.asarray(
            [[s.start, s.stop] for s in crop],
            dtype=np.int32,
        ),
        "spacing_zyx_um": np.asarray(spacing, dtype=np.float32),
        "dref_um": np.asarray([dref_um], dtype=np.float32),
    }

    for prefix, result in (
        ("original", original),
        ("xflip_back", xflip),
        ("yflip_back", yflip),
    ):
        for head, tensor in result["scalar"].items():
            payload[f"{prefix}_{head}"] = tensor.numpy().astype(np.float16)
        payload[f"{prefix}_supervoxels"] = (
            result["supervoxels"].numpy().astype(np.int32)
        )
        payload[f"{prefix}_partition"] = (
            result["partition"].numpy().astype(np.int32)
        )

    payload["absdiff_original_xflip_foreground"] = np.abs(
        payload["original_foreground"].astype(np.float32)
        - payload["xflip_back_foreground"].astype(np.float32)
    ).astype(np.float16)
    payload["absdiff_original_yflip_foreground"] = np.abs(
        payload["original_foreground"].astype(np.float32)
        - payload["yflip_back_foreground"].astype(np.float32)
    ).astype(np.float16)
    payload["absdiff_original_xflip_separator"] = np.abs(
        payload["original_separator"].astype(np.float32)
        - payload["xflip_back_separator"].astype(np.float32)
    ).astype(np.float16)
    payload["absdiff_original_yflip_separator"] = np.abs(
        payload["original_separator"].astype(np.float32)
        - payload["yflip_back_separator"].astype(np.float32)
    ).astype(np.float16)

    np.savez_compressed(path, **payload)


def _open_napari(npz_path: Path, title: str) -> None:
    import napari

    data = np.load(npz_path, allow_pickle=False)
    scale = tuple(float(v) for v in data["spacing_zyx_um"])

    viewer = napari.Viewer(title=title, ndisplay=3)

    viewer.add_image(
        data["normalized_raw"].astype(np.float32),
        name="normalized raw",
        scale=scale,
    )
    viewer.add_labels(
        data["source_labels"].astype(np.int32),
        name="source labels",
        scale=scale,
        visible=False,
    )

    for prefix, display in (
        ("original", "original"),
        ("xflip_back", "X-flip -> back"),
        ("yflip_back", "Y-flip -> back"),
    ):
        viewer.add_image(
            data[f"{prefix}_foreground"].astype(np.float32),
            name=f"{display} pred foreground",
            scale=scale,
            visible=(prefix == "original"),
            contrast_limits=(0, 1),
        )
        viewer.add_image(
            data[f"{prefix}_separator"].astype(np.float32),
            name=f"{display} pred separator",
            scale=scale,
            visible=False,
            contrast_limits=(0, 1),
        )
        viewer.add_image(
            data[f"{prefix}_seed"].astype(np.float32),
            name=f"{display} pred seed",
            scale=scale,
            visible=False,
            contrast_limits=(0, 1),
        )
        viewer.add_labels(
            data[f"{prefix}_supervoxels"].astype(np.int32),
            name=f"{display} supervoxels",
            scale=scale,
            visible=False,
        )
        viewer.add_labels(
            data[f"{prefix}_partition"].astype(np.int32),
            name=f"{display} spatial partition",
            scale=scale,
            visible=False,
        )

    viewer.add_image(
        data["absdiff_original_xflip_foreground"].astype(np.float32),
        name="|original - Xflipback| foreground",
        scale=scale,
        visible=False,
        contrast_limits=(0, 1),
    )
    viewer.add_image(
        data["absdiff_original_yflip_foreground"].astype(np.float32),
        name="|original - Yflipback| foreground",
        scale=scale,
        visible=False,
        contrast_limits=(0, 1),
    )
    viewer.add_image(
        data["absdiff_original_xflip_separator"].astype(np.float32),
        name="|original - Xflipback| separator",
        scale=scale,
        visible=False,
        contrast_limits=(0, 1),
    )
    viewer.add_image(
        data["absdiff_original_yflip_separator"].astype(np.float32),
        name="|original - Yflipback| separator",
        scale=scale,
        visible=False,
        contrast_limits=(0, 1),
    )

    print(
        "\n[Napari] Most useful comparison:\n"
        "  original pred foreground\n"
        "  X-flip -> back pred foreground\n"
        "  Y-flip -> back pred foreground\n"
        "  |original - Xflipback| foreground\n"
        "  |original - Yflipback| foreground\n"
        "\nIf the half-cell failure switches sides after X-flip-back, the effect "
        "is orientation-dependent.\n"
    )

    napari.run()
    data.close()


# ======================================================================================
# Main
# ======================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose STIR-Net BioHub X/Y reflection consistency."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--timepoint", type=int, default=0)
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=(1.625, 0.40625, 0.40625),
    )
    parser.add_argument(
        "--crop-mode",
        choices=("auto", "center", "explicit"),
        default="auto",
    )
    parser.add_argument("--z0", type=int, default=None)
    parser.add_argument("--y0", type=int, default=None)
    parser.add_argument("--x0", type=int, default=None)
    parser.add_argument(
        "--min-cell-voxels",
        type=int,
        default=32,
        help="Minimum source-object voxels for cell-side analysis.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--napari", action="store_true")
    args = parser.parse_args()

    sample_dir = I06.find_sample(args.sample_id, args.data_root)
    zarr_path = sample_dir / f"{args.sample_id}.zarr" / "0"

    import zarr

    volume = zarr.open_array(str(zarr_path), mode="r")
    if not 0 <= args.timepoint < int(volume.shape[0]):
        raise ValueError(
            f"timepoint {args.timepoint} is outside T={volume.shape[0]}"
        )

    raw = np.asarray(volume[args.timepoint])
    spacing = tuple(float(v) for v in args.spacing)

    checkpoint_path = _resolve(args.checkpoint)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    checkpoint, model, model_cfg, train_cfg = I06.E04._load_checkpoint_model(
        checkpoint_path,
        device,
    )
    step = int(checkpoint["global_step"])

    crop_shape = tuple(
        int(v)
        for v in train_cfg.curriculum.refinement_crop_shape_zyx
    )
    source_halo_um = float(
        getattr(
            train_cfg.curriculum,
            "refinement_crop_source_halo_um",
            4.0,
        )
    )
    foreground_threshold = float(
        model_cfg.partition.foreground_threshold
    )

    cache_path = (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / "06_biohub_spatial_checkpoint_inference"
        / "cache"
        / f"{args.sample_id}_t{args.timepoint:02d}.pt"
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = (
        _resolve(args.output_dir)
        if args.output_dir
        else (
            ROOT
            / "runs"
            / "stirnet"
            / "evaluation"
            / "08_biohub_flip_consistency_diagnosis"
            / f"{args.sample_id}_t{args.timepoint:02d}_step{step}_{stamp}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 112)
    print("STIR-Net Investigation 08 — BioHub X/Y flip consistency diagnosis")
    print("=" * 112)
    print(f"Sample               : {args.sample_id}")
    print(f"Timepoint            : {args.timepoint}")
    print(f"Frame shape          : {raw.shape}")
    print(f"Checkpoint           : {checkpoint_path}")
    print(f"Checkpoint step      : {step}")
    print(f"Device               : {device}")
    print(f"Production crop      : {crop_shape}")
    print(f"Foreground threshold : {foreground_threshold:.3f}")
    print(f"Output               : {output_dir}")
    print("=" * 112)

    source = I06.prepare_source(
        raw,
        spacing,
        args.sample_id,
        args.timepoint,
        cache_path,
    )
    current_full = source["instance_labels"][0].cpu().numpy()
    dref_um = float(source["dref_um"][0])
    source_meta = source["source_preprocessing_metadata"][0]

    if args.crop_mode == "auto":
        crop = I06.auto_crop(current_full, crop_shape)
    elif args.crop_mode == "center":
        crop = I06.center_crop(raw.shape, crop_shape)
    else:
        if None in (args.z0, args.y0, args.x0):
            raise ValueError(
                "--crop-mode explicit requires --z0 --y0 --x0"
            )
        crop = I06.explicit_crop(
            raw.shape,
            crop_shape,
            (args.z0, args.y0, args.x0),
        )

    base_batch, spatial_inputs, source_crop, source_context = (
        I06.build_crop_batch(
            source,
            crop,
            source_halo_um,
        )
    )

    print(
        f"[source] cache_hit={source_meta.get('source_cache_hit', False)} "
        f"dref={dref_um:.4f}um"
    )
    print(
        f"[crop] bounds={[(s.start, s.stop) for s in crop]} "
        f"source_instances={I06.positive_count(source_crop)} "
        f"source_fg={(source_crop > 0).mean():.4f}"
    )

    variants = {}
    for name, axis in (
        ("original", None),
        ("xflip_back", "x"),
        ("yflip_back", "y"),
    ):
        print(f"[model] {name} ...", flush=True)
        variants[name] = _run_variant(
            name=name,
            base_batch=base_batch,
            flip_axis=axis,
            model=model,
            train_cfg=train_cfg,
            device=device,
        )
        print(
            f"        time={variants[name]['elapsed_seconds']:.2f}s "
            f"SV={variants[name]['supervoxel_count']} "
            f"partition={variants[name]['partition_count']} "
            f"edges={variants[name]['edge_count']}"
        )

    source_mask = torch.from_numpy(source_crop > 0)

    thresholds = {
        "foreground": foreground_threshold,
        "surface": 0.50,
        "separator": 0.50,
        "seed": 0.50,
        "sdf": None,
    }

    consistency = {
        "xflip_back_vs_original": {},
        "yflip_back_vs_original": {},
    }

    for head in SCALAR_HEADS:
        consistency["xflip_back_vs_original"][head] = (
            _scalar_consistency(
                variants["original"]["scalar"][head],
                variants["xflip_back"]["scalar"][head],
                threshold=thresholds[head],
                source_mask=source_mask,
            )
        )
        consistency["yflip_back_vs_original"][head] = (
            _scalar_consistency(
                variants["original"]["scalar"][head],
                variants["yflip_back"]["scalar"][head],
                threshold=thresholds[head],
                source_mask=source_mask,
            )
        )

    behavior = {}
    for name in ("original", "xflip_back", "yflip_back"):
        behavior[name] = _foreground_behavior(
            variants[name]["scalar"]["foreground"],
            source_crop,
            foreground_threshold=foreground_threshold,
            min_cell_voxels=args.min_cell_voxels,
        )

    clues = _diagnostic_clues(
        original_behavior=behavior["original"],
        x_behavior=behavior["xflip_back"],
        y_behavior=behavior["yflip_back"],
        x_consistency=consistency["xflip_back_vs_original"],
        y_consistency=consistency["yflip_back_vs_original"],
    )

    summary = {
        "status": "success",
        "sample_id": args.sample_id,
        "timepoint": args.timepoint,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": step,
        "device": str(device),
        "spacing_zyx_um": spacing,
        "crop_bounds_zyx": [
            [int(s.start), int(s.stop)] for s in crop
        ],
        "source_context_bounds_zyx": [
            [int(s.start), int(s.stop)] for s in source_context
        ],
        "dref_um": dref_um,
        "foreground_threshold": foreground_threshold,
        "source_crop_instance_count": I06.positive_count(source_crop),
        "source_crop_foreground_fraction": float(
            (source_crop > 0).mean()
        ),
        "foreground_behavior": behavior,
        "flip_consistency": consistency,
        "variant_runtime": {
            name: {
                "elapsed_seconds": result["elapsed_seconds"],
                "peak_vram_gib": result["peak_vram_gib"],
                "supervoxel_count": result["supervoxel_count"],
                "partition_count": result["partition_count"],
                "edge_count": result["edge_count"],
            }
            for name, result in variants.items()
        },
        "diagnostic_clues": clues,
        "interpretation_note": (
            "Reflection inconsistency establishes orientation sensitivity. "
            "It does not alone prove whether that sensitivity is learned from "
            "single-volume training or caused by an orientation-sensitive "
            "implementation detail."
        ),
    }

    npz_path = output_dir / "flip_consistency.npz"
    _save_npz(
        npz_path,
        spatial_inputs=spatial_inputs,
        source_labels=source_crop,
        crop=crop,
        spacing=spacing,
        dref_um=dref_um,
        original=variants["original"],
        xflip=variants["xflip_back"],
        yflip=variants["yflip_back"],
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    x_fg = consistency["xflip_back_vs_original"]["foreground"]
    y_fg = consistency["yflip_back_vs_original"]["foreground"]

    print("\n" + "=" * 112)
    print("FLIP CONSISTENCY RESULT")
    print("=" * 112)
    print(
        f"Original source-cell FG preservation : "
        f"{behavior['original']['mean_source_cell_foreground_coverage']:.3f}"
    )
    print(
        f"Original X right-left bias           : "
        f"{behavior['original']['x_right_minus_left']:+.3f}"
    )
    print(
        f"Xflip-back X right-left bias         : "
        f"{behavior['xflip_back']['x_right_minus_left']:+.3f}"
    )
    print(
        f"Original Y high-low bias             : "
        f"{behavior['original']['y_high_minus_low']:+.3f}"
    )
    print(
        f"Yflip-back Y high-low bias           : "
        f"{behavior['yflip_back']['y_high_minus_low']:+.3f}"
    )
    print("-" * 112)
    print(
        f"X foreground consistency Dice        : "
        f"{x_fg['binary_dice']:.3f}"
    )
    print(
        f"X foreground consistency MAE         : "
        f"{x_fg['mae_all']:.4f}"
    )
    print(
        f"Y foreground consistency Dice        : "
        f"{y_fg['binary_dice']:.3f}"
    )
    print(
        f"Y foreground consistency MAE         : "
        f"{y_fg['mae_all']:.4f}"
    )
    print(
        f"X separator consistency Dice         : "
        f"{consistency['xflip_back_vs_original']['separator']['binary_dice']:.3f}"
    )
    print(
        f"Y separator consistency Dice         : "
        f"{consistency['yflip_back_vs_original']['separator']['binary_dice']:.3f}"
    )
    print("-" * 112)
    print("Diagnostic clues:")
    for clue in clues:
        print(f"  - {clue}")
    print(f"NPZ     : {npz_path}")
    print(f"Summary : {summary_path}")
    print("=" * 112)

    if args.napari:
        _open_napari(
            npz_path,
            title=(
                f"STIR-Net flip diagnosis — {args.sample_id} "
                f"t={args.timepoint} step={step}"
            ),
        )


if __name__ == "__main__":
    main()
