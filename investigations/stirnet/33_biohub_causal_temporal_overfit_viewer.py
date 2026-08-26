from __future__ import annotations

r"""
Investigation 33 — Napari 3-D viewer for Investigation-31 causal temporal overfit.

Purpose
-------
Investigation 31 deliberately trained the temporal branch on the 20 manually
corrected BioHub frames. Its scientific checkpoint is ``best.pt`` rather than
necessarily the final optimizer step.

This viewer reconstructs the partition output of that checkpoint without
running the spatial CNN, dense geometry, watershed, or spatial RAG network.

It renders:

    Frozen spatial baseline
    Manual corrected target
    FULL temporal prediction
    EMPTY temporal prediction
    SHUFFLED temporal prediction
    CONTENTLESS temporal prediction

and diagnostics restricted to the only task supervised by Investigation 31:
splits INSIDE an existing frozen spatial component.

Diagnostic colors
-----------------
GREEN
    Manual split boundary that FULL recovered.

RED
    Manual split boundary that FULL missed.

MAGENTA
    FULL introduced an extra split boundary not present in the manual target.

YELLOW
    FULL split boundary absent with CONTENTLESS temporal content.

CYAN
    FULL split boundary absent with SHUFFLED temporal content.

The last two layers are causal visualizations: they identify places where the
FULL prediction depends on temporally localized content instead of merely on
"temporal support exists".

Default checkpoint
------------------
    runs/stirnet/evaluation/
        31_biohub_causal_temporal_overfit/<sample>/best.pt

Output cache
------------
    runs/stirnet/evaluation/
        33_biohub_causal_temporal_overfit_viewer/<sample>/best/
            manifest.json
            predictions/
                full/temporal_partition_t000.npy
                empty/...
                shuffled/...
                contentless/...

The predictions are cached because they are cheap graph-only inference but are
still unnecessary to recompute every time Napari is opened.

Typical usage
-------------
From repository root:

    python .\investigations\stirnet\33_biohub_causal_temporal_overfit_viewer.py

Inspect final.pt instead:

    python .\investigations\stirnet\33_biohub_causal_temporal_overfit_viewer.py ^
        --checkpoint final

Open a subset:

    python .\investigations\stirnet\33_biohub_causal_temporal_overfit_viewer.py ^
        --timepoints 0-9

Force regeneration:

    python .\investigations\stirnet\33_biohub_causal_temporal_overfit_viewer.py ^
        --rebuild-predictions

Viewer-only, requiring the cached Investigation-33 predictions:

    python .\investigations\stirnet\33_biohub_causal_temporal_overfit_viewer.py ^
        --viewer-only

Keyboard shortcuts
------------------
    1 : FULL prediction
    2 : Manual corrected target
    3 : Frozen spatial baseline
    4 : SHUFFLED prediction
    5 : CONTENTLESS prediction
    6 : EMPTY prediction
    D : toggle diagnostic boundary overlays
"""

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_NAME = "33_biohub_causal_temporal_overfit_viewer"
INV31_SCRIPT_NAME = "31_biohub_causal_temporal_overfit"
DEFAULT_SAMPLE_ID = "44b6_0113de3b"
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)
ABLATIONS = ("full", "empty", "shuffled", "contentless")


# =============================================================================
# REPOSITORY / IMPORT HELPERS
# =============================================================================


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (
            (candidate / "learned").is_dir()
            and (candidate / "investigations").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (
            (candidate / "learned").is_dir()
            and (candidate / "investigations").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate

    raise RuntimeError("Could not resolve the cell-tracking repository root.")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INV31 = load_module(
    ROOT / "investigations" / "stirnet" / "31_biohub_causal_temporal_overfit.py",
    "_stirnet_inv31_for_inv33",
)
INV30 = INV31.inv30
V13 = load_module(
    ROOT
    / "investigations"
    / "stirnet"
    / "data"
    / "13_biohub_full_volume_spatial_results_viewer.py",
    "_stirnet_inv13_for_inv33",
)


# =============================================================================
# SERIALIZATION / PATHS
# =============================================================================


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if torch.is_tensor(value):
        if value.ndim == 0:
            return jsonable(value.detach().cpu().item())
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return str(value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(tmp, np.asarray(array), allow_pickle=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def default_inv31_output(sample_id: str) -> Path:
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / INV31_SCRIPT_NAME
        / sample_id
    ).resolve()


def resolve_checkpoint(
    inv31_output: Path,
    checkpoint: str,
) -> tuple[Path, str]:
    token = checkpoint.strip()
    lowered = token.lower()

    if lowered in {"best", "final", "latest"}:
        path = inv31_output / f"{lowered}.pt"
        label = lowered
    else:
        path = resolve(token)
        label = path.stem

    if not path.is_file():
        raise FileNotFoundError(
            f"Investigation-31 checkpoint does not exist: {path}"
        )
    return path.resolve(), label


def default_output(sample_id: str, checkpoint_label: str) -> Path:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in checkpoint_label
    )
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / SCRIPT_NAME
        / sample_id
        / safe
    ).resolve()


def prediction_path(
    output: Path,
    ablation: str,
    frame: int,
) -> Path:
    return (
        output
        / "predictions"
        / ablation
        / f"temporal_partition_t{frame:03d}.npy"
    )


# =============================================================================
# CHECKPOINT RECONSTRUCTION
# =============================================================================


def hydrate_model_config(payload: dict[str, Any]):
    """Hydrate the checkpoint ModelConfig while retaining new default fields."""
    cfg = INV31.ModelConfig()
    saved = payload.get("model_config")
    if not isinstance(saved, dict):
        raise KeyError("Checkpoint does not contain model_config")

    for section_name, section_values in saved.items():
        if not hasattr(cfg, section_name):
            continue
        section = getattr(cfg, section_name)
        if not isinstance(section_values, dict):
            continue
        for key, value in section_values.items():
            if hasattr(section, key):
                setattr(section, key, value)

    cfg.validate()
    return cfg


def checkpoint_args(
    payload: dict[str, Any],
    *,
    inv31_output: Path,
) -> argparse.Namespace:
    saved = payload.get("training_args", {})
    if not isinstance(saved, dict):
        raise TypeError("checkpoint['training_args'] must be a dict")

    values = dict(saved)
    values.setdefault("sample_id", payload.get("sample_id", DEFAULT_SAMPLE_ID))
    values.setdefault("frame_count", 20)
    values.setdefault("inv12", None)
    values.setdefault("inv24", None)
    values.setdefault("inv25", None)
    values.setdefault("annotations", None)
    values.setdefault("zarr", None)
    values.setdefault("inv30_cache", None)
    values.setdefault("spacing_zyx_um", payload.get(
        "spacing_zyx_um", DEFAULT_SPACING_ZYX_UM
    ))
    values.setdefault("dref_um", payload.get("dref_um"))
    values.setdefault("temporal_radius", 2)
    values.setdefault("trackastra_model", "ctc")
    values.setdefault("trackastra_mode", "greedy")
    values.setdefault("trackastra_device", "cuda")
    values.setdefault("rebuild_trackastra", False)
    values.setdefault("rebuild_temporal", False)
    values.setdefault("rebuild_movies", False)
    values.setdefault("complete_candidate_graph", False)
    values.setdefault(
        "spatial_prior_logit",
        INV30.DEFAULT_SPATIAL_PRIOR_LOGIT,
    )

    # Never rebuild preprocessing from an inspection script.
    values["rebuild_trackastra"] = False
    values["rebuild_temporal"] = False
    values["rebuild_movies"] = False
    values["output_resolved"] = inv31_output

    return argparse.Namespace(**values)


def resolve_frozen_paths(
    payload: dict[str, Any],
    *,
    inv31_output: Path,
):
    args = checkpoint_args(payload, inv31_output=inv31_output)
    data_args = INV31.make_inv30_data_args(args)
    paths = INV30.make_paths(data_args)
    return args, paths


# =============================================================================
# PREDICTION MATERIALIZATION
# =============================================================================


def predictions_complete(
    output: Path,
    frames: list[int],
) -> bool:
    return all(
        prediction_path(output, ablation, frame).is_file()
        for ablation in ABLATIONS
        for frame in frames
    )


def manifest_matches(
    output: Path,
    *,
    checkpoint_path: Path,
    checkpoint_step: int,
    frames: list[int],
) -> bool:
    path = output / "manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    return (
        Path(manifest.get("checkpoint_path", "")).resolve()
        == checkpoint_path.resolve()
        and int(manifest.get("checkpoint_step", -1)) == int(checkpoint_step)
        and list(manifest.get("frames", [])) == list(frames)
        and predictions_complete(output, frames)
    )


@torch.inference_mode()
def materialize_predictions(
    *,
    checkpoint_path: Path,
    checkpoint_payload: dict[str, Any],
    inv31_output: Path,
    output: Path,
    frames: list[int],
    device: torch.device,
    rebuild: bool,
) -> dict[str, Any]:
    checkpoint_step = int(checkpoint_payload.get("step", -1))
    if (
        not rebuild
        and manifest_matches(
            output,
            checkpoint_path=checkpoint_path,
            checkpoint_step=checkpoint_step,
            frames=frames,
        )
    ):
        print("[predictions] reusing Investigation-33 cached partitions")
        return json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )

    cfg = hydrate_model_config(checkpoint_payload)
    train_args, paths = resolve_frozen_paths(
        checkpoint_payload,
        inv31_output=inv31_output,
    )

    frame_count = int(train_args.frame_count)
    invalid = [frame for frame in frames if not 0 <= frame < frame_count]
    if invalid:
        raise IndexError(
            f"Requested frames outside checkpoint movie: {invalid}; "
            f"frame_count={frame_count}"
        )

    spacing = tuple(
        float(value)
        for value in checkpoint_payload.get(
            "spacing_zyx_um",
            train_args.spacing_zyx_um,
        )
    )
    dref_um = float(checkpoint_payload["dref_um"])

    missing = []
    for frame in frames:
        for path in (
            paths.rag_state(frame),
            paths.supervoxels(frame),
            paths.base_instances(frame),
            paths.manual_instances(frame),
            paths.temporal_cache(frame),
        ):
            if not path.is_file():
                missing.append(path)
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:30])
        raise FileNotFoundError(
            "Investigation-33 requires the frozen Investigation-30/31 "
            "artifacts but some are missing:\n"
            + preview
        )

    print()
    print("=" * 118)
    print("INVESTIGATION 33 — MATERIALIZING BEST-CHECKPOINT ABLATIONS")
    print("=" * 118)
    print(f"checkpoint : {checkpoint_path}")
    print(f"step       : {checkpoint_step}")
    print(f"device     : {device}")
    print(f"frames     : {frames}")
    print("spatial CNN / geometry / watershed / RAG net: NOT RUN")
    print("=" * 118)

    temporal_model = INV30.TemporalOnlyModel(cfg)
    temporal_model.load_state_dict(
        checkpoint_payload["temporal_model_state_dict"],
        strict=True,
    )
    frozen_spatial = INV30.FrozenCachedSpatialRepresentation(cfg)

    temporal_model.to(device).eval()
    frozen_spatial.to(device).eval()

    # Loading all 20 tiny temporal graph payloads is substantially cheaper
    # than touching the dense spatial model.
    temporal_payloads = INV30.load_all_temporal_payloads(
        paths,
        frame_count,
    )

    per_frame: list[dict[str, Any]] = []

    for frame in frames:
        case = INV30.load_spatial_case(
            paths,
            frame,
            spacing,
            float(train_args.spatial_prior_logit),
        )
        temporal_payload = temporal_payloads[frame]
        row: dict[str, Any] = {"timepoint": int(frame)}

        for ablation in ABLATIONS:
            _, reasoning = INV31.forward_eval_ablation(
                temporal_model,
                frozen_spatial,
                case,
                temporal_payload,
                device=device,
                dref_um=dref_um,
                ablation=ablation,
            )

            # Reuse exactly the same partition/rasterization contract as
            # Investigation 31's evaluator.
            accumulator = INV30.EvalAccumulator()
            predicted = INV30.accumulate_frame_metrics(
                accumulator,
                case,
                reasoning,
            )
            labels = INV30.rasterize_prediction(
                case,
                predicted,
            ).astype(np.int32, copy=False)

            target = prediction_path(output, ablation, frame)
            atomic_npy(target, labels)
            row[f"{ablation}_instance_count"] = int(labels.max())

        per_frame.append(row)
        print(
            f"[t{frame:03d}] "
            + " | ".join(
                f"{name}={row[f'{name}_instance_count']}"
                for name in ABLATIONS
            ),
            flush=True,
        )

    manifest = {
        "status": "success",
        "experiment": SCRIPT_NAME,
        "sample_id": str(checkpoint_payload.get("sample_id", train_args.sample_id)),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "checkpoint_investigation": checkpoint_payload.get("investigation"),
        "frames": list(frames),
        "spacing_zyx_um": spacing,
        "dref_um": dref_um,
        "prediction_ablations": list(ABLATIONS),
        "spatial_model_ran": False,
        "per_frame": per_frame,
        "checkpoint_metrics": checkpoint_payload.get("metrics", {}),
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


# =============================================================================
# PARTITION DIAGNOSTICS
# =============================================================================


def _pad(array, pad_width):
    try:
        import dask.array as da
        if isinstance(array, da.Array):
            return da.pad(array, pad_width, mode="constant")
    except ImportError:
        pass
    return np.pad(array, pad_width, mode="constant")


def new_internal_split_boundary_4d(base, labels):
    """Boundary added *inside* a positive frozen spatial component.

    Both arrays are [T,Z,Y,X]. Time is never treated as a spatial axis.
    """
    boundary = None

    for axis in (1, 2, 3):
        lower_slice = [slice(None)] * 4
        upper_slice = [slice(None)] * 4
        lower_slice[axis] = slice(0, -1)
        upper_slice[axis] = slice(1, None)

        base_a = base[tuple(lower_slice)]
        base_b = base[tuple(upper_slice)]
        label_a = labels[tuple(lower_slice)]
        label_b = labels[tuple(upper_slice)]

        changed = (
            (base_a > 0)
            & (base_a == base_b)
            & (label_a > 0)
            & (label_b > 0)
            & (label_a != label_b)
        )

        low_padding = [(0, 0)] * 4
        high_padding = [(0, 0)] * 4
        low_padding[axis] = (0, 1)
        high_padding[axis] = (1, 0)

        axis_boundary = (
            _pad(changed, low_padding)
            | _pad(changed, high_padding)
        )
        boundary = (
            axis_boundary
            if boundary is None
            else (boundary | axis_boundary)
        )

    return boundary


def new_internal_split_boundary_3d(
    base: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    if base.shape != labels.shape or base.ndim != 3:
        raise ValueError(
            f"Expected aligned 3-D labels, got {base.shape} and {labels.shape}"
        )

    boundary = np.zeros(base.shape, dtype=bool)
    for axis in range(3):
        lower_slice = [slice(None)] * 3
        upper_slice = [slice(None)] * 3
        lower_slice[axis] = slice(0, -1)
        upper_slice[axis] = slice(1, None)

        b0 = base[tuple(lower_slice)]
        b1 = base[tuple(upper_slice)]
        x0 = labels[tuple(lower_slice)]
        x1 = labels[tuple(upper_slice)]

        changed = (
            (b0 > 0)
            & (b0 == b1)
            & (x0 > 0)
            & (x1 > 0)
            & (x0 != x1)
        )

        low_target = [slice(None)] * 3
        high_target = [slice(None)] * 3
        low_target[axis] = slice(0, -1)
        high_target[axis] = slice(1, None)
        boundary[tuple(low_target)] |= changed
        boundary[tuple(high_target)] |= changed

    return boundary


def partitions_equivalent(
    a: np.ndarray,
    b: np.ndarray,
) -> bool:
    """Compare two 1-D partitions modulo arbitrary positive label IDs."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.shape != b.shape:
        return False
    if a.size == 0:
        return True

    # Treat foreground coverage differences as non-equivalent.
    if not np.array_equal(a > 0, b > 0):
        return False

    valid = (a > 0) & (b > 0)
    a = a[valid]
    b = b[valid]
    if a.size == 0:
        return True

    # A partition is identical modulo IDs iff mapping is functional in both
    # directions.
    order = np.lexsort((b, a))
    pairs = np.stack([a[order], b[order]], axis=1)
    pairs = np.unique(pairs, axis=0)

    a_values, a_counts = np.unique(pairs[:, 0], return_counts=True)
    b_values, b_counts = np.unique(pairs[:, 1], return_counts=True)
    del a_values, b_values

    return bool(
        np.all(a_counts == 1)
        and np.all(b_counts == 1)
    )


def component_audit(
    base: np.ndarray,
    manual: np.ndarray,
    variants: dict[str, np.ndarray],
) -> dict[str, Any]:
    base_ids = np.unique(base)
    base_ids = base_ids[base_ids > 0]

    changed_components = 0
    exact = {name: 0 for name in variants}
    accidental = {name: 0 for name in variants}

    for base_id in base_ids.tolist():
        mask = base == base_id
        manual_values = manual[mask]
        manual_ids = np.unique(manual_values[manual_values > 0])
        manual_changed = len(manual_ids) > 1

        if manual_changed:
            changed_components += 1

        for name, labels in variants.items():
            values = labels[mask]
            ids = np.unique(values[values > 0])
            variant_changed = len(ids) > 1

            if manual_changed:
                if partitions_equivalent(manual_values, values):
                    exact[name] += 1
            elif variant_changed:
                accidental[name] += 1

    return {
        "manual_changed_components": int(changed_components),
        **{
            f"{name}_exact_changed_components": int(value)
            for name, value in exact.items()
        },
        **{
            f"{name}_accidental_split_components": int(value)
            for name, value in accidental.items()
        },
    }


def audit_frames(
    *,
    paths,
    output: Path,
    frames: list[int],
) -> list[dict[str, Any]]:
    rows = []

    for frame in frames:
        base = np.asarray(
            np.load(
                paths.base_instances(frame),
                mmap_mode="r",
                allow_pickle=False,
            )
        )
        manual = np.asarray(
            np.load(
                paths.manual_instances(frame),
                mmap_mode="r",
                allow_pickle=False,
            )
        )
        variants = {
            name: np.asarray(
                np.load(
                    prediction_path(output, name, frame),
                    mmap_mode="r",
                    allow_pickle=False,
                )
            )
            for name in ABLATIONS
        }

        manual_boundary = new_internal_split_boundary_3d(base, manual)
        full_boundary = new_internal_split_boundary_3d(
            base,
            variants["full"],
        )
        shuffled_boundary = new_internal_split_boundary_3d(
            base,
            variants["shuffled"],
        )
        contentless_boundary = new_internal_split_boundary_3d(
            base,
            variants["contentless"],
        )

        recovered = manual_boundary & full_boundary
        missed = manual_boundary & ~full_boundary
        extra = full_boundary & ~manual_boundary
        full_not_shuffled = full_boundary & ~shuffled_boundary
        full_not_contentless = full_boundary & ~contentless_boundary

        row = {
            "timepoint": int(frame),
            "manual_split_boundary_voxels": int(manual_boundary.sum()),
            "full_recovered_boundary_voxels": int(recovered.sum()),
            "full_missed_boundary_voxels": int(missed.sum()),
            "full_extra_boundary_voxels": int(extra.sum()),
            "full_not_shuffled_boundary_voxels": int(
                full_not_shuffled.sum()
            ),
            "full_not_contentless_boundary_voxels": int(
                full_not_contentless.sum()
            ),
        }
        row.update(component_audit(base, manual, variants))
        rows.append(row)

    atomic_json(output / "viewer_audit.json", {"per_frame": rows})
    return rows


# =============================================================================
# NAPARI
# =============================================================================


def _exclusive_label_view(
    label_layers: dict[str, Any],
    selected: str,
) -> None:
    for name, layer in label_layers.items():
        layer.visible = name == selected


def open_viewer(
    *,
    sample_id: str,
    checkpoint_path: Path,
    checkpoint_payload: dict[str, Any],
    inv31_output: Path,
    output: Path,
    frames: list[int],
    spacing: tuple[float, float, float],
    audit_rows: list[dict[str, Any]],
) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is not installed in this environment."
        ) from exc

    _, paths = resolve_frozen_paths(
        checkpoint_payload,
        inv31_output=inv31_output,
    )

    scale_4d = (1.0, *spacing)

    raw, raw_backend = V13.load_raw_time_series(
        paths.zarr,
        frames,
    )
    base, base_backend = V13.stack_npy(
        [paths.base_instances(frame) for frame in frames],
        name="frozen spatial baseline",
    )
    manual, manual_backend = V13.stack_npy(
        [paths.manual_instances(frame) for frame in frames],
        name="manual corrected target",
    )
    predictions = {}
    prediction_backends = {}
    for name in ABLATIONS:
        predictions[name], prediction_backends[name] = V13.stack_npy(
            [prediction_path(output, name, frame) for frame in frames],
            name=f"{name} temporal prediction",
        )

    manual_boundary = new_internal_split_boundary_4d(base, manual)
    full_boundary = new_internal_split_boundary_4d(
        base,
        predictions["full"],
    )
    shuffled_boundary = new_internal_split_boundary_4d(
        base,
        predictions["shuffled"],
    )
    contentless_boundary = new_internal_split_boundary_4d(
        base,
        predictions["contentless"],
    )

    recovered = manual_boundary & full_boundary
    missed = manual_boundary & ~full_boundary
    extra = full_boundary & ~manual_boundary
    full_not_contentless = full_boundary & ~contentless_boundary
    full_not_shuffled = full_boundary & ~shuffled_boundary

    step = int(checkpoint_payload.get("step", -1))
    viewer = napari.Viewer(
        title=(
            f"STIR-Net causal temporal overfit | {sample_id} | "
            f"{checkpoint_path.name} step={step}"
        )
    )

    raw_layer = viewer.add_image(
        raw,
        name="Raw BioHub",
        scale=scale_4d,
        colormap="gray",
        opacity=0.72,
        visible=True,
    )

    base_layer = viewer.add_labels(
        base,
        name="3 | Frozen spatial baseline",
        scale=scale_4d,
        opacity=0.72,
        visible=False,
    )
    manual_layer = viewer.add_labels(
        manual,
        name="2 | Manual corrected target",
        scale=scale_4d,
        opacity=0.72,
        visible=False,
    )
    full_layer = viewer.add_labels(
        predictions["full"],
        name="1 | FULL temporal prediction",
        scale=scale_4d,
        opacity=0.72,
        visible=True,
    )
    shuffled_layer = viewer.add_labels(
        predictions["shuffled"],
        name="4 | SHUFFLED temporal prediction",
        scale=scale_4d,
        opacity=0.72,
        visible=False,
    )
    contentless_layer = viewer.add_labels(
        predictions["contentless"],
        name="5 | CONTENTLESS temporal prediction",
        scale=scale_4d,
        opacity=0.72,
        visible=False,
    )
    empty_layer = viewer.add_labels(
        predictions["empty"],
        name="6 | EMPTY temporal prediction",
        scale=scale_4d,
        opacity=0.72,
        visible=False,
    )

    label_layers = {
        "full": full_layer,
        "manual": manual_layer,
        "base": base_layer,
        "shuffled": shuffled_layer,
        "contentless": contentless_layer,
        "empty": empty_layer,
    }

    diagnostic_layers = []

    diagnostic_layers.append(
        viewer.add_image(
            recovered,
            name="GREEN | FULL recovered manual split boundary",
            scale=scale_4d,
            colormap="green",
            contrast_limits=(0.0, 1.0),
            opacity=1.0,
            blending="additive",
            visible=True,
        )
    )
    diagnostic_layers.append(
        viewer.add_image(
            missed,
            name="RED | FULL missed manual split boundary",
            scale=scale_4d,
            colormap="red",
            contrast_limits=(0.0, 1.0),
            opacity=1.0,
            blending="additive",
            visible=True,
        )
    )
    diagnostic_layers.append(
        viewer.add_image(
            extra,
            name="MAGENTA | FULL extra split boundary",
            scale=scale_4d,
            colormap="magenta",
            contrast_limits=(0.0, 1.0),
            opacity=1.0,
            blending="additive",
            visible=True,
        )
    )
    diagnostic_layers.append(
        viewer.add_image(
            full_not_contentless,
            name="YELLOW | FULL boundary absent in CONTENTLESS",
            scale=scale_4d,
            colormap="yellow",
            contrast_limits=(0.0, 1.0),
            opacity=0.78,
            blending="additive",
            visible=False,
        )
    )
    diagnostic_layers.append(
        viewer.add_image(
            full_not_shuffled,
            name="CYAN | FULL boundary absent in SHUFFLED",
            scale=scale_4d,
            colormap="cyan",
            contrast_limits=(0.0, 1.0),
            opacity=0.78,
            blending="additive",
            visible=False,
        )
    )

    # The manual boundary itself is useful when all correctness overlays are
    # hidden.
    diagnostic_layers.append(
        viewer.add_image(
            manual_boundary,
            name="Manual required internal split boundary",
            scale=scale_4d,
            colormap="gray",
            contrast_limits=(0.0, 1.0),
            opacity=0.65,
            blending="additive",
            visible=False,
        )
    )

    audit_by_frame = {
        int(row["timepoint"]): row
        for row in audit_rows
    }

    # Prefer a frame with an actual remaining FULL error. If the overfit is
    # perfect, start at the frame with the most manually required split surface.
    strongest_frame = max(
        frames,
        key=lambda frame: (
            audit_by_frame[frame]["full_missed_boundary_voxels"]
            + audit_by_frame[frame]["full_extra_boundary_voxels"],
            audit_by_frame[frame]["manual_split_boundary_voxels"],
        ),
    )
    viewer.dims.set_current_step(0, frames.index(strongest_frame))
    viewer.dims.ndisplay = 3

    # A small dock makes time-slider inspection much easier.
    try:
        from qtpy.QtWidgets import QLabel

        status = QLabel()
        status.setWordWrap(True)

        def update_status(event=None):
            del event
            local_t = int(viewer.dims.current_step[0])
            local_t = max(0, min(local_t, len(frames) - 1))
            frame = frames[local_t]
            row = audit_by_frame[frame]
            changed = row["manual_changed_components"]
            recovered_components = row["full_exact_changed_components"]
            status.setText(
                "<b>Investigation 33 — frame diagnostics</b><br>"
                f"Napari T index: {local_t}<br>"
                f"BioHub frame: t{frame:03d}<br><br>"
                f"Manual changed components: {changed}<br>"
                f"FULL exact changed comps: "
                f"{recovered_components}/{changed}<br>"
                f"FULL accidental split comps: "
                f"{row['full_accidental_split_components']}<br><br>"
                f"Manual split-boundary voxels: "
                f"{row['manual_split_boundary_voxels']:,}<br>"
                f"Recovered boundary voxels: "
                f"{row['full_recovered_boundary_voxels']:,}<br>"
                f"Missed boundary voxels: "
                f"{row['full_missed_boundary_voxels']:,}<br>"
                f"Extra boundary voxels: "
                f"{row['full_extra_boundary_voxels']:,}<br><br>"
                f"FULL-only vs shuffled: "
                f"{row['full_not_shuffled_boundary_voxels']:,}<br>"
                f"FULL-only vs contentless: "
                f"{row['full_not_contentless_boundary_voxels']:,}<br><br>"
                "<b>Keys</b>: 1 FULL, 2 Manual, 3 Spatial, "
                "4 Shuffled, 5 Contentless, 6 Empty, D diagnostics"
            )

        viewer.window.add_dock_widget(
            status,
            name="Temporal overfit diagnostics",
            area="right",
        )
        viewer.dims.events.current_step.connect(update_status)
        update_status()
    except Exception as exc:
        print(
            f"[viewer] diagnostic dock unavailable: {exc}",
            flush=True,
        )

    @viewer.bind_key("1")
    def _show_full(viewer_):
        del viewer_
        _exclusive_label_view(label_layers, "full")

    @viewer.bind_key("2")
    def _show_manual(viewer_):
        del viewer_
        _exclusive_label_view(label_layers, "manual")

    @viewer.bind_key("3")
    def _show_base(viewer_):
        del viewer_
        _exclusive_label_view(label_layers, "base")

    @viewer.bind_key("4")
    def _show_shuffled(viewer_):
        del viewer_
        _exclusive_label_view(label_layers, "shuffled")

    @viewer.bind_key("5")
    def _show_contentless(viewer_):
        del viewer_
        _exclusive_label_view(label_layers, "contentless")

    @viewer.bind_key("6")
    def _show_empty(viewer_):
        del viewer_
        _exclusive_label_view(label_layers, "empty")

    @viewer.bind_key("d")
    def _toggle_diagnostics(viewer_):
        del viewer_
        make_visible = not any(
            bool(layer.visible)
            for layer in diagnostic_layers[:3]
        )
        for layer in diagnostic_layers[:3]:
            layer.visible = make_visible

    print()
    print("=" * 118)
    print("INVESTIGATION 33 — NAPARI 3-D")
    print("=" * 118)
    print(f"checkpoint : {checkpoint_path}")
    print(f"step       : {step}")
    print(f"frames     : {frames}")
    print(f"start      : t{strongest_frame:03d}")
    print(
        "backends   : "
        f"raw={raw_backend}, base={base_backend}, manual={manual_backend}, "
        + ", ".join(
            f"{name}={prediction_backends[name]}"
            for name in ABLATIONS
        )
    )
    print("-" * 118)
    print("1 = FULL temporal prediction")
    print("2 = Manual corrected target")
    print("3 = Frozen spatial baseline")
    print("4 = SHUFFLED temporal prediction")
    print("5 = CONTENTLESS temporal prediction")
    print("6 = EMPTY temporal prediction")
    print("D = toggle GREEN/RED/MAGENTA correctness overlays")
    print("-" * 118)
    print("GREEN   = manual split boundary recovered by FULL")
    print("RED     = manual split boundary missed by FULL")
    print("MAGENTA = extra FULL split boundary")
    print("YELLOW  = FULL split absent with CONTENTLESS temporal evidence")
    print("CYAN    = FULL split absent with SHUFFLED temporal evidence")
    print("=" * 118)

    # Keep a direct reference so static analyzers do not treat the raw layer as
    # accidental. It is intentionally always visible beneath the selected
    # labels layer.
    _ = raw_layer

    napari.run()


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Napari 3-D viewer for Investigation-31 causal temporal overfit."
        )
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument(
        "--inv31-output",
        type=Path,
        default=None,
        help=(
            "Investigation-31 sample output. Default: "
            "runs/stirnet/evaluation/31_biohub_causal_temporal_overfit/<sample>"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="'best', 'final', 'latest', or an explicit .pt path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Investigation-33 cache/output directory.",
    )
    parser.add_argument(
        "--timepoints",
        default="all",
        help="all, comma-list (0,5,10), or ranges (0-9,12,15-19).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device used only while materializing graph predictions.",
    )
    parser.add_argument(
        "--rebuild-predictions",
        action="store_true",
    )
    parser.add_argument(
        "--viewer-only",
        action="store_true",
        help="Do not run graph inference; require cached Investigation-33 predictions.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Materialize/audit predictions without opening Napari.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    inv31_output = (
        resolve(args.inv31_output)
        if args.inv31_output is not None
        else default_inv31_output(args.sample_id)
    )
    if not inv31_output.is_dir():
        raise FileNotFoundError(
            f"Investigation-31 output not found: {inv31_output}"
        )

    checkpoint_path, checkpoint_label = resolve_checkpoint(
        inv31_output,
        args.checkpoint,
    )
    payload = torch_load(checkpoint_path)

    checkpoint_sample = str(
        payload.get("sample_id", args.sample_id)
    )
    if checkpoint_sample != args.sample_id:
        raise ValueError(
            f"Checkpoint sample={checkpoint_sample}, "
            f"but --sample-id={args.sample_id}"
        )

    train_args, paths = resolve_frozen_paths(
        payload,
        inv31_output=inv31_output,
    )
    frame_count = int(train_args.frame_count)
    available = list(range(frame_count))
    frames = V13.parse_timepoints(
        args.timepoints,
        available,
    )

    output = (
        resolve(args.output)
        if args.output is not None
        else default_output(args.sample_id, checkpoint_label)
    )
    output.mkdir(parents=True, exist_ok=True)

    checkpoint_step = int(payload.get("step", -1))

    if args.viewer_only:
        if not manifest_matches(
            output,
            checkpoint_path=checkpoint_path,
            checkpoint_step=checkpoint_step,
            frames=frames,
        ):
            raise FileNotFoundError(
                "--viewer-only requested, but cached predictions do not match "
                f"checkpoint={checkpoint_path}, step={checkpoint_step}, "
                f"frames={frames}. Run once without --viewer-only."
            )
        manifest = json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )
    else:
        device = torch.device(
            args.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        manifest = materialize_predictions(
            checkpoint_path=checkpoint_path,
            checkpoint_payload=payload,
            inv31_output=inv31_output,
            output=output,
            frames=frames,
            device=device,
            rebuild=args.rebuild_predictions,
        )

    audit_rows = audit_frames(
        paths=paths,
        output=output,
        frames=frames,
    )

    print()
    print("=" * 118)
    print("INVESTIGATION 33 — OVERFIT VISUAL AUDIT")
    print("=" * 118)
    print(f"sample     : {args.sample_id}")
    print(f"checkpoint : {checkpoint_path}")
    print(f"step       : {checkpoint_step}")
    print(f"output     : {output}")
    print("-" * 118)

    total_changed = sum(
        row["manual_changed_components"]
        for row in audit_rows
    )
    total_exact = sum(
        row["full_exact_changed_components"]
        for row in audit_rows
    )
    total_accidental = sum(
        row["full_accidental_split_components"]
        for row in audit_rows
    )
    total_missed_boundary = sum(
        row["full_missed_boundary_voxels"]
        for row in audit_rows
    )
    total_extra_boundary = sum(
        row["full_extra_boundary_voxels"]
        for row in audit_rows
    )

    print(
        f"manual changed components : {total_changed}"
    )
    print(
        f"FULL exact recovery        : {total_exact}/{total_changed}"
        if total_changed
        else "FULL exact recovery        : no manual split components"
    )
    print(
        f"FULL accidental split comps: {total_accidental}"
    )
    print(
        f"FULL missed boundary voxels: {total_missed_boundary:,}"
    )
    print(
        f"FULL extra boundary voxels : {total_extra_boundary:,}"
    )

    metrics = manifest.get("checkpoint_metrics", {})
    verdict = metrics.get("verdict", {}) if isinstance(metrics, dict) else {}
    if verdict:
        print("-" * 118)
        print(
            "checkpoint verdict         : "
            f"override={verdict.get('temporal_override_mechanism')}, "
            f"causal={verdict.get('causal_temporal_dependence')}, "
            f"strict={verdict.get('strict_investigation31_pass')}"
        )
    print("=" * 118)

    if args.no_viewer:
        return

    spacing = tuple(
        float(value)
        for value in payload.get(
            "spacing_zyx_um",
            DEFAULT_SPACING_ZYX_UM,
        )
    )

    open_viewer(
        sample_id=args.sample_id,
        checkpoint_path=checkpoint_path,
        checkpoint_payload=payload,
        inv31_output=inv31_output,
        output=output,
        frames=frames,
        spacing=spacing,
        audit_rows=audit_rows,
    )


if __name__ == "__main__":
    main()
