# STIRNET_GENERIC_BIOHUB_INFERENCE_V2
from __future__ import annotations

"""
Investigation 12 — full-volume STIR-Net spatial inference on the Stage-6 BioHub cache.

Purpose
-------
Run the known-good spatial checkpoint over every cached BioHub timepoint without
re-running the original Stage-1..6 preprocessing pipeline.

The script consumes the Stage-6 artifacts directly:

    data/sample/processed/stage_6_processed_dataset/<sample_id>/
        preprocessing/t000.npy
        masking/t000.npy
        segmentation/t000.npy
        ...
        preprocessing/t019.npy
        masking/t019.npy
        segmentation/t019.npy

For every frame it:
1. loads the Stage-6 preprocessed intensity volume and source segmentation;
2. builds the five STIR-Net spatial input channels from the full source volume;
3. runs the repository's production tiled spatial inference path;
4. saves dense geometry, watershed, partition, RAG and instance diagnostics;
5. writes a per-frame success record before moving to the next timepoint.

The output is intentionally diagnostic-rich.  RAG logits/edges are persisted so
merge thresholds and partition failures can be studied later without re-running
the CNN.

The run is resumable.  A frame with `_SUCCESS.json` is skipped unless
`--overwrite` is supplied.

Default milestone checkpoint directory:
    runs/stirnet/milestones/drosophila_12_spatial_v1

Typical command
---------------
From the repository root:

    python investigations/stirnet/data/12_biohub_full_volume_spatial_inference.py

Explicit checkpoint directory:

    python investigations/stirnet/data/12_biohub_full_volume_spatial_inference.py ^
        --checkpoint-dir runs/stirnet/milestones/drosophila_12_spatial_v1

A subset can be re-run during debugging:

    python investigations/stirnet/data/12_biohub_full_volume_spatial_inference.py ^
        --timepoints 3,7,12 --overwrite

Notes
-----
* This is spatial-only inference.  No temporal reasoning or refinement is run.
* Stage-6 `preprocessing` is already normalized/corrected to [0,1] by the
  canonical BioHub preprocessing pipeline, so it is used directly as channel 0.
* Source priors are generated from the Stage-6 instance segmentation over the
  COMPLETE volume before tiled CNN inference.  Crop/tile placement therefore
  does not redefine source EDT/marker geometry.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


SCRIPT_NAME = "12_biohub_full_volume_spatial_inference"
DEFAULT_SAMPLE = "44b6_0113de3b"
DEFAULT_EXPECTED_FRAMES = 20
DEFAULT_CHECKPOINT_DIR = "runs/stirnet/milestones/drosophila_12_spatial_v1"
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)

# These defaults match ModelConfig.InferenceConfig in the current repository.
DEFAULT_TILE_SHAPE_ZYX = (32, 128, 128)
DEFAULT_TILE_OVERLAP_ZYX = (8, 32, 32)
DEFAULT_TILE_HALO_ZYX = (4, 16, 16)
DEFAULT_TILE_BATCH_SIZE = 1


# ======================================================================================
# Repository / JSON utilities
# ======================================================================================


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (
            (candidate / "learned").is_dir()
            and (candidate / "src").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    cwd = Path.cwd().resolve()
    if (
        (cwd / "learned").is_dir()
        and (cwd / "src").is_dir()
        and (cwd / "pyproject.toml").is_file()
    ):
        return cwd
    raise RuntimeError("Could not find the cell-tracking repository root.")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return jsonable(tensor.item())
        return tensor.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(payload), sort_keys=True))
        handle.write("\n")
        handle.flush()


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# ======================================================================================
# Checkpoint loading
# ======================================================================================


def load_eval04():
    """Reuse the repository's checkpoint hydration compatibility layer."""
    path = ROOT / "investigations/stirnet/data/04_nis3d_geometry_checkpoint_eval.py"
    spec = importlib.util.spec_from_file_location("_eval04_for_investigation12", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import checkpoint helper from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_eval04_for_investigation12"] = module
    spec.loader.exec_module(module)
    return module


E04 = load_eval04()


def _load_raw_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint_model_for_inference(
    checkpoint_path: Path,
    device: torch.device,
):
    """Load experiment checkpoints without weakening strict model checks.

    Normal checkpoints are passed directly to Eval-04.

    Some investigation checkpoints intentionally store experiment metadata under
    ``training_config`` rather than a serialized TrainingConfig dataclass.
    Eval-04 correctly rejects those unknown fields. Only for that specific
    compatibility error, this function makes a temporary checkpoint copy with
    ``training_config`` omitted and retries. ModelConfig hydration and strict
    model-state loading remain unchanged. The original checkpoint is untouched.
    """

    try:
        checkpoint, model, model_cfg, train_cfg = E04._load_checkpoint_model(
            checkpoint_path,
            device,
        )
        return checkpoint, model, model_cfg, train_cfg, False
    except KeyError as exc:
        message = str(exc)
        if (
            "training_config." not in message
            or "unknown field" not in message
        ):
            raise

    payload = _load_raw_checkpoint(checkpoint_path)
    training_config = payload.get("training_config")
    if not isinstance(training_config, dict):
        raise RuntimeError(
            "Eval-04 rejected checkpoint configuration, but the checkpoint "
            "does not contain a dictionary training_config to strip."
        )

    stripped_fields = sorted(str(key) for key in training_config)
    compatibility_payload = dict(payload)
    compatibility_payload.pop("training_config", None)

    print(
        "[checkpoint] experiment-local training_config is not a TrainingConfig; "
        "retrying inference with a temporary compatibility copy.",
        flush=True,
    )
    print(
        "[checkpoint] stripped from temporary copy only: "
        + ", ".join(stripped_fields),
        flush=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="stirnet_inference_checkpoint_"
    ) as temporary_directory:
        temporary_path = (
            Path(temporary_directory)
            / checkpoint_path.name
        )
        torch.save(compatibility_payload, temporary_path)

        checkpoint, model, model_cfg, train_cfg = E04._load_checkpoint_model(
            temporary_path,
            device,
        )

    print(
        "[checkpoint] strict model load succeeded; original checkpoint "
        "remains unchanged.",
        flush=True,
    )
    return checkpoint, model, model_cfg, train_cfg, True


def validate_run_label(value: str | None) -> str | None:
    if value is None:
        return None
    label = value.strip()
    if not label:
        raise ValueError("--run-label cannot be empty")
    if label in {".", ".."}:
        raise ValueError("--run-label cannot be '.' or '..'")
    if any(
        not (character.isalnum() or character in "-_.")
        for character in label
    ):
        raise ValueError(
            "--run-label may contain only letters, digits, '-', '_' and '.'"
        )
    return label


def checkpoint_step_from_name(path: Path) -> int:
    name = path.name
    prefix = "checkpoint_step_"
    suffix = ".pt"
    if name.startswith(prefix) and name.endswith(suffix):
        token = name[len(prefix) : -len(suffix)]
        if token.isdigit():
            return int(token)
    return -1


def latest_checkpoint(directory: Path) -> Path:
    if not directory.is_dir():
        raise NotADirectoryError(f"Checkpoint directory does not exist: {directory}")

    direct = [
        path
        for path in directory.glob("checkpoint_step_*.pt")
        if path.is_file() and checkpoint_step_from_name(path) >= 0
    ]
    candidates = direct
    search_mode = "direct"
    if not candidates:
        candidates = [
            path
            for path in directory.rglob("checkpoint_step_*.pt")
            if path.is_file() and checkpoint_step_from_name(path) >= 0
        ]
        search_mode = "recursive"

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint_step_XXXXXX.pt found below {directory}"
        )

    selected = max(
        candidates,
        key=lambda path: (checkpoint_step_from_name(path), str(path)),
    )
    print(
        f"[checkpoint] {search_mode} search: {len(candidates)} candidate(s), "
        f"selected {selected.name}",
        flush=True,
    )
    return selected.resolve()


def resolve_checkpoint(
    checkpoint: str | None,
    checkpoint_dir: str | None,
) -> Path:
    if checkpoint is not None and checkpoint_dir is not None:
        raise ValueError("--checkpoint and --checkpoint-dir are mutually exclusive")

    if checkpoint_dir is not None:
        return latest_checkpoint(resolve(checkpoint_dir))

    candidate = resolve(checkpoint or DEFAULT_CHECKPOINT_DIR)
    if candidate.is_dir():
        return latest_checkpoint(candidate)
    if not candidate.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {candidate}")
    return candidate


# ======================================================================================
# Stage-6 input discovery
# ======================================================================================


def resolve_stage6_root(sample_id: str, override: str | None) -> Path:
    from src.io import PipelinePaths

    if override is None:
        root = PipelinePaths.discover(ROOT).processed_dataset(sample_id)
    else:
        supplied = resolve(override)
        if (supplied / "preprocessing").is_dir():
            root = supplied
        elif (supplied / sample_id / "preprocessing").is_dir():
            root = supplied / sample_id
        else:
            root = supplied

    required = ("preprocessing", "masking", "segmentation")
    missing = [name for name in required if not (root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Stage-6 root is incomplete: {root}. Missing: {', '.join(missing)}"
        )
    return root.resolve()


def discover_stage6_timepoints(
    stage6_root: Path,
    *,
    expected_frames: int | None,
) -> list[int]:
    series_names = ("preprocessing", "masking", "segmentation")
    sets: dict[str, set[int]] = {}
    for name in series_names:
        rows: set[int] = set()
        for path in (stage6_root / name).glob("t*.npy"):
            stem = path.stem
            if len(stem) >= 2 and stem[0] == "t" and stem[1:].isdigit():
                rows.add(int(stem[1:]))
        sets[name] = rows

    if not all(sets.values()):
        raise FileNotFoundError(
            "One or more Stage-6 series are empty: "
            + ", ".join(f"{name}={len(rows)}" for name, rows in sets.items())
        )

    reference = sets["preprocessing"]
    for name, rows in sets.items():
        if rows != reference:
            raise ValueError(
                "Stage-6 timepoint sets do not align: "
                + ", ".join(
                    f"{series}={sorted(values)}" for series, values in sets.items()
                )
            )

    frames = sorted(reference)
    if expected_frames is not None and len(frames) != expected_frames:
        raise ValueError(
            f"Expected {expected_frames} Stage-6 frames, found {len(frames)}: {frames}"
        )
    return frames


def parse_timepoints(text: str, available: list[int]) -> list[int]:
    token = text.strip().lower()
    if token in {"all", "*"}:
        return list(available)

    selected: set[int] = set()
    for item in token.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"Invalid timepoint range: {item}")
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(item))

    missing = sorted(selected.difference(available))
    if missing:
        raise ValueError(
            f"Requested timepoints are absent from Stage 6: {missing}; "
            f"available={available}"
        )
    if not selected:
        raise ValueError("No timepoints selected")
    return sorted(selected)


def frame_paths(stage6_root: Path, frame: int) -> dict[str, Path]:
    stem = f"t{frame:03d}.npy"
    return {
        "preprocessing": stage6_root / "preprocessing" / stem,
        "masking": stage6_root / "masking" / stem,
        "segmentation": stage6_root / "segmentation" / stem,
    }


# ======================================================================================
# STIR-Net input construction / inference
# ======================================================================================


def count_positive_labels(labels: np.ndarray) -> int:
    ids = np.unique(np.asarray(labels))
    return int(np.count_nonzero(ids > 0))


def build_stage6_spatial_input(
    preprocessed: np.ndarray,
    source_labels: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
) -> tuple[np.ndarray, float]:
    """Build full-volume channels before any CNN tiling."""
    from learned.stirnet.data.sample_builder import build_spatial_channels
    from learned.stirnet.data.targets import estimate_model_dref_um

    raw = np.asarray(preprocessed, dtype=np.float32)
    labels = np.asarray(source_labels)

    if raw.shape != labels.shape or raw.ndim != 3:
        raise ValueError(
            f"Stage-6 preprocessing/segmentation must align in 3-D; "
            f"got {raw.shape} and {labels.shape}"
        )

    # Canonical Stage-6 preprocessing already ends in [0,1]. Refuse obviously
    # incompatible data rather than silently applying a second normalization.
    finite = np.isfinite(raw)
    if not finite.all():
        raise ValueError("Stage-6 preprocessing contains NaN/Inf values")
    raw_min = float(raw.min()) if raw.size else 0.0
    raw_max = float(raw.max()) if raw.size else 0.0
    if raw_min < -1e-5 or raw_max > 1.0001:
        raise ValueError(
            "Stage-6 preprocessing is expected to be normalized to [0,1], "
            f"but observed range [{raw_min:.6g}, {raw_max:.6g}]"
        )

    dref_um = float(
        estimate_model_dref_um(
            labels,
            tuple(float(v) for v in spacing_zyx_um),
        )
    )
    spatial = build_spatial_channels(
        raw,
        labels,
        spacing_zyx_um,
        dref_um,
        derive_marker=True,
    )
    return np.ascontiguousarray(spatial, dtype=np.float32), dref_um


def parse_zyx(text: str, *, name: str) -> tuple[int, int, int]:
    rows = tuple(int(token.strip()) for token in text.split(","))
    if len(rows) != 3 or any(value < 0 for value in rows):
        raise ValueError(f"{name} must be three non-negative Z,Y,X integers")
    return rows


def build_inference_config(
    model_cfg,
    *,
    tile_shape_zyx: tuple[int, int, int],
    tile_overlap_zyx: tuple[int, int, int],
    tile_halo_zyx: tuple[int, int, int],
    tile_batch_size: int,
):
    if any(value <= 0 for value in tile_shape_zyx):
        raise ValueError("Tile shape values must be positive")
    if any(
        overlap >= tile
        for overlap, tile in zip(tile_overlap_zyx, tile_shape_zyx)
    ):
        raise ValueError("Every tile overlap must be smaller than its tile size")
    if any(
        2 * halo >= tile
        for halo, tile in zip(tile_halo_zyx, tile_shape_zyx)
    ):
        raise ValueError("Every tile halo must be smaller than half its tile size")
    if tile_batch_size < 1:
        raise ValueError("--tile-batch-size must be positive")

    return replace(
        model_cfg.inference,
        mode="tiled",
        tiled_dense_enabled=True,
        tile_shape_zyx=tile_shape_zyx,
        tile_overlap_zyx=tile_overlap_zyx,
        tile_halo_zyx=tile_halo_zyx,
        tile_batch_size=int(tile_batch_size),
    )


def amp_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext(), "fp32"
    if torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16), "bf16"
    return torch.autocast("cuda", dtype=torch.float16), "fp16"


def tensor_numpy(value, dtype=None) -> np.ndarray:
    tensor = torch.as_tensor(value).detach()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    array = tensor.cpu().numpy()
    return array.astype(dtype, copy=False) if dtype is not None else array


def run_tiled_spatial(
    model,
    spatial: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
    dref_um: float,
    *,
    device: torch.device,
    inference_cfg,
):
    from learned.stirnet.inference.tiled_dense import tiled_spatial_inference

    spatial_tensor = torch.from_numpy(spatial)[None].to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    )
    spacing_tensor = torch.tensor(
        [spacing_zyx_um],
        device=device,
        dtype=torch.float32,
    )
    dref_tensor = torch.tensor([dref_um], device=device, dtype=torch.float32)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    amp, amp_name = amp_context(device)
    started = time.perf_counter()
    with torch.inference_mode(), amp:
        result = tiled_spatial_inference(
            model,
            spatial_tensor,
            spacing_tensor,
            dref_tensor,
            config=inference_cfg,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    seconds = time.perf_counter() - started
    peak_gib = (
        float(torch.cuda.max_memory_allocated(device) / 2**30)
        if device.type == "cuda"
        else 0.0
    )
    return result, spatial_tensor, amp_name, seconds, peak_gib


# ======================================================================================
# Persistent diagnostics
# ======================================================================================


def save_frame_outputs(
    frame_dir: Path,
    *,
    spatial: np.ndarray,
    source_mask: np.ndarray,
    source_labels: np.ndarray,
    result,
    save_source_priors: bool,
) -> dict[str, Any]:
    """Persist enough state to debug dense geometry and RAG without rerunning CNN."""
    geometry = result.dense.geometry
    probabilities = geometry.probabilities()

    geometry_dir = frame_dir / "geometry"
    partition_dir = frame_dir / "partition"
    rag_dir = frame_dir / "rag"
    source_dir = frame_dir / "source_priors"

    # Dense learned geometry. Float16 roughly halves long-term diagnostic storage
    # while retaining far more precision than the visualization/debug tasks need.
    atomic_save_npy(
        geometry_dir / "foreground_probability.npy",
        tensor_numpy(probabilities["foreground"][0, 0], np.float16),
    )
    atomic_save_npy(
        geometry_dir / "surface_probability.npy",
        tensor_numpy(probabilities["surface"][0, 0], np.float16),
    )
    atomic_save_npy(
        geometry_dir / "separator_probability.npy",
        tensor_numpy(probabilities["separator"][0, 0], np.float16),
    )
    atomic_save_npy(
        geometry_dir / "seed_probability.npy",
        tensor_numpy(probabilities["seed"][0, 0], np.float16),
    )
    atomic_save_npy(
        geometry_dir / "sdf.npy",
        tensor_numpy(geometry.sdf[0, 0], np.float16),
    )
    atomic_save_npy(
        geometry_dir / "flow_zyx.npy",
        tensor_numpy(geometry.flow[0], np.float16),
    )
    atomic_save_npy(
        geometry_dir / "centroid_offset_zyx.npy",
        tensor_numpy(geometry.centroid_offset[0], np.float16),
    )

    supervoxels = tensor_numpy(result.supervoxel_labels[0], np.int32)
    spatial_labels = tensor_numpy(result.spatial_partition.labels[0], np.int32)
    atomic_save_npy(partition_dir / "watershed_supervoxels.npy", supervoxels)
    atomic_save_npy(partition_dir / "spatial_partition.npy", spatial_labels)

    # RAG / tokenizer state is compact and extremely useful for later diagnosis.
    rag = result.rag
    instances = result.provisional_instances
    partition = result.spatial_partition
    atomic_save_npz(
        rag_dir / "rag_state.npz",
        node_features=tensor_numpy(rag.node_features, np.float16),
        node_embeddings=tensor_numpy(rag.node_embeddings, np.float16),
        node_batch=tensor_numpy(rag.node_batch, np.int32),
        node_supervoxel_id=tensor_numpy(rag.node_supervoxel_id, np.int32),
        node_centroid_um=tensor_numpy(rag.node_centroid_um, np.float32),
        node_volume_voxels=tensor_numpy(rag.node_volume_voxels, np.int32),
        edge_index=tensor_numpy(rag.edge_index, np.int32),
        edge_features=tensor_numpy(rag.edge_features, np.float16),
        edge_embeddings=tensor_numpy(rag.edge_embeddings, np.float16),
        spatial_edge_logits=tensor_numpy(rag.spatial_edge_logits, np.float16),
        spatial_edge_probability=tensor_numpy(
            rag.spatial_edge_logits.float().sigmoid(), np.float16
        ),
        edge_batch=tensor_numpy(rag.edge_batch, np.int32),
        partition_node_component=tensor_numpy(partition.node_component, np.int32),
        partition_node_component_global=tensor_numpy(
            partition.node_component_global, np.int32
        ),
        provisional_tokens=tensor_numpy(instances.tokens, np.float16),
        provisional_ref_um=tensor_numpy(instances.ref_um, np.float32),
        provisional_local_ids=tensor_numpy(instances.local_ids, np.int32),
        provisional_quality_logits=tensor_numpy(
            instances.quality_logits, np.float16
        ),
        provisional_node_to_instance=tensor_numpy(
            instances.node_to_instance, np.int32
        ),
    )

    if save_source_priors:
        # Channel 0 is not duplicated: the canonical preprocessed volume already
        # exists in Stage 6. Foreground is also omitted because labels > 0 is exact.
        atomic_save_npy(
            source_dir / "edt_prior.npy",
            spatial[2].astype(np.float16, copy=False),
        )
        atomic_save_npy(
            source_dir / "boundary_prior.npy",
            spatial[3].astype(np.uint8, copy=False),
        )
        atomic_save_npy(
            source_dir / "marker_prior.npy",
            spatial[4].astype(np.uint8, copy=False),
        )

    pred_fg = tensor_numpy(
        probabilities["foreground"][0, 0], np.float32
    ) >= 0.5
    source_fg = np.asarray(source_labels) > 0
    denominator = int(pred_fg.sum()) + int(source_fg.sum())
    overlap = int(np.count_nonzero(pred_fg & source_fg))
    source_agreement_dice = (
        1.0 if denominator == 0 else (2.0 * overlap / denominator)
    )

    mask_bool = np.asarray(source_mask).astype(bool, copy=False)
    segmentation_outside_mask = int(np.count_nonzero(source_fg & ~mask_bool))

    edge_prob = rag.spatial_edge_logits.detach().float().sigmoid().cpu()
    metrics = {
        "source_instance_count": count_positive_labels(source_labels),
        "source_foreground_fraction": float(source_fg.mean()),
        "stage6_mask_foreground_fraction": float(mask_bool.mean()),
        "segmentation_voxels_outside_stage6_mask": segmentation_outside_mask,
        "pred_foreground_fraction_at_0_5": float(pred_fg.mean()),
        "pred_vs_source_foreground_dice_at_0_5": float(source_agreement_dice),
        "watershed_supervoxel_count": count_positive_labels(supervoxels),
        "spatial_partition_instance_count": count_positive_labels(spatial_labels),
        "rag_node_count": int(rag.node_features.shape[0]),
        "rag_edge_count": int(rag.edge_index.shape[1]),
        "rag_edge_probability_mean": (
            float(edge_prob.mean()) if edge_prob.numel() else float("nan")
        ),
        "rag_edge_probability_q10": (
            float(torch.quantile(edge_prob, 0.10))
            if edge_prob.numel()
            else float("nan")
        ),
        "rag_edge_probability_q50": (
            float(torch.quantile(edge_prob, 0.50))
            if edge_prob.numel()
            else float("nan")
        ),
        "rag_edge_probability_q90": (
            float(torch.quantile(edge_prob, 0.90))
            if edge_prob.numel()
            else float("nan")
        ),
        "rag_edge_probability_q99": (
            float(torch.quantile(edge_prob, 0.99))
            if edge_prob.numel()
            else float("nan")
        ),
        "provisional_instance_count": int(instances.tokens.shape[0]),
    }
    return metrics


# ======================================================================================
# Main
# ======================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run full-volume tiled STIR-Net spatial inference over the Stage-6 "
            "BioHub cache and persist diagnostic outputs."
        )
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint .pt file, or a directory containing checkpoints.",
    )
    checkpoint_group.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory; highest checkpoint_step_XXXXXX.pt is selected.",
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    parser.add_argument(
        "--stage6-root",
        default=None,
        help=(
            "Optional Stage-6 sample directory, or parent containing sample-id. "
            "Default comes from src.io.PipelinePaths."
        ),
    )
    parser.add_argument(
        "--timepoints",
        default="all",
        help='Timepoints: "all", comma list ("0,3,7"), or ranges ("0-19").',
    )
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=DEFAULT_EXPECTED_FRAMES,
        help="Require this many Stage-6 frames. Use 0 to disable the count check.",
    )
    parser.add_argument(
        "--spacing",
        default=",".join(str(v) for v in DEFAULT_SPACING_ZYX_UM),
        help="Physical spacing Z,Y,X in micrometres.",
    )
    parser.add_argument(
        "--tile-shape-zyx",
        default=",".join(str(v) for v in DEFAULT_TILE_SHAPE_ZYX),
    )
    parser.add_argument(
        "--tile-overlap-zyx",
        default=",".join(str(v) for v in DEFAULT_TILE_OVERLAP_ZYX),
    )
    parser.add_argument(
        "--tile-halo-zyx",
        default=",".join(str(v) for v in DEFAULT_TILE_HALO_ZYX),
    )
    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=DEFAULT_TILE_BATCH_SIZE,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Stable run directory. Default: "
            "runs/stirnet/evaluation/12_biohub_full_volume_spatial_inference/"
            "<sample>/stepXXXXXX, or <sample>/<run-label>/stepXXXXXX when "
            "--run-label is supplied."
        ),
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help=(
            "Optional collision-free experiment label used in the default "
            "output path. Useful when multiple checkpoints share the same "
            "global step, e.g. morphology_v2_h100 and morphology_v2_h150."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute frames that already contain _SUCCESS.json.",
    )
    parser.add_argument(
        "--no-source-priors",
        action="store_true",
        help="Do not save EDT/boundary/marker source priors.",
    )
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint(
        args.checkpoint,
        args.checkpoint_dir,
    )
    run_label = validate_run_label(args.run_label)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    print("[model] loading checkpoint ...", flush=True)
    (
        checkpoint,
        model,
        model_cfg,
        train_cfg,
        stripped_training_config,
    ) = load_checkpoint_model_for_inference(
        checkpoint_path,
        device,
    )
    model.eval()
    step = int(checkpoint["global_step"])

    spacing = tuple(float(v.strip()) for v in args.spacing.split(","))
    if len(spacing) != 3 or any(v <= 0 for v in spacing):
        raise ValueError("--spacing must contain three positive Z,Y,X values")

    tile_shape = parse_zyx(args.tile_shape_zyx, name="--tile-shape-zyx")
    tile_overlap = parse_zyx(
        args.tile_overlap_zyx, name="--tile-overlap-zyx"
    )
    tile_halo = parse_zyx(args.tile_halo_zyx, name="--tile-halo-zyx")
    inference_cfg = build_inference_config(
        model_cfg,
        tile_shape_zyx=tile_shape,
        tile_overlap_zyx=tile_overlap,
        tile_halo_zyx=tile_halo,
        tile_batch_size=args.tile_batch_size,
    )

    stage6_root = resolve_stage6_root(args.sample_id, args.stage6_root)
    available = discover_stage6_timepoints(
        stage6_root,
        expected_frames=(
            None if args.expected_frames == 0 else int(args.expected_frames)
        ),
    )
    selected = parse_timepoints(args.timepoints, available)

    if args.output_dir:
        output_root = resolve(args.output_dir)
    else:
        output_base = (
            ROOT
            / "runs"
            / "stirnet"
            / "evaluation"
            / SCRIPT_NAME
            / args.sample_id
        )
        if run_label is not None:
            output_base = output_base / run_label
        output_root = output_base / f"step{step:06d}"

    output_root.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "script": SCRIPT_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": args.sample_id,
        "stage6_root": str(stage6_root),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": step,
        "run_label": run_label,
        "checkpoint_training_config_stripped_for_inference": bool(
            stripped_training_config
        ),
        "device": str(device),
        "spacing_zyx_um": list(spacing),
        "available_timepoints": available,
        "selected_timepoints": selected,
        "expected_frames": (
            None if args.expected_frames == 0 else int(args.expected_frames)
        ),
        "tile_shape_zyx": list(tile_shape),
        "tile_overlap_zyx": list(tile_overlap),
        "tile_halo_zyx": list(tile_halo),
        "tile_batch_size": int(args.tile_batch_size),
        "save_source_priors": not args.no_source_priors,
        "spatial_merge_threshold": float(
            model_cfg.partition.spatial_merge_threshold
        ),
        "foreground_threshold": float(model_cfg.partition.foreground_threshold),
    }

    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    config_payload["run_config_hash"] = config_hash

    existing_manifest = output_root / "manifest.json"
    if existing_manifest.exists() and not args.overwrite:
        previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
        # Ignore timestamp/hash themselves when checking substantive compatibility.
        keys = (
            "sample_id",
            "checkpoint_path",
            "checkpoint_step",
            "run_label",
            "spacing_zyx_um",
            "tile_shape_zyx",
            "tile_overlap_zyx",
            "tile_halo_zyx",
            "tile_batch_size",
            "save_source_priors",
        )
        differences = [
            key
            for key in keys
            if previous.get(key) != config_payload.get(key)
        ]
        if differences:
            raise RuntimeError(
                "Output directory already contains a run with different settings "
                f"({differences}). Use a different --output-dir or --overwrite."
            )
    atomic_json(existing_manifest, config_payload)

    print("=" * 118)
    print("STIR-Net Investigation 12 — full-volume BioHub spatial inference")
    print("=" * 118)
    print("sample                    :", args.sample_id)
    print("stage-6 root              :", stage6_root)
    print("checkpoint                :", checkpoint_path)
    print("checkpoint step           :", step)
    print("run label                 :", run_label)
    print(
        "training-config compat    :",
        "stripped temporary copy"
        if stripped_training_config
        else "native",
    )
    print("device                    :", device)
    print("timepoints                :", selected)
    print("spacing ZYX um            :", spacing)
    print("tile shape ZYX            :", tile_shape)
    print("tile overlap ZYX          :", tile_overlap)
    print("tile halo ZYX             :", tile_halo)
    print("tile batch size           :", args.tile_batch_size)
    print("spatial merge threshold   :", model_cfg.partition.spatial_merge_threshold)
    print("output                    :", output_root)
    print("=" * 118)

    progress_path = output_root / "progress.jsonl"
    run_started = time.perf_counter()
    processed_now = 0
    skipped = 0

    for frame in tqdm(selected, desc="BioHub full-volume", unit="frame"):
        frame_dir = output_root / f"t{frame:03d}"
        success_path = frame_dir / "_SUCCESS.json"

        if success_path.is_file() and not args.overwrite:
            skipped += 1
            tqdm.write(f"[t{frame:03d}] already complete; skipping")
            continue

        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_started = time.perf_counter()
        paths = frame_paths(stage6_root, frame)

        try:
            load_started = time.perf_counter()
            preprocessed = np.load(
                paths["preprocessing"], mmap_mode="r", allow_pickle=False
            )
            stage6_mask = np.load(
                paths["masking"], mmap_mode="r", allow_pickle=False
            )
            source_labels = np.load(
                paths["segmentation"], mmap_mode="r", allow_pickle=False
            )
            if not (
                preprocessed.shape == stage6_mask.shape == source_labels.shape
            ):
                raise ValueError(
                    f"Stage-6 frame shapes disagree at t{frame:03d}: "
                    f"pre={preprocessed.shape}, mask={stage6_mask.shape}, "
                    f"seg={source_labels.shape}"
                )
            load_seconds = time.perf_counter() - load_started

            input_started = time.perf_counter()
            spatial, dref_um = build_stage6_spatial_input(
                preprocessed,
                source_labels,
                spacing,
            )
            input_seconds = time.perf_counter() - input_started

            result, spatial_gpu, amp_name, inference_seconds, peak_gib = (
                run_tiled_spatial(
                    model,
                    spatial,
                    spacing,
                    dref_um,
                    device=device,
                    inference_cfg=inference_cfg,
                )
            )

            save_started = time.perf_counter()
            metrics = save_frame_outputs(
                frame_dir,
                spatial=spatial,
                source_mask=stage6_mask,
                source_labels=source_labels,
                result=result,
                save_source_priors=not args.no_source_priors,
            )
            save_seconds = time.perf_counter() - save_started

            tile_count = int(result.dense.tile_count)
            total_seconds = time.perf_counter() - frame_started

            frame_summary = {
                "status": "success",
                "sample_id": args.sample_id,
                "timepoint": int(frame),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_step": step,
                "stage6_preprocessing_path": str(paths["preprocessing"]),
                "stage6_masking_path": str(paths["masking"]),
                "stage6_segmentation_path": str(paths["segmentation"]),
                "shape_zyx": list(preprocessed.shape),
                "spacing_zyx_um": list(spacing),
                "dref_um": float(dref_um),
                "tile_count": tile_count,
                "tile_shape_zyx": list(tile_shape),
                "tile_overlap_zyx": list(tile_overlap),
                "tile_halo_zyx": list(tile_halo),
                "tile_batch_size": int(args.tile_batch_size),
                "amp_dtype": amp_name,
                "load_seconds": float(load_seconds),
                "source_channel_build_seconds": float(input_seconds),
                "inference_seconds": float(inference_seconds),
                "save_seconds": float(save_seconds),
                "total_seconds": float(total_seconds),
                "peak_allocated_vram_gib": float(peak_gib),
                **metrics,
            }
            atomic_json(frame_dir / "frame_summary.json", frame_summary)
            atomic_json(success_path, frame_summary)
            append_jsonl(progress_path, frame_summary)
            processed_now += 1

            tqdm.write(
                f"[t{frame:03d}] done in {duration(total_seconds)} | "
                f"infer={inference_seconds:.2f}s | tiles={tile_count} | "
                f"source={metrics['source_instance_count']} -> "
                f"spatial={metrics['spatial_partition_instance_count']} | "
                f"VRAM={peak_gib:.2f} GiB"
            )

            # Release full-volume GPU state before the next timepoint.
            del result, spatial_gpu, spatial
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except BaseException as exc:
            failure = {
                "status": "failed",
                "sample_id": args.sample_id,
                "timepoint": int(frame),
                "checkpoint_step": step,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(frame_dir / "_FAILED.json", failure)
            append_jsonl(progress_path, failure)
            raise

    elapsed = time.perf_counter() - run_started

    # Rebuild an authoritative aggregate from durable per-frame success records.
    completed_records: list[dict[str, Any]] = []
    for frame in selected:
        path = output_root / f"t{frame:03d}" / "_SUCCESS.json"
        if path.is_file():
            completed_records.append(
                json.loads(path.read_text(encoding="utf-8"))
            )

    summary = {
        "status": (
            "success"
            if len(completed_records) == len(selected)
            else "incomplete"
        ),
        "sample_id": args.sample_id,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": step,
        "stage6_root": str(stage6_root),
        "output_root": str(output_root),
        "selected_timepoints": selected,
        "completed_timepoints": [
            int(row["timepoint"]) for row in completed_records
        ],
        "completed_count": len(completed_records),
        "selected_count": len(selected),
        "processed_this_invocation": processed_now,
        "skipped_existing_this_invocation": skipped,
        "elapsed_seconds_this_invocation": float(elapsed),
        "elapsed_human_this_invocation": duration(elapsed),
        "mean_inference_seconds": (
            float(
                np.mean(
                    [row["inference_seconds"] for row in completed_records]
                )
            )
            if completed_records
            else float("nan")
        ),
        "mean_total_seconds": (
            float(np.mean([row["total_seconds"] for row in completed_records]))
            if completed_records
            else float("nan")
        ),
        "max_peak_allocated_vram_gib": (
            float(
                max(
                    row["peak_allocated_vram_gib"]
                    for row in completed_records
                )
            )
            if completed_records
            else 0.0
        ),
        "source_instance_counts": {
            f"t{int(row['timepoint']):03d}": int(row["source_instance_count"])
            for row in completed_records
        },
        "spatial_partition_instance_counts": {
            f"t{int(row['timepoint']):03d}": int(
                row["spatial_partition_instance_count"]
            )
            for row in completed_records
        },
    }
    atomic_json(output_root / "summary.json", summary)

    print("\n" + "=" * 118)
    print("INVESTIGATION 12 COMPLETE")
    print("=" * 118)
    print("status                    :", summary["status"])
    print(
        "completed                 :",
        f"{summary['completed_count']}/{summary['selected_count']}",
    )
    print("processed this invocation :", processed_now)
    print("skipped existing          :", skipped)
    print("elapsed                   :", duration(elapsed))
    print(
        "mean inference / frame    :",
        f"{summary['mean_inference_seconds']:.2f}s",
    )
    print(
        "max peak VRAM             :",
        f"{summary['max_peak_allocated_vram_gib']:.2f} GiB",
    )
    print("output                    :", output_root)
    print("=" * 118)


if __name__ == "__main__":
    main()
