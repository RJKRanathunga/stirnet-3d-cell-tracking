from __future__ import annotations

"""
Investigation 18 — morphology-aware RAG full-volume BioHub A/B visualization.

Purpose
-------
Before treating Investigation-17 step 600 as a milestone, run it over the SAME
full BioHub sample and all 20 Stage-6 timepoints used by Investigation 12, then
open the complete T,Z,Y,X sequence in Napari for direct visual comparison.

This script reuses the already-tested production helpers:

    investigations/stirnet/data/12_biohub_full_volume_spatial_inference.py
        -> performs full-volume tiled spatial inference and persists diagnostics

    investigations/stirnet/data/13_biohub_full_volume_spatial_results_viewer.py
        -> provides lazy Dask/memmap loading helpers for Napari

Default morphology checkpoint
-----------------------------
    runs/stirnet/investigations/17_morphology_rag_multicrop_training/
        recovery/drosophila_12_morphology_rag_multicrop_v1/
        best_checkpoint.pt

with fallback to checkpoint_step_000600.pt.

Default baseline
----------------
    runs/stirnet/evaluation/12_biohub_full_volume_spatial_inference/
        44b6_0113de3b/step001000

New output
----------
    runs/stirnet/evaluation/18_morphology_rag_biohub_full_volume/
        44b6_0113de3b/step000600/

What happens
------------
1. Resolve the morphology-aware Investigation-17 checkpoint.
2. Run Investigation 12 on all 20 Stage-6 BioHub timepoints into a NEW
   Investigation-18 directory.
3. Compare the new run with the old step-1000 run:
   * verify watershed identity frame-by-frame;
   * verify RAG edge topology identity when possible;
   * count old->new merge-decision changes at p=0.845;
   * compare instance counts.
4. Write ab_comparison.json and ab_per_frame.jsonl.
5. Build explicit A/B difference masks from watershed-component membership
   and partition boundaries. This avoids relying on arbitrary label colors.
6. Open Napari with all 20 frames and direct A/B layers:
       Raw BioHub volume
       Stage-6 preprocessing
       Stage-6 source segmentation
       Shared watershed supervoxels
       OLD step-1000 spatial partition
       NEW morphology-RAG spatial partition
       NEW foreground / separator / surface / seed / SDF

Typical command
---------------
    python investigations/stirnet/18_morphology_rag_biohub_full_volume_visualization.py

Re-open already completed output:
    python investigations/stirnet/18_morphology_rag_biohub_full_volume_visualization.py ^
        --viewer-only

Recompute all 20 frames:
    python investigations/stirnet/18_morphology_rag_biohub_full_volume_visualization.py ^
        --overwrite

Use --no-viewer for inference/comparison only.
"""

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_NAME = "18_morphology_rag_biohub_full_volume"
DEFAULT_SAMPLE = "44b6_0113de3b"
DEFAULT_EXPECTED_FRAMES = 20
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)
DEFAULT_MERGE_THRESHOLD = 0.845

DEFAULT_MORPH_RECOVERY = (
    "runs/stirnet/investigations/17_morphology_rag_multicrop_training/"
    "recovery/drosophila_12_morphology_rag_multicrop_v1"
)
DEFAULT_TILE_SHAPE_ZYX = "32,128,128"
DEFAULT_TILE_OVERLAP_ZYX = "8,32,32"
DEFAULT_TILE_HALO_ZYX = "4,16,16"
DEFAULT_TILE_BATCH_SIZE = 1
DEFAULT_VECTOR_STRIDE_ZYX = (4, 12, 12)


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
    raise RuntimeError("Could not resolve the cell-tracking repository root.")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True))
        handle.write("\n")
        handle.flush()


def _load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _torch_checkpoint_step(path: Path) -> int:
    import torch
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return int(payload.get("global_step", -1))


def resolve_morphology_checkpoint(override: str | None) -> tuple[Path, int]:
    if override:
        path = resolve(override)
        if not path.is_file():
            raise FileNotFoundError(path)
        step = _torch_checkpoint_step(path)
        if step < 0:
            raise ValueError(f"Checkpoint has no valid global_step: {path}")
        return path, step

    recovery = resolve(DEFAULT_MORPH_RECOVERY)
    candidates = (
        recovery / "best_checkpoint.pt",
        recovery / "checkpoint_step_000600.pt",
    )
    for path in candidates:
        if path.is_file():
            step = _torch_checkpoint_step(path)
            if step >= 0:
                return path.resolve(), step

    fallback = sorted(recovery.glob("checkpoint_step_*.pt"))
    if fallback:
        path = fallback[-1].resolve()
        step = _torch_checkpoint_step(path)
        if step >= 0:
            return path, step

    raise FileNotFoundError(
        "Could not resolve Investigation-17 morphology checkpoint. "
        f"Checked {recovery}. Use --checkpoint."
    )


def resolve_baseline_dir(sample_id: str, override: str | None) -> Path | None:
    path = (
        resolve(override)
        if override
        else ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / "12_biohub_full_volume_spatial_inference"
        / sample_id
        / "step001000"
    )
    if not path.is_dir() or not (path / "summary.json").is_file():
        return None
    return path.resolve()


def new_output_dir(
    sample_id: str,
    checkpoint_step: int,
    override: str | None,
) -> Path:
    if override:
        return resolve(override)
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / SCRIPT_NAME
        / sample_id
        / f"step{checkpoint_step:06d}"
    ).resolve()


def completed_frames(root: Path) -> list[int]:
    frames = []
    if not root.is_dir():
        return frames
    for path in root.glob("t[0-9][0-9][0-9]"):
        if path.is_dir() and (path / "_SUCCESS.json").is_file():
            frames.append(int(path.name[1:]))
    return sorted(frames)



# ======================================================================================
# Investigation-12 checkpoint compatibility
# ======================================================================================

def prepare_inference_checkpoint(
    source_checkpoint: Path,
    *,
    output_dir: Path,
) -> Path:
    """Create an inference-only checkpoint compatible with Eval-04/Inv-12.

    Investigation 17 saves experiment-local metadata in ``training_config``:
    ``experiment`` and ``legacy_rag_frozen``. Investigation 12 loads checkpoints
    through the strict Eval-04 helper, which assumes every dictionary stored
    under ``training_config`` is a serialized TrainingConfig dataclass. Those
    experiment-local keys are therefore rejected before model loading.

    Spatial inference does not need optimizer/training metadata. Eval-04
    explicitly supports checkpoints with no training_config and falls back to
    evaluation-only TrainingConfig defaults.

    This function preserves the exact architecture, model state, model_config,
    global_step, checkpoint_version and epoch, while omitting only:
        training_config
        optimizer
        scheduler
        scaler

    The original Investigation-17 checkpoint is never modified.
    """
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    source_step = _torch_checkpoint_step(source_checkpoint)
    destination = (
        output_dir
        / f"_inference_checkpoint_step{source_step:06d}.pt"
    )

    # Reuse an already prepared copy when it is still valid.
    if (
        destination.is_file()
        and destination.stat().st_mtime_ns
        >= source_checkpoint.stat().st_mtime_ns
    ):
        try:
            try:
                cached = torch.load(
                    destination,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                cached = torch.load(destination, map_location="cpu")
            if (
                int(cached.get("global_step", -1)) == source_step
                and "model" in cached
                and "model_config" in cached
                and "training_config" not in cached
            ):
                print(
                    "[checkpoint] reusing inference-compatible copy:",
                    destination,
                    flush=True,
                )
                return destination.resolve()
        except Exception:
            pass

    try:
        payload = torch.load(
            source_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(source_checkpoint, map_location="cpu")

    required = ("architecture", "model", "global_step", "model_config")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(
            "Morphology checkpoint is missing inference fields: "
            + ", ".join(missing)
        )

    inference_payload = {
        "architecture": payload["architecture"],
        "checkpoint_version": payload.get("checkpoint_version"),
        "model": payload["model"],
        "global_step": int(payload["global_step"]),
        "epoch": int(payload.get("epoch", 0)),
        "model_config": payload["model_config"],
        "extra": {
            "inference_copy": True,
            "source_checkpoint": str(source_checkpoint),
            "source_extra": payload.get("extra", {}),
        },
    }

    temp = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        torch.save(inference_payload, temp)
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)

    # Verify the copy before launching the expensive 20-frame inference.
    try:
        verified = torch.load(
            destination,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        verified = torch.load(destination, map_location="cpu")

    if "training_config" in verified:
        raise RuntimeError(
            "Inference checkpoint unexpectedly contains training_config"
        )
    if int(verified.get("global_step", -1)) != source_step:
        raise RuntimeError("Inference checkpoint global_step changed")
    if payload["model"].keys() != verified["model"].keys():
        raise RuntimeError("Inference checkpoint model-state keys changed")
    for name, tensor in payload["model"].items():
        copied = verified["model"][name]
        if tensor.shape != copied.shape or tensor.dtype != copied.dtype:
            raise RuntimeError(
                f"Inference checkpoint tensor metadata changed: {name}"
            )

    print(
        "[checkpoint] created inference-compatible copy:",
        destination,
        flush=True,
    )
    print(
        "[checkpoint] model tensors preserved:",
        len(verified["model"]),
        "| stripped training/optimizer metadata only",
        flush=True,
    )
    return destination.resolve()

def run_full_volume_inference(
    *,
    checkpoint: Path,
    output_dir: Path,
    sample_id: str,
    stage6_root: str | None,
    device: str,
    timepoints: str,
    expected_frames: int,
    spacing: str,
    tile_shape_zyx: str,
    tile_overlap_zyx: str,
    tile_halo_zyx: str,
    tile_batch_size: int,
    overwrite: bool,
    no_source_priors: bool,
) -> None:
    script = ROOT / "investigations/stirnet/data/12_biohub_full_volume_spatial_inference.py"
    if not script.is_file():
        raise FileNotFoundError(
            "Investigation 18 reuses Investigation 12, but it is missing: "
            f"{script}"
        )

    inference_checkpoint = prepare_inference_checkpoint(
        checkpoint,
        output_dir=output_dir,
    )

    command = [
        sys.executable,
        str(script),
        "--checkpoint", str(inference_checkpoint),
        "--sample-id", sample_id,
        "--timepoints", timepoints,
        "--expected-frames", str(expected_frames),
        "--spacing", spacing,
        "--tile-shape-zyx", tile_shape_zyx,
        "--tile-overlap-zyx", tile_overlap_zyx,
        "--tile-halo-zyx", tile_halo_zyx,
        "--tile-batch-size", str(tile_batch_size),
        "--device", device,
        "--output-dir", str(output_dir),
    ]
    if stage6_root:
        command.extend(["--stage6-root", stage6_root])
    if overwrite:
        command.append("--overwrite")
    if no_source_priors:
        command.append("--no-source-priors")

    print("=" * 118, flush=True)
    print("INVESTIGATION 18 — MORPHOLOGY-AWARE FULL-VOLUME INFERENCE", flush=True)
    print("=" * 118, flush=True)
    print("source checkpoint    :", checkpoint, flush=True)
    print("inference checkpoint :", inference_checkpoint, flush=True)
    print("output               :", output_dir, flush=True)
    print("=" * 118, flush=True)

    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise RuntimeError(
            "Investigation-12 inference subprocess failed with exit code "
            f"{completed.returncode}"
        )


def _load_npy(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _positive_label_count(labels) -> int:
    ids = np.unique(np.asarray(labels))
    return int(np.count_nonzero(ids > 0))



def _partition_boundary_mask(labels: np.ndarray) -> np.ndarray:
    """6-neighbour internal instance boundary mask.

    Only boundaries between two positive labels are included. Foreground/background
    contours are intentionally excluded because Investigation 18 is diagnosing RAG
    split/merge changes, not dense-foreground changes.
    """
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError(f"Expected 3-D partition labels, got {labels.shape}")

    boundary = np.zeros(labels.shape, dtype=bool)
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        left = tuple(left)
        right = tuple(right)

        a = labels[left]
        b = labels[right]
        changed = (a > 0) & (b > 0) & (a != b)
        boundary[left] |= changed
        boundary[right] |= changed
    return boundary


def _supervoxel_component_maps(
    watershed: np.ndarray,
    partition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, frozenset[int]]]:
    """Map every watershed supervoxel to its final partition component.

    Partition integer IDs are arbitrary between old/new runs. We therefore compare
    component MEMBERSHIP SETS of supervoxel IDs, never the raw partition label IDs.
    """
    watershed = np.asarray(watershed)
    partition = np.asarray(partition)
    if watershed.shape != partition.shape:
        raise ValueError(
            f"Watershed/partition shape mismatch: {watershed.shape} vs {partition.shape}"
        )

    flat_ws = watershed.reshape(-1)
    flat_part = partition.reshape(-1)

    positive = flat_ws > 0
    if not bool(positive.any()):
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            {},
        )

    ws_positive = flat_ws[positive].astype(np.int64, copy=False)
    part_positive = flat_part[positive].astype(np.int64, copy=False)

    # A production partition should assign an entire supervoxel to one component.
    # Use the first voxel for the mapping, then verify uniformity cheaply through
    # unique (SV, component) pairs.
    sv_ids, first = np.unique(ws_positive, return_index=True)
    component_ids = part_positive[first]

    pairs = np.stack([ws_positive, part_positive], axis=1)
    unique_pairs = np.unique(pairs, axis=0)
    pair_counts: dict[int, int] = {}
    for sv_id in unique_pairs[:, 0].tolist():
        pair_counts[int(sv_id)] = pair_counts.get(int(sv_id), 0) + 1
    nonuniform = [sv for sv, count in pair_counts.items() if count > 1]
    if nonuniform:
        raise RuntimeError(
            "A final partition split one or more atomic watershed supervoxels, "
            f"which should not happen. Examples: {nonuniform[:10]}"
        )

    groups: dict[int, set[int]] = {}
    for sv_id, component_id in zip(sv_ids.tolist(), component_ids.tolist()):
        if int(component_id) <= 0:
            continue
        groups.setdefault(int(component_id), set()).add(int(sv_id))

    frozen_groups = {
        component_id: frozenset(members)
        for component_id, members in groups.items()
    }
    return (
        sv_ids.astype(np.int64, copy=False),
        component_ids.astype(np.int64, copy=False),
        frozen_groups,
    )


def _changed_supervoxel_ids(
    watershed: np.ndarray,
    old_partition: np.ndarray,
    new_partition: np.ndarray,
) -> np.ndarray:
    """Return SV IDs whose final component membership changed old->new."""
    old_sv, old_comp, old_groups = _supervoxel_component_maps(
        watershed,
        old_partition,
    )
    new_sv, new_comp, new_groups = _supervoxel_component_maps(
        watershed,
        new_partition,
    )

    if not np.array_equal(old_sv, new_sv):
        raise RuntimeError(
            "Old/new partitions are not defined on the same watershed supervoxel IDs"
        )

    changed: list[int] = []
    for sv_id, old_component, new_component in zip(
        old_sv.tolist(),
        old_comp.tolist(),
        new_comp.tolist(),
    ):
        old_members = old_groups.get(int(old_component), frozenset())
        new_members = new_groups.get(int(new_component), frozenset())
        if old_members != new_members:
            changed.append(int(sv_id))
    return np.asarray(changed, dtype=np.int64)


def _changed_edge_supervoxel_ids_and_points(
    *,
    rag_path: Path,
    other_rag_path: Path,
    merge_threshold: float,
    old_to_new_separate: bool,
    displayed_time_index: int | None,
    spacing_zyx_um: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoint SV IDs and optional Napari points for changed RAG edges.

    edge_index stores NODE indices. node_supervoxel_id is therefore mandatory;
    never assume node index + 1 equals the watershed label ID.
    """
    with np.load(rag_path, allow_pickle=False) as old_rag, np.load(
        other_rag_path,
        allow_pickle=False,
    ) as new_rag:
        old_edges = np.asarray(old_rag["edge_index"], dtype=np.int64)
        new_edges = np.asarray(new_rag["edge_index"], dtype=np.int64)
        if old_edges.shape != new_edges.shape or not np.array_equal(old_edges, new_edges):
            return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

        old_prob = np.asarray(
            old_rag["spatial_edge_probability"],
            dtype=np.float32,
        )
        new_prob = np.asarray(
            new_rag["spatial_edge_probability"],
            dtype=np.float32,
        )
        old_merge = old_prob >= float(merge_threshold)
        new_merge = new_prob >= float(merge_threshold)
        changed = (
            (old_merge & ~new_merge)
            if old_to_new_separate
            else (~old_merge & new_merge)
        )
        edge_ids = np.flatnonzero(changed)
        if edge_ids.size == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

        node_sv = np.asarray(
            old_rag["node_supervoxel_id"],
            dtype=np.int64,
        )
        endpoints = old_edges[:, edge_ids]
        sv_ids = np.unique(node_sv[endpoints.reshape(-1)])

        points = np.zeros((0, 4), dtype=np.float32)
        if (
            displayed_time_index is not None
            and spacing_zyx_um is not None
            and "node_centroid_um" in old_rag
        ):
            centroids_um = np.asarray(
                old_rag["node_centroid_um"],
                dtype=np.float32,
            )
            a = centroids_um[endpoints[0]]
            b = centroids_um[endpoints[1]]
            mid_um = 0.5 * (a + b)
            spacing = np.asarray(spacing_zyx_um, dtype=np.float32)
            mid_zyx = mid_um / spacing[None, :]
            points = np.concatenate(
                [
                    np.full(
                        (mid_zyx.shape[0], 1),
                        float(displayed_time_index),
                        dtype=np.float32,
                    ),
                    mid_zyx.astype(np.float32, copy=False),
                ],
                axis=1,
            )

        return sv_ids.astype(np.int64, copy=False), points


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_ab_difference_artifacts(
    *,
    baseline_root: Path,
    new_root: Path,
    frames: list[int],
    merge_threshold: float,
) -> dict[str, Any]:
    """Persist explicit old/new difference masks for the Napari viewer.

    These artifacts are tiny uint8 masks compared with the inference outputs and
    let Napari show ONLY changed boundaries/components instead of forcing a human
    to toggle two nearly identical full label volumes.
    """
    all_rows: list[dict[str, Any]] = []

    for frame in frames:
        old_frame = baseline_root / f"t{frame:03d}"
        new_frame = new_root / f"t{frame:03d}"
        diff_dir = new_frame / "ab_diff"
        diff_dir.mkdir(parents=True, exist_ok=True)

        watershed = np.asarray(
            _load_npy(new_frame / "partition/watershed_supervoxels.npy")
        )
        old_partition = np.asarray(
            _load_npy(old_frame / "partition/spatial_partition.npy")
        )
        new_partition = np.asarray(
            _load_npy(new_frame / "partition/spatial_partition.npy")
        )

        old_boundary = _partition_boundary_mask(old_partition)
        new_boundary = _partition_boundary_mask(new_partition)

        new_split_boundary = new_boundary & ~old_boundary
        removed_boundary = old_boundary & ~new_boundary

        changed_sv = _changed_supervoxel_ids(
            watershed,
            old_partition,
            new_partition,
        )
        changed_components = np.isin(watershed, changed_sv)

        old_rag = old_frame / "rag/rag_state.npz"
        new_rag = new_frame / "rag/rag_state.npz"
        split_sv, _ = _changed_edge_supervoxel_ids_and_points(
            rag_path=old_rag,
            other_rag_path=new_rag,
            merge_threshold=merge_threshold,
            old_to_new_separate=True,
            displayed_time_index=None,
            spacing_zyx_um=None,
        )
        merge_sv, _ = _changed_edge_supervoxel_ids_and_points(
            rag_path=old_rag,
            other_rag_path=new_rag,
            merge_threshold=merge_threshold,
            old_to_new_separate=False,
            displayed_time_index=None,
            spacing_zyx_um=None,
        )

        split_edge_supervoxels = np.isin(watershed, split_sv)
        merge_edge_supervoxels = np.isin(watershed, merge_sv)

        _save_npy_atomic(
            diff_dir / "new_split_boundaries.npy",
            new_split_boundary.astype(np.uint8, copy=False),
        )
        _save_npy_atomic(
            diff_dir / "removed_old_boundaries.npy",
            removed_boundary.astype(np.uint8, copy=False),
        )
        _save_npy_atomic(
            diff_dir / "changed_components.npy",
            changed_components.astype(np.uint8, copy=False),
        )
        _save_npy_atomic(
            diff_dir / "old_merge_new_separate_supervoxels.npy",
            split_edge_supervoxels.astype(np.uint8, copy=False),
        )
        _save_npy_atomic(
            diff_dir / "old_separate_new_merge_supervoxels.npy",
            merge_edge_supervoxels.astype(np.uint8, copy=False),
        )

        row = {
            "timepoint": int(frame),
            "changed_supervoxel_count": int(changed_sv.size),
            "new_split_boundary_voxel_count": int(new_split_boundary.sum()),
            "removed_old_boundary_voxel_count": int(removed_boundary.sum()),
            "old_merge_new_separate_supervoxel_count": int(split_sv.size),
            "old_separate_new_merge_supervoxel_count": int(merge_sv.size),
        }
        atomic_json(diff_dir / "summary.json", row)
        all_rows.append(row)

    summary = {
        "status": "success",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frames": frames,
        "frame_count": len(frames),
        "changed_supervoxel_count_total": int(
            sum(row["changed_supervoxel_count"] for row in all_rows)
        ),
        "new_split_boundary_voxel_count_total": int(
            sum(row["new_split_boundary_voxel_count"] for row in all_rows)
        ),
        "removed_old_boundary_voxel_count_total": int(
            sum(row["removed_old_boundary_voxel_count"] for row in all_rows)
        ),
        "per_frame": all_rows,
    }
    atomic_json(new_root / "ab_diff_summary.json", summary)
    return summary


def _collect_changed_edge_points(
    *,
    baseline_root: Path,
    new_root: Path,
    frames: list[int],
    merge_threshold: float,
    spacing_zyx_um: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    split_points: list[np.ndarray] = []
    merge_points: list[np.ndarray] = []

    for displayed_time_index, frame in enumerate(frames):
        old_rag = baseline_root / f"t{frame:03d}" / "rag/rag_state.npz"
        new_rag = new_root / f"t{frame:03d}" / "rag/rag_state.npz"

        _, points = _changed_edge_supervoxel_ids_and_points(
            rag_path=old_rag,
            other_rag_path=new_rag,
            merge_threshold=merge_threshold,
            old_to_new_separate=True,
            displayed_time_index=displayed_time_index,
            spacing_zyx_um=spacing_zyx_um,
        )
        if points.size:
            split_points.append(points)

        _, points = _changed_edge_supervoxel_ids_and_points(
            rag_path=old_rag,
            other_rag_path=new_rag,
            merge_threshold=merge_threshold,
            old_to_new_separate=False,
            displayed_time_index=displayed_time_index,
            spacing_zyx_um=spacing_zyx_um,
        )
        if points.size:
            merge_points.append(points)

    split = (
        np.concatenate(split_points, axis=0)
        if split_points
        else np.zeros((0, 4), dtype=np.float32)
    )
    merge = (
        np.concatenate(merge_points, axis=0)
        if merge_points
        else np.zeros((0, 4), dtype=np.float32)
    )
    return split, merge

def compare_frame(
    *,
    frame: int,
    baseline_root: Path,
    new_root: Path,
    merge_threshold: float,
) -> dict[str, Any]:
    old_frame = baseline_root / f"t{frame:03d}"
    new_frame = new_root / f"t{frame:03d}"

    old_ws = _load_npy(old_frame / "partition/watershed_supervoxels.npy")
    new_ws = _load_npy(new_frame / "partition/watershed_supervoxels.npy")
    old_part = _load_npy(old_frame / "partition/spatial_partition.npy")
    new_part = _load_npy(new_frame / "partition/spatial_partition.npy")

    if old_ws.shape != new_ws.shape:
        raise ValueError(
            f"t{frame:03d}: watershed shape mismatch "
            f"{old_ws.shape} vs {new_ws.shape}"
        )

    row: dict[str, Any] = {
        "timepoint": frame,
        "watershed_equal": bool(np.array_equal(old_ws, new_ws)),
        "baseline_supervoxel_count": _positive_label_count(old_ws),
        "new_supervoxel_count": _positive_label_count(new_ws),
        "baseline_partition_instance_count": _positive_label_count(old_part),
        "new_partition_instance_count": _positive_label_count(new_part),
    }
    row["partition_instance_delta_new_minus_old"] = (
        row["new_partition_instance_count"]
        - row["baseline_partition_instance_count"]
    )

    if row["watershed_equal"]:
        changed_sv = _changed_supervoxel_ids(old_ws, old_part, new_part)
        old_boundary = _partition_boundary_mask(old_part)
        new_boundary = _partition_boundary_mask(new_part)
        row["changed_supervoxel_count"] = int(changed_sv.size)
        row["partition_grouping_equal"] = bool(changed_sv.size == 0)
        row["new_split_boundary_voxel_count"] = int(
            np.count_nonzero(new_boundary & ~old_boundary)
        )
        row["removed_old_boundary_voxel_count"] = int(
            np.count_nonzero(old_boundary & ~new_boundary)
        )
    else:
        row["changed_supervoxel_count"] = None
        row["partition_grouping_equal"] = None

    old_rag_path = old_frame / "rag/rag_state.npz"
    new_rag_path = new_frame / "rag/rag_state.npz"
    if old_rag_path.is_file() and new_rag_path.is_file():
        with np.load(old_rag_path, allow_pickle=False) as old_rag, np.load(
            new_rag_path, allow_pickle=False
        ) as new_rag:
            old_edges = np.asarray(old_rag["edge_index"], dtype=np.int64)
            new_edges = np.asarray(new_rag["edge_index"], dtype=np.int64)
            old_prob = np.asarray(old_rag["spatial_edge_probability"], dtype=np.float32)
            new_prob = np.asarray(new_rag["spatial_edge_probability"], dtype=np.float32)

            topology_equal = (
                old_edges.shape == new_edges.shape
                and np.array_equal(old_edges, new_edges)
            )
            row["rag_edge_topology_equal"] = bool(topology_equal)
            row["baseline_rag_edge_count"] = int(old_edges.shape[1])
            row["new_rag_edge_count"] = int(new_edges.shape[1])

            if topology_equal and old_prob.shape == new_prob.shape:
                old_merge = old_prob >= float(merge_threshold)
                new_merge = new_prob >= float(merge_threshold)
                row.update(
                    {
                        "rag_probability_mean_old": float(old_prob.mean()) if old_prob.size else float("nan"),
                        "rag_probability_mean_new": float(new_prob.mean()) if new_prob.size else float("nan"),
                        "rag_probability_mean_delta": float((new_prob - old_prob).mean()) if old_prob.size else float("nan"),
                        "old_merge_new_separate_edge_count": int(np.count_nonzero(old_merge & ~new_merge)),
                        "old_separate_new_merge_edge_count": int(np.count_nonzero(~old_merge & new_merge)),
                        "unchanged_merge_edge_count": int(np.count_nonzero(old_merge & new_merge)),
                        "unchanged_separate_edge_count": int(np.count_nonzero(~old_merge & ~new_merge)),
                    }
                )

    for field in (
        "foreground_probability",
        "separator_probability",
        "surface_probability",
        "seed_probability",
        "sdf",
    ):
        old_path = old_frame / "geometry" / f"{field}.npy"
        new_path = new_frame / "geometry" / f"{field}.npy"
        if old_path.is_file() and new_path.is_file():
            old = np.asarray(_load_npy(old_path), dtype=np.float32)
            new = np.asarray(_load_npy(new_path), dtype=np.float32)
            if old.shape == new.shape:
                row[f"{field}_max_abs_delta"] = (
                    float(np.max(np.abs(new - old))) if old.size else 0.0
                )

    return row


def compare_runs(
    *,
    baseline_root: Path,
    new_root: Path,
    frames: list[int],
    merge_threshold: float,
) -> dict[str, Any]:
    per_frame_path = new_root / "ab_per_frame.jsonl"
    per_frame_path.unlink(missing_ok=True)
    rows = []
    for frame in frames:
        row = compare_frame(
            frame=frame,
            baseline_root=baseline_root,
            new_root=new_root,
            merge_threshold=merge_threshold,
        )
        rows.append(row)
        append_jsonl(per_frame_path, row)

    watershed_mismatch = [
        row["timepoint"] for row in rows if not row["watershed_equal"]
    ]
    topology_mismatch = [
        row["timepoint"]
        for row in rows
        if row.get("rag_edge_topology_equal") is False
    ]

    summary = {
        "status": "success",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_root": str(baseline_root),
        "new_root": str(new_root),
        "frames": frames,
        "merge_threshold": float(merge_threshold),
        "watershed_equal_all_frames": not watershed_mismatch,
        "watershed_mismatch_frames": watershed_mismatch,
        "rag_edge_topology_equal_all_comparable_frames": not topology_mismatch,
        "rag_edge_topology_mismatch_frames": topology_mismatch,
        "total_old_merge_new_separate_edges": int(
            sum(row.get("old_merge_new_separate_edge_count", 0) for row in rows)
        ),
        "total_old_separate_new_merge_edges": int(
            sum(row.get("old_separate_new_merge_edge_count", 0) for row in rows)
        ),
        "baseline_partition_instance_count_total": int(
            sum(row["baseline_partition_instance_count"] for row in rows)
        ),
        "new_partition_instance_count_total": int(
            sum(row["new_partition_instance_count"] for row in rows)
        ),
        "partition_instance_count_delta_total": int(
            sum(row["partition_instance_delta_new_minus_old"] for row in rows)
        ),
        "per_frame": rows,
    }
    diff_summary = ensure_ab_difference_artifacts(
        baseline_root=baseline_root,
        new_root=new_root,
        frames=frames,
        merge_threshold=merge_threshold,
    )
    summary["difference_artifacts"] = {
        "summary_path": str(new_root / "ab_diff_summary.json"),
        "changed_supervoxel_count_total": diff_summary[
            "changed_supervoxel_count_total"
        ],
        "new_split_boundary_voxel_count_total": diff_summary[
            "new_split_boundary_voxel_count_total"
        ],
        "removed_old_boundary_voxel_count_total": diff_summary[
            "removed_old_boundary_voxel_count_total"
        ],
    }
    atomic_json(new_root / "ab_comparison.json", summary)
    return summary


def parse_zyx_positive(text: str, *, name: str) -> tuple[int, int, int]:
    values = tuple(int(token.strip()) for token in text.split(","))
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError(f"{name} must contain three positive Z,Y,X integers")
    return values


def open_ab_viewer(
    *,
    sample_id: str,
    new_root: Path,
    baseline_root: Path | None,
    stage6_root_override: str | None,
    sample_zarr_override: str | None,
    timepoints: str,
    vectors: bool,
    vector_timepoint: int,
    vector_stride_zyx: tuple[int, int, int],
    vector_min_magnitude: float,
    flow_scale: float,
    centroid_offset_scale: float,
    no_geometry: bool,
    merge_threshold: float,
) -> None:
    viewer_script = ROOT / "investigations/stirnet/data/13_biohub_full_volume_spatial_results_viewer.py"
    V13 = _load_module(viewer_script, "_investigation13_for_investigation18")

    stage6_root = V13.resolve_stage6_root(sample_id, stage6_root_override)
    sample_zarr = V13.resolve_sample_zarr(sample_id, sample_zarr_override)
    available = V13.completed_timepoints(new_root)
    frames = V13.parse_timepoints(timepoints, available)

    if baseline_root is not None:
        baseline_available = set(V13.completed_timepoints(baseline_root))
        missing = [frame for frame in frames if frame not in baseline_available]
        if missing:
            raise FileNotFoundError(
                f"Baseline run is missing displayed frames: {missing}"
            )

    manifest = json.loads((new_root / "manifest.json").read_text(encoding="utf-8"))
    spacing = tuple(
        float(v)
        for v in manifest.get("spacing_zyx_um", DEFAULT_SPACING_ZYX_UM)
    )
    scale_4d = (1.0, *spacing)

    raw, _ = V13.load_raw_time_series(sample_zarr, frames)
    preprocessed, _ = V13.stack_npy(
        V13.stage6_paths(stage6_root, frames, "preprocessing"),
        name="Stage-6 preprocessing",
    )
    source_mask, _ = V13.stack_npy(
        V13.stage6_paths(stage6_root, frames, "masking"),
        name="Stage-6 mask",
    )
    source_labels, _ = V13.stack_npy(
        V13.stage6_paths(stage6_root, frames, "segmentation"),
        name="Stage-6 source segmentation",
    )
    watershed, _ = V13.stack_npy(
        V13.inference_paths(new_root, frames, "partition/watershed_supervoxels.npy"),
        name="shared watershed supervoxels",
    )
    new_partition, _ = V13.stack_npy(
        V13.inference_paths(new_root, frames, "partition/spatial_partition.npy"),
        name="new morphology-RAG partition",
    )
    old_partition = None
    new_split_boundaries = None
    removed_old_boundaries = None
    changed_components = None
    old_merge_new_separate_svs = None
    old_separate_new_merge_svs = None
    split_change_points = np.zeros((0, 4), dtype=np.float32)
    merge_change_points = np.zeros((0, 4), dtype=np.float32)

    if baseline_root is not None:
        old_partition, _ = V13.stack_npy(
            V13.inference_paths(
                baseline_root,
                frames,
                "partition/spatial_partition.npy",
            ),
            name="old step-1000 partition",
        )

        # viewer-only mode may be used with an older Investigation-18 output
        # created before explicit A/B masks existed. Build them on demand.
        ensure_ab_difference_artifacts(
            baseline_root=baseline_root,
            new_root=new_root,
            frames=frames,
            merge_threshold=merge_threshold,
        )

        new_split_boundaries, _ = V13.stack_npy(
            V13.inference_paths(
                new_root,
                frames,
                "ab_diff/new_split_boundaries.npy",
            ),
            name="A/B new split boundaries",
        )
        removed_old_boundaries, _ = V13.stack_npy(
            V13.inference_paths(
                new_root,
                frames,
                "ab_diff/removed_old_boundaries.npy",
            ),
            name="A/B removed old boundaries",
        )
        changed_components, _ = V13.stack_npy(
            V13.inference_paths(
                new_root,
                frames,
                "ab_diff/changed_components.npy",
            ),
            name="A/B changed components",
        )
        old_merge_new_separate_svs, _ = V13.stack_npy(
            V13.inference_paths(
                new_root,
                frames,
                "ab_diff/old_merge_new_separate_supervoxels.npy",
            ),
            name="A/B old-merge new-separate supervoxels",
        )
        old_separate_new_merge_svs, _ = V13.stack_npy(
            V13.inference_paths(
                new_root,
                frames,
                "ab_diff/old_separate_new_merge_supervoxels.npy",
            ),
            name="A/B old-separate new-merge supervoxels",
        )

        split_change_points, merge_change_points = _collect_changed_edge_points(
            baseline_root=baseline_root,
            new_root=new_root,
            frames=frames,
            merge_threshold=merge_threshold,
            spacing_zyx_um=spacing,
        )

    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is not installed. Install visualization dependencies and "
            "rerun Investigation 18 with --viewer-only."
        ) from exc

    viewer = napari.Viewer(
        title=(
            f"STIR-Net Morphology-RAG A/B | {sample_id} | "
            f"step {manifest.get('checkpoint_step', '?')}"
        )
    )
    viewer.add_image(
        raw,
        name="Raw BioHub volume",
        scale=scale_4d,
        colormap="gray",
        visible=False,
    )
    viewer.add_image(
        preprocessed,
        name="Stage-6 preprocessing",
        scale=scale_4d,
        colormap="gray",
        contrast_limits=(0.0, 1.0),
        visible=True,
    )
    viewer.add_labels(
        source_mask,
        name="Stage-6 binary mask",
        scale=scale_4d,
        opacity=0.25,
        visible=False,
    )
    viewer.add_labels(
        source_labels,
        name="Stage-6 source segmentation",
        scale=scale_4d,
        opacity=0.42,
        visible=False,
    )
    viewer.add_labels(
        watershed,
        name="Shared watershed supervoxels",
        scale=scale_4d,
        opacity=0.45,
        visible=False,
    )
    if old_partition is not None:
        viewer.add_labels(
            old_partition,
            name="OLD RAG spatial partition (step1000)",
            scale=scale_4d,
            opacity=0.58,
            visible=False,
        )
    viewer.add_labels(
        new_partition,
        name="NEW morphology-RAG spatial partition",
        scale=scale_4d,
        opacity=0.50,
        visible=True,
    )

    # Explicit A/B overlays. These are the important Investigation-18 layers:
    # they display ONLY locations where the old and new partition geometry differs.
    if new_split_boundaries is not None:
        viewer.add_image(
            new_split_boundaries,
            name="A/B NEW split boundaries (new only)",
            scale=scale_4d,
            colormap="green",
            contrast_limits=(0.0, 1.0),
            opacity=1.0,
            blending="additive",
            visible=True,
        )
        viewer.add_image(
            removed_old_boundaries,
            name="A/B NEW merge boundaries removed (old only)",
            scale=scale_4d,
            colormap="red",
            contrast_limits=(0.0, 1.0),
            opacity=1.0,
            blending="additive",
            visible=False,
        )
        viewer.add_image(
            changed_components,
            name="A/B changed final components",
            scale=scale_4d,
            colormap="cyan",
            contrast_limits=(0.0, 1.0),
            opacity=0.24,
            blending="additive",
            visible=False,
        )
        viewer.add_image(
            old_merge_new_separate_svs,
            name="A/B old-merge -> NEW-separate supervoxels",
            scale=scale_4d,
            colormap="yellow",
            contrast_limits=(0.0, 1.0),
            opacity=0.22,
            blending="additive",
            visible=True,
        )
        viewer.add_image(
            old_separate_new_merge_svs,
            name="A/B old-separate -> NEW-merge supervoxels",
            scale=scale_4d,
            colormap="magenta",
            contrast_limits=(0.0, 1.0),
            opacity=0.28,
            blending="additive",
            visible=False,
        )

        if split_change_points.size:
            viewer.add_points(
                split_change_points,
                name="A/B changed edges: old merge -> NEW separate",
                scale=scale_4d,
                size=6.0,
                face_color="yellow",
                border_color="black",
                opacity=0.95,
                visible=True,
            )
        if merge_change_points.size:
            viewer.add_points(
                merge_change_points,
                name="A/B changed edges: old separate -> NEW merge",
                scale=scale_4d,
                size=6.0,
                face_color="magenta",
                border_color="black",
                opacity=0.95,
                visible=False,
            )

    if not no_geometry:
        geometry_specs = (
            ("foreground probability", "geometry/foreground_probability.npy", "magenta", True, (0.0, 1.0)),
            ("separator probability", "geometry/separator_probability.npy", "yellow", False, (0.0, 1.0)),
            ("surface probability", "geometry/surface_probability.npy", "cyan", False, (0.0, 1.0)),
            ("seed probability", "geometry/seed_probability.npy", "green", False, (0.0, 1.0)),
            ("SDF", "geometry/sdf.npy", "turbo", False, None),
        )
        for layer_name, relative, colormap, visible, limits in geometry_specs:
            data, _ = V13.stack_npy(
                V13.inference_paths(new_root, frames, relative),
                name=layer_name,
            )
            kwargs = {
                "name": f"STIR-Net {layer_name}",
                "scale": scale_4d,
                "colormap": colormap,
                "visible": visible,
                "blending": "additive",
            }
            if limits is not None:
                kwargs["contrast_limits"] = limits
            viewer.add_image(data, **kwargs)

    if vectors:
        if vector_timepoint not in frames:
            raise ValueError(
                f"--vector-timepoint {vector_timepoint} is not displayed"
            )
        displayed_index = frames.index(vector_timepoint)
        frame_dir = new_root / f"t{vector_timepoint:03d}"
        flow_vectors = V13.build_vector_layer(
            frame_dir / "geometry/flow_zyx.npy",
            frame_dir / "geometry/foreground_probability.npy",
            displayed_time_index=displayed_index,
            stride_zyx=vector_stride_zyx,
            minimum_magnitude=vector_min_magnitude,
            vector_scale=flow_scale,
        )
        centroid_vectors = V13.build_vector_layer(
            frame_dir / "geometry/centroid_offset_zyx.npy",
            frame_dir / "geometry/foreground_probability.npy",
            displayed_time_index=displayed_index,
            stride_zyx=vector_stride_zyx,
            minimum_magnitude=vector_min_magnitude,
            vector_scale=centroid_offset_scale,
        )
        viewer.add_vectors(
            flow_vectors,
            name=f"flow vectors t{vector_timepoint:03d}",
            scale=scale_4d,
            edge_width=0.7,
            opacity=0.8,
            visible=True,
        )
        viewer.add_vectors(
            centroid_vectors,
            name=f"centroid-offset vectors t{vector_timepoint:03d}",
            scale=scale_4d,
            edge_width=0.7,
            opacity=0.8,
            visible=False,
        )

    start_display_index = 0
    if baseline_root is not None:
        diff_summary_path = new_root / "ab_diff_summary.json"
        if diff_summary_path.is_file():
            diff_summary = json.loads(
                diff_summary_path.read_text(encoding="utf-8")
            )
            per_frame_change = {
                int(row["timepoint"]): (
                    int(row["new_split_boundary_voxel_count"])
                    + int(row["removed_old_boundary_voxel_count"])
                )
                for row in diff_summary.get("per_frame", [])
            }
            if per_frame_change:
                max_frame = max(
                    frames,
                    key=lambda frame: per_frame_change.get(frame, 0),
                )
                start_display_index = frames.index(max_frame)
                print(
                    "[viewer] starting on strongest A/B-change frame: "
                    f"t{max_frame:03d} "
                    f"(boundary-difference voxels="
                    f"{per_frame_change.get(max_frame, 0):,})",
                    flush=True,
                )

    viewer.dims.set_current_step(0, start_display_index)
    viewer.dims.ndisplay = 3

    print(
        "[viewer] T index -> BioHub frame: "
        + ", ".join(f"{i}->t{frame:03d}" for i, frame in enumerate(frames)),
        flush=True,
    )
    if old_partition is not None:
        print(
            "[viewer] IMPORTANT A/B layers:\n"
            "         GREEN  = NEW split boundary that did not exist in OLD\n"
            "         YELLOW = supervoxels on old-merge -> NEW-separate RAG edges\n"
            "         RED    = OLD boundary removed by a NEW merge (hidden by default)\n"
            "         CYAN   = complete final components whose SV membership changed "
            "(hidden by default)",
            flush=True,
        )
        print(
            f"[viewer] changed-edge markers: "
            f"old-merge->new-separate={len(split_change_points):,}, "
            f"old-separate->new-merge={len(merge_change_points):,}",
            flush=True,
        )
    print(
        "[viewer] For separator-visible cases, enable "
        "'STIR-Net separator probability' together with the GREEN boundary layer.",
        flush=True,
    )
    napari.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Investigation-17 morphology-aware RAG over all 20 BioHub "
            "frames and open a direct old/new full-volume Napari comparison."
        )
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stage6-root", default=None)
    parser.add_argument("--sample-zarr", default=None)
    parser.add_argument("--timepoints", default="all")
    parser.add_argument("--expected-frames", type=int, default=DEFAULT_EXPECTED_FRAMES)
    parser.add_argument(
        "--spacing",
        default=",".join(str(v) for v in DEFAULT_SPACING_ZYX_UM),
    )
    parser.add_argument("--tile-shape-zyx", default=DEFAULT_TILE_SHAPE_ZYX)
    parser.add_argument("--tile-overlap-zyx", default=DEFAULT_TILE_OVERLAP_ZYX)
    parser.add_argument("--tile-halo-zyx", default=DEFAULT_TILE_HALO_ZYX)
    parser.add_argument("--tile-batch-size", type=int, default=DEFAULT_TILE_BATCH_SIZE)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--merge-threshold", type=float, default=DEFAULT_MERGE_THRESHOLD)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--viewer-only", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-comparison", action="store_true")
    parser.add_argument("--no-source-priors", action="store_true")
    parser.add_argument("--no-geometry", action="store_true")
    parser.add_argument("--vectors", action="store_true")
    parser.add_argument("--vector-timepoint", type=int, default=0)
    parser.add_argument(
        "--vector-stride-zyx",
        default=",".join(str(v) for v in DEFAULT_VECTOR_STRIDE_ZYX),
    )
    parser.add_argument("--vector-min-magnitude", type=float, default=0.05)
    parser.add_argument("--flow-scale", type=float, default=3.0)
    parser.add_argument("--centroid-offset-scale", type=float, default=1.0)
    args = parser.parse_args()

    if not 0.0 <= args.merge_threshold <= 1.0:
        raise ValueError("--merge-threshold must be in [0,1]")

    checkpoint, checkpoint_step = resolve_morphology_checkpoint(args.checkpoint)
    output_dir = new_output_dir(args.sample_id, checkpoint_step, args.output_dir)

    baseline_dir = None
    if not args.no_comparison:
        baseline_dir = resolve_baseline_dir(args.sample_id, args.baseline_dir)
        if baseline_dir is None:
            print(
                "[A/B warning] old step-1000 result not found; continuing "
                "with NEW-only viewing.",
                flush=True,
            )

    print("=" * 118)
    print("STIR-Net Investigation 18 — morphology-RAG BioHub full-volume visualization")
    print("=" * 118)
    print("sample                    :", args.sample_id)
    print("morphology checkpoint     :", checkpoint)
    print("checkpoint global step    :", checkpoint_step)
    print("old baseline              :", baseline_dir)
    print("new output                :", output_dir)
    print("timepoints                :", args.timepoints)
    print("expected frames           :", args.expected_frames)
    print("merge threshold           :", args.merge_threshold)
    print("=" * 118)

    started = time.perf_counter()

    if not args.viewer_only:
        run_full_volume_inference(
            checkpoint=checkpoint,
            output_dir=output_dir,
            sample_id=args.sample_id,
            stage6_root=args.stage6_root,
            device=args.device,
            timepoints=args.timepoints,
            expected_frames=args.expected_frames,
            spacing=args.spacing,
            tile_shape_zyx=args.tile_shape_zyx,
            tile_overlap_zyx=args.tile_overlap_zyx,
            tile_halo_zyx=args.tile_halo_zyx,
            tile_batch_size=args.tile_batch_size,
            overwrite=args.overwrite,
            no_source_priors=args.no_source_priors,
        )

    if not output_dir.is_dir():
        raise FileNotFoundError(f"Investigation-18 output does not exist: {output_dir}")

    frames = completed_frames(output_dir)
    if not frames:
        raise FileNotFoundError(f"No completed Investigation-18 frames below {output_dir}")

    if (
        args.timepoints.strip().lower() in {"all", "*"}
        and args.expected_frames > 0
        and len(frames) != args.expected_frames
    ):
        raise RuntimeError(
            f"Expected {args.expected_frames} completed frames, found "
            f"{len(frames)}: {frames}"
        )

    if baseline_dir is not None:
        baseline_frames = set(completed_frames(baseline_dir))
        common = [frame for frame in frames if frame in baseline_frames]
        if common:
            comparison = compare_runs(
                baseline_root=baseline_dir,
                new_root=output_dir,
                frames=common,
                merge_threshold=args.merge_threshold,
            )
            print("\n" + "=" * 118)
            print("INVESTIGATION 18 A/B SUMMARY")
            print("=" * 118)
            print("watershed identical       :", comparison["watershed_equal_all_frames"])
            print("watershed mismatch frames :", comparison["watershed_mismatch_frames"])
            print("RAG topology identical    :", comparison["rag_edge_topology_equal_all_comparable_frames"])
            print("old merge -> new separate :", comparison["total_old_merge_new_separate_edges"])
            print("old separate -> new merge :", comparison["total_old_separate_new_merge_edges"])
            print("sum instance count old    :", comparison["baseline_partition_instance_count_total"])
            print("sum instance count new    :", comparison["new_partition_instance_count_total"])
            print("sum instance-count delta  :", comparison["partition_instance_count_delta_total"])
            diff = comparison.get("difference_artifacts", {})
            print("changed supervoxels total :", diff.get("changed_supervoxel_count_total"))
            print("NEW-only boundary voxels  :", diff.get("new_split_boundary_voxel_count_total"))
            print("OLD-only boundary voxels  :", diff.get("removed_old_boundary_voxel_count_total"))
            print("comparison JSON           :", output_dir / "ab_comparison.json")
            print("difference summary        :", output_dir / "ab_diff_summary.json")
            print("=" * 118)

    print(
        f"[investigation18] preparation elapsed: "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )

    if args.no_viewer:
        return

    stride = parse_zyx_positive(args.vector_stride_zyx, name="--vector-stride-zyx")
    open_ab_viewer(
        sample_id=args.sample_id,
        new_root=output_dir,
        baseline_root=baseline_dir,
        stage6_root_override=args.stage6_root,
        sample_zarr_override=args.sample_zarr,
        timepoints=args.timepoints,
        vectors=args.vectors,
        vector_timepoint=args.vector_timepoint,
        vector_stride_zyx=stride,
        vector_min_magnitude=args.vector_min_magnitude,
        flow_scale=args.flow_scale,
        centroid_offset_scale=args.centroid_offset_scale,
        no_geometry=args.no_geometry,
        merge_threshold=args.merge_threshold,
    )


if __name__ == "__main__":
    main()
