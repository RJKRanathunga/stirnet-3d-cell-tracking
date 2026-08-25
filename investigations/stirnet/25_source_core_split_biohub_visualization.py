from __future__ import annotations

r"""
Investigation 25 — BioHub source-core split-only postfilter visualization.

Purpose
-------
Visualize the NEW asymmetric source-core split-only filter after the spatial
graph has already been solved with production multicut.

The comparison is:

    BEFORE
        h100 morphology RAG
        -> production signed multicut q=0.845

    AFTER
        exactly the same multicut partition
        -> source-core split-only final filter

No dense network inference is rerun.
No RAG network inference is rerun.
No multicut solve is rerun by this script.
No source evidence is ever fed back into graph reasoning.

This is intentionally an inference/postprocessing A/B experiment.

Default inputs
--------------
Multicut result from Investigation 24:

    runs/stirnet/evaluation/
        24_multicut_biohub_full_volume_visualization/
        44b6_0113de3b/
        h100_q0p845/

Historical h100 result, used ONLY for learned separator probability:

    runs/stirnet/evaluation/
        12_biohub_full_volume_spatial_inference/
        44b6_0113de3b/
        morphology_v2_h100/
        step000600/

Stage-6 data are discovered through PipelinePaths and supply:
    * preprocessing
    * original bright binary mask
    * original source connected-component segmentation

What Napari shows
-----------------
Core A/B layers:

    BEFORE multicut partition
        Spatial graph output before source-core split-only filtering.

    AFTER source-core split-only partition
        Final labels after applying the asymmetric postfilter.

    SPLIT-ONLY new instances ONLY
        Label-colored children created by applied split requests.
        Unchanged cells are transparent.

    GREEN — NEW split boundaries
        Boundary voxels introduced by the split-only filter.

    CYAN — applied candidate components
        Whole pre-filter multicut components that were accepted for splitting.

    YELLOW — rejected candidate components
        Components considered by the source-core logic but not split.

    Stage-6 source segmentation
        Original small bright connected components / "cores".

    Watershed supervoxels
        The exact supervoxels passed into graph reasoning / multicut.

    Separator probability
        Learned dense separator evidence used both to guide the split surface
        and to boost split confidence.

Point layers
------------
Each candidate component gets a point at its pre-filter component centroid.

Applied and rejected candidates are separate point layers. Napari properties
include:
    frame
    final_component_id
    source_core_count
    split_confidence
    source_score
    separator_score
    volume_ratio
    minimum_core_separation_dref
    status

This makes it easy to click a candidate and inspect WHY it was accepted/rejected.

Important scientific invariant
------------------------------
The postfilter is split-only. This script verifies per frame that no output
instance contains voxels from two different pre-filter multicut instances.

Typical commands
----------------
Run all 20 frames and open Napari:

    python .\investigations\stirnet\25_source_core_split_biohub_visualization.py

Generate artifacts only:

    python .\investigations\stirnet\25_source_core_split_biohub_visualization.py --no-viewer

Re-open existing artifacts:

    python .\investigations\stirnet\25_source_core_split_biohub_visualization.py --viewer-only

Inspect a subset:

    python .\investigations\stirnet\25_source_core_split_biohub_visualization.py --timepoints 0-4

Temporarily inspect a different confidence threshold WITHOUT changing production config:

    python .\investigations\stirnet\25_source_core_split_biohub_visualization.py `
        --confidence-threshold 0.72

Output
------
    runs/stirnet/evaluation/
        25_source_core_split_biohub_visualization/
        44b6_0113de3b/
        production_defaults/

or a threshold-specific directory when CLI overrides are used.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy import ndimage as ndi


SCRIPT_NAME = "25_source_core_split_biohub_visualization"
DEFAULT_SAMPLE = "44b6_0113de3b"
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)


# ======================================================================================
# Repository / generic helpers
# ======================================================================================


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (
            (candidate / "learned").is_dir()
            and (candidate / "src").is_dir()
            and (candidate / "investigations").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (
            (candidate / "learned").is_dir()
            and (candidate / "src").is_dir()
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


def _load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True))
        handle.write("\n")


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(temporary, np.asarray(array), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _positive_label_count(labels: np.ndarray) -> int:
    values = np.unique(labels)
    return int(np.count_nonzero(values > 0))


def _internal_boundary(labels: np.ndarray) -> np.ndarray:
    """Return both voxel sides of internal positive-label boundaries."""
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError(f"Expected [Z,Y,X] labels, got {labels.shape}")

    boundary = np.zeros(labels.shape, dtype=bool)
    for axis in range(3):
        lower_slice = [slice(None)] * 3
        upper_slice = [slice(None)] * 3
        lower_slice[axis] = slice(0, -1)
        upper_slice[axis] = slice(1, None)
        lower_slice = tuple(lower_slice)
        upper_slice = tuple(upper_slice)

        lower = labels[lower_slice]
        upper = labels[upper_slice]
        changed = (
            (lower > 0)
            & (upper > 0)
            & (lower != upper)
        )
        boundary[lower_slice] |= changed
        boundary[upper_slice] |= changed
    return boundary


def _verify_strict_refinement(before: np.ndarray, after: np.ndarray) -> None:
    """Every AFTER instance must have exactly one BEFORE parent."""
    positive = after > 0
    if not bool(positive.any()):
        return

    pairs = np.unique(
        np.stack(
            [
                after[positive].astype(np.int64, copy=False),
                before[positive].astype(np.int64, copy=False),
            ],
            axis=1,
        ),
        axis=0,
    )
    parent: dict[int, int] = {}
    for new_id, old_id in pairs.tolist():
        new_id = int(new_id)
        old_id = int(old_id)
        previous = parent.get(new_id)
        if previous is not None and previous != old_id:
            raise RuntimeError(
                "Split-only invariant violated: AFTER component "
                f"{new_id} contains BEFORE components {previous} and {old_id}"
            )
        parent[new_id] = old_id


def _component_centroid_voxel(
    labels: np.ndarray,
    component_id: int,
) -> np.ndarray:
    coordinates = np.argwhere(labels == int(component_id))
    if coordinates.size == 0:
        return np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
    return coordinates.mean(axis=0).astype(np.float32)


def _paint_component_mask(
    labels: np.ndarray,
    component_ids: Iterable[int],
) -> np.ndarray:
    ids = np.asarray(sorted(set(int(value) for value in component_ids)), dtype=np.int64)
    if ids.size == 0:
        return np.zeros(labels.shape, dtype=np.uint8)
    return np.isin(labels, ids).astype(np.uint8)


def _paint_after_children_only(
    before: np.ndarray,
    after: np.ndarray,
    applied_component_ids: Iterable[int],
) -> np.ndarray:
    """Keep AFTER labels only inside BEFORE components where a split was applied."""
    mask = _paint_component_mask(before, applied_component_ids).astype(bool)
    return np.where(mask, after, 0).astype(np.int32, copy=False)


def _safe_float(record: dict[str, Any], key: str, default: float = -1.0) -> float:
    value = record.get(key)
    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _property_rows(
    records: list[dict[str, Any]],
    *,
    frame: int,
) -> dict[str, np.ndarray]:
    return {
        "frame": np.asarray([int(frame)] * len(records), dtype=np.int32),
        "final_component_id": np.asarray(
            [int(row.get("final_component_id", -1)) for row in records],
            dtype=np.int32,
        ),
        "source_core_count": np.asarray(
            [int(row.get("source_core_count", 0)) for row in records],
            dtype=np.int32,
        ),
        "split_confidence": np.asarray(
            [_safe_float(row, "split_confidence") for row in records],
            dtype=np.float32,
        ),
        "source_score": np.asarray(
            [_safe_float(row, "source_score") for row in records],
            dtype=np.float32,
        ),
        "separator_score": np.asarray(
            [_safe_float(row, "separator_score") for row in records],
            dtype=np.float32,
        ),
        "volume_ratio": np.asarray(
            [
                _safe_float(
                    row,
                    "component_volume_ratio_to_single_core_median",
                )
                for row in records
            ],
            dtype=np.float32,
        ),
        "minimum_core_separation_dref": np.asarray(
            [_safe_float(row, "minimum_core_separation_dref") for row in records],
            dtype=np.float32,
        ),
        "status": np.asarray(
            [str(row.get("status", "unknown")) for row in records],
            dtype=object,
        ),
    }


def _concat_properties(
    rows: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not rows:
        return {
            "frame": np.zeros(0, dtype=np.int32),
            "final_component_id": np.zeros(0, dtype=np.int32),
            "source_core_count": np.zeros(0, dtype=np.int32),
            "split_confidence": np.zeros(0, dtype=np.float32),
            "source_score": np.zeros(0, dtype=np.float32),
            "separator_score": np.zeros(0, dtype=np.float32),
            "volume_ratio": np.zeros(0, dtype=np.float32),
            "minimum_core_separation_dref": np.zeros(0, dtype=np.float32),
            "status": np.zeros(0, dtype=object),
        }
    keys = rows[0].keys()
    return {
        key: np.concatenate([row[key] for row in rows], axis=0)
        for key in keys
    }


def default_multicut_root(sample_id: str) -> Path:
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / "24_multicut_biohub_full_volume_visualization"
        / sample_id
        / "h100_q0p845"
    ).resolve()


def default_h100_root(sample_id: str) -> Path:
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / "12_biohub_full_volume_spatial_inference"
        / sample_id
        / "morphology_v2_h100"
        / "step000600"
    ).resolve()


def output_variant_name(
    *,
    confidence_threshold: float | None,
    separator_boost: float | None,
) -> str:
    if confidence_threshold is None and separator_boost is None:
        return "supervoxel_graph_defaults"

    pieces = []
    if confidence_threshold is not None:
        pieces.append(
            "conf"
            + f"{confidence_threshold:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        )
    if separator_boost is not None:
        pieces.append(
            "sepboost"
            + f"{separator_boost:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        )
    return "_".join(pieces)


def default_output_root(
    sample_id: str,
    *,
    confidence_threshold: float | None,
    separator_boost: float | None,
) -> Path:
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / SCRIPT_NAME
        / sample_id
        / output_variant_name(
            confidence_threshold=confidence_threshold,
            separator_boost=separator_boost,
        )
    ).resolve()


# ======================================================================================
# Frame materialization
# ======================================================================================


def materialize_frame(
    *,
    frame: int,
    displayed_time_index: int,
    multicut_root: Path,
    h100_root: Path,
    stage6_root: Path,
    output_root: Path,
    spacing_zyx_um: tuple[float, float, float],
    confidence_threshold_override: float | None,
    separator_boost_override: float | None,
    overwrite: bool,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
]:
    from learned.stirnet.data.targets import estimate_model_dref_um
    from learned.stirnet.model.config import InferenceConfig
    from learned.stirnet.model.postprocess.source_core_split import (
        SourceCoreSplitOnlyFilter,
    )

    output_frame = output_root / f"t{frame:03d}"
    success_path = output_frame / "_SUCCESS.json"

    if success_path.is_file() and not overwrite:
        row = json.loads(success_path.read_text(encoding="utf-8"))
        candidate_path = output_frame / "candidate_points.npz"
        with np.load(candidate_path, allow_pickle=True) as payload:
            applied_points = np.asarray(payload["applied_points"], dtype=np.float32)
            rejected_points = np.asarray(payload["rejected_points"], dtype=np.float32)

            applied_properties = {}
            rejected_properties = {}
            for key in (
                "frame",
                "final_component_id",
                "source_core_count",
                "split_confidence",
                "source_score",
                "separator_score",
                "volume_ratio",
                "minimum_core_separation_dref",
                "status",
            ):
                applied_properties[key] = np.asarray(payload[f"applied_{key}"])
                rejected_properties[key] = np.asarray(payload[f"rejected_{key}"])
        return (
            row,
            applied_points,
            applied_properties,
            rejected_points,
            rejected_properties,
        )

    before_path = (
        multicut_root
        / f"t{frame:03d}"
        / "partition"
        / "spatial_partition.npy"
    )
    watershed_path = (
        multicut_root
        / f"t{frame:03d}"
        / "partition"
        / "watershed_supervoxels.npy"
    )
    separator_path = (
        h100_root
        / f"t{frame:03d}"
        / "geometry"
        / "separator_probability.npy"
    )
    source_mask_path = stage6_root / "masking" / f"t{frame:03d}.npy"
    source_segmentation_path = (
        stage6_root / "segmentation" / f"t{frame:03d}.npy"
    )

    for required in (
        before_path,
        watershed_path,
        separator_path,
        source_mask_path,
        source_segmentation_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    before = np.asarray(
        np.load(before_path, mmap_mode="r", allow_pickle=False),
        dtype=np.int32,
    )
    watershed_supervoxels = np.asarray(
        np.load(watershed_path, mmap_mode="r", allow_pickle=False),
        dtype=np.int64,
    )
    separator = np.asarray(
        np.load(separator_path, mmap_mode="r", allow_pickle=False),
        dtype=np.float32,
    )
    source_mask = np.asarray(
        np.load(source_mask_path, mmap_mode="r", allow_pickle=False),
    )
    source_segmentation = np.asarray(
        np.load(
            source_segmentation_path,
            mmap_mode="r",
            allow_pickle=False,
        ),
        dtype=np.int64,
    )

    if (
        before.shape != watershed_supervoxels.shape
        or before.shape != separator.shape
        or before.shape != source_mask.shape
        or before.shape != source_segmentation.shape
    ):
        raise ValueError(
            f"t{frame:03d}: shape mismatch: "
            f"before={before.shape}, separator={separator.shape}, "
            f"source_mask={source_mask.shape}, source_seg={source_segmentation.shape}"
        )

    spacing = tuple(float(value) for value in spacing_zyx_um)
    dref_um = float(
        estimate_model_dref_um(source_segmentation, spacing)
    )

    cfg = InferenceConfig()
    if confidence_threshold_override is not None:
        cfg.source_core_split_confidence_threshold = float(
            confidence_threshold_override
        )
    if separator_boost_override is not None:
        cfg.source_core_split_separator_boost = float(
            separator_boost_override
        )

    filt = SourceCoreSplitOnlyFilter(cfg)

    started = time.perf_counter()
    state = filt(
        [torch.as_tensor(before, dtype=torch.long)],
        torch.as_tensor(
            (source_mask > 0)[None],
            dtype=torch.float32,
        ),
        torch.as_tensor(
            separator[None],
            dtype=torch.float32,
        ),
        torch.tensor([spacing], dtype=torch.float32),
        torch.tensor([dref_um], dtype=torch.float32),
        supervoxel_labels=[
            torch.as_tensor(watershed_supervoxels, dtype=torch.long)
        ],
    )
    elapsed = time.perf_counter() - started

    after = state.labels[0].detach().cpu().numpy().astype(
        np.int32,
        copy=False,
    )
    _verify_strict_refinement(before, after)

    before_boundary = _internal_boundary(before)
    after_boundary = _internal_boundary(after)
    new_boundary = after_boundary & ~before_boundary
    removed_boundary = before_boundary & ~after_boundary

    # A correct split-only operation must never remove an existing internal
    # boundary between distinct BEFORE instances.
    removed_count = int(np.count_nonzero(removed_boundary))
    if removed_count:
        raise RuntimeError(
            f"t{frame:03d}: split-only invariant violation: "
            f"{removed_count} old boundary voxels were removed"
        )

    records = [dict(row) for row in state.records]
    applied_records = [row for row in records if bool(row.get("applied", False))]
    rejected_records = [row for row in records if not bool(row.get("applied", False))]

    applied_ids = {
        int(row["final_component_id"])
        for row in applied_records
        if "final_component_id" in row
    }
    rejected_ids = {
        int(row["final_component_id"])
        for row in rejected_records
        if "final_component_id" in row
    }

    applied_component_mask = _paint_component_mask(before, applied_ids)
    rejected_component_mask = _paint_component_mask(before, rejected_ids)
    split_children = _paint_after_children_only(
        before,
        after,
        applied_ids,
    )

    applied_points_rows = []
    for row in applied_records:
        centroid = _component_centroid_voxel(
            before,
            int(row["final_component_id"]),
        )
        if np.all(np.isfinite(centroid)):
            applied_points_rows.append(
                np.asarray(
                    [float(displayed_time_index), *centroid.tolist()],
                    dtype=np.float32,
                )
            )

    rejected_points_rows = []
    for row in rejected_records:
        if "final_component_id" not in row:
            continue
        centroid = _component_centroid_voxel(
            before,
            int(row["final_component_id"]),
        )
        if np.all(np.isfinite(centroid)):
            rejected_points_rows.append(
                np.asarray(
                    [float(displayed_time_index), *centroid.tolist()],
                    dtype=np.float32,
                )
            )

    applied_points = (
        np.stack(applied_points_rows, axis=0)
        if applied_points_rows
        else np.zeros((0, 4), dtype=np.float32)
    )
    rejected_points = (
        np.stack(rejected_points_rows, axis=0)
        if rejected_points_rows
        else np.zeros((0, 4), dtype=np.float32)
    )

    # Properties must align with the points. All applied records necessarily
    # carry final_component_id. Rejected rows generated by the production filter
    # also do, but filter explicitly to stay robust to future diagnostics.
    applied_point_records = [
        row
        for row in applied_records
        if "final_component_id" in row
    ]
    rejected_point_records = [
        row
        for row in rejected_records
        if "final_component_id" in row
    ]
    applied_properties = _property_rows(
        applied_point_records,
        frame=frame,
    )
    rejected_properties = _property_rows(
        rejected_point_records,
        frame=frame,
    )

    if len(applied_points) != len(applied_properties["frame"]):
        raise RuntimeError("Applied candidate point/property alignment failed")
    if len(rejected_points) != len(rejected_properties["frame"]):
        raise RuntimeError("Rejected candidate point/property alignment failed")

    output_frame.mkdir(parents=True, exist_ok=True)
    atomic_npy(
        output_frame / "partition/before_multicut.npy",
        before,
    )
    atomic_npy(
        output_frame / "partition/after_split_only.npy",
        after,
    )
    atomic_npy(
        output_frame / "diff/new_split_boundaries.npy",
        new_boundary.astype(np.uint8),
    )
    atomic_npy(
        output_frame / "diff/applied_candidate_components.npy",
        applied_component_mask,
    )
    atomic_npy(
        output_frame / "diff/rejected_candidate_components.npy",
        rejected_component_mask,
    )
    atomic_npy(
        output_frame / "diff/split_children_only.npy",
        split_children,
    )

    # Candidate point data are small and self-contained.
    candidate_payload: dict[str, Any] = {
        "applied_points": applied_points,
        "rejected_points": rejected_points,
    }
    for key, value in applied_properties.items():
        candidate_payload[f"applied_{key}"] = value
    for key, value in rejected_properties.items():
        candidate_payload[f"rejected_{key}"] = value
    np.savez_compressed(
        output_frame / "candidate_points.npz",
        **candidate_payload,
    )

    atomic_json(
        output_frame / "records.json",
        {
            "timepoint": int(frame),
            "dref_um": float(dref_um),
            "records": records,
        },
    )

    before_count = _positive_label_count(before)
    after_count = _positive_label_count(after)

    applied_confidence = [
        _safe_float(row, "split_confidence", default=float("nan"))
        for row in applied_records
        if math.isfinite(
            _safe_float(row, "split_confidence", default=float("nan"))
        )
    ]
    rejected_confidence = [
        _safe_float(row, "split_confidence", default=float("nan"))
        for row in rejected_records
        if math.isfinite(
            _safe_float(row, "split_confidence", default=float("nan"))
        )
    ]

    row = {
        "timepoint": int(frame),
        "dref_um": float(dref_um),
        "before_multicut_instance_count": int(before_count),
        "after_split_only_instance_count": int(after_count),
        "instance_delta_after_minus_before": int(after_count - before_count),
        "candidate_count": int(state.candidate_count),
        "applied_count": int(state.applied_count),
        "rejected_record_count": int(len(rejected_records)),
        "skipped_too_many_cores": int(state.skipped_too_many_cores),
        "new_split_boundary_voxel_count": int(np.count_nonzero(new_boundary)),
        "removed_old_boundary_voxel_count": int(removed_count),
        "applied_component_count": int(len(applied_ids)),
        "rejected_component_count": int(len(rejected_ids)),
        "applied_confidence_mean": (
            float(np.mean(applied_confidence))
            if applied_confidence
            else None
        ),
        "applied_confidence_min": (
            float(np.min(applied_confidence))
            if applied_confidence
            else None
        ),
        "rejected_confidence_mean": (
            float(np.mean(rejected_confidence))
            if rejected_confidence
            else None
        ),
        "rejected_confidence_max": (
            float(np.max(rejected_confidence))
            if rejected_confidence
            else None
        ),
        "filter_seconds": float(elapsed),
        "configuration": {
            "confidence_threshold": float(
                cfg.source_core_split_confidence_threshold
            ),
            "separator_boost": float(
                cfg.source_core_split_separator_boost
            ),
            "foreground_threshold": float(
                cfg.source_core_split_foreground_threshold
            ),
            "min_core_voxels": int(
                cfg.source_core_split_min_core_voxels
            ),
            "min_core_containment": float(
                cfg.source_core_split_min_core_containment
            ),
            "min_core_separation_dref": float(
                cfg.source_core_split_min_core_separation_dref
            ),
            "min_child_fraction": float(
                cfg.source_core_split_min_child_fraction
            ),
            "volume_ratio_center": float(
                cfg.source_core_split_volume_ratio_center
            ),
            "volume_ratio_softness": float(
                cfg.source_core_split_volume_ratio_softness
            ),
            "separator_support_threshold": float(
                cfg.source_core_split_separator_support_threshold
            ),
        },
    }
    atomic_json(success_path, row)

    return (
        row,
        applied_points,
        applied_properties,
        rejected_points,
        rejected_properties,
    )


def materialize_run(
    *,
    sample_id: str,
    multicut_root: Path,
    h100_root: Path,
    stage6_root: Path,
    output_root: Path,
    frames: list[int],
    spacing_zyx_um: tuple[float, float, float],
    confidence_threshold_override: float | None,
    separator_boost_override: float | None,
    overwrite: bool,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    per_frame_path = output_root / "per_frame.jsonl"
    per_frame_path.unlink(missing_ok=True)

    all_rows = []
    applied_point_rows = []
    rejected_point_rows = []
    applied_property_rows = []
    rejected_property_rows = []

    print("=" * 120)
    print("STIR-Net Investigation 25 — source-core split-only BioHub visualization")
    print("=" * 120)
    print("sample                :", sample_id)
    print("BEFORE multicut       :", multicut_root)
    print("separator source      :", h100_root)
    print("Stage-6 source priors :", stage6_root)
    print("output                :", output_root)
    print("frames                :", frames)
    print("confidence override   :", confidence_threshold_override)
    print("separator boost over. :", separator_boost_override)
    print("dense model rerun     : NO")
    print("RAG model rerun       : NO")
    print("multicut rerun        : NO")
    print("postfilter action     : SPLIT ONLY")
    print("=" * 120, flush=True)

    for displayed_index, frame in enumerate(frames):
        (
            row,
            applied_points,
            applied_properties,
            rejected_points,
            rejected_properties,
        ) = materialize_frame(
            frame=frame,
            displayed_time_index=displayed_index,
            multicut_root=multicut_root,
            h100_root=h100_root,
            stage6_root=stage6_root,
            output_root=output_root,
            spacing_zyx_um=spacing_zyx_um,
            confidence_threshold_override=confidence_threshold_override,
            separator_boost_override=separator_boost_override,
            overwrite=overwrite,
        )
        all_rows.append(row)
        append_jsonl(per_frame_path, row)

        if applied_points.size:
            applied_point_rows.append(applied_points)
            applied_property_rows.append(applied_properties)
        if rejected_points.size:
            rejected_point_rows.append(rejected_points)
            rejected_property_rows.append(rejected_properties)

        print(
            f"[t{frame:03d}] "
            f"instances {row['before_multicut_instance_count']} -> "
            f"{row['after_split_only_instance_count']} "
            f"(+{row['instance_delta_after_minus_before']}) | "
            f"candidates={row['candidate_count']} "
            f"applied={row['applied_count']} "
            f"rejected={row['rejected_record_count']} | "
            f"new-boundary={row['new_split_boundary_voxel_count']:,} | "
            f"{row['filter_seconds']:.3f}s",
            flush=True,
        )

    applied_points = (
        np.concatenate(applied_point_rows, axis=0)
        if applied_point_rows
        else np.zeros((0, 4), dtype=np.float32)
    )
    rejected_points = (
        np.concatenate(rejected_point_rows, axis=0)
        if rejected_point_rows
        else np.zeros((0, 4), dtype=np.float32)
    )
    applied_properties = _concat_properties(applied_property_rows)
    rejected_properties = _concat_properties(rejected_property_rows)

    point_payload: dict[str, Any] = {
        "applied_points": applied_points,
        "rejected_points": rejected_points,
    }
    for key, value in applied_properties.items():
        point_payload[f"applied_{key}"] = value
    for key, value in rejected_properties.items():
        point_payload[f"rejected_{key}"] = value
    np.savez_compressed(output_root / "candidate_points.npz", **point_payload)

    summary = {
        "status": "success",
        "experiment": SCRIPT_NAME,
        "sample_id": sample_id,
        "frames": frames,
        "frame_count": len(frames),
        "before_multicut_root": str(multicut_root),
        "separator_source_root": str(h100_root),
        "stage6_root": str(stage6_root),
        "output_root": str(output_root),
        "comparison_invariant": (
            "same multicut partition; source-core split-only postfilter only"
        ),
        "before_instance_count_total": int(
            sum(row["before_multicut_instance_count"] for row in all_rows)
        ),
        "after_instance_count_total": int(
            sum(row["after_split_only_instance_count"] for row in all_rows)
        ),
        "instance_delta_total": int(
            sum(row["instance_delta_after_minus_before"] for row in all_rows)
        ),
        "candidate_count_total": int(
            sum(row["candidate_count"] for row in all_rows)
        ),
        "applied_count_total": int(
            sum(row["applied_count"] for row in all_rows)
        ),
        "rejected_record_count_total": int(
            sum(row["rejected_record_count"] for row in all_rows)
        ),
        "new_split_boundary_voxel_count_total": int(
            sum(row["new_split_boundary_voxel_count"] for row in all_rows)
        ),
        "removed_old_boundary_voxel_count_total": int(
            sum(row["removed_old_boundary_voxel_count"] for row in all_rows)
        ),
        "filter_seconds_total": float(
            sum(row["filter_seconds"] for row in all_rows)
        ),
        "per_frame": all_rows,
    }
    atomic_json(output_root / "summary.json", summary)

    print("=" * 120)
    print("INVESTIGATION 25 SUMMARY")
    print("=" * 120)
    print(
        "instances total      :",
        summary["before_instance_count_total"],
        "->",
        summary["after_instance_count_total"],
        f"(delta={summary['instance_delta_total']:+d})",
    )
    print("candidate requests    :", summary["candidate_count_total"])
    print("applied split requests:", summary["applied_count_total"])
    print("rejected records      :", summary["rejected_record_count_total"])
    print(
        "new boundary voxels  :",
        f"{summary['new_split_boundary_voxel_count_total']:,}",
    )
    print(
        "removed boundaries   :",
        summary["removed_old_boundary_voxel_count_total"],
        "(must remain 0)",
    )
    print(
        "filter time total    :",
        f"{summary['filter_seconds_total']:.3f}s",
    )
    print("summary               :", output_root / "summary.json")
    print("=" * 120)

    return summary


# ======================================================================================
# Napari
# ======================================================================================


def _stack_npy(
    V13,
    root: Path,
    frames: list[int],
    relative: str,
    name: str,
):
    return V13.stack_npy(
        [
            root / f"t{frame:03d}" / relative
            for frame in frames
        ],
        name=name,
    )[0]


def _load_point_payload(output_root: Path):
    path = output_root / "candidate_points.npz"
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as payload:
        applied_points = np.asarray(
            payload["applied_points"],
            dtype=np.float32,
        )
        rejected_points = np.asarray(
            payload["rejected_points"],
            dtype=np.float32,
        )

        keys = (
            "frame",
            "final_component_id",
            "source_core_count",
            "split_confidence",
            "source_score",
            "separator_score",
            "volume_ratio",
            "minimum_core_separation_dref",
            "status",
        )
        applied_properties = {
            key: np.asarray(payload[f"applied_{key}"])
            for key in keys
        }
        rejected_properties = {
            key: np.asarray(payload[f"rejected_{key}"])
            for key in keys
        }

    return (
        applied_points,
        applied_properties,
        rejected_points,
        rejected_properties,
    )


def open_viewer(
    *,
    sample_id: str,
    multicut_root: Path,
    h100_root: Path,
    stage6_root: Path,
    output_root: Path,
    frames: list[int],
    sample_zarr_override: str | None,
    spacing_zyx_um: tuple[float, float, float],
) -> None:
    viewer_script = (
        ROOT
        / "investigations"
        / "stirnet"
        / "data"
        / "13_biohub_full_volume_spatial_results_viewer.py"
    )
    V13 = _load_module(
        viewer_script,
        "_investigation13_for_investigation25",
    )

    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is not installed. Run with --no-viewer or install "
            "the visualization dependencies."
        ) from exc

    sample_zarr = V13.resolve_sample_zarr(
        sample_id,
        sample_zarr_override,
    )
    scale_4d = (1.0, *spacing_zyx_um)

    raw, _ = V13.load_raw_time_series(sample_zarr, frames)
    preprocessed, _ = V13.stack_npy(
        V13.stage6_paths(stage6_root, frames, "preprocessing"),
        name="Stage-6 preprocessing",
    )
    source_mask, _ = V13.stack_npy(
        V13.stage6_paths(stage6_root, frames, "masking"),
        name="Stage-6 bright binary mask",
    )
    source_segmentation, _ = V13.stack_npy(
        V13.stage6_paths(stage6_root, frames, "segmentation"),
        name="Stage-6 source segmentation",
    )
    watershed_supervoxels = _stack_npy(
        V13,
        multicut_root,
        frames,
        "partition/watershed_supervoxels.npy",
        "watershed supervoxels",
    )

    before = _stack_npy(
        V13,
        output_root,
        frames,
        "partition/before_multicut.npy",
        "BEFORE multicut partition",
    )
    after = _stack_npy(
        V13,
        output_root,
        frames,
        "partition/after_split_only.npy",
        "AFTER split-only partition",
    )
    new_boundaries = _stack_npy(
        V13,
        output_root,
        frames,
        "diff/new_split_boundaries.npy",
        "split-only new boundaries",
    )
    applied_components = _stack_npy(
        V13,
        output_root,
        frames,
        "diff/applied_candidate_components.npy",
        "applied candidate components",
    )
    rejected_components = _stack_npy(
        V13,
        output_root,
        frames,
        "diff/rejected_candidate_components.npy",
        "rejected candidate components",
    )
    split_children = _stack_npy(
        V13,
        output_root,
        frames,
        "diff/split_children_only.npy",
        "split-only child instances",
    )

    separator_probability, _ = V13.stack_npy(
        V13.inference_paths(
            h100_root,
            frames,
            "geometry/separator_probability.npy",
        ),
        name="separator probability",
    )

    (
        applied_points,
        applied_properties,
        rejected_points,
        rejected_properties,
    ) = _load_point_payload(output_root)

    viewer = napari.Viewer(
        title=(
            f"STIR-Net Source-Core Split-Only A/B | {sample_id}"
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
        name="Stage-6 bright binary mask",
        scale=scale_4d,
        opacity=0.22,
        visible=False,
    )
    viewer.add_labels(
        source_segmentation,
        name="Stage-6 source bright cores",
        scale=scale_4d,
        opacity=0.70,
        visible=False,
    )
    viewer.add_labels(
        watershed_supervoxels,
        name="Watershed supervoxels",
        scale=scale_4d,
        opacity=0.55,
        visible=False,
    )

    viewer.add_labels(
        before,
        name="BEFORE: h100 + multicut",
        scale=scale_4d,
        opacity=0.55,
        visible=False,
    )
    viewer.add_labels(
        after,
        name="AFTER: + source-core split-only filter",
        scale=scale_4d,
        opacity=0.46,
        visible=True,
    )

    # Most useful layer: only the newly split components remain visible and
    # each postfilter child retains its label color.
    viewer.add_labels(
        split_children,
        name="SPLIT-ONLY new instances ONLY",
        scale=scale_4d,
        opacity=0.82,
        visible=True,
    )

    viewer.add_image(
        new_boundaries,
        name="SPLIT-ONLY NEW boundaries (green)",
        scale=scale_4d,
        colormap="green",
        contrast_limits=(0.0, 1.0),
        opacity=1.0,
        blending="additive",
        visible=True,
    )
    viewer.add_image(
        applied_components,
        name="Applied split candidate components (cyan)",
        scale=scale_4d,
        colormap="cyan",
        contrast_limits=(0.0, 1.0),
        opacity=0.20,
        blending="additive",
        visible=False,
    )
    viewer.add_image(
        rejected_components,
        name="Rejected split candidate components (yellow)",
        scale=scale_4d,
        colormap="yellow",
        contrast_limits=(0.0, 1.0),
        opacity=0.20,
        blending="additive",
        visible=False,
    )

    viewer.add_image(
        separator_probability,
        name="Learned separator probability",
        scale=scale_4d,
        colormap="yellow",
        contrast_limits=(0.0, 1.0),
        opacity=0.75,
        blending="additive",
        visible=False,
    )

    if applied_points.size:
        viewer.add_points(
            applied_points,
            name="Applied split candidates",
            scale=scale_4d,
            size=7.0,
            face_color="green",
            border_color="black",
            opacity=0.95,
            properties=applied_properties,
            visible=True,
        )

    if rejected_points.size:
        viewer.add_points(
            rejected_points,
            name="Rejected split candidates",
            scale=scale_4d,
            size=6.0,
            face_color="yellow",
            border_color="black",
            opacity=0.90,
            properties=rejected_properties,
            visible=True,
        )

    summary = json.loads(
        (output_root / "summary.json").read_text(encoding="utf-8")
    )
    change_by_frame = {
        int(row["timepoint"]): (
            int(row["applied_count"]),
            int(row["new_split_boundary_voxel_count"]),
            int(row["instance_delta_after_minus_before"]),
        )
        for row in summary["per_frame"]
    }

    strongest_frame = max(
        frames,
        key=lambda frame: change_by_frame.get(frame, (0, 0, 0)),
    )
    start_index = frames.index(strongest_frame)

    viewer.dims.set_current_step(0, start_index)
    viewer.dims.ndisplay = 3

    print(
        "[viewer] starting on strongest split-only frame: "
        f"t{strongest_frame:03d}; "
        f"applied={change_by_frame[strongest_frame][0]}, "
        f"new-boundary={change_by_frame[strongest_frame][1]:,}, "
        f"instance-delta={change_by_frame[strongest_frame][2]:+d}",
        flush=True,
    )
    print(
        "[viewer] IMPORTANT LAYERS:\n"
        "  AFTER: + source-core split-only filter\n"
        "      complete postfilter instance result.\n"
        "  SPLIT-ONLY new instances ONLY\n"
        "      label-colored children created only by accepted split requests.\n"
        "  GREEN boundaries\n"
        "      exact internal boundaries introduced by the postfilter.\n"
        "  Stage-6 source bright cores\n"
        "      original small bright connected components used as split evidence.\n"
        "  Watershed supervoxels\n"
        "      exact supervoxels entering graph reasoning / multicut.\n"
        "  Learned separator probability\n"
        "      independent learned geometry used to guide and boost the split.\n"
        "  Applied / Rejected candidate points\n"
        "      click a point and inspect split_confidence, source_score, "
        "separator_score, volume_ratio and core separation.",
        flush=True,
    )
    print(
        "[viewer] T index -> BioHub frame: "
        + ", ".join(
            f"{index}->t{frame:03d}"
            for index, frame in enumerate(frames)
        ),
        flush=True,
    )

    napari.run()


# ======================================================================================
# CLI
# ======================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply and visualize the production source-core split-only filter "
            "on top of Investigation-24 BioHub multicut partitions."
        )
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    parser.add_argument("--multicut-dir", default=None)
    parser.add_argument("--h100-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stage6-root", default=None)
    parser.add_argument("--sample-zarr", default=None)
    parser.add_argument("--timepoints", default="all")
    parser.add_argument(
        "--spacing",
        default=",".join(str(value) for value in DEFAULT_SPACING_ZYX_UM),
        help="Physical spacing in Z,Y,X um.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help=(
            "Optional Investigation-only override. Omit to use the current "
            "production InferenceConfig value."
        ),
    )
    parser.add_argument(
        "--separator-boost",
        type=float,
        default=None,
        help=(
            "Optional Investigation-only override. Omit to use the current "
            "production InferenceConfig value."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--viewer-only", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    if (
        args.confidence_threshold is not None
        and not 0.0 <= args.confidence_threshold <= 1.0
    ):
        raise ValueError("--confidence-threshold must be in [0,1]")
    if (
        args.separator_boost is not None
        and not 0.0 <= args.separator_boost <= 1.0
    ):
        raise ValueError("--separator-boost must be in [0,1]")

    spacing = tuple(
        float(token.strip()) for token in args.spacing.split(",")
    )
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError("--spacing must contain three positive Z,Y,X values")

    multicut_root = (
        resolve(args.multicut_dir)
        if args.multicut_dir
        else default_multicut_root(args.sample_id)
    )
    h100_root = (
        resolve(args.h100_dir)
        if args.h100_dir
        else default_h100_root(args.sample_id)
    )

    if not multicut_root.is_dir():
        raise FileNotFoundError(
            f"Investigation-24 multicut output not found: {multicut_root}\n"
            "Run Investigation 24 first or pass --multicut-dir."
        )
    if not h100_root.is_dir():
        raise FileNotFoundError(
            f"h100 geometry output not found: {h100_root}\n"
            "Pass --h100-dir explicitly."
        )

    viewer_script = (
        ROOT
        / "investigations"
        / "stirnet"
        / "data"
        / "13_biohub_full_volume_spatial_results_viewer.py"
    )
    V13 = _load_module(
        viewer_script,
        "_investigation13_paths_for_investigation25",
    )
    stage6_root = V13.resolve_stage6_root(
        args.sample_id,
        args.stage6_root,
    )

    available = V13.completed_timepoints(multicut_root)
    frames = V13.parse_timepoints(args.timepoints, available)

    output_root = (
        resolve(args.output_dir)
        if args.output_dir
        else default_output_root(
            args.sample_id,
            confidence_threshold=args.confidence_threshold,
            separator_boost=args.separator_boost,
        )
    )

    if args.viewer_only:
        if not (output_root / "summary.json").is_file():
            raise FileNotFoundError(
                f"--viewer-only requested but summary does not exist: "
                f"{output_root / 'summary.json'}"
            )
    else:
        materialize_run(
            sample_id=args.sample_id,
            multicut_root=multicut_root,
            h100_root=h100_root,
            stage6_root=stage6_root,
            output_root=output_root,
            frames=frames,
            spacing_zyx_um=spacing,
            confidence_threshold_override=args.confidence_threshold,
            separator_boost_override=args.separator_boost,
            overwrite=args.overwrite,
        )

    if not args.no_viewer:
        open_viewer(
            sample_id=args.sample_id,
            multicut_root=multicut_root,
            h100_root=h100_root,
            stage6_root=stage6_root,
            output_root=output_root,
            frames=frames,
            sample_zarr_override=args.sample_zarr,
            spacing_zyx_um=spacing,
        )


if __name__ == "__main__":
    main()
