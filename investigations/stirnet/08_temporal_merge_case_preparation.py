from __future__ import annotations

"""
Stage 08: prepare the first full-context temporal merge-correction case (v6 controlled-error).

Scientific purpose
------------------
The spatial overfit phase has already shown that STIR-Net can memorize a
current-frame scene.  The first temporal experiment should therefore isolate a
different question:

    Can temporal evidence correct a deliberately wrong TWO-instance spatial
    interpretation back to the GT-consistent ONE-cell interpretation?

The chosen scene is Stage-00 candidate 0 and, by default, GT cell 6.  The cell
was manually confirmed in Napari as the intended visually two-lobed / sparse
ONE-cell annotation.

This stage DOES NOT train any neural module and DOES NOT yet corrupt the
spatial RAG.

Instead it prepares both sides of the later temporal overfit:

A. Spatial preparation
   -------------------
   Reuse the saved Stage-00 shortlist, extract candidate 0, build the normal
   production geometry targets/five-channel spatial input, and save a dedicated
   Stage-01-compatible crop.

B. Full-volume temporal context
   ----------------------------
   Load the COMPLETE instance/GT movies and a temporal-v3 STIR-Net/Trackastra
   cache built from the full scene.  Starting from the full-volume temporal
   evidence, select a target-centred EGO GRAPH containing:

       - tracklets associated with the target current/source component;
       - nearby tracklets in physical space;
       - hypothesis-graph neighbours;
       - all detections belonging to those selected tracklets over time.

   This is intentionally different from cropping the movie BEFORE tracking.
   Trackastra / temporal evidence must come from the full uncropped scene so a
   broken predecessor, newborn branch, nearby competing track, or lineage
   hypothesis is not removed before STIR-Net can reason about it.

Why not feed the entire full-volume graph into the GNN?
-------------------------------------------------------
The production candidate detection graph is complete directed over the records
it receives.  A whole-scene five-frame graph can therefore become O(N^2).
Stage 08 keeps FULL-VOLUME evidence discovery, then extracts a bounded,
context-rich temporal ego graph for the target correction.

The later temporal overfit should do:

    candidate-0 spatial overfit -> correct ONE-cell solution
        -> freeze spatial modules
        -> cut one SAME-GT safe-supervoxel/RAG connection
        -> force TWO provisional spatial instances
        -> feed the Stage-08 full-volume-derived temporal ego graph
        -> train tokenizer + temporal modules
        -> require temporal final RAG to restore ONE cell.

Typical usage
-------------
Prepare/rebuild baseline full-volume temporal context:


    python investigations/stirnet/08_temporal_merge_case_preparation.py \
        --rebuild --require-temporal-cache

If auto-discovery does not find the cache:

    python investigations/stirnet/08_temporal_merge_case_preparation.py \
        --rebuild \
        --temporal-cache <path-to-full-volume-temporal-v3-cache.pt> \
        --require-temporal-cache

Visualize the FULL MOVIE with a Napari time slider and selected temporal tracks:

    python investigations/stirnet/08_temporal_merge_case_preparation.py \
        --visualize

Visualize only the old compact candidate-0 spatial crop:

    python investigations/stirnet/08_temporal_merge_case_preparation.py \
        --visualize-crop

Create the controlled TWO-instance error, rerun Trackastra, rebuild
the controlled temporal-v3 cache, and save the Stage-09 input:

    python investigations/stirnet/08_temporal_merge_case_preparation.py \
        --rebuild \
        --require-temporal-cache \
        --prepare-temporal-overfit \
        --require-controlled-ready

Visualize the controlled TWO->ONE case:

    python investigations/stirnet/08_temporal_merge_case_preparation.py \
        --visualize-controlled

Outputs
-------
Spatial crop:
    data/learned/stirnet/temporal_merge_debug_crop.pt

Case report:
    data/learned/stirnet/temporal_merge_case_selection.json

Temporal ego graph:
    data/learned/stirnet/temporal_merge_case/full_temporal_context.pt
    data/learned/stirnet/temporal_merge_case/full_temporal_context.json

The temporal context artifact stores graph/history tensors only.  It does NOT
duplicate the full movies; it records their source paths.

Important invariant
-------------------
Stage 08 never edits GT and never invents a voxel-plane split.  The controlled
TWO-instance mistake is created later at the safe-supervoxel/RAG level after
candidate-0 spatial overfit.
"""

import argparse
import importlib.util
import heapq
import json
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy import ndimage as ndi


# =============================================================================
# REPOSITORY / ADJACENT INVESTIGATION IMPORTS
# =============================================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _import_adjacent(module_name: str, filename: str):
    path = Path(__file__).with_name(filename)
    if not path.exists():
        raise FileNotFoundError(f"Missing required investigation: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


stage00 = _import_adjacent(
    "_stirnet_stage00_temporal_case",
    "00_GT_validation.py",
)
stage04 = _import_adjacent(
    "_stirnet_stage04_temporal_case",
    "04_rag_case_selection.py",
)
stage01 = _import_adjacent(
    "_stirnet_stage01_temporal_case",
    "01_dense_geometry_overfit.py",
)

from learned.stirnet.data.graph_builder import (
    AssociationRecord,
    DetectionRecord,
    build_temporal_graph,
)
from learned.stirnet.data.historical_instances import (
    build_historical_instance_grid,
)
from learned.stirnet.data.sample_builder import robust_normalize
from learned.stirnet.data.targets import extract_instance_metadata
from learned.stirnet.data.trackastra_cache import load_cache, save_cache
from learned.stirnet.model.geometry.derived import build_geometry_derived_cache
from learned.stirnet.model.partition.watershed import LearnedGeometryWatershed
from learned.stirnet.model.types import GeometryState


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_CANDIDATE_INDEX = 0
# Manually confirmed from the Stage-08A Napari inspection.
DEFAULT_TARGET_GT_ID = 6

DEFAULT_DATA_DIR = stage00.DEFAULT_DATA_DIR
DEFAULT_TIME_INDEX = stage00.DEFAULT_TIME_INDEX
DEFAULT_MAX_CANDIDATES = stage00.DEFAULT_MAX_CANDIDATES
DEFAULT_CONTEXT_MARGIN_DREF = stage00.DEFAULT_CONTEXT_MARGIN_DREF
DEFAULT_VOXEL_BUDGET = stage00.DEFAULT_VOXEL_BUDGET

DEFAULT_STAGE00_SAMPLE = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "debug_crop.pt"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "temporal_merge_debug_crop.pt"
)
DEFAULT_SELECTION_JSON = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "temporal_merge_case_selection.json"
)
DEFAULT_CONTEXT_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "temporal_merge_case"
)
DEFAULT_CONTEXT_OUTPUT = DEFAULT_CONTEXT_DIR / "full_temporal_context.pt"
DEFAULT_CONTEXT_JSON = DEFAULT_CONTEXT_DIR / "full_temporal_context.json"

# Controlled TWO-instance counterfactual + Trackastra rerun.
DEFAULT_CONTROLLED_DIR = DEFAULT_CONTEXT_DIR / "controlled_error"
DEFAULT_CONTROLLED_INPUT_MOVIE = (
    DEFAULT_CONTROLLED_DIR / "controlled_instance_movie_input.npy"
)
DEFAULT_CONTROLLED_TRACKED_MOVIE = (
    DEFAULT_CONTROLLED_DIR / "trackastra_masks_tracked.npy"
)
DEFAULT_CONTROLLED_GRAPH = (
    DEFAULT_CONTROLLED_DIR / "trackastra" / "track_graph.pkl"
)
DEFAULT_CONTROLLED_CACHE = (
    DEFAULT_CONTROLLED_DIR / "temporal_v3" / "temporal_graph.pt"
)
DEFAULT_CONTROLLED_CONTEXT_OUTPUT = (
    DEFAULT_CONTROLLED_DIR / "controlled_temporal_context.pt"
)
DEFAULT_CONTROLLED_CONTEXT_JSON = (
    DEFAULT_CONTROLLED_DIR / "controlled_temporal_context.json"
)
DEFAULT_CONTROLLED_SPLIT_ARTIFACT = (
    DEFAULT_CONTROLLED_DIR / "controlled_split.pt"
)
DEFAULT_CONTROLLED_SPLIT_JSON = (
    DEFAULT_CONTROLLED_DIR / "controlled_split.json"
)
DEFAULT_STAGE01_TEMPORAL_CHECKPOINT = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "dense_geometry_temporal_merge"
    / "latest_success.pt"
)

DEFAULT_TRACKASTRA_MODEL = "ctc"
DEFAULT_TRACKASTRA_MODE = "greedy"
DEFAULT_TRACKASTRA_DEVICE = "cuda"
DEFAULT_MIN_SPLIT_FRACTION = 0.15
DEFAULT_MIN_TARGET_SV_PURITY = 0.80
DEFAULT_MIN_TARGET_SV_OVERLAP_VOXELS = 16

DEFAULT_CONTEXT_RADIUS_DREF = 5.0
DEFAULT_MIN_CONTEXT_TRACKLETS = 12
DEFAULT_MAX_CONTEXT_TRACKLETS = 64
DEFAULT_HYPOTHESIS_HOPS = 1
DEFAULT_ANCHOR_OVERLAP = 0.05
DEFAULT_VISUALIZE_RADIUS_DREF = 4.5
DEFAULT_VISUALIZE_MARGIN_DREF = 1.25

SOURCE_MIN_OVERLAP_VOXELS = 8
SOURCE_MIN_GT_FRACTION = 0.03

LOBE_CORE_FRACTIONS = (0.22, 0.30, 0.38, 0.46, 0.54)
LOBE_MIN_COMPONENT_VOXELS = 8
LOBE_MIN_CELL_FRACTION = 0.02

TABLE_LIMIT = 40

REQUIRED_TEMPORAL_KEYS = {
    "graph_x",
    "graph_edge_index",
    "graph_edge_attr",
    "tracklet_id",
    "node_ids",
    "node_observed_ref_um",
    "node_time_offset",
    "temporal_ref_um",
    "temporal_status",
    "hypothesis_edge_index",
    "hypothesis_edge_attr",
}

NODE_ALIGNED_KEYS = (
    "graph_x",
    "node_event_features",
    "node_ids",
    "node_observed_ref_um",
    "node_time_offset",
    "node_instance_grid",
    "node_history_valid",
)

TRACKLET_ALIGNED_KEYS = (
    "temporal_ref_um",
    "temporal_status",
    "history_support",
    "history_support_valid",
    "history_support_dt",
    "history_support_center_um",
    "history_support_extent_um",
    "best_current_component_id",
    "best_component_overlap",
    "second_best_component_overlap",
)

OPTIONAL_FULL_RAW_NAMES = (
    "raw_norm_movie.npy",
    "raw_movie.npy",
    "image_movie.npy",
    "images.npy",
)


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def _roi_with_all_cells(
    instance_movie: np.ndarray,
    gt_movie: np.ndarray,
    spacing: np.ndarray,
    margin_um: float = 12.0,
) -> tuple[tuple[slice, slice, slice], np.ndarray, np.ndarray]:
    """Reproduce the canonical all-cell ROI used by temporal-v3 cache creation.

    Kept local on purpose: importing debugging.acceptance.first_overfit currently
    pulls a stale acceptance package dependency (RefinementCriterion), while this
    small helper itself has no such dependency.
    """
    full_shape = np.asarray(instance_movie.shape[-3:], dtype=int)
    low = full_shape.copy()
    high = np.zeros(3, dtype=int)

    for frame in range(len(instance_movie)):
        foreground = (
            (np.asarray(instance_movie[frame]) > 0)
            | (np.asarray(gt_movie[frame]) > 0)
        )
        coords = np.where(foreground)
        if len(coords[0]):
            low = np.minimum(
                low,
                np.asarray([axis.min() for axis in coords], dtype=int),
            )
            high = np.maximum(
                high,
                np.asarray([axis.max() + 1 for axis in coords], dtype=int),
            )

    margin = np.ceil(
        float(margin_um) / np.asarray(spacing, dtype=np.float32)
    ).astype(int)
    low = np.maximum(low - margin, 0)
    high = np.minimum(high + margin, full_shape)

    roi = tuple(
        slice(int(a), int(b))
        for a, b in zip(low.tolist(), high.tolist())
    )
    return roi, low, high


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _as_cpu_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value).detach().cpu()


def _tensor_rows(value: Any) -> int:
    tensor = torch.as_tensor(value)
    return int(tensor.shape[0]) if tensor.ndim else 0


def _slice_pairs(
    slices: tuple[slice, slice, slice],
) -> list[list[int]]:
    return [
        [int(axis_slice.start), int(axis_slice.stop)]
        for axis_slice in slices
    ]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(torch.as_tensor(value).item())
    except Exception:
        return float(default)


# =============================================================================
# STAGE-00 CANDIDATE-0 REUSE
# =============================================================================


def _stage04_args(args: argparse.Namespace) -> argparse.Namespace:
    """Only fields consumed by Stage-04's cached Stage-00 loader."""
    return argparse.Namespace(
        data_dir=args.data_dir,
        time_index=args.time_index,
        voxel_budget=args.voxel_budget,
        max_candidates=args.max_candidates,
        context_margin_dref=args.context_margin_dref,
        stage00_sample=args.stage00_sample,
    )


def load_candidate_scene(args: argparse.Namespace):
    base_scene, candidates = stage04.load_full_scene_and_candidates(
        _stage04_args(args)
    )
    indexed: dict[int, tuple[Any, dict[str, Any]]] = {}
    for ordinal, (crop, selection) in enumerate(candidates):
        selection = dict(selection)
        index = int(selection.get("candidate_index", ordinal))
        selection["candidate_index"] = index
        indexed[index] = (crop, selection)

    index = int(args.candidate_index)
    if index not in indexed:
        raise IndexError(
            f"candidate {index} unavailable; available={sorted(indexed)}"
        )

    crop, selection = indexed[index]
    return stage04.make_candidate_scene(
        base_scene,
        crop,
        selection,
        args.context_margin_dref,
    )


# =============================================================================
# COMPLETE-CELL / TWO-LOBE DIAGNOSTICS
# =============================================================================


def _current_overlap_rows(
    gt_mask: np.ndarray,
    current: np.ndarray,
) -> tuple[list[dict[str, Any]], float]:
    values, counts = np.unique(current[gt_mask], return_counts=True)
    cell_volume = int(gt_mask.sum())
    rows: list[dict[str, Any]] = []
    background_voxels = 0

    for value, count in zip(values.tolist(), counts.tolist()):
        value = int(value)
        count = int(count)
        if value <= 0:
            background_voxels += count
            continue
        fraction = count / max(cell_volume, 1)
        rows.append(
            {
                "source_id": value,
                "overlap_voxels": count,
                "fraction_of_gt": float(fraction),
                "meaningful": bool(
                    count >= SOURCE_MIN_OVERLAP_VOXELS
                    and fraction >= SOURCE_MIN_GT_FRACTION
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            row["overlap_voxels"],
            -row["source_id"],
        ),
        reverse=True,
    )
    background_fraction = background_voxels / max(cell_volume, 1)
    return rows, float(background_fraction)


def _physical_centroid(
    mask: np.ndarray,
    spacing_um: np.ndarray,
) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros(3, dtype=np.float32)
    return (
        coords.astype(np.float32).mean(axis=0)
        * spacing_um.astype(np.float32)
    )


def _lobe_analysis(
    mask: np.ndarray,
    spacing_um: np.ndarray,
    dref_um: float,
) -> dict[str, Any]:
    cell_volume = int(mask.sum())
    empty = {
        "available": False,
        "threshold_fraction": None,
        "component_count": 0,
        "core_fraction_of_cell": 0.0,
        "second_core_fraction_of_cell": 0.0,
        "second_core_fraction_of_core": 0.0,
        "core_separation_um": 0.0,
        "core_separation_dref": 0.0,
        "balance": 0.0,
        "score": 0.0,
        "core_labels": np.zeros(mask.shape, dtype=np.int16),
        "core_centers_vox": [],
    }
    if cell_volume == 0:
        return empty

    distance = ndi.distance_transform_edt(
        mask,
        sampling=tuple(float(v) for v in spacing_um),
    ).astype(np.float32)
    maximum = float(distance.max())
    if maximum <= 0:
        return empty

    structure = ndi.generate_binary_structure(3, 1)
    best: dict[str, Any] | None = None

    for threshold_fraction in LOBE_CORE_FRACTIONS:
        core = mask & (distance >= threshold_fraction * maximum)
        labels, count = ndi.label(core, structure=structure)
        if count < 2:
            continue

        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        component_ids = np.arange(1, count + 1, dtype=np.int64)
        order = component_ids[np.argsort(sizes[1:])[::-1]]
        meaningful = [
            int(component_id)
            for component_id in order.tolist()
            if (
                int(sizes[component_id]) >= LOBE_MIN_COMPONENT_VOXELS
                and int(sizes[component_id]) / max(cell_volume, 1)
                >= LOBE_MIN_CELL_FRACTION
            )
        ]
        if len(meaningful) < 2:
            continue

        first_id, second_id = meaningful[:2]
        first_size = int(sizes[first_id])
        second_size = int(sizes[second_id])
        core_size = int(core.sum())

        first_mask = labels == first_id
        second_mask = labels == second_id
        center_a_um = _physical_centroid(first_mask, spacing_um)
        center_b_um = _physical_centroid(second_mask, spacing_um)
        separation_um = float(np.linalg.norm(center_a_um - center_b_um))
        separation_dref = separation_um / max(float(dref_um), 1e-6)

        second_cell_fraction = second_size / max(cell_volume, 1)
        second_core_fraction = second_size / max(core_size, 1)
        core_fraction = core_size / max(cell_volume, 1)
        balance = min(first_size, second_size) / max(first_size, second_size)

        score = (
            3.0 * second_cell_fraction
            + 0.75 * min(separation_dref, 2.0)
            + 0.50 * balance
            + 0.35 * min(core_fraction, 0.60)
        )

        relabeled = np.zeros_like(labels, dtype=np.int16)
        relabeled[first_mask] = 1
        relabeled[second_mask] = 2

        centers_vox: list[list[float]] = []
        for component_mask in (first_mask, second_mask):
            coords = np.argwhere(component_mask)
            centers_vox.append(
                coords.astype(np.float32).mean(axis=0).tolist()
                if coords.size
                else [0.0, 0.0, 0.0]
            )

        row = {
            "available": True,
            "threshold_fraction": float(threshold_fraction),
            "component_count": int(len(meaningful)),
            "core_fraction_of_cell": float(core_fraction),
            "second_core_fraction_of_cell": float(second_cell_fraction),
            "second_core_fraction_of_core": float(second_core_fraction),
            "core_separation_um": float(separation_um),
            "core_separation_dref": float(separation_dref),
            "balance": float(balance),
            "score": float(score),
            "core_labels": relabeled,
            "core_centers_vox": centers_vox,
        }
        if best is None or float(row["score"]) > float(best["score"]):
            best = row

    return best if best is not None else empty


def analyze_complete_cells(
    gt: np.ndarray,
    current: np.ndarray,
    complete_gt_ids: list[int],
    spacing_um: np.ndarray,
    dref_um: float,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    lobe_maps: dict[int, np.ndarray] = {}

    for gt_id in complete_gt_ids:
        mask = gt == int(gt_id)
        volume = int(mask.sum())
        if volume <= 0:
            continue

        source_rows, background_fraction = _current_overlap_rows(
            mask,
            current,
        )
        meaningful_sources = [
            row for row in source_rows if row["meaningful"]
        ]
        second_source_fraction = (
            float(meaningful_sources[1]["fraction_of_gt"])
            if len(meaningful_sources) >= 2
            else 0.0
        )
        source_split_score = (
            4.0 * second_source_fraction
            + 0.35 * min(len(meaningful_sources), 3)
        )

        lobe = _lobe_analysis(mask, spacing_um, dref_um)
        lobe_maps[int(gt_id)] = np.asarray(
            lobe.pop("core_labels"),
            dtype=np.int16,
        )

        score = (
            3.0 * source_split_score
            + float(lobe["score"])
            - 0.75 * background_fraction
        )

        rows.append(
            {
                "gt_id": int(gt_id),
                "gt_voxels": volume,
                "current_positive_coverage": float(
                    1.0 - background_fraction
                ),
                "current_background_fraction": float(
                    background_fraction
                ),
                "source_overlaps": source_rows,
                "meaningful_source_ids": [
                    int(row["source_id"])
                    for row in meaningful_sources
                ],
                "meaningful_source_count": len(meaningful_sources),
                "second_source_fraction_of_gt": float(
                    second_source_fraction
                ),
                "lobe": lobe,
                "temporal_case_score": float(score),
            }
        )

    rows.sort(
        key=lambda row: (
            row["temporal_case_score"],
            row["second_source_fraction_of_gt"],
            row["lobe"]["second_core_fraction_of_cell"],
            -row["gt_id"],
        ),
        reverse=True,
    )
    return rows, lobe_maps


def select_target_gt(
    rows: list[dict[str, Any]],
    forced_gt_id: int | None,
) -> int:
    if not rows:
        raise RuntimeError("No complete GT cells are available in candidate 0.")

    available = {int(row["gt_id"]) for row in rows}
    if forced_gt_id is not None:
        if int(forced_gt_id) not in available:
            raise ValueError(
                f"--target-gt-id {forced_gt_id} is not a complete GT cell. "
                f"Available complete IDs: {sorted(available)}"
            )
        return int(forced_gt_id)

    return int(rows[0]["gt_id"])


def _row_for_gt(
    rows: list[dict[str, Any]],
    gt_id: int,
) -> dict[str, Any]:
    for row in rows:
        if int(row["gt_id"]) == int(gt_id):
            return row
    raise KeyError(gt_id)


def print_cell_table(
    rows: list[dict[str, Any]],
    target_gt_id: int,
) -> None:
    print("\n" + "=" * 132)
    print("Stage-08 complete-cell temporal merge candidates")
    print("=" * 132)
    print(
        f"{'pick':>4s} {'GT':>5s} {'voxels':>8s} "
        f"{'src#':>5s} {'source IDs':>18s} {'2ndSrc':>8s} "
        f"{'2lobes':>7s} {'2ndLobe':>8s} {'sep/dref':>8s} "
        f"{'coverage':>9s} {'score':>8s}"
    )
    print("-" * 132)

    for row in rows[:TABLE_LIMIT]:
        lobe = row["lobe"]
        source_ids = ",".join(
            str(v) for v in row["meaningful_source_ids"]
        ) or "-"
        print(
            f"{'*' if int(row['gt_id']) == int(target_gt_id) else '':>4s} "
            f"{int(row['gt_id']):5d} "
            f"{int(row['gt_voxels']):8d} "
            f"{int(row['meaningful_source_count']):5d} "
            f"{source_ids:>18.18s} "
            f"{float(row['second_source_fraction_of_gt']):8.3f} "
            f"{('yes' if lobe['available'] else 'no'):>7s} "
            f"{float(lobe['second_core_fraction_of_cell']):8.3f} "
            f"{float(lobe['core_separation_dref']):8.3f} "
            f"{float(row['current_positive_coverage']):9.3f} "
            f"{float(row['temporal_case_score']):8.3f}"
        )

    print("=" * 132)
    print("* = selected target GT cell")


def print_selected_detail(row: dict[str, Any]) -> None:
    lobe = row["lobe"]
    print("\nSelected spatial target details")
    print(f"  GT ID                         : {row['gt_id']}")
    print(f"  GT voxels                     : {row['gt_voxels']}")
    print(
        "  current/source positive cover : "
        f"{row['current_positive_coverage']:.4f}"
    )
    print(
        "  meaningful current/source IDs : "
        f"{row['meaningful_source_ids']}"
    )
    for source in row["source_overlaps"]:
        marker = "*" if source["meaningful"] else " "
        print(
            f"    {marker} source {source['source_id']:4d}: "
            f"{source['overlap_voxels']:6d} vox "
            f"({source['fraction_of_gt']:.3f} of GT)"
        )

    if lobe["available"]:
        print(
            "  two-lobe EDT evidence          : YES "
            f"(threshold={lobe['threshold_fraction']:.2f} max-EDT)"
        )
        print(
            "  second core / GT               : "
            f"{lobe['second_core_fraction_of_cell']:.4f}"
        )
        print(
            "  core separation                : "
            f"{lobe['core_separation_um']:.3f} um "
            f"({lobe['core_separation_dref']:.3f} dref)"
        )
    else:
        print("  two-lobe EDT evidence          : no stable 2-core split")


# =============================================================================
# FULL MOVIE / TARGET METADATA
# =============================================================================


def load_full_movies(
    data_dir: Path,
) -> tuple[np.memmap | np.ndarray, np.memmap | np.ndarray]:
    instance_path = data_dir / "instance_movie.npy"
    gt_path = data_dir / "gt_movie.npy"
    if not instance_path.exists() or not gt_path.exists():
        raise FileNotFoundError(
            "Stage 08 requires the full movies:\n"
            f"  {instance_path}\n"
            f"  {gt_path}"
        )

    instance_movie = np.load(instance_path, mmap_mode="r")
    gt_movie = np.load(gt_path, mmap_mode="r")
    if instance_movie.ndim != 4 or gt_movie.ndim != 4:
        raise ValueError(
            "instance_movie.npy and gt_movie.npy must have shape [T,Z,Y,X]"
        )
    if instance_movie.shape != gt_movie.shape:
        raise ValueError(
            "instance_movie and gt_movie must have identical shapes; "
            f"got {instance_movie.shape} vs {gt_movie.shape}"
        )
    return instance_movie, gt_movie


def full_target_metadata(
    instance_movie: np.ndarray,
    gt_movie: np.ndarray,
    *,
    time_index: int,
    target_gt_id: int,
    spacing_um: np.ndarray,
) -> dict[str, Any]:
    if not 0 <= time_index < gt_movie.shape[0]:
        raise IndexError(
            f"time_index={time_index} outside movie with T={gt_movie.shape[0]}"
        )

    gt_frame = np.asarray(gt_movie[time_index])
    current_frame = np.asarray(instance_movie[time_index])
    target_mask = gt_frame == int(target_gt_id)
    if not target_mask.any():
        raise RuntimeError(
            f"GT {target_gt_id} is absent from full GT frame t={time_index}."
        )

    coords = np.argwhere(target_mask).astype(np.float32)
    centroid_vox = coords.mean(axis=0)
    centroid_abs_um = centroid_vox * spacing_um.astype(np.float32)

    # IMPORTANT: temporal_v3 was built in the exact all-cell ROI used by
    # first_overfit._build_temporal_inputs(), NOT relative to the full-volume
    # center. DetectionRecord.position_um is:
    #
    #   (coords_full - roi_low) * spacing - roi_center_um
    #
    # Stage-08 v3 incorrectly used the full-volume center. That translated the
    # temporal graph away from the image and also made neighbourhood distances
    # wrong. Keep the canonical cache coordinate system explicitly here.
    temporal_roi, temporal_roi_low, temporal_roi_high = _roi_with_all_cells(
        instance_movie,
        gt_movie,
        spacing_um,
    )
    temporal_roi_shape = (
        temporal_roi_high - temporal_roi_low
    ).astype(np.float32)
    temporal_roi_center_um = (
        0.5
        * (temporal_roi_shape - 1.0)
        * spacing_um.astype(np.float32)
    )
    centroid_temporal_um = (
        (centroid_vox - temporal_roi_low.astype(np.float32))
        * spacing_um.astype(np.float32)
        - temporal_roi_center_um
    )

    source_rows, background_fraction = _current_overlap_rows(
        target_mask,
        current_frame,
    )
    meaningful = [row for row in source_rows if row["meaningful"]]
    source_ids = [int(row["source_id"]) for row in meaningful]
    if not source_ids and source_rows:
        source_ids = [int(source_rows[0]["source_id"])]

    bbox = ndi.find_objects(target_mask.astype(np.uint8))
    bbox_pairs: list[list[int]] | None = None
    if bbox and bbox[0] is not None:
        bbox_pairs = _slice_pairs(bbox[0])

    return {
        "target_gt_id": int(target_gt_id),
        "time_index": int(time_index),
        "movie_shape_tzyx": [int(v) for v in gt_movie.shape],
        "frame_shape_zyx": [int(v) for v in gt_frame.shape],
        "gt_voxels": int(target_mask.sum()),
        "target_centroid_vox_zyx": [
            float(v) for v in centroid_vox.tolist()
        ],
        "target_centroid_abs_um_zyx": [
            float(v) for v in centroid_abs_um.tolist()
        ],
        "target_centroid_temporal_um_zyx": [
            float(v) for v in centroid_temporal_um.tolist()
        ],
        "temporal_roi_low_zyx": [
            int(v) for v in temporal_roi_low.tolist()
        ],
        "temporal_roi_high_zyx": [
            int(v) for v in temporal_roi_high.tolist()
        ],
        "temporal_roi_shape_zyx": [
            int(v) for v in (temporal_roi_high - temporal_roi_low).tolist()
        ],
        "temporal_roi_center_um_zyx": [
            float(v) for v in temporal_roi_center_um.tolist()
        ],
        "target_bbox_zyx": bbox_pairs,
        "source_overlaps": source_rows,
        "target_source_ids": source_ids,
        "current_positive_coverage": float(1.0 - background_fraction),
    }


# =============================================================================
# TEMPORAL-V3 CACHE DISCOVERY / VALIDATION
# =============================================================================


def _candidate_temporal_cache_paths(
    data_dir: Path,
) -> list[Path]:
    roots = [
        data_dir,
        data_dir / "stirnet_source",
        data_dir.parent,
    ]
    patterns = (
        "*temporal*.pt",
        "*trackastra*.pt",
        "*cache*.pt",
    )
    found: dict[Path, float] = {}
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            try:
                iterator = root.rglob(pattern)
            except OSError:
                continue
            for path in iterator:
                if not path.is_file():
                    continue
                # Keep discovery bounded to plausible cache names.  In
                # particular, do not try every arbitrary training checkpoint.
                name = path.name.lower()
                if not any(
                    token in name
                    for token in ("temporal", "trackastra", "cache")
                ):
                    continue
                try:
                    found[path.resolve()] = path.stat().st_mtime
                except OSError:
                    continue
    return [
        path
        for path, _ in sorted(
            found.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def _validate_temporal_cache_payload(
    payload: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    missing = sorted(REQUIRED_TEMPORAL_KEYS - set(payload))
    if missing:
        failures.append("missing keys: " + ", ".join(missing))
        return failures

    try:
        n = int(torch.as_tensor(payload["graph_x"]).shape[0])
        m = int(torch.as_tensor(payload["temporal_ref_um"]).shape[0])
        if torch.as_tensor(payload["tracklet_id"]).shape != (n,):
            failures.append("tracklet_id does not align with graph_x")
        if torch.as_tensor(payload["node_time_offset"]).shape != (n,):
            failures.append("node_time_offset does not align with graph_x")
        if torch.as_tensor(payload["node_ids"]).shape != (n,):
            failures.append("node_ids does not align with graph_x")
        if torch.as_tensor(payload["node_observed_ref_um"]).shape != (n, 3):
            failures.append("node_observed_ref_um must have shape [N,3]")
        if torch.as_tensor(payload["temporal_ref_um"]).shape != (m, 3):
            failures.append("temporal_ref_um must have shape [M,3]")
        if torch.as_tensor(payload["temporal_status"]).shape[0] != m:
            failures.append("temporal_status does not align with tracklets")
        hidx = torch.as_tensor(payload["hypothesis_edge_index"])
        hattr = torch.as_tensor(payload["hypothesis_edge_attr"])
        if hidx.ndim != 2 or hidx.shape[0] != 2:
            failures.append("hypothesis_edge_index must have shape [2,H]")
        elif hattr.shape[0] != hidx.shape[1]:
            failures.append(
                "hypothesis_edge_attr does not align with hypothesis edges"
            )
    except Exception as exc:
        failures.append(f"tensor-contract error: {exc}")

    return failures


def resolve_temporal_cache(
    explicit_path: Path | None,
    data_dir: Path,
) -> tuple[Path | None, dict[str, Any] | None, list[str]]:
    notes: list[str] = []

    paths: list[Path]
    if explicit_path is not None:
        paths = [explicit_path.resolve()]
    else:
        paths = _candidate_temporal_cache_paths(data_dir)

    if not paths:
        notes.append(
            "No plausible temporal-v3 cache file was discovered automatically."
        )
        return None, None, notes

    for path in paths:
        if not path.exists():
            notes.append(f"missing cache candidate: {path}")
            continue
        try:
            # load_cache enforces the repository's current temporal cache
            # contract and intentionally rejects stale accepted-edge caches.
            payload = load_cache(path, map_location="cpu")
        except Exception as exc:
            notes.append(f"rejected {path}: {type(exc).__name__}: {exc}")
            if explicit_path is not None:
                break
            continue

        failures = _validate_temporal_cache_payload(payload)
        if failures:
            notes.append(
                f"rejected {path}: " + "; ".join(failures)
            )
            if explicit_path is not None:
                break
            continue

        notes.append(f"accepted temporal cache: {path}")
        return path, payload, notes

    return None, None, notes


# =============================================================================
# FULL-VOLUME-DERIVED TEMPORAL EGO GRAPH
# =============================================================================


def _load_trackastra_graph(data_dir: Path):
    path = data_dir / "trackastra" / "track_graph.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Trackastra graph not found:\n  {path}"
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


def validate_temporal_cache_coordinate_frame(
    payload: dict[str, Any],
    target_meta: dict[str, Any],
    track_graph,
    spacing_um: np.ndarray,
) -> dict[str, float]:
    """Verify cached node coordinates against the exact Trackastra geometry.

    Current temporal_v3 DetectionRecord.position_um is relative to the all-cell
    ROI center. This check prevents another silent coordinate-frame mismatch.
    """
    node_ids = torch.as_tensor(payload["node_ids"]).long().cpu().tolist()
    observed = (
        torch.as_tensor(payload["node_observed_ref_um"])
        .float()
        .cpu()
        .numpy()
    )
    roi_low = np.asarray(
        target_meta["temporal_roi_low_zyx"],
        dtype=np.float32,
    )
    roi_center_um = np.asarray(
        target_meta["temporal_roi_center_um_zyx"],
        dtype=np.float32,
    )
    spacing = np.asarray(spacing_um, dtype=np.float32)

    errors: list[float] = []
    compared = 0
    for row, node_id in enumerate(node_ids):
        if node_id not in track_graph:
            continue
        data = track_graph.nodes[node_id]
        coords_full = np.asarray(data["coords"], dtype=np.float32)
        expected = (
            (coords_full - roi_low) * spacing - roi_center_um
        )
        errors.append(
            float(np.linalg.norm(observed[row] - expected))
        )
        compared += 1

    if not errors:
        raise RuntimeError(
            "Could not validate temporal cache coordinates: none of the "
            "cached node_ids exist in track_graph.pkl."
        )

    diagnostics = {
        "compared_nodes": float(compared),
        "median_error_um": float(np.median(errors)),
        "p95_error_um": float(np.percentile(errors, 95)),
        "max_error_um": float(np.max(errors)),
    }
    # Values are generated from the same float32 Trackastra coordinates, so a
    # large discrepancy indicates a genuinely different cache frame/scene.
    if diagnostics["max_error_um"] > 0.10:
        raise RuntimeError(
            "Temporal cache coordinate contract does not match the current "
            "all-cell ROI / Trackastra graph. "
            f"median={diagnostics['median_error_um']:.4f} um, "
            f"max={diagnostics['max_error_um']:.4f} um"
        )
    return diagnostics


def choose_coordinate_mode(
    requested: str,
    payload: dict[str, Any],
    target_meta: dict[str, Any],
    anchor_tracklets: list[int],
) -> tuple[str, dict[str, float]]:
    # Kept as a compatibility wrapper for the existing CLI/report. Current
    # temporal_v3 has one canonical contract: all-cell-ROI-centred microns.
    if requested not in {"auto", "centered", "absolute"}:
        raise ValueError(requested)
    return "all_cell_roi_centered", {}


def target_ref_um(
    target_meta: dict[str, Any],
    coordinate_mode: str,
) -> np.ndarray:
    if coordinate_mode != "all_cell_roi_centered":
        raise ValueError(
            f"Unsupported temporal coordinate mode: {coordinate_mode}"
        )
    return np.asarray(
        target_meta["target_centroid_temporal_um_zyx"],
        dtype=np.float32,
    )


def find_anchor_tracklets(
    payload: dict[str, Any],
    target_meta: dict[str, Any],
    *,
    min_overlap: float,
) -> list[int]:
    source_ids = set(int(v) for v in target_meta["target_source_ids"])
    if not source_ids:
        return []

    m = int(torch.as_tensor(payload["temporal_ref_um"]).shape[0])
    best_component = payload.get("best_current_component_id")
    best_overlap = payload.get("best_component_overlap")

    anchors: list[int] = []
    if best_component is not None:
        components = torch.as_tensor(best_component).long().cpu()
        overlap = (
            torch.as_tensor(best_overlap).float().cpu()
            if best_overlap is not None
            else torch.ones((m,), dtype=torch.float32)
        )
        if components.shape == (m,) and overlap.shape == (m,):
            for index in range(m):
                if (
                    int(components[index]) in source_ids
                    and float(overlap[index]) >= min_overlap
                ):
                    anchors.append(index)

    return sorted(set(anchors))


def tracklet_distances_dref(
    payload: dict[str, Any],
    target_um: np.ndarray,
    dref_um: float,
) -> np.ndarray:
    refs = torch.as_tensor(
        payload["temporal_ref_um"],
        dtype=torch.float32,
    ).cpu().numpy()
    return np.linalg.norm(
        refs - target_um[None, :],
        axis=1,
    ) / max(float(dref_um), 1e-6)


def expand_hypothesis_hops(
    initial: set[int],
    hidx: torch.Tensor,
    hops: int,
) -> set[int]:
    selected = set(initial)
    frontier = set(initial)
    if not frontier or hops <= 0 or hidx.numel() == 0:
        return selected

    source = hidx[0].long().cpu().tolist()
    destination = hidx[1].long().cpu().tolist()

    for _ in range(hops):
        next_frontier: set[int] = set()
        for a, b in zip(source, destination):
            if a in frontier:
                next_frontier.add(int(b))
            if b in frontier:
                next_frontier.add(int(a))
        next_frontier -= selected
        if not next_frontier:
            break
        selected |= next_frontier
        frontier = next_frontier

    return selected


def select_context_tracklets(
    payload: dict[str, Any],
    target_meta: dict[str, Any],
    *,
    dref_um: float,
    coordinate_mode: str,
    anchors: list[int],
    radius_dref: float,
    min_tracklets: int,
    max_tracklets: int,
    hypothesis_hops: int,
) -> tuple[list[int], np.ndarray]:
    m = int(torch.as_tensor(payload["temporal_ref_um"]).shape[0])
    if m == 0:
        return [], np.zeros((0,), dtype=np.float32)

    target_um = target_ref_um(target_meta, coordinate_mode)
    distances = tracklet_distances_dref(
        payload,
        target_um,
        dref_um,
    )

    selected: set[int] = set(anchors)
    selected.update(
        int(index)
        for index in np.where(distances <= radius_dref)[0].tolist()
    )

    hidx = torch.as_tensor(payload["hypothesis_edge_index"]).long()
    selected = expand_hypothesis_hops(
        selected,
        hidx,
        hypothesis_hops,
    )

    nearest_order = np.argsort(distances)
    requested_min = min(max(min_tracklets, len(anchors)), m)
    for index in nearest_order.tolist():
        if len(selected) >= requested_min:
            break
        selected.add(int(index))

    # Bound compute while always preserving anchors.  Non-anchor context is
    # retained in increasing distance order.
    if len(selected) > max_tracklets:
        kept = set(anchors)
        for index in nearest_order.tolist():
            if index not in selected or index in kept:
                continue
            kept.add(int(index))
            if len(kept) >= max_tracklets:
                break
        selected = kept

    ordered = sorted(
        selected,
        key=lambda index: (
            0 if index in anchors else 1,
            float(distances[index]),
            int(index),
        ),
    )
    return ordered, distances


def _subset_first_dim(
    value: Any,
    indices: torch.Tensor,
) -> torch.Tensor:
    tensor = _as_cpu_tensor(value)
    return tensor.index_select(0, indices)


def subset_temporal_context(
    payload: dict[str, Any],
    selected_tracklets: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_tracklet_tensor = torch.tensor(
        selected_tracklets,
        dtype=torch.long,
    )
    selected_tracklet_set = set(selected_tracklets)

    old_tracklet_id = torch.as_tensor(payload["tracklet_id"]).long().cpu()
    node_keep = torch.tensor(
        [
            int(tracklet) in selected_tracklet_set
            for tracklet in old_tracklet_id.tolist()
        ],
        dtype=torch.bool,
    )
    old_node_rows = torch.nonzero(node_keep, as_tuple=False).flatten()

    old_to_new_tracklet = {
        int(old): int(new)
        for new, old in enumerate(selected_tracklets)
    }
    old_to_new_node = torch.full(
        (old_tracklet_id.shape[0],),
        -1,
        dtype=torch.long,
    )
    old_to_new_node[old_node_rows] = torch.arange(
        old_node_rows.numel(),
        dtype=torch.long,
    )

    context: dict[str, Any] = {}

    for key in NODE_ALIGNED_KEYS:
        if key in payload:
            value = torch.as_tensor(payload[key])
            if value.ndim and value.shape[0] == old_tracklet_id.shape[0]:
                context[key] = _subset_first_dim(
                    payload[key],
                    old_node_rows,
                )

    reindexed_tracklet_id = torch.tensor(
        [
            old_to_new_tracklet[int(value)]
            for value in old_tracklet_id[old_node_rows].tolist()
        ],
        dtype=torch.long,
    )
    context["tracklet_id"] = reindexed_tracklet_id

    for key in TRACKLET_ALIGNED_KEYS:
        if key in payload:
            value = torch.as_tensor(payload[key])
            if value.ndim and value.shape[0] == int(
                torch.as_tensor(payload["temporal_ref_um"]).shape[0]
            ):
                context[key] = _subset_first_dim(
                    payload[key],
                    selected_tracklet_tensor,
                )

    def subset_node_edges(
        index_key: str,
        attr_key: str,
    ) -> None:
        if index_key not in payload:
            return
        edge_index = torch.as_tensor(payload[index_key]).long().cpu()
        if edge_index.numel() == 0:
            context[index_key] = torch.zeros((2, 0), dtype=torch.long)
            if attr_key in payload:
                attr = torch.as_tensor(payload[attr_key]).cpu()
                context[attr_key] = attr[:0].clone()
            return

        keep = node_keep[edge_index[0]] & node_keep[edge_index[1]]
        kept_edges = edge_index[:, keep]
        context[index_key] = old_to_new_node[kept_edges]
        if attr_key in payload:
            context[attr_key] = (
                torch.as_tensor(payload[attr_key]).cpu()[keep].clone()
            )

    subset_node_edges("graph_edge_index", "graph_edge_attr")
    subset_node_edges(
        "accepted_association_edge_index",
        "accepted_association_edge_attr",
    )

    hidx = torch.as_tensor(payload["hypothesis_edge_index"]).long().cpu()
    if hidx.numel():
        tracklet_keep = torch.zeros(
            (
                int(
                    torch.as_tensor(payload["temporal_ref_um"]).shape[0]
                ),
            ),
            dtype=torch.bool,
        )
        tracklet_keep[selected_tracklet_tensor] = True
        keep = tracklet_keep[hidx[0]] & tracklet_keep[hidx[1]]
        kept = hidx[:, keep]
        reindex = torch.full(
            (tracklet_keep.shape[0],),
            -1,
            dtype=torch.long,
        )
        reindex[selected_tracklet_tensor] = torch.arange(
            len(selected_tracklets),
            dtype=torch.long,
        )
        context["hypothesis_edge_index"] = reindex[kept]
        context["hypothesis_edge_attr"] = (
            torch.as_tensor(payload["hypothesis_edge_attr"])
            .cpu()[keep]
            .clone()
        )
    else:
        context["hypothesis_edge_index"] = torch.zeros(
            (2, 0),
            dtype=torch.long,
        )
        hattr = torch.as_tensor(payload["hypothesis_edge_attr"]).cpu()
        context["hypothesis_edge_attr"] = hattr[:0].clone()

    context["temporal_batch"] = torch.zeros(
        (len(selected_tracklets),),
        dtype=torch.long,
    )

    mapping = {
        "original_tracklet_ids": [
            int(v) for v in selected_tracklets
        ],
        "original_node_rows": [
            int(v) for v in old_node_rows.tolist()
        ],
        "original_node_ids": [
            int(v)
            for v in torch.as_tensor(payload["node_ids"])
            .long()
            .cpu()[old_node_rows]
            .tolist()
        ],
    }
    return context, mapping


def tracklet_time_offsets(
    payload: dict[str, Any],
    tracklet_index: int,
) -> list[int]:
    tracklet_id = torch.as_tensor(payload["tracklet_id"]).long().cpu()
    offsets = torch.as_tensor(payload["node_time_offset"]).float().cpu()
    rows = offsets[tracklet_id == int(tracklet_index)]
    return sorted(
        set(int(round(float(value))) for value in rows.tolist())
    )


def temporal_context_report(
    payload: dict[str, Any],
    context: dict[str, Any],
    mapping: dict[str, Any],
    *,
    target_meta: dict[str, Any],
    anchors: list[int],
    selected_tracklets: list[int],
    distances_dref: np.ndarray,
    coordinate_mode: str,
    coordinate_diagnostics: dict[str, float],
    cache_path: Path,
) -> dict[str, Any]:
    status = torch.as_tensor(payload["temporal_status"]).float().cpu()
    best_component = payload.get("best_current_component_id")
    best_overlap = payload.get("best_component_overlap")
    second_overlap = payload.get("second_best_component_overlap")
    history_valid = payload.get("history_support_valid")

    best_component_t = (
        torch.as_tensor(best_component).long().cpu()
        if best_component is not None
        else None
    )
    best_overlap_t = (
        torch.as_tensor(best_overlap).float().cpu()
        if best_overlap is not None
        else None
    )
    second_overlap_t = (
        torch.as_tensor(second_overlap).float().cpu()
        if second_overlap is not None
        else None
    )
    history_valid_t = (
        torch.as_tensor(history_valid).bool().cpu()
        if history_valid is not None
        else None
    )

    rows: list[dict[str, Any]] = []
    for old_tracklet in selected_tracklets:
        times = tracklet_time_offsets(payload, old_tracklet)
        row: dict[str, Any] = {
            "original_tracklet_id": int(old_tracklet),
            "anchor": bool(old_tracklet in anchors),
            "distance_to_target_dref": float(
                distances_dref[old_tracklet]
            ),
            "time_offsets": times,
            "has_past": bool(any(value < 0 for value in times)),
            "has_current": bool(any(value == 0 for value in times)),
            "has_future": bool(any(value > 0 for value in times)),
        }

        if status.ndim == 2 and old_tracklet < status.shape[0]:
            values = status[old_tracklet].tolist()
            row["status"] = {
                "complete": bool(values[0] > 0.5) if len(values) > 0 else False,
                "interior_start": bool(values[1] > 0.5) if len(values) > 1 else False,
                "interior_end": bool(values[2] > 0.5) if len(values) > 2 else False,
                "gap": bool(values[3] > 0.5) if len(values) > 3 else False,
                "division": bool(values[4] > 0.5) if len(values) > 4 else False,
                "boundary": bool(values[5] > 0.5) if len(values) > 5 else False,
                "reaches_past_window": bool(values[6] > 0.5) if len(values) > 6 else False,
                "reaches_future_window": bool(values[7] > 0.5) if len(values) > 7 else False,
                "uncertain": bool(values[9] > 0.5) if len(values) > 9 else False,
            }

        if (
            best_component_t is not None
            and old_tracklet < best_component_t.shape[0]
        ):
            row["best_current_component_id"] = int(
                best_component_t[old_tracklet]
            )
        if (
            best_overlap_t is not None
            and old_tracklet < best_overlap_t.shape[0]
        ):
            row["best_component_overlap"] = float(
                best_overlap_t[old_tracklet]
            )
        if (
            second_overlap_t is not None
            and old_tracklet < second_overlap_t.shape[0]
        ):
            row["second_best_component_overlap"] = float(
                second_overlap_t[old_tracklet]
            )
        if (
            history_valid_t is not None
            and old_tracklet < history_valid_t.shape[0]
        ):
            row["valid_history_support_count"] = int(
                history_valid_t[old_tracklet].sum().item()
            )

        rows.append(row)

    anchor_rows = [row for row in rows if row["anchor"]]
    anchor_with_past = [
        row for row in anchor_rows if row["has_past"]
    ]

    # For the first TWO->ONE temporal correction, at least one target-associated
    # track with past evidence is the minimum useful condition.  Stage 09 will
    # impose the stronger learned correction criteria.
    temporal_ready = bool(anchor_rows and anchor_with_past)

    return {
        "format_version": 2,
        "stage": "08_temporal_merge_case_preparation",
        "temporal_ready": temporal_ready,
        "temporal_cache": str(cache_path),
        "coordinate_mode": coordinate_mode,
        "coordinate_diagnostics": coordinate_diagnostics,
        "target": target_meta,
        "anchor_tracklets": [int(v) for v in anchors],
        "selected_original_tracklets": [
            int(v) for v in selected_tracklets
        ],
        "tracklets": rows,
        "context_counts": {
            "tracklets": int(
                torch.as_tensor(context["temporal_ref_um"]).shape[0]
            ),
            "detection_nodes": int(
                torch.as_tensor(context["graph_x"]).shape[0]
            ),
            "detection_edges": int(
                torch.as_tensor(context["graph_edge_index"]).shape[1]
            ),
            "accepted_association_edges": int(
                torch.as_tensor(
                    context.get(
                        "accepted_association_edge_index",
                        torch.zeros((2, 0)),
                    )
                ).shape[1]
            ),
            "hypothesis_edges": int(
                torch.as_tensor(
                    context["hypothesis_edge_index"]
                ).shape[1]
            ),
        },
        "mapping": mapping,
        "readiness_reason": (
            "At least one target-associated tracklet has past evidence."
            if temporal_ready
            else (
                "No target-associated tracklet with past evidence was found. "
                "Do not start temporal overfit until the cache/association "
                "context is inspected or corrected."
            )
        ),
    }


def print_temporal_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 118)
    print("Stage-08 full-volume-derived temporal context")
    print("=" * 118)
    target = report["target"]
    counts = report["context_counts"]

    print(f"Target GT                       : {target['target_gt_id']}")
    print(f"Target frame                    : {target['time_index']}")
    print(f"Target source/current IDs       : {target['target_source_ids']}")
    print(f"Temporal coordinate convention  : {report['coordinate_mode']}")
    print(f"Anchor tracklets                : {report['anchor_tracklets']}")
    print(f"Context tracklets               : {counts['tracklets']}")
    print(f"Context detection nodes         : {counts['detection_nodes']}")
    print(f"Candidate detection edges       : {counts['detection_edges']}")
    print(
        f"Accepted Trackastra edges       : "
        f"{counts['accepted_association_edges']}"
    )
    print(f"Tracklet hypothesis edges       : {counts['hypothesis_edges']}")
    print("-" * 118)
    print(
        f"{'trk':>5s} {'A':>2s} {'dist':>7s} {'times':>15s} "
        f"{'past':>5s} {'cur':>4s} {'fut':>4s} "
        f"{'comp':>6s} {'ovlp':>7s} {'hist':>5s} "
        f"{'gap':>4s} {'div':>4s}"
    )
    print("-" * 118)

    for row in report["tracklets"]:
        status = row.get("status", {})
        print(
            f"{row['original_tracklet_id']:5d} "
            f"{'*' if row['anchor'] else '':>2s} "
            f"{row['distance_to_target_dref']:7.2f} "
            f"{str(row['time_offsets']):>15.15s} "
            f"{str(row['has_past']):>5s} "
            f"{str(row['has_current']):>4s} "
            f"{str(row['has_future']):>4s} "
            f"{str(row.get('best_current_component_id', '-')):>6s} "
            f"{row.get('best_component_overlap', 0.0):7.3f} "
            f"{row.get('valid_history_support_count', 0):5d} "
            f"{str(status.get('gap', False)):>4s} "
            f"{str(status.get('division', False)):>4s}"
        )

    print("-" * 118)
    print(
        f"TEMPORAL CONTEXT READY           : "
        f"{'YES' if report['temporal_ready'] else 'NO'}"
    )
    print(f"Reason                           : {report['readiness_reason']}")
    print("=" * 118)


# =============================================================================
# SPATIAL CROP PREPARATION
# =============================================================================


def prepare_spatial_crop(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    scene = load_candidate_scene(args)
    complete_ids, partial_ids = stage00._complete_and_partial_ids(
        scene.gt_full,
        scene.core_slices,
    )

    current_crop = np.asarray(
        scene.current_full[scene.core_slices]
    ).astype(np.int64, copy=True)
    gt_crop = np.asarray(
        scene.gt_full[scene.core_slices]
    ).astype(np.int64, copy=True)

    rows, _ = analyze_complete_cells(
        gt_crop,
        current_crop,
        complete_ids,
        scene.spacing_zyx_um,
        scene.dref_um,
    )
    target_gt_id = select_target_gt(rows, args.target_gt_id)
    selected = _row_for_gt(rows, target_gt_id)

    print_cell_table(rows, target_gt_id)
    print_selected_detail(selected)

    selection = dict(scene.selection)
    selection["temporal_merge_case"] = {
        "format_version": 2,
        "candidate_index": int(args.candidate_index),
        "target_gt_id": int(target_gt_id),
        "selection_mode": "manual_confirmed",
        "complete_gt_ids": [int(v) for v in complete_ids],
        "partial_gt_ids": [int(v) for v in partial_ids],
        "target_summary": selected,
        "candidate_ranking": rows,
        "preparation_note": (
            "Correct, uncorrupted spatial training sample. Later temporal "
            "overfit must create TWO provisional instances by cutting SAME-GT "
            "safe-supervoxel/RAG connectivity; do not edit GT/dense geometry."
        ),
    }
    scene = replace(scene, selection=selection)

    print("\n[spatial 1/3] Building production geometry targets ...", flush=True)
    targets = stage00.build_targets(scene)

    print("[spatial 2/3] Building production five-channel spatial inputs ...", flush=True)
    spatial_inputs = stage00.build_saved_spatial_inputs(scene)
    relative_core = stage00._relative_slices(
        scene.core_slices,
        scene.build_slices,
    )

    print("[spatial 3/3] Saving dedicated candidate-0 sample ...", flush=True)
    stage00.save_debug_sample(
        args.output,
        scene,
        targets,
        spatial_inputs,
        relative_core,
        source_data_dir=args.data_dir,
        time_index=args.time_index,
    )

    report = {
        "format_version": 2,
        "stage": "08_temporal_merge_case_preparation",
        "candidate_index": int(args.candidate_index),
        "target_gt_id": int(target_gt_id),
        "crop_shape_zyx": list(gt_crop.shape),
        "crop_slices_zyx": _slice_pairs(scene.core_slices),
        "spacing_zyx_um": [
            float(v) for v in scene.spacing_zyx_um.tolist()
        ],
        "dref_um": float(scene.dref_um),
        "complete_gt_ids": [int(v) for v in complete_ids],
        "partial_gt_ids": [int(v) for v in partial_ids],
        "selected_target": selected,
        "candidate_ranking": rows,
        "output_sample": str(args.output),
    }
    return args.output, report


# =============================================================================
# FULL TEMPORAL CONTEXT PREPARATION
# =============================================================================


def prepare_temporal_context(
    args: argparse.Namespace,
    spatial_report: dict[str, Any],
) -> dict[str, Any] | None:
    print("\n[temporal 1/5] Loading full uncropped instance/GT movies ...", flush=True)
    instance_movie, gt_movie = load_full_movies(args.data_dir)

    spacing = np.asarray(
        spatial_report["spacing_zyx_um"],
        dtype=np.float32,
    )
    dref_um = float(spatial_report["dref_um"])
    target_gt_id = int(spatial_report["target_gt_id"])

    target_meta = full_target_metadata(
        instance_movie,
        gt_movie,
        time_index=args.time_index,
        target_gt_id=target_gt_id,
        spacing_um=spacing,
    )
    print(
        f"[temporal] full movie shape={tuple(gt_movie.shape)} "
        f"target source IDs={target_meta['target_source_ids']}",
        flush=True,
    )

    print("[temporal 2/5] Resolving full-volume temporal-v3 cache ...", flush=True)
    cache_path, payload, cache_notes = resolve_temporal_cache(
        args.temporal_cache,
        args.data_dir,
    )
    for note in cache_notes:
        print(f"[temporal-cache] {note}")

    if payload is None or cache_path is None:
        message = (
            "\nTEMPORAL CONTEXT NOT READY.\n"
            "Stage 08 prepared the spatial crop and full-movie target metadata, "
            "but no compatible full-volume temporal-v3 cache was available.\n"
            "Re-run with:\n"
            "  --temporal-cache <path-to-full-volume-temporal-v3-cache.pt>\n"
            "The cache must contain the detection graph, tracklets, hypothesis "
            "graph, and history-support tensors produced from the uncropped scene."
        )
        print(message)
        missing_report = {
            "format_version": 2,
            "stage": "08_temporal_merge_case_preparation",
            "temporal_ready": False,
            "target": target_meta,
            "cache_notes": cache_notes,
            "readiness_reason": "No compatible full-volume temporal-v3 cache.",
        }
        atomic_json(args.context_json, missing_report)
        if args.require_temporal_cache:
            raise RuntimeError(message)
        return None

    print("[temporal 3/5] Validating temporal coordinates / finding target-associated tracklets ...", flush=True)
    track_graph = _load_trackastra_graph(args.data_dir)
    coordinate_diagnostics = validate_temporal_cache_coordinate_frame(
        payload,
        target_meta,
        track_graph,
        spacing,
    )
    print(
        "[temporal] coordinate validation "
        f"median={coordinate_diagnostics['median_error_um']:.5f} um "
        f"max={coordinate_diagnostics['max_error_um']:.5f} um",
        flush=True,
    )

    anchors = find_anchor_tracklets(
        payload,
        target_meta,
        min_overlap=args.anchor_overlap,
    )

    coordinate_mode, _ = choose_coordinate_mode(
        args.temporal_coordinates,
        payload,
        target_meta,
        anchors,
    )
    target_um = target_ref_um(target_meta, coordinate_mode)
    distances = tracklet_distances_dref(
        payload,
        target_um,
        dref_um,
    )

    # If projected-component association is unavailable, use nearby current
    # tracklets as an explicit fallback anchor candidate. This keeps the script
    # inspectable rather than silently claiming a confident association.
    anchor_fallback_used = False
    if not anchors:
        tracklet_id = torch.as_tensor(payload["tracklet_id"]).long().cpu()
        offsets = torch.as_tensor(payload["node_time_offset"]).float().cpu()
        current_tracklets = set(
            int(value)
            for value in tracklet_id[
                offsets.abs() < 0.5
            ].tolist()
        )
        nearby_current = sorted(
            current_tracklets,
            key=lambda index: float(distances[index]),
        )
        if nearby_current and float(distances[nearby_current[0]]) <= 1.5:
            anchors = [int(nearby_current[0])]
            anchor_fallback_used = True
            print(
                "[temporal] WARNING: projected current-component association "
                "did not identify an anchor; using nearest current tracklet "
                "as an inspection-only fallback.",
                flush=True,
            )

    print(
        f"[temporal] coordinate_mode={coordinate_mode} "
        f"anchors={anchors}",
        flush=True,
    )

    print("[temporal 4/5] Extracting target-centred temporal ego graph ...", flush=True)
    selected_tracklets, distances = select_context_tracklets(
        payload,
        target_meta,
        dref_um=dref_um,
        coordinate_mode=coordinate_mode,
        anchors=anchors,
        radius_dref=args.context_radius_dref,
        min_tracklets=args.min_context_tracklets,
        max_tracklets=args.max_context_tracklets,
        hypothesis_hops=args.hypothesis_hops,
    )
    context, mapping = subset_temporal_context(
        payload,
        selected_tracklets,
    )

    report = temporal_context_report(
        payload,
        context,
        mapping,
        target_meta=target_meta,
        anchors=anchors,
        selected_tracklets=selected_tracklets,
        distances_dref=distances,
        coordinate_mode=coordinate_mode,
        coordinate_diagnostics=coordinate_diagnostics,
        cache_path=cache_path,
    )
    report["anchor_fallback_used"] = bool(anchor_fallback_used)
    report["context_selection"] = {
        "radius_dref": float(args.context_radius_dref),
        "min_context_tracklets": int(args.min_context_tracklets),
        "max_context_tracklets": int(args.max_context_tracklets),
        "hypothesis_hops": int(args.hypothesis_hops),
        "anchor_overlap_min": float(args.anchor_overlap),
    }
    report["source_paths"] = {
        "instance_movie": str(args.data_dir / "instance_movie.npy"),
        "gt_movie": str(args.data_dir / "gt_movie.npy"),
        "spatial_crop": str(args.output),
        "temporal_cache": str(cache_path),
    }

    artifact = {
        "format_version": 2,
        "kind": "stirnet_stage08_full_temporal_context",
        "report": report,
        "target": target_meta,
        "spacing_um": torch.tensor(spacing, dtype=torch.float32),
        "dref_um": torch.tensor(dref_um, dtype=torch.float32),
        "target_time_index": int(args.time_index),
        "target_gt_id": int(target_gt_id),
        "target_source_ids": torch.tensor(
            target_meta["target_source_ids"],
            dtype=torch.long,
        ),
        "anchor_original_tracklet_ids": torch.tensor(
            anchors,
            dtype=torch.long,
        ),
        "selected_original_tracklet_ids": torch.tensor(
            selected_tracklets,
            dtype=torch.long,
        ),
        **context,
    }

    print("[temporal 5/5] Saving temporal ego-graph artifact ...", flush=True)
    atomic_torch_save(args.context_output, artifact)
    atomic_json(args.context_json, report)
    print_temporal_report(report)

    if not report["temporal_ready"] and args.require_temporal_cache:
        raise RuntimeError(
            "A compatible temporal cache was found, but the selected target "
            "has no target-associated tracklet with past evidence. Inspect the "
            "full-volume visualization/cache before temporal overfit."
        )

    return report



# =============================================================================
# CONTROLLED TWO-INSTANCE ERROR + TRACKASTRA RERUN
# =============================================================================


def _probability_to_logit(
    value: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    value = value.float().clamp(eps, 1.0 - eps)
    return torch.log(value) - torch.log1p(-value)


def _oracle_geometry_from_debug_batch(batch) -> GeometryState:
    return GeometryState(
        foreground_logits=_probability_to_logit(
            batch.targets["foreground"]
        ),
        surface_logits=_probability_to_logit(
            batch.targets["surface"]
        ),
        separator_logits=_probability_to_logit(
            batch.targets["separator"]
        ),
        sdf=batch.targets["sdf"].float(),
        flow=batch.targets["flow"].float(),
        centroid_offset=batch.targets["centroid_offset"].float(),
        seed_logits=_probability_to_logit(batch.targets["seed"]),
        features=None,
        feature_spacing_um=batch.spacing_um,
    )


@torch.no_grad()
def _learned_geometry_from_stage01(
    batch,
    cfg,
    checkpoint_path: Path,
    device: torch.device,
):
    checkpoint = torch_load(checkpoint_path)
    stage = str(checkpoint.get("stage", ""))
    if stage not in {"joint", "full"}:
        raise RuntimeError(
            "Controlled learned-supervoxel mode requires a successful "
            f"Stage-01 joint/full checkpoint; found stage={stage!r}."
        )

    sample_index = batch.selection.get("candidate_index")
    checkpoint_selection = dict(
        checkpoint.get("sample_selection", {})
    )
    checkpoint_index = checkpoint_selection.get("candidate_index")
    if (
        checkpoint_index is not None
        and sample_index is not None
        and int(checkpoint_index) != int(sample_index)
    ):
        raise RuntimeError(
            "Stage-01 checkpoint/sample candidate mismatch: "
            f"checkpoint={checkpoint_index}, sample={sample_index}."
        )

    model = stage01.DenseGeometryOnlyModel(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    acquisition = model.acquisition(batch.spacing_um, batch.dref_um)
    stem = model.evidence_stem(batch.spatial_inputs, acquisition)
    _, decoded = model.spatial_backbone(
        stem,
        batch.spacing_um,
        acquisition,
        padding_mask=None,
    )
    geometry = model.geometry_decoder(decoded.d0, acquisition)

    del model, acquisition, stem, decoded
    return geometry


@torch.no_grad()
def build_controlled_safe_supervoxels(
    args: argparse.Namespace,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Build production safe supervoxels for the selected candidate-0 crop.

    This is only used to define a controlled spatial error.  In auto mode a
    candidate-0 Stage-01 checkpoint is used if it already exists; otherwise
    oracle geometry targets are passed through the REAL production watershed +
    face-level safety guard.  GT is never edited.
    """
    if not args.output.exists():
        raise FileNotFoundError(
            f"Missing Stage-08 spatial crop:\n  {args.output}\n"
            "Run Stage 08 preparation first."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    cfg = stage01.build_debug_config()
    batch = stage01.load_debug_batch(args.output, device)

    if int(batch.selection.get("candidate_index", -1)) != int(
        args.candidate_index
    ):
        raise RuntimeError(
            "Controlled split sample is not the requested candidate: "
            f"{batch.selection.get('candidate_index')}."
        )

    source = args.supervoxel_source
    if source == "auto":
        source = (
            "learned"
            if args.stage01_checkpoint.exists()
            else "oracle"
        )

    if source == "learned":
        if not args.stage01_checkpoint.exists():
            raise FileNotFoundError(
                f"Missing candidate-0 Stage-01 checkpoint:\n"
                f"  {args.stage01_checkpoint}\n"
                "Use --supervoxel-source oracle for immediate controlled "
                "error preparation, or first overfit candidate 0."
            )
        geometry = _learned_geometry_from_stage01(
            batch,
            cfg,
            args.stage01_checkpoint,
            device,
        )
    elif source == "oracle":
        geometry = _oracle_geometry_from_debug_batch(batch)
    else:
        raise ValueError(source)

    derived = build_geometry_derived_cache(
        geometry,
        cfg.partition,
        padding_mask=None,
    )
    watershed = LearnedGeometryWatershed(
        cfg.partition,
        cfg.geometry,
    ).to(device)

    labels = watershed(
        geometry,
        batch.spacing_um,
        batch.dref_um,
        derived_cache=derived,
    )[0]

    diagnostics = {
        "source": source,
        "supervoxel_count": int(labels.max().item()),
        "crop_shape_zyx": [
            int(v) for v in labels.shape
        ],
    }
    return (
        labels.detach().cpu().numpy().astype(np.int32, copy=False),
        source,
        diagnostics,
    )


def target_supervoxel_rows(
    safe_supervoxels: np.ndarray,
    gt_crop: np.ndarray,
    target_gt_id: int,
    spacing_um: np.ndarray,
    dref_um: float,
) -> list[dict[str, Any]]:
    target = gt_crop == int(target_gt_id)
    target_voxels = max(int(target.sum()), 1)

    rows: list[dict[str, Any]] = []
    for supervoxel_id in np.unique(safe_supervoxels[target]):
        supervoxel_id = int(supervoxel_id)
        if supervoxel_id <= 0:
            continue

        sv = safe_supervoxels == supervoxel_id
        sv_volume = int(sv.sum())
        overlap = int(np.count_nonzero(sv & target))
        purity = overlap / max(sv_volume, 1)
        target_fraction = overlap / target_voxels

        coords = np.argwhere(sv & target)
        centroid_vox = (
            coords.astype(np.float32).mean(axis=0)
            if coords.size
            else np.zeros(3, dtype=np.float32)
        )
        centroid_um = centroid_vox * spacing_um.astype(np.float32)

        rows.append(
            {
                "supervoxel_id": supervoxel_id,
                "volume_voxels": sv_volume,
                "target_overlap_voxels": overlap,
                "purity_to_target": float(purity),
                "fraction_of_target": float(target_fraction),
                "centroid_vox_zyx": [
                    float(v) for v in centroid_vox.tolist()
                ],
                "centroid_um_zyx": [
                    float(v) for v in centroid_um.tolist()
                ],
                "eligible_seed": bool(
                    overlap >= DEFAULT_MIN_TARGET_SV_OVERLAP_VOXELS
                    and purity >= DEFAULT_MIN_TARGET_SV_PURITY
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            row["fraction_of_target"],
            row["purity_to_target"],
            -row["supervoxel_id"],
        ),
        reverse=True,
    )

    for row in rows:
        row["distance_to_largest_dref"] = 0.0
    if rows:
        ref = np.asarray(rows[0]["centroid_um_zyx"], dtype=np.float32)
        for row in rows:
            point = np.asarray(row["centroid_um_zyx"], dtype=np.float32)
            row["distance_to_largest_dref"] = float(
                np.linalg.norm(point - ref)
                / max(float(dref_um), 1e-6)
            )
    return rows


def print_target_supervoxels(
    rows: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 104)
    print("Target GT safe-supervoxel candidates for controlled split")
    print("=" * 104)
    print(
        f"{'SV':>5s} {'vox':>8s} {'GTvox':>8s} "
        f"{'purity':>8s} {'GTfrac':>8s} {'sepLargest':>10s} {'seed':>6s}"
    )
    print("-" * 104)
    for row in rows:
        print(
            f"{row['supervoxel_id']:5d} "
            f"{row['volume_voxels']:8d} "
            f"{row['target_overlap_voxels']:8d} "
            f"{row['purity_to_target']:8.3f} "
            f"{row['fraction_of_target']:8.3f} "
            f"{row['distance_to_largest_dref']:10.3f} "
            f"{str(row['eligible_seed']):>6s}"
        )
    print("=" * 104)


def choose_split_seed_supervoxels(
    rows: list[dict[str, Any]],
    forced: tuple[int, int] | None,
    dref_um: float,
) -> tuple[int, int, dict[str, Any]]:
    by_id = {
        int(row["supervoxel_id"]): row
        for row in rows
    }

    if forced is not None:
        a, b = (int(forced[0]), int(forced[1]))
        if a == b:
            raise ValueError("Split seed supervoxels must be different.")
        if a not in by_id or b not in by_id:
            raise ValueError(
                f"Requested seed SVs {a},{b} are not target-overlap "
                f"supervoxels. Available={sorted(by_id)}"
            )
        return a, b, {
            "selection": "manual",
            "pair_score": None,
        }

    eligible = [
        row for row in rows if row["eligible_seed"]
    ]
    if len(eligible) < 2:
        raise RuntimeError(
            "The selected GT cell does not contain at least two sufficiently "
            "large/pure safe supervoxels. Inspect the table and optionally "
            "supply --split-seed-supervoxels A B."
        )

    best_pair = None
    best_score = -float("inf")
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            a = eligible[i]
            b = eligible[j]
            ca = np.asarray(a["centroid_um_zyx"], dtype=np.float32)
            cb = np.asarray(b["centroid_um_zyx"], dtype=np.float32)
            separation_dref = float(
                np.linalg.norm(ca - cb)
                / max(float(dref_um), 1e-6)
            )
            fa = float(a["fraction_of_target"])
            fb = float(b["fraction_of_target"])
            balance = min(fa, fb) / max(max(fa, fb), 1e-8)

            # Prefer two substantial target pieces and, secondarily, spatially
            # separated seeds. This is deterministic and uses safe SVs only.
            score = (
                4.0 * min(fa, fb)
                + 1.25 * min(separation_dref, 3.0)
                + 0.75 * balance
                + 0.25 * (fa + fb)
            )
            if score > best_score:
                best_score = score
                best_pair = (
                    int(a["supervoxel_id"]),
                    int(b["supervoxel_id"]),
                    separation_dref,
                    balance,
                )

    assert best_pair is not None
    a, b, separation_dref, balance = best_pair
    return a, b, {
        "selection": "automatic",
        "pair_score": float(best_score),
        "seed_separation_dref": float(separation_dref),
        "seed_balance": float(balance),
    }


def _target_source_component(
    current_frame: np.ndarray,
    gt_frame: np.ndarray,
    target_gt_id: int,
    source_id: int,
) -> np.ndarray:
    source = current_frame == int(source_id)
    components, count = ndi.label(
        source,
        structure=ndi.generate_binary_structure(3, 1),
    )
    target = gt_frame == int(target_gt_id)

    if count == 0:
        raise RuntimeError(
            f"Source label {source_id} is absent from target frame."
        )

    best_component = None
    best_overlap = -1
    for component_id in range(1, count + 1):
        mask = components == component_id
        overlap = int(np.count_nonzero(mask & target))
        if overlap > best_overlap:
            best_overlap = overlap
            best_component = component_id

    if best_component is None or best_overlap <= 0:
        raise RuntimeError(
            f"No connected source-{source_id} component overlaps GT "
            f"{target_gt_id}."
        )
    return components == int(best_component)


def _multi_source_geodesic_partition(
    component_mask: np.ndarray,
    seed_a: np.ndarray,
    seed_b: np.ndarray,
    spacing_um: np.ndarray,
) -> np.ndarray:
    """Two-way 6-connected physical geodesic Voronoi inside one component.

    Output labels are 1/2. Starting from whole safe-supervoxel seed regions
    guarantees both pieces remain connected to their selected safe SV.
    """
    if component_mask.shape != seed_a.shape or seed_a.shape != seed_b.shape:
        raise ValueError("component and seed masks must share shape")
    if not seed_a.any() or not seed_b.any():
        raise RuntimeError("Both controlled split seeds must be non-empty.")
    if np.any(seed_a & seed_b):
        raise RuntimeError("Controlled split seeds overlap.")

    bbox = ndi.find_objects(component_mask.astype(np.uint8))
    if not bbox or bbox[0] is None:
        raise RuntimeError("Target source component is empty.")
    region = bbox[0]

    mask = component_mask[region]
    a = seed_a[region] & mask
    b = seed_b[region] & mask
    if not a.any() or not b.any():
        raise RuntimeError(
            "At least one safe-supervoxel seed does not intersect the "
            "target source component."
        )

    owner = np.zeros(mask.shape, dtype=np.uint8)
    distance = np.full(mask.shape, np.inf, dtype=np.float64)
    queue: list[tuple[float, int, int, int, int]] = []

    for label, seed in ((1, a), (2, b)):
        for z, y, x in np.argwhere(seed).tolist():
            owner[z, y, x] = label
            distance[z, y, x] = 0.0
            heapq.heappush(queue, (0.0, label, z, y, x))

    neighbours = (
        (-1, 0, 0, float(spacing_um[0])),
        (1, 0, 0, float(spacing_um[0])),
        (0, -1, 0, float(spacing_um[1])),
        (0, 1, 0, float(spacing_um[1])),
        (0, 0, -1, float(spacing_um[2])),
        (0, 0, 1, float(spacing_um[2])),
    )

    zmax, ymax, xmax = mask.shape
    tolerance = 1e-9

    while queue:
        dist, label, z, y, x = heapq.heappop(queue)
        if dist > distance[z, y, x] + tolerance:
            continue
        if owner[z, y, x] != label:
            continue

        for dz, dy, dx, cost in neighbours:
            nz, ny, nx = z + dz, y + dy, x + dx
            if (
                nz < 0
                or ny < 0
                or nx < 0
                or nz >= zmax
                or ny >= ymax
                or nx >= xmax
                or not mask[nz, ny, nx]
            ):
                continue

            candidate = dist + cost
            current = distance[nz, ny, nx]
            current_owner = int(owner[nz, ny, nx])
            better = candidate < current - tolerance
            tie = (
                abs(candidate - current) <= tolerance
                and (current_owner == 0 or label < current_owner)
            )
            if better or tie:
                distance[nz, ny, nx] = candidate
                owner[nz, ny, nx] = label
                heapq.heappush(
                    queue,
                    (candidate, label, nz, ny, nx),
                )

    if np.any(mask & (owner == 0)):
        raise RuntimeError(
            "Geodesic split failed to reach the full target source component."
        )

    result = np.zeros(component_mask.shape, dtype=np.uint8)
    result[region] = owner
    result[~component_mask] = 0
    return result


def _copy_movie_memmap(
    source: np.ndarray,
    output: Path,
) -> np.memmap:
    output.parent.mkdir(parents=True, exist_ok=True)
    movie = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=source.dtype,
        shape=source.shape,
    )
    for time_index in range(source.shape[0]):
        movie[time_index] = np.asarray(source[time_index])
    movie.flush()
    return movie


def prepare_controlled_split(
    args: argparse.Namespace,
    spatial_report: dict[str, Any],
) -> dict[str, Any]:
    print("\n" + "=" * 112)
    print("Stage 08C — controlled TWO-instance spatial error from safe supervoxels")
    print("=" * 112)

    instance_movie, gt_movie = load_full_movies(args.data_dir)
    spacing = np.asarray(
        spatial_report["spacing_zyx_um"],
        dtype=np.float32,
    )
    dref_um = float(spatial_report["dref_um"])
    target_gt_id = int(spatial_report["target_gt_id"])
    target_time = int(args.time_index)

    safe_sv, sv_source, sv_diagnostics = (
        build_controlled_safe_supervoxels(args)
    )

    crop_payload = torch_load(args.output)
    gt_crop = (
        torch.as_tensor(crop_payload["gt_labels"])
        .long()
        .cpu()
        .numpy()
    )

    rows = target_supervoxel_rows(
        safe_sv,
        gt_crop,
        target_gt_id,
        spacing,
        dref_um,
    )
    print_target_supervoxels(rows)

    forced = (
        tuple(args.split_seed_supervoxels)
        if args.split_seed_supervoxels is not None
        else None
    )
    seed_a_sv, seed_b_sv, seed_selection = (
        choose_split_seed_supervoxels(
            rows,
            forced,
            dref_um,
        )
    )

    original_frame = np.asarray(instance_movie[target_time])
    gt_frame = np.asarray(gt_movie[target_time])
    target_meta = full_target_metadata(
        instance_movie,
        gt_movie,
        time_index=target_time,
        target_gt_id=target_gt_id,
        spacing_um=spacing,
    )

    if not target_meta["target_source_ids"]:
        raise RuntimeError(
            "Selected GT has no positive source/current component."
        )
    source_id = int(
        max(
            target_meta["source_overlaps"],
            key=lambda row: row["overlap_voxels"],
        )["source_id"]
    )
    source_component = _target_source_component(
        original_frame,
        gt_frame,
        target_gt_id,
        source_id,
    )

    crop_slices = tuple(
        slice(int(pair[0]), int(pair[1]))
        for pair in spatial_report["crop_slices_zyx"]
    )
    if tuple(
        int(s.stop) - int(s.start)
        for s in crop_slices
    ) != tuple(safe_sv.shape):
        raise RuntimeError(
            "Saved crop_slices_zyx do not match safe-supervoxel crop shape."
        )

    seed_a_full = np.zeros(original_frame.shape, dtype=bool)
    seed_b_full = np.zeros(original_frame.shape, dtype=bool)
    seed_a_full[crop_slices] = safe_sv == int(seed_a_sv)
    seed_b_full[crop_slices] = safe_sv == int(seed_b_sv)
    seed_a_full &= source_component
    seed_b_full &= source_component

    partition = _multi_source_geodesic_partition(
        source_component,
        seed_a_full,
        seed_b_full,
        spacing,
    )
    part_a = partition == 1
    part_b = partition == 2

    count_a = int(part_a.sum())
    count_b = int(part_b.sum())
    total = count_a + count_b
    if total != int(source_component.sum()):
        raise RuntimeError(
            "Controlled split changed foreground support unexpectedly."
        )

    # Keep the larger piece on the original ID to minimize unrelated ID churn.
    if count_b > count_a:
        part_a, part_b = part_b, part_a
        count_a, count_b = count_b, count_a
        seed_a_sv, seed_b_sv = seed_b_sv, seed_a_sv

    fraction_a = count_a / max(total, 1)
    fraction_b = count_b / max(total, 1)
    if min(fraction_a, fraction_b) < float(args.min_split_fraction):
        raise RuntimeError(
            "Controlled safe-supervoxel split is too imbalanced: "
            f"{fraction_a:.3f}/{fraction_b:.3f}. "
            "Choose different --split-seed-supervoxels."
        )

    global_max_label = int(
        max(
            np.max(np.asarray(instance_movie[t]))
            for t in range(instance_movie.shape[0])
        )
    )
    new_source_id = global_max_label + 1
    if np.issubdtype(instance_movie.dtype, np.integer):
        dtype_max = int(np.iinfo(instance_movie.dtype).max)
        if new_source_id > dtype_max:
            raise RuntimeError(
                f"Cannot allocate controlled label {new_source_id} in "
                f"dtype {instance_movie.dtype}; maximum is {dtype_max}."
            )

    controlled = _copy_movie_memmap(
        instance_movie,
        args.controlled_input_movie,
    )
    controlled_frame = np.asarray(controlled[target_time])
    controlled_frame[source_component] = 0
    controlled_frame[part_a] = int(source_id)
    controlled_frame[part_b] = int(new_source_id)
    controlled[target_time] = controlled_frame
    controlled.flush()

    # Strict invariants.
    controlled_check = np.load(
        args.controlled_input_movie,
        mmap_mode="r",
    )
    for time_index in range(instance_movie.shape[0]):
        original_frame_check = np.asarray(instance_movie[time_index])
        controlled_frame_check = np.asarray(controlled_check[time_index])
        if not np.array_equal(
            controlled_frame_check > 0,
            original_frame_check > 0,
        ):
            raise RuntimeError(
                "Controlled split altered foreground/background support "
                f"in frame {time_index}."
            )
        if (
            time_index != target_time
            and not np.array_equal(
                controlled_frame_check,
                original_frame_check,
            )
        ):
            raise RuntimeError(
                f"Controlled split unexpectedly changed frame {time_index}."
            )

    target = gt_frame == target_gt_id
    target_piece_a_fraction = float(
        np.count_nonzero(part_a & target)
        / max(np.count_nonzero(target), 1)
    )
    target_piece_b_fraction = float(
        np.count_nonzero(part_b & target)
        / max(np.count_nonzero(target), 1)
    )

    bbox = ndi.find_objects(source_component.astype(np.uint8))[0]
    bbox_pairs = _slice_pairs(bbox)

    split_report = {
        "format_version": 1,
        "stage": "08C_controlled_split",
        "candidate_index": int(args.candidate_index),
        "target_gt_id": target_gt_id,
        "target_time_index": target_time,
        "supervoxel_source": sv_source,
        "supervoxel_diagnostics": sv_diagnostics,
        "target_supervoxels": rows,
        "seed_supervoxels": [
            int(seed_a_sv),
            int(seed_b_sv),
        ],
        "seed_selection": seed_selection,
        "original_source_id": int(source_id),
        "new_source_id": int(new_source_id),
        "component_bbox_zyx": bbox_pairs,
        "component_voxels": int(total),
        "part_a_voxels": int(count_a),
        "part_b_voxels": int(count_b),
        "part_a_fraction": float(fraction_a),
        "part_b_fraction": float(fraction_b),
        "part_a_gt_fraction": target_piece_a_fraction,
        "part_b_gt_fraction": target_piece_b_fraction,
        "controlled_input_movie": str(
            args.controlled_input_movie
        ),
        "invariant_foreground_unchanged": True,
        "invariant_other_frames_unchanged": True,
    }

    # Save only a compact target-component bbox mask, not another full movie.
    compact = {
        "format_version": 1,
        "report": split_report,
        "bbox_zyx": bbox_pairs,
        "controlled_partition_bbox": torch.from_numpy(
            partition[bbox].astype(np.uint8, copy=True)
        ),
        "safe_supervoxels_crop": torch.from_numpy(
            safe_sv.astype(np.int32, copy=True)
        ),
        "crop_slices_zyx": spatial_report["crop_slices_zyx"],
    }
    atomic_torch_save(args.controlled_split_artifact, compact)
    atomic_json(args.controlled_split_json, split_report)

    print(
        f"Safe-SV seeds                  : {seed_a_sv}, {seed_b_sv}"
    )
    print(
        f"Controlled source labels       : {source_id}, {new_source_id}"
    )
    print(
        f"Split fractions                : "
        f"{fraction_a:.3f} / {fraction_b:.3f}"
    )
    print(
        f"Controlled movie               : {args.controlled_input_movie}"
    )
    print("=" * 112)
    return split_report


def run_controlled_trackastra(
    args: argparse.Namespace,
) -> tuple[Any, np.ndarray]:
    args.controlled_graph.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        args.controlled_graph.exists()
        and args.controlled_tracked_movie.exists()
        and not args.force_trackastra
    ):
        print(
            "\n[Stage 08D] Reusing existing controlled Trackastra result."
        )
        with args.controlled_graph.open("rb") as handle:
            graph = pickle.load(handle)
        masks = np.load(
            args.controlled_tracked_movie,
            mmap_mode="r",
        )
        return graph, masks

    if not args.controlled_input_movie.exists():
        raise FileNotFoundError(
            f"Missing controlled input movie:\n"
            f"  {args.controlled_input_movie}"
        )

    try:
        from trackastra.model import Trackastra
    except ImportError as exc:
        raise RuntimeError(
            "Trackastra is not installed in this environment. "
            "Install/activate the same Trackastra environment used to create "
            "the baseline track_graph.pkl."
        ) from exc

    raw_path = args.data_dir / "raw_movie.npy"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    raw_movie = np.load(raw_path, mmap_mode="r")
    controlled_movie = np.load(
        args.controlled_input_movie,
        mmap_mode="r",
    )

    print("\n" + "=" * 112)
    print("Stage 08D — rerunning Trackastra on controlled TWO-instance movie")
    print("=" * 112)
    print(f"Model                          : {args.trackastra_model}")
    print(f"Mode                           : {args.trackastra_mode}")
    print(f"Device                         : {args.trackastra_device}")
    print(f"Raw                            : {raw_path}")
    print(f"Masks                          : {args.controlled_input_movie}")
    print("=" * 112)

    started = time.perf_counter()
    model = Trackastra.from_pretrained(
        args.trackastra_model,
        device=args.trackastra_device,
    )
    track_graph, masks_tracked = model.track(
        raw_movie,
        controlled_movie,
        mode=args.trackastra_mode,
    )

    with args.controlled_graph.open("wb") as handle:
        pickle.dump(track_graph, handle)

    np.save(
        args.controlled_tracked_movie,
        np.asarray(masks_tracked),
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"[Trackastra] nodes={track_graph.number_of_nodes()} "
        f"edges={track_graph.number_of_edges()} "
        f"time={_format_duration(time.perf_counter() - started)}"
    )
    print(f"[Trackastra] graph  : {args.controlled_graph}")
    print(f"[Trackastra] masks  : {args.controlled_tracked_movie}")

    return (
        track_graph,
        np.load(args.controlled_tracked_movie, mmap_mode="r"),
    )


def _graph_label_match_fraction(
    track_graph,
    movie: np.ndarray,
) -> float:
    matches = 0
    valid = 0
    shape = np.asarray(movie.shape[-3:], dtype=np.int64)

    for _, node in track_graph.nodes(data=True):
        time_index = int(node["time"])
        label_id = int(node["label"])
        coord = np.rint(
            np.asarray(node["coords"], dtype=np.float32)
        ).astype(np.int64)
        if (
            time_index < 0
            or time_index >= movie.shape[0]
            or np.any(coord < 0)
            or np.any(coord >= shape)
        ):
            continue
        valid += 1
        if int(movie[(time_index, *coord.tolist())]) == label_id:
            matches += 1

    return matches / max(valid, 1)


def choose_graph_mask_movie(
    args: argparse.Namespace,
    track_graph,
) -> tuple[np.ndarray, Path, dict[str, float]]:
    input_movie = np.load(
        args.controlled_input_movie,
        mmap_mode="r",
    )
    tracked_movie = np.load(
        args.controlled_tracked_movie,
        mmap_mode="r",
    )

    input_match = _graph_label_match_fraction(
        track_graph,
        input_movie,
    )
    tracked_match = _graph_label_match_fraction(
        track_graph,
        tracked_movie,
    )

    diagnostics = {
        "controlled_input_label_match": float(input_match),
        "trackastra_tracked_label_match": float(tracked_match),
    }
    if tracked_match >= input_match:
        return tracked_movie, args.controlled_tracked_movie, diagnostics
    return input_movie, args.controlled_input_movie, diagnostics


def _mean_velocity_from_track_graph(
    track_graph,
    node_position_abs_um: dict[int, np.ndarray],
    node_id: int,
    neighbours: list[int],
    *,
    forward: bool,
) -> np.ndarray:
    if not neighbours:
        return np.zeros(3, dtype=np.float32)

    time0 = int(track_graph.nodes[node_id]["time"])
    position0 = node_position_abs_um[node_id]
    values = []
    for other in neighbours:
        other = int(other)
        dt = abs(
            int(track_graph.nodes[other]["time"]) - time0
        )
        if dt <= 0:
            continue
        delta = node_position_abs_um[other] - position0
        values.append((delta if forward else -delta) / dt)

    return (
        np.mean(values, axis=0).astype(np.float32)
        if values
        else np.zeros(3, dtype=np.float32)
    )


def build_controlled_temporal_cache(
    args: argparse.Namespace,
    track_graph,
    graph_mask_movie: np.ndarray,
    spatial_report: dict[str, Any],
) -> dict[str, Any]:
    if (
        args.controlled_cache.exists()
        and not args.force_controlled_cache
    ):
        print(
            "\n[Stage 08E] Reusing existing controlled temporal-v3 cache."
        )
        return load_cache(
            args.controlled_cache,
            map_location="cpu",
        )

    raw_movie = np.load(
        args.data_dir / "raw_movie.npy",
        mmap_mode="r",
    )
    marker_path = args.data_dir / "markers_movie.npy"
    markers_movie = (
        np.load(marker_path, mmap_mode="r")
        if marker_path.exists()
        else None
    )
    gt_movie = np.load(
        args.data_dir / "gt_movie.npy",
        mmap_mode="r",
    )

    spacing = np.asarray(
        spatial_report["spacing_zyx_um"],
        dtype=np.float32,
    )
    dref_um = float(spatial_report["dref_um"])
    target_time = int(args.time_index)

    roi, roi_low, roi_high = _roi_with_all_cells(
        graph_mask_movie,
        gt_movie,
        spacing,
    )
    roi_shape = (roi_high - roi_low).astype(np.int64)
    roi_center_um = (
        0.5
        * (roi_shape.astype(np.float32) - 1.0)
        * spacing
    )
    full_shape = np.asarray(
        graph_mask_movie.shape[-3:],
        dtype=np.float32,
    )

    node_position_abs_um = {
        int(node_id): (
            np.asarray(node["coords"], dtype=np.float32)
            * spacing
        )
        for node_id, node in track_graph.nodes(data=True)
    }

    records: list[DetectionRecord] = []

    print("\n" + "=" * 112)
    print("Stage 08E — rebuilding STIR-Net temporal-v3 cache from controlled Trackastra")
    print("=" * 112)

    for local_time in range(len(graph_mask_movie)):
        print(
            f"[temporal-cache] frame {local_time + 1}/{len(graph_mask_movie)}",
            flush=True,
        )
        labels = np.asarray(
            graph_mask_movie[local_time][roi]
        ).astype(np.int32, copy=False)

        raw_norm = robust_normalize(
            np.asarray(raw_movie[local_time][roi])
        )
        marker = (
            (
                np.asarray(markers_movie[local_time][roi]) > 0
            ).astype(np.float32)
            if markers_movie is not None
            else np.zeros(labels.shape, dtype=np.float32)
        )

        metadata = extract_instance_metadata(
            labels,
            raw_norm,
            tuple(float(v) for v in spacing),
            dref_um,
            marker,
        )
        ids = metadata.ids.numpy()
        features = metadata.features.numpy()
        id_to_row = {
            int(instance_id): row
            for row, instance_id in enumerate(ids.tolist())
        }

        for node_id, node_data in track_graph.nodes(data=True):
            if int(node_data["time"]) != local_time:
                continue

            label_id = int(node_data["label"])
            if label_id not in id_to_row:
                continue

            row = id_to_row[label_id]
            coords_full = np.asarray(
                node_data["coords"],
                dtype=np.float32,
            )
            coords_roi = coords_full - roi_low.astype(np.float32)
            position_relative_um = (
                coords_roi * spacing - roi_center_um
            )
            component_voxels = int(
                np.count_nonzero(labels == label_id)
            )
            feature = features[row]

            lower_full_um = coords_full * spacing
            upper_full_um = (
                full_shape - 1.0 - coords_full
            ) * spacing
            lower_roi_um = coords_roi * spacing
            upper_roi_um = (
                roi_shape.astype(np.float32) - 1.0 - coords_roi
            ) * spacing

            predecessors = [
                int(value)
                for value in track_graph.predecessors(node_id)
            ]
            successors = [
                int(value)
                for value in track_graph.successors(node_id)
            ]

            distance_to_volume_boundary = float(
                np.min(
                    np.concatenate(
                        [lower_full_um, upper_full_um]
                    )
                )
            )

            # Build the compact historical instance grid from the CONTROLLED
            # Trackastra-compatible segmentation itself. This avoids stale
            # node-id assumptions from the baseline cache.
            history_grid, history_valid = (
                build_historical_instance_grid(
                    raw_norm,
                    labels,
                    label_id,
                    tuple(float(v) for v in spacing),
                    dref_um,
                    center_um=coords_roi * spacing,
                )
            )

            records.append(
                DetectionRecord(
                    node_id=int(node_id),
                    time_offset=local_time - target_time,
                    position_um=tuple(
                        position_relative_um.tolist()
                    ),
                    physical_volume_um3=(
                        component_voxels
                        * float(np.prod(spacing))
                    ),
                    bbox_um=tuple(
                        (feature[1:4] * dref_um).tolist()
                    ),
                    pca_axes_um=tuple(
                        (feature[4:7] * dref_um).tolist()
                    ),
                    elongation=float(feature[7]),
                    flatness=float(feature[8]),
                    solidity=float(feature[9]),
                    compactness=float(feature[10]),
                    intensity_mean=float(feature[11]),
                    intensity_std=float(feature[12]),
                    backward_velocity_um=tuple(
                        _mean_velocity_from_track_graph(
                            track_graph,
                            node_position_abs_um,
                            int(node_id),
                            predecessors,
                            forward=False,
                        ).tolist()
                    ),
                    forward_velocity_um=tuple(
                        _mean_velocity_from_track_graph(
                            track_graph,
                            node_position_abs_um,
                            int(node_id),
                            successors,
                            forward=True,
                        ).tolist()
                    ),
                    distance_to_volume_boundary_um=(
                        distance_to_volume_boundary
                    ),
                    distance_to_patch_boundary_um=float(
                        np.min(
                            np.concatenate(
                                [lower_roi_um, upper_roi_um]
                            )
                        )
                    ),
                    boundary_related=(
                        distance_to_volume_boundary <= 4.0
                    ),
                    instance_grid=history_grid,
                    history_valid=bool(history_valid),
                )
            )

    associations: list[AssociationRecord] = []
    for source, destination, edge_data in track_graph.edges(data=True):
        source = int(source)
        destination = int(destination)
        score = edge_data.get("weight")
        associations.append(
            AssociationRecord(
                src_node_id=source,
                dst_node_id=destination,
                score=(
                    None
                    if score is None
                    else float(score)
                ),
                relation=(
                    "division"
                    if track_graph.out_degree(source) > 1
                    else "temporal"
                ),
            )
        )

    current_target = np.asarray(
        graph_mask_movie[target_time][roi]
    ).astype(np.int32, copy=True)

    temporal = build_temporal_graph(
        records,
        associations,
        dref_um=dref_um,
        temporal_radius=2,
        k_spatial_neighbors=6,
        spatial_radius_dref=2.5,
        current_labels=current_target,
        spacing_um=tuple(float(v) for v in spacing),
    )
    save_cache(args.controlled_cache, temporal)

    print(
        f"[temporal-cache] records={len(records)} "
        f"tracklets={len(temporal['temporal_ref_um'])} "
        f"candidate_edges={temporal['graph_edge_index'].shape[1]} "
        f"hypothesis_edges={temporal['hypothesis_edge_index'].shape[1]}"
    )
    print(f"[temporal-cache] saved: {args.controlled_cache}")
    return temporal


def _piece_graph_component_map(
    controlled_input_movie: np.ndarray,
    graph_mask_movie: np.ndarray,
    *,
    time_index: int,
    controlled_source_ids: list[int],
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    input_frame = np.asarray(
        controlled_input_movie[time_index]
    )
    graph_frame = np.asarray(
        graph_mask_movie[time_index]
    )

    for source_id in controlled_source_ids:
        mask = input_frame == int(source_id)
        values, counts = np.unique(
            graph_frame[mask],
            return_counts=True,
        )
        positive = [
            (int(value), int(count))
            for value, count in zip(
                values.tolist(),
                counts.tolist(),
            )
            if int(value) > 0
        ]
        positive.sort(
            key=lambda pair: pair[1],
            reverse=True,
        )
        mapping[str(int(source_id))] = {
            "overlaps": [
                {
                    "graph_component_id": value,
                    "voxels": count,
                    "fraction_of_piece": (
                        count / max(int(mask.sum()), 1)
                    ),
                }
                for value, count in positive
            ],
            "dominant_graph_component_id": (
                positive[0][0]
                if positive
                else None
            ),
        }
    return mapping


def prepare_controlled_temporal_context(
    args: argparse.Namespace,
    spatial_report: dict[str, Any],
    split_report: dict[str, Any],
    track_graph,
    graph_mask_movie: np.ndarray,
    graph_mask_path: Path,
    temporal_payload: dict[str, Any],
    graph_mask_diagnostics: dict[str, float],
) -> dict[str, Any]:
    gt_movie = np.load(
        args.data_dir / "gt_movie.npy",
        mmap_mode="r",
    )
    spacing = np.asarray(
        spatial_report["spacing_zyx_um"],
        dtype=np.float32,
    )
    dref_um = float(spatial_report["dref_um"])
    target_gt_id = int(spatial_report["target_gt_id"])

    target_meta = full_target_metadata(
        graph_mask_movie,
        gt_movie,
        time_index=args.time_index,
        target_gt_id=target_gt_id,
        spacing_um=spacing,
    )

    coordinate_diagnostics = (
        validate_temporal_cache_coordinate_frame(
            temporal_payload,
            target_meta,
            track_graph,
            spacing,
        )
    )

    anchors = find_anchor_tracklets(
        temporal_payload,
        target_meta,
        min_overlap=args.anchor_overlap,
    )
    coordinate_mode, _ = choose_coordinate_mode(
        args.temporal_coordinates,
        temporal_payload,
        target_meta,
        anchors,
    )
    target_um = target_ref_um(
        target_meta,
        coordinate_mode,
    )
    distances = tracklet_distances_dref(
        temporal_payload,
        target_um,
        dref_um,
    )

    selected_tracklets, distances = select_context_tracklets(
        temporal_payload,
        target_meta,
        dref_um=dref_um,
        coordinate_mode=coordinate_mode,
        anchors=anchors,
        radius_dref=args.context_radius_dref,
        min_tracklets=args.min_context_tracklets,
        max_tracklets=args.max_context_tracklets,
        hypothesis_hops=args.hypothesis_hops,
    )
    context, mapping = subset_temporal_context(
        temporal_payload,
        selected_tracklets,
    )

    report = temporal_context_report(
        temporal_payload,
        context,
        mapping,
        target_meta=target_meta,
        anchors=anchors,
        selected_tracklets=selected_tracklets,
        distances_dref=distances,
        coordinate_mode=coordinate_mode,
        coordinate_diagnostics=coordinate_diagnostics,
        cache_path=args.controlled_cache,
    )

    controlled_input = np.load(
        args.controlled_input_movie,
        mmap_mode="r",
    )
    controlled_source_ids = [
        int(split_report["original_source_id"]),
        int(split_report["new_source_id"]),
    ]
    component_map = _piece_graph_component_map(
        controlled_input,
        graph_mask_movie,
        time_index=args.time_index,
        controlled_source_ids=controlled_source_ids,
    )

    # Associate graph/current component IDs back to tracklets.
    per_piece_tracklets: dict[str, list[dict[str, Any]]] = {}
    best_component = torch.as_tensor(
        temporal_payload["best_current_component_id"]
    ).long().cpu()
    best_overlap = torch.as_tensor(
        temporal_payload["best_component_overlap"]
    ).float().cpu()
    tracklet_id = torch.as_tensor(
        temporal_payload["tracklet_id"]
    ).long().cpu()
    offsets = torch.as_tensor(
        temporal_payload["node_time_offset"]
    ).float().cpu()

    for source_id in controlled_source_ids:
        dominant = component_map[str(source_id)][
            "dominant_graph_component_id"
        ]
        matches: list[dict[str, Any]] = []
        if dominant is not None:
            for tracklet in range(best_component.numel()):
                if (
                    int(best_component[tracklet]) == int(dominant)
                    and float(best_overlap[tracklet])
                    >= args.anchor_overlap
                ):
                    times = sorted(
                        set(
                            int(round(float(value)))
                            for value in offsets[
                                tracklet_id == tracklet
                            ].tolist()
                        )
                    )
                    matches.append(
                        {
                            "tracklet_id": int(tracklet),
                            "best_component_overlap": float(
                                best_overlap[tracklet]
                            ),
                            "time_offsets": times,
                            "has_past": bool(
                                any(value < 0 for value in times)
                            ),
                            "has_current": bool(
                                any(value == 0 for value in times)
                            ),
                            "has_future": bool(
                                any(value > 0 for value in times)
                            ),
                        }
                    )
        per_piece_tracklets[str(source_id)] = matches

    piece_tracklet_coverage = all(
        bool(per_piece_tracklets[str(source_id)])
        for source_id in controlled_source_ids
    )
    any_piece_has_past = any(
        row["has_past"]
        for rows in per_piece_tracklets.values()
        for row in rows
    )
    split_visible_in_graph = (
        len(set(
            component_map[str(source_id)][
                "dominant_graph_component_id"
            ]
            for source_id in controlled_source_ids
            if component_map[str(source_id)][
                "dominant_graph_component_id"
            ] is not None
        )) >= 2
    )

    controlled_ready = bool(
        len(target_meta["target_source_ids"]) >= 2
        and split_visible_in_graph
        and piece_tracklet_coverage
        and any_piece_has_past
    )

    report["controlled_error"] = {
        "controlled_source_ids": controlled_source_ids,
        "graph_mask_path": str(graph_mask_path),
        "graph_mask_label_match": graph_mask_diagnostics,
        "piece_graph_component_map": component_map,
        "piece_tracklets": per_piece_tracklets,
        "split_visible_in_graph": split_visible_in_graph,
        "each_piece_has_tracklet": piece_tracklet_coverage,
        "any_piece_has_past": any_piece_has_past,
    }
    report["controlled_temporal_ready"] = controlled_ready
    report["readiness_reason"] = (
        "Controlled TWO-instance split is visible to Trackastra/STIR-Net; "
        "both pieces have temporal tracklets and at least one has past evidence."
        if controlled_ready
        else (
            "Controlled temporal case is incomplete. Inspect split-to-graph "
            "component mapping and piece tracklets before Stage 09."
        )
    )
    report["source_paths"] = {
        "controlled_input_movie": str(
            args.controlled_input_movie
        ),
        "trackastra_tracked_movie": str(
            args.controlled_tracked_movie
        ),
        "graph_mask_movie": str(graph_mask_path),
        "track_graph": str(args.controlled_graph),
        "temporal_cache": str(args.controlled_cache),
        "controlled_split": str(
            args.controlled_split_artifact
        ),
    }

    artifact = {
        "format_version": 1,
        "kind": "stirnet_stage08_controlled_temporal_context",
        "report": report,
        "target": target_meta,
        "spacing_um": torch.tensor(
            spacing,
            dtype=torch.float32,
        ),
        "dref_um": torch.tensor(
            dref_um,
            dtype=torch.float32,
        ),
        "target_time_index": int(args.time_index),
        "target_gt_id": int(target_gt_id),
        "controlled_source_ids": torch.tensor(
            controlled_source_ids,
            dtype=torch.long,
        ),
        "anchor_original_tracklet_ids": torch.tensor(
            anchors,
            dtype=torch.long,
        ),
        "selected_original_tracklet_ids": torch.tensor(
            selected_tracklets,
            dtype=torch.long,
        ),
        **context,
    }

    atomic_torch_save(
        args.controlled_context_output,
        artifact,
    )
    atomic_json(
        args.controlled_context_json,
        report,
    )

    print_temporal_report(report)
    print("\nControlled split → temporal interpretation")
    print(
        f"  controlled source IDs        : {controlled_source_ids}"
    )
    print(
        f"  split visible in graph       : {split_visible_in_graph}"
    )
    print(
        f"  each piece has tracklet      : {piece_tracklet_coverage}"
    )
    print(
        f"  any target piece has past    : {any_piece_has_past}"
    )
    print(
        f"  CONTROLLED TEMPORAL READY    : "
        f"{'YES' if controlled_ready else 'NO'}"
    )

    if args.require_controlled_ready and not controlled_ready:
        raise RuntimeError(
            "Controlled temporal case did not satisfy Stage-08 readiness."
        )

    return report


def prepare_temporal_overfit_case(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Stage 08C-F convenience workflow.

    Assumes Stage-08A/B preparation exists (or rebuilds it through prepare()).
    """
    if not args.selection_json.exists() or not args.output.exists():
        print(
            "[controlled] Base Stage-08 preparation missing; "
            "building it first."
        )
        prepare(args)

    spatial_report = json.loads(
        args.selection_json.read_text(encoding="utf-8")
    )

    split_report = prepare_controlled_split(
        args,
        spatial_report,
    )

    track_graph, _ = run_controlled_trackastra(args)
    graph_mask_movie, graph_mask_path, graph_mask_diag = (
        choose_graph_mask_movie(
            args,
            track_graph,
        )
    )
    print(
        "[controlled] graph-label mask selection: "
        f"input={graph_mask_diag['controlled_input_label_match']:.3f}, "
        f"tracked={graph_mask_diag['trackastra_tracked_label_match']:.3f} "
        f"→ {graph_mask_path.name}"
    )

    temporal = build_controlled_temporal_cache(
        args,
        track_graph,
        graph_mask_movie,
        spatial_report,
    )

    report = prepare_controlled_temporal_context(
        args,
        spatial_report,
        split_report,
        track_graph,
        graph_mask_movie,
        graph_mask_path,
        temporal,
        graph_mask_diag,
    )

    case_report = dict(spatial_report)
    case_report["controlled_error"] = split_report
    case_report["controlled_temporal_summary"] = {
        "controlled_temporal_ready": bool(
            report.get("controlled_temporal_ready", False)
        ),
        "controlled_context_output": str(
            args.controlled_context_output
        ),
        "controlled_context_json": str(
            args.controlled_context_json
        ),
        "controlled_track_graph": str(
            args.controlled_graph
        ),
        "controlled_temporal_cache": str(
            args.controlled_cache
        ),
    }
    atomic_json(args.selection_json, case_report)

    print("\n" + "=" * 112)
    print("STAGE 08 CONTROLLED TEMPORAL-OVERFIT CASE")
    print("=" * 112)
    print(
        f"Controlled split ready         : YES"
    )
    print(
        f"Controlled Trackastra ready    : YES"
    )
    print(
        f"Controlled temporal context    : "
        f"{'YES' if report.get('controlled_temporal_ready') else 'NO'}"
    )
    print(
        f"Context artifact               : "
        f"{args.controlled_context_output}"
    )
    print(
        "\nVisualize with:\n"
        "  python investigations/stirnet/"
        "08_temporal_merge_case_preparation.py --visualize-controlled"
    )
    print("=" * 112)
    return report


def _controlled_split_masks_full(
    args: argparse.Namespace,
    frame_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    artifact = torch_load(
        args.controlled_split_artifact
    )
    bbox_pairs = artifact["bbox_zyx"]
    bbox = tuple(
        slice(int(pair[0]), int(pair[1]))
        for pair in bbox_pairs
    )
    local = (
        torch.as_tensor(
            artifact["controlled_partition_bbox"]
        )
        .cpu()
        .numpy()
    )
    a = np.zeros(frame_shape, dtype=np.uint8)
    b = np.zeros(frame_shape, dtype=np.uint8)
    a[bbox] = (local == 1).astype(np.uint8)
    b[bbox] = (local == 2).astype(np.uint8)
    return a, b


def visualize_controlled(args: argparse.Namespace) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is required for --visualize-controlled."
        ) from exc

    required = (
        args.controlled_context_output,
        args.controlled_split_artifact,
        args.controlled_input_movie,
        args.controlled_graph,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Controlled Stage-08 artifacts are missing:\n  "
            + "\n  ".join(str(path) for path in missing)
        )

    artifact = torch_load(
        args.controlled_context_output
    )
    report = artifact["report"]
    target = artifact["target"]
    spacing_np = (
        torch.as_tensor(artifact["spacing_um"])
        .float()
        .cpu()
        .numpy()
    )
    spacing = tuple(
        float(v) for v in spacing_np.tolist()
    )
    scale4d = (1.0, *spacing)
    dref_um = _safe_float(artifact["dref_um"])

    controlled_input = np.load(
        args.controlled_input_movie,
        mmap_mode="r",
    )
    gt_movie = np.load(
        args.data_dir / "gt_movie.npy",
        mmap_mode="r",
    )
    raw_movie = np.load(
        args.data_dir / "raw_movie.npy",
        mmap_mode="r",
    )

    with args.controlled_graph.open("rb") as handle:
        track_graph = pickle.load(handle)

    graph_mask_path = Path(
        report["controlled_error"]["graph_mask_path"]
    )
    graph_masks = np.load(
        graph_mask_path,
        mmap_mode="r",
    )

    display_tracklets = _display_tracklet_ids(
        report,
        args.visualize_radius_dref,
    )
    tracks_full, track_metadata = (
        _context_tracks_for_napari(
            artifact,
            track_graph,
            allowed_original_tracklets=display_tracklets,
        )
    )

    anchor_ids = set(
        int(v)
        for v in torch.as_tensor(
            artifact["anchor_original_tracklet_ids"]
        ).tolist()
    )
    anchor_tracks_full = (
        tracks_full[
            np.isin(
                tracks_full[:, 0].astype(np.int64),
                list(anchor_ids),
            )
        ]
        if tracks_full.size and anchor_ids
        else np.zeros((0, 5), dtype=np.float32)
    )

    frame_shape = tuple(
        int(v) for v in controlled_input.shape[-3:]
    )
    roi, roi_low, roi_high = _focus_visualization_roi(
        target,
        tracks_full,
        frame_shape,
        spacing_np,
        dref_um,
        args.visualize_margin_dref,
    )
    movie_roi = (slice(None),) + roi

    raw_roi = np.asarray(raw_movie[movie_roi])
    input_roi = np.asarray(
        controlled_input[movie_roi]
    )
    graph_roi = np.asarray(
        graph_masks[movie_roi]
    )
    gt_roi = np.asarray(
        gt_movie[movie_roi]
    )

    tracks = tracks_full.copy()
    if tracks.size:
        tracks[:, 2:5] -= roi_low[None].astype(np.float32)

    anchor_tracks = anchor_tracks_full.copy()
    if anchor_tracks.size:
        anchor_tracks[:, 2:5] -= roi_low[None].astype(np.float32)

    target_time = int(args.time_index)
    part_a_full, part_b_full = (
        _controlled_split_masks_full(
            args,
            frame_shape,
        )
    )
    roi_shape_zyx = tuple(
        int(s.stop) - int(s.start)
        for s in roi
    )
    display_shape = (
        int(controlled_input.shape[0]),
        *roi_shape_zyx,
    )
    part_a_roi = np.zeros(display_shape, dtype=np.uint8)
    part_b_roi = np.zeros(display_shape, dtype=np.uint8)
    part_a_roi[target_time] = part_a_full[roi]
    part_b_roi[target_time] = part_b_full[roi]

    target_gt_roi = np.zeros(display_shape, dtype=np.uint8)
    target_gt_roi[target_time] = (
        np.asarray(gt_movie[target_time][roi])
        == int(args.target_gt_id)
    ).astype(np.uint8)

    target_center_full = np.asarray(
        target["target_centroid_vox_zyx"],
        dtype=np.float32,
    )
    target_center_roi = (
        target_center_full - roi_low.astype(np.float32)
    )
    target_center = np.asarray(
        [
            float(target_time),
            *target_center_roi.tolist(),
        ],
        dtype=np.float32,
    )[None, :]

    viewer = napari.Viewer(
        title=(
            "STIR-Net Stage 08 — CONTROLLED TWO→ONE temporal case "
            f"(GT {args.target_gt_id}, t={target_time})"
        ),
        ndisplay=3,
    )
    viewer.add_image(
        raw_roi,
        name="Raw movie — target neighbourhood",
        colormap="gray",
        scale=scale4d,
        visible=True,
    )
    viewer.add_labels(
        input_roi,
        name="CONTROLLED Trackastra input instances",
        scale=scale4d,
        opacity=0.30,
        visible=True,
    )
    viewer.add_labels(
        graph_roi,
        name="Trackastra graph-compatible masks",
        scale=scale4d,
        opacity=0.25,
        visible=False,
    )
    viewer.add_labels(
        target_gt_roi,
        name=f"GT {args.target_gt_id} @ t={target_time} — ONE cell",
        scale=scale4d,
        opacity=0.70,
        visible=True,
    )
    viewer.add_labels(
        part_a_roi,
        name="CONTROLLED piece A",
        scale=scale4d,
        opacity=0.85,
        visible=True,
    )
    viewer.add_labels(
        part_b_roi,
        name="CONTROLLED piece B",
        scale=scale4d,
        opacity=0.85,
        visible=True,
    )

    if tracks.size:
        viewer.add_tracks(
            tracks,
            name="Controlled temporal context tracks",
            scale=scale4d,
            tail_length=max(8, controlled_input.shape[0] + 2),
            tail_width=2,
            visible=True,
        )
    if anchor_tracks.size:
        viewer.add_tracks(
            anchor_tracks,
            name="Controlled TARGET-associated tracks",
            scale=scale4d,
            tail_length=max(8, controlled_input.shape[0] + 2),
            tail_width=4,
            visible=True,
        )

    viewer.add_points(
        target_center,
        name="GT target centroid",
        scale=scale4d,
        size=6,
        face_color="transparent",
        border_color="white",
        visible=True,
    )

    try:
        viewer.dims.set_current_step(0, target_time)
    except Exception:
        pass

    print_temporal_report(report)
    print("\nControlled-case interpretation")
    print("  GT target                      : ONE biological instance")
    print("  Controlled piece A + piece B  : deliberate spatial over-split")
    print("  Trackastra tracks             : rerun AFTER injecting that split")
    print(
        "  Stage 09 objective            : temporal reasoning must merge "
        "the two provisional spatial pieces back into one final instance"
    )
    napari.run()



# =============================================================================
# VISUALIZATION
# =============================================================================


def _find_full_raw_movie(
    data_dir: Path,
    expected_shape: tuple[int, ...],
) -> Path | None:
    candidates: list[Path] = []
    for name in OPTIONAL_FULL_RAW_NAMES:
        candidates.append(data_dir / name)
        candidates.append(data_dir / "stirnet_source" / name)

    for path in candidates:
        if not path.exists():
            continue
        try:
            array = np.load(path, mmap_mode="r")
        except Exception:
            continue
        if tuple(array.shape) == expected_shape and array.ndim == 4:
            return path
    return None


def _context_tracks_for_napari(
    artifact: dict[str, Any],
    track_graph,
    *,
    allowed_original_tracklets: set[int] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build Napari tracks from raw Trackastra FULL-VOLUME voxel coords.

    Never reconstruct image coordinates from node_observed_ref_um here:
    those model coordinates are all-cell-ROI-centred microns.
    """
    original_tracklets = (
        torch.as_tensor(artifact["selected_original_tracklet_ids"])
        .long()
        .cpu()
    )
    local_tracklet_id = (
        torch.as_tensor(artifact["tracklet_id"])
        .long()
        .cpu()
    )
    node_ids = (
        torch.as_tensor(artifact["node_ids"])
        .long()
        .cpu()
    )

    rows: list[list[float]] = []
    metadata: list[dict[str, Any]] = []

    for row_index in range(node_ids.numel()):
        local_tid = int(local_tracklet_id[row_index])
        original_tid = int(original_tracklets[local_tid])
        if (
            allowed_original_tracklets is not None
            and original_tid not in allowed_original_tracklets
        ):
            continue

        node_id = int(node_ids[row_index])
        if node_id not in track_graph:
            continue
        node = track_graph.nodes[node_id]
        coords = np.asarray(node["coords"], dtype=np.float32)
        time_index = int(node["time"])
        label_id = int(node["label"])

        rows.append(
            [
                float(original_tid),
                float(time_index),
                float(coords[0]),
                float(coords[1]),
                float(coords[2]),
            ]
        )
        metadata.append(
            {
                "row_index": int(row_index),
                "node_id": node_id,
                "original_tracklet_id": original_tid,
                "time": time_index,
                "source_label": label_id,
                "coords_full_zyx": [
                    float(v) for v in coords.tolist()
                ],
            }
        )

    if not rows:
        return np.zeros((0, 5), dtype=np.float32), metadata

    tracks = np.asarray(rows, dtype=np.float32)
    order = np.lexsort((tracks[:, 1], tracks[:, 0]))
    tracks = tracks[order]
    metadata = [metadata[int(i)] for i in order.tolist()]
    return tracks, metadata


def _display_tracklet_ids(
    report: dict[str, Any],
    radius_dref: float,
) -> set[int]:
    anchors = set(int(v) for v in report.get("anchor_tracklets", []))
    selected = set(anchors)
    for row in report.get("tracklets", []):
        if float(row.get("distance_to_target_dref", 1e9)) <= radius_dref:
            selected.add(int(row["original_tracklet_id"]))
    return selected


def _focus_visualization_roi(
    target: dict[str, Any],
    tracks_full: np.ndarray,
    frame_shape_zyx: tuple[int, int, int],
    spacing_um: np.ndarray,
    dref_um: float,
    margin_dref: float,
) -> tuple[tuple[slice, slice, slice], np.ndarray, np.ndarray]:
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []

    bbox = target.get("target_bbox_zyx")
    if bbox:
        lows.append(np.asarray([axis[0] for axis in bbox], dtype=np.float32))
        highs.append(np.asarray([axis[1] for axis in bbox], dtype=np.float32))

    if tracks_full.size:
        coords = tracks_full[:, 2:5]
        lows.append(np.floor(coords.min(axis=0)))
        highs.append(np.ceil(coords.max(axis=0) + 1.0))

    if not lows:
        center = np.asarray(
            target["target_centroid_vox_zyx"],
            dtype=np.float32,
        )
        lows.append(center - 1.0)
        highs.append(center + 2.0)

    low = np.floor(np.min(np.stack(lows), axis=0)).astype(np.int64)
    high = np.ceil(np.max(np.stack(highs), axis=0)).astype(np.int64)

    margin_um = max(float(margin_dref) * float(dref_um), float(np.max(spacing_um)))
    margin_vox = np.ceil(
        margin_um / np.asarray(spacing_um, dtype=np.float32)
    ).astype(np.int64)

    full_shape = np.asarray(frame_shape_zyx, dtype=np.int64)
    low = np.maximum(low - margin_vox, 0)
    high = np.minimum(high + margin_vox, full_shape)

    roi = tuple(
        slice(int(a), int(b))
        for a, b in zip(low.tolist(), high.tolist())
    )
    return roi, low, high


def _linked_tracklet_mask_movie(
    instances_roi: np.ndarray,
    metadata: list[dict[str, Any]],
    roi_low: np.ndarray,
    *,
    allowed_tracklets: set[int],
) -> np.ndarray:
    result = np.zeros(instances_roi.shape, dtype=np.int32)
    ordered_tracklets = sorted(allowed_tracklets)
    display_id = {
        tracklet: index + 1
        for index, tracklet in enumerate(ordered_tracklets)
    }

    for row in metadata:
        tracklet = int(row["original_tracklet_id"])
        if tracklet not in allowed_tracklets:
            continue
        t = int(row["time"])
        label_id = int(row["source_label"])
        if not 0 <= t < instances_roi.shape[0]:
            continue
        mask = np.asarray(instances_roi[t]) == label_id
        result[t][mask] = int(display_id[tracklet])

    return result


def visualize_full(args: argparse.Namespace) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError("Napari is required for --visualize.") from exc

    if not args.context_output.exists():
        raise FileNotFoundError(
            f"Temporal context artifact not found:\n  {args.context_output}\n"
            "Run Stage 08 with a compatible --temporal-cache first."
        )

    artifact = torch_load(args.context_output)
    report = artifact["report"]
    instance_movie, gt_movie = load_full_movies(args.data_dir)
    shape = tuple(int(v) for v in gt_movie.shape)
    frame_shape = shape[1:]
    spacing_np = (
        torch.as_tensor(artifact["spacing_um"])
        .float()
        .cpu()
        .numpy()
    )
    spacing = tuple(float(v) for v in spacing_np.tolist())
    scale4d = (1.0, *spacing)
    dref_um = _safe_float(artifact["dref_um"])

    track_graph = _load_trackastra_graph(args.data_dir)

    display_tracklets = _display_tracklet_ids(
        report,
        args.visualize_radius_dref,
    )
    tracks_full, track_metadata = _context_tracks_for_napari(
        artifact,
        track_graph,
        allowed_original_tracklets=display_tracklets,
    )

    anchor_ids = set(
        int(v)
        for v in torch.as_tensor(
            artifact["anchor_original_tracklet_ids"]
        ).tolist()
    )
    anchor_tracks_full = (
        tracks_full[
            np.isin(
                tracks_full[:, 0].astype(np.int64),
                list(anchor_ids),
            )
        ]
        if tracks_full.size and anchor_ids
        else np.zeros((0, 5), dtype=np.float32)
    )

    target = artifact["target"]
    roi, roi_low, roi_high = _focus_visualization_roi(
        target,
        tracks_full,
        frame_shape,
        spacing_np,
        dref_um,
        args.visualize_margin_dref,
    )
    roi_shape = tuple(int(v) for v in (roi_high - roi_low).tolist())
    movie_roi = (slice(None),) + roi

    instances_roi = np.asarray(instance_movie[movie_roi])
    gt_roi = np.asarray(gt_movie[movie_roi])

    raw_path = _find_full_raw_movie(args.data_dir, shape)
    raw_roi = (
        np.asarray(np.load(raw_path, mmap_mode="r")[movie_roi])
        if raw_path is not None
        else None
    )

    tracks = tracks_full.copy()
    if tracks.size:
        tracks[:, 2:5] -= roi_low[None].astype(np.float32)

    anchor_tracks = anchor_tracks_full.copy()
    if anchor_tracks.size:
        anchor_tracks[:, 2:5] -= roi_low[None].astype(np.float32)

    target_center_full = np.asarray(
        target["target_centroid_vox_zyx"],
        dtype=np.float32,
    )
    target_center_roi = target_center_full - roi_low.astype(np.float32)
    target_center = np.asarray(
        [
            float(artifact["target_time_index"]),
            *target_center_roi.tolist(),
        ],
        dtype=np.float32,
    )[None, :]

    # Explicit target masks make the selected cell impossible to lose in a
    # many-cell scene.
    target_t = int(artifact["target_time_index"])
    target_gt_movie = np.zeros(instances_roi.shape, dtype=np.uint8)
    target_gt_movie[target_t] = (
        gt_roi[target_t] == int(artifact["target_gt_id"])
    ).astype(np.uint8)

    target_source_movie = np.zeros(instances_roi.shape, dtype=np.uint8)
    source_ids = [int(v) for v in target.get("target_source_ids", [])]
    if source_ids:
        target_source_movie[target_t] = np.isin(
            instances_roi[target_t],
            source_ids,
        ).astype(np.uint8)

    linked_context = _linked_tracklet_mask_movie(
        instances_roi,
        track_metadata,
        roi_low,
        allowed_tracklets=display_tracklets,
    )
    linked_anchor = _linked_tracklet_mask_movie(
        instances_roi,
        track_metadata,
        roi_low,
        allowed_tracklets=anchor_ids,
    )

    print(
        "[visualize] Track coordinate source: trackastra/track_graph.pkl "
        "(full-volume voxel coordinates)"
    )
    print(
        f"[visualize] Focus ROI zyx: low={roi_low.tolist()} "
        f"high={roi_high.tolist()} shape={roi_shape}"
    )
    print(
        f"[visualize] Showing {len(display_tracklets)} / "
        f"{report['context_counts']['tracklets']} temporal-context tracklets "
        f"within {args.visualize_radius_dref:.2f} dref (+ anchors)."
    )
    if raw_path is not None:
        print(f"[visualize] Raw movie: {raw_path}")

    viewer = napari.Viewer(
        title=(
            "STIR-Net Stage 08 — TARGET-FOCUSED temporal context "
            f"(GT {artifact['target_gt_id']}, t={target_t})"
        ),
        ndisplay=3,
    )

    if raw_roi is not None:
        viewer.add_image(
            raw_roi,
            name="Raw movie — target neighbourhood",
            colormap="gray",
            scale=scale4d,
            visible=True,
        )

    viewer.add_labels(
        instances_roi,
        name="Source instances — neighbourhood",
        scale=scale4d,
        opacity=0.30,
        visible=True,
    )
    viewer.add_labels(
        linked_context,
        name="Context track-linked source masks",
        scale=scale4d,
        opacity=0.40,
        visible=False,
    )
    viewer.add_labels(
        linked_anchor,
        name="TARGET anchor-track source masks",
        scale=scale4d,
        opacity=0.65,
        visible=True,
    )
    viewer.add_labels(
        gt_roi,
        name="GT neighbourhood (reference only)",
        scale=scale4d,
        opacity=0.25,
        visible=False,
    )
    viewer.add_labels(
        target_gt_movie,
        name=f"TARGET GT {artifact['target_gt_id']} @ t={target_t} — ONE cell",
        scale=scale4d,
        opacity=0.80,
        visible=True,
    )
    viewer.add_labels(
        target_source_movie,
        name=f"TARGET source {source_ids} @ t={target_t}",
        scale=scale4d,
        opacity=0.75,
        visible=False,
    )

    if tracks.size:
        viewer.add_tracks(
            tracks,
            name="Nearby temporal context tracks",
            scale=scale4d,
            tail_length=max(8, shape[0] + 2),
            tail_width=2,
            visible=True,
        )
    if anchor_tracks.size:
        viewer.add_tracks(
            anchor_tracks,
            name="TARGET-associated anchor track",
            scale=scale4d,
            tail_length=max(8, shape[0] + 2),
            tail_width=4,
            visible=True,
        )

    viewer.add_points(
        target_center,
        name="TARGET GT current centroid",
        scale=scale4d,
        size=6,
        face_color="transparent",
        border_color="white",
        visible=True,
    )

    try:
        viewer.dims.set_current_step(0, target_t)
    except Exception:
        pass

    print_temporal_report(report)
    print("\nNapari interpretation")
    print("  TARGET GT ... @ t=2              : exact selected one-cell GT mask")
    print("  TARGET anchor-track source masks  : source cell attached to anchor track across time")
    print("  TARGET-associated anchor track    : the relevant Trackastra trajectory")
    print("  Nearby temporal context tracks    : local neighbours used as context")
    print(
        "\nThis viewer is DISPLAY-cropped only. Stage-08 temporal evidence was "
        "still discovered from the full uncropped scene."
    )
    print(
        "The deliberate TWO-instance RAG split does not exist yet; it will be "
        "created only after candidate-0 spatial overfit."
    )
    napari.run()


def visualize_crop(args: argparse.Namespace) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError("Napari is required for --visualize-crop.") from exc

    if not args.output.exists():
        raise FileNotFoundError(
            f"Prepared crop not found:\n  {args.output}"
        )

    payload = torch_load(args.output)
    raw = torch.as_tensor(payload["raw_norm"]).float().numpy()
    current = torch.as_tensor(payload["current_labels"]).long().numpy()
    gt = torch.as_tensor(payload["gt_labels"]).long().numpy()
    spacing = torch.as_tensor(payload["spacing_um"]).float().numpy()
    scale = tuple(float(v) for v in spacing.tolist())

    selection = dict(payload.get("selection", {}))
    temporal_meta = dict(selection.get("temporal_merge_case", {}))
    target_gt_id = int(
        temporal_meta.get("target_gt_id", args.target_gt_id)
    )

    target_mask = (gt == target_gt_id).astype(np.int32)
    pieces = np.zeros(gt.shape, dtype=np.int32)
    target = gt == target_gt_id
    source_ids, counts = np.unique(
        current[target & (current > 0)],
        return_counts=True,
    )
    for display_id, source_id in enumerate(
        source_ids[np.argsort(counts)[::-1]].tolist(),
        1,
    ):
        pieces[target & (current == int(source_id))] = display_id

    viewer = napari.Viewer(
        title=f"STIR-Net Stage 08 — spatial crop GT {target_gt_id}",
        ndisplay=3,
    )
    viewer.add_image(raw, name="Raw", colormap="gray", scale=scale)
    viewer.add_labels(
        current,
        name="Current segmentation",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        gt,
        name="GT labels",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        target_mask,
        name=f"Selected GT {target_gt_id} (ONE cell)",
        scale=scale,
        visible=True,
    )
    viewer.add_labels(
        pieces,
        name="Current/source pieces inside selected GT",
        scale=scale,
        visible=True,
    )
    napari.run()


# =============================================================================
# MAIN PREPARATION
# =============================================================================


def _load_existing_spatial_report(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not args.selection_json.exists() or not args.output.exists():
        return None
    try:
        report = json.loads(args.selection_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    if int(report.get("format_version", 0)) < 2:
        return None
    if int(report.get("target_gt_id", -1)) != int(args.target_gt_id):
        return None
    return report


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()

    print("\n" + "=" * 112)
    print("STIR-Net Stage 08 — full-context temporal merge-case preparation")
    print("=" * 112)
    print(f"Candidate index               : {args.candidate_index}")
    print(f"Target GT                     : {args.target_gt_id}")
    print(f"Full scene                    : {args.data_dir}")
    print(f"Dedicated spatial crop        : {args.output}")
    print(f"Temporal context output       : {args.context_output}")
    print("Training                      : NONE")
    print("=" * 112)

    existing = None if args.rebuild else _load_existing_spatial_report(args)
    if existing is None:
        print("\n[stage 08A] Rebuilding candidate-0 spatial preparation ...")
        _, spatial_report = prepare_spatial_crop(args)
    else:
        spatial_report = existing
        print("\n[stage 08A] Reusing existing version-2 spatial preparation.")

    # Always refresh target/full-cache diagnostics when explicitly rebuilding
    # or when the context output/report is missing. Otherwise reuse context.
    context_needs_rebuild = (
        args.rebuild
        or not args.context_output.exists()
        or not args.context_json.exists()
        or args.temporal_cache is not None
    )

    temporal_report: dict[str, Any] | None = None
    if context_needs_rebuild:
        print("\n[stage 08B] Building full-volume-derived temporal context ...")
        temporal_report = prepare_temporal_context(args, spatial_report)
    elif args.context_json.exists():
        temporal_report = json.loads(
            args.context_json.read_text(encoding="utf-8")
        )
        print("\n[stage 08B] Reusing existing temporal context artifact.")
        if "context_counts" in temporal_report:
            print_temporal_report(temporal_report)

    # Merge temporal readiness/source paths into the case report without
    # changing the Stage-01-compatible .pt crop.
    final_report = dict(spatial_report)
    final_report["temporal_context_output"] = str(args.context_output)
    final_report["temporal_context_json"] = str(args.context_json)
    final_report["temporal_ready"] = bool(
        temporal_report
        and temporal_report.get("temporal_ready", False)
    )
    if temporal_report is not None:
        final_report["temporal_summary"] = {
            "temporal_cache": temporal_report.get("temporal_cache"),
            "coordinate_mode": temporal_report.get("coordinate_mode"),
            "anchor_tracklets": temporal_report.get("anchor_tracklets", []),
            "context_counts": temporal_report.get("context_counts", {}),
            "readiness_reason": temporal_report.get("readiness_reason"),
        }
    final_report["elapsed_seconds"] = float(time.perf_counter() - start)
    atomic_json(args.selection_json, final_report)

    print("\n" + "=" * 112)
    print("STAGE-08 STATUS")
    print("=" * 112)
    print(f"Spatial target/crop ready      : YES")
    print(
        f"Temporal context ready         : "
        f"{'YES' if final_report['temporal_ready'] else 'NO'}"
    )
    print(f"Case report                    : {args.selection_json}")
    print(f"Spatial crop                   : {args.output}")
    if args.context_output.exists():
        print(f"Temporal context              : {args.context_output}")
    print(f"Elapsed                        : {_format_duration(final_report['elapsed_seconds'])}")

    if final_report["temporal_ready"]:
        print(
            "\nStage 08 is ready for visual confirmation:\n"
            "  python investigations/stirnet/"
            "08_temporal_merge_case_preparation.py --visualize\n"
            "\nAfter the full temporal context is confirmed, candidate-0 spatial "
            "overfit can be run and the controlled RAG-level TWO-instance "
            "counterfactual can be prepared for Stage 09."
        )
    else:
        print(
            "\nDo NOT start temporal overfit yet. Supply/inspect the full-volume "
            "temporal-v3 cache and rebuild Stage 08."
        )
    print("=" * 112)
    return final_report


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare candidate 0 / GT 6 with full-volume-derived temporal "
            "neighbourhood evidence for the first TWO->ONE temporal overfit."
        )
    )
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=DEFAULT_CANDIDATE_INDEX,
    )
    parser.add_argument(
        "--target-gt-id",
        type=int,
        default=DEFAULT_TARGET_GT_ID,
        help="Manually confirmed target; default is GT 6.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=DEFAULT_TIME_INDEX,
    )
    parser.add_argument(
        "--voxel-budget",
        type=int,
        default=DEFAULT_VOXEL_BUDGET,
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
    )
    parser.add_argument(
        "--context-margin-dref",
        type=float,
        default=DEFAULT_CONTEXT_MARGIN_DREF,
    )
    parser.add_argument(
        "--stage00-sample",
        type=Path,
        default=DEFAULT_STAGE00_SAMPLE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=DEFAULT_SELECTION_JSON,
    )

    parser.add_argument(
        "--temporal-cache",
        type=Path,
        default=None,
        help=(
            "Full-volume temporal-v3 STIR-Net/Trackastra cache. "
            "If omitted, Stage 08 searches plausible cache names under the "
            "scene directory and its parent."
        ),
    )
    parser.add_argument(
        "--context-output",
        type=Path,
        default=DEFAULT_CONTEXT_OUTPUT,
    )
    parser.add_argument(
        "--context-json",
        type=Path,
        default=DEFAULT_CONTEXT_JSON,
    )
    parser.add_argument(
        "--context-radius-dref",
        type=float,
        default=DEFAULT_CONTEXT_RADIUS_DREF,
    )
    parser.add_argument(
        "--min-context-tracklets",
        type=int,
        default=DEFAULT_MIN_CONTEXT_TRACKLETS,
    )
    parser.add_argument(
        "--max-context-tracklets",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TRACKLETS,
    )
    parser.add_argument(
        "--hypothesis-hops",
        type=int,
        default=DEFAULT_HYPOTHESIS_HOPS,
    )
    parser.add_argument(
        "--anchor-overlap",
        type=float,
        default=DEFAULT_ANCHOR_OVERLAP,
    )
    parser.add_argument(
        "--temporal-coordinates",
        choices=("auto", "centered", "absolute"),
        default="auto",
        help=(
            "Coordinate convention for temporal_ref_um/node_observed_ref_um. "
            "Repository DetectionRecord uses target-centred coordinates; auto "
            "checks target-associated anchors when available."
        ),
    )
    parser.add_argument(
        "--require-temporal-cache",
        action="store_true",
        help=(
            "Fail instead of leaving Stage 08 spatial-only if no compatible "
            "full-volume temporal-v3 cache / target past evidence is available."
        ),
    )

    parser.add_argument(
        "--prepare-temporal-overfit",
        action="store_true",
        help=(
            "After normal Stage-08 preparation, create the controlled TWO-"
            "instance error from safe supervoxels, rerun Trackastra, rebuild "
            "the temporal-v3 cache, and save the Stage-09-ready ego graph."
        ),
    )
    parser.add_argument(
        "--supervoxel-source",
        choices=("auto", "oracle", "learned"),
        default="auto",
        help=(
            "Safe-SV source for defining the controlled split. auto uses a "
            "candidate-0 Stage-01 checkpoint if present, otherwise production "
            "watershed/guard over oracle geometry targets."
        ),
    )
    parser.add_argument(
        "--stage01-checkpoint",
        type=Path,
        default=DEFAULT_STAGE01_TEMPORAL_CHECKPOINT,
    )
    parser.add_argument(
        "--split-seed-supervoxels",
        type=int,
        nargs=2,
        default=None,
        metavar=("SV_A", "SV_B"),
        help="Manually choose the two safe supervoxels used as split seeds.",
    )
    parser.add_argument(
        "--min-split-fraction",
        type=float,
        default=DEFAULT_MIN_SPLIT_FRACTION,
    )

    parser.add_argument(
        "--controlled-input-movie",
        type=Path,
        default=DEFAULT_CONTROLLED_INPUT_MOVIE,
    )
    parser.add_argument(
        "--controlled-tracked-movie",
        type=Path,
        default=DEFAULT_CONTROLLED_TRACKED_MOVIE,
    )
    parser.add_argument(
        "--controlled-graph",
        type=Path,
        default=DEFAULT_CONTROLLED_GRAPH,
    )
    parser.add_argument(
        "--controlled-cache",
        type=Path,
        default=DEFAULT_CONTROLLED_CACHE,
    )
    parser.add_argument(
        "--controlled-context-output",
        type=Path,
        default=DEFAULT_CONTROLLED_CONTEXT_OUTPUT,
    )
    parser.add_argument(
        "--controlled-context-json",
        type=Path,
        default=DEFAULT_CONTROLLED_CONTEXT_JSON,
    )
    parser.add_argument(
        "--controlled-split-artifact",
        type=Path,
        default=DEFAULT_CONTROLLED_SPLIT_ARTIFACT,
    )
    parser.add_argument(
        "--controlled-split-json",
        type=Path,
        default=DEFAULT_CONTROLLED_SPLIT_JSON,
    )

    parser.add_argument(
        "--trackastra-model",
        type=str,
        default=DEFAULT_TRACKASTRA_MODEL,
        help="Trackastra pretrained model; default ctc for 2D/3D CTC-style tracking.",
    )
    parser.add_argument(
        "--trackastra-mode",
        choices=("greedy", "greedy_nodiv", "ilp"),
        default=DEFAULT_TRACKASTRA_MODE,
    )
    parser.add_argument(
        "--trackastra-device",
        type=str,
        default=DEFAULT_TRACKASTRA_DEVICE,
    )
    parser.add_argument(
        "--force-trackastra",
        action="store_true",
        help="Rerun Trackastra even if controlled graph/masks already exist.",
    )
    parser.add_argument(
        "--force-controlled-cache",
        action="store_true",
        help="Rebuild controlled temporal-v3 cache even if it already exists.",
    )
    parser.add_argument(
        "--require-controlled-ready",
        action="store_true",
        help="Fail unless the controlled split is visible to Trackastra and both pieces have temporal tracklets.",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Open a target-focused display ROI cut from the full movies, using "
            "raw Trackastra voxel coordinates so tracks align exactly."
        ),
    )
    parser.add_argument(
        "--visualize-radius-dref",
        type=float,
        default=DEFAULT_VISUALIZE_RADIUS_DREF,
        help=(
            "Display only context tracklets whose Stage-08 target distance is "
            "within this radius, plus all target anchors. Data extraction still "
            "uses the full prepared temporal context."
        ),
    )
    parser.add_argument(
        "--visualize-margin-dref",
        type=float,
        default=DEFAULT_VISUALIZE_MARGIN_DREF,
        help="Physical display margin around target + displayed trajectories.",
    )
    parser.add_argument(
        "--visualize-crop",
        action="store_true",
        help="Open only the compact candidate-0 spatial crop.",
    )
    parser.add_argument(
        "--visualize-controlled",
        action="store_true",
        help=(
            "Open the controlled TWO-instance movie together with the "
            "Trackastra rerun and target-focused temporal context."
        ),
    )
    parser.add_argument(
        "--list-cells",
        action="store_true",
        help="Print cell ranking during preparation; no separate action needed.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild both spatial preparation and temporal ego graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.data_dir = args.data_dir.resolve()
    args.stage00_sample = args.stage00_sample.resolve()
    args.output = args.output.resolve()
    args.selection_json = args.selection_json.resolve()
    args.context_output = args.context_output.resolve()
    args.context_json = args.context_json.resolve()
    args.stage01_checkpoint = args.stage01_checkpoint.resolve()
    args.controlled_input_movie = args.controlled_input_movie.resolve()
    args.controlled_tracked_movie = args.controlled_tracked_movie.resolve()
    args.controlled_graph = args.controlled_graph.resolve()
    args.controlled_cache = args.controlled_cache.resolve()
    args.controlled_context_output = args.controlled_context_output.resolve()
    args.controlled_context_json = args.controlled_context_json.resolve()
    args.controlled_split_artifact = args.controlled_split_artifact.resolve()
    args.controlled_split_json = args.controlled_split_json.resolve()
    if args.temporal_cache is not None:
        args.temporal_cache = args.temporal_cache.resolve()

    if args.context_radius_dref <= 0:
        raise ValueError("--context-radius-dref must be positive")
    if not 0.0 < args.min_split_fraction < 0.5:
        raise ValueError("--min-split-fraction must be in (0,0.5)")
    if args.visualize_radius_dref <= 0:
        raise ValueError("--visualize-radius-dref must be positive")
    if args.visualize_margin_dref <= 0:
        raise ValueError("--visualize-margin-dref must be positive")
    if args.min_context_tracklets < 1:
        raise ValueError("--min-context-tracklets must be >= 1")
    if args.max_context_tracklets < args.min_context_tracklets:
        raise ValueError(
            "--max-context-tracklets must be >= --min-context-tracklets"
        )
    if args.hypothesis_hops < 0:
        raise ValueError("--hypothesis-hops cannot be negative")
    if not 0.0 <= args.anchor_overlap <= 1.0:
        raise ValueError("--anchor-overlap must be in [0,1]")

    if args.candidate_index != DEFAULT_CANDIDATE_INDEX:
        print(
            f"[WARN] This experiment was designed for candidate "
            f"{DEFAULT_CANDIDATE_INDEX}; requested {args.candidate_index}."
        )

    if args.visualize:
        # Do not silently recompute expensive context just to visualize.
        if not args.context_output.exists():
            raise FileNotFoundError(
                f"Missing {args.context_output}. Run Stage 08 preparation first."
            )
        visualize_full(args)
        return

    if args.visualize_crop:
        if not args.output.exists():
            raise FileNotFoundError(
                f"Missing {args.output}. Run Stage 08 preparation first."
            )
        visualize_crop(args)
        return

    if args.visualize_controlled:
        visualize_controlled(args)
        return

    prepare(args)

    if args.prepare_temporal_overfit:
        prepare_temporal_overfit_case(args)


if __name__ == "__main__":
    main()
