from __future__ import annotations

"""Stage 04: select a useful STIR-Net RAG-debug crop from Stage-00 candidates.

This stage does NOT train the dense network.

Why oracle geometry is used here
--------------------------------
Stage 01 deliberately overfit one fixed crop. Running that checkpoint over all
other Stage-00 candidates would mostly measure one-crop generalization, which is
not the question we want to ask now.

Stage 04 instead reuses the Stage-00 candidate list, builds exact production
geometry targets for each candidate, converts them to oracle dense geometry,
then runs the exact Stage-03 marker-controlled watershed. It ranks candidates
by whether the resulting supervoxel graph contains:

    positive RAG edges:
        adjacent supervoxels belonging to the SAME GT cell -> MERGE

    negative RAG edges:
        adjacent supervoxels belonging to DIFFERENT GT cells -> KEEP SEPARATE

while first requiring a safe proposal:
    - complete-cell marker recall = 1
    - good complete-cell coverage
    - zero meaningful cross-GT supervoxels
    - complete cells recoverable by merging supervoxels

Outputs
-------
    data/learned/stirnet/rag_case_selection/case_scan.json
    data/learned/stirnet/rag_case_selection/selected_case.pt
    data/learned/stirnet/rag_debug_crop.pt
    data/learned/stirnet/rag_debug_crop_metadata.json

Typical use
-----------
    python investigations/stirnet/04_rag_case_selection.py

Visualize only the selected case:
    python investigations/stirnet/04_rag_case_selection.py --visualize

Visualize any Stage-00 candidate without rerunning the full scan:
    python investigations/stirnet/04_rag_case_selection.py --candidate-index 1 --visualize
    python investigations/stirnet/04_rag_case_selection.py --candidate-index 2 --visualize
    python investigations/stirnet/04_rag_case_selection.py --candidate-index 5 --visualize

The first view of a candidate builds and caches only that candidate. Reopening
the same candidate is then immediate.

Inspect more Stage-00 candidates:
    python investigations/stirnet/04_rag_case_selection.py --max-candidates 24

The script dynamically imports adjacent 00_GT_validation.py and
03_watershed_supervoxel_debug.py so target generation, candidate enumeration,
and watershed behavior stay synchronized with the investigations already used.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor


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
    "_stirnet_stage00_gt_validation",
    "00_GT_validation.py",
)
stage03 = _import_adjacent(
    "_stirnet_stage03_watershed_supervoxel_debug",
    "03_watershed_supervoxel_debug.py",
)


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_DATA_DIR = stage00.DEFAULT_DATA_DIR
DEFAULT_TIME_INDEX = stage00.DEFAULT_TIME_INDEX
DEFAULT_VOXEL_BUDGET = stage00.DEFAULT_VOXEL_BUDGET
DEFAULT_MAX_CANDIDATES = stage00.DEFAULT_MAX_CANDIDATES
DEFAULT_CONTEXT_MARGIN_DREF = stage00.DEFAULT_CONTEXT_MARGIN_DREF

DEFAULT_RESULT_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "rag_case_selection"
)
DEFAULT_SELECTED_SAMPLE = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "rag_debug_crop.pt"
)
DEFAULT_STAGE00_SAMPLE = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "debug_crop.pt"
)

# Candidate class priorities. Safety is always considered before edge richness.
CLASS_UNSAFE = 0
CLASS_SAFE_NEGATIVE_ONLY = 1
CLASS_SAFE_MERGE_ONLY = 2
CLASS_SAFE_BOTH_CLASSES = 3


# =============================================================================
# FILE HELPERS
# =============================================================================

def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _slice_pairs(
    slices: tuple[slice, slice, slice],
) -> list[list[int]]:
    return [[int(s.start), int(s.stop)] for s in slices]


def _slice_shape(
    slices: tuple[slice, slice, slice],
) -> tuple[int, int, int]:
    return tuple(int(s.stop) - int(s.start) for s in slices)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _progress(message: str) -> None:
    # flush=True is important when this is run from PowerShell/PyCharm so the
    # user sees long-running candidate stages immediately.
    print(message, flush=True)


# =============================================================================
# STAGE-00 SCENE / CANDIDATE REUSE
# =============================================================================

def _stage00_args(args: argparse.Namespace) -> argparse.Namespace:
    """Namespace containing the fields Stage-00 load_scene reads."""
    return argparse.Namespace(
        data_dir=args.data_dir,
        time_index=args.time_index,
        voxel_budget=args.voxel_budget,
        crop_shape=None,
        candidate_index=0,
        max_candidates=args.max_candidates,
        exclude_merge_source=[],
        context_margin_dref=args.context_margin_dref,
    )


def load_full_scene_and_candidates(
    args: argparse.Namespace,
):
    """Load full arrays and reuse the Stage-00 candidate list already on disk.

    This deliberately avoids calling Stage-00 ``load_scene()``, because that
    function re-runs the expensive candidate enumeration internally.
    """
    phase_start = time.perf_counter()

    _progress(
        "[setup] Loading saved Stage-00 candidate list from "
        f"{args.stage00_sample} ..."
    )
    if not args.stage00_sample.exists():
        raise FileNotFoundError(
            "Stage-00 reusable sample not found:\n"
            f"  {args.stage00_sample}\n"
            "Run investigations/stirnet/00_GT_validation.py once first."
        )

    saved_stage00 = torch_load(args.stage00_sample)
    saved_selection = saved_stage00.get("selection", {})
    summaries = list(saved_selection.get("candidate_summaries", []))
    if not summaries:
        raise RuntimeError(
            "The saved Stage-00 sample does not contain "
            "selection['candidate_summaries']. Re-run the current "
            "00_GT_validation.py once so the candidate list is persisted."
        )

    summaries = summaries[: int(args.max_candidates)]
    _progress(
        f"[setup] Reusing {len(summaries)} already-mined candidates; "
        "no candidate re-enumeration."
    )

    data_dir = args.data_dir.resolve()
    instance_path = data_dir / "instance_movie.npy"
    gt_path = data_dir / "gt_movie.npy"
    metadata_path = data_dir / "metadata.json"
    source_dir = data_dir / "stirnet_source"
    raw_path = source_dir / "raw_norm_target.npy"
    marker_path = source_dir / "marker_heatmap_target.npy"

    required = (
        instance_path,
        gt_path,
        metadata_path,
        raw_path,
        marker_path,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared STIR-Net data is incomplete. Missing:\n  "
            + "\n  ".join(str(path) for path in missing)
        )

    _progress("[setup] Memory-mapping full current/GT/source arrays ...")
    instance_movie = np.load(instance_path, mmap_mode="r")
    gt_movie = np.load(gt_path, mmap_mode="r")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    spacing = np.asarray(metadata["spacing_zyx_um"], dtype=np.float32)

    t = int(args.time_index)
    maximum_t = min(len(instance_movie), len(gt_movie)) - 1
    if not (0 <= t <= maximum_t):
        raise IndexError(f"time-index {t} is outside [0, {maximum_t}]")

    current_full = np.asarray(instance_movie[t])
    gt_full = np.asarray(gt_movie[t])
    raw_full = np.load(raw_path, mmap_mode="r")
    marker_full = np.load(marker_path, mmap_mode="r")

    if current_full.shape != gt_full.shape:
        raise RuntimeError(
            f"Current/GT shape mismatch: {current_full.shape} vs {gt_full.shape}"
        )
    if raw_full.shape != gt_full.shape:
        raise RuntimeError(
            f"Raw/GT shape mismatch: {raw_full.shape} vs {gt_full.shape}"
        )
    if marker_full.shape != gt_full.shape:
        raise RuntimeError(
            f"Marker/GT shape mismatch: {marker_full.shape} vs {gt_full.shape}"
        )

    dref_value = saved_stage00.get("dref_um")
    if isinstance(dref_value, torch.Tensor):
        dref_um = float(dref_value.item())
    elif dref_value is not None:
        dref_um = float(dref_value)
    else:
        dref_um = stage00.estimate_model_dref_um(
            current_full,
            tuple(float(v) for v in spacing),
        )

    candidates = []
    for ordinal, summary in enumerate(summaries):
        selection = dict(summary)
        selection["candidate_index"] = int(
            selection.get("candidate_index", ordinal)
        )
        pairs = selection["crop_slices_zyx"]
        crop = tuple(
            slice(int(pair[0]), int(pair[1]))
            for pair in pairs
        )
        candidates.append((crop, selection))

    if not candidates:
        raise RuntimeError("No Stage-00 candidate crops are available.")

    first_crop, first_selection = candidates[0]
    first_build = stage00._expand_build_region(
        gt_full,
        first_crop,
        spacing,
        dref_um,
        args.context_margin_dref,
    )

    base_scene = stage00.Scene(
        current_full=current_full,
        gt_full=gt_full,
        raw_full=raw_full,
        marker_full=marker_full,
        spacing_zyx_um=spacing,
        dref_um=float(dref_um),
        core_slices=first_crop,
        build_slices=first_build,
        selection=dict(first_selection),
    )

    _progress(
        f"[setup] Ready in "
        f"{_format_duration(time.perf_counter() - phase_start)} | "
        f"full shape={tuple(gt_full.shape)} | candidates={len(candidates)}"
    )
    return base_scene, candidates


def make_candidate_scene(
    base_scene,
    crop: tuple[slice, slice, slice],
    selection: dict[str, Any],
    context_margin_dref: float,
):
    selection = dict(selection)
    build = stage00._expand_build_region(
        base_scene.gt_full,
        crop,
        base_scene.spacing_zyx_um,
        base_scene.dref_um,
        context_margin_dref,
    )
    selection["build_slices_zyx"] = _slice_pairs(build)

    return stage00.Scene(
        current_full=base_scene.current_full,
        gt_full=base_scene.gt_full,
        raw_full=base_scene.raw_full,
        marker_full=base_scene.marker_full,
        spacing_zyx_um=base_scene.spacing_zyx_um,
        dref_um=base_scene.dref_um,
        core_slices=crop,
        build_slices=build,
        selection=selection,
    )


def target_batch_for_stage03(scene, targets) -> dict[str, Tensor]:
    relative_core = stage00._relative_slices(
        scene.core_slices,
        scene.build_slices,
    )
    cropped = stage00._cropped_target_tensors(targets, relative_core)
    # Stage-03 helpers consume batched [B,C,Z,Y,X] target tensors.
    return {
        name: value.unsqueeze(0).float().contiguous()
        for name, value in cropped.items()
    }


def batch_namespace_for_stage03(
    scene,
    target_batch: dict[str, Tensor],
):
    gt = np.asarray(scene.gt_full[scene.core_slices]).astype(
        np.int64, copy=True
    )
    return SimpleNamespace(
        spacing_um=torch.from_numpy(
            scene.spacing_zyx_um.copy()
        ).float()[None],
        dref_um=torch.tensor(
            [scene.dref_um], dtype=torch.float32
        ),
        gt_labels=torch.from_numpy(gt).long(),
        targets=target_batch,
    )


# =============================================================================
# SUPERVOXEL ADJACENCY / EDGE TARGETS
# =============================================================================

def supervoxel_adjacency(labels: np.ndarray) -> list[tuple[int, int]]:
    """Unique undirected 6-connected positive-label adjacency pairs."""
    labels = np.asarray(labels, dtype=np.int64)
    packed: list[np.ndarray] = []
    base = int(labels.max()) + 1

    if base <= 1:
        return []

    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)

        a = labels[tuple(lo)]
        b = labels[tuple(hi)]
        valid = (a > 0) & (b > 0) & (a != b)
        if not valid.any():
            continue

        left = np.minimum(a[valid], b[valid]).astype(np.int64)
        right = np.maximum(a[valid], b[valid]).astype(np.int64)
        packed.append(left * base + right)

    if not packed:
        return []

    unique = np.unique(np.concatenate(packed))
    return [
        (int(value // base), int(value % base))
        for value in unique.tolist()
    ]


def classify_edges(
    adjacency: list[tuple[int, int]],
    supervoxel_summary: dict[str, Any],
) -> dict[str, Any]:
    rows = {
        int(row["supervoxel_id"]): row
        for row in supervoxel_summary["nodes"]
    }

    all_positive: list[list[int]] = []
    all_negative: list[list[int]] = []
    invalid: list[list[int]] = []

    valid_positive: list[list[int]] = []
    valid_negative: list[list[int]] = []

    edge_rows: list[dict[str, Any]] = []

    for a, b in adjacency:
        ra = rows.get(a)
        rb = rows.get(b)
        if ra is None or rb is None:
            invalid.append([a, b])
            continue

        ga = int(ra["dominant_gt"])
        gb = int(rb["dominant_gt"])
        both_nodes_valid = bool(
            ra["passes_current_rag_validity"]
            and rb["passes_current_rag_validity"]
        )

        if ga <= 0 or gb <= 0:
            target = "invalid"
            invalid.append([a, b])
        elif ga == gb:
            target = "merge"
            all_positive.append([a, b])
            if both_nodes_valid:
                valid_positive.append([a, b])
        else:
            target = "separate"
            all_negative.append([a, b])
            if both_nodes_valid:
                valid_negative.append([a, b])

        edge_rows.append(
            {
                "supervoxel_a": a,
                "supervoxel_b": b,
                "dominant_gt_a": ga,
                "dominant_gt_b": gb,
                "target": target,
                "both_nodes_pass_rag_validity": both_nodes_valid,
            }
        )

    return {
        "adjacency_edge_count": len(adjacency),
        "merge_edge_count_all": len(all_positive),
        "separate_edge_count_all": len(all_negative),
        "invalid_edge_count": len(invalid),
        "merge_edge_count_valid": len(valid_positive),
        "separate_edge_count_valid": len(valid_negative),
        "merge_edges_valid": valid_positive,
        "separate_edges_valid": valid_negative,
        "edges": edge_rows,
    }


def edge_target_interface(
    labels: np.ndarray,
    edge_info: dict[str, Any],
) -> np.ndarray:
    """Voxel interface labels: 1=same-GT merge, 2=different-GT separate."""
    labels = np.asarray(labels, dtype=np.int64)
    result = np.zeros(labels.shape, dtype=np.uint8)

    pair_target: dict[tuple[int, int], int] = {}
    for edge in edge_info["edges"]:
        if not edge["both_nodes_pass_rag_validity"]:
            continue
        a = int(edge["supervoxel_a"])
        b = int(edge["supervoxel_b"])
        key = (min(a, b), max(a, b))
        if edge["target"] == "merge":
            pair_target[key] = 1
        elif edge["target"] == "separate":
            pair_target[key] = 2

    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)

        a = labels[tuple(lo)]
        b = labels[tuple(hi)]
        different = (a > 0) & (b > 0) & (a != b)
        if not different.any():
            continue

        coords = np.argwhere(different)
        avals = a[different]
        bvals = b[different]

        target_lo = result[tuple(lo)]
        target_hi = result[tuple(hi)]

        for coord, av, bv in zip(
            coords,
            avals.tolist(),
            bvals.tolist(),
        ):
            key = (min(int(av), int(bv)), max(int(av), int(bv)))
            value = pair_target.get(key, 0)
            if value == 0:
                continue
            idx = tuple(int(v) for v in coord)
            # Preserve merge interface as 1 unless a separate interface (2)
            # occupies the same voxel; separation is the stronger display cue.
            target_lo[idx] = max(int(target_lo[idx]), value)
            target_hi[idx] = max(int(target_hi[idx]), value)

        result[tuple(lo)] = target_lo
        result[tuple(hi)] = target_hi

    return result


# =============================================================================
# ORACLE STAGE-03 ANALYSIS
# =============================================================================

def analyze_candidate(
    scene,
    cfg,
    *,
    progress_prefix: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_start = time.perf_counter()

    step_start = time.perf_counter()
    _progress(f"{progress_prefix}  [1/4] building production oracle targets ...")
    targets = stage00.build_targets(scene)
    _progress(
        f"{progress_prefix}        targets done in "
        f"{_format_duration(time.perf_counter() - step_start)}"
    )
    target_batch = target_batch_for_stage03(scene, targets)
    batch = batch_namespace_for_stage03(scene, target_batch)

    complete_ids, partial_ids = stage00._complete_and_partial_ids(
        scene.gt_full,
        scene.core_slices,
    )

    step_start = time.perf_counter()
    _progress(f"{progress_prefix}  [2/4] constructing oracle geometry ...")
    oracle_geometry = stage03.oracle_geometry_from_targets(target_batch)
    _progress(
        f"{progress_prefix}        oracle geometry done in "
        f"{_format_duration(time.perf_counter() - step_start)}"
    )

    step_start = time.perf_counter()
    _progress(
        f"{progress_prefix}  [3/4] markers + watershed + supervoxel diagnostics ..."
    )
    ws_summary, ws_artifact = stage03.run_variant(
        name="oracle_all",
        geometry=oracle_geometry,
        batch=batch,
        cfg=cfg,
        complete_gt_ids=complete_ids,
    )
    _progress(
        f"{progress_prefix}        watershed done in "
        f"{_format_duration(time.perf_counter() - step_start)}"
    )

    step_start = time.perf_counter()
    _progress(f"{progress_prefix}  [4/4] building/classifying RAG adjacency ...")
    supervoxels = ws_artifact["supervoxels"].cpu().numpy().astype(
        np.int32, copy=False
    )
    adjacency = supervoxel_adjacency(supervoxels)
    edge_info = classify_edges(
        adjacency,
        ws_summary["supervoxels"],
    )
    _progress(
        f"{progress_prefix}        RAG edge scan done in "
        f"{_format_duration(time.perf_counter() - step_start)}"
    )

    marker = ws_summary["marker"]
    sv = ws_summary["supervoxels"]

    safe = bool(
        marker["complete_gt_marker_recall"] >= 1.0
        and sv["complete_gt_min_coverage"]
        >= stage03.MIN_COMPLETE_GT_COVERAGE
        and sv["cross_gt_unsafe_supervoxel_count"] == 0
        and sv["complete_gt_recoverable_fraction"] >= 1.0
    )

    pos = int(edge_info["merge_edge_count_valid"])
    neg = int(edge_info["separate_edge_count_valid"])

    if not safe:
        class_priority = CLASS_UNSAFE
        case_class = "unsafe"
    elif pos > 0 and neg > 0:
        class_priority = CLASS_SAFE_BOTH_CLASSES
        case_class = "safe_both_classes"
    elif pos > 0:
        class_priority = CLASS_SAFE_MERGE_ONLY
        case_class = "safe_merge_only"
    else:
        class_priority = CLASS_SAFE_NEGATIVE_ONLY
        case_class = "safe_negative_only"

    # Score is only used within/near the transparent class hierarchy above.
    # Positive merge edges are weighted more strongly because they are the key
    # behavior missing from the Stage-03 crop.
    utility_score = (
        1000.0 * class_priority
        + 50.0 * min(pos, 8)
        + 8.0 * min(neg, 16)
        + 2.0 * min(len(complete_ids), 12)
        - 0.15 * max(
            int(sv["supervoxel_count"]) - 24,
            0,
        )
    )

    summary = {
        "candidate_index": int(
            scene.selection.get("candidate_index", -1)
        ),
        "case_class": case_class,
        "class_priority": class_priority,
        "utility_score": float(utility_score),
        "safe_proposal": safe,
        "crop_slices_zyx": _slice_pairs(scene.core_slices),
        "crop_shape_zyx": list(_slice_shape(scene.core_slices)),
        "merge_source_id": int(scene.selection["merge_source_id"]),
        "merge_gt_ids": [
            int(v) for v in scene.selection["merge_gt_ids"]
        ],
        "free_gt_id": int(scene.selection["free_gt_id"]),
        "complete_gt_ids": complete_ids,
        "partial_gt_ids": partial_ids,
        "stage00_score": float(scene.selection["score"]),
        "oracle_marker_count": int(marker["marker_count"]),
        "oracle_supervoxel_count": int(sv["supervoxel_count"]),
        "oracle_marker_recall_complete": float(
            marker["complete_gt_marker_recall"]
        ),
        "oracle_min_complete_coverage": float(
            sv["complete_gt_min_coverage"]
        ),
        "oracle_cross_gt_supervoxel_count": int(
            sv["cross_gt_unsafe_supervoxel_count"]
        ),
        "oracle_recoverable_fraction": float(
            sv["complete_gt_recoverable_fraction"]
        ),
        "rag_edges": edge_info,
    }

    _progress(
        f"{progress_prefix}  candidate analysis finished in "
        f"{_format_duration(time.perf_counter() - candidate_start)}"
    )

    artifact = {
        "targets": targets,
        "target_batch": target_batch,
        "watershed_summary": ws_summary,
        "watershed_artifact": ws_artifact,
        "edge_info": edge_info,
        "edge_target_interface": torch.from_numpy(
            edge_target_interface(supervoxels, edge_info)
        ),
    }
    return summary, artifact


# =============================================================================
# RANKING / PRINTING
# =============================================================================

def rank_key(row: dict[str, Any]) -> tuple:
    edge = row["rag_edges"]
    return (
        int(row["class_priority"]),
        int(edge["merge_edge_count_valid"]),
        min(int(edge["separate_edge_count_valid"]), 16),
        float(row["oracle_min_complete_coverage"]),
        -int(row["oracle_supervoxel_count"]),
        float(row["stage00_score"]),
    )


def print_scan_table(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 150)
    print("Stage-04 candidate scan — oracle watershed -> RAG edge usefulness")
    print("=" * 150)
    print(
        f"{'rank':>4s} {'cand':>4s} {'class':22s} {'SV':>4s} "
        f"{'mergeE':>7s} {'sepE':>6s} {'adjE':>6s} "
        f"{'crossGT':>7s} {'minCov':>7s} {'recover':>7s} "
        f"{'mergeGT':>14s} {'free':>5s} {'shape':>18s}"
    )
    print("-" * 150)

    ordered = sorted(rows, key=rank_key, reverse=True)
    for rank, row in enumerate(ordered, 1):
        edge = row["rag_edges"]
        print(
            f"{rank:4d} "
            f"{int(row['candidate_index']):4d} "
            f"{row['case_class']:22s} "
            f"{int(row['oracle_supervoxel_count']):4d} "
            f"{int(edge['merge_edge_count_valid']):7d} "
            f"{int(edge['separate_edge_count_valid']):6d} "
            f"{int(edge['adjacency_edge_count']):6d} "
            f"{int(row['oracle_cross_gt_supervoxel_count']):7d} "
            f"{float(row['oracle_min_complete_coverage']):7.3f} "
            f"{float(row['oracle_recoverable_fraction']):7.3f} "
            f"{str(tuple(row['merge_gt_ids'])):>14s} "
            f"{int(row['free_gt_id']):5d} "
            f"{str(tuple(row['crop_shape_zyx'])):>18s}"
        )
    print("=" * 150)


# =============================================================================
# SAVE SELECTED CASE
# =============================================================================

def save_selected_case(
    args: argparse.Namespace,
    base_scene,
    candidate_record,
    candidate_summary: dict[str, Any],
    candidate_artifact: dict[str, Any],
) -> None:
    crop, original_selection = candidate_record
    selection = dict(original_selection)
    selection["stage04_case_class"] = candidate_summary["case_class"]
    selection["stage04_utility_score"] = candidate_summary["utility_score"]
    selection["stage04_rag_edge_counts"] = {
        "merge_valid": candidate_summary["rag_edges"][
            "merge_edge_count_valid"
        ],
        "separate_valid": candidate_summary["rag_edges"][
            "separate_edge_count_valid"
        ],
        "adjacency_total": candidate_summary["rag_edges"][
            "adjacency_edge_count"
        ],
    }
    selection["stage04_oracle_supervoxel_count"] = candidate_summary[
        "oracle_supervoxel_count"
    ]

    scene = make_candidate_scene(
        base_scene,
        crop,
        selection,
        args.context_margin_dref,
    )
    targets = stage00.build_targets(scene)
    spatial_inputs = stage00.build_saved_spatial_inputs(scene)
    relative_core = stage00._relative_slices(
        scene.core_slices,
        scene.build_slices,
    )

    stage00.save_debug_sample(
        args.output,
        scene,
        targets,
        spatial_inputs,
        relative_core,
        source_data_dir=args.data_dir,
        time_index=args.time_index,
    )

    current = np.asarray(
        scene.current_full[scene.core_slices]
    ).astype(np.int64, copy=True)
    gt = np.asarray(
        scene.gt_full[scene.core_slices]
    ).astype(np.int64, copy=True)
    raw = np.asarray(
        scene.raw_full[scene.core_slices]
    ).astype(np.float32, copy=True)

    visual_payload = {
        "format_version": 1,
        "kind": "stirnet_rag_case_selection",
        "candidate_summary": candidate_summary,
        "spacing_um": torch.from_numpy(
            scene.spacing_zyx_um.copy()
        ).float(),
        "raw": torch.from_numpy(raw).float(),
        "current_labels": torch.from_numpy(current).long(),
        "gt_labels": torch.from_numpy(gt).long(),
        "oracle_supervoxels": candidate_artifact[
            "watershed_artifact"
        ]["supervoxels"].long(),
        "rag_edge_target_interface": candidate_artifact[
            "edge_target_interface"
        ].to(torch.uint8),
    }
    atomic_torch_save(
        args.result_dir / "selected_case.pt",
        visual_payload,
    )


def candidate_artifact_path(
    result_dir: Path,
    candidate_index: int,
) -> Path:
    return (
        result_dir
        / "candidates"
        / f"candidate_{int(candidate_index):02d}.pt"
    )


def save_candidate_visual_artifact(
    path: Path,
    scene,
    summary: dict[str, Any],
    artifact: dict[str, Any],
) -> None:
    current = np.asarray(
        scene.current_full[scene.core_slices]
    ).astype(np.int64, copy=True)
    gt = np.asarray(
        scene.gt_full[scene.core_slices]
    ).astype(np.int64, copy=True)
    raw = np.asarray(
        scene.raw_full[scene.core_slices]
    ).astype(np.float32, copy=True)

    payload = {
        "format_version": 1,
        "kind": "stirnet_rag_case_candidate",
        "candidate_summary": summary,
        "spacing_um": torch.from_numpy(
            scene.spacing_zyx_um.copy()
        ).float(),
        "raw": torch.from_numpy(raw).float(),
        "current_labels": torch.from_numpy(current).long(),
        "gt_labels": torch.from_numpy(gt).long(),
        "oracle_supervoxels": artifact[
            "watershed_artifact"
        ]["supervoxels"].long(),
        "rag_edge_target_interface": artifact[
            "edge_target_interface"
        ].to(torch.uint8),
    }
    atomic_torch_save(path, payload)


def build_candidate_visual_artifact(
    args: argparse.Namespace,
    candidate_index: int,
) -> Path:
    """Analyze just one candidate and cache its Napari artifact.

    This avoids rerunning the full 12-candidate scan. On first inspection the
    chosen candidate still needs its oracle targets/watershed (~one candidate's
    runtime). Subsequent views load the cached .pt directly.
    """
    path = candidate_artifact_path(
        args.result_dir,
        candidate_index,
    )
    if path.exists() and not args.rebuild_candidate:
        _progress(
            f"[candidate {candidate_index}] Using cached artifact: {path}"
        )
        return path

    base_scene, candidates = load_full_scene_and_candidates(args)
    candidate_by_index = {
        int(selection.get("candidate_index", i)): (crop, selection)
        for i, (crop, selection) in enumerate(candidates)
    }

    if candidate_index not in candidate_by_index:
        available = sorted(candidate_by_index)
        raise IndexError(
            f"candidate-index {candidate_index} is unavailable. "
            f"Available indices: {available}"
        )

    crop, selection = candidate_by_index[candidate_index]
    selection = dict(selection)
    selection["candidate_index"] = int(candidate_index)

    scene = make_candidate_scene(
        base_scene,
        crop,
        selection,
        args.context_margin_dref,
    )

    _progress("")
    _progress(
        f"[candidate {candidate_index}] Building visualization artifact | "
        f"merge GT {tuple(selection['merge_gt_ids'])} | "
        f"free GT {selection['free_gt_id']} | shape {_slice_shape(crop)}"
    )

    cfg = stage03.stage01.build_debug_config()
    summary, artifact = analyze_candidate(
        scene,
        cfg,
        progress_prefix=f"[candidate {candidate_index}]",
    )

    save_candidate_visual_artifact(
        path,
        scene,
        summary,
        artifact,
    )
    _progress(
        f"[candidate {candidate_index}] Cached for future instant viewing: {path}"
    )
    return path


# =============================================================================
# NAPARI
# =============================================================================

def visualize_artifact(
    artifact_path: Path,
    *,
    title_prefix: str = "Stage-04 RAG case",
) -> None:
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing visualization artifact: {artifact_path}")

    try:
        import napari
    except ImportError as exc:
        raise RuntimeError("Napari is required for --visualize.") from exc

    data = torch_load(artifact_path)
    scale = tuple(float(v) for v in data["spacing_um"].tolist())

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(
        data["raw"].numpy(),
        name="Raw",
        scale=scale,
        colormap="gray",
        visible=True,
    )
    viewer.add_labels(
        data["current_labels"].numpy(),
        name="Input current labels",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        data["gt_labels"].numpy(),
        name="GT labels",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        data["oracle_supervoxels"].numpy(),
        name="Oracle supervoxels",
        scale=scale,
        visible=True,
    )
    viewer.add_labels(
        data["rag_edge_target_interface"].numpy(),
        name="RAG edge targets (1 merge, 2 separate)",
        scale=scale,
        visible=False,
    )

    summary = data["candidate_summary"]
    edge = summary["rag_edges"]

    print(f"\n{title_prefix}")
    print(f"  candidate index : {summary['candidate_index']}")
    print(f"  class           : {summary['case_class']}")
    print(
        "  valid edges     : "
        f"{edge['merge_edge_count_valid']} merge / "
        f"{edge['separate_edge_count_valid']} separate"
    )
    print(f"  supervoxels     : {summary['oracle_supervoxel_count']}")
    print("  Napari loads exactly 5 layers.")
    napari.run()


def visualize_selected(result_dir: Path) -> None:
    visualize_artifact(
        result_dir / "selected_case.pt",
        title_prefix="Selected Stage-04 RAG case",
    )


def visualize_candidate(
    args: argparse.Namespace,
    candidate_index: int,
) -> None:
    artifact_path = build_candidate_visual_artifact(
        args,
        candidate_index,
    )
    visualize_artifact(
        artifact_path,
        title_prefix=f"Stage-04 candidate {candidate_index}",
    )


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Stage-00 candidates with oracle Stage-03 watershed and "
            "select a useful RAG-debug case."
        )
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
        help=(
            "Existing debug_crop.pt containing Stage-00 candidate_summaries. "
            "Used to avoid expensive candidate re-enumeration."
        ),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SELECTED_SAMPLE,
        help="Saved reusable RAG-debug crop.",
    )
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=None,
        help=(
            "Stage-00 candidate to inspect. With --visualize, only this "
            "candidate is analyzed/cached and opened; the full scan is not rerun."
        ),
    )
    parser.add_argument(
        "--rebuild-candidate",
        action="store_true",
        help=(
            "Recompute a candidate visualization artifact even if it is cached."
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Open the selected case, or --candidate-index N when provided."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.stage00_sample = args.stage00_sample.resolve()
    args.result_dir = args.result_dir.resolve()
    args.output = args.output.resolve()

    if args.visualize:
        if args.candidate_index is None:
            visualize_selected(args.result_dir)
        else:
            visualize_candidate(
                args,
                int(args.candidate_index),
            )
        return

    if args.max_candidates < 1:
        raise ValueError("--max-candidates must be >= 1")
    if args.voxel_budget < 1:
        raise ValueError("--voxel-budget must be positive")

    print("\n" + "=" * 112)
    print("STIR-Net Stage 04 — select a RAG-debug case")
    print("=" * 112)
    print(f"Data dir              : {args.data_dir}")
    print(f"Time index            : {args.time_index}")
    print(f"Voxel budget          : {args.voxel_budget:,}")
    print(f"Stage-00 candidates   : up to {args.max_candidates}")
    print(f"Candidate source      : {args.stage00_sample}")
    print("Candidate mining      : reuse saved Stage-00 list")
    print("Geometry for scan     : ORACLE production targets")
    print("Watershed             : exact Stage-03 production path")
    print("Selection goal        : safe proposal + same-GT and different-GT RAG edges")
    print("Progress reporting    : per-candidate substages + running-average ETA")
    print("=" * 112, flush=True)

    start = time.perf_counter()
    base_scene, candidates = load_full_scene_and_candidates(args)

    cfg = stage03.stage01.build_debug_config()
    rows: list[dict[str, Any]] = []
    artifacts_by_index: dict[int, dict[str, Any]] = {}

    candidate_times: list[float] = []

    for ordinal, (crop, selection) in enumerate(candidates):
        selection = dict(selection)
        selection["candidate_index"] = int(
            selection.get("candidate_index", ordinal)
        )
        scene = make_candidate_scene(
            base_scene,
            crop,
            selection,
            args.context_margin_dref,
        )

        completed = ordinal
        total = len(candidates)
        percent = 100.0 * completed / max(total, 1)

        if candidate_times:
            mean_seconds = float(np.mean(candidate_times))
            eta_seconds = mean_seconds * (total - completed)
            eta_text = _format_duration(eta_seconds)
            avg_text = _format_duration(mean_seconds)
        else:
            eta_text = "estimating after first candidate"
            avg_text = "n/a"

        prefix = f"[{ordinal + 1:02d}/{total:02d}]"
        _progress("")
        _progress(
            f"{prefix} START candidate {selection['candidate_index']} | "
            f"{percent:5.1f}% complete | ETA {eta_text}"
        )
        _progress(
            f"{prefix}       merge GT {tuple(selection['merge_gt_ids'])} | "
            f"free GT {selection['free_gt_id']} | shape {_slice_shape(crop)} | "
            f"running avg {avg_text}"
        )

        candidate_start = time.perf_counter()
        summary, artifact = analyze_candidate(
            scene,
            cfg,
            progress_prefix=prefix,
        )
        candidate_elapsed = time.perf_counter() - candidate_start
        candidate_times.append(candidate_elapsed)

        rows.append(summary)
        artifacts_by_index[int(summary["candidate_index"])] = artifact

        mean_seconds = float(np.mean(candidate_times))
        remaining = total - (ordinal + 1)
        eta_seconds = mean_seconds * remaining
        edge = summary["rag_edges"]

        _progress(
            f"{prefix} DONE in {_format_duration(candidate_elapsed)} | "
            f"SV={summary['oracle_supervoxel_count']} | "
            f"mergeE={edge['merge_edge_count_valid']} | "
            f"sepE={edge['separate_edge_count_valid']} | "
            f"class={summary['case_class']}"
        )
        _progress(
            f"{prefix} Progress: {ordinal + 1}/{total} "
            f"({100.0 * (ordinal + 1) / total:5.1f}%) | "
            f"avg/candidate {_format_duration(mean_seconds)} | "
            f"estimated remaining {_format_duration(eta_seconds)}"
        )

    ordered = sorted(rows, key=rank_key, reverse=True)
    best = ordered[0]
    best_index = int(best["candidate_index"])
    candidate_by_index = {
        int(selection.get("candidate_index", i)): (crop, selection)
        for i, (crop, selection) in enumerate(candidates)
    }

    print_scan_table(rows)

    edge = best["rag_edges"]
    print("\nSelected candidate")
    print(f"  candidate index      : {best_index}")
    print(f"  class                : {best['case_class']}")
    print(f"  safe proposal        : {best['safe_proposal']}")
    print(f"  oracle supervoxels   : {best['oracle_supervoxel_count']}")
    print(
        "  valid RAG edges      : "
        f"{edge['merge_edge_count_valid']} merge / "
        f"{edge['separate_edge_count_valid']} separate"
    )
    print(f"  merge GT             : {tuple(best['merge_gt_ids'])}")
    print(f"  free GT              : {best['free_gt_id']}")
    print(f"  crop shape           : {tuple(best['crop_shape_zyx'])}")

    if best["case_class"] != "safe_both_classes":
        print(
            "\n[WARN] No scanned candidate provided a safe graph with both "
            "merge and separate edge targets. The best available case is still "
            "saved, but consider rerunning with --max-candidates 24 or broadening "
            "the search to the full volume before Stage-05 RAG overfit."
        )

    args.result_dir.mkdir(parents=True, exist_ok=True)

    scan_payload = {
        "format_version": 1,
        "kind": "stirnet_rag_case_selection_scan",
        "source_data_dir": str(args.data_dir),
        "time_index": int(args.time_index),
        "voxel_budget": int(args.voxel_budget),
        "candidate_count": len(rows),
        "selection_rule": {
            "primary": (
                "safe proposal, then both edge classes, then maximize valid "
                "same-GT merge edges, then different-GT separation edges"
            ),
            "oracle_geometry": True,
            "reason": (
                "Stage-01 checkpoint intentionally overfit one crop; oracle "
                "geometry isolates RAG-case suitability from dense-network "
                "generalization."
            ),
        },
        "selected_candidate_index": best_index,
        "selected_case_class": best["case_class"],
        "ranked_candidates": ordered,
        "candidate_elapsed_seconds": [
            float(value) for value in candidate_times
        ],
        "mean_candidate_seconds": (
            float(np.mean(candidate_times)) if candidate_times else 0.0
        ),
        "elapsed_seconds": float(time.perf_counter() - start),
    }
    atomic_json(
        args.result_dir / "case_scan.json",
        scan_payload,
    )

    save_selected_case(
        args,
        base_scene,
        candidate_by_index[best_index],
        best,
        artifacts_by_index[best_index],
    )

    elapsed = time.perf_counter() - start
    print("\n" + "=" * 112)
    print("Stage-04 case selection complete")
    print("=" * 112)
    print(f"Scan report           : {args.result_dir / 'case_scan.json'}")
    print(f"Selected-case artifact: {args.result_dir / 'selected_case.pt'}")
    print(f"Reusable RAG crop     : {args.output}")
    print(f"Elapsed               : {elapsed:.2f} s")
    print("\nVisualize selected case with:")
    print(
        "  python investigations/stirnet/"
        "04_rag_case_selection.py --visualize"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
