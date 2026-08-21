from __future__ import annotations

"""
Stage 05: validate the STIR-Net post-watershed Supervoxel Safety Guard.

Purpose
-------
Stage 04 selects a useful RAG-debug case, but its Stage-03 helper deliberately
stops at the preliminary watershed + cleanup result. It therefore does NOT
exercise the new post-watershed safety guard.

This stage answers one narrow question:

    Does the new Supervoxel Safety Guard eliminate dangerous cross-GT atomic
    supervoxels without destroying foreground coverage or causing pathological
    fragmentation?

The experiment uses ORACLE dense geometry on the same saved Stage-00 candidate
set. No neural-network training occurs here. Oracle geometry isolates the
deterministic proposal machinery from one-crop dense-network generalization.

For every candidate this stage measures:

    oracle geometry
        -> markers
        -> preliminary marker-controlled watershed
        -> tiny-region cleanup
        -> PRELIMINARY supervoxels
        -> Supervoxel Safety Guard
        -> GUARDED supervoxels

Primary acceptance criterion
----------------------------
    final guarded cross-GT atomic supervoxels == 0
    across every scanned candidate.

Secondary checks
----------------
    * foreground support is exactly preserved by the guard
    * the guard never merges two preliminary supervoxels
    * every visible GT cell remains recoverable by unions of guarded SVs
    * fragmentation remains bounded
    * production LearnedGeometryWatershed agrees exactly with the explicit
      preliminary+guard diagnostic path

Typical use
-----------
Run the full saved Stage-00 candidate set:

    python investigations/stirnet/06_supervoxel_safety_guard_validation.py

Inspect one candidate:

    python investigations/stirnet/06_supervoxel_safety_guard_validation.py \
        --candidate-index 8 --visualize

Force recomputation of one cached candidate:

    python investigations/stirnet/06_supervoxel_safety_guard_validation.py \
        --candidate-index 8 --visualize --rebuild-candidate

Outputs
-------
    data/learned/stirnet/supervoxel_safety_guard_validation/
        validation_report.json
        candidates/candidate_XX.pt
        candidates/candidate_XX.json

Napari intentionally contains only six layers:
    Raw
    GT labels
    Preliminary supervoxels
    Guarded supervoxels
    Safety barrier
    Separator probability
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
    "_stirnet_stage00_gt_validation_stage05",
    "00_GT_validation.py",
)
stage03 = _import_adjacent(
    "_stirnet_stage03_watershed_stage05",
    "03_watershed_supervoxel_debug.py",
)
stage04 = _import_adjacent(
    "_stirnet_stage04_rag_case_stage05",
    "04_rag_case_selection.py",
)

try:
    from learned.stirnet.model.geometry.derived import build_geometry_derived_cache
    from learned.stirnet.model.partition.supervoxel_guard import (
        build_supervoxel_barrier,
        split_preliminary_supervoxels,
    )
    from learned.stirnet.model.partition.watershed import LearnedGeometryWatershed
except ImportError as exc:
    raise RuntimeError(
        "The Supervoxel Safety Guard production patch is not available.\n"
        "Apply apply_stirnet_supervoxel_safety_guard.py first, then rerun Stage 05."
    ) from exc


# =============================================================================
# DEFAULTS / HISTORICAL REGRESSION SETS
# =============================================================================

DEFAULT_DATA_DIR = stage00.DEFAULT_DATA_DIR
DEFAULT_TIME_INDEX = stage00.DEFAULT_TIME_INDEX
DEFAULT_MAX_CANDIDATES = stage00.DEFAULT_MAX_CANDIDATES
DEFAULT_CONTEXT_MARGIN_DREF = stage00.DEFAULT_CONTEXT_MARGIN_DREF

DEFAULT_STAGE00_SAMPLE = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "debug_crop.pt"
)

DEFAULT_RESULT_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "supervoxel_safety_guard_validation"
)

# From the pre-guard oracle scans. These are labels for reporting only; the
# current result is always recomputed and never inferred from these sets.
PREVIOUSLY_UNSAFE_CANDIDATES = frozenset({3, 4, 6, 7, 8, 9, 10})
PREVIOUS_CALIBRATION_RESIDUAL_CANDIDATES = frozenset({7, 8, 9, 10})

# Fragmentation is not a hard correctness failure until it becomes excessive.
# This threshold is deliberately generous; the graph can merge extra SVs.
DEFAULT_OVERFRAGMENTED_SV_PER_GT = 8


# =============================================================================
# GENERIC HELPERS
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
    print(message, flush=True)


def _slice_shape(slices: tuple[slice, slice, slice]) -> tuple[int, int, int]:
    return tuple(int(s.stop) - int(s.start) for s in slices)


def _tensor_cpu(value: Tensor, *, half: bool = False) -> Tensor:
    out = value.detach().cpu()
    if half and out.is_floating_point():
        out = out.half()
    return out


def candidate_pt_path(result_dir: Path, candidate_index: int) -> Path:
    return result_dir / "candidates" / f"candidate_{candidate_index:02d}.pt"


def candidate_json_path(result_dir: Path, candidate_index: int) -> Path:
    return result_dir / "candidates" / f"candidate_{candidate_index:02d}.json"


# =============================================================================
# STAGE-00 / STAGE-04 REUSE
# =============================================================================


def build_stage04_args(args: argparse.Namespace) -> argparse.Namespace:
    # stage04.load_full_scene_and_candidates reads these fields.
    return argparse.Namespace(
        data_dir=args.data_dir,
        time_index=args.time_index,
        voxel_budget=stage00.DEFAULT_VOXEL_BUDGET,
        max_candidates=args.max_candidates,
        context_margin_dref=args.context_margin_dref,
        stage00_sample=args.stage00_sample,
        result_dir=stage04.DEFAULT_RESULT_DIR,
        output=stage04.DEFAULT_SELECTED_SAMPLE,
        candidate_index=None,
        rebuild_candidate=False,
        visualize=False,
    )


def load_scene_index(args: argparse.Namespace):
    base_scene, candidates = stage04.load_full_scene_and_candidates(
        build_stage04_args(args)
    )
    by_index: dict[int, tuple[tuple[slice, slice, slice], dict[str, Any]]] = {}
    for ordinal, (crop, selection) in enumerate(candidates):
        selection = dict(selection)
        index = int(selection.get("candidate_index", ordinal))
        selection["candidate_index"] = index
        by_index[index] = (crop, selection)
    return base_scene, candidates, by_index


def make_scene(
    base_scene,
    crop: tuple[slice, slice, slice],
    selection: dict[str, Any],
    context_margin_dref: float,
):
    return stage04.make_candidate_scene(
        base_scene,
        crop,
        selection,
        context_margin_dref,
    )


def build_oracle_candidate(scene):
    """Build exact production targets and crop them to the Stage-00 core."""
    targets = stage00.build_targets(scene)
    target_batch = stage04.target_batch_for_stage03(scene, targets)
    batch = stage04.batch_namespace_for_stage03(scene, target_batch)
    geometry = stage03.oracle_geometry_from_targets(target_batch)
    complete_ids, partial_ids = stage00._complete_and_partial_ids(
        scene.gt_full,
        scene.core_slices,
    )
    return targets, target_batch, batch, geometry, complete_ids, partial_ids


# =============================================================================
# SAFETY / FRAGMENTATION DIAGNOSTICS
# =============================================================================


def _visible_recoverability(
    sv_summary: dict[str, Any],
) -> tuple[int, int, float, list[int]]:
    rows = sv_summary["per_gt"]
    total = len(rows)
    recovered: list[int] = []
    missing: list[int] = []
    for gt_id_text, row in rows.items():
        gt_id = int(gt_id_text)
        if bool(row["recoverable_by_merging"]):
            recovered.append(gt_id)
        else:
            missing.append(gt_id)
    fraction = len(recovered) / max(total, 1)
    return len(recovered), total, float(fraction), missing


def _fragmentation_summary(
    sv_summary: dict[str, Any],
    *,
    overfragmented_threshold: int,
) -> dict[str, Any]:
    counts = [
        int(row["meaningful_supervoxel_count"])
        for row in sv_summary["per_gt"].values()
    ]
    if not counts:
        return {
            "visible_gt_count": 0,
            "mean_meaningful_sv_per_gt": 0.0,
            "median_meaningful_sv_per_gt": 0.0,
            "max_meaningful_sv_per_gt": 0,
            "overfragmented_gt_count": 0,
            "overfragmented_gt_ids": [],
        }

    over_ids = [
        int(gt_id)
        for gt_id, row in sv_summary["per_gt"].items()
        if int(row["meaningful_supervoxel_count"]) > overfragmented_threshold
    ]
    return {
        "visible_gt_count": len(counts),
        "mean_meaningful_sv_per_gt": float(np.mean(counts)),
        "median_meaningful_sv_per_gt": float(np.median(counts)),
        "max_meaningful_sv_per_gt": int(max(counts)),
        "overfragmented_gt_count": len(over_ids),
        "overfragmented_gt_ids": over_ids,
    }


def _cross_gt_ids(summary: dict[str, Any]) -> list[int]:
    return [
        int(row["supervoxel_id"])
        for row in summary["nodes"]
        if bool(row["cross_gt_unsafe"])
    ]


def _descendant_mapping(
    preliminary: np.ndarray,
    guarded: np.ndarray,
) -> tuple[dict[int, list[int]], int]:
    """Map each preliminary SV to guarded descendants and detect illegal merges."""
    mapping: dict[int, list[int]] = {}
    for pre_id in range(1, int(preliminary.max()) + 1):
        descendants = np.unique(guarded[preliminary == pre_id])
        descendants = descendants[descendants > 0]
        mapping[pre_id] = [int(v) for v in descendants.tolist()]

    merge_violations = 0
    for guarded_id in range(1, int(guarded.max()) + 1):
        ancestors = np.unique(preliminary[guarded == guarded_id])
        ancestors = ancestors[ancestors > 0]
        if ancestors.size > 1:
            merge_violations += 1
    return mapping, merge_violations


def _repair_rows(
    preliminary_summary: dict[str, Any],
    guarded_summary: dict[str, Any],
    descendant_mapping: dict[int, list[int]],
) -> list[dict[str, Any]]:
    pre_rows = {
        int(row["supervoxel_id"]): row
        for row in preliminary_summary["nodes"]
    }
    guard_rows = {
        int(row["supervoxel_id"]): row
        for row in guarded_summary["nodes"]
    }

    rows: list[dict[str, Any]] = []
    for pre_id, descendants in descendant_mapping.items():
        pre = pre_rows.get(pre_id)
        if pre is None:
            continue
        if len(descendants) <= 1 and not bool(pre["cross_gt_unsafe"]):
            continue
        descendant_cross = [
            int(sv_id)
            for sv_id in descendants
            if bool(guard_rows.get(sv_id, {}).get("cross_gt_unsafe", False))
        ]
        rows.append(
            {
                "preliminary_supervoxel_id": int(pre_id),
                "preliminary_cross_gt_unsafe": bool(pre["cross_gt_unsafe"]),
                "preliminary_meaningful_gt_ids": [
                    int(v) for v in pre["meaningful_gt_ids"]
                ],
                "guarded_descendant_ids": descendants,
                "guarded_cross_gt_descendant_ids": descendant_cross,
                "split_by_guard": len(descendants) > 1,
                "cross_gt_repaired": bool(
                    pre["cross_gt_unsafe"] and not descendant_cross
                ),
            }
        )
    return rows


def _cue_counts(
    barrier: np.ndarray,
    evidence: dict[str, np.ndarray],
    foreground: np.ndarray,
) -> dict[str, int]:
    fg = np.asarray(foreground, dtype=bool)
    return {
        "barrier_voxels_in_foreground": int(np.sum(barrier & fg)),
        "strong_separator_voxels_in_foreground": int(
            np.sum(evidence["strong_separator"] & fg)
        ),
        "separator_corroborated_voxels_in_foreground": int(
            np.sum(evidence["separator_corroborated"] & fg)
        ),
        "geometry_only_voxels_in_foreground": int(
            np.sum(evidence["geometry_only"] & fg)
        ),
        "valley_support_voxels_in_foreground": int(
            np.sum(evidence["valley_support"] & fg)
        ),
    }


def _true_volume_boundary_ids(
    gt_full: np.ndarray,
    crop: tuple[slice, slice, slice],
) -> list[int]:
    """Visible IDs in crop that also touch the true acquisition-volume boundary."""
    visible = {
        int(v)
        for v in np.unique(gt_full[crop]).tolist()
        if int(v) > 0
    }
    if not visible:
        return []
    boundary_ids: set[int] = set()
    for face in (
        gt_full[0],
        gt_full[-1],
        gt_full[:, 0],
        gt_full[:, -1],
        gt_full[:, :, 0],
        gt_full[:, :, -1],
    ):
        boundary_ids.update(
            int(v) for v in np.unique(face).tolist() if int(v) > 0
        )
    return sorted(visible & boundary_ids)


# =============================================================================
# ONE-CANDIDATE ANALYSIS
# =============================================================================


@torch.no_grad()
def analyze_candidate(
    scene,
    cfg,
    *,
    overfragmented_threshold: int,
    verify_production_path: bool,
    progress_prefix: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.perf_counter()

    _progress(f"{progress_prefix}  [1/6] building oracle production targets ...")
    t0 = time.perf_counter()
    (
        targets,
        target_batch,
        batch,
        geometry,
        complete_ids,
        partial_ids,
    ) = build_oracle_candidate(scene)
    target_seconds = time.perf_counter() - t0
    _progress(
        f"{progress_prefix}        targets + oracle geometry: "
        f"{_format_duration(target_seconds)}"
    )

    _progress(
        f"{progress_prefix}  [2/6] deriving marker score / energy / markers ..."
    )
    t0 = time.perf_counter()
    cache = build_geometry_derived_cache(
        geometry,
        cfg.partition,
        padding_mask=None,
    )
    foreground = cache.foreground_mask[0]
    markers = stage03.build_exact_markers(
        cache.seed_score[0, 0],
        foreground,
        batch.spacing_um[0],
        batch.dref_um[0],
        cfg,
    )
    marker_seconds = time.perf_counter() - t0
    _progress(
        f"{progress_prefix}        markers={int(markers.max())} in "
        f"{_format_duration(marker_seconds)}"
    )

    _progress(
        f"{progress_prefix}  [3/6] running PRELIMINARY watershed (guard bypassed) ..."
    )
    t0 = time.perf_counter()
    raw_watershed, preliminary = stage03.run_exact_watershed(
        cache.watershed_energy[0, 0],
        markers,
        foreground,
        cfg,
    )
    preliminary_seconds = time.perf_counter() - t0
    _progress(
        f"{progress_prefix}        raw={int(raw_watershed.max())} "
        f"preliminary={int(preliminary.max())} in "
        f"{_format_duration(preliminary_seconds)}"
    )

    gt_np = batch.gt_labels.detach().cpu().numpy().astype(
        np.int32, copy=False
    )
    marker_summary = stage03.marker_diagnostics(
        markers,
        gt_np,
        complete_ids,
    )
    preliminary_summary, _ = stage03.supervoxel_diagnostics(
        preliminary,
        gt_np,
        complete_ids,
        cfg,
    )

    _progress(
        f"{progress_prefix}  [4/6] applying Supervoxel Safety Guard explicitly ..."
    )
    t0 = time.perf_counter()
    separator_np = (
        cache.separator_prob[0, 0].detach().float().cpu().numpy()
    )
    seed_np = cache.seed_prob[0, 0].detach().float().cpu().numpy()
    sdf_norm_np = (
        cache.sdf_normalized[0, 0].detach().float().cpu().numpy()
    )
    flow_np = geometry.flow[0].detach().float().cpu().numpy()
    centroid_np = (
        geometry.centroid_offset[0].detach().float().cpu().numpy()
    )
    spacing_np = batch.spacing_um[0].detach().float().cpu().numpy()
    dref_value = float(batch.dref_um[0].item())

    barrier, evidence = build_supervoxel_barrier(
        separator_np,
        centroid_np,
        flow_np,
        seed_np,
        sdf_norm_np,
        spacing_np,
        dref_value,
        cfg.partition,
    )
    guarded, guard_diagnostics = split_preliminary_supervoxels(
        preliminary,
        separator_np,
        centroid_np,
        flow_np,
        seed_np,
        sdf_norm_np,
        spacing_np,
        dref_value,
        cfg.partition,
    )
    guard_seconds = time.perf_counter() - t0

    guarded_summary, _ = stage03.supervoxel_diagnostics(
        guarded,
        gt_np,
        complete_ids,
        cfg,
    )
    _progress(
        f"{progress_prefix}        guarded={int(guarded.max())} | "
        f"split prelim SV={guard_diagnostics.split_supervoxel_count} | "
        f"added SV={guard_diagnostics.added_supervoxel_count} | "
        f"{_format_duration(guard_seconds)}"
    )

    _progress(f"{progress_prefix}  [5/6] validating safety invariants ...")
    pre_positive = preliminary > 0
    guarded_positive = guarded > 0
    foreground_support_equal = bool(
        np.array_equal(pre_positive, guarded_positive)
    )
    descendant_mapping, merge_violations = _descendant_mapping(
        preliminary,
        guarded,
    )
    repair_rows = _repair_rows(
        preliminary_summary,
        guarded_summary,
        descendant_mapping,
    )
    repaired_cross_gt_count = sum(
        bool(row["cross_gt_repaired"]) for row in repair_rows
    )

    pre_recovered, pre_total, pre_visible_fraction, pre_missing_visible = (
        _visible_recoverability(preliminary_summary)
    )
    (
        guarded_recovered,
        guarded_total,
        guarded_visible_fraction,
        guarded_missing_visible,
    ) = _visible_recoverability(guarded_summary)

    preliminary_fragmentation = _fragmentation_summary(
        preliminary_summary,
        overfragmented_threshold=overfragmented_threshold,
    )
    guarded_fragmentation = _fragmentation_summary(
        guarded_summary,
        overfragmented_threshold=overfragmented_threshold,
    )

    cue_counts = _cue_counts(
        barrier,
        evidence,
        foreground.detach().cpu().numpy(),
    )

    production_exact_match = None
    production_supervoxel_count = None
    production_seconds = None

    _progress(
        f"{progress_prefix}  [6/6] "
        + (
            "verifying exact production LearnedGeometryWatershed path ..."
            if verify_production_path
            else "production-path verification skipped ..."
        )
    )
    if verify_production_path:
        t0 = time.perf_counter()
        production_watershed = LearnedGeometryWatershed(cfg.partition)
        production_labels = production_watershed(
            geometry,
            batch.spacing_um,
            batch.dref_um,
            padding_mask=None,
            derived_cache=cache,
        )[0].detach().cpu().numpy().astype(np.int32, copy=False)
        production_seconds = time.perf_counter() - t0
        production_exact_match = bool(
            np.array_equal(production_labels, guarded)
        )
        production_supervoxel_count = int(production_labels.max())
        if not production_exact_match:
            raise RuntimeError(
                f"{progress_prefix} explicit safety-guard path does not match "
                "production LearnedGeometryWatershed output. This must be "
                "resolved before interpreting the diagnostic."
            )

    pre_cross = int(
        preliminary_summary["cross_gt_unsafe_supervoxel_count"]
    )
    post_cross = int(
        guarded_summary["cross_gt_unsafe_supervoxel_count"]
    )

    strict_pass = bool(
        post_cross == 0
        and foreground_support_equal
        and merge_violations == 0
        and guarded_visible_fraction >= 1.0
        and (
            production_exact_match is not False
        )
    )

    true_boundary_ids = _true_volume_boundary_ids(
        scene.gt_full,
        scene.core_slices,
    )

    candidate_index = int(scene.selection.get("candidate_index", -1))
    summary = {
        "candidate_index": candidate_index,
        "previous_status": (
            "previous_calibration_residual"
            if candidate_index in PREVIOUS_CALIBRATION_RESIDUAL_CANDIDATES
            else (
                "previously_unsafe"
                if candidate_index in PREVIOUSLY_UNSAFE_CANDIDATES
                else "previously_safe"
            )
        ),
        "strict_pass": strict_pass,
        "crop_shape_zyx": list(_slice_shape(scene.core_slices)),
        "merge_source_id": int(scene.selection["merge_source_id"]),
        "merge_gt_ids": [
            int(v) for v in scene.selection["merge_gt_ids"]
        ],
        "free_gt_id": int(scene.selection["free_gt_id"]),
        "complete_gt_ids": [int(v) for v in complete_ids],
        "partial_gt_ids": [int(v) for v in partial_ids],
        "true_volume_boundary_gt_ids_visible": true_boundary_ids,
        "marker": {
            "marker_count": int(marker_summary["marker_count"]),
            "complete_gt_marker_recall": float(
                marker_summary["complete_gt_marker_recall"]
            ),
            "markers_on_background": int(
                marker_summary["markers_on_background"]
            ),
        },
        "preliminary": {
            "raw_watershed_count": int(raw_watershed.max()),
            "supervoxel_count": int(preliminary.max()),
            "cross_gt_unsafe_supervoxel_count": pre_cross,
            "cross_gt_unsafe_supervoxel_ids": _cross_gt_ids(
                preliminary_summary
            ),
            "complete_gt_min_coverage": float(
                preliminary_summary["complete_gt_min_coverage"]
            ),
            "complete_gt_recoverable_fraction": float(
                preliminary_summary[
                    "complete_gt_recoverable_fraction"
                ]
            ),
            "visible_gt_recoverable_count": pre_recovered,
            "visible_gt_count": pre_total,
            "visible_gt_recoverable_fraction": pre_visible_fraction,
            "visible_gt_unrecoverable_ids": pre_missing_visible,
            "rag_valid_supervoxel_fraction": float(
                preliminary_summary[
                    "rag_valid_supervoxel_fraction"
                ]
            ),
            "fragmentation": preliminary_fragmentation,
        },
        "guard": {
            "split_preliminary_supervoxel_count": int(
                guard_diagnostics.split_supervoxel_count
            ),
            "added_supervoxel_count": int(
                guard_diagnostics.added_supervoxel_count
            ),
            "barrier_voxel_count": int(
                guard_diagnostics.barrier_voxel_count
            ),
            "cue_counts": cue_counts,
            "repaired_cross_gt_preliminary_supervoxel_count": int(
                repaired_cross_gt_count
            ),
            "repair_rows": repair_rows,
        },
        "guarded": {
            "supervoxel_count": int(guarded.max()),
            "cross_gt_unsafe_supervoxel_count": post_cross,
            "cross_gt_unsafe_supervoxel_ids": _cross_gt_ids(
                guarded_summary
            ),
            "complete_gt_min_coverage": float(
                guarded_summary["complete_gt_min_coverage"]
            ),
            "complete_gt_recoverable_fraction": float(
                guarded_summary["complete_gt_recoverable_fraction"]
            ),
            "visible_gt_recoverable_count": guarded_recovered,
            "visible_gt_count": guarded_total,
            "visible_gt_recoverable_fraction": guarded_visible_fraction,
            "visible_gt_unrecoverable_ids": guarded_missing_visible,
            "rag_valid_supervoxel_fraction": float(
                guarded_summary["rag_valid_supervoxel_fraction"]
            ),
            "fragmentation": guarded_fragmentation,
        },
        "invariants": {
            "foreground_support_exactly_preserved": foreground_support_equal,
            "guard_merge_violation_count": int(merge_violations),
            "production_path_exact_match": production_exact_match,
            "production_supervoxel_count": production_supervoxel_count,
        },
        "timing_seconds": {
            "targets_and_oracle_geometry": float(target_seconds),
            "marker_generation": float(marker_seconds),
            "preliminary_watershed": float(preliminary_seconds),
            "safety_guard": float(guard_seconds),
            "production_path_verification": (
                None
                if production_seconds is None
                else float(production_seconds)
            ),
            "total": float(time.perf_counter() - start),
        },
    }

    artifact = {
        "format_version": 1,
        "kind": "stirnet_supervoxel_safety_guard_validation_candidate",
        "summary": summary,
        "spacing_um": batch.spacing_um[0].detach().cpu().float(),
        "raw": torch.from_numpy(
            np.asarray(scene.raw_full[scene.core_slices]).astype(
                np.float32, copy=True
            )
        ),
        "gt_labels": batch.gt_labels.detach().cpu().long(),
        "markers": torch.from_numpy(
            markers.astype(np.int32, copy=False)
        ),
        "preliminary_supervoxels": torch.from_numpy(
            preliminary.astype(np.int32, copy=False)
        ),
        "guarded_supervoxels": torch.from_numpy(
            guarded.astype(np.int32, copy=False)
        ),
        "safety_barrier": torch.from_numpy(
            barrier.astype(np.uint8, copy=False)
        ),
        "separator_probability": _tensor_cpu(
            cache.separator_prob[0, 0], half=True
        ),
        "centroid_vote_disagreement": torch.from_numpy(
            evidence["centroid_vote_disagreement"].astype(
                np.float32, copy=False
            )
        ).half(),
        "flow_disagreement": torch.from_numpy(
            evidence["flow_disagreement"].astype(
                np.float32, copy=False
            )
        ).half(),
    }
    return summary, artifact


# =============================================================================
# REPORTING / ACCEPTANCE
# =============================================================================


def print_candidate_table(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 150)
    print("Stage-05 Supervoxel Safety Guard validation")
    print("=" * 150)
    print(
        f"{'cand':>4s} {'previous':>26s} "
        f"{'preSV':>6s} {'postSV':>7s} "
        f"{'preX':>5s} {'postX':>6s} "
        f"{'splits':>6s} {'added':>6s} "
        f"{'visRec':>7s} {'maxSV/GT':>8s} "
        f"{'RAGvalid':>8s} {'PASS':>6s}"
    )
    print("-" * 150)
    for row in sorted(rows, key=lambda x: int(x["candidate_index"])):
        pre = row["preliminary"]
        guard = row["guard"]
        post = row["guarded"]
        print(
            f"{int(row['candidate_index']):4d} "
            f"{row['previous_status']:>26s} "
            f"{int(pre['supervoxel_count']):6d} "
            f"{int(post['supervoxel_count']):7d} "
            f"{int(pre['cross_gt_unsafe_supervoxel_count']):5d} "
            f"{int(post['cross_gt_unsafe_supervoxel_count']):6d} "
            f"{int(guard['split_preliminary_supervoxel_count']):6d} "
            f"{int(guard['added_supervoxel_count']):6d} "
            f"{float(post['visible_gt_recoverable_fraction']):7.3f} "
            f"{int(post['fragmentation']['max_meaningful_sv_per_gt']):8d} "
            f"{float(post['rag_valid_supervoxel_fraction']):8.3f} "
            f"{'YES' if row['strict_pass'] else 'NO':>6s}"
        )
    print("=" * 150)


def aggregate_report(
    rows: list[dict[str, Any]],
    cfg,
    *,
    overfragmented_threshold: int,
) -> dict[str, Any]:
    pre_cross = sum(
        int(row["preliminary"]["cross_gt_unsafe_supervoxel_count"])
        for row in rows
    )
    post_cross = sum(
        int(row["guarded"]["cross_gt_unsafe_supervoxel_count"])
        for row in rows
    )
    repaired = sum(
        int(
            row["guard"][
                "repaired_cross_gt_preliminary_supervoxel_count"
            ]
        )
        for row in rows
    )
    added_sv = sum(
        int(row["guard"]["added_supervoxel_count"])
        for row in rows
    )
    unsafe_indices = [
        int(row["candidate_index"])
        for row in rows
        if int(row["guarded"]["cross_gt_unsafe_supervoxel_count"]) > 0
    ]
    failed_indices = [
        int(row["candidate_index"])
        for row in rows
        if not bool(row["strict_pass"])
    ]
    regression_failures = [
        int(row["candidate_index"])
        for row in rows
        if (
            row["previous_status"] == "previously_safe"
            and not bool(row["strict_pass"])
        )
    ]
    residual_repair_failures = sorted(
        set(unsafe_indices) & PREVIOUS_CALIBRATION_RESIDUAL_CANDIDATES
    )

    foreground_ok = all(
        bool(
            row["invariants"][
                "foreground_support_exactly_preserved"
            ]
        )
        for row in rows
    )
    merge_ok = all(
        int(row["invariants"]["guard_merge_violation_count"]) == 0
        for row in rows
    )
    production_ok = all(
        row["invariants"]["production_path_exact_match"] is not False
        for row in rows
    )
    visible_recoverability_ok = all(
        float(row["guarded"]["visible_gt_recoverable_fraction"]) >= 1.0
        for row in rows
    )

    max_fragmentation = max(
        (
            int(
                row["guarded"]["fragmentation"][
                    "max_meaningful_sv_per_gt"
                ]
            )
            for row in rows
        ),
        default=0,
    )
    overfragmented_gt_total = sum(
        int(
            row["guarded"]["fragmentation"][
                "overfragmented_gt_count"
            ]
        )
        for row in rows
    )

    strict_pass = bool(
        post_cross == 0
        and not failed_indices
        and foreground_ok
        and merge_ok
        and production_ok
        and visible_recoverability_ok
    )

    return {
        "stage": "05_supervoxel_safety_guard_validation",
        "candidate_count": len(rows),
        "strict_pass": strict_pass,
        "acceptance": {
            "all_guarded_cross_gt_atomic_supervoxels_zero": post_cross == 0,
            "foreground_support_preserved_all_candidates": foreground_ok,
            "guard_never_merged_preliminary_supervoxels": merge_ok,
            "all_visible_gt_recoverable": visible_recoverability_ok,
            "explicit_path_matches_production": production_ok,
        },
        "preliminary_cross_gt_supervoxel_total": int(pre_cross),
        "guarded_cross_gt_supervoxel_total": int(post_cross),
        "repaired_cross_gt_preliminary_supervoxel_total": int(repaired),
        "added_supervoxel_total": int(added_sv),
        "guarded_unsafe_candidate_indices": unsafe_indices,
        "strict_failed_candidate_indices": failed_indices,
        "previously_safe_regression_failure_indices": regression_failures,
        "previous_calibration_residual_repair_failure_indices": (
            residual_repair_failures
        ),
        "fragmentation": {
            "soft_overfragmented_threshold_sv_per_gt": int(
                overfragmented_threshold
            ),
            "maximum_meaningful_sv_per_gt": int(max_fragmentation),
            "overfragmented_visible_gt_total": int(
                overfragmented_gt_total
            ),
        },
        "partition_config": {
            key: value
            for key, value in vars(cfg.partition).items()
            if isinstance(value, (bool, int, float, str))
        },
        "candidates": rows,
    }


def print_conclusion(report: dict[str, Any]) -> None:
    print("\n" + "=" * 112)
    print("STAGE-05 CONCLUSION")
    print("=" * 112)
    print(
        "Cross-GT atomic SVs : "
        f"{report['preliminary_cross_gt_supervoxel_total']} preliminary "
        f"-> {report['guarded_cross_gt_supervoxel_total']} guarded"
    )
    print(
        "Cross-GT prelim SVs repaired by guard : "
        f"{report['repaired_cross_gt_preliminary_supervoxel_total']}"
    )
    print(
        "Extra guarded supervoxels             : "
        f"{report['added_supervoxel_total']}"
    )
    print(
        "Max meaningful SV / visible GT         : "
        f"{report['fragmentation']['maximum_meaningful_sv_per_gt']}"
    )
    print(
        "Guarded unsafe candidates              : "
        f"{report['guarded_unsafe_candidate_indices']}"
    )
    print(
        "Previously-safe regression failures    : "
        f"{report['previously_safe_regression_failure_indices']}"
    )
    print(
        "Residual-set repair failures           : "
        f"{report['previous_calibration_residual_repair_failure_indices']}"
    )

    if report["strict_pass"]:
        print(
            "\nPASS: the tested oracle candidate set contains zero dangerous "
            "cross-GT atomic supervoxels after the guard, foreground support "
            "is preserved, the guard never merges preliminary regions, all "
            "visible GT cells remain recoverable, and the explicit diagnostic "
            "path matches the production watershed."
        )
        print(
            "Next: freeze the proposal policy, rebuild Stage-04 candidate "
            "artifacts under this policy, then continue to exact RAG "
            "construction diagnostics."
        )
    else:
        print(
            "\nNOT PASS: do not freeze the proposal stage yet. Inspect the "
            "candidate JSON/PT artifacts for the remaining failure mode(s) "
            "and tune the guard from evidence rather than weakening the "
            "cross-GT safety criterion."
        )
    print("=" * 112)


# =============================================================================
# CACHING / VISUALIZATION
# =============================================================================


def save_candidate(
    result_dir: Path,
    summary: dict[str, Any],
    artifact: dict[str, Any],
) -> None:
    index = int(summary["candidate_index"])
    atomic_json(candidate_json_path(result_dir, index), summary)
    atomic_torch_save(candidate_pt_path(result_dir, index), artifact)


def visualize_candidate_artifact(path: Path) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError("Napari is required for --visualize.") from exc

    if not path.exists():
        raise FileNotFoundError(path)

    data = torch_load(path)
    summary = data["summary"]
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
        data["gt_labels"].numpy(),
        name="GT labels",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        data["preliminary_supervoxels"].numpy(),
        name="Preliminary supervoxels",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        data["guarded_supervoxels"].numpy(),
        name="Guarded supervoxels",
        scale=scale,
        visible=True,
    )
    viewer.add_labels(
        data["safety_barrier"].numpy().astype(np.uint8),
        name="Safety barrier",
        scale=scale,
        visible=False,
    )
    viewer.add_image(
        data["separator_probability"].float().numpy(),
        name="Separator probability",
        scale=scale,
        colormap="viridis",
        contrast_limits=(0.0, 1.0),
        visible=False,
    )

    print("\nStage-05 candidate visualization")
    print(f"  candidate          : {summary['candidate_index']}")
    print(
        "  cross-GT SV        : "
        f"{summary['preliminary']['cross_gt_unsafe_supervoxel_count']} "
        f"-> {summary['guarded']['cross_gt_unsafe_supervoxel_count']}"
    )
    print(
        "  supervoxels        : "
        f"{summary['preliminary']['supervoxel_count']} "
        f"-> {summary['guarded']['supervoxel_count']}"
    )
    print(
        "  guard split/added  : "
        f"{summary['guard']['split_preliminary_supervoxel_count']} / "
        f"{summary['guard']['added_supervoxel_count']}"
    )
    print(f"  strict pass        : {summary['strict_pass']}")
    print("  Napari loads exactly 6 layers.")
    napari.run()


def build_or_load_one_candidate(
    args: argparse.Namespace,
    base_scene,
    candidate_by_index,
    cfg,
    candidate_index: int,
) -> tuple[dict[str, Any], Path]:
    if candidate_index not in candidate_by_index:
        raise IndexError(
            f"candidate-index {candidate_index} unavailable; "
            f"available={sorted(candidate_by_index)}"
        )

    pt_path = candidate_pt_path(args.result_dir, candidate_index)
    json_path = candidate_json_path(args.result_dir, candidate_index)

    if pt_path.exists() and json_path.exists() and not args.rebuild_candidate:
        _progress(
            f"[candidate {candidate_index}] Using cached Stage-05 artifact: "
            f"{pt_path}"
        )
        data = torch_load(pt_path)
        return data["summary"], pt_path

    crop, selection = candidate_by_index[candidate_index]
    scene = make_scene(
        base_scene,
        crop,
        selection,
        args.context_margin_dref,
    )
    summary, artifact = analyze_candidate(
        scene,
        cfg,
        overfragmented_threshold=args.overfragmented_threshold,
        verify_production_path=not args.skip_production_verify,
        progress_prefix=f"[candidate {candidate_index}]",
    )
    save_candidate(args.result_dir, summary, artifact)
    return summary, pt_path


# =============================================================================
# CLI / MAIN
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate STIR-Net post-watershed Supervoxel Safety Guard "
            "against the saved Stage-00 oracle candidate set."
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
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
    )
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=None,
        help=(
            "Analyze only one Stage-00 candidate. Useful with --visualize."
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open the requested candidate in a minimal six-layer Napari view.",
    )
    parser.add_argument(
        "--rebuild-candidate",
        action="store_true",
        help="Ignore the Stage-05 per-candidate cache and recompute.",
    )
    parser.add_argument(
        "--skip-production-verify",
        action="store_true",
        help=(
            "Skip the second production LearnedGeometryWatershed call. "
            "Normally keep verification enabled."
        ),
    )
    parser.add_argument(
        "--overfragmented-threshold",
        type=int,
        default=DEFAULT_OVERFRAGMENTED_SV_PER_GT,
        help=(
            "Soft diagnostic threshold for meaningful SVs per visible GT cell."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.stage00_sample = args.stage00_sample.resolve()
    args.result_dir = args.result_dir.resolve()

    if args.max_candidates < 1:
        raise ValueError("--max-candidates must be >= 1")
    if args.overfragmented_threshold < 1:
        raise ValueError("--overfragmented-threshold must be >= 1")

    print("\n" + "=" * 112)
    print("STIR-Net Stage 05 — Supervoxel Safety Guard validation")
    print("=" * 112)
    print(f"Data dir                  : {args.data_dir}")
    print(f"Time index                : {args.time_index}")
    print(f"Candidate source          : {args.stage00_sample}")
    print(f"Candidates                : up to {args.max_candidates}")
    print("Dense geometry            : ORACLE production targets")
    print("Training                   : NONE")
    print("Preliminary proposal       : Stage-03 exact marker watershed + cleanup")
    print("Safety guard               : production supervoxel_guard.py")
    print(
        "Production path verify    : "
        f"{'OFF' if args.skip_production_verify else 'ON'}"
    )
    print(
        "Primary acceptance        : zero guarded cross-GT atomic supervoxels"
    )
    print("=" * 112, flush=True)

    start = time.perf_counter()
    base_scene, candidates, candidate_by_index = load_scene_index(args)
    cfg = stage03.stage01.build_debug_config()

    if not bool(getattr(cfg.partition, "supervoxel_guard_enabled", False)):
        raise RuntimeError(
            "Current PartitionConfig has supervoxel_guard_enabled=False. "
            "Apply/enable the safety-guard patch before running Stage 05."
        )

    if args.candidate_index is not None:
        summary, path = build_or_load_one_candidate(
            args,
            base_scene,
            candidate_by_index,
            cfg,
            int(args.candidate_index),
        )
        print_candidate_table([summary])
        one_report = aggregate_report(
            [summary],
            cfg,
            overfragmented_threshold=args.overfragmented_threshold,
        )
        print_conclusion(one_report)
        if args.visualize:
            visualize_candidate_artifact(path)
        return

    if args.visualize:
        raise ValueError(
            "--visualize requires --candidate-index N so the visual target "
            "is unambiguous."
        )

    rows: list[dict[str, Any]] = []
    candidate_times: list[float] = []

    ordered_candidates = sorted(candidate_by_index.items())
    total = len(ordered_candidates)

    for ordinal, (candidate_index, (crop, selection)) in enumerate(
        ordered_candidates, 1
    ):
        prefix = f"[{ordinal:02d}/{total:02d}]"
        if candidate_times:
            mean_seconds = float(np.mean(candidate_times))
            eta = mean_seconds * (total - ordinal + 1)
            eta_text = _format_duration(eta)
        else:
            eta_text = "estimating"

        _progress("")
        _progress(
            f"{prefix} START candidate {candidate_index} | "
            f"previous={'residual' if candidate_index in PREVIOUS_CALIBRATION_RESIDUAL_CANDIDATES else ('unsafe' if candidate_index in PREVIOUSLY_UNSAFE_CANDIDATES else 'safe')} | "
            f"shape={_slice_shape(crop)} | ETA {eta_text}"
        )

        candidate_start = time.perf_counter()
        # Full scans intentionally recompute unless the user explicitly wants
        # per-candidate cache reuse via --candidate-index. This prevents a stale
        # validation report after changing guard thresholds/configuration.
        scene = make_scene(
            base_scene,
            crop,
            selection,
            args.context_margin_dref,
        )
        summary, artifact = analyze_candidate(
            scene,
            cfg,
            overfragmented_threshold=args.overfragmented_threshold,
            verify_production_path=not args.skip_production_verify,
            progress_prefix=prefix,
        )
        save_candidate(args.result_dir, summary, artifact)
        rows.append(summary)

        elapsed = time.perf_counter() - candidate_start
        candidate_times.append(elapsed)
        mean_seconds = float(np.mean(candidate_times))
        remaining = total - ordinal
        eta = mean_seconds * remaining

        _progress(
            f"{prefix} DONE {_format_duration(elapsed)} | "
            f"crossGT "
            f"{summary['preliminary']['cross_gt_unsafe_supervoxel_count']} "
            f"-> {summary['guarded']['cross_gt_unsafe_supervoxel_count']} | "
            f"SV {summary['preliminary']['supervoxel_count']} "
            f"-> {summary['guarded']['supervoxel_count']} | "
            f"PASS={summary['strict_pass']} | "
            f"remaining {_format_duration(eta)}"
        )

    print_candidate_table(rows)

    report = aggregate_report(
        rows,
        cfg,
        overfragmented_threshold=args.overfragmented_threshold,
    )
    report["elapsed_seconds"] = float(time.perf_counter() - start)

    report_path = args.result_dir / "validation_report.json"
    atomic_json(report_path, report)

    print_conclusion(report)
    print(f"\nSaved report: {report_path}")
    print(
        f"Elapsed: {_format_duration(report['elapsed_seconds'])}"
    )


if __name__ == "__main__":
    main()
