from __future__ import annotations

"""Stage 05: validate face-level STIR-Net supervoxel safety.

Oracle dense geometry is used so this stage isolates deterministic proposal
machinery from dense-network generalization.

The legacy cross-GT supervoxel count remains visible, but it is warning-only.
PASS/FAIL is based on atomic recoverability: if every supervoxel must eventually
belong to one final cell, what fraction of each complete GT cell can still be
represented by assigning each SV to its dominant GT identity?

Run:
    python investigations/stirnet/06_supervoxel_safety_guard_validation.py

Visualize one candidate:
    python investigations/stirnet/06_supervoxel_safety_guard_validation.py \
        --candidate-index 7 --visualize
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

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


stage00 = _import_adjacent("_stirnet_s00_face05", "00_GT_validation.py")
stage03 = _import_adjacent("_stirnet_s03_face05", "03_watershed_supervoxel_debug.py")
stage04 = _import_adjacent("_stirnet_s04_face05", "04_rag_case_selection.py")

from learned.stirnet.model.geometry.derived import build_geometry_derived_cache
from learned.stirnet.model.partition.supervoxel_guard import (
    build_supervoxel_face_cuts,
    face_cuts_to_voxel_proxy,
    split_preliminary_supervoxels,
)
from learned.stirnet.model.partition.watershed import LearnedGeometryWatershed

DEFAULT_DATA_DIR = stage00.DEFAULT_DATA_DIR
DEFAULT_TIME_INDEX = stage00.DEFAULT_TIME_INDEX
DEFAULT_MAX_CANDIDATES = stage00.DEFAULT_MAX_CANDIDATES
DEFAULT_CONTEXT_MARGIN_DREF = stage00.DEFAULT_CONTEXT_MARGIN_DREF
DEFAULT_STAGE00_SAMPLE = REPOSITORY_ROOT / "data/learned/stirnet/debug_crop.pt"
DEFAULT_RESULT_DIR = REPOSITORY_ROOT / "data/learned/stirnet/supervoxel_safety_guard_validation"
DEFAULT_MIN_ATOMIC_RECOVERABLE_FRACTION = 0.95
DEFAULT_OVERFRAGMENTED_SV_PER_GT = 8


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


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _shape(slices):
    return tuple(int(v.stop) - int(v.start) for v in slices)


def _candidate_pt(result_dir: Path, index: int) -> Path:
    return result_dir / "candidates" / f"candidate_{index:02d}.pt"


def _candidate_json(result_dir: Path, index: int) -> Path:
    return result_dir / "candidates" / f"candidate_{index:02d}.json"


def _stage04_args(args):
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


def load_candidates(args):
    base_scene, candidates = stage04.load_full_scene_and_candidates(_stage04_args(args))
    indexed = {}
    for ordinal, (crop, selection) in enumerate(candidates):
        selection = dict(selection)
        index = int(selection.get("candidate_index", ordinal))
        selection["candidate_index"] = index
        indexed[index] = (crop, selection)
    return base_scene, indexed


def make_scene(base_scene, crop, selection, context_margin_dref):
    return stage04.make_candidate_scene(base_scene, crop, selection, context_margin_dref)


def build_oracle(scene):
    targets = stage00.build_targets(scene)
    target_batch = stage04.target_batch_for_stage03(scene, targets)
    batch = stage04.batch_namespace_for_stage03(scene, target_batch)
    geometry = stage03.oracle_geometry_from_targets(target_batch)
    complete_ids, partial_ids = stage00._complete_and_partial_ids(
        scene.gt_full, scene.core_slices
    )
    return batch, geometry, complete_ids, partial_ids


def _descendant_mapping(preliminary, guarded):
    mapping = {}
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


def _fragmentation(summary, threshold):
    counts = [int(row["meaningful_supervoxel_count"]) for row in summary["per_gt"].values()]
    return {
        "mean_meaningful_sv_per_gt": float(np.mean(counts)) if counts else 0.0,
        "max_meaningful_sv_per_gt": max(counts, default=0),
        "overfragmented_gt_ids": [
            int(gt_id)
            for gt_id, row in summary["per_gt"].items()
            if int(row["meaningful_supervoxel_count"]) > threshold
        ],
    }


def _internal_face_count(labels, faces):
    total = 0
    for axis, field in enumerate(faces):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        a = labels[tuple(lower)]
        b = labels[tuple(upper)]
        total += int(np.sum(field & (a > 0) & (a == b)))
    return total


def _cue_counts(labels, cuts, evidence):
    result = {"cut_faces_inside_preliminary_sv": _internal_face_count(labels, cuts)}
    for key in ("strong_separator", "separator_corroborated", "geometry_only"):
        result[f"{key}_faces_inside_preliminary_sv"] = _internal_face_count(
            labels, evidence[key]
        )
    return result


@torch.no_grad()
def analyze_candidate(
    scene,
    cfg,
    *,
    minimum_atomic_recoverable_fraction: float,
    overfragmented_threshold: int,
    verify_production_path: bool,
    prefix: str,
):
    start = time.perf_counter()
    print(f"{prefix} [1/6] oracle targets + geometry ...", flush=True)
    batch, geometry, complete_ids, partial_ids = build_oracle(scene)

    print(f"{prefix} [2/6] derived geometry + markers ...", flush=True)
    cache = build_geometry_derived_cache(geometry, cfg.partition, padding_mask=None)
    foreground = cache.foreground_mask[0]
    markers = stage03.build_exact_markers(
        cache.seed_score[0, 0], foreground, batch.spacing_um[0], batch.dref_um[0], cfg
    )

    print(f"{prefix} [3/6] preliminary watershed ...", flush=True)
    raw, preliminary = stage03.run_exact_watershed(
        cache.watershed_energy[0, 0], markers, foreground, cfg
    )
    gt_np = batch.gt_labels.cpu().numpy().astype(np.int32, copy=False)
    marker_summary = stage03.marker_diagnostics(markers, gt_np, complete_ids)
    preliminary_summary, preliminary_maps = stage03.supervoxel_diagnostics(
        preliminary, gt_np, complete_ids, cfg
    )

    separator = cache.separator_prob[0, 0].float().cpu().numpy()
    seed = cache.seed_prob[0, 0].float().cpu().numpy()
    sdf_normalized = cache.sdf_normalized[0, 0].float().cpu().numpy()
    flow = geometry.flow[0].float().cpu().numpy()
    centroid = geometry.centroid_offset[0].float().cpu().numpy()
    spacing = batch.spacing_um[0].float().cpu().numpy()
    dref = float(batch.dref_um[0].item())
    sigma = float(cfg.geometry.separator_target_sigma_um)

    print(f"{prefix} [4/6] face-level safety guard ...", flush=True)
    cuts, evidence = build_supervoxel_face_cuts(
        separator, centroid, flow, seed, sdf_normalized, spacing, dref, cfg.partition,
        separator_sigma_um=sigma,
    )
    guarded, guard_diag = split_preliminary_supervoxels(
        preliminary, separator, centroid, flow, seed, sdf_normalized, spacing,
        dref, cfg.partition, separator_sigma_um=sigma,
    )
    guarded_summary, guarded_maps = stage03.supervoxel_diagnostics(
        guarded, gt_np, complete_ids, cfg
    )

    print(f"{prefix} [5/6] invariants + severity ...", flush=True)
    foreground_preserved = bool(np.array_equal(preliminary > 0, guarded > 0))
    descendants, merge_violations = _descendant_mapping(preliminary, guarded)

    production_exact_match = None
    print(
        f"{prefix} [6/6] " + (
            "production-path equality ..." if verify_production_path
            else "production equality skipped ..."
        ),
        flush=True,
    )
    if verify_production_path:
        production = LearnedGeometryWatershed(cfg.partition, cfg.geometry)
        production_labels = production(
            geometry, batch.spacing_um, batch.dref_um,
            padding_mask=None, derived_cache=cache,
        )[0].cpu().numpy().astype(np.int32, copy=False)
        production_exact_match = bool(np.array_equal(production_labels, guarded))
        if not production_exact_match:
            raise RuntimeError(f"{prefix} explicit face guard != production watershed")

    marker_ok = marker_summary["complete_gt_marker_recall"] >= 1.0
    coverage_ok = guarded_summary["complete_gt_min_coverage"] >= stage03.MIN_COMPLETE_GT_COVERAGE
    atomic_min = float(guarded_summary["atomic_min_complete_recoverable_fraction"])
    atomic_ok = atomic_min >= minimum_atomic_recoverable_fraction
    strict_pass = bool(
        marker_ok and coverage_ok and atomic_ok and foreground_preserved
        and merge_violations == 0 and production_exact_match is not False
    )

    summary = {
        "candidate_index": int(scene.selection.get("candidate_index", -1)),
        "strict_pass": strict_pass,
        "crop_shape_zyx": list(_shape(scene.core_slices)),
        "merge_gt_ids": [int(v) for v in scene.selection["merge_gt_ids"]],
        "complete_gt_ids": [int(v) for v in complete_ids],
        "partial_gt_ids": [int(v) for v in partial_ids],
        "marker": {
            "count": int(marker_summary["marker_count"]),
            "complete_gt_recall": float(marker_summary["complete_gt_marker_recall"]),
        },
        "preliminary": {
            "raw_watershed_count": int(raw.max()),
            "supervoxel_count": int(preliminary.max()),
            "legacy_cross_gt_count": int(preliminary_summary["cross_gt_unsafe_supervoxel_count"]),
            "atomic_min_complete_recoverable_fraction": float(
                preliminary_summary["atomic_min_complete_recoverable_fraction"]
            ),
            "atomic_global_recoverable_fraction": float(
                preliminary_summary["atomic_global_recoverable_fraction"]
            ),
            "atomic_total_unrecoverable_gt_voxels": int(
                preliminary_summary["atomic_total_unrecoverable_gt_voxels"]
            ),
            "fragmentation": _fragmentation(preliminary_summary, overfragmented_threshold),
        },
        "guard": {
            "cut_face_count": int(guard_diag.cut_face_count),
            "split_preliminary_supervoxel_count": int(guard_diag.split_supervoxel_count),
            "added_supervoxel_count": int(guard_diag.added_supervoxel_count),
            "suppressed_pathological_split_count": int(
                guard_diag.suppressed_pathological_split_count
            ),
            "cue_counts": _cue_counts(preliminary, cuts, evidence),
            "descendants": {str(k): v for k, v in descendants.items() if len(v) > 1},
        },
        "guarded": {
            "supervoxel_count": int(guarded.max()),
            "legacy_cross_gt_count": int(guarded_summary["cross_gt_unsafe_supervoxel_count"]),
            "atomic_min_complete_recoverable_fraction": atomic_min,
            "atomic_mean_complete_recoverable_fraction": float(
                guarded_summary["atomic_mean_complete_recoverable_fraction"]
            ),
            "atomic_global_recoverable_fraction": float(
                guarded_summary["atomic_global_recoverable_fraction"]
            ),
            "atomic_total_unrecoverable_gt_voxels": int(
                guarded_summary["atomic_total_unrecoverable_gt_voxels"]
            ),
            "complete_gt_min_coverage": float(guarded_summary["complete_gt_min_coverage"]),
            "fragmentation": _fragmentation(guarded_summary, overfragmented_threshold),
            "per_gt": guarded_summary["per_gt"],
        },
        "acceptance": {
            "minimum_atomic_recoverable_fraction": float(minimum_atomic_recoverable_fraction),
            "marker_ok": bool(marker_ok),
            "coverage_ok": bool(coverage_ok),
            "atomic_ok": bool(atomic_ok),
            "foreground_preserved": foreground_preserved,
            "guard_merge_violation_count": int(merge_violations),
            "production_exact_match": production_exact_match,
        },
        "elapsed_seconds": float(time.perf_counter() - start),
    }

    artifact = {
        "format_version": 2,
        "kind": "stirnet_face_supervoxel_guard_validation",
        "summary": summary,
        "spacing_um": batch.spacing_um[0].cpu().float(),
        "raw": torch.from_numpy(
            np.asarray(scene.raw_full[scene.core_slices]).astype(np.float32, copy=True)
        ),
        "gt_labels": batch.gt_labels.cpu().long(),
        "preliminary_supervoxels": torch.from_numpy(preliminary.astype(np.int32, copy=False)),
        "guarded_supervoxels": torch.from_numpy(guarded.astype(np.int32, copy=False)),
        "cut_face_proxy": torch.from_numpy(face_cuts_to_voxel_proxy(cuts, preliminary.shape)),
        "separator_probability": cache.separator_prob[0, 0].cpu().half(),
        "preliminary_atomic_assignment_error": torch.from_numpy(
            preliminary_maps["atomic_assignment_error"].astype(np.uint8, copy=False)
        ),
        "guarded_atomic_assignment_error": torch.from_numpy(
            guarded_maps["atomic_assignment_error"].astype(np.uint8, copy=False)
        ),
    }
    return summary, artifact


def save_candidate(result_dir, summary, artifact):
    index = int(summary["candidate_index"])
    atomic_json(_candidate_json(result_dir, index), summary)
    atomic_torch_save(_candidate_pt(result_dir, index), artifact)


def print_table(rows):
    print("\n" + "=" * 142)
    print("Stage-05 face-level supervoxel safety validation")
    print("=" * 142)
    print(
        f"{'cand':>4s} {'preSV':>6s} {'postSV':>7s} {'preX':>5s} {'postX':>6s} "
        f"{'cuts':>6s} {'splits':>6s} {'added':>6s} {'preAtom':>8s} "
        f"{'postAtom':>9s} {'maxSV/GT':>8s} {'PASS':>6s}"
    )
    print("-" * 142)
    for row in sorted(rows, key=lambda x: x["candidate_index"]):
        print(
            f"{row['candidate_index']:4d} "
            f"{row['preliminary']['supervoxel_count']:6d} "
            f"{row['guarded']['supervoxel_count']:7d} "
            f"{row['preliminary']['legacy_cross_gt_count']:5d} "
            f"{row['guarded']['legacy_cross_gt_count']:6d} "
            f"{row['guard']['cut_face_count']:6d} "
            f"{row['guard']['split_preliminary_supervoxel_count']:6d} "
            f"{row['guard']['added_supervoxel_count']:6d} "
            f"{row['preliminary']['atomic_min_complete_recoverable_fraction']:8.3f} "
            f"{row['guarded']['atomic_min_complete_recoverable_fraction']:9.3f} "
            f"{row['guarded']['fragmentation']['max_meaningful_sv_per_gt']:8d} "
            f"{'YES' if row['strict_pass'] else 'NO':>6s}"
        )
    print("=" * 142)


def aggregate(rows, threshold):
    failed = [int(row["candidate_index"]) for row in rows if not row["strict_pass"]]
    return {
        "stage": "05_supervoxel_safety_guard_validation",
        "metric_policy": {
            "legacy_cross_gt_is_warning_only": True,
            "minimum_complete_gt_atomic_recoverable_fraction": float(threshold),
        },
        "candidate_count": len(rows),
        "strict_pass": not failed,
        "failed_candidate_indices": failed,
        "legacy_cross_gt_total_preliminary": int(sum(
            row["preliminary"]["legacy_cross_gt_count"] for row in rows
        )),
        "legacy_cross_gt_total_guarded": int(sum(
            row["guarded"]["legacy_cross_gt_count"] for row in rows
        )),
        "minimum_guarded_complete_gt_atomic_recoverable_fraction": float(min(
            (row["guarded"]["atomic_min_complete_recoverable_fraction"] for row in rows),
            default=1.0,
        )),
        "total_added_supervoxels": int(sum(
            row["guard"]["added_supervoxel_count"] for row in rows
        )),
        "candidates": rows,
    }


def visualize(path: Path):
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError("Napari is required for --visualize.") from exc

    data = torch_load(path)
    scale = tuple(float(v) for v in data["spacing_um"].tolist())
    summary = data["summary"]
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(data["raw"].numpy(), name="Raw", scale=scale, colormap="gray", visible=True)
    viewer.add_labels(data["gt_labels"].numpy(), name="GT labels", scale=scale, visible=False)
    viewer.add_labels(
        data["preliminary_supervoxels"].numpy(), name="Preliminary supervoxels",
        scale=scale, visible=False,
    )
    viewer.add_labels(
        data["guarded_supervoxels"].numpy(), name="Guarded supervoxels",
        scale=scale, visible=True,
    )
    viewer.add_labels(
        data["cut_face_proxy"].numpy().astype(np.uint8),
        name="Cut faces (lower-voxel proxy)", scale=scale, visible=False,
    )
    viewer.add_labels(
        data["guarded_atomic_assignment_error"].numpy().astype(np.uint8),
        name="Atomic assignment error (1=wrong owner, 2=missing FG)",
        scale=scale, visible=False,
    )
    viewer.add_image(
        data["separator_probability"].float().numpy(), name="Separator probability",
        scale=scale, colormap="viridis", contrast_limits=(0.0, 1.0), visible=False,
    )
    print(f"\nCandidate {summary['candidate_index']}")
    print(
        "  legacy cross-GT: "
        f"{summary['preliminary']['legacy_cross_gt_count']} -> "
        f"{summary['guarded']['legacy_cross_gt_count']}"
    )
    print(
        "  atomic minimum : "
        f"{summary['preliminary']['atomic_min_complete_recoverable_fraction']:.3f} -> "
        f"{summary['guarded']['atomic_min_complete_recoverable_fraction']:.3f}"
    )
    print(f"  strict pass    : {summary['strict_pass']}")
    print("  Napari loads exactly 7 layers.")
    napari.run()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--time-index", type=int, default=DEFAULT_TIME_INDEX)
    p.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    p.add_argument("--context-margin-dref", type=float, default=DEFAULT_CONTEXT_MARGIN_DREF)
    p.add_argument("--stage00-sample", type=Path, default=DEFAULT_STAGE00_SAMPLE)
    p.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    p.add_argument("--candidate-index", type=int, default=None)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--rebuild-candidate", action="store_true")
    p.add_argument("--skip-production-verify", action="store_true")
    p.add_argument(
        "--min-atomic-recoverable-fraction", type=float,
        default=DEFAULT_MIN_ATOMIC_RECOVERABLE_FRACTION,
    )
    p.add_argument(
        "--overfragmented-threshold", type=int,
        default=DEFAULT_OVERFRAGMENTED_SV_PER_GT,
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.stage00_sample = args.stage00_sample.resolve()
    args.result_dir = args.result_dir.resolve()
    if not 0.0 < args.min_atomic_recoverable_fraction <= 1.0:
        raise ValueError("--min-atomic-recoverable-fraction must be in (0,1]")

    print("\n" + "=" * 112)
    print("STIR-Net Stage 05 — face-level Supervoxel Safety Guard")
    print("=" * 112)
    print(f"Candidate source      : {args.stage00_sample}")
    print("Dense geometry        : ORACLE production targets")
    print("Training              : NONE")
    print("Separator topology    : normalized + ridge-thinned voxel faces")
    print(
        "Acceptance            : min complete-cell atomic recoverability "
        f">= {args.min_atomic_recoverable_fraction:.3f}"
    )
    print("Legacy cross-GT count : warning only")
    print("=" * 112, flush=True)

    base_scene, indexed = load_candidates(args)
    cfg = stage03.stage01.build_debug_config()

    if args.candidate_index is not None:
        index = int(args.candidate_index)
        if index not in indexed:
            raise IndexError(f"candidate {index} unavailable; available={sorted(indexed)}")
        path = _candidate_pt(args.result_dir, index)
        if path.exists() and not args.rebuild_candidate:
            data = torch_load(path)
            summary = data["summary"]
        else:
            crop, selection = indexed[index]
            scene = make_scene(base_scene, crop, selection, args.context_margin_dref)
            summary, artifact = analyze_candidate(
                scene, cfg,
                minimum_atomic_recoverable_fraction=args.min_atomic_recoverable_fraction,
                overfragmented_threshold=args.overfragmented_threshold,
                verify_production_path=not args.skip_production_verify,
                prefix=f"[candidate {index}]",
            )
            save_candidate(args.result_dir, summary, artifact)
        print_table([summary])
        if args.visualize:
            visualize(path)
        return

    if args.visualize:
        raise ValueError("--visualize requires --candidate-index N")

    rows = []
    timings = []
    start = time.perf_counter()
    ordered = sorted(indexed.items())
    total = len(ordered)
    for ordinal, (index, (crop, selection)) in enumerate(ordered, 1):
        eta = float(np.mean(timings)) * (total - ordinal + 1) if timings else 0.0
        eta_text = _duration(eta) if timings else "estimating"
        print(
            f"\n[{ordinal:02d}/{total:02d}] START candidate {index} "
            f"shape={_shape(crop)} ETA {eta_text}", flush=True,
        )
        t0 = time.perf_counter()
        scene = make_scene(base_scene, crop, selection, args.context_margin_dref)
        summary, artifact = analyze_candidate(
            scene, cfg,
            minimum_atomic_recoverable_fraction=args.min_atomic_recoverable_fraction,
            overfragmented_threshold=args.overfragmented_threshold,
            verify_production_path=not args.skip_production_verify,
            prefix=f"[{ordinal:02d}/{total:02d}]",
        )
        save_candidate(args.result_dir, summary, artifact)
        rows.append(summary)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        print(
            f"[{ordinal:02d}/{total:02d}] DONE {_duration(elapsed)} | "
            f"legacyX {summary['preliminary']['legacy_cross_gt_count']} -> "
            f"{summary['guarded']['legacy_cross_gt_count']} | "
            f"atomic {summary['preliminary']['atomic_min_complete_recoverable_fraction']:.3f} -> "
            f"{summary['guarded']['atomic_min_complete_recoverable_fraction']:.3f} | "
            f"PASS={summary['strict_pass']}", flush=True,
        )

    print_table(rows)
    report = aggregate(rows, args.min_atomic_recoverable_fraction)
    report["elapsed_seconds"] = float(time.perf_counter() - start)
    report_path = args.result_dir / "validation_report.json"
    atomic_json(report_path, report)

    print("\n" + "=" * 112)
    print("STAGE-05 CONCLUSION")
    print("=" * 112)
    print(
        "Legacy cross-GT SVs : "
        f"{report['legacy_cross_gt_total_preliminary']} -> "
        f"{report['legacy_cross_gt_total_guarded']} (warning metric)"
    )
    print(
        "Worst complete-cell atomic recoverability : "
        f"{report['minimum_guarded_complete_gt_atomic_recoverable_fraction']:.3f}"
    )
    print(f"Added supervoxels                       : {report['total_added_supervoxels']}")
    print(f"Failed candidates                       : {report['failed_candidate_indices']}")
    print("\nPASS" if report["strict_pass"] else "\nNOT PASS")
    print("=" * 112)
    print(f"Saved report: {report_path}")
    print(f"Elapsed: {_duration(report['elapsed_seconds'])}")


if __name__ == "__main__":
    main()
