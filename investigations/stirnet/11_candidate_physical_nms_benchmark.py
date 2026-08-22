from __future__ import annotations

"""Stage 11: candidate-based exact physical-distance marker NMS benchmark.

Purpose
-------
This stage isolates ONLY watershed marker generation. It does not train STIR-Net
and it does not rerun the dense CNN. It reuses the fixed Stage-03 diagnostic
artifact so every marker method receives exactly the same seed-score,
foreground, watershed-energy, spacing, and GT context.

The experiment tests the proposed replacement:

    seed score volume
        -> cheap 3x3x3 local maxima on GPU
        -> threshold + foreground mask
        -> candidate peak coordinates
        -> exact Euclidean physical-distance NMS on candidates
        -> disconnected-foreground seed repair
        -> final markers

Against:

    Experiment-31 fast rectangular physical max-pool marker backend

If the Stage-03 artifact was generated with the reference ellipsoidal backend,
its already-saved marker map is also reported as a zero-cost reference baseline;
Stage 11 never reruns the expensive full-volume SciPy ellipsoidal maximum filter.

Two tests are performed:

1. REAL DEBUG CROP — correctness / proposal quality
   - predicted seed score
   - oracle-marker-score variant when available
   - marker recall on complete GT cells
   - downstream watershed proposal safety/recoverability
   - marker-set physical agreement
   - detailed NMS timing and CUDA memory

2. OPTIONAL LARGE STRESS TEST — timing only
   The real Stage-03 score/foreground tensors are tiled to a user-selectable
   volume shape (default: the Stage-10 useful-ROI shape 58x651x692). This tests
   scaling on a local GPU without rerunning the CNN or GT preprocessing.

Typical use
-----------
    python investigations/stirnet/11_candidate_physical_nms_benchmark.py

Skip the large local stress test:
    python investigations/stirnet/11_candidate_physical_nms_benchmark.py --skip-stress

Change repeats:
    python investigations/stirnet/11_candidate_physical_nms_benchmark.py --repeats 10 --stress-repeats 3

Notes
-----
- The experimental candidate NMS lives only in this investigation file. It does
  NOT modify production seeds.py yet.
- NMS uses a cKDTree over physical (micrometre) candidate coordinates. The
  suppression radius is therefore exactly Euclidean in physical space rather
  than an axis-aligned voxel rectangle.
- Ties are deterministic: higher score wins; equal score uses ascending flat
  voxel index.
"""

import argparse
import copy
import gc
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from torch import Tensor


# =============================================================================
# REPOSITORY / ADJACENT STAGE IMPORT
# =============================================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

STAGE03_PATH = Path(__file__).with_name("03_watershed_supervoxel_debug.py")
if not STAGE03_PATH.exists():
    raise FileNotFoundError(f"Missing required adjacent investigation: {STAGE03_PATH}")

_spec = importlib.util.spec_from_file_location(
    "_stirnet_stage03_watershed_supervoxel_debug", STAGE03_PATH
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not import {STAGE03_PATH}")
stage03 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = stage03
_spec.loader.exec_module(stage03)

from learned.stirnet.model.partition.seeds import build_markers_fast


DEFAULT_STAGE03_ARTIFACT = (
    stage03.DEFAULT_RESULT_DIR / "watershed_supervoxel_debug.pt"
)
DEFAULT_STAGE03_SUMMARY = (
    stage03.DEFAULT_RESULT_DIR / "watershed_supervoxel_debug.json"
)
DEFAULT_SAMPLE_PATH = stage03.DEFAULT_SAMPLE_PATH
DEFAULT_RESULT_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "candidate_physical_nms"
)

DEFAULT_VARIANTS = ("predicted", "oracle_marker_score")
DEFAULT_STRESS_SHAPE = (58, 651, 692)
PHYSICAL_EPS_UM = 1e-6


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


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_allocated_mib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / 1024**2)


def format_seconds(value: float) -> str:
    if value < 0.001:
        return f"{value * 1e6:.1f} us"
    if value < 1.0:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.3f} s"


def median_or_none(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


class SyncWallProfiler:
    """Tiny profiler implementing the profile(name) API used by seeds.py."""

    def __init__(self, device: torch.device):
        self.device = device
        self.timings: dict[str, float] = {}

    @contextmanager
    def profile(self, name: str):
        sync_cuda(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            sync_cuda(self.device)
            elapsed = time.perf_counter() - start
            self.timings[name] = self.timings.get(name, 0.0) + elapsed


# =============================================================================
# EXPERIMENTAL CANDIDATE-BASED PHYSICAL NMS
# =============================================================================


@torch.no_grad()
def build_markers_candidate_physical(
    score: Tensor,
    foreground: Tensor,
    spacing_um: Tensor,
    radius_um: float,
    threshold: float,
    max_markers: int,
    *,
    collect_timings: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Candidate-based exact Euclidean physical-distance NMS.

    The expensive full-volume neighborhood is intentionally only 3x3x3.
    Physical-radius suppression happens on the much smaller candidate list.

    This is a *greedy marker NMS* rather than the current SciPy ellipsoidal
    maximum-filter implementation. It guarantees that the NMS-selected marker
    coordinates are separated by >= radius_um in exact physical Euclidean
    distance (before the existing disconnected-component seed repair).
    """
    if score.ndim != 3 or foreground.ndim != 3:
        raise ValueError("score and foreground must both be [Z,Y,X]")
    if tuple(score.shape) != tuple(foreground.shape):
        raise ValueError("score and foreground shapes must match")
    if radius_um <= 0:
        raise ValueError("radius_um must be positive")
    if max_markers < 1:
        raise ValueError("max_markers must be positive")

    device = score.device
    shape = tuple(int(v) for v in score.shape)
    spacing_np = spacing_um.detach().float().cpu().numpy().astype(np.float64)
    if spacing_np.shape != (3,) or np.any(spacing_np <= 0):
        raise ValueError(f"Invalid spacing_um={spacing_np}")

    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    # ------------------------------------------------------------------
    # A. Cheap native-grid local maxima on GPU.
    # ------------------------------------------------------------------
    sync_cuda(device)
    start = time.perf_counter()
    score32 = score.float()
    pooled = F.max_pool3d(
        score32[None, None], kernel_size=3, stride=1, padding=1
    )[0, 0]
    candidate_mask = (
        (score32 >= pooled - 1e-7)
        & foreground.bool()
        & (score32 >= float(threshold))
    )
    sync_cuda(device)
    timings["gpu_3x3_local_max_threshold"] = time.perf_counter() - start

    # ------------------------------------------------------------------
    # B. Move the cheap boolean local-max mask to CPU and collapse connected
    #    plateau voxels to one representative peak. This keeps the later
    #    physical candidate list small even for BF16/FP16 score plateaus.
    # ------------------------------------------------------------------
    sync_cuda(device)
    start = time.perf_counter()
    candidate_np = candidate_mask.detach().cpu().numpy()
    sync_cuda(device)
    timings["candidate_mask_transfer_to_cpu"] = time.perf_counter() - start

    start = time.perf_counter()
    plateau_labels, plateau_count = ndi.label(candidate_np)
    candidate_voxel_coords = np.argwhere(candidate_np).astype(np.int64, copy=False)
    timings["candidate_plateau_connected_components"] = time.perf_counter() - start

    sync_cuda(device)
    start = time.perf_counter()
    if len(candidate_voxel_coords):
        candidate_coords_gpu = torch.from_numpy(candidate_voxel_coords).to(device=device)
        candidate_values = score32[
            candidate_coords_gpu[:, 0],
            candidate_coords_gpu[:, 1],
            candidate_coords_gpu[:, 2],
        ]
        candidate_values_np = (
            candidate_values.detach().cpu().numpy().astype(np.float32, copy=False)
        )
    else:
        candidate_values_np = np.zeros((0,), dtype=np.float32)
    sync_cuda(device)
    timings["candidate_score_gather"] = time.perf_counter() - start

    start = time.perf_counter()
    if len(candidate_voxel_coords):
        component_ids = plateau_labels[tuple(candidate_voxel_coords.T)].astype(
            np.int64, copy=False
        )
        flat_voxels = np.ravel_multi_index(candidate_voxel_coords.T, shape)
        # Primary key is component id; within each plateau choose maximum score,
        # then the smallest flat index for deterministic equal-score ties.
        order = np.lexsort(
            (flat_voxels, -candidate_values_np.astype(np.float64), component_ids)
        )
        ordered_components = component_ids[order]
        first = np.ones(len(order), dtype=bool)
        if len(order) > 1:
            first[1:] = ordered_components[1:] != ordered_components[:-1]
        representative_rows = order[first]
        coords_np = candidate_voxel_coords[representative_rows]
        values_np = candidate_values_np[representative_rows]
    else:
        coords_np = np.zeros((0, 3), dtype=np.int64)
        values_np = np.zeros((0,), dtype=np.float32)
    timings["candidate_plateau_representatives"] = time.perf_counter() - start

    del pooled, candidate_mask, candidate_np

    # ------------------------------------------------------------------
    # C. Exact physical-distance greedy suppression using cKDTree.
    # ------------------------------------------------------------------
    start = time.perf_counter()
    if len(coords_np):
        physical_um = coords_np.astype(np.float64) * spacing_np[None]
        tree = cKDTree(physical_um)
    else:
        physical_um = np.zeros((0, 3), dtype=np.float64)
        tree = None
    timings["physical_kdtree_build"] = time.perf_counter() - start

    start = time.perf_counter()
    selected_rows: list[int] = []
    if len(coords_np):
        flat_indices = np.ravel_multi_index(coords_np.T, shape)
        # np.lexsort uses the last key as primary: score descending, then flat
        # index ascending for deterministic equal-score resolution.
        order = np.lexsort((flat_indices, -values_np.astype(np.float64)))
        suppressed = np.zeros(len(coords_np), dtype=bool)

        for row in order.tolist():
            if suppressed[row]:
                continue
            selected_rows.append(row)
            if len(selected_rows) >= max_markers:
                break
            neighbors = tree.query_ball_point(
                physical_um[row], r=float(radius_um) + PHYSICAL_EPS_UM
            )
            if neighbors:
                suppressed[np.asarray(neighbors, dtype=np.int64)] = True

    selected_coords = (
        coords_np[np.asarray(selected_rows, dtype=np.int64)]
        if selected_rows
        else np.zeros((0, 3), dtype=np.int64)
    )
    timings["exact_physical_candidate_nms"] = time.perf_counter() - start

    # Exact separation check for the NMS-selected peaks before component repair.
    minimum_selected_distance_um: float | None = None
    violating_pairs = 0
    if len(selected_coords) >= 2:
        selected_physical = selected_coords.astype(np.float64) * spacing_np[None]
        selected_tree = cKDTree(selected_physical)
        nearest = selected_tree.query(selected_physical, k=2)[0][:, 1]
        minimum_selected_distance_um = float(nearest.min())
        violating_pairs = len(
            selected_tree.query_pairs(
                r=max(float(radius_um) - PHYSICAL_EPS_UM, 0.0),
                output_type="set",
            )
        )

    # ------------------------------------------------------------------
    # D. Materialize point markers and preserve existing foreground-component
    #    repair semantics. The repair is deliberately timed separately.
    # ------------------------------------------------------------------
    start = time.perf_counter()
    markers = np.zeros(shape, dtype=np.int32)
    for marker_id, point in enumerate(selected_coords, 1):
        markers[tuple(int(v) for v in point)] = marker_id
    timings["marker_materialize"] = time.perf_counter() - start

    sync_cuda(device)
    start = time.perf_counter()
    foreground_np = foreground.bool().detach().cpu().numpy()
    sync_cuda(device)
    timings["foreground_transfer_to_cpu"] = time.perf_counter() - start

    start = time.perf_counter()
    foreground_cc, foreground_count = ndi.label(foreground_np)
    timings["foreground_connected_components"] = time.perf_counter() - start

    start = time.perf_counter()
    present = np.unique(foreground_cc[markers > 0])
    missing = np.setdiff1d(
        np.arange(1, foreground_count + 1), present, assume_unique=False
    )
    repair_count = 0
    if len(missing):
        sync_cuda(device)
        score_np = score32.detach().cpu().numpy()
        sync_cuda(device)
        next_id = int(markers.max()) + 1
        flat_fg = foreground_cc.ravel()
        flat_score = score_np.ravel()
        flat_markers = markers.ravel()
        for component_id in missing.tolist():
            flat = np.flatnonzero(flat_fg == component_id)
            if flat.size == 0:
                continue
            best_flat = flat[int(np.argmax(flat_score[flat]))]
            flat_markers[best_flat] = next_id
            next_id += 1
            repair_count += 1
    timings["missing_component_seed_repair"] = time.perf_counter() - start

    timings["total"] = time.perf_counter() - total_start
    detail = {
        "shape_zyx": list(shape),
        "raw_candidate_voxel_count": int(len(candidate_voxel_coords)),
        "plateau_candidate_count": int(plateau_count),
        "nms_candidate_count": int(len(coords_np)),
        "nms_selected_count": int(len(selected_coords)),
        "missing_component_repair_count": int(repair_count),
        "final_marker_count": int(markers.max()),
        "foreground_component_count": int(foreground_count),
        "radius_um": float(radius_um),
        "threshold": float(threshold),
        "minimum_nms_selected_pair_distance_um": minimum_selected_distance_um,
        "nms_selected_pairs_below_radius": int(violating_pairs),
        "timings_seconds": timings if collect_timings else {},
    }
    return markers, detail


# =============================================================================
# BENCHMARK / COMPARISON HELPERS
# =============================================================================


def call_fast_markers(
    score: Tensor,
    foreground: Tensor,
    spacing_um: Tensor,
    radius_um: float,
    threshold: float,
    max_markers: int,
    *,
    detailed: bool,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    profiler = SyncWallProfiler(device) if detailed else None
    sync_cuda(device)
    started = time.perf_counter()
    markers = build_markers_fast(
        score,
        foreground,
        spacing_um,
        radius_um,
        threshold,
        max_markers,
        stage_profiler=profiler,
        profile_prefix="fast",
    )
    sync_cuda(device)
    total = time.perf_counter() - started
    detail = {
        "final_marker_count": int(markers.max()),
        "timings_seconds": {"total": total},
    }
    if profiler is not None:
        detail["timings_seconds"].update(profiler.timings)
    return markers, detail


def benchmark_method(
    fn: Callable[[], tuple[np.ndarray, dict[str, Any]]],
    *,
    repeats: int,
    device: torch.device,
    warmup: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    # Warm-up is intentionally excluded from statistics. For the large stress
    # test we can disable it to avoid paying twice for an intentionally heavy
    # backend on a small local GPU.
    if warmup:
        warm_markers, _ = fn()
        del warm_markers
        sync_cuda(device)

    totals: list[float] = []
    peaks: list[float] = []
    result_markers: np.ndarray | None = None
    result_detail: dict[str, Any] | None = None

    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync_cuda(device)
        started = time.perf_counter()
        markers, detail = fn()
        sync_cuda(device)
        elapsed = time.perf_counter() - started
        totals.append(float(elapsed))
        peaks.append(peak_allocated_mib(device))
        result_markers = markers
        result_detail = detail

    assert result_markers is not None and result_detail is not None
    summary = dict(result_detail)
    summary["benchmark"] = {
        "repeats": repeats,
        "median_seconds": float(statistics.median(totals)),
        "min_seconds": float(min(totals)),
        "max_seconds": float(max(totals)),
        "median_peak_allocated_mib": float(statistics.median(peaks)),
        "max_peak_allocated_mib": float(max(peaks)),
        "all_seconds": totals,
    }
    return result_markers, summary


def marker_representatives(
    markers: np.ndarray,
    score_np: np.ndarray,
) -> np.ndarray:
    count = int(markers.max())
    if count <= 0:
        return np.zeros((0, 3), dtype=np.int64)
    output: list[tuple[int, int, int]] = []
    for marker_id in range(1, count + 1):
        flat = np.flatnonzero(markers.ravel() == marker_id)
        if flat.size == 0:
            continue
        values = score_np.ravel()[flat]
        best_value = values.max()
        best = flat[values == best_value]
        best_flat = int(best.min())
        output.append(tuple(int(v) for v in np.unravel_index(best_flat, markers.shape)))
    return np.asarray(output, dtype=np.int64).reshape(-1, 3)


def marker_set_comparison(
    a: np.ndarray,
    b: np.ndarray,
    score_np: np.ndarray,
    spacing_um: np.ndarray,
    radius_um: float,
) -> dict[str, Any]:
    a_coords = marker_representatives(a, score_np)
    b_coords = marker_representatives(b, score_np)
    if len(a_coords) == 0 or len(b_coords) == 0:
        return {
            "marker_count_a": int(len(a_coords)),
            "marker_count_b": int(len(b_coords)),
            "median_a_to_b_um": None,
            "p95_a_to_b_um": None,
            "median_b_to_a_um": None,
            "p95_b_to_a_um": None,
            "a_fraction_within_radius_of_b": 0.0 if len(a_coords) else 1.0,
            "b_fraction_within_radius_of_a": 0.0 if len(b_coords) else 1.0,
        }

    a_um = a_coords.astype(np.float64) * spacing_um[None]
    b_um = b_coords.astype(np.float64) * spacing_um[None]
    a_to_b = cKDTree(b_um).query(a_um, k=1)[0]
    b_to_a = cKDTree(a_um).query(b_um, k=1)[0]
    return {
        "marker_count_a": int(len(a_coords)),
        "marker_count_b": int(len(b_coords)),
        "median_a_to_b_um": float(np.median(a_to_b)),
        "p95_a_to_b_um": float(np.percentile(a_to_b, 95)),
        "median_b_to_a_um": float(np.median(b_to_a)),
        "p95_b_to_a_um": float(np.percentile(b_to_a, 95)),
        "a_fraction_within_radius_of_b": float(np.mean(a_to_b <= radius_um + PHYSICAL_EPS_UM)),
        "b_fraction_within_radius_of_a": float(np.mean(b_to_a <= radius_um + PHYSICAL_EPS_UM)),
    }


def downstream_proposal_diagnostics(
    markers: np.ndarray,
    *,
    energy_np: np.ndarray,
    foreground_np: np.ndarray,
    gt_np: np.ndarray,
    complete_gt_ids: list[int],
    cfg,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw = stage03.skimage_watershed(
        energy_np,
        markers=markers,
        mask=foreground_np,
        connectivity=1,
    ).astype(np.int32)
    clean = stage03._merge_tiny_regions(
        raw.astype(np.int32, copy=False),
        cfg.partition.min_supervoxel_voxels,
    )
    watershed_seconds = time.perf_counter() - started
    marker_summary = stage03.marker_diagnostics(
        markers, gt_np, complete_gt_ids
    )
    sv_summary, _ = stage03.supervoxel_diagnostics(
        clean, gt_np, complete_gt_ids, cfg
    )
    return {
        "marker_count": marker_summary["marker_count"],
        "complete_gt_marker_recall": marker_summary[
            "complete_gt_marker_recall"
        ],
        "markers_on_background": marker_summary["markers_on_background"],
        "watershed_seconds": float(watershed_seconds),
        "supervoxel_count": int(sv_summary["supervoxel_count"]),
        "cross_gt_unsafe_supervoxel_count": int(
            sv_summary["cross_gt_unsafe_supervoxel_count"]
        ),
        "complete_gt_min_coverage": sv_summary["complete_gt_min_coverage"],
        "complete_gt_recoverable_fraction": float(
            sv_summary["complete_gt_recoverable_fraction"]
        ),
    }


def quality_pass(candidate: dict[str, Any], fast: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if candidate["complete_gt_marker_recall"] + 1e-9 < fast["complete_gt_marker_recall"]:
        failures.append(
            "candidate complete-GT marker recall is below fast backend"
        )
    if (
        candidate["cross_gt_unsafe_supervoxel_count"]
        > fast["cross_gt_unsafe_supervoxel_count"]
    ):
        failures.append(
            "candidate creates more meaningful cross-GT supervoxels than fast backend"
        )
    if (
        candidate["complete_gt_recoverable_fraction"] + 1e-3
        < fast["complete_gt_recoverable_fraction"]
    ):
        failures.append(
            "candidate complete-GT recoverability is below fast backend"
        )
    return not failures, failures


def tile_to_shape(value: Tensor, target_shape: tuple[int, int, int]) -> Tensor:
    if value.ndim != 3:
        raise ValueError("tile_to_shape expects [Z,Y,X]")
    repeats = tuple(
        int(math.ceil(target / source))
        for target, source in zip(target_shape, value.shape)
    )
    tiled = value.repeat(*repeats)
    return tiled[
        : target_shape[0],
        : target_shape[1],
        : target_shape[2],
    ].contiguous()


# =============================================================================
# PRINTING
# =============================================================================


def print_method_timing(name: str, detail: dict[str, Any]) -> None:
    benchmark = detail.get("benchmark", {})
    print(f"\n{name}")
    print("-" * 82)
    if benchmark:
        print(
            f"median total             : {format_seconds(benchmark['median_seconds'])} "
            f"(min {format_seconds(benchmark['min_seconds'])}, "
            f"max {format_seconds(benchmark['max_seconds'])})"
        )
        print(
            f"peak allocated           : {benchmark['median_peak_allocated_mib']:.1f} MiB median / "
            f"{benchmark['max_peak_allocated_mib']:.1f} MiB max"
        )
    if "raw_candidate_voxel_count" in detail:
        print(f"raw candidate voxels     : {detail['raw_candidate_voxel_count']}")
        print(f"candidate plateaus       : {detail.get('plateau_candidate_count', 'n/a')}")
        print(f"physical-NMS candidates  : {detail.get('nms_candidate_count', 'n/a')}")
        print(f"NMS-selected peaks       : {detail['nms_selected_count']}")
        print(f"component repairs        : {detail['missing_component_repair_count']}")
    print(f"final markers            : {detail.get('final_marker_count')}")
    timings = detail.get("timings_seconds", {})
    for key, seconds in sorted(timings.items(), key=lambda item: -item[1]):
        if key == "total":
            continue
        print(f"  {key:<42} {format_seconds(float(seconds)):>12}")


def print_quality_table(rows: list[dict[str, Any]]) -> None:
    print("\nProposal-quality comparison (same watershed energy, only markers differ)")
    print("-" * 116)
    print(
        f"{'method':<24} {'markers':>8} {'GTrecall':>10} {'SV':>7} "
        f"{'unsafeSV':>9} {'minCov':>10} {'recover':>10} {'ws sec':>9}"
    )
    print("-" * 116)
    for row in rows:
        min_cov = row["complete_gt_min_coverage"]
        min_cov_text = "n/a" if min_cov is None else f"{float(min_cov):.4f}"
        print(
            f"{row['method']:<24} {row['marker_count']:>8d} "
            f"{row['complete_gt_marker_recall']:>10.4f} "
            f"{row['supervoxel_count']:>7d} "
            f"{row['cross_gt_unsafe_supervoxel_count']:>9d} "
            f"{min_cov_text:>10} "
            f"{row['complete_gt_recoverable_fraction']:>10.4f} "
            f"{row['watershed_seconds']:>9.3f}"
        )
    print("-" * 116)


# =============================================================================
# MAIN
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 11: candidate exact-physical NMS vs fast rectangular NMS."
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_STAGE03_ARTIFACT)
    parser.add_argument("--stage03-summary", type=Path, default=DEFAULT_STAGE03_SUMMARY)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help="Stage-03 artifact variants to test.",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--stress-repeats", type=int, default=1)
    parser.add_argument(
        "--stress-shape",
        nargs=3,
        type=int,
        metavar=("Z", "Y", "X"),
        default=list(DEFAULT_STRESS_SHAPE),
        help="Large timing-only tiled shape; default matches Stage-10 useful ROI.",
    )
    parser.add_argument("--skip-stress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.stress_repeats < 1:
        raise ValueError("repeat counts must be >= 1")
    stress_shape = tuple(int(v) for v in args.stress_shape)
    if any(v < 1 for v in stress_shape):
        raise ValueError("stress dimensions must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Stage 11 is intentionally a local-GPU benchmark, but CUDA is unavailable."
        )
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()

    if not args.artifact.exists():
        raise FileNotFoundError(
            f"Missing Stage-03 artifact: {args.artifact}\n"
            "Run: python investigations/stirnet/03_watershed_supervoxel_debug.py"
        )
    if not args.sample_path.exists():
        raise FileNotFoundError(f"Missing Stage-03 sample: {args.sample_path}")

    artifact = torch_load(args.artifact)
    variants = dict(artifact.get("variants", {}))
    if not variants:
        raise RuntimeError("Stage-03 artifact has no variants")

    stage03_summary = {}
    if args.stage03_summary.exists():
        stage03_summary = json.loads(args.stage03_summary.read_text(encoding="utf-8"))

    spacing_cpu = torch.as_tensor(artifact["spacing_um"]).float().reshape(3)
    dref_um = float(torch.as_tensor(artifact["dref_um"]).item())
    spacing_np = spacing_cpu.numpy().astype(np.float64)
    complete_gt_ids = [int(v) for v in artifact.get("complete_gt_ids", [])]

    cfg = stage03.stage01.build_debug_config()
    radius_um = float(cfg.partition.seed_min_distance_dref * dref_um)
    threshold = float(cfg.partition.seed_threshold)
    max_markers = int(cfg.partition.max_supervoxels)

    # GT is loaded from the already-prepared Stage-00/01 sample. No target
    # generation and no dense forward is performed in Stage 11.
    batch = stage03.stage01.load_debug_batch(args.sample_path, torch.device("cpu"))
    gt_np = batch.gt_labels.detach().cpu().numpy().astype(np.int32, copy=False)

    requested = [name for name in args.variants if name in variants]
    missing = [name for name in args.variants if name not in variants]
    if missing:
        print(f"Warning: Stage-03 artifact does not contain variants: {missing}")
    if not requested:
        raise RuntimeError("None of the requested variants exist in the Stage-03 artifact")

    saved_backend = (
        stage03_summary.get("partition_config", {}).get("watershed_backend", "unknown")
    )

    print("\n" + "=" * 118)
    print("STIR-Net Stage 11 — candidate exact-physical marker NMS benchmark")
    print("=" * 118)
    print(f"Device                   : {torch.cuda.get_device_name(device)}")
    props = torch.cuda.get_device_properties(device)
    print(f"VRAM                     : {props.total_memory / 1024**3:.2f} GiB")
    print(f"Stage-03 artifact        : {args.artifact}")
    print(f"Sample                   : {args.sample_path}")
    print(f"Variants                 : {requested}")
    print(f"Spacing Z,Y,X            : {tuple(float(v) for v in spacing_np)} um")
    print(f"dref                     : {dref_um:.6f} um")
    print(f"Physical NMS radius      : {radius_um:.6f} um")
    print(f"Seed threshold           : {threshold:.4f}")
    print(f"Max markers              : {max_markers}")
    print(f"Saved Stage-03 backend   : {saved_backend}")
    print(f"Benchmark repeats        : {args.repeats}")
    print(
        "Stress test              : "
        + ("disabled" if args.skip_stress else f"{stress_shape}, repeats={args.stress_repeats}")
    )
    print("=" * 118)

    result: dict[str, Any] = {
        "format_version": 1,
        "kind": "stirnet_stage11_candidate_physical_nms",
        "device": torch.cuda.get_device_name(device),
        "total_vram_gib": float(props.total_memory / 1024**3),
        "artifact_path": str(args.artifact),
        "sample_path": str(args.sample_path),
        "saved_stage03_backend": saved_backend,
        "spacing_zyx_um": [float(v) for v in spacing_np],
        "dref_um": dref_um,
        "radius_um": radius_um,
        "seed_threshold": threshold,
        "max_markers": max_markers,
        "variants": {},
    }
    marker_artifact: dict[str, Any] = {
        "format_version": 1,
        "kind": "stirnet_stage11_candidate_physical_nms",
        "spacing_um": spacing_cpu,
        "dref_um": torch.tensor(dref_um),
        "radius_um": radius_um,
        "variants": {},
    }

    overall_quality_pass = True
    all_failures: list[str] = []

    for variant_name in requested:
        print("\n" + "#" * 118)
        print(f"VARIANT: {variant_name}")
        print("#" * 118)

        variant = variants[variant_name]
        score_cpu = torch.as_tensor(variant["seed_score"]).float().contiguous()
        foreground_cpu = torch.as_tensor(variant["foreground_mask"]).bool().contiguous()
        energy_np = (
            torch.as_tensor(variant["watershed_energy"])
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        foreground_np = foreground_cpu.numpy()
        score_np = score_cpu.numpy().astype(np.float32, copy=False)
        saved_markers = (
            torch.as_tensor(variant["markers"])
            .cpu()
            .numpy()
            .astype(np.int32, copy=False)
        )

        score = score_cpu.to(device, non_blocking=False)
        foreground = foreground_cpu.to(device, non_blocking=False)
        spacing = spacing_cpu.to(device)

        def candidate_call():
            return build_markers_candidate_physical(
                score,
                foreground,
                spacing,
                radius_um,
                threshold,
                max_markers,
                collect_timings=True,
            )

        def fast_call():
            return call_fast_markers(
                score,
                foreground,
                spacing,
                radius_um,
                threshold,
                max_markers,
                detailed=True,
                device=device,
            )

        candidate_markers, candidate_detail = benchmark_method(
            candidate_call, repeats=args.repeats, device=device
        )
        fast_markers, fast_detail = benchmark_method(
            fast_call, repeats=args.repeats, device=device
        )

        print_method_timing("candidate exact-physical NMS", candidate_detail)
        print_method_timing("existing fast rectangular NMS", fast_detail)

        candidate_diag = downstream_proposal_diagnostics(
            candidate_markers,
            energy_np=energy_np,
            foreground_np=foreground_np,
            gt_np=gt_np,
            complete_gt_ids=complete_gt_ids,
            cfg=cfg,
        )
        fast_diag = downstream_proposal_diagnostics(
            fast_markers,
            energy_np=energy_np,
            foreground_np=foreground_np,
            gt_np=gt_np,
            complete_gt_ids=complete_gt_ids,
            cfg=cfg,
        )
        saved_diag = downstream_proposal_diagnostics(
            saved_markers,
            energy_np=energy_np,
            foreground_np=foreground_np,
            gt_np=gt_np,
            complete_gt_ids=complete_gt_ids,
            cfg=cfg,
        )

        rows = []
        for method, diag in (
            ("candidate_physical", candidate_diag),
            ("fast_rectangular", fast_diag),
            (f"saved_stage03_{saved_backend}", saved_diag),
        ):
            rows.append({"method": method, **diag})
        print_quality_table(rows)

        physical_check_pass = (
            candidate_detail.get("nms_selected_pairs_below_radius", 0) == 0
        )
        qpass, qfailures = quality_pass(candidate_diag, fast_diag)
        if not physical_check_pass:
            qfailures.append(
                "candidate NMS retained at least one selected pair below the exact physical radius"
            )
            qpass = False

        overall_quality_pass &= qpass
        for failure in qfailures:
            all_failures.append(f"{variant_name}: {failure}")

        print("\nCandidate-vs-fast verdict")
        print(f"  physical-spacing invariant : {'PASS' if physical_check_pass else 'FAIL'}")
        print(f"  proposal quality           : {'PASS' if qpass else 'FAIL'}")
        if qfailures:
            for failure in qfailures:
                print(f"  - {failure}")
        else:
            print("  - complete-cell marker recall is not worse than fast")
            print("  - cross-GT proposal safety is not worse than fast")
            print("  - complete-cell recoverability is not worse than fast")

        candidate_vs_fast = marker_set_comparison(
            candidate_markers,
            fast_markers,
            score_np,
            spacing_np,
            radius_um,
        )
        candidate_vs_saved = marker_set_comparison(
            candidate_markers,
            saved_markers,
            score_np,
            spacing_np,
            radius_um,
        )

        result["variants"][variant_name] = {
            "shape_zyx": list(score.shape),
            "candidate": candidate_detail,
            "fast": fast_detail,
            "candidate_proposal": candidate_diag,
            "fast_proposal": fast_diag,
            "saved_stage03_proposal": saved_diag,
            "candidate_vs_fast_marker_set": candidate_vs_fast,
            "candidate_vs_saved_marker_set": candidate_vs_saved,
            "quality_pass": bool(qpass),
            "quality_failures": qfailures,
        }
        marker_artifact["variants"][variant_name] = {
            "seed_score": score_cpu.half(),
            "foreground_mask": foreground_cpu.to(torch.uint8),
            "candidate_markers": torch.from_numpy(candidate_markers),
            "fast_markers": torch.from_numpy(fast_markers),
            "saved_stage03_markers": torch.from_numpy(saved_markers.copy()),
        }

        del score, foreground
        gc.collect()
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Large tiled timing-only stress test.
    # -------------------------------------------------------------------------
    stress_result: dict[str, Any] | None = None
    if not args.skip_stress:
        source_variant_name = requested[0]
        source_variant = variants[source_variant_name]
        base_score = torch.as_tensor(source_variant["seed_score"]).float().contiguous()
        base_fg = torch.as_tensor(source_variant["foreground_mask"]).bool().contiguous()

        print("\n" + "=" * 118)
        print("LARGE TIMING-ONLY STRESS TEST")
        print("=" * 118)
        print(f"Source variant           : {source_variant_name}")
        print(f"Base shape               : {tuple(base_score.shape)}")
        print(f"Tiled target shape       : {stress_shape}")
        print(f"Target voxels            : {math.prod(stress_shape):,}")
        print("GT / watershed quality   : intentionally not evaluated on tiled synthetic scene")

        stress_score_cpu = tile_to_shape(base_score, stress_shape)
        stress_fg_cpu = tile_to_shape(base_fg, stress_shape)
        stress_score = stress_score_cpu.to(device)
        stress_fg = stress_fg_cpu.to(device)
        spacing = spacing_cpu.to(device)

        stress_result = {
            "shape_zyx": list(stress_shape),
            "voxel_count": int(math.prod(stress_shape)),
            "source_variant": source_variant_name,
        }

        try:
            candidate_markers, candidate_detail = benchmark_method(
                lambda: build_markers_candidate_physical(
                    stress_score,
                    stress_fg,
                    spacing,
                    radius_um,
                    threshold,
                    max_markers,
                    collect_timings=True,
                ),
                repeats=args.stress_repeats,
                device=device,
                warmup=False,
            )
            stress_result["candidate"] = candidate_detail
            print_method_timing("candidate exact-physical NMS [stress]", candidate_detail)
            del candidate_markers
        except torch.OutOfMemoryError as exc:
            stress_result["candidate"] = {
                "status": "oom",
                "exception": str(exc),
            }
            print(f"\nCandidate stress test OOM: {exc}")
            torch.cuda.empty_cache()

        try:
            fast_markers, fast_detail = benchmark_method(
                lambda: call_fast_markers(
                    stress_score,
                    stress_fg,
                    spacing,
                    radius_um,
                    threshold,
                    max_markers,
                    detailed=True,
                    device=device,
                ),
                repeats=args.stress_repeats,
                device=device,
                warmup=False,
            )
            stress_result["fast"] = fast_detail
            print_method_timing("existing fast rectangular NMS [stress]", fast_detail)
            del fast_markers
        except torch.OutOfMemoryError as exc:
            stress_result["fast"] = {
                "status": "oom",
                "exception": str(exc),
            }
            print(f"\nFast stress test OOM: {exc}")
            torch.cuda.empty_cache()

        result["stress_test"] = stress_result
        del stress_score, stress_fg, stress_score_cpu, stress_fg_cpu
        gc.collect()
        torch.cuda.empty_cache()

    result["overall_quality_pass"] = bool(overall_quality_pass)
    result["failures"] = all_failures

    args.result_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.result_dir / "stage11_candidate_physical_nms.json"
    artifact_path = args.result_dir / "stage11_candidate_physical_nms.pt"
    atomic_json(json_path, result)
    atomic_torch_save(artifact_path, marker_artifact)

    print("\n" + "=" * 118)
    print("Stage 11 summary")
    print("=" * 118)
    print(f"Quality verdict          : {'PASS' if overall_quality_pass else 'NEEDS REVIEW'}")
    if all_failures:
        for failure in all_failures:
            print(f"  - {failure}")
    else:
        print("  - candidate physical NMS preserved or improved fast-backend proposal quality")
        print("  - exact physical minimum-distance invariant passed")
    print(f"JSON                     : {json_path}")
    print(f"Compact marker artifact  : {artifact_path}")
    print("=" * 118)


if __name__ == "__main__":
    main()
