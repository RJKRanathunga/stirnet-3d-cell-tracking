from __future__ import annotations

"""
STIR-Net Stage 14 — CPU vs GPU geometry-target preprocessing benchmark.

Why this exists
---------------
STIR-Net now has an exact CuPy EDT backend that can accelerate production
geometry-target construction without changing target definitions.  We have
used the CPU/SciPy path extensively, but the GPU path has not yet been
validated as a complete production pipeline.

Stage 14 benchmarks the SAME target builder three ways:

    scipy
        Force every EDT to SciPy/CPU. This is the reference.

    cupy
        Force every EDT to CuPy/GPU. This is a stress/correctness test of the
        GPU implementation; it is not necessarily the fastest policy because
        many per-cell EDTs are tiny.

    auto
        Production hybrid policy. Large EDTs go to CuPy and small EDTs stay on
        SciPy according to TrainingConfig.geometry_target_gpu_min_voxels.

The benchmark is deliberately target-construction only. It does NOT run the
STIR-Net network, watershed, RAG, backward, or optimizer.

Default benchmark scenes
------------------------
1. TRAINING CROP:
       the current merge-aware 32 x 192 x 192 crop selected from the prepared
       BlastoSPIM frame used in Stage 13.

2. USEFUL ROI:
       full current/source bounding box + 12 um physical margin. On the
       BlastoSPIM reference frame this is close to the large ROI used in
       Stage 10 and is useful for testing GPU scaling.

For each scene Stage 14:
    - runs CPU/SciPy FIRST;
    - runs forced CuPy/GPU SECOND;
    - runs production AUTO THIRD;
    - instruments every EDT call to count CPU/GPU dispatch;
    - compares every generated target tensor against the CPU reference;
    - reports wall time, speed-up, EDT call statistics, and CuPy pool peak;
    - writes a JSON report.

Typical usage
-------------
    python investigations/stirnet/14_geometry_target_cpu_gpu_benchmark.py

Crop only:
    python investigations/stirnet/14_geometry_target_cpu_gpu_benchmark.py \
        --scope crop

Large useful ROI only:
    python investigations/stirnet/14_geometry_target_cpu_gpu_benchmark.py \
        --scope roi

More timing repeats:
    python investigations/stirnet/14_geometry_target_cpu_gpu_benchmark.py \
        --cpu-repeats 3 --gpu-repeats 5

Outputs
-------
data/learned/stirnet/geometry_target_backend_benchmark/stage14_report.json
"""

import argparse
from contextlib import contextmanager
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch


# =============================================================================
# REPOSITORY
# =============================================================================


def _repo_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "learned" / "stirnet").is_dir():
            return candidate
    raise RuntimeError("Could not locate the cell-tracking repository root.")


ROOT = _repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from learned.stirnet import StirNetConfig
from learned.stirnet.data.targets import estimate_model_dref_um
from learned.stirnet.model.geometry import edt_backend as edt_backend_module
from learned.stirnet.model.geometry.edt_backend import (
    cupy_edt_available,
    distance_transform_edt,
    geometry_edt_backend,
    release_cupy_memory,
)
from learned.stirnet.training import TrainingConfig
from learned.stirnet.training.merge_aware_crops import (
    build_merge_aware_crop_manifest,
)
from learned.stirnet.training.prepared_geometry import (
    build_prepared_geometry_targets,
)


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_DATA_DIR = (
    ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "first_overfit"
    / "BlastoSPIM1_F22_030_034"
)
DEFAULT_TIME_INDEX = 2
DEFAULT_CROP_SHAPE = (32, 192, 192)
DEFAULT_ROI_MARGIN_UM = 12.0
DEFAULT_RESULT_DIR = (
    ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "geometry_target_backend_benchmark"
)

# Numerical comparison thresholds. Distances should agree much more tightly
# than these. A tiny separator mismatch can occur if nearest-index tie breaking
# differs for points that are exactly equidistant.
FLOAT_ATOL = 2e-5
FLOAT_RTOL = 2e-5
SEPARATOR_TIE_FRACTION_WARN = 1e-4


# =============================================================================
# SERIALIZATION / FORMATTING
# =============================================================================


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _fmt_seconds(value: float) -> str:
    if value < 1e-3:
        return f"{value * 1e6:.1f} us"
    if value < 1:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.3f} s"


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


# =============================================================================
# EDT INSTRUMENTATION
# =============================================================================


class EDTTrace:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.cupy_pool_peak_bytes = 0

    def add(
        self,
        *,
        backend: str,
        shape: tuple[int, ...],
        voxels: int,
        return_indices: bool,
        seconds: float,
    ) -> None:
        self.calls.append(
            {
                "backend": backend,
                "shape": list(shape),
                "voxels": int(voxels),
                "return_indices": bool(return_indices),
                "seconds": float(seconds),
            }
        )

    def observe_cupy_pool(self) -> None:
        try:
            cp, _ = edt_backend_module._load_cupy()
            if cp is None:
                return
            pool = cp.get_default_memory_pool()
            self.cupy_pool_peak_bytes = max(
                self.cupy_pool_peak_bytes,
                int(pool.total_bytes()),
            )
        except Exception:
            pass

    def summary(self) -> dict[str, Any]:
        cpu = [row for row in self.calls if row["backend"] == "scipy"]
        gpu = [row for row in self.calls if row["backend"] == "cupy"]
        return {
            "total_calls": len(self.calls),
            "scipy_calls": len(cpu),
            "cupy_calls": len(gpu),
            "scipy_voxels": int(sum(row["voxels"] for row in cpu)),
            "cupy_voxels": int(sum(row["voxels"] for row in gpu)),
            "scipy_seconds": float(sum(row["seconds"] for row in cpu)),
            "cupy_seconds": float(sum(row["seconds"] for row in gpu)),
            "return_indices_calls": int(
                sum(bool(row["return_indices"]) for row in self.calls)
            ),
            "cupy_pool_peak_mib": self.cupy_pool_peak_bytes / 1024**2,
            "largest_calls": sorted(
                self.calls,
                key=lambda row: row["voxels"],
                reverse=True,
            )[:12],
        }


@contextmanager
def _trace_edt_calls():
    """Instrument actual backend dispatch without changing target semantics."""
    trace = EDTTrace()
    original_scipy = edt_backend_module._scipy_edt
    original_cupy = edt_backend_module._cupy_edt

    def scipy_wrapper(
        image,
        *,
        sampling=None,
        return_distances=True,
        return_indices=False,
    ):
        array = np.asarray(image)
        started = time.perf_counter()
        result = original_scipy(
            array,
            sampling=sampling,
            return_distances=return_distances,
            return_indices=return_indices,
        )
        elapsed = time.perf_counter() - started
        trace.add(
            backend="scipy",
            shape=tuple(int(v) for v in array.shape),
            voxels=int(array.size),
            return_indices=return_indices,
            seconds=elapsed,
        )
        return result

    def cupy_wrapper(
        image,
        *,
        sampling=None,
        return_distances=True,
        return_indices=False,
    ):
        array = np.asarray(image)
        started = time.perf_counter()
        result = original_cupy(
            array,
            sampling=sampling,
            return_distances=return_distances,
            return_indices=return_indices,
        )
        elapsed = time.perf_counter() - started
        trace.observe_cupy_pool()
        trace.add(
            backend="cupy",
            shape=tuple(int(v) for v in array.shape),
            voxels=int(array.size),
            return_indices=return_indices,
            seconds=elapsed,
        )
        return result

    edt_backend_module._scipy_edt = scipy_wrapper
    edt_backend_module._cupy_edt = cupy_wrapper
    try:
        yield trace
    finally:
        edt_backend_module._scipy_edt = original_scipy
        edt_backend_module._cupy_edt = original_cupy


# =============================================================================
# DATA / SCENE CONSTRUCTION
# =============================================================================


def _load_reference_frame(
    data_dir: Path,
    time_index: int,
) -> dict[str, Any]:
    instance_path = data_dir / "instance_movie.npy"
    gt_path = data_dir / "gt_movie.npy"
    metadata_path = data_dir / "metadata.json"
    for path in (instance_path, gt_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required prepared Stage-13 input is missing: {path}"
            )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    spacing = np.asarray(metadata["spacing_zyx_um"], dtype=np.float32)

    instance_movie = np.load(instance_path, mmap_mode="r")
    gt_movie = np.load(gt_path, mmap_mode="r")
    if not 0 <= time_index < min(len(instance_movie), len(gt_movie)):
        raise IndexError(
            f"time_index={time_index} outside prepared movie range"
        )

    current = np.asarray(instance_movie[time_index], dtype=np.int64)
    gt = np.asarray(gt_movie[time_index], dtype=np.int64)

    started = time.perf_counter()
    dref_um = float(
        estimate_model_dref_um(
            current,
            tuple(float(v) for v in spacing),
        )
    )
    dref_seconds = time.perf_counter() - started

    return {
        "current": current,
        "gt": gt,
        "spacing": spacing,
        "dref_um": dref_um,
        "dref_seconds": dref_seconds,
        "shape": tuple(int(v) for v in gt.shape),
    }


def _training_crop_scene(
    frame: dict[str, Any],
    crop_shape: tuple[int, int, int],
    train_cfg: TrainingConfig,
) -> dict[str, Any]:
    current = frame["current"]
    gt = frame["gt"]
    spacing = frame["spacing"]
    cfg = train_cfg.curriculum

    manifest = build_merge_aware_crop_manifest(
        torch.from_numpy(gt).unsqueeze(0),
        current_labels=torch.from_numpy(current).unsqueeze(0),
        spacing_um=torch.from_numpy(spacing).unsqueeze(0),
        crop_shape_zyx=crop_shape,
        min_complete_cells=cfg.refinement_crop_min_complete_cells,
        preferred_complete_cells=cfg.refinement_crop_preferred_complete_cells,
        views_per_cell=cfg.refinement_crop_views_per_cell,
        context_um=cfg.refinement_crop_context_um,
        merge_min_overlap_voxels=cfg.refinement_crop_merge_min_overlap_voxels,
        merge_min_gt_fraction=cfg.refinement_crop_merge_min_gt_fraction,
    )
    rows = list(manifest.records[0])
    if not rows:
        raise RuntimeError("merge-aware crop planner returned no crops")
    merge_rows = [row for row in rows if row.merge_source_ids]
    record = merge_rows[0] if merge_rows else rows[0]
    slices = record.slices_zyx

    return {
        "name": "training_crop",
        "gt": np.asarray(gt[slices], dtype=np.int64).copy(),
        "current": np.asarray(current[slices], dtype=np.int64).copy(),
        "spacing": spacing.copy(),
        "dref_um": frame["dref_um"],
        "slices": [
            [int(axis.start), int(axis.stop)] for axis in slices
        ],
        "candidate_type": record.candidate_type,
        "merge_source_ids": list(record.merge_source_ids),
        "complete_cell_ids": list(record.complete_cell_ids),
        "partial_cell_ids": list(record.partial_cell_ids),
    }


def _useful_roi_scene(
    frame: dict[str, Any],
    margin_um: float,
) -> dict[str, Any]:
    current = frame["current"]
    gt = frame["gt"]
    spacing = frame["spacing"]

    positive = np.argwhere(current > 0)
    if positive.size == 0:
        raise RuntimeError("reference current segmentation contains no foreground")

    low = positive.min(axis=0)
    high = positive.max(axis=0) + 1
    margin_vox = np.ceil(
        float(margin_um) / np.maximum(spacing, 1e-6)
    ).astype(np.int64)

    shape = np.asarray(current.shape, dtype=np.int64)
    low = np.maximum(low - margin_vox, 0)
    high = np.minimum(high + margin_vox, shape)
    slices = tuple(
        slice(int(lo), int(hi)) for lo, hi in zip(low, high)
    )

    return {
        "name": "useful_roi",
        "gt": np.asarray(gt[slices], dtype=np.int64).copy(),
        "current": np.asarray(current[slices], dtype=np.int64).copy(),
        "spacing": spacing.copy(),
        "dref_um": frame["dref_um"],
        "slices": [
            [int(axis.start), int(axis.stop)] for axis in slices
        ],
        "margin_um": float(margin_um),
    }


# =============================================================================
# TARGET BUILD / COMPARISON
# =============================================================================


def _build_targets_once(
    scene: dict[str, Any],
    *,
    backend: str,
    geometry_cfg,
    train_cfg: TrainingConfig,
) -> tuple[Any, float, dict[str, Any]]:
    gt = torch.from_numpy(scene["gt"]).unsqueeze(0)
    current = torch.from_numpy(scene["current"]).unsqueeze(0)
    spacing = torch.from_numpy(scene["spacing"]).unsqueeze(0)
    dref = torch.tensor([float(scene["dref_um"])], dtype=torch.float32)

    # Clean allocator state between timed backend runs.
    release_cupy_memory()
    gc.collect()

    with _trace_edt_calls() as trace:
        started = time.perf_counter()
        targets = build_prepared_geometry_targets(
            gt,
            spacing,
            dref,
            current_labels=current,
            geometry_config=geometry_cfg,
            backend=backend,
            gpu_min_voxels=train_cfg.geometry_target_gpu_min_voxels,
            device=torch.device("cpu"),
        )
        seconds = time.perf_counter() - started

    return targets, seconds, trace.summary()


def _field_comparison(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    name: str,
) -> dict[str, Any]:
    ref = reference.detach().cpu()
    cand = candidate.detach().cpu()

    if ref.shape != cand.shape:
        return {
            "shape_match": False,
            "reference_shape": list(ref.shape),
            "candidate_shape": list(cand.shape),
            "pass": False,
        }

    if ref.dtype == torch.bool or not ref.dtype.is_floating_point:
        equal = torch.equal(ref, cand)
        differing = int((ref != cand).sum().item())
        total = int(ref.numel())
        return {
            "shape_match": True,
            "exact_equal": bool(equal),
            "differing_voxels": differing,
            "differing_fraction": differing / max(total, 1),
            "pass": bool(equal),
        }

    ref64 = ref.double()
    cand64 = cand.double()
    finite = bool(torch.isfinite(cand64).all())
    delta = (ref64 - cand64).abs()
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    mean_abs = float(delta.mean().item()) if delta.numel() else 0.0
    close = torch.isclose(
        ref64,
        cand64,
        atol=FLOAT_ATOL,
        rtol=FLOAT_RTOL,
    )
    differing = int((~close).sum().item())
    total = int(close.numel())
    differing_fraction = differing / max(total, 1)

    # Separator is allowed to expose a tiny tie-breaking difference separately;
    # it is reported rather than silently treated as identical.
    if name == "separator":
        passed = finite and differing_fraction <= SEPARATOR_TIE_FRACTION_WARN
    else:
        passed = finite and bool(close.all())

    return {
        "shape_match": True,
        "finite": finite,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "outside_tolerance_voxels": differing,
        "outside_tolerance_fraction": differing_fraction,
        "pass": bool(passed),
    }


def _compare_targets(reference, candidate) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in reference.__dict__:
        fields[name] = _field_comparison(
            getattr(reference, name),
            getattr(candidate, name),
            name,
        )
    return {
        "pass": all(row.get("pass", False) for row in fields.values()),
        "fields": fields,
    }


def _warmup_cupy() -> tuple[bool, float, str | None]:
    if not cupy_edt_available():
        return False, 0.0, "CuPy EDT unavailable"

    synthetic = np.ones((32, 64, 64), dtype=bool)
    synthetic[:, 32, 32] = False

    release_cupy_memory()
    started = time.perf_counter()
    try:
        with geometry_edt_backend("cupy", gpu_min_voxels=1):
            distance_transform_edt(
                synthetic,
                sampling=(2.0, 0.4, 0.4),
            )
        elapsed = time.perf_counter() - started
        return True, elapsed, None
    except Exception as exc:
        return False, time.perf_counter() - started, (
            f"{type(exc).__name__}: {exc}"
        )
    finally:
        release_cupy_memory()


def _benchmark_backend(
    scene: dict[str, Any],
    *,
    backend: str,
    repeats: int,
    geometry_cfg,
    train_cfg: TrainingConfig,
) -> tuple[Any | None, dict[str, Any]]:
    times: list[float] = []
    traces: list[dict[str, Any]] = []
    last_targets = None

    try:
        for repeat in range(repeats):
            print(
                f"      {backend:5s} repeat {repeat + 1}/{repeats} ...",
                end="",
                flush=True,
            )
            targets, seconds, trace = _build_targets_once(
                scene,
                backend=backend,
                geometry_cfg=geometry_cfg,
                train_cfg=train_cfg,
            )
            times.append(seconds)
            traces.append(trace)
            last_targets = targets
            print(f" {_fmt_seconds(seconds)}", flush=True)

        med = _median(times)
        best_index = min(
            range(len(times)),
            key=lambda index: abs(times[index] - med),
        )
        representative_trace = traces[best_index]
        result = {
            "status": "PASS",
            "backend": backend,
            "repeats": repeats,
            "times_seconds": times,
            "median_seconds": med,
            "min_seconds": min(times),
            "max_seconds": max(times),
            "representative_edt_trace": representative_trace,
            "all_edt_traces": traces,
        }
        return last_targets, result

    except Exception as exc:
        result = {
            "status": "ERROR",
            "backend": backend,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "completed_times_seconds": times,
            "completed_edt_traces": traces,
        }
        print(
            f"\n      {backend} FAILED: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return last_targets, result
    finally:
        release_cupy_memory()
        gc.collect()


# =============================================================================
# ONE SCENE
# =============================================================================


def _run_scene(
    scene: dict[str, Any],
    *,
    cpu_repeats: int,
    gpu_repeats: int,
    geometry_cfg,
    train_cfg: TrainingConfig,
) -> dict[str, Any]:
    shape = tuple(int(v) for v in scene["gt"].shape)
    voxels = int(np.prod(shape))
    gt_count = int(np.unique(scene["gt"][scene["gt"] > 0]).size)
    current_count = int(
        np.unique(scene["current"][scene["current"] > 0]).size
    )

    print("\n" + "=" * 118)
    print(
        f"SCENE: {scene['name']} | shape={shape} | "
        f"voxels={voxels:,} | GT={gt_count} | current={current_count}"
    )
    if scene.get("candidate_type"):
        print(
            f"crop type={scene['candidate_type']} | "
            f"merge sources={scene.get('merge_source_ids', [])} | "
            f"complete={len(scene.get('complete_cell_ids', []))} | "
            f"partial={len(scene.get('partial_cell_ids', []))}"
        )
    print("=" * 118)

    # Explicit requested order: CPU first, GPU second.
    print("\n[1/3] CPU reference — forced SciPy")
    cpu_targets, cpu = _benchmark_backend(
        scene,
        backend="scipy",
        repeats=cpu_repeats,
        geometry_cfg=geometry_cfg,
        train_cfg=train_cfg,
    )
    if cpu_targets is None or cpu["status"] != "PASS":
        raise RuntimeError(
            f"CPU reference failed for scene {scene['name']}; "
            "cannot perform a meaningful GPU comparison."
        )

    print("\n[2/3] GPU validation — forced CuPy")
    gpu_targets, gpu = _benchmark_backend(
        scene,
        backend="cupy",
        repeats=gpu_repeats,
        geometry_cfg=geometry_cfg,
        train_cfg=train_cfg,
    )
    gpu_comparison = None
    if gpu_targets is not None and gpu["status"] == "PASS":
        gpu_comparison = _compare_targets(cpu_targets, gpu_targets)

    print("\n[3/3] Production policy — AUTO hybrid")
    auto_targets, auto = _benchmark_backend(
        scene,
        backend="auto",
        repeats=gpu_repeats,
        geometry_cfg=geometry_cfg,
        train_cfg=train_cfg,
    )
    auto_comparison = None
    if auto_targets is not None and auto["status"] == "PASS":
        auto_comparison = _compare_targets(cpu_targets, auto_targets)

    cpu_median = float(cpu["median_seconds"])

    def speedup(row: dict[str, Any]) -> float | None:
        if row.get("status") != "PASS":
            return None
        med = float(row["median_seconds"])
        return cpu_median / med if med > 0 else None

    result = {
        "scene": {
            key: value
            for key, value in scene.items()
            if key not in {"gt", "current", "spacing"}
        },
        "shape_zyx": list(shape),
        "voxel_count": voxels,
        "gt_instance_count": gt_count,
        "current_instance_count": current_count,
        "spacing_zyx_um": scene["spacing"].tolist(),
        "cpu_scipy": cpu,
        "gpu_cupy": gpu,
        "auto_hybrid": auto,
        "gpu_vs_cpu": gpu_comparison,
        "auto_vs_cpu": auto_comparison,
        "gpu_speedup_vs_cpu": speedup(gpu),
        "auto_speedup_vs_cpu": speedup(auto),
    }

    print("\n" + "-" * 118)
    print("RESULT")
    print("-" * 118)
    print(f"CPU/SciPy median     : {_fmt_seconds(cpu_median)}")
    if gpu.get("status") == "PASS":
        print(
            f"GPU/CuPy median      : "
            f"{_fmt_seconds(gpu['median_seconds'])} "
            f"({result['gpu_speedup_vs_cpu']:.2f}x vs CPU)"
        )
        print(
            f"GPU parity           : "
            f"{'PASS' if gpu_comparison and gpu_comparison['pass'] else 'CHECK'}"
        )
    else:
        print("GPU/CuPy             : FAILED")
    if auto.get("status") == "PASS":
        print(
            f"AUTO median          : "
            f"{_fmt_seconds(auto['median_seconds'])} "
            f"({result['auto_speedup_vs_cpu']:.2f}x vs CPU)"
        )
        trace = auto["representative_edt_trace"]
        print(
            f"AUTO EDT dispatch    : "
            f"CPU {trace['scipy_calls']} calls / "
            f"GPU {trace['cupy_calls']} calls"
        )
        print(
            f"AUTO parity          : "
            f"{'PASS' if auto_comparison and auto_comparison['pass'] else 'CHECK'}"
        )
    else:
        print("AUTO                 : FAILED")
    print("-" * 118)

    return result


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "--scope",
        choices=("crop", "roi", "both"),
        default="both",
        help="Benchmark the training crop, useful ROI, or both.",
    )
    parser.add_argument(
        "--crop-shape",
        nargs=3,
        type=int,
        metavar=("Z", "Y", "X"),
        default=DEFAULT_CROP_SHAPE,
    )
    parser.add_argument(
        "--roi-margin-um",
        type=float,
        default=DEFAULT_ROI_MARGIN_UM,
    )
    parser.add_argument(
        "--cpu-repeats",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--gpu-repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--gpu-min-voxels",
        type=int,
        default=None,
        help=(
            "Override TrainingConfig.geometry_target_gpu_min_voxels for AUTO. "
            "Default keeps the repository production value."
        ),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
    )
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    args = parse_args()
    if args.cpu_repeats < 1 or args.gpu_repeats < 1:
        raise ValueError("repeat counts must be positive")
    if args.roi_margin_um < 0:
        raise ValueError("roi margin cannot be negative")

    geometry_cfg = StirNetConfig().geometry
    train_cfg = TrainingConfig()
    if args.gpu_min_voxels is not None:
        train_cfg.geometry_target_gpu_min_voxels = int(args.gpu_min_voxels)
    train_cfg.validate()

    print("\n" + "=" * 118)
    print("STIR-Net Stage 14 — CPU vs GPU geometry-target preprocessing")
    print("=" * 118)
    print(f"Repository                : {ROOT}")
    print(f"Data                      : {args.data_dir}")
    print(f"Time index                : {args.time_index}")
    print(f"Scope                     : {args.scope}")
    print(f"CPU backend               : SciPy")
    print(f"GPU backend               : CuPy")
    print(f"Production backend        : auto")
    print(
        f"AUTO GPU threshold        : "
        f"{train_cfg.geometry_target_gpu_min_voxels:,} voxels"
    )
    print(f"CPU repeats               : {args.cpu_repeats}")
    print(f"GPU/AUTO repeats          : {args.gpu_repeats}")
    print(f"torch                     : {torch.__version__}")
    print(f"CUDA available            : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU                       : {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(
            f"GPU VRAM                  : "
            f"{total / 1024**3:.2f} GiB total / "
            f"{free / 1024**3:.2f} GiB free"
        )
    print(f"CuPy EDT available        : {cupy_edt_available()}")
    print("=" * 118, flush=True)

    if not cupy_edt_available():
        raise RuntimeError(
            "CuPy EDT is unavailable. Stage 14 requires the GPU backend so "
            "CPU-vs-GPU comparison would be meaningless."
        )

    print("\n[setup] Warming up CuPy EDT once (not included in benchmark) ...")
    warmup_ok, warmup_seconds, warmup_error = _warmup_cupy()
    print(
        f"[setup] CuPy warmup: "
        f"{'PASS' if warmup_ok else 'FAIL'} in "
        f"{_fmt_seconds(warmup_seconds)}"
    )
    if not warmup_ok:
        raise RuntimeError(f"CuPy EDT warmup failed: {warmup_error}")

    print("\n[setup] Loading prepared reference frame ...")
    started = time.perf_counter()
    frame = _load_reference_frame(args.data_dir, args.time_index)
    load_seconds = time.perf_counter() - started
    print(
        f"[setup] shape={frame['shape']} "
        f"spacing={tuple(float(v) for v in frame['spacing'])} um "
        f"dref={frame['dref_um']:.4f} um "
        f"load={_fmt_seconds(load_seconds)}"
    )

    scenes: list[dict[str, Any]] = []
    if args.scope in {"crop", "both"}:
        print("[setup] Selecting current merge-aware training crop ...")
        scenes.append(
            _training_crop_scene(
                frame,
                tuple(int(v) for v in args.crop_shape),
                train_cfg,
            )
        )
    if args.scope in {"roi", "both"}:
        print("[setup] Building current/source useful ROI ...")
        scenes.append(
            _useful_roi_scene(
                frame,
                float(args.roi_margin_um),
            )
        )

    report: dict[str, Any] = {
        "format_version": 1,
        "stage": "stirnet_stage14_geometry_target_cpu_gpu_benchmark",
        "repository": str(ROOT),
        "data_dir": str(args.data_dir),
        "time_index": int(args.time_index),
        "reference_full_shape_zyx": list(frame["shape"]),
        "spacing_zyx_um": frame["spacing"].tolist(),
        "dref_um": float(frame["dref_um"]),
        "dref_seconds": float(frame["dref_seconds"]),
        "cupy_edt_available": bool(cupy_edt_available()),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "auto_gpu_min_voxels": int(
            train_cfg.geometry_target_gpu_min_voxels
        ),
        "cupy_warmup_seconds": float(warmup_seconds),
        "scenes": {},
    }

    failures: list[str] = []

    for scene in scenes:
        result = _run_scene(
            scene,
            cpu_repeats=args.cpu_repeats,
            gpu_repeats=args.gpu_repeats,
            geometry_cfg=geometry_cfg,
            train_cfg=train_cfg,
        )
        report["scenes"][scene["name"]] = result

        gpu = result["gpu_cupy"]
        auto = result["auto_hybrid"]
        if gpu.get("status") != "PASS":
            failures.append(f"{scene['name']}: forced CuPy backend failed")
        elif not result.get("gpu_vs_cpu", {}).get("pass", False):
            failures.append(
                f"{scene['name']}: forced CuPy targets differ from CPU reference"
            )
        if auto.get("status") != "PASS":
            failures.append(f"{scene['name']}: AUTO backend failed")
        elif not result.get("auto_vs_cpu", {}).get("pass", False):
            failures.append(
                f"{scene['name']}: AUTO targets differ from CPU reference"
            )

    report["failures"] = failures
    report["passed"] = not failures

    result_path = args.result_dir / "stage14_report.json"
    _atomic_json(result_path, report)

    print("\n" + "=" * 118)
    print("STAGE 14 CONCLUSION")
    print("=" * 118)
    for name, row in report["scenes"].items():
        cpu = row["cpu_scipy"]
        gpu = row["gpu_cupy"]
        auto = row["auto_hybrid"]
        print(f"{name}:")
        print(
            f"  CPU/SciPy : {_fmt_seconds(cpu['median_seconds'])}"
        )
        if gpu.get("status") == "PASS":
            print(
                f"  GPU/CuPy  : {_fmt_seconds(gpu['median_seconds'])} "
                f"| speedup={row['gpu_speedup_vs_cpu']:.2f}x "
                f"| parity={'PASS' if row['gpu_vs_cpu']['pass'] else 'CHECK'}"
            )
        else:
            print("  GPU/CuPy  : FAILED")
        if auto.get("status") == "PASS":
            trace = auto["representative_edt_trace"]
            print(
                f"  AUTO      : {_fmt_seconds(auto['median_seconds'])} "
                f"| speedup={row['auto_speedup_vs_cpu']:.2f}x "
                f"| CPU/GPU EDT calls="
                f"{trace['scipy_calls']}/{trace['cupy_calls']} "
                f"| parity="
                f"{'PASS' if row['auto_vs_cpu']['pass'] else 'CHECK'}"
            )
        else:
            print("  AUTO      : FAILED")

    print(f"\nAcceptance : {'PASS' if report['passed'] else 'CHECK/FAIL'}")
    if failures:
        for failure in failures:
            print(f"  - {failure}")
    print(f"JSON       : {result_path}")
    print("=" * 118)

    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
