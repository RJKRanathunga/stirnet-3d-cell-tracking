from __future__ import annotations

r"""
Investigation 34 — serial-vs-pipelined spatial inference benchmark.

Purpose
-------
Measure how much wall-clock time can be saved by overlapping preparation of
future BioHub frames on CPU with STIR-Net spatial inference on the GPU.

This is intentionally an EXECUTION-ONLY experiment.  It does not patch the
production repository pipeline.  The scientific operations are unchanged:

    CPU preparation
        load_timepoint
        -> preprocess_volume
        -> create_binary_mask
        -> source segment_instances with geometric completion DISABLED
        -> exact five-channel STIR input construction

    GPU/main-thread inference
        tiled STIR-Net spatial inference
        -> RAG / signed multicut

    CPU/main-thread postprocess
        -> source-core split-only filter
        -> detect_cells
        -> extract_cell_features

The only change is scheduling.  While frame t is running STIR-Net on CUDA,
one worker prepares frame t+1.

Correctness guard
-----------------
By default this script REQUIRES the serial Investigation-32 cache and verifies
for every tested frame that the pipelined execution produces bit-identical:

    source_instances.npy
    multicut_instances.npy
    final_instances.npy

Any mismatch aborts immediately.

The serial cache produced by Investigation 32 currently lives under its
historical SCRIPT_NAME directory:

    runs/stirnet/evaluation/
        30_kaggle_spatial_runtime_napari/<sample>/

Typical usage
-------------
Recommended first benchmark, 10 frames:

    python .\investigations\stirnet\34_kaggle_spatial_parallel_benchmark.py ^
        --frames 0-9

Five-frame quick check:

    python .\investigations\stirnet\34_kaggle_spatial_parallel_benchmark.py ^
        --frames 0-4

Force a clean rerun:

    python .\investigations\stirnet\34_kaggle_spatial_parallel_benchmark.py ^
        --frames 0-9 --overwrite

Open the generated output in Napari after benchmarking:

    python .\investigations\stirnet\34_kaggle_spatial_parallel_benchmark.py ^
        --frames 0-9 --viewer --overwrite

Try two CPU preparation workers if desired:

    python .\investigations\stirnet\34_kaggle_spatial_parallel_benchmark.py ^
        --frames 0-9 --prep-workers 2 --prefetch-depth 2 --overwrite

Notes
-----
- One preparation worker is the primary experiment.  Local preparation was
  ~12 s/frame while spatial inference was ~24 s/frame, so one worker should
  theoretically be sufficient to hide preparation.
- More preparation workers can increase GIL/CPU contention and memory pressure.
- Model loading is excluded from both serial-baseline and parallel timing.
- The summary reports real total wall time and warm steady-state throughput.
"""

import argparse
import csv
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_NAME = "34_kaggle_spatial_parallel_benchmark"
INV32_FILENAME = "32_kaggle_spatial_runtime_napari.py"
EXPECTED_REPO_SHA = "6d5e89cbb56e5d4dc051dcdf9e613b12ad018b02"

DEFAULT_SAMPLE_ID = "44b6_0113de3b"
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)
DEFAULT_TILE_SHAPE_ZYX = (32, 128, 128)
DEFAULT_TILE_OVERLAP_ZYX = (8, 32, 32)
DEFAULT_TILE_HALO_ZYX = (4, 16, 16)
DEFAULT_TILE_BATCH_SIZE = 1

KAGGLE_TOTAL_FRAMES = 400
KAGGLE_LIMIT_SECONDS = 12 * 60 * 60
KAGGLE_FRAME_BUDGET_SECONDS = KAGGLE_LIMIT_SECONDS / KAGGLE_TOTAL_FRAMES


# =============================================================================
# Repository / Investigation-32 reuse
# =============================================================================


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

    raise RuntimeError("Could not resolve repository root")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INV32 = load_module(
    ROOT / "investigations" / "stirnet" / INV32_FILENAME,
    "_stirnet_inv32_for_inv34",
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def current_git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    value = completed.stdout.strip()
    return value or None


def parse_triplet(text: str, *, cast, name: str):
    values = tuple(cast(token.strip()) for token in str(text).split(","))
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 comma-separated values")
    return values


def parse_frames(text: str, frame_count: int) -> list[int]:
    return INV32.parse_frames(text, frame_count)


def resolve_sample_zarr(sample_id: str, override: str | None) -> Path:
    return INV32.resolve_sample_zarr(sample_id, override)


def resolve_checkpoint(override: str | None) -> Path:
    return INV32.resolve_checkpoint(override)


def default_output(sample_id: str) -> Path:
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / SCRIPT_NAME
        / sample_id
    ).resolve()


def default_serial_baseline(sample_id: str) -> Path:
    # Investigation 32 currently retains SCRIPT_NAME =
    # "30_kaggle_spatial_runtime_napari", and therefore writes here.
    return INV32.default_output(sample_id)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    INV32.atomic_npy(path, array)


def atomic_json(path: Path, payload: Any) -> None:
    INV32.atomic_json(path, payload)


def count_labels(labels: np.ndarray) -> int:
    return INV32.count_labels(labels)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


# =============================================================================
# Exact source-segmentation configuration
# =============================================================================


_segmentation_config_module = import_module("src.03_segmentation.config")
SOURCE_SEGMENTATION_CONFIG = replace(
    _segmentation_config_module.DEFAULT_SEGMENTATION_CONFIG,
    enable_geometric_completion=False,
)


# =============================================================================
# Prepared-frame contract
# =============================================================================


@dataclass
class PreparedFrame:
    frame: int
    preprocessed: np.ndarray
    source_mask: np.ndarray
    source_labels: np.ndarray
    spatial: np.ndarray
    dref_um: float

    load_seconds: float
    preprocess_seconds: float
    mask_seconds: float
    source_segment_seconds: float
    build_input_seconds: float
    total_prepare_seconds: float


def prepare_frame(
    frame: int,
    *,
    sample_zarr: Path,
    spacing: tuple[float, float, float],
    helper: Any,
) -> PreparedFrame:
    """
    CPU-only preparation for one frame.

    IMPORTANT: this function must never touch CUDA.  It is safe to execute in
    the preparation thread while the main thread owns PyTorch CUDA inference.
    """
    from src.api import (
        create_binary_mask,
        preprocess_volume,
        segment_instances,
    )
    from src.io import load_timepoint

    started_total = time.perf_counter()

    t = time.perf_counter()
    raw = load_timepoint(sample_zarr, frame)
    load_seconds = time.perf_counter() - t

    t = time.perf_counter()
    preprocessed = preprocess_volume(raw)
    preprocess_seconds = time.perf_counter() - t

    t = time.perf_counter()
    source_mask = create_binary_mask(preprocessed)
    mask_seconds = time.perf_counter() - t

    t = time.perf_counter()
    source_labels = segment_instances(
        source_mask,
        config=SOURCE_SEGMENTATION_CONFIG,
    )
    source_segment_seconds = time.perf_counter() - t

    t = time.perf_counter()
    spatial, dref_um = helper.build_stage6_spatial_input(
        preprocessed,
        source_labels,
        spacing,
    )
    build_input_seconds = time.perf_counter() - t

    total_prepare_seconds = time.perf_counter() - started_total

    # The raw frame is not needed after preprocessing.  Keeping only the
    # prepared scientific inputs bounds look-ahead RAM consumption.
    del raw

    return PreparedFrame(
        frame=int(frame),
        preprocessed=preprocessed,
        source_mask=source_mask,
        source_labels=source_labels,
        spatial=spatial,
        dref_um=float(dref_um),
        load_seconds=float(load_seconds),
        preprocess_seconds=float(preprocess_seconds),
        mask_seconds=float(mask_seconds),
        source_segment_seconds=float(source_segment_seconds),
        build_input_seconds=float(build_input_seconds),
        total_prepare_seconds=float(total_prepare_seconds),
    )


# =============================================================================
# Serial baseline cache
# =============================================================================


def load_serial_baseline(
    baseline_root: Path,
    frames: list[int],
) -> tuple[dict[int, dict[str, Any]], dict[str, float]]:
    rows: dict[int, dict[str, Any]] = {}

    for frame in frames:
        frame_dir = baseline_root / f"t{frame:03d}"
        success = frame_dir / "_SUCCESS.json"
        required_arrays = (
            frame_dir / "source_instances.npy",
            frame_dir / "multicut_instances.npy",
            frame_dir / "final_instances.npy",
        )
        if not success.is_file() or not all(path.is_file() for path in required_arrays):
            raise FileNotFoundError(
                "Investigation-32 serial baseline is incomplete for "
                f"t={frame:03d}: {frame_dir}\n"
                "Run Investigation 32 for the same frames first."
            )

        row = json.loads(success.read_text(encoding="utf-8"))
        if int(row.get("frame", -1)) != int(frame):
            raise RuntimeError(
                f"Serial baseline frame metadata mismatch in {success}"
            )
        rows[int(frame)] = row

    pipeline = np.asarray(
        [float(rows[frame]["pipeline_seconds"]) for frame in frames],
        dtype=np.float64,
    )
    spatial = np.asarray(
        [float(rows[frame]["spatial_call_wall_seconds"]) for frame in frames],
        dtype=np.float64,
    )
    prep = np.asarray(
        [
            float(rows[frame]["load_seconds"])
            + float(rows[frame]["preprocess_seconds"])
            + float(rows[frame]["mask_seconds"])
            + float(rows[frame]["source_segment_seconds"])
            + float(rows[frame]["build_input_seconds"])
            for frame in frames
        ],
        dtype=np.float64,
    )

    warm_pipeline = pipeline[1:] if pipeline.size >= 3 else pipeline
    warm_spatial = spatial[1:] if spatial.size >= 3 else spatial
    warm_prep = prep[1:] if prep.size >= 3 else prep

    summary = {
        "total_pipeline_seconds": float(pipeline.sum()),
        "mean_pipeline_seconds": float(pipeline.mean()),
        "warm_mean_pipeline_seconds": float(warm_pipeline.mean()),
        "mean_spatial_seconds": float(spatial.mean()),
        "warm_mean_spatial_seconds": float(warm_spatial.mean()),
        "mean_prepare_seconds": float(prep.mean()),
        "warm_mean_prepare_seconds": float(warm_prep.mean()),
    }
    return rows, summary


def assert_exact_baseline(
    *,
    baseline_root: Path,
    frame: int,
    source_labels: np.ndarray,
    multicut_labels: np.ndarray,
    final_labels: np.ndarray,
) -> None:
    frame_dir = baseline_root / f"t{frame:03d}"

    expected_source = np.load(
        frame_dir / "source_instances.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    expected_multicut = np.load(
        frame_dir / "multicut_instances.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    expected_final = np.load(
        frame_dir / "final_instances.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    checks = (
        ("source", expected_source, source_labels),
        ("multicut", expected_multicut, multicut_labels),
        ("final", expected_final, final_labels),
    )
    for name, expected, actual in checks:
        if expected.shape != actual.shape:
            raise RuntimeError(
                f"t={frame:03d} {name} shape mismatch: "
                f"serial={expected.shape}, parallel={actual.shape}"
            )
        if not np.array_equal(expected, actual):
            changed = int(np.count_nonzero(np.asarray(expected) != actual))
            raise RuntimeError(
                f"t={frame:03d} {name} differs from serial baseline: "
                f"changed_voxels={changed:,}"
            )


# =============================================================================
# Pipelined benchmark
# =============================================================================


def run_parallel_benchmark(
    *,
    sample_id: str,
    sample_zarr: Path,
    checkpoint: Path,
    output_root: Path,
    baseline_root: Path,
    frames: list[int],
    spacing: tuple[float, float, float],
    tile_shape: tuple[int, int, int],
    tile_overlap: tuple[int, int, int],
    tile_halo: tuple[int, int, int],
    tile_batch_size: int,
    prep_workers: int,
    prefetch_depth: int,
    diagnostics: str,
) -> dict[str, Any]:
    from src.api import detect_cells, extract_cell_features

    if prep_workers < 1:
        raise ValueError("--prep-workers must be >= 1")
    if prefetch_depth < 1:
        raise ValueError("--prefetch-depth must be >= 1")
    if prefetch_depth < prep_workers:
        raise ValueError(
            "--prefetch-depth must be >= --prep-workers so every worker "
            "can receive useful work"
        )

    serial_rows, serial_summary = load_serial_baseline(
        baseline_root,
        frames,
    )

    runner = INV32.load_kaggle_runner()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    print("[init] loading exact Kaggle spatial runtime ...", flush=True)
    started_model = time.perf_counter()
    runtime, helper = runner._load_model_runtime(
        ROOT,
        checkpoint,
        device=device,
        tile_shape=tile_shape,
        tile_overlap=tile_overlap,
        tile_halo=tile_halo,
        tile_batch_size=tile_batch_size,
    )
    model_load_seconds = time.perf_counter() - started_model

    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output_root / "manifest.json",
        {
            "script": SCRIPT_NAME,
            "repository_head": current_git_sha(),
            "expected_repository_head": EXPECTED_REPO_SHA,
            "sample_id": sample_id,
            "sample_zarr": str(sample_zarr),
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(runtime.checkpoint_step),
            "serial_baseline_root": str(baseline_root),
            "frames": frames,
            "spacing_zyx_um": list(spacing),
            "tile_shape_zyx": list(tile_shape),
            "tile_overlap_zyx": list(tile_overlap),
            "tile_halo_zyx": list(tile_halo),
            "tile_batch_size": int(tile_batch_size),
            "prep_workers": int(prep_workers),
            "prefetch_depth": int(prefetch_depth),
            "diagnostics": diagnostics,
            "model_load_seconds": float(model_load_seconds),
        },
    )

    print("=" * 126, flush=True)
    print("INVESTIGATION 34 — CPU PREPARATION / GPU INFERENCE PIPELINE", flush=True)
    print("=" * 126, flush=True)
    print(f"sample             : {sample_id}", flush=True)
    print(f"frames             : {frames[0]}..{frames[-1]} ({len(frames)} selected)", flush=True)
    print(f"serial baseline    : {baseline_root}", flush=True)
    print(f"output             : {output_root}", flush=True)
    print(f"checkpoint step    : {runtime.checkpoint_step}", flush=True)
    print(f"GPU                : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"prep workers       : {prep_workers}", flush=True)
    print(f"prefetch depth     : {prefetch_depth}", flush=True)
    print(f"model load         : {model_load_seconds:.2f}s (excluded from benchmark)", flush=True)
    print(
        f"serial warm mean   : {serial_summary['warm_mean_pipeline_seconds']:.2f}s/frame",
        flush=True,
    )
    print(
        f"serial warm prep   : {serial_summary['warm_mean_prepare_seconds']:.2f}s/frame",
        flush=True,
    )
    print(
        f"serial warm spatial: {serial_summary['warm_mean_spatial_seconds']:.2f}s/frame",
        flush=True,
    )
    print("=" * 126, flush=True)

    rows: list[dict[str, Any]] = []
    completion_times: list[float] = []

    pending: dict[int, Future[PreparedFrame]] = {}
    next_submit_index = 0

    def fill_prefetch(executor: ThreadPoolExecutor) -> None:
        nonlocal next_submit_index
        while (
            len(pending) < prefetch_depth
            and next_submit_index < len(frames)
        ):
            frame_to_submit = int(frames[next_submit_index])
            if frame_to_submit not in pending:
                pending[frame_to_submit] = executor.submit(
                    prepare_frame,
                    frame_to_submit,
                    sample_zarr=sample_zarr,
                    spacing=spacing,
                    helper=helper,
                )
            next_submit_index += 1

    benchmark_started = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=prep_workers,
        thread_name_prefix="stirnet-prep",
    ) as executor:
        fill_prefetch(executor)

        for index, frame in enumerate(frames, start=1):
            frame = int(frame)
            if frame not in pending:
                raise RuntimeError(
                    f"Internal prefetch error: t={frame:03d} was not submitted"
                )

            wait_started = time.perf_counter()
            prepared = pending.pop(frame).result()
            prep_wait_seconds = time.perf_counter() - wait_started

            if prepared.frame != frame:
                raise RuntimeError(
                    f"Prepared frame mismatch: expected {frame}, got {prepared.frame}"
                )

            # Start the next CPU preparation BEFORE launching CUDA for this frame.
            fill_prefetch(executor)

            frame_main_started = time.perf_counter()

            t = time.perf_counter()
            (
                result,
                spatial_gpu,
                amp_name,
                reported_inference_seconds,
                peak_gib,
            ) = helper.run_tiled_spatial(
                runtime.model,
                prepared.spatial,
                spacing,
                prepared.dref_um,
                device=runtime.device,
                inference_cfg=runtime.inference_cfg,
            )
            spatial_call_wall_seconds = time.perf_counter() - t

            t = time.perf_counter()
            before = runner._tensor_numpy(
                result.spatial_partition.labels[0],
                np.int32,
            )
            watershed = runner._tensor_numpy(
                result.supervoxel_labels[0],
                np.int32,
            )
            separator_probability = runner._tensor_numpy(
                result.dense.geometry.probabilities()["separator"][0, 0],
                np.float32,
            )
            unpack_seconds = time.perf_counter() - t

            t = time.perf_counter()
            final_labels, split_diag = runner._apply_source_core_split_only(
                before,
                watershed,
                separator_probability,
                prepared.source_mask,
                prepared.source_labels,
                spacing,
                prepared.dref_um,
            )
            split_filter_seconds = time.perf_counter() - t

            t = time.perf_counter()
            cells = detect_cells(final_labels)
            detect_cells_seconds = time.perf_counter() - t

            t = time.perf_counter()
            cells = extract_cell_features(
                cells,
                final_labels,
                prepared.preprocessed,
            )
            extract_features_seconds = time.perf_counter() - t

            if cells.empty:
                raise RuntimeError(
                    f"{sample_id} t={frame}: no final cells"
                )

            main_compute_seconds = time.perf_counter() - frame_main_started

            # Strong acceptance condition: scheduling must not alter the
            # scientific result by even one label voxel.
            assert_exact_baseline(
                baseline_root=baseline_root,
                frame=frame,
                source_labels=prepared.source_labels,
                multicut_labels=before,
                final_labels=final_labels,
            )

            t = time.perf_counter()
            frame_dir = output_root / f"t{frame:03d}"
            frame_dir.mkdir(parents=True, exist_ok=True)

            atomic_npy(
                frame_dir / "source_instances.npy",
                prepared.source_labels.astype(np.int32, copy=False),
            )
            atomic_npy(
                frame_dir / "multicut_instances.npy",
                before.astype(np.int32, copy=False),
            )
            atomic_npy(
                frame_dir / "final_instances.npy",
                final_labels.astype(np.int32, copy=False),
            )
            cells.to_csv(frame_dir / "cells.csv", index=False)

            if diagnostics == "full":
                atomic_npy(
                    frame_dir / "preprocessed.npy",
                    np.asarray(prepared.preprocessed, dtype=np.float16),
                )
                atomic_npy(
                    frame_dir / "source_mask.npy",
                    np.asarray(prepared.source_mask > 0, dtype=np.uint8),
                )
                atomic_npy(
                    frame_dir / "watershed_supervoxels.npy",
                    watershed.astype(np.int32, copy=False),
                )
                atomic_npy(
                    frame_dir / "separator_probability.npy",
                    np.asarray(separator_probability, dtype=np.float16),
                )

            save_seconds = time.perf_counter() - t

            # Real steady-state throughput is measured at the end of each
            # completed frame, including the tiny required save work.
            completed_at = time.perf_counter()
            completion_times.append(completed_at)

            prep_hidden_seconds = max(
                prepared.total_prepare_seconds - prep_wait_seconds,
                0.0,
            )
            prep_hidden_fraction = (
                prep_hidden_seconds / prepared.total_prepare_seconds
                if prepared.total_prepare_seconds > 0
                else 0.0
            )

            serial = serial_rows[frame]
            row = {
                "frame": frame,
                "source_instances": count_labels(prepared.source_labels),
                "multicut_instances": count_labels(before),
                "final_instances": count_labels(final_labels),
                "split_candidates": int(split_diag["candidate_count"]),
                "splits_applied": int(split_diag["applied_count"]),
                "prepare_total_seconds": float(prepared.total_prepare_seconds),
                "prepare_wait_seconds": float(prep_wait_seconds),
                "prepare_hidden_seconds": float(prep_hidden_seconds),
                "prepare_hidden_fraction": float(prep_hidden_fraction),
                "load_seconds": float(prepared.load_seconds),
                "preprocess_seconds": float(prepared.preprocess_seconds),
                "mask_seconds": float(prepared.mask_seconds),
                "source_segment_seconds": float(prepared.source_segment_seconds),
                "build_input_seconds": float(prepared.build_input_seconds),
                "spatial_call_wall_seconds": float(spatial_call_wall_seconds),
                "reported_inference_seconds": float(reported_inference_seconds),
                "unpack_seconds": float(unpack_seconds),
                "split_filter_seconds": float(split_filter_seconds),
                "detect_cells_seconds": float(detect_cells_seconds),
                "extract_features_seconds": float(extract_features_seconds),
                "main_compute_seconds": float(main_compute_seconds),
                "save_seconds": float(save_seconds),
                "serial_pipeline_seconds": float(serial["pipeline_seconds"]),
                "serial_spatial_seconds": float(
                    serial["spatial_call_wall_seconds"]
                ),
                "exact_source": True,
                "exact_multicut": True,
                "exact_final": True,
                "amp_dtype": str(amp_name),
                "peak_allocated_vram_gib": float(peak_gib),
            }
            rows.append(row)
            atomic_json(frame_dir / "_SUCCESS.json", row)
            save_csv(output_root / "frame_timings.csv", rows)

            interval = (
                completion_times[-1] - completion_times[-2]
                if len(completion_times) >= 2
                else completed_at - benchmark_started
            )

            print(
                f"[{index}/{len(frames)} t={frame:03d}] "
                f"prep={prepared.total_prepare_seconds:5.1f}s "
                f"WAIT={prep_wait_seconds:5.2f}s "
                f"hidden={100.0*prep_hidden_fraction:5.1f}% | "
                f"spatial={spatial_call_wall_seconds:5.1f}s "
                f"post={unpack_seconds + split_filter_seconds + detect_cells_seconds + extract_features_seconds:4.1f}s "
                f"save={save_seconds:4.1f}s | "
                f"interval={interval:5.1f}s | "
                f"EXACT=yes VRAM={peak_gib:.2f}GiB",
                flush=True,
            )

            del result, spatial_gpu, separator_probability, watershed
            del before, final_labels, cells, prepared
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    benchmark_total_seconds = time.perf_counter() - benchmark_started

    if not rows:
        raise RuntimeError("No frames were benchmarked")

    # Warm throughput: the first frame necessarily pays the initial CPU
    # preparation.  For a 400-frame hidden-test run the intervals after the
    # first completion are much more representative.
    if len(completion_times) >= 2:
        warm_parallel_seconds = (
            completion_times[-1] - completion_times[0]
        ) / (len(completion_times) - 1)
    else:
        warm_parallel_seconds = benchmark_total_seconds

    warm_rows = rows[1:] if len(rows) >= 3 else rows

    mean_prepare = float(
        np.mean([row["prepare_total_seconds"] for row in rows])
    )
    warm_mean_prepare = float(
        np.mean([row["prepare_total_seconds"] for row in warm_rows])
    )
    warm_mean_wait = float(
        np.mean([row["prepare_wait_seconds"] for row in warm_rows])
    )
    warm_hidden_fraction = float(
        1.0
        - (
            sum(row["prepare_wait_seconds"] for row in warm_rows)
            / max(
                sum(row["prepare_total_seconds"] for row in warm_rows),
                1e-12,
            )
        )
    )

    mean_parallel_spatial = float(
        np.mean([row["spatial_call_wall_seconds"] for row in rows])
    )
    warm_parallel_spatial = float(
        np.mean([row["spatial_call_wall_seconds"] for row in warm_rows])
    )

    serial_warm = float(serial_summary["warm_mean_pipeline_seconds"])
    serial_total = float(serial_summary["total_pipeline_seconds"])

    speedup_total = serial_total / benchmark_total_seconds
    speedup_warm = serial_warm / warm_parallel_seconds
    saved_per_frame_warm = serial_warm - warm_parallel_seconds
    saved_400_hours = saved_per_frame_warm * KAGGLE_TOTAL_FRAMES / 3600.0
    parallel_400_hours = (
        warm_parallel_seconds * KAGGLE_TOTAL_FRAMES / 3600.0
    )
    serial_400_hours = (
        serial_warm * KAGGLE_TOTAL_FRAMES / 3600.0
    )

    spatial_slowdown_ratio = (
        warm_parallel_spatial
        / max(float(serial_summary["warm_mean_spatial_seconds"]), 1e-12)
    )

    summary = {
        "sample_id": sample_id,
        "frames": frames,
        "frame_count": len(frames),
        "prep_workers": prep_workers,
        "prefetch_depth": prefetch_depth,
        "model_load_seconds_excluded": float(model_load_seconds),
        "serial_baseline": serial_summary,
        "parallel_total_wall_seconds": float(benchmark_total_seconds),
        "parallel_mean_wall_seconds_including_initial_fill": float(
            benchmark_total_seconds / len(frames)
        ),
        "parallel_warm_throughput_seconds_per_frame": float(
            warm_parallel_seconds
        ),
        "parallel_mean_prepare_seconds": mean_prepare,
        "parallel_warm_mean_prepare_seconds": warm_mean_prepare,
        "parallel_warm_mean_prepare_wait_seconds": warm_mean_wait,
        "parallel_warm_prepare_hidden_fraction": warm_hidden_fraction,
        "parallel_mean_spatial_seconds": mean_parallel_spatial,
        "parallel_warm_mean_spatial_seconds": warm_parallel_spatial,
        "parallel_vs_serial_spatial_time_ratio": spatial_slowdown_ratio,
        "speedup_total_selected_frames": float(speedup_total),
        "speedup_warm": float(speedup_warm),
        "warm_seconds_saved_per_frame": float(saved_per_frame_warm),
        "serial_estimated_400_frame_hours": float(serial_400_hours),
        "parallel_estimated_400_frame_hours": float(parallel_400_hours),
        "estimated_400_frame_hours_saved": float(saved_400_hours),
        "kaggle_frame_budget_seconds_before_tracking": float(
            KAGGLE_FRAME_BUDGET_SECONDS
        ),
        "all_source_arrays_exact": True,
        "all_multicut_arrays_exact": True,
        "all_final_arrays_exact": True,
    }
    atomic_json(output_root / "summary.json", summary)

    print("", flush=True)
    print("=" * 126, flush=True)
    print("PARALLELIZATION RESULT", flush=True)
    print("=" * 126, flush=True)
    print(
        f"serial baseline warm         : {serial_warm:8.2f} s/frame",
        flush=True,
    )
    print(
        f"parallel warm throughput     : {warm_parallel_seconds:8.2f} s/frame",
        flush=True,
    )
    print(
        f"warm speedup                 : {speedup_warm:8.3f} x",
        flush=True,
    )
    print(
        f"warm saving                  : {saved_per_frame_warm:8.2f} s/frame",
        flush=True,
    )
    print(
        f"CPU preparation hidden       : {100.0 * warm_hidden_fraction:8.2f} %",
        flush=True,
    )
    print(
        f"warm prep wait               : {warm_mean_wait:8.3f} s/frame",
        flush=True,
    )
    print(
        f"serial spatial mean          : "
        f"{serial_summary['warm_mean_spatial_seconds']:8.2f} s/frame",
        flush=True,
    )
    print(
        f"parallel spatial mean        : {warm_parallel_spatial:8.2f} s/frame",
        flush=True,
    )
    print(
        f"spatial contention ratio     : {spatial_slowdown_ratio:8.3f} x",
        flush=True,
    )
    print(
        f"serial 400-frame estimate    : {serial_400_hours:8.2f} h",
        flush=True,
    )
    print(
        f"parallel 400-frame estimate  : {parallel_400_hours:8.2f} h",
        flush=True,
    )
    print(
        f"estimated time saved / 400   : {saved_400_hours:8.2f} h",
        flush=True,
    )
    print(
        "scientific equivalence       : EXACT source + multicut + final",
        flush=True,
    )
    print(f"results                      : {output_root}", flush=True)
    print("=" * 126, flush=True)

    return summary


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--sample-zarr", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--frames", default="0-4")
    parser.add_argument("--output", default=None)
    parser.add_argument("--serial-baseline", default=None)

    parser.add_argument("--spacing", default="1.625,0.40625,0.40625")
    parser.add_argument("--tile-shape", default="32,128,128")
    parser.add_argument("--tile-overlap", default="8,32,32")
    parser.add_argument("--tile-halo", default="4,16,16")
    parser.add_argument("--tile-batch-size", type=int, default=1)

    parser.add_argument(
        "--prep-workers",
        type=int,
        default=1,
        help="CPU frame-preparation threads. Start with 1.",
    )
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=1,
        help="Number of future prepared frames allowed in flight.",
    )
    parser.add_argument(
        "--diagnostics",
        choices=("basic", "full"),
        default="basic",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete any previous Investigation-34 output before timing.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the generated source/multicut/final arrays in Napari afterward.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    head = current_git_sha()
    print(f"Repository HEAD: {head or 'unknown'}", flush=True)
    print(f"Benchmark basis: {EXPECTED_REPO_SHA}", flush=True)
    if head is not None and head != EXPECTED_REPO_SHA:
        print(
            "WARNING: repository HEAD differs from the commit this Investigation "
            "34 file was generated against.",
            flush=True,
        )

    spacing = tuple(
        float(v)
        for v in parse_triplet(args.spacing, cast=float, name="--spacing")
    )
    tile_shape = tuple(
        int(v)
        for v in parse_triplet(args.tile_shape, cast=int, name="--tile-shape")
    )
    tile_overlap = tuple(
        int(v)
        for v in parse_triplet(
            args.tile_overlap,
            cast=int,
            name="--tile-overlap",
        )
    )
    tile_halo = tuple(
        int(v)
        for v in parse_triplet(args.tile_halo, cast=int, name="--tile-halo")
    )

    sample_zarr = resolve_sample_zarr(args.sample_id, args.sample_zarr)
    checkpoint = resolve_checkpoint(args.checkpoint)

    from src.io import open_sample

    image = open_sample(sample_zarr)
    if len(image.shape) != 4:
        raise ValueError(f"Expected [T,Z,Y,X], got {image.shape}")
    frames = parse_frames(args.frames, int(image.shape[0]))

    output_root = (
        default_output(args.sample_id)
        if args.output is None
        else resolve(args.output)
    )
    baseline_root = (
        default_serial_baseline(args.sample_id)
        if args.serial_baseline is None
        else resolve(args.serial_baseline)
    )

    if output_root.exists():
        if not args.overwrite:
            completed = sorted(output_root.glob("t*/_SUCCESS.json"))
            if completed:
                raise RuntimeError(
                    f"Previous benchmark output exists at {output_root}. "
                    "Use --overwrite for a clean timing run."
                )
        else:
            shutil.rmtree(output_root)

    summary = run_parallel_benchmark(
        sample_id=args.sample_id,
        sample_zarr=sample_zarr,
        checkpoint=checkpoint,
        output_root=output_root,
        baseline_root=baseline_root,
        frames=frames,
        spacing=spacing,
        tile_shape=tile_shape,
        tile_overlap=tile_overlap,
        tile_halo=tile_halo,
        tile_batch_size=int(args.tile_batch_size),
        prep_workers=int(args.prep_workers),
        prefetch_depth=int(args.prefetch_depth),
        diagnostics=args.diagnostics,
    )

    if args.viewer:
        INV32.open_viewer(
            sample_zarr=sample_zarr,
            output_root=output_root,
            spacing=spacing,
        )

    # Explicit nonzero failure would already have raised.  Reaching here means
    # every selected frame was exactly equivalent to the serial baseline.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
