from __future__ import annotations

"""
Investigation 30 — local profiling + Napari inspection of the frozen Kaggle
spatial STIR-Net inference path.

This intentionally reuses kaggle/run_submission.py internals so the scientific
path matches the Kaggle baseline:

    raw BioHub Zarr
      -> canonical preprocessing
      -> binary mask + source watershed
      -> tiled STIR-Net spatial inference
      -> signed multicut
      -> source-instance anchored split-only postfilter
      -> detection + cell feature extraction

Tracking is intentionally NOT run here.  This script answers two questions first:
1. Where is per-frame runtime being spent?
2. Do source -> multicut -> final instances look biologically reasonable?

Examples
--------
Profile 5 frames with full visual diagnostics and open Napari:

    python investigations/stirnet/32_kaggle_spatial_runtime_napari.py ^
        --frames 0-4 --diagnostics full

Run all 100 frames with lighter disk output, then open Napari:

    python investigations/stirnet/32_kaggle_spatial_runtime_napari.py ^
        --frames all --diagnostics basic

Profile only, no viewer:

    python investigations/stirnet/32_kaggle_spatial_runtime_napari.py ^
        --frames 0-4 --diagnostics full --no-viewer

Reopen a completed/partial run without recomputing:

    python investigations/stirnet/32_kaggle_spatial_runtime_napari.py ^
        --viewer-only
"""

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


SCRIPT_NAME = "30_kaggle_spatial_runtime_napari"
DEFAULT_SAMPLE_ID = "44b6_0113de3b"
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)
DEFAULT_TILE_SHAPE_ZYX = (32, 128, 128)
DEFAULT_TILE_OVERLAP_ZYX = (8, 32, 32)
DEFAULT_TILE_HALO_ZYX = (4, 16, 16)
DEFAULT_TILE_BATCH_SIZE = 1
KAGGLE_TOTAL_FRAMES = 400
KAGGLE_LIMIT_SECONDS = 12 * 60 * 60
KAGGLE_FRAME_BUDGET_SECONDS = KAGGLE_LIMIT_SECONDS / KAGGLE_TOTAL_FRAMES


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
    if (
        (cwd / "learned").is_dir()
        and (cwd / "src").is_dir()
        and (cwd / "pyproject.toml").is_file()
    ):
        return cwd
    raise RuntimeError("Could not resolve repository root")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def parse_triplet(text: str, *, cast, name: str):
    values = tuple(cast(token.strip()) for token in str(text).split(","))
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 comma-separated values")
    return values


def parse_frames(text: str, frame_count: int) -> list[int]:
    token = str(text).strip().lower()
    if token in {"all", "*"}:
        return list(range(frame_count))

    selected: set[int] = set()
    for piece in token.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            left, right = piece.split("-", 1)
            start = int(left)
            stop = int(right)
            if stop < start:
                raise ValueError(f"Invalid frame range {piece!r}")
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(piece))

    frames = sorted(selected)
    if not frames:
        raise ValueError("No frames selected")
    bad = [frame for frame in frames if frame < 0 or frame >= frame_count]
    if bad:
        raise IndexError(f"Frames outside [0,{frame_count - 1}]: {bad}")
    return frames


def _load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_kaggle_runner():
    path = ROOT / "kaggle" / "run_submission.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Copy the current frozen Kaggle "
            "run_submission.py into repo_root/kaggle first."
        )
    return _load_module(path, "_investigation30_kaggle_runner")


def resolve_sample_zarr(sample_id: str, override: str | None) -> Path:
    if override:
        path = resolve(override)
    else:
        from src.io import PipelinePaths
        path = PipelinePaths.discover(ROOT).sample_zarr(sample_id)
    if path.name == "0":
        path = path.parent
    if not (path / "0").exists():
        raise FileNotFoundError(f"BioHub Zarr array not found: {path / '0'}")
    return path.resolve()


def resolve_checkpoint(override: str | None) -> Path:
    if override:
        path = resolve(override)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    recovery = (
        ROOT
        / "runs"
        / "stirnet"
        / "investigations"
        / "19_morphology_rag_v2_headroom_training"
        / "recovery"
        / "drosophila_12_morphology_rag_v2_headroom_h100"
    )
    candidates = (
        recovery / "best_checkpoint.pt",
        recovery / "checkpoint_step_000600.pt",
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()

    fallback = sorted(recovery.glob("checkpoint_step_*.pt"))
    if fallback:
        return fallback[-1].resolve()

    raise FileNotFoundError(
        "Could not resolve morphology-v2 h100 checkpoint. "
        f"Checked {recovery}. Use --checkpoint."
    )


def default_output(sample_id: str) -> Path:
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / SCRIPT_NAME
        / sample_id
    ).resolve()


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(tmp, np.asarray(array), allow_pickle=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def count_labels(labels: np.ndarray) -> int:
    values = np.unique(labels)
    return int(np.count_nonzero(values > 0))


def stage_timer():
    return time.perf_counter()


def elapsed(start: float) -> float:
    return float(time.perf_counter() - start)


def save_timings_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def pipeline_profile(
    *,
    sample_id: str,
    sample_zarr: Path,
    checkpoint: Path,
    output_root: Path,
    frames: list[int],
    spacing: tuple[float, float, float],
    tile_shape: tuple[int, int, int],
    tile_overlap: tuple[int, int, int],
    tile_halo: tuple[int, int, int],
    tile_batch_size: int,
    diagnostics: str,
    overwrite: bool,
) -> None:
    from src.api import (
        create_binary_mask,
        detect_cells,
        extract_cell_features,
        preprocess_volume,
        segment_instances,
    )
    from src.io import load_timepoint

    runner = load_kaggle_runner()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local profiler")

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
    model_load_seconds = elapsed(started_model)

    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output_root / "manifest.json",
        {
            "script": SCRIPT_NAME,
            "sample_id": sample_id,
            "sample_zarr": str(sample_zarr),
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(runtime.checkpoint_step),
            "spacing_zyx_um": list(spacing),
            "tile_shape_zyx": list(tile_shape),
            "tile_overlap_zyx": list(tile_overlap),
            "tile_halo_zyx": list(tile_halo),
            "tile_batch_size": int(tile_batch_size),
            "diagnostics": diagnostics,
            "requested_frames": frames,
            "model_load_seconds": model_load_seconds,
            "kaggle_total_frames": KAGGLE_TOTAL_FRAMES,
            "kaggle_limit_seconds": KAGGLE_LIMIT_SECONDS,
            "kaggle_frame_budget_seconds_ignoring_tracking": KAGGLE_FRAME_BUDGET_SECONDS,
        },
    )

    timing_rows: list[dict[str, Any]] = []
    run_started = time.perf_counter()

    print("=" * 118, flush=True)
    print("LOCAL KAGGLE-SPATIAL PROFILER", flush=True)
    print("=" * 118, flush=True)
    print(f"sample          : {sample_id}", flush=True)
    print(f"zarr            : {sample_zarr}", flush=True)
    print(f"checkpoint      : {checkpoint}", flush=True)
    print(f"checkpoint step : {runtime.checkpoint_step}", flush=True)
    print(f"frames          : {frames[0]}..{frames[-1]} ({len(frames)} selected)", flush=True)
    print(f"diagnostics     : {diagnostics}", flush=True)
    print(f"GPU             : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"model load      : {model_load_seconds:.2f}s", flush=True)
    print(f"12h budget      : {KAGGLE_FRAME_BUDGET_SECONDS:.2f}s/frame BEFORE tracking", flush=True)
    print("=" * 118, flush=True)

    for index, frame in enumerate(frames, start=1):
        frame_dir = output_root / f"t{frame:03d}"
        success_path = frame_dir / "_SUCCESS.json"
        if success_path.is_file() and not overwrite:
            row = json.loads(success_path.read_text(encoding="utf-8"))
            timing_rows.append(row)
            print(f"[{index}/{len(frames)} t={frame:03d}] cached", flush=True)
            continue

        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_started = time.perf_counter()

        t = stage_timer()
        raw = load_timepoint(sample_zarr, frame)
        load_seconds = elapsed(t)

        t = stage_timer()
        preprocessed = preprocess_volume(raw)
        preprocess_seconds = elapsed(t)

        t = stage_timer()
        source_mask = create_binary_mask(preprocessed)
        mask_seconds = elapsed(t)

        t = stage_timer()
        source_labels = segment_instances(source_mask)
        source_segment_seconds = elapsed(t)

        t = stage_timer()
        spatial, dref_um = helper.build_stage6_spatial_input(
            preprocessed,
            source_labels,
            spacing,
        )
        build_input_seconds = elapsed(t)

        t = stage_timer()
        result, spatial_gpu, amp_name, reported_inference_seconds, peak_gib = (
            helper.run_tiled_spatial(
                runtime.model,
                spatial,
                spacing,
                dref_um,
                device=runtime.device,
                inference_cfg=runtime.inference_cfg,
            )
        )
        spatial_call_wall_seconds = elapsed(t)

        t = stage_timer()
        before = runner._tensor_numpy(result.spatial_partition.labels[0], np.int32)
        watershed = runner._tensor_numpy(result.supervoxel_labels[0], np.int32)
        separator_probability = runner._tensor_numpy(
            result.dense.geometry.probabilities()["separator"][0, 0],
            np.float32,
        )
        unpack_seconds = elapsed(t)

        t = stage_timer()
        final_labels, split_diag = runner._apply_source_core_split_only(
            before,
            watershed,
            separator_probability,
            source_mask,
            source_labels,
            spacing,
            dref_um,
        )
        split_filter_seconds = elapsed(t)

        t = stage_timer()
        cells = detect_cells(final_labels)
        detect_cells_seconds = elapsed(t)

        t = stage_timer()
        cells = extract_cell_features(cells, final_labels, preprocessed)
        extract_features_seconds = elapsed(t)

        if cells.empty:
            raise RuntimeError(f"{sample_id} t={frame}: no final cells")

        # This is the comparable Kaggle per-frame compute path.  Extra visual
        # diagnostic writes below are deliberately excluded.
        pipeline_seconds = elapsed(frame_started)

        t = stage_timer()
        atomic_npy(frame_dir / "source_instances.npy", source_labels.astype(np.int32, copy=False))
        atomic_npy(frame_dir / "multicut_instances.npy", before.astype(np.int32, copy=False))
        atomic_npy(frame_dir / "final_instances.npy", final_labels.astype(np.int32, copy=False))
        cells.to_csv(frame_dir / "cells.csv", index=False)

        if diagnostics == "full":
            atomic_npy(
                frame_dir / "preprocessed.npy",
                np.asarray(preprocessed, dtype=np.float16),
            )
            atomic_npy(
                frame_dir / "source_mask.npy",
                np.asarray(source_mask > 0, dtype=np.uint8),
            )
            atomic_npy(
                frame_dir / "watershed_supervoxels.npy",
                watershed.astype(np.int32, copy=False),
            )
            atomic_npy(
                frame_dir / "separator_probability.npy",
                np.asarray(separator_probability, dtype=np.float16),
            )
        diagnostic_save_seconds = elapsed(t)

        row = {
            "frame": int(frame),
            "source_instances": count_labels(source_labels),
            "multicut_instances": count_labels(before),
            "final_instances": count_labels(final_labels),
            "split_candidates": int(split_diag["candidate_count"]),
            "splits_applied": int(split_diag["applied_count"]),
            "load_seconds": load_seconds,
            "preprocess_seconds": preprocess_seconds,
            "mask_seconds": mask_seconds,
            "source_segment_seconds": source_segment_seconds,
            "build_input_seconds": build_input_seconds,
            "spatial_call_wall_seconds": spatial_call_wall_seconds,
            "reported_inference_seconds": float(reported_inference_seconds),
            "unpack_seconds": unpack_seconds,
            "split_filter_seconds": split_filter_seconds,
            "detect_cells_seconds": detect_cells_seconds,
            "extract_features_seconds": extract_features_seconds,
            "pipeline_seconds": pipeline_seconds,
            "diagnostic_save_seconds": diagnostic_save_seconds,
            "amp_dtype": str(amp_name),
            "peak_allocated_vram_gib": float(peak_gib),
        }
        atomic_json(success_path, row)
        timing_rows.append(row)
        save_timings_csv(output_root / "frame_timings.csv", timing_rows)

        completed = len(timing_rows)
        mean_pipeline = float(np.mean([float(r["pipeline_seconds"]) for r in timing_rows]))
        estimate_400_h = mean_pipeline * KAGGLE_TOTAL_FRAMES / 3600.0
        remaining_local = mean_pipeline * max(len(frames) - completed, 0)

        print(
            f"[{index}/{len(frames)} t={frame:03d}] "
            f"source={row['source_instances']} -> multicut={row['multicut_instances']} "
            f"-> final={row['final_instances']} splits={row['splits_applied']} | "
            f"load={load_seconds:.1f}s prep={preprocess_seconds:.1f}s "
            f"mask={mask_seconds:.1f}s srcseg={source_segment_seconds:.1f}s "
            f"input={build_input_seconds:.1f}s spatial={spatial_call_wall_seconds:.1f}s "
            f"split={split_filter_seconds:.1f}s feat={extract_features_seconds:.1f}s | "
            f"PIPE={pipeline_seconds:.1f}s save={diagnostic_save_seconds:.1f}s | "
            f"mean={mean_pipeline:.1f}s est400={estimate_400_h:.2f}h "
            f"localETA={remaining_local/60:.1f}m VRAM={peak_gib:.2f}GiB",
            flush=True,
        )

        del result, spatial_gpu, spatial
        del separator_probability, watershed, before, final_labels
        del raw, preprocessed, source_mask, source_labels, cells
        torch.cuda.empty_cache()
        gc.collect()

    if not timing_rows:
        raise RuntimeError("No timing rows produced")

    pipeline_values = np.asarray(
        [float(row["pipeline_seconds"]) for row in timing_rows],
        dtype=np.float64,
    )
    spatial_values = np.asarray(
        [float(row["spatial_call_wall_seconds"]) for row in timing_rows],
        dtype=np.float64,
    )

    # Report a warm estimate too, excluding the first measured frame when
    # enough frames are available.
    warm_values = pipeline_values[1:] if pipeline_values.size >= 3 else pipeline_values

    summary = {
        "sample_id": sample_id,
        "frames_completed": len(timing_rows),
        "frames": [int(row["frame"]) for row in timing_rows],
        "mean_pipeline_seconds": float(pipeline_values.mean()),
        "median_pipeline_seconds": float(np.median(pipeline_values)),
        "mean_spatial_call_seconds": float(spatial_values.mean()),
        "warm_mean_pipeline_seconds": float(warm_values.mean()),
        "warm_estimated_400_frame_hours_spatial_stage_only": float(
            warm_values.mean() * KAGGLE_TOTAL_FRAMES / 3600.0
        ),
        "required_mean_seconds_per_frame_for_12h_ignoring_tracking": float(
            KAGGLE_FRAME_BUDGET_SECONDS
        ),
        "run_wall_seconds": elapsed(run_started),
        "model_load_seconds": model_load_seconds,
    }
    atomic_json(output_root / "summary.json", summary)

    print("", flush=True)
    print("=" * 118, flush=True)
    print("PROFILE SUMMARY", flush=True)
    print("=" * 118, flush=True)
    print(f"mean pipeline/frame       : {summary['mean_pipeline_seconds']:.2f}s", flush=True)
    print(f"median pipeline/frame     : {summary['median_pipeline_seconds']:.2f}s", flush=True)
    print(f"warm mean/frame           : {summary['warm_mean_pipeline_seconds']:.2f}s", flush=True)
    print(f"mean spatial call/frame   : {summary['mean_spatial_call_seconds']:.2f}s", flush=True)
    print(
        "400-frame extrapolation   : "
        f"{summary['warm_estimated_400_frame_hours_spatial_stage_only']:.2f}h "
        "(before tracking)",
        flush=True,
    )
    print(
        "required for 12h          : "
        f"< {KAGGLE_FRAME_BUDGET_SECONDS:.2f}s/frame BEFORE tracking",
        flush=True,
    )
    print(f"results                   : {output_root}", flush=True)
    print("=" * 118, flush=True)


def _load_memmap(path: Path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def stack_npy(paths: Iterable[Path], *, name: str):
    paths = list(paths)
    arrays = [_load_memmap(path) for path in paths]
    if not arrays:
        raise ValueError(f"No {name} arrays")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError(f"{name} shape mismatch")

    try:
        import dask.array as da
        chunks = (
            min(8, shape[0]),
            min(128, shape[1]),
            min(128, shape[2]),
        )
        return da.stack(
            [da.from_array(array, chunks=chunks, asarray=False) for array in arrays],
            axis=0,
        )
    except ImportError:
        print(f"[viewer] Dask unavailable; eagerly stacking {name}.", flush=True)
        return np.stack([np.asarray(array) for array in arrays], axis=0)


def available_frames(output_root: Path) -> list[int]:
    frames = []
    for path in output_root.glob("t[0-9][0-9][0-9]"):
        if path.is_dir() and (path / "_SUCCESS.json").is_file():
            frames.append(int(path.name[1:]))
    return sorted(frames)


def open_viewer(
    *,
    sample_zarr: Path,
    output_root: Path,
    spacing: tuple[float, float, float],
) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError("Napari is required for viewing") from exc

    frames = available_frames(output_root)
    if not frames:
        raise FileNotFoundError(f"No completed frames in {output_root}")

    source = stack_npy(
        [output_root / f"t{f:03d}" / "source_instances.npy" for f in frames],
        name="source instances",
    )
    multicut = stack_npy(
        [output_root / f"t{f:03d}" / "multicut_instances.npy" for f in frames],
        name="multicut instances",
    )
    final = stack_npy(
        [output_root / f"t{f:03d}" / "final_instances.npy" for f in frames],
        name="final instances",
    )

    try:
        import dask.array as da
        raw_all = da.from_zarr(str(sample_zarr / "0"))
        raw = raw_all[frames]
        raw_backend = "dask-zarr"
    except ImportError:
        from src.io import open_sample
        sample = open_sample(sample_zarr)
        raw = np.stack([np.asarray(sample[f]) for f in frames], axis=0)
        raw_backend = "numpy"

    scale_4d = (1.0, *spacing)
    viewer = napari.Viewer(title=f"{SCRIPT_NAME} — {sample_zarr.name}")

    viewer.add_image(
        raw,
        name=f"Raw ({raw_backend})",
        scale=scale_4d,
        blending="additive",
    )
    viewer.add_labels(
        source,
        name="Source instances",
        scale=scale_4d,
        opacity=1.0,
        visible=False,
    )
    viewer.add_labels(
        multicut,
        name="Multicut instances",
        scale=scale_4d,
        opacity=1.0,
        visible=False,
    )
    viewer.add_labels(
        final,
        name="Final split-only instances",
        scale=scale_4d,
        opacity=1.0,
        visible=True,
    )

    full = all(
        (output_root / f"t{f:03d}" / "separator_probability.npy").is_file()
        for f in frames
    )
    if full:
        preprocessed = stack_npy(
            [output_root / f"t{f:03d}" / "preprocessed.npy" for f in frames],
            name="preprocessed",
        )
        source_mask = stack_npy(
            [output_root / f"t{f:03d}" / "source_mask.npy" for f in frames],
            name="source mask",
        )
        watershed = stack_npy(
            [output_root / f"t{f:03d}" / "watershed_supervoxels.npy" for f in frames],
            name="watershed",
        )
        separator = stack_npy(
            [output_root / f"t{f:03d}" / "separator_probability.npy" for f in frames],
            name="separator",
        )
        viewer.add_image(
            preprocessed,
            name="Preprocessed",
            scale=scale_4d,
            visible=False,
        )
        viewer.add_labels(
            source_mask,
            name="Source binary mask",
            scale=scale_4d,
            opacity=0.55,
            visible=False,
        )
        viewer.add_labels(
            watershed,
            name="Watershed supervoxels",
            scale=scale_4d,
            opacity=1.0,
            visible=False,
        )
        viewer.add_image(
            separator,
            name="Separator probability",
            scale=scale_4d,
            opacity=0.65,
            visible=False,
        )

    print(f"[viewer] frames represented in T slider: {frames}", flush=True)
    print(
        "[viewer] Final instances are visible by default; source and multicut "
        "are available for direct A/B inspection.",
        flush=True,
    )
    napari.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--sample-zarr", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--frames", default="0-4")
    parser.add_argument(
        "--diagnostics",
        choices=("basic", "full"),
        default="full",
        help="basic saves source/multicut/final; full also saves preprocessed, mask, watershed, separator",
    )
    parser.add_argument("--spacing", default="1.625,0.40625,0.40625")
    parser.add_argument("--tile-shape", default="32,128,128")
    parser.add_argument("--tile-overlap", default="8,32,32")
    parser.add_argument("--tile-halo", default="4,16,16")
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--viewer-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    spacing = parse_triplet(args.spacing, cast=float, name="--spacing")
    tile_shape = parse_triplet(args.tile_shape, cast=int, name="--tile-shape")
    tile_overlap = parse_triplet(args.tile_overlap, cast=int, name="--tile-overlap")
    tile_halo = parse_triplet(args.tile_halo, cast=int, name="--tile-halo")

    sample_zarr = resolve_sample_zarr(args.sample_id, args.sample_zarr)
    output_root = (
        default_output(args.sample_id)
        if args.output is None
        else resolve(args.output)
    )

    from src.io import open_sample
    image = open_sample(sample_zarr)
    if len(image.shape) != 4:
        raise ValueError(f"Expected [T,Z,Y,X], got {image.shape}")
    frame_count = int(image.shape[0])

    if not args.viewer_only:
        checkpoint = resolve_checkpoint(args.checkpoint)
        frames = parse_frames(args.frames, frame_count)
        pipeline_profile(
            sample_id=args.sample_id,
            sample_zarr=sample_zarr,
            checkpoint=checkpoint,
            output_root=output_root,
            frames=frames,
            spacing=tuple(float(v) for v in spacing),
            tile_shape=tuple(int(v) for v in tile_shape),
            tile_overlap=tuple(int(v) for v in tile_overlap),
            tile_halo=tuple(int(v) for v in tile_halo),
            tile_batch_size=int(args.tile_batch_size),
            diagnostics=args.diagnostics,
            overwrite=bool(args.overwrite),
        )

    if not args.no_viewer:
        open_viewer(
            sample_zarr=sample_zarr,
            output_root=output_root,
            spacing=tuple(float(v) for v in spacing),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
