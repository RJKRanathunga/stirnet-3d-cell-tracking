from __future__ import annotations

r"""
Investigation 36 — current spatial STIR-Net + Trackastra visualization on the
20-frame BioHub sample.

Default sample
--------------
    44b6_0113de3b

Scientific path
---------------
For all requested frames:

    raw BioHub
      -> canonical preprocessing
      -> binary foreground mask
      -> source segmentation
           geometric completion EXPLICITLY DISABLED
      -> five-channel STIR-Net input
      -> current tiled spatial STIR-Net
      -> learned watershed / RAG / signed multicut
      -> source-instance anchored split-only filter
      -> final spatial instances + cells

The frame preparation uses the validated Investigation-34 schedule:

    one CPU preparation worker for t+1
        ||
    main-thread CUDA spatial inference for t

After the complete spatial movie exists:

    raw movie + final spatial instance movie
      -> Trackastra (same defaults as Investigation 30: ctc / greedy / cuda)
      -> track graph + tracked masks
      -> Napari tracklets

Visualization intentionally follows notebooks/09_visualization.ipynb:
- Raw Volume
- Preprocessed Volume
- Binary Mask
- Spatial Final Instances
- Trackastra Tracked Masks
- Tracks - all
- Centroids - all
- Ended Tracks / Centroids
- New Tracks / Centroids
- Boundary Entry Tracks / Centroids
- Boundary Exit Tracks / Centroids
- same physical Z/Y/X scaling and endpoint grouping
- cell-volume and tracking-scene extractors when available

This investigation does NOT run learned temporal STIR-Net reasoning and does
NOT run the Stage-7/8 legacy tracking stack.  Trackastra is the temporal
association shown here.

Typical run
-----------
From repository root:

    python .\investigations\stirnet\36_biohub_spatial_trackastra_visualization.py

Force a fresh spatial + Trackastra run:

    python .\investigations\stirnet\36_biohub_spatial_trackastra_visualization.py ^
        --overwrite-spatial --rebuild-trackastra

Re-open completed results without inference:

    python .\investigations\stirnet\36_biohub_spatial_trackastra_visualization.py --viewer-only

Run/cache only, do not open Napari:

    python .\investigations\stirnet\36_biohub_spatial_trackastra_visualization.py ^
        --no-viewer
"""

import argparse
import gc
import importlib.util
import json
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SCRIPT_NAME = "36_biohub_spatial_trackastra_visualization"
DEFAULT_SAMPLE_ID = "44b6_0113de3b"
DEFAULT_FRAME_COUNT = 20
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)

DEFAULT_TILE_SHAPE_ZYX = (32, 128, 128)
DEFAULT_TILE_OVERLAP_ZYX = (8, 32, 32)
DEFAULT_TILE_HALO_ZYX = (4, 16, 16)
DEFAULT_TILE_BATCH_SIZE = 1

DEFAULT_TRACKASTRA_MODEL = "ctc"
DEFAULT_TRACKASTRA_MODE = "greedy"
DEFAULT_TRACKASTRA_DEVICE = "cuda"

BOUNDARY_MARGIN_UM = 4.0
SHOW_BOUNDARY_TRACKS = False
SCENE_PADDING_ZYX = (2, 12, 12)


# =============================================================================
# Repository / path helpers
# =============================================================================


def repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = (here.parent, *here.parents, Path.cwd().resolve())
    for candidate in candidates:
        if (
            (candidate / "learned" / "stirnet").is_dir()
            and (candidate / "src").is_dir()
            and (candidate / "investigations" / "stirnet").is_dir()
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
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_kaggle_runner():
    return _load_module(
        ROOT / "kaggle" / "run_submission.py",
        "_inv36_kaggle_runner",
    )


def resolve_sample_zarr(sample_id: str, override: str | None) -> Path:
    if override is not None:
        path = resolve(override)
    else:
        path = (
            ROOT
            / "data"
            / "sample"
            / "biohub_5samples_20timepoints"
            / "train"
            / sample_id
            / f"{sample_id}.zarr"
        ).resolve()
        if not path.exists():
            from src.io import PipelinePaths

            path = PipelinePaths.discover(ROOT).sample_zarr(sample_id).resolve()

    if path.name == "0":
        path = path.parent
    if not (path / "0").exists():
        raise FileNotFoundError(f"BioHub Zarr array not found: {path / '0'}")
    return path


def resolve_checkpoint(override: str | None) -> Path:
    if override is not None:
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
        "Could not resolve the morphology-v2 h100 checkpoint. "
        "Pass --checkpoint explicitly."
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


def parse_triplet(text: str, *, cast, name: str):
    values = tuple(cast(token.strip()) for token in str(text).split(","))
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 comma-separated values")
    return values


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds >= 3600:
        return f"{seconds / 3600.0:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds:.1f} s"


# =============================================================================
# Output contract
# =============================================================================


@dataclass(frozen=True)
class OutputPaths:
    root: Path

    @property
    def movies(self) -> Path:
        return self.root / "movies"

    @property
    def cells_dir(self) -> Path:
        return self.root / "cells"

    @property
    def trackastra_dir(self) -> Path:
        return self.root / "trackastra"

    @property
    def raw(self) -> Path:
        return self.movies / "raw.npy"

    @property
    def preprocessed(self) -> Path:
        return self.movies / "preprocessed.npy"

    @property
    def binary_mask(self) -> Path:
        return self.movies / "binary_mask.npy"

    @property
    def source_instances(self) -> Path:
        return self.movies / "source_instances.npy"

    @property
    def final_instances(self) -> Path:
        return self.movies / "final_instances.npy"

    @property
    def cells_csv(self) -> Path:
        return self.root / "cells_all.csv"

    @property
    def spatial_summary(self) -> Path:
        return self.root / "spatial_summary.json"

    @property
    def spatial_success(self) -> Path:
        return self.root / "_SPATIAL_SUCCESS.json"

    @property
    def track_graph(self) -> Path:
        return self.trackastra_dir / "track_graph.pkl"

    @property
    def tracked_masks(self) -> Path:
        return self.trackastra_dir / "tracked_masks.npy"

    @property
    def napari_tracks(self) -> Path:
        return self.trackastra_dir / "napari_tracks.npy"

    @property
    def napari_graph(self) -> Path:
        return self.trackastra_dir / "napari_graph.json"

    @property
    def tracks_csv(self) -> Path:
        return self.trackastra_dir / "tracks.csv"

    @property
    def trackastra_summary(self) -> Path:
        return self.trackastra_dir / "summary.json"


def spatial_cache_complete(paths: OutputPaths, frame_count: int) -> bool:
    required = (
        paths.raw,
        paths.preprocessed,
        paths.binary_mask,
        paths.source_instances,
        paths.final_instances,
        paths.cells_csv,
        paths.spatial_success,
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        final = np.load(paths.final_instances, mmap_mode="r", allow_pickle=False)
        return int(final.shape[0]) == int(frame_count)
    except Exception:
        return False


def trackastra_cache_complete(paths: OutputPaths) -> bool:
    return (
        paths.track_graph.is_file()
        and paths.tracked_masks.is_file()
        and paths.napari_tracks.is_file()
        and paths.tracks_csv.is_file()
    )


# =============================================================================
# Investigation-34 CPU preparation
# =============================================================================


_segmentation_config_module = import_module("src.03_segmentation.config")
SOURCE_SEGMENTATION_CONFIG = replace(
    _segmentation_config_module.DEFAULT_SEGMENTATION_CONFIG,
    enable_geometric_completion=False,
)


@dataclass
class PreparedFrame:
    frame: int
    raw: np.ndarray
    preprocessed: np.ndarray
    source_mask: np.ndarray
    source_labels: np.ndarray
    spatial: np.ndarray
    dref_um: float
    prepare_seconds: float


def prepare_frame(
    frame: int,
    *,
    sample_zarr: Path,
    helper: Any,
    spacing: tuple[float, float, float],
) -> PreparedFrame:
    """CPU-only preparation. This function must never touch CUDA."""
    from src.api import (
        create_binary_mask,
        preprocess_volume,
        segment_instances,
    )
    from src.io import load_timepoint

    started = time.perf_counter()
    raw = np.asarray(load_timepoint(sample_zarr, frame))
    preprocessed = preprocess_volume(raw)
    source_mask = create_binary_mask(preprocessed)
    source_labels = segment_instances(
        source_mask,
        config=SOURCE_SEGMENTATION_CONFIG,
    )
    spatial, dref_um = helper.build_stage6_spatial_input(
        preprocessed,
        source_labels,
        spacing,
    )
    return PreparedFrame(
        frame=int(frame),
        raw=raw,
        preprocessed=preprocessed,
        source_mask=source_mask,
        source_labels=source_labels,
        spatial=spatial,
        dref_um=float(dref_um),
        prepare_seconds=float(time.perf_counter() - started),
    )


def _create_movie(
    path: Path,
    *,
    dtype,
    shape: tuple[int, ...],
) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )


# =============================================================================
# Spatial inference
# =============================================================================


def run_spatial_movie(
    *,
    sample_id: str,
    sample_zarr: Path,
    checkpoint: Path,
    output: OutputPaths,
    frame_count: int,
    spacing: tuple[float, float, float],
    tile_shape: tuple[int, int, int],
    tile_overlap: tuple[int, int, int],
    tile_halo: tuple[int, int, int],
    tile_batch_size: int,
    device: torch.device,
) -> None:
    from src.api import detect_cells, extract_cell_features
    from src.io import open_sample

    runner = load_kaggle_runner()

    print("[spatial] loading current STIR-Net runtime ...", flush=True)
    runtime, helper = runner._load_model_runtime(
        ROOT,
        checkpoint,
        device=device,
        tile_shape=tile_shape,
        tile_overlap=tile_overlap,
        tile_halo=tile_halo,
        tile_batch_size=tile_batch_size,
    )

    image = open_sample(sample_zarr)
    if len(image.shape) != 4:
        raise ValueError(f"Expected [T,Z,Y,X], got shape={image.shape}")
    if frame_count > int(image.shape[0]):
        raise ValueError(
            f"Requested {frame_count} frames but movie has only {image.shape[0]}"
        )

    spatial_shape = tuple(int(v) for v in image.shape[-3:])
    movie_shape = (frame_count, *spatial_shape)

    output.root.mkdir(parents=True, exist_ok=True)
    output.movies.mkdir(parents=True, exist_ok=True)
    output.cells_dir.mkdir(parents=True, exist_ok=True)

    raw_movie = _create_movie(
        output.raw,
        dtype=np.dtype(image.dtype),
        shape=movie_shape,
    )
    preprocessed_movie = _create_movie(
        output.preprocessed,
        dtype=np.float16,
        shape=movie_shape,
    )
    binary_movie = _create_movie(
        output.binary_mask,
        dtype=np.uint8,
        shape=movie_shape,
    )
    source_movie = _create_movie(
        output.source_instances,
        dtype=np.int32,
        shape=movie_shape,
    )
    final_movie = _create_movie(
        output.final_instances,
        dtype=np.int32,
        shape=movie_shape,
    )

    frame_rows: list[dict[str, Any]] = []
    cells_all: list[pd.DataFrame] = []

    run_started = time.perf_counter()
    completion_times: list[float] = []

    print("=" * 124, flush=True)
    print("INVESTIGATION 36 — 20-FRAME CURRENT SPATIAL STIR-NET", flush=True)
    print("=" * 124, flush=True)
    print(f"sample           : {sample_id}", flush=True)
    print(f"zarr             : {sample_zarr}", flush=True)
    print(f"frames           : 0..{frame_count - 1}", flush=True)
    print(f"checkpoint       : {checkpoint}", flush=True)
    print(f"checkpoint step  : {runtime.checkpoint_step}", flush=True)
    print(f"device           : {runtime.device}", flush=True)
    print("prep scheduling  : 1 CPU worker / 1-frame lookahead", flush=True)
    print("source geometry  : DISABLED", flush=True)
    print("=" * 124, flush=True)

    future: Future[PreparedFrame] | None = None
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="inv36-prep",
    ) as executor:
        future = executor.submit(
            prepare_frame,
            0,
            sample_zarr=sample_zarr,
            helper=helper,
            spacing=spacing,
        )

        for index, frame in enumerate(range(frame_count), start=1):
            wait_started = time.perf_counter()
            prepared = future.result()
            prep_wait_seconds = time.perf_counter() - wait_started

            if prepared.frame != frame:
                raise RuntimeError(
                    f"Preparation ordering failure: expected {frame}, got "
                    f"{prepared.frame}"
                )

            # Submit t+1 before CUDA starts t.
            if frame + 1 < frame_count:
                future = executor.submit(
                    prepare_frame,
                    frame + 1,
                    sample_zarr=sample_zarr,
                    helper=helper,
                    spacing=spacing,
                )

            frame_started = time.perf_counter()

            inference_started = time.perf_counter()
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
            spatial_wall_seconds = time.perf_counter() - inference_started

            before = runner._tensor_numpy(
                result.spatial_partition.labels[0],
                np.int32,
            )
            watershed = runner._tensor_numpy(
                result.supervoxel_labels[0],
                np.int64,
            )
            separator_probability = runner._tensor_numpy(
                result.dense.geometry.probabilities()["separator"][0, 0],
                np.float32,
            )

            post_started = time.perf_counter()
            final_labels, split_diag = runner._apply_source_core_split_only(
                before,
                watershed,
                separator_probability,
                prepared.source_mask,
                prepared.source_labels,
                spacing,
                prepared.dref_um,
            )

            cells = detect_cells(final_labels)
            cells = extract_cell_features(
                cells,
                final_labels,
                prepared.preprocessed,
            )
            if cells.empty:
                raise RuntimeError(
                    f"{sample_id} t={frame}: no cells after spatial inference"
                )
            post_seconds = time.perf_counter() - post_started

            # Persist the complete 20-frame visualization contract.
            raw_movie[frame] = prepared.raw
            preprocessed_movie[frame] = np.asarray(
                prepared.preprocessed,
                dtype=np.float16,
            )
            binary_movie[frame] = np.asarray(
                prepared.source_mask > 0,
                dtype=np.uint8,
            )
            source_movie[frame] = prepared.source_labels.astype(
                np.int32,
                copy=False,
            )
            final_movie[frame] = final_labels.astype(np.int32, copy=False)

            cells_frame = cells.copy()
            cells_frame["frame"] = int(frame)
            cells_all.append(cells_frame)
            atomic_csv(
                output.cells_dir / f"t{frame:03d}.csv",
                cells_frame,
            )

            completion_times.append(time.perf_counter())
            interval_seconds = (
                completion_times[-1] - completion_times[-2]
                if len(completion_times) >= 2
                else completion_times[-1] - run_started
            )
            main_seconds = time.perf_counter() - frame_started
            hidden_fraction = (
                max(prepared.prepare_seconds - prep_wait_seconds, 0.0)
                / prepared.prepare_seconds
                if prepared.prepare_seconds > 0
                else 0.0
            )

            row = {
                "frame": int(frame),
                "source_instances": int(
                    np.count_nonzero(np.unique(prepared.source_labels) > 0)
                ),
                "multicut_instances": int(
                    np.count_nonzero(np.unique(before) > 0)
                ),
                "final_instances": int(
                    np.count_nonzero(np.unique(final_labels) > 0)
                ),
                "splits_applied": int(split_diag["applied_count"]),
                "prepare_seconds": float(prepared.prepare_seconds),
                "prepare_wait_seconds": float(prep_wait_seconds),
                "prepare_hidden_fraction": float(hidden_fraction),
                "spatial_wall_seconds": float(spatial_wall_seconds),
                "reported_inference_seconds": float(reported_inference_seconds),
                "post_seconds": float(post_seconds),
                "main_seconds": float(main_seconds),
                "completion_interval_seconds": float(interval_seconds),
                "amp_dtype": str(amp_name),
                "peak_allocated_vram_gib": float(peak_gib),
            }
            frame_rows.append(row)

            print(
                f"[{index:02d}/{frame_count:02d} t={frame:03d}] "
                f"source={row['source_instances']} -> "
                f"multicut={row['multicut_instances']} -> "
                f"final={row['final_instances']} splits={row['splits_applied']} | "
                f"prep={prepared.prepare_seconds:5.1f}s "
                f"wait={prep_wait_seconds:4.2f}s "
                f"hidden={100.0 * hidden_fraction:5.1f}% | "
                f"spatial={spatial_wall_seconds:5.1f}s "
                f"post={post_seconds:4.1f}s "
                f"interval={interval_seconds:5.1f}s "
                f"VRAM={peak_gib:.2f}GiB",
                flush=True,
            )

            del result, spatial_gpu, separator_probability, watershed
            del before, final_labels, cells, prepared
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    raw_movie.flush()
    preprocessed_movie.flush()
    binary_movie.flush()
    source_movie.flush()
    final_movie.flush()

    del raw_movie, preprocessed_movie, binary_movie, source_movie, final_movie

    combined_cells = pd.concat(cells_all, ignore_index=True)
    atomic_csv(output.cells_csv, combined_cells)

    elapsed = time.perf_counter() - run_started
    warm_intervals = np.asarray(
        [row["completion_interval_seconds"] for row in frame_rows[1:]],
        dtype=np.float64,
    )
    summary = {
        "sample_id": sample_id,
        "frame_count": frame_count,
        "spacing_zyx_um": list(spacing),
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(runtime.checkpoint_step),
        "elapsed_seconds": float(elapsed),
        "elapsed": format_seconds(elapsed),
        "warm_mean_completion_interval_seconds": (
            float(warm_intervals.mean()) if warm_intervals.size else None
        ),
        "frames": frame_rows,
        "scientific_path": {
            "source_geometric_completion": False,
            "parallel_preparation_workers": 1,
            "parallel_prefetch_depth": 1,
            "temporal_stirnet": False,
        },
    }
    atomic_json(output.spatial_summary, summary)
    atomic_json(
        output.spatial_success,
        {
            "status": "success",
            "sample_id": sample_id,
            "frame_count": frame_count,
            "elapsed_seconds": float(elapsed),
        },
    )

    # Release the spatial model before Trackastra claims the same GPU.
    del runtime
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[spatial] complete: {frame_count} frames in {format_seconds(elapsed)}",
        flush=True,
    )


# =============================================================================
# Trackastra
# =============================================================================


def run_trackastra(
    *,
    output: OutputPaths,
    model_name: str,
    mode: str,
    device: str,
    rebuild: bool,
) -> None:
    if trackastra_cache_complete(output) and not rebuild:
        print("[trackastra] reusing cached graph/masks/tracks", flush=True)
        return

    output.trackastra_dir.mkdir(parents=True, exist_ok=True)

    try:
        from trackastra.model import Trackastra
        from trackastra.tracking.utils import graph_to_napari_tracks
    except ImportError as exc:
        raise RuntimeError(
            "Trackastra is required for Investigation 36. "
            "Activate the repository environment containing trackastra==0.5.5."
        ) from exc

    raw_movie = np.load(output.raw, mmap_mode="r", allow_pickle=False)
    final_movie = np.load(
        output.final_instances,
        mmap_mode="r",
        allow_pickle=False,
    )

    print("", flush=True)
    print("=" * 108, flush=True)
    print("INVESTIGATION 36 — TRACKASTRA", flush=True)
    print("=" * 108, flush=True)
    print(f"model     : {model_name}", flush=True)
    print(f"mode      : {mode}", flush=True)
    print(f"device    : {device}", flush=True)
    print(f"raw       : {output.raw}", flush=True)
    print(f"instances : {output.final_instances}", flush=True)
    print("=" * 108, flush=True)

    started = time.perf_counter()
    model = Trackastra.from_pretrained(model_name, device=device)
    track_graph, tracked_masks = model.track(
        raw_movie,
        final_movie,
        mode=mode,
    )
    track_seconds = time.perf_counter() - started

    with output.track_graph.open("wb") as handle:
        pickle.dump(track_graph, handle)

    np.save(
        output.tracked_masks,
        np.asarray(tracked_masks),
        allow_pickle=False,
    )

    napari_tracks, napari_graph, _properties = graph_to_napari_tracks(
        track_graph
    )
    napari_tracks = np.asarray(napari_tracks, dtype=np.float64)
    if napari_tracks.ndim != 2 or napari_tracks.shape[1] != 5:
        raise RuntimeError(
            "Expected Trackastra 3-D Napari tracks with columns "
            "[track_id,time,z,y,x], got shape="
            f"{napari_tracks.shape}"
        )
    np.save(output.napari_tracks, napari_tracks, allow_pickle=False)

    # JSON is only a human-readable/debug copy; Napari tracks are the primary
    # representation. Trackastra uses integer child->parent relations.
    serializable_graph = {
        str(int(child)): (
            [int(v) for v in parent]
            if isinstance(parent, (list, tuple, set))
            else int(parent)
        )
        for child, parent in napari_graph.items()
    }
    atomic_json(output.napari_graph, serializable_graph)

    tracks_df = pd.DataFrame(
        napari_tracks,
        columns=["track_id", "frame", "z", "y", "x"],
    )
    tracks_df["track_id"] = tracks_df["track_id"].astype(np.int64)
    tracks_df["frame"] = tracks_df["frame"].astype(np.int64)

    cells = pd.read_csv(output.cells_csv)
    from src.api import prepare_visualization_data

    visualization = prepare_visualization_data(
        tracks_df,
        cells,
        assign_cell_ids=True,
    )
    atomic_csv(output.tracks_csv, visualization.tracks)

    summary = {
        "model": model_name,
        "mode": mode,
        "device": device,
        "seconds": float(track_seconds),
        "elapsed": format_seconds(track_seconds),
        "graph_nodes": int(track_graph.number_of_nodes()),
        "graph_edges": int(track_graph.number_of_edges()),
        "napari_track_rows": int(len(napari_tracks)),
        "napari_tracklets": int(
            np.unique(napari_tracks[:, 0]).size
            if napari_tracks.size
            else 0
        ),
        "napari_parent_relations": int(len(napari_graph)),
    }
    atomic_json(output.trackastra_summary, summary)

    del model, tracked_masks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"[trackastra] nodes={summary['graph_nodes']} "
        f"edges={summary['graph_edges']} "
        f"tracklets={summary['napari_tracklets']} "
        f"time={summary['elapsed']}",
        flush=True,
    )


# =============================================================================
# Notebook-09 style visualization
# =============================================================================


def open_napari_viewer(
    *,
    sample_id: str,
    output: OutputPaths,
    spacing: tuple[float, float, float],
) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is required to open Investigation 36. "
            "Install/activate the repository visualization environment."
        ) from exc

    from src.api import prepare_visualization_data

    raw = np.load(output.raw, mmap_mode="r", allow_pickle=False)
    preprocessed = np.load(
        output.preprocessed,
        mmap_mode="r",
        allow_pickle=False,
    )
    binary_mask = np.load(
        output.binary_mask,
        mmap_mode="r",
        allow_pickle=False,
    )
    source_instances = np.load(
        output.source_instances,
        mmap_mode="r",
        allow_pickle=False,
    )
    final_instances = np.load(
        output.final_instances,
        mmap_mode="r",
        allow_pickle=False,
    )
    tracked_masks = np.load(
        output.tracked_masks,
        mmap_mode="r",
        allow_pickle=False,
    )

    cells = pd.read_csv(output.cells_csv)
    tracks = pd.read_csv(output.tracks_csv)

    visualization = prepare_visualization_data(
        tracks,
        cells,
        assign_cell_ids=False,
        spatial_shape_zyx=raw.shape[-3:],
        voxel_size_zyx=spacing,
        boundary_margin_um=BOUNDARY_MARGIN_UM,
    )
    endpoint_groups = visualization.endpoint_groups
    if endpoint_groups is None:
        raise RuntimeError("Endpoint grouping unexpectedly returned None")

    scale_tzyx = (1.0, *spacing)

    print("", flush=True)
    print("=" * 108, flush=True)
    print("INVESTIGATION 36 — NOTEBOOK-09 STYLE VISUALIZATION", flush=True)
    print("=" * 108, flush=True)
    print(f"Voxel size ZYX          : {spacing}", flush=True)
    print(
        "Trackastra tracklets    : "
        f"{visualization.tracks['track_id'].nunique()}",
        flush=True,
    )
    print(
        "New failure candidates  : "
        f"{endpoint_groups.new_failure_tracks.track_id.nunique()}",
        flush=True,
    )
    print(
        "Ended failure candidates: "
        f"{endpoint_groups.ended_failure_tracks.track_id.nunique()}",
        flush=True,
    )
    print(
        "Boundary entries        : "
        f"{endpoint_groups.boundary_entry_tracks.track_id.nunique()}",
        flush=True,
    )
    print(
        "Boundary exits          : "
        f"{endpoint_groups.boundary_exit_tracks.track_id.nunique()}",
        flush=True,
    )
    print("=" * 108, flush=True)

    viewer = napari.Viewer(ndisplay=3)

    # Notebook 09 computes these percentiles from the complete raw movie.
    low, high = np.percentile(np.asarray(raw), [1.0, 99.8])
    viewer.add_image(
        raw,
        name="Raw Volume",
        scale=scale_tzyx,
        rendering="mip",
        colormap="gray",
        contrast_limits=[float(low), float(high)],
    )
    viewer.add_image(
        preprocessed,
        name="Preprocessed Volume",
        scale=scale_tzyx,
        rendering="mip",
        colormap="gray",
        contrast_limits=(0.0, 1.0),
        visible=False,
    )
    viewer.add_labels(
        binary_mask,
        name="Binary Mask",
        scale=scale_tzyx,
        visible=False,
    )
    viewer.add_labels(
        source_instances,
        name="Source Instances",
        scale=scale_tzyx,
        visible=False,
    )
    viewer.add_labels(
        final_instances,
        name="Spatial Final Instances",
        scale=scale_tzyx,
        visible=False,
    )
    viewer.add_labels(
        tracked_masks,
        name="Trackastra Tracked Masks",
        scale=scale_tzyx,
        visible=False,
    )

    all_tracks_layer = viewer.add_tracks(
        visualization.tracks_array,
        name="Tracks - all",
        scale=scale_tzyx,
        tail_length=20,
    )
    all_tracks_layer.visible = False

    all_centers_layer = viewer.add_points(
        visualization.points_array,
        name="Centroids - all",
        scale=scale_tzyx,
        size=4,
        face_color="red",
        properties={
            "track_id": visualization.track_ids,
            "cell_id": visualization.tracks["cell_id"].to_numpy(),
        },
        text={
            "string": "{cell_id}",
            "size": 8,
            "color": "white",
            "anchor": "center",
        },
    )
    all_centers_layer.visible = False

    napari_layers = import_module("src.09_visualization.napari_layers")
    add_track_group = napari_layers.add_track_group

    add_track_group(
        viewer,
        endpoint_groups.ended_failure_tracks,
        track_name="Ended Tracks",
        point_name="Ended Centroids",
        color="red",
        scale=scale_tzyx,
    )
    add_track_group(
        viewer,
        endpoint_groups.new_failure_tracks,
        track_name="New Tracks",
        point_name="New Centroids",
        color="lime",
        scale=scale_tzyx,
    )
    add_track_group(
        viewer,
        endpoint_groups.boundary_entry_tracks,
        track_name="Boundary Entry Tracks",
        point_name="Boundary Entry Centroids",
        color="cyan",
        scale=scale_tzyx,
        visible=SHOW_BOUNDARY_TRACKS,
    )
    add_track_group(
        viewer,
        endpoint_groups.boundary_exit_tracks,
        track_name="Boundary Exit Tracks",
        point_name="Boundary Exit Centroids",
        color="orange",
        scale=scale_tzyx,
        visible=SHOW_BOUNDARY_TRACKS,
    )

    # Preserve Notebook-09 diagnostic tooling, but do not make viewer startup
    # depend on these optional widgets.
    try:
        from diagnostics.cell_volume_extraction import (
            add_cell_volume_extractor,
        )
        from diagnostics.tracking_scene_extraction import (
            add_tracking_scene_extractor,
        )

        add_tracking_scene_extractor(
            viewer=viewer,
            cells=cells,
            instance_labels_volume=final_instances,
            binary_mask_volume=binary_mask,
            image_volumes={
                "raw": raw,
                "preprocessed": preprocessed,
            },
            sample_id=sample_id,
            save_root=output.root / "tracking_scenes",
            voxel_size_zyx=spacing,
            default_padding_zyx=SCENE_PADDING_ZYX,
            cell_id_column="cell_id",
            frame_column="frame",
            source_metadata={
                "investigation": SCRIPT_NAME,
                "cells_csv": output.cells_csv,
                "tracks_csv": output.tracks_csv,
                "source_zarr": str(resolve_sample_zarr(sample_id, None)),
                "spatial_final_instances": output.final_instances,
                "trackastra_graph": output.track_graph,
            },
        )
        add_cell_volume_extractor(
            viewer=viewer,
            cells=cells,
            image_volume=raw,
            sample_id=sample_id,
            preprocessed_volume=preprocessed,
            binary_mask_volume=binary_mask,
            instance_labels_volume=final_instances,
            voxel_size_zyx=spacing,
        )
        print("[viewer] Notebook-09 diagnostic extractors attached.", flush=True)
    except Exception as exc:
        print(
            "[viewer warning] diagnostic extractors could not be attached: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

    print(
        "[viewer] Raw visible by default. Toggle 'Spatial Final Instances', "
        "'Trackastra Tracked Masks', and 'Tracks - all' for direct inspection.",
        flush=True,
    )
    napari.run()


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current spatial STIR-Net over the 20-frame BioHub sample, "
            "run Trackastra, and open Notebook-09 style Napari visualization."
        )
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--sample-zarr", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--frame-count",
        type=int,
        default=DEFAULT_FRAME_COUNT,
    )
    parser.add_argument("--output", default=None)

    parser.add_argument(
        "--spacing",
        default="1.625,0.40625,0.40625",
    )
    parser.add_argument("--tile-shape", default="32,128,128")
    parser.add_argument("--tile-overlap", default="8,32,32")
    parser.add_argument("--tile-halo", default="4,16,16")
    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=DEFAULT_TILE_BATCH_SIZE,
    )
    parser.add_argument(
        "--spatial-device",
        default="cuda",
    )

    parser.add_argument(
        "--trackastra-model",
        default=DEFAULT_TRACKASTRA_MODEL,
    )
    parser.add_argument(
        "--trackastra-mode",
        default=DEFAULT_TRACKASTRA_MODE,
    )
    parser.add_argument(
        "--trackastra-device",
        default=DEFAULT_TRACKASTRA_DEVICE,
    )

    parser.add_argument(
        "--overwrite-spatial",
        action="store_true",
        help="Discard/recompute the cached 20-frame spatial inference.",
    )
    parser.add_argument(
        "--rebuild-trackastra",
        action="store_true",
        help="Rerun Trackastra even when cached graph/masks exist.",
    )
    parser.add_argument(
        "--viewer-only",
        action="store_true",
        help="Skip inference/Trackastra and open an existing completed cache.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run/cache everything but do not start Napari.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.frame_count < 1:
        raise ValueError("--frame-count must be positive")

    spacing = tuple(
        float(v)
        for v in parse_triplet(
            args.spacing,
            cast=float,
            name="--spacing",
        )
    )
    tile_shape = tuple(
        int(v)
        for v in parse_triplet(
            args.tile_shape,
            cast=int,
            name="--tile-shape",
        )
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
        for v in parse_triplet(
            args.tile_halo,
            cast=int,
            name="--tile-halo",
        )
    )

    output_root = (
        default_output(args.sample_id)
        if args.output is None
        else resolve(args.output)
    )
    output = OutputPaths(output_root)

    sample_zarr = resolve_sample_zarr(
        args.sample_id,
        args.sample_zarr,
    )

    if args.viewer_only:
        if not spatial_cache_complete(output, args.frame_count):
            raise FileNotFoundError(
                f"Incomplete spatial cache below {output.root}"
            )
        if not trackastra_cache_complete(output):
            raise FileNotFoundError(
                f"Incomplete Trackastra cache below {output.trackastra_dir}"
            )
    else:
        if args.overwrite_spatial and output.root.exists():
            # Spatial results and Trackastra are coupled.  A new spatial movie
            # invalidates the old Trackastra graph, so remove the whole cache.
            shutil.rmtree(output.root)

        if spatial_cache_complete(output, args.frame_count):
            print(
                f"[spatial] reusing completed cache: {output.root}",
                flush=True,
            )
        else:
            checkpoint = resolve_checkpoint(args.checkpoint)
            spatial_device = torch.device(args.spatial_device)
            if (
                spatial_device.type == "cuda"
                and not torch.cuda.is_available()
            ):
                raise RuntimeError(
                    "CUDA requested for spatial inference but unavailable."
                )
            torch.set_grad_enabled(False)
            run_spatial_movie(
                sample_id=args.sample_id,
                sample_zarr=sample_zarr,
                checkpoint=checkpoint,
                output=output,
                frame_count=args.frame_count,
                spacing=spacing,
                tile_shape=tile_shape,
                tile_overlap=tile_overlap,
                tile_halo=tile_halo,
                tile_batch_size=int(args.tile_batch_size),
                device=spatial_device,
            )

        # A Trackastra rebuild is automatically needed after --overwrite-spatial.
        run_trackastra(
            output=output,
            model_name=args.trackastra_model,
            mode=args.trackastra_mode,
            device=args.trackastra_device,
            rebuild=bool(
                args.rebuild_trackastra or args.overwrite_spatial
            ),
        )

    print("", flush=True)
    print("=" * 108, flush=True)
    print("INVESTIGATION 36 READY", flush=True)
    print("=" * 108, flush=True)
    print(f"sample       : {args.sample_id}", flush=True)
    print(f"frames       : {args.frame_count}", flush=True)
    print(f"output       : {output.root}", flush=True)
    print(f"spatial      : {output.final_instances}", flush=True)
    print(f"cells        : {output.cells_csv}", flush=True)
    print(f"track graph  : {output.track_graph}", flush=True)
    print(f"tracked mask : {output.tracked_masks}", flush=True)
    print(f"tracks       : {output.tracks_csv}", flush=True)
    print("=" * 108, flush=True)

    if not args.no_viewer:
        open_napari_viewer(
            sample_id=args.sample_id,
            output=output,
            spacing=spacing,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
