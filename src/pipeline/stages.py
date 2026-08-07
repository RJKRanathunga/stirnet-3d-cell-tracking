"""Adapters from the full-dataset runner to the repository's current stage APIs."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.api import (
    FourDGraphConfig,
    GraphTrackingConfig,
    run_cell_lineage,
    run_cell_tracking,
    run_track_reconciliation,
    run_track_stitching,
)
from src.dataset_processing import process_dataset
from src.io import (
    load_csv,
    load_json,
    load_optional_csv,
    open_sample,
    save_lineage_result,
    save_stitching_result,
    save_track_reconciliation_result,
    save_tracking_result,
)
from src.io.arrays import list_timepoint_files
from src.io.stage_io import CELL_COLUMNS

from .config import FullPipelineConfig
from .dataset import FullDatasetSample
from .paths import SampleOutputPaths


TrackReconciliationConfig = import_module(
    "src.11_track_reconciliation.step01_config"
).TrackReconciliationConfig


@dataclass(frozen=True)
class ProcessedState:
    root: Path
    cell_files: tuple[Path, ...]
    segmentation_files: tuple[Path, ...]
    time_frames: tuple[pd.DataFrame, ...]


@dataclass(frozen=True)
class Stage7State:
    tracks: pd.DataFrame
    global_motion: pd.DataFrame
    association_events: pd.DataFrame
    association_candidates: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Stage8State:
    tracks: pd.DataFrame
    segmentation_events: pd.DataFrame


@dataclass(frozen=True)
class Stage10State:
    division_events: pd.DataFrame
    lineage_edges: pd.DataFrame
    track_lineage: pd.DataFrame
    protected_tracks: pd.DataFrame


@dataclass
class PipelineState:
    processed: ProcessedState | None = None
    stage7: Any | None = None
    stage8: Any | None = None
    stage10: Any | None = None
    stage11: Any | None = None


def load_processed_state(directory: str | Path) -> ProcessedState:
    root = Path(directory)
    cell_files = tuple(list_timepoint_files(root / "cells", "csv"))
    segmentation_files = tuple(
        list_timepoint_files(root / "segmentation", "npy")
    )
    if len(cell_files) != len(segmentation_files):
        raise ValueError(
            "The number of Stage 6 cell tables does not match segmentation volumes: "
            f"{len(cell_files)} vs {len(segmentation_files)}"
        )
    frames = tuple(
        load_csv(path, required_columns=CELL_COLUMNS)
        for path in cell_files
    )
    return ProcessedState(
        root=root,
        cell_files=cell_files,
        segmentation_files=segmentation_files,
        time_frames=frames,
    )


def load_stage7_state(directory: str | Path) -> Stage7State:
    root = Path(directory)
    return Stage7State(
        tracks=load_csv(root / "tracks.csv"),
        global_motion=load_optional_csv(root / "global_motion.csv"),
        association_events=load_optional_csv(root / "association_events.csv"),
        association_candidates=load_optional_csv(
            root / "association_candidates.csv"
        ),
        metadata=load_json(root / "metadata.json"),
    )


def load_stage8_state(directory: str | Path) -> Stage8State:
    root = Path(directory)
    return Stage8State(
        tracks=load_csv(root / "tracks.csv"),
        segmentation_events=load_optional_csv(root / "segmentation_events.csv"),
    )


def load_stage10_state(directory: str | Path) -> Stage10State:
    root = Path(directory)
    return Stage10State(
        division_events=load_csv(root / "division_events.csv"),
        lineage_edges=load_csv(root / "lineage_edges.csv"),
        track_lineage=load_csv(root / "track_lineage.csv"),
        protected_tracks=load_csv(root / "protected_tracks.csv"),
    )


def _frame_progress(sample_id: str):
    def progress(frames: range):
        total = len(frames)
        for frame in frames:
            print(
                f"    [{sample_id}] Stage 1-6 frame {int(frame) + 1}/{total}",
                flush=True,
            )
            yield frame

    return progress


def _graph_config(config: FullPipelineConfig) -> GraphTrackingConfig:
    four_d = FourDGraphConfig(
        window_size=int(config.graph_window_size),
        lookback_frames=int(config.graph_window_size) // 2,
        lookahead_frames=int(config.graph_window_size) // 2,
        maximum_gap_frames=int(config.graph_maximum_gap_frames),
        solver_time_limit_seconds=float(
            config.graph_solver_time_limit_seconds
        ),
        iterative_solver_iterations=int(
            config.graph_iterative_fallback_iterations
        ),
        save_detailed_debug_artifacts=bool(config.graph_save_debug_npz),
    )
    return GraphTrackingConfig(
        mode=config.graph_mode,
        algorithm=config.graph_algorithm,
        four_d=four_d,
    )


def run_stage6(
    sample: FullDatasetSample,
    paths: SampleOutputPaths,
    config: FullPipelineConfig,
) -> ProcessedState:
    output = paths.stage_directory(6)
    process_dataset(
        sample.zarr_path,
        output,
        progress=_frame_progress(sample.sample_id),
    )
    return load_processed_state(output)


def run_stage7(
    sample: FullDatasetSample,
    paths: SampleOutputPaths,
    config: FullPipelineConfig,
    processed: ProcessedState,
):
    result = run_cell_tracking(
        list(processed.time_frames),
        sample_id=sample.sample_id,
        graph_config=_graph_config(config),
    )
    save_tracking_result(result, paths.stage_directory(7))
    return result


def run_stage8(
    sample: FullDatasetSample,
    paths: SampleOutputPaths,
    processed: ProcessedState,
    stage7,
):
    result = run_track_stitching(
        stage7.tracks,
        list(processed.time_frames),
        processed.segmentation_files,
        sample_id=sample.sample_id,
    )
    save_stitching_result(result, paths.stage_directory(8))
    return result


def run_stage10(
    sample: FullDatasetSample,
    paths: SampleOutputPaths,
    processed: ProcessedState,
    stage8,
):
    raw_volume = open_sample(sample.zarr_path)
    result = run_cell_lineage(
        stage8.tracks,
        list(processed.time_frames),
        raw_volume,
        processed.segmentation_files,
        segmentation_events=stage8.segmentation_events,
        sample_id=sample.sample_id,
    )
    save_lineage_result(result, paths.stage_directory(10))
    return result


def run_stage11(
    sample: FullDatasetSample,
    paths: SampleOutputPaths,
    config: FullPipelineConfig,
    processed: ProcessedState,
    stage7,
    stage8,
    stage10,
):
    if not processed.segmentation_files:
        raise ValueError("Stage 11 requires at least one segmentation frame")
    spatial_shape_zyx = tuple(
        int(value)
        for value in np.load(
            processed.segmentation_files[0],
            mmap_mode="r",
            allow_pickle=False,
        ).shape
    )
    reconciliation_config = TrackReconciliationConfig(
        policy=config.reconciliation_policy
    )
    result = run_track_reconciliation(
        stage8.tracks,
        list(processed.time_frames),
        segmentation_events=stage8.segmentation_events,
        division_events=stage10.division_events,
        lineage_edges=stage10.lineage_edges,
        track_lineage=stage10.track_lineage,
        protected_tracks=stage10.protected_tracks,
        global_motion=stage7.global_motion,
        association_events=stage7.association_events,
        association_candidates=stage7.association_candidates,
        spatial_shape_zyx=spatial_shape_zyx,
        sample_id=sample.sample_id,
        config=reconciliation_config,
    )
    save_track_reconciliation_result(result, paths.stage_directory(11))
    return result


__all__ = [
    "PipelineState",
    "ProcessedState",
    "Stage7State",
    "Stage8State",
    "Stage10State",
    "load_processed_state",
    "load_stage7_state",
    "load_stage8_state",
    "load_stage10_state",
    "run_stage6",
    "run_stage7",
    "run_stage8",
    "run_stage10",
    "run_stage11",
]
