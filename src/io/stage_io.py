"""Stage-oriented loaders and savers for the existing artifact layout."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd

from .arrays import list_timepoint_files, load_npy, save_npy
from .paths import PipelinePaths
from .tables import load_csv, load_json, load_optional_csv, save_csv, save_json


CELL_COLUMNS = (
    "cell_id", "volume_voxels", "z_min", "y_min", "x_min",
    "z_max", "y_max", "x_max", "centroid_z", "centroid_y", "centroid_x",
)
TRACK_COLUMNS = ("track_id", "frame", "cell", "cell_id", "z", "y", "x", "volume")


@dataclass(frozen=True)
class ProcessedDatasetInputs:
    root: Path
    cell_files: tuple[Path, ...]
    segmentation_files: tuple[Path, ...]
    time_frames: tuple[pd.DataFrame, ...]


@dataclass(frozen=True)
class Stage8Outputs:
    detections: pd.DataFrame
    tracks: pd.DataFrame
    segmentation_events: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Stage10Outputs:
    division_candidates: pd.DataFrame
    division_events: pd.DataFrame
    lineage_edges: pd.DataFrame
    track_lineage: pd.DataFrame
    protected_tracks: pd.DataFrame
    metadata: dict[str, Any]


def load_processed_dataset_inputs(
    sample_id: str,
    *,
    paths: PipelinePaths | None = None,
) -> ProcessedDatasetInputs:
    resolved = paths or PipelinePaths.discover()
    root = resolved.processed_dataset(sample_id)
    cell_files = tuple(list_timepoint_files(root / "cells", "csv"))
    segmentation_files = tuple(list_timepoint_files(root / "segmentation", "npy"))
    if len(cell_files) != len(segmentation_files):
        raise ValueError(
            "The number of segmentation volumes does not match the number of cell tables: "
            f"{len(segmentation_files)} vs {len(cell_files)}"
        )
    frames = tuple(load_csv(path, required_columns=CELL_COLUMNS) for path in cell_files)
    return ProcessedDatasetInputs(root, cell_files, segmentation_files, frames)


def load_stage7_detections(
    sample_id: str,
    *,
    paths: PipelinePaths | None = None,
) -> list[pd.DataFrame]:
    return list(load_processed_dataset_inputs(sample_id, paths=paths).time_frames)


def load_stage8_outputs(*, paths: PipelinePaths | None = None) -> Stage8Outputs:
    resolved = paths or PipelinePaths.discover()
    root = resolved.stage8_stitching
    return Stage8Outputs(
        detections=load_csv(root / "detections.csv", required_columns=("frame", *CELL_COLUMNS)),
        tracks=load_csv(root / "tracks.csv", required_columns=TRACK_COLUMNS),
        segmentation_events=load_optional_csv(root / "segmentation_events.csv"),
        metadata=load_json(root / "metadata.json"),
    )


def save_processed_frame(
    root: str | Path,
    frame: int,
    *,
    preprocessed,
    binary_mask,
    instance_labels,
    cells: pd.DataFrame,
) -> None:
    directory = Path(root)
    stem = f"t{int(frame):03d}"
    save_npy(preprocessed, directory / "preprocessing" / f"{stem}.npy")
    save_npy(binary_mask, directory / "masking" / f"{stem}.npy")
    save_npy(instance_labels, directory / "segmentation" / f"{stem}.npy")
    save_csv(cells, directory / "cells" / f"{stem}.csv")


def save_tracking_result(result, directory: str | Path) -> None:
    root = Path(directory)
    mapping = {
        "tracks": "tracks.csv",
        "boundary_events": "boundary_events.csv",
        "boundary_predictions": "boundary_predictions.csv",
        "missing_predictions": "missing_predictions.csv",
        "boundary_counts": "boundary_detection_counts.csv",
        "global_motion": "global_motion.csv",
        "tracking_diagnostics": "tracking_diagnostics.csv",
        "association_events": "association_events.csv",
        "association_candidates": "association_candidates.csv",
        "track_states": "track_states.csv",
    }
    for attribute, filename in mapping.items():
        save_csv(getattr(result, attribute), root / filename)
    save_json(result.metadata, root / "metadata.json")


def save_stitching_result(result, directory: str | Path) -> None:
    root = Path(directory)
    mapping = {
        "detections": "detections.csv",
        "tracks": "tracks.csv",
        "merge_onset_candidates": "merge_onset_candidates.csv",
        "merge_onsets": "merge_onsets.csv",
        "segmentation_events": "segmentation_events.csv",
        "merge_center_trajectories": "merge_center_trajectories.csv",
        "merge_split_links": "merge_split_links.csv",
        "merge_track_repairs": "merge_track_repairs.csv",
        "merge_trace_failures": "merge_trace_failures.csv",
    }
    for attribute, filename in mapping.items():
        save_csv(getattr(result, attribute), root / filename)
    save_json(result.metadata, root / "metadata.json")


def save_lineage_result(result, directory: str | Path) -> None:
    """Save the five stable Stage 10 tables and metadata object."""

    root = Path(directory)
    mapping = {
        "division_candidates": "division_candidates.csv",
        "division_events": "division_events.csv",
        "lineage_edges": "lineage_edges.csv",
        "track_lineage": "track_lineage.csv",
        "protected_tracks": "protected_tracks.csv",
    }
    for attribute, filename in mapping.items():
        save_csv(getattr(result, attribute), root / filename)
    save_json(result.metadata, root / "metadata.json")


def load_stage10_outputs(*, paths: PipelinePaths | None = None) -> Stage10Outputs:
    """Load Stage 10 artifacts while validating every public table schema."""

    resolved = paths or PipelinePaths.discover()
    root = resolved.stage10_lineage
    schemas = import_module("src.10_cell_lineage.step01_config")
    return Stage10Outputs(
        division_candidates=load_csv(
            root / "division_candidates.csv",
            required_columns=schemas.DIVISION_CANDIDATE_COLUMNS,
        ),
        division_events=load_csv(
            root / "division_events.csv",
            required_columns=schemas.DIVISION_EVENT_COLUMNS,
        ),
        lineage_edges=load_csv(
            root / "lineage_edges.csv",
            required_columns=schemas.LINEAGE_EDGE_COLUMNS,
        ),
        track_lineage=load_csv(
            root / "track_lineage.csv",
            required_columns=schemas.TRACK_LINEAGE_COLUMNS,
        ),
        protected_tracks=load_csv(
            root / "protected_tracks.csv",
            required_columns=schemas.PROTECTED_TRACK_COLUMNS,
        ),
        metadata=load_json(root / "metadata.json"),
    )
