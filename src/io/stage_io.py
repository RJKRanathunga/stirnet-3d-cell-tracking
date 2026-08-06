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
class Stage7Outputs:
    tracks: pd.DataFrame
    boundary_events: pd.DataFrame
    boundary_predictions: pd.DataFrame
    missing_predictions: pd.DataFrame
    boundary_counts: pd.DataFrame
    global_motion: pd.DataFrame
    tracking_diagnostics: pd.DataFrame
    association_events: pd.DataFrame
    association_candidates: pd.DataFrame
    track_states: pd.DataFrame
    graph_transition_summary: pd.DataFrame
    graph_candidate_evidence: pd.DataFrame
    graph_anchor_votes: pd.DataFrame
    graph_boundary_hypotheses: pd.DataFrame
    graph_refinement_events: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Stage10Outputs:
    division_candidates: pd.DataFrame
    division_events: pd.DataFrame
    lineage_edges: pd.DataFrame
    track_lineage: pd.DataFrame
    protected_tracks: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Stage11Outputs:
    tracks: pd.DataFrame
    segmentation_events: pd.DataFrame
    division_events: pd.DataFrame
    lineage_edges: pd.DataFrame
    track_lineage: pd.DataFrame
    protected_tracks: pd.DataFrame
    endpoint_classifications: pd.DataFrame
    continuation_candidates: pd.DataFrame
    continuation_decisions: pd.DataFrame
    track_id_remap: pd.DataFrame
    unresolved_endings: pd.DataFrame
    validation_results: pd.DataFrame
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


def load_stage7_outputs(*, paths: PipelinePaths | None = None) -> Stage7Outputs:
    """Load Stage 7 tracks plus optional diagnostic evidence tables."""

    resolved = paths or PipelinePaths.discover()
    root = resolved.stage7_tracking
    return Stage7Outputs(
        tracks=load_csv(root / "tracks.csv", required_columns=TRACK_COLUMNS),
        boundary_events=load_optional_csv(root / "boundary_events.csv"),
        boundary_predictions=load_optional_csv(root / "boundary_predictions.csv"),
        missing_predictions=load_optional_csv(root / "missing_predictions.csv"),
        boundary_counts=load_optional_csv(root / "boundary_detection_counts.csv"),
        global_motion=load_optional_csv(root / "global_motion.csv"),
        tracking_diagnostics=load_optional_csv(root / "tracking_diagnostics.csv"),
        association_events=load_optional_csv(root / "association_events.csv"),
        association_candidates=load_optional_csv(root / "association_candidates.csv"),
        track_states=load_optional_csv(root / "track_states.csv"),
        graph_transition_summary=load_optional_csv(
            root / "graph_transition_summary.csv"
        ),
        graph_candidate_evidence=load_optional_csv(
            root / "graph_candidate_evidence.csv"
        ),
        graph_anchor_votes=load_optional_csv(
            root / "graph_anchor_votes.csv"
        ),
        graph_boundary_hypotheses=load_optional_csv(
            root / "graph_boundary_hypotheses.csv"
        ),
        graph_refinement_events=load_optional_csv(
            root / "graph_refinement_events.csv"
        ),
        metadata=load_json(root / "metadata.json"),
    )


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
        "graph_transition_summary": "graph_transition_summary.csv",
        "graph_candidate_evidence": "graph_candidate_evidence.csv",
        "graph_anchor_votes": "graph_anchor_votes.csv",
        "graph_boundary_hypotheses": "graph_boundary_hypotheses.csv",
        "graph_refinement_events": "graph_refinement_events.csv",
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


def save_track_reconciliation_result(result, directory: str | Path) -> None:
    """Save stable Stage 11 tables without modifying Stage 8 or Stage 10."""

    root = Path(directory)
    mapping = {
        "tracks": "tracks.csv",
        "segmentation_events": "segmentation_events.csv",
        "division_events": "division_events.csv",
        "lineage_edges": "lineage_edges.csv",
        "track_lineage": "track_lineage.csv",
        "protected_tracks": "protected_tracks.csv",
        "endpoint_classifications": "endpoint_classifications.csv",
        "continuation_candidates": "continuation_candidates.csv",
        "continuation_decisions": "continuation_decisions.csv",
        "track_id_remap": "track_id_remap.csv",
        "unresolved_endings": "unresolved_endings.csv",
        "validation_results": "validation_results.csv",
    }
    for attribute, filename in mapping.items():
        save_csv(getattr(result, attribute), root / filename)
    save_json(result.metadata, root / "metadata.json")


def load_stage11_outputs(*, paths: PipelinePaths | None = None) -> Stage11Outputs:
    """Load and validate all stable Stage 11 artifact schemas."""

    resolved = paths or PipelinePaths.discover()
    root = resolved.stage11_reconciliation
    schemas = import_module("src.11_track_reconciliation.step01_config")
    lineage_schemas = import_module("src.10_cell_lineage.step01_config")
    return Stage11Outputs(
        tracks=load_csv(root / "tracks.csv", required_columns=TRACK_COLUMNS),
        segmentation_events=load_optional_csv(root / "segmentation_events.csv"),
        division_events=load_csv(
            root / "division_events.csv",
            required_columns=lineage_schemas.DIVISION_EVENT_COLUMNS,
        ),
        lineage_edges=load_csv(
            root / "lineage_edges.csv",
            required_columns=lineage_schemas.LINEAGE_EDGE_COLUMNS,
        ),
        track_lineage=load_csv(
            root / "track_lineage.csv",
            required_columns=lineage_schemas.TRACK_LINEAGE_COLUMNS,
        ),
        protected_tracks=load_csv(
            root / "protected_tracks.csv",
            required_columns=lineage_schemas.PROTECTED_TRACK_COLUMNS,
        ),
        endpoint_classifications=load_csv(
            root / "endpoint_classifications.csv",
            required_columns=schemas.ENDPOINT_CLASSIFICATION_COLUMNS,
        ),
        continuation_candidates=load_csv(
            root / "continuation_candidates.csv",
            required_columns=schemas.CONTINUATION_CANDIDATE_COLUMNS,
        ),
        continuation_decisions=load_csv(
            root / "continuation_decisions.csv",
            required_columns=schemas.CONTINUATION_DECISION_COLUMNS,
        ),
        track_id_remap=load_csv(
            root / "track_id_remap.csv",
            required_columns=schemas.TRACK_ID_REMAP_COLUMNS,
        ),
        unresolved_endings=load_csv(
            root / "unresolved_endings.csv",
            required_columns=schemas.UNRESOLVED_ENDING_COLUMNS,
        ),
        validation_results=load_csv(
            root / "validation_results.csv",
            required_columns=schemas.VALIDATION_RESULT_COLUMNS,
        ),
        metadata=load_json(root / "metadata.json"),
    )
