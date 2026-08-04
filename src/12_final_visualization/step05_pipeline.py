"""Stable Stage 12 preparation entry point."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.diagnostics import Provenance, StageTrace

from .step01_config import FinalVisualizationConfig
from .step02_matching import prepare_visualization_tracks
from .step03_endpoint_audit import audit_final_tracks
from .step04_napari_data import (
    FinalTrackGroups,
    build_track_groups,
    to_napari_points,
    to_napari_tracks,
)


@dataclass(frozen=True)
class FinalVisualizationData:
    tracks: pd.DataFrame
    track_summary: pd.DataFrame
    diagnostic_events: pd.DataFrame
    failure_events: pd.DataFrame
    warning_events: pd.DataFrame
    groups: FinalTrackGroups
    tracks_array: np.ndarray
    points_array: np.ndarray
    track_ids: np.ndarray
    cell_ids: np.ndarray
    original_to_canonical: dict[int, int]
    lineage_graph: dict[int, list[int]]
    voxel_size_zyx: tuple[float, float, float]
    sequence_first_frame: int
    sequence_last_frame: int
    summary: dict[str, object]


def _build_lineage_graph(
    lineage_edges: pd.DataFrame | None, final_track_ids: set[int]
) -> dict[int, list[int]]:
    if lineage_edges is None or lineage_edges.empty:
        return {}
    required = {"parent_track_id", "child_track_id"}
    missing = sorted(required - set(lineage_edges.columns))
    if missing:
        raise ValueError(f"lineage_edges is missing required columns: {missing}")
    graph: dict[int, set[int]] = {}
    for row in lineage_edges.itertuples(index=False):
        parent = int(row.parent_track_id)
        child = int(row.child_track_id)
        if parent in final_track_ids and child in final_track_ids and parent != child:
            graph.setdefault(child, set()).add(parent)
    return {child: sorted(parents) for child, parents in sorted(graph.items())}


def prepare_final_visualization_data(
    tracks: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    endpoint_classifications: pd.DataFrame | None = None,
    continuation_decisions: pd.DataFrame | None = None,
    track_id_remap: pd.DataFrame | None = None,
    unresolved_endings: pd.DataFrame | None = None,
    division_events: pd.DataFrame | None = None,
    lineage_edges: pd.DataFrame | None = None,
    segmentation_events: pd.DataFrame | None = None,
    spatial_shape_zyx: tuple[int, int, int] | None = None,
    sequence_first_frame: int = 0,
    sequence_last_frame: int | None = None,
    config: FinalVisualizationConfig | None = None,
    return_diagnostics: bool = False,
):
    """Prepare final Stage 11 outputs for Napari and residual-failure review."""

    resolved = config or FinalVisualizationConfig()
    prepared, match_distances = prepare_visualization_tracks(tracks, cells)
    if prepared.empty:
        raise ValueError("tracks must contain at least one final observation")
    if sequence_last_frame is None:
        sequence_last_frame = int(prepared["frame"].max())
    sequence_first_frame = int(sequence_first_frame)
    sequence_last_frame = int(sequence_last_frame)
    if sequence_last_frame < sequence_first_frame:
        raise ValueError("sequence_last_frame must not be before sequence_first_frame")
    if spatial_shape_zyx is not None:
        if len(spatial_shape_zyx) != 3 or any(int(value) <= 0 for value in spatial_shape_zyx):
            raise ValueError("spatial_shape_zyx must contain three positive dimensions")

    track_summary, events, mapping = audit_final_tracks(
        prepared,
        cells,
        spatial_shape_zyx=spatial_shape_zyx,
        sequence_first_frame=sequence_first_frame,
        sequence_last_frame=sequence_last_frame,
        endpoint_classifications=endpoint_classifications,
        continuation_decisions=continuation_decisions,
        track_id_remap=track_id_remap,
        unresolved_endings=unresolved_endings,
        division_events=division_events,
        segmentation_events=segmentation_events,
        config=resolved,
    )
    groups = build_track_groups(prepared, track_summary)
    lineage_graph = _build_lineage_graph(
        lineage_edges, set(prepared["track_id"].astype(int).unique())
    )
    failure_events = events[events["is_failure"].astype(bool)].copy()
    warning_events = events[events["severity"] == "warning"].copy()
    summary = {
        "final_track_count": int(prepared["track_id"].nunique()),
        "final_observation_count": int(len(prepared)),
        "suspicious_birth_count": int(track_summary["suspicious_start"].sum()),
        "suspicious_termination_count": int(track_summary["suspicious_end"].sum()),
        "boundary_entry_count": int((track_summary["start_classification"] == "boundary_entry").sum()),
        "boundary_exit_count": int((track_summary["end_classification"] == "boundary_exit").sum()),
        "division_related_track_count": int(track_summary["division_related"].sum()),
        "merge_related_track_count": int(track_summary["merge_related"].sum()),
        "stage11_modified_track_count": int(track_summary["stage11_modified"].sum()),
        "forced_repair_track_count": int((track_summary["forced_repair_count"] > 0).sum()),
        "unresolved_ending_count": int(track_summary["unresolved_ending"].sum()),
        "short_lived_track_count": int(track_summary["short_lived"].sum()),
        "temporal_gap_track_count": int(track_summary["has_temporal_gap"].sum()),
        "failure_event_count": int(len(failure_events)),
        "warning_event_count": int(len(warning_events)),
        "unmatched_cell_id_count": int((prepared["cell_id"] < 0).sum()),
    }
    result = FinalVisualizationData(
        tracks=prepared,
        track_summary=track_summary,
        diagnostic_events=events,
        failure_events=failure_events,
        warning_events=warning_events,
        groups=groups,
        tracks_array=to_napari_tracks(prepared),
        points_array=to_napari_points(prepared),
        track_ids=prepared["track_id"].to_numpy(dtype=int),
        cell_ids=prepared["cell_id"].to_numpy(dtype=int),
        original_to_canonical=mapping,
        lineage_graph=lineage_graph,
        voxel_size_zyx=tuple(float(value) for value in resolved.voxel_size_zyx_um),
        sequence_first_frame=sequence_first_frame,
        sequence_last_frame=sequence_last_frame,
        summary=summary,
    )
    if not return_diagnostics:
        return result

    trace = StageTrace(
        stage_name="12_final_visualization",
        inputs={
            "tracks": tracks,
            "cells": cells,
            "endpoint_classifications": endpoint_classifications,
            "track_id_remap": track_id_remap,
            "unresolved_endings": unresolved_endings,
        },
        outputs={
            "track_summary": track_summary,
            "diagnostic_events": events,
            "failure_events": failure_events,
        },
        intermediates={
            "prepared_tracks": prepared,
            "cell_match_distances": match_distances,
            "original_to_canonical": mapping,
            "lineage_graph": lineage_graph,
        },
        metrics=summary.copy(),
        provenance={
            f"event:{row.event_id}": Provenance(
                source_type="final_tracking_diagnostic",
                source_stage="12_final_visualization",
                source_frame=int(row.frame),
                source_track_ids=(int(row.track_id),),
                details={
                    "event_type": str(row.event_type),
                    "classification": str(row.classification),
                    "reason": str(row.reason),
                },
            )
            for row in events.itertuples(index=False)
        },
    )
    return result, trace
