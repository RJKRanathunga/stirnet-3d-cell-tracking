"""Stable visualization-data preparation entry point."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.diagnostics import Provenance, StageTrace

from .step01_matching import assign_nearest_cell_ids
from .step02_endpoints import (
    BOUNDARY_MARGIN_UM,
    VOXEL_SIZE_ZYX,
    EndpointTrackGroups,
    prepare_endpoint_track_groups,
    to_napari_points,
    to_napari_tracks,
)


@dataclass(frozen=True)
class VisualizationData:
    tracks: pd.DataFrame
    tracks_array: np.ndarray
    points_array: np.ndarray
    track_ids: np.ndarray
    cell_ids: np.ndarray
    endpoint_groups: EndpointTrackGroups | None
    voxel_size_zyx: tuple[float, float, float]


def prepare_visualization_data(
    tracks: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    assign_cell_ids: bool = True,
    spatial_shape_zyx=None,
    voxel_size_zyx=VOXEL_SIZE_ZYX,
    boundary_margin_um: float = BOUNDARY_MARGIN_UM,
    return_diagnostics: bool = False,
):
    if assign_cell_ids:
        prepared, distances = assign_nearest_cell_ids(
            tracks, cells, return_distances=True
        )
    else:
        prepared = tracks.copy()
        distances = pd.Series(np.nan, index=prepared.index, dtype=float)
    endpoint_groups = None
    if spatial_shape_zyx is not None:
        endpoint_groups = prepare_endpoint_track_groups(
            prepared,
            cells,
            spatial_shape_zyx,
            voxel_size_zyx=voxel_size_zyx,
            boundary_margin_um=boundary_margin_um,
        )
    result = VisualizationData(
        tracks=prepared,
        tracks_array=to_napari_tracks(prepared),
        points_array=to_napari_points(prepared),
        track_ids=prepared["track_id"].to_numpy(),
        cell_ids=prepared["cell_id"].to_numpy(),
        endpoint_groups=endpoint_groups,
        voxel_size_zyx=tuple(float(value) for value in voxel_size_zyx),
    )
    if not return_diagnostics:
        return result
    trace = StageTrace(
        stage_name="09_visualization",
        inputs={"tracks": tracks, "cells": cells},
        outputs={"tracks_array": result.tracks_array, "points_array": result.points_array},
        intermediates={"cell_match_distances": distances},
        metrics={
            "track_rows": len(prepared),
            "unmatched_rows": int((prepared["cell_id"] == -1).sum()),
            "maximum_match_distance": (
                float(distances.max()) if distances.notna().any() else np.nan
            ),
            "new_failure_tracks": (
                int(result.endpoint_groups.new_failure_tracks["track_id"].nunique())
                if result.endpoint_groups is not None else 0
            ),
            "ended_failure_tracks": (
                int(result.endpoint_groups.ended_failure_tracks["track_id"].nunique())
                if result.endpoint_groups is not None else 0
            ),
            "boundary_entry_tracks": (
                int(result.endpoint_groups.boundary_entry_tracks["track_id"].nunique())
                if result.endpoint_groups is not None else 0
            ),
            "boundary_exit_tracks": (
                int(result.endpoint_groups.boundary_exit_tracks["track_id"].nunique())
                if result.endpoint_groups is not None else 0
            ),
        },
        provenance={
            f"row:{index}": Provenance(
                source_type="visualization_nearest_cell_match",
                source_stage="09_visualization",
                source_frame=int(row.frame),
                source_cell_id=(int(row.cell_id) if int(row.cell_id) >= 0 else None),
                source_track_ids=(int(row.track_id),),
                details={"distance_voxels": distances.loc[index]},
            )
            for index, row in prepared.iterrows()
        } if return_diagnostics else {},
    )
    return result, trace
