"""Stable visualization-data preparation entry point."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.diagnostics import Provenance, StageTrace

from .step01_matching import assign_nearest_cell_ids
from .step02_endpoints import to_napari_points, to_napari_tracks


@dataclass(frozen=True)
class VisualizationData:
    tracks: pd.DataFrame
    tracks_array: np.ndarray
    points_array: np.ndarray
    track_ids: np.ndarray


def prepare_visualization_data(
    tracks: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    assign_cell_ids: bool = True,
    return_diagnostics: bool = False,
):
    if assign_cell_ids:
        prepared, distances = assign_nearest_cell_ids(
            tracks, cells, return_distances=True
        )
    else:
        prepared = tracks.copy()
        distances = pd.Series(np.nan, index=prepared.index, dtype=float)
    result = VisualizationData(
        tracks=prepared,
        tracks_array=to_napari_tracks(prepared),
        points_array=to_napari_points(prepared),
        track_ids=prepared["track_id"].to_numpy(),
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
