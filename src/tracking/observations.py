"""Conversions between tracker coordinates and source-instance observations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def assign_nearest_instance_ids(
    tracks: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    output_column: str = "cell_id",
    return_distances: bool = False,
):
    """Attach the nearest source-instance id to each tracker observation."""
    required_tracks = {"frame", "z", "y", "x"}
    required_cells = {
        "frame", "cell_id", "centroid_z", "centroid_y", "centroid_x"
    }
    missing_tracks = required_tracks - set(tracks.columns)
    missing_cells = required_cells - set(cells.columns)
    if missing_tracks:
        raise ValueError(f"tracks missing required columns: {sorted(missing_tracks)}")
    if missing_cells:
        raise ValueError(f"cells missing required columns: {sorted(missing_cells)}")

    result = tracks.copy()
    result[output_column] = -1
    distances_by_index = pd.Series(np.nan, index=result.index, dtype=float)
    for frame, frame_tracks in result.groupby("frame"):
        frame_cells = cells[cells["frame"] == frame]
        if frame_cells.empty:
            continue
        tree = cKDTree(
            frame_cells[["centroid_z", "centroid_y", "centroid_x"]].to_numpy()
        )
        distances, indices = tree.query(
            frame_tracks[["z", "y", "x"]].to_numpy()
        )
        result.loc[frame_tracks.index, output_column] = (
            frame_cells["cell_id"].to_numpy()[indices]
        )
        distances_by_index.loc[frame_tracks.index] = distances
    if return_distances:
        return result, distances_by_index
    return result


__all__ = ["assign_nearest_instance_ids"]
