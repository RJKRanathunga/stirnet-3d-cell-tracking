"""Cell matching migrated from visualization notebook cell 7."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def assign_nearest_cell_ids(
    tracks: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    return_distances: bool = False,
):
    result = tracks.copy()
    result["cell_id"] = -1
    distances_by_index = pd.Series(np.nan, index=result.index, dtype=float)
    for frame, frame_tracks in result.groupby("frame"):
        frame_cells = cells[cells["frame"] == frame]
        if frame_cells.empty:
            continue
        tree = cKDTree(
            frame_cells[["centroid_z", "centroid_y", "centroid_x"]].to_numpy()
        )
        distances, indices = tree.query(frame_tracks[["z", "y", "x"]].to_numpy())
        result.loc[frame_tracks.index, "cell_id"] = frame_cells[
            "cell_id"
        ].to_numpy()[indices]
        distances_by_index.loc[frame_tracks.index] = distances
    if return_distances:
        return result, distances_by_index
    return result
