"""Track validation and conservative cell-ID enrichment for visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


TRACK_REQUIRED_COLUMNS = ("track_id", "frame", "z", "y", "x")
CELL_CENTROID_COLUMNS = ("centroid_z", "centroid_y", "centroid_x")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def prepare_visualization_tracks(
    tracks: pd.DataFrame,
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate final tracks and fill only missing/invalid cell IDs.

    Stage 11 normally preserves ``cell_id``. Nearest-centroid matching is used only
    as a visualization fallback and its distance is returned for diagnostics.
    """

    if not isinstance(tracks, pd.DataFrame) or not isinstance(cells, pd.DataFrame):
        raise TypeError("tracks and cells must be pandas DataFrames")
    _require_columns(tracks, TRACK_REQUIRED_COLUMNS, "tracks")
    _require_columns(cells, ("frame", "cell_id", *CELL_CENTROID_COLUMNS), "cells")

    result = tracks.copy(deep=True)
    for column in ("track_id", "frame"):
        numeric = pd.to_numeric(result[column], errors="raise")
        if numeric.isna().any() or not np.allclose(numeric, np.round(numeric)):
            raise ValueError(f"tracks.{column} must contain finite integer values")
        result[column] = numeric.astype(int)
    if result.duplicated(["track_id", "frame"]).any():
        duplicate = result.loc[result.duplicated(["track_id", "frame"], keep=False), ["track_id", "frame"]]
        raise ValueError(
            "final tracks contain duplicate track/frame observations: "
            f"{duplicate.head(5).to_dict('records')}"
        )
    if "cell_id" not in result.columns:
        result["cell_id"] = -1
    result["cell_id"] = pd.to_numeric(result["cell_id"], errors="coerce").fillna(-1).astype(int)
    distances = pd.Series(np.nan, index=result.index, dtype=float)

    cell_ids_by_frame = {
        int(frame): set(pd.to_numeric(group["cell_id"], errors="coerce").dropna().astype(int))
        for frame, group in cells.groupby("frame", sort=False)
    }
    invalid = result.apply(
        lambda row: int(row.cell_id) < 0
        or int(row.cell_id) not in cell_ids_by_frame.get(int(row.frame), set()),
        axis=1,
    )
    for frame, frame_tracks in result.loc[invalid].groupby("frame", sort=False):
        frame_cells = cells[cells["frame"] == frame]
        if frame_cells.empty:
            continue
        coordinates = frame_cells[list(CELL_CENTROID_COLUMNS)].to_numpy(dtype=float)
        finite = np.isfinite(coordinates).all(axis=1)
        frame_cells = frame_cells.loc[finite]
        coordinates = coordinates[finite]
        if len(frame_cells) == 0:
            continue
        tree = cKDTree(coordinates)
        query = frame_tracks[["z", "y", "x"]].to_numpy(dtype=float)
        distance, indices = tree.query(query)
        result.loc[frame_tracks.index, "cell_id"] = frame_cells["cell_id"].to_numpy()[indices]
        distances.loc[frame_tracks.index] = distance

    return result.sort_values(["track_id", "frame"], kind="stable").reset_index(drop=True), distances
