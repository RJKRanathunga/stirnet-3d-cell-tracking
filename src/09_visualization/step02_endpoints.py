"""Track endpoint extraction and boundary classification."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_track_endpoints(tracks_df, track_ids, endpoint):
    selected = tracks_df[tracks_df["track_id"].isin(track_ids)]
    if selected.empty:
        return selected.copy()
    grouped_frames = selected.groupby("track_id")["frame"]
    if endpoint == "start":
        row_indices = grouped_frames.idxmin()
    elif endpoint == "end":
        row_indices = grouped_frames.idxmax()
    else:
        raise ValueError("endpoint must be either 'start' or 'end'")
    return selected.loc[row_indices].sort_values("track_id").reset_index(drop=True)


def classify_boundary_endpoints(
    endpoint_rows,
    cells_df,
    spatial_shape_zyx,
    voxel_size_zyx,
    boundary_margin_um,
):
    result = endpoint_rows.copy()
    if result.empty:
        result["boundary_distance_um"] = pd.Series(dtype=float)
        result["is_boundary_endpoint"] = pd.Series(dtype=bool)
        return result
    voxel_size = np.asarray(voxel_size_zyx, dtype=float)
    spatial_shape = np.asarray(spatial_shape_zyx, dtype=float)
    coordinates = result[["z", "y", "x"]].to_numpy(dtype=float)
    lower = coordinates * voxel_size
    upper = (spatial_shape - 1 - coordinates) * voxel_size
    centroid_distance = np.minimum(lower, upper).min(axis=1)
    bbox_columns = ["z_min", "y_min", "x_min", "z_max", "y_max", "x_max"]
    bbox_available = (
        "cell_id" in result.columns
        and "frame" in cells_df.columns
        and "cell_id" in cells_df.columns
        and set(bbox_columns).issubset(cells_df.columns)
    )
    if bbox_available:
        lookup = cells_df[["frame", "cell_id", *bbox_columns]].drop_duplicates(
            subset=["frame", "cell_id"]
        )
        result = result.merge(lookup, on=["frame", "cell_id"], how="left")
        bbox_min = result[["z_min", "y_min", "x_min"]].to_numpy(dtype=float)
        bbox_max = result[["z_max", "y_max", "x_max"]].to_numpy(dtype=float)
        bbox_distance = np.minimum(
            bbox_min * voxel_size,
            (spatial_shape - bbox_max) * voxel_size,
        ).min(axis=1)
        result["boundary_distance_um"] = np.where(
            np.isfinite(bbox_distance), bbox_distance, centroid_distance
        )
    else:
        result["boundary_distance_um"] = centroid_distance
    result["is_boundary_endpoint"] = (
        result["boundary_distance_um"] <= boundary_margin_um
    )
    return result


def to_napari_tracks(tracks: pd.DataFrame) -> np.ndarray:
    return tracks[["track_id", "frame", "z", "y", "x"]].to_numpy(dtype=float)


def to_napari_points(tracks: pd.DataFrame) -> np.ndarray:
    return tracks[["frame", "z", "y", "x"]].to_numpy(dtype=float)
