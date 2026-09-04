"""Track endpoint extraction and boundary classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


VOXEL_SIZE_ZYX = (1.625, 0.40625, 0.40625)
BOUNDARY_MARGIN_UM = 4.0

@dataclass(frozen=True)
class EndpointTrackGroups:
    track_summary: pd.DataFrame
    new_track_endpoints: pd.DataFrame
    ended_track_endpoints: pd.DataFrame
    new_failure_tracks: pd.DataFrame
    ended_failure_tracks: pd.DataFrame
    boundary_entry_tracks: pd.DataFrame
    boundary_exit_tracks: pd.DataFrame


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


def prepare_endpoint_track_groups(
    tracks: pd.DataFrame,
    cells: pd.DataFrame,
    spatial_shape_zyx,
    *,
    voxel_size_zyx=VOXEL_SIZE_ZYX,
    boundary_margin_um: float = BOUNDARY_MARGIN_UM,
) -> EndpointTrackGroups:
    """Reproduce the notebook's boundary-filtered birth/death grouping."""

    track_summary = (
        tracks.groupby("track_id")
        .agg(first_frame=("frame", "min"), last_frame=("frame", "max"))
        .reset_index()
    )
    first_frame = tracks["frame"].min()
    last_frame = tracks["frame"].max()
    all_new_track_ids = track_summary.loc[
        track_summary["first_frame"] > first_frame, "track_id"
    ]
    all_ended_track_ids = track_summary.loc[
        track_summary["last_frame"] < last_frame, "track_id"
    ]
    new_endpoints = get_track_endpoints(tracks, all_new_track_ids, "start")
    ended_endpoints = get_track_endpoints(tracks, all_ended_track_ids, "end")
    new_endpoints = classify_boundary_endpoints(
        new_endpoints,
        cells,
        spatial_shape_zyx,
        voxel_size_zyx,
        boundary_margin_um,
    )
    ended_endpoints = classify_boundary_endpoints(
        ended_endpoints,
        cells,
        spatial_shape_zyx,
        voxel_size_zyx,
        boundary_margin_um,
    )
    boundary_entry_ids = new_endpoints.loc[
        new_endpoints["is_boundary_endpoint"], "track_id"
    ]
    boundary_exit_ids = ended_endpoints.loc[
        ended_endpoints["is_boundary_endpoint"], "track_id"
    ]
    new_failure_ids = new_endpoints.loc[
        ~new_endpoints["is_boundary_endpoint"], "track_id"
    ]
    ended_failure_ids = ended_endpoints.loc[
        ~ended_endpoints["is_boundary_endpoint"], "track_id"
    ]

    def selected(track_ids) -> pd.DataFrame:
        return tracks[tracks["track_id"].isin(track_ids)].copy()

    return EndpointTrackGroups(
        track_summary=track_summary,
        new_track_endpoints=new_endpoints,
        ended_track_endpoints=ended_endpoints,
        new_failure_tracks=selected(new_failure_ids),
        ended_failure_tracks=selected(ended_failure_ids),
        boundary_entry_tracks=selected(boundary_entry_ids),
        boundary_exit_tracks=selected(boundary_exit_ids),
    )


def to_napari_tracks(tracks: pd.DataFrame) -> np.ndarray:
    return tracks[["track_id", "frame", "z", "y", "x"]].to_numpy(dtype=float)


def to_napari_points(tracks: pd.DataFrame) -> np.ndarray:
    return tracks[["frame", "z", "y", "x"]].to_numpy(dtype=float)
