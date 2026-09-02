from __future__ import annotations

"""Notebook-09-equivalent Trackastra endpoint diagnostics for curation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataset_curation.annotation.tracks.graph import AnnotationError, Node
from dataset_curation.config import DEFAULT_SPACING_ZYX_UM


BOUNDARY_MARGIN_UM = 4.0

CATEGORY_COLORS = {
    "default": "white",
    "broken": "red",
    "new": "lime",
    "boundary_entry": "cyan",
    "boundary_exit": "orange",
}


@dataclass(frozen=True)
class EndpointTrackGroups:
    track_summary: pd.DataFrame
    new_track_endpoints: pd.DataFrame
    ended_track_endpoints: pd.DataFrame
    new_failure_tracks: pd.DataFrame
    ended_failure_tracks: pd.DataFrame
    boundary_entry_tracks: pd.DataFrame
    boundary_exit_tracks: pd.DataFrame


def normalize_cells(cells: pd.DataFrame) -> pd.DataFrame:
    required = {
        "frame",
        "cell_id",
        "centroid_z",
        "centroid_y",
        "centroid_x",
    }
    missing = sorted(required - set(cells.columns))
    if missing:
        raise AnnotationError(
            f"cells_all.csv is missing columns: {missing}"
        )

    result = cells.copy()
    result["frame"] = result["frame"].astype(np.int64)
    result["cell_id"] = result["cell_id"].astype(np.int64)

    duplicate = result.duplicated(
        ["frame", "cell_id"],
        keep=False,
    )
    if duplicate.any():
        rows = result.loc[
            duplicate,
            ["frame", "cell_id"],
        ].head(10)
        raise AnnotationError(
            "cells_all.csv contains duplicate (frame, cell_id) detections:\n"
            + rows.to_string(index=False)
        )
    return result


def normalize_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    required = {
        "track_id",
        "frame",
        "cell_id",
        "z",
        "y",
        "x",
    }
    missing = sorted(required - set(tracks.columns))
    if missing:
        raise AnnotationError(
            f"trackastra/tracks.csv is missing columns: {missing}"
        )

    result = tracks.copy()
    result["track_id"] = result["track_id"].astype(np.int64)
    result["frame"] = result["frame"].astype(np.int64)
    result["cell_id"] = result["cell_id"].astype(np.int64)
    return result.loc[
        result["cell_id"] > 0
    ].copy()


def _get_track_endpoints(
    tracks: pd.DataFrame,
    track_ids,
    endpoint: str,
) -> pd.DataFrame:
    selected = tracks[
        tracks["track_id"].isin(track_ids)
    ]
    if selected.empty:
        return selected.copy()

    grouped = selected.groupby("track_id")["frame"]
    if endpoint == "start":
        row_indices = grouped.idxmin()
    elif endpoint == "end":
        row_indices = grouped.idxmax()
    else:
        raise ValueError(
            "endpoint must be 'start' or 'end'"
        )
    return (
        selected.loc[row_indices]
        .sort_values("track_id")
        .reset_index(drop=True)
    )


def _classify_boundary_endpoints(
    endpoint_rows: pd.DataFrame,
    cells: pd.DataFrame,
    spatial_shape_zyx,
    *,
    voxel_size_zyx,
    boundary_margin_um: float,
) -> pd.DataFrame:
    result = endpoint_rows.copy()
    if result.empty:
        result["boundary_distance_um"] = pd.Series(dtype=float)
        result["is_boundary_endpoint"] = pd.Series(dtype=bool)
        return result

    voxel_size = np.asarray(
        voxel_size_zyx,
        dtype=np.float64,
    )
    spatial_shape = np.asarray(
        spatial_shape_zyx,
        dtype=np.float64,
    )
    coordinates = result[
        ["z", "y", "x"]
    ].to_numpy(dtype=np.float64)

    lower = coordinates * voxel_size
    upper = (
        spatial_shape
        - 1.0
        - coordinates
    ) * voxel_size
    centroid_distance = np.minimum(
        lower,
        upper,
    ).min(axis=1)

    bbox_columns = [
        "z_min",
        "y_min",
        "x_min",
        "z_max",
        "y_max",
        "x_max",
    ]
    bbox_available = (
        "cell_id" in result.columns
        and "frame" in cells.columns
        and "cell_id" in cells.columns
        and set(bbox_columns).issubset(cells.columns)
    )

    if bbox_available:
        lookup = cells[
            [
                "frame",
                "cell_id",
                *bbox_columns,
            ]
        ].drop_duplicates(
            subset=[
                "frame",
                "cell_id",
            ]
        )
        result = result.merge(
            lookup,
            on=[
                "frame",
                "cell_id",
            ],
            how="left",
        )
        bbox_min = result[
            ["z_min", "y_min", "x_min"]
        ].to_numpy(dtype=np.float64)
        bbox_max = result[
            ["z_max", "y_max", "x_max"]
        ].to_numpy(dtype=np.float64)
        bbox_distance = np.minimum(
            bbox_min * voxel_size,
            (
                spatial_shape
                - bbox_max
            ) * voxel_size,
        ).min(axis=1)

        result["boundary_distance_um"] = np.where(
            np.isfinite(bbox_distance),
            bbox_distance,
            centroid_distance,
        )
    else:
        result["boundary_distance_um"] = centroid_distance

    result["is_boundary_endpoint"] = (
        result["boundary_distance_um"]
        <= float(boundary_margin_um)
    )
    return result


def prepare_endpoint_track_groups(
    tracks: pd.DataFrame,
    cells: pd.DataFrame,
    spatial_shape_zyx,
    *,
    voxel_size_zyx=DEFAULT_SPACING_ZYX_UM,
    boundary_margin_um: float = BOUNDARY_MARGIN_UM,
) -> EndpointTrackGroups:
    """
    Reproduce notebooks/09_visualization.ipynb endpoint grouping.

    - prematurely ended non-boundary tracks -> broken
    - newly started non-boundary tracks -> new
    - boundary starts/exits stay separate
    """
    tracks = normalize_tracks(tracks)
    cells = normalize_cells(cells)

    if tracks.empty:
        empty = tracks.copy()
        return EndpointTrackGroups(
            track_summary=pd.DataFrame(
                columns=[
                    "track_id",
                    "first_frame",
                    "last_frame",
                ]
            ),
            new_track_endpoints=empty.copy(),
            ended_track_endpoints=empty.copy(),
            new_failure_tracks=empty.copy(),
            ended_failure_tracks=empty.copy(),
            boundary_entry_tracks=empty.copy(),
            boundary_exit_tracks=empty.copy(),
        )

    track_summary = (
        tracks.groupby("track_id")
        .agg(
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
        )
        .reset_index()
    )

    first_frame = int(tracks["frame"].min())
    last_frame = int(tracks["frame"].max())

    new_track_ids = track_summary.loc[
        track_summary["first_frame"] > first_frame,
        "track_id",
    ]
    ended_track_ids = track_summary.loc[
        track_summary["last_frame"] < last_frame,
        "track_id",
    ]

    new_endpoints = _get_track_endpoints(
        tracks,
        new_track_ids,
        "start",
    )
    ended_endpoints = _get_track_endpoints(
        tracks,
        ended_track_ids,
        "end",
    )

    new_endpoints = _classify_boundary_endpoints(
        new_endpoints,
        cells,
        spatial_shape_zyx,
        voxel_size_zyx=voxel_size_zyx,
        boundary_margin_um=boundary_margin_um,
    )
    ended_endpoints = _classify_boundary_endpoints(
        ended_endpoints,
        cells,
        spatial_shape_zyx,
        voxel_size_zyx=voxel_size_zyx,
        boundary_margin_um=boundary_margin_um,
    )

    boundary_entry_ids = new_endpoints.loc[
        new_endpoints["is_boundary_endpoint"],
        "track_id",
    ]
    boundary_exit_ids = ended_endpoints.loc[
        ended_endpoints["is_boundary_endpoint"],
        "track_id",
    ]
    new_failure_ids = new_endpoints.loc[
        ~new_endpoints["is_boundary_endpoint"],
        "track_id",
    ]
    ended_failure_ids = ended_endpoints.loc[
        ~ended_endpoints["is_boundary_endpoint"],
        "track_id",
    ]

    def selected(track_ids) -> pd.DataFrame:
        return tracks[
            tracks["track_id"].isin(track_ids)
        ].copy()

    return EndpointTrackGroups(
        track_summary=track_summary,
        new_track_endpoints=new_endpoints,
        ended_track_endpoints=ended_endpoints,
        new_failure_tracks=selected(new_failure_ids),
        ended_failure_tracks=selected(ended_failure_ids),
        boundary_entry_tracks=selected(boundary_entry_ids),
        boundary_exit_tracks=selected(boundary_exit_ids),
    )


def diagnostic_node_categories(
    groups: EndpointTrackGroups,
) -> dict[Node, str]:
    """
    Category for center coloring.

    Notebook-style failure categories have priority over boundary categories.
    """
    result: dict[Node, str] = {}

    def assign(frame: pd.DataFrame, category: str) -> None:
        for row in frame.itertuples(index=False):
            result[
                (
                    int(row.frame),
                    int(row.cell_id),
                )
            ] = category

    assign(
        groups.boundary_entry_tracks,
        "boundary_entry",
    )
    assign(
        groups.boundary_exit_tracks,
        "boundary_exit",
    )
    assign(
        groups.new_failure_tracks,
        "new",
    )
    assign(
        groups.ended_failure_tracks,
        "broken",
    )
    return result
