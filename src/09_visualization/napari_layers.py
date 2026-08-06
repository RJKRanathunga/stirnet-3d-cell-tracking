"""Shared Napari layer construction for Stage 9 diagnostics."""

from __future__ import annotations

from typing import Any

import pandas as pd


def add_track_group(
    viewer: Any,
    frame: pd.DataFrame,
    *,
    track_name: str,
    point_name: str,
    color: str,
    scale,
    visible: bool = True,
    tail_length: int = 20,
) -> None:
    """Add one Stage 9 track group with its matching centroid points."""

    if frame.empty:
        return
    track_layer = viewer.add_tracks(
        frame[["track_id", "frame", "z", "y", "x"]].to_numpy(float),
        name=track_name,
        scale=scale,
        tail_length=tail_length,
    )
    point_layer = viewer.add_points(
        frame[["frame", "z", "y", "x"]].to_numpy(float),
        name=point_name,
        scale=scale,
        size=4,
        face_color=color,
        properties={
            "track_id": frame["track_id"].to_numpy(),
            "cell_id": frame["cell_id"].to_numpy(),
        },
    )
    track_layer.visible = visible
    point_layer.visible = visible
