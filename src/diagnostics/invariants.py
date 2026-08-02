"""Optional lightweight pipeline-boundary validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_detection_table(cells: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required = {"cell_id", "centroid_z", "centroid_y", "centroid_x"}
    missing = sorted(required.difference(cells.columns))
    if missing:
        errors.append(f"missing detection columns: {missing}")
        return errors
    if cells["cell_id"].duplicated().any():
        errors.append("cell_id values are not unique within the frame")
    if not np.isfinite(cells[["centroid_z", "centroid_y", "centroid_x"]]).all().all():
        errors.append("detection coordinates contain non-finite values")
    return errors


def validate_tracks(tracks: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required = {"track_id", "frame", "cell", "cell_id", "z", "y", "x"}
    missing = sorted(required.difference(tracks.columns))
    if missing:
        errors.append(f"missing track columns: {missing}")
        return errors
    if tracks.duplicated(["track_id", "frame"]).any():
        errors.append("a track contains multiple rows in one frame")
    observed = tracks
    if "is_virtual_merge" in tracks.columns:
        observed = tracks[~tracks["is_virtual_merge"].fillna(False).astype(bool)]
        virtual = tracks[tracks["is_virtual_merge"].fillna(False).astype(bool)]
        required_provenance = {"merge_event_id", "source_track_id", "source_merged_cell_id"}
        if not required_provenance.issubset(virtual.columns):
            errors.append("virtual merge rows are missing provenance columns")
    if observed.duplicated(["frame", "cell_id"]).any():
        errors.append("one observed detection is assigned more than once in a frame")
    return errors
