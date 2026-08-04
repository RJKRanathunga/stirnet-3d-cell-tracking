"""Deterministic track groups and Napari-compatible arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FinalTrackGroups:
    suspicious_birth_tracks: pd.DataFrame
    suspicious_termination_tracks: pd.DataFrame
    boundary_entry_tracks: pd.DataFrame
    boundary_exit_tracks: pd.DataFrame
    division_tracks: pd.DataFrame
    merge_tracks: pd.DataFrame
    stage11_modified_tracks: pd.DataFrame
    forced_repair_tracks: pd.DataFrame
    unresolved_tracks: pd.DataFrame
    short_lived_tracks: pd.DataFrame
    temporal_gap_tracks: pd.DataFrame


def to_napari_tracks(tracks: pd.DataFrame) -> np.ndarray:
    if tracks.empty:
        return np.empty((0, 5), dtype=float)
    return tracks[["track_id", "frame", "z", "y", "x"]].to_numpy(dtype=float)


def to_napari_points(tracks: pd.DataFrame) -> np.ndarray:
    if tracks.empty:
        return np.empty((0, 4), dtype=float)
    return tracks[["frame", "z", "y", "x"]].to_numpy(dtype=float)


def select_tracks(tracks: pd.DataFrame, track_ids: pd.Series | np.ndarray | list[int]) -> pd.DataFrame:
    ids = set(pd.to_numeric(pd.Series(track_ids), errors="coerce").dropna().astype(int))
    return tracks[tracks["track_id"].isin(ids)].copy()


def build_track_groups(tracks: pd.DataFrame, summary: pd.DataFrame) -> FinalTrackGroups:
    def selected(mask) -> pd.DataFrame:
        return select_tracks(tracks, summary.loc[mask, "track_id"])

    return FinalTrackGroups(
        suspicious_birth_tracks=selected(summary["suspicious_start"].astype(bool)),
        suspicious_termination_tracks=selected(summary["suspicious_end"].astype(bool)),
        boundary_entry_tracks=selected(summary["start_classification"] == "boundary_entry"),
        boundary_exit_tracks=selected(summary["end_classification"] == "boundary_exit"),
        division_tracks=selected(summary["division_related"].astype(bool)),
        merge_tracks=selected(summary["merge_related"].astype(bool)),
        stage11_modified_tracks=selected(summary["stage11_modified"].astype(bool)),
        forced_repair_tracks=selected(summary["forced_repair_count"] > 0),
        unresolved_tracks=selected(summary["unresolved_ending"].astype(bool)),
        short_lived_tracks=selected(summary["short_lived"].astype(bool)),
        temporal_gap_tracks=selected(summary["has_temporal_gap"].astype(bool)),
    )
