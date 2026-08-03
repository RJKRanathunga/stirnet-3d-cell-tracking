"""Selection and temporal analysis of stable Stage 7 tracks."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .statistics import temporal_summary


def _contains_virtual_or_merge(values: pd.Series) -> bool:
    text = " ".join(values.fillna("").astype(str).str.lower().tolist())
    return any(token in text for token in ("virtual", "merge", "division", "split"))


def select_track_candidates(
    tracks: pd.DataFrame,
    frame_ids: tuple[int, ...],
    *,
    manual_track_ids: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Select complete, unique, boundary-free observed tracks.

    The automatic result is a high-quality candidate set, not ground truth.
    ``manual_track_ids`` overrides automatic selection after structural checks.
    """
    relevant = tracks[tracks["frame"].isin(frame_ids)].copy()
    required_frames = set(int(frame) for frame in frame_ids)
    manual = set(int(track_id) for track_id in manual_track_ids)
    rows: list[dict[str, object]] = []

    for track_id, group in relevant.groupby("track_id", sort=True):
        observed_frames = set(group["frame"].astype(int).tolist())
        complete = observed_frames == required_frames
        unique_per_frame = not group["frame"].duplicated().any()
        boundary_free = (
            not bool(group["touches_boundary"].fillna(False).astype(bool).any())
            if "touches_boundary" in group.columns
            else True
        )
        observed_only = (
            not _contains_virtual_or_merge(group["match_type"])
            if "match_type" in group.columns
            else True
        )
        selected_automatically = (
            complete and unique_per_frame and boundary_free and observed_only
        )
        selected = (
            int(track_id) in manual
            if manual
            else selected_automatically
        )

        reasons = []
        if not complete:
            reasons.append("not_complete")
        if not unique_per_frame:
            reasons.append("duplicate_frame")
        if not boundary_free:
            reasons.append("touches_boundary")
        if not observed_only:
            reasons.append("virtual_or_event_related")
        if manual and int(track_id) not in manual:
            reasons.append("not_manually_selected")

        rows.append(
            {
                "track_id": int(track_id),
                "record_count": int(len(group)),
                "unique_frame_count": int(group["frame"].nunique()),
                "first_frame": int(group["frame"].min()),
                "last_frame": int(group["frame"].max()),
                "complete": bool(complete),
                "unique_per_frame": bool(unique_per_frame),
                "boundary_free": bool(boundary_free),
                "observed_only": bool(observed_only),
                "selected_automatically": bool(selected_automatically),
                "selected": bool(selected),
                "rejection_reasons": "|".join(reasons),
            }
        )

    return pd.DataFrame(rows)


def attach_track_ids(cell_stats: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Attach Stage 7 track IDs to per-cell measurements."""
    mapping = tracks[["frame", "cell_id", "track_id"]].copy()
    mapping = mapping.drop_duplicates(["frame", "cell_id"], keep=False)
    return cell_stats.merge(mapping, on=["frame", "cell_id"], how="left")


def compute_track_temporal_statistics(
    tracked_cell_stats: pd.DataFrame,
    selected_track_ids: Iterable[int],
    *,
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    """Summarize temporal stability for each selected track and method."""
    selected = set(int(value) for value in selected_track_ids)
    data = tracked_cell_stats[
        tracked_cell_stats["track_id"].isin(selected)
    ].copy()
    rows: list[dict[str, object]] = []

    for (track_id, method), group in data.groupby(
        ["track_id", "method"], sort=True
    ):
        group = group.sort_values("frame")
        for feature in feature_names:
            if feature not in group.columns:
                continue
            finite = group[["frame", feature]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            summary = temporal_summary(
                finite["frame"].to_numpy(),
                finite[feature].to_numpy(),
            )
            rows.append(
                {
                    "track_id": int(track_id),
                    "method": method,
                    "feature": feature,
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def compute_between_within_ratio(
    tracked_cell_stats: pd.DataFrame,
    selected_track_ids: Iterable[int],
    *,
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    """Compare retained between-track variation with within-track variation."""
    selected = set(int(value) for value in selected_track_ids)
    data = tracked_cell_stats[
        tracked_cell_stats["track_id"].isin(selected)
    ].copy()
    rows: list[dict[str, object]] = []

    for method, method_data in data.groupby("method", sort=True):
        for feature in feature_names:
            if feature not in method_data.columns:
                continue
            matrix = method_data.pivot_table(
                index="track_id",
                columns="frame",
                values=feature,
                aggfunc="first",
            )
            if matrix.empty:
                continue
            track_means = matrix.mean(axis=1, skipna=True)
            track_variances = matrix.var(axis=1, ddof=0, skipna=True)
            between = float(track_means.var(ddof=0))
            within = float(track_variances.mean())
            ratio = between / within if within > 1e-12 else math.inf
            rows.append(
                {
                    "method": method,
                    "feature": feature,
                    "track_count": int(matrix.shape[0]),
                    "between_track_variance": between,
                    "mean_within_track_variance": within,
                    "between_within_ratio": float(ratio),
                }
            )
    return pd.DataFrame(rows)
