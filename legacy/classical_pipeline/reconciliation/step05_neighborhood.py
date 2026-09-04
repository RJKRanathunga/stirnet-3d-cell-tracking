"""Stable-neighbour anchor evidence for Stage 11."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .step01_config import TrackReconciliationConfig
from .step02_observations import as_bool, observation_is_boundary, physical_position


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = 0.5 * float(ordered_weights.sum())
    index = int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _motion_stability(group: pd.DataFrame, config: TrackReconciliationConfig) -> float:
    ordered = group.sort_values("frame", kind="mergesort").tail(5)
    if len(ordered) < 3:
        return 0.6
    positions = ordered[["z", "y", "x"]].to_numpy(dtype=float) * np.asarray(
        config.voxel_size_zyx_um, dtype=float
    )
    frames = ordered["frame"].to_numpy(dtype=float)
    velocities = np.diff(positions, axis=0) / np.diff(frames)[:, None]
    center = np.median(velocities, axis=0)
    dispersion = float(np.median(np.linalg.norm(velocities - center, axis=1)))
    return float(math.exp(-dispersion / config.neighborhood_score_scale_um))


def anchor_evidence(
    observations: pd.DataFrame,
    source_end: pd.Series,
    target_start: pd.Series,
    *,
    source_track_id: int,
    target_track_id: int,
    protected_track_ids: set[int],
    spatial_shape_zyx: tuple[int, int, int] | None,
    config: TrackReconciliationConfig,
) -> dict[str, object]:
    """Build a robust local continuation prediction from persistent anchors."""

    source_frame = int(source_end["frame"])
    target_frame = int(target_start["frame"])
    source_position = physical_position(source_end, config)
    target_position = physical_position(target_start, config)
    source_rows = observations.loc[
        (observations["frame"] == source_frame)
        & ~observations["is_virtual_merge"].map(as_bool)
    ]
    target_rows = observations.loc[
        (observations["frame"] == target_frame)
        & ~observations["is_virtual_merge"].map(as_bool)
    ]
    target_by_track = {
        int(row["track_id"]): row for _, row in target_rows.iterrows()
    }
    nearby_count = 0
    records: list[dict[str, object]] = []
    excluded = {int(source_track_id), int(target_track_id)} | set(protected_track_ids)
    for _, anchor_start in source_rows.sort_values("track_id", kind="mergesort").iterrows():
        track_id = int(anchor_start["track_id"])
        if track_id in {int(source_track_id), int(target_track_id)}:
            continue
        distance = float(np.linalg.norm(physical_position(anchor_start, config) - source_position))
        if distance > config.anchor_search_radius_um:
            continue
        nearby_count += 1
        anchor_end = target_by_track.get(track_id)
        if anchor_end is None or track_id in excluded:
            continue
        if observation_is_boundary(anchor_start, spatial_shape_zyx, config):
            continue
        if observation_is_boundary(anchor_end, spatial_shape_zyx, config):
            continue
        group = observations.loc[
            (observations["track_id"] == track_id)
            & ~observations["is_virtual_merge"].map(as_bool)
        ]
        association_quality = 0.5
        if "association_probability" in group:
            values = pd.to_numeric(group["association_probability"], errors="coerce")
            values = values[np.isfinite(values)]
            if len(values):
                association_quality = float(np.clip(values.tail(4).median(), 0.0, 1.0))
        persistence = min(1.0, len(group) / max(target_frame - source_frame + 2, 2))
        stability = _motion_stability(group, config)
        proximity = math.exp(-distance / config.anchor_search_radius_um)
        weight = max(
            1e-6,
            0.40 * proximity + 0.20 * association_quality
            + 0.20 * persistence + 0.20 * stability,
        )
        anchor_source_position = physical_position(anchor_start, config)
        anchor_target_position = physical_position(anchor_end, config)
        prediction = anchor_target_position + (source_position - anchor_source_position)
        source_distance = float(np.linalg.norm(source_position - anchor_source_position))
        target_distance = float(np.linalg.norm(target_position - anchor_target_position))
        records.append({
            "track_id": track_id,
            "distance": distance,
            "weight": weight,
            "prediction": prediction,
            "distance_error": abs(target_distance - source_distance),
        })
    records.sort(key=lambda row: (-float(row["weight"]), float(row["distance"]), int(row["track_id"])))
    records = records[: config.maximum_anchor_count]
    if not records:
        return {
            "anchor_count": 0,
            "anchor_track_ids": "",
            "anchor_predicted_z_um": math.nan,
            "anchor_predicted_y_um": math.nan,
            "anchor_predicted_x_um": math.nan,
            "anchor_prediction_error_um": math.nan,
            "neighborhood_distance_error_um": math.nan,
            "local_survival_ratio": 0.0 if nearby_count else math.nan,
        }
    predictions = np.asarray([row["prediction"] for row in records], dtype=float)
    weights = np.asarray([row["weight"] for row in records], dtype=float)
    combined = np.asarray([
        _weighted_median(predictions[:, axis], weights) for axis in range(3)
    ])
    distance_errors = np.asarray([row["distance_error"] for row in records], dtype=float)
    return {
        "anchor_count": int(len(records)),
        "anchor_track_ids": "|".join(
            str(track_id) for track_id in sorted(int(row["track_id"]) for row in records)
        ),
        "anchor_predicted_z_um": float(combined[0]),
        "anchor_predicted_y_um": float(combined[1]),
        "anchor_predicted_x_um": float(combined[2]),
        "anchor_prediction_error_um": float(np.linalg.norm(combined - target_position)),
        "neighborhood_distance_error_um": _weighted_median(distance_errors, weights),
        "local_survival_ratio": float(len(records) / nearby_count) if nearby_count else math.nan,
    }
