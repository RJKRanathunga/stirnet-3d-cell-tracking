"""Topology, geometry, persistence, and divergence calculations for Stage 10."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
import pandas as pd

from .step01_config import CellLineageConfig
from .step02_observations import as_bool, physical_distance
from .step05_scoring import relative_error, safe_ratio


@dataclass(frozen=True)
class CandidateGenerationResult:
    records: list[dict[str, object]]
    eligible_parent_count: int
    topology_parent_count: int
    boundary_rejection_count: int
    virtual_observation_rejection_count: int


def predict_parent_position(
    parent_observations: pd.DataFrame,
    target_frame: int,
    config: CellLineageConfig,
) -> dict[str, object]:
    """Extrapolate median physical velocity from recent real observations."""

    usable = parent_observations.loc[
        ~parent_observations["is_virtual_merge"].map(as_bool)
    ].sort_values("frame", kind="mergesort").tail(config.parent_motion_history)
    final = parent_observations.sort_values("frame", kind="mergesort").iloc[-1]
    final_point = final[["z", "y", "x"]].to_numpy(dtype=float)
    predicted = final_point.copy()
    used_velocity = False
    if len(usable) >= 2:
        frames = usable["frame"].to_numpy(dtype=float)
        physical = (
            usable[["z", "y", "x"]].to_numpy(dtype=float)
            * np.asarray(config.voxel_size_zyx_um, dtype=float)
        )
        frame_gaps = np.diff(frames)
        valid = frame_gaps > 0
        if np.any(valid):
            velocities = np.diff(physical, axis=0)[valid] / frame_gaps[valid, None]
            velocity = np.median(velocities, axis=0)
            extrapolation = int(target_frame) - int(final["frame"])
            predicted_physical = (
                final_point * np.asarray(config.voxel_size_zyx_um, dtype=float)
                + velocity * extrapolation
            )
            predicted = predicted_physical / np.asarray(config.voxel_size_zyx_um)
            used_velocity = True
    return {
        "predicted_parent_z": float(predicted[0]),
        "predicted_parent_y": float(predicted[1]),
        "predicted_parent_x": float(predicted[2]),
        "prediction_history_count": int(len(usable)),
        "prediction_used_velocity": used_velocity,
    }


def parent_volume_evidence(
    parent_observations: pd.DataFrame,
    config: CellLineageConfig,
) -> dict[str, object]:
    """Summarize recent real parent volumes without gating on enlargement."""

    real = parent_observations.loc[
        ~parent_observations["is_virtual_merge"].map(as_bool)
    ].sort_values("frame", kind="mergesort").tail(config.parent_volume_history)
    values = pd.to_numeric(real["volume"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values) & (values > 0)
    values = values[finite]
    frames = real["frame"].to_numpy(dtype=float)[finite]
    reference = float(np.median(values)) if values.size else math.nan
    final_volume = float(parent_observations.sort_values("frame").iloc[-1]["volume"])
    slope = (
        float(np.polyfit(frames, values, 1)[0])
        if values.size >= 2 and np.unique(frames).size >= 2
        else math.nan
    )
    return {
        "parent_reference_volume": reference,
        "parent_final_volume": final_volume,
        "parent_recent_volume_count": int(values.size),
        "parent_final_volume_ratio": safe_ratio(final_volume, reference),
        "parent_recent_volume_slope": slope,
    }


def child_persistence_and_divergence(
    observations: pd.DataFrame,
    child_track_a: int,
    child_track_b: int,
    birth_frame: int,
    sequence_last_frame: int,
    config: CellLineageConfig,
) -> dict[str, object]:
    """Measure the weaker child's persistence and paired physical separation."""

    window_end = min(birth_frame + config.future_child_horizon, sequence_last_frame)
    selected = observations.loc[
        observations["track_id"].isin([child_track_a, child_track_b])
        & observations["frame"].between(birth_frame, window_end)
    ]
    a = selected.loc[selected["track_id"] == child_track_a].sort_values("frame")
    b = selected.loc[selected["track_id"] == child_track_b].sort_values("frame")
    count_a = int(len(a))
    count_b = int(len(b))
    minimum_count = min(count_a, count_b)
    paired = a[["frame", "z", "y", "x"]].merge(
        b[["frame", "z", "y", "x"]], on="frame", suffixes=("_a", "_b")
    ).sort_values("frame")
    separations: list[float] = []
    for row in paired.itertuples(index=False):
        separations.append(physical_distance(
            np.asarray([row.z_a, row.y_a, row.x_a]),
            np.asarray([row.z_b, row.y_b, row.x_b]),
            config.voxel_size_zyx_um,
        ))
    separation_array = np.asarray(separations, dtype=float)
    slope = math.nan
    gain = math.nan
    increase_count = 0
    decrease_count = 0
    if separation_array.size >= 2:
        paired_frames = paired["frame"].to_numpy(dtype=float)
        slope = float(np.polyfit(paired_frames, separation_array, 1)[0])
        gain = float(separation_array[-1] - separation_array[0])
        changes = np.diff(separation_array)
        increase_count = int(np.count_nonzero(changes > 0))
        decrease_count = int(np.count_nonzero(changes < 0))
    return {
        "child_a_observation_count_in_window": count_a,
        "child_b_observation_count_in_window": count_b,
        "minimum_child_observation_count": minimum_count,
        "both_children_persist": bool(
            minimum_count >= config.minimum_child_observations
        ),
        "future_window_truncated": bool(
            sequence_last_frame < birth_frame + config.future_child_horizon
        ),
        "child_a_last_observed_frame_in_window": (
            int(a["frame"].max()) if count_a else math.nan
        ),
        "child_b_last_observed_frame_in_window": (
            int(b["frame"].max()) if count_b else math.nan
        ),
        "paired_future_frame_count": int(separation_array.size),
        "last_available_separation_um": (
            float(separation_array[-1]) if separation_array.size else math.nan
        ),
        "maximum_separation_um": (
            float(np.max(separation_array)) if separation_array.size else math.nan
        ),
        "minimum_separation_um": (
            float(np.min(separation_array)) if separation_array.size else math.nan
        ),
        "separation_gain_um": gain,
        "separation_slope_um_per_frame": slope,
        "separation_increase_count": increase_count,
        "separation_decrease_count": decrease_count,
    }


def _candidate_geometry(
    parent_reference_volume: float,
    predicted_zyx: np.ndarray,
    child_a: pd.Series,
    child_b: pd.Series,
    config: CellLineageConfig,
) -> dict[str, float]:
    volume_a = float(child_a["volume"])
    volume_b = float(child_b["volume"])
    combined = volume_a + volume_b
    point_a = child_a[["z", "y", "x"]].to_numpy(dtype=float)
    point_b = child_b[["z", "y", "x"]].to_numpy(dtype=float)
    if not math.isfinite(combined) or combined <= 0:
        weighted = np.full(3, np.nan)
    else:
        weighted = (volume_a * point_a + volume_b * point_b) / combined
    return {
        "child_a_volume": volume_a,
        "child_b_volume": volume_b,
        "combined_child_volume": combined,
        "combined_volume_ratio": safe_ratio(combined, parent_reference_volume),
        "combined_volume_relative_error": relative_error(combined, parent_reference_volume),
        "child_a_volume_fraction": safe_ratio(volume_a, parent_reference_volume),
        "child_b_volume_fraction": safe_ratio(volume_b, parent_reference_volume),
        "child_a_distance_um": physical_distance(point_a, predicted_zyx, config.voxel_size_zyx_um),
        "child_b_distance_um": physical_distance(point_b, predicted_zyx, config.voxel_size_zyx_um),
        "weighted_child_centroid_z": float(weighted[0]),
        "weighted_child_centroid_y": float(weighted[1]),
        "weighted_child_centroid_x": float(weighted[2]),
        "weighted_centroid_error_um": physical_distance(
            weighted, predicted_zyx, config.voxel_size_zyx_um
        ),
        "birth_separation_um": physical_distance(point_a, point_b, config.voxel_size_zyx_um),
    }


def generate_candidates(
    observations: pd.DataFrame,
    summary: pd.DataFrame,
    sample_id: str,
    config: CellLineageConfig,
) -> CandidateGenerationResult:
    """Generate deterministic one-ended-parent to two-started-child pairs."""

    if observations.empty:
        return CandidateGenerationResult([], 0, 0, 0, 0)
    sequence_last_frame = int(observations["frame"].max())
    starts_by_frame: dict[int, pd.DataFrame] = {}
    for frame, rows in summary.groupby("first_frame", sort=True):
        indices = rows["first_observation_index"].astype(int).tolist()
        starts_by_frame[int(frame)] = observations.loc[indices].sort_values(
            "track_id", kind="mergesort"
        )

    eligible = summary.loc[
        (summary["last_frame"] < sequence_last_frame)
        & (summary["observation_count"] >= config.minimum_parent_observations)
    ].copy()
    parent_boundary = eligible["last_is_boundary"].astype(bool)
    parent_virtual = eligible["last_is_virtual"].astype(bool)
    boundary_rejection_keys = {
        ("parent", int(track_id), int(frame))
        for track_id, frame in eligible.loc[parent_boundary, ["track_id", "last_frame"]].itertuples(index=False)
    }
    virtual_rejection_keys = {
        ("parent", int(track_id), int(frame))
        for track_id, frame in eligible.loc[parent_virtual, ["track_id", "last_frame"]].itertuples(index=False)
    }
    eligible = eligible.loc[~parent_boundary & ~parent_virtual]

    records: list[dict[str, object]] = []
    topology_parents = 0
    summary_lookup = summary.set_index("track_id", drop=False)
    for parent_summary in eligible.sort_values("track_id").itertuples(index=False):
        parent_id = int(parent_summary.track_id)
        parent_end = int(parent_summary.last_frame)
        birth_frame = parent_end + config.division_frame_gap
        starts = starts_by_frame.get(birth_frame, observations.iloc[0:0]).copy()
        if starts.empty:
            continue
        starts = starts.loc[starts["track_id"] != parent_id]
        child_ids = starts["track_id"].astype(int)
        child_summary = summary_lookup.loc[child_ids]
        began_on_target = child_summary["first_frame"].to_numpy(dtype=int) == birth_frame
        starts = starts.loc[began_on_target]
        if starts.empty:
            continue
        boundary_flags = starts["track_id"].map(
            summary_lookup["first_is_boundary"].astype(bool)
        ).astype(bool)
        virtual_flags = starts["track_id"].map(
            summary_lookup["first_is_virtual"].astype(bool)
        ).astype(bool)
        boundary_rejection_keys.update(
            ("child", int(track_id), birth_frame)
            for track_id in starts.loc[boundary_flags, "track_id"]
        )
        virtual_rejection_keys.update(
            ("child", int(track_id), birth_frame)
            for track_id in starts.loc[virtual_flags, "track_id"]
        )
        starts = starts.loc[~boundary_flags & ~virtual_flags].copy()
        if len(starts) < 2:
            continue
        topology_parents += 1

        parent_rows = observations.loc[observations["track_id"] == parent_id]
        prediction = predict_parent_position(parent_rows, birth_frame, config)
        predicted = np.asarray([
            prediction["predicted_parent_z"], prediction["predicted_parent_y"],
            prediction["predicted_parent_x"],
        ], dtype=float)
        volume = parent_volume_evidence(parent_rows, config)
        starts["_distance_to_prediction"] = [
            physical_distance(
                row[["z", "y", "x"]].to_numpy(dtype=float),
                predicted,
                config.voxel_size_zyx_um,
            )
            for _, row in starts.iterrows()
        ]
        nearby = starts.loc[
            starts["_distance_to_prediction"] <= config.child_search_radius_um
        ].sort_values("track_id", kind="mergesort")
        if len(nearby) < 2:
            continue

        for index_a, index_b in combinations(nearby.index.tolist(), 2):
            first = nearby.loc[index_a]
            second = nearby.loc[index_b]
            if int(first["track_id"]) <= int(second["track_id"]):
                child_a, child_b = first, second
            else:
                child_a, child_b = second, first
            child_a_id = int(child_a["track_id"])
            child_b_id = int(child_b["track_id"])
            geometry = _candidate_geometry(
                float(volume["parent_reference_volume"]), predicted,
                child_a, child_b, config,
            )
            persistence = child_persistence_and_divergence(
                observations, child_a_id, child_b_id, birth_frame,
                sequence_last_frame, config,
            )
            records.append({
                "candidate_id": f"p{parent_id}_f{birth_frame}_c{child_a_id}_{child_b_id}",
                "sample_id": str(sample_id),
                "parent_track_id": parent_id,
                "parent_end_frame": parent_end,
                "parent_observation_count": int(parent_summary.observation_count),
                "child_birth_frame": birth_frame,
                "child_track_a": child_a_id,
                "child_track_b": child_b_id,
                "parent_is_boundary": False,
                "parent_is_virtual": False,
                "child_a_is_boundary": False,
                "child_b_is_boundary": False,
                "child_a_is_virtual": False,
                "child_b_is_virtual": False,
                "transition_overlaps_virtual_merge": False,
                **prediction,
                **volume,
                **geometry,
                **persistence,
            })

    records.sort(key=lambda row: (
        int(row["parent_track_id"]), int(row["child_birth_frame"]),
        int(row["child_track_a"]), int(row["child_track_b"]),
    ))
    return CandidateGenerationResult(
        records=records,
        eligible_parent_count=int(len(eligible)),
        topology_parent_count=topology_parents,
        boundary_rejection_count=len(boundary_rejection_keys),
        virtual_observation_rejection_count=len(virtual_rejection_keys),
    )
