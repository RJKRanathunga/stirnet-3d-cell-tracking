"""Physical forward and backward motion evidence for continuation candidates."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .step01_config import TrackReconciliationConfig
from .step02_observations import as_bool, physical_position


def _robust_velocity(
    observations: pd.DataFrame,
    history: int,
    config: TrackReconciliationConfig,
) -> tuple[np.ndarray | None, int]:
    ordered = observations.sort_values("frame", kind="mergesort").tail(history)
    if len(ordered) < 2:
        return None, int(len(ordered))
    frames = ordered["frame"].to_numpy(dtype=float)
    positions = ordered[["z", "y", "x"]].to_numpy(dtype=float) * np.asarray(
        config.voxel_size_zyx_um, dtype=float
    )
    deltas = np.diff(positions, axis=0)
    frame_deltas = np.diff(frames)
    valid = np.isfinite(frame_deltas) & (frame_deltas > 0)
    if not valid.any():
        return None, int(len(ordered))
    velocities = deltas[valid] / frame_deltas[valid, None]
    return np.median(velocities, axis=0), int(len(ordered))


def _global_shift_lookup(global_motion: pd.DataFrame | None) -> dict[tuple[int, int], np.ndarray]:
    if global_motion is None or global_motion.empty:
        return {}
    frame_columns = {"from_frame", "to_frame"}
    if not frame_columns.issubset(global_motion.columns):
        return {}
    alternatives = (
        ("final_shift_z_um", "final_shift_y_um", "final_shift_x_um"),
        ("global_shift_z_um", "global_shift_y_um", "global_shift_x_um"),
        ("shift_z_um", "shift_y_um", "shift_x_um"),
    )
    columns = next((cols for cols in alternatives if set(cols).issubset(global_motion.columns)), None)
    if columns is None:
        return {}
    lookup: dict[tuple[int, int], np.ndarray] = {}
    ordered = global_motion.sort_values(["from_frame", "to_frame"], kind="mergesort")
    for row in ordered.itertuples(index=False):
        shift = np.asarray([getattr(row, name) for name in columns], dtype=float)
        if np.all(np.isfinite(shift)):
            lookup[(int(row.from_frame), int(row.to_frame))] = shift
    return lookup


def _cumulative_global(
    lookup: dict[tuple[int, int], np.ndarray],
    from_frame: int,
    to_frame: int,
) -> tuple[np.ndarray, bool]:
    if to_frame <= from_frame:
        return np.zeros(3, dtype=float), False
    shifts = []
    for frame in range(from_frame + 1, to_frame + 1):
        value = lookup.get((frame - 1, frame))
        if value is None:
            return np.zeros(3, dtype=float), False
        shifts.append(value)
    return np.sum(shifts, axis=0), bool(shifts)


def forward_prediction(
    source_real: pd.DataFrame,
    target_start: pd.Series,
    global_motion: pd.DataFrame | None,
    config: TrackReconciliationConfig,
) -> dict[str, object]:
    """Predict a source endpoint without double-counting global displacement."""

    last = source_real.sort_values("frame", kind="mergesort").iloc[-1]
    source_frame = int(last["frame"])
    target_frame = int(target_start["frame"])
    gap = target_frame - source_frame
    source_position = physical_position(last, config)
    target_position = physical_position(target_start, config)
    lookup = _global_shift_lookup(global_motion)
    global_displacement, used_global = _cumulative_global(
        lookup, source_frame, target_frame
    )

    relative_velocity: np.ndarray | None = None
    used_stored_relative = False
    relative_columns = (
        "relative_velocity_z_um_per_frame", "relative_velocity_y_um_per_frame",
        "relative_velocity_x_um_per_frame",
    )
    if (
        "relative_velocity_valid" in last.index
        and as_bool(last["relative_velocity_valid"])
        and all(column in last.index for column in relative_columns)
    ):
        candidate = last[list(relative_columns)].to_numpy(dtype=float)
        if np.all(np.isfinite(candidate)):
            relative_velocity = candidate
            used_stored_relative = True

    total_velocity, history_count = _robust_velocity(
        source_real, config.source_motion_history, config
    )
    if relative_velocity is None and total_velocity is not None:
        if used_global:
            recent = source_real.sort_values("frame", kind="mergesort").tail(
                config.source_motion_history
            )
            historical_global: list[np.ndarray] = []
            frames = recent["frame"].astype(int).tolist()
            complete = True
            for previous, current in zip(frames[:-1], frames[1:]):
                displacement, available = _cumulative_global(lookup, previous, current)
                if not available:
                    complete = False
                    break
                historical_global.append(displacement / (current - previous))
            relative_velocity = (
                total_velocity - np.median(historical_global, axis=0)
                if complete and historical_global else np.zeros(3, dtype=float)
            )
        else:
            relative_velocity = total_velocity
    if relative_velocity is None:
        relative_velocity = np.zeros(3, dtype=float)

    residual = np.zeros(3, dtype=float)
    residual_columns = (
        "position_residual_ema_z_um", "position_residual_ema_y_um",
        "position_residual_ema_x_um",
    )
    if all(column in last.index for column in residual_columns):
        try:
            samples = int(last.get("position_residual_samples", 0))
            candidate = last[list(residual_columns)].to_numpy(dtype=float)
            if samples > 0 and np.all(np.isfinite(candidate)):
                residual = candidate
        except (TypeError, ValueError):
            pass

    predicted = (
        source_position + global_displacement + relative_velocity * gap + residual
    )
    error = float(np.linalg.norm(predicted - target_position))
    error_ema = 0.0
    try:
        value = float(last.get("relative_velocity_error_ema_um", 0.0))
        if math.isfinite(value) and value > 0:
            error_ema = value
    except (TypeError, ValueError):
        pass
    uncertainty = (
        config.forward_uncertainty_base_um
        + config.forward_uncertainty_per_gap_um * math.sqrt(max(gap, 1))
        + error_ema * max(gap, 1)
    )
    return {
        "forward_predicted_z_um": float(predicted[0]),
        "forward_predicted_y_um": float(predicted[1]),
        "forward_predicted_x_um": float(predicted[2]),
        "forward_error_um": error,
        "forward_history_count": history_count,
        "forward_used_global_motion": bool(used_global),
        "forward_used_relative_velocity": bool(used_stored_relative),
        "forward_uncertainty_um": float(uncertainty),
    }


def backward_prediction(
    target_real: pd.DataFrame,
    source_end: pd.Series,
    config: TrackReconciliationConfig,
) -> dict[str, object]:
    """Propagate early target motion backward to the source ending frame."""

    ordered = target_real.sort_values("frame", kind="mergesort").head(
        config.target_backward_history
    )
    source_position = physical_position(source_end, config)
    if len(ordered) < 2:
        return {
            "backward_predicted_z_um": math.nan,
            "backward_predicted_y_um": math.nan,
            "backward_predicted_x_um": math.nan,
            "backward_error_um": math.nan,
            "backward_history_count": int(len(ordered)),
            "backward_prediction_available": False,
        }
    velocity, history_count = _robust_velocity(
        ordered, config.target_backward_history, config
    )
    if velocity is None:
        return {
            "backward_predicted_z_um": math.nan,
            "backward_predicted_y_um": math.nan,
            "backward_predicted_x_um": math.nan,
            "backward_error_um": math.nan,
            "backward_history_count": history_count,
            "backward_prediction_available": False,
        }
    first = ordered.iloc[0]
    frame_gap = int(first["frame"]) - int(source_end["frame"])
    predicted = physical_position(first, config) - velocity * frame_gap
    return {
        "backward_predicted_z_um": float(predicted[0]),
        "backward_predicted_y_um": float(predicted[1]),
        "backward_predicted_x_um": float(predicted[2]),
        "backward_error_um": float(np.linalg.norm(predicted - source_position)),
        "backward_history_count": history_count,
        "backward_prediction_available": True,
    }


def bidirectional_disagreement(
    forward: dict[str, object],
    backward: dict[str, object],
    source_end: pd.Series,
    target_start: pd.Series,
    config: TrackReconciliationConfig,
) -> float:
    if not bool(backward["backward_prediction_available"]):
        return math.nan
    forward_target = np.asarray([
        forward["forward_predicted_z_um"], forward["forward_predicted_y_um"],
        forward["forward_predicted_x_um"],
    ], dtype=float)
    backward_source = np.asarray([
        backward["backward_predicted_z_um"], backward["backward_predicted_y_um"],
        backward["backward_predicted_x_um"],
    ], dtype=float)
    source_position = physical_position(source_end, config)
    target_position = physical_position(target_start, config)
    forward_displacement = forward_target - source_position
    backward_implied_displacement = target_position - backward_source
    return float(np.linalg.norm(forward_displacement - backward_implied_displacement))
