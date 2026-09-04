"""Strict validation for the Stage 7 graph-refinement integration contract."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_ASSIGNMENT_KEYS = (
    "rows",
    "cols",
    "missed_state_indices",
    "birth_detection_indices",
    "pair_cost_matrix",
    "miss_costs",
    "birth_costs",
    "safety_invalid",
    "distance_matrix",
    "association_probabilities",
    "track_candidate_probabilities",
    "detection_candidate_probabilities",
    "track_probability_margins",
)

REQUIRED_DETECTION_COLUMNS = (
    "centroid_z",
    "centroid_y",
    "centroid_x",
    "volume_voxels",
)


def validate_transition_inputs(
    *,
    eligible_states: list[dict],
    detections: pd.DataFrame,
    base_assignment: dict,
    volume_shape_zyx: np.ndarray,
    voxel_size_zyx_um: np.ndarray,
) -> None:
    missing = [key for key in REQUIRED_ASSIGNMENT_KEYS if key not in base_assignment]
    if missing:
        raise KeyError(f"base_assignment is missing required keys: {missing}")
    missing_columns = [column for column in REQUIRED_DETECTION_COLUMNS if column not in detections.columns]
    if missing_columns:
        raise KeyError(f"detections are missing required columns: {missing_columns}")

    track_count = len(eligible_states)
    detection_count = len(detections)
    matrix_shape = (track_count, detection_count)
    for key in (
        "pair_cost_matrix",
        "safety_invalid",
        "distance_matrix",
        "association_probabilities",
        "track_candidate_probabilities",
        "detection_candidate_probabilities",
    ):
        if np.asarray(base_assignment[key]).shape != matrix_shape:
            raise ValueError(f"{key} must have shape {matrix_shape}, got {np.asarray(base_assignment[key]).shape}")
    if np.asarray(base_assignment["miss_costs"]).shape != (track_count,):
        raise ValueError("miss_costs must have one value per eligible state")
    if np.asarray(base_assignment["birth_costs"]).shape != (detection_count,):
        raise ValueError("birth_costs must have one value per detection")
    if np.asarray(base_assignment["track_probability_margins"]).shape != (track_count,):
        raise ValueError("track_probability_margins must have one value per eligible state")

    for index, state in enumerate(eligible_states):
        if "track_id" not in state:
            raise KeyError(f"eligible state {index} is missing track_id")
        if "last_frame" not in state:
            raise KeyError(f"eligible state {index} is missing last_frame")
        if "last_position_physical" not in state and "last_detection" not in state:
            raise KeyError(f"eligible state {index} needs last_position_physical or last_detection")

    shape = np.asarray(volume_shape_zyx)
    spacing = np.asarray(voxel_size_zyx_um)
    if shape.shape != (3,) or spacing.shape != (3,):
        raise ValueError("volume_shape_zyx and voxel_size_zyx_um must have shape (3,)")
    if np.any(shape <= 0) or np.any(spacing <= 0):
        raise ValueError("volume shape and voxel size must be positive")
