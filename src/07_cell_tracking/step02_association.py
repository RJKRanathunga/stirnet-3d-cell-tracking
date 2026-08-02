"""Association helpers migrated verbatim from Stage 7 notebook cell 8."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .step01_config import *

# ============================================================
# Boundary metadata, motion modelling, and assignment helpers
# ============================================================

SIZE_FEATURES = [
    "extent",
    "equivalent_radius",
]

SHAPE_FEATURES = [
    "elongation",
    "flatness",
    "anisotropy",
    "solidity",
    "compactness",
]

INTENSITY_FEATURES = [
    "intensity_mean",
    "intensity_std",
    "intensity_cv",
]

BBOX_FEATURES = [
    "bbox_depth",
    "bbox_height",
    "bbox_width",
]

TEMPLATE_FEATURES = sorted(
    set(
        SIZE_FEATURES
        + SHAPE_FEATURES
        + INTENSITY_FEATURES
        + BBOX_FEATURES
    )
)


def annotate_boundary_metadata(
    detections: pd.DataFrame,
) -> pd.DataFrame:
    """Add boundary-contact information to one frame's detections."""

    required_bbox_columns = {
        "z_min", "z_max",
        "y_min", "y_max",
        "x_min", "x_max",
    }

    missing = required_bbox_columns.difference(
        detections.columns
    )

    if missing:
        raise KeyError(
            "Boundary-aware tracking requires bounding-box columns. "
            f"Missing: {sorted(missing)}"
        )

    result = detections.copy()

    margin_zyx = np.ceil(
        BOUNDARY_MARGIN_UM / VOXEL_SIZE_ZYX
    ).astype(int)

    z_size, y_size, x_size = VOLUME_SHAPE_ZYX
    z_margin, y_margin, x_margin = margin_zyx

    # The stored bbox maxima are treated as upper/exclusive bounds.
    result["touches_z_min"] = result["z_min"] <= z_margin
    result["touches_z_max"] = result["z_max"] >= (
        z_size - z_margin
    )

    result["touches_y_min"] = result["y_min"] <= y_margin
    result["touches_y_max"] = result["y_max"] >= (
        y_size - y_margin
    )

    result["touches_x_min"] = result["x_min"] <= x_margin
    result["touches_x_max"] = result["x_max"] >= (
        x_size - x_margin
    )

    face_columns = [
        "touches_z_min",
        "touches_z_max",
        "touches_y_min",
        "touches_y_max",
        "touches_x_min",
        "touches_x_max",
    ]

    result["touches_boundary"] = (
        result[face_columns].any(axis=1)
    )

    result["boundary_face_count"] = (
        result[face_columns].sum(axis=1).astype(int)
    )

    face_names = [
        "z_min", "z_max",
        "y_min", "y_max",
        "x_min", "x_max",
    ]

    result["boundary_faces"] = [
        "|".join(
            face_name
            for face_name, flag in zip(
                face_names,
                flags,
            )
            if bool(flag)
        )
        for flags in result[face_columns].to_numpy()
    ]

    centroids = result[
        ["centroid_z", "centroid_y", "centroid_x"]
    ].to_numpy(dtype=float)

    lower_distance_um = (
        centroids * VOXEL_SIZE_ZYX
    )

    upper_distance_um = (
        (
            VOLUME_SHAPE_ZYX
            - 1
            - centroids
        )
        * VOXEL_SIZE_ZYX
    )

    result["distance_to_boundary_um"] = np.min(
        np.concatenate(
            [lower_distance_um, upper_distance_um],
            axis=1,
        ),
        axis=1,
    )

    result["is_boundary_partial"] = (
        result["touches_boundary"]
    )

    return result


def parse_boundary_faces(value) -> set[str]:
    """Convert a pipe-separated boundary-face string into a set."""

    if value is None or pd.isna(value):
        return set()

    text = str(value).strip()

    if not text:
        return set()

    return {
        face
        for face in text.split("|")
        if face
    }


def physical_coordinates(
    detections: pd.DataFrame,
) -> np.ndarray:
    """Return centroid coordinates in physical (Z, Y, X) units."""

    coords_voxel = detections[
        ["centroid_z", "centroid_y", "centroid_x"]
    ].to_numpy(dtype=float)

    return coords_voxel * VOXEL_SIZE_ZYX


def vector_angle_degrees(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return the angle between two vectors in degrees."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    denominator = (
        np.linalg.norm(first)
        * np.linalg.norm(second)
    )

    if denominator <= EPS:
        return np.nan

    cosine = float(
        np.dot(first, second) / denominator
    )

    return float(
        np.degrees(
            np.arccos(
                np.clip(cosine, -1.0, 1.0)
            )
        )
    )


def robust_displacement_summary(
    displacements: np.ndarray,
) -> dict:
    """Robustly summarize a collection of 3-D displacement vectors."""

    displacements = np.asarray(
        displacements,
        dtype=float,
    ).reshape(-1, 3)

    if len(displacements) == 0:
        return {
            "shift": np.zeros(3, dtype=float),
            "inlier_mask": np.zeros(0, dtype=bool),
            "inlier_count": 0,
            "dispersion_um": np.nan,
            "inlier_radius_um": np.nan,
            "confidence": 0.0,
        }

    initial_center = np.median(
        displacements,
        axis=0,
    )

    residual_norms = np.linalg.norm(
        displacements - initial_center[None, :],
        axis=1,
    )

    residual_median = float(
        np.median(residual_norms)
    )

    residual_mad = float(
        np.median(
            np.abs(
                residual_norms - residual_median
            )
        )
    )

    robust_sigma = 1.4826 * residual_mad

    inlier_radius_um = max(
        GLOBAL_SHIFT_MIN_INLIER_RADIUS_UM,
        (
            residual_median
            + GLOBAL_SHIFT_MAD_SCALE
            * robust_sigma
        ),
    )

    inlier_mask = (
        residual_norms <= inlier_radius_um
    )

    if not np.any(inlier_mask):
        inlier_mask = np.ones(
            len(displacements),
            dtype=bool,
        )

    inlier_displacements = displacements[
        inlier_mask
    ]

    shift = np.median(
        inlier_displacements,
        axis=0,
    )

    dispersion_um = float(
        np.median(
            np.linalg.norm(
                inlier_displacements
                - shift[None, :],
                axis=1,
            )
        )
    )

    pair_confidence = min(
        len(inlier_displacements)
        / max(
            GLOBAL_SHIFT_CONFIDENCE_PAIR_COUNT,
            1,
        ),
        1.0,
    )

    dispersion_confidence = float(
        np.exp(
            -dispersion_um
            / max(
                GLOBAL_SHIFT_CONFIDENCE_DISPERSION_UM,
                EPS,
            )
        )
    )

    confidence = float(
        np.clip(
            pair_confidence
            * dispersion_confidence,
            0.0,
            1.0,
        )
    )

    return {
        "shift": shift.astype(float),
        "inlier_mask": inlier_mask,
        "inlier_count": int(
            inlier_mask.sum()
        ),
        "dispersion_um": dispersion_um,
        "inlier_radius_um": float(
            inlier_radius_um
        ),
        "confidence": confidence,
    }


def estimate_global_shift_physical(
    previous_positions: np.ndarray,
    current_positions: np.ndarray,
) -> dict:
    """Estimate current global displacement using mutual nearest neighbours."""

    previous_positions = np.asarray(
        previous_positions,
        dtype=float,
    ).reshape(-1, 3)

    current_positions = np.asarray(
        current_positions,
        dtype=float,
    ).reshape(-1, 3)

    if (
        len(previous_positions) == 0
        or len(current_positions) == 0
    ):
        return {
            "shift": np.zeros(3, dtype=float),
            "method": "no_data",
            "pair_count": 0,
            "inlier_count": 0,
            "dispersion_um": np.nan,
            "inlier_radius_um": np.nan,
            "confidence": 0.0,
        }

    distances = cdist(
        previous_positions,
        current_positions,
    )

    nearest_current = np.argmin(
        distances,
        axis=1,
    )

    nearest_previous = np.argmin(
        distances,
        axis=0,
    )

    matched_displacements = []

    for previous_index, current_index in enumerate(
        nearest_current
    ):
        if (
            nearest_previous[current_index]
            != previous_index
        ):
            continue

        if (
            distances[
                previous_index,
                current_index,
            ]
            > GLOBAL_SHIFT_MAX_PAIR_DISTANCE_UM
        ):
            continue

        matched_displacements.append(
            current_positions[current_index]
            - previous_positions[previous_index]
        )

    if matched_displacements:
        matched_displacements = np.asarray(
            matched_displacements,
            dtype=float,
        )

        summary = robust_displacement_summary(
            matched_displacements
        )

        return {
            "shift": summary["shift"],
            "method": "mutual_nearest_neighbour",
            "pair_count": int(
                len(matched_displacements)
            ),
            "inlier_count": summary[
                "inlier_count"
            ],
            "dispersion_um": summary[
                "dispersion_um"
            ],
            "inlier_radius_um": summary[
                "inlier_radius_um"
            ],
            "confidence": summary[
                "confidence"
            ],
        }

    # Low-confidence fallback that still responds to a sudden direction
    # change instead of blindly repeating the previous frame's shift.
    centroid_shift = (
        np.median(current_positions, axis=0)
        - np.median(previous_positions, axis=0)
    )

    return {
        "shift": centroid_shift.astype(float),
        "method": "median_centroid_fallback",
        "pair_count": 0,
        "inlier_count": 0,
        "dispersion_um": np.nan,
        "inlier_radius_um": np.nan,
        "confidence": 0.05,
    }


def make_feature_template(
    detection: pd.Series,
) -> dict[str, float]:
    """Create a numeric feature template from one detection."""

    template = {}

    for feature in TEMPLATE_FEATURES:
        if feature not in detection.index:
            continue

        value = detection[feature]

        if pd.notna(value):
            template[feature] = float(value)

    return template


def update_feature_template(
    state: dict,
    detection: pd.Series,
) -> None:
    """Update a reliable template using only fully visible detections."""

    if bool(detection["touches_boundary"]):
        return

    values = make_feature_template(detection)

    if not state["template_reliable"]:
        state["template"] = values
        state["template_reliable"] = True
        state["template_count"] = 1
        return

    for feature, value in values.items():
        old_value = state["template"].get(
            feature,
            value,
        )

        state["template"][feature] = (
            (1.0 - TEMPLATE_EMA_ALPHA)
            * old_value
            + TEMPLATE_EMA_ALPHA
            * value
        )

    state["template_count"] += 1


def make_track_state(
    *,
    track_id: int,
    frame: int,
    detection: pd.Series,
) -> dict:
    """Create the persistent state for one new track."""

    position_voxel = detection[
        ["centroid_z", "centroid_y", "centroid_x"]
    ].to_numpy(dtype=float)

    position_physical = (
        position_voxel * VOXEL_SIZE_ZYX
    )

    is_boundary = bool(
        detection["touches_boundary"]
    )

    return {
        "track_id": int(track_id),
        "active": True,
        "last_frame": int(frame),
        "last_position_voxel": position_voxel,
        "last_position_physical": position_physical,
        "previous_position_physical": None,
        "relative_velocity_physical": np.zeros(
            3,
            dtype=float,
        ),
        "relative_velocity_valid": False,
        "relative_velocity_samples": 0,
        "relative_velocity_error_ema": 0.0,
        "relative_velocity_updates_rejected": 0,
        "last_relative_update_frame": None,
        "position_residual_ema_zyx": np.zeros(
            3,
            dtype=float,
        ),
        "position_residual_samples": 0,
        "last_detection": detection.to_dict(),
        "missed_frames": 0,
        "boundary_pending": is_boundary,
        "interior_pending": False,
        "last_boundary_faces": parse_boundary_faces(
            detection["boundary_faces"]
        ),
        "template": make_feature_template(detection),
        "template_reliable": not is_boundary,
        "template_count": 1 if not is_boundary else 0,
    }

def state_reference_value(
    state: dict,
    feature: str,
) -> float:
    """Read a feature from the reliable template or last observation."""

    if (
        state["template_reliable"]
        and feature in state["template"]
    ):
        return float(state["template"][feature])

    value = state["last_detection"].get(
        feature,
        np.nan,
    )

    return float(value) if pd.notna(value) else np.nan


def cumulative_global_displacement(
    *,
    start_frame: int,
    target_frame: int,
    global_shift_history: dict[int, np.ndarray],
) -> np.ndarray:
    """Sum frame-to-frame global shifts from start_frame to target_frame."""

    if target_frame <= start_frame:
        return np.zeros(3, dtype=float)

    missing_frames = [
        frame
        for frame in range(
            int(start_frame) + 1,
            int(target_frame) + 1,
        )
        if frame not in global_shift_history
    ]

    if missing_frames:
        raise KeyError(
            "Global-shift history is incomplete for frames "
            f"{missing_frames}."
        )

    return np.sum(
        np.vstack(
            [
                global_shift_history[frame]
                for frame in range(
                    int(start_frame) + 1,
                    int(target_frame) + 1,
                )
            ]
        ),
        axis=0,
    )


def relative_motion_confidence(
    state: dict,
    frame_gap: int,
) -> float:
    """Return confidence in a track's relative-motion correction."""

    if not state["relative_velocity_valid"]:
        return 0.0

    sample_confidence = min(
        state["relative_velocity_samples"]
        / max(
            RELATIVE_FULL_CONFIDENCE_SAMPLES,
            1,
        ),
        1.0,
    )

    error_confidence = float(
        np.exp(
            -state["relative_velocity_error_ema"]
            / max(
                RELATIVE_ERROR_CONFIDENCE_SCALE_UM,
                EPS,
            )
        )
    )

    gap_confidence = (
        RELATIVE_MOTION_GAP_DECAY
        ** max(int(frame_gap) - 1, 0)
    )

    boundary_confidence = 1.0

    if (
        state["boundary_pending"]
        or bool(
            state["last_detection"].get(
                "touches_boundary",
                False,
            )
        )
    ):
        boundary_confidence = (
            BOUNDARY_RELATIVE_MOTION_CONFIDENCE_SCALE
        )

    return float(
        np.clip(
            sample_confidence
            * error_confidence
            * gap_confidence
            * boundary_confidence,
            0.0,
            1.0,
        )
    )


def predict_state_components(
    *,
    state: dict,
    target_frame: int,
    global_shift_history: dict[int, np.ndarray],
) -> dict:
    """Return global, relative, and final prediction components."""

    frame_gap = (
        int(target_frame)
        - int(state["last_frame"])
    )

    if frame_gap <= 0:
        return {
            "predicted_position": state[
                "last_position_physical"
            ].copy(),
            "global_displacement": np.zeros(
                3,
                dtype=float,
            ),
            "relative_displacement": np.zeros(
                3,
                dtype=float,
            ),
            "relative_weight": 0.0,
            "frame_gap": frame_gap,
        }

    global_displacement = (
        cumulative_global_displacement(
            start_frame=state["last_frame"],
            target_frame=target_frame,
            global_shift_history=global_shift_history,
        )
    )

    relative_weight = (
        relative_motion_confidence(
            state,
            frame_gap,
        )
    )

    relative_displacement = (
        relative_weight
        * state["relative_velocity_physical"]
        * frame_gap
    )

    predicted_position = (
        state["last_position_physical"]
        + global_displacement
        + relative_displacement
    )

    return {
        "predicted_position": predicted_position,
        "global_displacement": global_displacement,
        "relative_displacement": relative_displacement,
        "relative_weight": relative_weight,
        "frame_gap": frame_gap,
    }


def predict_state_position(
    state: dict,
    target_frame: int,
    global_shift_history: dict[int, np.ndarray],
) -> np.ndarray:
    """Predict a track position using global plus relative motion."""

    return predict_state_components(
        state=state,
        target_frame=target_frame,
        global_shift_history=global_shift_history,
    )["predicted_position"]


def normalized_pairwise_cost(
    previous_values: np.ndarray,
    current_values: np.ndarray,
) -> np.ndarray:
    """Relative absolute difference with robust NaN handling."""

    previous_values = np.asarray(
        previous_values,
        dtype=float,
    )

    current_values = np.asarray(
        current_values,
        dtype=float,
    )

    cost = (
        np.abs(
            previous_values[:, None]
            - current_values[None, :]
        )
        / (
            np.maximum(
                np.abs(previous_values[:, None]),
                np.abs(current_values[None, :]),
            )
            + EPS
        )
    )

    return np.nan_to_num(
        cost,
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )


def feature_group_cost(
    states: list[dict],
    detections: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """Average relative cost over available features in one group."""

    costs = []

    for feature in feature_names:
        if feature not in detections.columns:
            continue

        previous_values = np.asarray(
            [
                state_reference_value(
                    state,
                    feature,
                )
                for state in states
            ],
            dtype=float,
        )

        current_values = detections[
            feature
        ].to_numpy(dtype=float)

        costs.append(
            normalized_pairwise_cost(
                previous_values,
                current_values,
            )
        )

    if not costs:
        return np.zeros(
            (len(states), len(detections)),
            dtype=float,
        )

    return np.mean(
        np.stack(costs, axis=0),
        axis=0,
    )


def relative_motion_consistency_cost(
    *,
    states: list[dict],
    current_positions: np.ndarray,
    current_frame: int,
    global_shift_history: dict[int, np.ndarray],
    prediction_components: list[dict],
) -> np.ndarray:
    """Compare candidate residual motion after removing global motion."""

    result = np.zeros(
        (len(states), len(current_positions)),
        dtype=float,
    )

    for state_index, state in enumerate(states):
        confidence = prediction_components[
            state_index
        ]["relative_weight"]

        if confidence <= EPS:
            continue

        frame_gap = (
            current_frame
            - state["last_frame"]
        )

        global_displacement = (
            prediction_components[
                state_index
            ]["global_displacement"]
        )

        expected_relative_displacement = (
            prediction_components[
                state_index
            ]["relative_displacement"]
        )

        candidate_relative_displacements = (
            current_positions
            - state[
                "last_position_physical"
            ][None, :]
            - global_displacement[None, :]
        )

        residual_error = np.linalg.norm(
            candidate_relative_displacements
            - expected_relative_displacement[
                None,
                :
            ],
            axis=1,
        )

        normalized_error = np.clip(
            residual_error
            / max(
                RELATIVE_MOTION_COST_SCALE_UM
                * max(frame_gap, 1),
                EPS,
            ),
            0.0,
            1.0,
        )

        # Confidence controls how strongly this term can influence
        # assignment. Uncertain relative motion contributes little.
        result[state_index] = (
            confidence * normalized_error
        )

    return result


def boundary_face_cost(
    states: list[dict],
    detections: pd.DataFrame,
) -> np.ndarray:
    """Penalize jumps between incompatible volume faces."""

    current_faces = [
        parse_boundary_faces(value)
        for value in detections["boundary_faces"]
    ]

    current_boundary = detections[
        "touches_boundary"
    ].to_numpy(dtype=bool)

    result = np.zeros(
        (len(states), len(detections)),
        dtype=float,
    )

    for state_index, state in enumerate(states):
        previous_faces = state[
            "last_boundary_faces"
        ]

        previous_boundary = bool(
            state["last_detection"].get(
                "touches_boundary",
                False,
            )
        ) or state["boundary_pending"]

        if not previous_boundary:
            continue

        for detection_index, faces in enumerate(
            current_faces
        ):
            # Moving from a boundary into the interior is valid.
            if not current_boundary[detection_index]:
                continue

            if previous_faces and faces:
                if previous_faces.isdisjoint(faces):
                    result[
                        state_index,
                        detection_index,
                    ] = 1.0

    return result


def student_t_scalar_cost(
    residual: np.ndarray,
    scale: np.ndarray | float,
    degrees_of_freedom: float,
) -> np.ndarray:
    """Heavy-tailed scalar negative-log cost, excluding constants."""

    residual = np.asarray(residual, dtype=float)
    scale = np.maximum(
        np.asarray(scale, dtype=float),
        EPS,
    )

    standardized_squared = (
        residual / scale
    ) ** 2

    return (
        0.5
        * (degrees_of_freedom + 1.0)
        * np.log1p(
            standardized_squared
            / degrees_of_freedom
        )
    )


def student_t_vector_cost(
    mahalanobis_squared: np.ndarray,
    degrees_of_freedom: float,
    dimension: int,
) -> np.ndarray:
    """Multivariate Student-t cost from squared Mahalanobis distance."""

    mahalanobis_squared = np.asarray(
        mahalanobis_squared,
        dtype=float,
    )

    return (
        0.5
        * (degrees_of_freedom + dimension)
        * np.log1p(
            mahalanobis_squared
            / degrees_of_freedom
        )
    )


def negative_log_probability(
    probability: np.ndarray | float,
) -> np.ndarray:
    """Stable -log(probability)."""

    return -np.log(
        np.clip(
            probability,
            PROBABILITY_FLOOR,
            1.0,
        )
    )


def track_position_sigma_zyx(
    *,
    state: dict,
    prediction_component: dict,
    global_shift_confidence: float,
) -> np.ndarray:
    """Adaptive physical position uncertainty for one track."""

    frame_gap = max(
        int(prediction_component["frame_gap"]),
        1,
    )

    sigma = (
        BASE_POSITION_SIGMA_ZYX_UM
        * np.sqrt(frame_gap)
    ).astype(float)

    sigma += (
        POSITION_SIGMA_GLOBAL_UNCERTAINTY_UM
        * (1.0 - np.clip(
            global_shift_confidence,
            0.0,
            1.0,
        ))
    )

    sigma += (
        POSITION_SIGMA_PER_MISSING_FRAME_UM
        * max(frame_gap - 1, 0)
    )

    if state["position_residual_samples"] > 0:
        sigma += (
            POSITION_SIGMA_TRACK_RESIDUAL_WEIGHT
            * state["position_residual_ema_zyx"]
        )

    if (
        state["boundary_pending"]
        or bool(
            state["last_detection"].get(
                "touches_boundary",
                False,
            )
        )
    ):
        sigma *= POSITION_SIGMA_BOUNDARY_SCALE

    return np.maximum(sigma, 0.25)


def track_miss_probability(
    *,
    state: dict,
    global_shift_confidence: float,
) -> float:
    """Context-dependent prior probability that a track is missed."""

    is_boundary_context = (
        state["boundary_pending"]
        or bool(
            state["last_detection"].get(
                "touches_boundary",
                False,
            )
        )
    )

    if state["missed_frames"] > 0:
        probability = (
            MISS_PROBABILITY_PENDING_BOUNDARY
            if is_boundary_context
            else MISS_PROBABILITY_PENDING_INTERIOR
        )
    else:
        probability = (
            MISS_PROBABILITY_BOUNDARY
            if is_boundary_context
            else MISS_PROBABILITY_INTERIOR
        )

    probability += (
        MISS_GLOBAL_UNCERTAINTY_BONUS
        * (1.0 - np.clip(
            global_shift_confidence,
            0.0,
            1.0,
        ))
    )

    return float(
        np.clip(
            probability,
            PROBABILITY_FLOOR,
            0.95,
        )
    )


def detection_birth_probability(
    detection: pd.Series,
) -> float:
    """Context-dependent prior probability that a detection is new."""

    probability = (
        BIRTH_PROBABILITY_BOUNDARY
        if bool(detection["touches_boundary"])
        else BIRTH_PROBABILITY_INTERIOR
    )

    return float(
        np.clip(
            probability,
            PROBABILITY_FLOOR,
            0.95,
        )
    )


def normalized_choice_probabilities(
    *,
    pair_costs: np.ndarray,
    alternative_cost: float,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Normalize candidate scores against one miss/birth alternative."""

    pair_costs = np.asarray(pair_costs, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    probabilities = np.zeros_like(
        pair_costs,
        dtype=float,
    )

    finite_costs = pair_costs[valid_mask]

    all_costs = np.concatenate(
        [
            finite_costs,
            np.asarray([alternative_cost]),
        ]
    )

    minimum_cost = float(np.min(all_costs))
    scores = np.exp(-(all_costs - minimum_cost))
    denominator = float(np.sum(scores))

    if finite_costs.size:
        probabilities[valid_mask] = (
            scores[:-1] / denominator
        )

    alternative_probability = float(
        scores[-1] / denominator
    )

    return probabilities, alternative_probability


def augmented_assignment(
    *,
    pair_cost_matrix: np.ndarray,
    miss_costs: np.ndarray,
    birth_costs: np.ndarray,
) -> dict:
    """Solve one-to-one pair/miss/birth assignment jointly."""

    track_count, detection_count = (
        pair_cost_matrix.shape
    )

    augmented_cost = np.full(
        (
            track_count + detection_count,
            detection_count + track_count,
        ),
        INVALID_COST,
        dtype=float,
    )

    augmented_cost[
        :track_count,
        :detection_count,
    ] = pair_cost_matrix

    # Each track has one private miss column.
    if track_count:
        augmented_cost[
            np.arange(track_count),
            detection_count + np.arange(track_count),
        ] = miss_costs

    # Each detection has one private birth row.
    if detection_count:
        augmented_cost[
            track_count + np.arange(detection_count),
            np.arange(detection_count),
        ] = birth_costs

    # Dummy birth rows and miss columns absorb one another at zero cost.
    augmented_cost[
        track_count:,
        detection_count:,
    ] = 0.0

    augmented_rows, augmented_cols = (
        linear_sum_assignment(
            augmented_cost
        )
    )

    pair_rows = []
    pair_cols = []
    missed_state_indices = []
    birth_detection_indices = []

    for row, col in zip(
        augmented_rows,
        augmented_cols,
    ):
        if row < track_count and col < detection_count:
            if pair_cost_matrix[row, col] < INVALID_COST:
                pair_rows.append(int(row))
                pair_cols.append(int(col))
            else:
                raise RuntimeError(
                    "Augmented assignment selected an invalid pair."
                )
        elif row < track_count and col >= detection_count:
            missed_state_indices.append(int(row))
        elif row >= track_count and col < detection_count:
            birth_detection_indices.append(int(col))

    return {
        "rows": np.asarray(pair_rows, dtype=int),
        "cols": np.asarray(pair_cols, dtype=int),
        "missed_state_indices": np.asarray(
            sorted(missed_state_indices),
            dtype=int,
        ),
        "birth_detection_indices": np.asarray(
            sorted(birth_detection_indices),
            dtype=int,
        ),
        "augmented_cost_matrix": augmented_cost,
        "augmented_rows": augmented_rows,
        "augmented_cols": augmented_cols,
        "objective_cost": float(
            np.sum(
                augmented_cost[
                    augmented_rows,
                    augmented_cols,
                ]
            )
        ),
    }

def assign_track_states(
    *,
    states: list[dict],
    detections: pd.DataFrame,
    current_frame: int,
    global_shift_history: dict[int, np.ndarray],
    global_shift_confidence: float,
):
    """Assign tracks using soft pair likelihoods and explicit miss/birth options."""

    track_count = len(states)
    detection_count = len(detections)

    if track_count == 0 or detection_count == 0:
        # Preserve explicit miss/birth decisions even when one side is empty.
        rows = np.array([], dtype=int)
        cols = np.array([], dtype=int)
        missed = np.arange(track_count, dtype=int)
        births = np.arange(detection_count, dtype=int)

        miss_probabilities = np.asarray(
            [
                track_miss_probability(
                    state=state,
                    global_shift_confidence=global_shift_confidence,
                )
                for state in states
            ],
            dtype=float,
        )

        birth_probabilities = np.asarray(
            [
                detection_birth_probability(
                    detections.iloc[index]
                )
                for index in range(detection_count)
            ],
            dtype=float,
        )

        return {
            "rows": rows,
            "cols": cols,
            "missed_state_indices": missed,
            "birth_detection_indices": births,
            "distance_matrix": np.empty((track_count, detection_count)),
            "position_sigma_zyx": np.empty((track_count, 3)),
            "mahalanobis_squared": np.empty((track_count, detection_count)),
            "volume_ratio": np.empty((track_count, detection_count)),
            "log_volume_change": np.empty((track_count, detection_count)),
            "pair_cost_matrix": np.empty((track_count, detection_count)),
            "cost_matrix": np.empty((track_count, detection_count)),
            "position_cost": np.empty((track_count, detection_count)),
            "volume_cost": np.empty((track_count, detection_count)),
            "size_cost": np.empty((track_count, detection_count)),
            "shape_cost": np.empty((track_count, detection_count)),
            "intensity_cost": np.empty((track_count, detection_count)),
            "bbox_cost": np.empty((track_count, detection_count)),
            "motion_cost": np.empty((track_count, detection_count)),
            "face_cost": np.empty((track_count, detection_count)),
            "safety_invalid": np.empty((track_count, detection_count), dtype=bool),
            "absolute_distance_limit": np.empty((track_count, detection_count)),
            "absolute_volume_ratio_limit": np.empty((track_count, detection_count)),
            "predicted_positions": np.empty((track_count, 3)),
            "global_only_positions": np.empty((track_count, 3)),
            "relative_motion_weights": np.empty(track_count),
            "boundary_related": np.empty((track_count, detection_count), dtype=bool),
            "prediction_components": [],
            "miss_probabilities": miss_probabilities,
            "miss_costs": negative_log_probability(miss_probabilities),
            "birth_probabilities": birth_probabilities,
            "birth_costs": negative_log_probability(birth_probabilities),
            "track_candidate_probabilities": np.empty((track_count, detection_count)),
            "detection_candidate_probabilities": np.empty((track_count, detection_count)),
            "association_probabilities": np.empty((track_count, detection_count)),
            "track_no_match_probabilities": miss_probabilities.copy(),
            "detection_birth_choice_probabilities": birth_probabilities.copy(),
            "track_probability_margins": np.zeros(track_count),
            "objective_cost": float(
                np.sum(negative_log_probability(miss_probabilities))
                + np.sum(negative_log_probability(birth_probabilities))
            ),
        }

    prediction_components = [
        predict_state_components(
            state=state,
            target_frame=current_frame,
            global_shift_history=global_shift_history,
        )
        for state in states
    ]

    predicted_positions = np.vstack(
        [
            item["predicted_position"]
            for item in prediction_components
        ]
    )

    global_only_positions = np.vstack(
        [
            state["last_position_physical"]
            + item["global_displacement"]
            for state, item in zip(
                states,
                prediction_components,
            )
        ]
    )

    relative_motion_weights = np.asarray(
        [
            item["relative_weight"]
            for item in prediction_components
        ],
        dtype=float,
    )

    current_positions = physical_coordinates(detections)

    residual_vectors = (
        current_positions[None, :, :]
        - predicted_positions[:, None, :]
    )

    distance_matrix = np.linalg.norm(
        residual_vectors,
        axis=2,
    )

    position_sigma_zyx = np.vstack(
        [
            track_position_sigma_zyx(
                state=state,
                prediction_component=component,
                global_shift_confidence=global_shift_confidence,
            )
            for state, component in zip(
                states,
                prediction_components,
            )
        ]
    )

    mahalanobis_squared = np.sum(
        (
            residual_vectors
            / position_sigma_zyx[:, None, :]
        ) ** 2,
        axis=2,
    )

    position_cost = student_t_vector_cost(
        mahalanobis_squared,
        POSITION_STUDENT_T_DOF,
        dimension=3,
    )

    frame_gaps = np.asarray(
        [
            current_frame - state["last_frame"]
            for state in states
        ],
        dtype=int,
    )

    previous_boundary = np.asarray(
        [
            bool(
                state["last_detection"].get(
                    "touches_boundary",
                    False,
                )
            )
            or state["boundary_pending"]
            for state in states
        ],
        dtype=bool,
    )

    current_boundary = detections[
        "touches_boundary"
    ].to_numpy(dtype=bool)

    boundary_related = (
        previous_boundary[:, None]
        | current_boundary[None, :]
    )

    previous_volume = np.asarray(
        [
            state_reference_value(
                state,
                "volume_voxels",
            )
            for state in states
        ],
        dtype=float,
    )

    current_volume = detections[
        "volume_voxels"
    ].to_numpy(dtype=float)

    safe_previous_volume = np.maximum(
        previous_volume,
        EPS,
    )
    safe_current_volume = np.maximum(
        current_volume,
        EPS,
    )

    log_volume_change = np.abs(
        np.log(
            safe_current_volume[None, :]
            / safe_previous_volume[:, None]
        )
    )

    volume_ratio = np.exp(log_volume_change)

    volume_scale = np.where(
        boundary_related,
        VOLUME_LOG_SCALE_BOUNDARY,
        VOLUME_LOG_SCALE_INTERIOR,
    )

    volume_cost = student_t_scalar_cost(
        log_volume_change,
        volume_scale,
        VOLUME_STUDENT_T_DOF,
    )

    size_residual = feature_group_cost(
        states,
        detections,
        SIZE_FEATURES,
    )
    shape_residual = feature_group_cost(
        states,
        detections,
        SHAPE_FEATURES,
    )
    intensity_residual = feature_group_cost(
        states,
        detections,
        INTENSITY_FEATURES,
    )
    bbox_residual = feature_group_cost(
        states,
        detections,
        BBOX_FEATURES,
    )

    size_cost = student_t_scalar_cost(
        size_residual,
        SIZE_RELATIVE_SCALE,
        FEATURE_STUDENT_T_DOF,
    )
    shape_cost = student_t_scalar_cost(
        shape_residual,
        SHAPE_RELATIVE_SCALE,
        FEATURE_STUDENT_T_DOF,
    )
    intensity_cost = student_t_scalar_cost(
        intensity_residual,
        INTENSITY_RELATIVE_SCALE,
        FEATURE_STUDENT_T_DOF,
    )
    bbox_cost = student_t_scalar_cost(
        bbox_residual,
        BBOX_RELATIVE_SCALE,
        FEATURE_STUDENT_T_DOF,
    )

    motion_cost = relative_motion_consistency_cost(
        states=states,
        current_positions=current_positions,
        current_frame=current_frame,
        global_shift_history=global_shift_history,
        prediction_components=prediction_components,
    )

    face_cost = boundary_face_cost(
        states,
        detections,
    )

    interior_cost = (
        INTERIOR_W_POSITION * position_cost
        + INTERIOR_W_VOLUME * volume_cost
        + INTERIOR_W_SIZE * size_cost
        + INTERIOR_W_SHAPE * shape_cost
        + INTERIOR_W_INTENSITY * intensity_cost
        + INTERIOR_W_BBOX * bbox_cost
        + INTERIOR_W_MOTION * motion_cost
    )

    boundary_cost = (
        BOUNDARY_W_POSITION * position_cost
        + BOUNDARY_W_VOLUME * volume_cost
        + BOUNDARY_W_MOTION * motion_cost
        + BOUNDARY_W_INTENSITY * intensity_cost
        + BOUNDARY_W_FACE * face_cost
    )

    pair_cost_matrix = np.where(
        boundary_related,
        boundary_cost,
        interior_cost,
    )

    absolute_distance_limit = np.where(
        boundary_related,
        (
            ABSOLUTE_MAX_DISTANCE_BOUNDARY_UM
            + ABSOLUTE_DISTANCE_PER_MISSING_FRAME_UM
            * np.maximum(frame_gaps[:, None] - 1, 0)
        ),
        (
            ABSOLUTE_MAX_DISTANCE_INTERIOR_UM
            + ABSOLUTE_DISTANCE_PER_MISSING_FRAME_UM
            * np.maximum(frame_gaps[:, None] - 1, 0)
        ),
    )

    absolute_volume_ratio_limit = np.where(
        boundary_related,
        ABSOLUTE_MAX_VOLUME_RATIO_BOUNDARY,
        ABSOLUTE_MAX_VOLUME_RATIO_INTERIOR,
    )

    safety_invalid = (
        (distance_matrix > absolute_distance_limit)
        | (volume_ratio > absolute_volume_ratio_limit)
        | ~np.isfinite(pair_cost_matrix)
    )

    pair_cost_matrix = pair_cost_matrix.copy()
    pair_cost_matrix[safety_invalid] = INVALID_COST

    miss_probabilities = np.asarray(
        [
            track_miss_probability(
                state=state,
                global_shift_confidence=global_shift_confidence,
            )
            for state in states
        ],
        dtype=float,
    )
    miss_costs = negative_log_probability(
        miss_probabilities
    )

    birth_probabilities = np.asarray(
        [
            detection_birth_probability(
                detections.iloc[index]
            )
            for index in range(detection_count)
        ],
        dtype=float,
    )
    birth_costs = negative_log_probability(
        birth_probabilities
    )

    assignment_result = augmented_assignment(
        pair_cost_matrix=pair_cost_matrix,
        miss_costs=miss_costs,
        birth_costs=birth_costs,
    )

    valid_pair_mask = ~safety_invalid

    track_candidate_probabilities = np.zeros_like(
        pair_cost_matrix,
        dtype=float,
    )
    track_no_match_probabilities = np.zeros(
        track_count,
        dtype=float,
    )

    for state_index in range(track_count):
        (
            track_candidate_probabilities[state_index],
            track_no_match_probabilities[state_index],
        ) = normalized_choice_probabilities(
            pair_costs=pair_cost_matrix[state_index],
            alternative_cost=float(miss_costs[state_index]),
            valid_mask=valid_pair_mask[state_index],
        )

    detection_candidate_probabilities = np.zeros_like(
        pair_cost_matrix,
        dtype=float,
    )
    detection_birth_choice_probabilities = np.zeros(
        detection_count,
        dtype=float,
    )

    for detection_index in range(detection_count):
        column_probabilities, birth_choice_probability = (
            normalized_choice_probabilities(
                pair_costs=pair_cost_matrix[:, detection_index],
                alternative_cost=float(birth_costs[detection_index]),
                valid_mask=valid_pair_mask[:, detection_index],
            )
        )

        detection_candidate_probabilities[
            :, detection_index
        ] = column_probabilities
        detection_birth_choice_probabilities[
            detection_index
        ] = birth_choice_probability

    association_probabilities = np.sqrt(
        track_candidate_probabilities
        * detection_candidate_probabilities
    )

    # Margin of the globally selected decision relative to that track's
    # best local alternative. It may be negative when the global one-to-one
    # solution deliberately chooses a locally second-best option to avoid a
    # worse conflict elsewhere.
    track_probability_margins = np.zeros(
        track_count,
        dtype=float,
    )

    selected_detection_by_state = {
        int(state_index): int(detection_index)
        for state_index, detection_index in zip(
            assignment_result["rows"],
            assignment_result["cols"],
        )
    }

    missed_state_set = set(
        assignment_result["missed_state_indices"].tolist()
    )

    for state_index in range(track_count):
        selected_detection = selected_detection_by_state.get(
            state_index
        )

        if selected_detection is not None:
            selected_probability = float(
                track_candidate_probabilities[
                    state_index,
                    selected_detection,
                ]
            )

            alternative_probabilities = np.concatenate(
                [
                    np.delete(
                        track_candidate_probabilities[state_index],
                        selected_detection,
                    ),
                    np.asarray([
                        track_no_match_probabilities[state_index]
                    ]),
                ]
            )
        elif state_index in missed_state_set:
            selected_probability = float(
                track_no_match_probabilities[state_index]
            )
            alternative_probabilities = (
                track_candidate_probabilities[state_index]
            )
        else:
            # Defensive fallback; every track row should be either matched
            # or assigned to its private miss column.
            selected_probability = 0.0
            alternative_probabilities = (
                track_candidate_probabilities[state_index]
            )

        best_alternative = (
            float(np.max(alternative_probabilities))
            if alternative_probabilities.size
            else 0.0
        )

        track_probability_margins[state_index] = (
            selected_probability - best_alternative
        )

    return {
        **assignment_result,
        "distance_matrix": distance_matrix,
        "position_sigma_zyx": position_sigma_zyx,
        "mahalanobis_squared": mahalanobis_squared,
        "volume_ratio": volume_ratio,
        "log_volume_change": log_volume_change,
        "pair_cost_matrix": pair_cost_matrix,
        # Compatibility alias for existing downstream code.
        "cost_matrix": pair_cost_matrix,
        "position_cost": position_cost,
        "volume_cost": volume_cost,
        "size_cost": size_cost,
        "shape_cost": shape_cost,
        "intensity_cost": intensity_cost,
        "bbox_cost": bbox_cost,
        "motion_cost": motion_cost,
        "face_cost": face_cost,
        "safety_invalid": safety_invalid,
        "absolute_distance_limit": absolute_distance_limit,
        "absolute_volume_ratio_limit": absolute_volume_ratio_limit,
        "predicted_positions": predicted_positions,
        "global_only_positions": global_only_positions,
        "relative_motion_weights": relative_motion_weights,
        "boundary_related": boundary_related,
        "prediction_components": prediction_components,
        "miss_probabilities": miss_probabilities,
        "miss_costs": miss_costs,
        "birth_probabilities": birth_probabilities,
        "birth_costs": birth_costs,
        "track_candidate_probabilities": track_candidate_probabilities,
        "detection_candidate_probabilities": detection_candidate_probabilities,
        "association_probabilities": association_probabilities,
        "track_no_match_probabilities": track_no_match_probabilities,
        "detection_birth_choice_probabilities": detection_birth_choice_probabilities,
        "track_probability_margins": track_probability_margins,
    }

def refine_global_shift_from_assignment(
    *,
    states: list[dict],
    detections: pd.DataFrame,
    current_frame: int,
    assignment: dict,
) -> dict:
    """Refine global shift using confident selected immediate-frame matches."""

    current_positions = physical_coordinates(
        detections
    )

    candidate_displacements = []
    candidate_pairs = []

    def collect_candidates(
        *,
        allow_boundary: bool,
    ) -> None:
        candidate_displacements.clear()
        candidate_pairs.clear()

        for state_index, detection_index in zip(
            assignment["rows"],
            assignment["cols"],
        ):
            state = states[int(state_index)]

            if state["last_frame"] != current_frame - 1:
                continue

            detection = detections.iloc[int(detection_index)]

            previous_boundary = bool(
                state["last_detection"].get(
                    "touches_boundary",
                    False,
                )
            )
            current_boundary = bool(
                detection["touches_boundary"]
            )

            if (
                not allow_boundary
                and (previous_boundary or current_boundary)
            ):
                continue

            association_probability = assignment[
                "association_probabilities"
            ][state_index, detection_index]

            probability_margin = assignment[
                "track_probability_margins"
            ][state_index]

            prediction_error = assignment[
                "distance_matrix"
            ][state_index, detection_index]

            if (
                association_probability
                < GLOBAL_REFINEMENT_MIN_ASSOCIATION_PROBABILITY
            ):
                continue

            if (
                probability_margin
                < GLOBAL_REFINEMENT_MIN_PROBABILITY_MARGIN
            ):
                continue

            if (
                prediction_error
                > GLOBAL_REFINEMENT_MAX_PREDICTION_ERROR_UM
            ):
                continue

            candidate_displacements.append(
                current_positions[int(detection_index)]
                - state["last_position_physical"]
            )
            candidate_pairs.append(
                (int(state_index), int(detection_index))
            )

    collect_candidates(allow_boundary=False)

    if len(candidate_displacements) < GLOBAL_REFINEMENT_MIN_MATCHES:
        collect_candidates(allow_boundary=True)

    if len(candidate_displacements) < GLOBAL_REFINEMENT_MIN_MATCHES:
        return {
            "accepted": False,
            "shift": np.zeros(3, dtype=float),
            "candidate_count": int(len(candidate_displacements)),
            "inlier_count": 0,
            "dispersion_um": np.nan,
            "confidence": 0.0,
        }

    summary = robust_displacement_summary(
        np.asarray(candidate_displacements, dtype=float)
    )

    accepted = (
        summary["inlier_count"]
        >= GLOBAL_REFINEMENT_MIN_MATCHES
    )

    return {
        "accepted": bool(accepted),
        "shift": summary["shift"],
        "candidate_count": int(len(candidate_displacements)),
        "inlier_count": summary["inlier_count"],
        "dispersion_um": summary["dispersion_um"],
        "confidence": summary["confidence"],
    }

def assignment_summary(
    assignment: dict,
) -> dict:
    """Return compact quality statistics for one probabilistic assignment."""

    rows = assignment["rows"]
    cols = assignment["cols"]

    if len(rows) == 0:
        median_distance = np.inf
        mean_pair_cost = np.inf
        median_probability = 0.0
    else:
        median_distance = float(
            np.median(
                assignment["distance_matrix"][rows, cols]
            )
        )
        mean_pair_cost = float(
            np.mean(
                assignment["pair_cost_matrix"][rows, cols]
            )
        )
        median_probability = float(
            np.median(
                assignment["association_probabilities"][rows, cols]
            )
        )

    decision_count = (
        len(assignment["rows"])
        + len(assignment["missed_state_indices"])
        + len(assignment["birth_detection_indices"])
    )

    normalized_objective = (
        assignment["objective_cost"]
        / max(decision_count, 1)
    )

    return {
        "matches": int(len(rows)),
        "misses": int(len(assignment["missed_state_indices"])),
        "births": int(len(assignment["birth_detection_indices"])),
        "median_distance_um": median_distance,
        "mean_pair_cost": mean_pair_cost,
        "median_association_probability": median_probability,
        "objective_cost": float(assignment["objective_cost"]),
        "normalized_objective": float(normalized_objective),
    }

def should_use_refined_assignment(
    *,
    initial_assignment: dict,
    refined_assignment: dict,
) -> bool:
    """Accept refinement when its joint pair/miss/birth objective improves."""

    initial = assignment_summary(initial_assignment)
    refined = assignment_summary(refined_assignment)

    if (
        refined["matches"]
        < initial["matches"] - GLOBAL_REFINEMENT_MAX_MATCH_LOSS
    ):
        return False

    if (
        refined["normalized_objective"]
        < initial["normalized_objective"]
        - GLOBAL_REFINEMENT_OBJECTIVE_TOLERANCE
    ):
        return True

    if refined["matches"] > initial["matches"]:
        return (
            refined["median_distance_um"]
            <= initial["median_distance_um"] + 0.50
        )

    return (
        refined["matches"] == initial["matches"]
        and refined["median_association_probability"]
        > initial["median_association_probability"]
        and refined["median_distance_um"]
        <= initial["median_distance_um"] + 0.25
    )

def update_matched_state(
    *,
    state: dict,
    detection: pd.Series,
    current_frame: int,
    global_shift_history: dict[int, np.ndarray],
    global_shift_confidence: float,
    predicted_position: np.ndarray,
    match_distance_um: float,
    match_cost: float,
    association_probability: float,
    probability_margin: float,
    boundary_related: bool,
) -> dict:
    """Update a selected match and conditionally learn motion/uncertainty."""

    previous_boundary = bool(
        state["last_detection"].get(
            "touches_boundary",
            False,
        )
    )

    was_reacquired = state["missed_frames"] > 0
    was_boundary_pending = bool(state["boundary_pending"])
    was_interior_pending = bool(state["interior_pending"])

    new_position_voxel = detection[
        ["centroid_z", "centroid_y", "centroid_x"]
    ].to_numpy(dtype=float)

    new_position_physical = (
        new_position_voxel * VOXEL_SIZE_ZYX
    )

    frame_gap = current_frame - state["last_frame"]

    global_displacement = cumulative_global_displacement(
        start_frame=state["last_frame"],
        target_frame=current_frame,
        global_shift_history=global_shift_history,
    )

    observed_relative_velocity = (
        new_position_physical
        - state["last_position_physical"]
        - global_displacement
    ) / max(frame_gap, 1)

    observed_relative_speed = float(
        np.linalg.norm(observed_relative_velocity)
    )

    current_boundary = bool(
        detection["touches_boundary"]
    )

    reliable_relative_update = (
        frame_gap == 1
        and not previous_boundary
        and not current_boundary
        and not boundary_related
        and global_shift_confidence
        >= RELATIVE_UPDATE_MIN_GLOBAL_CONFIDENCE
        and association_probability
        >= RELATIVE_UPDATE_MIN_ASSOCIATION_PROBABILITY
        and probability_margin
        >= RELATIVE_UPDATE_MIN_PROBABILITY_MARGIN
        and match_distance_um
        <= RELATIVE_UPDATE_MAX_DISTANCE_UM
        and match_cost
        <= RELATIVE_UPDATE_MAX_PAIR_COST
        and observed_relative_speed
        <= MAX_RELATIVE_VELOCITY_UM_PER_FRAME
    )

    relative_velocity_updated = False

    if reliable_relative_update:
        if state["relative_velocity_valid"]:
            innovation = float(
                np.linalg.norm(
                    observed_relative_velocity
                    - state["relative_velocity_physical"]
                )
            )

            state["relative_velocity_error_ema"] = (
                (1.0 - RELATIVE_ERROR_EMA_ALPHA)
                * state["relative_velocity_error_ema"]
                + RELATIVE_ERROR_EMA_ALPHA
                * innovation
            )

            state["relative_velocity_physical"] = (
                (1.0 - RELATIVE_VELOCITY_EMA_ALPHA)
                * state["relative_velocity_physical"]
                + RELATIVE_VELOCITY_EMA_ALPHA
                * observed_relative_velocity
            )
        else:
            state["relative_velocity_physical"] = (
                observed_relative_velocity
            )
            state["relative_velocity_valid"] = True
            state["relative_velocity_error_ema"] = 0.0

        state["relative_velocity_samples"] += 1
        state["last_relative_update_frame"] = int(current_frame)
        relative_velocity_updated = True
    else:
        state["relative_velocity_updates_rejected"] += 1

    prediction_residual_abs = np.abs(
        new_position_physical
        - np.asarray(predicted_position, dtype=float)
    )

    position_residual_updated = False

    if (
        frame_gap == 1
        and association_probability
        >= POSITION_RESIDUAL_UPDATE_MIN_ASSOCIATION_PROBABILITY
        and probability_margin
        >= POSITION_RESIDUAL_UPDATE_MIN_MARGIN
    ):
        if state["position_residual_samples"] == 0:
            state["position_residual_ema_zyx"] = (
                prediction_residual_abs
            )
        else:
            state["position_residual_ema_zyx"] = (
                (1.0 - POSITION_RESIDUAL_EMA_ALPHA)
                * state["position_residual_ema_zyx"]
                + POSITION_RESIDUAL_EMA_ALPHA
                * prediction_residual_abs
            )

        state["position_residual_samples"] += 1
        position_residual_updated = True

    state["previous_position_physical"] = (
        state["last_position_physical"].copy()
    )
    state["last_position_voxel"] = new_position_voxel
    state["last_position_physical"] = new_position_physical
    state["last_frame"] = int(current_frame)
    state["last_detection"] = detection.to_dict()
    state["missed_frames"] = 0
    state["boundary_pending"] = current_boundary
    state["interior_pending"] = False
    state["last_boundary_faces"] = parse_boundary_faces(
        detection["boundary_faces"]
    )

    update_feature_template(state, detection)

    return {
        "was_reacquired": was_reacquired,
        "was_boundary_pending": was_boundary_pending,
        "was_interior_pending": was_interior_pending,
        "previous_boundary": previous_boundary,
        "relative_velocity_updated": relative_velocity_updated,
        "position_residual_updated": position_residual_updated,
        "prediction_residual_abs_zyx": prediction_residual_abs,
        "observed_relative_velocity": observed_relative_velocity,
        "observed_relative_speed": observed_relative_speed,
        "global_displacement": global_displacement,
    }

# __CODE_APPEND_SENTINEL__
