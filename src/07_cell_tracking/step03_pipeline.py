"""Behavior-preserving Stage 7 tracking entry point."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.diagnostics import DecisionRecord, Provenance, StageTrace

from .step01_config import *
from .step02_association import *


@dataclass(frozen=True)
class TrackingResult:
    tracks: pd.DataFrame
    boundary_events: pd.DataFrame
    boundary_predictions: pd.DataFrame
    missing_predictions: pd.DataFrame
    boundary_counts: pd.DataFrame
    global_motion: pd.DataFrame
    tracking_diagnostics: pd.DataFrame
    association_events: pd.DataFrame
    association_candidates: pd.DataFrame
    track_states: pd.DataFrame
    metadata: dict
    summary: dict


def run_cell_tracking(
    time_frames: list[pd.DataFrame],
    *,
    sample_id: str = "44b6_0113de3b",
    return_diagnostics: bool = False,
):
    """Run the notebook's tracking algorithm without changing its ordering."""

    time_frames = [
        annotate_boundary_metadata(frame)
        for frame in time_frames
    ]

    boundary_counts = pd.DataFrame(
        {
            "frame": np.arange(len(time_frames)),
            "detections": [len(frame) for frame in time_frames],
            "boundary_detections": [
                int(frame["touches_boundary"].sum())
                for frame in time_frames
            ],
        }
    )

    # ============================================================
    # Boundary-aware probabilistic tracking with global-relative motion
    # ============================================================
    
    track_states: dict[int, dict] = {}
    track_records: list[dict] = []
    boundary_events: list[dict] = []
    missing_predictions: list[dict] = []
    global_motion_records: list[dict] = []
    tracking_diagnostic_records: list[dict] = []
    association_event_records: list[dict] = []
    association_candidate_records: list[dict] = []
    
    # Maps target frame t to the global shift from t-1 -> t.
    global_shift_history: dict[int, np.ndarray] = {}
    
    next_track_id = 0
    
    
    def append_track_record(
        *,
        track_id: int,
        frame: int,
        cell_index: int,
        detection: pd.Series,
        state: dict,
        match_type: str,
        match_distance_um: float | None,
        match_cost: float | None,
        association_probability: float | None,
        track_candidate_probability: float | None,
        detection_candidate_probability: float | None,
        probability_margin: float | None,
        position_cost: float | None,
        volume_cost: float | None,
        shape_cost: float | None,
        global_shift_physical: np.ndarray | None,
        global_shift_confidence: float | None,
        relative_velocity_updated: bool,
    ) -> None:
        """Append one observed detection to the output track table."""
    
        if global_shift_physical is None:
            global_shift_physical = np.full(3, np.nan, dtype=float)
    
        next_frame_relative_confidence = relative_motion_confidence(
            state,
            frame_gap=1,
        )
    
        track_records.append(
            {
                "track_id": int(track_id),
                "frame": int(frame),
                "cell": int(cell_index),
                "cell_id": int(detection.get("cell_id", cell_index)),
                "z": float(detection["centroid_z"]),
                "y": float(detection["centroid_y"]),
                "x": float(detection["centroid_x"]),
                "volume": float(detection["volume_voxels"]),
                "touches_boundary": bool(detection["touches_boundary"]),
                "boundary_faces": str(detection["boundary_faces"]),
                "distance_to_boundary_um": float(
                    detection["distance_to_boundary_um"]
                ),
                "boundary_state": (
                    "ACTIVE_BOUNDARY"
                    if bool(detection["touches_boundary"])
                    else "ACTIVE_INTERIOR"
                ),
                "match_type": str(match_type),
                "match_distance_um": (
                    np.nan if match_distance_um is None
                    else float(match_distance_um)
                ),
                "match_cost": (
                    np.nan if match_cost is None
                    else float(match_cost)
                ),
                "association_probability": (
                    np.nan if association_probability is None
                    else float(association_probability)
                ),
                "track_candidate_probability": (
                    np.nan if track_candidate_probability is None
                    else float(track_candidate_probability)
                ),
                "detection_candidate_probability": (
                    np.nan if detection_candidate_probability is None
                    else float(detection_candidate_probability)
                ),
                "probability_margin": (
                    np.nan if probability_margin is None
                    else float(probability_margin)
                ),
                "position_cost": (
                    np.nan if position_cost is None
                    else float(position_cost)
                ),
                "volume_cost": (
                    np.nan if volume_cost is None
                    else float(volume_cost)
                ),
                "shape_cost": (
                    np.nan if shape_cost is None
                    else float(shape_cost)
                ),
                "template_reliable": bool(state["template_reliable"]),
                "global_shift_z_um": float(global_shift_physical[0]),
                "global_shift_y_um": float(global_shift_physical[1]),
                "global_shift_x_um": float(global_shift_physical[2]),
                "global_shift_confidence": (
                    np.nan if global_shift_confidence is None
                    else float(global_shift_confidence)
                ),
                "relative_velocity_z_um_per_frame": float(
                    state["relative_velocity_physical"][0]
                ),
                "relative_velocity_y_um_per_frame": float(
                    state["relative_velocity_physical"][1]
                ),
                "relative_velocity_x_um_per_frame": float(
                    state["relative_velocity_physical"][2]
                ),
                "relative_velocity_valid": bool(
                    state["relative_velocity_valid"]
                ),
                "relative_velocity_samples": int(
                    state["relative_velocity_samples"]
                ),
                "relative_velocity_error_ema_um": float(
                    state["relative_velocity_error_ema"]
                ),
                "relative_motion_confidence_next_frame": float(
                    next_frame_relative_confidence
                ),
                "relative_velocity_updated": bool(
                    relative_velocity_updated
                ),
                "position_residual_ema_z_um": float(
                    state["position_residual_ema_zyx"][0]
                ),
                "position_residual_ema_y_um": float(
                    state["position_residual_ema_zyx"][1]
                ),
                "position_residual_ema_x_um": float(
                    state["position_residual_ema_zyx"][2]
                ),
                "position_residual_samples": int(
                    state["position_residual_samples"]
                ),
            }
        )
    
    
    def safe_median_nearest_distance(
        distance_matrix: np.ndarray,
    ) -> float:
        """Median row-wise nearest distance, or NaN for an empty matrix."""
    
        if (
            distance_matrix.size == 0
            or distance_matrix.shape[0] == 0
            or distance_matrix.shape[1] == 0
        ):
            return np.nan
    
        return float(
            np.median(np.min(distance_matrix, axis=1))
        )
    
    
    def record_top_candidates(
        *,
        assignment: dict,
        eligible_states: list[dict],
        detections: pd.DataFrame,
        current_frame: int,
    ) -> None:
        """Save the most relevant candidate pairs for later failure diagnosis."""
    
        if not SAVE_TOP_ASSOCIATION_CANDIDATES:
            return
    
        selected_by_state = {
            int(state_index): int(detection_index)
            for state_index, detection_index in zip(
                assignment["rows"],
                assignment["cols"],
            )
        }
    
        owner_by_detection = {
            int(detection_index): int(state_index)
            for state_index, detection_index in zip(
                assignment["rows"],
                assignment["cols"],
            )
        }
    
        for state_index, state in enumerate(eligible_states):
            valid_indices = np.flatnonzero(
                ~assignment["safety_invalid"][state_index]
            )
    
            ordered = valid_indices[
                np.argsort(
                    assignment["pair_cost_matrix"][
                        state_index,
                        valid_indices,
                    ]
                )
            ] if valid_indices.size else np.array([], dtype=int)
    
            keep = set(
                ordered[
                    :TOP_ASSOCIATION_CANDIDATES_PER_TRACK
                ].tolist()
            )
    
            selected_detection = selected_by_state.get(state_index)
            if selected_detection is not None:
                keep.add(selected_detection)
    
            for rank, detection_index in enumerate(
                sorted(
                    keep,
                    key=lambda index: assignment[
                        "pair_cost_matrix"
                    ][state_index, index],
                ),
                start=1,
            ):
                detection = detections.iloc[detection_index]
                owner_state_index = owner_by_detection.get(detection_index)
    
                association_candidate_records.append(
                    {
                        "from_frame": int(state["last_frame"]),
                        "to_frame": int(current_frame),
                        "track_id": int(state["track_id"]),
                        "detection_position_index": int(detection_index),
                        "detection_cell_index": int(
                            detections.index[detection_index]
                        ),
                        "candidate_rank_by_pair_cost": int(rank),
                        "selected": bool(
                            selected_detection == detection_index
                        ),
                        "detection_selected_by_track_id": (
                            np.nan
                            if owner_state_index is None
                            else int(
                                eligible_states[
                                    owner_state_index
                                ]["track_id"]
                            )
                        ),
                        "pair_cost": float(
                            assignment["pair_cost_matrix"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "association_probability": float(
                            assignment["association_probabilities"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "track_candidate_probability": float(
                            assignment["track_candidate_probabilities"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "detection_candidate_probability": float(
                            assignment["detection_candidate_probabilities"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "track_no_match_probability": float(
                            assignment["track_no_match_probabilities"][
                                state_index
                            ]
                        ),
                        "probability_margin": float(
                            assignment["track_probability_margins"][
                                state_index
                            ]
                        ),
                        "distance_um": float(
                            assignment["distance_matrix"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "mahalanobis_squared": float(
                            assignment["mahalanobis_squared"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "volume_ratio": float(
                            assignment["volume_ratio"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "log_volume_change": float(
                            assignment["log_volume_change"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "position_cost": float(
                            assignment["position_cost"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "volume_cost": float(
                            assignment["volume_cost"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "size_cost": float(
                            assignment["size_cost"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "shape_cost": float(
                            assignment["shape_cost"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "intensity_cost": float(
                            assignment["intensity_cost"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "bbox_cost": float(
                            assignment["bbox_cost"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "motion_cost": float(
                            assignment["motion_cost"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "face_cost": float(
                            assignment["face_cost"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "boundary_related": bool(
                            assignment["boundary_related"][
                                state_index,
                                detection_index,
                            ]
                        ),
                        "candidate_touches_boundary": bool(
                            detection["touches_boundary"]
                        ),
                    }
                )
    
    
    # ------------------------------------------------------------
    # Initialize tracks from the first frame
    # ------------------------------------------------------------
    
    first_frame = time_frames[0]
    
    for cell_index, detection in first_frame.iterrows():
        track_id = next_track_id
        next_track_id += 1
    
        state = make_track_state(
            track_id=track_id,
            frame=0,
            detection=detection,
        )
        track_states[track_id] = state
    
        match_type = (
            "boundary_entry"
            if bool(detection["touches_boundary"])
            else "initial"
        )
    
        append_track_record(
            track_id=track_id,
            frame=0,
            cell_index=int(cell_index),
            detection=detection,
            state=state,
            match_type=match_type,
            match_distance_um=None,
            match_cost=None,
            association_probability=None,
            track_candidate_probability=None,
            detection_candidate_probability=None,
            probability_margin=None,
            position_cost=None,
            volume_cost=None,
            shape_cost=None,
            global_shift_physical=None,
            global_shift_confidence=None,
            relative_velocity_updated=False,
        )
    
        if bool(detection["touches_boundary"]):
            boundary_events.append(
                {
                    "track_id": track_id,
                    "frame": 0,
                    "event_type": "boundary_entry",
                    "boundary_faces": detection["boundary_faces"],
                    "missing_frames": 0,
                    "reacquired_frame": np.nan,
                    "confidence": np.nan,
                }
            )
    
    
    # ------------------------------------------------------------
    # Process each subsequent frame
    # ------------------------------------------------------------
    
    for current_frame in range(1, len(time_frames)):
        detections = time_frames[current_frame]
    
        eligible_states = []
    
        for state in track_states.values():
            if not state["active"]:
                continue
    
            frame_gap = current_frame - state["last_frame"]
    
            if frame_gap == 1:
                eligible_states.append(state)
                continue
    
            if (
                state["boundary_pending"]
                and frame_gap <= BOUNDARY_MAX_MISSING_FRAMES + 1
            ):
                eligible_states.append(state)
                continue
    
            if (
                state["interior_pending"]
                and frame_gap <= INTERIOR_MAX_MISSING_FRAMES + 1
            ):
                eligible_states.append(state)
                continue
    
            state["active"] = False
    
        previous_frame_positions = np.asarray(
            [
                state["last_position_physical"]
                for state in eligible_states
                if state["last_frame"] == current_frame - 1
            ],
            dtype=float,
        ).reshape(-1, 3)
    
        current_positions = physical_coordinates(detections)
    
        # --------------------------------------------------------
        # Pass 1: robust global shift and provisional assignment
        # --------------------------------------------------------
    
        initial_global_estimate = estimate_global_shift_physical(
            previous_frame_positions,
            current_positions,
        )
    
        initial_global_shift = initial_global_estimate["shift"]
        initial_global_confidence = float(
            initial_global_estimate["confidence"]
        )
    
        global_shift_history[current_frame] = (
            initial_global_shift.copy()
        )
    
        initial_assignment = assign_track_states(
            states=eligible_states,
            detections=detections,
            current_frame=current_frame,
            global_shift_history=global_shift_history,
            global_shift_confidence=initial_global_confidence,
        )
    
        # --------------------------------------------------------
        # Pass 2: refine global shift from confident assignments
        # --------------------------------------------------------
    
        refinement = refine_global_shift_from_assignment(
            states=eligible_states,
            detections=detections,
            current_frame=current_frame,
            assignment=initial_assignment,
        )
    
        refinement_used = False
        assignment = initial_assignment
        final_global_shift = initial_global_shift.copy()
        final_global_confidence = initial_global_confidence
    
        if refinement["accepted"]:
            global_shift_history[current_frame] = (
                refinement["shift"].copy()
            )
    
            refined_assignment = assign_track_states(
                states=eligible_states,
                detections=detections,
                current_frame=current_frame,
                global_shift_history=global_shift_history,
                global_shift_confidence=float(
                    refinement["confidence"]
                ),
            )
    
            if should_use_refined_assignment(
                initial_assignment=initial_assignment,
                refined_assignment=refined_assignment,
            ):
                assignment = refined_assignment
                final_global_shift = refinement["shift"].copy()
                final_global_confidence = float(
                    refinement["confidence"]
                )
                refinement_used = True
            else:
                global_shift_history[current_frame] = (
                    initial_global_shift.copy()
                )
    
        previous_global_shift = global_shift_history.get(
            current_frame - 1,
            np.zeros(3, dtype=float),
        )
    
        direction_change_deg = (
            vector_angle_degrees(
                previous_global_shift,
                final_global_shift,
            )
            if current_frame > 1
            else np.nan
        )
    
        shift_change_um = float(
            np.linalg.norm(
                final_global_shift - previous_global_shift
            )
        )
    
        initial_assignment_stats = assignment_summary(
            initial_assignment
        )
        final_assignment_stats = assignment_summary(
            assignment
        )
    
        global_motion_records.append(
            {
                "from_frame": int(current_frame - 1),
                "to_frame": int(current_frame),
                "initial_method": initial_global_estimate["method"],
                "initial_pair_count": int(
                    initial_global_estimate["pair_count"]
                ),
                "initial_inlier_count": int(
                    initial_global_estimate["inlier_count"]
                ),
                "initial_dispersion_um": float(
                    initial_global_estimate["dispersion_um"]
                ),
                "initial_confidence": initial_global_confidence,
                "initial_shift_z_um": float(initial_global_shift[0]),
                "initial_shift_y_um": float(initial_global_shift[1]),
                "initial_shift_x_um": float(initial_global_shift[2]),
                "refinement_accepted": bool(refinement["accepted"]),
                "refinement_used": bool(refinement_used),
                "refinement_candidate_count": int(
                    refinement["candidate_count"]
                ),
                "refinement_inlier_count": int(
                    refinement["inlier_count"]
                ),
                "refinement_dispersion_um": float(
                    refinement["dispersion_um"]
                ),
                "refinement_confidence": float(
                    refinement["confidence"]
                ),
                "final_shift_z_um": float(final_global_shift[0]),
                "final_shift_y_um": float(final_global_shift[1]),
                "final_shift_x_um": float(final_global_shift[2]),
                "final_shift_magnitude_um": float(
                    np.linalg.norm(final_global_shift)
                ),
                "final_confidence": final_global_confidence,
                "shift_change_from_previous_um": shift_change_um,
                "direction_change_deg": float(direction_change_deg),
                "initial_matches": int(initial_assignment_stats["matches"]),
                "final_matches": int(final_assignment_stats["matches"]),
                "initial_misses": int(initial_assignment_stats["misses"]),
                "final_misses": int(final_assignment_stats["misses"]),
                "initial_births": int(initial_assignment_stats["births"]),
                "final_births": int(final_assignment_stats["births"]),
                "initial_normalized_objective": float(
                    initial_assignment_stats["normalized_objective"]
                ),
                "final_normalized_objective": float(
                    final_assignment_stats["normalized_objective"]
                ),
                "initial_median_match_distance_um": float(
                    initial_assignment_stats["median_distance_um"]
                ),
                "final_median_match_distance_um": float(
                    final_assignment_stats["median_distance_um"]
                ),
                "final_median_association_probability": float(
                    final_assignment_stats[
                        "median_association_probability"
                    ]
                ),
            }
        )
    
        record_top_candidates(
            assignment=assignment,
            eligible_states=eligible_states,
            detections=detections,
            current_frame=current_frame,
        )
    
        distance_matrix = assignment["distance_matrix"]
        global_only_distances = (
            cdist(
                assignment["global_only_positions"],
                current_positions,
            )
            if len(eligible_states) > 0 and len(detections) > 0
            else np.empty((len(eligible_states), len(detections)))
        )
    
        prediction_median_um = safe_median_nearest_distance(
            distance_matrix
        )
        global_only_median_um = safe_median_nearest_distance(
            global_only_distances
        )
    
        relative_weights = assignment["relative_motion_weights"]
        median_relative_weight = (
            float(np.median(relative_weights))
            if len(relative_weights) > 0
            else np.nan
        )
    
        rows = assignment["rows"]
        cols = assignment["cols"]
        missed_state_indices = assignment[
            "missed_state_indices"
        ]
        birth_detection_indices = assignment[
            "birth_detection_indices"
        ]
    
        matched_state_indices = set(rows.tolist())
        matched_detection_indices = set(cols.tolist())
        relative_updates_this_frame = 0
    
        # --------------------------------------------------------
        # Continue selected matches
        # --------------------------------------------------------
    
        for state_index, detection_index in zip(rows, cols):
            state = eligible_states[int(state_index)]
            detection = detections.iloc[int(detection_index)]
    
            previous_faces = "|".join(
                sorted(state["last_boundary_faces"])
            )
    
            distance_um = float(
                assignment["distance_matrix"][
                    state_index,
                    detection_index,
                ]
            )
            pair_cost = float(
                assignment["pair_cost_matrix"][
                    state_index,
                    detection_index,
                ]
            )
            association_probability = float(
                assignment["association_probabilities"][
                    state_index,
                    detection_index,
                ]
            )
            track_candidate_probability = float(
                assignment["track_candidate_probabilities"][
                    state_index,
                    detection_index,
                ]
            )
            detection_candidate_probability = float(
                assignment["detection_candidate_probabilities"][
                    state_index,
                    detection_index,
                ]
            )
            probability_margin = float(
                assignment["track_probability_margins"][
                    state_index
                ]
            )
            is_boundary_related = bool(
                assignment["boundary_related"][
                    state_index,
                    detection_index,
                ]
            )
    
            update_result = update_matched_state(
                state=state,
                detection=detection,
                current_frame=current_frame,
                global_shift_history=global_shift_history,
                global_shift_confidence=final_global_confidence,
                predicted_position=assignment[
                    "predicted_positions"
                ][state_index],
                match_distance_um=distance_um,
                match_cost=pair_cost,
                association_probability=association_probability,
                probability_margin=probability_margin,
                boundary_related=is_boundary_related,
            )
    
            relative_velocity_updated = bool(
                update_result["relative_velocity_updated"]
            )
            relative_updates_this_frame += int(
                relative_velocity_updated
            )
    
            if update_result["was_reacquired"]:
                if update_result["was_boundary_pending"]:
                    match_type = "boundary_reacquired"
                else:
                    match_type = "interior_reacquired"
            elif is_boundary_related:
                match_type = "boundary_partial"
            else:
                match_type = "normal"
    
            append_track_record(
                track_id=state["track_id"],
                frame=current_frame,
                cell_index=int(detections.index[detection_index]),
                detection=detection,
                state=state,
                match_type=match_type,
                match_distance_um=distance_um,
                match_cost=pair_cost,
                association_probability=association_probability,
                track_candidate_probability=track_candidate_probability,
                detection_candidate_probability=detection_candidate_probability,
                probability_margin=probability_margin,
                position_cost=float(
                    assignment["position_cost"][
                        state_index,
                        detection_index,
                    ]
                ),
                volume_cost=float(
                    assignment["volume_cost"][
                        state_index,
                        detection_index,
                    ]
                ),
                shape_cost=float(
                    assignment["shape_cost"][
                        state_index,
                        detection_index,
                    ]
                ),
                global_shift_physical=final_global_shift,
                global_shift_confidence=final_global_confidence,
                relative_velocity_updated=relative_velocity_updated,
            )
    
            association_event_records.append(
                {
                    "from_frame": int(current_frame - 1),
                    "to_frame": int(current_frame),
                    "decision_type": "match",
                    "track_id": int(state["track_id"]),
                    "detection_position_index": int(detection_index),
                    "detection_cell_index": int(
                        detections.index[detection_index]
                    ),
                    "pair_cost": pair_cost,
                    "decision_cost": pair_cost,
                    "association_probability": association_probability,
                    "track_candidate_probability": track_candidate_probability,
                    "detection_candidate_probability": detection_candidate_probability,
                    "alternative_probability": float(
                        assignment["track_no_match_probabilities"][
                            state_index
                        ]
                    ),
                    "probability_margin": probability_margin,
                    "distance_um": distance_um,
                    "volume_ratio": float(
                        assignment["volume_ratio"][
                            state_index,
                            detection_index,
                        ]
                    ),
                    "position_cost": float(
                        assignment["position_cost"][
                            state_index,
                            detection_index,
                        ]
                    ),
                    "volume_cost": float(
                        assignment["volume_cost"][
                            state_index,
                            detection_index,
                        ]
                    ),
                    "shape_cost": float(
                        assignment["shape_cost"][
                            state_index,
                            detection_index,
                        ]
                    ),
                    "boundary_related": is_boundary_related,
                }
            )
    
            current_boundary = bool(detection["touches_boundary"])
            previous_boundary = bool(
                update_result["previous_boundary"]
            )
    
            if (
                update_result["was_reacquired"]
                and update_result["was_boundary_pending"]
            ):
                boundary_events.append(
                    {
                        "track_id": state["track_id"],
                        "frame": current_frame,
                        "event_type": "boundary_reacquired",
                        "boundary_faces": detection["boundary_faces"],
                        "missing_frames": 0,
                        "reacquired_frame": current_frame,
                        "confidence": association_probability,
                    }
                )
            elif previous_boundary and not current_boundary:
                boundary_events.append(
                    {
                        "track_id": state["track_id"],
                        "frame": current_frame,
                        "event_type": "entered_interior",
                        "boundary_faces": previous_faces,
                        "missing_frames": 0,
                        "reacquired_frame": np.nan,
                        "confidence": association_probability,
                    }
                )
            elif not previous_boundary and current_boundary:
                boundary_events.append(
                    {
                        "track_id": state["track_id"],
                        "frame": current_frame,
                        "event_type": "boundary_exit_started",
                        "boundary_faces": detection["boundary_faces"],
                        "missing_frames": 0,
                        "reacquired_frame": np.nan,
                        "confidence": association_probability,
                    }
                )
    
        # --------------------------------------------------------
        # Apply explicit miss decisions and preserve short memory
        # --------------------------------------------------------
    
        for state_index in missed_state_indices:
            state = eligible_states[int(state_index)]
    
            last_was_boundary = bool(
                state["last_detection"].get(
                    "touches_boundary",
                    False,
                )
            )
            boundary_context = (
                state["boundary_pending"]
                or last_was_boundary
            )
    
            state["missed_frames"] += 1
    
            if boundary_context:
                state["boundary_pending"] = True
                state["interior_pending"] = False
                pending_kind = "boundary"
                maximum_missing_frames = BOUNDARY_MAX_MISSING_FRAMES
            else:
                state["boundary_pending"] = False
                state["interior_pending"] = True
                pending_kind = "interior"
                maximum_missing_frames = INTERIOR_MAX_MISSING_FRAMES
    
            prediction = predict_state_components(
                state=state,
                target_frame=current_frame,
                global_shift_history=global_shift_history,
            )
            predicted_position = prediction["predicted_position"]
    
            missing_predictions.append(
                {
                    "track_id": state["track_id"],
                    "frame": current_frame,
                    "pending_kind": pending_kind,
                    "predicted_z": predicted_position[0] / VOXEL_SIZE_ZYX[0],
                    "predicted_y": predicted_position[1] / VOXEL_SIZE_ZYX[1],
                    "predicted_x": predicted_position[2] / VOXEL_SIZE_ZYX[2],
                    "global_displacement_z_um": float(
                        prediction["global_displacement"][0]
                    ),
                    "global_displacement_y_um": float(
                        prediction["global_displacement"][1]
                    ),
                    "global_displacement_x_um": float(
                        prediction["global_displacement"][2]
                    ),
                    "relative_displacement_z_um": float(
                        prediction["relative_displacement"][0]
                    ),
                    "relative_displacement_y_um": float(
                        prediction["relative_displacement"][1]
                    ),
                    "relative_displacement_x_um": float(
                        prediction["relative_displacement"][2]
                    ),
                    "relative_motion_weight": float(
                        prediction["relative_weight"]
                    ),
                    "missed_frames": int(state["missed_frames"]),
                    "miss_prior_probability": float(
                        assignment["miss_probabilities"][state_index]
                    ),
                    "miss_choice_probability": float(
                        assignment["track_no_match_probabilities"][
                            state_index
                        ]
                    ),
                    "miss_cost": float(
                        assignment["miss_costs"][state_index]
                    ),
                    "boundary_faces": "|".join(
                        sorted(state["last_boundary_faces"])
                    ),
                }
            )
    
            association_event_records.append(
                {
                    "from_frame": int(current_frame - 1),
                    "to_frame": int(current_frame),
                    "decision_type": "miss",
                    "track_id": int(state["track_id"]),
                    "detection_position_index": np.nan,
                    "detection_cell_index": np.nan,
                    "pair_cost": np.nan,
                    "decision_cost": float(
                        assignment["miss_costs"][state_index]
                    ),
                    "association_probability": np.nan,
                    "track_candidate_probability": np.nan,
                    "detection_candidate_probability": np.nan,
                    "alternative_probability": float(
                        assignment["track_no_match_probabilities"][
                            state_index
                        ]
                    ),
                    "probability_margin": float(
                        assignment["track_probability_margins"][
                            state_index
                        ]
                    ),
                    "distance_um": np.nan,
                    "volume_ratio": np.nan,
                    "position_cost": np.nan,
                    "volume_cost": np.nan,
                    "shape_cost": np.nan,
                    "boundary_related": boundary_context,
                }
            )
    
            if boundary_context:
                boundary_events.append(
                    {
                        "track_id": state["track_id"],
                        "frame": current_frame,
                        "event_type": "boundary_missing",
                        "boundary_faces": "|".join(
                            sorted(state["last_boundary_faces"])
                        ),
                        "missing_frames": state["missed_frames"],
                        "reacquired_frame": np.nan,
                        "confidence": float(
                            assignment["track_no_match_probabilities"][
                                state_index
                            ]
                        ),
                    }
                )
    
            if state["missed_frames"] > maximum_missing_frames:
                state["active"] = False
    
                if boundary_context:
                    boundary_events.append(
                        {
                            "track_id": state["track_id"],
                            "frame": current_frame,
                            "event_type": "boundary_exit_confirmed",
                            "boundary_faces": "|".join(
                                sorted(state["last_boundary_faces"])
                            ),
                            "missing_frames": state["missed_frames"],
                            "reacquired_frame": np.nan,
                            "confidence": np.nan,
                        }
                    )
    
        # --------------------------------------------------------
        # Create tracks from explicit birth decisions
        # --------------------------------------------------------
    
        new_track_count = 0
    
        for detection_index in birth_detection_indices:
            detection = detections.iloc[int(detection_index)]
    
            track_id = next_track_id
            next_track_id += 1
            new_track_count += 1
    
            state = make_track_state(
                track_id=track_id,
                frame=current_frame,
                detection=detection,
            )
            track_states[track_id] = state
    
            is_boundary = bool(detection["touches_boundary"])
            match_type = (
                "boundary_entry" if is_boundary
                else "new_interior"
            )
    
            append_track_record(
                track_id=track_id,
                frame=current_frame,
                cell_index=int(detections.index[detection_index]),
                detection=detection,
                state=state,
                match_type=match_type,
                match_distance_um=None,
                match_cost=float(
                    assignment["birth_costs"][detection_index]
                ),
                association_probability=None,
                track_candidate_probability=None,
                detection_candidate_probability=None,
                probability_margin=None,
                position_cost=None,
                volume_cost=None,
                shape_cost=None,
                global_shift_physical=final_global_shift,
                global_shift_confidence=final_global_confidence,
                relative_velocity_updated=False,
            )
    
            association_event_records.append(
                {
                    "from_frame": int(current_frame - 1),
                    "to_frame": int(current_frame),
                    "decision_type": "birth",
                    "track_id": int(track_id),
                    "detection_position_index": int(detection_index),
                    "detection_cell_index": int(
                        detections.index[detection_index]
                    ),
                    "pair_cost": np.nan,
                    "decision_cost": float(
                        assignment["birth_costs"][detection_index]
                    ),
                    "association_probability": np.nan,
                    "track_candidate_probability": np.nan,
                    "detection_candidate_probability": np.nan,
                    "alternative_probability": float(
                        assignment[
                            "detection_birth_choice_probabilities"
                        ][detection_index]
                    ),
                    "probability_margin": np.nan,
                    "distance_um": np.nan,
                    "volume_ratio": np.nan,
                    "position_cost": np.nan,
                    "volume_cost": np.nan,
                    "shape_cost": np.nan,
                    "boundary_related": is_boundary,
                }
            )
    
            if is_boundary:
                boundary_events.append(
                    {
                        "track_id": track_id,
                        "frame": current_frame,
                        "event_type": "boundary_entry",
                        "boundary_faces": detection["boundary_faces"],
                        "missing_frames": 0,
                        "reacquired_frame": np.nan,
                        "confidence": float(
                            assignment[
                                "detection_birth_choice_probabilities"
                            ][detection_index]
                        ),
                    }
                )
    
        valid_candidate_tracks = int(
            (~assignment["safety_invalid"]).any(axis=1).sum()
        ) if len(eligible_states) and len(detections) else 0
    
        matched_probabilities = (
            assignment["association_probabilities"][rows, cols]
            if len(rows)
            else np.asarray([], dtype=float)
        )
    
        tracking_diagnostic_records.append(
            {
                "from_frame": int(current_frame - 1),
                "to_frame": int(current_frame),
                "eligible_tracks": int(len(eligible_states)),
                "detections": int(len(detections)),
                "matches": int(len(rows)),
                "selected_misses": int(len(missed_state_indices)),
                "selected_births": int(len(birth_detection_indices)),
                "new_tracks": int(new_track_count),
                "tracks_with_safety_valid_candidate": valid_candidate_tracks,
                "median_nearest_prediction_distance_um": prediction_median_um,
                "median_nearest_global_only_distance_um": global_only_median_um,
                "median_relative_motion_weight": median_relative_weight,
                "median_selected_association_probability": (
                    float(np.median(matched_probabilities))
                    if matched_probabilities.size else np.nan
                ),
                "minimum_selected_association_probability": (
                    float(np.min(matched_probabilities))
                    if matched_probabilities.size else np.nan
                ),
                "normalized_assignment_objective": float(
                    final_assignment_stats["normalized_objective"]
                ),
                "relative_velocity_updates": int(
                    relative_updates_this_frame
                ),
                "global_shift_confidence": float(
                    final_global_confidence
                ),
                "global_direction_change_deg": float(
                    direction_change_deg
                ),
                "global_shift_change_um": float(shift_change_um),
                "refinement_used": bool(refinement_used),
            }
        )
    
        print(
            f"\nDiagnostics {current_frame - 1:03d}"
            f"->{current_frame:03d}"
        )
        print(
            "Final global shift ZYX (Âµm):",
            np.round(final_global_shift, 3),
            "| magnitude:",
            round(float(np.linalg.norm(final_global_shift)), 3),
            "| confidence:",
            round(final_global_confidence, 3),
        )
    
        if current_frame > 1:
            print(
                "Global direction change:",
                "nan" if np.isnan(direction_change_deg)
                else round(direction_change_deg, 1),
                "degrees | shift-vector change:",
                round(shift_change_um, 3),
                "Âµm",
            )
    
        print(
            "Global refinement:",
            "used" if refinement_used else "not used",
            "| candidates:",
            refinement["candidate_count"],
            "| initial/final matches:",
            initial_assignment_stats["matches"],
            "/",
            final_assignment_stats["matches"],
        )
        print(
            "Median nearest distance using final/global-only predictor:",
            "nan" if np.isnan(prediction_median_um)
            else round(prediction_median_um, 3),
            "/",
            "nan" if np.isnan(global_only_median_um)
            else round(global_only_median_um, 3),
            "Âµm",
        )
        print(
            "Decisions:",
            len(rows),
            "matches |",
            len(missed_state_indices),
            "misses |",
            len(birth_detection_indices),
            "births",
        )
        print(
            "Median selected association probability:",
            "nan" if not matched_probabilities.size
            else round(float(np.median(matched_probabilities)), 3),
            "| normalized objective:",
            round(final_assignment_stats["normalized_objective"], 3),
        )
        print(
            "Tracks with at least one broad-safety-valid candidate:",
            valid_candidate_tracks,
            "/",
            len(eligible_states),
            "| reliable velocity updates:",
            relative_updates_this_frame,
        )
    
    
    tracks = pd.DataFrame(track_records)
    boundary_events = pd.DataFrame(boundary_events)
    missing_predictions = pd.DataFrame(missing_predictions)
    boundary_predictions = (
        missing_predictions[
            missing_predictions["pending_kind"] == "boundary"
        ].copy()
        if not missing_predictions.empty
        else pd.DataFrame()
    )
    global_motion = pd.DataFrame(global_motion_records)
    tracking_diagnostics = pd.DataFrame(
        tracking_diagnostic_records
    )
    association_events = pd.DataFrame(
        association_event_records
    )
    association_candidates = pd.DataFrame(
        association_candidate_records
    )
    
    # ============================================================
    # Tracking summary
    # ============================================================
    
    summary = {
        "track_records": len(tracks),
        "unique_tracks": tracks["track_id"].nunique(),
        "boundary_track_records": int(
            tracks["touches_boundary"].sum()
        ),
        "boundary_reacquisitions": int(
            (
                boundary_events["event_type"]
                == "boundary_reacquired"
            ).sum()
        ) if not boundary_events.empty else 0,
        "interior_reacquisitions": int(
            (tracks["match_type"] == "interior_reacquired").sum()
        ),
        "confirmed_boundary_exits": int(
            (
                boundary_events["event_type"]
                == "boundary_exit_confirmed"
            ).sum()
        ) if not boundary_events.empty else 0,
        "selected_misses": int(
            (association_events["decision_type"] == "miss").sum()
        ) if not association_events.empty else 0,
        "selected_births": int(
            (association_events["decision_type"] == "birth").sum()
        ) if not association_events.empty else 0,
        "median_association_probability": float(
            tracks["association_probability"].median()
        ),
        "relative_velocity_updates": int(
            tracks["relative_velocity_updated"].sum()
        ),
        "tracks_with_relative_velocity": int(
            sum(
                state["relative_velocity_valid"]
                for state in track_states.values()
            )
        ),
        "mean_global_shift_confidence": float(
            global_motion["final_confidence"].mean()
        ) if not global_motion.empty else np.nan,
        "largest_global_direction_change_deg": float(
            global_motion["direction_change_deg"].max()
        ) if not global_motion.empty else np.nan,
        "global_refinements_used": int(
            global_motion["refinement_used"].sum()
        ) if not global_motion.empty else 0,
    }
    
    pd.Series(summary)
    
    final_track_states = pd.DataFrame(
        [
            {
                "track_id": state["track_id"],
                "active": state["active"],
                "last_frame": state["last_frame"],
                "missed_frames": state["missed_frames"],
                "boundary_pending": state["boundary_pending"],
                "interior_pending": state["interior_pending"],
                "last_boundary_faces": "|".join(
                    sorted(state["last_boundary_faces"])
                ),
                "template_reliable": state["template_reliable"],
                "template_count": state["template_count"],
                "relative_velocity_valid": state[
                    "relative_velocity_valid"
                ],
                "relative_velocity_samples": state[
                    "relative_velocity_samples"
                ],
                "relative_velocity_error_ema_um": state[
                    "relative_velocity_error_ema"
                ],
                "relative_velocity_updates_rejected": state[
                    "relative_velocity_updates_rejected"
                ],
                "last_relative_update_frame": state[
                    "last_relative_update_frame"
                ],
                "relative_velocity_z_um_per_frame": float(
                    state["relative_velocity_physical"][0]
                ),
                "relative_velocity_y_um_per_frame": float(
                    state["relative_velocity_physical"][1]
                ),
                "relative_velocity_x_um_per_frame": float(
                    state["relative_velocity_physical"][2]
                ),
                "relative_motion_confidence_next_frame": (
                    relative_motion_confidence(state, frame_gap=1)
                ),
                "position_residual_ema_z_um": float(
                    state["position_residual_ema_zyx"][0]
                ),
                "position_residual_ema_y_um": float(
                    state["position_residual_ema_zyx"][1]
                ),
                "position_residual_ema_x_um": float(
                    state["position_residual_ema_zyx"][2]
                ),
                "position_residual_samples": int(
                    state["position_residual_samples"]
                ),
            }
            for state in track_states.values()
        ]
    ).sort_values("track_id")
    
    
    metadata = {
        "architecture": (
            "boundary_aware_probabilistic_global_relative_motion_tracking"
        ),
        "association_probabilities_are_calibrated": False,
        "sample_id": sample_id,
        "volume_shape_zyx": VOLUME_SHAPE_ZYX.tolist(),
        "voxel_size_zyx": VOXEL_SIZE_ZYX.tolist(),
        "global_shift": {
            "max_pair_distance_um": GLOBAL_SHIFT_MAX_PAIR_DISTANCE_UM,
            "mad_scale": GLOBAL_SHIFT_MAD_SCALE,
            "min_inlier_radius_um": GLOBAL_SHIFT_MIN_INLIER_RADIUS_UM,
            "confidence_pair_count": GLOBAL_SHIFT_CONFIDENCE_PAIR_COUNT,
            "confidence_dispersion_um": GLOBAL_SHIFT_CONFIDENCE_DISPERSION_UM,
            "refinement_min_matches": GLOBAL_REFINEMENT_MIN_MATCHES,
            "refinement_min_association_probability": (
                GLOBAL_REFINEMENT_MIN_ASSOCIATION_PROBABILITY
            ),
            "refinement_min_probability_margin": (
                GLOBAL_REFINEMENT_MIN_PROBABILITY_MARGIN
            ),
            "refinement_max_prediction_error_um": (
                GLOBAL_REFINEMENT_MAX_PREDICTION_ERROR_UM
            ),
        },
        "position_likelihood": {
            "base_sigma_zyx_um": BASE_POSITION_SIGMA_ZYX_UM.tolist(),
            "global_uncertainty_um": (
                POSITION_SIGMA_GLOBAL_UNCERTAINTY_UM
            ),
            "per_missing_frame_um": (
                POSITION_SIGMA_PER_MISSING_FRAME_UM
            ),
            "boundary_scale": POSITION_SIGMA_BOUNDARY_SCALE,
            "track_residual_weight": (
                POSITION_SIGMA_TRACK_RESIDUAL_WEIGHT
            ),
            "student_t_degrees_of_freedom": POSITION_STUDENT_T_DOF,
        },
        "volume_likelihood": {
            "interior_log_scale": VOLUME_LOG_SCALE_INTERIOR,
            "boundary_log_scale": VOLUME_LOG_SCALE_BOUNDARY,
            "student_t_degrees_of_freedom": VOLUME_STUDENT_T_DOF,
        },
        "miss_priors": {
            "interior": MISS_PROBABILITY_INTERIOR,
            "boundary": MISS_PROBABILITY_BOUNDARY,
            "pending_interior": MISS_PROBABILITY_PENDING_INTERIOR,
            "pending_boundary": MISS_PROBABILITY_PENDING_BOUNDARY,
            "global_uncertainty_bonus": MISS_GLOBAL_UNCERTAINTY_BONUS,
        },
        "birth_priors": {
            "interior": BIRTH_PROBABILITY_INTERIOR,
            "boundary": BIRTH_PROBABILITY_BOUNDARY,
        },
        "broad_safety_gates": {
            "max_distance_interior_um": (
                ABSOLUTE_MAX_DISTANCE_INTERIOR_UM
            ),
            "max_distance_boundary_um": (
                ABSOLUTE_MAX_DISTANCE_BOUNDARY_UM
            ),
            "distance_per_missing_frame_um": (
                ABSOLUTE_DISTANCE_PER_MISSING_FRAME_UM
            ),
            "max_volume_ratio_interior": (
                ABSOLUTE_MAX_VOLUME_RATIO_INTERIOR
            ),
            "max_volume_ratio_boundary": (
                ABSOLUTE_MAX_VOLUME_RATIO_BOUNDARY
            ),
        },
        "memory": {
            "interior_max_missing_frames": INTERIOR_MAX_MISSING_FRAMES,
            "boundary_max_missing_frames": BOUNDARY_MAX_MISSING_FRAMES,
        },
        "relative_motion": {
            "velocity_ema_alpha": RELATIVE_VELOCITY_EMA_ALPHA,
            "error_ema_alpha": RELATIVE_ERROR_EMA_ALPHA,
            "full_confidence_samples": RELATIVE_FULL_CONFIDENCE_SAMPLES,
            "error_confidence_scale_um": RELATIVE_ERROR_CONFIDENCE_SCALE_UM,
            "gap_decay": RELATIVE_MOTION_GAP_DECAY,
            "boundary_confidence_scale": (
                BOUNDARY_RELATIVE_MOTION_CONFIDENCE_SCALE
            ),
            "update_min_global_confidence": (
                RELATIVE_UPDATE_MIN_GLOBAL_CONFIDENCE
            ),
            "update_min_association_probability": (
                RELATIVE_UPDATE_MIN_ASSOCIATION_PROBABILITY
            ),
            "update_min_probability_margin": (
                RELATIVE_UPDATE_MIN_PROBABILITY_MARGIN
            ),
            "update_max_distance_um": RELATIVE_UPDATE_MAX_DISTANCE_UM,
            "max_velocity_um_per_frame": (
                MAX_RELATIVE_VELOCITY_UM_PER_FRAME
            ),
        },
        "boundary_margin_um": BOUNDARY_MARGIN_UM,
        "candidate_diagnostics": {
            "saved": SAVE_TOP_ASSOCIATION_CANDIDATES,
            "top_per_track": TOP_ASSOCIATION_CANDIDATES_PER_TRACK,
        },
        "track_records": int(len(tracks)),
        "unique_tracks": int(tracks["track_id"].nunique()),
        "association_events": int(len(association_events)),
        "association_candidates": int(len(association_candidates)),
        "selected_misses": int(
            (association_events["decision_type"] == "miss").sum()
        ) if not association_events.empty else 0,
        "selected_births": int(
            (association_events["decision_type"] == "birth").sum()
        ) if not association_events.empty else 0,
        "boundary_events": int(len(boundary_events)),
        "boundary_reacquisitions": int(
            (
                boundary_events["event_type"]
                == "boundary_reacquired"
            ).sum()
        ) if not boundary_events.empty else 0,
        "interior_reacquisitions": int(
            (tracks["match_type"] == "interior_reacquired").sum()
        ),
        "relative_velocity_updates": int(
            tracks["relative_velocity_updated"].sum()
        ),
        "global_refinements_used": int(
            global_motion["refinement_used"].sum()
        ) if not global_motion.empty else 0,
    }
    
    
    result = TrackingResult(
        tracks=tracks,
        boundary_events=boundary_events,
        boundary_predictions=boundary_predictions,
        missing_predictions=missing_predictions,
        boundary_counts=boundary_counts,
        global_motion=global_motion,
        tracking_diagnostics=tracking_diagnostics,
        association_events=association_events,
        association_candidates=association_candidates,
        track_states=final_track_states,
        metadata=metadata,
        summary=summary,
    )
    if not return_diagnostics:
        return result

    decisions = [
        DecisionRecord(
            decision_type=str(row.decision_type),
            outcome=str(row.decision_type),
            frame=int(row.to_frame),
            subject_id=int(row.track_id),
            metrics={
                "association_probability": row.association_probability,
                "decision_cost": row.decision_cost,
            },
            provenance=Provenance(
                source_type="tracking_decision",
                source_stage="07_cell_tracking",
                source_frame=int(row.to_frame),
                source_track_ids=(int(row.track_id),),
            ),
        )
        for row in association_events.itertuples()
    ] if not association_events.empty else []
    trace = StageTrace(
        stage_name="07_cell_tracking",
        inputs={"time_frames": time_frames},
        outputs={"tracks": tracks},
        intermediates={
            "global_motion": global_motion,
            "association_events": association_events,
            "association_candidates": association_candidates,
            "missing_predictions": missing_predictions,
            "boundary_events": boundary_events,
        },
        metrics=summary,
        decisions=decisions,
    )
    return result, trace
