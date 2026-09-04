"""Behavior-preserving Stage 7 tracking entry point."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.diagnostics import DecisionRecord, Provenance, StageTrace

from .graph_tracking import (
    GraphRefinementResult,
    GraphTrackingConfig,
    refine_transition_with_graph,
)
from .graph_tracking.diagnostics import (
    empty_anchor_votes,
    empty_boundary_hypotheses,
    empty_candidate_evidence,
    empty_refinement_events,
    empty_transition_summary,
)
from .graph_tracking.four_d import (
    FourDGraphResult,
    TransitionEvidence,
    run_four_d_graph_tracking,
)
from .graph_tracking.four_d.diagnostics import (
    empty_assignment_changes as empty_graph4d_assignment_changes,
    empty_boundary_events as empty_graph4d_boundary_events,
    empty_component_summary as empty_graph4d_component_summary,
    empty_solver_diagnostics as empty_graph4d_solver_diagnostics,
    empty_temporal_edges as empty_graph4d_temporal_edges,
    empty_track_id_map as empty_graph4d_track_id_map,
    empty_window_summary as empty_graph4d_window_summary,
)
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
    graph_transition_summary: pd.DataFrame
    graph_candidate_evidence: pd.DataFrame
    graph_anchor_votes: pd.DataFrame
    graph_boundary_hypotheses: pd.DataFrame
    graph_refinement_events: pd.DataFrame
    graph4d_window_summary: pd.DataFrame
    graph4d_component_summary: pd.DataFrame
    graph4d_temporal_edges: pd.DataFrame
    graph4d_assignment_changes: pd.DataFrame
    graph4d_boundary_events: pd.DataFrame
    graph4d_solver_diagnostics: pd.DataFrame
    graph4d_track_id_map: pd.DataFrame
    metadata: dict
    summary: dict
    transition_evidence: tuple[TransitionEvidence, ...] = field(
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    graph4d_debug_artifacts: dict[str, dict[str, np.ndarray]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )


def _run_provisional_or_pairwise_tracking(
    time_frames: list[pd.DataFrame],
    *,
    sample_id: str = "44b6_0113de3b",
    graph_config: GraphTrackingConfig | None = None,
    return_diagnostics: bool = False,
) -> TrackingResult | tuple[TrackingResult, StageTrace]:
    """Run the notebook's tracking algorithm without changing its ordering."""

    graph_config = graph_config or GraphTrackingConfig(mode="disabled")

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
    graph_transition_summary_tables: list[pd.DataFrame] = []
    graph_candidate_evidence_tables: list[pd.DataFrame] = []
    graph_anchor_vote_tables: list[pd.DataFrame] = []
    graph_boundary_hypothesis_tables: list[pd.DataFrame] = []
    graph_refinement_event_tables: list[pd.DataFrame] = []
    transition_evidence_records: list[TransitionEvidence] = []
    
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


    def build_graph_diagnostic_context(
        graph_result: GraphRefinementResult | None,
        eligible_states: list[dict],
        assignment: dict,
    ) -> dict[str, Any]:
        """Index one transition's graph evidence for stable Stage 7 records."""

        base_assignment = (
            graph_result.base_assignment
            if graph_result is not None
            else assignment
        )
        graph_assignment = (
            graph_result.graph_assignment
            if graph_result is not None
            else assignment
        )

        def decision_sets(value: dict) -> tuple[dict[int, int], set[int], set[int]]:
            matches = {
                int(row): int(col)
                for row, col in zip(value["rows"], value["cols"])
            }
            misses = {
                int(index)
                for index in value["missed_state_indices"]
            }
            births = {
                int(index)
                for index in value["birth_detection_indices"]
            }
            return matches, misses, births

        candidate_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
        boundary_by_state: dict[int, dict[str, Any]] = {}
        boundary_by_detection: dict[int, dict[str, Any]] = {}
        changed_states: set[int] = set()
        changed_detections: set[int] = set()

        if graph_result is not None:
            for row in graph_result.candidate_evidence.to_dict("records"):
                candidate_by_pair[
                    (
                        int(row["source_state_index"]),
                        int(row["candidate_detection_index"]),
                    )
                ] = row

            state_index_by_track_id = {
                int(state["track_id"]): state_index
                for state_index, state in enumerate(eligible_states)
            }
            for row in graph_result.boundary_hypotheses.to_dict("records"):
                track_id = row.get("track_id")
                detection_index = row.get("detection_index")
                if pd.notna(track_id):
                    state_index = state_index_by_track_id.get(int(track_id))
                    if state_index is not None:
                        boundary_by_state[state_index] = row
                if pd.notna(detection_index):
                    boundary_by_detection[int(detection_index)] = row

            for row in graph_result.refinement_events.to_dict("records"):
                if not bool(row["changed"]):
                    continue
                if row["entity_type"] == "track":
                    changed_states.add(int(row["entity_index"]))
                elif row["entity_type"] == "detection":
                    changed_detections.add(int(row["entity_index"]))

        base_matches, base_misses, base_births = decision_sets(base_assignment)
        graph_matches, graph_misses, graph_births = decision_sets(graph_assignment)
        return {
            "enabled": graph_result is not None,
            "candidate_by_pair": candidate_by_pair,
            "boundary_by_state": boundary_by_state,
            "boundary_by_detection": boundary_by_detection,
            "changed_states": changed_states,
            "changed_detections": changed_detections,
            "base_matches": base_matches,
            "base_misses": base_misses,
            "base_births": base_births,
            "graph_matches": graph_matches,
            "graph_misses": graph_misses,
            "graph_births": graph_births,
        }


    def graph_decision_diagnostics(
        *,
        context: dict[str, Any],
        decision_type: str,
        state_index: int | None = None,
        detection_index: int | None = None,
    ) -> dict[str, Any]:
        """Return graph fields shared by candidate and decision diagnostics."""

        pair = None
        if state_index is not None and detection_index is not None:
            pair = context["candidate_by_pair"].get(
                (int(state_index), int(detection_index))
            )

        boundary = None
        if decision_type == "miss" and state_index is not None:
            boundary = context["boundary_by_state"].get(int(state_index))
        elif decision_type == "birth" and detection_index is not None:
            boundary = context["boundary_by_detection"].get(
                int(detection_index)
            )
        elif decision_type in {"match", "candidate"}:
            if detection_index is not None:
                boundary = context["boundary_by_detection"].get(
                    int(detection_index)
                )
            if boundary is None and state_index is not None:
                boundary = context["boundary_by_state"].get(int(state_index))

        if decision_type in {"match", "candidate"}:
            base_selected = (
                state_index is not None
                and detection_index is not None
                and context["base_matches"].get(int(state_index))
                == int(detection_index)
            )
            graph_selected = (
                state_index is not None
                and detection_index is not None
                and context["graph_matches"].get(int(state_index))
                == int(detection_index)
            )
        elif decision_type == "miss":
            base_selected = (
                state_index is not None
                and int(state_index) in context["base_misses"]
            )
            graph_selected = (
                state_index is not None
                and int(state_index) in context["graph_misses"]
            )
        else:
            base_selected = (
                detection_index is not None
                and int(detection_index) in context["base_births"]
            )
            graph_selected = (
                detection_index is not None
                and int(detection_index) in context["graph_births"]
            )

        changed = (
            state_index is not None
            and int(state_index) in context["changed_states"]
        ) or (
            detection_index is not None
            and int(detection_index) in context["changed_detections"]
        )
        boundary_confidence = (
            float(boundary["confidence"])
            if boundary is not None
            else np.nan
        )
        return {
            "graph_available": bool(pair is not None or boundary is not None),
            "graph_anchor_count": (
                int(pair["anchor_count"])
                if pair is not None
                else int(boundary["anchor_count"])
                if boundary is not None
                else 0
            ),
            "graph_inlier_count": (
                int(pair["inlier_count"])
                if pair is not None
                else int(boundary["inlier_count"])
                if boundary is not None
                else 0
            ),
            "graph_vote_score": (
                float(pair["vote_score"])
                if pair is not None
                else np.nan
            ),
            "graph_confidence": (
                float(pair["graph_confidence"])
                if pair is not None
                else boundary_confidence
            ),
            "graph_cost_delta": (
                float(pair["graph_cost_delta"])
                if pair is not None
                else np.nan
            ),
            "base_selected": bool(base_selected),
            "graph_selected": bool(graph_selected),
            "assignment_changed_by_graph": bool(changed),
            "graph_boundary_event_type": (
                str(boundary["decision"])
                if boundary is not None
                else ""
            ),
            "graph_boundary_confidence": boundary_confidence,
        }
    
    
    def record_top_candidates(
        *,
        assignment: dict,
        eligible_states: list[dict],
        detections: pd.DataFrame,
        current_frame: int,
        graph_context: dict[str, Any],
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
                        **graph_decision_diagnostics(
                            context=graph_context,
                            decision_type="candidate",
                            state_index=state_index,
                            detection_index=int(detection_index),
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

        # Graph refinement is deliberately inserted after the final base
        # global-shift assignment is selected and before any diagnostics or
        # track-state mutation consume the assignment.
        graph_result: GraphRefinementResult | None = None
        if (
            graph_config.mode != "disabled"
            and graph_config.algorithm == "pairwise"
        ):
            graph_result = refine_transition_with_graph(
                eligible_states=eligible_states,
                detections=detections,
                current_frame=current_frame,
                base_assignment=assignment,
                volume_shape_zyx=VOLUME_SHAPE_ZYX,
                voxel_size_zyx_um=VOXEL_SIZE_ZYX,
                config=graph_config,
                assignment_solver=stage7_assignment_solver,
            )
            graph_transition_summary_tables.append(
                graph_result.transition_summary
            )
            graph_candidate_evidence_tables.append(
                graph_result.candidate_evidence
            )
            graph_anchor_vote_tables.append(graph_result.anchor_votes)
            graph_boundary_hypothesis_tables.append(
                graph_result.boundary_hypotheses
            )
            graph_refinement_event_tables.append(
                graph_result.refinement_events
            )
            assignment = graph_result.assignment

        graph_context = build_graph_diagnostic_context(
            graph_result,
            eligible_states,
            assignment,
        )

        # Keep a compact immutable snapshot for the post-sequence 4D solver.
        # Dense evidence is deliberately retained only in memory and is never
        # written to the default Stage 7 CSV artifact set.
        detection_probability_margins = np.zeros(len(detections), dtype=float)
        for detection_index in range(len(detections)):
            choices = assignment["detection_candidate_probabilities"][
                :, detection_index
            ]
            birth_choice = float(
                assignment["detection_birth_choice_probabilities"][
                    detection_index
                ]
            )
            selected_state = next(
                (
                    int(state_index)
                    for state_index, selected_detection in zip(
                        assignment["rows"], assignment["cols"]
                    )
                    if int(selected_detection) == detection_index
                ),
                None,
            )
            if selected_state is None:
                selected_probability = birth_choice
                alternatives = choices
            else:
                selected_probability = float(choices[selected_state])
                alternatives = np.concatenate(
                    [np.delete(choices, selected_state), [birth_choice]]
                )
            detection_probability_margins[detection_index] = (
                selected_probability
                - (float(np.max(alternatives)) if alternatives.size else 0.0)
            )

        transition_evidence_records.append(
            TransitionEvidence(
                from_frame=int(current_frame - 1),
                to_frame=int(current_frame),
                source_track_ids=np.asarray(
                    [state["track_id"] for state in eligible_states],
                    dtype=np.int64,
                ),
                source_frames=np.asarray(
                    [state["last_frame"] for state in eligible_states],
                    dtype=np.int32,
                ),
                target_detection_indices=np.arange(
                    len(detections), dtype=np.int32
                ),
                target_cell_ids=np.asarray(
                    [
                        detections.iloc[index].get("cell_id", index + 1)
                        for index in range(len(detections))
                    ],
                    dtype=np.int64,
                ),
                pair_cost_matrix=np.asarray(
                    assignment["pair_cost_matrix"], dtype=float
                ).copy(),
                safety_invalid=np.asarray(
                    assignment["safety_invalid"], dtype=bool
                ).copy(),
                distance_matrix=np.asarray(
                    assignment["distance_matrix"], dtype=float
                ).copy(),
                association_probabilities=np.asarray(
                    assignment["association_probabilities"], dtype=float
                ).copy(),
                track_candidate_probabilities=np.asarray(
                    assignment["track_candidate_probabilities"], dtype=float
                ).copy(),
                detection_candidate_probabilities=np.asarray(
                    assignment["detection_candidate_probabilities"], dtype=float
                ).copy(),
                track_probability_margins=np.asarray(
                    assignment["track_probability_margins"], dtype=float
                ).copy(),
                detection_probability_margins=(
                    detection_probability_margins.copy()
                ),
                miss_costs=np.asarray(
                    assignment["miss_costs"], dtype=float
                ).copy(),
                birth_costs=np.asarray(
                    assignment["birth_costs"], dtype=float
                ).copy(),
                global_only_positions_zyx_um=np.asarray(
                    assignment["global_only_positions"], dtype=float
                ).copy(),
                relative_motion_predicted_positions_zyx_um=np.asarray(
                    assignment["predicted_positions"], dtype=float
                ).copy(),
                selected_rows=np.asarray(
                    assignment["rows"], dtype=np.int32
                ).copy(),
                selected_columns=np.asarray(
                    assignment["cols"], dtype=np.int32
                ).copy(),
                missed_rows=np.asarray(
                    assignment["missed_state_indices"], dtype=np.int32
                ).copy(),
                birth_columns=np.asarray(
                    assignment["birth_detection_indices"], dtype=np.int32
                ).copy(),
                global_shift_zyx_um=np.asarray(
                    final_global_shift, dtype=float
                ).copy(),
                global_shift_confidence=float(final_global_confidence),
            )
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
            graph_context=graph_context,
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
            graph_exit_was_pending = (
                state["graph_boundary_hypothesis"] == "exit"
            )
            pending_graph_exit_face = state["graph_boundary_face"]
            pending_graph_exit_confidence = float(
                state["graph_boundary_confidence"]
            )
    
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
                    **graph_decision_diagnostics(
                        context=graph_context,
                        decision_type="match",
                        state_index=int(state_index),
                        detection_index=int(detection_index),
                    ),
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

            if graph_exit_was_pending and update_result["was_reacquired"]:
                boundary_events.append(
                    {
                        "track_id": state["track_id"],
                        "frame": current_frame,
                        "event_type": "graph_exit_rejected_reacquired",
                        "boundary_faces": pending_graph_exit_face,
                        "missing_frames": 0,
                        "reacquired_frame": current_frame,
                        "confidence": pending_graph_exit_confidence,
                    }
                )
            if graph_exit_was_pending:
                state["graph_boundary_hypothesis"] = ""
                state["graph_boundary_face"] = ""
                state["graph_boundary_confidence"] = 0.0
                state["graph_boundary_since_frame"] = None
                state["graph_anchor_support_count"] = 0
    
        # --------------------------------------------------------
        # Apply explicit miss decisions and preserve short memory
        # --------------------------------------------------------
    
        for state_index in missed_state_indices:
            state = eligible_states[int(state_index)]
            graph_exit_row = graph_context["boundary_by_state"].get(
                int(state_index)
            )
            graph_exit_supported = bool(
                graph_config.mode == "apply"
                and graph_exit_row is not None
                and graph_exit_row["event_type"] == "exit"
                and bool(graph_exit_row["supported"])
            )
            graph_exit_pending = bool(
                graph_exit_supported
                or state["graph_boundary_hypothesis"] == "exit"
            )
    
            last_was_boundary = bool(
                state["last_detection"].get(
                    "touches_boundary",
                    False,
                )
            )
            boundary_context = (
                state["boundary_pending"]
                or last_was_boundary
                or graph_exit_pending
            )

            if graph_exit_supported:
                state["graph_last_confidence"] = float(
                    graph_exit_row["confidence"]
                )
                state["graph_anchor_support_count"] = int(
                    graph_exit_row["anchor_count"]
                )
                state["graph_boundary_hypothesis"] = "exit"
                state["graph_boundary_face"] = str(
                    graph_exit_row["boundary_face"]
                )
                state["graph_boundary_confidence"] = float(
                    graph_exit_row["confidence"]
                )
                if state["graph_boundary_since_frame"] is None:
                    state["graph_boundary_since_frame"] = int(current_frame)
    
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
                    "graph_boundary_hypothesis": state[
                        "graph_boundary_hypothesis"
                    ],
                    "graph_boundary_face": state["graph_boundary_face"],
                    "graph_boundary_confidence": float(
                        state["graph_boundary_confidence"]
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
                    **graph_decision_diagnostics(
                        context=graph_context,
                        decision_type="miss",
                        state_index=int(state_index),
                    ),
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

            if graph_exit_supported:
                boundary_events.append(
                    {
                        "track_id": state["track_id"],
                        "frame": current_frame,
                        "event_type": "graph_exit_predicted",
                        "boundary_faces": state["graph_boundary_face"],
                        "missing_frames": state["missed_frames"],
                        "reacquired_frame": np.nan,
                        "confidence": state["graph_boundary_confidence"],
                    }
                )
            if state["graph_boundary_hypothesis"] == "exit":
                boundary_events.append(
                    {
                        "track_id": state["track_id"],
                        "frame": current_frame,
                        "event_type": "graph_exit_pending",
                        "boundary_faces": state["graph_boundary_face"],
                        "missing_frames": state["missed_frames"],
                        "reacquired_frame": np.nan,
                        "confidence": state["graph_boundary_confidence"],
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
                if state["graph_boundary_hypothesis"] == "exit":
                    boundary_events.append(
                        {
                            "track_id": state["track_id"],
                            "frame": current_frame,
                            "event_type": "graph_exit_confirmed",
                            "boundary_faces": state["graph_boundary_face"],
                            "missing_frames": state["missed_frames"],
                            "reacquired_frame": np.nan,
                            "confidence": state[
                                "graph_boundary_confidence"
                            ],
                        }
                    )
                    state["graph_boundary_hypothesis"] = "exit_confirmed"
    
        # --------------------------------------------------------
        # Create tracks from explicit birth decisions
        # --------------------------------------------------------
    
        new_track_count = 0
    
        for detection_index in birth_detection_indices:
            detection = detections.iloc[int(detection_index)]
            graph_entry_row = graph_context[
                "boundary_by_detection"
            ].get(int(detection_index))
            graph_entry_supported = bool(
                graph_config.mode == "apply"
                and graph_entry_row is not None
                and graph_entry_row["event_type"] == "entry"
                and bool(graph_entry_row["supported"])
            )
    
            track_id = next_track_id
            next_track_id += 1
            new_track_count += 1
    
            state = make_track_state(
                track_id=track_id,
                frame=current_frame,
                detection=detection,
            )
            if graph_entry_supported:
                state["graph_last_confidence"] = float(
                    graph_entry_row["confidence"]
                )
                state["graph_anchor_support_count"] = int(
                    graph_entry_row["anchor_count"]
                )
                state["graph_boundary_hypothesis"] = "entry_supported"
                state["graph_boundary_face"] = str(
                    graph_entry_row["boundary_face"]
                )
                state["graph_boundary_confidence"] = float(
                    graph_entry_row["confidence"]
                )
                state["graph_boundary_since_frame"] = int(current_frame)
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
                    **graph_decision_diagnostics(
                        context=graph_context,
                        decision_type="birth",
                        detection_index=int(detection_index),
                    ),
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
            if graph_entry_supported:
                boundary_events.append(
                    {
                        "track_id": track_id,
                        "frame": current_frame,
                        "event_type": "graph_entry_supported",
                        "boundary_faces": state["graph_boundary_face"],
                        "missing_frames": 0,
                        "reacquired_frame": np.nan,
                        "confidence": state["graph_boundary_confidence"],
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
                "graph_mode": graph_config.mode,
                "graph_candidate_evidence": (
                    int(len(graph_result.candidate_evidence))
                    if graph_result is not None else 0
                ),
                "graph_anchor_votes": (
                    int(len(graph_result.anchor_votes))
                    if graph_result is not None else 0
                ),
                "graph_boundary_hypotheses": (
                    int(len(graph_result.boundary_hypotheses))
                    if graph_result is not None else 0
                ),
                "graph_assignment_changes": (
                    int(graph_result.refinement_events["changed"].sum())
                    if graph_result is not None
                    and not graph_result.refinement_events.empty
                    else 0
                ),
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

    def concatenate_graph_tables(
        tables: list[pd.DataFrame],
        empty_factory: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        populated = [table for table in tables if not table.empty]
        if not populated:
            return empty_factory()
        return pd.concat(populated, ignore_index=True)

    graph_transition_summary = concatenate_graph_tables(
        graph_transition_summary_tables,
        empty_transition_summary,
    )
    graph_candidate_evidence = concatenate_graph_tables(
        graph_candidate_evidence_tables,
        empty_candidate_evidence,
    )
    graph_anchor_votes = concatenate_graph_tables(
        graph_anchor_vote_tables,
        empty_anchor_votes,
    )
    graph_boundary_hypotheses = concatenate_graph_tables(
        graph_boundary_hypothesis_tables,
        empty_boundary_hypotheses,
    )
    graph_refinement_events = concatenate_graph_tables(
        graph_refinement_event_tables,
        empty_refinement_events,
    )

    graph_refinement_change_count = int(
        graph_refinement_events["changed"].sum()
    ) if not graph_refinement_events.empty else 0
    graph_supported_entry_count = int(
        (
            (graph_boundary_hypotheses["event_type"] == "entry")
            & graph_boundary_hypotheses["supported"].astype(bool)
        ).sum()
    ) if not graph_boundary_hypotheses.empty else 0
    graph_supported_exit_count = int(
        (
            (graph_boundary_hypotheses["event_type"] == "exit")
            & graph_boundary_hypotheses["supported"].astype(bool)
        ).sum()
    ) if not graph_boundary_hypotheses.empty else 0
    
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
            tracks["association_probability"].dropna().median()
        ) if tracks["association_probability"].notna().any() else np.nan,
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
        "graph_mode": graph_config.mode,
        "graph_algorithm": graph_config.algorithm,
        "graph_transitions": int(len(graph_transition_summary)),
        "graph_candidate_evidence": int(len(graph_candidate_evidence)),
        "graph_assignment_changes": graph_refinement_change_count,
        "graph_supported_entries": graph_supported_entry_count,
        "graph_supported_exits": graph_supported_exit_count,
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
                "graph_last_confidence": float(
                    state["graph_last_confidence"]
                ),
                "graph_anchor_support_count": int(
                    state["graph_anchor_support_count"]
                ),
                "graph_boundary_hypothesis": state[
                    "graph_boundary_hypothesis"
                ],
                "graph_boundary_face": state["graph_boundary_face"],
                "graph_boundary_confidence": float(
                    state["graph_boundary_confidence"]
                ),
                "graph_boundary_since_frame": state[
                    "graph_boundary_since_frame"
                ],
            }
            for state in track_states.values()
        ]
    ).sort_values("track_id")
    
    
    metadata = {
        "architecture": (
            "boundary_aware_probabilistic_global_relative_motion_tracking"
            "_with_optional_graph_transition_refinement"
        ),
        "association_probabilities_are_calibrated": False,
        "sample_id": sample_id,
        "volume_shape_zyx": VOLUME_SHAPE_ZYX.tolist(),
        "voxel_size_zyx": VOXEL_SIZE_ZYX.tolist(),
        "graph_tracking": {
            "mode": graph_config.mode,
            "algorithm": graph_config.algorithm,
            "enabled": graph_config.mode != "disabled",
            "configuration": asdict(graph_config),
            "transition_count": int(len(graph_transition_summary)),
            "candidate_evidence_count": int(
                len(graph_candidate_evidence)
            ),
            "anchor_vote_count": int(len(graph_anchor_votes)),
            "boundary_hypothesis_count": int(
                len(graph_boundary_hypotheses)
            ),
            "refinement_change_count": graph_refinement_change_count,
            "graph_supported_entry_count": graph_supported_entry_count,
            "graph_supported_exit_count": graph_supported_exit_count,
        },
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
        graph_transition_summary=graph_transition_summary,
        graph_candidate_evidence=graph_candidate_evidence,
        graph_anchor_votes=graph_anchor_votes,
        graph_boundary_hypotheses=graph_boundary_hypotheses,
        graph_refinement_events=graph_refinement_events,
        graph4d_window_summary=empty_graph4d_window_summary(),
        graph4d_component_summary=empty_graph4d_component_summary(),
        graph4d_temporal_edges=empty_graph4d_temporal_edges(),
        graph4d_assignment_changes=empty_graph4d_assignment_changes(),
        graph4d_boundary_events=empty_graph4d_boundary_events(),
        graph4d_solver_diagnostics=empty_graph4d_solver_diagnostics(),
        graph4d_track_id_map=empty_graph4d_track_id_map(),
        metadata=metadata,
        summary=summary,
        transition_evidence=tuple(transition_evidence_records),
        graph4d_debug_artifacts={},
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
                "graph_available": bool(row.graph_available),
                "graph_confidence": row.graph_confidence,
                "graph_cost_delta": row.graph_cost_delta,
                "base_selected": bool(row.base_selected),
                "graph_selected": bool(row.graph_selected),
                "assignment_changed_by_graph": bool(
                    row.assignment_changed_by_graph
                ),
                "graph_boundary_event_type": (
                    row.graph_boundary_event_type
                ),
                "graph_boundary_confidence": (
                    row.graph_boundary_confidence
                ),
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
            "graph_transition_summary": graph_transition_summary,
            "graph_candidate_evidence": graph_candidate_evidence,
            "graph_anchor_votes": graph_anchor_votes,
            "graph_boundary_hypotheses": graph_boundary_hypotheses,
            "graph_refinement_events": graph_refinement_events,
        },
        metrics=summary,
        decisions=decisions,
    )
    return result, trace


def _graph4d_node_descriptors(
    time_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frame, detections in enumerate(time_frames):
        for detection_index in range(len(detections)):
            rows.append({
                "node_index": len(rows),
                "frame": frame,
                "detection_position_index": detection_index,
                "detection_cell_index": detections.index[detection_index],
            })
    return pd.DataFrame(rows)


def _blank_record(columns: list[str]) -> dict[str, object]:
    return {column: np.nan for column in columns}


def _rebuild_graph4d_association_events(
    provisional: TrackingResult,
    four_d: FourDGraphResult,
    time_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    essential = [
        "from_frame", "to_frame", "decision_type", "track_id",
        "detection_position_index", "detection_cell_index", "pair_cost",
        "decision_cost", "association_probability", "probability_margin",
        "distance_um", "boundary_related", "graph4d_edge_index",
        "graph4d_frame_gap", "graph4d_component_id", "graph4d_window_id",
        "graph4d_solver_type", "graph4d_changed_from_provisional",
    ]
    columns = list(dict.fromkeys([
        *provisional.association_events.columns.tolist(), *essential
    ]))
    nodes = _graph4d_node_descriptors(time_frames).set_index("node_index")
    selected_edges = four_d.temporal_edges.loc[
        four_d.temporal_edges["optimized_selected"].astype(bool)
    ]
    incoming = set(selected_edges["target_node"].astype(int))
    outgoing = set(selected_edges["source_node"].astype(int))
    rows: list[dict[str, object]] = []
    for edge in selected_edges.sort_values(
        ["target_frame", "source_node", "target_node"], kind="mergesort"
    ).itertuples(index=False):
        target = nodes.loc[int(edge.target_node)]
        record = _blank_record(columns)
        record.update({
            "from_frame": int(edge.source_frame),
            "to_frame": int(edge.target_frame),
            "decision_type": "match",
            "track_id": int(four_d.node_to_track_id[int(edge.target_node)]),
            "detection_position_index": int(target["detection_position_index"]),
            "detection_cell_index": target["detection_cell_index"],
            "pair_cost": float(edge.total_effective_cost),
            "decision_cost": float(edge.total_effective_cost),
            "association_probability": edge.provisional_probability,
            "probability_margin": edge.provisional_margin,
            "distance_um": float(edge.displacement_um),
            "boundary_related": False,
            "graph_available": True,
            "base_selected": bool(edge.provisional_selected),
            "graph_selected": True,
            "assignment_changed_by_graph": bool(edge.changed_from_provisional),
            "graph4d_edge_index": int(edge.edge_index),
            "graph4d_frame_gap": int(edge.frame_gap),
            "graph4d_component_id": int(edge.component_id),
            "graph4d_window_id": int(edge.window_id),
            "graph4d_solver_type": str(edge.solver_type),
            "graph4d_changed_from_provisional": bool(edge.changed_from_provisional),
        })
        rows.append(record)

    if not nodes.empty:
        first_frame = int(nodes["frame"].min())
        last_frame = int(nodes["frame"].max())
        for node, descriptor in nodes.iterrows():
            node = int(node)
            frame = int(descriptor["frame"])
            if frame > first_frame and node not in incoming:
                record = _blank_record(columns)
                record.update({
                    "from_frame": frame - 1,
                    "to_frame": frame,
                    "decision_type": "birth",
                    "track_id": int(four_d.node_to_track_id[node]),
                    "detection_position_index": int(descriptor["detection_position_index"]),
                    "detection_cell_index": descriptor["detection_cell_index"],
                    "decision_cost": float(provisional.metadata["birth_priors"]["interior"]),
                    "boundary_related": False,
                    "graph_available": True,
                    "base_selected": False,
                    "graph_selected": True,
                    "assignment_changed_by_graph": True,
                    "graph4d_frame_gap": 0,
                    "graph4d_solver_type": "windowed_4d_event",
                    "graph4d_changed_from_provisional": True,
                })
                rows.append(record)
            if frame < last_frame and node not in outgoing:
                record = _blank_record(columns)
                record.update({
                    "from_frame": frame,
                    "to_frame": frame + 1,
                    "decision_type": "miss",
                    "track_id": int(four_d.node_to_track_id[node]),
                    "detection_position_index": np.nan,
                    "detection_cell_index": np.nan,
                    "decision_cost": float(provisional.metadata["miss_priors"]["interior"]),
                    "boundary_related": False,
                    "graph_available": True,
                    "base_selected": False,
                    "graph_selected": True,
                    "assignment_changed_by_graph": True,
                    "graph4d_frame_gap": 0,
                    "graph4d_solver_type": "windowed_4d_event",
                    "graph4d_changed_from_provisional": True,
                })
                rows.append(record)
    decision_order = {"match": 0, "miss": 1, "birth": 2}
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    result["_decision_order"] = result["decision_type"].map(decision_order)
    return result.sort_values(
        ["to_frame", "_decision_order", "track_id", "detection_position_index"],
        kind="mergesort",
    ).drop(columns="_decision_order").reset_index(drop=True)


def _rebuild_graph4d_association_candidates(
    provisional: TrackingResult,
    four_d: FourDGraphResult,
    time_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    essential = [
        "from_frame", "to_frame", "track_id", "detection_position_index",
        "detection_cell_index", "candidate_rank_by_pair_cost", "pair_cost",
        "distance_um", "association_probability", "probability_margin",
        "frame_gap", "provisional_selected", "optimized_selected",
        "graph_expanded", "component_id", "window_id", "solver_type",
        "changed_from_provisional", "unary_cost", "motion_cost", "graph_cost",
        "persistent_relation_cost", "boundary_cost", "total_effective_cost",
    ]
    columns = list(dict.fromkeys([
        *provisional.association_candidates.columns.tolist(), *essential
    ]))
    nodes = _graph4d_node_descriptors(time_frames).set_index("node_index")
    rows: list[dict[str, object]] = []
    for edge in four_d.temporal_edges.itertuples(index=False):
        target = nodes.loc[int(edge.target_node)]
        record = _blank_record(columns)
        record.update({
            "from_frame": int(edge.source_frame),
            "to_frame": int(edge.target_frame),
            "track_id": int(four_d.node_to_track_id[int(edge.source_node)]),
            "detection_position_index": int(target["detection_position_index"]),
            "detection_cell_index": target["detection_cell_index"],
            "pair_cost": float(edge.base_stage7_cost) if pd.notna(edge.base_stage7_cost) else float(edge.unary_cost),
            "distance_um": float(edge.displacement_um),
            "association_probability": edge.provisional_probability,
            "probability_margin": edge.provisional_margin,
            "frame_gap": int(edge.frame_gap),
            "provisional_selected": bool(edge.provisional_selected),
            "optimized_selected": bool(edge.optimized_selected),
            "graph_expanded": bool(edge.graph_expanded),
            "component_id": int(edge.component_id),
            "window_id": int(edge.window_id),
            "solver_type": str(edge.solver_type),
            "changed_from_provisional": bool(edge.changed_from_provisional),
            "unary_cost": float(edge.unary_cost),
            "motion_cost": float(edge.motion_cost),
            "graph_cost": float(edge.graph_cost),
            "persistent_relation_cost": float(edge.persistent_relation_cost),
            "boundary_cost": float(edge.boundary_cost),
            "total_effective_cost": float(edge.total_effective_cost),
            "base_selected": bool(edge.provisional_selected),
            "graph_selected": bool(edge.optimized_selected),
            "graph_available": True,
            "assignment_changed_by_graph": bool(edge.changed_from_provisional),
        })
        rows.append(record)
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    result = result.sort_values(
        ["from_frame", "track_id", "to_frame", "pair_cost", "detection_position_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    result["candidate_rank_by_pair_cost"] = (
        result.groupby(["from_frame", "track_id"], sort=False).cumcount() + 1
    )
    return result


def _rebuild_graph4d_track_states(
    provisional: TrackingResult,
    optimized_tracks: pd.DataFrame,
) -> pd.DataFrame:
    columns = list(provisional.track_states.columns)
    for required in ("track_id", "active", "last_frame", "missed_frames"):
        if required not in columns:
            columns.append(required)
    rows: list[dict[str, object]] = []
    sequence_last = int(optimized_tracks["frame"].max()) if not optimized_tracks.empty else -1
    for track_id, group in optimized_tracks.groupby("track_id", sort=True):
        last = group.sort_values("frame", kind="mergesort").iloc[-1]
        record = _blank_record(columns)
        record.update({
            "track_id": int(track_id),
            "active": int(last["frame"]) == sequence_last,
            "last_frame": int(last["frame"]),
            "missed_frames": 0,
            "boundary_pending": bool(last.get("touches_boundary", False)),
            "interior_pending": False,
            "last_boundary_faces": str(last.get("boundary_faces", "")),
            "template_reliable": not bool(last.get("touches_boundary", False)),
            "template_count": int(len(group)),
            "relative_velocity_valid": False,
            "relative_velocity_samples": 0,
            "graph_last_confidence": 0.0,
            "graph_anchor_support_count": 0,
            "graph_boundary_hypothesis": "",
            "graph_boundary_face": "",
            "graph_boundary_confidence": 0.0,
        })
        rows.append(record)
    return pd.DataFrame(rows, columns=columns).sort_values(
        "track_id", kind="mergesort"
    ).reset_index(drop=True)


def _rebuild_graph4d_tracking_diagnostics(
    provisional: TrackingResult,
    association_events: pd.DataFrame,
) -> pd.DataFrame:
    columns = list(provisional.tracking_diagnostics.columns)
    for required in (
        "from_frame", "to_frame", "matches", "selected_misses",
        "selected_births", "graph_mode", "graph_algorithm",
        "graph4d_selected_gap_edges",
    ):
        if required not in columns:
            columns.append(required)
    rows: list[dict[str, object]] = []
    if association_events.empty:
        return pd.DataFrame(columns=columns)
    for to_frame, group in association_events.groupby("to_frame", sort=True):
        record = _blank_record(columns)
        record.update({
            "from_frame": int(to_frame) - 1,
            "to_frame": int(to_frame),
            "matches": int((group["decision_type"] == "match").sum()),
            "selected_misses": int((group["decision_type"] == "miss").sum()),
            "selected_births": int((group["decision_type"] == "birth").sum()),
            "new_tracks": int((group["decision_type"] == "birth").sum()),
            "graph_mode": "apply",
            "graph_algorithm": "windowed_4d",
            "graph4d_selected_gap_edges": int(
                ((group["decision_type"] == "match") & (group["graph4d_frame_gap"] > 1)).sum()
            ),
        })
        rows.append(record)
    return pd.DataFrame(rows, columns=columns)


def _attach_four_d_result(
    *,
    provisional: TrackingResult,
    four_d: FourDGraphResult,
    time_frames: list[pd.DataFrame],
    graph_config: GraphTrackingConfig,
    provisional_runtime_seconds: float,
) -> TrackingResult:
    apply_mode = graph_config.mode == "apply"
    metadata = dict(provisional.metadata)
    four_d_metadata = {
        key: value for key, value in four_d.metadata.items()
        if key != "configuration"
    }
    runtime = dict(four_d_metadata.get("runtime_seconds_by_phase", {}))
    runtime["provisional_tracking"] = provisional_runtime_seconds
    four_d_metadata["runtime_seconds_by_phase"] = runtime
    metadata["architecture"] = (
        "boundary_aware_probabilistic_provisional_tracking_with_windowed_4d_post_optimization"
    )
    metadata["graph_tracking"] = {
        "mode": graph_config.mode,
        "algorithm": graph_config.algorithm,
        "enabled": True,
        "configuration": asdict(graph_config),
        **four_d_metadata,
    }
    summary = dict(provisional.summary)
    summary.update({
        "graph_mode": graph_config.mode,
        "graph_algorithm": graph_config.algorithm,
        "graph4d_assignment_changes": len(four_d.assignment_changes),
        "graph4d_selected_gap_edges": four_d.summary["selected_gap_edges"],
        "graph4d_solver_fallback_rate": four_d.summary["solver_fallback_rate"],
        "runtime_seconds_by_phase": runtime,
    })

    values: dict[str, Any] = {
        "graph4d_window_summary": four_d.window_summary,
        "graph4d_component_summary": four_d.component_summary,
        "graph4d_temporal_edges": four_d.temporal_edges,
        "graph4d_assignment_changes": four_d.assignment_changes,
        "graph4d_boundary_events": four_d.boundary_events,
        "graph4d_solver_diagnostics": four_d.solver_diagnostics,
        "graph4d_track_id_map": four_d.provisional_to_optimized_track_map,
        "graph4d_debug_artifacts": four_d.debug_artifacts,
        "metadata": metadata,
        "summary": summary,
    }
    if apply_mode:
        association_events = _rebuild_graph4d_association_events(
            provisional, four_d, time_frames
        )
        association_candidates = _rebuild_graph4d_association_candidates(
            provisional, four_d, time_frames
        )
        converted_boundary = pd.DataFrame(columns=provisional.boundary_events.columns)
        if not four_d.boundary_events.empty:
            converted_rows = []
            for row in four_d.boundary_events.itertuples(index=False):
                converted_rows.append({
                    "track_id": int(row.optimized_track_id),
                    "frame": int(row.frame),
                    "event_type": str(row.event_type),
                    "boundary_faces": str(row.boundary_face),
                    "missing_frames": 0,
                    "reacquired_frame": np.nan,
                    "confidence": np.nan,
                })
            converted_boundary = pd.DataFrame(converted_rows)
            for column in provisional.boundary_events.columns:
                if column not in converted_boundary:
                    converted_boundary[column] = np.nan
            converted_boundary = converted_boundary[
                provisional.boundary_events.columns
            ]
        values.update({
            "tracks": four_d.optimized_tracks,
            "boundary_events": converted_boundary,
            "boundary_predictions": provisional.boundary_predictions.iloc[0:0].copy(),
            "missing_predictions": provisional.missing_predictions.iloc[0:0].copy(),
            "tracking_diagnostics": _rebuild_graph4d_tracking_diagnostics(
                provisional, association_events
            ),
            "association_events": association_events,
            "association_candidates": association_candidates,
            "track_states": _rebuild_graph4d_track_states(
                provisional, four_d.optimized_tracks
            ),
        })
        summary.update({
            "track_records": len(four_d.optimized_tracks),
            "unique_tracks": int(four_d.optimized_tracks["track_id"].nunique()),
            "selected_misses": int((association_events["decision_type"] == "miss").sum()),
            "selected_births": int((association_events["decision_type"] == "birth").sum()),
        })
    return replace(provisional, **values)


def _four_d_trace(
    result: TrackingResult,
    time_frames: list[pd.DataFrame],
) -> StageTrace:
    decisions: list[DecisionRecord] = []
    for row in result.graph4d_assignment_changes.itertuples(index=False):
        decisions.append(DecisionRecord(
            decision_type=str(row.change_type),
            outcome="selected" if row.change_type == "continuation_added" else "rejected",
            frame=int(row.target_frame),
            subject_id=int(row.optimized_track_id),
            metrics={
                "source_node": int(row.source_node),
                "target_node": int(row.target_node),
                "component_id": int(row.component_id),
                "window_id": int(row.window_id),
                "solver_type": str(row.solver_type),
            },
            provenance=Provenance(
                source_type="graph4d_temporal_edge",
                source_stage="07_cell_tracking",
                source_frame=int(row.target_frame),
                source_track_ids=(int(row.provisional_track_id),),
                details={
                    "source_detection": int(row.source_detection_index),
                    "target_detection": int(row.target_detection_index),
                },
            ),
        ))
    selected_gaps = result.graph4d_temporal_edges.loc[
        result.graph4d_temporal_edges["optimized_selected"].astype(bool)
        & (result.graph4d_temporal_edges["frame_gap"] > 1)
    ]
    for row in selected_gaps.itertuples(index=False):
        decisions.append(DecisionRecord(
            decision_type="selected_gap_edge",
            outcome="selected",
            frame=int(row.target_frame),
            subject_id=int(row.edge_index),
            metrics={"frame_gap": int(row.frame_gap), "total_cost": float(row.total_effective_cost)},
            provenance=Provenance(
                source_type="graph4d_temporal_edge",
                source_stage="07_cell_tracking",
                source_frame=int(row.target_frame),
                source_candidate_id=int(row.edge_index),
                details={
                    "source_frame": int(row.source_frame),
                    "source_detection": int(row.source_detection_index),
                    "target_detection": int(row.target_detection_index),
                    "component_id": int(row.component_id),
                    "window_id": int(row.window_id),
                },
            ),
        ))
    for row in result.graph4d_boundary_events.itertuples(index=False):
        decisions.append(DecisionRecord(
            decision_type=str(row.event_type),
            outcome=str(row.boundary_face),
            frame=int(row.frame),
            subject_id=int(row.optimized_track_id),
            metrics={"cost": row.cost, "distance_to_face_um": row.distance_to_face_um},
            provenance=Provenance(
                source_type="graph4d_boundary_event",
                source_stage="07_cell_tracking",
                source_frame=int(row.frame),
                source_track_ids=(int(row.optimized_track_id),),
                details={
                    "detection_index": int(row.detection_index),
                    "component_id": int(row.component_id),
                    "window_id": int(row.window_id),
                },
            ),
        ))
    for row in result.graph4d_component_summary.itertuples(index=False):
        if bool(row.fallback_used):
            decisions.append(DecisionRecord(
                decision_type=(
                    "solver_failure" if row.status == "provisional_fallback"
                    else "fallback_component"
                ),
                outcome=str(row.status),
                frame=int(row.minimum_frame),
                subject_id=int(row.component_id),
                reason=str(row.failure_message),
                metrics={"node_count": int(row.node_count), "edge_count": int(row.temporal_edge_count)},
                provenance=Provenance(
                    source_type="graph4d_component",
                    source_stage="07_cell_tracking",
                    source_frame=int(row.minimum_frame),
                    source_candidate_id=int(row.component_id),
                ),
            ))
            if row.status == "provisional_fallback":
                decisions.append(DecisionRecord(
                    decision_type="unresolved_ambiguity",
                    outcome="provisional_component_retained",
                    frame=int(row.minimum_frame),
                    subject_id=int(row.component_id),
                    reason=str(row.failure_message),
                    metrics={
                        "minimum_frame": int(row.minimum_frame),
                        "maximum_frame": int(row.maximum_frame),
                    },
                    provenance=Provenance(
                        source_type="graph4d_component",
                        source_stage="07_cell_tracking",
                        source_frame=int(row.minimum_frame),
                        source_candidate_id=int(row.component_id),
                    ),
                ))
    return StageTrace(
        stage_name="07_cell_tracking",
        inputs={"time_frames": time_frames},
        outputs={"tracks": result.tracks},
        intermediates={
            "global_motion": result.global_motion,
            "association_events": result.association_events,
            "association_candidates": result.association_candidates,
            "boundary_events": result.boundary_events,
            "graph4d_window_summary": result.graph4d_window_summary,
            "graph4d_component_summary": result.graph4d_component_summary,
            "graph4d_temporal_edges": result.graph4d_temporal_edges,
            "graph4d_assignment_changes": result.graph4d_assignment_changes,
            "graph4d_boundary_events": result.graph4d_boundary_events,
            "graph4d_solver_diagnostics": result.graph4d_solver_diagnostics,
            "graph4d_track_id_map": result.graph4d_track_id_map,
        },
        metrics=result.summary,
        decisions=decisions,
    )


def run_cell_tracking(
    time_frames: list[pd.DataFrame],
    *,
    sample_id: str = "44b6_0113de3b",
    graph_config: GraphTrackingConfig | None = None,
    return_diagnostics: bool = False,
) -> TrackingResult | tuple[TrackingResult, StageTrace]:
    """Run provisional Stage 7 and optionally post-process it with a 4D graph."""

    config = graph_config or GraphTrackingConfig(mode="disabled")
    if config.mode == "disabled" or config.algorithm == "pairwise":
        return _run_provisional_or_pairwise_tracking(
            time_frames,
            sample_id=sample_id,
            graph_config=config,
            return_diagnostics=return_diagnostics,
        )

    provisional_started = __import__("time").perf_counter()
    provisional = _run_provisional_or_pairwise_tracking(
        time_frames,
        sample_id=sample_id,
        graph_config=GraphTrackingConfig(mode="disabled"),
        return_diagnostics=False,
    )
    provisional_runtime = __import__("time").perf_counter() - provisional_started
    four_d = run_four_d_graph_tracking(
        time_frames=time_frames,
        provisional_tracks=provisional.tracks,
        transition_evidence=provisional.transition_evidence,
        spatial_shape_zyx=VOLUME_SHAPE_ZYX,
        voxel_size_zyx_um=VOXEL_SIZE_ZYX,
        config=config.four_d,
    )
    result = _attach_four_d_result(
        provisional=provisional,
        four_d=four_d,
        time_frames=time_frames,
        graph_config=config,
        provisional_runtime_seconds=provisional_runtime,
    )
    if not return_diagnostics:
        return result
    return result, _four_d_trace(result, time_frames)
