"""Integration-ready graph refinement for one Stage 7 frame transition."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .anchors import ambiguous_entities, select_temporal_anchors
from .assignment import (
    AssignmentSolver,
    populate_assignment_probabilities,
    solve_assignment,
)
from .boundary import entry_hypothesis, exit_hypothesis
from .config import GraphTrackingConfig
from .diagnostics import (
    empty_anchor_votes,
    empty_boundary_hypotheses,
    empty_candidate_evidence,
    empty_refinement_events,
    table,
)
from .geometry import boundary_coverage_score
from .schemas import (
    GRAPH_ANCHOR_VOTE_COLUMNS,
    GRAPH_BOUNDARY_HYPOTHESIS_COLUMNS,
    GRAPH_CANDIDATE_EVIDENCE_COLUMNS,
    GRAPH_REFINEMENT_EVENT_COLUMNS,
    GRAPH_TRANSITION_SUMMARY_COLUMNS,
)
from .scoring import combined_graph_score, graph_confidence, graph_cost_delta
from .spatial_graph import build_spatial_graph
from .types import GraphRefinementResult
from .validation import validate_transition_inputs
from .voting import build_backward_vote_bundle, build_forward_vote_bundle, evaluate_candidate


def _state_position(state: dict) -> np.ndarray:
    value = state.get("last_position_physical")
    if value is None:
        detection = state["last_detection"]
        value = np.asarray([detection["centroid_z"], detection["centroid_y"], detection["centroid_x"]], dtype=float)
    return np.asarray(value, dtype=float)


def _source_graph(eligible_states: list[dict], current_frame: int, config: GraphTrackingConfig):
    positions = np.asarray([_state_position(state) for state in eligible_states], dtype=float).reshape(-1, 3)
    volumes = np.asarray([
        float(state.get("last_detection", {}).get("volume_voxels", state.get("volume", np.nan)))
        for state in eligible_states
    ], dtype=float)
    volumes[~np.isfinite(volumes)] = 1.0
    boundary = np.asarray([bool(state.get("last_detection", {}).get("touches_boundary", False)) for state in eligible_states])
    faces = tuple(str(state.get("last_detection", {}).get("boundary_faces", "")) for state in eligible_states)
    return build_spatial_graph(
        frame=current_frame - 1,
        node_ids=np.asarray([int(state["track_id"]) for state in eligible_states], dtype=np.int64),
        positions_zyx_um=positions,
        volumes=volumes,
        touches_boundary=boundary,
        boundary_faces=faces,
        config=config,
    )


def _target_graph(detections: pd.DataFrame, current_frame: int, voxel_size_zyx_um: np.ndarray, config: GraphTrackingConfig):
    if len(detections):
        positions_voxel = detections[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
        positions = positions_voxel * np.asarray(voxel_size_zyx_um, dtype=float)[None, :]
    else:
        positions = np.empty((0, 3), dtype=float)
    volumes = detections.get("volume_voxels", pd.Series(np.ones(len(detections)))).to_numpy(dtype=float)
    boundary = detections.get("touches_boundary", pd.Series(np.zeros(len(detections), dtype=bool))).to_numpy(dtype=bool)
    faces = tuple(str(value) for value in detections.get("boundary_faces", pd.Series([""] * len(detections))).tolist())
    return build_spatial_graph(
        frame=current_frame,
        node_ids=np.arange(len(detections), dtype=np.int64),
        positions_zyx_um=positions,
        volumes=volumes,
        touches_boundary=boundary,
        boundary_faces=faces,
        config=config,
    )


def _decision_maps(assignment: dict) -> tuple[dict[int, int], set[int], set[int]]:
    matches = {int(row): int(col) for row, col in zip(assignment.get("rows", []), assignment.get("cols", []))}
    misses = {int(value) for value in assignment.get("missed_state_indices", [])}
    births = {int(value) for value in assignment.get("birth_detection_indices", [])}
    return matches, misses, births


def _refinement_events(
    *,
    eligible_states: list[dict],
    base_assignment: dict,
    graph_assignment: dict,
    current_frame: int,
) -> pd.DataFrame:
    base_matches, base_misses, base_births = _decision_maps(base_assignment)
    graph_matches, graph_misses, graph_births = _decision_maps(graph_assignment)
    rows: list[dict[str, Any]] = []
    for state_index, state in enumerate(eligible_states):
        base_target = base_matches.get(state_index)
        graph_target = graph_matches.get(state_index)
        base_decision = "match" if base_target is not None else "miss" if state_index in base_misses else "unassigned"
        graph_decision = "match" if graph_target is not None else "miss" if state_index in graph_misses else "unassigned"
        base_cost = float(base_assignment["pair_cost_matrix"][state_index, base_target]) if base_target is not None else float(base_assignment["miss_costs"][state_index])
        graph_cost = float(graph_assignment["pair_cost_matrix"][state_index, graph_target]) if graph_target is not None else float(graph_assignment["miss_costs"][state_index])
        rows.append({
            "from_frame": current_frame - 1,
            "to_frame": current_frame,
            "entity_type": "track",
            "entity_index": state_index,
            "track_id": int(state["track_id"]),
            "base_decision": base_decision,
            "graph_decision": graph_decision,
            "changed": bool(base_decision != graph_decision or base_target != graph_target),
            "base_target_index": np.nan if base_target is None else base_target,
            "graph_target_index": np.nan if graph_target is None else graph_target,
            "base_cost": base_cost,
            "graph_cost": graph_cost,
        })
    for detection_index in sorted(base_births | graph_births):
        rows.append({
            "from_frame": current_frame - 1,
            "to_frame": current_frame,
            "entity_type": "detection",
            "entity_index": detection_index,
            "track_id": np.nan,
            "base_decision": "birth" if detection_index in base_births else "matched",
            "graph_decision": "birth" if detection_index in graph_births else "matched",
            "changed": bool((detection_index in base_births) != (detection_index in graph_births)),
            "base_target_index": np.nan,
            "graph_target_index": np.nan,
            "base_cost": float(base_assignment["birth_costs"][detection_index]),
            "graph_cost": float(graph_assignment["birth_costs"][detection_index]),
        })
    return table(rows, GRAPH_REFINEMENT_EVENT_COLUMNS)


def _disabled_result(base_assignment: dict, current_frame: int, config: GraphTrackingConfig) -> GraphRefinementResult:
    summary = table([{
        "from_frame": current_frame - 1, "to_frame": current_frame, "mode": config.mode,
        "source_nodes": 0, "target_nodes": 0, "source_edges": 0, "target_edges": 0,
        "temporal_anchors": 0, "ambiguous_tracks": 0, "ambiguous_detections": 0,
        "candidate_pairs_scored": 0, "exit_hypotheses": 0, "entry_hypotheses": 0,
        "assignment_changes": 0, "base_matches": len(base_assignment.get("rows", [])),
        "graph_matches": len(base_assignment.get("rows", [])),
        "base_misses": len(base_assignment.get("missed_state_indices", [])),
        "graph_misses": len(base_assignment.get("missed_state_indices", [])),
        "base_births": len(base_assignment.get("birth_detection_indices", [])),
        "graph_births": len(base_assignment.get("birth_detection_indices", [])),
    }], GRAPH_TRANSITION_SUMMARY_COLUMNS)
    return GraphRefinementResult(
        assignment=base_assignment,
        base_assignment=base_assignment,
        graph_assignment=base_assignment,
        candidate_evidence=empty_candidate_evidence(),
        anchor_votes=empty_anchor_votes(),
        boundary_hypotheses=empty_boundary_hypotheses(),
        refinement_events=empty_refinement_events(),
        transition_summary=summary,
        metadata={"mode": config.mode, "applied": False},
    )


def refine_transition_with_graph(
    *,
    eligible_states: list[dict],
    detections: pd.DataFrame,
    current_frame: int,
    base_assignment: dict,
    volume_shape_zyx: np.ndarray,
    voxel_size_zyx_um: np.ndarray,
    config: GraphTrackingConfig | None = None,
    assignment_solver: AssignmentSolver | None = None,
) -> GraphRefinementResult:
    """Refine one Stage 7 transition with vector-preserving graph evidence.

    The input assignment is the current Stage 7 output after global-motion
    refinement. In shadow mode, the returned effective assignment remains the
    base assignment while diagnostics expose the graph alternative.
    """
    config = config or GraphTrackingConfig()
    validate_transition_inputs(
        eligible_states=eligible_states,
        detections=detections,
        base_assignment=base_assignment,
        volume_shape_zyx=volume_shape_zyx,
        voxel_size_zyx_um=voxel_size_zyx_um,
    )
    if config.mode == "disabled":
        return _disabled_result(base_assignment, current_frame, config)

    source_graph = _source_graph(eligible_states, current_frame, config)
    target_graph = _target_graph(detections, current_frame, voxel_size_zyx_um, config)
    ambiguous_tracks, ambiguous_detections = ambiguous_entities(base_assignment, config)
    anchors = select_temporal_anchors(
        eligible_states=eligible_states,
        detections=detections,
        current_frame=current_frame,
        assignment=base_assignment,
        ambiguous_tracks=ambiguous_tracks,
        ambiguous_detections=ambiguous_detections,
        config=config,
    )
    anchors_by_source = {anchor.source_node_index: anchor for anchor in anchors}
    anchors_by_target = {anchor.target_node_index: anchor for anchor in anchors}

    pair_cost = np.asarray(base_assignment["pair_cost_matrix"], dtype=float).copy()
    miss_cost = np.asarray(base_assignment["miss_costs"], dtype=float).copy()
    birth_cost = np.asarray(base_assignment["birth_costs"], dtype=float).copy()
    safety_invalid = np.asarray(base_assignment["safety_invalid"], dtype=bool)
    base_matches, base_misses, base_births = _decision_maps(base_assignment)

    candidate_rows: list[dict[str, Any]] = []
    vote_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    forward_bundles: dict[int, Any] = {}

    tracks_to_score = set(ambiguous_tracks) | set(base_misses)
    for state_index in sorted(tracks_to_score):
        bundle = build_forward_vote_bundle(
            source_graph=source_graph,
            target_graph=target_graph,
            source_node_index=state_index,
            anchors_by_source_index=anchors_by_source,
            config=config,
        )
        if bundle is None:
            continue
        forward_bundles[state_index] = bundle
        valid_candidates = np.flatnonzero(~safety_invalid[state_index])
        if valid_candidates.size:
            ordered = valid_candidates[np.argsort(pair_cost[state_index, valid_candidates], kind="mergesort")]
            keep = set(int(value) for value in ordered[: config.save_top_candidates_per_track])
            selected = base_matches.get(state_index)
            if selected is not None:
                keep.add(int(selected))
            keep.update(int(value) for value in ambiguous_detections if not safety_invalid[state_index, int(value)])
            valid_candidates = np.asarray(sorted(keep, key=lambda value: (pair_cost[state_index, value], value)), dtype=int)
        for detection_index in valid_candidates:
            evidence = evaluate_candidate(
                bundle=bundle,
                source_graph=source_graph,
                target_graph=target_graph,
                candidate_node_index=int(detection_index),
                config=config,
            )
            source_coverage = boundary_coverage_score(
                source_graph.positions_zyx_um[state_index],
                search_radius_um=config.maximum_radius_um,
                volume_shape_zyx=volume_shape_zyx,
                voxel_size_zyx_um=voxel_size_zyx_um,
            )
            target_coverage = boundary_coverage_score(
                target_graph.positions_zyx_um[detection_index],
                search_radius_um=config.maximum_radius_um,
                volume_shape_zyx=volume_shape_zyx,
                voxel_size_zyx_um=voxel_size_zyx_um,
            )
            confidence = graph_confidence(
                anchor_count=len(bundle.weights),
                inlier_fraction=bundle.consensus.inlier_weight_fraction,
                dispersion_um=bundle.consensus.dispersion_um,
                source_coverage=source_coverage,
                target_coverage=target_coverage,
                config=config,
            )
            score = combined_graph_score(evidence, config)
            delta = graph_cost_delta(score, confidence, config)
            pair_cost[state_index, detection_index] += delta
            candidate_rows.append({
                "from_frame": current_frame - 1, "to_frame": current_frame,
                "source_track_id": int(eligible_states[state_index]["track_id"]),
                "source_state_index": state_index, "candidate_detection_index": int(detection_index),
                "anchor_count": len(bundle.weights), "inlier_count": int(bundle.consensus.inlier_mask.sum()),
                "anchor_track_ids": "|".join(str(int(value)) for value in bundle.anchor_track_ids),
                "vote_score": evidence.vote_score, "vector_score": evidence.vector_score,
                "radial_score": evidence.radial_score, "relative_volume_score": evidence.relative_volume_score,
                "consensus_score": evidence.consensus_score, "deformation_score": evidence.deformation_score,
                "graph_score": score, "graph_confidence": confidence, "graph_cost_delta": delta,
                "consensus_z_um": bundle.consensus.position_zyx_um[0],
                "consensus_y_um": bundle.consensus.position_zyx_um[1],
                "consensus_x_um": bundle.consensus.position_zyx_um[2],
                "consensus_dispersion_um": bundle.consensus.dispersion_um,
                "deformation_transform_type": "none" if bundle.local_transform is None else bundle.local_transform.transform_type,
                "deformation_condition_number": np.nan if bundle.local_transform is None else bundle.local_transform.condition_number,
                "deformation_fit_residual_um": np.nan if bundle.local_transform is None else bundle.local_transform.residual_um,
                "deformation_prediction_z_um": bundle.deformation_prediction_zyx_um[0],
                "deformation_prediction_y_um": bundle.deformation_prediction_zyx_um[1],
                "deformation_prediction_x_um": bundle.deformation_prediction_zyx_um[2],
                "source_boundary_coverage": source_coverage, "target_boundary_coverage": target_coverage,
                "base_selected": base_matches.get(state_index) == int(detection_index), "graph_selected": False,
            })
            if config.save_all_anchor_votes:
                for anchor_index, anchor_track_id in enumerate(bundle.anchor_track_ids):
                    residual = evidence.residual_vectors_zyx_um[anchor_index]
                    source_relative = bundle.source_relative_vectors_zyx_um[anchor_index]
                    vote = bundle.vote_positions_zyx_um[anchor_index]
                    vote_rows.append({
                        "from_frame": current_frame - 1, "to_frame": current_frame,
                        "source_track_id": int(eligible_states[state_index]["track_id"]),
                        "source_state_index": state_index, "candidate_detection_index": int(detection_index),
                        "anchor_track_id": int(anchor_track_id),
                        "source_relative_z_um": source_relative[0], "source_relative_y_um": source_relative[1], "source_relative_x_um": source_relative[2],
                        "vote_z_um": vote[0], "vote_y_um": vote[1], "vote_x_um": vote[2],
                        "candidate_residual_z_um": residual[0], "candidate_residual_y_um": residual[1], "candidate_residual_x_um": residual[2],
                        "candidate_residual_norm_um": evidence.residual_norms_um[anchor_index],
                        "relative_volume_log_error": abs(
                            np.log(max(target_graph.volumes[bundle.anchor_target_indices[anchor_index]], 1.0e-9) / max(target_graph.volumes[detection_index], 1.0e-9))
                            - np.log(max(source_graph.volumes[bundle.anchor_source_indices[anchor_index]], 1.0e-9) / max(source_graph.volumes[state_index], 1.0e-9))
                        ),
                        "anchor_weight": evidence.weights[anchor_index], "inlier": bool(evidence.inlier_mask[anchor_index]),
                    })

        if state_index in base_misses:
            hypothesis = exit_hypothesis(
                bundle=bundle,
                source_graph=source_graph,
                source_node_index=state_index,
                volume_shape_zyx=volume_shape_zyx,
                voxel_size_zyx_um=voxel_size_zyx_um,
                config=config,
            )
            if hypothesis is not None:
                if hypothesis["supported"]:
                    miss_cost[state_index] -= config.boundary_exit_maximum_miss_cost_reduction * float(hypothesis["confidence"])
                predicted = hypothesis["predicted_position_zyx_um"]
                boundary_rows.append({
                    "event_type": hypothesis["event_type"], "from_frame": current_frame - 1, "to_frame": current_frame,
                    "track_id": int(eligible_states[state_index]["track_id"]), "detection_index": np.nan,
                    "boundary_face": hypothesis["boundary_face"], "predicted_z_um": predicted[0], "predicted_y_um": predicted[1], "predicted_x_um": predicted[2],
                    "outside_distance_um": hypothesis["outside_distance_um"], "anchor_count": hypothesis["anchor_count"],
                    "inlier_count": hypothesis["inlier_count"], "consensus": hypothesis["consensus"],
                    "directional_agreement": hypothesis["directional_agreement"], "confidence": hypothesis["confidence"],
                    "supported": hypothesis["supported"], "decision": "graph_exit_predicted" if hypothesis["supported"] else "insufficient_exit_evidence",
                })

    detections_to_score = set(ambiguous_detections) | set(base_births)
    for detection_index in sorted(detections_to_score):
        bundle = build_backward_vote_bundle(
            source_graph=source_graph,
            target_graph=target_graph,
            target_node_index=detection_index,
            anchors_by_target_index=anchors_by_target,
            config=config,
        )
        if bundle is None:
            continue
        hypothesis = entry_hypothesis(
            bundle=bundle,
            target_graph=target_graph,
            target_node_index=detection_index,
            volume_shape_zyx=volume_shape_zyx,
            voxel_size_zyx_um=voxel_size_zyx_um,
            config=config,
        )
        if hypothesis is None:
            continue
        if hypothesis["event_type"] == "entry" and hypothesis["supported"]:
            birth_cost[detection_index] -= config.boundary_entry_maximum_birth_cost_reduction * float(hypothesis["confidence"])
            decision = "graph_entry_supported"
        elif hypothesis["event_type"] == "inside_predecessor":
            birth_cost[detection_index] += config.inside_backward_vote_birth_penalty * float(hypothesis["confidence"])
            decision = "inside_predecessor_vote"
        else:
            decision = "insufficient_entry_evidence"
        predicted = hypothesis["predicted_position_zyx_um"]
        boundary_rows.append({
            "event_type": hypothesis["event_type"], "from_frame": current_frame - 1, "to_frame": current_frame,
            "track_id": np.nan, "detection_index": detection_index, "boundary_face": hypothesis["boundary_face"],
            "predicted_z_um": predicted[0], "predicted_y_um": predicted[1], "predicted_x_um": predicted[2],
            "outside_distance_um": hypothesis["outside_distance_um"], "anchor_count": hypothesis["anchor_count"],
            "inlier_count": hypothesis["inlier_count"], "consensus": hypothesis["consensus"],
            "directional_agreement": hypothesis["directional_agreement"], "confidence": hypothesis["confidence"],
            "supported": hypothesis["supported"], "decision": decision,
        })

    graph_costs_changed = not (
        np.array_equal(pair_cost, base_assignment["pair_cost_matrix"])
        and np.array_equal(miss_cost, base_assignment["miss_costs"])
        and np.array_equal(birth_cost, base_assignment["birth_costs"])
    )

    # Keep all non-ambiguous base matches fixed. They provide the trusted
    # temporal correspondences used by the graph and must not be rewritten by
    # the same refinement pass.
    locked_safety_invalid = safety_invalid.copy()
    solver_miss_cost = miss_cost.copy()
    solver_birth_cost = birth_cost.copy()
    locked_matches = {
        int(state_index): int(detection_index)
        for state_index, detection_index in base_matches.items()
        if state_index not in ambiguous_tracks and detection_index not in ambiguous_detections
    }
    for state_index, detection_index in locked_matches.items():
        locked_safety_invalid[state_index, :] = True
        locked_safety_invalid[:, detection_index] = True
        locked_safety_invalid[state_index, detection_index] = False
        solver_miss_cost[state_index] = config.invalid_cost
        solver_birth_cost[detection_index] = config.invalid_cost

    if graph_costs_changed:
        graph_assignment = solve_assignment(
            pair_cost,
            solver_miss_cost,
            solver_birth_cost,
            safety_invalid=locked_safety_invalid,
            config=config,
            solver=assignment_solver,
        )
        # Solver-only locks must not alter reported probabilities or margins.
        graph_assignment = populate_assignment_probabilities(
            graph_assignment,
            pair_cost_matrix=pair_cost,
            miss_costs=miss_cost,
            birth_costs=birth_cost,
            safety_invalid=safety_invalid,
        )
        # Preserve all non-assignment evidence produced by current Stage 7.
        graph_assignment = {**base_assignment, **graph_assignment}
        graph_assignment["graph_solver_safety_invalid"] = (
            locked_safety_invalid
        )
    else:
        # Exact fallback: no evidence means no numerical or ordering changes.
        graph_assignment = base_assignment
    graph_matches, _, _ = _decision_maps(graph_assignment)
    for row in candidate_rows:
        row["graph_selected"] = graph_matches.get(int(row["source_state_index"])) == int(row["candidate_detection_index"])

    candidate_table = table(candidate_rows, GRAPH_CANDIDATE_EVIDENCE_COLUMNS)
    vote_table = table(vote_rows, GRAPH_ANCHOR_VOTE_COLUMNS)
    boundary_table = table(boundary_rows, GRAPH_BOUNDARY_HYPOTHESIS_COLUMNS)
    events = (
        _refinement_events(
            eligible_states=eligible_states,
            base_assignment=base_assignment,
            graph_assignment=graph_assignment,
            current_frame=current_frame,
        )
        if graph_costs_changed
        else empty_refinement_events()
    )
    changes = int(events["changed"].sum()) if not events.empty else 0
    summary = table([{
        "from_frame": current_frame - 1, "to_frame": current_frame, "mode": config.mode,
        "source_nodes": source_graph.node_count, "target_nodes": target_graph.node_count,
        "source_edges": source_graph.edge_count, "target_edges": target_graph.edge_count,
        "temporal_anchors": len(anchors), "ambiguous_tracks": len(ambiguous_tracks),
        "ambiguous_detections": len(ambiguous_detections), "candidate_pairs_scored": len(candidate_table),
        "exit_hypotheses": int((boundary_table["event_type"] == "exit").sum()) if not boundary_table.empty else 0,
        "entry_hypotheses": int((boundary_table["event_type"] == "entry").sum()) if not boundary_table.empty else 0,
        "assignment_changes": changes, "base_matches": len(base_assignment.get("rows", [])),
        "graph_matches": len(graph_assignment.get("rows", [])),
        "base_misses": len(base_assignment.get("missed_state_indices", [])),
        "graph_misses": len(graph_assignment.get("missed_state_indices", [])),
        "base_births": len(base_assignment.get("birth_detection_indices", [])),
        "graph_births": len(graph_assignment.get("birth_detection_indices", [])),
    }], GRAPH_TRANSITION_SUMMARY_COLUMNS)
    effective = graph_assignment if config.mode == "apply" else base_assignment
    return GraphRefinementResult(
        assignment=effective,
        base_assignment=base_assignment,
        graph_assignment=graph_assignment,
        candidate_evidence=candidate_table,
        anchor_votes=vote_table,
        boundary_hypotheses=boundary_table,
        refinement_events=events,
        transition_summary=summary,
        metadata={
            "mode": config.mode,
            "applied": config.mode == "apply",
            "source_graph_edges": source_graph.edge_count,
            "target_graph_edges": target_graph.edge_count,
            "temporal_anchor_count": len(anchors),
            "assignment_changes": changes,
        },
    )
