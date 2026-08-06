"""Reliable temporal-anchor selection from the existing Stage 7 assignment."""

from __future__ import annotations

import math

import numpy as np

from .config import GraphTrackingConfig
from .types import TemporalAnchor


def _selected_pairs(assignment: dict) -> list[tuple[int, int]]:
    return [
        (int(row), int(col))
        for row, col in zip(assignment.get("rows", []), assignment.get("cols", []))
    ]


def selected_pair_maps(assignment: dict) -> tuple[dict[int, int], dict[int, int]]:
    by_track = dict(_selected_pairs(assignment))
    by_detection = {detection: track for track, detection in by_track.items()}
    return by_track, by_detection


def ambiguous_entities(
    assignment: dict,
    config: GraphTrackingConfig,
) -> tuple[set[int], set[int]]:
    """Find rows/columns that should remain eligible for graph refinement."""
    pair_cost = np.asarray(assignment["pair_cost_matrix"], dtype=float)
    safety_invalid = np.asarray(assignment["safety_invalid"], dtype=bool)
    association = np.asarray(assignment["association_probabilities"], dtype=float)
    margins = np.asarray(assignment["track_probability_margins"], dtype=float)
    track_prob = np.asarray(assignment["track_candidate_probabilities"], dtype=float)
    detection_prob = np.asarray(assignment["detection_candidate_probabilities"], dtype=float)
    selected_by_track, owner_by_detection = selected_pair_maps(assignment)
    missed = {int(value) for value in assignment.get("missed_state_indices", [])}
    births = {int(value) for value in assignment.get("birth_detection_indices", [])}

    ambiguous_tracks: set[int] = set()
    ambiguous_detections: set[int] = set()

    for track_index in range(pair_cost.shape[0]):
        selected_detection = selected_by_track.get(track_index)
        valid = np.flatnonzero(~safety_invalid[track_index])
        if selected_detection is None:
            if track_index in missed and valid.size:
                ambiguous_tracks.add(track_index)
                miss_cost = float(assignment["miss_costs"][track_index])
                ambiguous_detections.update(
                    int(value)
                    for value in valid
                    if float(pair_cost[track_index, int(value)])
                    <= miss_cost + config.ambiguous_cost_gap
                )
            continue
        selected_probability = float(association[track_index, selected_detection])
        if selected_probability < config.ambiguous_probability_threshold or float(margins[track_index]) < config.ambiguous_margin_threshold:
            ambiguous_tracks.add(track_index)
            ambiguous_detections.add(selected_detection)
        if valid.size >= 2:
            ordered = np.sort(pair_cost[track_index, valid])
            if float(ordered[1] - ordered[0]) <= config.ambiguous_cost_gap:
                ambiguous_tracks.add(track_index)
                best_cost = float(ordered[0])
                ambiguous_detections.update(
                    int(value)
                    for value in valid
                    if float(pair_cost[track_index, int(value)])
                    <= best_cost + config.ambiguous_cost_gap
                )

    for detection_index in range(pair_cost.shape[1]):
        valid = np.flatnonzero(~safety_invalid[:, detection_index])
        if detection_index in births and valid.size:
            ambiguous_detections.add(detection_index)
            # Stage 7 safety gates are deliberately broad. A birth can
            # therefore be safety-valid for many strong, unrelated matches;
            # propagating ambiguity to every such row removes all temporal
            # anchors in dense scenes. Only rows for which this birth is a
            # locally competitive alternative should become ambiguous.
            for value in valid:
                track_index = int(value)
                selected_detection = selected_by_track.get(track_index)
                selected_cost = (
                    float(pair_cost[track_index, selected_detection])
                    if selected_detection is not None
                    else float(assignment["miss_costs"][track_index])
                )
                if (
                    float(pair_cost[track_index, detection_index])
                    <= selected_cost + config.ambiguous_cost_gap
                ):
                    ambiguous_tracks.add(track_index)
            continue
        owner = owner_by_detection.get(detection_index)
        if owner is not None and float(detection_prob[owner, detection_index]) < config.ambiguous_probability_threshold:
            ambiguous_detections.add(detection_index)
            ambiguous_tracks.add(owner)
        if valid.size >= 2:
            ordered = np.sort(pair_cost[valid, detection_index])
            if float(ordered[1] - ordered[0]) <= config.ambiguous_cost_gap:
                ambiguous_detections.add(detection_index)
                best_cost = float(ordered[0])
                ambiguous_tracks.update(
                    int(value)
                    for value in valid
                    if float(pair_cost[int(value), detection_index])
                    <= best_cost + config.ambiguous_cost_gap
                )

    return ambiguous_tracks, ambiguous_detections


def select_temporal_anchors(
    *,
    eligible_states: list[dict],
    detections,
    current_frame: int,
    assignment: dict,
    ambiguous_tracks: set[int],
    ambiguous_detections: set[int],
    config: GraphTrackingConfig,
) -> list[TemporalAnchor]:
    anchors: list[TemporalAnchor] = []
    for state_index, detection_index in _selected_pairs(assignment):
        if state_index in ambiguous_tracks or detection_index in ambiguous_detections:
            continue
        state = eligible_states[state_index]
        if int(state.get("last_frame", -1)) != current_frame - 1:
            continue
        detection = detections.iloc[detection_index]
        source_boundary = bool(state.get("last_detection", {}).get("touches_boundary", False))
        target_boundary = bool(detection.get("touches_boundary", False))
        if not config.anchor_allow_boundary and (source_boundary or target_boundary):
            continue
        probability = float(assignment["association_probabilities"][state_index, detection_index])
        margin = float(assignment["track_probability_margins"][state_index])
        position_error = float(assignment["distance_matrix"][state_index, detection_index])
        if probability < config.anchor_minimum_association_probability:
            continue
        if margin < config.anchor_minimum_probability_margin:
            continue
        if position_error > config.anchor_maximum_position_error_um:
            continue
        margin_score = 1.0 - math.exp(-max(margin, 0.0) / max(config.anchor_margin_scale, 1.0e-9))
        position_score = math.exp(-position_error / max(config.anchor_position_scale_um, 1.0e-9))
        motion_confidence = float(state.get("relative_motion_confidence_next_frame", 1.0))
        if not np.isfinite(motion_confidence):
            motion_confidence = 1.0
        reliability = float(np.clip(probability * margin_score * position_score * max(motion_confidence, 0.25), 1.0e-6, 1.0))
        anchors.append(
            TemporalAnchor(
                track_id=int(state["track_id"]),
                source_node_index=state_index,
                target_node_index=detection_index,
                association_probability=probability,
                probability_margin=margin,
                position_error_um=position_error,
                reliability=reliability,
            )
        )
    anchors.sort(key=lambda anchor: (-anchor.reliability, anchor.track_id))
    return anchors


def locked_base_matches(
    assignment: dict,
    config: GraphTrackingConfig,
) -> dict[int, int]:
    locked: dict[int, int] = {}
    for state_index, detection_index in _selected_pairs(assignment):
        probability = float(assignment["association_probabilities"][state_index, detection_index])
        margin = float(assignment["track_probability_margins"][state_index])
        error = float(assignment["distance_matrix"][state_index, detection_index])
        if (
            probability >= config.lock_minimum_association_probability
            and margin >= config.lock_minimum_probability_margin
            and error <= config.lock_maximum_position_error_um
        ):
            locked[state_index] = detection_index
    return locked
