"""Augmented assignment and graph-refinement solution helpers."""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import GraphTrackingConfig


AssignmentSolver = Callable[[np.ndarray, np.ndarray, np.ndarray], dict]


def augmented_assignment(
    pair_cost_matrix: np.ndarray,
    miss_costs: np.ndarray,
    birth_costs: np.ndarray,
    *,
    invalid_cost: float = 1.0e6,
) -> dict:
    """Solve one-to-one matches with private miss and birth alternatives."""
    pair = np.asarray(pair_cost_matrix, dtype=float)
    miss = np.asarray(miss_costs, dtype=float)
    birth = np.asarray(birth_costs, dtype=float)
    track_count, detection_count = pair.shape
    total = track_count + detection_count
    matrix = np.full((total, total), invalid_cost, dtype=float)
    matrix[:track_count, :detection_count] = pair
    if track_count:
        matrix[np.arange(track_count), detection_count + np.arange(track_count)] = miss
    if detection_count:
        matrix[track_count + np.arange(detection_count), np.arange(detection_count)] = birth
    if track_count and detection_count:
        matrix[track_count:, detection_count:] = 0.0
    rows_all, cols_all = linear_sum_assignment(matrix)

    matches: list[tuple[int, int]] = []
    misses: list[int] = []
    births: list[int] = []
    for row, col in zip(rows_all, cols_all):
        if row < track_count and col < detection_count:
            matches.append((int(row), int(col)))
        elif row < track_count and col >= detection_count:
            misses.append(int(row))
        elif row >= track_count and col < detection_count:
            births.append(int(col))
    matches.sort()
    misses.sort()
    births.sort()
    return {
        "rows": np.asarray([row for row, _ in matches], dtype=np.int32),
        "cols": np.asarray([col for _, col in matches], dtype=np.int32),
        "missed_state_indices": np.asarray(misses, dtype=np.int32),
        "birth_detection_indices": np.asarray(births, dtype=np.int32),
        "objective_cost": float(matrix[rows_all, cols_all].sum()),
    }


def normalized_choice_probabilities(
    pair_costs: np.ndarray,
    alternative_cost: float,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    costs = np.asarray(pair_costs, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    probabilities = np.zeros_like(costs, dtype=float)
    values = [float(alternative_cost)]
    valid_indices = np.flatnonzero(valid)
    values.extend(float(costs[index]) for index in valid_indices)
    values_array = np.asarray(values, dtype=float)
    minimum = float(np.min(values_array))
    weights = np.exp(-(values_array - minimum))
    weights /= float(weights.sum())
    alternative_probability = float(weights[0])
    if valid_indices.size:
        probabilities[valid_indices] = weights[1:]
    return probabilities, alternative_probability


def populate_assignment_probabilities(
    assignment: dict,
    *,
    pair_cost_matrix: np.ndarray,
    miss_costs: np.ndarray,
    birth_costs: np.ndarray,
    safety_invalid: np.ndarray,
) -> dict:
    pair = np.asarray(pair_cost_matrix, dtype=float)
    valid = ~np.asarray(safety_invalid, dtype=bool)
    track_count, detection_count = pair.shape
    track_candidate = np.zeros_like(pair)
    track_no_match = np.zeros(track_count)
    for index in range(track_count):
        track_candidate[index], track_no_match[index] = normalized_choice_probabilities(pair[index], float(miss_costs[index]), valid[index])
    detection_candidate = np.zeros_like(pair)
    detection_birth = np.zeros(detection_count)
    for index in range(detection_count):
        detection_candidate[:, index], detection_birth[index] = normalized_choice_probabilities(pair[:, index], float(birth_costs[index]), valid[:, index])
    association = np.sqrt(track_candidate * detection_candidate)

    selected = {int(row): int(col) for row, col in zip(assignment["rows"], assignment["cols"])}
    missed = {int(value) for value in assignment["missed_state_indices"]}
    margins = np.zeros(track_count)
    for index in range(track_count):
        if index in selected:
            col = selected[index]
            chosen = float(track_candidate[index, col])
            alternatives = np.concatenate([np.delete(track_candidate[index], col), np.asarray([track_no_match[index]])])
        elif index in missed:
            chosen = float(track_no_match[index])
            alternatives = track_candidate[index]
        else:
            chosen = 0.0
            alternatives = track_candidate[index]
        margins[index] = chosen - (float(np.max(alternatives)) if alternatives.size else 0.0)

    updated = dict(assignment)
    updated.update(
        {
            "pair_cost_matrix": pair,
            "cost_matrix": pair,
            "miss_costs": np.asarray(miss_costs, dtype=float),
            "birth_costs": np.asarray(birth_costs, dtype=float),
            "track_candidate_probabilities": track_candidate,
            "detection_candidate_probabilities": detection_candidate,
            "association_probabilities": association,
            "track_no_match_probabilities": track_no_match,
            "detection_birth_choice_probabilities": detection_birth,
            "track_probability_margins": margins,
        }
    )
    return updated


def solve_assignment(
    pair_cost_matrix: np.ndarray,
    miss_costs: np.ndarray,
    birth_costs: np.ndarray,
    *,
    safety_invalid: np.ndarray,
    config: GraphTrackingConfig,
    solver: AssignmentSolver | None = None,
) -> dict:
    effective_pair = np.asarray(pair_cost_matrix, dtype=float).copy()
    effective_pair[np.asarray(safety_invalid, dtype=bool)] = config.invalid_cost
    if solver is None:
        result = augmented_assignment(effective_pair, miss_costs, birth_costs, invalid_cost=config.invalid_cost)
    else:
        result = solver(effective_pair, np.asarray(miss_costs, dtype=float), np.asarray(birth_costs, dtype=float))
    return populate_assignment_probabilities(
        result,
        pair_cost_matrix=effective_pair,
        miss_costs=miss_costs,
        birth_costs=birth_costs,
        safety_invalid=safety_invalid,
    )


def bipartite_components(valid_mask: np.ndarray, tracks: set[int], detections: set[int]) -> list[tuple[list[int], list[int]]]:
    valid = np.asarray(valid_mask, dtype=bool)
    remaining_tracks = set(tracks)
    remaining_detections = set(detections)
    components: list[tuple[list[int], list[int]]] = []
    while remaining_tracks or remaining_detections:
        if remaining_tracks:
            queue = deque([("t", remaining_tracks.pop())])
        else:
            queue = deque([("d", remaining_detections.pop())])
        component_tracks: set[int] = set()
        component_detections: set[int] = set()
        while queue:
            kind, index = queue.popleft()
            if kind == "t":
                if index in component_tracks:
                    continue
                component_tracks.add(index)
                neighbors = [int(value) for value in np.flatnonzero(valid[index]) if int(value) in detections]
                for neighbor in neighbors:
                    if neighbor not in component_detections:
                        remaining_detections.discard(neighbor)
                        queue.append(("d", neighbor))
            else:
                if index in component_detections:
                    continue
                component_detections.add(index)
                neighbors = [int(value) for value in np.flatnonzero(valid[:, index]) if int(value) in tracks]
                for neighbor in neighbors:
                    if neighbor not in component_tracks:
                        remaining_tracks.discard(neighbor)
                        queue.append(("t", neighbor))
        components.append((sorted(component_tracks), sorted(component_detections)))
    return components
