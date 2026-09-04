"""Sparse temporal continuation candidate generation and graph expansion."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import FourDGraphConfig
from .spatial_relations import neighbor_nodes
from .types import ObservationStore, SpatialFrameGraph, TemporalEdge, TransitionEvidence


_SHAPE_COLUMNS = (
    "equivalent_radius", "axis_major", "axis_middle", "axis_minor",
    "elongation", "flatness", "anisotropy", "solidity", "compactness",
)
_INTENSITY_COLUMNS = (
    "intensity_mean", "intensity_median", "intensity_std", "intensity_iqr",
    "intensity_cv",
)


def _finite_feature_error(
    observations: ObservationStore,
    source: int,
    target: int,
    columns: tuple[str, ...],
    feature_values: dict[str, np.ndarray],
) -> float:
    values: list[float] = []
    for column in columns:
        values_array = feature_values.get(column)
        if values_array is None:
            continue
        first = float(values_array[source])
        second = float(values_array[target])
        if math.isfinite(first) and math.isfinite(second):
            values.append(abs(second - first) / max(abs(first), abs(second), 1e-8))
    return float(np.median(values)) if values else math.nan


def provisional_edges(observations: ObservationStore) -> set[tuple[int, int]]:
    selected: set[tuple[int, int]] = set()
    table = observations.table
    valid = table.loc[table["provisional_track_id"] >= 0]
    for _, group in valid.groupby("provisional_track_id", sort=True):
        nodes = group.sort_values(["frame", "detection_index"], kind="mergesort")[
            "node_index"
        ].astype(int).tolist()
        selected.update(zip(nodes[:-1], nodes[1:]))
    return selected


def _source_node_lookup(observations: ObservationStore) -> dict[tuple[int, int], int]:
    return {
        (int(track_id), int(frame)): int(node)
        for node, (track_id, frame) in enumerate(zip(
            observations.provisional_track_ids, observations.frames
        ))
        if int(track_id) >= 0
    }


def _candidate_indices(
    evidence: TransitionEvidence,
    config: FourDGraphConfig,
) -> set[tuple[int, int]]:
    pair = evidence.pair_cost_matrix
    valid = ~evidence.safety_invalid
    keep: set[tuple[int, int]] = {
        (int(row), int(col))
        for row, col in zip(evidence.selected_rows, evidence.selected_columns)
    }
    for row in range(pair.shape[0]):
        columns = np.flatnonzero(valid[row])
        if not len(columns):
            continue
        ordered = columns[np.argsort(pair[row, columns], kind="mergesort")]
        keep.update((row, int(col)) for col in ordered[: config.temporal_candidates_per_source])
        reference = min(float(pair[row, ordered[0]]), float(evidence.miss_costs[row]))
        keep.update(
            (row, int(col)) for col in ordered
            if float(pair[row, col]) <= reference + config.locally_competitive_cost_delta
        )
    for col in range(pair.shape[1]):
        rows = np.flatnonzero(valid[:, col])
        if not len(rows):
            continue
        ordered = rows[np.argsort(pair[rows, col], kind="mergesort")]
        keep.update((int(row), col) for row in ordered[: config.temporal_candidates_per_target])
        reference = min(float(pair[ordered[0], col]), float(evidence.birth_costs[col]))
        keep.update(
            (int(row), col) for row in ordered
            if float(pair[row, col]) <= reference + config.locally_competitive_cost_delta
        )
    return keep


def _edge_from_evidence(
    *,
    source: int,
    target: int,
    row: int | None,
    col: int | None,
    evidence: TransitionEvidence | None,
    observations: ObservationStore,
    selected: set[tuple[int, int]],
    graph_expanded: bool,
    config: FourDGraphConfig,
    global_shift: np.ndarray,
    feature_values: dict[str, np.ndarray],
) -> TemporalEdge:
    gap = int(observations.frames[target] - observations.frames[source])
    delta = observations.positions_zyx_um[target] - observations.positions_zyx_um[source]
    displacement = float(np.linalg.norm(delta))
    volume_error = float(abs(math.log(
        max(observations.volumes[target], 1e-9)
        / max(observations.volumes[source], 1e-9)
    )))
    shape_error = _finite_feature_error(
        observations, source, target, _SHAPE_COLUMNS, feature_values
    )
    intensity_error = _finite_feature_error(
        observations, source, target, _INTENSITY_COLUMNS, feature_values
    )
    small_reliability = min(
        float(observations.small_cell_reliability[source]),
        float(observations.small_cell_reliability[target]),
    )
    appearance = small_reliability * (
        volume_error
        + (shape_error if math.isfinite(shape_error) else 0.0)
        + 0.5 * (intensity_error if math.isfinite(intensity_error) else 0.0)
    )
    predicted = observations.positions_zyx_um[source] + global_shift
    global_residual = float(np.linalg.norm(observations.positions_zyx_um[target] - predicted))
    relative_residual = global_residual
    base = math.nan
    probability = math.nan
    margin = math.nan
    safety = True
    if evidence is not None and row is not None and col is not None:
        safety = not bool(evidence.safety_invalid[row, col])
        base = float(evidence.pair_cost_matrix[row, col])
        probability = float(evidence.association_probabilities[row, col])
        margin = float(evidence.track_probability_margins[row])
        if row < len(evidence.relative_motion_predicted_positions_zyx_um):
            relative_residual = float(np.linalg.norm(
                observations.positions_zyx_um[target]
                - evidence.relative_motion_predicted_positions_zyx_um[row]
            ))
    maximum = (
        config.adjacent_maximum_distance_um if gap == 1
        else config.gap_maximum_distance_um_per_frame * gap
    )
    safety = safety and gap <= config.maximum_gap_frames and displacement <= maximum
    motion = 0.20 * global_residual + 0.15 * relative_residual
    unary = base if math.isfinite(base) else motion + appearance
    unary += config.gap_penalty_per_missing_frame * max(gap - 1, 0)
    if (source, target) in selected:
        unary -= 0.05
    return TemporalEdge(
        edge_index=-1,
        source=source,
        target=target,
        frame_gap=gap,
        provisional_selected=(source, target) in selected,
        graph_expanded=graph_expanded,
        hard_safety_valid=bool(safety),
        base_stage7_cost=base,
        unary_cost=float(config.unary_weight * unary),
        motion_cost=0.0,
        graph_cost=0.0,
        persistent_relation_cost=0.0,
        boundary_cost=0.0,
        displacement_um=displacement,
        global_motion_residual_um=global_residual,
        relative_motion_residual_um=relative_residual,
        volume_log_error=volume_error,
        shape_error=shape_error,
        intensity_error=intensity_error,
        provisional_probability=probability,
        provisional_margin=margin,
    )


def _global_displacement(
    start: int,
    target: int,
    shift_by_target_frame: dict[int, np.ndarray],
) -> np.ndarray:
    values = [
        shift_by_target_frame.get(frame, np.zeros(3, dtype=float))
        for frame in range(start + 1, target + 1)
    ]
    return np.sum(values, axis=0) if values else np.zeros(3, dtype=float)


def _expanded_pairs(
    *,
    observations: ObservationStore,
    spatial_graphs: dict[int, SpatialFrameGraph],
    existing: set[tuple[int, int]],
    selected: set[tuple[int, int]],
    config: FourDGraphConfig,
) -> set[tuple[int, int]]:
    if not config.enable_candidate_expansion:
        return set()
    selected_target = {source: target for source, target in selected}
    result: set[tuple[int, int]] = set()
    target_trees = {
        frame: cKDTree(observations.positions_zyx_um[nodes])
        for frame, nodes in observations.nodes_by_frame.items()
        if len(nodes)
    }
    for source in range(observations.node_count):
        source_frame = int(observations.frames[source])
        target_frame = source_frame + 1
        if target_frame not in observations.nodes_by_frame:
            continue
        graph = spatial_graphs[source_frame]
        votes: list[np.ndarray] = []
        for neighbor in neighbor_nodes(graph, source):
            target_neighbor = selected_target.get(int(neighbor))
            if target_neighbor is None or int(observations.frames[target_neighbor]) != target_frame:
                continue
            relative = observations.positions_zyx_um[source] - observations.positions_zyx_um[neighbor]
            votes.append(observations.positions_zyx_um[target_neighbor] + relative)
        if len(votes) < 2:
            continue
        votes_array = np.asarray(votes)
        center = np.median(votes_array, axis=0)
        residual = np.linalg.norm(votes_array - center, axis=1)
        mad = float(np.median(np.abs(residual - np.median(residual))))
        inliers = residual <= max(config.graph_expansion_radius_um, 2.5 * 1.4826 * mad)
        if not np.any(inliers):
            continue
        center = np.median(votes_array[inliers], axis=0)
        target_nodes = observations.nodes_by_frame[target_frame]
        tree = target_trees[target_frame]
        for local in tree.query_ball_point(center, config.graph_expansion_radius_um):
            pair = (source, int(target_nodes[int(local)]))
            if pair not in existing:
                result.add(pair)
    return result


def build_temporal_candidates(
    *,
    observations: ObservationStore,
    transition_evidence: tuple[TransitionEvidence, ...],
    spatial_graphs: dict[int, SpatialFrameGraph],
    config: FourDGraphConfig,
) -> list[TemporalEdge]:
    """Build the deterministic union of Stage 7, competitive, gap and graph candidates."""

    selected = provisional_edges(observations)
    source_lookup = _source_node_lookup(observations)
    shift_by_frame = {
        item.to_frame: np.asarray(item.global_shift_zyx_um, dtype=float)
        for item in transition_evidence
    }
    feature_values = {
        column: pd.to_numeric(
            observations.table[column], errors="coerce"
        ).to_numpy(dtype=float)
        for column in (*_SHAPE_COLUMNS, *_INTENSITY_COLUMNS)
        if column in observations.table
    }
    candidates: dict[tuple[int, int], tuple[TransitionEvidence | None, int | None, int | None, bool]] = {}
    explicit_validity: dict[tuple[int, int], bool] = {}
    for evidence in sorted(transition_evidence, key=lambda value: value.to_frame):
        for row, col in _candidate_indices(evidence, config):
            source = source_lookup.get((
                int(evidence.source_track_ids[row]), int(evidence.source_frames[row])
            ))
            target = observations.node_by_frame_detection.get((evidence.to_frame, col))
            if source is None or target is None:
                continue
            gap = int(observations.frames[target] - observations.frames[source])
            if gap < 1 or gap > config.maximum_gap_frames:
                continue
            key = (source, target)
            explicit_validity[key] = not bool(evidence.safety_invalid[row, col])
            previous = candidates.get(key)
            if previous is None or float(evidence.pair_cost_matrix[row, col]) < float(
                previous[0].pair_cost_matrix[previous[1], previous[2]]
            ):
                candidates[key] = (evidence, row, col, False)

    # A selected provisional continuation is never pruned by top-K selection.
    for source, target in selected:
        if int(observations.frames[target] - observations.frames[source]) <= config.maximum_gap_frames:
            candidates.setdefault((source, target), (None, None, None, False))

    expanded = _expanded_pairs(
        observations=observations,
        spatial_graphs=spatial_graphs,
        existing=set(candidates),
        selected=selected,
        config=config,
    )
    for pair in expanded:
        if explicit_validity.get(pair, True):
            candidates[pair] = (None, None, None, True)

    edges: list[TemporalEdge] = []
    for source, target in sorted(candidates, key=lambda pair: (
        int(observations.frames[pair[0]]), pair[0],
        int(observations.frames[pair[1]]), pair[1],
    )):
        evidence, row, col, graph_expanded = candidates[(source, target)]
        displacement = _global_displacement(
            int(observations.frames[source]), int(observations.frames[target]), shift_by_frame
        )
        edge = _edge_from_evidence(
            source=source,
            target=target,
            row=row,
            col=col,
            evidence=evidence,
            observations=observations,
            selected=selected,
            graph_expanded=graph_expanded,
            config=config,
            global_shift=displacement,
            feature_values=feature_values,
        )
        if edge.hard_safety_valid:
            edges.append(replace(edge, edge_index=len(edges)))
    return edges
