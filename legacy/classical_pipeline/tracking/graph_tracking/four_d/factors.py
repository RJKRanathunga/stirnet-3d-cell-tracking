"""Sparse trajectory, spatial-deformation, and persistent relation factors."""

from __future__ import annotations

import math

import numpy as np

from .config import FourDGraphConfig
from .types import (
    AmbiguityComponent,
    NeighborRelationHistory,
    ObservationStore,
    PairFactor,
    SpatialFrameGraph,
    TemporalEdge,
)


def _robust_scaled(value: float, scale: float) -> float:
    normalized = max(value, 0.0) / max(scale, 1e-9)
    return 0.5 * normalized * normalized if normalized <= 1.0 else normalized - 0.5


def _trajectory_factors(
    edges: list[TemporalEdge],
    edge_indices: set[int],
    observations: ObservationStore,
    config: FourDGraphConfig,
    limit: int,
) -> list[PairFactor]:
    incoming: dict[int, list[TemporalEdge]] = {}
    outgoing: dict[int, list[TemporalEdge]] = {}
    for index in sorted(edge_indices):
        edge = edges[index]
        incoming.setdefault(edge.target, []).append(edge)
        outgoing.setdefault(edge.source, []).append(edge)
    result: list[PairFactor] = []
    for middle in sorted(set(incoming) & set(outgoing)):
        first_edges = sorted(
            incoming[middle],
            key=lambda edge: (not edge.provisional_selected, edge.total_cost, edge.edge_index),
        )[: config.maximum_factor_candidates_per_node]
        second_edges = sorted(
            outgoing[middle],
            key=lambda edge: (not edge.provisional_selected, edge.total_cost, edge.edge_index),
        )[: config.maximum_factor_candidates_per_node]
        for first in first_edges:
            first_velocity = (
                observations.positions_zyx_um[middle]
                - observations.positions_zyx_um[first.source]
            ) / first.frame_gap
            for second in second_edges:
                second_velocity = (
                    observations.positions_zyx_um[second.target]
                    - observations.positions_zyx_um[middle]
                ) / second.frame_gap
                acceleration = float(np.linalg.norm(second_velocity - first_velocity))
                reliability = float(observations.small_cell_reliability[middle])
                cost = config.trajectory_weight * reliability * (
                    _robust_scaled(acceleration, config.acceleration_scale_um)
                    - config.compatible_factor_reward
                )
                result.append(PairFactor(
                    first_edge=first.edge_index,
                    second_edge=second.edge_index,
                    cost=float(cost),
                    factor_type="trajectory",
                ))
                if len(result) >= limit:
                    return result
    return result


def _spatial_factors(
    edges: list[TemporalEdge],
    edge_indices: set[int],
    observations: ObservationStore,
    spatial_graphs: dict[int, SpatialFrameGraph],
    histories: dict[tuple[int, int], NeighborRelationHistory],
    config: FourDGraphConfig,
    limit: int,
) -> list[PairFactor]:
    outgoing: dict[int, list[TemporalEdge]] = {}
    for index in sorted(edge_indices):
        outgoing.setdefault(edges[index].source, []).append(edges[index])
    for source in outgoing:
        outgoing[source] = sorted(
            outgoing[source],
            key=lambda edge: (not edge.provisional_selected, edge.total_cost, edge.edge_index),
        )[: config.maximum_factor_candidates_per_node]
    result: list[PairFactor] = []
    for frame, graph in sorted(spatial_graphs.items()):
        for spatial_index, (source, neighbor) in enumerate(zip(
            graph.edge_sources, graph.edge_targets
        )):
            if int(source) not in outgoing or int(neighbor) not in outgoing:
                continue
            source_vector = observations.positions_zyx_um[neighbor] - observations.positions_zyx_um[source]
            source_distance = float(np.linalg.norm(source_vector))
            first_track = int(observations.provisional_track_ids[source])
            second_track = int(observations.provisional_track_ids[neighbor])
            history_key = tuple(sorted((first_track, second_track)))
            history = histories.get(history_key)
            persistence = history.confidence if history is not None else 0.0
            relation_key = history_key if history is not None else None
            for first in outgoing[int(source)]:
                for second in outgoing[int(neighbor)]:
                    if first.frame_gap != second.frame_gap:
                        continue
                    if observations.frames[first.target] != observations.frames[second.target]:
                        continue
                    target_vector = (
                        observations.positions_zyx_um[second.target]
                        - observations.positions_zyx_um[first.target]
                    )
                    vector_error = float(np.linalg.norm(target_vector - source_vector))
                    radial_error = abs(float(np.linalg.norm(target_vector)) - source_distance)
                    source_norm = max(source_distance, 1e-9)
                    target_norm = max(float(np.linalg.norm(target_vector)), 1e-9)
                    cosine = float(np.clip(
                        np.dot(source_vector, target_vector) / (source_norm * target_norm), -1.0, 1.0
                    ))
                    angular_error = math.acos(cosine) / math.pi
                    visibility = max(float(graph.boundary_observability[spatial_index]), 0.15)
                    base_loss = (
                        _robust_scaled(vector_error, config.spatial_vector_scale_um)
                        + 0.5 * _robust_scaled(radial_error, config.relation_distance_scale_um)
                        + 0.5 * angular_error
                    )
                    cost = config.spatial_weight * visibility * (
                        base_loss - config.compatible_factor_reward
                    )
                    if history is not None:
                        history_vector = history.vector_median_zyx_um
                        if first_track > second_track:
                            history_vector = -history_vector
                        history_error = float(np.linalg.norm(target_vector - history_vector))
                        cost += config.persistent_relation_weight * persistence * (
                            _robust_scaled(history_error, config.relation_vector_scale_um)
                            - config.compatible_factor_reward
                        )
                    result.append(PairFactor(
                        first_edge=first.edge_index,
                        second_edge=second.edge_index,
                        cost=float(cost),
                        factor_type="spatial_persistent" if history is not None else "spatial",
                        supporting_relation=relation_key,
                    ))
                    if len(result) >= limit:
                        return result
    return result


def build_pair_factors(
    *,
    component: AmbiguityComponent,
    edges: list[TemporalEdge],
    observations: ObservationStore,
    spatial_graphs: dict[int, SpatialFrameGraph],
    histories: dict[tuple[int, int], NeighborRelationHistory],
    config: FourDGraphConfig,
) -> list[PairFactor]:
    """Construct only factors supported by actual trajectory/spatial structure."""

    indices = set(int(value) for value in component.edge_indices)
    factors: list[PairFactor] = []
    if config.trajectory_weight > 0:
        trajectory_limit = max(1, config.maximum_exact_pair_factors // 2)
        factors.extend(_trajectory_factors(
            edges, indices, observations, config, trajectory_limit
        ))
    if config.spatial_weight > 0 or config.persistent_relation_weight > 0:
        spatial_limit = max(0, config.maximum_exact_pair_factors - len(factors))
        if spatial_limit > 0:
            factors.extend(_spatial_factors(
                edges, indices, observations, spatial_graphs, histories, config,
                spatial_limit,
            ))
    factors.sort(key=lambda value: (
        value.factor_type, value.first_edge, value.second_edge, value.cost
    ))
    # Retain the most informative factors deterministically when capped.
    if len(factors) > config.maximum_exact_pair_factors:
        ranked = sorted(
            factors,
            key=lambda value: (-abs(value.cost), value.factor_type, value.first_edge, value.second_edge),
        )[: config.maximum_exact_pair_factors]
        factors = sorted(ranked, key=lambda value: (
            value.factor_type, value.first_edge, value.second_edge
        ))
    return factors
