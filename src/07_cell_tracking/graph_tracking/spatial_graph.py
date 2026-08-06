"""Sparse spatial graph construction with shared node storage."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .config import GraphTrackingConfig
from .types import SpatialGraph


def _validate_nodes(
    node_ids: np.ndarray,
    positions_zyx_um: np.ndarray,
    volumes: np.ndarray,
    touches_boundary: np.ndarray,
    boundary_faces: tuple[str, ...],
) -> None:
    positions = np.asarray(positions_zyx_um)
    count = int(positions.shape[0])
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_zyx_um must have shape (N, 3)")
    if len(node_ids) != count or len(volumes) != count or len(touches_boundary) != count:
        raise ValueError("node arrays must have the same length")
    if len(boundary_faces) != count:
        raise ValueError("boundary_faces must have one entry per node")


def build_spatial_graph(
    *,
    frame: int,
    node_ids: np.ndarray,
    positions_zyx_um: np.ndarray,
    volumes: np.ndarray,
    touches_boundary: np.ndarray,
    boundary_faces: tuple[str, ...],
    config: GraphTrackingConfig,
) -> SpatialGraph:
    """Build a deterministic radius-limited sparse undirected graph."""
    node_ids = np.asarray(node_ids, dtype=np.int64)
    positions = np.asarray(positions_zyx_um, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    touches_boundary = np.asarray(touches_boundary, dtype=bool)
    _validate_nodes(node_ids, positions, volumes, touches_boundary, boundary_faces)
    count = len(node_ids)

    if count == 0:
        return SpatialGraph(
            frame=int(frame),
            node_ids=node_ids,
            positions_zyx_um=positions.reshape(0, 3),
            volumes=volumes,
            touches_boundary=touches_boundary,
            boundary_faces=boundary_faces,
            edge_source_indices=np.empty(0, dtype=np.int32),
            edge_target_indices=np.empty(0, dtype=np.int32),
            edge_vectors_zyx_um=np.empty((0, 3), dtype=np.float64),
            edge_distances_um=np.empty(0, dtype=np.float64),
            adjacency_indptr=np.zeros(1, dtype=np.int32),
            adjacency_edge_indices=np.empty(0, dtype=np.int32),
        )

    tree = cKDTree(positions)
    selected_by_node: list[list[int]] = []
    for index, position in enumerate(positions):
        candidates = tree.query_ball_point(position, r=config.maximum_radius_um)
        candidates = [int(other) for other in candidates if int(other) != index]
        candidates.sort(key=lambda other: (float(np.linalg.norm(positions[other] - position)), int(node_ids[other])))
        selected_by_node.append(candidates[: config.maximum_neighbors])

    edges: set[tuple[int, int]] = set()
    for source, candidates in enumerate(selected_by_node):
        for target in candidates:
            if config.prefer_mutual_neighbors and source not in selected_by_node[target]:
                continue
            a, b = sorted((source, target))
            edges.add((a, b))

    ordered_edges = sorted(edges, key=lambda pair: (int(node_ids[pair[0]]), int(node_ids[pair[1]])))
    sources = np.asarray([pair[0] for pair in ordered_edges], dtype=np.int32)
    targets = np.asarray([pair[1] for pair in ordered_edges], dtype=np.int32)
    vectors = positions[targets] - positions[sources] if ordered_edges else np.empty((0, 3), dtype=float)
    distances = np.linalg.norm(vectors, axis=1) if ordered_edges else np.empty(0, dtype=float)

    adjacency: dict[int, list[int]] = defaultdict(list)
    for edge_index, (source, target) in enumerate(ordered_edges):
        adjacency[source].append(edge_index)
        adjacency[target].append(edge_index)
    adjacency_lists = [sorted(adjacency[index]) for index in range(count)]
    indptr = np.zeros(count + 1, dtype=np.int32)
    for index, values in enumerate(adjacency_lists):
        indptr[index + 1] = indptr[index] + len(values)
    edge_indices = np.asarray([edge for values in adjacency_lists for edge in values], dtype=np.int32)

    return SpatialGraph(
        frame=int(frame),
        node_ids=node_ids,
        positions_zyx_um=positions,
        volumes=volumes,
        touches_boundary=touches_boundary,
        boundary_faces=boundary_faces,
        edge_source_indices=sources,
        edge_target_indices=targets,
        edge_vectors_zyx_um=np.asarray(vectors, dtype=np.float64),
        edge_distances_um=np.asarray(distances, dtype=np.float64),
        adjacency_indptr=indptr,
        adjacency_edge_indices=edge_indices,
    )


def neighbor_indices(graph: SpatialGraph, node_index: int) -> np.ndarray:
    if not 0 <= node_index < graph.node_count:
        raise IndexError(f"node index {node_index} is outside graph")
    start = int(graph.adjacency_indptr[node_index])
    stop = int(graph.adjacency_indptr[node_index + 1])
    result: list[int] = []
    for edge_index in graph.adjacency_edge_indices[start:stop]:
        source = int(graph.edge_source_indices[edge_index])
        target = int(graph.edge_target_indices[edge_index])
        result.append(target if source == node_index else source)
    result.sort(key=lambda index: (float(np.linalg.norm(graph.positions_zyx_um[index] - graph.positions_zyx_um[node_index])), int(graph.node_ids[index])))
    return np.asarray(result, dtype=np.int32)


def relative_vector(graph: SpatialGraph, source_index: int, target_index: int) -> np.ndarray:
    return graph.positions_zyx_um[target_index] - graph.positions_zyx_um[source_index]
