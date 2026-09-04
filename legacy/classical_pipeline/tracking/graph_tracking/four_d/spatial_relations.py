"""Sparse physical neighbour graphs for every observed 3D frame."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .config import FourDGraphConfig
from .types import ObservationStore, SpatialFrameGraph


def build_spatial_relations(
    observations: ObservationStore,
    config: FourDGraphConfig,
) -> dict[int, SpatialFrameGraph]:
    result: dict[int, SpatialFrameGraph] = {}
    for frame in sorted(observations.nodes_by_frame):
        nodes = observations.nodes_by_frame[frame]
        positions = observations.positions_zyx_um[nodes]
        selected: list[list[int]] = []
        if len(nodes):
            tree = cKDTree(positions)
            for local_index, position in enumerate(positions):
                candidates = [
                    int(value) for value in tree.query_ball_point(
                        position, config.spatial_maximum_radius_um
                    ) if int(value) != local_index
                ]
                candidates.sort(key=lambda value: (
                    float(np.linalg.norm(positions[value] - position)),
                    int(nodes[value]),
                ))
                selected.append(candidates[: config.spatial_maximum_neighbors])
        pairs = sorted({
            tuple(sorted((int(nodes[source]), int(nodes[target]))))
            for source, candidates in enumerate(selected)
            for target in candidates
        })
        sources = np.asarray([pair[0] for pair in pairs], dtype=np.int32)
        targets = np.asarray([pair[1] for pair in pairs], dtype=np.int32)
        vectors = (
            observations.positions_zyx_um[targets] - observations.positions_zyx_um[sources]
            if pairs else np.empty((0, 3), dtype=float)
        )
        distances = np.linalg.norm(vectors, axis=1) if pairs else np.empty(0)
        units = np.divide(
            vectors,
            distances[:, None],
            out=np.zeros_like(vectors),
            where=distances[:, None] > 0,
        )
        relative_volume = (
            np.log(np.maximum(observations.volumes[targets], 1e-9))
            - np.log(np.maximum(observations.volumes[sources], 1e-9))
            if pairs else np.empty(0)
        )
        visibility = (
            (observations.distances_to_faces_um[sources] > config.spatial_maximum_radius_um)
            & (observations.distances_to_faces_um[targets] > config.spatial_maximum_radius_um)
            if pairs else np.empty((0, 6), dtype=bool)
        )
        observability = visibility.mean(axis=1) if pairs else np.empty(0)

        adjacency: dict[int, list[int]] = defaultdict(list)
        local_by_global = {int(node): local for local, node in enumerate(nodes)}
        for edge_index, (source, target) in enumerate(pairs):
            adjacency[local_by_global[source]].append(edge_index)
            adjacency[local_by_global[target]].append(edge_index)
        indptr = np.zeros(len(nodes) + 1, dtype=np.int32)
        flat: list[int] = []
        for local in range(len(nodes)):
            values = sorted(adjacency[local])
            flat.extend(values)
            indptr[local + 1] = len(flat)
        result[frame] = SpatialFrameGraph(
            frame=frame,
            node_indices=nodes.copy(),
            edge_sources=sources,
            edge_targets=targets,
            relative_vectors_zyx_um=np.asarray(vectors, dtype=float),
            distances_um=np.asarray(distances, dtype=float),
            unit_directions=np.asarray(units, dtype=float),
            relative_log_volumes=np.asarray(relative_volume, dtype=float),
            boundary_observability=np.asarray(observability, dtype=float),
            visibility_masks=np.asarray(visibility, dtype=bool),
            adjacency_indptr=indptr,
            adjacency_edge_indices=np.asarray(flat, dtype=np.int32),
        )
    return result


def neighbor_nodes(graph: SpatialFrameGraph, node: int) -> np.ndarray:
    local_matches = np.flatnonzero(graph.node_indices == int(node))
    if not len(local_matches):
        return np.empty(0, dtype=np.int32)
    local = int(local_matches[0])
    result: list[int] = []
    for edge_index in graph.adjacency_edge_indices[
        graph.adjacency_indptr[local]:graph.adjacency_indptr[local + 1]
    ]:
        source = int(graph.edge_sources[edge_index])
        target = int(graph.edge_targets[edge_index])
        result.append(target if source == node else source)
    return np.asarray(result, dtype=np.int32)
