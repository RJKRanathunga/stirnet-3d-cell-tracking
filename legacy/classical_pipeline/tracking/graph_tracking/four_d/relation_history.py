"""Persistent neighbour-identity and relative-vector histories."""

from __future__ import annotations

import math

import numpy as np

from .config import FourDGraphConfig
from .types import NeighborRelationHistory, ObservationStore, SpatialFrameGraph


def build_relation_histories(
    *,
    observations: ObservationStore,
    spatial_graphs: dict[int, SpatialFrameGraph],
    config: FourDGraphConfig,
) -> dict[tuple[int, int], NeighborRelationHistory]:
    records: dict[tuple[int, int], list[tuple[int, np.ndarray, float, float, np.ndarray, float]]] = {}
    for frame, graph in sorted(spatial_graphs.items()):
        for edge_index, (source, target) in enumerate(zip(graph.edge_sources, graph.edge_targets)):
            first = int(observations.provisional_track_ids[source])
            second = int(observations.provisional_track_ids[target])
            if first < 0 or second < 0 or first == second:
                continue
            if first < second:
                key = (first, second)
                vector = observations.positions_zyx_um[target] - observations.positions_zyx_um[source]
                volume = math.log(max(observations.volumes[target], 1e-9) / max(observations.volumes[source], 1e-9))
            else:
                key = (second, first)
                vector = observations.positions_zyx_um[source] - observations.positions_zyx_um[target]
                volume = math.log(max(observations.volumes[source], 1e-9) / max(observations.volumes[target], 1e-9))
            records.setdefault(key, []).append((
                frame, vector, float(np.linalg.norm(vector)), volume,
                graph.visibility_masks[edge_index],
                float(graph.boundary_observability[edge_index]),
            ))

    result: dict[tuple[int, int], NeighborRelationHistory] = {}
    for key in sorted(records):
        rows = sorted(records[key], key=lambda value: value[0])[-config.relation_history_frames:]
        frames = np.asarray([value[0] for value in rows], dtype=np.int32)
        vectors = np.asarray([value[1] for value in rows], dtype=float)
        distances = np.asarray([value[2] for value in rows], dtype=float)
        volumes = np.asarray([value[3] for value in rows], dtype=float)
        visibility = np.asarray([value[4] for value in rows], dtype=bool)
        coverage = np.asarray([value[5] for value in rows], dtype=float)
        median = np.median(vectors, axis=0)
        dispersion = float(np.median(np.linalg.norm(vectors - median, axis=1)))
        persistence = len(rows)
        persistence_score = min(1.0, persistence / config.relation_history_frames)
        smoothness = math.exp(-dispersion / max(config.relation_vector_scale_um, 1e-9))
        confidence = float(np.clip(
            persistence_score * smoothness * max(float(np.mean(coverage)), 0.15), 0.0, 1.0
        ))
        if persistence < config.minimum_persistent_relation_frames:
            confidence *= 0.25
        result[key] = NeighborRelationHistory(
            provisional_track_a=key[0],
            provisional_track_b=key[1],
            observed_frames=frames,
            relative_vectors_zyx_um=vectors,
            distances_um=distances,
            relative_log_volume_ratios=volumes,
            visibility_masks=visibility,
            boundary_coverage=coverage,
            persistence_count=persistence,
            vector_median_zyx_um=median,
            robust_vector_dispersion_um=dispersion,
            confidence=confidence,
        )
    return result
