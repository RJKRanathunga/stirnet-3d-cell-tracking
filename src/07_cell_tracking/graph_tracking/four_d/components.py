"""Bounded multi-frame ambiguity component construction."""

from __future__ import annotations

import numpy as np

from .config import FourDGraphConfig
from .spatial_relations import neighbor_nodes
from .types import AmbiguityComponent, ObservationStore, SpatialFrameGraph, TemporalEdge


def build_ambiguity_components(
    *,
    seeds: set[int],
    observations: ObservationStore,
    edges: list[TemporalEdge],
    spatial_graphs: dict[int, SpatialFrameGraph],
    config: FourDGraphConfig,
) -> list[AmbiguityComponent]:
    temporal: dict[int, set[int]] = {}
    for edge in edges:
        temporal.setdefault(edge.source, set()).add(edge.target)
        temporal.setdefault(edge.target, set()).add(edge.source)

    regions: list[set[int]] = []
    for seed in sorted(seeds):
        region = {seed}
        frontier = {seed}
        for _ in range(config.ambiguity_temporal_hops):
            expanded = {neighbor for node in frontier for neighbor in temporal.get(node, ())}
            frontier = expanded - region
            region |= expanded
        for _ in range(config.ambiguity_spatial_hops):
            expanded: set[int] = set()
            for node in tuple(region):
                graph = spatial_graphs[int(observations.frames[node])]
                expanded.update(int(value) for value in neighbor_nodes(graph, node))
            region |= expanded
        regions.append(region)

    merged: list[set[int]] = []
    for region in regions:
        overlaps = [index for index, current in enumerate(merged) if current & region]
        if not overlaps:
            merged.append(set(region))
            continue
        combined = set(region)
        for index in reversed(overlaps):
            combined |= merged.pop(index)
        merged.append(combined)

    # A provisional edge is a boundary condition only when entirely outside a
    # component. Pull its other endpoint into a region to avoid half-rewriting it.
    selected = [edge for edge in edges if edge.provisional_selected]
    changed = True
    while changed:
        changed = False
        for region in merged:
            for edge in selected:
                if (edge.source in region) != (edge.target in region):
                    region.update((edge.source, edge.target))
                    changed = True
        # Closure can make independently grown regions overlap.
        next_merged: list[set[int]] = []
        for region in merged:
            overlaps = [index for index, current in enumerate(next_merged) if current & region]
            if not overlaps:
                next_merged.append(set(region))
            else:
                combined = set(region)
                for index in reversed(overlaps):
                    combined |= next_merged.pop(index)
                next_merged.append(combined)
        if len(next_merged) != len(merged):
            changed = True
        merged = next_merged

    ordered = sorted(merged, key=lambda values: (
        min(int(observations.frames[node]) for node in values), min(values)
    ))
    result: list[AmbiguityComponent] = []
    for component_id, nodes_set in enumerate(ordered):
        nodes = np.asarray(sorted(nodes_set), dtype=np.int32)
        edge_indices = np.asarray([
            edge.edge_index for edge in edges
            if edge.source in nodes_set and edge.target in nodes_set
        ], dtype=np.int32)
        component_seeds = np.asarray(sorted(nodes_set & seeds), dtype=np.int32)
        result.append(AmbiguityComponent(
            component_id=component_id,
            node_indices=nodes,
            edge_indices=edge_indices,
            seed_nodes=component_seeds,
            minimum_frame=int(observations.frames[nodes].min()),
            maximum_frame=int(observations.frames[nodes].max()),
        ))
    return result

