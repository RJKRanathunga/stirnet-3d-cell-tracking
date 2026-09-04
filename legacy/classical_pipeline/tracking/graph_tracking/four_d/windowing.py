"""Deterministic overlapping temporal windows and central commit ranges."""

from __future__ import annotations

import numpy as np

from .config import FourDGraphConfig
from .types import AmbiguityComponent, ObservationStore, TemporalEdge, TemporalWindow


def build_windows(
    component: AmbiguityComponent,
    *,
    observations: ObservationStore,
    edges: list[TemporalEdge],
    config: FourDGraphConfig,
) -> list[TemporalWindow]:
    if component.maximum_frame - component.minimum_frame + 1 <= config.window_size:
        return [TemporalWindow(
            window_id=0,
            component_id=component.component_id,
            start_frame=component.minimum_frame,
            end_frame=component.maximum_frame,
            commit_start_frame=component.minimum_frame,
            commit_end_frame=component.maximum_frame,
            node_indices=component.node_indices.copy(),
            edge_indices=component.edge_indices.copy(),
        )]

    result: list[TemporalWindow] = []
    for center in range(component.minimum_frame, component.maximum_frame + 1):
        start = max(component.minimum_frame, center - config.lookback_frames)
        end = min(component.maximum_frame, center + config.lookahead_frames)
        if start == component.minimum_frame:
            end = min(component.maximum_frame, start + config.window_size - 1)
        if end == component.maximum_frame:
            start = max(component.minimum_frame, end - config.window_size + 1)
        node_mask = (
            (observations.frames[component.node_indices] >= start)
            & (observations.frames[component.node_indices] <= end)
        )
        nodes = component.node_indices[node_mask]
        node_set = set(int(value) for value in nodes)
        edge_indices = np.asarray([
            index for index in component.edge_indices
            if edges[int(index)].source in node_set and edges[int(index)].target in node_set
        ], dtype=np.int32)
        result.append(TemporalWindow(
            window_id=len(result),
            component_id=component.component_id,
            start_frame=start,
            end_frame=end,
            commit_start_frame=center,
            commit_end_frame=center,
            node_indices=nodes,
            edge_indices=edge_indices,
        ))
    return result
