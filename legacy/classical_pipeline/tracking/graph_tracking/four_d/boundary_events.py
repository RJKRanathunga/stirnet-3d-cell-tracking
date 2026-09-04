"""Face-specific boundary entry/exit alternatives and costs."""

from __future__ import annotations

import math

import numpy as np

from ..geometry import FACE_NAMES, FACE_NORMALS
from .config import FourDGraphConfig
from .types import ObservationStore, TemporalEdge


def _direction_support(
    node: int,
    face: str,
    edges: list[TemporalEdge],
    observations: ObservationStore,
    *,
    entering: bool,
) -> float:
    normal = FACE_NORMALS[face]
    values: list[float] = []
    for edge in edges:
        if entering and edge.source == node:
            displacement = observations.positions_zyx_um[edge.target] - observations.positions_zyx_um[node]
            values.append(float(np.dot(displacement, normal) < 0.0))
        elif not entering and edge.target == node:
            displacement = observations.positions_zyx_um[node] - observations.positions_zyx_um[edge.source]
            values.append(float(np.dot(displacement, normal) > 0.0))
    return float(np.mean(values)) if values else 0.5


def start_event_options(
    node: int,
    *,
    edges: list[TemporalEdge],
    observations: ObservationStore,
    config: FourDGraphConfig,
    first_frame: int | None = None,
) -> list[tuple[str, str, float, float]]:
    frame = int(observations.frames[node])
    if first_frame is None:
        first_frame = int(observations.frames.min()) if observations.node_count else frame
    result = [("birth", "", float(config.generic_birth_cost), math.nan)]
    if frame == first_frame:
        result.append(("sequence_start", "", 0.0, math.nan))
    if config.enable_boundary_events:
        for face_index, face in enumerate(FACE_NAMES):
            distance = float(observations.distances_to_faces_um[node, face_index])
            if distance < -1e-9 or distance > config.boundary_margin_um:
                continue
            support = _direction_support(
                node, face, edges, observations, entering=True
            )
            cost = (
                config.boundary_event_base_cost
                + distance / max(config.boundary_margin_um, 1e-9)
                + config.boundary_direction_weight * (1.0 - support)
            )
            result.append(("boundary_entry", face, float(cost), support))
    return result


def end_event_options(
    node: int,
    *,
    edges: list[TemporalEdge],
    observations: ObservationStore,
    config: FourDGraphConfig,
    last_frame: int | None = None,
) -> list[tuple[str, str, float, float]]:
    frame = int(observations.frames[node])
    if last_frame is None:
        last_frame = int(observations.frames.max()) if observations.node_count else frame
    result = [("death", "", float(config.generic_death_cost), math.nan)]
    if frame == last_frame:
        result.append(("sequence_end", "", 0.0, math.nan))
    if config.enable_boundary_events:
        for face_index, face in enumerate(FACE_NAMES):
            distance = float(observations.distances_to_faces_um[node, face_index])
            if distance < -1e-9 or distance > config.boundary_margin_um:
                continue
            support = _direction_support(
                node, face, edges, observations, entering=False
            )
            cost = (
                config.boundary_event_base_cost
                + distance / max(config.boundary_margin_um, 1e-9)
                + config.boundary_direction_weight * (1.0 - support)
            )
            result.append(("boundary_exit", face, float(cost), support))
    return result
