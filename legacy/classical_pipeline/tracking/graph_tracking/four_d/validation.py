"""Strict graph and optimized Stage 7 output validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..geometry import FACE_NAMES
from .config import FourDGraphConfig
from .types import ObservationStore, TemporalEdge


REQUIRED_TRACK_COLUMNS = (
    "track_id", "frame", "cell", "cell_id", "z", "y", "x", "volume",
)


def validate_optimized_solution(
    *,
    observations: ObservationStore,
    edges: list[TemporalEdge],
    selected_edge_indices: set[int],
    optimized_tracks: pd.DataFrame,
    node_to_track_id: np.ndarray,
    boundary_events: pd.DataFrame,
    track_id_map: pd.DataFrame,
    config: FourDGraphConfig,
) -> dict[str, object]:
    if len(node_to_track_id) != observations.node_count or np.any(node_to_track_id < 0):
        raise ValueError("Every observation must have exactly one optimized track ID")
    if len(optimized_tracks) != observations.node_count:
        raise ValueError("Optimized tracks do not contain exactly one row per observation")
    missing_columns = [column for column in REQUIRED_TRACK_COLUMNS if column not in optimized_tracks]
    if missing_columns:
        raise ValueError(f"Optimized tracks are missing downstream columns: {missing_columns}")
    if optimized_tracks[list(REQUIRED_TRACK_COLUMNS[:4])].isna().any().any():
        raise ValueError("Optimized tracks contain NaN identifiers")
    duplicate_observations = optimized_tracks.duplicated(["frame", "cell"], keep=False)
    if duplicate_observations.any():
        raise ValueError("An observation appears more than once in optimized tracks")
    duplicate_track_frames = optimized_tracks.duplicated(["track_id", "frame"], keep=False)
    if duplicate_track_frames.any():
        raise ValueError("An optimized track has two observations in the same frame")

    incoming: dict[int, int] = {}
    outgoing: dict[int, int] = {}
    for edge_index in sorted(selected_edge_indices):
        if not 0 <= edge_index < len(edges):
            raise ValueError(f"Selected edge index {edge_index} is invalid")
        edge = edges[edge_index]
        if not edge.hard_safety_valid:
            raise ValueError(f"Safety-invalid edge {edge_index} was selected")
        if not 1 <= edge.frame_gap <= config.maximum_gap_frames:
            raise ValueError(f"Selected edge {edge_index} exceeds the configured gap")
        if observations.frames[edge.target] <= observations.frames[edge.source]:
            raise ValueError(f"Selected edge {edge_index} is cyclic or non-forward")
        incoming[edge.target] = incoming.get(edge.target, 0) + 1
        outgoing[edge.source] = outgoing.get(edge.source, 0) + 1
    if any(value > 1 for value in incoming.values()):
        raise ValueError("A node has multiple selected predecessors")
    if any(value > 1 for value in outgoing.values()):
        raise ValueError("A node has multiple selected successors")

    if not boundary_events.empty:
        invalid_faces = set(boundary_events["boundary_face"].dropna().astype(str)) - set(FACE_NAMES) - {""}
        if invalid_faces:
            raise ValueError(f"Boundary events contain invalid faces: {sorted(invalid_faces)}")
    if not track_id_map.empty and track_id_map[
        ["provisional_track_id", "optimized_track_id"]
    ].isna().any().any():
        raise ValueError("Track ID mapping contains NaN identifiers")

    starts = []
    for track_id, group in optimized_tracks.groupby("track_id", sort=True):
        first = group.sort_values(["frame", "cell"], kind="mergesort").iloc[0]
        starts.append((int(first["frame"]), int(first["cell"]), int(track_id)))
    ordered = sorted(starts)
    if [value[2] for value in ordered] != list(range(len(ordered))):
        raise ValueError("Optimized track IDs are not deterministic start-order IDs")
    return {
        "valid": True,
        "observation_count": observations.node_count,
        "track_count": int(optimized_tracks["track_id"].nunique()),
        "selected_edge_count": len(selected_edge_indices),
        "selected_gap_edge_count": sum(edges[index].frame_gap > 1 for index in selected_edge_indices),
    }
