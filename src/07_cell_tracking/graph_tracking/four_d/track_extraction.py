"""Deterministic one-to-one path extraction and provisional ID mapping."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .diagnostics import empty_track_id_map
from .schemas import TRACK_ID_MAP_COLUMNS
from .types import ExtractedTracks, ObservationStore, TemporalEdge


def extract_tracks(
    *,
    observations: ObservationStore,
    edges: list[TemporalEdge],
    selected_edge_indices: set[int],
    provisional_tracks: pd.DataFrame,
) -> ExtractedTracks:
    predecessor: dict[int, int] = {}
    successor: dict[int, int] = {}
    selected_edge_by_target: dict[int, TemporalEdge] = {}
    for edge_index in sorted(selected_edge_indices):
        edge = edges[edge_index]
        if edge.target in predecessor:
            raise ValueError(f"Observation {edge.target} has multiple selected predecessors")
        if edge.source in successor:
            raise ValueError(f"Observation {edge.source} has multiple selected successors")
        predecessor[edge.target] = edge.source
        successor[edge.source] = edge.target
        selected_edge_by_target[edge.target] = edge

    starts = sorted(
        (node for node in range(observations.node_count) if node not in predecessor),
        key=lambda node: (
            int(observations.frames[node]), int(observations.detection_indices[node]), node
        ),
    )
    node_to_track = np.full(observations.node_count, -1, dtype=np.int64)
    for track_id, start in enumerate(starts):
        node = start
        visited: set[int] = set()
        while node not in visited:
            visited.add(node)
            if node_to_track[node] >= 0:
                raise ValueError(f"Observation {node} is shared by extracted tracks")
            node_to_track[node] = track_id
            if node not in successor:
                break
            node = successor[node]
    if np.any(node_to_track < 0):
        missing = np.flatnonzero(node_to_track < 0).tolist()
        raise ValueError(f"Unassigned observations after track extraction: {missing[:20]}")

    provisional_lookup = {
        (int(row["frame"]), row["cell"]): row
        for _, row in provisional_tracks.iterrows()
    }
    records: list[dict[str, object]] = []
    for node in range(observations.node_count):
        observation = observations.table.iloc[node]
        key = (int(observation["frame"]), observation["cell_index"])
        source = provisional_lookup.get(key)
        if source is None:
            record: dict[str, object] = {
                "frame": int(observation["frame"]),
                "cell": observation["cell_index"],
                "cell_id": int(observation["cell_id"]),
                "z": float(observation["centroid_z"]),
                "y": float(observation["centroid_y"]),
                "x": float(observation["centroid_x"]),
                "volume": float(observation["volume_voxels"]),
                "touches_boundary": bool(observation["touches_boundary"]),
                "boundary_faces": str(observation["boundary_faces"]),
            }
        else:
            record = source.to_dict()
        record["track_id"] = int(node_to_track[node])
        edge = selected_edge_by_target.get(node)
        if edge is None:
            record["match_type"] = (
                "initial" if int(observations.frames[node]) == int(observations.frames.min())
                else "graph4d_track_start"
            )
            record["match_distance_um"] = math.nan
            record["match_cost"] = math.nan
            record["association_probability"] = math.nan
            record["probability_margin"] = math.nan
        else:
            record["match_type"] = (
                "graph4d_gap_reacquired" if edge.frame_gap > 1 else "graph4d_continuation"
            )
            record["match_distance_um"] = edge.displacement_um
            record["match_cost"] = edge.total_cost
            record["association_probability"] = edge.provisional_probability
            record["probability_margin"] = edge.provisional_margin
        records.append(record)
    optimized = pd.DataFrame(records)
    optimized = optimized.sort_values(
        ["frame", "track_id", "cell"], kind="mergesort"
    ).reset_index(drop=True)

    map_rows: list[dict[str, int]] = []
    mapping_frame = pd.DataFrame({
        "provisional_track_id": observations.provisional_track_ids,
        "optimized_track_id": node_to_track,
    })
    mapping_frame = mapping_frame.loc[mapping_frame["provisional_track_id"] >= 0]
    provisional_counts = mapping_frame.groupby("provisional_track_id").size()
    optimized_counts = mapping_frame.groupby("optimized_track_id").size()
    for (provisional_id, optimized_id), group in mapping_frame.groupby(
        ["provisional_track_id", "optimized_track_id"], sort=True
    ):
        map_rows.append({
            "provisional_track_id": int(provisional_id),
            "optimized_track_id": int(optimized_id),
            "shared_observations": int(len(group)),
            "provisional_observations": int(provisional_counts.loc[provisional_id]),
            "optimized_observations": int(optimized_counts.loc[optimized_id]),
        })
    id_map = (
        pd.DataFrame(map_rows, columns=TRACK_ID_MAP_COLUMNS)
        if map_rows else empty_track_id_map()
    )
    return ExtractedTracks(
        optimized_tracks=optimized,
        node_to_track_id=node_to_track,
        track_id_map=id_map,
        selected_successor=successor,
        selected_predecessor=predecessor,
    )

