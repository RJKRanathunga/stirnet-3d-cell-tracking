from __future__ import annotations

"""Track node/edge semantics and Trackastra detection-edge construction."""

from typing import Any, Iterable

import pandas as pd

Node = tuple[int, int]  # (frame, spatial_instance_id)

Edge = tuple[Node, Node]  # directed earlier -> later

class AnnotationError(RuntimeError):
    pass

def _canonical_edge(a: Node, b: Node) -> Edge:
    if a == b:
        raise AnnotationError("A track edge cannot connect a detection to itself.")
    if a[0] == b[0]:
        raise AnnotationError(
            "Track edges must connect different frames; both selections are at "
            f"t={a[0]}."
        )
    return (a, b) if a[0] < b[0] else (b, a)

def _track_nodes(tracks: pd.DataFrame) -> dict[int, list[Node]]:
    by_track: dict[int, list[Node]] = {}
    for track_id, group in tracks.groupby("track_id", sort=False):
        ordered = group.sort_values("frame")
        nodes: list[Node] = []
        seen: set[Node] = set()
        for row in ordered.itertuples(index=False):
            node = (int(row.frame), int(row.cell_id))
            if node not in seen:
                nodes.append(node)
                seen.add(node)
        by_track[int(track_id)] = nodes
    return by_track

def build_trackastra_detection_edges(
    tracks: pd.DataFrame,
    lineage_payload: dict[str, Any],
) -> set[Edge]:
    """Convert Trackastra tracklets + child->parent lineage into detection edges."""
    by_track = _track_nodes(tracks)
    edges: set[Edge] = set()

    # Consecutive detections within each Trackastra tracklet.
    for nodes in by_track.values():
        for left, right in zip(nodes[:-1], nodes[1:]):
            if left[0] != right[0]:
                edges.add(_canonical_edge(left, right))

    # graph_to_napari_tracks() stores child_track_id -> parent_track_id.
    for child_text, parents_raw in lineage_payload.items():
        child_id = int(child_text)
        parents = (
            list(parents_raw)
            if isinstance(parents_raw, (list, tuple, set))
            else [parents_raw]
        )
        child_nodes = by_track.get(child_id, [])
        if not child_nodes:
            continue
        child_first = min(child_nodes, key=lambda node: node[0])

        for parent_raw in parents:
            parent_id = int(parent_raw)
            parent_nodes = by_track.get(parent_id, [])
            earlier = [node for node in parent_nodes if node[0] < child_first[0]]
            if not earlier:
                continue
            parent_last = max(earlier, key=lambda node: node[0])
            edges.add(_canonical_edge(parent_last, child_first))

    return edges
