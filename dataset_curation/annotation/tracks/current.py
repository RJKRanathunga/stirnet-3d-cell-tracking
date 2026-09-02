from __future__ import annotations

"""Canonical corrected-track materialization for annotation visualization."""

from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dataset_curation.annotation.tracks.diagnostics import EndpointTrackGroups
from dataset_curation.annotation.tracks.graph import Edge, Node
from dataset_curation.annotation.tracks.storage import _atomic_csv


_TRACK_COLUMNS = (
    "track_id",
    "frame",
    "cell_id",
    "z",
    "y",
    "x",
)


def _continuation_components(
    valid_nodes: set[Node],
    active_edges: set[Edge],
) -> list[set[Node]]:
    """
    Build maximal non-branching corrected tracklets.

    A manual Continue is an ordinary edge, so it merges the two tracklets.
    Branch/division edges split tracklets at the parent/daughter boundary.
    """
    incoming: dict[Node, set[Node]] = defaultdict(set)
    outgoing: dict[Node, set[Node]] = defaultdict(set)

    for left, right in active_edges:
        outgoing[left].add(right)
        incoming[right].add(left)

    continuation_edges = {
        (left, right)
        for left, right in active_edges
        if len(outgoing[left]) == 1
        and len(incoming[right]) == 1
    }

    adjacency: dict[Node, set[Node]] = defaultdict(set)
    for left, right in continuation_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    remaining = set(valid_nodes)
    components: list[set[Node]] = []

    while remaining:
        seed = min(remaining)
        visited = {seed}
        queue: deque[Node] = deque([seed])

        while queue:
            node = queue.popleft()
            for neighbour in adjacency.get(node, ()):
                if neighbour in remaining and neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)

        remaining.difference_update(visited)
        components.append(visited)

    components.sort(key=lambda component: min(component))
    return components


def build_current_track_table(
    *,
    valid_nodes: Iterable[Node],
    active_edges: Iterable[Edge],
    centers: dict[Node, np.ndarray],
) -> pd.DataFrame:
    """
    Materialize the current corrected graph as deterministic Napari tracklets.

    The table is annotation-owned. It is rebuilt after graph/spatial edits and
    is never written back into the immutable preprocessed Trackastra output.
    """
    nodes = {
        (int(node[0]), int(node[1]))
        for node in valid_nodes
    }
    edges = {
        (
            (int(edge[0][0]), int(edge[0][1])),
            (int(edge[1][0]), int(edge[1][1])),
        )
        for edge in active_edges
    }

    rows: list[dict[str, float | int]] = []
    components = _continuation_components(nodes, edges)

    for track_id, component in enumerate(components, start=1):
        for node in sorted(component):
            center = centers.get(node)
            if center is None:
                continue
            point = np.asarray(center, dtype=np.float64)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                continue

            rows.append(
                {
                    "track_id": int(track_id),
                    "frame": int(node[0]),
                    "cell_id": int(node[1]),
                    "z": float(point[0]),
                    "y": float(point[1]),
                    "x": float(point[2]),
                }
            )

    return pd.DataFrame(rows, columns=_TRACK_COLUMNS)


def persist_current_track_table(
    path: Path,
    tracks: pd.DataFrame,
) -> None:
    _atomic_csv(
        Path(path),
        tracks.loc[:, list(_TRACK_COLUMNS)].copy(),
    )


def _endpoint_row(
    group: pd.DataFrame,
    *,
    first: bool,
) -> pd.Series:
    ordered = group.sort_values(
        ["frame", "cell_id"],
        kind="stable",
    )
    return ordered.iloc[0 if first else -1]


def prepare_current_endpoint_groups(
    tracks: pd.DataFrame,
    *,
    unresolved_start_nodes: set[Node],
    unresolved_end_nodes: set[Node],
    boundary_entry_nodes: set[Node],
    boundary_exit_nodes: set[Node],
    hidden_nodes: set[Node],
    ignored_start_nodes: set[Node] | None = None,
    ignored_end_nodes: set[Node] | None = None,
) -> EndpointTrackGroups:
    """
    Diagnose the CURRENT corrected tracklets, not the original Trackastra CSV.

    Therefore a Continue behaves exactly like a normal stored connection:
    the two former tracklets merge. If another break exists later, the merged
    tracklet is what appears in Broken Tracks. If no break remains and the
    component is complete, it disappears into Hidden tracks.
    """
    ignored_start_nodes = set(
        ignored_start_nodes or ()
    )
    ignored_end_nodes = set(
        ignored_end_nodes or ()
    )

    if tracks.empty:
        empty = tracks.copy()
        return EndpointTrackGroups(
            track_summary=pd.DataFrame(
                columns=["track_id", "first_frame", "last_frame"]
            ),
            new_track_endpoints=empty.copy(),
            ended_track_endpoints=empty.copy(),
            new_failure_tracks=empty.copy(),
            ended_failure_tracks=empty.copy(),
            boundary_entry_tracks=empty.copy(),
            boundary_exit_tracks=empty.copy(),
        )

    summaries: list[dict[str, int]] = []
    new_endpoint_rows: list[pd.Series] = []
    end_endpoint_rows: list[pd.Series] = []
    new_failure_ids: list[int] = []
    broken_ids: list[int] = []
    boundary_entry_ids: list[int] = []
    boundary_exit_ids: list[int] = []

    for raw_track_id, group in tracks.groupby("track_id", sort=True):
        track_id = int(raw_track_id)
        start = _endpoint_row(group, first=True)
        end = _endpoint_row(group, first=False)

        start_node = (
            int(start["frame"]),
            int(start["cell_id"]),
        )
        end_node = (
            int(end["frame"]),
            int(end["cell_id"]),
        )

        summaries.append(
            {
                "track_id": track_id,
                "first_frame": int(start_node[0]),
                "last_frame": int(end_node[0]),
            }
        )
        new_endpoint_rows.append(start)
        end_endpoint_rows.append(end)

        # Completed components are represented only by Hidden tracks.
        if start_node in hidden_nodes and end_node in hidden_nodes:
            continue

        if (
            start_node in unresolved_start_nodes
            and start_node not in ignored_start_nodes
        ):
            new_failure_ids.append(track_id)
        elif start_node in boundary_entry_nodes:
            boundary_entry_ids.append(track_id)

        if (
            end_node in unresolved_end_nodes
            and end_node not in ignored_end_nodes
        ):
            broken_ids.append(track_id)
        elif end_node in boundary_exit_nodes:
            boundary_exit_ids.append(track_id)

    def rows_to_frame(rows: list[pd.Series]) -> pd.DataFrame:
        if not rows:
            return tracks.iloc[0:0].copy()
        return pd.DataFrame(rows).reset_index(drop=True)

    def selected(track_ids: list[int]) -> pd.DataFrame:
        if not track_ids:
            return tracks.iloc[0:0].copy()
        return tracks[
            tracks["track_id"].isin(track_ids)
        ].copy()

    return EndpointTrackGroups(
        track_summary=pd.DataFrame(
            summaries,
            columns=["track_id", "first_frame", "last_frame"],
        ),
        new_track_endpoints=rows_to_frame(new_endpoint_rows),
        ended_track_endpoints=rows_to_frame(end_endpoint_rows),
        new_failure_tracks=selected(new_failure_ids),
        ended_failure_tracks=selected(broken_ids),
        boundary_entry_tracks=selected(boundary_entry_ids),
        boundary_exit_tracks=selected(boundary_exit_ids),
    )

def filter_diagnostic_rows_for_frame(
    frame: pd.DataFrame,
    *,
    category: str,
    current_frame: int,
    horizon_frames: int,
) -> pd.DataFrame:
    """Return only diagnostic rows relevant to the current annotation frame."""
    if frame.empty:
        return frame.copy()

    category = str(category)
    current_frame = int(current_frame)
    horizon_frames = max(int(horizon_frames), 0)

    if category not in {"broken", "new"}:
        lower = current_frame - horizon_frames
        return frame.loc[
            (frame["frame"] >= lower)
            & (frame["frame"] <= current_frame)
        ].copy()

    endpoint_by_track = (
        frame.groupby("track_id")["frame"].max()
        if category == "broken"
        else frame.groupby("track_id")["frame"].min()
    )

    if category == "broken":
        active_ids = endpoint_by_track.index[
            (endpoint_by_track >= current_frame)
            & (
                endpoint_by_track
                <= current_frame + horizon_frames
            )
        ]
    else:
        active_ids = endpoint_by_track.index[
            (endpoint_by_track <= current_frame)
            & (
                endpoint_by_track
                >= current_frame - horizon_frames
            )
        ]

    if len(active_ids) == 0:
        return frame.iloc[0:0].copy()

    lower = current_frame - horizon_frames
    return frame.loc[
        frame["track_id"].isin(active_ids)
        & (frame["frame"] >= lower)
        & (frame["frame"] <= current_frame)
    ].copy()

def diagnostic_events_for_node(
    groups: EndpointTrackGroups,
    node: Node,
    *,
    current_frame: int | None = None,
    horizon_frames: int | None = None,
) -> list[tuple[str, Node]]:
    node = (
        int(node[0]),
        int(node[1]),
    )

    result: list[tuple[str, Node]] = []

    for category, full_frame, endpoint_kind in (
        ("new", groups.new_failure_tracks, "start"),
        ("broken", groups.ended_failure_tracks, "end"),
    ):
        if full_frame.empty:
            continue

        visible_frame = full_frame
        if (
            current_frame is not None
            and horizon_frames is not None
        ):
            visible_frame = filter_diagnostic_rows_for_frame(
                full_frame,
                category=category,
                current_frame=int(current_frame),
                horizon_frames=int(horizon_frames),
            )

        if visible_frame.empty:
            continue

        matches = visible_frame.loc[
            (visible_frame["frame"].astype(int) == node[0])
            & (visible_frame["cell_id"].astype(int) == node[1])
        ]
        if matches.empty:
            continue

        for raw_track_id in matches["track_id"].unique().tolist():
            track = full_frame.loc[
                full_frame["track_id"] == raw_track_id
            ]
            if track.empty:
                continue

            ordered = track.sort_values(
                ["frame", "cell_id"],
                kind="stable",
            )
            row = (
                ordered.iloc[0]
                if endpoint_kind == "start"
                else ordered.iloc[-1]
            )
            result.append(
                (
                    category,
                    (
                        int(row["frame"]),
                        int(row["cell_id"]),
                    ),
                )
            )

    return list(dict.fromkeys(result))
