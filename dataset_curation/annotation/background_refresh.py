from __future__ import annotations

"""Background derivation/export for interactive dataset curation."""

from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dataset_curation.annotation.layers import (
    edges_to_tracks_array,
    nodes_to_points_array,
    track_frame_arrays,
)
from dataset_curation.annotation.tracks.current import (
    build_current_track_table,
    persist_current_track_table,
    prepare_current_endpoint_groups,
)
from dataset_curation.annotation.tracks.diagnostics import EndpointTrackGroups
from dataset_curation.annotation.tracks.graph import (
    Edge,
    Node,
    _canonical_edge,
)
from dataset_curation.annotation.tracks.storage import (
    OutputPaths,
    _atomic_csv,
    _parse_node,
)


@dataclass(frozen=True)
class TrackRefreshSnapshot:
    generation: int
    reason: str
    output: OutputPaths
    valid_nodes: frozenset[Node]
    base_edges: frozenset[Edge]
    forced_edges: frozenset[Edge]
    broken_edges: frozenset[Edge]
    boundary_entry_nodes: frozenset[Node]
    boundary_exit_nodes: frozenset[Node]
    frame_count: int
    birth_events: tuple[dict[str, Any], ...]
    ignored_events: tuple[dict[str, Any], ...]
    centers: dict[Node, tuple[float, float, float]]


@dataclass(frozen=True)
class TrackGraphAnalysis:
    active_edges: frozenset[Edge]
    hidden_nodes: frozenset[Node]
    hidden_edges: frozenset[Edge]
    visible_edges: frozenset[Edge]
    unresolved_start_nodes: frozenset[Node]
    unresolved_end_nodes: frozenset[Node]


@dataclass(frozen=True)
class TrackRefreshResult:
    generation: int
    reason: str
    current_tracks: pd.DataFrame
    diagnostics: EndpointTrackGroups
    all_tracks_array: np.ndarray
    all_points_array: np.ndarray
    all_properties: dict[str, np.ndarray]
    active_tracks_array: np.ndarray
    hidden_tracks_array: np.ndarray
    hidden_points_array: np.ndarray
    visible_edge_count: int
    hidden_node_count: int
    unresolved_start_count: int
    unresolved_end_count: int


def capture_track_refresh_snapshot(
    *,
    generation: int,
    reason: str,
    track_session,
    track_centers: dict[Node, np.ndarray],
) -> TrackRefreshSnapshot:
    centers: dict[Node, tuple[float, float, float]] = {}
    for raw_node, raw_center in track_centers.items():
        node = (
            int(raw_node[0]),
            int(raw_node[1]),
        )
        point = np.asarray(
            raw_center,
            dtype=np.float64,
        )
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            continue
        centers[node] = (
            float(point[0]),
            float(point[1]),
            float(point[2]),
        )

    return TrackRefreshSnapshot(
        generation=int(generation),
        reason=str(reason),
        output=track_session.output,
        valid_nodes=frozenset(
            (
                int(node[0]),
                int(node[1]),
            )
            for node in track_session.valid_nodes
        ),
        base_edges=frozenset(
            _canonical_edge(edge[0], edge[1])
            for edge in track_session.base_edges
        ),
        forced_edges=frozenset(
            _canonical_edge(edge[0], edge[1])
            for edge in track_session.forced_edges
        ),
        broken_edges=frozenset(
            _canonical_edge(edge[0], edge[1])
            for edge in track_session.broken_edges
        ),
        boundary_entry_nodes=frozenset(
            (
                int(node[0]),
                int(node[1]),
            )
            for node in track_session.boundary_entry_nodes
        ),
        boundary_exit_nodes=frozenset(
            (
                int(node[0]),
                int(node[1]),
            )
            for node in track_session.boundary_exit_nodes
        ),
        frame_count=int(track_session.frame_count),
        birth_events=tuple(
            deepcopy(track_session.birth_events)
        ),
        ignored_events=tuple(
            deepcopy(track_session.ignored_events)
        ),
        centers=centers,
    )


def analyze_track_snapshot(
    snapshot: TrackRefreshSnapshot,
) -> TrackGraphAnalysis:
    candidate = (
        set(snapshot.base_edges)
        | set(snapshot.forced_edges)
    ) - set(snapshot.broken_edges)

    active_edges: set[Edge] = {
        edge
        for edge in candidate
        if (
            edge[0] in snapshot.valid_nodes
            and edge[1] in snapshot.valid_nodes
        )
    }

    adjacency: dict[Node, set[Node]] = defaultdict(set)
    incoming: set[Node] = set()
    outgoing: set[Node] = set()

    for left, right in active_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
        outgoing.add(left)
        incoming.add(right)

    remaining = set(snapshot.valid_nodes)
    hidden_nodes: set[Node] = set()
    unresolved_starts: set[Node] = set()
    unresolved_ends: set[Node] = set()

    while remaining:
        seed = next(iter(remaining))
        visited: set[Node] = {seed}
        queue: deque[Node] = deque([seed])

        while queue:
            node = queue.popleft()
            for neighbour in adjacency.get(node, ()):
                if (
                    neighbour in remaining
                    and neighbour not in visited
                ):
                    visited.add(neighbour)
                    queue.append(neighbour)

        remaining.difference_update(visited)

        starts = {
            node
            for node in visited
            if node not in incoming
        }
        ends = {
            node
            for node in visited
            if node not in outgoing
        }

        bad_starts = {
            node
            for node in starts
            if not (
                int(node[0]) == 0
                or node in snapshot.boundary_entry_nodes
            )
        }
        bad_ends = {
            node
            for node in ends
            if not (
                int(node[0]) == snapshot.frame_count - 1
                or node in snapshot.boundary_exit_nodes
            )
        }

        unresolved_starts.update(bad_starts)
        unresolved_ends.update(bad_ends)

        if visited and not bad_starts and not bad_ends:
            hidden_nodes.update(visited)

    hidden_edges = {
        edge
        for edge in active_edges
        if (
            edge[0] in hidden_nodes
            and edge[1] in hidden_nodes
        )
    }
    visible_edges = active_edges - hidden_edges

    return TrackGraphAnalysis(
        active_edges=frozenset(active_edges),
        hidden_nodes=frozenset(hidden_nodes),
        hidden_edges=frozenset(hidden_edges),
        visible_edges=frozenset(visible_edges),
        unresolved_start_nodes=frozenset(
            unresolved_starts
        ),
        unresolved_end_nodes=frozenset(
            unresolved_ends
        ),
    )


def _birth_edges(
    snapshot: TrackRefreshSnapshot,
) -> set[Edge]:
    result: set[Edge] = set()
    for event in snapshot.birth_events:
        parent = _parse_node(
            event["parent"]
        )
        for raw_daughter in event.get(
            "daughters",
            [],
        ):
            result.add(
                _canonical_edge(
                    parent,
                    _parse_node(
                        raw_daughter
                    ),
                )
            )
    return result


def _ignored_nodes(
    snapshot: TrackRefreshSnapshot,
    event_type: str,
) -> set[Node]:
    return {
        _parse_node(event["endpoint"])
        for event in snapshot.ignored_events
        if str(event.get("event_type")) == event_type
    }


def export_track_snapshot(
    snapshot: TrackRefreshSnapshot,
    analysis: TrackGraphAnalysis,
    current_tracks: pd.DataFrame,
) -> None:
    persist_current_track_table(
        snapshot.output.current_tracks_csv,
        current_tracks,
    )

    birth_edges = _birth_edges(
        snapshot
    )

    active_rows: list[dict[str, Any]] = []
    for edge in sorted(
        analysis.active_edges
    ):
        if edge in birth_edges:
            origin = "birth"
        elif edge in snapshot.forced_edges:
            origin = "manual_continue"
        else:
            origin = "trackastra"

        active_rows.append(
            {
                "source_frame": int(edge[0][0]),
                "source_cell_id": int(edge[0][1]),
                "target_frame": int(edge[1][0]),
                "target_cell_id": int(edge[1][1]),
                "frame_gap": int(
                    edge[1][0]
                    - edge[0][0]
                ),
                "origin": origin,
            }
        )

    _atomic_csv(
        snapshot.output.corrected_edges_csv,
        pd.DataFrame(
            active_rows,
            columns=[
                "source_frame",
                "source_cell_id",
                "target_frame",
                "target_cell_id",
                "frame_gap",
                "origin",
            ],
        ),
    )

    override_rows: list[dict[str, Any]] = []

    for edge in sorted(
        set(snapshot.forced_edges)
        - birth_edges
    ):
        override_rows.append(
            {
                "action": "CONTINUE",
                "source_frame": int(edge[0][0]),
                "source_cell_id": int(edge[0][1]),
                "target_frame": int(edge[1][0]),
                "target_cell_id": int(edge[1][1]),
                "frame_gap": int(
                    edge[1][0]
                    - edge[0][0]
                ),
            }
        )

    for edge in sorted(
        snapshot.broken_edges
    ):
        override_rows.append(
            {
                "action": "BREAK",
                "source_frame": int(edge[0][0]),
                "source_cell_id": int(edge[0][1]),
                "target_frame": int(edge[1][0]),
                "target_cell_id": int(edge[1][1]),
                "frame_gap": int(
                    edge[1][0]
                    - edge[0][0]
                ),
            }
        )

    for event in snapshot.birth_events:
        parent = _parse_node(
            event["parent"]
        )
        for raw_daughter in event.get(
            "daughters",
            [],
        ):
            daughter = _parse_node(
                raw_daughter
            )
            override_rows.append(
                {
                    "action": "BIRTH",
                    "source_frame": int(parent[0]),
                    "source_cell_id": int(parent[1]),
                    "target_frame": int(daughter[0]),
                    "target_cell_id": int(daughter[1]),
                    "frame_gap": int(
                        daughter[0]
                        - parent[0]
                    ),
                }
            )

    _atomic_csv(
        snapshot.output.overrides_csv,
        pd.DataFrame(
            override_rows,
            columns=[
                "action",
                "source_frame",
                "source_cell_id",
                "target_frame",
                "target_cell_id",
                "frame_gap",
            ],
        ),
    )

    birth_rows: list[dict[str, int]] = []
    for event_id, event in enumerate(
        snapshot.birth_events,
        start=1,
    ):
        parent = _parse_node(
            event["parent"]
        )
        daughters = [
            _parse_node(value)
            for value in event.get(
                "daughters",
                [],
            )
        ]
        if len(daughters) != 2:
            continue

        birth_rows.append(
            {
                "event_id": int(event_id),
                "parent_frame": int(parent[0]),
                "parent_cell_id": int(parent[1]),
                "daughter_frame": int(
                    daughters[0][0]
                ),
                "daughter_cell_id_1": int(
                    daughters[0][1]
                ),
                "daughter_cell_id_2": int(
                    daughters[1][1]
                ),
            }
        )

    _atomic_csv(
        snapshot.output.birth_events_csv,
        pd.DataFrame(
            birth_rows,
            columns=[
                "event_id",
                "parent_frame",
                "parent_cell_id",
                "daughter_frame",
                "daughter_cell_id_1",
                "daughter_cell_id_2",
            ],
        ),
    )

    ignored_rows: list[dict[str, Any]] = []
    for event_id, event in enumerate(
        snapshot.ignored_events,
        start=1,
    ):
        endpoint = _parse_node(
            event["endpoint"]
        )
        selected = _parse_node(
            event.get(
                "selected_node",
                event["endpoint"],
            )
        )
        ignored_rows.append(
            {
                "event_id": int(event_id),
                "event_type": str(
                    event.get(
                        "event_type",
                        "instance",
                    )
                ),
                "event_frame": int(
                    endpoint[0]
                ),
                "event_cell_id": int(
                    endpoint[1]
                ),
                "selected_frame": int(
                    selected[0]
                ),
                "selected_cell_id": int(
                    selected[1]
                ),
                "source_mode": str(
                    event.get(
                        "source_mode",
                        "tracking",
                    )
                ),
                "review_status": str(
                    event.get(
                        "review_status",
                        "pending",
                    )
                ),
            }
        )

    _atomic_csv(
        snapshot.output.ignored_events_csv,
        pd.DataFrame(
            ignored_rows,
            columns=[
                "event_id",
                "event_type",
                "event_frame",
                "event_cell_id",
                "selected_frame",
                "selected_cell_id",
                "source_mode",
                "review_status",
            ],
        ),
    )


def build_track_refresh_result(
    snapshot: TrackRefreshSnapshot,
) -> TrackRefreshResult:
    analysis = analyze_track_snapshot(
        snapshot
    )

    centers = {
        node: np.asarray(
            point,
            dtype=np.float64,
        )
        for node, point in snapshot.centers.items()
    }

    current_tracks = build_current_track_table(
        valid_nodes=snapshot.valid_nodes,
        active_edges=analysis.active_edges,
        centers=centers,
    )

    diagnostics = prepare_current_endpoint_groups(
        current_tracks,
        unresolved_start_nodes=set(
            analysis.unresolved_start_nodes
        ),
        unresolved_end_nodes=set(
            analysis.unresolved_end_nodes
        ),
        boundary_entry_nodes=set(
            snapshot.boundary_entry_nodes
        ),
        boundary_exit_nodes=set(
            snapshot.boundary_exit_nodes
        ),
        hidden_nodes=set(
            analysis.hidden_nodes
        ),
        ignored_start_nodes=_ignored_nodes(
            snapshot,
            "new",
        ),
        ignored_end_nodes=_ignored_nodes(
            snapshot,
            "broken",
        ),
    )

    (
        all_tracks_array,
        all_points_array,
        all_properties,
    ) = track_frame_arrays(
        current_tracks
    )

    active_tracks_array = edges_to_tracks_array(
        analysis.visible_edges,
        centers,
    )
    hidden_tracks_array = edges_to_tracks_array(
        analysis.hidden_edges,
        centers,
    )
    hidden_points_array = nodes_to_points_array(
        analysis.hidden_nodes,
        centers,
    )

    export_track_snapshot(
        snapshot,
        analysis,
        current_tracks,
    )

    return TrackRefreshResult(
        generation=int(
            snapshot.generation
        ),
        reason=str(
            snapshot.reason
        ),
        current_tracks=current_tracks,
        diagnostics=diagnostics,
        all_tracks_array=all_tracks_array,
        all_points_array=all_points_array,
        all_properties=all_properties,
        active_tracks_array=active_tracks_array,
        hidden_tracks_array=hidden_tracks_array,
        hidden_points_array=hidden_points_array,
        visible_edge_count=len(
            analysis.visible_edges
        ),
        hidden_node_count=len(
            analysis.hidden_nodes
        ),
        unresolved_start_count=len(
            analysis.unresolved_start_nodes
        ),
        unresolved_end_count=len(
            analysis.unresolved_end_nodes
        ),
    )


class TrackRefreshCoordinator:
    """
    One serialized worker with latest-state coalescing.

    While one generation is running, repeated requests replace one pending
    snapshot. Intermediate pending generations are never submitted.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=(
                "dataset-curation-refresh"
            ),
        )
        self._future: Future[
            TrackRefreshResult
        ] | None = None
        self._pending: TrackRefreshSnapshot | None = None
        self._latest_generation = 0
        self._closed = False

    @property
    def latest_generation(self) -> int:
        return int(
            self._latest_generation
        )

    @property
    def busy(self) -> bool:
        return (
            self._future is not None
            or self._pending is not None
        )

    def request(
        self,
        snapshot: TrackRefreshSnapshot,
    ) -> None:
        if self._closed:
            return

        self._latest_generation = max(
            int(self._latest_generation),
            int(snapshot.generation),
        )

        if self._future is None:
            self._future = self._executor.submit(
                build_track_refresh_result,
                snapshot,
            )
            return

        self._pending = snapshot

    def poll(
        self,
    ) -> TrackRefreshResult | None:
        future = self._future
        if future is None or not future.done():
            return None

        self._future = None
        pending = self._pending
        self._pending = None

        # Start the coalesced latest state even if the completed generation
        # raises. One bad derived refresh must not permanently stall later
        # annotations.
        if (
            not self._closed
            and pending is not None
        ):
            self._future = self._executor.submit(
                build_track_refresh_result,
                pending,
            )

        # future.result() is called only after completion and on Napari's
        # QTimer/main thread, so errors can be surfaced without cross-thread UI.
        return future.result()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending = None
        self._executor.shutdown(
            wait=False,
            cancel_futures=True,
        )
