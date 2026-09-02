from __future__ import annotations

"""Persistent corrected tracking graph with dynamic spatial-node synchronization."""

from collections import defaultdict, deque
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from dataset_curation.annotation.tracks.graph import (
    AnnotationError,
    Edge,
    Node,
    _canonical_edge,
)
from dataset_curation.annotation.tracks.storage import (
    OutputPaths,
    _atomic_csv,
    _atomic_json,
    _edge_json,
    _node_json,
    _parse_edge,
    _parse_node,
)


class TrackAnnotationSession:
    SCHEMA_VERSION = 2

    def __init__(
        self,
        *,
        sample_id: str,
        source_root: Path,
        output: OutputPaths,
        valid_nodes: set[Node],
        base_edges: set[Edge],
        resume: bool,
    ) -> None:
        self.sample_id = str(sample_id)
        self.source_root = Path(source_root)
        self.output = output
        self.valid_nodes = {
            (int(node[0]), int(node[1]))
            for node in valid_nodes
        }
        self.base_edges = {
            _canonical_edge(edge[0], edge[1])
            for edge in base_edges
        }

        self.forced_edges: set[Edge] = set()
        self.broken_edges: set[Edge] = set()
        self.completed_nodes: set[Node] = set()
        self.selections: list[Node] = []
        self.history: list[dict[str, Any]] = []

        if resume and output.state_json.is_file():
            self._load()
        else:
            self.persist()

    def _validate_node(self, node: Node) -> None:
        node = (int(node[0]), int(node[1]))
        if node not in self.valid_nodes:
            raise AnnotationError(
                f"Detection (t={node[0]}, instance={node[1]}) is not present "
                "in the current corrected spatial annotation."
            )

    def _validate_edge(self, edge: Edge) -> None:
        self._validate_node(edge[0])
        self._validate_node(edge[1])

    @property
    def active_edges(self) -> set[Edge]:
        """
        Corrected edges whose two detections still exist spatially.

        Historical Trackastra/manual edges are preserved in state so spatial
        undo can make them valid again, but invalid endpoints are not rendered
        or exported as active.
        """
        candidate = (
            self.base_edges
            | self.forced_edges
        ) - self.broken_edges
        return {
            edge
            for edge in candidate
            if edge[0] in self.valid_nodes
            and edge[1] in self.valid_nodes
        }

    @property
    def visible_edges(self) -> set[Edge]:
        return {
            edge
            for edge in self.active_edges
            if edge[0] not in self.completed_nodes
            and edge[1] not in self.completed_nodes
        }

    @property
    def hidden_edges(self) -> set[Edge]:
        return {
            edge
            for edge in self.active_edges
            if edge[0] in self.completed_nodes
            and edge[1] in self.completed_nodes
        }

    @property
    def visible_nodes(self) -> set[Node]:
        return self.valid_nodes - self.completed_nodes

    @property
    def hidden_nodes(self) -> set[Node]:
        return self.valid_nodes & self.completed_nodes

    def set_frame_nodes(
        self,
        frame: int,
        instance_ids: Iterable[int],
    ) -> tuple[set[Node], set[Node]]:
        """
        Synchronize track-selectable detections after a spatial edit.

        Existing graph overrides are retained as historical state. Edges with a
        removed endpoint simply become inactive; if a spatial undo restores the
        node they become active again.
        """
        frame = int(frame)
        new_nodes = {
            (frame, int(instance_id))
            for instance_id in instance_ids
            if int(instance_id) > 0
        }
        old_nodes = {
            node
            for node in self.valid_nodes
            if node[0] == frame
        }
        if new_nodes == old_nodes:
            return set(), set()

        removed = old_nodes - new_nodes
        added = new_nodes - old_nodes

        self.valid_nodes.difference_update(old_nodes)
        self.valid_nodes.update(new_nodes)

        self.selections = [
            node
            for node in self.selections
            if node in self.valid_nodes
            and node not in self.completed_nodes
        ][:2]

        # State JSON does not persist valid_nodes because those are authoritative
        # from the corrected spatial raster. Refresh exports only.
        self._export_tables()
        return removed, added

    def add_selection(self, node: Node) -> None:
        node = (int(node[0]), int(node[1]))
        self._validate_node(node)
        if node in self.completed_nodes:
            raise AnnotationError(
                f"t={node[0]} instance={node[1]} belongs to a completed track. "
                "Undo Complete Track before editing it."
            )
        if node in self.selections:
            raise AnnotationError(
                f"t={node[0]} instance={node[1]} is already selected."
            )
        if len(self.selections) >= 2:
            raise AnnotationError(
                "Two cells are already selected. Press Continue Track, "
                "Break Track, or Reset Track."
            )
        self.selections.append(node)
        self.persist()

    def reset_selections(self) -> None:
        if self.selections:
            self.selections.clear()
            self.persist()

    def _selected_edge(self) -> Edge:
        if len(self.selections) != 2:
            raise AnnotationError(
                "Continue/Break requires exactly two selected cells; "
                f"currently selected: {len(self.selections)}."
            )
        edge = _canonical_edge(
            self.selections[0],
            self.selections[1],
        )
        self._validate_edge(edge)
        return edge

    def connect_selected(self) -> Edge:
        edge = self._selected_edge()
        previous_forced = edge in self.forced_edges
        previous_broken = edge in self.broken_edges

        if edge in self.active_edges and not previous_broken:
            raise AnnotationError(
                "That connection is already active in the corrected graph."
            )

        self.forced_edges.add(edge)
        self.broken_edges.discard(edge)
        self.history.append(
            {
                "type": "connect",
                "edge": _edge_json(edge),
                "previous_forced": bool(previous_forced),
                "previous_broken": bool(previous_broken),
                "focus_frame": int(edge[0][0]),
            }
        )
        self.selections.clear()
        self.persist()
        return edge

    def break_selected(self) -> Edge:
        edge = self._selected_edge()
        previous_forced = edge in self.forced_edges
        previous_broken = edge in self.broken_edges

        if edge not in self.active_edges:
            raise AnnotationError(
                "That connection is already absent from the corrected graph."
            )

        self.broken_edges.add(edge)
        self.forced_edges.discard(edge)
        self.history.append(
            {
                "type": "break",
                "edge": _edge_json(edge),
                "previous_forced": bool(previous_forced),
                "previous_broken": bool(previous_broken),
                "focus_frame": int(edge[0][0]),
            }
        )
        self.selections.clear()
        self.persist()
        return edge

    def _component(self, seed: Node) -> set[Node]:
        adjacency: dict[Node, set[Node]] = defaultdict(set)
        for left, right in self.active_edges:
            adjacency[left].add(right)
            adjacency[right].add(left)

        visited = {seed}
        queue: deque[Node] = deque([seed])
        while queue:
            node = queue.popleft()
            for neighbour in adjacency.get(node, ()):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        return visited

    def complete_selected_components(self) -> set[Node]:
        if not self.selections:
            raise AnnotationError(
                "Select at least one cell from the verified track before "
                "pressing Complete Track."
            )

        newly_completed: set[Node] = set()
        for node in self.selections:
            self._validate_node(node)
            newly_completed.update(
                self._component(node)
            )
        newly_completed -= self.completed_nodes

        if not newly_completed:
            raise AnnotationError(
                "The selected corrected component is already complete."
            )

        focus_frame = min(
            node[0]
            for node in self.selections
        )
        self.completed_nodes.update(
            newly_completed
        )
        self.history.append(
            {
                "type": "complete",
                "nodes": [
                    _node_json(node)
                    for node in sorted(newly_completed)
                ],
                "focus_frame": int(focus_frame),
            }
        )
        self.selections.clear()
        self.persist()
        return newly_completed

    def undo(self) -> dict[str, Any]:
        if not self.history:
            raise AnnotationError(
                "There is no saved track operation to undo."
            )

        op = self.history.pop()
        op_type = str(op.get("type"))

        if op_type in {"connect", "break"}:
            edge = _parse_edge(op["edge"])
            if bool(op.get("previous_forced", False)):
                self.forced_edges.add(edge)
            else:
                self.forced_edges.discard(edge)

            if bool(op.get("previous_broken", False)):
                self.broken_edges.add(edge)
            else:
                self.broken_edges.discard(edge)

        elif op_type == "complete":
            nodes = {
                _parse_node(value)
                for value in op.get("nodes", [])
            }
            self.completed_nodes.difference_update(
                nodes
            )
        else:
            raise AnnotationError(
                f"Unknown track operation type: {op_type!r}"
            )

        self.selections.clear()
        self.persist()
        return op

    def persist(self) -> None:
        self.output.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "source_root": str(self.source_root),
            "node_identity": [
                "frame",
                "spatial_instance_id",
            ],
            "base_edge_count": int(
                len(self.base_edges)
            ),
            "forced_edges": [
                _edge_json(edge)
                for edge in sorted(self.forced_edges)
            ],
            "broken_edges": [
                _edge_json(edge)
                for edge in sorted(self.broken_edges)
            ],
            "completed_nodes": [
                _node_json(node)
                for node in sorted(self.completed_nodes)
            ],
            "selections": [
                _node_json(node)
                for node in self.selections
            ],
            "history": self.history,
        }
        _atomic_json(
            self.output.state_json,
            payload,
        )
        self._export_tables()

    def _export_tables(self) -> None:
        active_rows: list[dict[str, Any]] = []
        for edge in sorted(self.active_edges):
            origin = (
                "manual_continue"
                if edge in self.forced_edges
                else "trackastra"
            )
            active_rows.append(
                {
                    "source_frame": edge[0][0],
                    "source_cell_id": edge[0][1],
                    "target_frame": edge[1][0],
                    "target_cell_id": edge[1][1],
                    "frame_gap": (
                        edge[1][0]
                        - edge[0][0]
                    ),
                    "origin": origin,
                }
            )

        _atomic_csv(
            self.output.corrected_edges_csv,
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
        for edge in sorted(self.forced_edges):
            override_rows.append(
                {
                    "action": "CONTINUE",
                    "source_frame": edge[0][0],
                    "source_cell_id": edge[0][1],
                    "target_frame": edge[1][0],
                    "target_cell_id": edge[1][1],
                    "frame_gap": edge[1][0] - edge[0][0],
                }
            )
        for edge in sorted(self.broken_edges):
            override_rows.append(
                {
                    "action": "BREAK",
                    "source_frame": edge[0][0],
                    "source_cell_id": edge[0][1],
                    "target_frame": edge[1][0],
                    "target_cell_id": edge[1][1],
                    "frame_gap": edge[1][0] - edge[0][0],
                }
            )

        _atomic_csv(
            self.output.overrides_csv,
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

        completed_rows = [
            {
                "frame": node[0],
                "cell_id": node[1],
            }
            for node in sorted(
                self.completed_nodes
                & self.valid_nodes
            )
        ]
        _atomic_csv(
            self.output.completed_nodes_csv,
            pd.DataFrame(
                completed_rows,
                columns=[
                    "frame",
                    "cell_id",
                ],
            ),
        )

    def _load(self) -> None:
        payload = json.loads(
            self.output.state_json.read_text(
                encoding="utf-8"
            )
        )
        if int(
            payload.get(
                "schema_version",
                -1,
            )
        ) != self.SCHEMA_VERSION:
            raise AnnotationError(
                f"Unsupported track annotation schema in "
                f"{self.output.state_json}: "
                f"{payload.get('schema_version')!r}. "
                "This unified annotator intentionally does not carry legacy "
                "track-state compatibility."
            )
        if str(
            payload.get("sample_id")
        ) != self.sample_id:
            raise AnnotationError(
                "Existing track annotation belongs to sample "
                f"{payload.get('sample_id')!r}, not {self.sample_id!r}."
            )

        self.forced_edges = {
            _parse_edge(value)
            for value in payload.get(
                "forced_edges",
                [],
            )
        }
        self.broken_edges = {
            _parse_edge(value)
            for value in payload.get(
                "broken_edges",
                [],
            )
        }
        self.completed_nodes = {
            _parse_node(value)
            for value in payload.get(
                "completed_nodes",
                [],
            )
        }
        self.selections = [
            _parse_node(value)
            for value in payload.get(
                "selections",
                [],
            )
        ]
        self.selections = [
            node
            for node in self.selections
            if node in self.valid_nodes
            and node not in self.completed_nodes
        ][:2]
        self.history = list(
            payload.get(
                "history",
                [],
            )
        )

        # Historical forced/broken/completed nodes may temporarily be spatially
        # invalid after a split. They are intentionally retained; active_edges
        # filters them against current valid_nodes.
        self.persist()
