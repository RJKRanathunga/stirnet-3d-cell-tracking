from __future__ import annotations

"""Persistent Trackastra association-correction session state."""

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

import json

from collections import defaultdict, deque

from pathlib import Path

from typing import Any, Iterable

import pandas as pd

class TrackAnnotationSession:
    SCHEMA_VERSION = 1

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
        self.sample_id = sample_id
        self.source_root = source_root
        self.output = output
        self.valid_nodes = set(valid_nodes)
        self.base_edges = set(base_edges)

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
        if node not in self.valid_nodes:
            raise AnnotationError(
                f"Detection (t={node[0]}, instance={node[1]}) is not present "
                "in cells_all.csv."
            )

    def _validate_edge(self, edge: Edge) -> None:
        self._validate_node(edge[0])
        self._validate_node(edge[1])

    @property
    def active_edges(self) -> set[Edge]:
        return (self.base_edges | self.forced_edges) - self.broken_edges

    def add_selection(self, node: Node) -> None:
        self._validate_node(node)
        if node in self.completed_nodes:
            raise AnnotationError(
                f"t={node[0]} instance={node[1]} belongs to a completed track. "
                "Undo Complete Track if you need to edit it again."
            )
        if node in self.selections:
            raise AnnotationError(
                f"t={node[0]} instance={node[1]} is already selected."
            )
        if len(self.selections) >= 2:
            raise AnnotationError(
                "Two cells are already selected. Press Connect, Break, or Reset."
            )
        self.selections.append(node)
        self.persist()

    def reset_selections(self) -> None:
        self.selections.clear()
        self.persist()

    def _selected_edge(self) -> Edge:
        if len(self.selections) != 2:
            raise AnnotationError(
                f"Connect/Break requires exactly two selected cells; "
                f"currently selected: {len(self.selections)}."
            )
        edge = _canonical_edge(self.selections[0], self.selections[1])
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
            newly_completed.update(self._component(node))
        newly_completed -= self.completed_nodes

        if not newly_completed:
            raise AnnotationError("The selected corrected component is already complete.")

        focus_frame = min(node[0] for node in self.selections)
        self.completed_nodes.update(newly_completed)
        self.history.append(
            {
                "type": "complete",
                "nodes": [_node_json(node) for node in sorted(newly_completed)],
                "focus_frame": int(focus_frame),
            }
        )
        self.selections.clear()
        self.persist()
        return newly_completed

    def undo(self) -> dict[str, Any]:
        if not self.history:
            raise AnnotationError("There is no saved operation to undo.")

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
            nodes = {_parse_node(value) for value in op.get("nodes", [])}
            self.completed_nodes.difference_update(nodes)
        else:
            raise AnnotationError(f"Unknown operation type in history: {op_type!r}")

        self.selections.clear()
        self.persist()
        return op

    def persist(self) -> None:
        self.output.root.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "source_root": str(self.source_root),
            "node_identity": ["frame", "spatial_instance_id"],
            "base_edge_count": int(len(self.base_edges)),
            "forced_edges": [_edge_json(edge) for edge in sorted(self.forced_edges)],
            "broken_edges": [_edge_json(edge) for edge in sorted(self.broken_edges)],
            "completed_nodes": [
                _node_json(node) for node in sorted(self.completed_nodes)
            ],
            "selections": [_node_json(node) for node in self.selections],
            "history": self.history,
        }
        _atomic_json(self.output.state_json, payload)
        self._export_tables()

    def _export_tables(self) -> None:
        active_rows: list[dict[str, Any]] = []
        for edge in sorted(self.active_edges):
            origin = "manual_connect" if edge in self.forced_edges else "trackastra"
            active_rows.append(
                {
                    "source_frame": edge[0][0],
                    "source_cell_id": edge[0][1],
                    "target_frame": edge[1][0],
                    "target_cell_id": edge[1][1],
                    "frame_gap": edge[1][0] - edge[0][0],
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
                    "action": "CONNECT",
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
            {"frame": node[0], "cell_id": node[1]}
            for node in sorted(self.completed_nodes)
        ]
        _atomic_csv(
            self.output.completed_nodes_csv,
            pd.DataFrame(completed_rows, columns=["frame", "cell_id"]),
        )

    def _load(self) -> None:
        payload = json.loads(self.output.state_json.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise AnnotationError(
                f"Unsupported annotation schema in {self.output.state_json}: "
                f"{payload.get('schema_version')!r}"
            )
        if str(payload.get("sample_id")) != self.sample_id:
            raise AnnotationError(
                "Existing annotation file belongs to sample "
                f"{payload.get('sample_id')!r}, not {self.sample_id!r}."
            )

        self.forced_edges = {
            _parse_edge(value) for value in payload.get("forced_edges", [])
        }
        self.broken_edges = {
            _parse_edge(value) for value in payload.get("broken_edges", [])
        }
        self.completed_nodes = {
            _parse_node(value) for value in payload.get("completed_nodes", [])
        }
        self.selections = [
            _parse_node(value) for value in payload.get("selections", [])
        ][:2]
        self.history = list(payload.get("history", []))

        for edge in self.forced_edges | self.broken_edges:
            self._validate_edge(edge)
        for node in self.completed_nodes | set(self.selections):
            self._validate_node(node)

        self.persist()
