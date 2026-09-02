from __future__ import annotations

"""Persistent corrected tracking graph with automatic completion detection."""

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
    """
    Corrected tracking graph.

    A component is hidden automatically when all of its temporal starts/ends are
    legitimate:
      * start at first movie frame or a notebook-09 boundary entry
      * end at last movie frame or a notebook-09 boundary exit

    Continue and Birth can resolve internal broken endpoints. Break can create a
    new unresolved internal start/end, so that component becomes visible again.
    """

    SCHEMA_VERSION = 4

    def __init__(
        self,
        *,
        sample_id: str,
        source_root: Path,
        output: OutputPaths,
        valid_nodes: set[Node],
        base_edges: set[Edge],
        frame_count: int,
        boundary_entry_nodes: set[Node],
        boundary_exit_nodes: set[Node],
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
        self.frame_count = int(frame_count)
        if self.frame_count < 1:
            raise ValueError("frame_count must be >= 1")

        self.boundary_entry_nodes = {
            (int(node[0]), int(node[1]))
            for node in boundary_entry_nodes
        }
        self.boundary_exit_nodes = {
            (int(node[0]), int(node[1]))
            for node in boundary_exit_nodes
        }

        self.forced_edges: set[Edge] = set()
        self.broken_edges: set[Edge] = set()
        self.birth_events: list[dict[str, Any]] = []
        self.ignored_events: list[dict[str, Any]] = []
        self.selections: list[Node] = []
        self.history: list[dict[str, Any]] = []
        self._analysis_cache: dict[str, Any] | None = None
        self._analysis_rebuilds = 0

        # Manual completion is intentionally gone. Remove its old export if a
        # previous experimental unified session left one behind.
        obsolete_completed = self.output.root / "completed_nodes.csv"
        obsolete_completed.unlink(missing_ok=True)

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

    def _invalidate_analysis(self) -> None:
        self._analysis_cache = None

    def _analysis(self) -> dict[str, Any]:
        cached = self._analysis_cache
        if cached is not None:
            return cached

        candidate = (self.base_edges | self.forced_edges) - self.broken_edges
        active_edges: set[Edge] = {
            edge
            for edge in candidate
            if edge[0] in self.valid_nodes and edge[1] in self.valid_nodes
        }

        adjacency: dict[Node, set[Node]] = defaultdict(set)
        incoming: set[Node] = set()
        outgoing: set[Node] = set()

        for left, right in active_edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
            outgoing.add(left)
            incoming.add(right)

        remaining = set(self.valid_nodes)
        components: list[frozenset[Node]] = []
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
                    if neighbour in remaining and neighbour not in visited:
                        visited.add(neighbour)
                        queue.append(neighbour)

            remaining.difference_update(visited)
            component = frozenset(visited)
            components.append(component)

            starts = {node for node in component if node not in incoming}
            ends = {node for node in component if node not in outgoing}

            bad_starts = {
                node for node in starts if not self._legitimate_start(node)
            }
            bad_ends = {
                node for node in ends if not self._legitimate_end(node)
            }

            unresolved_starts.update(bad_starts)
            unresolved_ends.update(bad_ends)

            if component and not bad_starts and not bad_ends:
                hidden_nodes.update(component)

        hidden_edges = {
            edge
            for edge in active_edges
            if edge[0] in hidden_nodes and edge[1] in hidden_nodes
        }
        visible_edges = active_edges - hidden_edges

        result: dict[str, Any] = {
            'active_edges': active_edges,
            'components': tuple(components),
            'incoming': incoming,
            'outgoing': outgoing,
            'hidden_nodes': hidden_nodes,
            'hidden_edges': hidden_edges,
            'visible_edges': visible_edges,
            'unresolved_start_nodes': unresolved_starts,
            'unresolved_end_nodes': unresolved_ends,
        }
        self._analysis_cache = result
        self._analysis_rebuilds += 1

        print(
            '[tracks] graph analysis '
            f'#{self._analysis_rebuilds}: '
            f'nodes={len(self.valid_nodes)} '
            f'edges={len(active_edges)} '
            f'components={len(components)} '
            f'hidden_nodes={len(hidden_nodes)} '
            f'unresolved_starts={len(unresolved_starts)} '
            f'unresolved_ends={len(unresolved_ends)}',
            flush=True,
        )
        return result

    @property
    def active_edges(self) -> set[Edge]:
        return self._analysis()['active_edges']

    def _component_sets(self) -> list[set[Node]]:
        return [set(component) for component in self._analysis()['components']]

    def _component_temporal_endpoints(
        self,
        component: set[Node],
    ) -> tuple[set[Node], set[Node]]:
        analysis = self._analysis()
        incoming = analysis['incoming']
        outgoing = analysis['outgoing']
        starts = {node for node in component if node not in incoming}
        ends = {node for node in component if node not in outgoing}
        return starts, ends

    def _legitimate_start(self, node: Node) -> bool:
        return int(node[0]) == 0 or node in self.boundary_entry_nodes

    def _legitimate_end(self, node: Node) -> bool:
        return (
            int(node[0]) == self.frame_count - 1
            or node in self.boundary_exit_nodes
        )

    @property
    def unresolved_start_nodes(self) -> set[Node]:
        return self._analysis()['unresolved_start_nodes']

    @property
    def unresolved_end_nodes(self) -> set[Node]:
        return self._analysis()['unresolved_end_nodes']

    def _component_complete(self, component: set[Node]) -> bool:
        return bool(component) and component <= self._analysis()['hidden_nodes']

    @property
    def hidden_nodes(self) -> set[Node]:
        return self._analysis()['hidden_nodes']

    @property
    def visible_nodes(self) -> set[Node]:
        return self.valid_nodes - self._analysis()['hidden_nodes']

    @property
    def hidden_edges(self) -> set[Edge]:
        return self._analysis()['hidden_edges']

    @property
    def visible_edges(self) -> set[Edge]:
        return self._analysis()['visible_edges']

    @property
    def birth_edges(self) -> set[Edge]:
        result: set[Edge] = set()
        for event in self.birth_events:
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
                        _parse_node(raw_daughter),
                    )
                )
        return result

    @property
    def ignored_start_nodes(self) -> set[Node]:
        return {
            _parse_node(event["endpoint"])
            for event in self.ignored_events
            if str(event.get("event_type")) == "new"
        }


    @property
    def ignored_end_nodes(self) -> set[Node]:
        return {
            _parse_node(event["endpoint"])
            for event in self.ignored_events
            if str(event.get("event_type")) == "broken"
        }


    def ignore_events(
        self,
        *,
        selected_node: Node,
        source_mode: str,
        events: Iterable[tuple[str, Node]],
    ) -> list[dict[str, Any]]:
        selected_node = (
            int(selected_node[0]),
            int(selected_node[1]),
        )
        self._validate_node(selected_node)

        source_mode = str(source_mode)
        if source_mode not in {"spatial", "tracking"}:
            raise AnnotationError(
                f"Unsupported Ignore source mode: {source_mode!r}."
            )

        normalized: list[tuple[str, Node]] = []
        for event_type, endpoint in events:
            event_type = str(event_type)
            if event_type not in {"broken", "new"}:
                continue
            endpoint = (
                int(endpoint[0]),
                int(endpoint[1]),
            )
            self._validate_node(endpoint)
            normalized.append((event_type, endpoint))

        if not normalized:
            normalized = [("instance", selected_node)]

        existing = {
            (
                str(event.get("event_type")),
                _parse_node(event["endpoint"]),
            )
            for event in self.ignored_events
            if "endpoint" in event
        }

        added: list[dict[str, Any]] = []
        for event_type, endpoint in normalized:
            key = (event_type, endpoint)
            if key in existing:
                continue

            record = {
                "event_type": event_type,
                "endpoint": _node_json(endpoint),
                "selected_node": _node_json(selected_node),
                "source_mode": source_mode,
                "review_status": "pending",
            }
            self.ignored_events.append(record)
            added.append(record)
            existing.add(key)

        if added:
            self.persist()

        return added

    def set_frame_nodes(
        self,
        frame: int,
        instance_ids: Iterable[int],
    ) -> tuple[set[Node], set[Node]]:
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
        self._invalidate_analysis()

        self.selections = [
            node
            for node in self.selections
            if node in self.valid_nodes
        ][:3]

        self.persist()
        return removed, added

    def add_selection(self, node: Node) -> None:
        node = (int(node[0]), int(node[1]))
        self._validate_node(node)

        if node in self.selections:
            raise AnnotationError(
                f"t={node[0]} instance={node[1]} is already selected."
            )
        if len(self.selections) >= 3:
            raise AnnotationError(
                "Three track cells are already selected. "
                "Use Birth or Reset Track."
            )

        self.selections.append(node)

    def reset_selections(self) -> None:
        self.selections.clear()

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
        self._invalidate_analysis()
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

        if edge in self.birth_edges:
            raise AnnotationError(
                "That edge belongs to an annotated Birth event. "
                "Undo Birth before changing a parent-to-daughter edge."
            )

        if edge not in self.active_edges:
            raise AnnotationError(
                "That connection is already absent from the corrected graph."
            )

        self.broken_edges.add(edge)
        self.forced_edges.discard(edge)
        self._invalidate_analysis()
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

    def _birth_from_selections(
        self,
    ) -> tuple[Node, tuple[Node, Node]]:
        if len(self.selections) != 3:
            raise AnnotationError(
                "Birth requires exactly three selected cells: "
                "one parent in the earlier frame and two daughters "
                "in the next consecutive frame."
            )

        by_frame: dict[int, list[Node]] = defaultdict(list)
        for node in self.selections:
            self._validate_node(node)
            by_frame[int(node[0])].append(node)

        if len(by_frame) != 2:
            raise AnnotationError(
                "Birth selections must occupy exactly two frames."
            )

        frames = sorted(by_frame)
        earlier, later = frames
        if later != earlier + 1:
            raise AnnotationError(
                "Birth requires consecutive frames; "
                f"selected frames are {earlier} and {later}."
            )

        if (
            len(by_frame[earlier]) != 1
            or len(by_frame[later]) != 2
        ):
            raise AnnotationError(
                "Birth direction is parent -> two daughters: "
                "select exactly one cell in the earlier frame and "
                "two cells in the next frame."
            )

        parent = by_frame[earlier][0]
        daughters = tuple(
            sorted(
                by_frame[later],
                key=lambda node: node[1],
            )
        )
        if daughters[0] == daughters[1]:
            raise AnnotationError(
                "The two daughter cells must be different."
            )

        return parent, (
            daughters[0],
            daughters[1],
        )

    def birth_selected(
        self,
    ) -> dict[str, Any]:
        parent, daughters = (
            self._birth_from_selections()
        )

        event = {
            "parent": _node_json(parent),
            "daughters": [
                _node_json(daughters[0]),
                _node_json(daughters[1]),
            ],
        }

        for existing in self.birth_events:
            if existing == event:
                raise AnnotationError(
                    "That birth event is already annotated."
                )

        edges = (
            _canonical_edge(
                parent,
                daughters[0],
            ),
            _canonical_edge(
                parent,
                daughters[1],
            ),
        )

        edge_states: list[dict[str, Any]] = []
        for edge in edges:
            previous_forced = (
                edge in self.forced_edges
            )
            previous_broken = (
                edge in self.broken_edges
            )
            edge_states.append(
                {
                    "edge": _edge_json(edge),
                    "previous_forced": bool(
                        previous_forced
                    ),
                    "previous_broken": bool(
                        previous_broken
                    ),
                }
            )
            self.forced_edges.add(edge)
            self.broken_edges.discard(edge)

        self.birth_events.append(event)
        self._invalidate_analysis()
        self.history.append(
            {
                "type": "birth",
                "event": event,
                "edge_states": edge_states,
                "focus_frame": int(parent[0]),
            }
        )
        self.selections.clear()
        self.persist()

        return {
            "parent": parent,
            "daughters": daughters,
            "edges": edges,
        }

    def undo(self) -> dict[str, Any]:
        if not self.history:
            raise AnnotationError(
                "There is no saved track operation to undo."
            )

        op = self.history.pop()
        op_type = str(op.get("type"))

        if op_type in {
            "connect",
            "break",
        }:
            edge = _parse_edge(
                op["edge"]
            )
            if bool(
                op.get(
                    "previous_forced",
                    False,
                )
            ):
                self.forced_edges.add(edge)
            else:
                self.forced_edges.discard(edge)

            if bool(
                op.get(
                    "previous_broken",
                    False,
                )
            ):
                self.broken_edges.add(edge)
            else:
                self.broken_edges.discard(edge)

        elif op_type == "birth":
            event = dict(
                op.get(
                    "event",
                    {},
                )
            )
            removed = False
            for index in range(
                len(self.birth_events) - 1,
                -1,
                -1,
            ):
                if self.birth_events[index] == event:
                    self.birth_events.pop(index)
                    removed = True
                    break
            if not removed:
                raise AnnotationError(
                    "Cannot undo Birth because its event record is missing."
                )

            for state in op.get(
                "edge_states",
                [],
            ):
                edge = _parse_edge(
                    state["edge"]
                )
                if bool(
                    state.get(
                        "previous_forced",
                        False,
                    )
                ):
                    self.forced_edges.add(edge)
                else:
                    self.forced_edges.discard(edge)

                if bool(
                    state.get(
                        "previous_broken",
                        False,
                    )
                ):
                    self.broken_edges.add(edge)
                else:
                    self.broken_edges.discard(edge)

        else:
            raise AnnotationError(
                f"Unknown track operation type: {op_type!r}"
            )

        self._invalidate_analysis()
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
            "frame_count": int(
                self.frame_count
            ),
            "base_edge_count": int(
                len(self.base_edges)
            ),
            "forced_edges": [
                _edge_json(edge)
                for edge in sorted(
                    self.forced_edges
                )
            ],
            "broken_edges": [
                _edge_json(edge)
                for edge in sorted(
                    self.broken_edges
                )
            ],
            "birth_events": self.birth_events,
            "ignored_events": self.ignored_events,
            "selections": [
                _node_json(node)
                for node in self.selections
            ],
            "history": self.history,
            "automatic_hidden_nodes": int(
                len(self.hidden_nodes)
            ),
            "unresolved_start_nodes": [
                _node_json(node)
                for node in sorted(
                    self.unresolved_start_nodes
                )
            ],
            "unresolved_end_nodes": [
                _node_json(node)
                for node in sorted(
                    self.unresolved_end_nodes
                )
            ],
        }
        _atomic_json(
            self.output.state_json,
            payload,
        )
        self._export_tables()

    def _export_tables(self) -> None:
        birth_edges = self.birth_edges

        active_rows: list[dict[str, Any]] = []
        for edge in sorted(
            self.active_edges
        ):
            if edge in birth_edges:
                origin = "birth"
            elif edge in self.forced_edges:
                origin = "manual_continue"
            else:
                origin = "trackastra"

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
        for edge in sorted(
            self.forced_edges
            - birth_edges
        ):
            override_rows.append(
                {
                    "action": "CONTINUE",
                    "source_frame": edge[0][0],
                    "source_cell_id": edge[0][1],
                    "target_frame": edge[1][0],
                    "target_cell_id": edge[1][1],
                    "frame_gap": (
                        edge[1][0]
                        - edge[0][0]
                    ),
                }
            )

        for edge in sorted(
            self.broken_edges
        ):
            override_rows.append(
                {
                    "action": "BREAK",
                    "source_frame": edge[0][0],
                    "source_cell_id": edge[0][1],
                    "target_frame": edge[1][0],
                    "target_cell_id": edge[1][1],
                    "frame_gap": (
                        edge[1][0]
                        - edge[0][0]
                    ),
                }
            )

        for event_index, event in enumerate(
            self.birth_events,
            start=1,
        ):
            parent = _parse_node(
                event["parent"]
            )
            for raw_daughter in event[
                "daughters"
            ]:
                daughter = _parse_node(
                    raw_daughter
                )
                override_rows.append(
                    {
                        "action": "BIRTH",
                        "source_frame": parent[0],
                        "source_cell_id": parent[1],
                        "target_frame": daughter[0],
                        "target_cell_id": daughter[1],
                        "frame_gap": (
                            daughter[0]
                            - parent[0]
                        ),
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

        birth_rows: list[dict[str, int]] = []
        for event_index, event in enumerate(
            self.birth_events,
            start=1,
        ):
            parent = _parse_node(
                event["parent"]
            )
            daughters = [
                _parse_node(value)
                for value in event[
                    "daughters"
                ]
            ]
            birth_rows.append(
                {
                    "event_id": int(
                        event_index
                    ),
                    "parent_frame": int(
                        parent[0]
                    ),
                    "parent_cell_id": int(
                        parent[1]
                    ),
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
            self.output.birth_events_csv,
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
            self.ignored_events,
            start=1,
        ):
            endpoint = _parse_node(event["endpoint"])
            selected = _parse_node(
                event.get(
                    "selected_node",
                    event["endpoint"],
                )
            )
            ignored_rows.append(
                {
                    "event_id": int(event_id),
                    "event_type": str(event.get("event_type", "instance")),
                    "event_frame": int(endpoint[0]),
                    "event_cell_id": int(endpoint[1]),
                    "selected_frame": int(selected[0]),
                    "selected_cell_id": int(selected[1]),
                    "source_mode": str(event.get("source_mode", "tracking")),
                    "review_status": str(event.get("review_status", "pending")),
                }
            )

        _atomic_csv(
            self.output.ignored_events_csv,
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

    def _load(self) -> None:
        payload = json.loads(
            self.output.state_json.read_text(
                encoding="utf-8"
            )
        )
        loaded_schema = int(
            payload.get(
                "schema_version",
                -1,
            )
        )

        # Schema 4 is an additive extension of schema 3: it adds
        # ignored_events only. A schema-3 annotation is therefore interpreted
        # exactly as the same annotation with ignored_events == [].
        if loaded_schema not in {
            3,
            self.SCHEMA_VERSION,
        }:
            raise AnnotationError(
                f"Unsupported track annotation schema in "
                f"{self.output.state_json}: "
                f"{payload.get('schema_version')!r}. "
                "Only the lossless schema-3 -> schema-4 Ignore upgrade "
                "is supported."
            )

        if str(
            payload.get(
                "sample_id"
            )
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

        raw_birth_events = payload.get(
            "birth_events",
            [],
        )
        if not isinstance(
            raw_birth_events,
            list,
        ):
            raise AnnotationError(
                "birth_events must be a list."
            )
        self.birth_events = [
            {
                "parent": _node_json(
                    _parse_node(
                        event["parent"]
                    )
                ),
                "daughters": [
                    _node_json(
                        _parse_node(value)
                    )
                    for value in event.get(
                        "daughters",
                        [],
                    )
                ],
            }
            for event in raw_birth_events
        ]

        for event in self.birth_events:
            if len(
                event["daughters"]
            ) != 2:
                raise AnnotationError(
                    "Every birth event must contain exactly two daughters."
                )

        raw_ignored_events = payload.get(
            "ignored_events",
            [],
        )
        if not isinstance(raw_ignored_events, list):
            raise AnnotationError(
                "ignored_events must be a list."
            )

        self.ignored_events = []
        for raw_event in raw_ignored_events:
            event_type = str(raw_event.get("event_type", ""))
            if event_type not in {"broken", "new", "instance"}:
                raise AnnotationError(
                    f"Invalid ignored event type: {event_type!r}."
                )

            endpoint = _parse_node(raw_event["endpoint"])
            selected_node = _parse_node(
                raw_event.get(
                    "selected_node",
                    raw_event["endpoint"],
                )
            )
            self.ignored_events.append(
                {
                    "event_type": event_type,
                    "endpoint": _node_json(endpoint),
                    "selected_node": _node_json(selected_node),
                    "source_mode": str(
                        raw_event.get("source_mode", "tracking")
                    ),
                    "review_status": str(
                        raw_event.get("review_status", "pending")
                    ),
                }
            )

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
        ][:3]
        self.history = list(
            payload.get(
                "history",
                [],
            )
        )

        self.persist()
