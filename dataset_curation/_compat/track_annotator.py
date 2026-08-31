from __future__ import annotations

# DATASET_CURATION_REFACTOR_CURRENT_V1: migrated current track annotator

r"""
Interactive BioHub track annotator built on Investigation 36 outputs.

Purpose
-------
Correct Trackastra associations without mutating the cached Trackastra graph.
The stable annotation node is a spatial detection:

    (frame, spatial_instance_id)

This makes the resulting supervision independent of Trackastra's current
track IDs and suitable for later tracker fine-tuning / graph supervision.

Default source
--------------
    runs/stirnet/evaluation/
        36_biohub_spatial_trackastra_visualization/
        <sample-id>/

Required source files
---------------------
    movies/raw.npy
    movies/binary_mask.npy
    movies/final_instances.npy
    cells_all.csv
    trackastra/tracked_masks.npy              (optional display layer)
    trackastra/napari_tracks.npy              (optional original-track display)
    trackastra/napari_graph.json
    trackastra/tracks.csv

Interaction
-----------
1. Rotate the 3-D volume and click a cell.
2. Move to a previous/next frame (or a farther frame for a gap) and click the
   corresponding cell.
3. Press Connect or Break.
4. When a corrected connected track/lineage component has been checked, select
   any cell in it and press Complete Track. Completed detections disappear from
   the annotation candidate layers.

Picking
-------
- Normal mode: choose the current-frame cell centroid closest to the mouse ray.
- If the "Binary Mask" layer is visible: explicitly ray-march to the FIRST
  foreground voxel. If that voxel belongs to a spatial instance, that instance
  is selected. Otherwise the nearest current-frame centroid to the hit point is
  used as a fallback.
- Click-drag remains normal Napari camera navigation; only a plain left click
  selects a cell.

Selection persistence
---------------------
Selections are stored as (frame, instance_id). Their highlight disappears when
another frame is displayed and reappears when returning to the selected frame.
Connect/Break reset the active pair. Reset (or Escape) clears the active pair.

Graph semantics
---------------
- Connections may span arbitrary frame gaps.
- One-to-many edges are allowed, so division supervision can be entered by
  connecting a parent independently to both daughters.
- Connect and Break are stored as overrides on top of the Trackastra graph.
- The Trackastra cache is NEVER edited in place.
- Complete Track operates on the corrected undirected connected component of
  the currently selected cell(s).
- Undo is global/LIFO and persists because operation history is saved.

Persisted outputs
-----------------
By default:
    evaluation/segmentation/track_annotations/<sample-id>/

Files:
    track_annotations.json   complete resumable state + operation history
    corrected_edges.csv      currently active corrected positive edges
    edge_overrides.csv       manual CONNECT / BREAK overrides
    completed_nodes.csv      detections hidden as completed

Typical run
-----------
From repository root:

    python .\evaluation\segmentation\scripts\04_track_annotator.py

Use a different Investigation-36 cache:

    python .\evaluation\segmentation\scripts\04_track_annotator.py ^
        --source-root runs/stirnet/evaluation/36_biohub_spatial_trackastra_visualization/44b6_0113de3b
"""

import argparse
import colorsys
import json
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import napari
import numpy as np
import pandas as pd

try:
    from magicgui.widgets import Container, Label, PushButton
except ImportError as exc:
    raise ImportError(
        "magicgui is required for the track annotation panel. It normally "
        "comes with Napari. Install it with: pip install magicgui"
    ) from exc

try:
    from qtpy.QtWidgets import QSizePolicy
except ImportError:
    QSizePolicy = None


# =============================================================================
# Configuration / paths
# =============================================================================


from dataset_curation._repo import repo_root as _dataset_curation_repo_root

REPO_ROOT = _dataset_curation_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SAMPLE_ID = "44b6_0113de3b"
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)
DEFAULT_MAX_RAY_DISTANCE_UM = 8.0
DEFAULT_INV36_ROOT = (
    REPO_ROOT
    / "runs"
    / "stirnet"
    / "evaluation"
    / "36_biohub_spatial_trackastra_visualization"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "evaluation"
    / "segmentation"
    / "track_annotations"
)

Node = tuple[int, int]  # (frame, spatial_instance_id)
Edge = tuple[Node, Node]  # directed earlier -> later


class AnnotationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourcePaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "movies" / "raw.npy"

    @property
    def binary_mask(self) -> Path:
        return self.root / "movies" / "binary_mask.npy"

    @property
    def final_instances(self) -> Path:
        return self.root / "movies" / "final_instances.npy"

    @property
    def cells_csv(self) -> Path:
        return self.root / "cells_all.csv"

    @property
    def tracked_masks(self) -> Path:
        return self.root / "trackastra" / "tracked_masks.npy"

    @property
    def napari_tracks(self) -> Path:
        return self.root / "trackastra" / "napari_tracks.npy"

    @property
    def napari_graph(self) -> Path:
        return self.root / "trackastra" / "napari_graph.json"

    @property
    def tracks_csv(self) -> Path:
        return self.root / "trackastra" / "tracks.csv"


@dataclass(frozen=True)
class OutputPaths:
    root: Path

    @property
    def state_json(self) -> Path:
        return self.root / "track_annotations.json"

    @property
    def corrected_edges_csv(self) -> Path:
        return self.root / "corrected_edges.csv"

    @property
    def overrides_csv(self) -> Path:
        return self.root / "edge_overrides.csv"

    @property
    def completed_nodes_csv(self) -> Path:
        return self.root / "completed_nodes.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Trackastra association annotator for BioHub."
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Investigation-36 sample output directory. Default: "
            "runs/stirnet/evaluation/36_biohub_spatial_trackastra_visualization/"
            "<sample-id>."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Annotation output directory. Default: "
            "evaluation/segmentation/track_annotations/<sample-id>."
        ),
    )
    parser.add_argument(
        "--spacing-zyx",
        default=",".join(str(v) for v in DEFAULT_SPACING_ZYX_UM),
        help="Physical voxel spacing in micrometres as z,y,x.",
    )
    parser.add_argument(
        "--max-ray-distance-um",
        type=float,
        default=DEFAULT_MAX_RAY_DISTANCE_UM,
        help=(
            "Maximum allowed centroid-to-ray distance in normal picking mode. "
            "Use <=0 to disable the distance guard."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing track_annotations.json and start fresh.",
    )
    return parser.parse_args()


def _resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _parse_spacing(text: str) -> tuple[float, float, float]:
    values = tuple(float(token.strip()) for token in str(text).split(","))
    if len(values) != 3 or any(value <= 0 for value in values):
        raise AnnotationError(
            f"--spacing-zyx must contain three positive values, got {text!r}."
        )
    return values


def resolve_paths(args: argparse.Namespace) -> tuple[SourcePaths, OutputPaths]:
    source_root = (
        DEFAULT_INV36_ROOT / args.sample_id
        if args.source_root is None
        else _resolve_repo_path(args.source_root)
    ).resolve()
    output_root = (
        DEFAULT_OUTPUT_ROOT / args.sample_id
        if args.output_dir is None
        else _resolve_repo_path(args.output_dir)
    ).resolve()

    source = SourcePaths(source_root)
    output = OutputPaths(output_root)

    required = (
        source.raw,
        source.binary_mask,
        source.final_instances,
        source.cells_csv,
        source.napari_graph,
        source.tracks_csv,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        lines = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "Investigation-36 cache is incomplete. Missing:\n" + lines
        )

    output.root.mkdir(parents=True, exist_ok=True)
    return source, output


# =============================================================================
# Serialization helpers
# =============================================================================


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _node_json(node: Node) -> list[int]:
    return [int(node[0]), int(node[1])]


def _edge_json(edge: Edge) -> list[list[int]]:
    return [_node_json(edge[0]), _node_json(edge[1])]


def _parse_node(value: Any) -> Node:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise AnnotationError(f"Invalid serialized node: {value!r}")
    return int(value[0]), int(value[1])


def _parse_edge(value: Any) -> Edge:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise AnnotationError(f"Invalid serialized edge: {value!r}")
    return _canonical_edge(_parse_node(value[0]), _parse_node(value[1]))


def _canonical_edge(a: Node, b: Node) -> Edge:
    if a == b:
        raise AnnotationError("A track edge cannot connect a detection to itself.")
    if a[0] == b[0]:
        raise AnnotationError(
            "Track edges must connect different frames; both selections are at "
            f"t={a[0]}."
        )
    return (a, b) if a[0] < b[0] else (b, a)


# =============================================================================
# Source graph construction
# =============================================================================


def _validate_source_arrays(
    raw: np.ndarray,
    binary_mask: np.ndarray,
    instances: np.ndarray,
) -> None:
    if raw.ndim != 4:
        raise AnnotationError(f"raw.npy must be (T,Z,Y,X), got {raw.shape}")
    if binary_mask.shape != raw.shape:
        raise AnnotationError(
            f"binary mask shape {binary_mask.shape} != raw shape {raw.shape}"
        )
    if instances.shape != raw.shape:
        raise AnnotationError(
            f"final instances shape {instances.shape} != raw shape {raw.shape}"
        )


def _normalize_cells(cells: pd.DataFrame) -> pd.DataFrame:
    required = {
        "frame",
        "cell_id",
        "centroid_z",
        "centroid_y",
        "centroid_x",
    }
    missing = sorted(required - set(cells.columns))
    if missing:
        raise AnnotationError(f"cells_all.csv is missing columns: {missing}")

    result = cells.copy()
    result["frame"] = result["frame"].astype(np.int64)
    result["cell_id"] = result["cell_id"].astype(np.int64)

    duplicate = result.duplicated(["frame", "cell_id"], keep=False)
    if duplicate.any():
        rows = result.loc[duplicate, ["frame", "cell_id"]].head(10)
        raise AnnotationError(
            "cells_all.csv contains duplicate (frame, cell_id) detections:\n"
            + rows.to_string(index=False)
        )
    return result


def _normalize_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    required = {"track_id", "frame", "cell_id", "z", "y", "x"}
    missing = sorted(required - set(tracks.columns))
    if missing:
        raise AnnotationError(f"trackastra/tracks.csv is missing columns: {missing}")

    result = tracks.copy()
    result["track_id"] = result["track_id"].astype(np.int64)
    result["frame"] = result["frame"].astype(np.int64)
    result["cell_id"] = result["cell_id"].astype(np.int64)
    result = result.loc[result["cell_id"] > 0].copy()
    return result


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


# =============================================================================
# Annotation state
# =============================================================================


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


# =============================================================================
# Ray picking
# =============================================================================


def _ray_intersections_data(layer, event) -> tuple[np.ndarray, np.ndarray] | None:
    view_direction = getattr(event, "view_direction", None)
    dims_displayed = getattr(event, "dims_displayed", None)
    if view_direction is None or dims_displayed is None:
        return None

    try:
        start, end = layer.get_ray_intersections(
            position=event.position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=True,
        )
    except TypeError:
        try:
            start, end = layer.get_ray_intersections(
                event.position,
                view_direction,
                dims_displayed,
                world=True,
            )
        except Exception:
            return None
    except Exception:
        return None

    if start is None or end is None:
        return None
    start = np.asarray(start, dtype=np.float64).reshape(-1)
    end = np.asarray(end, dtype=np.float64).reshape(-1)
    if start.shape != end.shape or not np.all(np.isfinite(start)) or not np.all(
        np.isfinite(end)
    ):
        return None
    return start, end


def _first_foreground_point_along_ray(
    mask: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    samples_per_voxel: float = 4.0,
) -> np.ndarray | None:
    data = np.asarray(mask)
    if start.size != data.ndim or end.size != data.ndim:
        return None

    delta = end - start
    max_axis_distance = float(np.max(np.abs(delta)))
    sample_count = max(
        2,
        int(np.ceil(max_axis_distance * float(samples_per_voxel))) + 1,
    )
    shape = np.asarray(data.shape, dtype=np.int64)

    for alpha in np.linspace(0.0, 1.0, sample_count, dtype=np.float64):
        point = start + alpha * delta
        index = np.rint(point).astype(np.int64)
        if np.any(index < 0) or np.any(index >= shape):
            continue
        if int(data[tuple(index.tolist())]) > 0:
            return point
    return None


def _distance_points_to_segment(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    direction = end - start
    denom = float(np.dot(direction, direction))
    if denom <= 1e-12:
        return np.linalg.norm(points - start[None, :], axis=1)
    alpha = ((points - start[None, :]) @ direction) / denom
    alpha = np.clip(alpha, 0.0, 1.0)
    closest = start[None, :] + alpha[:, None] * direction[None, :]
    return np.linalg.norm(points - closest, axis=1)


class DetectionPicker:
    def __init__(
        self,
        *,
        cells: pd.DataFrame,
        instances: np.ndarray,
        binary_mask: np.ndarray,
        spacing_zyx: tuple[float, float, float],
        max_ray_distance_um: float,
        session: TrackAnnotationSession,
    ) -> None:
        self.cells = cells
        self.instances = instances
        self.binary_mask = binary_mask
        self.spacing = np.asarray(spacing_zyx, dtype=np.float64)
        self.max_ray_distance_um = float(max_ray_distance_um)
        self.session = session

        self.rows_by_frame: dict[int, pd.DataFrame] = {
            int(frame): group.reset_index(drop=True)
            for frame, group in cells.groupby("frame", sort=False)
        }

    def _nearest_to_point(self, frame: int, point_zyx: np.ndarray) -> Node:
        rows = self.rows_by_frame.get(frame)
        if rows is None or rows.empty:
            raise AnnotationError(f"No detections are available at t={frame}.")

        centers = rows[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(float)
        distances = np.linalg.norm(
            (centers - np.asarray(point_zyx)[None, :]) * self.spacing[None, :],
            axis=1,
        )
        order = np.argsort(distances)
        for index in order:
            node = (frame, int(rows.iloc[int(index)]["cell_id"]))
            if node not in self.session.completed_nodes:
                return node
        raise AnnotationError(f"Every detection at t={frame} is already completed.")

    def _nearest_to_ray(
        self,
        frame: int,
        start_zyx: np.ndarray,
        end_zyx: np.ndarray,
    ) -> tuple[Node, float]:
        rows = self.rows_by_frame.get(frame)
        if rows is None or rows.empty:
            raise AnnotationError(f"No detections are available at t={frame}.")

        active_indices = [
            index
            for index, row in rows.iterrows()
            if (frame, int(row.cell_id)) not in self.session.completed_nodes
        ]
        if not active_indices:
            raise AnnotationError(f"Every detection at t={frame} is already completed.")

        active = rows.loc[active_indices].reset_index(drop=True)
        centers = active[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(float)
        centers_um = centers * self.spacing[None, :]
        start_um = np.asarray(start_zyx, dtype=np.float64) * self.spacing
        end_um = np.asarray(end_zyx, dtype=np.float64) * self.spacing
        distances = _distance_points_to_segment(centers_um, start_um, end_um)
        best = int(np.argmin(distances))
        distance = float(distances[best])

        if self.max_ray_distance_um > 0 and distance > self.max_ray_distance_um:
            raise AnnotationError(
                f"Closest cell center is {distance:.2f} um from the click ray, "
                f"above the {self.max_ray_distance_um:.2f} um guard. Click closer "
                "to the target cell or increase --max-ray-distance-um."
            )

        node = (frame, int(active.iloc[best]["cell_id"]))
        return node, distance

    def pick(
        self,
        *,
        frame: int,
        event,
        binary_layer,
        reference_layer,
    ) -> tuple[Node, str]:
        ray = _ray_intersections_data(reference_layer, event)

        # Binary-mask mode: frontmost foreground is authoritative.
        if bool(binary_layer.visible):
            binary_ray = _ray_intersections_data(binary_layer, event) or ray
            if binary_ray is None:
                raise AnnotationError(
                    "Could not resolve a 3-D camera ray for Binary Mask picking."
                )
            hit = _first_foreground_point_along_ray(
                self.binary_mask,
                binary_ray[0],
                binary_ray[1],
            )
            if hit is None:
                raise AnnotationError("The click ray did not hit Binary Mask foreground.")

            index = np.rint(hit).astype(np.int64)
            index[0] = int(frame)
            shape = np.asarray(self.instances.shape, dtype=np.int64)
            index = np.clip(index, 0, shape - 1)
            instance_id = int(self.instances[tuple(index.tolist())])

            if instance_id > 0:
                node = (frame, instance_id)
                self.session._validate_node(node)
                if node in self.session.completed_nodes:
                    raise AnnotationError(
                        f"The frontmost mask hit is t={frame} instance={instance_id}, "
                        "which belongs to a completed track."
                    )
                return node, "binary-mask front hit"

            node = self._nearest_to_point(frame, hit[-3:])
            return node, "binary-mask hit -> nearest centroid"

        # Normal mode: closest center to the click ray.
        if ray is None:
            # 2-D / compatibility fallback: convert the click position to layer
            # data coordinates and use that as a degenerate ray point.
            try:
                point = np.asarray(reference_layer.world_to_data(event.position), dtype=float)
            except Exception as exc:
                raise AnnotationError("Could not resolve mouse position in data space.") from exc
            if point.size < 3:
                raise AnnotationError("Mouse position does not contain Z/Y/X coordinates.")
            node = self._nearest_to_point(frame, point[-3:])
            return node, "nearest centroid to click point"

        node, distance = self._nearest_to_ray(frame, ray[0][-3:], ray[1][-3:])
        return node, f"nearest centroid to ray ({distance:.2f} um)"


# =============================================================================
# Napari display helpers
# =============================================================================


def _display_color(label_id: int) -> tuple[float, float, float, float]:
    hue = (0.03 + int(label_id) * 0.6180339887498949) % 1.0
    saturation = 0.82 if int(label_id) % 2 == 0 else 0.70
    value = 0.98 if int(label_id) % 3 else 0.86
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return float(r), float(g), float(b), 0.62


def _apply_current_frame_colors(layer, labels: np.ndarray) -> None:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    mapping: dict[int, tuple[float, float, float, float]] = {
        0: (0.0, 0.0, 0.0, 0.0)
    }
    for label_id in ids.tolist():
        mapping[int(label_id)] = _display_color(int(label_id))
    try:
        layer.color = mapping
    except Exception:
        pass
    layer.refresh()


def _edge_tracks_array(
    edges: Iterable[Edge],
    centers: dict[Node, np.ndarray],
    hidden_nodes: set[Node],
) -> np.ndarray:
    rows: list[list[float]] = []
    track_id = 1
    for left, right in sorted(edges):
        if left in hidden_nodes or right in hidden_nodes:
            continue
        if left not in centers or right not in centers:
            continue
        rows.append([float(track_id), float(left[0]), *centers[left].tolist()])
        rows.append([float(track_id), float(right[0]), *centers[right].tolist()])
        track_id += 1
    if not rows:
        return np.empty((0, 5), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _active_points_array(
    centers: dict[Node, np.ndarray],
    hidden_nodes: set[Node],
) -> np.ndarray:
    rows = [
        [float(node[0]), *center.tolist()]
        for node, center in sorted(centers.items())
        if node not in hidden_nodes
    ]
    return np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 4), float)


def _make_label_shrinkable(widget) -> None:
    native = getattr(widget, "native", None)
    if native is None:
        return
    try:
        native.setWordWrap(True)
        native.setMinimumWidth(0)
    except Exception:
        pass
    if QSizePolicy is not None:
        try:
            native.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        except Exception:
            pass


# =============================================================================
# Viewer
# =============================================================================


def open_viewer(
    *,
    source: SourcePaths,
    output: OutputPaths,
    sample_id: str,
    spacing_zyx: tuple[float, float, float],
    max_ray_distance_um: float,
    resume: bool,
) -> None:
    raw = np.load(source.raw, mmap_mode="r", allow_pickle=False)
    binary_mask = np.load(source.binary_mask, mmap_mode="r", allow_pickle=False)
    instances = np.load(source.final_instances, mmap_mode="r", allow_pickle=False)
    _validate_source_arrays(raw, binary_mask, instances)

    cells = _normalize_cells(pd.read_csv(source.cells_csv))
    tracks = _normalize_tracks(pd.read_csv(source.tracks_csv))
    lineage_payload = json.loads(source.napari_graph.read_text(encoding="utf-8"))
    if not isinstance(lineage_payload, dict):
        raise AnnotationError(
            f"Expected dict in {source.napari_graph}, got "
            f"{type(lineage_payload).__name__}."
        )

    valid_nodes: set[Node] = {
        (int(row.frame), int(row.cell_id))
        for row in cells.itertuples(index=False)
    }
    centers: dict[Node, np.ndarray] = {
        (int(row.frame), int(row.cell_id)): np.asarray(
            [row.centroid_z, row.centroid_y, row.centroid_x],
            dtype=np.float64,
        )
        for row in cells.itertuples(index=False)
    }
    base_edges = build_trackastra_detection_edges(tracks, lineage_payload)

    session = TrackAnnotationSession(
        sample_id=sample_id,
        source_root=source.root,
        output=output,
        valid_nodes=valid_nodes,
        base_edges=base_edges,
        resume=resume,
    )
    picker = DetectionPicker(
        cells=cells,
        instances=instances,
        binary_mask=binary_mask,
        spacing_zyx=spacing_zyx,
        max_ray_distance_um=max_ray_distance_um,
        session=session,
    )

    scale_tzyx = (1.0, *spacing_zyx)
    spatial_shape = tuple(int(v) for v in raw.shape[-3:])

    viewer = napari.Viewer(ndisplay=3)
    low, high = np.percentile(np.asarray(raw), [1.0, 99.8])
    raw_layer = viewer.add_image(
        raw,
        name="Raw Volume",
        scale=scale_tzyx,
        rendering="mip",
        colormap="gray",
        contrast_limits=[float(low), float(high)],
    )
    binary_layer = viewer.add_labels(
        binary_mask,
        name="Binary Mask",
        scale=scale_tzyx,
        visible=False,
        opacity=0.35,
    )

    # Current-frame-only candidate layer. It is rebuilt when t changes so
    # completed detections can disappear without copying the full 4-D label movie.
    active_cells_layer = viewer.add_labels(
        np.zeros(spatial_shape, dtype=np.int32),
        name="Active Spatial Instances",
        scale=spacing_zyx,
        opacity=0.62,
    )

    if source.tracked_masks.is_file():
        tracked_masks = np.load(source.tracked_masks, mmap_mode="r", allow_pickle=False)
        viewer.add_labels(
            tracked_masks,
            name="Trackastra Tracked Masks (original)",
            scale=scale_tzyx,
            visible=False,
            opacity=0.45,
        )

    if source.napari_tracks.is_file():
        original_tracks = np.load(source.napari_tracks, mmap_mode="r", allow_pickle=False)
        original_layer = viewer.add_tracks(
            np.asarray(original_tracks),
            name="Trackastra Tracks (original)",
            scale=scale_tzyx,
            tail_length=int(raw.shape[0]),
        )
        original_layer.visible = False

    corrected_tracks_layer = viewer.add_tracks(
        _edge_tracks_array(session.active_edges, centers, session.completed_nodes),
        name="Corrected Edges - active",
        scale=scale_tzyx,
        tail_length=int(raw.shape[0]),
    )
    corrected_tracks_layer.visible = True

    active_points_layer = viewer.add_points(
        _active_points_array(centers, session.completed_nodes),
        name="Centroids - active",
        scale=scale_tzyx,
        size=3.5,
        face_color="red",
        opacity=0.8,
    )

    selection_a_layer = viewer.add_labels(
        np.zeros(spatial_shape, dtype=np.uint8),
        name="Selected Cell A",
        scale=spacing_zyx,
        opacity=0.88,
        color={0: (0, 0, 0, 0), 1: (1.0, 0.95, 0.05, 1.0)},
    )
    selection_b_layer = viewer.add_labels(
        np.zeros(spatial_shape, dtype=np.uint8),
        name="Selected Cell B",
        scale=spacing_zyx,
        opacity=0.88,
        color={0: (0, 0, 0, 0), 1: (1.0, 0.05, 0.75, 1.0)},
    )

    frame_label = Label(value="")
    selection_label = Label(value="Selected: none")
    graph_label = Label(value="")
    status_label = Label(
        value=(
            "Click cell A, move in time, click cell B, then Connect or Break. "
            "Escape resets the pair."
        )
    )
    connect_button = PushButton(text="Connect")
    break_button = PushButton(text="Break")
    complete_button = PushButton(text="Complete Track")
    undo_button = PushButton(text="Undo")
    reset_button = PushButton(text="Reset")

    for widget in (frame_label, selection_label, graph_label, status_label):
        _make_label_shrinkable(widget)

    panel = Container(
        widgets=[
            frame_label,
            selection_label,
            graph_label,
            connect_button,
            break_button,
            complete_button,
            undo_button,
            reset_button,
            status_label,
        ],
        layout="vertical",
    )
    dock = viewer.window.add_dock_widget(panel, name="Track annotation", area="right")
    try:
        dock.setMinimumWidth(300)
    except Exception:
        pass

    last_frame = [-1]

    def current_frame() -> int:
        return int(round(viewer.dims.current_step[0]))

    def refresh_selection_layers() -> None:
        frame = current_frame()
        zero = np.zeros(spatial_shape, dtype=np.uint8)
        layers = (selection_a_layer, selection_b_layer)
        for slot, layer in enumerate(layers):
            if slot < len(session.selections):
                node = session.selections[slot]
                if node[0] == frame:
                    layer.data = (np.asarray(instances[frame]) == node[1]).astype(
                        np.uint8, copy=False
                    )
                else:
                    layer.data = zero
            else:
                layer.data = zero
            layer.refresh()

    def refresh_current_candidate_cells() -> None:
        frame = current_frame()
        frame_labels = np.asarray(instances[frame])
        completed_ids = {
            node[1] for node in session.completed_nodes if node[0] == frame
        }
        if completed_ids:
            display = np.asarray(frame_labels).copy()
            display[np.isin(display, np.fromiter(completed_ids, dtype=np.int64))] = 0
        else:
            display = np.asarray(frame_labels)
        active_cells_layer.data = display
        _apply_current_frame_colors(active_cells_layer, display)

    def refresh_graph_layers() -> None:
        corrected_tracks_layer.data = _edge_tracks_array(
            session.active_edges,
            centers,
            session.completed_nodes,
        )
        corrected_tracks_layer.refresh()
        active_points_layer.data = _active_points_array(
            centers,
            session.completed_nodes,
        )
        active_points_layer.refresh()

    def refresh_status_labels() -> None:
        frame = current_frame()
        active_here = sum(
            1 for node in valid_nodes if node[0] == frame and node not in session.completed_nodes
        )
        completed_here = sum(1 for node in session.completed_nodes if node[0] == frame)
        frame_label.value = (
            f"Frame t={frame} / {raw.shape[0] - 1} | active cells={active_here} | "
            f"completed here={completed_here}"
        )
        if session.selections:
            selection_label.value = "Selected: " + " | ".join(
                f"{chr(65 + i)}=(t={node[0]}, id={node[1]})"
                for i, node in enumerate(session.selections)
            )
        else:
            selection_label.value = "Selected: none"
        graph_label.value = (
            f"Graph: active edges={len(session.active_edges)} | "
            f"manual connects={len(session.forced_edges)} | "
            f"manual breaks={len(session.broken_edges)} | "
            f"completed nodes={len(session.completed_nodes)} | "
            f"undo depth={len(session.history)}"
        )
        try:
            undo_button.enabled = bool(session.history)
            connect_button.enabled = len(session.selections) == 2
            break_button.enabled = len(session.selections) == 2
            complete_button.enabled = bool(session.selections)
        except Exception:
            pass

    def refresh_all(*, force_candidate: bool = True) -> None:
        if force_candidate:
            refresh_current_candidate_cells()
        refresh_selection_layers()
        refresh_graph_layers()
        refresh_status_labels()

    def show_error(exc: Exception) -> None:
        status_label.value = "ERROR: " + str(exc)
        print("\n[track annotation error]")
        print(exc)

    @viewer.mouse_drag_callbacks.append
    def ray_pick_cell(_viewer, event):
        button = getattr(event, "button", None)
        button_text = str(button).lower()
        is_left = (
            button is None
            or button == 1
            or button_text == "1"
            or "left" in button_text
        )
        if not is_left:
            return

        dragged = False
        yield
        while getattr(event, "type", None) == "mouse_move":
            dragged = True
            yield
        if dragged:
            return

        try:
            node, mode = picker.pick(
                frame=current_frame(),
                event=event,
                binary_layer=binary_layer,
                reference_layer=raw_layer,
            )
            session.add_selection(node)
        except Exception as exc:
            show_error(exc)
            return

        refresh_selection_layers()
        refresh_status_labels()
        slot = chr(64 + len(session.selections))
        status_label.value = (
            f"Selected {slot}: t={node[0]} instance={node[1]} via {mode}."
        )

    def connect_selected() -> None:
        try:
            edge = session.connect_selected()
        except Exception as exc:
            show_error(exc)
            return
        refresh_all(force_candidate=False)
        status_label.value = (
            f"CONNECTED t={edge[0][0]} id={edge[0][1]} -> "
            f"t={edge[1][0]} id={edge[1][1]} "
            f"(gap={edge[1][0] - edge[0][0]})."
        )

    def break_selected() -> None:
        try:
            edge = session.break_selected()
        except Exception as exc:
            show_error(exc)
            return
        refresh_all(force_candidate=False)
        status_label.value = (
            f"BROKE t={edge[0][0]} id={edge[0][1]} -> "
            f"t={edge[1][0]} id={edge[1][1]}."
        )

    def complete_selected() -> None:
        try:
            nodes = session.complete_selected_components()
        except Exception as exc:
            show_error(exc)
            return
        refresh_all(force_candidate=True)
        status_label.value = (
            f"Completed and hid corrected component: {len(nodes)} newly hidden "
            "detections."
        )

    def undo_last() -> None:
        try:
            op = session.undo()
        except Exception as exc:
            show_error(exc)
            return
        focus_frame = int(op.get("focus_frame", current_frame()))
        focus_frame = int(np.clip(focus_frame, 0, raw.shape[0] - 1))
        viewer.dims.set_current_step(0, focus_frame)
        last_frame[0] = -1
        refresh_all(force_candidate=True)
        status_label.value = f"Undid last {op.get('type', 'operation')}."

    def reset_selected() -> None:
        session.reset_selections()
        refresh_selection_layers()
        refresh_status_labels()
        status_label.value = "Selections reset."

    connect_button.clicked.connect(connect_selected)
    break_button.clicked.connect(break_selected)
    complete_button.clicked.connect(complete_selected)
    undo_button.clicked.connect(undo_last)
    reset_button.clicked.connect(reset_selected)

    @viewer.bind_key("Escape", overwrite=True)
    def _reset_shortcut(_viewer):
        reset_selected()

    def on_dims_change(_event=None) -> None:
        frame = current_frame()
        if frame == last_frame[0]:
            return
        last_frame[0] = frame
        refresh_current_candidate_cells()
        refresh_selection_layers()
        refresh_status_labels()

    viewer.dims.events.current_step.connect(on_dims_change)

    # Initial state and resume selection highlights.
    refresh_all(force_candidate=True)

    print("=" * 96)
    print("BIOHUB TRACK ANNOTATOR")
    print("=" * 96)
    print(f"sample             : {sample_id}")
    print(f"source             : {source.root}")
    print(f"output             : {output.root}")
    print(f"detections         : {len(valid_nodes)}")
    print(f"Trackastra edges   : {len(base_edges)}")
    print(f"manual connects    : {len(session.forced_edges)}")
    print(f"manual breaks      : {len(session.broken_edges)}")
    print(f"completed nodes    : {len(session.completed_nodes)}")
    print(f"max ray distance   : {max_ray_distance_um} um")
    print("Binary Mask visible: frontmost-mask-hit picking")
    print("Binary Mask hidden : closest-centroid-to-ray picking")
    print("=" * 96)

    napari.run()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    source, output = resolve_paths(args)
    spacing = _parse_spacing(args.spacing_zyx)
    open_viewer(
        source=source,
        output=output,
        sample_id=args.sample_id,
        spacing_zyx=spacing,
        max_ray_distance_um=float(args.max_ray_distance_um),
        resume=not bool(args.no_resume),
    )


if __name__ == "__main__":
    main()
