from __future__ import annotations

r"""
Investigation 40 — visualize Investigation-35 concrete training data quality.

Purpose
-------
This script inspects the synthetic temporal training data created by:

    investigations/stirnet/35_biohub_temporal_merge_synthesis.py

It is designed to answer questions like:

    - Are the selected A+B merge cases visually correct?
    - Are many training events "ideal isolated" one-frame merges?
    - How many events are repeated across neighboring frames?
    - How often is one of the partner cells also corrupted in t-1 / t+1?
    - What does Trackastra actually see on the concrete corrupted movie?
    - Does the selected synthetic distribution resemble the real BioHub task?

The viewer focuses on ONE concrete variant at a time.

Displayed data
--------------
    Raw Volume
    Manual True Instances
    Synthetic Training Instances
    Synthetic Event Points
    Trackastra Evidence Masks
    Trackastra Evidence Tracks
    Selected A
    Selected B
    Selected A+B Union
    Selected Trackastra Overlap

The selected event can be changed using keyboard shortcuts:

    N / ]     next event
    P / [     previous event
"""

import argparse
import dataclasses
import importlib.util
import json
import os
import pickle
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents, Path.cwd().resolve()):
        if (
            (candidate / "learned" / "stirnet").is_dir()
            and (candidate / "investigations" / "stirnet").is_dir()
            and (candidate / "src").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    raise RuntimeError("Could not resolve repository root")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INV35 = load_module(
    ROOT / "investigations" / "stirnet" / "35_biohub_temporal_merge_synthesis.py",
    "_inv40_current_inv35",
)

SCRIPT_NAME = "40_biohub_training_data_visualization"
DEFAULT_SAMPLE_ID = "44b6_0113de3b"
DEFAULT_FRAME_COUNT = 20
DEFAULT_TRACKASTRA_MODEL = "ctc"
DEFAULT_TRACKASTRA_MODE = "greedy"
DEFAULT_TRACKASTRA_DEVICE = "cuda"


def format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds >= 3600:
        return f"{seconds / 3600.0:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds:.1f} s"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def centroid_zyx(mask: np.ndarray) -> tuple[float, float, float]:
    zyx = np.argwhere(mask)
    if len(zyx) == 0:
        return (float("nan"), float("nan"), float("nan"))
    center = zyx.mean(axis=0)
    return float(center[0]), float(center[1]), float(center[2])


def unique_positive(values: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(values) if int(v) > 0]


@dataclass(frozen=True)
class MergeEventRecord:
    variant_index: int
    split: str
    event_index: int
    frame: int
    a: int
    b: int
    representative: int
    interface_edges: int
    category: str
    repeated_prev: bool
    repeated_next: bool
    neighbor_corrupted_prev: bool
    neighbor_corrupted_next: bool
    isolated_prev: bool
    isolated_next: bool
    voxels_a: int
    voxels_b: int
    volume_ratio: float
    union_voxels: int
    overlap_track_labels: tuple[int, ...]
    overlap_track_count: int
    track_node_count: int
    indegree_sum: int
    outdegree_sum: int
    centroid_z: float
    centroid_y: float
    centroid_x: float


@dataclass(frozen=True)
class Paths:
    sample_id: str
    inv35_output: Path
    variant_index: int
    inspection_root: Path

    @property
    def manual_movie(self) -> Path:
        return self.inv35_output / "movies" / "manual.npy"

    @property
    def raw_movie(self) -> Path:
        return self.inv35_output / "movies" / "raw.npy"

    @property
    def touching_pairs(self) -> Path:
        return self.inv35_output / "touching_pairs.json"

    @property
    def variant_dir(self) -> Path:
        return self.inv35_output / "variants" / f"variant_{self.variant_index:03d}"

    @property
    def merge_plan(self) -> Path:
        return self.variant_dir / "merge_plan.json"

    @property
    def track_graph(self) -> Path:
        return self.variant_dir / "track_graph.pkl"

    @property
    def synthetic_movie(self) -> Path:
        return self.inspection_root / "synthetic_instances.npy"

    @property
    def tracked_masks(self) -> Path:
        return self.inspection_root / "tracked_masks.npy"

    @property
    def napari_tracks(self) -> Path:
        return self.inspection_root / "napari_tracks.npy"

    @property
    def napari_graph(self) -> Path:
        return self.inspection_root / "napari_graph.json"

    @property
    def summary_json(self) -> Path:
        return self.inspection_root / "summary.json"

    @property
    def events_csv(self) -> Path:
        return self.inspection_root / "events.csv"

    @property
    def success_json(self) -> Path:
        return self.inspection_root / "_SUCCESS.json"


def default_inv35_output(sample_id: str) -> Path:
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / "35_biohub_temporal_merge_synthesis"
        / sample_id
        / "concrete_cutkeep_v3"
    ).resolve()


def default_output_root(sample_id: str, variant_index: int) -> Path:
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / SCRIPT_NAME
        / sample_id
        / f"variant_{variant_index:03d}"
    ).resolve()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def variant_from_split(inv35_output: Path, split: str, ordinal: int) -> int:
    manifest = load_json(inv35_output / "dataset_manifest.json")
    split = str(split).lower()
    indices = [int(plan["index"]) for plan in manifest["plans"] if str(plan["split"]).lower() == split]
    if not indices:
        raise ValueError(f"No variants found for split={split!r}")
    if ordinal < 0 or ordinal >= len(indices):
        raise ValueError(f"Split {split!r} has only {len(indices)} variants; ordinal {ordinal} is invalid")
    return int(indices[ordinal])


def load_plan(path: Path) -> tuple[str, dict[int, list[dict[str, Any]]]]:
    payload = load_json(path)
    split = str(payload["split"])
    events = {int(t): [dict(row) for row in rows] for t, rows in payload["events_by_frame"].items()}
    return split, events


def load_touching_catalog(path: Path) -> dict[int, dict[tuple[int, int], dict[str, Any]]]:
    payload = load_json(path)
    out: dict[int, dict[tuple[int, int], dict[str, Any]]] = {}
    for t, rows in payload["frames"].items():
        table: dict[tuple[int, int], dict[str, Any]] = {}
        for row in rows:
            a = int(row["a"])
            b = int(row["b"])
            table[(min(a, b), max(a, b))] = dict(row)
        out[int(t)] = table
    return out


def movie_shape_ok(path: Path, expected_t: int) -> bool:
    if not path.is_file():
        return False
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return int(arr.shape[0]) == int(expected_t)
    except Exception:
        return False


def inspection_cache_complete(paths: Paths, frame_count: int) -> bool:
    return (
        movie_shape_ok(paths.synthetic_movie, frame_count)
        and movie_shape_ok(paths.tracked_masks, frame_count)
        and paths.napari_tracks.is_file()
        and paths.napari_graph.is_file()
        and paths.events_csv.is_file()
        and paths.summary_json.is_file()
        and paths.success_json.is_file()
    )


def build_synthetic_movie(*, paths: Paths, frame_count: int, rebuild: bool) -> None:
    if movie_shape_ok(paths.synthetic_movie, frame_count) and not rebuild:
        return

    manual = np.load(paths.manual_movie, mmap_mode="r", allow_pickle=False)
    _split, events_by_frame = load_plan(paths.merge_plan)

    paths.inspection_root.mkdir(parents=True, exist_ok=True)
    first = np.asarray(manual[0])
    synthetic = np.lib.format.open_memmap(
        paths.synthetic_movie,
        mode="w+",
        dtype=np.int32,
        shape=(frame_count, *first.shape),
    )
    for t in range(frame_count):
        synthetic[t] = INV35.apply_merge_events(
            np.asarray(manual[t]),
            [
                INV35.MergeEvent(
                    frame=int(row["frame"]),
                    a=int(row["a"]),
                    b=int(row["b"]),
                    representative=int(row["representative"]),
                    interface_edges=int(row["interface_edges"]),
                )
                for row in events_by_frame.get(t, [])
            ],
        )
    synthetic.flush()


def rerun_trackastra_if_needed(
    *,
    paths: Paths,
    frame_count: int,
    model_name: str,
    mode: str,
    device: str,
    rebuild: bool,
) -> None:
    if (
        movie_shape_ok(paths.tracked_masks, frame_count)
        and paths.napari_tracks.is_file()
        and paths.napari_graph.is_file()
        and not rebuild
    ):
        print("[trackastra] reusing inspection cache", flush=True)
        return

    try:
        from trackastra.model import Trackastra
        from trackastra.tracking.utils import graph_to_napari_tracks
    except ImportError as exc:
        raise RuntimeError("Trackastra is required for Investigation 40") from exc

    raw = np.load(paths.raw_movie, mmap_mode="r", allow_pickle=False)
    synthetic = np.load(paths.synthetic_movie, mmap_mode="r", allow_pickle=False)

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 40 — RERUN TRACKASTRA FOR VISUALIZATION", flush=True)
    print("=" * 112, flush=True)
    print(f"variant  : {paths.variant_index:03d}", flush=True)
    print(f"model    : {model_name}", flush=True)
    print(f"mode     : {mode}", flush=True)
    print(f"device   : {device}", flush=True)
    print("=" * 112, flush=True)

    started = time.perf_counter()
    tracker = Trackastra.from_pretrained(model_name, device=device)
    graph, tracked_masks = tracker.track(raw, synthetic, mode=mode)
    INV35.annotate_graph_fast(graph, np.asarray(tracked_masks))

    with paths.track_graph.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(paths.tracked_masks, np.asarray(tracked_masks), allow_pickle=False)
    napari_tracks, napari_graph, _props = graph_to_napari_tracks(graph)
    np.save(paths.napari_tracks, np.asarray(napari_tracks, dtype=np.float64), allow_pickle=False)
    serializable_graph = {
        str(int(child)): ([int(v) for v in parent] if isinstance(parent, (list, tuple, set)) else int(parent))
        for child, parent in napari_graph.items()
    }
    atomic_json(paths.napari_graph, serializable_graph)
    elapsed = time.perf_counter() - started
    print(f"[trackastra] nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} time={format_seconds(elapsed)}", flush=True)


def build_frame_event_maps(events_by_frame: dict[int, list[dict[str, Any]]]) -> tuple[dict[int, set[tuple[int, int]]], dict[int, set[int]]]:
    pair_map: dict[int, set[tuple[int, int]]] = {}
    cell_map: dict[int, set[int]] = {}
    for t, rows in events_by_frame.items():
        pair_map[int(t)] = {(min(int(row["a"]), int(row["b"])), max(int(row["a"]), int(row["b"]))) for row in rows}
        cell_map[int(t)] = {int(v) for row in rows for v in (row["a"], row["b"])}
    return pair_map, cell_map


def event_category(
    *,
    repeated_prev: bool,
    repeated_next: bool,
    neighbor_corrupted_prev: bool,
    neighbor_corrupted_next: bool,
    frame: int,
    frame_count: int,
) -> tuple[str, bool, bool]:
    isolated_prev = frame == 0 or not neighbor_corrupted_prev
    isolated_next = frame == frame_count - 1 or not neighbor_corrupted_next
    if repeated_prev or repeated_next:
        return "REPEATED_PAIR", isolated_prev, isolated_next
    if isolated_prev and isolated_next:
        return "IDEAL_ISOLATED", isolated_prev, isolated_next
    if isolated_prev != isolated_next:
        return "ONE_SIDED_ADJACENT", isolated_prev, isolated_next
    return "NEIGHBOR_CORRUPTED", isolated_prev, isolated_next


def overlap_track_labels_for_union(tracked_frame: np.ndarray, union_mask: np.ndarray) -> tuple[int, ...]:
    return tuple(unique_positive(np.asarray(tracked_frame)[np.asarray(union_mask)]))


def graph_degree_maps(graph) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    indegree: dict[tuple[int, int], int] = defaultdict(int)
    outdegree: dict[tuple[int, int], int] = defaultdict(int)
    for source, destination in graph.edges():
        s = graph.nodes[int(source)]
        d = graph.nodes[int(destination)]
        outdegree[(int(s["time"]), int(s["label"]))] += 1
        indegree[(int(d["time"]), int(d["label"]))] += 1
    return indegree, outdegree


def build_event_table(*, paths: Paths, frame_count: int) -> pd.DataFrame:
    split, events_by_frame = load_plan(paths.merge_plan)
    touching = load_touching_catalog(paths.touching_pairs)
    manual = np.load(paths.manual_movie, mmap_mode="r", allow_pickle=False)
    tracked = np.load(paths.tracked_masks, mmap_mode="r", allow_pickle=False)

    with paths.track_graph.open("rb") as handle:
        graph = pickle.load(handle)

    indegree_map, outdegree_map = graph_degree_maps(graph)
    pair_map, cell_map = build_frame_event_maps(events_by_frame)

    nodes_by_tl: dict[tuple[int, int], list[int]] = defaultdict(list)
    for node_id, data in graph.nodes(data=True):
        nodes_by_tl[(int(data["time"]), int(data["label"]))].append(int(node_id))

    rows: list[dict[str, Any]] = []
    event_index = 0

    for t in range(frame_count):
        current_rows = events_by_frame.get(t, [])
        prev_pairs = pair_map.get(t - 1, set())
        next_pairs = pair_map.get(t + 1, set())
        prev_cells = cell_map.get(t - 1, set())
        next_cells = cell_map.get(t + 1, set())

        for row in current_rows:
            a = int(row["a"])
            b = int(row["b"])
            representative = int(row["representative"])
            key = (min(a, b), max(a, b))
            manual_frame = np.asarray(manual[t])

            mask_a = manual_frame == a
            mask_b = manual_frame == b
            union_mask = mask_a | mask_b

            voxels_a = int(mask_a.sum())
            voxels_b = int(mask_b.sum())
            union_voxels = int(union_mask.sum())
            volume_ratio = max(voxels_a / max(voxels_b, 1), voxels_b / max(voxels_a, 1))
            center = centroid_zyx(union_mask)

            repeated_prev = key in prev_pairs
            repeated_next = key in next_pairs
            neighbor_corrupted_prev = (a in prev_cells) or (b in prev_cells)
            neighbor_corrupted_next = (a in next_cells) or (b in next_cells)
            category, isolated_prev, isolated_next = event_category(
                repeated_prev=repeated_prev,
                repeated_next=repeated_next,
                neighbor_corrupted_prev=neighbor_corrupted_prev,
                neighbor_corrupted_next=neighbor_corrupted_next,
                frame=t,
                frame_count=frame_count,
            )

            overlap_labels = overlap_track_labels_for_union(np.asarray(tracked[t]), union_mask)
            node_count = 0
            indegree_sum = 0
            outdegree_sum = 0
            for label in overlap_labels:
                nodes = nodes_by_tl.get((t, int(label)), [])
                node_count += len(nodes)
                indegree_sum += indegree_map.get((t, int(label)), 0)
                outdegree_sum += outdegree_map.get((t, int(label)), 0)

            catalog_row = touching.get(t, {}).get(key, None)
            interface_edges = int(catalog_row.get("interface_edges", row.get("interface_edges", 0))) if catalog_row is not None else int(row.get("interface_edges", 0))

            rows.append(dataclasses.asdict(MergeEventRecord(
                variant_index=int(paths.variant_index),
                split=split,
                event_index=int(event_index),
                frame=int(t),
                a=a,
                b=b,
                representative=representative,
                interface_edges=interface_edges,
                category=category,
                repeated_prev=bool(repeated_prev),
                repeated_next=bool(repeated_next),
                neighbor_corrupted_prev=bool(neighbor_corrupted_prev),
                neighbor_corrupted_next=bool(neighbor_corrupted_next),
                isolated_prev=bool(isolated_prev),
                isolated_next=bool(isolated_next),
                voxels_a=voxels_a,
                voxels_b=voxels_b,
                volume_ratio=float(volume_ratio),
                union_voxels=union_voxels,
                overlap_track_labels=tuple(int(v) for v in overlap_labels),
                overlap_track_count=int(len(overlap_labels)),
                track_node_count=int(node_count),
                indegree_sum=int(indegree_sum),
                outdegree_sum=int(outdegree_sum),
                centroid_z=float(center[0]),
                centroid_y=float(center[1]),
                centroid_x=float(center[2]),
            )))
            event_index += 1

    table = pd.DataFrame(rows)
    atomic_csv(paths.events_csv, table)
    return table


def summarize_event_table(table: pd.DataFrame, *, paths: Paths, frame_count: int) -> None:
    split = str(table["split"].iloc[0]) if not table.empty else "unknown"
    summary = {
        "sample_id": paths.sample_id,
        "variant_index": int(paths.variant_index),
        "split": split,
        "frame_count": int(frame_count),
        "event_count": int(len(table)),
        "category_counts": {str(k): int(v) for k, v in table["category"].value_counts().sort_index().to_dict().items()} if not table.empty else {},
        "mean_events_per_frame": float(len(table) / frame_count) if frame_count else 0.0,
        "frames_with_no_events": int(frame_count - table["frame"].nunique()) if not table.empty else int(frame_count),
        "repeated_pair_fraction": float(((table["repeated_prev"]) | (table["repeated_next"])).mean()) if not table.empty else 0.0,
        "neighbor_corrupted_fraction": float((((table["neighbor_corrupted_prev"]) | (table["neighbor_corrupted_next"])).mean())) if not table.empty else 0.0,
        "ideal_isolated_fraction": float((table["category"] == "IDEAL_ISOLATED").mean()) if not table.empty else 0.0,
        "mean_overlap_track_count": float(table["overlap_track_count"].mean()) if not table.empty else 0.0,
        "median_union_voxels": float(table["union_voxels"].median()) if not table.empty else 0.0,
    }
    atomic_json(paths.summary_json, summary)
    atomic_json(paths.success_json, {"status": "success"})


def print_summary(paths: Paths) -> None:
    summary = load_json(paths.summary_json)
    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 40 — TRAINING-DATA QUALITY SUMMARY", flush=True)
    print("=" * 112, flush=True)
    print(f"sample                    : {summary['sample_id']}", flush=True)
    print(f"variant                   : {summary['variant_index']:03d} ({summary['split']})", flush=True)
    print(f"events                    : {summary['event_count']}", flush=True)
    print(f"mean events / frame       : {summary['mean_events_per_frame']:.2f}", flush=True)
    print(f"frames with no events     : {summary['frames_with_no_events']}", flush=True)
    print(f"ideal isolated fraction   : {100.0 * summary['ideal_isolated_fraction']:.1f}%", flush=True)
    print(f"repeated pair fraction    : {100.0 * summary['repeated_pair_fraction']:.1f}%", flush=True)
    print(f"neighbor corrupted frac   : {100.0 * summary['neighbor_corrupted_fraction']:.1f}%", flush=True)
    print(f"mean Trackastra overlap   : {summary['mean_overlap_track_count']:.2f}", flush=True)
    print(f"median union voxels       : {summary['median_union_voxels']:.1f}", flush=True)
    categories = summary.get("category_counts", {})
    if categories:
        print("category counts           :", flush=True)
        for key, value in sorted(categories.items()):
            print(f"  {key:<22} {value}", flush=True)
    print("=" * 112, flush=True)


def build_points_properties(table: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "event_index": table["event_index"].to_numpy(np.int64),
        "frame": table["frame"].to_numpy(np.int64),
        "a": table["a"].to_numpy(np.int64),
        "b": table["b"].to_numpy(np.int64),
        "category": table["category"].astype(str).to_numpy(),
        "union_voxels": table["union_voxels"].to_numpy(np.int64),
        "track_overlap": table["overlap_track_count"].to_numpy(np.int64),
    }


def build_face_colors(categories: Sequence[str]) -> list[str]:
    colors: list[str] = []
    for category in categories:
        if category == "IDEAL_ISOLATED":
            colors.append("lime")
        elif category == "ONE_SIDED_ADJACENT":
            colors.append("yellow")
        elif category == "NEIGHBOR_CORRUPTED":
            colors.append("orange")
        elif category == "REPEATED_PAIR":
            colors.append("magenta")
        else:
            colors.append("cyan")
    return colors


def mask_from_labels(labels: np.ndarray, ids: Iterable[int]) -> np.ndarray:
    result = np.zeros_like(np.asarray(labels), dtype=np.uint8)
    ids = [int(v) for v in ids if int(v) > 0]
    if not ids:
        return result
    result[np.isin(np.asarray(labels), np.asarray(ids))] = 1
    return result


def open_viewer(*, paths: Paths, spacing: tuple[float, float, float]) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError("Napari is required for Investigation 40") from exc

    raw = np.load(paths.raw_movie, mmap_mode="r", allow_pickle=False)
    manual = np.load(paths.manual_movie, mmap_mode="r", allow_pickle=False)
    synthetic = np.load(paths.synthetic_movie, mmap_mode="r", allow_pickle=False)
    tracked = np.load(paths.tracked_masks, mmap_mode="r", allow_pickle=False)
    tracks = np.load(paths.napari_tracks, allow_pickle=False)
    table = pd.read_csv(paths.events_csv)

    scale = (1.0, *spacing)
    viewer = napari.Viewer(ndisplay=3)

    low, high = np.percentile(np.asarray(raw), [1.0, 99.8])
    viewer.add_image(raw, name="Raw Volume", scale=scale, rendering="mip", colormap="gray", contrast_limits=[float(low), float(high)])
    viewer.add_labels(manual, name="Manual True Instances", scale=scale, opacity=1.0, visible=False)
    viewer.add_labels(synthetic, name="Synthetic Training Instances", scale=scale, opacity=1.0, visible=True)
    viewer.add_labels(tracked, name="Trackastra Evidence Masks", scale=scale, opacity=1.0, visible=False)

    if tracks.size:
        tracks_layer = viewer.add_tracks(tracks, name="Trackastra Evidence Tracks", scale=scale, tail_length=20)
        tracks_layer.visible = False

    points = np.column_stack([
        table["frame"].to_numpy(float),
        table["centroid_z"].to_numpy(float),
        table["centroid_y"].to_numpy(float),
        table["centroid_x"].to_numpy(float),
    ]) if not table.empty else np.zeros((0, 4), dtype=float)

    points_layer = viewer.add_points(
        points,
        name="Synthetic Event Points",
        scale=scale,
        size=5,
        face_color=build_face_colors(table["category"].astype(str).tolist()),
        properties=build_points_properties(table),
        text={"string": "{event_index}", "size": 8, "color": "white"},
    )

    selected_a_layer = viewer.add_labels(np.zeros_like(np.asarray(manual), dtype=np.uint8), name="Selected A", scale=scale, opacity=1.0, visible=True)
    selected_b_layer = viewer.add_labels(np.zeros_like(np.asarray(manual), dtype=np.uint8), name="Selected B", scale=scale, opacity=1.0, visible=True)
    selected_union_layer = viewer.add_labels(np.zeros_like(np.asarray(manual), dtype=np.uint8), name="Selected A+B Union", scale=scale, opacity=1.0, visible=True)
    selected_overlap_layer = viewer.add_labels(np.zeros_like(np.asarray(tracked), dtype=np.uint8), name="Selected Trackastra Overlap", scale=scale, opacity=1.0, visible=False)

    current = {"row": 0 if len(table) else -1}

    def show_row(row_index: int) -> None:
        if len(table) == 0:
            return
        row_index = int(row_index) % len(table)
        current["row"] = row_index
        row = table.iloc[row_index]

        t = int(row["frame"])
        a = int(row["a"])
        b = int(row["b"])
        overlap_raw = str(row["overlap_track_labels"]).strip()

        viewer.dims.set_point(0, float(t))
        mask_a_4d = np.zeros_like(np.asarray(manual), dtype=np.uint8)
        mask_b_4d = np.zeros_like(np.asarray(manual), dtype=np.uint8)
        mask_u_4d = np.zeros_like(np.asarray(manual), dtype=np.uint8)
        mask_o_4d = np.zeros_like(np.asarray(tracked), dtype=np.uint8)

        frame_manual = np.asarray(manual[t])
        frame_tracked = np.asarray(tracked[t])

        mask_a = frame_manual == a
        mask_b = frame_manual == b
        mask_u = mask_a | mask_b
        mask_a_4d[t] = mask_a.astype(np.uint8)
        mask_b_4d[t] = mask_b.astype(np.uint8)
        mask_u_4d[t] = mask_u.astype(np.uint8)

        labels: list[int] = []
        if overlap_raw and overlap_raw not in ("()", "nan", "NaN"):
            cleaned = overlap_raw.strip("()")
            if cleaned:
                labels = [int(token.strip()) for token in cleaned.split(",") if token.strip()]
        if labels:
            mask_o_4d[t] = mask_from_labels(frame_tracked, labels)

        selected_a_layer.data = mask_a_4d
        selected_b_layer.data = mask_b_4d
        selected_union_layer.data = mask_u_4d
        selected_overlap_layer.data = mask_o_4d
        points_layer.selected_data = {row_index}

        print(
            f"[event {row_index:03d}] t={t:03d} A={a} B={b} "
            f"cat={row['category']} union={int(row['union_voxels'])} "
            f"track_overlap={int(row['overlap_track_count'])} "
            f"in={int(row['indegree_sum'])} out={int(row['outdegree_sum'])}",
            flush=True,
        )

    if len(table):
        show_row(0)

    @viewer.bind_key("]")
    @viewer.bind_key("n")
    def _next(_viewer):
        if len(table):
            show_row(current["row"] + 1)

    @viewer.bind_key("[")
    @viewer.bind_key("p")
    def _prev(_viewer):
        if len(table):
            show_row(current["row"] - 1)

    @points_layer.mouse_double_click_callbacks.append
    def _on_double_click(layer, event):
        if len(layer.selected_data) == 1:
            show_row(next(iter(layer.selected_data)))

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 40 — VIEWER HELP", flush=True)
    print("=" * 112, flush=True)
    print("N / ] : next synthetic merge event", flush=True)
    print("P / [ : previous synthetic merge event", flush=True)
    print("Green  points : ideal isolated events", flush=True)
    print("Yellow points : one-sided adjacent corruption", flush=True)
    print("Orange points : partner cell also corrupted in both adjacent directions", flush=True)
    print("Magenta points: same A+B pair repeated in neighboring frame(s)", flush=True)
    print("=" * 112, flush=True)

    napari.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize Investigation-35 synthetic temporal training data.")
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--spacing", default="1.625,0.40625,0.40625")
    parser.add_argument("--inv35-output", default=None)
    parser.add_argument("--variant-index", type=int, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--split-ordinal", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--trackastra-model", default=DEFAULT_TRACKASTRA_MODEL)
    parser.add_argument("--trackastra-mode", default=DEFAULT_TRACKASTRA_MODE)
    parser.add_argument("--trackastra-device", default=DEFAULT_TRACKASTRA_DEVICE)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--viewer-only", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    return parser


def parse_spacing(text: str) -> tuple[float, float, float]:
    rows = tuple(float(token.strip()) for token in str(text).split(","))
    if len(rows) != 3:
        raise ValueError("--spacing must have 3 comma-separated values")
    return rows


def main() -> int:
    args = build_parser().parse_args()

    spacing = parse_spacing(args.spacing)
    inv35_output = resolve(args.inv35_output) if args.inv35_output is not None else default_inv35_output(args.sample_id)
    if not inv35_output.is_dir():
        raise FileNotFoundError(inv35_output)

    variant_index = int(args.variant_index) if args.variant_index is not None else variant_from_split(inv35_output, args.split, int(args.split_ordinal))
    output = resolve(args.output) if args.output is not None else default_output_root(args.sample_id, variant_index)

    paths = Paths(
        sample_id=str(args.sample_id),
        inv35_output=inv35_output,
        variant_index=int(variant_index),
        inspection_root=output,
    )

    if not paths.manual_movie.is_file():
        raise FileNotFoundError(f"Manual movie not found: {paths.manual_movie}")
    if not paths.raw_movie.is_file():
        raise FileNotFoundError(f"Raw movie not found: {paths.raw_movie}")
    if not paths.merge_plan.is_file():
        raise FileNotFoundError(f"Merge plan not found: {paths.merge_plan}")

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 40 — TRAINING-DATA INSPECTION", flush=True)
    print("=" * 112, flush=True)
    print(f"repository      : {ROOT}", flush=True)
    print(f"sample          : {args.sample_id}", flush=True)
    print(f"inv35 output    : {inv35_output}", flush=True)
    print(f"variant index   : {variant_index:03d}", flush=True)
    print(f"inspection root : {output}", flush=True)
    print("=" * 112, flush=True)

    if args.viewer_only:
        if not inspection_cache_complete(paths, args.frame_count):
            raise FileNotFoundError(f"Incomplete inspection cache: {paths.inspection_root}")
    else:
        if args.rebuild and paths.inspection_root.exists():
            shutil.rmtree(paths.inspection_root)

        build_synthetic_movie(paths=paths, frame_count=args.frame_count, rebuild=bool(args.rebuild))
        rerun_trackastra_if_needed(
            paths=paths,
            frame_count=args.frame_count,
            model_name=args.trackastra_model,
            mode=args.trackastra_mode,
            device=args.trackastra_device,
            rebuild=bool(args.rebuild),
        )
        table = build_event_table(paths=paths, frame_count=args.frame_count)
        summarize_event_table(table, paths=paths, frame_count=args.frame_count)

    print_summary(paths)
    print(f"[artifacts] events table : {paths.events_csv}", flush=True)
    print(f"[artifacts] summary     : {paths.summary_json}", flush=True)

    if not args.no_viewer:
        open_viewer(paths=paths, spacing=spacing)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
