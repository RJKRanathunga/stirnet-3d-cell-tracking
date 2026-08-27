from __future__ import annotations

r"""
Investigation 38 — concrete BioHub merge variants + temporal CUT/KEEP training.

The dataset construction is deliberately simple:

    clean annotated BioHub instances
      -> randomly merge non-overlapping touching true-cell pairs for one frame
      -> run Trackastra on each concrete corrupted movie
      -> train the production temporal stack on those real Trackastra histories

Supervision is defined directly from the clean annotation ID carried by every
atomic RAG supervoxel:

    same true cell ID                              -> KEEP
    different true cell IDs inside one merged comp -> CUT

Temporal action is split-only relative to the current synthetic component:
cross-current-component RAG edges are immutable CUT.  The temporal branch is
therefore never asked to repair general spatial over-segmentation.

The raw movie is not duplicated on disk.  Each variant stores only its merge
plan, Trackastra graph, compact temporal metadata, and temporal graph files.  A
temporary synthetic label movie exists only while Trackastra is running.

Default dataset:
    8 train variants, 2 validation variants, 2 held-out test variants.

This script intentionally reuses the expensive real frozen spatial/RAG cache
already prepared by Investigation 35, but does not reuse its dynamic synthetic
merge graphs.
"""

import argparse
import dataclasses
import gc
import importlib.util
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents, Path.cwd().resolve()):
        if (
            (candidate / "learned" / "stirnet").is_dir()
            and (candidate / "investigations" / "stirnet").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    raise RuntimeError("Could not resolve repository root")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INV35 = load_module(
    ROOT / "investigations/stirnet/35_biohub_temporal_merge_synthesis.py",
    "_inv35_for_inv38",
)

from learned.stirnet.data.graph_builder import build_temporal_graph, sequence_available_time_offsets
from learned.stirnet.training import TrainingConfig
from learned.stirnet.training.checkpoint import save_checkpoint


SCRIPT_NAME = "38_biohub_concrete_merge_variant_training"
OBJECTIVE_VERSION = 1
DATASET_VERSION = 1
OBSERVER_VERSION = 1

DEFAULT_SAMPLE = "44b6_0113de3b"
DEFAULT_FRAMES = 20
DEFAULT_SPACING = (1.625, 0.40625, 0.40625)
DEFAULT_TEMPORAL_RADIUS = 2
DEFAULT_TRAIN_VARIANTS = 8
DEFAULT_VAL_VARIANTS = 2
DEFAULT_TEST_VARIANTS = 2
DEFAULT_MERGE_FRACTION = 0.10
DEFAULT_MAX_MERGES_PER_FRAME = 20
DEFAULT_MIN_VOXELS = 64
DEFAULT_MAX_VOLUME_RATIO = 3.0
DEFAULT_MAX_DISTANCE_DREF = 2.5
DEFAULT_TRACKASTRA_MODEL = "ctc"
DEFAULT_TRACKASTRA_MODE = "greedy"
DEFAULT_TRACKASTRA_DEVICE = "cuda"
DEFAULT_STEPS = 1000
DEFAULT_LR = 2e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_ACCUMULATE = 4
DEFAULT_EVAL_EVERY = 100
DEFAULT_PRINT_EVERY = 10
DEFAULT_VAL_CASES = 20
DEFAULT_SYNTHETIC_LOGIT = 3.5
DEFAULT_PRESERVE_EDGES = 512
DEFAULT_PRESERVE_RATIO = 12
DEFAULT_PRESERVATION_WEIGHT = 2.0
DEFAULT_SPLIT_WEIGHT = 0.05
DEFAULT_NOOP_WEIGHT = 0.50
DEFAULT_CORRUPTED_GATE_WEIGHT = 0.05
DEFAULT_MARGIN_WEIGHT = 0.50
DEFAULT_MARGIN = 1.0
DEFAULT_SEED = 20260827


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if torch.is_tensor(x):
        y = x.detach().cpu()
        return jsonable(y.item()) if y.ndim == 0 else jsonable(y.tolist())
    if isinstance(x, Path):
        return str(x)
    if dataclasses.is_dataclass(x):
        return jsonable(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    return str(x)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(tmp, np.asarray(array), allow_pickle=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# STIRNET_INV38_INV35_PICKLE_COMPAT_V1
class _Inv35CacheUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "__main__" and name == "TemporalStatic":
            return INV35.TemporalStatic
        return super().find_class(module, name)


def load_inv35_temporal_static(path: Path):
    with path.open("rb") as handle:
        return _Inv35CacheUnpickler(handle).load()


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def parse_spacing(value: str) -> tuple[float, float, float]:
    values = tuple(float(v.strip()) for v in value.split(","))
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError("--spacing must be three positive comma-separated values")
    return values


def autocast_for(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "fp32":
        return nullcontext()
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)


def make_grad_scaler(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda" and amp_dtype == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


@dataclass(frozen=True)
class Paths:
    sample: str
    inv35_root: Path
    output: Path
    init_checkpoint: Path

    @property
    def inv35_cache(self): return self.inv35_root / "cache"
    @property
    def preparation(self): return self.inv35_root / "preparation.json"
    @property
    def raw_movie(self): return self.inv35_cache / "raw_movie.npy"
    @property
    def manual_movie(self): return self.inv35_cache / "manual_movie.npy"
    @property
    def clean_track_graph(self): return self.inv35_cache / "trackastra/track_graph.pkl"
    @property
    def clean_temporal_static(self): return self.inv35_cache / "temporal_static.pkl"
    @property
    def spatial(self): return self.inv35_cache / "spatial"
    def graph_cache(self, t): return self.spatial / f"t{t:03d}/frozen_graph.pt"
    def spatial_inputs(self, t): return self.spatial / f"t{t:03d}/spatial_inputs.npy"
    def old_obs(self, t, name): return self.spatial / f"t{t:03d}/{name}"
    @property
    def touching_pairs(self): return self.output / "touching_pairs.json"
    @property
    def dataset_manifest(self): return self.output / "dataset_manifest.json"
    def variant_dir(self, v): return self.output / "variants" / f"variant_{v:03d}"
    def merge_plan(self, v): return self.variant_dir(v) / "merge_plan.json"
    def variant_graph(self, v): return self.variant_dir(v) / "track_graph.pkl"
    def variant_static(self, v): return self.variant_dir(v) / "temporal_static.pkl"
    def variant_temporal_graph(self, v, t): return self.variant_dir(v) / "temporal_graphs" / f"t{t:03d}.pt"
    def variant_success(self, v): return self.variant_dir(v) / "_SUCCESS.json"
    def observer_dir(self, t): return self.output / "observer" / f"t{t:03d}"
    def observer_file(self, t, name): return self.observer_dir(t) / name
    @property
    def training_history(self): return self.output / "training_history.json"
    @property
    def validation_history(self): return self.output / "validation_history.json"
    @property
    def best_metrics(self): return self.output / "best_metrics.json"
    @property
    def latest(self): return self.output / "latest.pt"
    @property
    def best(self): return self.output / "best.pt"
    @property
    def final(self): return self.output / "final.pt"
    @property
    def test_metrics(self): return self.output / "test_metrics.json"


def make_paths(args: argparse.Namespace) -> Paths:
    sample = args.sample_id
    inv35_root = resolve(args.inv35_root) if args.inv35_root else (
        ROOT / "runs/stirnet/evaluation/35_biohub_temporal_merge_synthesis" / sample
    ).resolve()
    output = resolve(args.output) if args.output else (
        ROOT / "runs/stirnet/evaluation" / SCRIPT_NAME / sample
    ).resolve()
    if args.init_checkpoint:
        init_checkpoint = resolve(args.init_checkpoint)
    else:
        init_checkpoint = (inv35_root / "best.pt").resolve()
    return Paths(sample, inv35_root, output, init_checkpoint)


def validate_inputs(paths: Paths, frames: int) -> None:
    required = [
        paths.preparation, paths.raw_movie, paths.manual_movie,
        paths.clean_track_graph, paths.clean_temporal_static, paths.init_checkpoint,
    ]
    for t in range(frames):
        required += [paths.graph_cache(t), paths.spatial_inputs(t)]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Investigation 38 needs the completed Investigation-35 real spatial cache. Missing:\n"
            + "\n".join(f"  {p}" for p in missing[:50])
        )


@dataclass(frozen=True)
class TouchPair:
    frame: int
    a: int
    b: int
    clean_node_a: int
    clean_node_b: int
    interface_edges: int
    voxels_a: int
    voxels_b: int
    volume_ratio: float
    distance_dref: float


def manual_stats(labels: np.ndarray, spacing: Sequence[float]):
    spacing = np.asarray(spacing, np.float32)
    center = 0.5 * (np.asarray(labels.shape, np.float32) - 1.0) * spacing
    counts, centroids = {}, {}
    ids, cnts = np.unique(labels[labels > 0], return_counts=True)
    for label, count in zip(ids.tolist(), cnts.tolist()):
        coords = np.argwhere(labels == int(label))
        counts[int(label)] = int(count)
        centroids[int(label)] = coords.astype(np.float32).mean(0) * spacing - center
    return counts, centroids


def build_touching_catalog(paths: Paths, *, frames: int, spacing, dref_um: float,
                           min_voxels: int, max_volume_ratio: float,
                           max_distance_dref: float, rebuild: bool):
    if paths.touching_pairs.is_file() and not rebuild:
        payload = json.loads(paths.touching_pairs.read_text(encoding="utf-8"))
        if int(payload.get("version", -1)) == DATASET_VERSION:
            result = {int(t): [TouchPair(**row) for row in rows] for t, rows in payload["frames"].items()}
            print(f"[touching pairs] reuse total={sum(map(len, result.values()))}")
            return result

    clean_static = load_inv35_temporal_static(
        paths.clean_temporal_static
    )
    with paths.clean_track_graph.open("rb") as handle:
        clean_graph = pickle.load(handle)
    manual_movie = np.load(paths.manual_movie, mmap_mode="r", allow_pickle=False)
    result, serial = {}, {}

    for t in range(frames):
        cache = torch_load(paths.graph_cache(t)); rag = cache["rag"]
        manual = np.asarray(manual_movie[t])
        sv = rag.supervoxel_labels[0].detach().cpu().numpy().astype(np.int64, copy=False)
        lookup = INV35.sv_label_lookup(sv, manual, name=f"inv38 t={t} manual")
        node_sv = rag.node_supervoxel_id.detach().cpu().numpy().astype(np.int64, copy=False)
        node_manual = lookup[node_sv]
        counts, centroids = manual_stats(manual, spacing)
        pair_edges = defaultdict(int)
        src = rag.edge_index[0].detach().cpu().numpy(); dst = rag.edge_index[1].detach().cpu().numpy()
        for u, v in zip(src.tolist(), dst.tolist()):
            a, b = int(node_manual[u]), int(node_manual[v])
            if a > 0 and b > 0 and a != b:
                pair_edges[(min(a, b), max(a, b))] += 1
        rows = []
        for (a, b), edge_count in sorted(pair_edges.items()):
            if counts.get(a, 0) < min_voxels or counts.get(b, 0) < min_voxels:
                continue
            na = clean_static.manual_to_node.get(t, {}).get(a)
            nb = clean_static.manual_to_node.get(t, {}).get(b)
            if na is None or nb is None:
                continue
            if any((clean_graph.in_degree(na) > 1, clean_graph.out_degree(na) > 1,
                    clean_graph.in_degree(nb) > 1, clean_graph.out_degree(nb) > 1)):
                continue
            ratio = max(counts[a] / max(counts[b], 1), counts[b] / max(counts[a], 1))
            if ratio > max_volume_ratio:
                continue
            distance = float(np.linalg.norm(centroids[a] - centroids[b]) / max(dref_um, 1e-6))
            if distance > max_distance_dref:
                continue
            rows.append(TouchPair(t, a, b, int(na), int(nb), int(edge_count),
                                  counts[a], counts[b], float(ratio), distance))
        result[t] = rows
        serial[str(t)] = [dataclasses.asdict(row) for row in rows]
        print(f"[touching pairs t={t:03d}] {len(rows)}", flush=True)

    atomic_json(paths.touching_pairs, {
        "version": DATASET_VERSION, "sample_id": paths.sample,
        "dref_um": dref_um, "frames": serial,
        "total_pairs": sum(len(v) for v in result.values()),
        "min_voxels": min_voxels, "max_volume_ratio": max_volume_ratio,
        "max_distance_dref": max_distance_dref,
    })
    return result


@dataclass(frozen=True)
class MergeEvent:
    frame: int
    a: int
    b: int
    representative: int
    clean_node_a: int
    clean_node_b: int
    interface_edges: int


@dataclass(frozen=True)
class VariantPlan:
    index: int
    split: str
    events_by_frame: dict[int, tuple[MergeEvent, ...]]
    @property
    def merge_count(self): return sum(len(x) for x in self.events_by_frame.values())


def next_clean_nodes(graph, node: int, t: int) -> set[int]:
    return {int(v) for v in graph.successors(int(node)) if int(graph.nodes[int(v)]["time"]) == t + 1}


def plan_one_variant(index: int, split: str, catalog, clean_graph, *, frames: int,
                     merge_fraction: float, max_merges_per_frame: int, seed: int):
    events, blocked = {}, set()
    for t in range(frames):
        candidates = list(catalog.get(t, ()))
        rng = random.Random(seed + 1_000_003 * index + 10_007 * t)
        rng.shuffle(candidates)
        target = min(max_merges_per_frame, max(1 if candidates else 0, int(round(len(candidates) * merge_fraction))))
        selected, used = [], set()
        for pair in candidates:
            if len(selected) >= target:
                break
            if pair.a in used or pair.b in used:
                continue
            if pair.clean_node_a in blocked or pair.clean_node_b in blocked:
                continue
            selected.append(MergeEvent(t, pair.a, pair.b, min(pair.a, pair.b),
                                       pair.clean_node_a, pair.clean_node_b, pair.interface_edges))
            used.update((pair.a, pair.b))
        events[t] = tuple(selected)
        next_blocked = set()
        for event in selected:
            next_blocked |= next_clean_nodes(clean_graph, event.clean_node_a, t)
            next_blocked |= next_clean_nodes(clean_graph, event.clean_node_b, t)
        blocked = next_blocked
    return VariantPlan(index, split, events)


def generate_plans(paths: Paths, catalog, *, frames: int, train_variants: int,
                   val_variants: int, test_variants: int, merge_fraction: float,
                   max_merges_per_frame: int, seed: int, rebuild: bool):
    with paths.clean_track_graph.open("rb") as handle:
        clean_graph = pickle.load(handle)
    splits = ["train"] * train_variants + ["val"] * val_variants + ["test"] * test_variants
    plans = [plan_one_variant(i, split, catalog, clean_graph, frames=frames,
                              merge_fraction=merge_fraction,
                              max_merges_per_frame=max_merges_per_frame, seed=seed)
             for i, split in enumerate(splits)]
    manifest = {
        "version": DATASET_VERSION, "sample_id": paths.sample, "frame_count": frames,
        "train_variants": train_variants, "val_variants": val_variants,
        "test_variants": test_variants, "merge_fraction": merge_fraction,
        "max_merges_per_frame": max_merges_per_frame, "seed": seed,
        "plans": [{"index": p.index, "split": p.split, "merge_count": p.merge_count,
                   "events_by_frame": {str(t): [dataclasses.asdict(e) for e in p.events_by_frame[t]] for t in range(frames)}}
                  for p in plans],
    }
    if paths.dataset_manifest.is_file() and not rebuild:
        old = json.loads(paths.dataset_manifest.read_text(encoding="utf-8"))
        if old != manifest:
            raise RuntimeError("Existing dataset manifest differs. Use --rebuild-dataset.")
    else:
        if rebuild and (paths.output / "variants").exists():
            shutil.rmtree(paths.output / "variants")
        if rebuild and (paths.output / "observer").exists():
            shutil.rmtree(paths.output / "observer")
        atomic_json(paths.dataset_manifest, manifest)
    for p in plans:
        pdir = paths.variant_dir(p.index); pdir.mkdir(parents=True, exist_ok=True)
        atomic_json(paths.merge_plan(p.index), {
            "version": DATASET_VERSION, "variant_index": p.index, "split": p.split,
            "merge_count": p.merge_count,
            "events_by_frame": {str(t): [dataclasses.asdict(e) for e in p.events_by_frame[t]] for t in range(frames)},
        })
    print("\n" + "=" * 112)
    print("INVESTIGATION 38 — VARIANT PLAN")
    print("=" * 112)
    for split in ("train", "val", "test"):
        rows = [p for p in plans if p.split == split]
        print(f"{split:5s} variants={len(rows):2d} merge_events={sum(p.merge_count for p in rows)}")
    print("=" * 112)
    return plans


def variant_ready(paths: Paths, plan: VariantPlan, frames: int) -> bool:
    return (
        paths.variant_success(plan.index).is_file()
        and paths.variant_graph(plan.index).is_file()
        and paths.variant_static(plan.index).is_file()
        and all(paths.variant_temporal_graph(plan.index, t).is_file() for t in range(frames))
    )


def apply_events(clean: np.ndarray, events: Sequence[MergeEvent]) -> np.ndarray:
    out = np.asarray(clean).astype(np.int32, copy=True)
    for e in events:
        mask = (out == int(e.a)) | (out == int(e.b))
        out[mask] = int(e.representative)
    return out


@dataclass
class VariantTemporalStatic:
    records: dict[int, Any]
    associations: list[Any]
    nodes_by_time: dict[int, list[int]]


def build_variant_temporal_graphs(paths: Paths, plan: VariantPlan, *, track_graph,
                                  tracked_masks: np.ndarray, records, associations,
                                  nodes_by_time, frames: int, temporal_radius: int,
                                  spacing, dref_um: float):
    for target_t in range(frames):
        available = sequence_available_time_offsets(target_t, frames, temporal_radius)
        abs_times = [target_t + int(dt) for dt in available]
        selected_ids, selected_records = set(), []
        for absolute_t in abs_times:
            for node_id in nodes_by_time.get(absolute_t, []):
                base = records.get(int(node_id))
                if base is None:
                    continue
                selected_ids.add(int(node_id))
                selected_records.append(replace(base, time_offset=absolute_t - target_t))
        selected_assoc = [a for a in associations if int(a.src_node_id) in selected_ids and int(a.dst_node_id) in selected_ids]
        graph = build_temporal_graph(
            selected_records, selected_assoc,
            dref_um=float(dref_um), temporal_radius=int(temporal_radius),
            available_time_offsets=available, k_spatial_neighbors=6,
            spatial_radius_dref=2.5,
            current_labels=np.asarray(tracked_masks[target_t]),
            spacing_um=tuple(float(v) for v in spacing),
            candidate_graph_enabled=False,
        )
        graph["temporal_batch"] = torch.zeros(len(graph["temporal_ref_um"]), dtype=torch.long)
        graph["target_time_index"] = int(target_t)
        graph["available_time_offsets"] = torch.tensor(available, dtype=torch.long)
        graph["_inv38_dataset_version"] = DATASET_VERSION
        graph["_variant_index"] = int(plan.index)
        atomic_torch(paths.variant_temporal_graph(plan.index, target_t), graph)


def prepare_variants(paths: Paths, plans: Sequence[VariantPlan], *, frames: int,
                     temporal_radius: int, spacing, dref_um: float,
                     trackastra_model: str, trackastra_mode: str,
                     trackastra_device: str, rebuild: bool):
    todo = [p for p in plans if rebuild or not variant_ready(paths, p, frames)]
    if not todo:
        print("[variants] all concrete Trackastra variants ready", flush=True)
        return
    try:
        from trackastra.model import Trackastra
    except ImportError as exc:
        raise RuntimeError("Trackastra is required for Investigation 38") from exc

    raw = np.load(paths.raw_movie, mmap_mode="r", allow_pickle=False)
    manual = np.load(paths.manual_movie, mmap_mode="r", allow_pickle=False)
    print("\n" + "=" * 112)
    print("INVESTIGATION 38 — TRACKASTRA ON CONCRETE MERGE VARIANTS")
    print("=" * 112)
    print(f"variants={ [p.index for p in todo] }")
    print(f"model={trackastra_model} mode={trackastra_mode} device={trackastra_device}")
    print("=" * 112)
    tracker = Trackastra.from_pretrained(trackastra_model, device=trackastra_device)

    for position, plan in enumerate(todo, 1):
        started = time.perf_counter()
        pdir = paths.variant_dir(plan.index); pdir.mkdir(parents=True, exist_ok=True)
        tmp_path = pdir / "_synthetic_labels_tmp.npy"
        tmp_path.unlink(missing_ok=True)
        synthetic = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.int32, shape=manual.shape)
        for t in range(frames):
            synthetic[t] = apply_events(np.asarray(manual[t]), plan.events_by_frame.get(t, ()))
        synthetic.flush()
        print(f"[variant {plan.index:03d} {plan.split}] {position}/{len(todo)} merges={plan.merge_count}", flush=True)
        track_graph, tracked_masks = tracker.track(raw, synthetic, mode=trackastra_mode)
        with paths.variant_graph(plan.index).open("wb") as handle:
            pickle.dump(track_graph, handle)

        records = INV35.INV30.build_static_detection_records(
            track_graph, np.asarray(tracked_masks), raw,
            tuple(float(v) for v in spacing), float(dref_um),
        )
        associations = INV35.INV30.trackastra_associations(track_graph)
        nodes_by_time = defaultdict(list)
        for node_id, row in track_graph.nodes(data=True):
            nodes_by_time[int(row["time"])].append(int(node_id))
        nodes_by_time = {int(t): sorted(map(int, rows)) for t, rows in nodes_by_time.items()}
        static = VariantTemporalStatic(records, associations, nodes_by_time)
        with paths.variant_static(plan.index).open("wb") as handle:
            pickle.dump(static, handle)
        build_variant_temporal_graphs(
            paths, plan, track_graph=track_graph, tracked_masks=np.asarray(tracked_masks),
            records=records, associations=associations, nodes_by_time=nodes_by_time,
            frames=frames, temporal_radius=temporal_radius, spacing=spacing, dref_um=dref_um,
        )
        atomic_json(paths.variant_success(plan.index), {
            "version": DATASET_VERSION, "variant_index": plan.index, "split": plan.split,
            "merge_count": plan.merge_count,
            "trackastra_nodes": int(track_graph.number_of_nodes()),
            "trackastra_edges": int(track_graph.number_of_edges()),
            "seconds": time.perf_counter() - started,
        })
        del synthetic, track_graph, tracked_masks, records, associations, static
        tmp_path.unlink(missing_ok=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[variant {plan.index:03d}] ready time={duration(time.perf_counter() - started)}", flush=True)

    del tracker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def reference_union(paths: Paths, plans: Sequence[VariantPlan], t: int) -> np.ndarray:
    rows = {}
    for p in plans:
        graph = torch_load(paths.variant_temporal_graph(p.index, t))
        refs = torch.as_tensor(graph["temporal_ref_um"]).detach().float().cpu().numpy()
        for ref in refs:
            rows.setdefault(INV35.reference_key(ref), np.asarray(ref, np.float32))
    if not rows:
        return np.zeros((0, 3), np.float32)
    return np.stack([rows[k] for k in sorted(rows)]).astype(np.float32, copy=False)


def observer_files(paths: Paths, t: int):
    return {
        "refs": paths.observer_file(t, "ref_um.npy"),
        "d1": paths.observer_file(t, "d1.npy"),
        "d2": paths.observer_file(t, "d2.npy"),
        "hidden": paths.observer_file(t, "hidden.npy"),
        "explicit": paths.observer_file(t, "explicit.npy"),
        "meta": paths.observer_file(t, "meta.json"),
    }


def observer_ready(paths: Paths, t: int, refs: np.ndarray) -> bool:
    files = observer_files(paths, t)
    if any(not p.is_file() for p in files.values()):
        return False
    cached = np.load(files["refs"], mmap_mode="r", allow_pickle=False)
    if cached.shape != refs.shape:
        return False
    return [INV35.reference_key(x) for x in cached] == [INV35.reference_key(x) for x in refs]


def load_observer(paths: Paths, t: int, device: torch.device):
    f = observer_files(paths, t)
    def arr(name):
        return np.array(np.load(f[name], mmap_mode="r", allow_pickle=False), dtype=np.float32, order="C", copy=True)
    refs, d1, d2, hidden, explicit = arr("refs"), arr("d1"), arr("d2"), arr("hidden"), arr("explicit")
    return INV35.ObserverRawLookup(
        ref_um=torch.from_numpy(refs).to(device), d1=torch.from_numpy(d1).to(device),
        d2=torch.from_numpy(d2).to(device), hidden=torch.from_numpy(hidden).to(device),
        explicit=torch.from_numpy(explicit).to(device),
        key_to_row={INV35.reference_key(ref): i for i, ref in enumerate(refs)},
    )


@torch.inference_mode()
def prepare_observer(paths: Paths, plans: Sequence[VariantPlan], *, frames: int,
                     spacing, dref_um: float, model, device: torch.device,
                     amp_dtype: str, rebuild: bool):
    spacing_t = torch.tensor([spacing], device=device, dtype=torch.float32)
    dref_t = torch.tensor([dref_um], device=device, dtype=torch.float32)
    inference_cfg = dataclasses.replace(model.cfg.inference, mode="tiled", tiled_dense_enabled=True)
    print("\n" + "=" * 112)
    print("INVESTIGATION 38 — OBSERVER CACHE FOR ACTUAL VARIANT REFERENCES")
    print("=" * 112)
    for t in range(frames):
        refs = reference_union(paths, plans, t)
        if not rebuild and observer_ready(paths, t, refs):
            meta = json.loads(observer_files(paths, t)["meta"].read_text(encoding="utf-8"))
            print(f"[observer t={t:03d}] reuse refs={len(refs)} new={meta.get('newly_sampled_count', '?')}")
            continue
        started = time.perf_counter()
        old_names = {
            "refs": "observer_ref_um.npy", "d1": "observer_d1_raw.npy",
            "d2": "observer_d2_raw.npy", "hidden": "observer_hidden_raw.npy",
            "explicit": "observer_explicit_raw.npy",
        }
        old_paths = {k: paths.old_obs(t, v) for k, v in old_names.items()}
        old_ok = all(p.is_file() for p in old_paths.values())
        if old_ok:
            old_refs = np.load(old_paths["refs"], mmap_mode="r", allow_pickle=False)
            old_map = {INV35.reference_key(ref): i for i, ref in enumerate(old_refs)}
            old_d1 = np.load(old_paths["d1"], mmap_mode="r", allow_pickle=False)
            old_d2 = np.load(old_paths["d2"], mmap_mode="r", allow_pickle=False)
            old_hidden = np.load(old_paths["hidden"], mmap_mode="r", allow_pickle=False)
            old_explicit = np.load(old_paths["explicit"], mmap_mode="r", allow_pickle=False)
        else:
            old_map = {}; old_d1 = old_d2 = old_hidden = old_explicit = None
        c1, c2, ch = int(model.cfg.spatial.channels[1]), int(model.cfg.spatial.channels[2]), int(model.cfg.geometry.hidden_channels)
        d1_out = np.zeros((len(refs), c1), np.float16)
        d2_out = np.zeros((len(refs), c2), np.float16)
        hidden_out = np.zeros((len(refs), ch), np.float16)
        explicit_out = np.zeros((len(refs), 11), np.float16)
        missing, reused = [], 0
        for row, ref in enumerate(refs):
            source = old_map.get(INV35.reference_key(ref))
            if source is None:
                missing.append(row); continue
            d1_out[row] = np.asarray(old_d1[source], np.float16)
            d2_out[row] = np.asarray(old_d2[source], np.float16)
            hidden_out[row] = np.asarray(old_hidden[source], np.float16)
            explicit_out[row] = np.asarray(old_explicit[source], np.float16)
            reused += 1
        if missing:
            spatial_np = np.array(np.load(paths.spatial_inputs(t), mmap_mode="r", allow_pickle=False),
                                  dtype=np.float32, order="C", copy=True)
            spatial = torch.from_numpy(spatial_np)[None].to(device)
            missing_refs = refs[np.asarray(missing, np.int64)]
            with autocast_for(device, amp_dtype):
                d1, d2, hidden, explicit = INV35.sample_raw_observer_features(
                    model, spatial, spacing_t, dref_t, missing_refs, config=inference_cfg
                )
            rows = np.asarray(missing, np.int64)
            d1_out[rows], d2_out[rows], hidden_out[rows], explicit_out[rows] = d1, d2, hidden, explicit
            del spatial
            if device.type == "cuda": torch.cuda.empty_cache()
        f = observer_files(paths, t); paths.observer_dir(t).mkdir(parents=True, exist_ok=True)
        atomic_npy(f["refs"], refs); atomic_npy(f["d1"], d1_out); atomic_npy(f["d2"], d2_out)
        atomic_npy(f["hidden"], hidden_out); atomic_npy(f["explicit"], explicit_out)
        atomic_json(f["meta"], {
            "version": OBSERVER_VERSION, "timepoint": t, "reference_count": len(refs),
            "reused_inv35_count": reused, "newly_sampled_count": len(missing),
            "seconds": time.perf_counter() - started,
        })
        print(f"[observer t={t:03d}] refs={len(refs)} reused35={reused} new={len(missing)} time={duration(time.perf_counter() - started)}", flush=True)
    print("=" * 112)


@dataclass
class RuntimeFrame:
    t: int
    rag: Any
    actual_partition: Any
    manual: np.ndarray
    node_manual: Tensor
    observer: Any


class RuntimeFrameLoader:
    def __init__(self, paths: Paths, device: torch.device):
        self.paths, self.device, self.current = paths, device, None

    def load(self, t: int) -> RuntimeFrame:
        t = int(t)
        if self.current is not None and self.current.t == t:
            return self.current
        if self.current is not None:
            del self.current; self.current = None
            if self.device.type == "cuda": torch.cuda.empty_cache()
        cache = torch_load(self.paths.graph_cache(t))
        rag = INV35.tree_device_training_float(cache["rag"], self.device, torch.float32)
        actual = INV35.tree_device_training_float(cache["actual_partition"], self.device, torch.float32)
        manual_movie = np.load(self.paths.manual_movie, mmap_mode="r", allow_pickle=False)
        manual = np.asarray(manual_movie[t])
        sv = rag.supervoxel_labels[0].detach().cpu().numpy().astype(np.int64, copy=False)
        lookup = INV35.sv_label_lookup(sv, manual, name=f"inv38 runtime t={t} manual")
        node_sv = rag.node_supervoxel_id.detach().cpu().numpy().astype(np.int64, copy=False)
        node_manual = torch.as_tensor(lookup[node_sv], device=self.device, dtype=torch.long)
        observer = load_observer(self.paths, t, self.device)
        self.current = RuntimeFrame(t, rag, actual, manual, node_manual, observer)
        return self.current


@dataclass
class ConcreteCase:
    frame: int
    variant_index: int
    events: tuple[MergeEvent, ...]
    rag: Any
    partition: Any
    target_keep: Tensor
    editable_mask: Tensor
    correction_mask: Tensor
    preservation_mask: Tensor
    split_target: Tensor
    node_current_component: Tensor


def build_case(runtime: RuntimeFrame, *, variant_index: int, events: Sequence[MergeEvent], synthetic_logit: float):
    rag, node_manual = runtime.rag, runtime.node_manual
    representative = {}
    for e in events:
        representative[int(e.a)] = int(e.representative)
        representative[int(e.b)] = int(e.representative)
    actual = runtime.actual_partition.node_component_global
    keys = []
    for node in range(int(node_manual.numel())):
        mid = int(node_manual[node].item())
        keys.append(("manual", representative.get(mid, mid)) if mid > 0 else ("extra", int(actual[node].item())))
    key_to_comp, comp_manual, node_comp_values = {}, [], []
    for node, key in enumerate(keys):
        comp = key_to_comp.get(key)
        if comp is None:
            comp = len(key_to_comp); key_to_comp[key] = comp; comp_manual.append(set())
        node_comp_values.append(comp)
        mid = int(node_manual[node].item())
        if mid > 0: comp_manual[comp].add(mid)
    node_comp = torch.tensor(node_comp_values, device=rag.node_features.device, dtype=torch.long)
    src, dst = rag.edge_index
    same_current = node_comp[src] == node_comp[dst]
    positive = rag.spatial_edge_logits.new_full(rag.spatial_edge_logits.shape, float(synthetic_logit))
    negative = rag.spatial_edge_logits.new_full(rag.spatial_edge_logits.shape, -float(synthetic_logit))
    spatial_logits = torch.where(same_current, positive, negative)
    synthetic_rag = replace(rag, spatial_edge_logits=spatial_logits)
    ncomp = len(key_to_comp)
    tiny = (torch.arange(1, ncomp + 1, device=rag.node_features.device, dtype=torch.long).reshape(1, 1, -1)
            if ncomp else torch.zeros((1, 1, 1), device=rag.node_features.device, dtype=torch.long))
    partition = INV35.PartitionState(
        labels=[tiny], node_component=node_comp, node_component_global=node_comp.clone(),
        component_count_per_batch=torch.tensor([ncomp], device=rag.node_features.device, dtype=torch.long),
        edge_logits=spatial_logits,
    )
    msrc, mdst = node_manual[src], node_manual[dst]
    valid = (msrc > 0) & (mdst > 0)
    target_keep = valid & (msrc == mdst)
    editable = valid & same_current
    correction = editable & ~target_keep
    preservation = editable & target_keep
    split_target = spatial_logits.new_zeros((ncomp,))
    for comp, mids in enumerate(comp_manual): split_target[comp] = float(len(mids) > 1)
    return ConcreteCase(runtime.t, variant_index, tuple(events), synthetic_rag, partition,
                        target_keep, editable, correction, preservation, split_target, node_comp)


@dataclass
class ConcreteForward:
    full_reasoning: Any
    full_logits: Tensor
    contentless_reasoning: Any
    contentless_logits: Tensor


def split_only_logits(case: ConcreteCase, proposed: Tensor) -> Tensor:
    return torch.where(case.editable_mask, proposed, case.rag.spatial_edge_logits.detach())


def load_variant_temporal_graph(paths: Paths, variant: int, t: int):
    return torch_load(paths.variant_temporal_graph(variant, t))


def forward_case(model, *, runtime: RuntimeFrame, case: ConcreteCase, graph,
                 spacing, dref_um: float, device: torch.device):
    decoded, geometry = INV35.dummy_geometry_and_decode(model, case.rag.node_features)
    spacing_t = torch.tensor([spacing], device=device, dtype=torch.float32)
    dref_t = torch.tensor([dref_um], device=device, dtype=torch.float32)
    instances = model.instance_tokenizer(case.partition, case.rag, decoded, geometry,
                                         spacing_t, dref_t, profile_prefix="inv38_tokenizer")
    temporal_input = INV35.temporal_input_from_graph(model, graph, device)
    temporal_base = model.temporal_encoder(temporal_input)
    full_temporal = INV35.observe_from_raw_lookup(model, temporal_base, runtime.observer)
    full_reasoning = model.instance_temporal(instances, case.rag, full_temporal, dref_t)
    full_logits = split_only_logits(case, full_reasoning.final_edge_logits)
    contentless_base = INV35.contentless_temporal_state(temporal_base)
    contentless_temporal = INV35.observe_from_raw_lookup(model, contentless_base, runtime.observer)
    contentless_reasoning = model.instance_temporal(instances, case.rag, contentless_temporal, dref_t)
    contentless_logits = split_only_logits(case, contentless_reasoning.final_edge_logits)
    return ConcreteForward(full_reasoning, full_logits, contentless_reasoning, contentless_logits)


@dataclass
class ConcreteLoss:
    total: Tensor
    correction: Tensor
    preservation: Tensor
    split: Tensor
    noop: Tensor
    gate: Tensor
    margin: Tensor
    correction_edges: int
    preservation_edges: int


def sample_rows(index: Tensor, count: int, rng: random.Random):
    if count <= 0 or index.numel() == 0: return index[:0]
    if int(index.numel()) <= count: return index
    rows = rng.sample(range(int(index.numel())), count)
    return index[torch.tensor(rows, device=index.device, dtype=torch.long)]


def compute_loss(case: ConcreteCase, forward: ConcreteForward, *, preserve_edges: int,
                 preserve_ratio: int, preservation_weight: float, split_weight: float,
                 noop_weight: float, gate_weight: float, margin_weight: float,
                 margin: float, rng: random.Random):
    full, corrupt = forward.full_logits, forward.contentless_logits
    zero = full.sum() * 0.0
    corr_idx = torch.nonzero(case.correction_mask, as_tuple=False).flatten()
    keep_idx_all = torch.nonzero(case.preservation_mask, as_tuple=False).flatten()
    budget = preserve_edges
    if corr_idx.numel(): budget = min(budget, max(int(corr_idx.numel()) * preserve_ratio, int(corr_idx.numel())))
    keep_idx = sample_rows(keep_idx_all, budget, rng)
    correction = (F.binary_cross_entropy_with_logits(full[corr_idx], torch.zeros_like(full[corr_idx]))
                  if corr_idx.numel() else zero)
    preservation = (F.binary_cross_entropy_with_logits(full[keep_idx], torch.ones_like(full[keep_idx]))
                    if keep_idx.numel() else zero)
    positives = torch.nonzero(case.split_target > 0.5, as_tuple=False).flatten()
    negatives = torch.nonzero(case.split_target <= 0.5, as_tuple=False).flatten()
    if positives.numel():
        negatives = sample_rows(negatives, max(int(positives.numel()) * 8, 16), rng)
        sidx = torch.cat([positives, negatives])
    else:
        sidx = sample_rows(negatives, min(int(negatives.numel()), 64), rng)
    split = (F.binary_cross_entropy_with_logits(forward.full_reasoning.split_logits[sidx],
                                                case.split_target[sidx].to(forward.full_reasoning.split_logits.dtype))
             if sidx.numel() else zero)
    editable = torch.nonzero(case.editable_mask, as_tuple=False).flatten()
    if editable.numel():
        noop = F.smooth_l1_loss(corrupt[editable], case.rag.spatial_edge_logits[editable].detach(), beta=0.5)
        gate = forward.contentless_reasoning.edge_temporal_gate[editable].square().mean()
    else:
        noop = gate = zero
    causal = (F.relu(float(margin) - (corrupt[corr_idx].detach() - full[corr_idx])).mean()
              if corr_idx.numel() else zero)
    total = correction + preservation_weight * preservation + split_weight * split + noop_weight * noop + gate_weight * gate + margin_weight * causal
    return ConcreteLoss(total, correction, preservation, split, noop, gate, causal,
                        int(corr_idx.numel()), int(keep_idx.numel()))


@dataclass(frozen=True)
class VariantFrameCase:
    variant_index: int
    split: str
    frame: int
    events: tuple[MergeEvent, ...]


def index_cases(plans: Sequence[VariantPlan], frames: int):
    result = {"train": [], "val": [], "test": []}
    for p in plans:
        for t in range(frames):
            events = tuple(p.events_by_frame.get(t, ()))
            if events:
                result[p.split].append(VariantFrameCase(p.index, p.split, t, events))
    return result


def pair_exact(runtime: RuntimeFrame, predicted, event: MergeEvent) -> bool:
    seen = set(); node_manual = runtime.node_manual; component = predicted.node_component_global
    for mid in (event.a, event.b):
        rows = torch.nonzero(node_manual == int(mid), as_tuple=False).flatten()
        if rows.numel() == 0: return False
        values = torch.unique(component[rows])
        if values.numel() != 1: return False
        comp = int(values.item())
        if comp in seen: return False
        seen.add(comp)
        members = torch.nonzero(component == comp, as_tuple=False).flatten()
        positive = node_manual[members]; positive = positive[positive > 0]
        if bool((positive != int(mid)).any()): return False
    return True


def clean_split_counts(runtime: RuntimeFrame, predicted, events: Sequence[MergeEvent]):
    merged = {int(v) for e in events for v in (e.a, e.b)}
    total = split = 0; comp = predicted.node_component_global
    for mid in torch.unique(runtime.node_manual).tolist():
        mid = int(mid)
        if mid <= 0 or mid in merged: continue
        rows = torch.nonzero(runtime.node_manual == mid, as_tuple=False).flatten()
        if rows.numel():
            total += 1; split += int(torch.unique(comp[rows]).numel() > 1)
    return total, split


def split_only_violations(case: ConcreteCase, predicted) -> int:
    current, final = case.node_current_component, predicted.node_component_global
    violations = 0
    for fid in torch.unique(final).tolist():
        rows = torch.nonzero(final == int(fid), as_tuple=False).flatten()
        if rows.numel() > 1 and torch.unique(current[rows]).numel() > 1: violations += 1
    return violations


@torch.no_grad()
def evaluate(model, *, paths: Paths, cases: Sequence[VariantFrameCase], loader: RuntimeFrameLoader,
             spacing, dref_um: float, synthetic_logit: float, device: torch.device,
             maximum_cases: int, seed: int):
    INV35.set_temporal_train_mode(model, False)
    ordered = sorted(cases, key=lambda c: ((c.variant_index * 1_000_003 + c.frame * 10_007 + seed) % 2_147_483_647))
    if maximum_cases > 0: ordered = ordered[:min(maximum_cases, len(ordered))]
    if not ordered: raise RuntimeError("No evaluation cases")
    cut_total = cut_correct = keep_total = keep_correct = contentless_cut_correct = 0
    pair_total = pair_ok = clean_total = clean_split = violations = 0
    threshold = float(model.cfg.partition.final_merge_threshold)
    for spec in ordered:
        runtime = loader.load(spec.frame)
        case = build_case(runtime, variant_index=spec.variant_index, events=spec.events, synthetic_logit=synthetic_logit)
        graph = load_variant_temporal_graph(paths, spec.variant_index, spec.frame)
        fw = forward_case(model, runtime=runtime, case=case, graph=graph, spacing=spacing, dref_um=dref_um, device=device)
        pred_keep = fw.full_logits.sigmoid() >= threshold
        cont_keep = fw.contentless_logits.sigmoid() >= threshold
        cut_total += int(case.correction_mask.sum().item())
        cut_correct += int((~pred_keep[case.correction_mask]).sum().item())
        contentless_cut_correct += int((~cont_keep[case.correction_mask]).sum().item())
        keep_total += int(case.preservation_mask.sum().item())
        keep_correct += int(pred_keep[case.preservation_mask].sum().item())
        predicted = model.partitioner(case.rag, fw.full_logits, model.cfg.partition.final_merge_threshold, stage="final")
        for event in spec.events:
            pair_total += 1; pair_ok += int(pair_exact(runtime, predicted, event))
        a, b = clean_split_counts(runtime, predicted, spec.events); clean_total += a; clean_split += b
        violations += split_only_violations(case, predicted)
    cut_acc = cut_correct / max(cut_total, 1); keep_acc = keep_correct / max(keep_total, 1)
    content_acc = contentless_cut_correct / max(cut_total, 1); exact = pair_ok / max(pair_total, 1)
    false_split = clean_split / max(clean_total, 1); gap = cut_acc - content_acc
    strict = bool(cut_acc >= .90 and keep_acc >= .98 and exact >= .80 and false_split <= .02 and gap >= .15 and violations == 0)
    score = 2.5 * cut_acc + 2.0 * exact + 1.5 * keep_acc - 4.0 * false_split + 1.5 * gap
    INV35.set_temporal_train_mode(model, True)
    return {
        "objective_version": OBJECTIVE_VERSION, "cases": len(ordered), "merge_pairs": pair_total,
        "cut_edges": cut_total, "keep_edges": keep_total, "cut_accuracy": float(cut_acc),
        "keep_accuracy": float(keep_acc), "exact_merge_recovery": float(exact),
        "clean_false_split_rate": float(false_split), "contentless_cut_accuracy": float(content_acc),
        "full_minus_contentless_cut": float(gap), "split_only_violations": int(violations),
        "strict_pass": strict, "checkpoint_score": float(score),
    }


def print_metrics(title: str, step: int, m):
    print("\n" + "=" * 112); print(f"{title} @ STEP {step}"); print("=" * 112)
    print(f"CUT accuracy             : {m['cut_accuracy']:.4f}")
    print(f"KEEP accuracy            : {m['keep_accuracy']:.4f}")
    print(f"exact merge recovery     : {m['exact_merge_recovery']:.4f}")
    print(f"clean false split        : {m['clean_false_split_rate']:.4f}")
    print(f"CONTENTLESS CUT accuracy : {m['contentless_cut_accuracy']:.4f}")
    print(f"FULL - CONTENTLESS CUT   : {m['full_minus_contentless_cut']:+.4f}")
    print(f"split-only violations    : {m['split_only_violations']}")
    print(f"STRICT PASS              : {m['strict_pass']}")
    print(f"checkpoint score         : {m['checkpoint_score']:.5f}")
    print("=" * 112)


def make_training_config(args: argparse.Namespace) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.lr = float(args.lr)
    cfg.weight_decay = float(args.weight_decay)
    cfg.max_grad_norm = float(args.grad_clip)
    cfg.amp_dtype = str(args.amp_dtype)
    cfg.curriculum.fixed_stage = "instance_temporal"
    cfg.curriculum.instance_temporal_detached_spatial = True
    cfg.curriculum.instance_temporal_freeze_spatial = True
    cfg.loss.temporal_causal_enabled = True
    cfg.loss.temporal_causal_noop_weight = float(args.noop_weight)
    cfg.loss.temporal_causal_corrupted_gate_weight = float(args.corrupted_gate_weight)
    cfg.loss.temporal_causal_margin_weight = float(args.margin_weight)
    cfg.loss.temporal_causal_margin = float(args.margin)
    cfg.loss.temporal_causal_corruptions = ("contentless",)
    cfg.loss.temporal_causal_seed = int(args.seed) + 38_000
    cfg.validate()
    return cfg


def save_training(path: Path, *, model, optimizer, scaler, step: int,
                  training_config: TrainingConfig, paths: Paths,
                  args: argparse.Namespace, dref_um: float, metrics,
                  parameter_audit):
    save_checkpoint(
        path, model=model, optimizer=optimizer, scaler=scaler, step=int(step), epoch=0,
        model_config=model.cfg, training_config=training_config,
        extra={
            "investigation": SCRIPT_NAME, "objective_version": OBJECTIVE_VERSION,
            "sample_id": paths.sample, "dref_um": float(dref_um),
            "spacing_zyx_um": tuple(float(v) for v in args.spacing),
            "initializer": str(paths.init_checkpoint),
            "dataset_manifest": file_signature(paths.dataset_manifest),
            "parameter_audit": parameter_audit, "validation_metrics": metrics or {},
            "notes": {
                "concrete_trackastra_variants": True,
                "dynamic_synthetic_graph_construction": False,
                "temporal_action": "split_only",
                "cross_current_component_edges": "immutable_cut",
                "corruptions": ["contentless"],
                "raw_movie_duplicated": False,
                "spatial_parameters_optimized": False,
            },
        },
    )


def load_model(paths: Paths, device: torch.device, resume: Path | None):
    path = resolve(resume) if resume else paths.init_checkpoint
    payload, model, _, _, _ = INV35.INV12.load_checkpoint_model_for_inference(path, device)
    return path, payload, model


def train(paths: Paths, plans: Sequence[VariantPlan], case_index, *, spacing,
          dref_um: float, device: torch.device, args: argparse.Namespace):
    resume = resolve(args.resume) if args.resume else None
    loaded_path, payload, model = load_model(paths, device, resume)
    training_config = make_training_config(args)
    trainable = INV35.configure_temporal_training(model, device)
    parameter_audit = INV35.temporal_parameter_audit(model)
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = make_grad_scaler(device, str(args.amp_dtype))
    start_step = 0
    if resume:
        start_step = int(payload.get("global_step", 0))
        if "optimizer" in payload: optimizer.load_state_dict(payload["optimizer"])
        if "scaler" in payload:
            try: scaler.load_state_dict(payload["scaler"])
            except Exception as exc: print(f"[resume] scaler not restored: {exc}")
        print(f"[resume] {loaded_path} after step {start_step}")

    train_cases, val_cases, test_cases = case_index["train"], case_index["val"], case_index["test"]
    if not train_cases or not val_cases or not test_cases:
        raise RuntimeError("train/val/test each need at least one variant-frame case")
    by_frame = defaultdict(list)
    for row in train_cases: by_frame[row.frame].append(row)
    frames = sorted(by_frame)
    loader = RuntimeFrameLoader(paths, device)

    history, validation_history = [], []
    if resume and paths.training_history.is_file():
        try: history = list(json.loads(paths.training_history.read_text(encoding="utf-8")))
        except Exception: pass
    if resume and paths.validation_history.is_file():
        try: validation_history = list(json.loads(paths.validation_history.read_text(encoding="utf-8")))
        except Exception: pass
    best_score, best_strict = -float("inf"), False
    if paths.best_metrics.is_file():
        try:
            old = json.loads(paths.best_metrics.read_text(encoding="utf-8"))
            if int(old.get("objective_version", -1)) == OBJECTIVE_VERSION:
                best_score = float(old.get("checkpoint_score", -float("inf")))
                best_strict = bool(old.get("strict_pass", False))
        except Exception: pass

    print("\n" + "=" * 112)
    print("INVESTIGATION 38 — CONCRETE CUT/KEEP TEMPORAL TRAINING")
    print("=" * 112)
    print(f"device                  : {device}")
    print(f"initializer             : {loaded_path}")
    print(f"optimizer steps         : {args.steps}")
    print(f"microcases / step       : {args.accumulate_cases}")
    print(f"train variant-frames    : {len(train_cases)}")
    print(f"val variant-frames      : {len(val_cases)}")
    print(f"test variant-frames     : {len(test_cases)}")
    print(f"trainable parameters    : {parameter_audit['trainable_parameter_count']['total']:,}")
    print(f"learning rate           : {args.lr:g}")
    print(f"synthetic spatial logit : +/-{args.synthetic_spatial_logit:g}")
    print(f"preservation weight     : {args.preservation_weight:g}")
    print("temporal action         : SPLIT ONLY")
    print("causal negative         : CONTENTLESS only")
    print("spatial CNN in training : NO")
    print("=" * 112)

    run_started = time.perf_counter()
    for step in range(start_step + 1, int(args.steps) + 1):
        step_started = time.perf_counter()
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        rng = random.Random(int(args.seed) + 15_485_863 * step)
        frame = rng.choice(frames); runtime = loader.load(frame); candidates = by_frame[frame]
        optimizer.zero_grad(set_to_none=True)
        sums = {k: 0.0 for k in ("loss", "correction", "preservation", "split", "noop", "gate", "margin")}
        micro_meta = []
        for micro in range(int(args.accumulate_cases)):
            spec = rng.choice(candidates)
            case = build_case(runtime, variant_index=spec.variant_index, events=spec.events,
                              synthetic_logit=float(args.synthetic_spatial_logit))
            if not bool(case.correction_mask.any()):
                raise RuntimeError(f"variant={spec.variant_index} t={spec.frame} has merge events but no CUT RAG edges")
            graph = load_variant_temporal_graph(paths, spec.variant_index, spec.frame)
            with autocast_for(device, str(args.amp_dtype)):
                fw = forward_case(model, runtime=runtime, case=case, graph=graph,
                                  spacing=spacing, dref_um=dref_um, device=device)
                loss = compute_loss(
                    case, fw, preserve_edges=int(args.preserve_edges), preserve_ratio=int(args.preserve_ratio),
                    preservation_weight=float(args.preservation_weight), split_weight=float(args.split_weight),
                    noop_weight=float(args.noop_weight), gate_weight=float(args.corrupted_gate_weight),
                    margin_weight=float(args.margin_weight), margin=float(args.margin), rng=rng,
                )
                scaled = loss.total / float(args.accumulate_cases)
            if not bool(torch.isfinite(scaled.detach())):
                raise FloatingPointError(f"Non-finite loss at step={step} micro={micro}")
            scaler.scale(scaled).backward()
            sums["loss"] += float(loss.total.detach().float().cpu())
            sums["correction"] += float(loss.correction.detach().float().cpu())
            sums["preservation"] += float(loss.preservation.detach().float().cpu())
            sums["split"] += float(loss.split.detach().float().cpu())
            sums["noop"] += float(loss.noop.detach().float().cpu())
            sums["gate"] += float(loss.gate.detach().float().cpu())
            sums["margin"] += float(loss.margin.detach().float().cpu())
            micro_meta.append({"variant": spec.variant_index, "frame": spec.frame,
                               "merge_count": len(spec.events), "cut_edges": loss.correction_edges,
                               "keep_edges": loss.preservation_edges})
        scaler.unscale_(optimizer)
        grad_t = torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
        grad = float(torch.as_tensor(grad_t).detach().cpu())
        if not math.isfinite(grad): raise FloatingPointError(f"Non-finite grad at step={step}")
        scaler.step(optimizer); scaler.update()
        denom = max(int(args.accumulate_cases), 1)
        row = {
            "step": step, "frame": frame,
            "loss": sums["loss"] / denom,
            "correction_loss": sums["correction"] / denom,
            "preservation_loss": sums["preservation"] / denom,
            "split_loss": sums["split"] / denom,
            "contentless_noop": sums["noop"] / denom,
            "contentless_gate": sums["gate"] / denom,
            "causal_margin": sums["margin"] / denom,
            "grad_norm": grad, "step_seconds": time.perf_counter() - step_started,
            "microcases": micro_meta,
        }
        history.append(row)
        if step == 1 or step % int(args.print_every) == 0:
            print(f"[step {step:05d}/{args.steps}] t={frame:02d} loss={row['loss']:.5f} "
                  f"cut={row['correction_loss']:.5f} keep={row['preservation_loss']:.5f} "
                  f"noop={row['contentless_noop']:.5f} margin={row['causal_margin']:.5f} "
                  f"grad={grad:.3f} elapsed={duration(time.perf_counter() - run_started)}", flush=True)
        if step % 50 == 0: atomic_json(paths.training_history, history)
        if step % int(args.eval_every) == 0 or step == int(args.steps):
            metrics = evaluate(
                model, paths=paths, cases=val_cases, loader=loader, spacing=spacing,
                dref_um=dref_um, synthetic_logit=float(args.synthetic_spatial_logit),
                device=device, maximum_cases=int(args.val_cases), seed=int(args.seed) + step,
            )
            print_metrics("INVESTIGATION 38 VALIDATION", step, metrics)
            validation_history.append({"step": step, "metrics": metrics})
            atomic_json(paths.validation_history, validation_history); atomic_json(paths.training_history, history)
            save_training(paths.latest, model=model, optimizer=optimizer, scaler=scaler, step=step,
                          training_config=training_config, paths=paths, args=args, dref_um=dref_um,
                          metrics=metrics, parameter_audit=parameter_audit)
            score, strict = float(metrics["checkpoint_score"]), bool(metrics["strict_pass"])
            if (strict and not best_strict) or (strict == best_strict and score > best_score):
                best_score, best_strict = score, strict
                save_training(paths.best, model=model, optimizer=optimizer, scaler=scaler, step=step,
                              training_config=training_config, paths=paths, args=args, dref_um=dref_um,
                              metrics=metrics, parameter_audit=parameter_audit)
                atomic_json(paths.best_metrics, metrics)
                print(f"[best] step={step} score={score:.5f} strict={strict}")

    atomic_json(paths.training_history, history); atomic_json(paths.validation_history, validation_history)
    final_metrics = validation_history[-1]["metrics"] if validation_history else {}
    save_training(paths.final, model=model, optimizer=optimizer, scaler=scaler,
                  step=max(start_step, int(args.steps)), training_config=training_config,
                  paths=paths, args=args, dref_um=dref_um, metrics=final_metrics,
                  parameter_audit=parameter_audit)
    test_metrics = evaluate(
        model, paths=paths, cases=test_cases, loader=loader, spacing=spacing,
        dref_um=dref_um, synthetic_logit=float(args.synthetic_spatial_logit),
        device=device, maximum_cases=0, seed=int(args.seed) + 38_999_999,
    )
    print_metrics("INVESTIGATION 38 HELD-OUT VARIANT TEST", max(start_step, int(args.steps)), test_metrics)
    atomic_json(paths.test_metrics, test_metrics)
    print("\n" + "=" * 112)
    print("INVESTIGATION 38 TRAINING COMPLETE")
    print("=" * 112)
    print(f"best   : {paths.best}")
    print(f"latest : {paths.latest}")
    print(f"final  : {paths.final}")
    print(f"test   : {paths.test_metrics}")
    print("=" * 112)


def prepare(paths: Paths, args: argparse.Namespace, spacing, device: torch.device):
    preparation = json.loads(paths.preparation.read_text(encoding="utf-8"))
    dref_um = float(preparation["dref_um"])
    catalog = build_touching_catalog(
        paths, frames=int(args.frame_count), spacing=spacing, dref_um=dref_um,
        min_voxels=int(args.min_voxels), max_volume_ratio=float(args.max_volume_ratio),
        max_distance_dref=float(args.max_distance_dref), rebuild=bool(args.rebuild_dataset),
    )
    plans = generate_plans(
        paths, catalog, frames=int(args.frame_count), train_variants=int(args.train_variants),
        val_variants=int(args.val_variants), test_variants=int(args.test_variants),
        merge_fraction=float(args.merge_fraction), max_merges_per_frame=int(args.max_merges_per_frame),
        seed=int(args.seed), rebuild=bool(args.rebuild_dataset),
    )
    prepare_variants(
        paths, plans, frames=int(args.frame_count), temporal_radius=int(args.temporal_radius),
        spacing=spacing, dref_um=dref_um, trackastra_model=str(args.trackastra_model),
        trackastra_mode=str(args.trackastra_mode), trackastra_device=str(args.trackastra_device),
        rebuild=bool(args.rebuild_dataset),
    )
    _, observer_model, _, _, _ = INV35.INV12.load_checkpoint_model_for_inference(paths.init_checkpoint, device)
    observer_model.eval()
    prepare_observer(
        paths, plans, frames=int(args.frame_count), spacing=spacing, dref_um=dref_um,
        model=observer_model, device=device, amp_dtype=str(args.observer_amp_dtype),
        rebuild=bool(args.rebuild_observer or args.rebuild_dataset),
    )
    del observer_model
    if device.type == "cuda": torch.cuda.empty_cache()
    gc.collect()
    cases = index_cases(plans, int(args.frame_count))
    print("\n" + "=" * 112)
    print("INVESTIGATION 38 PREPARATION READY")
    print("=" * 112)
    print(f"dref                 : {dref_um:.5f} um")
    for split in ("train", "val", "test"):
        rows = cases[split]
        print(f"{split:5s} variant-frames : {len(rows):3d} | merge_events={sum(len(r.events) for r in rows)}")
    print(f"dataset              : {paths.output}")
    print("=" * 112)
    return dref_um, plans, cases


def build_parser():
    p = argparse.ArgumentParser(description="Concrete BioHub merge variants + split-only temporal CUT/KEEP training")
    p.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    p.add_argument("--frame-count", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--spacing", default="1.625,0.40625,0.40625")
    p.add_argument("--inv35-root", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--init-checkpoint", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--temporal-radius", type=int, default=DEFAULT_TEMPORAL_RADIUS)
    p.add_argument("--train-variants", type=int, default=DEFAULT_TRAIN_VARIANTS)
    p.add_argument("--val-variants", type=int, default=DEFAULT_VAL_VARIANTS)
    p.add_argument("--test-variants", type=int, default=DEFAULT_TEST_VARIANTS)
    p.add_argument("--merge-fraction", type=float, default=DEFAULT_MERGE_FRACTION)
    p.add_argument("--max-merges-per-frame", type=int, default=DEFAULT_MAX_MERGES_PER_FRAME)
    p.add_argument("--min-voxels", type=int, default=DEFAULT_MIN_VOXELS)
    p.add_argument("--max-volume-ratio", type=float, default=DEFAULT_MAX_VOLUME_RATIO)
    p.add_argument("--max-distance-dref", type=float, default=DEFAULT_MAX_DISTANCE_DREF)
    p.add_argument("--trackastra-model", default=DEFAULT_TRACKASTRA_MODEL)
    p.add_argument("--trackastra-mode", default=DEFAULT_TRACKASTRA_MODE)
    p.add_argument("--trackastra-device", default=DEFAULT_TRACKASTRA_DEVICE)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp-dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    p.add_argument("--observer-amp-dtype", choices=("fp32", "fp16", "bf16"),
                   default="fp16" if torch.cuda.is_available() else "fp32")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    p.add_argument("--accumulate-cases", type=int, default=DEFAULT_ACCUMULATE)
    p.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    p.add_argument("--print-every", type=int, default=DEFAULT_PRINT_EVERY)
    p.add_argument("--val-cases", type=int, default=DEFAULT_VAL_CASES,
                   help="0 evaluates all held-out validation variant-frames")
    p.add_argument("--synthetic-spatial-logit", type=float, default=DEFAULT_SYNTHETIC_LOGIT)
    p.add_argument("--preserve-edges", type=int, default=DEFAULT_PRESERVE_EDGES)
    p.add_argument("--preserve-ratio", type=int, default=DEFAULT_PRESERVE_RATIO)
    p.add_argument("--preservation-weight", type=float, default=DEFAULT_PRESERVATION_WEIGHT)
    p.add_argument("--split-weight", type=float, default=DEFAULT_SPLIT_WEIGHT)
    p.add_argument("--noop-weight", type=float, default=DEFAULT_NOOP_WEIGHT)
    p.add_argument("--corrupted-gate-weight", type=float, default=DEFAULT_CORRUPTED_GATE_WEIGHT)
    p.add_argument("--margin-weight", type=float, default=DEFAULT_MARGIN_WEIGHT)
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--rebuild-dataset", action="store_true")
    p.add_argument("--rebuild-observer", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    return p


def validate_args(args):
    if args.frame_count < 3: raise ValueError("--frame-count must be >= 3")
    if args.temporal_radius < 1: raise ValueError("--temporal-radius must be positive")
    if min(args.train_variants, args.val_variants, args.test_variants) < 1:
        raise ValueError("train/val/test variant counts must each be >= 1")
    if not 0 < args.merge_fraction <= 1: raise ValueError("--merge-fraction must be in (0,1]")
    if args.max_merges_per_frame < 1: raise ValueError("--max-merges-per-frame must be >=1")
    if args.steps < 1 or args.accumulate_cases < 1: raise ValueError("steps/accumulate-cases must be positive")
    if args.eval_every < 1 or args.print_every < 1: raise ValueError("eval/print cadence must be positive")


def main() -> int:
    args = build_parser().parse_args(); validate_args(args)
    args.spacing = parse_spacing(args.spacing)
    paths = make_paths(args); validate_inputs(paths, int(args.frame_count)); paths.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    print("\n" + "=" * 112)
    print("INVESTIGATION 38 — CONCRETE BIOHUB MERGE-VARIANT TRAINING")
    print("=" * 112)
    print(f"repository : {ROOT}")
    print(f"sample     : {paths.sample}")
    print(f"Inv35 base : {paths.inv35_root}")
    print(f"initializer: {paths.init_checkpoint}")
    print(f"output     : {paths.output}")
    print("=" * 112)
    dref_um, plans, cases = prepare(paths, args, args.spacing, device)
    if args.prepare_only:
        print("Stopped after --prepare-only. No training was run.")
        return 0
    train(paths, plans, cases, spacing=args.spacing, dref_um=dref_um, device=device, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
