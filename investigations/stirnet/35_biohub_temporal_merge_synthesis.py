from __future__ import annotations

r"""
Investigation 35 — production STIR-Net temporal training with controlled BioHub merges.

Goal
----
Train the REAL production temporal stack on target-domain BioHub motion by
manufacturing many controlled under-segmentation failures from the refined
20-frame BioHub instance annotations.

Unlike Investigation 31, this uses the real production:
    InstanceTokenizer
    HistoricalInstanceEncoder
    TemporalGraphEncoder
    TemporalSpatialObserver
    InstanceTemporalReasoner
and real frozen spatial RAG node/edge embeddings/statistics from the mature
spatial checkpoint.

Synthetic example
-----------------
Clean pseudo-GT:
    A   B

Synthetic current spatial partition:
    A+B

Temporal evidence:
    t-2      t-1       t       t+1      t+2
     A        A       A+B       A        A
     B        B                 B        B

The target-frame temporal graph is leak-free: clean target detections A and B
are removed and replaced by ONE merged detection. Their temporal associations
are rewired to that merged detection. Context frames stay clean.

Frozen vs trainable
-------------------
FROZEN:
    acquisition / evidence stem / spatial backbone / geometry decoder
    watershed / RAG builder / spatial RAG network / refinement

TRAINABLE:
    InstanceTokenizer
    HistoricalInstanceEncoder
    TemporalGraphEncoder
    TemporalSpatialObserver
    InstanceTemporalReasoner

Preparation
-----------
1. Build refined-manual + raw movies.
2. Run Trackastra once on the refined manual movie.
3. Recompute REAL frozen RAG embeddings/statistics over the exact Investigation-24
   atomic supervoxels. Dense geometry uses production tiled inference, so the
   one-time preparation is suitable for the full BioHub volume.
4. Cache the explicit 11-channel dense geometry and 5-channel spatial input.
5. Build adjacent high-confidence pair/triple merge candidate pools.

During training, D1/D2/hidden-geometry observer samples are produced lazily from
frozen spatial tiles and cached by physical reference coordinate. The trainable
observer projections/message/gate remain live and receive gradients.

Temporal causal objective
-------------------------
FULL temporal state:
    - all synthetic correction edges supervised
    - balanced preservation edges supervised
    - weak split-head supervision

CORRUPTED state (alternating CONTENTLESS / SHUFFLED):
    - corruption is applied BEFORE TemporalSpatialObserver
    - spatial observer evidence remains available
    - corrupted final logits must regress to synthetic spatial logits
    - corrupted temporal gate is discouraged
    - FULL must beat corrupted by a directional margin on correction edges

Applying corruption before the observer is deliberately stronger than the
Investigation-31 diagnostic: static current-frame observer evidence alone is
not allowed to solve the synthetic merge.

Dynamic difficulty
------------------
    - pair merges
    - occasional connected 3-cell merges
    - clean preservation cases
    - full finite context
    - past-only context
    - future-only context
    - random context-frame dropout

Validation
----------
A deterministic held-out adjacent-pair pool reports:
    FULL correction-edge accuracy
    preservation-edge accuracy
    exact recovery of selected merged cells
    SHUFFLED correction accuracy
    CONTENTLESS correction accuracy
    FULL-SHUFFLED and FULL-CONTENTLESS causal gaps

This is same-movie held-out corruption validation, NOT cross-movie generalization.

Recommended usage
-----------------
From repository root:

    python .\investigations\stirnet\35_biohub_temporal_merge_synthesis.py --prepare-only

Then train:

    python .\investigations\stirnet\35_biohub_temporal_merge_synthesis.py

Smoke run:

    python .\investigations\stirnet\35_biohub_temporal_merge_synthesis.py ^
        --steps 100 --eval-every 50 --val-cases 8

Default outputs
---------------
    runs/stirnet/evaluation/35_biohub_temporal_merge_synthesis/<sample>/
        preparation.json
        candidate_manifest.json
        cache/
            manual_movie.npy
            raw_movie.npy
            raw_normalized/
            trackastra/
            temporal_static.pkl
            spatial/tXXX/
                frozen_graph.pt
                spatial_inputs.npy
                explicit_geometry.npy
                meta.json
        training_history.json
        validation_history.json
        latest.pt
        best.pt
        best_metrics.json
        final.pt
"""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


# =============================================================================
# Repository / production imports
# =============================================================================


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (
            (candidate / "learned").is_dir()
            and (candidate / "investigations").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (
            (candidate / "learned").is_dir()
            and (candidate / "investigations").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    raise RuntimeError("Could not resolve repository root")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


INV12 = load_module(
    ROOT / "investigations/stirnet/data/12_biohub_full_volume_spatial_inference.py",
    "_inv12_for_inv35",
)
V13 = load_module(
    ROOT / "investigations/stirnet/data/13_biohub_full_volume_spatial_results_viewer.py",
    "_inv13_for_inv35",
)
INV30 = load_module(
    ROOT / "investigations/stirnet/30_biohub_temporal_partition_overfit.py",
    "_inv30_for_inv35",
)

from src.io import load_timepoint

from learned.stirnet.data.graph_builder import (
    AssociationRecord,
    DetectionRecord,
    build_temporal_graph,
    sequence_available_time_offsets,
)
from learned.stirnet.data.historical_instances import (
    TEMPORAL_CACHE_CONTRACT_VERSION,
    build_historical_instance_grid,
)
from learned.stirnet.data.sample_builder import build_spatial_channels, robust_normalize
from learned.stirnet.data.targets import estimate_model_dref_um, extract_instance_metadata
from learned.stirnet.inference.tiled_dense import (
    generate_dense_tiles,
    stream_tiled_label_feature_stats,
    tile_blend_weight,
    tiled_dense_geometry,
)
from learned.stirnet.model.geometry.derived import build_geometry_derived_cache
from learned.stirnet.model.partition.statistics import build_supervoxel_statistics
from learned.stirnet.model.temporal.observer import _sample_explicit_geometry, _sample_local_grid
from learned.stirnet.model.types import (
    GeometryState,
    PartitionState,
    RAGState,
    SpatialDecodeState,
    TemporalInput,
    TemporalState,
)
from learned.stirnet.training import TrainingConfig
from learned.stirnet.training.checkpoint import save_checkpoint
from learned.stirnet.training.temporal_causal import (
    contentless_temporal_state,
    shuffled_temporal_state,
)


# =============================================================================
# Defaults
# =============================================================================

SCRIPT_NAME = "35_biohub_temporal_merge_synthesis"
DEFAULT_SAMPLE = "44b6_0113de3b"
DEFAULT_FRAMES = 20
DEFAULT_SPACING = (1.625, 0.40625, 0.40625)
DEFAULT_SPATIAL_CHECKPOINT = ROOT / "runs/stirnet/milestones/drosophila_12_spatial_v1"
DEFAULT_INV24 = ROOT / "runs/stirnet/evaluation/24_multicut_biohub_full_volume_visualization"
DEFAULT_ANNOTATIONS = ROOT / "evaluation/segmentation/annotations"
DEFAULT_OUTPUT = ROOT / "runs/stirnet/evaluation" / SCRIPT_NAME

DEFAULT_STEPS = 5000
DEFAULT_LR = 2e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_ACCUMULATE = 4
DEFAULT_FRAME_BLOCK = 8
DEFAULT_EVAL_EVERY = 250
DEFAULT_PRINT_EVERY = 10
DEFAULT_VAL_CASES = 32

DEFAULT_SYNTHETIC_LOGIT = 3.5
DEFAULT_CLEAN_FRACTION = 0.25
DEFAULT_TRIPLE_FRACTION = 0.10
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_MIN_VOXELS = 64
DEFAULT_MAX_VOLUME_RATIO = 3.0
DEFAULT_MAX_DISTANCE_DREF = 2.50
DEFAULT_TEMPORAL_NEIGHBORHOOD_DREF = 4.0
DEFAULT_LOCAL_EDGE_RADIUS_DREF = 4.5
DEFAULT_MAX_TRIPLES_PER_FRAME = 32

DEFAULT_PRESERVE_EDGES = 512
DEFAULT_PRESERVE_RATIO = 12
DEFAULT_SPLIT_WEIGHT = 0.05
DEFAULT_NOOP_WEIGHT = 0.50
DEFAULT_CORRUPTED_GATE_WEIGHT = 0.05
DEFAULT_MARGIN_WEIGHT = 0.50
DEFAULT_MARGIN = 1.0

CORRUPTIONS = ("contentless", "shuffled")
CACHE_VERSION = 2
OBSERVER_CACHE_VERSION = 1


# =============================================================================
# Generic helpers
# =============================================================================


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if torch.is_tensor(value):
        x = value.detach().cpu()
        return jsonable(x.item()) if x.ndim == 0 else jsonable(x.tolist())
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)


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


def torch_load(path: Path, map_location="cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def stable_fraction(*values: Any) -> float:
    digest = hashlib.sha256("|".join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def map_tree(value: Any, fn):
    if torch.is_tensor(value):
        return fn(value)
    if dataclasses.is_dataclass(value):
        return type(value)(**{
            field.name: map_tree(getattr(value, field.name), fn)
            for field in dataclasses.fields(value)
        })
    if isinstance(value, dict):
        return {k: map_tree(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [map_tree(v, fn) for v in value]
    if isinstance(value, tuple):
        return tuple(map_tree(v, fn) for v in value)
    return value


def tree_cpu(value: Any) -> Any:
    return map_tree(value, lambda x: x.detach().cpu())


def tree_device(value: Any, device: torch.device) -> Any:
    return map_tree(value, lambda x: x.to(device, non_blocking=True))


# STIRNET_INV35_FROZEN_CACHE_TRAINING_DTYPE_V1
def tree_device_training_float(
    value: Any,
    device: torch.device,
    floating_dtype: torch.dtype,
) -> Any:
    # Restore floating cache tensors to the dtype used by trainable modules.
    def convert(tensor: Tensor) -> Tensor:
        if tensor.is_floating_point():
            return tensor.to(
                device=device,
                dtype=floating_dtype,
                non_blocking=True,
            )
        return tensor.to(
            device=device,
            non_blocking=True,
        )

    return map_tree(value, convert)


def autocast_for(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "fp32":
        return nullcontext()
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)


# =============================================================================
# Paths / checkpoint
# =============================================================================


@dataclass(frozen=True)
class Paths:
    sample: str
    output: Path
    annotations: Path
    inv24: Path
    stage6: Path
    zarr: Path
    checkpoint: Path

    @property
    def cache(self): return self.output / "cache"
    @property
    def manual_movie(self): return self.cache / "manual_movie.npy"
    @property
    def manual_movie_meta(self): return self.cache / "manual_movie_meta.json"
    @property
    def raw_movie(self): return self.cache / "raw_movie.npy"
    @property
    def temporal_static(self): return self.cache / "temporal_static.pkl"
    @property
    def trackastra(self): return self.cache / "trackastra"
    @property
    def track_graph(self): return self.trackastra / "track_graph.pkl"
    @property
    def tracked_masks(self): return self.trackastra / "tracked_masks.npy"
    @property
    def spatial_cache(self): return self.cache / "spatial"
    @property
    def raw_norm_cache(self): return self.cache / "raw_normalized"
    @property
    def candidate_manifest(self): return self.output / "candidate_manifest.json"

    def manual(self, t): return self.annotations / f"manual_instances_t{t:03d}.npy"
    def supervoxels(self, t): return self.inv24 / f"t{t:03d}/partition/watershed_supervoxels.npy"
    def preprocessed(self, t): return self.stage6 / "preprocessing" / f"t{t:03d}.npy"
    def source(self, t): return self.stage6 / "segmentation" / f"t{t:03d}.npy"
    def raw_norm(self, t): return self.raw_norm_cache / f"t{t:03d}.npy"
    def frame_dir(self, t): return self.spatial_cache / f"t{t:03d}"
    def graph_cache(self, t): return self.frame_dir(t) / "frozen_graph.pt"
    def spatial_inputs(self, t): return self.frame_dir(t) / "spatial_inputs.npy"
    def explicit_geometry(self, t): return self.frame_dir(t) / "explicit_geometry.npy"
    def observer_refs(self, t): return self.frame_dir(t) / "observer_ref_um.npy"
    def observer_d1(self, t): return self.frame_dir(t) / "observer_d1_raw.npy"
    def observer_d2(self, t): return self.frame_dir(t) / "observer_d2_raw.npy"
    def observer_hidden(self, t): return self.frame_dir(t) / "observer_hidden_raw.npy"
    def observer_explicit(self, t): return self.frame_dir(t) / "observer_explicit_raw.npy"
    def observer_meta(self, t): return self.frame_dir(t) / "observer_meta.json"
    def frame_meta(self, t): return self.frame_dir(t) / "meta.json"


def resolve_checkpoint(value: Path | None) -> Path:
    root = DEFAULT_SPATIAL_CHECKPOINT if value is None else resolve(value)
    if root.is_file():
        return root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    for name in ("best.pt", "final.pt", "latest.pt", "checkpoint.pt"):
        candidate = root / name
        if candidate.is_file():
            return candidate.resolve()
    candidates = list(root.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoint below {root}")
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name)).resolve()


def make_paths(args) -> Paths:
    sample = args.sample_id
    output = resolve(args.output) if args.output else (DEFAULT_OUTPUT / sample).resolve()
    annotations = resolve(args.annotations) if args.annotations else (DEFAULT_ANNOTATIONS / sample).resolve()
    inv24 = resolve(args.inv24) if args.inv24 else (DEFAULT_INV24 / sample / "h100_q0p845").resolve()
    stage6 = V13.resolve_stage6_root(sample, None if args.stage6_root is None else str(args.stage6_root))
    zarr = V13.resolve_sample_zarr(sample, None if args.sample_zarr is None else str(args.sample_zarr))
    return Paths(sample, output, annotations, inv24, stage6, zarr, resolve_checkpoint(args.checkpoint))


def validate_inputs(paths: Paths, frame_count: int) -> None:
    missing = []
    for t in range(frame_count):
        for path in (paths.manual(t), paths.supervoxels(t), paths.preprocessed(t), paths.source(t)):
            if not path.is_file(): missing.append(path)
    if not paths.zarr.exists(): missing.append(paths.zarr)
    if not paths.checkpoint.is_file(): missing.append(paths.checkpoint)
    if missing:
        preview = "\n".join(f"  {p}" for p in missing[:40])
        raise FileNotFoundError("Missing Investigation-35 inputs:\n" + preview)


def load_spatial_model(checkpoint: Path, device: torch.device):
    checkpoint_payload, model, _, _, _ = INV12.load_checkpoint_model_for_inference(checkpoint, device)
    model.eval()
    return checkpoint_payload, model


# =============================================================================
# Movie / raw caches
# =============================================================================


def stack_movie(target: Path, frame_paths: Sequence[Path], *, rebuild: bool) -> Path:
    if target.is_file() and not rebuild:
        existing = np.load(target, mmap_mode="r")
        if existing.shape[0] == len(frame_paths): return target
    first = np.load(frame_paths[0], mmap_mode="r", allow_pickle=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    movie = np.lib.format.open_memmap(target, mode="w+", dtype=first.dtype, shape=(len(frame_paths), *first.shape))
    for index, path in enumerate(frame_paths):
        frame = np.load(path, mmap_mode="r", allow_pickle=False)
        if frame.shape != first.shape: raise ValueError(f"Frame shape mismatch: {path}")
        movie[index] = frame
    movie.flush(); del movie
    return target


def build_raw_movie(paths: Paths, frame_count: int, shape: tuple[int, int, int], *, rebuild: bool) -> Path:
    if paths.raw_movie.is_file() and not rebuild:
        x = np.load(paths.raw_movie, mmap_mode="r")
        if x.shape == (frame_count, *shape): return paths.raw_movie
    first = np.asarray(load_timepoint(paths.zarr, 0))
    if first.shape != shape: raise ValueError(f"Raw/manual shape mismatch: {first.shape} vs {shape}")
    paths.raw_movie.parent.mkdir(parents=True, exist_ok=True)
    movie = np.lib.format.open_memmap(paths.raw_movie, mode="w+", dtype=first.dtype, shape=(frame_count, *shape))
    movie[0] = first
    for t in range(1, frame_count):
        print(f"[raw] t={t:03d}", flush=True)
        frame = np.asarray(load_timepoint(paths.zarr, t))
        if frame.shape != shape: raise ValueError(f"Raw shape changed at t={t}")
        movie[t] = frame
    movie.flush(); del movie
    return paths.raw_movie


def prepare_raw_norm(paths: Paths, frame_count: int, *, rebuild: bool) -> None:
    paths.raw_norm_cache.mkdir(parents=True, exist_ok=True)
    raw = np.load(paths.raw_movie, mmap_mode="r")
    for t in range(frame_count):
        target = paths.raw_norm(t)
        if target.is_file() and not rebuild: continue
        print(f"[raw norm] t={t:03d}", flush=True)
        atomic_npy(target, robust_normalize(np.asarray(raw[t])).astype(np.float32))


def resolve_dref(paths: Paths, frame_count: int, spacing, explicit: float | None) -> tuple[float, list[float]]:
    if explicit is not None:
        if explicit <= 0: raise ValueError("--dref-um must be positive")
        return float(explicit), [float(explicit)] * frame_count
    values = []
    for t in range(frame_count):
        source = np.load(paths.source(t), mmap_mode="r", allow_pickle=False)
        values.append(float(estimate_model_dref_um(np.asarray(source), tuple(spacing))))
    return float(np.median(np.asarray(values))), values


# =============================================================================
# Trackastra + static temporal metadata
# =============================================================================


def prepare_trackastra(paths: Paths, *, model_name: str, mode: str, device: str, rebuild: bool):
    paths.trackastra.mkdir(parents=True, exist_ok=True)
    if paths.track_graph.is_file() and paths.tracked_masks.is_file() and not rebuild:
        print("[trackastra] reuse")
        with paths.track_graph.open("rb") as handle: graph = pickle.load(handle)
        return graph, np.load(paths.tracked_masks, mmap_mode="r")
    try:
        from trackastra.model import Trackastra
    except ImportError as exc:
        raise RuntimeError("Trackastra is required once for Investigation 35 preparation") from exc
    raw = np.load(paths.raw_movie, mmap_mode="r")
    manual = np.load(paths.manual_movie, mmap_mode="r")
    started = time.perf_counter()
    model = Trackastra.from_pretrained(model_name, device=device)
    graph, masks = model.track(raw, manual, mode=mode)
    with paths.track_graph.open("wb") as handle: pickle.dump(graph, handle)
    np.save(paths.tracked_masks, np.asarray(masks), allow_pickle=False)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"[trackastra] nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} time={duration(time.perf_counter()-started)}")
    return graph, np.load(paths.tracked_masks, mmap_mode="r")


@dataclass
class TemporalStatic:
    records: dict[int, DetectionRecord]
    associations: list[AssociationRecord]
    nodes_by_time: dict[int, list[int]]
    manual_to_node: dict[int, dict[int, int]]
    node_to_manual: dict[int, dict[int, int]]
    maximum_node_id: int


def one_to_one_overlap(manual: np.ndarray, tracked: np.ndarray, purity: float = 0.98):
    m = np.asarray(manual, np.int64).reshape(-1)
    t = np.asarray(tracked, np.int64).reshape(-1)
    positive = (m > 0) & (t > 0)
    if not positive.any(): return {}, {}
    pairs, counts = np.unique(np.stack([m[positive], t[positive]], 1), axis=0, return_counts=True)
    mc = dict(zip(*np.unique(m[m > 0], return_counts=True)))
    tc = dict(zip(*np.unique(t[t > 0], return_counts=True)))
    bm, bt = defaultdict(list), defaultdict(list)
    for (mi, ti), count in zip(pairs.tolist(), counts.tolist()):
        bm[int(mi)].append((int(ti), int(count))); bt[int(ti)].append((int(mi), int(count)))
    m2t, t2m = {}, {}
    for mi, rows in bm.items():
        if len(rows) != 1: continue
        ti, n = rows[0]
        if len(bt[ti]) != 1 or bt[ti][0][0] != mi: continue
        if n / max(int(mc[mi]), 1) < purity or n / max(int(tc[ti]), 1) < purity: continue
        m2t[mi] = ti; t2m[ti] = mi
    return m2t, t2m


def prepare_temporal_static(paths: Paths, track_graph, tracked_movie, raw_movie, spacing, dref_um, frame_count: int, *, rebuild: bool) -> TemporalStatic:
    if paths.temporal_static.is_file() and not rebuild:
        print("[temporal static] reuse")
        with paths.temporal_static.open("rb") as handle: return pickle.load(handle)
    records = INV30.build_static_detection_records(track_graph, tracked_movie, raw_movie, tuple(spacing), float(dref_um))
    associations = INV30.trackastra_associations(track_graph)
    nodes_by_time = defaultdict(list)
    time_label_to_node = {}
    for node_id, data in track_graph.nodes(data=True):
        node_id = int(node_id); t = int(data["time"]); label = int(data["label"])
        nodes_by_time[t].append(node_id); time_label_to_node[(t, label)] = node_id
    manual_movie = np.load(paths.manual_movie, mmap_mode="r")
    manual_to_node, node_to_manual = {}, {}
    for t in range(frame_count):
        m2track, track2m = one_to_one_overlap(np.asarray(manual_movie[t]), np.asarray(tracked_movie[t]))
        m2n, n2m = {}, {}
        for mid, track_label in m2track.items():
            node = time_label_to_node.get((t, track_label))
            if node is not None and node in records:
                m2n[mid] = node; n2m[node] = mid
        manual_to_node[t] = m2n; node_to_manual[t] = n2m
        count = int(np.unique(np.asarray(manual_movie[t])[np.asarray(manual_movie[t]) > 0]).size)
        print(f"[temporal map t={t:03d}] {len(m2n)}/{count} manual cells mapped 1:1", flush=True)
    result = TemporalStatic(
        records=records,
        associations=associations,
        nodes_by_time={int(k): sorted(map(int, v)) for k, v in nodes_by_time.items()},
        manual_to_node=manual_to_node,
        node_to_manual=node_to_manual,
        maximum_node_id=max([int(v) for v in track_graph.nodes] or [0]),
    )
    with paths.temporal_static.open("wb") as handle: pickle.dump(result, handle)
    return result

# =============================================================================
# Frozen real spatial cache over exact Investigation-24 atomic supervoxels
# =============================================================================


def frame_cache_matches(paths: Paths, t: int, checkpoint_sig: dict, dref_um: float) -> bool:
    required = (paths.graph_cache(t), paths.spatial_inputs(t), paths.explicit_geometry(t), paths.frame_meta(t))
    if any(not p.is_file() for p in required): return False
    try: meta = json.loads(paths.frame_meta(t).read_text(encoding="utf-8"))
    except Exception: return False
    return (
        meta.get("cache_version") == CACHE_VERSION
        and meta.get("checkpoint_signature") == checkpoint_sig
        and abs(float(meta.get("dref_um", -1)) - float(dref_um)) < 1e-6
    )


def prepare_spatial_cache(paths: Paths, *, frame_count: int, spacing, dref_um: float, device: torch.device, amp_dtype: str, rebuild: bool) -> None:
    paths.spatial_cache.mkdir(parents=True, exist_ok=True)
    signature = file_signature(paths.checkpoint)
    todo = [t for t in range(frame_count) if rebuild or not frame_cache_matches(paths, t, signature, dref_um)]
    if not todo:
        print("[spatial cache] all frames ready")
        return

    _, model = load_spatial_model(paths.checkpoint, device)
    # Force the production bounded dense path during one-time preparation.
    inference_cfg = dataclasses.replace(
        model.cfg.inference,
        mode="tiled",
        tiled_dense_enabled=True,
    )
    spacing_t = torch.tensor([spacing], device=device, dtype=torch.float32)
    dref_t = torch.tensor([dref_um], device=device, dtype=torch.float32)

    print("=" * 118)
    print("INVESTIGATION 35 — ONE-TIME REAL SPATIAL CACHE")
    print(f"checkpoint : {paths.checkpoint}")
    print(f"device     : {device}")
    print(f"frames     : {todo}")
    print("watershed  : exact Investigation-24 atoms; watershed itself is NOT rerun")
    print("=" * 118)

    for t in todo:
        frame_dir = paths.frame_dir(t); frame_dir.mkdir(parents=True, exist_ok=True)
        pre = np.asarray(np.load(paths.preprocessed(t), mmap_mode="r", allow_pickle=False), np.float32)
        source = np.asarray(np.load(paths.source(t), mmap_mode="r", allow_pickle=False))
        manual = np.asarray(np.load(paths.manual(t), mmap_mode="r", allow_pickle=False))
        sv_np = np.asarray(np.load(paths.supervoxels(t), mmap_mode="r", allow_pickle=False), np.int64)
        if not (pre.shape == source.shape == manual.shape == sv_np.shape):
            raise ValueError(f"t={t}: spatial input / manual / SV shape mismatch")

        spatial_np = build_spatial_channels(pre, source, tuple(spacing), float(dref_um), derive_marker=True)
        # Float16 disk cache is only for future frozen tile replays; preparation itself uses float32.
        atomic_npy(paths.spatial_inputs(t), spatial_np.astype(np.float16))
        spatial = torch.from_numpy(np.ascontiguousarray(spatial_np))[None].to(device)
        sv = torch.from_numpy(sv_np.copy()).to(device=device, dtype=torch.long)

        started = time.perf_counter()
        with torch.inference_mode(), autocast_for(device, amp_dtype):
            dense = tiled_dense_geometry(model, spatial, spacing_t, dref_t, config=inference_cfg)
            derived = build_geometry_derived_cache(dense.geometry, model.cfg.partition)
            streamed = stream_tiled_label_feature_stats(
                model,
                spatial,
                spacing_t,
                dref_t,
                [sv],
                dense.blend_weight_sum,
                config=inference_cfg,
            )
            statistics = build_supervoxel_statistics(
                [sv],
                spatial,
                dense.geometry,
                spacing_t,
                None,
                derived=derived,
                pooled_scales=streamed.pooled_scales,
                pooled_counts=streamed.counts_scales,
            )
            dummy = spatial.new_zeros((1, model.cfg.spatial.channels[0], 1, 1, 1))
            rag = model.rag_builder(
                [sv],
                dummy,
                spatial,
                dense.geometry,
                spacing_t,
                dref_t,
                pooled_d0_by_batch=streamed.pooled_scales[0],
                statistics_by_batch=statistics,
                derived_cache=derived,
                profile_prefix="inv35",
            )
            rag = model.rag_network(rag)
            actual_partition = model.partitioner(
                rag,
                rag.spatial_edge_logits,
                model.cfg.partition.spatial_merge_threshold,
                stage="spatial",
            )

        explicit = torch.cat([
            dense.geometry.foreground_logits.sigmoid(),
            dense.geometry.surface_logits.sigmoid(),
            dense.geometry.separator_logits.sigmoid(),
            dense.geometry.sdf,
            dense.geometry.flow,
            dense.geometry.centroid_offset,
            dense.geometry.seed_logits.sigmoid(),
        ], dim=1)
        atomic_npy(paths.explicit_geometry(t), explicit[0].detach().to(torch.float16).cpu().numpy())
        atomic_torch(paths.graph_cache(t), {
            "cache_version": CACHE_VERSION,
            "rag": tree_cpu(rag),
            "actual_partition": tree_cpu(actual_partition),
        })
        meta = {
            "cache_version": CACHE_VERSION,
            "checkpoint_signature": signature,
            "timepoint": t,
            "dref_um": float(dref_um),
            "spacing_zyx_um": list(map(float, spacing)),
            "node_count": int(rag.node_features.shape[0]),
            "edge_count": int(rag.edge_index.shape[1]),
            "actual_spatial_instances": int(actual_partition.labels[0].max().item()),
            "tile_shape_zyx": list(inference_cfg.tile_shape_zyx),
            "tile_overlap_zyx": list(inference_cfg.tile_overlap_zyx),
            "tile_halo_zyx": list(inference_cfg.tile_halo_zyx),
            "tile_batch_size": int(inference_cfg.tile_batch_size),
            "seconds": time.perf_counter() - started,
        }
        atomic_json(paths.frame_meta(t), meta)
        print(f"[spatial cache t={t:03d}] nodes={meta['node_count']} edges={meta['edge_count']} instances={meta['actual_spatial_instances']} time={duration(meta['seconds'])}", flush=True)

        del spatial, sv, dense, derived, streamed, statistics, rag, actual_partition, explicit
        if device.type == "cuda": torch.cuda.empty_cache()

    del model
    if device.type == "cuda": torch.cuda.empty_cache()


# =============================================================================
# Candidate pool
# =============================================================================


def sv_label_lookup(supervoxels: np.ndarray, labels: np.ndarray, *, name: str) -> np.ndarray:
    sv = np.asarray(supervoxels, np.int64).reshape(-1)
    lab = np.asarray(labels, np.int64).reshape(-1)
    max_sv = int(sv.max(initial=0))
    lo = np.full(max_sv + 1, np.iinfo(np.int64).max, np.int64)
    hi = np.full(max_sv + 1, -1, np.int64)
    positive = sv > 0
    np.minimum.at(lo, sv[positive], lab[positive]); np.maximum.at(hi, sv[positive], lab[positive])
    present = hi >= 0
    bad = present & (lo != hi)
    if bad.any():
        raise RuntimeError(f"{name}: manual labels split atomic SVs; examples={np.flatnonzero(bad)[:20].tolist()}")
    result = np.zeros(max_sv + 1, np.int64); result[present] = hi[present]
    return result


@dataclass(frozen=True)
class PairCandidate:
    frame: int
    a: int
    b: int
    node_a: int
    node_b: int
    interface_edges: int
    voxels_a: int
    voxels_b: int
    volume_ratio: float
    distance_dref: float
    split: str


@dataclass
class FrameCandidates:
    frame: int
    node_manual: Tensor
    manual_ids: tuple[int, ...]
    pairs: list[PairCandidate]
    triples: list[tuple[int, int, int]]


@dataclass
class CandidateManifest:
    frames: dict[int, FrameCandidates]
    train_pairs: list[PairCandidate]
    val_pairs: list[PairCandidate]


def manual_stats(labels: np.ndarray, spacing) -> tuple[dict[int, int], dict[int, np.ndarray]]:
    spacing = np.asarray(spacing, np.float32)
    center = 0.5 * (np.asarray(labels.shape, np.float32) - 1) * spacing
    counts, centroids = {}, {}
    for label in np.unique(labels[labels > 0]).tolist():
        coords = np.argwhere(labels == label)
        counts[int(label)] = int(len(coords))
        centroids[int(label)] = coords.astype(np.float32).mean(0) * spacing - center
    return counts, centroids


def node_has_context(track_graph, node_id: int, target_t: int, radius: int) -> bool:
    for other in list(track_graph.predecessors(node_id)) + list(track_graph.successors(node_id)):
        dt = int(track_graph.nodes[int(other)]["time"]) - target_t
        if 0 < abs(dt) <= radius: return True
    return False


def reconstruct_manifest(paths: Paths, payload: dict) -> CandidateManifest:
    frames, train, val = {}, [], []
    for row in payload["frames"]:
        t = int(row["frame"])
        cache = torch_load(paths.graph_cache(t)); rag: RAGState = cache["rag"]
        manual = np.asarray(np.load(paths.manual(t), mmap_mode="r", allow_pickle=False))
        sv = rag.supervoxel_labels[0].detach().cpu().numpy()
        lookup = sv_label_lookup(sv, manual, name=f"t={t} manual")
        node_manual = torch.as_tensor(lookup[rag.node_supervoxel_id.detach().cpu().numpy()], dtype=torch.long)
        raw_pairs = [PairCandidate(**item) for item in row["pairs"]]
        val_keys = {tuple(map(int, key)) for key in payload.get("val_pair_keys", [])}
        pairs = []
        for pair in raw_pairs:
            key = (int(pair.frame), int(pair.a), int(pair.b))
            effective = dataclasses.replace(pair, split=("val" if key in val_keys else pair.split))
            pairs.append(effective)
            (val if effective.split == "val" else train).append(effective)
        frames[t] = FrameCandidates(
            t,
            node_manual,
            tuple(map(int, row["manual_ids"])),
            pairs,
            [tuple(map(int, triple)) for triple in row["triples"]],
        )
    return CandidateManifest(frames, train, val)


def build_candidates(paths: Paths, *, track_graph, temporal_static: TemporalStatic, frame_count: int, spacing, dref_um: float, temporal_radius: int, min_voxels: int, max_volume_ratio: float, max_distance_dref: float, val_fraction: float, rebuild: bool) -> CandidateManifest:
    expected_manual_signatures = [file_signature(paths.manual(t)) for t in range(frame_count)]
    if paths.candidate_manifest.is_file() and not rebuild:
        try:
            cached_payload = json.loads(paths.candidate_manifest.read_text(encoding="utf-8"))
            if cached_payload.get("manual_signatures") == expected_manual_signatures:
                return reconstruct_manifest(paths, cached_payload)
            print("[candidates] manual annotations changed; rebuilding candidate manifest")
        except Exception:
            print("[candidates] cached manifest is invalid; rebuilding")

    frames, train_pairs, val_pairs, json_frames = {}, [], [], []
    for t in range(frame_count):
        cache = torch_load(paths.graph_cache(t)); rag: RAGState = cache["rag"]
        manual = np.asarray(np.load(paths.manual(t), mmap_mode="r", allow_pickle=False))
        sv = rag.supervoxel_labels[0].detach().cpu().numpy()
        lookup = sv_label_lookup(sv, manual, name=f"t={t} manual")
        node_sv = rag.node_supervoxel_id.detach().cpu().numpy()
        node_manual_np = lookup[node_sv]
        node_manual = torch.as_tensor(node_manual_np, dtype=torch.long)
        counts, centroids = manual_stats(manual, spacing)

        pair_edges = defaultdict(int)
        src = rag.edge_index[0].detach().cpu().numpy(); dst = rag.edge_index[1].detach().cpu().numpy()
        for u, v in zip(src.tolist(), dst.tolist()):
            a, b = int(node_manual_np[u]), int(node_manual_np[v])
            if a <= 0 or b <= 0 or a == b: continue
            pair_edges[tuple(sorted((a, b)))] += 1

        adjacency = defaultdict(set); pairs = []
        for (a, b), edge_count in sorted(pair_edges.items()):
            if counts.get(a, 0) < min_voxels or counts.get(b, 0) < min_voxels: continue
            node_a = temporal_static.manual_to_node.get(t, {}).get(a)
            node_b = temporal_static.manual_to_node.get(t, {}).get(b)
            if node_a is None or node_b is None: continue
            # Keep this first production training stage focused on ordinary
            # continuation/merge phenotypes. Division topology is valuable,
            # but mixing it into controlled merge synthesis would make the
            # target semantics ambiguous.
            if (
                int(track_graph.in_degree(node_a)) > 1
                or int(track_graph.out_degree(node_a)) > 1
                or int(track_graph.in_degree(node_b)) > 1
                or int(track_graph.out_degree(node_b)) > 1
            ):
                continue
            if not node_has_context(track_graph, node_a, t, temporal_radius): continue
            if not node_has_context(track_graph, node_b, t, temporal_radius): continue
            ratio = max(counts[a] / max(counts[b], 1), counts[b] / max(counts[a], 1))
            if ratio > max_volume_ratio: continue
            distance = float(np.linalg.norm(centroids[a] - centroids[b]) / max(dref_um, 1e-6))
            if distance > max_distance_dref: continue
            split = "val" if stable_fraction("inv35", t, a, b) < val_fraction else "train"
            pair = PairCandidate(t, a, b, int(node_a), int(node_b), int(edge_count), counts[a], counts[b], float(ratio), distance, split)
            pairs.append(pair); (val_pairs if split == "val" else train_pairs).append(pair)
            adjacency[a].add(b); adjacency[b].add(a)

        triples = set()
        for middle, neighbours in adjacency.items():
            neighbours = sorted(neighbours)
            for i in range(len(neighbours)):
                for j in range(i + 1, len(neighbours)):
                    triple = tuple(sorted((neighbours[i], middle, neighbours[j])))
                    if len(set(triple)) == 3 and all(mid in temporal_static.manual_to_node.get(t, {}) for mid in triple):
                        triples.add(triple)
        triples = sorted(triples)
        manual_ids = tuple(map(int, np.unique(node_manual_np[node_manual_np > 0]).tolist()))
        frames[t] = FrameCandidates(t, node_manual, manual_ids, pairs, triples)
        json_frames.append({
            "frame": t,
            "manual_ids": list(manual_ids),
            "pairs": [dataclasses.asdict(pair) for pair in pairs],
            "triples": [list(x) for x in triples],
        })
        print(f"[candidates t={t:03d}] pairs={len(pairs)} triples={len(triples)}", flush=True)

    if not train_pairs: raise RuntimeError("No training merge candidates survived filtering")
    if not val_pairs:
        ordered = sorted(train_pairs, key=lambda p: stable_fraction("fallback-val", p.frame, p.a, p.b))
        selected = ordered[:min(16, len(ordered))]
        selected_keys = {(p.frame, p.a, p.b) for p in selected}
        val_pairs = [dataclasses.replace(p, split="val") for p in selected]
        train_pairs = [p for p in train_pairs if (p.frame, p.a, p.b) not in selected_keys]
        # Keep per-frame rows consistent with the top-level split.
        for frame_data in frames.values():
            frame_data.pairs = [
                dataclasses.replace(p, split="val")
                if (p.frame, p.a, p.b) in selected_keys
                else p
                for p in frame_data.pairs
            ]
        for frame_row in json_frames:
            frame_row["pairs"] = [
                {**item, "split": ("val" if (int(item["frame"]), int(item["a"]), int(item["b"])) in selected_keys else item["split"])}
                for item in frame_row["pairs"]
            ]
        print("[candidates] WARNING: no hash-held-out pairs; moved a deterministic subset to validation")

    atomic_json(paths.candidate_manifest, {
        "format_version": 1,
        "sample_id": paths.sample,
        "dref_um": dref_um,
        "filters": {
            "min_voxels": min_voxels,
            "max_volume_ratio": max_volume_ratio,
            "max_distance_dref": max_distance_dref,
            "val_fraction": val_fraction,
        },
        "train_pair_count": len(train_pairs),
        "val_pair_count": len(val_pairs),
        "val_pair_keys": [[int(p.frame), int(p.a), int(p.b)] for p in val_pairs],
        "manual_signatures": expected_manual_signatures,
        "frames": json_frames,
    })
    return CandidateManifest(frames, train_pairs, val_pairs)


# =============================================================================
# Controlled local temporal-graph synthesis
# =============================================================================


def reference_key(ref_um: Sequence[float], scale: float = 10_000.0) -> tuple[int, int, int]:
    values = np.asarray(ref_um, np.float64)
    return tuple(int(v) for v in np.rint(values * float(scale)).astype(np.int64))


def context_variants(target_t: int, frame_count: int, temporal_radius: int) -> list[tuple[str, tuple[int, ...]]]:
    full = tuple(sequence_available_time_offsets(target_t, frame_count, temporal_radius))
    past = tuple(v for v in full if v <= 0)
    future = tuple(v for v in full if v >= 0)
    rows: list[tuple[str, tuple[int, ...]]] = [("full", full)]
    if len(past) > 1 and past != full:
        rows.append(("past_only", past))
    if len(future) > 1 and future != full:
        rows.append(("future_only", future))

    negative = [v for v in full if v < 0]
    positive = [v for v in full if v > 0]
    if negative:
        drop = max(negative)  # nearest past
        variant = tuple(v for v in full if v != drop)
        if len(variant) > 1:
            rows.append(("drop_nearest_past", variant))
    if positive:
        drop = min(positive)  # nearest future
        variant = tuple(v for v in full if v != drop)
        if len(variant) > 1:
            rows.append(("drop_nearest_future", variant))

    dedup: list[tuple[str, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    for name, offsets in rows:
        if offsets not in seen:
            seen.add(offsets)
            dedup.append((name, offsets))
    return dedup


def choose_context_variant(target_t: int, frame_count: int, temporal_radius: int, rng: random.Random) -> tuple[str, tuple[int, ...]]:
    rows = context_variants(target_t, frame_count, temporal_radius)
    by_name = {name: offsets for name, offsets in rows}
    draw = rng.random()
    if draw < 0.50 or len(rows) == 1:
        return "full", by_name["full"]
    if draw < 0.67 and "past_only" in by_name:
        return "past_only", by_name["past_only"]
    if draw < 0.84 and "future_only" in by_name:
        return "future_only", by_name["future_only"]
    dropout = [(name, offsets) for name, offsets in rows if name.startswith("drop_")]
    if dropout:
        return rng.choice(dropout)
    non_full = [(name, offsets) for name, offsets in rows if name != "full"]
    return rng.choice(non_full) if non_full else ("full", by_name["full"])


def anchor_for_manual_ids(temporal_static: TemporalStatic, frame: int, manual_ids: Sequence[int]) -> np.ndarray:
    rows = []
    weights = []
    for manual_id in manual_ids:
        node_id = temporal_static.manual_to_node.get(int(frame), {}).get(int(manual_id))
        if node_id is None:
            continue
        record = temporal_static.records[int(node_id)]
        rows.append(np.asarray(record.position_um, np.float32))
        weights.append(max(float(record.physical_volume_um3), 1e-6))
    if not rows:
        raise RuntimeError(f"No temporal target nodes for frame={frame}, manual_ids={tuple(manual_ids)}")
    weight = np.asarray(weights, np.float64)
    weight /= weight.sum()
    return np.sum(np.stack(rows).astype(np.float64) * weight[:, None], axis=0).astype(np.float32)


def member_target_nodes(temporal_static: TemporalStatic, frame: int, manual_ids: Sequence[int]) -> tuple[int, ...]:
    nodes = []
    for manual_id in manual_ids:
        node_id = temporal_static.manual_to_node.get(int(frame), {}).get(int(manual_id))
        if node_id is None:
            raise RuntimeError(f"Manual instance {manual_id} at t={frame} has no clean Trackastra node")
        nodes.append(int(node_id))
    return tuple(nodes)


def forced_lineage_nodes(track_graph, target_nodes: Sequence[int], absolute_times: set[int]) -> set[int]:
    forced = {int(v) for v in target_nodes}
    queue = list(forced)
    while queue:
        node = queue.pop()
        neighbours = list(track_graph.predecessors(node)) + list(track_graph.successors(node))
        for other in neighbours:
            other = int(other)
            if other in forced:
                continue
            if int(track_graph.nodes[other]["time"]) not in absolute_times:
                continue
            forced.add(other)
            queue.append(other)
    return forced


def merged_detection_record(
    temporal_static: TemporalStatic,
    *,
    frame: int,
    member_nodes: Sequence[int],
    node_id: int,
) -> DetectionRecord:
    members = [temporal_static.records[int(v)] for v in member_nodes]
    volumes = np.asarray([max(float(r.physical_volume_um3), 1e-6) for r in members], np.float64)
    weights = volumes / volumes.sum()
    positions = np.stack([np.asarray(r.position_um, np.float32) for r in members])
    position = np.sum(positions * weights[:, None], axis=0)

    # Approximate the union bounding box in the same centred physical frame.
    bbox = np.stack([np.asarray(r.bbox_um, np.float32) for r in members])
    lower = positions - 0.5 * bbox
    upper = positions + 0.5 * bbox
    union_bbox = upper.max(axis=0) - lower.min(axis=0)

    pca = np.stack([np.asarray(r.pca_axes_um, np.float32) for r in members])
    pca_union = np.maximum(pca.max(axis=0), union_bbox * 0.50)

    means = np.asarray([float(r.intensity_mean) for r in members], np.float64)
    stds = np.asarray([float(r.intensity_std) for r in members], np.float64)
    mean_i = float(np.sum(weights * means))
    second = float(np.sum(weights * (stds**2 + means**2)))
    std_i = float(math.sqrt(max(second - mean_i**2, 0.0)))

    def weighted_vector(name: str) -> tuple[float, float, float]:
        values = np.stack([np.asarray(getattr(r, name), np.float32) for r in members])
        result = np.sum(values * weights[:, None], axis=0)
        return tuple(float(v) for v in result)

    def weighted_scalar(name: str) -> float:
        values = np.asarray([float(getattr(r, name)) for r in members], np.float64)
        return float(np.sum(weights * values))

    first_grid = next((r.instance_grid for r in members if r.instance_grid is not None), None)
    grid_size = int(torch.as_tensor(first_grid).shape[-1]) if first_grid is not None else 12

    # Deliberately remove current-frame compact-history content. The target
    # merged detection is needed for correct graph topology, but the temporal
    # branch should learn from surrounding observations, not a hidden current
    # morphology descriptor that disappears under a different tracker.
    zero_grid = torch.zeros((4, grid_size, grid_size, grid_size), dtype=torch.float32)

    return DetectionRecord(
        node_id=int(node_id),
        time_offset=0,
        position_um=tuple(float(v) for v in position),
        physical_volume_um3=float(volumes.sum()),
        bbox_um=tuple(float(v) for v in union_bbox),
        pca_axes_um=tuple(float(v) for v in pca_union),
        elongation=weighted_scalar("elongation"),
        flatness=weighted_scalar("flatness"),
        solidity=weighted_scalar("solidity"),
        compactness=weighted_scalar("compactness"),
        intensity_mean=mean_i,
        intensity_std=std_i,
        backward_velocity_um=weighted_vector("backward_velocity_um"),
        forward_velocity_um=weighted_vector("forward_velocity_um"),
        distance_to_volume_boundary_um=min(float(r.distance_to_volume_boundary_um) for r in members),
        distance_to_patch_boundary_um=min(float(r.distance_to_patch_boundary_um) for r in members),
        boundary_related=any(bool(r.boundary_related) for r in members),
        instance_grid=zero_grid,
        history_valid=False,
    )


def _patch_synthetic_same_component_features(graph: dict[str, Any], merge_ids: Sequence[int]) -> dict[str, Any]:
    if not merge_ids:
        return graph
    merge_set = {int(v) for v in merge_ids}
    merged_id = min(merge_set)
    original = torch.as_tensor(graph["best_current_component_id"], dtype=torch.long).clone()
    remapped = original.clone()
    mask = torch.zeros_like(remapped, dtype=torch.bool)
    for value in merge_set:
        mask |= remapped == int(value)
    remapped[mask] = int(merged_id)
    graph["best_current_component_id"] = remapped

    hidx = torch.as_tensor(graph["hypothesis_edge_index"], dtype=torch.long)
    hattr = torch.as_tensor(graph["hypothesis_edge_attr"]).clone()
    if hidx.numel() and hattr.numel():
        src, dst = hidx
        synthetic_same = (remapped[src] > 0) & (remapped[src] == remapped[dst])
        newly_same = synthetic_same & (original[src] != original[dst])
        if bool(newly_same.any()):
            best = torch.as_tensor(graph["best_component_overlap"], dtype=hattr.dtype)
            confidence = torch.minimum(best[src], best[dst]).clamp(0.0, 1.0)
            confidence = torch.where(confidence > 0, confidence, torch.ones_like(confidence))
            hattr[newly_same, 4] = confidence[newly_same]
            # Representative past/future volumes sum to the synthetic current
            # merged component by construction; ratio ~1 is the intended cue.
            hattr[newly_same, 18] = 1.0
            hattr[newly_same, 21] = 1.0
        graph["hypothesis_edge_attr"] = hattr
    return graph


def build_local_temporal_graph(
    *,
    temporal_static: TemporalStatic,
    track_graph,
    manual_labels: np.ndarray,
    target_t: int,
    frame_count: int,
    temporal_radius: int,
    available_offsets: Sequence[int],
    anchor_ids: Sequence[int],
    merge_ids: Sequence[int],
    spacing: Sequence[float],
    dref_um: float,
    neighbourhood_dref: float,
    complete_candidate_graph: bool = False,
) -> dict[str, Any]:
    available_offsets = tuple(sorted({int(v) for v in available_offsets}))
    if 0 not in available_offsets:
        raise ValueError("available_offsets must contain 0")
    absolute_times = {int(target_t) + int(v) for v in available_offsets}
    anchor = anchor_for_manual_ids(temporal_static, target_t, anchor_ids)
    merge_nodes = member_target_nodes(temporal_static, target_t, merge_ids) if merge_ids else ()
    anchor_nodes = member_target_nodes(temporal_static, target_t, anchor_ids)
    forced = forced_lineage_nodes(track_graph, anchor_nodes, absolute_times)

    selected: set[int] = set()
    radius_um = float(neighbourhood_dref) * float(dref_um)
    for absolute_t in sorted(absolute_times):
        for node_id in temporal_static.nodes_by_time.get(absolute_t, []):
            record = temporal_static.records.get(int(node_id))
            if record is None:
                continue
            distance = float(np.linalg.norm(np.asarray(record.position_um, np.float32) - anchor))
            if distance <= radius_um or int(node_id) in forced:
                selected.add(int(node_id))
    selected.update(int(v) for v in anchor_nodes)

    replacement: dict[int, int] = {}
    merged_node_id: int | None = None
    records: list[DetectionRecord] = []

    if merge_nodes:
        digest = int.from_bytes(
            hashlib.sha256(
                (f"{target_t}:" + ",".join(str(v) for v in sorted(merge_nodes))).encode("utf-8")
            ).digest()[:4],
            "big",
        )
        merged_node_id = int(temporal_static.maximum_node_id + 1 + target_t * 1_000_000 + digest % 900_000)
        for node in merge_nodes:
            replacement[int(node)] = merged_node_id

    for node_id in sorted(selected):
        data = track_graph.nodes[int(node_id)]
        absolute_t = int(data["time"])
        if absolute_t not in absolute_times:
            continue
        if node_id in replacement:
            continue
        base = temporal_static.records.get(int(node_id))
        if base is None:
            continue
        records.append(replace(base, time_offset=absolute_t - int(target_t)))

    if merge_nodes:
        assert merged_node_id is not None
        records.append(
            merged_detection_record(
                temporal_static,
                frame=target_t,
                member_nodes=merge_nodes,
                node_id=merged_node_id,
            )
        )

    final_ids = {int(r.node_id) for r in records}
    associations: dict[tuple[int, int, str], AssociationRecord] = {}
    for row in temporal_static.associations:
        src0, dst0 = int(row.src_node_id), int(row.dst_node_id)
        if src0 not in selected or dst0 not in selected:
            continue
        src = replacement.get(src0, src0)
        dst = replacement.get(dst0, dst0)
        if src == dst or src not in final_ids or dst not in final_ids:
            continue
        rewired = src != src0 or dst != dst0
        relation = "temporal" if rewired else row.relation
        key = (int(src), int(dst), str(relation))
        existing = associations.get(key)
        if existing is None:
            associations[key] = AssociationRecord(src, dst, row.score, relation)
        else:
            old_score = -float("inf") if existing.score is None else float(existing.score)
            new_score = -float("inf") if row.score is None else float(row.score)
            if new_score > old_score:
                associations[key] = AssociationRecord(src, dst, row.score, relation)

    graph = build_temporal_graph(
        records,
        list(associations.values()),
        dref_um=float(dref_um),
        temporal_radius=int(temporal_radius),
        available_time_offsets=available_offsets,
        k_spatial_neighbors=6,
        spatial_radius_dref=2.5,
        current_labels=np.asarray(manual_labels),
        spacing_um=tuple(float(v) for v in spacing),
        candidate_graph_enabled=bool(complete_candidate_graph),
    )
    graph["temporal_batch"] = torch.zeros(len(graph["temporal_ref_um"]), dtype=torch.long)
    graph["target_time_index"] = int(target_t)
    graph["available_time_offsets"] = torch.tensor(available_offsets, dtype=torch.long)
    graph["anchor_um"] = torch.as_tensor(anchor, dtype=torch.float32)
    graph["merge_manual_ids"] = torch.tensor(tuple(int(v) for v in merge_ids), dtype=torch.long)
    return _patch_synthetic_same_component_features(graph, merge_ids)


# =============================================================================
# Precompute raw spatial evidence at every temporal reference used by training
# =============================================================================


def observer_cache_matches(paths: Paths, t: int, *, dref_um: float, neighbourhood_dref: float) -> bool:
    required = (
        paths.observer_refs(t),
        paths.observer_d1(t),
        paths.observer_d2(t),
        paths.observer_hidden(t),
        paths.observer_explicit(t),
        paths.observer_meta(t),
    )
    if any(not p.is_file() for p in required):
        return False
    try:
        meta = json.loads(paths.observer_meta(t).read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        int(meta.get("observer_cache_version", -1)) == OBSERVER_CACHE_VERSION
        and abs(float(meta.get("dref_um", -1)) - float(dref_um)) < 1e-6
        and abs(float(meta.get("neighbourhood_dref", -1)) - float(neighbourhood_dref)) < 1e-6
        and meta.get("candidate_manifest_signature") == file_signature(paths.candidate_manifest)
        and meta.get("checkpoint_signature") == file_signature(paths.checkpoint)
    )


def selected_triples(frame_data: FrameCandidates, maximum: int) -> list[tuple[int, int, int]]:
    ordered = sorted(
        frame_data.triples,
        key=lambda triple: stable_fraction("inv35-triple", frame_data.frame, *triple),
    )
    return ordered[: max(int(maximum), 0)]


def collect_frame_reference_superset(
    *,
    paths: Paths,
    frame_data: FrameCandidates,
    temporal_static: TemporalStatic,
    track_graph,
    frame_count: int,
    temporal_radius: int,
    spacing: Sequence[float],
    dref_um: float,
    neighbourhood_dref: float,
    max_triples: int,
    complete_candidate_graph: bool,
) -> np.ndarray:
    t = int(frame_data.frame)
    manual = np.load(paths.manual(t), mmap_mode="r", allow_pickle=False)
    keys: dict[tuple[int, int, int], np.ndarray] = {}
    variants = context_variants(t, frame_count, temporal_radius)

    def add_graph(anchor_ids: Sequence[int], merge_ids: Sequence[int], offsets: Sequence[int]) -> None:
        graph = build_local_temporal_graph(
            temporal_static=temporal_static,
            track_graph=track_graph,
            manual_labels=manual,
            target_t=t,
            frame_count=frame_count,
            temporal_radius=temporal_radius,
            available_offsets=offsets,
            anchor_ids=anchor_ids,
            merge_ids=merge_ids,
            spacing=spacing,
            dref_um=dref_um,
            neighbourhood_dref=neighbourhood_dref,
            complete_candidate_graph=complete_candidate_graph,
        )
        refs = torch.as_tensor(graph["temporal_ref_um"]).detach().cpu().numpy()
        for ref in refs:
            keys.setdefault(reference_key(ref), np.asarray(ref, np.float32))

    # Both the positive synthetic merge and the corresponding clean close-pair
    # negative are included in the reference superset.
    for pair in frame_data.pairs:
        anchor = (int(pair.a), int(pair.b))
        for _, offsets in variants:
            add_graph(anchor, anchor, offsets)
            add_graph(anchor, (), offsets)

    for triple in selected_triples(frame_data, max_triples):
        for _, offsets in variants:
            add_graph(triple, triple, offsets)

    if not keys:
        return np.zeros((0, 3), dtype=np.float32)
    ordered = [keys[key] for key in sorted(keys)]
    return np.stack(ordered).astype(np.float32, copy=False)


@torch.inference_mode()
def sample_raw_observer_features(
    model,
    spatial_inputs: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    refs_um: np.ndarray,
    *,
    config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirror production stream_tiled_observation_cache before trainable projections."""
    count = int(len(refs_um))
    c1 = int(model.cfg.spatial.channels[1])
    c2 = int(model.cfg.spatial.channels[2])
    ch = int(model.cfg.geometry.hidden_channels)
    if count == 0:
        return (
            np.zeros((0, c1), np.float16),
            np.zeros((0, c2), np.float16),
            np.zeros((0, ch), np.float16),
            np.zeros((0, 11), np.float16),
        )

    refs = torch.from_numpy(np.asarray(refs_um, np.float32)).to(spatial_inputs.device)
    shape = tuple(int(v) for v in spatial_inputs.shape[-3:])
    specs = generate_dense_tiles(1, shape, config)
    assignments: dict[int, list[int]] = defaultdict(list)
    extent = (torch.as_tensor(shape, device=refs.device).float() - 1) * spacing_um[0].float()
    maximum_voxel = (
        torch.as_tensor(
            shape,
            device=refs.device,
            dtype=torch.long,
        )
        - 1
    )

    # STIRNET_TILED_OBSERVER_OUTSIDE_REFERENCE_V1
    # Match production tiled observer routing. Tracklet target references may
    # be interpolated/extrapolated outside the FOV. Clamp ONLY the routing
    # voxel so a boundary tile is selected; keep refs[row] unchanged for the
    # actual physical observer sample.
    for row in range(count):
        voxel_unclamped = torch.round(
            (refs[row] + 0.5 * extent)
            / spacing_um[0].float().clamp_min(1e-6)
        ).long()
        voxel = torch.minimum(
            voxel_unclamped.clamp_min(0),
            maximum_voxel,
        )
        best_index = None
        best_weight = -1.0
        for spec_index, spec in enumerate(specs):
            if not all(
                int(spec.slices_zyx[axis].start) <= int(voxel[axis]) < int(spec.slices_zyx[axis].stop)
                for axis in range(3)
            ):
                continue
            local = tuple(int(voxel[axis]) - int(spec.slices_zyx[axis].start) for axis in range(3))
            weight = float(tile_blend_weight(spec, shape, config.tile_halo_zyx, device=refs.device)[local].item())
            if weight > best_weight:
                best_weight = weight
                best_index = spec_index
        if best_index is None:
            raise RuntimeError(f"Observer reference is outside all tiles: {refs[row].tolist()}")
        assignments[int(best_index)].append(row)

    d1_out = np.zeros((count, c1), np.float16)
    d2_out = np.zeros((count, c2), np.float16)
    hidden_out = np.zeros((count, ch), np.float16)
    explicit_out = np.zeros((count, 11), np.float16)
    global_center = 0.5 * (torch.as_tensor(shape, device=refs.device).float() - 1)

    for spec_index, rows in assignments.items():
        spec = specs[spec_index]
        tile = spatial_inputs[
            0:1,
            :,
            spec.slices_zyx[0],
            spec.slices_zyx[1],
            spec.slices_zyx[2],
        ]
        output = model(tile, spacing_um, dref_um, execution_stage="geometry")
        if output.geometry.features is None:
            raise RuntimeError("Geometry hidden features are required for TemporalSpatialObserver")
        row_index = torch.tensor(rows, device=refs.device, dtype=torch.long)
        tile_center = torch.tensor(
            [0.5 * (int(axis.start) + int(axis.stop) - 1) for axis in spec.slices_zyx],
            device=refs.device,
            dtype=torch.float32,
        )
        shift_um = (tile_center - global_center) * spacing_um[0].float()
        local_refs = refs[row_index] - shift_um
        radius = dref_um[0].float() * float(model.cfg.temporal.observation_radius_dref)
        radius_vec = radius.expand(len(rows))

        d1 = _sample_local_grid(
            output.decoded_spatial.d1[0],
            local_refs,
            output.spatial_pyramid.spacings_um[1][0],
            radius_vec,
        )
        d2 = _sample_local_grid(
            output.decoded_spatial.d2[0],
            local_refs,
            output.spatial_pyramid.spacings_um[2][0],
            radius_vec,
        )
        hidden_spacing = (
            spacing_um[0]
            if output.geometry.feature_spacing_um is None
            else output.geometry.feature_spacing_um[0]
        )
        hidden = _sample_local_grid(
            output.geometry.features[0],
            local_refs,
            hidden_spacing,
            radius_vec,
        )
        explicit = _sample_explicit_geometry(
            output.geometry,
            0,
            local_refs,
            spacing_um[0],
            radius_vec,
        )

        rows_np = np.asarray(rows, np.int64)
        d1_out[rows_np] = d1.detach().float().cpu().numpy().astype(np.float16)
        d2_out[rows_np] = d2.detach().float().cpu().numpy().astype(np.float16)
        hidden_out[rows_np] = hidden.detach().float().cpu().numpy().astype(np.float16)
        explicit_out[rows_np] = explicit.detach().float().cpu().numpy().astype(np.float16)

    return d1_out, d2_out, hidden_out, explicit_out


def prepare_observer_raw_cache(
    paths: Paths,
    *,
    manifest: CandidateManifest,
    temporal_static: TemporalStatic,
    track_graph,
    frame_count: int,
    temporal_radius: int,
    spacing: Sequence[float],
    dref_um: float,
    neighbourhood_dref: float,
    max_triples: int,
    device: torch.device,
    amp_dtype: str,
    complete_candidate_graph: bool,
    rebuild: bool,
) -> None:
    todo = [
        t
        for t in range(frame_count)
        if rebuild or not observer_cache_matches(paths, t, dref_um=dref_um, neighbourhood_dref=neighbourhood_dref)
    ]
    if not todo:
        print("[observer raw cache] all frames ready")
        return

    _, model = load_spatial_model(paths.checkpoint, device)
    inference_cfg = dataclasses.replace(model.cfg.inference, mode="tiled", tiled_dense_enabled=True)
    spacing_t = torch.tensor([spacing], device=device, dtype=torch.float32)
    dref_t = torch.tensor([float(dref_um)], device=device, dtype=torch.float32)

    print("=" * 118)
    print("INVESTIGATION 35 — PRECOMPUTE FROZEN OBSERVER RAW SAMPLES")
    print("These samples are BEFORE the trainable observer projections/message/gate.")
    print(f"frames : {todo}")
    print("=" * 118)

    for t in todo:
        started = time.perf_counter()
        refs = collect_frame_reference_superset(
            paths=paths,
            frame_data=manifest.frames[t],
            temporal_static=temporal_static,
            track_graph=track_graph,
            frame_count=frame_count,
            temporal_radius=temporal_radius,
            spacing=spacing,
            dref_um=dref_um,
            neighbourhood_dref=neighbourhood_dref,
            max_triples=max_triples,
            complete_candidate_graph=complete_candidate_graph,
        )
        spatial_np = np.asarray(np.load(paths.spatial_inputs(t), mmap_mode="r", allow_pickle=False), np.float32)
        spatial = torch.from_numpy(np.ascontiguousarray(spatial_np))[None].to(device)
        with torch.inference_mode(), autocast_for(device, amp_dtype):
            d1, d2, hidden, explicit = sample_raw_observer_features(
                model,
                spatial,
                spacing_t,
                dref_t,
                refs,
                config=inference_cfg,
            )
        atomic_npy(paths.observer_refs(t), refs.astype(np.float32, copy=False))
        atomic_npy(paths.observer_d1(t), d1)
        atomic_npy(paths.observer_d2(t), d2)
        atomic_npy(paths.observer_hidden(t), hidden)
        atomic_npy(paths.observer_explicit(t), explicit)
        meta = {
            "observer_cache_version": OBSERVER_CACHE_VERSION,
            "timepoint": int(t),
            "dref_um": float(dref_um),
            "neighbourhood_dref": float(neighbourhood_dref),
            "reference_key_scale": 10_000.0,
            "reference_count": int(len(refs)),
            "candidate_manifest_signature": file_signature(paths.candidate_manifest),
            "checkpoint_signature": file_signature(paths.checkpoint),
            "seconds": time.perf_counter() - started,
        }
        atomic_json(paths.observer_meta(t), meta)
        print(f"[observer t={t:03d}] refs={len(refs)} time={duration(meta['seconds'])}", flush=True)
        del spatial
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# Runtime frozen frame + trainable observer lookup
# =============================================================================


@dataclass
class ObserverRawLookup:
    ref_um: Tensor
    d1: Tensor
    d2: Tensor
    hidden: Tensor
    explicit: Tensor
    key_to_row: dict[tuple[int, int, int], int]

    @classmethod
    def load(cls, paths: Paths, t: int, device: torch.device) -> "ObserverRawLookup":
        refs_np = np.array(
            np.load(paths.observer_refs(t), mmap_mode="r", allow_pickle=False),
            dtype=np.float32,
            order="C",
            copy=True,
        )
        d1_np = np.array(
            np.load(paths.observer_d1(t), mmap_mode="r", allow_pickle=False),
            dtype=np.float32,
            order="C",
            copy=True,
        )
        d2_np = np.array(
            np.load(paths.observer_d2(t), mmap_mode="r", allow_pickle=False),
            dtype=np.float32,
            order="C",
            copy=True,
        )
        hidden_np = np.array(
            np.load(paths.observer_hidden(t), mmap_mode="r", allow_pickle=False),
            dtype=np.float32,
            order="C",
            copy=True,
        )
        explicit_np = np.array(
            np.load(paths.observer_explicit(t), mmap_mode="r", allow_pickle=False),
            dtype=np.float32,
            order="C",
            copy=True,
        )
        key_to_row = {reference_key(ref): i for i, ref in enumerate(refs_np)}
        return cls(
            ref_um=torch.from_numpy(refs_np).to(device),
            d1=torch.from_numpy(d1_np).to(device),
            d2=torch.from_numpy(d2_np).to(device),
            hidden=torch.from_numpy(hidden_np).to(device),
            explicit=torch.from_numpy(explicit_np).to(device),
            key_to_row=key_to_row,
        )

    def rows_for(self, refs_um: Tensor) -> Tensor:
        refs_np = refs_um.detach().float().cpu().numpy()
        rows = []
        for ref in refs_np:
            key = reference_key(ref)
            row = self.key_to_row.get(key)
            if row is None:
                # Float32 graph reconstruction should normally be exact under
                # 1e-4 um quantization. The nearest fallback is diagnostic only.
                cached = self.ref_um.detach().float().cpu().numpy()
                if len(cached):
                    distance = np.linalg.norm(cached - ref[None], axis=1)
                    nearest = int(np.argmin(distance))
                    if float(distance[nearest]) <= 5e-4:
                        row = nearest
                if row is None:
                    raise KeyError(
                        "Temporal reference was not present in the prepared observer cache: "
                        f"ref={ref.tolist()}. Re-run with --rebuild-observer-cache."
                    )
            rows.append(int(row))
        return torch.tensor(rows, device=refs_um.device, dtype=torch.long)


@dataclass
class RuntimeFrame:
    t: int
    rag: RAGState
    actual_partition: PartitionState
    node_manual: Tensor
    manual: np.ndarray
    observer: ObserverRawLookup


class RuntimeFrameLoader:
    def __init__(self, paths: Paths, manifest: CandidateManifest, device: torch.device):
        self.paths = paths
        self.manifest = manifest
        self.device = device
        self.current: RuntimeFrame | None = None

    def load(self, t: int) -> RuntimeFrame:
        t = int(t)
        if self.current is not None and self.current.t == t:
            return self.current
        if self.current is not None:
            del self.current
            self.current = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        payload = torch_load(self.paths.graph_cache(t), map_location="cpu")

        # Spatial preparation may persist floating RAG statistics/embeddings in
        # FP16. Training feeds them into trainable FP32 modules, so restore the
        # complete floating cache tree to FP32 at this boundary.
        training_float_dtype = torch.float32
        rag: RAGState = tree_device_training_float(
            payload["rag"],
            self.device,
            training_float_dtype,
        )
        actual: PartitionState = tree_device_training_float(
            payload["actual_partition"],
            self.device,
            training_float_dtype,
        )
        node_manual = self.manifest.frames[t].node_manual.to(self.device)
        manual = np.load(self.paths.manual(t), mmap_mode="r", allow_pickle=False)
        observer = ObserverRawLookup.load(self.paths, t, self.device)
        self.current = RuntimeFrame(t, rag, actual, node_manual, manual, observer)
        return self.current


@dataclass
class SyntheticCaseState:
    frame: int
    merge_ids: tuple[int, ...]
    anchor_ids: tuple[int, ...]
    anchor_um: Tensor
    rag: RAGState
    partition: PartitionState
    target_keep: Tensor
    valid_edge_mask: Tensor
    correction_edge_mask: Tensor
    local_edge_mask: Tensor
    instance_split_target: Tensor
    component_manual_ids: tuple[tuple[int, ...], ...]


def build_synthetic_partition(
    runtime: RuntimeFrame,
    *,
    temporal_static: TemporalStatic,
    merge_ids: Sequence[int],
    anchor_ids: Sequence[int],
    dref_um: float,
    synthetic_spatial_logit: float,
    local_edge_radius_dref: float,
) -> SyntheticCaseState:
    rag = runtime.rag
    node_manual = runtime.node_manual
    merge_set = {int(v) for v in merge_ids}
    merge_tag = min(merge_set) if merge_set else None
    actual_global = runtime.actual_partition.node_component_global

    keys: list[tuple[str, int]] = []
    for node in range(int(node_manual.numel())):
        manual_id = int(node_manual[node].item())
        if manual_id > 0:
            key_id = merge_tag if merge_tag is not None and manual_id in merge_set else manual_id
            keys.append(("manual", int(key_id)))
        else:
            # Preserve unlabeled/spatial-only nodes according to the frozen
            # spatial partition rather than fabricating pseudo-GT identities.
            keys.append(("extra", int(actual_global[node].item())))

    key_to_component: dict[tuple[str, int], int] = {}
    component_manual: list[set[int]] = []
    node_component_values: list[int] = []
    for node, key in enumerate(keys):
        component = key_to_component.get(key)
        if component is None:
            component = len(key_to_component)
            key_to_component[key] = component
            component_manual.append(set())
        node_component_values.append(component)
        manual_id = int(node_manual[node].item())
        if manual_id > 0:
            component_manual[component].add(manual_id)

    node_component = torch.tensor(node_component_values, device=rag.node_features.device, dtype=torch.long)
    component_count = len(key_to_component)
    src, dst = rag.edge_index
    same_component = node_component[src] == node_component[dst]
    positive = rag.spatial_edge_logits.new_full(rag.spatial_edge_logits.shape, float(synthetic_spatial_logit))
    negative = rag.spatial_edge_logits.new_full(rag.spatial_edge_logits.shape, -float(synthetic_spatial_logit))
    spatial_logits = torch.where(same_component, positive, negative)
    synthetic_rag = replace(rag, spatial_edge_logits=spatial_logits)

    # Tokenizer uses only the max label/device when compact RAG statistics are
    # available. A tiny representative map avoids copying the native volume.
    if component_count:
        tiny_labels = torch.arange(1, component_count + 1, device=rag.node_features.device, dtype=torch.long).reshape(1, 1, -1)
    else:
        tiny_labels = torch.zeros((1, 1, 1), device=rag.node_features.device, dtype=torch.long)
    partition = PartitionState(
        labels=[tiny_labels],
        node_component=node_component,
        node_component_global=node_component.clone(),
        component_count_per_batch=torch.tensor([component_count], device=rag.node_features.device, dtype=torch.long),
        edge_logits=spatial_logits,
    )

    manual_src = node_manual[src]
    manual_dst = node_manual[dst]
    valid = (manual_src > 0) & (manual_dst > 0)
    target_keep = valid & (manual_src == manual_dst)
    correction = valid & (same_component != target_keep)

    anchor_np = anchor_for_manual_ids(temporal_static, runtime.t, anchor_ids)
    anchor = torch.as_tensor(anchor_np, device=rag.node_features.device, dtype=torch.float32)
    midpoint = 0.5 * (rag.node_centroid_um[src].float() + rag.node_centroid_um[dst].float())
    local_edge = torch.linalg.vector_norm(midpoint - anchor[None], dim=-1) <= float(local_edge_radius_dref) * float(dref_um)
    local_edge |= correction

    split_target = spatial_logits.new_zeros((component_count,))
    for component, manual_ids in enumerate(component_manual):
        split_target[component] = float(len(manual_ids) > 1)

    return SyntheticCaseState(
        frame=runtime.t,
        merge_ids=tuple(int(v) for v in merge_ids),
        anchor_ids=tuple(int(v) for v in anchor_ids),
        anchor_um=anchor,
        rag=synthetic_rag,
        partition=partition,
        target_keep=target_keep,
        valid_edge_mask=valid,
        correction_edge_mask=correction,
        local_edge_mask=local_edge,
        instance_split_target=split_target,
        component_manual_ids=tuple(tuple(sorted(v)) for v in component_manual),
    )


def dummy_geometry_and_decode(model, reference: Tensor) -> tuple[SpatialDecodeState, GeometryState]:
    channels = model.cfg.spatial.channels
    decoded = SpatialDecodeState(
        d0=reference.new_zeros((1, channels[0], 1, 1, 1)),
        d1=reference.new_zeros((1, channels[1], 1, 1, 1)),
        d2=reference.new_zeros((1, channels[2], 1, 1, 1)),
    )
    geometry = GeometryState(
        foreground_logits=reference.new_zeros((1, 1, 1, 1, 1)),
        surface_logits=reference.new_zeros((1, 1, 1, 1, 1)),
        separator_logits=reference.new_zeros((1, 1, 1, 1, 1)),
        sdf=reference.new_zeros((1, 1, 1, 1, 1)),
        flow=reference.new_zeros((1, 3, 1, 1, 1)),
        centroid_offset=reference.new_zeros((1, 3, 1, 1, 1)),
        seed_logits=reference.new_zeros((1, 1, 1, 1, 1)),
        features=None,
        feature_spacing_um=None,
    )
    return decoded, geometry


def temporal_input_from_graph(model, graph: dict[str, Any], device: torch.device) -> TemporalInput:
    grid = torch.as_tensor(graph["node_instance_grid"]).to(device=device, dtype=torch.float32)
    valid = torch.as_tensor(graph["node_history_valid"]).to(device=device, dtype=torch.bool)
    history = model.history_encoder(grid, valid)
    return TemporalInput(
        graph_x=torch.as_tensor(graph["graph_x"]).to(device=device, dtype=torch.float32),
        graph_edge_index=torch.as_tensor(graph["graph_edge_index"]).to(device=device, dtype=torch.long),
        graph_edge_attr=torch.as_tensor(graph["graph_edge_attr"]).to(device=device, dtype=torch.float32),
        hypothesis_edge_index=torch.as_tensor(graph["hypothesis_edge_index"]).to(device=device, dtype=torch.long),
        hypothesis_edge_attr=torch.as_tensor(graph["hypothesis_edge_attr"]).to(device=device, dtype=torch.float32),
        tracklet_id=torch.as_tensor(graph["tracklet_id"]).to(device=device, dtype=torch.long),
        temporal_ref_um=torch.as_tensor(graph["temporal_ref_um"]).to(device=device, dtype=torch.float32),
        temporal_status=torch.as_tensor(graph["temporal_status"]).to(device=device, dtype=torch.float32),
        temporal_batch=torch.as_tensor(graph["temporal_batch"]).to(device=device, dtype=torch.long),
        node_history_embedding=history,
    )


def observe_from_raw_lookup(model, temporal: TemporalState, lookup: ObserverRawLookup) -> TemporalState:
    if temporal.is_empty:
        return temporal
    rows = lookup.rows_for(temporal.ref_um)
    observer = model.temporal_observer
    p1 = observer.d1_proj(lookup.d1[rows])
    p2 = observer.d2_proj(lookup.d2[rows])
    pg = observer.geometry_proj(lookup.hidden[rows]) + observer.geometry_field_proj(lookup.explicit[rows])
    message = observer.message(torch.cat([p1, p2, pg], dim=-1))
    message = message.to(temporal.tokens.dtype)
    gate = observer.gate(torch.cat([temporal.tokens, message, temporal.reliability], dim=-1))
    return replace(temporal, tokens=observer.norm(temporal.tokens + gate * message))


@dataclass
class ProductionTemporalForward:
    instances: Any
    temporal_base: TemporalState
    full_temporal: TemporalState
    full_reasoning: Any
    corrupted_temporal: TemporalState
    corrupted_reasoning: Any


def production_temporal_forward(
    model,
    *,
    case: SyntheticCaseState,
    temporal_graph: dict[str, Any],
    observer_lookup: ObserverRawLookup,
    spacing: Sequence[float],
    dref_um: float,
    device: torch.device,
    corruption: str,
    corruption_seed: int,
) -> ProductionTemporalForward:
    reference = case.rag.node_features
    decoded, geometry = dummy_geometry_and_decode(model, reference)
    spacing_t = torch.tensor([spacing], device=device, dtype=torch.float32)
    dref_t = torch.tensor([float(dref_um)], device=device, dtype=torch.float32)

    instances = model.instance_tokenizer(
        case.partition,
        case.rag,
        decoded,
        geometry,
        spacing_t,
        dref_t,
        profile_prefix="inv35_tokenizer",
    )
    temporal_input = temporal_input_from_graph(model, temporal_graph, device)
    temporal_base = model.temporal_encoder(temporal_input)
    full_temporal = observe_from_raw_lookup(model, temporal_base, observer_lookup)
    full_reasoning = model.instance_temporal(instances, case.rag, full_temporal, dref_t)

    if corruption == "contentless":
        corrupted_base = contentless_temporal_state(temporal_base)
    elif corruption == "shuffled":
        corrupted_base = shuffled_temporal_state(temporal_base, seed=int(corruption_seed))
    else:
        raise ValueError(f"Unknown corruption: {corruption}")
    corrupted_temporal = observe_from_raw_lookup(model, corrupted_base, observer_lookup)
    corrupted_reasoning = model.instance_temporal(instances, case.rag, corrupted_temporal, dref_t)
    return ProductionTemporalForward(
        instances,
        temporal_base,
        full_temporal,
        full_reasoning,
        corrupted_temporal,
        corrupted_reasoning,
    )


# =============================================================================
# Causal synthetic-merge objective
# =============================================================================


@dataclass
class SyntheticLoss:
    total: Tensor
    edge: Tensor
    correction: Tensor
    preservation: Tensor
    split: Tensor
    corrupted_noop: Tensor
    corrupted_gate: Tensor
    causal_margin: Tensor
    correction_edges: int
    preservation_edges: int


def _sample_tensor_rows(index: Tensor, count: int, rng: random.Random) -> Tensor:
    if count <= 0 or index.numel() == 0:
        return index[:0]
    if int(index.numel()) <= int(count):
        return index
    rows = rng.sample(range(int(index.numel())), int(count))
    return index[torch.tensor(rows, device=index.device, dtype=torch.long)]


def preservation_indices(
    case: SyntheticCaseState,
    *,
    maximum: int,
    correction_ratio: int,
    rng: random.Random,
) -> Tensor:
    preserve = case.valid_edge_mask & case.local_edge_mask & ~case.correction_edge_mask
    keep = torch.nonzero(preserve & case.target_keep, as_tuple=False).flatten()
    cut = torch.nonzero(preserve & ~case.target_keep, as_tuple=False).flatten()
    correction_count = int(case.correction_edge_mask.sum().item())
    budget = int(maximum)
    if correction_count > 0:
        budget = min(budget, max(correction_count * int(correction_ratio), correction_count))
    if budget <= 0:
        return keep[:0]

    keep_budget = budget // 2
    cut_budget = budget - keep_budget
    selected_keep = _sample_tensor_rows(keep, keep_budget, rng)
    selected_cut = _sample_tensor_rows(cut, cut_budget, rng)
    selected = torch.cat([selected_keep, selected_cut])
    if int(selected.numel()) < budget:
        used = torch.zeros_like(preserve)
        if selected.numel():
            used[selected] = True
        remaining = torch.nonzero(preserve & ~used, as_tuple=False).flatten()
        selected = torch.cat([
            selected,
            _sample_tensor_rows(remaining, budget - int(selected.numel()), rng),
        ])
    return selected


def split_indices(target: Tensor, *, negative_ratio: int, rng: random.Random) -> Tensor:
    positive = torch.nonzero(target > 0.5, as_tuple=False).flatten()
    negative = torch.nonzero(target <= 0.5, as_tuple=False).flatten()
    if positive.numel():
        negative = _sample_tensor_rows(
            negative,
            max(int(positive.numel()) * int(negative_ratio), 8),
            rng,
        )
        return torch.cat([positive, negative])
    return _sample_tensor_rows(negative, min(int(negative.numel()), 64), rng)


def synthetic_causal_loss(
    case: SyntheticCaseState,
    forward: ProductionTemporalForward,
    *,
    preserve_edges: int,
    preserve_ratio: int,
    split_weight: float,
    noop_weight: float,
    corrupted_gate_weight: float,
    margin_weight: float,
    margin: float,
    rng: random.Random,
) -> SyntheticLoss:
    full = forward.full_reasoning
    corrupted = forward.corrupted_reasoning
    device = full.final_edge_logits.device
    zero = full.final_edge_logits.sum() * 0.0

    correction_index = torch.nonzero(
        case.correction_edge_mask & case.local_edge_mask,
        as_tuple=False,
    ).flatten()
    preserve_index = preservation_indices(
        case,
        maximum=preserve_edges,
        correction_ratio=preserve_ratio,
        rng=rng,
    )

    def bce(index: Tensor) -> Tensor:
        if index.numel() == 0:
            return zero
        target = case.target_keep[index].to(full.final_edge_logits.dtype)
        return F.binary_cross_entropy_with_logits(full.final_edge_logits[index], target)

    correction_loss = bce(correction_index)
    preservation_loss = bce(preserve_index)
    if correction_index.numel() and preserve_index.numel():
        edge_loss = correction_loss + preservation_loss
    elif correction_index.numel():
        edge_loss = correction_loss
    else:
        edge_loss = preservation_loss

    sidx = split_indices(case.instance_split_target, negative_ratio=8, rng=rng)
    split_loss = (
        F.binary_cross_entropy_with_logits(
            full.split_logits[sidx],
            case.instance_split_target[sidx].to(full.split_logits.dtype),
        )
        if sidx.numel()
        else zero
    )

    causal_valid = case.valid_edge_mask & case.local_edge_mask
    valid_index = torch.nonzero(causal_valid, as_tuple=False).flatten()
    if valid_index.numel():
        corrupted_noop = F.smooth_l1_loss(
            corrupted.final_edge_logits[valid_index],
            case.rag.spatial_edge_logits[valid_index].detach(),
            beta=0.5,
        )
        corrupted_gate = corrupted.edge_temporal_gate[valid_index].square().mean()
    else:
        corrupted_noop = zero
        corrupted_gate = zero

    if correction_index.numel():
        target = case.target_keep[correction_index].to(full.final_edge_logits.dtype)
        direction = target.mul(2.0).sub(1.0)
        improvement = direction * (
            full.final_edge_logits[correction_index]
            - corrupted.final_edge_logits[correction_index].detach()
        )
        causal_margin = F.relu(float(margin) - improvement).mean()
    else:
        causal_margin = zero

    total = (
        edge_loss
        + float(split_weight) * split_loss
        + float(noop_weight) * corrupted_noop
        + float(corrupted_gate_weight) * corrupted_gate
        + float(margin_weight) * causal_margin
    )
    return SyntheticLoss(
        total=total,
        edge=edge_loss,
        correction=correction_loss,
        preservation=preservation_loss,
        split=split_loss,
        corrupted_noop=corrupted_noop,
        corrupted_gate=corrupted_gate,
        causal_margin=causal_margin,
        correction_edges=int(correction_index.numel()),
        preservation_edges=int(preserve_index.numel()),
    )


# =============================================================================
# Evaluation
# =============================================================================


@dataclass
class EvalAccumulator35:
    cases: int = 0
    correction_edges: int = 0
    full_correction_correct: int = 0
    shuffled_correction_correct: int = 0
    contentless_correction_correct: int = 0
    preservation_edges: int = 0
    preservation_correct: int = 0
    selected_exact: int = 0
    local_clean_components: int = 0
    local_clean_split: int = 0
    max_empty_noop_error: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        corr = max(self.correction_edges, 1)
        pres = max(self.preservation_edges, 1)
        cases = max(self.cases, 1)
        clean = max(self.local_clean_components, 1)
        full = self.full_correction_correct / corr
        shuffled = self.shuffled_correction_correct / corr
        contentless = self.contentless_correction_correct / corr
        return {
            "cases": self.cases,
            "correction_edges": self.correction_edges,
            "full_correction_accuracy": float(full),
            "shuffled_correction_accuracy": float(shuffled),
            "contentless_correction_accuracy": float(contentless),
            "full_minus_shuffled": float(full - shuffled),
            "full_minus_contentless": float(full - contentless),
            "preservation_edges": self.preservation_edges,
            "preservation_accuracy": float(self.preservation_correct / pres),
            "selected_component_exact_rate": float(self.selected_exact / cases),
            "local_clean_component_split_rate": float(self.local_clean_split / clean),
            "max_empty_noop_error": float(self.max_empty_noop_error),
        }


def selected_component_exact(
    runtime: RuntimeFrame,
    case: SyntheticCaseState,
    predicted: PartitionState,
) -> bool:
    node_manual = runtime.node_manual
    component = predicted.node_component_global
    selected_components: set[int] = set()
    for manual_id in case.merge_ids:
        rows = torch.nonzero(node_manual == int(manual_id), as_tuple=False).flatten()
        if rows.numel() == 0:
            return False
        values = torch.unique(component[rows])
        if values.numel() != 1:
            return False
        comp = int(values.item())
        if comp in selected_components:
            return False
        selected_components.add(comp)
        members = torch.nonzero(component == comp, as_tuple=False).flatten()
        positive_manual = node_manual[members]
        if bool((positive_manual != int(manual_id)).any()):
            return False
    return True


def local_clean_split_counts(
    runtime: RuntimeFrame,
    case: SyntheticCaseState,
    predicted: PartitionState,
    dref_um: float,
    radius_dref: float,
) -> tuple[int, int]:
    node_manual = runtime.node_manual
    component = predicted.node_component_global
    node_distance = torch.linalg.vector_norm(runtime.rag.node_centroid_um.float() - case.anchor_um[None], dim=-1)
    local = node_distance <= float(radius_dref) * float(dref_um)
    total = split = 0
    for manual_id in torch.unique(node_manual[local]).tolist():
        manual_id = int(manual_id)
        if manual_id <= 0 or manual_id in case.merge_ids:
            continue
        rows = torch.nonzero(node_manual == manual_id, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        total += 1
        split += int(torch.unique(component[rows]).numel() > 1)
    return total, split


def correction_correct(
    reasoning,
    case: SyntheticCaseState,
    *,
    merge_threshold: float,
) -> tuple[int, int]:
    mask = case.correction_edge_mask & case.local_edge_mask
    count = int(mask.sum().item())
    if not count:
        return 0, 0
    predicted_keep = (
        reasoning.final_edge_logits[mask].sigmoid() >= float(merge_threshold)
    )
    target_keep = case.target_keep[mask]
    return int((predicted_keep == target_keep).sum().item()), count


def preservation_correct(
    reasoning,
    case: SyntheticCaseState,
    *,
    merge_threshold: float,
) -> tuple[int, int]:
    mask = case.valid_edge_mask & case.local_edge_mask & ~case.correction_edge_mask
    count = int(mask.sum().item())
    if not count:
        return 0, 0
    predicted_keep = (
        reasoning.final_edge_logits[mask].sigmoid() >= float(merge_threshold)
    )
    target_keep = case.target_keep[mask]
    return int((predicted_keep == target_keep).sum().item()), count


def trainable_modules(model) -> tuple[Any, ...]:
    return (
        model.instance_tokenizer,
        model.history_encoder,
        model.temporal_encoder,
        model.temporal_observer,
        model.instance_temporal,
    )


def set_temporal_train_mode(model, training: bool) -> None:
    for module in trainable_modules(model):
        module.train(training)


@torch.no_grad()
def evaluate_model35(
    model,
    *,
    paths: Paths,
    manifest: CandidateManifest,
    loader: RuntimeFrameLoader,
    temporal_static: TemporalStatic,
    track_graph,
    frame_count: int,
    temporal_radius: int,
    spacing: Sequence[float],
    dref_um: float,
    neighbourhood_dref: float,
    local_edge_radius_dref: float,
    synthetic_spatial_logit: float,
    val_cases: int,
    device: torch.device,
    seed: int,
    complete_candidate_graph: bool,
) -> dict[str, Any]:
    set_temporal_train_mode(model, False)
    ordered = sorted(
        manifest.val_pairs,
        key=lambda p: stable_fraction("inv35-eval", seed, p.frame, p.a, p.b),
    )
    selected = ordered[: min(int(val_cases), len(ordered))]
    if not selected:
        raise RuntimeError("No validation pair candidates are available")

    acc = EvalAccumulator35()
    for case_index, pair in enumerate(selected):
        runtime = loader.load(pair.frame)
        merge_ids = (int(pair.a), int(pair.b))
        synthetic = build_synthetic_partition(
            runtime,
            temporal_static=temporal_static,
            merge_ids=merge_ids,
            anchor_ids=merge_ids,
            dref_um=dref_um,
            synthetic_spatial_logit=synthetic_spatial_logit,
            local_edge_radius_dref=local_edge_radius_dref,
        )
        graph = build_local_temporal_graph(
            temporal_static=temporal_static,
            track_graph=track_graph,
            manual_labels=runtime.manual,
            target_t=pair.frame,
            frame_count=frame_count,
            temporal_radius=temporal_radius,
            available_offsets=sequence_available_time_offsets(pair.frame, frame_count, temporal_radius),
            anchor_ids=merge_ids,
            merge_ids=merge_ids,
            spacing=spacing,
            dref_um=dref_um,
            neighbourhood_dref=neighbourhood_dref,
            complete_candidate_graph=complete_candidate_graph,
        )

        # Build shared instance + temporal-base state once.
        reference = synthetic.rag.node_features
        decoded, geometry = dummy_geometry_and_decode(model, reference)
        spacing_t = torch.tensor([spacing], device=device, dtype=torch.float32)
        dref_t = torch.tensor([float(dref_um)], device=device, dtype=torch.float32)
        instances = model.instance_tokenizer(
            synthetic.partition,
            synthetic.rag,
            decoded,
            geometry,
            spacing_t,
            dref_t,
            profile_prefix="inv35_eval_tokenizer",
        )
        temporal_base = model.temporal_encoder(temporal_input_from_graph(model, graph, device))
        full_temporal = observe_from_raw_lookup(model, temporal_base, runtime.observer)
        full = model.instance_temporal(instances, synthetic.rag, full_temporal, dref_t)

        shuffled_base = shuffled_temporal_state(temporal_base, seed=seed + 1009 * case_index)
        shuffled = model.instance_temporal(
            instances,
            synthetic.rag,
            observe_from_raw_lookup(model, shuffled_base, runtime.observer),
            dref_t,
        )
        contentless_base = contentless_temporal_state(temporal_base)
        contentless = model.instance_temporal(
            instances,
            synthetic.rag,
            observe_from_raw_lookup(model, contentless_base, runtime.observer),
            dref_t,
        )
        empty = model.instance_temporal(
            instances,
            synthetic.rag,
            model.temporal_encoder.empty(device, full.final_edge_logits.dtype),
            dref_t,
        )

        empty_error = (
            float((empty.final_edge_logits - synthetic.rag.spatial_edge_logits).abs().max().item())
            if synthetic.rag.spatial_edge_logits.numel()
            else 0.0
        )
        acc.max_empty_noop_error = max(acc.max_empty_noop_error, empty_error)

        final_threshold = float(model.cfg.partition.final_merge_threshold)
        full_correct, correction_count = correction_correct(
            full,
            synthetic,
            merge_threshold=final_threshold,
        )
        shuffled_correct, _ = correction_correct(
            shuffled,
            synthetic,
            merge_threshold=final_threshold,
        )
        contentless_correct, _ = correction_correct(
            contentless,
            synthetic,
            merge_threshold=final_threshold,
        )
        preserve_correct_count, preserve_count = preservation_correct(
            full,
            synthetic,
            merge_threshold=final_threshold,
        )
        acc.cases += 1
        acc.correction_edges += correction_count
        acc.full_correction_correct += full_correct
        acc.shuffled_correction_correct += shuffled_correct
        acc.contentless_correction_correct += contentless_correct
        acc.preservation_edges += preserve_count
        acc.preservation_correct += preserve_correct_count

        predicted = model.partitioner(
            synthetic.rag,
            full.final_edge_logits,
            model.cfg.partition.final_merge_threshold,
            stage="final",
        )
        acc.selected_exact += int(selected_component_exact(runtime, synthetic, predicted))
        clean_total, clean_split = local_clean_split_counts(
            runtime,
            synthetic,
            predicted,
            dref_um,
            local_edge_radius_dref,
        )
        acc.local_clean_components += clean_total
        acc.local_clean_split += clean_split

    metrics = acc.as_dict()
    metrics["empty_exact_noop"] = bool(metrics["max_empty_noop_error"] == 0.0)
    min_gap = min(metrics["full_minus_shuffled"], metrics["full_minus_contentless"])
    metrics["minimum_causal_gap"] = float(min_gap)
    metrics["strict_pass"] = bool(
        metrics["empty_exact_noop"]
        and metrics["full_correction_accuracy"] >= 0.90
        and metrics["preservation_accuracy"] >= 0.98
        and metrics["selected_component_exact_rate"] >= 0.80
        and metrics["local_clean_component_split_rate"] <= 0.02
        and min_gap >= 0.15
    )
    metrics["checkpoint_score"] = float(
        2.5 * metrics["full_correction_accuracy"]
        + 2.0 * metrics["selected_component_exact_rate"]
        + 1.0 * metrics["preservation_accuracy"]
        - 4.0 * metrics["local_clean_component_split_rate"]
        + 1.5 * metrics["full_minus_shuffled"]
        + 1.5 * metrics["full_minus_contentless"]
    )
    set_temporal_train_mode(model, True)
    return metrics


def print_eval35(step: int, metrics: dict[str, Any]) -> None:
    print()
    print("=" * 118)
    print(f"INVESTIGATION 35 VALIDATION @ STEP {step}")
    print("=" * 118)
    print(f"FULL correction      : {metrics['full_correction_accuracy']:.4f}")
    print(f"preservation         : {metrics['preservation_accuracy']:.4f}")
    print(f"selected exact       : {metrics['selected_component_exact_rate']:.4f}")
    print(f"local clean split    : {metrics['local_clean_component_split_rate']:.4f}")
    print(f"SHUFFLED correction  : {metrics['shuffled_correction_accuracy']:.4f}")
    print(f"CONTENTLESS correction: {metrics['contentless_correction_accuracy']:.4f}")
    print(f"FULL - SHUFFLED      : {metrics['full_minus_shuffled']:+.4f}")
    print(f"FULL - CONTENTLESS   : {metrics['full_minus_contentless']:+.4f}")
    print(f"EMPTY exact no-op    : {metrics['empty_exact_noop']}")
    print(f"STRICT PASS          : {metrics['strict_pass']}")
    print(f"checkpoint score     : {metrics['checkpoint_score']:.5f}")
    print("=" * 118)


# =============================================================================
# Production temporal-stage optimization / checkpointing
# =============================================================================


def configure_temporal_training(model, device: torch.device) -> list[Tensor]:
    """Freeze the mature spatial system and move only temporal-stage modules.

    The complete model remains checkpoint-compatible, but spatial parameters stay
    on CPU and require no gradients.  This makes Investigation 35 a genuine
    temporal-stage trainer rather than a hidden joint fine-tune.
    """
    model.cpu()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    parameters: list[Tensor] = []
    seen: set[int] = set()
    for module in trainable_modules(model):
        module.to(device)
        module.train(True)
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)

    if not parameters:
        raise RuntimeError("No trainable temporal-stage parameters were found")
    return parameters


def temporal_parameter_audit(model) -> dict[str, Any]:
    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    allowed_prefixes = (
        "instance_tokenizer.",
        "history_encoder.",
        "temporal_encoder.",
        "temporal_observer.",
        "instance_temporal.",
    )
    unexpected = [
        name
        for name in trainable_names
        if not name.startswith(allowed_prefixes)
    ]
    if unexpected:
        raise RuntimeError(
            "Investigation 35 unexpectedly made spatial parameters trainable: "
            + ", ".join(unexpected[:20])
        )
    counts = {}
    for prefix in allowed_prefixes:
        counts[prefix[:-1]] = sum(
            int(parameter.numel())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith(prefix)
        )
    counts["total"] = sum(counts.values())
    return {
        "trainable_parameter_count": counts,
        "trainable_parameter_names": trainable_names,
    }


def make_training_config(args: argparse.Namespace) -> TrainingConfig:
    config = TrainingConfig()
    config.lr = float(args.lr)
    config.weight_decay = float(args.weight_decay)
    config.max_grad_norm = float(args.grad_clip)
    config.amp_dtype = str(args.amp_dtype)
    config.curriculum.fixed_stage = "instance_temporal"
    config.curriculum.instance_temporal_detached_spatial = True
    config.curriculum.instance_temporal_freeze_spatial = True
    config.loss.temporal_causal_enabled = True
    config.loss.temporal_causal_noop_weight = float(args.noop_weight)
    config.loss.temporal_causal_corrupted_gate_weight = float(
        args.corrupted_gate_weight
    )
    config.loss.temporal_causal_margin_weight = float(args.margin_weight)
    config.loss.temporal_causal_margin = float(args.margin)
    config.loss.temporal_causal_corruptions = tuple(CORRUPTIONS)
    config.loss.temporal_causal_seed = int(args.seed) + 35_000
    config.validate()
    return config


def make_grad_scaler(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda" and amp_dtype == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        # Compatibility with older PyTorch installations.
        return torch.cuda.amp.GradScaler(enabled=enabled)


def checkpoint_extra(
    *,
    paths: Paths,
    args: argparse.Namespace,
    dref_um: float,
    metrics: dict[str, Any] | None,
    parameter_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "investigation": SCRIPT_NAME,
        "curriculum_stage": "instance_temporal",
        "sample_id": paths.sample,
        "spatial_checkpoint": str(paths.checkpoint),
        "dref_um": float(dref_um),
        "spacing_zyx_um": tuple(float(v) for v in args.spacing),
        "synthetic_spatial_logit": float(args.synthetic_spatial_logit),
        "temporal_neighbourhood_dref": float(args.temporal_neighbourhood_dref),
        "local_edge_radius_dref": float(args.local_edge_radius_dref),
        "manual_signatures": [
            file_signature(paths.manual(t))
            for t in range(int(args.frame_count))
        ],
        "parameter_audit": parameter_audit,
        "validation_metrics": metrics or {},
        "notes": {
            "spatial_parameters_optimized": False,
            "spatial_cnn_runs_during_training": False,
            "target_frame_temporal_detection_leakage": False,
            "corruption_location": (
                "after TemporalGraphEncoder and before TemporalSpatialObserver"
            ),
            "corruptions": list(CORRUPTIONS),
            "full_keep_gate_penalty": False,
        },
    }


def save_training_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scaler,
    step: int,
    training_config: TrainingConfig,
    paths: Paths,
    args: argparse.Namespace,
    dref_um: float,
    metrics: dict[str, Any] | None,
    parameter_audit: dict[str, Any],
) -> None:
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=int(step),
        epoch=0,
        model_config=model.cfg,
        training_config=training_config,
        extra=checkpoint_extra(
            paths=paths,
            args=args,
            dref_um=dref_um,
            metrics=metrics,
            parameter_audit=parameter_audit,
        ),
    )


def safe_training_triples(
    frame_data: FrameCandidates,
    manifest: CandidateManifest,
    maximum: int,
) -> list[tuple[int, int, int]]:
    """Exclude triples that contain a held-out validation pair."""
    val_keys = {
        (int(pair.frame), min(int(pair.a), int(pair.b)), max(int(pair.a), int(pair.b)))
        for pair in manifest.val_pairs
    }
    result: list[tuple[int, int, int]] = []
    for triple in selected_triples(frame_data, maximum):
        a, b, c = map(int, triple)
        pair_keys = (
            (frame_data.frame, min(a, b), max(a, b)),
            (frame_data.frame, min(a, c), max(a, c)),
            (frame_data.frame, min(b, c), max(b, c)),
        )
        if any(key in val_keys for key in pair_keys):
            continue
        result.append((a, b, c))
    return result


def target_nodes_have_context(
    track_graph,
    target_nodes: Sequence[int],
    *,
    target_t: int,
    available_offsets: Sequence[int],
) -> bool:
    absolute_times = {
        int(target_t) + int(offset)
        for offset in available_offsets
        if int(offset) != 0
    }
    if not absolute_times:
        return False
    for node_id in target_nodes:
        found = False
        neighbours = list(track_graph.predecessors(int(node_id))) + list(
            track_graph.successors(int(node_id))
        )
        for other in neighbours:
            if int(track_graph.nodes[int(other)]["time"]) in absolute_times:
                found = True
                break
        if not found:
            return False
    return True


def choose_supported_context(
    *,
    target_t: int,
    frame_count: int,
    temporal_radius: int,
    target_nodes: Sequence[int],
    track_graph,
    rng: random.Random,
) -> tuple[str, tuple[int, ...]]:
    name, offsets = choose_context_variant(
        target_t,
        frame_count,
        temporal_radius,
        rng,
    )
    if target_nodes_have_context(
        track_graph,
        target_nodes,
        target_t=target_t,
        available_offsets=offsets,
    ):
        return name, offsets
    full = tuple(
        sequence_available_time_offsets(
            target_t,
            frame_count,
            temporal_radius,
        )
    )
    return "full_fallback", full


def pair_rows_by_frame(
    pairs: Sequence[PairCandidate],
) -> dict[int, list[PairCandidate]]:
    result: dict[int, list[PairCandidate]] = defaultdict(list)
    for pair in pairs:
        result[int(pair.frame)].append(pair)
    return dict(result)


def sample_train_pair(
    rows: Sequence[PairCandidate],
    rng: random.Random,
) -> PairCandidate:
    if not rows:
        raise RuntimeError("Cannot sample an empty training-pair list")
    # Slightly favor pairs with a larger true interface: they are more likely
    # to resemble a realistic under-segmentation than point-contact neighbours.
    weights = [max(float(row.interface_edges), 1.0) ** 0.5 for row in rows]
    return rng.choices(list(rows), weights=weights, k=1)[0]


@dataclass(frozen=True)
class TrainMicroSpec:
    frame: int
    anchor_ids: tuple[int, ...]
    merge_ids: tuple[int, ...]
    kind: str
    context_name: str
    available_offsets: tuple[int, ...]
    corruption: str


def make_micro_spec(
    *,
    frame: int,
    frame_data: FrameCandidates,
    train_pairs: Sequence[PairCandidate],
    manifest: CandidateManifest,
    temporal_static: TemporalStatic,
    track_graph,
    frame_count: int,
    temporal_radius: int,
    max_triples: int,
    triple_fraction: float,
    clean: bool,
    corruption: str,
    rng: random.Random,
) -> TrainMicroSpec:
    pair = sample_train_pair(train_pairs, rng)
    anchor_ids: tuple[int, ...] = (int(pair.a), int(pair.b))
    merge_ids: tuple[int, ...] = ()
    kind = "clean_close_pair"

    if not clean:
        triples = safe_training_triples(
            frame_data,
            manifest,
            max_triples,
        )
        if triples and rng.random() < float(triple_fraction):
            anchor_ids = tuple(map(int, rng.choice(triples)))
            merge_ids = anchor_ids
            kind = "synthetic_three_cell_merge"
        else:
            merge_ids = anchor_ids
            kind = "synthetic_two_cell_merge"

    target_nodes = member_target_nodes(
        temporal_static,
        frame,
        anchor_ids,
    )
    context_name, offsets = choose_supported_context(
        target_t=frame,
        frame_count=frame_count,
        temporal_radius=temporal_radius,
        target_nodes=target_nodes,
        track_graph=track_graph,
        rng=rng,
    )
    return TrainMicroSpec(
        frame=int(frame),
        anchor_ids=anchor_ids,
        merge_ids=merge_ids,
        kind=kind,
        context_name=context_name,
        available_offsets=tuple(offsets),
        corruption=str(corruption),
    )


def run_train_microcase(
    model,
    *,
    runtime: RuntimeFrame,
    spec: TrainMicroSpec,
    temporal_static: TemporalStatic,
    track_graph,
    args: argparse.Namespace,
    dref_um: float,
    device: torch.device,
    rng: random.Random,
    corruption_seed: int,
) -> tuple[SyntheticLoss, dict[str, Any]]:
    case = build_synthetic_partition(
        runtime,
        temporal_static=temporal_static,
        merge_ids=spec.merge_ids,
        anchor_ids=spec.anchor_ids,
        dref_um=dref_um,
        synthetic_spatial_logit=float(args.synthetic_spatial_logit),
        local_edge_radius_dref=float(args.local_edge_radius_dref),
    )

    if spec.merge_ids and not bool(case.correction_edge_mask.any()):
        raise RuntimeError(
            "A selected positive merge has no correction RAG edge; "
            f"t={runtime.t}, merge_ids={spec.merge_ids}. "
            "The candidate manifest is inconsistent with the frozen graph."
        )

    temporal_graph = build_local_temporal_graph(
        temporal_static=temporal_static,
        track_graph=track_graph,
        manual_labels=runtime.manual,
        target_t=runtime.t,
        frame_count=int(args.frame_count),
        temporal_radius=int(args.temporal_radius),
        available_offsets=spec.available_offsets,
        anchor_ids=spec.anchor_ids,
        merge_ids=spec.merge_ids,
        spacing=args.spacing,
        dref_um=dref_um,
        neighbourhood_dref=float(args.temporal_neighbourhood_dref),
        complete_candidate_graph=bool(args.complete_candidate_graph),
    )

    forward = production_temporal_forward(
        model,
        case=case,
        temporal_graph=temporal_graph,
        observer_lookup=runtime.observer,
        spacing=args.spacing,
        dref_um=dref_um,
        device=device,
        corruption=spec.corruption,
        corruption_seed=int(corruption_seed),
    )

    loss = synthetic_causal_loss(
        case,
        forward,
        preserve_edges=int(args.preserve_edges),
        preserve_ratio=int(args.preserve_ratio),
        split_weight=float(args.split_weight),
        noop_weight=float(args.noop_weight),
        corrupted_gate_weight=float(args.corrupted_gate_weight),
        margin_weight=float(args.margin_weight),
        margin=float(args.margin),
        rng=rng,
    )

    correction_mask = case.correction_edge_mask & case.local_edge_mask
    correction_gate = (
        float(
            forward.full_reasoning.edge_temporal_gate[
                correction_mask
            ].detach().mean().cpu()
        )
        if bool(correction_mask.any())
        else 0.0
    )
    meta = {
        "frame": int(runtime.t),
        "kind": spec.kind,
        "anchor_ids": list(spec.anchor_ids),
        "merge_ids": list(spec.merge_ids),
        "context": spec.context_name,
        "available_offsets": list(spec.available_offsets),
        "corruption": spec.corruption,
        "correction_edges": int(loss.correction_edges),
        "preservation_edges": int(loss.preservation_edges),
        "mean_full_correction_gate": correction_gate,
        "temporal_nodes": int(torch.as_tensor(temporal_graph["graph_x"]).shape[0]),
        "tracklets": int(torch.as_tensor(temporal_graph["temporal_ref_um"]).shape[0]),
    }
    return loss, meta


def training_metrics_row(
    *,
    step: int,
    frame: int,
    accumulated: dict[str, float],
    microcases: list[dict[str, Any]],
    grad_norm: float,
    lr: float,
    step_seconds: float,
    device: torch.device,
) -> dict[str, Any]:
    denominator = max(len(microcases), 1)
    row = {
        "step": int(step),
        "frame": int(frame),
        "loss": accumulated["total"] / denominator,
        "edge_loss": accumulated["edge"] / denominator,
        "correction_loss": accumulated["correction"] / denominator,
        "preservation_loss": accumulated["preservation"] / denominator,
        "split_loss": accumulated["split"] / denominator,
        "corrupted_noop": accumulated["noop"] / denominator,
        "corrupted_gate": accumulated["corrupted_gate"] / denominator,
        "causal_margin": accumulated["margin"] / denominator,
        "grad_norm": float(grad_norm),
        "lr": float(lr),
        "step_seconds": float(step_seconds),
        "microcases": microcases,
    }
    if device.type == "cuda":
        row["peak_allocated_mb"] = float(
            torch.cuda.max_memory_allocated(device) / (1024**2)
        )
        row["peak_reserved_mb"] = float(
            torch.cuda.max_memory_reserved(device) / (1024**2)
        )
    return row


def train_model35(
    model,
    *,
    paths: Paths,
    manifest: CandidateManifest,
    temporal_static: TemporalStatic,
    track_graph,
    dref_um: float,
    device: torch.device,
    args: argparse.Namespace,
    resume_checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    training_config = make_training_config(args)
    trainable = configure_temporal_training(model, device)
    parameter_audit = temporal_parameter_audit(model)

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scaler = make_grad_scaler(device, str(args.amp_dtype))

    start_step = 0
    if resume_checkpoint is not None:
        start_step = int(resume_checkpoint.get("global_step", 0))
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if "scaler" in resume_checkpoint:
            try:
                scaler.load_state_dict(resume_checkpoint["scaler"])
            except Exception as exc:
                print(
                    f"[resume] scaler state was not restored: {exc}",
                    flush=True,
                )
        print(f"[resume] continuing after optimizer step {start_step}")

    if start_step >= int(args.steps):
        print(
            f"[resume] checkpoint step {start_step} already reaches --steps={args.steps}; "
            "running validation only."
        )

    pair_by_frame = pair_rows_by_frame(manifest.train_pairs)
    train_frames = sorted(frame for frame, rows in pair_by_frame.items() if rows)
    if not train_frames:
        raise RuntimeError("No frames contain training-pair candidates")

    frame_weights = np.asarray(
        [max(len(pair_by_frame[frame]), 1) for frame in train_frames],
        dtype=np.float64,
    )
    frame_weights /= frame_weights.sum()

    loader = RuntimeFrameLoader(paths, manifest, device)
    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    history_path = paths.output / "training_history.json"
    validation_path = paths.output / "validation_history.json"

    if history_path.is_file() and resume_checkpoint is not None:
        try:
            history = list(json.loads(history_path.read_text(encoding="utf-8")))
        except Exception:
            history = []
    if validation_path.is_file() and resume_checkpoint is not None:
        try:
            validation_history = list(
                json.loads(validation_path.read_text(encoding="utf-8"))
            )
        except Exception:
            validation_history = []

    best_score = -float("inf")
    best_strict = False
    best_metrics_path = paths.output / "best_metrics.json"
    if best_metrics_path.is_file():
        try:
            old_best = json.loads(best_metrics_path.read_text(encoding="utf-8"))
            best_score = float(old_best.get("checkpoint_score", -float("inf")))
            best_strict = bool(old_best.get("strict_pass", False))
        except Exception:
            pass

    print()
    print("=" * 118)
    print("INVESTIGATION 35 — PRODUCTION TEMPORAL MERGE-SYNTHESIS TRAINING")
    print("=" * 118)
    print(f"device                   : {device}")
    print(f"AMP                      : {args.amp_dtype}")
    print(f"optimizer steps          : {args.steps}")
    print(f"microcases / step        : {args.accumulate_cases}")
    print(f"steps / loaded frame     : {args.steps_per_frame_block}")
    print(f"train pair candidates    : {len(manifest.train_pairs)}")
    print(f"held-out pair candidates : {len(manifest.val_pairs)}")
    print(f"trainable parameters     : {parameter_audit['trainable_parameter_count']['total']:,}")
    print(f"learning rate            : {args.lr:g}")
    print(f"synthetic spatial logit  : +/-{args.synthetic_spatial_logit:g}")
    print(f"clean close-pair fraction: {args.clean_fraction:.2f}")
    print(f"three-cell merge fraction: {args.triple_fraction:.2f}")
    print("spatial CNN during train : NO")
    print("spatial parameters       : FROZEN")
    print("=" * 118)

    # Deterministic starting point for frame-block rotation.  The random state
    # is intentionally a function of requested seed + step, so resuming does not
    # depend on Python's pickled RNG internals.
    current_frame: int | None = None
    run_started = time.perf_counter()

    for step in range(start_step + 1, int(args.steps) + 1):
        step_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        if (
            current_frame is None
            or (step - 1) % int(args.steps_per_frame_block) == 0
        ):
            # Use a step-derived generator so frame selection is reproducible
            # even after resume.
            block_rng = np.random.default_rng(
                int(args.seed) + 71_003 * ((step - 1) // int(args.steps_per_frame_block))
            )
            current_frame = int(
                block_rng.choice(train_frames, p=frame_weights)
            )

        runtime = loader.load(current_frame)
        optimizer.zero_grad(set_to_none=True)
        # Step-local RNG makes synthetic sampling exactly reproducible across
        # resume: step N gets the same pair/context/corruption-side sampling
        # whether reached continuously or loaded from a checkpoint.
        step_rng = random.Random(
            int(args.seed) + 15_485_863 * int(step)
        )

        accumulated = {
            "total": 0.0,
            "edge": 0.0,
            "correction": 0.0,
            "preservation": 0.0,
            "split": 0.0,
            "noop": 0.0,
            "corrupted_gate": 0.0,
            "margin": 0.0,
        }
        micro_meta: list[dict[str, Any]] = []

        for micro in range(int(args.accumulate_cases)):
            # Every optimizer update starts with a positive correction case.
            # With >=2 microcases it also always contains one clean close-pair
            # preservation case.  Remaining cases follow clean_fraction.
            if micro == 0:
                clean = False
            elif micro == 1 and int(args.accumulate_cases) >= 2:
                clean = True
            else:
                clean = step_rng.random() < float(args.clean_fraction)

            corruption = CORRUPTIONS[(step + micro) % len(CORRUPTIONS)]
            spec = make_micro_spec(
                frame=current_frame,
                frame_data=manifest.frames[current_frame],
                train_pairs=pair_by_frame[current_frame],
                manifest=manifest,
                temporal_static=temporal_static,
                track_graph=track_graph,
                frame_count=int(args.frame_count),
                temporal_radius=int(args.temporal_radius),
                max_triples=int(args.max_triples_per_frame),
                triple_fraction=float(args.triple_fraction),
                clean=clean,
                corruption=corruption,
                rng=step_rng,
            )

            with autocast_for(device, str(args.amp_dtype)):
                loss, meta = run_train_microcase(
                    model,
                    runtime=runtime,
                    spec=spec,
                    temporal_static=temporal_static,
                    track_graph=track_graph,
                    args=args,
                    dref_um=dref_um,
                    device=device,
                    rng=step_rng,
                    corruption_seed=(
                        int(args.seed)
                        + 1_000_003 * int(step)
                        + 10_007 * int(micro)
                    ),
                )
                scaled_loss = loss.total / float(args.accumulate_cases)

            if not bool(torch.isfinite(scaled_loss.detach())):
                raise FloatingPointError(
                    f"Non-finite Investigation-35 loss at step={step}, micro={micro}"
                )
            scaler.scale(scaled_loss).backward()

            accumulated["total"] += float(loss.total.detach().float().cpu())
            accumulated["edge"] += float(loss.edge.detach().float().cpu())
            accumulated["correction"] += float(
                loss.correction.detach().float().cpu()
            )
            accumulated["preservation"] += float(
                loss.preservation.detach().float().cpu()
            )
            accumulated["split"] += float(loss.split.detach().float().cpu())
            accumulated["noop"] += float(
                loss.corrupted_noop.detach().float().cpu()
            )
            accumulated["corrupted_gate"] += float(
                loss.corrupted_gate.detach().float().cpu()
            )
            accumulated["margin"] += float(
                loss.causal_margin.detach().float().cpu()
            )
            micro_meta.append(meta)

        scaler.unscale_(optimizer)
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            trainable,
            float(args.grad_clip),
        )
        grad_norm = float(torch.as_tensor(grad_norm_tensor).detach().cpu())
        if not math.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"Non-finite Investigation-35 gradient norm at step={step}"
            )

        scaler.step(optimizer)
        scaler.update()

        row = training_metrics_row(
            step=step,
            frame=current_frame,
            accumulated=accumulated,
            microcases=micro_meta,
            grad_norm=grad_norm,
            lr=float(optimizer.param_groups[0]["lr"]),
            step_seconds=time.perf_counter() - step_started,
            device=device,
        )
        history.append(row)

        if step == 1 or step % int(args.print_every) == 0:
            elapsed = duration(time.perf_counter() - run_started)
            print(
                f"[step {step:05d}/{args.steps}] "
                f"t={current_frame:02d} "
                f"loss={row['loss']:.5f} "
                f"corr={row['correction_loss']:.5f} "
                f"pres={row['preservation_loss']:.5f} "
                f"noop={row['corrupted_noop']:.5f} "
                f"margin={row['causal_margin']:.5f} "
                f"negGate={row['corrupted_gate']:.5f} "
                f"grad={row['grad_norm']:.3f} "
                f"elapsed={elapsed}",
                flush=True,
            )

        if step % 50 == 0:
            atomic_json(history_path, history)

        if step % int(args.eval_every) == 0 or step == int(args.steps):
            metrics = evaluate_model35(
                model,
                paths=paths,
                manifest=manifest,
                loader=loader,
                temporal_static=temporal_static,
                track_graph=track_graph,
                frame_count=int(args.frame_count),
                temporal_radius=int(args.temporal_radius),
                spacing=args.spacing,
                dref_um=dref_um,
                neighbourhood_dref=float(args.temporal_neighbourhood_dref),
                local_edge_radius_dref=float(args.local_edge_radius_dref),
                synthetic_spatial_logit=float(args.synthetic_spatial_logit),
                val_cases=int(args.val_cases),
                device=device,
                seed=int(args.seed) + int(step),
                complete_candidate_graph=bool(args.complete_candidate_graph),
            )
            print_eval35(step, metrics)
            validation_history.append({"step": int(step), "metrics": metrics})
            atomic_json(validation_path, validation_history)
            atomic_json(history_path, history)

            save_training_checkpoint(
                paths.output / "latest.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                training_config=training_config,
                paths=paths,
                args=args,
                dref_um=dref_um,
                metrics=metrics,
                parameter_audit=parameter_audit,
            )

            score = float(metrics["checkpoint_score"])
            strict = bool(metrics["strict_pass"])
            improved = (
                (strict and not best_strict)
                or (strict == best_strict and score > best_score)
            )
            if improved:
                best_score = score
                best_strict = strict
                save_training_checkpoint(
                    paths.output / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    training_config=training_config,
                    paths=paths,
                    args=args,
                    dref_um=dref_um,
                    metrics=metrics,
                    parameter_audit=parameter_audit,
                )
                atomic_json(paths.output / "best_metrics.json", metrics)
                print(
                    f"[best] step={step} score={score:.5f} strict={strict}",
                    flush=True,
                )

    # Validation-only resume path.
    if start_step >= int(args.steps):
        metrics = evaluate_model35(
            model,
            paths=paths,
            manifest=manifest,
            loader=loader,
            temporal_static=temporal_static,
            track_graph=track_graph,
            frame_count=int(args.frame_count),
            temporal_radius=int(args.temporal_radius),
            spacing=args.spacing,
            dref_um=dref_um,
            neighbourhood_dref=float(args.temporal_neighbourhood_dref),
            local_edge_radius_dref=float(args.local_edge_radius_dref),
            synthetic_spatial_logit=float(args.synthetic_spatial_logit),
            val_cases=int(args.val_cases),
            device=device,
            seed=int(args.seed) + int(start_step),
            complete_candidate_graph=bool(args.complete_candidate_graph),
        )
        print_eval35(start_step, metrics)
        validation_history.append({"step": int(start_step), "metrics": metrics})

    final_metrics = (
        validation_history[-1]["metrics"]
        if validation_history
        else {}
    )
    atomic_json(history_path, history)
    atomic_json(validation_path, validation_history)
    save_training_checkpoint(
        paths.output / "final.pt",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=max(start_step, int(args.steps)),
        training_config=training_config,
        paths=paths,
        args=args,
        dref_um=dref_um,
        metrics=final_metrics,
        parameter_audit=parameter_audit,
    )

    print()
    print("=" * 118)
    print("INVESTIGATION 35 TRAINING COMPLETE")
    print("=" * 118)
    print(f"best   : {paths.output / 'best.pt'}")
    print(f"latest : {paths.output / 'latest.pt'}")
    print(f"final  : {paths.output / 'final.pt'}")
    print("=" * 118)
    return final_metrics


# =============================================================================
# Preparation orchestration
# =============================================================================


def prepare_investigation35(
    args: argparse.Namespace,
    paths: Paths,
) -> tuple[float, Any, TemporalStatic, CandidateManifest]:
    validate_inputs(paths, int(args.frame_count))
    paths.output.mkdir(parents=True, exist_ok=True)
    paths.cache.mkdir(parents=True, exist_ok=True)

    frame_count = int(args.frame_count)
    spacing = tuple(float(v) for v in args.spacing)

    current_manual_signatures = [
        file_signature(paths.manual(t))
        for t in range(frame_count)
    ]
    cached_manual_signatures = None
    if paths.manual_movie_meta.is_file():
        try:
            cached_manual_signatures = json.loads(
                paths.manual_movie_meta.read_text(encoding="utf-8")
            ).get("manual_signatures")
        except Exception:
            cached_manual_signatures = None
    manual_movie_changed = (
        bool(args.rebuild_movies)
        or not paths.manual_movie.is_file()
        or cached_manual_signatures != current_manual_signatures
    )
    stack_movie(
        paths.manual_movie,
        [paths.manual(t) for t in range(frame_count)],
        rebuild=manual_movie_changed,
    )
    atomic_json(
        paths.manual_movie_meta,
        {
            "sample_id": paths.sample,
            "manual_signatures": current_manual_signatures,
        },
    )
    if manual_movie_changed:
        print("[manual movie] rebuilt because source annotations changed")
    manual_movie = np.load(paths.manual_movie, mmap_mode="r")
    build_raw_movie(
        paths,
        frame_count,
        tuple(int(v) for v in manual_movie.shape[-3:]),
        rebuild=bool(args.rebuild_movies),
    )
    prepare_raw_norm(
        paths,
        frame_count,
        rebuild=bool(args.rebuild_movies),
    )
    raw_movie = np.load(paths.raw_movie, mmap_mode="r")

    dref_um, per_frame_dref = resolve_dref(
        paths,
        frame_count,
        spacing,
        args.dref_um,
    )
    if not math.isfinite(dref_um) or dref_um <= 0:
        raise RuntimeError(f"Invalid resolved dref: {dref_um}")
    print(
        f"[scale] fixed movie dref={dref_um:.5f} um; "
        f"source-frame range={min(per_frame_dref):.5f}..{max(per_frame_dref):.5f}",
        flush=True,
    )

    track_graph, tracked_movie = prepare_trackastra(
        paths,
        model_name=str(args.trackastra_model),
        mode=str(args.trackastra_mode),
        device=str(args.trackastra_device),
        rebuild=(bool(args.rebuild_trackastra) or manual_movie_changed),
    )
    temporal_static = prepare_temporal_static(
        paths,
        track_graph,
        tracked_movie,
        raw_movie,
        spacing,
        dref_um,
        frame_count,
        rebuild=(
            bool(args.rebuild_trackastra)
            or bool(args.rebuild_temporal_static)
            or manual_movie_changed
        ),
    )

    prepare_device = torch.device(args.prepare_device)
    prepare_spatial_cache(
        paths,
        frame_count=frame_count,
        spacing=spacing,
        dref_um=dref_um,
        device=prepare_device,
        amp_dtype=str(args.prepare_amp_dtype),
        rebuild=bool(args.rebuild_spatial_cache),
    )

    manifest = build_candidates(
        paths,
        track_graph=track_graph,
        temporal_static=temporal_static,
        frame_count=frame_count,
        spacing=spacing,
        dref_um=dref_um,
        temporal_radius=int(args.temporal_radius),
        min_voxels=int(args.min_voxels),
        max_volume_ratio=float(args.max_volume_ratio),
        max_distance_dref=float(args.max_distance_dref),
        val_fraction=float(args.val_fraction),
        rebuild=(
            bool(args.rebuild_candidates)
            or bool(args.rebuild_spatial_cache)
            or bool(args.rebuild_trackastra)
        ),
    )

    prepare_observer_raw_cache(
        paths,
        manifest=manifest,
        temporal_static=temporal_static,
        track_graph=track_graph,
        frame_count=frame_count,
        temporal_radius=int(args.temporal_radius),
        spacing=spacing,
        dref_um=dref_um,
        neighbourhood_dref=float(args.temporal_neighbourhood_dref),
        max_triples=int(args.max_triples_per_frame),
        device=prepare_device,
        amp_dtype=str(args.prepare_amp_dtype),
        complete_candidate_graph=bool(args.complete_candidate_graph),
        rebuild=(
            bool(args.rebuild_observer_cache)
            or bool(args.rebuild_candidates)
            or bool(args.rebuild_spatial_cache)
            or bool(args.rebuild_trackastra)
        ),
    )

    observer_counts = {}
    for t in range(frame_count):
        try:
            meta = json.loads(paths.observer_meta(t).read_text(encoding="utf-8"))
            observer_counts[f"t{t:03d}"] = int(meta.get("reference_count", 0))
        except Exception:
            observer_counts[f"t{t:03d}"] = -1

    preparation = {
        "format_version": 1,
        "investigation": SCRIPT_NAME,
        "sample_id": paths.sample,
        "checkpoint": file_signature(paths.checkpoint),
        "manual_signatures": [
            file_signature(paths.manual(t))
            for t in range(frame_count)
        ],
        "spacing_zyx_um": spacing,
        "dref_um": float(dref_um),
        "per_frame_source_dref_um": per_frame_dref,
        "frame_count": frame_count,
        "temporal_radius": int(args.temporal_radius),
        "temporal_cache_contract": int(TEMPORAL_CACHE_CONTRACT_VERSION),
        "train_pair_candidates": len(manifest.train_pairs),
        "val_pair_candidates": len(manifest.val_pairs),
        "observer_reference_counts": observer_counts,
        "trackastra_model": str(args.trackastra_model),
        "trackastra_mode": str(args.trackastra_mode),
        "synthetic_training": {
            "synthetic_spatial_logit": float(args.synthetic_spatial_logit),
            "clean_fraction": float(args.clean_fraction),
            "triple_fraction": float(args.triple_fraction),
            "temporal_neighbourhood_dref": float(
                args.temporal_neighbourhood_dref
            ),
            "local_edge_radius_dref": float(args.local_edge_radius_dref),
        },
        "trainable_modules": [
            "instance_tokenizer",
            "history_encoder",
            "temporal_encoder",
            "temporal_observer",
            "instance_temporal",
        ],
        "frozen_modules": [
            "acquisition",
            "evidence_stem",
            "spatial_backbone",
            "geometry_decoder",
            "watershed",
            "rag_builder",
            "rag_network",
            "local_refiner",
        ],
        "training_runs_spatial_cnn": False,
    }
    atomic_json(paths.output / "preparation.json", preparation)

    print()
    print("=" * 118)
    print("INVESTIGATION 35 PREPARATION READY")
    print("=" * 118)
    print(f"train pair candidates : {len(manifest.train_pairs)}")
    print(f"val pair candidates   : {len(manifest.val_pairs)}")
    print(f"dref                  : {dref_um:.5f} um")
    print(f"cache                 : {paths.cache}")
    print("=" * 118)
    return dref_um, track_graph, temporal_static, manifest


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the production STIR-Net temporal branch on controlled, "
            "leak-free BioHub merge synthesis."
        )
    )

    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--inv24", type=Path, default=None)
    parser.add_argument("--stage6-root", type=Path, default=None)
    parser.add_argument("--sample-zarr", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)

    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=DEFAULT_SPACING,
        metavar=("Z", "Y", "X"),
    )
    parser.add_argument(
        "--dref-um",
        type=float,
        default=None,
        help=(
            "Optional fixed non-GT dref. Default is the median dref estimated "
            "from Stage-6 source segmentation over the movie."
        ),
    )
    parser.add_argument("--temporal-radius", type=int, default=2)

    parser.add_argument("--trackastra-model", default="ctc")
    parser.add_argument("--trackastra-mode", default="greedy")
    parser.add_argument(
        "--trackastra-device",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )

    parser.add_argument(
        "--prepare-device",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device for one-time frozen spatial/observer cache preparation.",
    )
    parser.add_argument(
        "--prepare-amp-dtype",
        choices=("fp32", "fp16", "bf16"),
        default=("fp16" if torch.cuda.is_available() else "fp32"),
    )
    parser.add_argument("--prepare-only", action="store_true")

    parser.add_argument("--rebuild-movies", action="store_true")
    parser.add_argument("--rebuild-trackastra", action="store_true")
    parser.add_argument("--rebuild-temporal-static", action="store_true")
    parser.add_argument("--rebuild-spatial-cache", action="store_true")
    parser.add_argument("--rebuild-candidates", action="store_true")
    parser.add_argument("--rebuild-observer-cache", action="store_true")

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--accumulate-cases", type=int, default=DEFAULT_ACCUMULATE)
    parser.add_argument("--steps-per-frame-block", type=int, default=DEFAULT_FRAME_BLOCK)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--print-every", type=int, default=DEFAULT_PRINT_EVERY)
    parser.add_argument("--val-cases", type=int, default=DEFAULT_VAL_CASES)

    parser.add_argument(
        "--synthetic-spatial-logit",
        type=float,
        default=DEFAULT_SYNTHETIC_LOGIT,
    )
    parser.add_argument("--clean-fraction", type=float, default=DEFAULT_CLEAN_FRACTION)
    parser.add_argument("--triple-fraction", type=float, default=DEFAULT_TRIPLE_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--min-voxels", type=int, default=DEFAULT_MIN_VOXELS)
    parser.add_argument(
        "--max-volume-ratio",
        type=float,
        default=DEFAULT_MAX_VOLUME_RATIO,
    )
    parser.add_argument(
        "--max-distance-dref",
        type=float,
        default=DEFAULT_MAX_DISTANCE_DREF,
    )
    parser.add_argument(
        "--temporal-neighbourhood-dref",
        type=float,
        default=DEFAULT_TEMPORAL_NEIGHBORHOOD_DREF,
    )
    parser.add_argument(
        "--local-edge-radius-dref",
        type=float,
        default=DEFAULT_LOCAL_EDGE_RADIUS_DREF,
    )
    parser.add_argument(
        "--max-triples-per-frame",
        type=int,
        default=DEFAULT_MAX_TRIPLES_PER_FRAME,
    )

    parser.add_argument("--preserve-edges", type=int, default=DEFAULT_PRESERVE_EDGES)
    parser.add_argument("--preserve-ratio", type=int, default=DEFAULT_PRESERVE_RATIO)
    parser.add_argument("--split-weight", type=float, default=DEFAULT_SPLIT_WEIGHT)
    parser.add_argument("--noop-weight", type=float, default=DEFAULT_NOOP_WEIGHT)
    parser.add_argument(
        "--corrupted-gate-weight",
        type=float,
        default=DEFAULT_CORRUPTED_GATE_WEIGHT,
    )
    parser.add_argument("--margin-weight", type=float, default=DEFAULT_MARGIN_WEIGHT)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)

    parser.add_argument(
        "--complete-candidate-graph",
        action="store_true",
        help=(
            "Use the expensive complete temporal candidate graph. Default uses "
            "accepted Trackastra links plus local graph-builder candidates."
        ),
    )
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
        help="Temporal-stage training precision. fp32 is the conservative default.",
    )
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--resume", type=Path, default=None)

    args = parser.parse_args()

    if args.frame_count < 3:
        parser.error("--frame-count must be >= 3")
    if args.temporal_radius < 1:
        parser.error("--temporal-radius must be >= 1")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if args.grad_clip <= 0:
        parser.error("--grad-clip must be positive")
    if args.accumulate_cases < 1:
        parser.error("--accumulate-cases must be positive")
    if args.steps_per_frame_block < 1:
        parser.error("--steps-per-frame-block must be positive")
    if args.eval_every < 1 or args.print_every < 1 or args.val_cases < 1:
        parser.error("evaluation/print counts must be positive")
    if args.synthetic_spatial_logit <= 0:
        parser.error("--synthetic-spatial-logit must be positive")
    if not 0.0 <= args.clean_fraction < 1.0:
        parser.error("--clean-fraction must be in [0,1)")
    if not 0.0 <= args.triple_fraction <= 1.0:
        parser.error("--triple-fraction must be in [0,1]")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be in (0,1)")
    if args.min_voxels < 1:
        parser.error("--min-voxels must be positive")
    if args.max_volume_ratio < 1.0:
        parser.error("--max-volume-ratio must be >= 1")
    if args.max_distance_dref <= 0:
        parser.error("--max-distance-dref must be positive")
    if args.temporal_neighbourhood_dref <= 0 or args.local_edge_radius_dref <= 0:
        parser.error("temporal/local physical radii must be positive")
    if args.max_triples_per_frame < 0:
        parser.error("--max-triples-per-frame cannot be negative")
    if args.preserve_edges < 0 or args.preserve_ratio < 1:
        parser.error("preservation sampling values are invalid")
    if any(
        value < 0
        for value in (
            args.split_weight,
            args.noop_weight,
            args.corrupted_gate_weight,
            args.margin_weight,
            args.margin,
        )
    ):
        parser.error("loss weights / margin cannot be negative")
    if any(float(v) <= 0 for v in args.spacing):
        parser.error("--spacing must contain three positive values")

    return args


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    seed_all(int(args.seed))

    if int(TEMPORAL_CACHE_CONTRACT_VERSION) < 4:
        raise RuntimeError(
            "Investigation 35 requires the finite-window temporal availability "
            "patch (temporal cache contract v4 or newer)."
        )

    paths = make_paths(args)
    print("=" * 118)
    print("INVESTIGATION 35 — BIOHUB CONTROLLED TEMPORAL MERGE TRAINING")
    print("=" * 118)
    print(f"repository : {ROOT}")
    print(f"sample     : {paths.sample}")
    print(f"checkpoint : {paths.checkpoint}")
    print(f"output     : {paths.output}")
    print("=" * 118)

    dref_um, track_graph, temporal_static, manifest = prepare_investigation35(
        args,
        paths,
    )

    if args.prepare_only:
        print("Stopped after --prepare-only. No training was run.")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    if (
        device.type == "cuda"
        and args.amp_dtype == "bf16"
        and hasattr(torch.cuda, "is_bf16_supported")
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "--amp-dtype bf16 was requested but this CUDA device does not support BF16"
        )

    # Always hydrate the architecture from the mature spatial checkpoint.  A
    # resume checkpoint then replaces its complete model state, preserving exact
    # architecture validation while keeping the spatial source of truth explicit.
    _, model = load_spatial_model(paths.checkpoint, torch.device("cpu"))
    resume_checkpoint = None
    if args.resume is not None:
        resume_path = resolve(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        resume_checkpoint = torch_load(resume_path, map_location="cpu")
        architecture = resume_checkpoint.get("architecture")
        if architecture is not None and architecture != "spatial_first_v2":
            raise ValueError(
                f"Resume checkpoint has unexpected architecture: {architecture!r}"
            )
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        extra = dict(resume_checkpoint.get("extra", {}))
        old_investigation = extra.get("investigation")
        if old_investigation not in {None, SCRIPT_NAME}:
            raise ValueError(
                "--resume points to a checkpoint from a different experiment: "
                f"{old_investigation!r}"
            )

    train_model35(
        model,
        paths=paths,
        manifest=manifest,
        temporal_static=temporal_static,
        track_graph=track_graph,
        dref_um=dref_um,
        device=device,
        args=args,
        resume_checkpoint=resume_checkpoint,
    )


if __name__ == "__main__":
    main()
