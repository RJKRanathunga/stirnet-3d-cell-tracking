from __future__ import annotations

r"""
Investigation 37 — real BioHub temporal STIR-Net diagnostic visualization.

Evaluates the Investigation-35 temporal checkpoint on the real frozen spatial
RAG, using Trackastra run on that exact spatial-baseline movie. The script does
not synthesize merges and does not train anything.

Diagnostic component classes:
  FIXED_EXPECTED         spatial baseline wrong -> temporal final exact
  MISSED_WRONG           spatial baseline wrong -> temporal final still wrong
  PRESERVED_CORRECT      spatial baseline correct -> temporal final still exact
  UNEXPECTED_REGRESSION  spatial baseline correct -> temporal final becomes wrong

The available manual annotations are split-correction annotations rather than
an independent complete biological ground truth. Therefore this script is a
same-sample diagnostic against the currently available manual corrections.

Typical usage:
  python .\investigations\stirnet\37_biohub_real_temporal_diagnostic_visualization.py

Re-open cached results:
  python .\investigations\stirnet\37_biohub_real_temporal_diagnostic_visualization.py --viewer-only

Run without Napari:
  python .\investigations\stirnet\37_biohub_real_temporal_diagnostic_visualization.py --no-viewer
"""

import argparse
import dataclasses
import gc
import importlib.util
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

SCRIPT_NAME = "37_biohub_real_temporal_diagnostic_visualization"
DEFAULT_SAMPLE = "44b6_0113de3b"
DEFAULT_FRAME_COUNT = 20
DEFAULT_SPACING = (1.625, 0.40625, 0.40625)
DEFAULT_TEMPORAL_RADIUS = 2

CATEGORY_FIXED = "fixed_expected"
CATEGORY_MISSED = "missed_wrong"
CATEGORY_PRESERVED = "preserved_correct"
CATEGORY_UNEXPECTED = "unexpected_regression"
CATEGORY_CODE = {
    CATEGORY_FIXED: 1,
    CATEGORY_MISSED: 2,
    CATEGORY_PRESERVED: 3,
    CATEGORY_UNEXPECTED: 4,
}


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
    "_inv35_for_inv37",
)
INV30 = load_module(
    ROOT / "investigations" / "stirnet" / "30_biohub_temporal_partition_overfit.py",
    "_inv30_for_inv37",
)

from learned.stirnet.data.graph_builder import (
    build_temporal_graph,
    sequence_available_time_offsets,
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
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


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp)
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


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def parse_spacing(text: str) -> tuple[float, float, float]:
    values = tuple(float(v.strip()) for v in str(text).split(","))
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError("--spacing must contain three positive comma-separated values")
    return values


def positive_count(labels: np.ndarray) -> int:
    values = np.unique(np.asarray(labels))
    return int(np.count_nonzero(values > 0))


@dataclass(frozen=True)
class Paths:
    sample: str
    inv35_root: Path
    output: Path
    checkpoint: Path
    annotations: Path

    @property
    def inv35_cache(self): return self.inv35_root / "cache"
    @property
    def raw_movie(self): return self.inv35_cache / "raw_movie.npy"
    @property
    def manual_movie(self): return self.inv35_cache / "manual_movie.npy"
    @property
    def inv35_spatial(self): return self.inv35_cache / "spatial"
    def graph_cache(self, t): return self.inv35_spatial / f"t{t:03d}" / "frozen_graph.pt"
    def spatial_inputs(self, t): return self.inv35_spatial / f"t{t:03d}" / "spatial_inputs.npy"
    def old_observer_refs(self, t): return self.inv35_spatial / f"t{t:03d}" / "observer_ref_um.npy"
    def old_observer_d1(self, t): return self.inv35_spatial / f"t{t:03d}" / "observer_d1_raw.npy"
    def old_observer_d2(self, t): return self.inv35_spatial / f"t{t:03d}" / "observer_d2_raw.npy"
    def old_observer_hidden(self, t): return self.inv35_spatial / f"t{t:03d}" / "observer_hidden_raw.npy"
    def old_observer_explicit(self, t): return self.inv35_spatial / f"t{t:03d}" / "observer_explicit_raw.npy"
    def manual(self, t): return self.annotations / f"manual_instances_t{t:03d}.npy"
    @property
    def movies(self): return self.output / "movies"
    @property
    def spatial_baseline(self): return self.movies / "spatial_baseline.npy"
    @property
    def temporal_final(self): return self.movies / "temporal_final.npy"
    @property
    def spatial_wrong(self): return self.movies / "spatial_wrong.npy"
    @property
    def temporal_fixed(self): return self.movies / "temporal_fixed.npy"
    @property
    def temporal_missed(self): return self.movies / "temporal_missed.npy"
    @property
    def temporal_preserved(self): return self.movies / "temporal_preserved.npy"
    @property
    def temporal_unexpected(self): return self.movies / "temporal_unexpected.npy"
    @property
    def trackastra(self): return self.output / "trackastra"
    @property
    def track_graph(self): return self.trackastra / "track_graph.pkl"
    @property
    def tracked_masks(self): return self.trackastra / "tracked_masks.npy"
    @property
    def napari_tracks(self): return self.trackastra / "napari_tracks.npy"
    @property
    def temporal_metadata(self): return self.output / "temporal_metadata.pkl"
    @property
    def temporal_graphs(self): return self.output / "temporal_graphs"
    def temporal_graph(self, t): return self.temporal_graphs / f"t{t:03d}.pt"
    @property
    def observer(self): return self.output / "observer"
    def observer_dir(self, t): return self.observer / f"t{t:03d}"
    def observer_refs(self, t): return self.observer_dir(t) / "ref_um.npy"
    def observer_d1(self, t): return self.observer_dir(t) / "d1.npy"
    def observer_d2(self, t): return self.observer_dir(t) / "d2.npy"
    def observer_hidden(self, t): return self.observer_dir(t) / "hidden.npy"
    def observer_explicit(self, t): return self.observer_dir(t) / "explicit.npy"
    def observer_meta(self, t): return self.observer_dir(t) / "meta.json"
    @property
    def cases_csv(self): return self.output / "cases.csv"
    @property
    def edges_csv(self): return self.output / "edge_diagnostics.csv"
    @property
    def frame_metrics_csv(self): return self.output / "frame_metrics.csv"
    @property
    def summary(self): return self.output / "summary.json"
    @property
    def success(self): return self.output / "_SUCCESS.json"


def make_paths(args: argparse.Namespace) -> Paths:
    sample = str(args.sample_id)
    inv35_root = (
        resolve(args.inv35_root)
        if args.inv35_root is not None
        else (ROOT / "runs" / "stirnet" / "evaluation" / "35_biohub_temporal_merge_synthesis" / sample).resolve()
    )
    checkpoint = resolve(args.checkpoint) if args.checkpoint is not None else (inv35_root / "best.pt").resolve()
    output = (
        resolve(args.output)
        if args.output is not None
        else (ROOT / "runs" / "stirnet" / "evaluation" / SCRIPT_NAME / sample).resolve()
    )
    annotations = (
        resolve(args.annotations)
        if args.annotations is not None
        else (ROOT / "evaluation" / "segmentation" / "annotations" / sample).resolve()
    )
    return Paths(sample, inv35_root, output, checkpoint, annotations)


def validate_inputs(paths: Paths, frame_count: int) -> None:
    missing: list[Path] = []
    for path in (paths.raw_movie, paths.manual_movie, paths.checkpoint):
        if not path.is_file(): missing.append(path)
    for t in range(frame_count):
        for path in (paths.graph_cache(t), paths.spatial_inputs(t), paths.manual(t)):
            if not path.is_file(): missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Investigation-37 required artifacts are missing:\n" +
            "\n".join(f"  {path}" for path in missing[:40])
        )


def materialize_partition_labels(rag, partition) -> np.ndarray:
    labels = partition.labels[0]
    if torch.is_tensor(labels) and labels.ndim >= 3 and int(labels.numel()) > 8:
        return labels.detach().cpu().numpy().astype(np.int32, copy=False)
    sv = rag.supervoxel_labels[0].detach().cpu().numpy().astype(np.int64, copy=False)
    node_sv = rag.node_supervoxel_id.detach().cpu().numpy().astype(np.int64, copy=False)
    comp = partition.node_component_global.detach().cpu().numpy().astype(np.int64, copy=False)
    lut = np.zeros(int(sv.max(initial=0)) + 1, dtype=np.int32)
    lut[node_sv] = comp.astype(np.int32) + 1
    return lut[sv]


def build_spatial_baseline_movie(paths: Paths, frame_count: int, rebuild: bool) -> None:
    if paths.spatial_baseline.is_file() and not rebuild:
        movie = np.load(paths.spatial_baseline, mmap_mode="r", allow_pickle=False)
        if int(movie.shape[0]) == frame_count:
            print("[spatial baseline] reuse", flush=True)
            return
    first = torch_load(paths.graph_cache(0))
    first_labels = materialize_partition_labels(first["rag"], first["actual_partition"])
    paths.movies.mkdir(parents=True, exist_ok=True)
    movie = np.lib.format.open_memmap(
        paths.spatial_baseline,
        mode="w+",
        dtype=np.int32,
        shape=(frame_count, *first_labels.shape),
    )
    for t in range(frame_count):
        payload = torch_load(paths.graph_cache(t))
        labels = materialize_partition_labels(payload["rag"], payload["actual_partition"])
        movie[t] = labels
        print(f"[spatial baseline t={t:03d}] instances={positive_count(labels)}", flush=True)
    movie.flush(); del movie


def trackastra_ready(paths: Paths) -> bool:
    return paths.track_graph.is_file() and paths.tracked_masks.is_file() and paths.napari_tracks.is_file()


def run_trackastra(paths: Paths, model_name: str, mode: str, device: str, rebuild: bool) -> None:
    if trackastra_ready(paths) and not rebuild:
        print("[trackastra] reuse", flush=True)
        return
    try:
        from trackastra.model import Trackastra
        from trackastra.tracking.utils import graph_to_napari_tracks
    except ImportError as exc:
        raise RuntimeError("Trackastra is required for Investigation 37") from exc
    paths.trackastra.mkdir(parents=True, exist_ok=True)
    raw = np.load(paths.raw_movie, mmap_mode="r", allow_pickle=False)
    baseline = np.load(paths.spatial_baseline, mmap_mode="r", allow_pickle=False)
    print("[trackastra] running on exact frozen spatial RAG movie ...", flush=True)
    started = time.perf_counter()
    model = Trackastra.from_pretrained(model_name, device=device)
    graph, tracked_masks = model.track(raw, baseline, mode=mode)
    with paths.track_graph.open("wb") as handle: pickle.dump(graph, handle)
    np.save(paths.tracked_masks, np.asarray(tracked_masks), allow_pickle=False)
    tracks, napari_graph, _ = graph_to_napari_tracks(graph)
    np.save(paths.napari_tracks, np.asarray(tracks, np.float64), allow_pickle=False)
    atomic_json(paths.trackastra / "napari_graph.json", {str(int(k)): v for k, v in napari_graph.items()})
    print(f"[trackastra] nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} time={duration(time.perf_counter()-started)}", flush=True)
    del model, tracked_masks
    if torch.cuda.is_available(): torch.cuda.empty_cache()


@dataclass
class TemporalMetadata:
    records: dict[int, Any]
    associations: list[Any]
    nodes_by_time: dict[int, list[int]]


def prepare_temporal_metadata(paths: Paths, spacing, dref_um: float, rebuild: bool) -> TemporalMetadata:
    if paths.temporal_metadata.is_file() and not rebuild:
        with paths.temporal_metadata.open("rb") as handle: payload = pickle.load(handle)
        if abs(float(payload["dref_um"]) - float(dref_um)) < 1e-6:
            print("[temporal metadata] reuse", flush=True)
            return TemporalMetadata(payload["records"], payload["associations"], payload["nodes_by_time"])
    with paths.track_graph.open("rb") as handle: graph = pickle.load(handle)
    tracked = np.load(paths.tracked_masks, mmap_mode="r", allow_pickle=False)
    raw = np.load(paths.raw_movie, mmap_mode="r", allow_pickle=False)
    records = INV30.build_static_detection_records(graph, tracked, raw, tuple(spacing), float(dref_um))
    associations = INV30.trackastra_associations(graph)
    nodes_by_time: dict[int, list[int]] = defaultdict(list)
    for node_id, row in graph.nodes(data=True): nodes_by_time[int(row["time"])].append(int(node_id))
    nodes_by_time = {int(t): sorted(rows) for t, rows in nodes_by_time.items()}
    payload = {"dref_um": float(dref_um), "records": records, "associations": associations, "nodes_by_time": nodes_by_time}
    with paths.temporal_metadata.open("wb") as handle: pickle.dump(payload, handle)
    return TemporalMetadata(records, associations, nodes_by_time)


def build_real_temporal_graph(paths: Paths, metadata: TemporalMetadata, target_t: int, frame_count: int, temporal_radius: int, spacing, dref_um: float, complete_candidate_graph: bool, rebuild: bool) -> dict[str, Any]:
    cache = paths.temporal_graph(target_t)
    if cache.is_file() and not rebuild:
        graph = torch_load(cache)
        if int(graph.get("_inv37_version", 0)) == 1: return graph
    available = sequence_available_time_offsets(target_t, frame_count, temporal_radius)
    records = []
    selected: set[int] = set()
    for dt in available:
        absolute_t = target_t + int(dt)
        for node_id in metadata.nodes_by_time.get(absolute_t, []):
            base = metadata.records.get(int(node_id))
            if base is None: continue
            selected.add(int(node_id))
            records.append(replace(base, time_offset=int(dt)))
    associations = [a for a in metadata.associations if int(a.src_node_id) in selected and int(a.dst_node_id) in selected]
    baseline = np.load(paths.spatial_baseline, mmap_mode="r", allow_pickle=False)
    graph = build_temporal_graph(
        records,
        associations,
        dref_um=float(dref_um),
        temporal_radius=int(temporal_radius),
        available_time_offsets=available,
        k_spatial_neighbors=6,
        spatial_radius_dref=2.5,
        current_labels=np.asarray(baseline[target_t]),
        spacing_um=tuple(float(v) for v in spacing),
        candidate_graph_enabled=bool(complete_candidate_graph),
    )
    graph["temporal_batch"] = torch.zeros(len(graph["temporal_ref_um"]), dtype=torch.long)
    graph["target_time_index"] = int(target_t)
    graph["available_time_offsets"] = torch.tensor(available, dtype=torch.long)
    graph["_inv37_version"] = 1
    paths.temporal_graphs.mkdir(parents=True, exist_ok=True)
    atomic_torch(cache, graph)
    print(f"[temporal graph t={target_t:03d}] detections={len(records)} tracklets={len(graph['temporal_ref_um'])} available={available}", flush=True)
    return graph


def observer_ready(paths: Paths, t: int) -> bool:
    return all(p.is_file() for p in (
        paths.observer_refs(t), paths.observer_d1(t), paths.observer_d2(t),
        paths.observer_hidden(t), paths.observer_explicit(t), paths.observer_meta(t)
    ))


def ref_keys(refs: np.ndarray): return [INV35.reference_key(row) for row in np.asarray(refs, np.float32)]


def observer_matches(paths: Paths, t: int, refs: np.ndarray) -> bool:
    if not observer_ready(paths, t): return False
    try:
        cached = np.load(paths.observer_refs(t), mmap_mode="r", allow_pickle=False)
        return cached.shape == refs.shape and ref_keys(cached) == ref_keys(refs)
    except Exception:
        return False


def load_observer(paths: Paths, t: int, device: torch.device):
    arrays = []
    for p in (paths.observer_refs(t), paths.observer_d1(t), paths.observer_d2(t), paths.observer_hidden(t), paths.observer_explicit(t)):
        arrays.append(np.array(np.load(p, mmap_mode="r", allow_pickle=False), dtype=np.float32, order="C", copy=True))
    refs, d1, d2, hidden, explicit = arrays
    return INV35.ObserverRawLookup(
        ref_um=torch.from_numpy(refs).to(device),
        d1=torch.from_numpy(d1).to(device),
        d2=torch.from_numpy(d2).to(device),
        hidden=torch.from_numpy(hidden).to(device),
        explicit=torch.from_numpy(explicit).to(device),
        key_to_row={INV35.reference_key(ref): i for i, ref in enumerate(refs)},
    )


@torch.inference_mode()
def prepare_observer(paths: Paths, t: int, graph: dict[str, Any], model, spacing, dref_um: float, device: torch.device, amp_dtype: str, rebuild: bool):
    refs = torch.as_tensor(graph["temporal_ref_um"]).detach().cpu().numpy().astype(np.float32, copy=False)
    if not rebuild and observer_matches(paths, t, refs):
        print(f"[observer real t={t:03d}] reuse refs={len(refs)}", flush=True)
        return load_observer(paths, t, device)
    started = time.perf_counter()
    old_paths = (
        paths.old_observer_refs(t), paths.old_observer_d1(t), paths.old_observer_d2(t),
        paths.old_observer_hidden(t), paths.old_observer_explicit(t)
    )
    old_available = all(p.is_file() for p in old_paths)
    if old_available:
        old_refs = np.load(old_paths[0], mmap_mode="r", allow_pickle=False)
        old_features = [np.load(p, mmap_mode="r", allow_pickle=False) for p in old_paths[1:]]
        old_map = {INV35.reference_key(ref): i for i, ref in enumerate(old_refs)}
    else:
        old_features = []
        old_map = {}
    c1 = int(model.cfg.spatial.channels[1]); c2 = int(model.cfg.spatial.channels[2]); ch = int(model.cfg.geometry.hidden_channels)
    d1_out = np.zeros((len(refs), c1), np.float16)
    d2_out = np.zeros((len(refs), c2), np.float16)
    hidden_out = np.zeros((len(refs), ch), np.float16)
    explicit_out = np.zeros((len(refs), 11), np.float16)
    missing = []
    reused = 0
    for row, ref in enumerate(refs):
        src = old_map.get(INV35.reference_key(ref))
        if src is None:
            missing.append(row); continue
        d1_out[row] = np.asarray(old_features[0][src], np.float16)
        d2_out[row] = np.asarray(old_features[1][src], np.float16)
        hidden_out[row] = np.asarray(old_features[2][src], np.float16)
        explicit_out[row] = np.asarray(old_features[3][src], np.float16)
        reused += 1
    if missing:
        missing_refs = refs[np.asarray(missing, np.int64)]
        spatial_np = np.array(np.load(paths.spatial_inputs(t), mmap_mode="r", allow_pickle=False), dtype=np.float32, order="C", copy=True)
        spatial = torch.from_numpy(spatial_np)[None].to(device)
        spacing_t = torch.tensor([spacing], device=device, dtype=torch.float32)
        dref_t = torch.tensor([float(dref_um)], device=device, dtype=torch.float32)
        inference_cfg = dataclasses.replace(model.cfg.inference, mode="tiled", tiled_dense_enabled=True)
        with INV35.autocast_for(device, amp_dtype):
            d1, d2, hidden, explicit = INV35.sample_raw_observer_features(model, spatial, spacing_t, dref_t, missing_refs, config=inference_cfg)
        rows = np.asarray(missing, np.int64)
        d1_out[rows] = d1; d2_out[rows] = d2; hidden_out[rows] = hidden; explicit_out[rows] = explicit
        del spatial
        if device.type == "cuda": torch.cuda.empty_cache()
    paths.observer_dir(t).mkdir(parents=True, exist_ok=True)
    atomic_npy(paths.observer_refs(t), refs)
    atomic_npy(paths.observer_d1(t), d1_out)
    atomic_npy(paths.observer_d2(t), d2_out)
    atomic_npy(paths.observer_hidden(t), hidden_out)
    atomic_npy(paths.observer_explicit(t), explicit_out)
    atomic_json(paths.observer_meta(t), {
        "version": 1, "frame": int(t), "reference_count": int(len(refs)),
        "reused_inv35": int(reused), "sampled_new": int(len(missing)),
        "seconds": float(time.perf_counter()-started), "checkpoint": file_signature(paths.checkpoint)
    })
    print(f"[observer real t={t:03d}] refs={len(refs)} reused_inv35={reused} sampled={len(missing)} time={duration(time.perf_counter()-started)}", flush=True)
    return load_observer(paths, t, device)


@dataclass
class FrameInference:
    rag: Any
    spatial_partition: Any
    temporal_partition: Any
    reasoning: Any
    node_manual: torch.Tensor
    spatial_labels: np.ndarray
    temporal_labels: np.ndarray
    manual_labels: np.ndarray


@torch.inference_mode()
def infer_frame(paths: Paths, t: int, graph, observer, model, spacing, dref_um: float, device: torch.device) -> FrameInference:
    payload = torch_load(paths.graph_cache(t))
    rag = INV35.tree_device_training_float(payload["rag"], device, torch.float32)
    spatial_partition = INV35.tree_device_training_float(payload["actual_partition"], device, torch.float32)
    manual = np.asarray(np.load(paths.manual(t), mmap_mode="r", allow_pickle=False))
    sv = rag.supervoxel_labels[0].detach().cpu().numpy().astype(np.int64, copy=False)
    node_sv = rag.node_supervoxel_id.detach().cpu().numpy().astype(np.int64, copy=False)
    manual_by_sv = INV35.sv_label_lookup(sv, manual, name=f"inv37 t={t} manual")
    node_manual = torch.as_tensor(manual_by_sv[node_sv], device=device, dtype=torch.long)
    decoded, geometry = INV35.dummy_geometry_and_decode(model, rag.node_features)
    spacing_t = torch.tensor([spacing], device=device, dtype=torch.float32)
    dref_t = torch.tensor([float(dref_um)], device=device, dtype=torch.float32)
    instances = model.instance_tokenizer(spatial_partition, rag, decoded, geometry, spacing_t, dref_t, profile_prefix="inv37_tokenizer")
    temporal_input = INV35.temporal_input_from_graph(model, graph, device)
    temporal_base = model.temporal_encoder(temporal_input)
    temporal = INV35.observe_from_raw_lookup(model, temporal_base, observer)
    reasoning = model.instance_temporal(instances, rag, temporal, dref_t)
    temporal_partition = model.partitioner(rag, reasoning.final_edge_logits, model.cfg.partition.final_merge_threshold, stage="final")
    return FrameInference(
        rag, spatial_partition, temporal_partition, reasoning, node_manual,
        materialize_partition_labels(rag, spatial_partition),
        materialize_partition_labels(rag, temporal_partition), manual
    )


def sets_by_key(keys: np.ndarray, values: np.ndarray) -> dict[int, set[int]]:
    result: dict[int, set[int]] = defaultdict(set)
    for k, v in zip(keys.tolist(), values.tolist()):
        if int(k) > 0 and int(v) > 0: result[int(k)].add(int(v))
    return dict(result)


def manual_exact(mid: int, manual_to_comp: dict[int, set[int]], comp_to_manual: dict[int, set[int]]) -> bool:
    comps = manual_to_comp.get(int(mid), set())
    if len(comps) != 1: return False
    comp = next(iter(comps))
    return comp_to_manual.get(comp, set()) == {int(mid)}


def baseline_error_type(component: int, base_to_manual, manual_to_base) -> str:
    mids = base_to_manual.get(component, set())
    if not mids: return "unmapped"
    merge = len(mids) > 1
    split = any(len(manual_to_base.get(mid, set())) > 1 for mid in mids)
    if merge and split: return "mixed_merge_split"
    if merge: return "merge"
    if split: return "split"
    return "correct"


def case_diagnostics(t: int, inference: FrameInference, spacing) -> tuple[list[dict[str, Any]], np.ndarray]:
    rag = inference.rag
    base = inference.spatial_partition.node_component_global.detach().cpu().numpy().astype(np.int64) + 1
    final = inference.temporal_partition.node_component_global.detach().cpu().numpy().astype(np.int64) + 1
    manual = inference.node_manual.detach().cpu().numpy().astype(np.int64)
    base_to_manual = sets_by_key(base, manual); manual_to_base = sets_by_key(manual, base)
    final_to_manual = sets_by_key(final, manual); manual_to_final = sets_by_key(manual, final)
    node_sv = rag.node_supervoxel_id.detach().cpu().numpy().astype(np.int64)
    sv = rag.supervoxel_labels[0].detach().cpu().numpy().astype(np.int64, copy=False)
    counts = np.bincount(sv.reshape(-1), minlength=int(sv.max(initial=0))+1)
    node_weights = counts[node_sv].astype(np.float64)
    centroids = rag.node_centroid_um.detach().float().cpu().numpy()
    shape = np.asarray(sv.shape, np.float32); spacing_np = np.asarray(spacing, np.float32)
    volume_center = 0.5 * (shape-1.0) * spacing_np
    category_node = np.zeros(len(base), np.uint8)
    rows = []
    for component in sorted(int(v) for v in np.unique(base) if int(v) > 0):
        nodes = np.flatnonzero(base == component)
        mids = sorted(base_to_manual.get(component, set()))
        error_type = baseline_error_type(component, base_to_manual, manual_to_base)
        baseline_wrong = error_type != "correct"
        final_exact = bool(mids) and all(manual_exact(mid, manual_to_final, final_to_manual) for mid in mids)
        if baseline_wrong: category = CATEGORY_FIXED if final_exact else CATEGORY_MISSED
        else: category = CATEGORY_PRESERVED if final_exact else CATEGORY_UNEXPECTED
        category_node[nodes] = CATEGORY_CODE[category]
        w = np.maximum(node_weights[nodes], 1.0); w = w / w.sum()
        center_um = np.sum(centroids[nodes].astype(np.float64) * w[:,None], axis=0).astype(np.float32)
        center_vox = (center_um + volume_center) / spacing_np
        final_components = sorted(int(v) for v in np.unique(final[nodes]) if int(v) > 0)
        rows.append({
            "case_id": int(t*100000 + component), "frame": int(t), "spatial_component": int(component),
            "category": category, "baseline_error_type": error_type,
            "manual_ids": "|".join(map(str,mids)), "manual_id_count": len(mids),
            "temporal_components": "|".join(map(str,final_components)), "temporal_component_count": len(final_components),
            "node_count": len(nodes), "voxel_count": int(node_weights[nodes].sum()),
            "center_z": float(center_vox[0]), "center_y": float(center_vox[1]), "center_x": float(center_vox[2]),
            "baseline_wrong": bool(baseline_wrong), "temporal_exact": bool(final_exact),
        })
    lut = np.zeros(int(sv.max(initial=0))+1, np.uint8); lut[node_sv] = category_node
    return rows, lut[sv]


def edge_diagnostics(t: int, inference: FrameInference, spacing) -> list[dict[str, Any]]:
    rag = inference.rag
    src_t, dst_t = rag.edge_index
    src = src_t.detach().cpu().numpy().astype(np.int64); dst = dst_t.detach().cpu().numpy().astype(np.int64)
    manual = inference.node_manual.detach().cpu().numpy().astype(np.int64)
    base = inference.spatial_partition.node_component_global.detach().cpu().numpy().astype(np.int64)
    final = inference.temporal_partition.node_component_global.detach().cpu().numpy().astype(np.int64)
    valid = (manual[src] > 0) & (manual[dst] > 0)
    target_keep = manual[src] == manual[dst]; spatial_keep = base[src] == base[dst]; temporal_keep = final[src] == final[dst]
    spatial_wrong = valid & (spatial_keep != target_keep); final_correct = valid & (temporal_keep == target_keep)
    category = np.full(len(src), "", dtype=object)
    category[spatial_wrong & final_correct] = CATEGORY_FIXED
    category[spatial_wrong & ~final_correct] = CATEGORY_MISSED
    category[~spatial_wrong & valid & final_correct] = CATEGORY_PRESERVED
    category[~spatial_wrong & valid & ~final_correct] = CATEGORY_UNEXPECTED
    centroids = rag.node_centroid_um.detach().float().cpu().numpy(); midpoint_um = 0.5*(centroids[src]+centroids[dst])
    shape = np.asarray(inference.spatial_labels.shape, np.float32); spacing_np = np.asarray(spacing,np.float32)
    midpoint_vox = (midpoint_um + 0.5*(shape-1.0)*spacing_np) / spacing_np
    spatial_logit = rag.spatial_edge_logits.detach().float().cpu().numpy()
    final_logit = inference.reasoning.final_edge_logits.detach().float().cpu().numpy()
    gate = inference.reasoning.edge_temporal_gate.detach().float().cpu().numpy()
    delta = inference.reasoning.edge_temporal_delta.detach().float().cpu().numpy()
    rows=[]
    for edge in np.flatnonzero(valid):
        rows.append({
            "frame":int(t),"edge_index":int(edge),"category":str(category[edge]),
            "src_node":int(src[edge]),"dst_node":int(dst[edge]),"src_manual":int(manual[src[edge]]),"dst_manual":int(manual[dst[edge]]),
            "target_keep":bool(target_keep[edge]),"spatial_keep":bool(spatial_keep[edge]),"temporal_keep":bool(temporal_keep[edge]),
            "spatial_logit":float(spatial_logit[edge]),"temporal_final_logit":float(final_logit[edge]),
            "temporal_gate":float(gate[edge]),"temporal_delta":float(delta[edge]),
            "z":float(midpoint_vox[edge,0]),"y":float(midpoint_vox[edge,1]),"x":float(midpoint_vox[edge,2]),
        })
    return rows


def create_movie(path: Path, dtype, shape):
    path.parent.mkdir(parents=True, exist_ok=True); path.unlink(missing_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def results_complete(paths: Paths, frame_count: int) -> bool:
    required=(paths.temporal_final,paths.spatial_wrong,paths.temporal_fixed,paths.temporal_missed,paths.temporal_preserved,paths.temporal_unexpected,paths.cases_csv,paths.edges_csv,paths.frame_metrics_csv,paths.summary,paths.success)
    if not all(p.is_file() for p in required): return False
    try: return int(np.load(paths.temporal_final,mmap_mode="r").shape[0]) == frame_count
    except Exception: return False


def evaluate(paths: Paths, frame_count: int, spacing, temporal_radius: int, device: torch.device, observer_amp_dtype: str, complete_candidate_graph: bool, rebuild_temporal: bool, rebuild_observer: bool) -> None:
    prep = json.loads((paths.inv35_root / "preparation.json").read_text(encoding="utf-8")); dref_um = float(prep["dref_um"])
    metadata = prepare_temporal_metadata(paths, spacing, dref_um, rebuild_temporal)
    print("[model] loading temporal checkpoint ...", flush=True)
    _, model = INV35.load_spatial_model(paths.checkpoint, device); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    baseline = np.load(paths.spatial_baseline,mmap_mode="r",allow_pickle=False); shape=tuple(int(v) for v in baseline.shape)
    temporal_movie=create_movie(paths.temporal_final,np.int32,shape)
    wrong_movie=create_movie(paths.spatial_wrong,np.uint8,shape)
    fixed_movie=create_movie(paths.temporal_fixed,np.uint8,shape)
    missed_movie=create_movie(paths.temporal_missed,np.uint8,shape)
    preserved_movie=create_movie(paths.temporal_preserved,np.uint8,shape)
    unexpected_movie=create_movie(paths.temporal_unexpected,np.uint8,shape)
    case_rows=[]; edge_rows=[]; frame_rows=[]
    started_all=time.perf_counter()
    print("="*118); print("INVESTIGATION 37 — REAL TEMPORAL RAG EVALUATION"); print("="*118)
    print(f"checkpoint : {paths.checkpoint}\ndref       : {dref_um:.5f} um\nspacing    : {spacing}\nradius     : {temporal_radius}")
    print("="*118)
    for t in range(frame_count):
        started=time.perf_counter()
        graph=build_real_temporal_graph(paths,metadata,t,frame_count,temporal_radius,spacing,dref_um,complete_candidate_graph,rebuild_temporal)
        observer=prepare_observer(paths,t,graph,model,spacing,dref_um,device,observer_amp_dtype,rebuild_observer)
        inf=infer_frame(paths,t,graph,observer,model,spacing,dref_um,device)
        temporal_movie[t]=inf.temporal_labels
        cases,cat=case_diagnostics(t,inf,spacing); edges=edge_diagnostics(t,inf,spacing)
        wrong_movie[t]=np.isin(cat,[CATEGORY_CODE[CATEGORY_FIXED],CATEGORY_CODE[CATEGORY_MISSED]]).astype(np.uint8)
        fixed_movie[t]=(cat==CATEGORY_CODE[CATEGORY_FIXED]).astype(np.uint8)
        missed_movie[t]=(cat==CATEGORY_CODE[CATEGORY_MISSED]).astype(np.uint8)
        preserved_movie[t]=(cat==CATEGORY_CODE[CATEGORY_PRESERVED]).astype(np.uint8)
        unexpected_movie[t]=(cat==CATEGORY_CODE[CATEGORY_UNEXPECTED]).astype(np.uint8)
        case_rows.extend(cases); edge_rows.extend(edges)
        cc=pd.Series([r["category"] for r in cases],dtype="object").value_counts(); ec=pd.Series([r["category"] for r in edges],dtype="object").value_counts()
        row={
            "frame":int(t),"spatial_instances":positive_count(inf.spatial_labels),"temporal_instances":positive_count(inf.temporal_labels),"manual_instances":positive_count(inf.manual_labels),
            "case_fixed_expected":int(cc.get(CATEGORY_FIXED,0)),"case_missed_wrong":int(cc.get(CATEGORY_MISSED,0)),"case_preserved_correct":int(cc.get(CATEGORY_PRESERVED,0)),"case_unexpected_regression":int(cc.get(CATEGORY_UNEXPECTED,0)),
            "edge_fixed_expected":int(ec.get(CATEGORY_FIXED,0)),"edge_missed_wrong":int(ec.get(CATEGORY_MISSED,0)),"edge_preserved_correct":int(ec.get(CATEGORY_PRESERVED,0)),"edge_unexpected_regression":int(ec.get(CATEGORY_UNEXPECTED,0)),
            "mean_temporal_gate":float(inf.reasoning.edge_temporal_gate.detach().float().mean().cpu()) if inf.reasoning.edge_temporal_gate.numel() else 0.0,
            "seconds":float(time.perf_counter()-started),
        }; frame_rows.append(row)
        print(f"[{t+1:02d}/{frame_count:02d} t={t:03d}] spatial={row['spatial_instances']} temporal={row['temporal_instances']} | fixed={row['case_fixed_expected']} missed={row['case_missed_wrong']} unexpected={row['case_unexpected_regression']} preserved={row['case_preserved_correct']} | {duration(row['seconds'])}",flush=True)
        del graph,observer,inf
        if device.type=="cuda": torch.cuda.empty_cache()
        gc.collect()
    for m in (temporal_movie,wrong_movie,fixed_movie,missed_movie,preserved_movie,unexpected_movie): m.flush()
    del temporal_movie,wrong_movie,fixed_movie,missed_movie,preserved_movie,unexpected_movie
    cases_df=pd.DataFrame(case_rows); edges_df=pd.DataFrame(edge_rows); frames_df=pd.DataFrame(frame_rows)
    atomic_csv(paths.cases_csv,cases_df); atomic_csv(paths.edges_csv,edges_df); atomic_csv(paths.frame_metrics_csv,frames_df)
    cc=cases_df["category"].value_counts().to_dict() if not cases_df.empty else {}
    ec=edges_df["category"].value_counts().to_dict() if not edges_df.empty else {}
    wrong=int(cc.get(CATEGORY_FIXED,0)+cc.get(CATEGORY_MISSED,0)); correct=int(cc.get(CATEGORY_PRESERVED,0)+cc.get(CATEGORY_UNEXPECTED,0))
    summary={
        "investigation":SCRIPT_NAME,"sample_id":paths.sample,"frame_count":frame_count,"checkpoint":file_signature(paths.checkpoint),"dref_um":dref_um,"spacing_zyx_um":spacing,"temporal_radius":temporal_radius,"elapsed_seconds":float(time.perf_counter()-started_all),
        "case_counts":{k:int(v) for k,v in cc.items()},"edge_counts":{k:int(v) for k,v in ec.items()},
        "component_level":{
            "baseline_wrong_cases":wrong,"fixed_expected_cases":int(cc.get(CATEGORY_FIXED,0)),"missed_wrong_cases":int(cc.get(CATEGORY_MISSED,0)),
            "baseline_correct_cases":correct,"preserved_correct_cases":int(cc.get(CATEGORY_PRESERVED,0)),"unexpected_regressions":int(cc.get(CATEGORY_UNEXPECTED,0)),
            "wrong_case_fix_rate":float(cc.get(CATEGORY_FIXED,0)/wrong) if wrong else None,
            "correct_case_preservation_rate":float(cc.get(CATEGORY_PRESERVED,0)/correct) if correct else None,
        },
        "notes":{
            "training":False,"synthetic_merge_used_for_inference":False,"real_trackastra_on_exact_spatial_rag_movie":True,
            "manual_annotation_scope":"split-correction annotations; not independent complete GT",
            "temporal_stage":"RAG temporal reasoning + final partition; no source-core split-only postfilter",
        },
    }
    atomic_json(paths.summary,summary); atomic_json(paths.success,{"status":"success","sample_id":paths.sample,"checkpoint":file_signature(paths.checkpoint)})
    print("="*118); print("INVESTIGATION 37 COMPLETE"); print("="*118)
    print(f"spatial wrong: {wrong} | fixed={cc.get(CATEGORY_FIXED,0)} missed={cc.get(CATEGORY_MISSED,0)}")
    print(f"spatial correct: {correct} | preserved={cc.get(CATEGORY_PRESERVED,0)} unexpected={cc.get(CATEGORY_UNEXPECTED,0)}")
    print(f"summary: {paths.summary}"); print("="*118)
    del model
    if device.type=="cuda": torch.cuda.empty_cache()


def add_case_points(viewer, df, category, name, color, scale, visible):
    rows=df.loc[df["category"]==category].copy()
    if rows.empty: return None
    layer=viewer.add_points(
        rows[["frame","center_z","center_y","center_x"]].to_numpy(float),name=name,scale=scale,size=7,face_color=color,border_color="white",border_width=0.15,
        properties={"case_id":rows["case_id"].to_numpy(),"category":rows["category"].astype(str).to_numpy(),"error_type":rows["baseline_error_type"].astype(str).to_numpy(),"manual_ids":rows["manual_ids"].astype(str).to_numpy(),"spatial_component":rows["spatial_component"].to_numpy()},
        text={"string":"{case_id}","size":8,"color":"white","anchor":"upper_left"},
    ); layer.visible=visible; return layer


def add_edge_points(viewer, df, category, name, color, scale, visible):
    rows=df.loc[df["category"]==category].copy()
    if rows.empty: return None
    layer=viewer.add_points(
        rows[["frame","z","y","x"]].to_numpy(float),name=name,scale=scale,size=3.5,face_color=color,border_color="white",border_width=0.1,
        properties={"edge_index":rows["edge_index"].to_numpy(),"category":rows["category"].astype(str).to_numpy(),"target_keep":rows["target_keep"].to_numpy(),"spatial_keep":rows["spatial_keep"].to_numpy(),"temporal_keep":rows["temporal_keep"].to_numpy(),"gate":rows["temporal_gate"].to_numpy(),"delta":rows["temporal_delta"].to_numpy()},
    ); layer.visible=visible; return layer


def attach_case_navigator(viewer, cases: pd.DataFrame, spacing) -> None:
    try:
        from qtpy.QtWidgets import QComboBox,QHBoxLayout,QLabel,QPushButton,QVBoxLayout,QWidget
    except Exception as exc:
        print(f"[viewer warning] case navigator unavailable: {exc}",flush=True); return
    class Navigator(QWidget):
        def __init__(self):
            super().__init__(); self.rows=cases.copy(); self.filtered=pd.DataFrame(); self.index=0
            self.combo=QComboBox(); self.combo.addItems(["Issues only","Fixed expected","Missed wrong","Unexpected regression","Preserved correct","All cases"])
            self.prev=QPushButton("Previous"); self.nextb=QPushButton("Next"); self.info=QLabel(""); self.info.setWordWrap(True)
            buttons=QHBoxLayout(); buttons.addWidget(self.prev); buttons.addWidget(self.nextb)
            layout=QVBoxLayout(); layout.addWidget(self.combo); layout.addLayout(buttons); layout.addWidget(self.info); self.setLayout(layout)
            self.combo.currentIndexChanged.connect(self.refilter); self.prev.clicked.connect(self.previous); self.nextb.clicked.connect(self.next); self.refilter()
        def refilter(self):
            choice=self.combo.currentText()
            if choice=="Issues only": mask=self.rows["category"].isin([CATEGORY_FIXED,CATEGORY_MISSED,CATEGORY_UNEXPECTED])
            elif choice=="Fixed expected": mask=self.rows["category"]==CATEGORY_FIXED
            elif choice=="Missed wrong": mask=self.rows["category"]==CATEGORY_MISSED
            elif choice=="Unexpected regression": mask=self.rows["category"]==CATEGORY_UNEXPECTED
            elif choice=="Preserved correct": mask=self.rows["category"]==CATEGORY_PRESERVED
            else: mask=np.ones(len(self.rows),dtype=bool)
            self.filtered=self.rows.loc[mask].sort_values(["frame","case_id"]).reset_index(drop=True); self.index=0; self.show_current()
        def show_current(self):
            if self.filtered.empty: self.info.setText("No cases in this category."); return
            self.index%=len(self.filtered); row=self.filtered.iloc[self.index]; frame=int(row["frame"]); viewer.dims.set_current_step(0,frame)
            try: viewer.camera.center=(float(row["center_z"])*spacing[0],float(row["center_y"])*spacing[1],float(row["center_x"])*spacing[2])
            except Exception: pass
            self.info.setText(f"{self.index+1}/{len(self.filtered)}\ncase={int(row['case_id'])} frame={frame}\ncategory={row['category']}\nbaseline error={row['baseline_error_type']}\nspatial component={int(row['spatial_component'])}\nmanual IDs={row['manual_ids']}\ntemporal components={row['temporal_components']}")
        def previous(self):
            if self.filtered.empty:return
            self.index=(self.index-1)%len(self.filtered); self.show_current()
        def next(self):
            if self.filtered.empty:return
            self.index=(self.index+1)%len(self.filtered); self.show_current()
    widget=Navigator(); viewer.window.add_dock_widget(widget,area="right",name="Case Navigator"); viewer._inv37_navigator=widget


def open_viewer(paths: Paths, spacing) -> None:
    try: import napari
    except ImportError as exc: raise RuntimeError("Napari is required for Investigation 37") from exc
    raw=np.load(paths.raw_movie,mmap_mode="r",allow_pickle=False); manual=np.load(paths.manual_movie,mmap_mode="r",allow_pickle=False)
    spatial=np.load(paths.spatial_baseline,mmap_mode="r",allow_pickle=False); temporal=np.load(paths.temporal_final,mmap_mode="r",allow_pickle=False)
    tracked=np.load(paths.tracked_masks,mmap_mode="r",allow_pickle=False); wrong=np.load(paths.spatial_wrong,mmap_mode="r",allow_pickle=False)
    fixed=np.load(paths.temporal_fixed,mmap_mode="r",allow_pickle=False); missed=np.load(paths.temporal_missed,mmap_mode="r",allow_pickle=False)
    preserved=np.load(paths.temporal_preserved,mmap_mode="r",allow_pickle=False); unexpected=np.load(paths.temporal_unexpected,mmap_mode="r",allow_pickle=False)
    cases=pd.read_csv(paths.cases_csv); edges=pd.read_csv(paths.edges_csv); tracks=np.load(paths.napari_tracks,mmap_mode="r",allow_pickle=False)
    scale=(1.0,*spacing); sample=np.asarray(raw[:,::2,::4,::4]); low,high=np.percentile(sample,[1.0,99.8])
    viewer=napari.Viewer(ndisplay=3)
    viewer.add_image(raw,name="Raw Volume",scale=scale,rendering="mip",colormap="gray",contrast_limits=[float(low),float(high)])
    viewer.add_labels(manual,name="Manual Annotations",scale=scale,opacity=1.0,visible=False)
    viewer.add_labels(spatial,name="Spatial Baseline RAG",scale=scale,opacity=1.0,visible=False)
    viewer.add_labels(temporal,name="Temporal Final RAG",scale=scale,opacity=1.0,visible=True)
    viewer.add_labels(tracked,name="Trackastra Tracked Masks",scale=scale,opacity=1.0,visible=False)
    viewer.add_labels(wrong,name="Spatial Wrong Before Temporal",scale=scale,opacity=0.80,visible=False)
    viewer.add_labels(fixed,name="Temporal Fixed Expected",scale=scale,opacity=0.90,visible=True)
    viewer.add_labels(missed,name="Temporal Missed Wrong",scale=scale,opacity=0.90,visible=True)
    viewer.add_labels(preserved,name="Temporal Preserved Correct",scale=scale,opacity=0.65,visible=False)
    viewer.add_labels(unexpected,name="Temporal Unexpected Regression",scale=scale,opacity=0.95,visible=True)
    if tracks.ndim==2 and tracks.shape[1]==5 and len(tracks):
        layer=viewer.add_tracks(tracks,name="Tracks - Spatial Baseline",scale=scale,tail_length=20); layer.visible=False
    add_case_points(viewer,cases,CATEGORY_FIXED,"Case Centers - Fixed","lime",scale,True)
    add_case_points(viewer,cases,CATEGORY_MISSED,"Case Centers - Missed","magenta",scale,True)
    add_case_points(viewer,cases,CATEGORY_UNEXPECTED,"Case Centers - Unexpected","red",scale,True)
    add_case_points(viewer,cases,CATEGORY_PRESERVED,"Case Centers - Preserved","cyan",scale,False)
    if not edges.empty:
        add_edge_points(viewer,edges,CATEGORY_FIXED,"Edge Fixes - Expected","lime",scale,False)
        add_edge_points(viewer,edges,CATEGORY_MISSED,"Edge Misses - Wrong","magenta",scale,True)
        add_edge_points(viewer,edges,CATEGORY_UNEXPECTED,"Edge Regressions - Unexpected","red",scale,True)
        add_edge_points(viewer,edges,CATEGORY_PRESERVED,"Edges - Preserved Correct","cyan",scale,False)
    attach_case_navigator(viewer,cases,spacing)
    summary=json.loads(paths.summary.read_text(encoding="utf-8")); c=summary["component_level"]
    print(f"[viewer] spatial wrong={c['baseline_wrong_cases']} fixed={c['fixed_expected_cases']} missed={c['missed_wrong_cases']} | spatial correct={c['baseline_correct_cases']} preserved={c['preserved_correct_cases']} unexpected={c['unexpected_regressions']}",flush=True)
    napari.run()


def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Real BioHub temporal STIR-Net diagnostic visualization")
    p.add_argument("--sample-id",default=DEFAULT_SAMPLE); p.add_argument("--frame-count",type=int,default=DEFAULT_FRAME_COUNT); p.add_argument("--spacing",default="1.625,0.40625,0.40625")
    p.add_argument("--inv35-root",type=Path,default=None); p.add_argument("--checkpoint",type=Path,default=None); p.add_argument("--annotations",type=Path,default=None); p.add_argument("--output",type=Path,default=None)
    p.add_argument("--temporal-radius",type=int,default=DEFAULT_TEMPORAL_RADIUS); p.add_argument("--complete-candidate-graph",action="store_true")
    p.add_argument("--device",default=("cuda" if torch.cuda.is_available() else "cpu")); p.add_argument("--observer-amp-dtype",choices=("fp32","fp16","bf16"),default=("fp16" if torch.cuda.is_available() else "fp32"))
    p.add_argument("--trackastra-model",default="ctc"); p.add_argument("--trackastra-mode",default="greedy"); p.add_argument("--trackastra-device",default="cuda")
    p.add_argument("--rebuild-spatial-baseline",action="store_true"); p.add_argument("--rebuild-trackastra",action="store_true"); p.add_argument("--rebuild-temporal",action="store_true"); p.add_argument("--rebuild-observer",action="store_true")
    p.add_argument("--viewer-only",action="store_true"); p.add_argument("--no-viewer",action="store_true")
    return p


def main() -> int:
    args=build_parser().parse_args(); spacing=parse_spacing(args.spacing); paths=make_paths(args); validate_inputs(paths,args.frame_count); paths.output.mkdir(parents=True,exist_ok=True)
    if args.viewer_only:
        if not results_complete(paths,args.frame_count): raise FileNotFoundError(f"Incomplete Investigation-37 results below {paths.output}")
        if not trackastra_ready(paths): raise FileNotFoundError(f"Incomplete Trackastra cache below {paths.trackastra}")
        open_viewer(paths,spacing); return 0
    build_spatial_baseline_movie(paths,args.frame_count,bool(args.rebuild_spatial_baseline))
    run_trackastra(paths,str(args.trackastra_model),str(args.trackastra_mode),str(args.trackastra_device),bool(args.rebuild_trackastra or args.rebuild_spatial_baseline))
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    evaluate(paths,args.frame_count,spacing,args.temporal_radius,device,str(args.observer_amp_dtype),bool(args.complete_candidate_graph),bool(args.rebuild_temporal or args.rebuild_trackastra or args.rebuild_spatial_baseline),bool(args.rebuild_observer))
    if not args.no_viewer: open_viewer(paths,spacing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
