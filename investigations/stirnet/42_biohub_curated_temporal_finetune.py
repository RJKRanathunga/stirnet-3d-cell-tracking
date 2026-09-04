from __future__ import annotations

r"""
Investigation 42 — BioHub curated real temporal fine-tuning.

Scientific question
-------------------
Given the ORIGINAL frozen STIR-Net spatial RAG and the ORIGINAL Trackastra graph,
can STIR-Net's temporal stack learn to repair REAL manually annotated spatial
merges without introducing false splits into trusted correctly segmented cells?

This experiment is deliberately split-only:

    current A+B  -> temporal may produce A | B
    current A|B  -> temporal may NOT produce A+B

Data roles
----------
INPUT:
    * raw/frozen spatial model state from canonical dataset-curation inference
    * original Trackastra graph from canonical dataset-curation inference

TARGET:
    * manually corrected instance frame when manual_instances_tXXX.npy exists
    * otherwise the canonical final_instances frame, but ONLY for frames that
      are explicitly declared reviewed by --reviewed-frames

EXCLUDED:
    * every cell referenced by an Ignore event in track_annotations.json
    * every RAG edge touching an ignored/invalid target node
    * hallucinated/background target nodes

The manually corrected tracking graph (Continue/Break/Birth) is NOT used as
model input. Temporal evidence remains the original Trackastra graph so train
and inference conditions match.

Implementation strategy
-----------------------
Investigation 35 already contains the current direct Trackastra -> TemporalInput
adapter, compact frozen-spatial cache format, InstanceTokenizer-compatible
runtime, checkpoint hydration, and several safe utilities. Investigation 42
reuses those stable helpers, but replaces synthetic merges with real curated
CUT/KEEP targets.

Before training, the script:
    1. verifies annotation/inference binding,
    2. resolves the exact spatial checkpoint recorded by spatial_summary.json,
    3. rebuilds/reuses a compact frozen RAG cache directly on the persisted
       curation supervoxels (watershed is never replayed),
    4. requires cached atomic supervoxels to match the persisted curation
       supervoxels EXACTLY,
    5. matches original Trackastra detections directly to persisted final-instance centroids,
    6. resolves Ignore cells against corrected reviewed targets,
    7. audits train/validation CUT and KEEP edges.

Typical commands
----------------
Audit + prepare only:

    python .\investigations\stirnet\42_biohub_curated_temporal_finetune.py --audit-only

Small real-data overfit/smoke run:

    python .\investigations\stirnet\42_biohub_curated_temporal_finetune.py `
        --steps 300 `
        --eval-every 50 `
        --print-every 10

First full run:

    python .\investigations\stirnet\42_biohub_curated_temporal_finetune.py `
        --steps 1500 `
        --eval-every 100 `
        --print-every 10

Defaults are intentionally specific to the currently curated volume:
    sample          : train/44b6_0113de3b
    reviewed frames : 0-39
    train targets   : 2-25
    validation      : 30-39
    temporal radius : 2

The 26-29 guard interval prevents train/validation temporal windows from
sharing frames when radius=2.
"""

import argparse
import dataclasses
import gc
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


# =============================================================================
# Repository / current code
# =============================================================================


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents, Path.cwd().resolve()):
        if (
            (candidate / "learned" / "stirnet").is_dir()
            and (candidate / "dataset_curation").is_dir()
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
    "_inv42_inv35_helpers",
)

from dataset_curation.catalog import BioHubCatalog
from dataset_curation.config import DEFAULT_SPACING_ZYX_UM
from dataset_curation.inference.backends.stirnet_trackastra import (
    resolve_default_checkpoint,
)
from learned.stirnet.model.types import PartitionState, RAGState, TemporalInput
from learned.stirnet.inference import SpatialInferenceConfig, load_spatial_runtime
from learned.stirnet.inference.spatial_input import (
    canonical_source_segmentation_config,
    prepare_spatial_frame,
)
from learned.stirnet.inference.tiled_dense import (
    _spatial_from_streamed_stats,
    stream_tiled_label_feature_stats,
    tiled_dense_geometry,
)
from learned.stirnet.training import TrainingConfig
from learned.stirnet.training.checkpoint import save_checkpoint
from learned.stirnet.training.temporal_causal import (
    causal_temporal_loss_terms,
    corrupt_temporal_state,
)
from src.io import load_timepoint


SCRIPT_NAME = "42_biohub_curated_temporal_finetune"
OBJECTIVE_VERSION = 4
DEFAULT_SAMPLE = "44b6_0113de3b"
DEFAULT_SPLIT = "train"
DEFAULT_ANNOTATION_SET = "main"
DEFAULT_REVIEWED = "0-39"
DEFAULT_TRAIN = "2-25"
DEFAULT_VAL = "30-39"
DEFAULT_TEMPORAL_RADIUS = 2
DEFAULT_STEPS = 1500
DEFAULT_LR = 2.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_ACCUMULATE = 4
DEFAULT_PRESERVE_EDGES = 512
DEFAULT_PRESERVE_RATIO = 12
DEFAULT_PRESERVATION_WEIGHT = 2.0
DEFAULT_CANDIDATE_WEIGHT = 0.0
DEFAULT_CANDIDATE_PRESERVATION_WEIGHT = 1.0
DEFAULT_GATE_HELP_WEIGHT = 1.0
DEFAULT_GATE_SUPPRESS_WEIGHT = 1.0
DEFAULT_GATE_CORRECTION_SCALE = 4.0
DEFAULT_SPLIT_WEIGHT = 0.05
DEFAULT_CUT_FRAME_PROBABILITY = 0.80
DEFAULT_EVAL_EVERY = 100
DEFAULT_PRINT_EVERY = 10
DEFAULT_SEED = 20260903


# =============================================================================
# Generic helpers
# =============================================================================


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def atomic_json(path: Path, payload: Any) -> None:
    INV35.atomic_json(path, payload)


def torch_load(path: Path, map_location="cpu") -> Any:
    return INV35.torch_load(path, map_location=map_location)


def duration(seconds: float) -> str:
    return INV35.duration(seconds)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_frame_spec(text: str) -> tuple[int, ...]:
    """Parse `0-4,7,10-12` into sorted unique frame IDs."""
    values: set[int] = set()
    for raw in str(text).split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            a, b = int(left), int(right)
            if b < a:
                raise ValueError(f"Invalid descending frame range: {token!r}")
            values.update(range(a, b + 1))
        else:
            values.add(int(token))
    if not values:
        raise ValueError("Frame specification is empty")
    if min(values) < 0:
        raise ValueError("Frame IDs must be non-negative")
    return tuple(sorted(values))


def parse_spacing(text: str) -> tuple[float, float, float]:
    values = tuple(float(v.strip()) for v in str(text).split(","))
    if len(values) != 3 or any(v <= 0 or not math.isfinite(v) for v in values):
        raise ValueError("--spacing must contain three positive finite Z,Y,X values")
    return values


def json_load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def json_load_list(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Expected JSON array in {path}")
    return payload


def sampled(index: Tensor, maximum: int, rng: random.Random) -> Tensor:
    if int(index.numel()) <= int(maximum):
        return index
    rows = rng.sample(range(int(index.numel())), int(maximum))
    return index[torch.tensor(rows, device=index.device, dtype=torch.long)]


def bool_mean(value: Tensor) -> float:
    return float(value.float().mean().detach().cpu()) if value.numel() else 0.0


# =============================================================================
# Paths / provenance
# =============================================================================


@dataclass(frozen=True)
class Paths:
    sample: str
    split: str
    annotation_set: str
    data_root: Path
    zarr: Path
    preprocessed: Path
    supervoxels: Path
    base_instances: Path
    cells_csv: Path
    tracks_csv: Path
    track_graph: Path
    spatial_summary: Path
    inference_manifest: Path
    annotation_root: Path
    instance_annotations: Path
    track_annotations: Path
    annotation_manifest: Path
    output: Path
    spatial_cache: Path
    raw_prepare_movie: Path
    checkpoint: Path
    checkpoint_sha256: str
    checkpoint_step: int
    frame_count: int

    def graph_cache(self, t: int) -> Path:
        return self.spatial_cache / f"t{int(t):03d}" / "frozen_graph.pt"

    def spatial_meta(self, t: int) -> Path:
        return self.spatial_cache / f"t{int(t):03d}" / "meta.json"

    def manual_frame(self, t: int) -> Path:
        return self.instance_annotations / f"manual_instances_t{int(t):03d}.npy"

    @property
    def spatial_operations(self) -> Path:
        return self.instance_annotations / "spatial_operations.json"

    @property
    def track_state(self) -> Path:
        return self.track_annotations / "track_annotations.json"

    @property
    def manifest(self) -> Path:
        return self.output / "dataset_manifest.json"

    @property
    def audit_json(self) -> Path:
        return self.output / "audit.json"

    @property
    def training_history(self) -> Path:
        return self.output / "training_history.json"

    @property
    def validation_history(self) -> Path:
        return self.output / "validation_history.json"

    @property
    def baseline_metrics(self) -> Path:
        return self.output / "baseline_metrics.json"

    @property
    def initial_temporal_metrics(self) -> Path:
        return self.output / "initial_temporal_metrics.json"

    @property
    def best_metrics(self) -> Path:
        return self.output / "best_metrics.json"

    @property
    def latest(self) -> Path:
        return self.output / "latest.pt"

    @property
    def best(self) -> Path:
        return self.output / "best.pt"

    @property
    def best_safe(self) -> Path:
        return self.output / "best_safe.pt"

    @property
    def best_safe_metrics(self) -> Path:
        return self.output / "best_safe_metrics.json"

    @property
    def final(self) -> Path:
        return self.output / "final.pt"


def resolve_checkpoint_from_curation(record_paths, override: Path | None) -> tuple[Path, str, int]:
    if not record_paths.spatial_summary.is_file():
        raise FileNotFoundError(record_paths.spatial_summary)

    summary = json_load(record_paths.spatial_summary)
    expected_sha = str(summary.get("checkpoint_sha256", "")).strip()
    expected_step = int(summary.get("checkpoint_step", -1))
    recorded = Path(str(summary.get("checkpoint", ""))).expanduser()

    candidates: list[Path] = []
    if override is not None:
        candidates.append(resolve(override))
    if str(recorded):
        candidates.append(recorded)
    try:
        candidates.append(resolve_default_checkpoint())
    except Exception:
        pass

    seen: set[Path] = set()
    existing: list[Path] = []
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            existing.append(candidate)

    if not existing:
        raise FileNotFoundError(
            "Could not locate the spatial checkpoint recorded by curation. "
            "Pass --checkpoint explicitly."
        )

    for candidate in existing:
        actual_sha = sha256(candidate)
        if expected_sha and actual_sha != expected_sha:
            if override is not None and candidate == resolve(override):
                raise RuntimeError(
                    "--checkpoint does not match the checkpoint SHA used by the "
                    f"annotated inference.\nexpected={expected_sha}\nactual  ={actual_sha}\n"
                    f"path    ={candidate}"
                )
            continue
        return candidate, actual_sha, expected_step

    raise RuntimeError(
        "No available checkpoint matches spatial_summary.json. "
        f"Expected SHA-256: {expected_sha or '<missing>'}"
    )


def make_paths(args: argparse.Namespace) -> Paths:
    catalog = BioHubCatalog(args.data_root)
    record = catalog.get(str(args.sample_id), split=str(args.split))
    if record.frame_count is None:
        raise RuntimeError(f"Could not determine frame count for {record.volume_id}")
    if not record.paths.inference_complete(frame_count=record.frame_count):
        raise RuntimeError(
            f"Canonical inference is incomplete for {record.split}/{record.volume_id}"
        )

    annotation_root = record.paths.annotation_set(str(args.annotation_set))
    annotation_manifest = record.paths.annotation_manifest(str(args.annotation_set))
    if not annotation_manifest.is_file():
        raise FileNotFoundError(annotation_manifest)

    ann = json_load(annotation_manifest)
    inference_id = record.paths.inference_id()
    if str(ann.get("base_inference_id", "")) != str(inference_id):
        raise RuntimeError(
            "Annotation set is not bound to the current canonical inference: "
            f"annotation={ann.get('base_inference_id')!r}, current={inference_id!r}"
        )

    checkpoint, checkpoint_hash, checkpoint_step = resolve_checkpoint_from_curation(
        record.paths,
        args.checkpoint,
    )

    output = (
        resolve(args.output)
        if args.output is not None
        else (
            ROOT
            / "runs"
            / "stirnet"
            / "investigations"
            / SCRIPT_NAME
            / record.volume_id
        ).resolve()
    )
    spatial_cache = (
        resolve(args.spatial_cache_root)
        if args.spatial_cache_root is not None
        else (output / "spatial_cache").resolve()
    )

    return Paths(
        sample=record.volume_id,
        split=record.split,
        annotation_set=str(args.annotation_set),
        data_root=record.paths.data_root,
        zarr=record.paths.zarr,
        preprocessed=record.paths.preprocessed_root,
        supervoxels=record.paths.supervoxels,
        base_instances=record.paths.final_instances,
        cells_csv=record.paths.cells_csv,
        tracks_csv=record.paths.tracks_csv,
        track_graph=record.paths.track_graph,
        spatial_summary=record.paths.spatial_summary,
        inference_manifest=record.paths.inference_manifest,
        annotation_root=annotation_root,
        instance_annotations=record.paths.instance_annotations(str(args.annotation_set)),
        track_annotations=record.paths.track_annotations(str(args.annotation_set)),
        annotation_manifest=annotation_manifest,
        output=output,
        spatial_cache=spatial_cache,
        raw_prepare_movie=output / "_prepare" / "raw_reviewed_prefix.npy",
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        checkpoint_step=checkpoint_step,
        frame_count=int(record.frame_count),
    )


# =============================================================================
# Reviewed target / Ignore state
# =============================================================================


class TargetFrameStore:
    def __init__(
        self,
        paths: Paths,
        reviewed_frames: Iterable[int],
        ignored_ids: dict[int, set[int]],
    ) -> None:
        self.paths = paths
        self.reviewed = set(int(v) for v in reviewed_frames)
        self.ignored_ids = {int(k): set(map(int, v)) for k, v in ignored_ids.items()}
        self.base = np.load(paths.base_instances, mmap_mode="r", allow_pickle=False)
        self._current_t: int | None = None
        self._current: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.base.shape)

    def frame(self, t: int) -> np.ndarray:
        t = int(t)
        if t not in self.reviewed:
            raise RuntimeError(f"Frame {t} is not declared reviewed")
        if self._current_t == t and self._current is not None:
            return self._current
        manual = self.paths.manual_frame(t)
        if manual.is_file():
            data = np.load(manual, mmap_mode="r", allow_pickle=False)
        else:
            data = self.base[t]
        data = np.asarray(data)
        if data.shape != self.base.shape[1:]:
            raise RuntimeError(
                f"Target frame t={t} has shape {data.shape}; expected {self.base.shape[1:]}"
            )
        self._current_t = t
        self._current = data
        return data

    def ignored_for_frame(self, t: int) -> set[int]:
        return set(self.ignored_ids.get(int(t), set()))


def load_ignored_ids(paths: Paths, reviewed_frames: set[int]) -> tuple[dict[int, set[int]], list[dict[str, Any]]]:
    if not paths.track_state.is_file():
        return {}, []
    payload = json_load(paths.track_state)
    events = list(payload.get("ignored_events", []))
    by_frame: dict[int, set[int]] = defaultdict(set)
    normalized: list[dict[str, Any]] = []

    def add_node(raw: Any, source: str, event_index: int) -> None:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return
        frame, cell_id = int(raw[0]), int(raw[1])
        if frame not in reviewed_frames or cell_id <= 0:
            return
        by_frame[frame].add(cell_id)
        normalized.append(
            {
                "event_index": int(event_index),
                "source": source,
                "frame": frame,
                "cell_id": cell_id,
            }
        )

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        add_node(event.get("endpoint"), "endpoint", index)
        add_node(event.get("selected_node"), "selected_node", index)

    return dict(by_frame), normalized


def validate_ignored_ids(store: TargetFrameStore) -> dict[str, Any]:
    missing: list[tuple[int, int]] = []
    resolved = 0
    for frame, ids in sorted(store.ignored_ids.items()):
        if frame not in store.reviewed:
            continue
        target = store.frame(frame)
        present = set(int(v) for v in np.unique(target).tolist() if int(v) > 0)
        for cell_id in sorted(ids):
            if cell_id in present:
                resolved += 1
            else:
                missing.append((frame, cell_id))
    if missing:
        preview = ", ".join(f"t={t}:id={i}" for t, i in missing[:20])
        raise RuntimeError(
            "Ignore records could not be resolved against the current corrected "
            f"targets. Refine/review them before training. Examples: {preview}"
        )
    return {
        "ignored_unique_cells": int(sum(len(v) for v in store.ignored_ids.values())),
        "ignored_resolved_cells": int(resolved),
        "ignored_unresolved_cells": 0,
    }


def load_spatial_operations(paths: Paths) -> list[dict[str, Any]]:
    if not paths.spatial_operations.is_file():
        return []
    payload = json_load(paths.spatial_operations)
    return [row for row in payload.get("operations", []) if isinstance(row, dict)]


# =============================================================================
# Frozen spatial cache — persisted curation supervoxels are authoritative
# =============================================================================

# The annotation was created from the persisted atomic supervoxels below
# E:/data/biohub/preprocessed/.../movies/supervoxels.npy.  Re-running watershed
# is NOT a valid way to recover that atomic graph because repository-side
# non-parametric proposal logic may change after inference (for example the
# September-03 tiny-supervoxel agglomeration default).  Investigation 42
# therefore recomputes only dense/model features and rebuilds the RAG ON TOP OF
# the persisted labels.  The persisted final_instances movie similarly defines
# the frozen current partition that the human reviewed.
CACHE_CONTRACT = "persisted_curation_supervoxels_v2"
CACHE_VERSION = 2


def _cache_frame_compatible(paths: Paths, t: int) -> bool:
    graph_path = paths.graph_cache(t)
    meta_path = paths.spatial_meta(t)
    if not graph_path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json_load(meta_path)
        return (
            int(meta.get("version", -1)) == CACHE_VERSION
            and str(meta.get("cache_contract", "")) == CACHE_CONTRACT
            and str(meta.get("checkpoint_sha256", "")) == paths.checkpoint_sha256
            and int(meta.get("checkpoint_step", -1)) == int(paths.checkpoint_step)
            and str(meta.get("source_supervoxels", ""))
            == str(paths.supervoxels.resolve())
            and str(meta.get("source_final_instances", ""))
            == str(paths.base_instances.resolve())
        )
    except Exception:
        return False


def cache_complete(paths: Paths, frames: Iterable[int]) -> bool:
    return all(_cache_frame_compatible(paths, int(t)) for t in frames)


def _tiny_partition_labels(
    node_component: Tensor,
    *,
    edge_logits: Tensor,
) -> PartitionState:
    """Build the compact PartitionState used by InstanceTokenizer statistics."""
    if node_component.numel():
        component_count = int(node_component.max().item()) + 1
    else:
        component_count = 0

    labels = (
        torch.arange(
            1,
            component_count + 1,
            device=node_component.device,
            dtype=torch.long,
        ).reshape(1, 1, -1)
        if component_count
        else torch.zeros((1, 1, 1), device=node_component.device, dtype=torch.long)
    )
    return PartitionState(
        labels=[labels],
        node_component=node_component,
        node_component_global=node_component.clone(),
        component_count_per_batch=torch.tensor(
            [component_count],
            device=node_component.device,
            dtype=torch.long,
        ),
        edge_logits=edge_logits,
    )


def partition_from_persisted_final(
    rag: RAGState,
    *,
    supervoxels_zyx: np.ndarray,
    final_instances_zyx: np.ndarray,
) -> tuple[PartitionState, dict[str, int]]:
    """Recover the exact reviewed current partition from persisted labels.

    A positive atomic supervoxel must not be split across two persisted final
    instance IDs.  Whole supervoxels that were removed by a downstream
    existence filter may map to background; those are conservatively assigned
    independent unsupervised components rather than being merged into a trusted
    cell.
    """
    lookup = INV35.sv_label_lookup(
        np.asarray(supervoxels_zyx, dtype=np.int64),
        np.asarray(final_instances_zyx, dtype=np.int64),
        name="persisted final-instance partition",
    )
    node_sv = (
        rag.node_supervoxel_id.detach().cpu().numpy().astype(np.int64, copy=False)
    )
    if node_sv.size and int(node_sv.max()) >= int(lookup.shape[0]):
        raise RuntimeError(
            "RAG node_supervoxel_id exceeds the persisted supervoxel lookup."
        )
    node_final = lookup[node_sv] if node_sv.size else np.zeros((0,), np.int64)

    key_to_component: dict[tuple[str, int], int] = {}
    components: list[int] = []
    positive_final_ids: set[int] = set()
    background_nodes = 0

    for sv_id, final_id in zip(node_sv.tolist(), node_final.tolist()):
        final_id = int(final_id)
        if final_id > 0:
            key = ("final", final_id)
            positive_final_ids.add(final_id)
        else:
            # This node is not a trusted current cell.  Keep it isolated so the
            # split-only temporal experiment cannot accidentally use it as a
            # must-link bridge between reviewed cells.
            key = ("background_sv", int(sv_id))
            background_nodes += 1
        component = key_to_component.get(key)
        if component is None:
            component = len(key_to_component)
            key_to_component[key] = component
        components.append(int(component))

    node_component = torch.tensor(
        components,
        device=rag.node_features.device,
        dtype=torch.long,
    )
    partition = _tiny_partition_labels(
        node_component,
        edge_logits=rag.spatial_edge_logits,
    )
    return partition, {
        "persisted_final_instance_count": int(len(positive_final_ids)),
        "partition_component_count": int(len(key_to_component)),
        "background_rag_node_count": int(background_nodes),
    }


def _cache_amp_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext(), "fp32"
    if torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16), "bf16"
    return torch.autocast("cuda", dtype=torch.float16), "fp16"


def prepare_spatial_cache(
    paths: Paths,
    *,
    frames: tuple[int, ...],
    spacing: tuple[float, float, float],
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Build compact RAG/statistics on the EXACT persisted curation atoms.

    Dense geometry and D0/D1/D2 features are recomputed from the recorded
    checkpoint, but watershed is intentionally bypassed.  This makes the cache
    stable against later changes to non-parametric proposal code while keeping
    the learned spatial evidence used by InstanceTokenizer/RAGNetwork.
    """
    requested = tuple(sorted(set(int(t) for t in frames)))
    force = bool(args.rebuild_spatial_cache)
    todo = [
        t for t in requested
        if force or not _cache_frame_compatible(paths, t)
    ]
    if not todo:
        print("[spatial cache] requested persisted-label frames already exist", flush=True)
        return

    tile_shape = INV35.parse_zyx_ints(args.tile_shape_zyx, name="--tile-shape-zyx")
    tile_overlap = INV35.parse_zyx_ints(args.tile_overlap_zyx, name="--tile-overlap-zyx")
    tile_halo = INV35.parse_zyx_ints(args.tile_halo_zyx, name="--tile-halo-zyx")
    inference_request = SpatialInferenceConfig(
        spacing_zyx_um=tuple(float(v) for v in spacing),
        tile_shape_zyx=tile_shape,
        tile_overlap_zyx=tile_overlap,
        tile_halo_zyx=tile_halo,
        tile_batch_size=int(args.tile_batch_size),
    )
    runtime = load_spatial_runtime(
        paths.checkpoint,
        device=device,
        config=inference_request,
    )
    if runtime.checkpoint_sha256 != paths.checkpoint_sha256:
        raise RuntimeError(
            "Loaded spatial runtime checkpoint SHA differs from curation provenance."
        )
    segmentation_config = canonical_source_segmentation_config()

    persisted_sv_movie = np.load(paths.supervoxels, mmap_mode="r", allow_pickle=False)
    persisted_final_movie = np.load(paths.base_instances, mmap_mode="r", allow_pickle=False)

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 42 — BUILD RAG ON PERSISTED CURATION SUPERVOXELS", flush=True)
    print("=" * 112, flush=True)
    print(f"checkpoint      : {paths.checkpoint}", flush=True)
    print(f"checkpoint SHA  : {paths.checkpoint_sha256[:16]}...", flush=True)
    print(f"frames          : {todo}", flush=True)
    print(f"supervoxels     : {paths.supervoxels}", flush=True)
    print(f"final instances : {paths.base_instances}", flush=True)
    print("watershed       : BYPASSED (persisted atomic labels are authoritative)", flush=True)
    print("=" * 112, flush=True)

    paths.spatial_cache.mkdir(parents=True, exist_ok=True)

    for ordinal, t in enumerate(todo, 1):
        started = time.perf_counter()
        print(f"[persisted cache t={t:03d}] {ordinal}/{len(todo)} prepare", flush=True)

        prepared = prepare_spatial_frame(
            paths.zarr,
            int(t),
            config=inference_request,
            segmentation_config=segmentation_config,
        )
        spatial_tensor = torch.from_numpy(prepared.spatial)[None].to(
            device=runtime.device,
            dtype=torch.float32,
        )
        spacing_t = torch.tensor(
            [spacing],
            device=runtime.device,
            dtype=torch.float32,
        )
        dref_t = torch.tensor(
            [float(prepared.dref_um)],
            device=runtime.device,
            dtype=torch.float32,
        )

        persisted_sv_np = np.asarray(
            persisted_sv_movie[int(t)],
            dtype=np.int64,
        )
        persisted_final_np = np.asarray(
            persisted_final_movie[int(t)],
            dtype=np.int64,
        )
        persisted_sv_t = torch.from_numpy(
            np.ascontiguousarray(persisted_sv_np)
        ).to(device=runtime.device, dtype=torch.long)

        if runtime.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(runtime.device)

        amp, amp_name = _cache_amp_context(runtime.device)
        infer_started = time.perf_counter()
        with torch.inference_mode(), amp:
            dense = tiled_dense_geometry(
                runtime.model,
                spatial_tensor,
                spacing_t,
                dref_t,
                config=runtime.inference_cfg,
            )
            streamed = stream_tiled_label_feature_stats(
                runtime.model,
                spatial_tensor,
                spacing_t,
                dref_t,
                [persisted_sv_t],
                dense.blend_weight_sum,
                config=runtime.inference_cfg,
            )
            rag, _model_partition, _instances, _dummy = _spatial_from_streamed_stats(
                runtime.model,
                spatial_tensor,
                spacing_t,
                dref_t,
                dense.geometry,
                [persisted_sv_t],
                streamed,
            )
            actual_partition, partition_diag = partition_from_persisted_final(
                rag,
                supervoxels_zyx=persisted_sv_np,
                final_instances_zyx=persisted_final_np,
            )

        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
        infer_seconds = time.perf_counter() - infer_started
        peak_gib = (
            float(torch.cuda.max_memory_allocated(runtime.device) / 2**30)
            if runtime.device.type == "cuda"
            else 0.0
        )

        # Exact identity is now guaranteed by construction, but verify it before
        # persisting the cache so later code never relies on an implicit promise.
        cached_sv = (
            rag.supervoxel_labels[0].detach().cpu().numpy().astype(np.int64, copy=False)
        )
        mismatch = int(np.count_nonzero(cached_sv != persisted_sv_np))
        if mismatch:
            raise RuntimeError(
                f"Internal error: persisted-label RAG changed {mismatch} supervoxel voxels at t={t}."
            )

        frame_dir = paths.spatial_cache / f"t{t:03d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        INV35.atomic_torch_save(
            paths.graph_cache(t),
            {
                "version": CACHE_VERSION,
                "cache_contract": CACHE_CONTRACT,
                "rag": INV35.cpu_detached_tree(rag),
                "actual_partition": INV35.cpu_detached_tree(actual_partition),
            },
        )
        atomic_json(
            paths.spatial_meta(t),
            {
                "version": CACHE_VERSION,
                "cache_contract": CACHE_CONTRACT,
                "sample_id": paths.sample,
                "timepoint": int(t),
                "checkpoint": str(paths.checkpoint.resolve()),
                "checkpoint_step": int(paths.checkpoint_step),
                "checkpoint_sha256": paths.checkpoint_sha256,
                "source_supervoxels": str(paths.supervoxels.resolve()),
                "source_final_instances": str(paths.base_instances.resolve()),
                "spacing_zyx_um": [float(v) for v in spacing],
                "dref_um": float(prepared.dref_um),
                "rag_nodes": int(rag.node_features.shape[0]),
                "rag_edges": int(rag.edge_index.shape[1]),
                "persisted_supervoxel_count": int(np.unique(persisted_sv_np[persisted_sv_np > 0]).size),
                **partition_diag,
                "watershed_replayed": False,
                "persisted_atomic_labels_authoritative": True,
                "amp_dtype": amp_name,
                "feature_rebuild_seconds": float(infer_seconds),
                "preparation_seconds": float(prepared.preparation_seconds),
                "peak_allocated_vram_gib": float(peak_gib),
                "total_seconds": float(time.perf_counter() - started),
            },
        )

        print(
            f"[persisted cache t={t:03d}] DONE "
            f"SV={int(np.unique(persisted_sv_np[persisted_sv_np > 0]).size)} "
            f"nodes={int(rag.node_features.shape[0])} "
            f"edges={int(rag.edge_index.shape[1])} "
            f"current={partition_diag['persisted_final_instance_count']} "
            f"dref={float(prepared.dref_um):.4f}um "
            f"features={infer_seconds:.1f}s "
            f"total={duration(time.perf_counter() - started)}",
            flush=True,
        )

        del (
            prepared,
            spatial_tensor,
            spacing_t,
            dref_t,
            persisted_sv_t,
            dense,
            streamed,
            rag,
            _model_partition,
            _instances,
            _dummy,
            actual_partition,
        )
        gc.collect()
        if runtime.device.type == "cuda":
            torch.cuda.empty_cache()

    del runtime
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def validate_spatial_cache(paths: Paths, frames: Iterable[int]) -> dict[str, Any]:
    persisted_sv = np.load(paths.supervoxels, mmap_mode="r", allow_pickle=False)
    persisted_final = np.load(paths.base_instances, mmap_mode="r", allow_pickle=False)
    compared = 0
    total_voxels = 0
    total_mismatch = 0
    partition_mismatch_nodes = 0

    for t in frames:
        t = int(t)
        if not _cache_frame_compatible(paths, t):
            raise RuntimeError(
                f"Spatial cache t={t} is not a {CACHE_CONTRACT} cache."
            )
        payload = torch_load(paths.graph_cache(t))
        rag: RAGState = payload["rag"]
        actual: PartitionState = payload["actual_partition"]

        cached = (
            rag.supervoxel_labels[0]
            .detach()
            .cpu()
            .numpy()
            .astype(persisted_sv.dtype, copy=False)
        )
        expected = np.asarray(persisted_sv[t])
        if cached.shape != expected.shape:
            raise RuntimeError(
                f"Spatial cache t={t} supervoxel shape mismatch: {cached.shape} vs {expected.shape}"
            )
        mismatch = int(np.count_nonzero(cached != expected))
        compared += 1
        total_voxels += int(expected.size)
        total_mismatch += mismatch
        if mismatch:
            raise RuntimeError(
                "Persisted-label cache failed atomic identity check at "
                f"t={t}: mismatched_voxels={mismatch}."
            )

        # Validate that the cached current component assignment still exactly
        # represents persisted final_instances at the RAG-node level.
        lookup = INV35.sv_label_lookup(
            np.asarray(expected, dtype=np.int64),
            np.asarray(persisted_final[t], dtype=np.int64),
            name=f"validate persisted final t={t}",
        )
        node_sv = rag.node_supervoxel_id.detach().cpu().numpy().astype(np.int64, copy=False)
        node_final = lookup[node_sv]
        current = actual.node_component_global.detach().cpu().numpy().astype(np.int64, copy=False)
        final_to_component: dict[int, int] = {}
        for row, final_id in enumerate(node_final.tolist()):
            final_id = int(final_id)
            if final_id <= 0:
                continue
            comp = int(current[row])
            previous = final_to_component.setdefault(final_id, comp)
            if previous != comp:
                partition_mismatch_nodes += 1
        if partition_mismatch_nodes:
            raise RuntimeError(
                "Cached current partition does not preserve persisted final-instance identity."
            )

    return {
        "cache_contract": CACHE_CONTRACT,
        "frames_compared": int(compared),
        "voxels_compared": int(total_voxels),
        "mismatched_voxels": int(total_mismatch),
        "partition_mismatch_nodes": int(partition_mismatch_nodes),
        "exact_supervoxel_match": True,
        "persisted_final_partition": True,
    }


# =============================================================================
# Original Trackastra graph enrichment + coordinate audit
# =============================================================================


def load_track_graph(paths: Paths):
    with paths.track_graph.open("rb") as handle:
        return pickle.load(handle)


def relevant_temporal_frames(
    target_frames: Iterable[int],
    *,
    radius: int,
    frame_count: int,
) -> set[int]:
    result: set[int] = set()
    for t in target_frames:
        result.update(
            range(max(0, int(t) - radius), min(frame_count, int(t) + radius + 1))
        )
    return result


def audit_and_enrich_track_graph(
    graph,
    *,
    paths: Paths,
    target_frames: Iterable[int],
    spacing: Sequence[float],
    temporal_radius: int,
    max_error_um: float,
) -> dict[str, Any]:
    """Bind Trackastra graph nodes directly to persisted final instances.

    ``final_instances.npy`` is the strongest available identity contract here:
    it is the exact frozen spatial mask movie used by dataset curation and the
    mask movie supplied to Trackastra for this inference.

    Neither Trackastra graph ``label`` nor production ``tracks.csv`` cell_id is
    treated as authoritative.  ``tracks.csv`` is derived later by a nearest-cell
    visualization matcher and can become stale if spatial artifacts are repaired
    without rebuilding Trackastra.

    Per relevant frame we therefore:
      1. compute exact positive-instance centroids/volumes from
         ``final_instances.npy``;
      2. one-to-one Hungarian-match Trackastra graph coordinates to those
         persisted centroids using PHYSICAL Z/Y/X distance;
      3. copy the persisted centroid/volume into the graph node for temporal
         feature construction;
      4. remove isolated unmatched or >max_error_um graph nodes from the
         in-memory graph only.

    A high p95 matched-coordinate error still aborts because it indicates a
    systemic coordinate convention/inference mismatch.  Isolated outliers are
    excluded rather than snapped to a questionable cell.
    """
    from scipy.optimize import linear_sum_assignment
    from skimage.measure import regionprops_table

    relevant = relevant_temporal_frames(
        target_frames,
        radius=int(temporal_radius),
        frame_count=paths.frame_count,
    )
    base = np.load(
        paths.base_instances,
        mmap_mode="r",
        allow_pickle=False,
    )
    spacing_np = np.asarray(spacing, dtype=np.float64)

    if tuple(int(v) for v in base.shape)[0] != int(paths.frame_count):
        raise RuntimeError(
            "final_instances.npy frame count does not match the canonical "
            f"volume: labels={base.shape[0]} expected={paths.frame_count}"
        )

    matched_errors: list[float] = []
    dropped: list[dict[str, Any]] = []
    retained = 0
    total_graph_nodes = 0
    total_final_instances = 0
    unmatched_graph_nodes = 0
    unmatched_final_instances = 0
    invalid_coordinate_nodes = 0
    instance_ids_by_frame: dict[int, set[int]] = {}

    # The graph is loaded fresh from the immutable pickle for every run, so
    # enriching/removing nodes here never modifies dataset-curation artifacts.
    for t in sorted(relevant):
        labels = np.asarray(base[int(t)])
        props = regionprops_table(
            labels,
            properties=("label", "centroid", "area"),
        )
        cell_ids = np.asarray(props["label"], dtype=np.int64)
        if cell_ids.size:
            cell_coords = np.stack(
                [
                    np.asarray(props["centroid-0"], dtype=np.float64),
                    np.asarray(props["centroid-1"], dtype=np.float64),
                    np.asarray(props["centroid-2"], dtype=np.float64),
                ],
                axis=1,
            )
            cell_volumes = np.asarray(props["area"], dtype=np.int64)
        else:
            cell_coords = np.zeros((0, 3), dtype=np.float64)
            cell_volumes = np.zeros((0,), dtype=np.int64)

        instance_ids_by_frame[int(t)] = {
            int(v) for v in cell_ids.tolist()
        }
        total_final_instances += int(cell_ids.size)

        frame_node_ids = [
            int(node_id)
            for node_id, data in graph.nodes(data=True)
            if int(data.get("time", -10**9)) == int(t)
        ]
        total_graph_nodes += len(frame_node_ids)

        valid_node_ids: list[int] = []
        graph_coords_rows: list[np.ndarray] = []
        invalid_this_frame: list[int] = []

        for node_id in frame_node_ids:
            try:
                coords = np.asarray(
                    INV35.get_coords_zyx(graph.nodes[node_id]),
                    dtype=np.float64,
                ).reshape(-1)
            except Exception:
                coords = np.zeros((0,), dtype=np.float64)

            if coords.size != 3 or not np.isfinite(coords).all():
                invalid_this_frame.append(int(node_id))
                continue
            valid_node_ids.append(int(node_id))
            graph_coords_rows.append(coords)

        if invalid_this_frame:
            invalid_coordinate_nodes += len(invalid_this_frame)
            unmatched_graph_nodes += len(invalid_this_frame)
            for node_id in invalid_this_frame:
                dropped.append(
                    {
                        "node_id": int(node_id),
                        "frame": int(t),
                        "reason": "invalid_coordinate",
                    }
                )

        if not valid_node_ids:
            unmatched_final_instances += int(cell_ids.size)
            continue

        if cell_ids.size == 0:
            unmatched_graph_nodes += len(valid_node_ids)
            for node_id in valid_node_ids:
                dropped.append(
                    {
                        "node_id": int(node_id),
                        "frame": int(t),
                        "reason": "no_persisted_instance_in_frame",
                    }
                )
            continue

        graph_coords = np.stack(graph_coords_rows, axis=0)
        delta_um = (
            graph_coords[:, None, :] - cell_coords[None, :, :]
        ) * spacing_np[None, None, :]
        cost_um = np.linalg.norm(delta_um, axis=-1)

        graph_rows, cell_rows = linear_sum_assignment(cost_um)
        assigned_graph = set(int(v) for v in graph_rows.tolist())
        assigned_cells = set(int(v) for v in cell_rows.tolist())

        unmatched_graph_rows = [
            row
            for row in range(len(valid_node_ids))
            if row not in assigned_graph
        ]
        unmatched_cell_rows = [
            row
            for row in range(int(cell_ids.size))
            if row not in assigned_cells
        ]

        unmatched_graph_nodes += len(unmatched_graph_rows)
        unmatched_final_instances += len(unmatched_cell_rows)

        for row in unmatched_graph_rows:
            dropped.append(
                {
                    "node_id": int(valid_node_ids[row]),
                    "frame": int(t),
                    "reason": "unmatched_graph_detection",
                    "trackastra_coords_zyx": [
                        float(v) for v in graph_coords[row]
                    ],
                }
            )

        for graph_row, cell_row in zip(
            graph_rows.tolist(),
            cell_rows.tolist(),
        ):
            node_id = int(valid_node_ids[int(graph_row)])
            cell_id = int(cell_ids[int(cell_row)])
            canonical = cell_coords[int(cell_row)]
            volume = int(cell_volumes[int(cell_row)])
            error = float(cost_um[int(graph_row), int(cell_row)])
            matched_errors.append(error)

            if error > float(max_error_um):
                dropped.append(
                    {
                        "node_id": int(node_id),
                        "frame": int(t),
                        "cell_id": int(cell_id),
                        "reason": "coordinate_outlier",
                        "error_um": float(error),
                        "trackastra_coords_zyx": [
                            float(v) for v in graph_coords[int(graph_row)]
                        ],
                        "canonical_coords_zyx": [
                            float(v) for v in canonical
                        ],
                    }
                )
                continue

            data = graph.nodes[node_id]
            # INV35.get_coords_zyx() intentionally prefers this override.
            data["inv35_coords_zyx"] = canonical.astype(np.float32)
            data["inv35_volume_voxels"] = int(volume)
            data["inv42_cell_id"] = int(cell_id)
            data["inv42_match_error_um"] = float(error)
            data["inv42_identity_source"] = (
                "persisted_final_instances_hungarian"
            )
            retained += 1

    if not matched_errors:
        raise RuntimeError(
            "No Trackastra graph detections in the requested temporal windows "
            "could be matched to persisted final_instances.npy."
        )

    error_array = np.asarray(matched_errors, dtype=np.float64)
    p95 = float(np.percentile(error_array, 95))
    if p95 > float(max_error_um):
        raise RuntimeError(
            "Trackastra/final-instance coordinate convention is SYSTEMICALLY "
            "mismatched: "
            f"median={float(np.median(error_array)):.4f} um, "
            f"p95={p95:.4f} um, "
            f"max={float(np.max(error_array)):.4f} um, "
            f"allowed_p95={float(max_error_um):.4f} um."
        )

    dropped_node_ids = sorted(
        {
            int(row["node_id"])
            for row in dropped
            if "node_id" in row
        }
    )
    if dropped_node_ids:
        graph.remove_nodes_from(dropped_node_ids)
        print(
            "[Trackastra coordinates] excluded non-canonical temporal nodes: "
            f"{len(dropped_node_ids)}/{total_graph_nodes} "
            f"(threshold={float(max_error_um):.3f}um).",
            flush=True,
        )
        for row in sorted(
            dropped,
            key=lambda item: float(item.get("error_um", -1.0)),
            reverse=True,
        )[:10]:
            detail = (
                f" error={float(row['error_um']):.4f}um"
                if "error_um" in row
                else ""
            )
            print(
                "  "
                f"node={row.get('node_id')} "
                f"t={int(row.get('frame', -1)):03d} "
                f"reason={row.get('reason')}{detail}",
                flush=True,
            )

    # Secondary diagnostics only.  These files are useful for explaining stale
    # curation artifacts but never define temporal identity for Investigation 42.
    stale_tracks_rows = 0
    stale_tracks_examples: list[dict[str, int]] = []
    tracks_diag_error: str | None = None
    if paths.tracks_csv.is_file():
        try:
            tracks = pd.read_csv(paths.tracks_csv)
            required = {"frame", "cell_id"}
            if not required.issubset(tracks.columns):
                raise ValueError(
                    "missing columns "
                    + str(sorted(required - set(tracks.columns)))
                )
            for row in tracks.itertuples(index=False):
                frame = int(row.frame)
                if frame not in relevant:
                    continue
                cell_id = int(row.cell_id)
                if cell_id <= 0:
                    continue
                if cell_id not in instance_ids_by_frame.get(frame, set()):
                    stale_tracks_rows += 1
                    if len(stale_tracks_examples) < 20:
                        stale_tracks_examples.append(
                            {
                                "frame": int(frame),
                                "cell_id": int(cell_id),
                            }
                        )
        except Exception as exc:
            tracks_diag_error = f"{type(exc).__name__}: {exc}"

    stale_cells_rows = 0
    stale_cells_examples: list[dict[str, int]] = []
    cells_diag_error: str | None = None
    if paths.cells_csv.is_file():
        try:
            cells = pd.read_csv(paths.cells_csv)
            required = {"frame", "cell_id"}
            if not required.issubset(cells.columns):
                raise ValueError(
                    "missing columns "
                    + str(sorted(required - set(cells.columns)))
                )
            for row in cells.itertuples(index=False):
                frame = int(row.frame)
                if frame not in relevant:
                    continue
                cell_id = int(row.cell_id)
                if cell_id <= 0:
                    continue
                if cell_id not in instance_ids_by_frame.get(frame, set()):
                    stale_cells_rows += 1
                    if len(stale_cells_examples) < 20:
                        stale_cells_examples.append(
                            {
                                "frame": int(frame),
                                "cell_id": int(cell_id),
                            }
                        )
        except Exception as exc:
            cells_diag_error = f"{type(exc).__name__}: {exc}"

    if stale_tracks_rows or stale_cells_rows:
        print(
            "[curation artifact audit] non-authoritative CSV rows are stale "
            "relative to final_instances.npy: "
            f"tracks.csv={stale_tracks_rows}, cells_all.csv={stale_cells_rows}. "
            "Investigation 42 uses final_instances.npy directly.",
            flush=True,
        )

    retained_fraction = float(
        retained / max(total_graph_nodes, 1)
    )
    return {
        "identity_contract": (
            "trackastra_graph_hungarian_to_persisted_final_instances"
        ),
        "coordinate_source_for_training": (
            "centroid_computed_directly_from_final_instances_npy"
        ),
        "bad_coordinate_policy": (
            "drop_in_memory_temporal_node_and_incident_edges"
        ),
        "relevant_temporal_frames": sorted(int(v) for v in relevant),
        "graph_nodes_in_relevant_frames": int(total_graph_nodes),
        "final_instances_in_relevant_frames": int(total_final_instances),
        "compared_nodes": int(len(matched_errors)),
        "retained_nodes": int(retained),
        "retained_fraction_of_graph_nodes": float(retained_fraction),
        "dropped_coordinate_outlier_count": int(
            sum(row.get("reason") == "coordinate_outlier" for row in dropped)
        ),
        "invalid_coordinate_node_count": int(invalid_coordinate_nodes),
        "unmatched_graph_node_count": int(unmatched_graph_nodes),
        "unmatched_final_instance_count": int(unmatched_final_instances),
        "total_dropped_graph_node_count": int(len(dropped_node_ids)),
        "median_error_um": float(np.median(error_array)),
        "p95_error_um": float(p95),
        "max_error_um": float(np.max(error_array)),
        "allowed_p95_error_um": float(max_error_um),
        "stale_tracks_csv_rows": int(stale_tracks_rows),
        "stale_tracks_csv_examples": stale_tracks_examples,
        "tracks_csv_diagnostic_error": tracks_diag_error,
        "stale_cells_csv_rows": int(stale_cells_rows),
        "stale_cells_csv_examples": stale_cells_examples,
        "cells_csv_diagnostic_error": cells_diag_error,
        "worst_dropped_coordinate_outliers": sorted(
            [
                row for row in dropped
                if row.get("reason") == "coordinate_outlier"
            ],
            key=lambda row: float(row.get("error_um", 0.0)),
            reverse=True,
        )[:20],
    }


# =============================================================================
# Real curated runtime / CUT-KEEP targets
# =============================================================================


@dataclass
class RuntimeFrame:
    t: int
    rag: RAGState
    actual_partition: PartitionState
    target: np.ndarray
    node_target: Tensor
    node_valid: Tensor
    ignored_ids: tuple[int, ...]
    dref_um: float


class RuntimeLoader:
    def __init__(self, paths: Paths, target_store: TargetFrameStore, device: torch.device):
        self.paths = paths
        self.target_store = target_store
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

        payload = torch_load(self.paths.graph_cache(t))
        rag: RAGState = INV35.tree_to_device_fp32(payload["rag"], self.device)
        actual: PartitionState = INV35.tree_to_device_fp32(
            payload["actual_partition"], self.device
        )
        if rag.statistics is None:
            raise RuntimeError(
                f"Frozen cache t={t} lacks compact RAG statistics required by InstanceTokenizer"
            )

        target = np.asarray(self.target_store.frame(t))
        supervoxels = (
            rag.supervoxel_labels[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        lookup = INV35.sv_label_lookup(
            supervoxels,
            target,
            name=f"inv42 target t={t}",
        )
        node_sv = (
            rag.node_supervoxel_id
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        node_target = torch.as_tensor(
            lookup[node_sv],
            device=self.device,
            dtype=torch.long,
        )

        ignored = tuple(sorted(self.target_store.ignored_for_frame(t)))
        if ignored:
            ignored_tensor = torch.tensor(ignored, device=self.device, dtype=torch.long)
            node_ignored = torch.isin(node_target, ignored_tensor)
        else:
            node_ignored = torch.zeros_like(node_target, dtype=torch.bool)
        node_valid = (node_target > 0) & ~node_ignored

        dref_um = float(json_load(self.paths.spatial_meta(t))["dref_um"])
        self.current = RuntimeFrame(
            t=t,
            rag=rag,
            actual_partition=actual,
            target=target,
            node_target=node_target,
            node_valid=node_valid,
            ignored_ids=ignored,
            dref_um=dref_um,
        )
        return self.current


@dataclass
class RealCase:
    rag: RAGState
    partition: PartitionState
    target_keep: Tensor
    edge_valid: Tensor
    same_current: Tensor
    editable: Tensor
    cut_mask: Tensor
    keep_mask: Tensor
    split_target: Tensor
    split_valid: Tensor
    metric_component_valid: Tensor
    node_current_component: Tensor

    @property
    def bad_components(self) -> Tensor:
        return torch.nonzero(
            self.split_valid & self.metric_component_valid & (self.split_target > 0.5),
            as_tuple=False,
        ).flatten()

    @property
    def clean_components(self) -> Tensor:
        return torch.nonzero(
            self.split_valid & self.metric_component_valid & (self.split_target <= 0.5),
            as_tuple=False,
        ).flatten()


def build_real_case(runtime: RuntimeFrame) -> RealCase:
    rag = runtime.rag
    current = runtime.actual_partition.node_component_global.long()
    src, dst = rag.edge_index

    edge_valid = runtime.node_valid[src] & runtime.node_valid[dst]
    target_keep = edge_valid & (runtime.node_target[src] == runtime.node_target[dst])
    same_current = current[src] == current[dst]
    editable = edge_valid & same_current
    cut_mask = editable & ~target_keep
    keep_mask = editable & target_keep

    component_count = int(
        runtime.actual_partition.component_count_per_batch.sum().item()
    )
    split_target = rag.spatial_edge_logits.new_zeros((component_count,))
    split_valid = torch.zeros(component_count, device=current.device, dtype=torch.bool)
    metric_valid = torch.zeros_like(split_valid)

    # Target IDs that span multiple current components indicate an over-split
    # situation. Split-only temporal reasoning cannot repair that, so those
    # components remain usable for edge supervision but are excluded from the
    # exact component-recovery metric.
    target_components: dict[int, set[int]] = defaultdict(set)
    for target_id in torch.unique(runtime.node_target[runtime.node_valid]).tolist():
        target_id = int(target_id)
        rows = torch.nonzero(
            runtime.node_valid & (runtime.node_target == target_id),
            as_tuple=False,
        ).flatten()
        target_components[target_id].update(int(v) for v in torch.unique(current[rows]).tolist())

    for comp in range(component_count):
        rows = torch.nonzero(current == comp, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        fully_trusted = bool(runtime.node_valid[rows].all())
        if not fully_trusted:
            continue
        ids = torch.unique(runtime.node_target[rows])
        ids = ids[ids > 0]
        if ids.numel() == 0:
            continue
        split_valid[comp] = True
        split_target[comp] = float(ids.numel() >= 2)
        metric_valid[comp] = all(
            len(target_components.get(int(target_id), set())) == 1
            for target_id in ids.tolist()
        )

    return RealCase(
        rag=rag,
        partition=runtime.actual_partition,
        target_keep=target_keep,
        edge_valid=edge_valid,
        same_current=same_current,
        editable=editable,
        cut_mask=cut_mask,
        keep_mask=keep_mask,
        split_target=split_target,
        split_valid=split_valid,
        metric_component_valid=metric_valid,
        node_current_component=current,
    )


# =============================================================================
# Temporal forward / split-only enforcement
# =============================================================================


@dataclass
class EncodedCase:
    instances: Any
    temporal: Any
    dref_t: Tensor


@dataclass
class ReasonedCase:
    reasoning: Any
    final_logits: Tensor
    candidate_logits: Tensor
    gate_logits: Tensor
    base_gate: Tensor


class SelectiveWriteGateCalibrator(nn.Module):
    """Residual calibration of the frozen step-300 temporal gate.

    The original production gate only sees:
        spatial uncertainty, left temporal support, right temporal support,
        same-provisional-instance.

    V10 showed that those four inputs cannot separate helpful from harmful
    candidate writes on the curated validation split.

    This experiment keeps the complete temporal candidate frozen and adds only
    a small residual gate calibrator that additionally sees:
        * frozen RAG edge embedding,
        * spatial merge margin,
        * temporal-candidate merge margin,
        * signed/absolute candidate displacement,
        * original gate probability/logit,
        * original gated-final margin,
        * spatial uncertainty,
        * candidate-vs-spatial probability displacement.

    The output layer is initialized to exactly zero, therefore correction=0 and
    the initial calibrated gate is numerically the original step-300 gate.
    """

    SCALAR_DIM = 9

    def __init__(
        self,
        edge_embedding_dim: int,
        *,
        correction_scale: float,
    ) -> None:
        super().__init__()
        edge_embedding_dim = int(edge_embedding_dim)
        if edge_embedding_dim <= 0:
            raise ValueError("edge_embedding_dim must be positive")
        self.edge_embedding_dim = edge_embedding_dim
        self.correction_scale = float(correction_scale)
        if not math.isfinite(self.correction_scale) or self.correction_scale <= 0:
            raise ValueError("correction_scale must be positive")

        self.edge_norm = nn.LayerNorm(edge_embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(edge_embedding_dim + self.SCALAR_DIM, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        last = self.net[-1]
        if not isinstance(last, nn.Linear):
            raise RuntimeError("Unexpected selective gate output module")
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        *,
        edge_embedding: Tensor,
        scalar_features: Tensor,
        base_gate: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if edge_embedding.ndim != 2:
            raise ValueError("edge_embedding must be [E,D]")
        if scalar_features.ndim != 2:
            raise ValueError("scalar_features must be [E,F]")
        if edge_embedding.shape[0] != scalar_features.shape[0]:
            raise ValueError("Selective gate edge/scalar row mismatch")
        if int(edge_embedding.shape[1]) != self.edge_embedding_dim:
            raise ValueError(
                "Selective gate edge embedding dimension mismatch: "
                f"got {edge_embedding.shape[1]}, expected {self.edge_embedding_dim}"
            )
        if int(scalar_features.shape[1]) != self.SCALAR_DIM:
            raise ValueError(
                "Selective gate scalar feature dimension mismatch: "
                f"got {scalar_features.shape[1]}, expected {self.SCALAR_DIM}"
            )

        edge = self.edge_norm(edge_embedding.float())
        scalars = scalar_features.float()
        residual_raw = self.net(
            torch.cat([edge, scalars], dim=-1)
        ).squeeze(-1)
        correction = self.correction_scale * torch.tanh(residual_raw)

        base = base_gate.float().clamp(1.0e-5, 1.0 - 1.0e-5)
        base_logit = torch.logit(base)
        calibrated_logit = base_logit + correction
        calibrated_gate = torch.sigmoid(calibrated_logit)
        return calibrated_logit, calibrated_gate


def selective_gate_scalar_features(
    model,
    *,
    case: RealCase,
    base_reasoning,
    candidate_logits: Tensor,
) -> Tensor:
    threshold = float(model.cfg.partition.final_merge_threshold)
    if not 0.0 < threshold < 1.0:
        raise RuntimeError(
            "final_merge_threshold must lie strictly inside (0,1) "
            "for selective gate calibration"
        )
    threshold_logit = math.log(threshold / (1.0 - threshold))

    spatial = case.rag.spatial_edge_logits.detach().float()
    candidate = candidate_logits.detach().float()
    delta = candidate - spatial
    base_gate = base_reasoning.edge_temporal_gate.detach().float().clamp(
        1.0e-5,
        1.0 - 1.0e-5,
    )
    base_gate_logit = torch.logit(base_gate)
    base_final = base_reasoning.final_edge_logits.detach().float()

    spatial_prob = torch.sigmoid(spatial)
    candidate_prob = torch.sigmoid(candidate)

    # Logit-valued features are bounded/scaled so no isolated confident edge can
    # dominate the small calibration MLP.
    def margin(value: Tensor) -> Tensor:
        return ((value - float(threshold_logit)).clamp(-10.0, 10.0) / 5.0)

    return torch.stack(
        [
            margin(spatial),
            margin(candidate),
            delta.clamp(-10.0, 10.0) / 5.0,
            delta.abs().clamp(0.0, 10.0) / 5.0,
            base_gate,
            base_gate_logit.clamp(-10.0, 10.0) / 5.0,
            margin(base_final),
            torch.exp(-spatial.abs()).clamp(0.0, 1.0),
            (candidate_prob - spatial_prob).clamp(-1.0, 1.0),
        ],
        dim=-1,
    )


def encode_case(
    model,
    *,
    runtime: RuntimeFrame,
    case: RealCase,
    temporal_input: TemporalInput,
    spacing: Sequence[float],
    device: torch.device,
) -> EncodedCase:
    decoded, geometry = INV35.dummy_geometry_and_decode(model, case.rag.node_features)
    spacing_t = torch.tensor([spacing], device=device, dtype=torch.float32)
    dref_t = torch.tensor([float(runtime.dref_um)], device=device, dtype=torch.float32)
    instances = model.instance_tokenizer(
        case.partition,
        case.rag,
        decoded,
        geometry,
        spacing_t,
        dref_t,
        profile_prefix="inv42_curated_tokenizer",
    )
    temporal = model.temporal_encoder(temporal_input)
    return EncodedCase(instances=instances, temporal=temporal, dref_t=dref_t)


def reason_case(
    model,
    gate_calibrator: SelectiveWriteGateCalibrator,
    *,
    case: RealCase,
    encoded: EncodedCase,
    temporal=None,
) -> ReasonedCase:
    state = encoded.temporal if temporal is None else temporal

    # The complete step-300 temporal stack is frozen.  Only the experiment-local
    # gate calibrator below receives gradients.
    base_reasoning = model.instance_temporal(
        encoded.instances,
        case.rag,
        state,
        encoded.dref_t,
    )
    candidate_logits = (
        case.rag.spatial_edge_logits
        + base_reasoning.edge_temporal_delta
    )
    scalar_features = selective_gate_scalar_features(
        model,
        case=case,
        base_reasoning=base_reasoning,
        candidate_logits=candidate_logits,
    )
    gate_logits, calibrated_gate = gate_calibrator(
        edge_embedding=case.rag.edge_embeddings.detach(),
        scalar_features=scalar_features,
        base_gate=base_reasoning.edge_temporal_gate.detach(),
    )

    calibrated_final = (
        case.rag.spatial_edge_logits
        + calibrated_gate.to(case.rag.spatial_edge_logits.dtype)
        * base_reasoning.edge_temporal_delta
    )
    reasoning = dataclasses.replace(
        base_reasoning,
        edge_temporal_gate=calibrated_gate.to(
            base_reasoning.edge_temporal_gate.dtype
        ),
        final_edge_logits=calibrated_final,
    )

    # Hard split-only + Ignore invariant: only trusted edges already within the
    # same frozen spatial component may be modified by temporal reasoning.
    final_logits = torch.where(
        case.editable,
        calibrated_final,
        case.rag.spatial_edge_logits,
    )
    return ReasonedCase(
        reasoning=reasoning,
        final_logits=final_logits,
        candidate_logits=candidate_logits,
        gate_logits=gate_logits,
        base_gate=base_reasoning.edge_temporal_gate.detach(),
    )


def make_temporal_input(
    graph,
    *,
    runtime: RuntimeFrame,
    paths: Paths,
    temporal_radius: int,
    spacing: Sequence[float],
    device: torch.device,
) -> TemporalInput:
    return INV35.direct_temporal_input(
        graph,
        target_t=runtime.t,
        frame_count=paths.frame_count,
        temporal_radius=int(temporal_radius),
        spacing=spacing,
        dref_um=float(runtime.dref_um),
        shape_zyx=runtime.target.shape,
        device=device,
    )


# =============================================================================
# Loss
# =============================================================================


@dataclass
class LossResult:
    total: Tensor
    cut: Tensor
    keep: Tensor
    candidate_cut: Tensor
    candidate_keep: Tensor
    candidate_total: Tensor
    gate_help: Tensor
    gate_suppress: Tensor
    gate_total: Tensor
    split: Tensor
    causal: Tensor
    causal_noop: Tensor
    causal_gate: Tensor
    causal_margin: Tensor
    cut_edges: int
    keep_edges: int
    gate_help_edges: int
    gate_suppress_edges: int
    corruption: str


def training_loss(
    model,
    gate_calibrator: SelectiveWriteGateCalibrator,
    *,
    case: RealCase,
    encoded: EncodedCase,
    full: ReasonedCase,
    args: argparse.Namespace,
    rng: random.Random,
    corruption: str,
) -> LossResult:
    logits = full.final_logits
    zero = logits.sum() * 0.0

    cut_index = torch.nonzero(case.cut_mask, as_tuple=False).flatten()
    keep_index = torch.nonzero(case.keep_mask, as_tuple=False).flatten()
    keep_budget = int(args.preserve_edges)
    if cut_index.numel():
        keep_budget = min(
            keep_budget,
            max(int(cut_index.numel()) * int(args.preserve_ratio), int(cut_index.numel())),
        )
    keep_index = sampled(keep_index, keep_budget, rng)

    cut = (
        F.binary_cross_entropy_with_logits(logits[cut_index], torch.zeros_like(logits[cut_index]))
        if cut_index.numel()
        else zero
    )
    keep = (
        F.binary_cross_entropy_with_logits(logits[keep_index], torch.ones_like(logits[keep_index]))
        if keep_index.numel()
        else zero
    )

    # Direct temporal-candidate supervision.
    #
    # InstanceTemporalReasoner exposes:
    #   edge_temporal_delta = temporal_candidate - spatial_edge_logits
    #
    # so the ungated candidate is reconstructed exactly here.  This auxiliary
    # loss avoids the gradient-starvation failure observed when preservation
    # pressure drives edge_temporal_gate close to zero: the candidate continues
    # learning the temporal CUT/KEEP signal even while the final write gate is
    # conservative.
    candidate_logits = full.candidate_logits
    candidate_cut = (
        F.binary_cross_entropy_with_logits(
            candidate_logits[cut_index],
            torch.zeros_like(candidate_logits[cut_index]),
        )
        if cut_index.numel()
        else zero
    )
    candidate_keep = (
        F.binary_cross_entropy_with_logits(
            candidate_logits[keep_index],
            torch.ones_like(candidate_logits[keep_index]),
        )
        if keep_index.numel()
        else zero
    )
    candidate_total = (
        candidate_cut
        + float(args.candidate_preservation_weight) * candidate_keep
    )

    # Gate-only selective-write supervision.
    #
    # The step-300 temporal candidate is intentionally frozen.  Train the gate
    # to WRITE only when that frozen candidate fixes a spatial decision, and to
    # SUPPRESS when the candidate would damage an already-correct spatial edge.
    #
    # Edges where candidate and spatial are both correct or both wrong carry no
    # oracle gate target: changing the gate cannot improve their binary class.
    threshold = float(model.cfg.partition.final_merge_threshold)
    with torch.no_grad():
        target_keep_bool = case.target_keep.bool()
        spatial_keep_prediction = (
            torch.sigmoid(case.rag.spatial_edge_logits) >= threshold
        )
        candidate_keep_prediction = (
            torch.sigmoid(candidate_logits) >= threshold
        )
        spatial_correct = spatial_keep_prediction == target_keep_bool
        candidate_correct = candidate_keep_prediction == target_keep_bool
        gate_help_mask = (
            case.editable
            & candidate_correct
            & ~spatial_correct
        )
        gate_suppress_mask = (
            case.editable
            & ~candidate_correct
            & spatial_correct
        )

    gate_probability = full.reasoning.edge_temporal_gate
    gate_logits = full.gate_logits
    gate_help_index = torch.nonzero(
        gate_help_mask,
        as_tuple=False,
    ).flatten()
    gate_suppress_index = torch.nonzero(
        gate_suppress_mask,
        as_tuple=False,
    ).flatten()

    gate_help = (
        F.binary_cross_entropy_with_logits(
            gate_logits[gate_help_index],
            torch.ones_like(gate_logits[gate_help_index]),
        )
        if gate_help_index.numel()
        else zero
    )
    gate_suppress = (
        F.binary_cross_entropy_with_logits(
            gate_logits[gate_suppress_index],
            torch.zeros_like(gate_logits[gate_suppress_index]),
        )
        if gate_suppress_index.numel()
        else zero
    )
    gate_total = (
        float(args.gate_help_weight) * gate_help
        + float(args.gate_suppress_weight) * gate_suppress
    )

    positive = torch.nonzero(
        case.split_valid & (case.split_target > 0.5), as_tuple=False
    ).flatten()
    negative = torch.nonzero(
        case.split_valid & (case.split_target <= 0.5), as_tuple=False
    ).flatten()
    if positive.numel():
        negative = sampled(
            negative,
            min(int(negative.numel()), max(16, int(positive.numel()) * 8)),
            rng,
        )
        split_index = torch.cat([positive, negative])
    else:
        split_index = sampled(negative, min(int(negative.numel()), 64), rng)

    split = (
        F.binary_cross_entropy_with_logits(
            full.reasoning.split_logits[split_index],
            case.split_target[split_index].to(full.reasoning.split_logits.dtype),
        )
        if split_index.numel() and float(args.split_weight) > 0
        else zero
    )

    causal_total = zero
    causal_noop = zero.detach()
    causal_gate = zero.detach()
    causal_margin = zero.detach()
    if not bool(args.no_causal) and case.editable.any():
        corrupted_temporal = corrupt_temporal_state(
            encoded.temporal,
            corruption=corruption,
            seed=int(args.seed) + rng.randrange(1_000_000_000),
        )
        corrupted = reason_case(
            model,
            gate_calibrator,
            case=case,
            encoded=encoded,
            temporal=corrupted_temporal,
        )
        terms = causal_temporal_loss_terms(
            spatial_edge_logits=case.rag.spatial_edge_logits,
            spatial_same_component=case.same_current,
            full_reasoning=full.reasoning,
            corrupted_reasoning=corrupted.reasoning,
            target=case.target_keep.to(case.rag.spatial_edge_logits.dtype),
            valid=case.editable,
            noop_weight=float(args.causal_noop_weight),
            corrupted_gate_weight=float(args.causal_gate_weight),
            margin_weight=float(args.causal_margin_weight),
            margin=float(args.causal_margin),
        )
        causal_total = terms.total
        causal_noop = terms.noop
        causal_gate = terms.corrupted_gate
        causal_margin = terms.margin

    total = (
        cut
        + float(args.preservation_weight) * keep
        + gate_total
        + causal_total
    )
    return LossResult(
        total=total,
        cut=cut,
        keep=keep,
        candidate_cut=candidate_cut,
        candidate_keep=candidate_keep,
        candidate_total=candidate_total,
        gate_help=gate_help,
        gate_suppress=gate_suppress,
        gate_total=gate_total,
        split=split,
        causal=causal_total,
        causal_noop=causal_noop,
        causal_gate=causal_gate,
        causal_margin=causal_margin,
        cut_edges=int(cut_index.numel()),
        keep_edges=int(keep_index.numel()),
        gate_help_edges=int(gate_help_index.numel()),
        gate_suppress_edges=int(gate_suppress_index.numel()),
        corruption=str(corruption),
    )


# =============================================================================
# Metrics
# =============================================================================


def enforce_split_only(case: RealCase, predicted: PartitionState) -> PartitionState:
    return INV35.enforce_split_only_partition(case, predicted)


def exact_component_counts(
    runtime: RuntimeFrame,
    case: RealCase,
    predicted: PartitionState,
) -> tuple[int, int, int, int]:
    bad_total = bad_exact = clean_total = clean_split = 0
    current = case.node_current_component

    for comp in case.bad_components.tolist():
        rows = torch.nonzero(current == int(comp), as_tuple=False).flatten()
        target_ids = torch.unique(runtime.node_target[rows])
        target_ids = target_ids[target_ids > 0]
        bad_total += 1
        seen: set[int] = set()
        exact = True
        for target_id in target_ids.tolist():
            target_rows = rows[runtime.node_target[rows] == int(target_id)]
            components = torch.unique(predicted.node_component_global[target_rows])
            if components.numel() != 1:
                exact = False
                break
            final_component = int(components.item())
            if final_component in seen:
                exact = False
                break
            seen.add(final_component)
            members = torch.nonzero(
                predicted.node_component_global == final_component,
                as_tuple=False,
            ).flatten()
            trusted_members = members[runtime.node_valid[members]]
            member_targets = runtime.node_target[trusted_members]
            if bool((member_targets != int(target_id)).any()):
                exact = False
                break
        bad_exact += int(exact)

    for comp in case.clean_components.tolist():
        rows = torch.nonzero(current == int(comp), as_tuple=False).flatten()
        clean_total += 1
        clean_split += int(torch.unique(predicted.node_component_global[rows]).numel() > 1)

    return bad_total, bad_exact, clean_total, clean_split


def update_metric_accumulator(
    acc: dict[str, float],
    *,
    model,
    runtime: RuntimeFrame,
    case: RealCase,
    logits: Tensor,
) -> None:
    threshold = float(model.cfg.partition.final_merge_threshold)
    keep_prediction = torch.sigmoid(logits) >= threshold
    acc["cut_total"] += int(case.cut_mask.sum().item())
    acc["cut_correct"] += int((~keep_prediction[case.cut_mask]).sum().item())
    acc["keep_total"] += int(case.keep_mask.sum().item())
    acc["keep_correct"] += int(keep_prediction[case.keep_mask].sum().item())

    predicted = model.partitioner(
        case.rag,
        logits,
        model.cfg.partition.final_merge_threshold,
        stage="final",
    )
    predicted = enforce_split_only(case, predicted)
    bad_total, bad_exact, clean_total, clean_split = exact_component_counts(
        runtime, case, predicted
    )
    acc["bad_total"] += bad_total
    acc["bad_exact"] += bad_exact
    acc["clean_total"] += clean_total
    acc["clean_split"] += clean_split
    acc["violations"] += INV35.split_only_violation_count(case, predicted)


def finalize_metrics(acc: dict[str, float]) -> dict[str, float]:
    cut_accuracy = acc["cut_correct"] / max(acc["cut_total"], 1)
    keep_accuracy = acc["keep_correct"] / max(acc["keep_total"], 1)
    exact = acc["bad_exact"] / max(acc["bad_total"], 1)
    false_split = acc["clean_split"] / max(acc["clean_total"], 1)
    return {
        "cut_edges": int(acc["cut_total"]),
        "keep_edges": int(acc["keep_total"]),
        "cut_accuracy": float(cut_accuracy),
        "keep_accuracy": float(keep_accuracy),
        "bad_components": int(acc["bad_total"]),
        "exact_bad_component_recovery": float(exact),
        "clean_components": int(acc["clean_total"]),
        "clean_false_split_rate": float(false_split),
        "split_only_violations": int(acc["violations"]),
    }


def metric_accumulator() -> dict[str, float]:
    return {
        "cut_total": 0,
        "cut_correct": 0,
        "keep_total": 0,
        "keep_correct": 0,
        "bad_total": 0,
        "bad_exact": 0,
        "clean_total": 0,
        "clean_split": 0,
        "violations": 0,
    }


@torch.no_grad()
def evaluate_spatial_baseline(
    model,
    *,
    frames: Sequence[int],
    loader: RuntimeLoader,
) -> dict[str, Any]:
    acc = metric_accumulator()
    for t in frames:
        runtime = loader.load(t)
        case = build_real_case(runtime)
        update_metric_accumulator(
            acc,
            model=model,
            runtime=runtime,
            case=case,
            logits=case.rag.spatial_edge_logits,
        )
    result = finalize_metrics(acc)
    result["mode"] = "frozen_spatial"
    return result


@torch.no_grad()
def evaluate_temporal(
    model,
    gate_calibrator: SelectiveWriteGateCalibrator,
    *,
    frames: Sequence[int],
    loader: RuntimeLoader,
    graph,
    paths: Paths,
    temporal_radius: int,
    spacing: Sequence[float],
    device: torch.device,
) -> dict[str, Any]:
    set_selective_gate_mode(model, gate_calibrator, False)
    real_acc = metric_accumulator()
    candidate_acc = metric_accumulator()
    contentless_acc = metric_accumulator()
    shuffled_acc = metric_accumulator()
    gate_help_sum = 0.0
    gate_help_count = 0
    gate_suppress_sum = 0.0
    gate_suppress_count = 0
    base_gate_help_sum = 0.0
    base_gate_suppress_sum = 0.0

    for t in frames:
        runtime = loader.load(t)
        case = build_real_case(runtime)
        temporal_input = make_temporal_input(
            graph,
            runtime=runtime,
            paths=paths,
            temporal_radius=temporal_radius,
            spacing=spacing,
            device=device,
        )
        encoded = encode_case(
            model,
            runtime=runtime,
            case=case,
            temporal_input=temporal_input,
            spacing=spacing,
            device=device,
        )
        real = reason_case(model, gate_calibrator, case=case, encoded=encoded)
        update_metric_accumulator(
            real_acc, model=model, runtime=runtime, case=case, logits=real.final_logits
        )
        candidate_logits = real.candidate_logits
        update_metric_accumulator(
            candidate_acc,
            model=model,
            runtime=runtime,
            case=case,
            logits=candidate_logits,
        )

        threshold = float(model.cfg.partition.final_merge_threshold)
        target_keep_bool = case.target_keep.bool()
        spatial_keep_prediction = (
            torch.sigmoid(case.rag.spatial_edge_logits) >= threshold
        )
        candidate_keep_prediction = (
            torch.sigmoid(candidate_logits) >= threshold
        )
        spatial_correct = spatial_keep_prediction == target_keep_bool
        candidate_correct = candidate_keep_prediction == target_keep_bool
        help_mask = (
            case.editable
            & candidate_correct
            & ~spatial_correct
        )
        suppress_mask = (
            case.editable
            & ~candidate_correct
            & spatial_correct
        )
        gate = real.reasoning.edge_temporal_gate
        base_gate = real.base_gate
        if bool(help_mask.any()):
            gate_help_sum += float(
                gate[help_mask].sum().detach().float().cpu()
            )
            base_gate_help_sum += float(
                base_gate[help_mask].sum().detach().float().cpu()
            )
            gate_help_count += int(help_mask.sum().item())
        if bool(suppress_mask.any()):
            gate_suppress_sum += float(
                gate[suppress_mask].sum().detach().float().cpu()
            )
            base_gate_suppress_sum += float(
                base_gate[suppress_mask].sum().detach().float().cpu()
            )
            gate_suppress_count += int(suppress_mask.sum().item())

        for name, acc, seed in (
            ("contentless", contentless_acc, 42_100_001 + int(t)),
            ("shuffled", shuffled_acc, 42_200_001 + int(t)),
        ):
            corrupted_state = corrupt_temporal_state(
                encoded.temporal,
                corruption=name,
                seed=seed,
            )
            corrupted = reason_case(
                model,
                gate_calibrator,
                case=case,
                encoded=encoded,
                temporal=corrupted_state,
            )
            update_metric_accumulator(
                acc,
                model=model,
                runtime=runtime,
                case=case,
                logits=corrupted.final_logits,
            )

    real = finalize_metrics(real_acc)
    candidate = finalize_metrics(candidate_acc)
    contentless = finalize_metrics(contentless_acc)
    shuffled = finalize_metrics(shuffled_acc)
    control_exact = 0.5 * (
        contentless["exact_bad_component_recovery"]
        + shuffled["exact_bad_component_recovery"]
    )
    control_cut = 0.5 * (
        contentless["cut_accuracy"] + shuffled["cut_accuracy"]
    )
    score = (
        3.0 * real["cut_accuracy"]
        + 2.0 * real["exact_bad_component_recovery"]
        + 1.5 * real["keep_accuracy"]
        - 4.0 * real["clean_false_split_rate"]
        - 0.75 * control_exact
        - 0.25 * control_cut
    )
    safe = bool(
        real["cut_accuracy"] > 0.0
        and real["keep_accuracy"] >= 0.98
        and real["clean_false_split_rate"] <= 0.005
        and real["split_only_violations"] == 0
    )
    strict = bool(
        real["cut_accuracy"] >= 0.80
        and real["keep_accuracy"] >= 0.98
        and real["exact_bad_component_recovery"] >= 0.60
        and real["clean_false_split_rate"] <= 0.02
        and real["split_only_violations"] == 0
    )
    gate_help_mean = (
        gate_help_sum / max(gate_help_count, 1)
    )
    gate_suppress_mean = (
        gate_suppress_sum / max(gate_suppress_count, 1)
    )
    base_gate_help_mean = (
        base_gate_help_sum / max(gate_help_count, 1)
    )
    base_gate_suppress_mean = (
        base_gate_suppress_sum / max(gate_suppress_count, 1)
    )
    gate_selection = {
        "helpful_edges": int(gate_help_count),
        "harmful_edges": int(gate_suppress_count),
        "helpful_gate_mean": float(gate_help_mean),
        "harmful_gate_mean": float(gate_suppress_mean),
        "gate_separation": float(
            gate_help_mean - gate_suppress_mean
        ),
        "base_helpful_gate_mean": float(base_gate_help_mean),
        "base_harmful_gate_mean": float(base_gate_suppress_mean),
        "base_gate_separation": float(
            base_gate_help_mean - base_gate_suppress_mean
        ),
    }

    set_selective_gate_mode(model, gate_calibrator, True)
    return {
        "real": real,
        "candidate": candidate,
        "gate_selection": gate_selection,
        "contentless": contentless,
        "shuffled": shuffled,
        "checkpoint_score": float(score),
        "safe_pass": bool(safe),
        "strict_pass": bool(strict),
    }



def assert_candidate_invariant(
    reference: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """Frozen candidate must remain bit-for-bit identical in discrete metrics."""
    keys = (
        "cut_edges",
        "keep_edges",
        "cut_accuracy",
        "keep_accuracy",
        "bad_components",
        "exact_bad_component_recovery",
        "clean_components",
        "clean_false_split_rate",
        "split_only_violations",
    )
    mismatches = {
        key: (reference.get(key), current.get(key))
        for key in keys
        if reference.get(key) != current.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "Frozen temporal candidate changed during enhanced-gate training: "
            f"{mismatches}"
        )


def print_temporal_metrics(title: str, step: int, metrics: dict[str, Any]) -> None:
    real = metrics["real"]
    candidate = metrics.get("candidate", {})
    gate_selection = metrics.get("gate_selection", {})
    contentless = metrics["contentless"]
    shuffled = metrics["shuffled"]
    print("\n" + "=" * 112, flush=True)
    print(f"{title} @ STEP {step}", flush=True)
    print("=" * 112, flush=True)
    print(f"real CUT accuracy          : {real['cut_accuracy']:.4f}", flush=True)
    print(f"real KEEP accuracy         : {real['keep_accuracy']:.4f}", flush=True)
    print(f"real exact split recovery  : {real['exact_bad_component_recovery']:.4f}", flush=True)
    print(f"real clean false split     : {real['clean_false_split_rate']:.4f}", flush=True)
    if candidate:
        print(f"candidate CUT accuracy     : {candidate['cut_accuracy']:.4f}", flush=True)
        print(f"candidate KEEP accuracy    : {candidate['keep_accuracy']:.4f}", flush=True)
        print(
            f"candidate exact recovery   : "
            f"{candidate['exact_bad_component_recovery']:.4f}",
            flush=True,
        )
    if gate_selection:
        print(
            f"gate helpful mean          : "
            f"{gate_selection['helpful_gate_mean']:.4f} "
            f"(n={gate_selection['helpful_edges']})",
            flush=True,
        )
        print(
            f"gate harmful mean          : "
            f"{gate_selection['harmful_gate_mean']:.4f} "
            f"(n={gate_selection['harmful_edges']})",
            flush=True,
        )
        print(
            f"gate separation            : "
            f"{gate_selection['gate_separation']:.4f}",
            flush=True,
        )
        print(
            f"frozen base gate separation: "
            f"{gate_selection['base_gate_separation']:.4f}",
            flush=True,
        )
    print(f"contentless CUT accuracy   : {contentless['cut_accuracy']:.4f}", flush=True)
    print(f"contentless exact recovery : {contentless['exact_bad_component_recovery']:.4f}", flush=True)
    print(f"shuffled CUT accuracy      : {shuffled['cut_accuracy']:.4f}", flush=True)
    print(f"shuffled exact recovery    : {shuffled['exact_bad_component_recovery']:.4f}", flush=True)
    print(f"split-only violations      : {real['split_only_violations']}", flush=True)
    print(f"SAFE PASS                  : {metrics.get('safe_pass', False)}", flush=True)
    print(f"STRICT PASS                : {metrics['strict_pass']}", flush=True)
    print(f"checkpoint score           : {metrics['checkpoint_score']:.5f}", flush=True)
    print("=" * 112, flush=True)


# =============================================================================
# Dataset audit
# =============================================================================


def frame_operation_counts(operations: Sequence[dict[str, Any]]) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for op in operations:
        if "timepoint" not in op:
            continue
        t = int(op["timepoint"])
        result[t][str(op.get("type", "unknown"))] += 1
    return {t: dict(rows) for t, rows in result.items()}


def audit_dataset(
    *,
    train_frames: Sequence[int],
    val_frames: Sequence[int],
    loader: RuntimeLoader,
    operations: Sequence[dict[str, Any]],
    ignore_metrics: dict[str, Any],
    spatial_cache_metrics: dict[str, Any],
    coordinate_metrics: dict[str, Any],
) -> dict[str, Any]:
    per_frame: dict[str, Any] = {}
    totals = {
        "train": {"cut": 0, "keep": 0, "bad_components": 0, "clean_components": 0},
        "val": {"cut": 0, "keep": 0, "bad_components": 0, "clean_components": 0},
    }
    operation_counts = frame_operation_counts(operations)

    for split_name, frames in (("train", train_frames), ("val", val_frames)):
        for t in frames:
            runtime = loader.load(t)
            case = build_real_case(runtime)
            row = {
                "split": split_name,
                "cut_edges": int(case.cut_mask.sum().item()),
                "keep_edges": int(case.keep_mask.sum().item()),
                "editable_edges": int(case.editable.sum().item()),
                "valid_nodes": int(runtime.node_valid.sum().item()),
                "total_nodes": int(runtime.node_valid.numel()),
                "ignored_ids": list(runtime.ignored_ids),
                "bad_components": int(case.bad_components.numel()),
                "clean_components": int(case.clean_components.numel()),
                "operations": operation_counts.get(int(t), {}),
            }
            per_frame[str(int(t))] = row
            totals[split_name]["cut"] += row["cut_edges"]
            totals[split_name]["keep"] += row["keep_edges"]
            totals[split_name]["bad_components"] += row["bad_components"]
            totals[split_name]["clean_components"] += row["clean_components"]

    split_ops = [op for op in operations if op.get("type") == "split"]
    split_train = sum(int(op.get("timepoint", -1)) in set(train_frames) for op in split_ops)
    split_val = sum(int(op.get("timepoint", -1)) in set(val_frames) for op in split_ops)

    if totals["train"]["cut"] <= 0:
        raise RuntimeError("Training frame set contains zero trusted real CUT edges")
    if totals["val"]["cut"] <= 0:
        raise RuntimeError(
            "Validation frame set contains zero trusted real CUT edges. "
            "Choose a different --val-frames split before training."
        )

    return {
        "train_frames": list(map(int, train_frames)),
        "val_frames": list(map(int, val_frames)),
        "manual_split_operations": {
            "total": int(len(split_ops)),
            "train": int(split_train),
            "val": int(split_val),
        },
        "edge_totals": totals,
        "ignore": ignore_metrics,
        "spatial_cache": spatial_cache_metrics,
        "coordinates": coordinate_metrics,
        "per_frame": per_frame,
    }


def print_audit(audit: dict[str, Any]) -> None:
    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 42 — CURATED REAL TEMPORAL DATA AUDIT", flush=True)
    print("=" * 112, flush=True)
    print(f"train frames                 : {audit['train_frames']}", flush=True)
    print(f"validation frames            : {audit['val_frames']}", flush=True)
    print(f"manual split ops train       : {audit['manual_split_operations']['train']}", flush=True)
    print(f"manual split ops validation  : {audit['manual_split_operations']['val']}", flush=True)
    print(f"trusted CUT edges train      : {audit['edge_totals']['train']['cut']}", flush=True)
    print(f"trusted KEEP edges train     : {audit['edge_totals']['train']['keep']}", flush=True)
    print(f"trusted CUT edges validation : {audit['edge_totals']['val']['cut']}", flush=True)
    print(f"trusted KEEP edges validation: {audit['edge_totals']['val']['keep']}", flush=True)
    print(f"bad components train         : {audit['edge_totals']['train']['bad_components']}", flush=True)
    print(f"bad components validation    : {audit['edge_totals']['val']['bad_components']}", flush=True)
    print(f"ignored cells excluded       : {audit['ignore']['ignored_unique_cells']}", flush=True)
    print(f"ignored unresolved           : {audit['ignore']['ignored_unresolved_cells']}", flush=True)
    print(f"atomic SV mismatched voxels  : {audit['spatial_cache']['mismatched_voxels']}", flush=True)
    print(
        "Trackastra coordinate error  : "
        f"median={audit['coordinates']['median_error_um']:.4f}um "
        f"p95={audit['coordinates']['p95_error_um']:.4f}um "
        f"max={audit['coordinates']['max_error_um']:.4f}um",
        flush=True,
    )
    print(
        "Trackastra dropped nodes     : "
        f"{audit['coordinates'].get('total_dropped_graph_node_count', 0)}/"
        f"{audit['coordinates'].get('graph_nodes_in_relevant_frames', 0)} "
        "(outliers/unmatched nodes excluded; retained nodes use persisted centroids)",
        flush=True,
    )
    print(
        "stale production CSV rows    : "
        f"tracks={audit['coordinates'].get('stale_tracks_csv_rows', 0)} "
        f"cells={audit['coordinates'].get('stale_cells_csv_rows', 0)} "
        "(diagnostic only)",
        flush=True,
    )
    print("=" * 112, flush=True)


# =============================================================================
# Model / checkpoint
# =============================================================================


def load_model(checkpoint: Path, device: torch.device):
    return INV35.load_model(checkpoint, device)


def set_selective_gate_mode(
    model,
    gate_calibrator: SelectiveWriteGateCalibrator,
    training: bool,
) -> None:
    """Frozen base model, train/eval only the experiment-local calibrator."""
    model.eval()
    gate_calibrator.train(bool(training))


def configure_temporal_training(
    model,
    gate_calibrator: SelectiveWriteGateCalibrator,
) -> list[Tensor]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    trainable: list[Tensor] = []
    for parameter in gate_calibrator.parameters():
        parameter.requires_grad_(True)
        trainable.append(parameter)
    gate_calibrator.train(True)
    return trainable


def parameter_audit(
    model,
    gate_calibrator: SelectiveWriteGateCalibrator,
) -> dict[str, int]:
    frozen_model_trainable = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )
    if frozen_model_trainable != 0:
        raise RuntimeError(
            "Selective gate parameter audit failed: base STIR-Net has "
            f"{frozen_model_trainable} trainable parameters."
        )

    calibrator = int(
        sum(
            parameter.numel()
            for parameter in gate_calibrator.parameters()
            if parameter.requires_grad
        )
    )
    return {
        "base_stirnet": 0,
        "selective_gate_calibrator": calibrator,
        "total": calibrator,
    }


def training_config(args: argparse.Namespace) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.lr = float(args.lr)
    cfg.weight_decay = float(args.weight_decay)
    cfg.max_grad_norm = float(args.grad_clip)
    cfg.amp_dtype = str(args.amp_dtype)
    cfg.curriculum.fixed_stage = "instance_temporal"
    cfg.curriculum.instance_temporal_detached_spatial = True
    cfg.curriculum.instance_temporal_freeze_spatial = True
    cfg.loss.temporal_causal_enabled = not bool(args.no_causal)
    cfg.loss.temporal_causal_noop_weight = float(args.causal_noop_weight)
    cfg.loss.temporal_causal_corrupted_gate_weight = float(args.causal_gate_weight)
    cfg.loss.temporal_causal_margin_weight = float(args.causal_margin_weight)
    cfg.loss.temporal_causal_margin = float(args.causal_margin)
    cfg.validate()
    return cfg


def save_training_state(
    path: Path,
    *,
    gate_calibrator: SelectiveWriteGateCalibrator,
    optimizer,
    scaler,
    step: int,
    config: TrainingConfig,
    paths: Paths,
    args: argparse.Namespace,
    audit: dict[str, int],
    validation: dict[str, Any] | None,
    base_checkpoint: Path,
) -> None:
    """Persist only the small experiment-local gate head.

    The frozen step-300 STIR-Net checkpoint remains the immutable initializer.
    This avoids creating a checkpoint that pretends the experiment-local head
    is part of the production model architecture.
    """
    payload = {
        "format": "inv42_selective_write_gate_v1",
        "investigation": SCRIPT_NAME,
        "objective_version": OBJECTIVE_VERSION,
        "objective": (
            "curated_real_split_only_temporal_enhanced_gate_calibration"
        ),
        "global_step": int(step),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": sha256(base_checkpoint),
        "gate_class": "SelectiveWriteGateCalibrator",
        "gate_edge_embedding_dim": int(
            gate_calibrator.edge_embedding_dim
        ),
        "gate_correction_scale": float(
            gate_calibrator.correction_scale
        ),
        "gate_state_dict": gate_calibrator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "training_config": config,
        "args": vars(args),
        "sample_id": paths.sample,
        "split": paths.split,
        "annotation_set": paths.annotation_set,
        "parameter_audit": audit,
        "validation_metrics": validation or {},
        "notes": {
            "base_stirnet_frozen": True,
            "temporal_candidate_frozen": True,
            "production_edge_gate_frozen": True,
            "experiment_local_gate": True,
            "temporal_action": "split_only",
            "manual_tracking_overrides_used": False,
            "temporal_observer": "bypassed",
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_gate_training_state(
    path: Path,
    *,
    gate_calibrator: SelectiveWriteGateCalibrator,
    optimizer,
    scaler,
    expected_base_checkpoint: Path,
) -> int:
    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected gate checkpoint dict in {path}")
    if payload.get("format") != "inv42_selective_write_gate_v1":
        raise RuntimeError(
            f"Unsupported gate checkpoint format in {path}: "
            f"{payload.get('format')!r}"
        )
    if int(payload.get("objective_version", -1)) != OBJECTIVE_VERSION:
        raise RuntimeError(
            "Gate checkpoint objective-version mismatch: "
            f"{payload.get('objective_version')} != {OBJECTIVE_VERSION}"
        )

    expected_sha = sha256(expected_base_checkpoint)
    recorded_sha = str(payload.get("base_checkpoint_sha256", ""))
    if recorded_sha != expected_sha:
        raise RuntimeError(
            "Gate checkpoint was trained against a different frozen base "
            f"checkpoint: recorded={recorded_sha[:16]} "
            f"expected={expected_sha[:16]}"
        )

    recorded_dim = int(
        payload.get("gate_edge_embedding_dim", -1)
    )
    if recorded_dim != gate_calibrator.edge_embedding_dim:
        raise RuntimeError(
            "Gate checkpoint edge-embedding dimension mismatch: "
            f"{recorded_dim} != {gate_calibrator.edge_embedding_dim}"
        )

    gate_calibrator.load_state_dict(
        payload["gate_state_dict"],
        strict=True,
    )
    optimizer.load_state_dict(payload["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = float(group["lr"])
    try:
        scaler.load_state_dict(payload["scaler"])
    except Exception as exc:
        print(
            f"[gate resume] scaler state not restored: {exc}",
            flush=True,
        )
    return int(payload.get("global_step", 0))


# =============================================================================
# Training
# =============================================================================


def choose_training_frame(
    rng: random.Random,
    train_frames: Sequence[int],
    cut_frames: Sequence[int],
    cut_probability: float,
) -> int:
    if cut_frames and rng.random() < float(cut_probability):
        return int(rng.choice(list(cut_frames)))
    return int(rng.choice(list(train_frames)))


def train(
    *,
    paths: Paths,
    train_frames: Sequence[int],
    val_frames: Sequence[int],
    loader: RuntimeLoader,
    graph,
    spacing: Sequence[float],
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    checkpoint_path = (
        resolve(args.resume)
        if args.resume is not None
        else paths.checkpoint
    )
    checkpoint_payload, model = load_model(checkpoint_path, device)

    edge_embedding_dim = int(
        model.cfg.partition.rag_hidden_dim
    )
    gate_calibrator = SelectiveWriteGateCalibrator(
        edge_embedding_dim,
        correction_scale=float(args.gate_correction_scale),
    ).to(device)

    trainable = configure_temporal_training(
        model,
        gate_calibrator,
    )
    audit = parameter_audit(
        model,
        gate_calibrator,
    )
    config = training_config(args)

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scaler = INV35.make_grad_scaler(device, str(args.amp_dtype))

    start_step = int(
        checkpoint_payload.get("global_step", 0)
    )
    if args.gate_resume is not None:
        gate_resume = resolve(args.gate_resume)
        start_step = load_gate_training_state(
            gate_resume,
            gate_calibrator=gate_calibrator,
            optimizer=optimizer,
            scaler=scaler,
            expected_base_checkpoint=checkpoint_path,
        )
        for group in optimizer.param_groups:
            group["lr"] = float(args.lr)
            group["weight_decay"] = float(args.weight_decay)
        print(
            f"[gate resume] restored {gate_resume}; "
            f"step={start_step} lr={float(args.lr):.3g} "
            f"weight_decay={float(args.weight_decay):.3g}",
            flush=True,
        )
    else:
        print(
            "[gate] frozen base checkpoint loaded; enhanced calibrator starts "
            "as an exact zero-residual reproduction of the step-300 gate.",
            flush=True,
        )

    cut_frames = []
    for t in train_frames:
        runtime = loader.load(t)
        if build_real_case(runtime).cut_mask.any():
            cut_frames.append(int(t))

    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_strict = False
    best_safe_score = -float("inf")

    # Preserve prior run history/best selection when continuing in the same
    # output directory.  Resume must never make a worse future evaluation
    # overwrite an already-better best.pt merely because best_score reset.
    if args.gate_resume is not None:
        if paths.training_history.is_file():
            try:
                history = list(
                    json_load_list(paths.training_history)
                )
            except Exception as exc:
                print(f"[resume] training history not restored: {exc}", flush=True)
        if paths.validation_history.is_file():
            try:
                validation_history = list(
                    json_load_list(paths.validation_history)
                )
            except Exception as exc:
                print(f"[resume] validation history not restored: {exc}", flush=True)
        if paths.best_metrics.is_file():
            try:
                previous_best = json_load(paths.best_metrics)
                best_score = float(
                    previous_best.get("checkpoint_score", -float("inf"))
                )
                best_strict = bool(
                    previous_best.get("strict_pass", False)
                )
                print(
                    f"[resume] existing best preserved: score={best_score:.5f} "
                    f"strict={best_strict}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[resume] previous best metrics not restored: {exc}", flush=True)
        if paths.best_safe_metrics.is_file():
            try:
                previous_safe = json_load(paths.best_safe_metrics)
                best_safe_score = float(
                    previous_safe.get(
                        "checkpoint_score",
                        -float("inf"),
                    )
                )
                print(
                    f"[resume] existing safe best preserved: "
                    f"score={best_safe_score:.5f}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[resume] previous safe best metrics not restored: {exc}",
                    flush=True,
                )

    baseline = evaluate_spatial_baseline(model, frames=val_frames, loader=loader)
    atomic_json(paths.baseline_metrics, baseline)
    print("\n[frozen spatial validation baseline]", flush=True)
    print(json.dumps(baseline, indent=2), flush=True)

    initial_temporal = evaluate_temporal(
        model,
        gate_calibrator,
        frames=val_frames,
        loader=loader,
        graph=graph,
        paths=paths,
        temporal_radius=int(args.temporal_radius),
        spacing=spacing,
        device=device,
    )
    atomic_json(paths.initial_temporal_metrics, initial_temporal)
    print_temporal_metrics("INVESTIGATION 42 INITIAL TEMPORAL", start_step, initial_temporal)
    candidate_reference = dict(
        initial_temporal["candidate"]
    )

    initial_score = float(initial_temporal["checkpoint_score"])
    initial_strict = bool(initial_temporal["strict_pass"])
    initial_is_better = (
        (initial_strict and not best_strict)
        or (initial_strict == best_strict and initial_score > best_score)
    )
    if initial_is_better:
        best_score = initial_score
        best_strict = initial_strict

        # When resuming into a NEW output directory, there is no local best.pt
        # yet. Persist the loaded initializer as the run's baseline best so a
        # later objective phase cannot finish without a valid best checkpoint.
        # When resuming in-place and best.pt already exists, leave it untouched.
        if not paths.best.is_file():
            save_training_state(
                paths.best,
                gate_calibrator=gate_calibrator,
                optimizer=optimizer,
                scaler=scaler,
                step=start_step,
                config=config,
                paths=paths,
                args=args,
                audit=audit,
                validation=initial_temporal,
                base_checkpoint=checkpoint_path,
            )
            atomic_json(
                paths.best_metrics,
                initial_temporal,
            )
            print(
                f"[best baseline saved] step={start_step} "
                f"score={best_score:.5f}",
                flush=True,
            )
        else:
            print(
                f"[best baseline] step={start_step} score={best_score:.5f} "
                f"strict={best_strict}",
                flush=True,
            )

        if (
            bool(initial_temporal.get("safe_pass", False))
            and initial_score > best_safe_score
        ):
            best_safe_score = initial_score
            if not paths.best_safe.is_file():
                save_training_state(
                    paths.best_safe,
                    gate_calibrator=gate_calibrator,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=start_step,
                    config=config,
                    paths=paths,
                    args=args,
                    audit=audit,
                    validation=initial_temporal,
                    base_checkpoint=checkpoint_path,
                )
                atomic_json(
                    paths.best_safe_metrics,
                    initial_temporal,
                )
                print(
                    f"[best-safe baseline saved] step={start_step} "
                    f"score={best_safe_score:.5f}",
                    flush=True,
                )

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 42 — REAL CURATED TEMPORAL CUT/KEEP TRAINING", flush=True)
    print("=" * 112, flush=True)
    print(f"device                    : {device}", flush=True)
    print(f"initializer                : {checkpoint_path}", flush=True)
    print(f"base spatial checkpoint    : {paths.checkpoint}", flush=True)
    print(f"optimizer steps            : {args.steps}", flush=True)
    print(f"microcases / step          : {args.accumulate_cases}", flush=True)
    print(f"train frames               : {list(train_frames)}", flush=True)
    print(f"real-CUT train frames      : {cut_frames}", flush=True)
    print(f"validation frames          : {list(val_frames)}", flush=True)
    print(f"trainable parameters       : {audit['total']:,}", flush=True)
    print(f"CUT-frame sampling prob    : {args.cut_frame_probability:.2f}", flush=True)
    print(f"learning rate              : {optimizer.param_groups[0]['lr']:.3g}", flush=True)
    print(f"weight decay               : {optimizer.param_groups[0]['weight_decay']:.3g}", flush=True)
    print(f"preservation weight        : {args.preservation_weight:g}", flush=True)
    print("temporal candidate         : FROZEN step-300", flush=True)
    print("production edge gate       : FROZEN step-300", flush=True)
    print(
        f"enhanced gate params       : {audit['selective_gate_calibrator']:,}",
        flush=True,
    )
    print(
        f"gate correction scale      : {args.gate_correction_scale:g}",
        flush=True,
    )
    print(f"gate helpful weight        : {args.gate_help_weight:g}", flush=True)
    print(f"gate suppress weight       : {args.gate_suppress_weight:g}", flush=True)
    print(f"causal temporal objective  : {not bool(args.no_causal)}", flush=True)
    print("temporal action            : SPLIT ONLY", flush=True)
    print("manual Continue/Break input: NO", flush=True)
    print("temporal observer          : BYPASSED", flush=True)
    print("=" * 112, flush=True)

    run_started = time.perf_counter()
    for step in range(start_step + 1, int(args.steps) + 1):
        step_started = time.perf_counter()
        rng = random.Random(int(args.seed) + 15_485_863 * int(step))
        t = choose_training_frame(
            rng,
            train_frames,
            cut_frames,
            float(args.cut_frame_probability),
        )
        runtime = loader.load(t)
        case = build_real_case(runtime)
        temporal_input = make_temporal_input(
            graph,
            runtime=runtime,
            paths=paths,
            temporal_radius=int(args.temporal_radius),
            spacing=spacing,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        sums = defaultdict(float)
        corruption_rows: list[str] = []

        for micro in range(int(args.accumulate_cases)):
            corruption = ("contentless", "shuffled")[(step + micro) % 2]
            with INV35.autocast_for(device, str(args.amp_dtype)):
                encoded = encode_case(
                    model,
                    runtime=runtime,
                    case=case,
                    temporal_input=temporal_input,
                    spacing=spacing,
                    device=device,
                )
                full = reason_case(model, gate_calibrator, case=case, encoded=encoded)
                loss = training_loss(
                    model,
                    gate_calibrator,
                    case=case,
                    encoded=encoded,
                    full=full,
                    args=args,
                    rng=rng,
                    corruption=corruption,
                )
                scaled = loss.total / float(args.accumulate_cases)

            if not bool(torch.isfinite(scaled.detach())):
                raise FloatingPointError(f"Non-finite loss at step={step}, micro={micro}")
            scaler.scale(scaled).backward()
            sums["loss"] += float(loss.total.detach().float().cpu())
            sums["cut"] += float(loss.cut.detach().float().cpu())
            sums["keep"] += float(loss.keep.detach().float().cpu())
            sums["candidate_cut"] += float(
                loss.candidate_cut.detach().float().cpu()
            )
            sums["candidate_keep"] += float(
                loss.candidate_keep.detach().float().cpu()
            )
            sums["candidate"] += float(
                loss.candidate_total.detach().float().cpu()
            )
            sums["gate_help"] += float(
                loss.gate_help.detach().float().cpu()
            )
            sums["gate_suppress"] += float(
                loss.gate_suppress.detach().float().cpu()
            )
            sums["gate"] += float(
                loss.gate_total.detach().float().cpu()
            )
            sums["gate_help_edges"] += float(loss.gate_help_edges)
            sums["gate_suppress_edges"] += float(loss.gate_suppress_edges)
            sums["split"] += float(loss.split.detach().float().cpu())
            sums["causal"] += float(loss.causal.detach().float().cpu())
            sums["causal_noop"] += float(loss.causal_noop.detach().float().cpu())
            sums["causal_gate"] += float(loss.causal_gate.detach().float().cpu())
            sums["causal_margin"] += float(loss.causal_margin.detach().float().cpu())
            sums["cut_edges"] += int(loss.cut_edges)
            sums["keep_edges"] += int(loss.keep_edges)
            corruption_rows.append(loss.corruption)

        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
        grad_norm = float(torch.as_tensor(grad).detach().cpu())
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient at step={step}")
        scaler.step(optimizer)
        scaler.update()

        denom = max(int(args.accumulate_cases), 1)
        row = {
            "step": int(step),
            "frame": int(t),
            "has_real_cut": bool(case.cut_mask.any()),
            "loss": sums["loss"] / denom,
            "cut_loss": sums["cut"] / denom,
            "keep_loss": sums["keep"] / denom,
            "candidate_cut_loss": sums["candidate_cut"] / denom,
            "candidate_keep_loss": sums["candidate_keep"] / denom,
            "candidate_loss": sums["candidate"] / denom,
            "gate_help_loss": sums["gate_help"] / denom,
            "gate_suppress_loss": sums["gate_suppress"] / denom,
            "gate_loss": sums["gate"] / denom,
            "gate_help_edges": sums["gate_help_edges"] / denom,
            "gate_suppress_edges": sums["gate_suppress_edges"] / denom,
            "split_loss": sums["split"] / denom,
            "causal_loss": sums["causal"] / denom,
            "causal_noop": sums["causal_noop"] / denom,
            "causal_gate": sums["causal_gate"] / denom,
            "causal_margin": sums["causal_margin"] / denom,
            "cut_edges": int(sums["cut_edges"] / denom),
            "keep_edges": int(sums["keep_edges"] / denom),
            "corruptions": corruption_rows,
            "grad_norm": grad_norm,
            "step_seconds": float(time.perf_counter() - step_started),
        }
        history.append(row)

        if step == 1 or step % int(args.print_every) == 0:
            print(
                f"[step {step:05d}/{args.steps}] "
                f"t={t:02d} cut_case={int(row['has_real_cut'])} "
                f"loss={row['loss']:.5f} cut={row['cut_loss']:.5f} "
                f"keep={row['keep_loss']:.5f} gate={row['gate_loss']:.5f} "
                f"causal={row['causal_loss']:.5f} "
                f"grad={row['grad_norm']:.3f} "
                f"elapsed={duration(time.perf_counter() - run_started)}",
                flush=True,
            )

        if step % 50 == 0:
            atomic_json(paths.training_history, history)

        if step % int(args.eval_every) == 0 or step == int(args.steps):
            metrics = evaluate_temporal(
                model,
                gate_calibrator,
                frames=val_frames,
                loader=loader,
                graph=graph,
                paths=paths,
                temporal_radius=int(args.temporal_radius),
                spacing=spacing,
                device=device,
            )
            assert_candidate_invariant(
                candidate_reference,
                metrics["candidate"],
            )
            print_temporal_metrics("INVESTIGATION 42 VALIDATION", step, metrics)
            validation_history.append({"step": int(step), "metrics": metrics})
            atomic_json(paths.validation_history, validation_history)
            atomic_json(paths.training_history, history)

            save_training_state(
                paths.latest,
                gate_calibrator=gate_calibrator,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                config=config,
                paths=paths,
                args=args,
                audit=audit,
                validation=metrics,
                base_checkpoint=checkpoint_path,
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
                save_training_state(
                    paths.best,
                    gate_calibrator=gate_calibrator,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    config=config,
                    paths=paths,
                    args=args,
                    audit=audit,
                    validation=metrics,
                    base_checkpoint=checkpoint_path,
                )
                atomic_json(paths.best_metrics, metrics)
                print(f"[best] step={step} score={score:.5f} strict={strict}", flush=True)

            safe = bool(metrics.get("safe_pass", False))
            if safe and score > best_safe_score:
                best_safe_score = score
                save_training_state(
                    paths.best_safe,
                    gate_calibrator=gate_calibrator,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    config=config,
                    paths=paths,
                    args=args,
                    audit=audit,
                    validation=metrics,
                    base_checkpoint=checkpoint_path,
                )
                atomic_json(
                    paths.best_safe_metrics,
                    metrics,
                )
                print(
                    f"[best-safe] step={step} score={score:.5f}",
                    flush=True,
                )

    atomic_json(paths.training_history, history)
    atomic_json(paths.validation_history, validation_history)
    final_metrics = validation_history[-1]["metrics"] if validation_history else {}
    save_training_state(
        paths.final,
        gate_calibrator=gate_calibrator,
        optimizer=optimizer,
        scaler=scaler,
        step=max(start_step, int(args.steps)),
        config=config,
        paths=paths,
        args=args,
        audit=audit,
        validation=final_metrics,
        base_checkpoint=checkpoint_path,
    )

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 42 TRAINING COMPLETE", flush=True)
    print("=" * 112, flush=True)
    print(f"best gate      : {paths.best}", flush=True)
    print(f"best safe gate : {paths.best_safe}", flush=True)
    print(f"latest gate    : {paths.latest}", flush=True)
    print(f"final gate     : {paths.final}", flush=True)
    print("=" * 112, flush=True)


# =============================================================================
# CLI / main
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train STIR-Net temporal CUT/KEEP reasoning on real curated BioHub splits."
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--annotation-set", default=DEFAULT_ANNOTATION_SET)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--spacing", default=",".join(str(v) for v in DEFAULT_SPACING_ZYX_UM))

    parser.add_argument("--reviewed-frames", default=DEFAULT_REVIEWED)
    parser.add_argument("--train-frames", default=DEFAULT_TRAIN)
    parser.add_argument("--val-frames", default=DEFAULT_VAL)
    parser.add_argument("--temporal-radius", type=int, default=DEFAULT_TEMPORAL_RADIUS)

    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--spatial-cache-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--gate-resume",
        type=Path,
        default=None,
        help=(
            "Resume an Investigation-42 V11 selective-gate checkpoint. "
            "--resume must still identify the frozen base STIR-Net checkpoint."
        ),
    )

    parser.add_argument("--tile-shape-zyx", default="32,128,128")
    parser.add_argument("--tile-overlap-zyx", default="8,32,32")
    parser.add_argument("--tile-halo-zyx", default="4,16,16")
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--rebuild-spatial-cache", action="store_true")
    parser.add_argument("--keep-raw-prepare-movie", action="store_true")

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--amp-dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--accumulate-cases", type=int, default=DEFAULT_ACCUMULATE)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--print-every", type=int, default=DEFAULT_PRINT_EVERY)

    parser.add_argument("--preserve-edges", type=int, default=DEFAULT_PRESERVE_EDGES)
    parser.add_argument("--preserve-ratio", type=int, default=DEFAULT_PRESERVE_RATIO)
    parser.add_argument("--preservation-weight", type=float, default=DEFAULT_PRESERVATION_WEIGHT)
    parser.add_argument("--candidate-weight", type=float, default=DEFAULT_CANDIDATE_WEIGHT)
    parser.add_argument(
        "--candidate-preservation-weight",
        type=float,
        default=DEFAULT_CANDIDATE_PRESERVATION_WEIGHT,
    )
    parser.add_argument(
        "--gate-help-weight",
        type=float,
        default=DEFAULT_GATE_HELP_WEIGHT,
    )
    parser.add_argument(
        "--gate-suppress-weight",
        type=float,
        default=DEFAULT_GATE_SUPPRESS_WEIGHT,
    )
    parser.add_argument(
        "--gate-correction-scale",
        type=float,
        default=DEFAULT_GATE_CORRECTION_SCALE,
    )
    parser.add_argument("--split-weight", type=float, default=DEFAULT_SPLIT_WEIGHT)
    parser.add_argument("--cut-frame-probability", type=float, default=DEFAULT_CUT_FRAME_PROBABILITY)

    parser.add_argument("--no-causal", action="store_true")
    parser.add_argument("--causal-noop-weight", type=float, default=0.50)
    parser.add_argument("--causal-gate-weight", type=float, default=0.05)
    parser.add_argument("--causal-margin-weight", type=float, default=0.50)
    parser.add_argument("--causal-margin", type=float, default=1.0)

    parser.add_argument(
        "--max-coordinate-error-um",
        type=float,
        default=0.50,
        help=(
            "Maximum allowed P95 raw Trackastra-vs-canonical coordinate error. "
            "Identity is matched exactly by (time,label); isolated raw coordinate "
            "outliers are reported and replaced with canonical curation centroids."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--audit-only", action="store_true")
    return parser


def validate_args(
    args: argparse.Namespace,
    *,
    reviewed: tuple[int, ...],
    train_frames: tuple[int, ...],
    val_frames: tuple[int, ...],
) -> None:
    reviewed_set = set(reviewed)
    if not set(train_frames) <= reviewed_set or not set(val_frames) <= reviewed_set:
        raise ValueError("Train/validation frames must be subsets of --reviewed-frames")
    if set(train_frames) & set(val_frames):
        raise ValueError("Train and validation frame sets overlap")
    if args.temporal_radius < 1:
        raise ValueError("--temporal-radius must be positive")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.accumulate_cases < 1:
        raise ValueError("--accumulate-cases must be positive")
    if args.eval_every < 1 or args.print_every < 1:
        raise ValueError("--eval-every/--print-every must be positive")
    if not 0.0 <= float(args.cut_frame_probability) <= 1.0:
        raise ValueError("--cut-frame-probability must be in [0,1]")
    if float(args.candidate_weight) < 0:
        raise ValueError("--candidate-weight must be >= 0")
    if float(args.candidate_preservation_weight) < 0:
        raise ValueError("--candidate-preservation-weight must be >= 0")
    if float(args.gate_help_weight) < 0:
        raise ValueError("--gate-help-weight must be >= 0")
    if float(args.gate_suppress_weight) < 0:
        raise ValueError("--gate-suppress-weight must be >= 0")
    if float(args.gate_correction_scale) <= 0:
        raise ValueError("--gate-correction-scale must be > 0")
    if float(args.max_coordinate_error_um) <= 0:
        raise ValueError("--max-coordinate-error-um must be positive")

    # Hard temporal leakage guard: no frame may be used as temporal evidence
    # by both train and validation target windows.
    train_evidence = relevant_temporal_frames(
        train_frames,
        radius=int(args.temporal_radius),
        frame_count=max(reviewed) + int(args.temporal_radius) + 2,
    )
    val_evidence = relevant_temporal_frames(
        val_frames,
        radius=int(args.temporal_radius),
        frame_count=max(reviewed) + int(args.temporal_radius) + 2,
    )
    overlap = sorted(train_evidence & val_evidence)
    if overlap:
        raise ValueError(
            "Train/validation temporal windows overlap. Add a wider guard interval. "
            f"Shared evidence frames: {overlap}"
        )


def write_manifest(
    paths: Paths,
    *,
    reviewed: Sequence[int],
    train_frames: Sequence[int],
    val_frames: Sequence[int],
    spacing: Sequence[float],
    args: argparse.Namespace,
) -> None:
    payload = {
        "version": OBJECTIVE_VERSION,
        "investigation": SCRIPT_NAME,
        "sample_id": paths.sample,
        "split": paths.split,
        "annotation_set": paths.annotation_set,
        "reviewed_frames": list(map(int, reviewed)),
        "train_frames": list(map(int, train_frames)),
        "val_frames": list(map(int, val_frames)),
        "temporal_radius": int(args.temporal_radius),
        "spacing_zyx_um": list(map(float, spacing)),
        "base_inference_manifest": str(paths.inference_manifest),
        "annotation_manifest": str(paths.annotation_manifest),
        "spatial_checkpoint": str(paths.checkpoint),
        "spatial_checkpoint_sha256": paths.checkpoint_sha256,
        "spatial_checkpoint_step": int(paths.checkpoint_step),
        "spatial_cache": str(paths.spatial_cache),
        "track_graph": str(paths.track_graph),
        "target_contract": {
            "manual_frame_if_present": True,
            "accepted_reviewed_frame_falls_back_to_base": True,
            "ignored_cells_excluded": True,
            "unreviewed_frames_supervised": False,
        },
        "input_contract": {
            "original_spatial": True,
            "original_trackastra": True,
            "manual_track_overrides": False,
            "split_only": True,
        },
    }
    if paths.manifest.is_file():
        old = json_load(paths.manifest)
        if old != payload:
            raise RuntimeError(
                "Existing Investigation-42 manifest differs from requested dataset. "
                "Use a different --output directory or remove the old run directory."
            )
    else:
        atomic_json(paths.manifest, payload)


def main() -> int:
    args = build_parser().parse_args()
    reviewed = parse_frame_spec(args.reviewed_frames)
    train_frames = parse_frame_spec(args.train_frames)
    val_frames = parse_frame_spec(args.val_frames)
    validate_args(
        args,
        reviewed=reviewed,
        train_frames=train_frames,
        val_frames=val_frames,
    )
    spacing = parse_spacing(args.spacing)
    paths = make_paths(args)
    paths.output.mkdir(parents=True, exist_ok=True)

    if max(reviewed) >= paths.frame_count:
        raise ValueError(
            f"Reviewed frame {max(reviewed)} exceeds movie last frame {paths.frame_count - 1}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 42 — BIOHUB CURATED REAL TEMPORAL FINE-TUNING", flush=True)
    print("=" * 112, flush=True)
    print(f"repository        : {ROOT}", flush=True)
    print(f"volume            : {paths.split}/{paths.sample}", flush=True)
    print(f"annotation set    : {paths.annotation_set}", flush=True)
    print(f"reviewed frames   : {list(reviewed)}", flush=True)
    print(f"train frames      : {list(train_frames)}", flush=True)
    print(f"validation frames : {list(val_frames)}", flush=True)
    print(f"checkpoint        : {paths.checkpoint}", flush=True)
    print(f"checkpoint SHA    : {paths.checkpoint_sha256[:16]}...", flush=True)
    print(f"spatial cache     : {paths.spatial_cache}", flush=True)
    print(f"output            : {paths.output}", flush=True)
    print("=" * 112, flush=True)

    write_manifest(
        paths,
        reviewed=reviewed,
        train_frames=train_frames,
        val_frames=val_frames,
        spacing=spacing,
        args=args,
    )

    reviewed_set = set(reviewed)
    ignored_ids, ignored_records = load_ignored_ids(paths, reviewed_set)
    target_store = TargetFrameStore(paths, reviewed, ignored_ids)
    ignore_metrics = validate_ignored_ids(target_store)
    ignore_metrics["ignore_records_touching_reviewed_frames"] = int(len(ignored_records))

    # Only train/validation target frames need a compact spatial RAG cache.
    requested_cache_frames = tuple(sorted(set(train_frames) | set(val_frames)))
    prepare_spatial_cache(
        paths,
        frames=requested_cache_frames,
        spacing=spacing,
        device=device,
        args=args,
    )
    spatial_cache_metrics = validate_spatial_cache(paths, requested_cache_frames)

    graph = load_track_graph(paths)
    coordinate_metrics = audit_and_enrich_track_graph(
        graph,
        paths=paths,
        target_frames=requested_cache_frames,
        spacing=spacing,
        temporal_radius=int(args.temporal_radius),
        max_error_um=float(args.max_coordinate_error_um),
    )

    loader = RuntimeLoader(paths, target_store, device)
    operations = load_spatial_operations(paths)
    audit = audit_dataset(
        train_frames=train_frames,
        val_frames=val_frames,
        loader=loader,
        operations=operations,
        ignore_metrics=ignore_metrics,
        spatial_cache_metrics=spatial_cache_metrics,
        coordinate_metrics=coordinate_metrics,
    )
    atomic_json(paths.audit_json, audit)
    print_audit(audit)

    if args.audit_only:
        print(
            "Audit passed. No training was started because --audit-only was supplied.",
            flush=True,
        )
        return 0

    train(
        paths=paths,
        train_frames=train_frames,
        val_frames=val_frames,
        loader=loader,
        graph=graph,
        spacing=spacing,
        device=device,
        args=args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
