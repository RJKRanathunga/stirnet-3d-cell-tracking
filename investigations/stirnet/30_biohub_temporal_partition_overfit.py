from __future__ import annotations

r"""
Investigation 30 — real BioHub temporal-only partition overfit.

Scientific question
-------------------
Can STIR-Net's temporal branch learn to split the comparatively rare merged
instances that were manually corrected across the 20-frame BioHub movie,
WITHOUT rerunning or training the spatial CNN / watershed / spatial RAG?

This investigation deliberately freezes the current spatial result at the
artifact level:

    Investigation 25:
        current instances
        tXXX/partition/after_split_only.npy

    Investigation 24:
        atomic supervoxels
        tXXX/partition/watershed_supervoxels.npy

    Investigation 12:
        saved h100 RAG topology + edge probabilities
        tXXX/rag/rag_state.npz

No spatial model is instantiated anywhere in this file.

Manual supervision
------------------
The canonical targets are:

    evaluation/segmentation/annotations/<sample>/
        manual_instances_t000.npy
        ...
        manual_instances_t019.npy

The manual annotations intentionally preserve the strong spatial result except
where merged instances were split. A very small number of hallucinated cells
may remain. Therefore this first temporal experiment is SPLIT/PARTITION ONLY.

It does NOT supervise:
    - instance existence / hallucination deletion,
    - birth/death correctness,
    - recovery of missing cells,
    - lineage/division correctness.

Only RAG edges whose endpoints already belong to the SAME Investigation-25
instance are supervised:

    same manual instance  -> KEEP edge
    different manual IDs  -> CUT edge

Cross-base-instance edges are ignored by the loss and are never allowed to
merge components during evaluation. This prevents the pseudo-GT from silently
teaching unannotated false-split corrections.

Temporal inputs
---------------
If a cached Trackastra movie/graph is unavailable, this script runs Trackastra
ONCE on the frozen Investigation-25 instance movie and caches its output.
That is temporal preprocessing, not spatial inference.

For every target frame, a target-relative temporal graph is then built with the
patched finite-window availability contract. With radius=2:

    t=0  :  0 +1 +2
    t=1  : -1  0 +1 +2
    t=2  : -2 -1  0 +1 +2
    ...
    t=18 : -2 -1  0 +1
    t=19 : -2 -1  0

The graph is built from Trackastra output/current segmentation only. Manual
annotations are NEVER used to construct temporal evidence.

Trainable modules
-----------------
ONLY temporal-side production modules are optimized:

    HistoricalInstanceEncoder
    TemporalGraphEncoder
    InstanceTemporalReasoner

The production reasoner expects 96-D RAG node/edge embeddings and 128-D
instance tokens. Investigation-12's compact rag_state.npz intentionally stores
topology/probabilities rather than the full learned embedding tensors, so this
script supplies deterministic, NON-TRAINABLE projections of cached spatial
geometry into those widths. They are buffers, not parameters.

This keeps the experiment temporal-only. Importantly, an EMPTY temporal state
still produces the exact frozen spatial edge logits because the production
reasoner gates temporal writes to zero when temporal support is absent.

Evaluation / causal checks
--------------------------
The script evaluates:

1. FULL TEMPORAL
   Normal learned temporal content.

2. EMPTY TEMPORAL
   Strict no-temporal ablation. final_edge_logits must equal the frozen spatial
   logits exactly. If this fails, the experiment is invalid.

3. SHUFFLED TEMPORAL CONTENT
   Tracklet tokens/salience/reliability are permuted across the fixed physical
   tracklet reference positions. If FULL beats SHUFFLED, the correction is
   using temporally localized content rather than just "temporal support exists".

Reported metrics include:
    - manual cut-edge recall,
    - keep-edge preservation,
    - exact recovery of manually split base components,
    - accidental split rate on unchanged base components,
    - temporal gate on CUT vs KEEP edges,
    - FULL vs EMPTY vs SHUFFLED comparison.

This is an OVERFIT experiment. Success proves capacity and temporal-path
functionality on the 20 annotated frames; it does not prove generalization.

Typical usage
-------------
From repository root:

    python .\investigations\stirnet\30_biohub_temporal_partition_overfit.py

Shorter first smoke run:

    python .\investigations\stirnet\30_biohub_temporal_partition_overfit.py ^
        --steps 300 --eval-every 100

Force rebuilding Trackastra + temporal-v4 caches:

    python .\investigations\stirnet\30_biohub_temporal_partition_overfit.py ^
        --rebuild-trackastra --rebuild-temporal

Use the complete O(N^2) detection candidate graph instead of the lighter
accepted-association + same-frame-neighbour graph:

    python .\investigations\stirnet\30_biohub_temporal_partition_overfit.py ^
        --complete-candidate-graph

The lighter graph is the default for this full 20-frame overfit because it
preserves Trackastra temporal links, local same-frame context, the 22-D
tracklet hypothesis graph, and historical-instance evidence without creating
~N^2 detection edges for every target frame.
"""

import argparse
import dataclasses
import inspect
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


# =============================================================================
# REPOSITORY / PRODUCTION IMPORTS
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

    raise RuntimeError("Could not resolve cell-tracking repository root.")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from learned.stirnet.data.sample_builder import robust_normalize
from learned.stirnet.data.targets import (
    estimate_model_dref_um,
    extract_instance_metadata,
)
from learned.stirnet.data.trackastra_cache import load_cache, save_cache
from learned.stirnet.model.config import ModelConfig
from learned.stirnet.model.temporal.fusion import InstanceTemporalReasoner
from learned.stirnet.model.temporal.graph_encoder import TemporalGraphEncoder
from learned.stirnet.model.temporal.history import HistoricalInstanceEncoder
from learned.stirnet.model.types import (
    InstanceState,
    RAGState,
    TemporalInput,
    TemporalState,
)


# =============================================================================
# DEFAULTS
# =============================================================================

SCRIPT_NAME = "30_biohub_temporal_partition_overfit"
DEFAULT_SAMPLE_ID = "44b6_0113de3b"
DEFAULT_FRAME_COUNT = 20
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)
DEFAULT_TEMPORAL_RADIUS = 2

DEFAULT_INV12_ROOT = (
    ROOT
    / "runs"
    / "stirnet"
    / "evaluation"
    / "12_biohub_full_volume_spatial_inference"
)
DEFAULT_INV24_ROOT = (
    ROOT
    / "runs"
    / "stirnet"
    / "evaluation"
    / "24_multicut_biohub_full_volume_visualization"
)
DEFAULT_INV25_ROOT = (
    ROOT
    / "runs"
    / "stirnet"
    / "evaluation"
    / "25_source_core_split_biohub_visualization"
)
DEFAULT_ANNOTATION_ROOT = (
    ROOT / "evaluation" / "segmentation" / "annotations"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "runs" / "stirnet" / "evaluation" / SCRIPT_NAME
)

DEFAULT_INV12_VARIANT = Path("morphology_v2_h100") / "step000600"
DEFAULT_INV24_VARIANT = "h100_q0p845"
DEFAULT_INV25_VARIANT = "source_instance_anchors_supervoxel_graph_defaults"

DEFAULT_STEPS = 2500
DEFAULT_LR = 2.0e-3
DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_EVAL_EVERY = 250
DEFAULT_CORRECTION_FRAME_PROB = 0.70
DEFAULT_CLEAN_KEEP_EDGES = 512
DEFAULT_KEEP_TO_CUT_RATIO = 12
DEFAULT_SPLIT_LOSS_WEIGHT = 0.20
DEFAULT_KEEP_GATE_WEIGHT = 0.03
DEFAULT_GRAD_CLIP = 5.0

# This is intentionally a partition-consistent prior rather than the raw h100
# probability. Investigation 25 is the frozen "current spatial answer", so all
# RAG connections inside one current instance begin as confident KEEP and all
# cross-instance connections begin as confident CUT. The raw h100 probability
# is still supplied as a cached edge feature.
DEFAULT_SPATIAL_PRIOR_LOGIT = 3.5

FIXED_PROJECTION_SEED = 20260826


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return _jsonable(value.detach().cpu().item())
        return [_jsonable(v) for v in value.detach().cpu().tolist()]
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def probability_to_logit_np(
    probability: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    p = np.clip(np.asarray(probability, np.float32), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


# =============================================================================
# PATHS
# =============================================================================


@dataclass(frozen=True)
class Paths:
    sample_id: str
    inv12: Path
    inv24: Path
    inv25: Path
    annotations: Path
    zarr: Path
    output: Path
    trackastra_dir: Path
    temporal_cache_dir: Path

    def rag_state(self, t: int) -> Path:
        return self.inv12 / f"t{t:03d}" / "rag" / "rag_state.npz"

    def supervoxels(self, t: int) -> Path:
        return (
            self.inv24
            / f"t{t:03d}"
            / "partition"
            / "watershed_supervoxels.npy"
        )

    def base_instances(self, t: int) -> Path:
        return (
            self.inv25
            / f"t{t:03d}"
            / "partition"
            / "after_split_only.npy"
        )

    def manual_instances(self, t: int) -> Path:
        return self.annotations / f"manual_instances_t{t:03d}.npy"

    def temporal_cache(self, t: int) -> Path:
        return self.temporal_cache_dir / f"temporal_t{t:03d}.pt"


def make_paths(args: argparse.Namespace) -> Paths:
    sample = args.sample_id

    inv12 = (
        resolve(args.inv12)
        if args.inv12 is not None
        else (
            DEFAULT_INV12_ROOT
            / sample
            / DEFAULT_INV12_VARIANT
        ).resolve()
    )
    inv24 = (
        resolve(args.inv24)
        if args.inv24 is not None
        else (
            DEFAULT_INV24_ROOT
            / sample
            / DEFAULT_INV24_VARIANT
        ).resolve()
    )
    inv25 = (
        resolve(args.inv25)
        if args.inv25 is not None
        else (
            DEFAULT_INV25_ROOT
            / sample
            / DEFAULT_INV25_VARIANT
        ).resolve()
    )
    annotations = (
        resolve(args.annotations)
        if args.annotations is not None
        else (DEFAULT_ANNOTATION_ROOT / sample).resolve()
    )
    zarr = (
        resolve(args.zarr)
        if args.zarr is not None
        else (
            ROOT
            / "data"
            / "sample"
            / "biohub_5samples_20timepoints"
            / "train"
            / sample
            / f"{sample}.zarr"
        ).resolve()
    )
    output = (
        resolve(args.output)
        if args.output is not None
        else (DEFAULT_OUTPUT_ROOT / sample).resolve()
    )

    return Paths(
        sample_id=sample,
        inv12=inv12,
        inv24=inv24,
        inv25=inv25,
        annotations=annotations,
        zarr=zarr,
        output=output,
        trackastra_dir=output / "trackastra",
        temporal_cache_dir=output / "temporal_v4",
    )


def validate_required_artifacts(
    paths: Paths,
    frame_count: int,
) -> None:
    missing: list[Path] = []
    for t in range(frame_count):
        for path in (
            paths.rag_state(t),
            paths.supervoxels(t),
            paths.base_instances(t),
            paths.manual_instances(t),
        ):
            if not path.is_file():
                missing.append(path)

    if not paths.zarr.exists():
        missing.append(paths.zarr)

    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:40])
        suffix = (
            f"\n  ... and {len(missing) - 40} more"
            if len(missing) > 40
            else ""
        )
        raise FileNotFoundError(
            "Investigation-30 required artifacts are missing:\n"
            + preview
            + suffix
        )


# =============================================================================
# FROZEN MOVIE PREPARATION
# =============================================================================


def assemble_base_instance_movie(
    paths: Paths,
    frame_count: int,
    *,
    rebuild: bool,
) -> Path:
    output = paths.output / "frozen_base_instance_movie.npy"
    if output.is_file() and not rebuild:
        movie = np.load(output, mmap_mode="r")
        if movie.shape[0] == frame_count:
            return output

    first = np.load(paths.base_instances(0), mmap_mode="r", allow_pickle=False)
    shape = (frame_count, *first.shape)
    movie = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=first.dtype,
        shape=shape,
    )

    for t in range(frame_count):
        frame = np.asarray(
            np.load(
                paths.base_instances(t),
                mmap_mode="r",
                allow_pickle=False,
            )
        )
        if frame.shape != first.shape:
            raise ValueError(
                f"Base instance shape changed at t={t}: "
                f"{frame.shape} vs {first.shape}"
            )
        movie[t] = frame

    movie.flush()
    del movie
    return output


def assemble_raw_movie(
    paths: Paths,
    frame_count: int,
    expected_shape: tuple[int, int, int],
    *,
    rebuild: bool,
) -> Path:
    output = paths.output / "raw_movie.npy"
    if output.is_file() and not rebuild:
        movie = np.load(output, mmap_mode="r")
        if movie.shape == (frame_count, *expected_shape):
            return output

    first = np.asarray(load_timepoint(paths.zarr, 0))
    if first.shape != expected_shape:
        raise ValueError(
            f"Raw/base shape mismatch: raw={first.shape}, base={expected_shape}"
        )

    movie = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=first.dtype,
        shape=(frame_count, *first.shape),
    )
    movie[0] = first

    for t in range(1, frame_count):
        print(f"[raw] loading t={t}", flush=True)
        frame = np.asarray(load_timepoint(paths.zarr, t))
        if frame.shape != expected_shape:
            raise ValueError(
                f"Raw shape changed at t={t}: {frame.shape} vs {expected_shape}"
            )
        movie[t] = frame

    movie.flush()
    del movie
    return output


def resolve_movie_dref_um(
    base_movie: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    values = []
    for t in range(base_movie.shape[0]):
        value = estimate_model_dref_um(
            np.asarray(base_movie[t]),
            spacing,
        )
        if np.isfinite(value) and value > 0:
            values.append(float(value))
    if not values:
        raise RuntimeError("Could not estimate model dref from current segmentation")
    return float(np.median(np.asarray(values, np.float64)))


# =============================================================================
# TRACKASTRA
# =============================================================================


def prepare_trackastra(
    paths: Paths,
    raw_path: Path,
    base_path: Path,
    *,
    model_name: str,
    mode: str,
    device: str,
    rebuild: bool,
) -> tuple[Path, Path]:
    paths.trackastra_dir.mkdir(parents=True, exist_ok=True)
    graph_path = paths.trackastra_dir / "track_graph.pkl"
    masks_path = paths.trackastra_dir / "tracked_masks.npy"

    if graph_path.is_file() and masks_path.is_file() and not rebuild:
        print("[trackastra] reusing cached graph/masks")
        return graph_path, masks_path

    try:
        from trackastra.model import Trackastra
    except ImportError as exc:
        raise RuntimeError(
            "Trackastra cache is absent and trackastra is not importable. "
            "Install/activate the same Trackastra environment used elsewhere "
            "in this repository, or provide the cached Investigation-30 "
            "trackastra/track_graph.pkl + tracked_masks.npy."
        ) from exc

    raw_movie = np.load(raw_path, mmap_mode="r")
    base_movie = np.load(base_path, mmap_mode="r")

    print()
    print("=" * 100)
    print("Investigation 30 — Trackastra preprocessing")
    print("=" * 100)
    print(f"model  : {model_name}")
    print(f"mode   : {mode}")
    print(f"device : {device}")
    print(f"raw    : {raw_path}")
    print(f"masks  : {base_path}")
    print("=" * 100)

    started = time.perf_counter()
    model = Trackastra.from_pretrained(model_name, device=device)
    track_graph, tracked_masks = model.track(
        raw_movie,
        base_movie,
        mode=mode,
    )

    with graph_path.open("wb") as handle:
        pickle.dump(track_graph, handle)

    np.save(
        masks_path,
        np.asarray(tracked_masks),
        allow_pickle=False,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"[trackastra] nodes={track_graph.number_of_nodes()} "
        f"edges={track_graph.number_of_edges()} "
        f"time={format_seconds(time.perf_counter() - started)}"
    )
    print(f"[trackastra] graph : {graph_path}")
    print(f"[trackastra] masks : {masks_path}")
    return graph_path, masks_path


# =============================================================================
# TEMPORAL CACHE BUILDING
# =============================================================================


def _mean_track_velocity(
    track_graph,
    positions_abs_um: dict[int, np.ndarray],
    node_id: int,
    neighbours: list[int],
    *,
    forward: bool,
) -> np.ndarray:
    if not neighbours:
        return np.zeros(3, dtype=np.float32)

    time0 = int(track_graph.nodes[node_id]["time"])
    pos0 = positions_abs_um[node_id]
    rows = []

    for other in neighbours:
        other = int(other)
        dt = abs(int(track_graph.nodes[other]["time"]) - time0)
        if dt <= 0:
            continue
        delta = positions_abs_um[other] - pos0
        rows.append((delta if forward else -delta) / float(dt))

    if not rows:
        return np.zeros(3, dtype=np.float32)
    return np.mean(np.stack(rows), axis=0).astype(np.float32)


def build_static_detection_records(
    track_graph,
    tracked_movie: np.ndarray,
    raw_movie: np.ndarray,
    spacing: tuple[float, float, float],
    dref_um: float,
) -> dict[int, DetectionRecord]:
    """Build each Trackastra detection once in full-volume centered coordinates."""
    spacing_np = np.asarray(spacing, dtype=np.float32)
    volume_shape = np.asarray(tracked_movie.shape[-3:], dtype=np.float32)
    center_um = 0.5 * (volume_shape - 1.0) * spacing_np

    positions_abs_um = {
        int(node_id): (
            np.asarray(node["coords"], dtype=np.float32) * spacing_np
        )
        for node_id, node in track_graph.nodes(data=True)
    }

    nodes_by_time: dict[int, list[int]] = {}
    for node_id, node in track_graph.nodes(data=True):
        nodes_by_time.setdefault(int(node["time"]), []).append(int(node_id))

    result: dict[int, DetectionRecord] = {}

    for t in range(tracked_movie.shape[0]):
        labels = np.asarray(tracked_movie[t]).astype(np.int32, copy=False)
        raw_norm = robust_normalize(np.asarray(raw_movie[t]))

        print(
            f"[temporal metadata] frame {t + 1}/{tracked_movie.shape[0]} "
            f"track nodes={len(nodes_by_time.get(t, []))}",
            flush=True,
        )

        metadata = extract_instance_metadata(
            labels,
            raw_norm,
            spacing,
            dref_um,
            np.zeros(labels.shape, dtype=np.float32),
        )
        ids = metadata.ids.cpu().numpy()
        features = metadata.features.cpu().numpy()
        id_to_row = {
            int(instance_id): row
            for row, instance_id in enumerate(ids.tolist())
        }

        for node_id in nodes_by_time.get(t, []):
            node = track_graph.nodes[node_id]
            label_id = int(node["label"])

            if label_id not in id_to_row:
                # A graph/mask mismatch should be rare. Refuse silently
                # fabricating metadata for a missing component.
                continue

            row = id_to_row[label_id]
            feature = features[row]
            coords_vox = np.asarray(node["coords"], dtype=np.float32)
            coords_abs_um = coords_vox * spacing_np
            position_centered_um = coords_abs_um - center_um

            lower_um = coords_vox * spacing_np
            upper_um = (volume_shape - 1.0 - coords_vox) * spacing_np
            distance_boundary_um = float(
                np.min(np.concatenate([lower_um, upper_um]))
            )

            component_voxels = int(np.count_nonzero(labels == label_id))
            predecessors = [
                int(value)
                for value in track_graph.predecessors(node_id)
            ]
            successors = [
                int(value)
                for value in track_graph.successors(node_id)
            ]

            history_grid, history_valid = build_historical_instance_grid(
                raw_norm,
                labels,
                label_id,
                spacing,
                dref_um,
                center_um=coords_abs_um,
            )

            result[node_id] = DetectionRecord(
                node_id=node_id,
                time_offset=0,  # target-relative offset is set later
                position_um=tuple(position_centered_um.tolist()),
                physical_volume_um3=(
                    component_voxels * float(np.prod(spacing_np))
                ),
                bbox_um=tuple(
                    (feature[1:4] * dref_um).tolist()
                ),
                pca_axes_um=tuple(
                    (feature[4:7] * dref_um).tolist()
                ),
                elongation=float(feature[7]),
                flatness=float(feature[8]),
                solidity=float(feature[9]),
                compactness=float(feature[10]),
                intensity_mean=float(feature[11]),
                intensity_std=float(feature[12]),
                backward_velocity_um=tuple(
                    _mean_track_velocity(
                        track_graph,
                        positions_abs_um,
                        node_id,
                        predecessors,
                        forward=False,
                    ).tolist()
                ),
                forward_velocity_um=tuple(
                    _mean_track_velocity(
                        track_graph,
                        positions_abs_um,
                        node_id,
                        successors,
                        forward=True,
                    ).tolist()
                ),
                distance_to_volume_boundary_um=distance_boundary_um,
                distance_to_patch_boundary_um=distance_boundary_um,
                boundary_related=(distance_boundary_um <= 4.0),
                instance_grid=history_grid,
                history_valid=bool(history_valid),
            )

    missing = track_graph.number_of_nodes() - len(result)
    if missing:
        print(
            f"[temporal metadata] WARNING: skipped {missing} Trackastra nodes "
            "whose label was absent from tracked_masks."
        )

    return result


def trackastra_associations(track_graph) -> list[AssociationRecord]:
    rows = []
    for source, destination, edge_data in track_graph.edges(data=True):
        source = int(source)
        destination = int(destination)
        score = edge_data.get("weight")
        rows.append(
            AssociationRecord(
                src_node_id=source,
                dst_node_id=destination,
                score=None if score is None else float(score),
                relation=(
                    "division"
                    if track_graph.out_degree(source) > 1
                    else "temporal"
                ),
            )
        )
    return rows


def build_temporal_caches(
    paths: Paths,
    *,
    track_graph,
    tracked_movie: np.ndarray,
    raw_movie: np.ndarray,
    frame_count: int,
    spacing: tuple[float, float, float],
    dref_um: float,
    temporal_radius: int,
    complete_candidate_graph: bool,
    rebuild: bool,
) -> None:
    paths.temporal_cache_dir.mkdir(parents=True, exist_ok=True)

    existing = [
        paths.temporal_cache(t).is_file()
        for t in range(frame_count)
    ]
    if all(existing) and not rebuild:
        # Validate one cache now; load_cache will reject pre-v4 artifacts.
        load_cache(paths.temporal_cache(0))
        print(
            f"[temporal cache] reusing {frame_count} contract-v"
            f"{TEMPORAL_CACHE_CONTRACT_VERSION} target caches"
        )
        return

    signature = inspect.signature(build_temporal_graph)
    if "available_time_offsets" not in signature.parameters:
        raise RuntimeError(
            "Your finite-window temporal patch is not present: "
            "build_temporal_graph() has no available_time_offsets parameter."
        )

    static_records = build_static_detection_records(
        track_graph,
        tracked_movie,
        raw_movie,
        spacing,
        dref_um,
    )
    associations = trackastra_associations(track_graph)

    nodes_by_time: dict[int, list[int]] = {}
    for node_id, node in track_graph.nodes(data=True):
        nodes_by_time.setdefault(int(node["time"]), []).append(int(node_id))

    for target_t in range(frame_count):
        cache_path = paths.temporal_cache(target_t)
        if cache_path.is_file() and not rebuild:
            try:
                load_cache(cache_path)
                print(f"[temporal cache] reuse t={target_t}")
                continue
            except Exception:
                pass

        available = sequence_available_time_offsets(
            target_t,
            frame_count,
            temporal_radius,
        )
        absolute_times = [
            target_t + offset
            for offset in available
        ]

        records: list[DetectionRecord] = []
        selected_node_ids: set[int] = set()

        for absolute_t in absolute_times:
            for node_id in nodes_by_time.get(absolute_t, []):
                base = static_records.get(node_id)
                if base is None:
                    continue
                selected_node_ids.add(node_id)
                records.append(
                    dataclasses.replace(
                        base,
                        time_offset=absolute_t - target_t,
                    )
                )

        selected_associations = [
            row
            for row in associations
            if (
                row.src_node_id in selected_node_ids
                and row.dst_node_id in selected_node_ids
            )
        ]

        print(
            f"[temporal cache] build t={target_t:02d} "
            f"available={available} detections={len(records)} "
            f"associations={len(selected_associations)}",
            flush=True,
        )

        graph = build_temporal_graph(
            records,
            selected_associations,
            dref_um=dref_um,
            temporal_radius=temporal_radius,
            available_time_offsets=available,
            k_spatial_neighbors=6,
            spatial_radius_dref=2.5,
            current_labels=np.asarray(tracked_movie[target_t]),
            spacing_um=spacing,
            candidate_graph_enabled=complete_candidate_graph,
        )
        graph["temporal_batch"] = torch.zeros(
            len(graph["temporal_ref_um"]),
            dtype=torch.long,
        )
        graph["target_time_index"] = int(target_t)
        graph["available_time_offsets"] = torch.tensor(
            available,
            dtype=torch.int64,
        )
        save_cache(cache_path, graph)

    print(
        f"[temporal cache] ready: {paths.temporal_cache_dir}"
    )


# =============================================================================
# SPATIAL CASE EXTRACTION — ARTIFACTS ONLY, NO MODEL
# =============================================================================


def sv_label_lookup(
    supervoxels: np.ndarray,
    labels: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Return label[sv_id], requiring each positive atomic SV to be pure."""
    sv = np.asarray(supervoxels, dtype=np.int64).reshape(-1)
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    if sv.shape != lab.shape:
        raise ValueError(f"{name}: supervoxel/label shape mismatch")

    max_sv = int(sv.max(initial=0))
    minimum = np.full(max_sv + 1, np.iinfo(np.int64).max, dtype=np.int64)
    maximum = np.full(max_sv + 1, -1, dtype=np.int64)

    positive = sv > 0
    np.minimum.at(minimum, sv[positive], lab[positive])
    np.maximum.at(maximum, sv[positive], lab[positive])

    present = maximum >= 0
    impure = present & (minimum != maximum)
    if np.any(impure):
        bad = np.flatnonzero(impure)[:20].tolist()
        raise RuntimeError(
            f"{name}: labels split atomic supervoxels; examples={bad}"
        )

    result = np.zeros(max_sv + 1, dtype=np.int64)
    result[present] = maximum[present]
    return result


def sv_voxel_counts(
    supervoxels: np.ndarray,
    max_sv: int,
) -> np.ndarray:
    return np.bincount(
        np.asarray(supervoxels, dtype=np.int64).reshape(-1),
        minlength=max_sv + 1,
    ).astype(np.int64, copy=False)


def compute_sv_centroids_um(
    supervoxels: np.ndarray,
    node_sv: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """Fallback if compact rag_state lacks node_centroid_um."""
    labels = np.asarray(supervoxels, dtype=np.int64)
    spacing_np = np.asarray(spacing, dtype=np.float64)
    shape = np.asarray(labels.shape, dtype=np.float64)
    center_vox = 0.5 * (shape - 1.0)
    max_sv = int(labels.max(initial=0))

    flat = labels.reshape(-1)
    positive = flat > 0
    flat_pos = flat[positive]

    z, y, x = np.indices(labels.shape, sparse=False)
    coordinates = (
        z.reshape(-1)[positive],
        y.reshape(-1)[positive],
        x.reshape(-1)[positive],
    )
    count = np.bincount(flat_pos, minlength=max_sv + 1).astype(np.float64)

    sums = []
    for values in coordinates:
        sums.append(
            np.bincount(
                flat_pos,
                weights=values.astype(np.float64),
                minlength=max_sv + 1,
            )
        )
    centroid_vox = np.stack(sums, axis=1) / np.maximum(count[:, None], 1.0)
    centered_um = (centroid_vox - center_vox[None]) * spacing_np[None]
    return centered_um[node_sv].astype(np.float32)


@dataclass
class SpatialFrameCase:
    t: int
    shape_zyx: tuple[int, int, int]
    supervoxels: np.ndarray
    base_labels: np.ndarray
    manual_labels: np.ndarray

    node_sv: Tensor
    node_centroid_um: Tensor
    node_volume_voxels: Tensor

    edge_index: Tensor
    raw_spatial_probability: Tensor
    spatial_edge_logits: Tensor

    node_base_label: Tensor
    node_manual_label: Tensor
    node_instance_index: Tensor
    instance_base_ids: Tensor
    instance_split_target: Tensor

    eligible_edge_mask: Tensor
    keep_edge_mask: Tensor
    cut_edge_mask: Tensor
    edge_base_label: Tensor
    changed_base_ids: tuple[int, ...]

    @property
    def cut_count(self) -> int:
        return int(self.cut_edge_mask.sum().item())

    @property
    def keep_count(self) -> int:
        return int(self.keep_edge_mask.sum().item())

    @property
    def changed_component_count(self) -> int:
        return len(self.changed_base_ids)


def load_spatial_case(
    paths: Paths,
    t: int,
    spacing: tuple[float, float, float],
    spatial_prior_logit: float,
) -> SpatialFrameCase:
    supervoxels = np.asarray(
        np.load(
            paths.supervoxels(t),
            mmap_mode="r",
            allow_pickle=False,
        )
    )
    base = np.asarray(
        np.load(
            paths.base_instances(t),
            mmap_mode="r",
            allow_pickle=False,
        )
    )
    manual = np.asarray(
        np.load(
            paths.manual_instances(t),
            mmap_mode="r",
            allow_pickle=False,
        )
    )

    if not (
        supervoxels.shape == base.shape == manual.shape
    ):
        raise ValueError(
            f"t={t}: shape mismatch "
            f"sv={supervoxels.shape}, base={base.shape}, manual={manual.shape}"
        )

    if not np.array_equal(base > 0, manual > 0):
        difference = int(np.count_nonzero((base > 0) != (manual > 0)))
        raise RuntimeError(
            f"t={t}: manual annotation changed foreground support at "
            f"{difference} voxels. Investigation 30 expects split-only labels."
        )

    base_by_sv = sv_label_lookup(
        supervoxels,
        base,
        name=f"t={t} base",
    )
    manual_by_sv = sv_label_lookup(
        supervoxels,
        manual,
        name=f"t={t} manual",
    )

    with np.load(paths.rag_state(t), allow_pickle=False) as rag:
        required = (
            "node_supervoxel_id",
            "edge_index",
            "spatial_edge_probability",
        )
        missing = [key for key in required if key not in rag]
        if missing:
            raise KeyError(
                f"t={t} rag_state.npz missing required keys: {missing}"
            )

        node_sv_np = np.asarray(
            rag["node_supervoxel_id"],
            dtype=np.int64,
        ).reshape(-1)
        edge_index_np = np.asarray(
            rag["edge_index"],
            dtype=np.int64,
        )
        probability_np = np.asarray(
            rag["spatial_edge_probability"],
            dtype=np.float32,
        ).reshape(-1)
        centroid_np = (
            np.asarray(
                rag["node_centroid_um"],
                dtype=np.float32,
            )
            if "node_centroid_um" in rag
            else None
        )

    if edge_index_np.ndim != 2 or edge_index_np.shape[0] != 2:
        raise ValueError(
            f"t={t}: RAG edge_index must have shape [2,E]"
        )
    if edge_index_np.shape[1] != len(probability_np):
        raise ValueError(
            f"t={t}: RAG edge/probability length mismatch"
        )
    if node_sv_np.size and (
        node_sv_np.min() <= 0
        or node_sv_np.max() >= len(base_by_sv)
    ):
        raise RuntimeError(
            f"t={t}: RAG node supervoxel IDs do not align with "
            "Investigation-24 watershed IDs."
        )

    if centroid_np is None:
        centroid_np = compute_sv_centroids_um(
            supervoxels,
            node_sv_np,
            spacing,
        )

    counts = sv_voxel_counts(
        supervoxels,
        int(supervoxels.max(initial=0)),
    )
    node_volume_np = counts[node_sv_np].astype(np.float32)

    node_base_np = base_by_sv[node_sv_np]
    node_manual_np = manual_by_sv[node_sv_np]

    if np.any(node_base_np <= 0):
        bad = node_sv_np[node_base_np <= 0][:20].tolist()
        raise RuntimeError(
            f"t={t}: positive RAG nodes outside current base instances: {bad}"
        )
    if np.any(node_manual_np <= 0):
        bad = node_sv_np[node_manual_np <= 0][:20].tolist()
        raise RuntimeError(
            f"t={t}: positive RAG nodes outside manual foreground: {bad}"
        )

    base_ids = np.unique(node_base_np)
    base_ids = base_ids[base_ids > 0]
    base_to_index = {
        int(label): index
        for index, label in enumerate(base_ids.tolist())
    }
    node_instance_np = np.asarray(
        [base_to_index[int(label)] for label in node_base_np],
        dtype=np.int64,
    )

    split_target_np = np.zeros(len(base_ids), dtype=np.float32)
    changed_base_ids: list[int] = []

    for index, base_id in enumerate(base_ids.tolist()):
        manual_ids = np.unique(
            node_manual_np[node_base_np == int(base_id)]
        )
        manual_ids = manual_ids[manual_ids > 0]
        if len(manual_ids) > 1:
            split_target_np[index] = 1.0
            changed_base_ids.append(int(base_id))

    src = edge_index_np[0]
    dst = edge_index_np[1]
    same_base = node_base_np[src] == node_base_np[dst]
    same_manual = node_manual_np[src] == node_manual_np[dst]

    eligible = (
        same_base
        & (node_base_np[src] > 0)
        & (node_manual_np[src] > 0)
        & (node_manual_np[dst] > 0)
    )
    keep = eligible & same_manual
    cut = eligible & ~same_manual

    # Exact frozen current-partition prior. Raw h100 probability remains
    # available as a fixed edge embedding feature.
    spatial_logits = np.where(
        same_base,
        float(spatial_prior_logit),
        -float(spatial_prior_logit),
    ).astype(np.float32)

    edge_base = np.where(
        same_base,
        node_base_np[src],
        0,
    ).astype(np.int64)

    return SpatialFrameCase(
        t=t,
        shape_zyx=tuple(int(v) for v in supervoxels.shape),
        supervoxels=supervoxels,
        base_labels=base,
        manual_labels=manual,
        node_sv=torch.from_numpy(node_sv_np.copy()).long(),
        node_centroid_um=torch.from_numpy(centroid_np.copy()).float(),
        node_volume_voxels=torch.from_numpy(node_volume_np.copy()).float(),
        edge_index=torch.from_numpy(edge_index_np.copy()).long(),
        raw_spatial_probability=torch.from_numpy(
            probability_np.copy()
        ).float(),
        spatial_edge_logits=torch.from_numpy(
            spatial_logits.copy()
        ).float(),
        node_base_label=torch.from_numpy(node_base_np.copy()).long(),
        node_manual_label=torch.from_numpy(node_manual_np.copy()).long(),
        node_instance_index=torch.from_numpy(
            node_instance_np.copy()
        ).long(),
        instance_base_ids=torch.from_numpy(base_ids.copy()).long(),
        instance_split_target=torch.from_numpy(
            split_target_np.copy()
        ).float(),
        eligible_edge_mask=torch.from_numpy(eligible.copy()).bool(),
        keep_edge_mask=torch.from_numpy(keep.copy()).bool(),
        cut_edge_mask=torch.from_numpy(cut.copy()).bool(),
        edge_base_label=torch.from_numpy(edge_base.copy()).long(),
        changed_base_ids=tuple(changed_base_ids),
    )


# =============================================================================
# DETERMINISTIC FROZEN CACHED-SPATIAL REPRESENTATION
# =============================================================================


class FixedRandomProjection(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        seed: int,
    ):
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        weight = torch.randn(
            in_dim,
            out_dim,
            generator=generator,
            dtype=torch.float32,
        ) / math.sqrt(max(in_dim, 1))
        bias = torch.randn(
            out_dim,
            generator=generator,
            dtype=torch.float32,
        ) * 0.05
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(x @ self.weight + self.bias)


class FrozenCachedSpatialRepresentation(nn.Module):
    """No trainable parameters; converts cached graph geometry to model widths."""

    NODE_RAW_DIM = 6
    EDGE_RAW_DIM = 7
    INSTANCE_RAW_DIM = 8

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.node_projection = FixedRandomProjection(
            self.NODE_RAW_DIM,
            cfg.partition.rag_hidden_dim,
            seed=FIXED_PROJECTION_SEED + 1,
        )
        self.edge_projection = FixedRandomProjection(
            self.EDGE_RAW_DIM,
            cfg.partition.rag_hidden_dim,
            seed=FIXED_PROJECTION_SEED + 2,
        )
        self.instance_projection = FixedRandomProjection(
            self.INSTANCE_RAW_DIM,
            cfg.instances.d_model,
            seed=FIXED_PROJECTION_SEED + 3,
        )

    @staticmethod
    def _instance_geometry(
        case: SpatialFrameCase,
        node_centroid: Tensor,
        node_volume: Tensor,
        dref_um: Tensor,
    ) -> tuple[Tensor, Tensor]:
        instance_index = case.node_instance_index.to(node_centroid.device)
        m = int(case.instance_base_ids.numel())

        volume_sum = node_volume.new_zeros((m,))
        volume_sum.index_add_(0, instance_index, node_volume)

        weighted_centroid = node_centroid.new_zeros((m, 3))
        weighted_centroid.index_add_(
            0,
            instance_index,
            node_centroid * node_volume[:, None],
        )
        ref = weighted_centroid / volume_sum[:, None].clamp_min(1.0)

        # Min/max centroid extents by current instance.
        minimum = node_centroid.new_full((m, 3), torch.inf)
        maximum = node_centroid.new_full((m, 3), -torch.inf)
        if node_centroid.shape[0]:
            minimum.scatter_reduce_(
                0,
                instance_index[:, None].expand(-1, 3),
                node_centroid,
                reduce="amin",
                include_self=True,
            )
            maximum.scatter_reduce_(
                0,
                instance_index[:, None].expand(-1, 3),
                node_centroid,
                reduce="amax",
                include_self=True,
            )
        extent = (maximum - minimum).nan_to_num(0.0) / dref_um.clamp_min(1e-6)

        node_count = node_volume.new_zeros((m,))
        node_count.index_add_(
            0,
            instance_index,
            torch.ones_like(node_volume),
        )
        median_volume = volume_sum.median().clamp_min(1.0)

        raw = torch.cat(
            [
                ref / dref_um.clamp_min(1e-6),
                (
                    torch.log1p(volume_sum)
                    - torch.log1p(median_volume)
                )[:, None],
                extent,
                torch.log1p(node_count)[:, None],
            ],
            dim=-1,
        )
        return raw, ref

    def forward(
        self,
        case: SpatialFrameCase,
        *,
        device: torch.device,
        dref_um_value: float,
    ) -> tuple[RAGState, InstanceState]:
        node_centroid = case.node_centroid_um.to(device)
        node_volume = case.node_volume_voxels.to(device)
        edge_index = case.edge_index.to(device)
        probability = case.raw_spatial_probability.to(device)
        spatial_logits = case.spatial_edge_logits.to(device)
        dref = torch.tensor(
            float(dref_um_value),
            device=device,
            dtype=torch.float32,
        )

        median_node_volume = node_volume.median().clamp_min(1.0)
        node_degree = node_volume.new_zeros((node_volume.shape[0],))
        if edge_index.shape[1]:
            ones = node_volume.new_ones((edge_index.shape[1],))
            node_degree.index_add_(0, edge_index[0], ones)
            node_degree.index_add_(0, edge_index[1], ones)

        base_index = case.node_instance_index.to(device)
        m = int(case.instance_base_ids.numel())
        component_volume = node_volume.new_zeros((m,))
        component_volume.index_add_(0, base_index, node_volume)
        node_component_volume = component_volume[base_index]

        node_raw = torch.cat(
            [
                node_centroid / dref.clamp_min(1e-6),
                (
                    torch.log1p(node_volume)
                    - torch.log1p(median_node_volume)
                )[:, None],
                torch.log1p(node_degree)[:, None],
                (
                    torch.log1p(node_component_volume)
                    - torch.log1p(component_volume.median().clamp_min(1.0))
                )[:, None],
            ],
            dim=-1,
        )
        node_embeddings = self.node_projection(node_raw)

        if edge_index.shape[1]:
            src, dst = edge_index
            delta = (
                node_centroid[dst] - node_centroid[src]
            ) / dref.clamp_min(1e-6)
            distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            log_volume_ratio = torch.log(
                node_volume[dst].clamp_min(1.0)
                / node_volume[src].clamp_min(1.0)
            )[:, None]
            raw_h100_logit = torch.logit(
                probability.clamp(1e-5, 1.0 - 1e-5)
            )[:, None]
            same_base = (
                case.node_base_label[edge_index[0].cpu()]
                == case.node_base_label[edge_index[1].cpu()]
            ).to(device=device, dtype=torch.float32)[:, None]

            edge_raw = torch.cat(
                [
                    torch.tanh(raw_h100_logit / 4.0),
                    delta,
                    distance,
                    log_volume_ratio,
                    same_base,
                ],
                dim=-1,
            )
            edge_embeddings = self.edge_projection(edge_raw)
        else:
            edge_raw = node_raw.new_zeros((0, self.EDGE_RAW_DIM))
            edge_embeddings = node_embeddings.new_zeros(
                (0, self.cfg.partition.rag_hidden_dim)
            )

        instance_raw, instance_ref = self._instance_geometry(
            case,
            node_centroid,
            node_volume,
            dref,
        )
        instance_tokens = self.instance_projection(instance_raw)

        rag = RAGState(
            node_features=node_raw,
            node_embeddings=node_embeddings,
            node_batch=torch.zeros(
                node_embeddings.shape[0],
                device=device,
                dtype=torch.long,
            ),
            node_supervoxel_id=case.node_sv.to(device),
            node_centroid_um=node_centroid,
            node_volume_voxels=node_volume,
            edge_index=edge_index,
            edge_features=edge_raw,
            edge_embeddings=edge_embeddings,
            spatial_edge_logits=spatial_logits,
            edge_batch=torch.zeros(
                edge_index.shape[1],
                device=device,
                dtype=torch.long,
            ),
            supervoxel_labels=[
                torch.as_tensor(
                    np.asarray(case.supervoxels),
                    device=device,
                    dtype=torch.long,
                )
            ],
            node_offsets=torch.tensor(
                [0, node_embeddings.shape[0]],
                device=device,
                dtype=torch.long,
            ),
            statistics=None,
        )

        instances = InstanceState(
            tokens=instance_tokens,
            ref_um=instance_ref,
            batch_index=torch.zeros(
                m,
                device=device,
                dtype=torch.long,
            ),
            local_ids=case.instance_base_ids.to(device),
            quality_logits=torch.zeros(
                m,
                device=device,
                dtype=instance_tokens.dtype,
            ),
            labels=[
                torch.as_tensor(
                    np.asarray(case.base_labels),
                    device=device,
                    dtype=torch.long,
                )
            ],
            token_offsets=torch.tensor(
                [0, m],
                device=device,
                dtype=torch.long,
            ),
            node_to_instance=case.node_instance_index.to(device),
            spatial_tokens=instance_tokens,
        )
        return rag, instances


# =============================================================================
# TEMPORAL MODULE WRAPPER
# =============================================================================


class TemporalOnlyModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.history = HistoricalInstanceEncoder(
            cfg.history,
            cfg.temporal,
        )
        self.encoder = TemporalGraphEncoder(cfg.temporal)
        self.reasoner = InstanceTemporalReasoner(
            cfg.temporal,
            cfg.instances,
            cfg.partition,
            cfg.refinement,
        )

    def encode_temporal(
        self,
        payload: dict[str, Any],
        *,
        device: torch.device,
    ) -> TemporalState:
        graph_x = torch.as_tensor(payload["graph_x"]).to(
            device=device,
            dtype=torch.float32,
        )
        edge_index = torch.as_tensor(
            payload["graph_edge_index"]
        ).to(device=device, dtype=torch.long)
        edge_attr = torch.as_tensor(
            payload["graph_edge_attr"]
        ).to(device=device, dtype=torch.float32)
        tracklet_id = torch.as_tensor(
            payload["tracklet_id"]
        ).to(device=device, dtype=torch.long)
        temporal_ref = torch.as_tensor(
            payload["temporal_ref_um"]
        ).to(device=device, dtype=torch.float32)
        temporal_status = torch.as_tensor(
            payload["temporal_status"]
        ).to(device=device, dtype=torch.float32)
        temporal_batch = torch.zeros(
            temporal_ref.shape[0],
            device=device,
            dtype=torch.long,
        )

        grids = torch.as_tensor(
            payload["node_instance_grid"]
        ).to(device=device)
        valid = torch.as_tensor(
            payload["node_history_valid"]
        ).to(device=device, dtype=torch.bool)
        history_embedding = self.history(grids, valid)

        hypothesis_index = torch.as_tensor(
            payload["hypothesis_edge_index"]
        ).to(device=device, dtype=torch.long)
        hypothesis_attr = torch.as_tensor(
            payload["hypothesis_edge_attr"]
        ).to(device=device, dtype=torch.float32)

        temporal_input = TemporalInput(
            graph_x=graph_x,
            graph_edge_index=edge_index,
            graph_edge_attr=edge_attr,
            tracklet_id=tracklet_id,
            temporal_ref_um=temporal_ref,
            temporal_status=temporal_status,
            temporal_batch=temporal_batch,
            node_history_embedding=history_embedding,
            hypothesis_edge_index=hypothesis_index,
            hypothesis_edge_attr=hypothesis_attr,
        )
        return self.encoder(temporal_input)

    def empty_temporal(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> TemporalState:
        return self.encoder.empty(device, dtype)


def shuffled_temporal_state(
    temporal: TemporalState,
    *,
    seed: int,
) -> TemporalState:
    m = temporal.tokens.shape[0]
    if m <= 1:
        return temporal

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(m, generator=generator).to(
        temporal.tokens.device
    )

    # Keep physical ref_um fixed. Shuffle the content/reliability attached to
    # those positions. That destroys tracklet-content/location correspondence.
    return TemporalState(
        tokens=temporal.tokens[permutation],
        ref_um=temporal.ref_um,
        batch_index=temporal.batch_index,
        salience=temporal.salience[permutation],
        reliability=temporal.reliability[permutation],
        status=temporal.status[permutation],
        node_tokens=temporal.node_tokens,
    )


# =============================================================================
# LOSS
# =============================================================================


@dataclass
class LossBreakdown:
    total: Tensor
    edge: Tensor
    cut: Tensor
    keep: Tensor
    split: Tensor
    keep_gate: Tensor
    selected_cut_edges: int
    selected_keep_edges: int


def choose_keep_indices(
    case: SpatialFrameCase,
    *,
    clean_keep_edges: int,
    keep_to_cut_ratio: int,
    rng: random.Random,
) -> Tensor:
    keep = torch.nonzero(
        case.keep_edge_mask,
        as_tuple=False,
    ).flatten()
    if keep.numel() == 0:
        return keep

    changed = set(case.changed_base_ids)
    edge_base = case.edge_base_label

    changed_keep = torch.tensor(
        [
            int(index)
            for index in keep.tolist()
            if int(edge_base[index]) in changed
        ],
        dtype=torch.long,
    )

    ordinary_keep = torch.tensor(
        [
            int(index)
            for index in keep.tolist()
            if int(edge_base[index]) not in changed
        ],
        dtype=torch.long,
    )

    if case.cut_count > 0:
        desired_ordinary = max(
            clean_keep_edges // 2,
            keep_to_cut_ratio * case.cut_count,
        )
    else:
        desired_ordinary = clean_keep_edges

    if ordinary_keep.numel() > desired_ordinary:
        selected = rng.sample(
            ordinary_keep.tolist(),
            desired_ordinary,
        )
        ordinary_keep = torch.tensor(
            selected,
            dtype=torch.long,
        )

    if changed_keep.numel() == 0:
        return ordinary_keep
    if ordinary_keep.numel() == 0:
        return changed_keep
    return torch.cat([changed_keep, ordinary_keep], dim=0)


def balanced_split_indices(
    split_target: Tensor,
    *,
    negative_ratio: int,
    rng: random.Random,
) -> Tensor:
    positive = torch.nonzero(
        split_target > 0.5,
        as_tuple=False,
    ).flatten()
    negative = torch.nonzero(
        split_target <= 0.5,
        as_tuple=False,
    ).flatten()

    if positive.numel() == 0:
        limit = min(128, negative.numel())
    else:
        limit = min(
            negative.numel(),
            max(16, negative_ratio * positive.numel()),
        )

    if negative.numel() > limit:
        negative = torch.tensor(
            rng.sample(negative.tolist(), int(limit)),
            dtype=torch.long,
        )

    if positive.numel() and negative.numel():
        return torch.cat([positive, negative])
    return positive if positive.numel() else negative


def temporal_partition_loss(
    case: SpatialFrameCase,
    reasoning,
    *,
    device: torch.device,
    clean_keep_edges: int,
    keep_to_cut_ratio: int,
    split_loss_weight: float,
    keep_gate_weight: float,
    rng: random.Random,
) -> LossBreakdown:
    cut_index = torch.nonzero(
        case.cut_edge_mask,
        as_tuple=False,
    ).flatten().to(device)
    keep_index = choose_keep_indices(
        case,
        clean_keep_edges=clean_keep_edges,
        keep_to_cut_ratio=keep_to_cut_ratio,
        rng=rng,
    ).to(device)

    zero = reasoning.final_edge_logits.sum() * 0.0

    if cut_index.numel():
        cut_loss = F.binary_cross_entropy_with_logits(
            reasoning.final_edge_logits[cut_index],
            torch.zeros(
                cut_index.numel(),
                device=device,
                dtype=reasoning.final_edge_logits.dtype,
            ),
        )
    else:
        cut_loss = zero

    if keep_index.numel():
        keep_loss = F.binary_cross_entropy_with_logits(
            reasoning.final_edge_logits[keep_index],
            torch.ones(
                keep_index.numel(),
                device=device,
                dtype=reasoning.final_edge_logits.dtype,
            ),
        )
        keep_gate = reasoning.edge_temporal_gate[keep_index].square().mean()
    else:
        keep_loss = zero
        keep_gate = zero

    if cut_index.numel() and keep_index.numel():
        edge_loss = cut_loss + keep_loss
    elif cut_index.numel():
        edge_loss = cut_loss
    else:
        edge_loss = keep_loss

    split_target = case.instance_split_target
    split_index = balanced_split_indices(
        split_target,
        negative_ratio=8,
        rng=rng,
    ).to(device)

    if split_index.numel():
        split_loss = F.binary_cross_entropy_with_logits(
            reasoning.split_logits[split_index],
            split_target.to(device)[split_index],
        )
    else:
        split_loss = zero

    total = (
        edge_loss
        + split_loss_weight * split_loss
        + keep_gate_weight * keep_gate
    )

    return LossBreakdown(
        total=total,
        edge=edge_loss,
        cut=cut_loss,
        keep=keep_loss,
        split=split_loss,
        keep_gate=keep_gate,
        selected_cut_edges=int(cut_index.numel()),
        selected_keep_edges=int(keep_index.numel()),
    )


# =============================================================================
# PARTITION EVALUATION
# =============================================================================


def predicted_node_components_within_base(
    case: SpatialFrameCase,
    final_logits: Tensor,
) -> np.ndarray:
    """Split-only graph partition: never merge different frozen base instances."""
    edge_index = case.edge_index.cpu().numpy()
    logits = final_logits.detach().float().cpu().numpy()
    node_base = case.node_base_label.cpu().numpy()

    n = len(node_base)
    uf = UnionFind(n)

    for edge_row in range(edge_index.shape[1]):
        a = int(edge_index[0, edge_row])
        b = int(edge_index[1, edge_row])

        if node_base[a] <= 0 or node_base[a] != node_base[b]:
            continue
        if logits[edge_row] >= 0.0:
            uf.union(a, b)

    return np.asarray([uf.find(i) for i in range(n)], dtype=np.int64)


def component_partition_exact(
    node_rows: np.ndarray,
    predicted_component: np.ndarray,
    manual_node_label: np.ndarray,
) -> bool:
    pred_groups: dict[int, set[int]] = {}
    manual_groups: dict[int, set[int]] = {}

    for node in node_rows.tolist():
        pred_groups.setdefault(
            int(predicted_component[node]),
            set(),
        ).add(int(node))
        manual_groups.setdefault(
            int(manual_node_label[node]),
            set(),
        ).add(int(node))

    pred_signature = {
        tuple(sorted(group))
        for group in pred_groups.values()
    }
    manual_signature = {
        tuple(sorted(group))
        for group in manual_groups.values()
    }
    return pred_signature == manual_signature


def rasterize_prediction(
    case: SpatialFrameCase,
    predicted_component: np.ndarray,
) -> np.ndarray:
    """Rasterize split-only node partition to a compact label volume."""
    node_sv = case.node_sv.cpu().numpy()
    node_base = case.node_base_label.cpu().numpy()

    max_sv = int(np.asarray(case.supervoxels).max(initial=0))
    sv_to_output = np.zeros(max_sv + 1, dtype=np.int32)

    next_id = 1
    for base_id in sorted(
        int(v) for v in np.unique(node_base) if int(v) > 0
    ):
        rows = np.flatnonzero(node_base == base_id)
        roots = sorted(
            set(int(predicted_component[row]) for row in rows.tolist())
        )
        root_to_label = {
            root: next_id + index
            for index, root in enumerate(roots)
        }
        next_id += len(roots)

        for row in rows.tolist():
            sv_to_output[int(node_sv[row])] = root_to_label[
                int(predicted_component[row])
            ]

    ws = np.asarray(case.supervoxels, dtype=np.int64)
    return sv_to_output[ws]


@dataclass
class EvalAccumulator:
    cut_total: int = 0
    cut_correct: int = 0
    keep_total: int = 0
    keep_correct: int = 0

    changed_total: int = 0
    changed_exact: int = 0
    unchanged_total: int = 0
    unchanged_split: int = 0

    cut_gate_sum: float = 0.0
    cut_gate_count: int = 0
    keep_gate_sum: float = 0.0
    keep_gate_count: int = 0

    max_empty_noop_error: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "cut_edges": self.cut_total,
            "cut_edge_recall": (
                self.cut_correct / self.cut_total
                if self.cut_total
                else 1.0
            ),
            "keep_edges": self.keep_total,
            "keep_edge_accuracy": (
                self.keep_correct / self.keep_total
                if self.keep_total
                else 1.0
            ),
            "changed_components": self.changed_total,
            "changed_components_exact": self.changed_exact,
            "changed_component_exact_rate": (
                self.changed_exact / self.changed_total
                if self.changed_total
                else 1.0
            ),
            "unchanged_components": self.unchanged_total,
            "unchanged_components_accidentally_split": self.unchanged_split,
            "unchanged_component_split_rate": (
                self.unchanged_split / self.unchanged_total
                if self.unchanged_total
                else 0.0
            ),
            "mean_cut_gate": (
                self.cut_gate_sum / self.cut_gate_count
                if self.cut_gate_count
                else 0.0
            ),
            "mean_keep_gate": (
                self.keep_gate_sum / self.keep_gate_count
                if self.keep_gate_count
                else 0.0
            ),
            "max_empty_noop_error": self.max_empty_noop_error,
        }


def accumulate_frame_metrics(
    accumulator: EvalAccumulator,
    case: SpatialFrameCase,
    reasoning,
) -> np.ndarray:
    logits = reasoning.final_edge_logits.detach().float().cpu()
    gate = reasoning.edge_temporal_gate.detach().float().cpu()

    cut = case.cut_edge_mask
    keep = case.keep_edge_mask

    accumulator.cut_total += int(cut.sum().item())
    accumulator.cut_correct += int(
        (logits[cut] < 0).sum().item()
    )
    accumulator.keep_total += int(keep.sum().item())
    accumulator.keep_correct += int(
        (logits[keep] >= 0).sum().item()
    )

    if cut.any():
        accumulator.cut_gate_sum += float(gate[cut].sum())
        accumulator.cut_gate_count += int(cut.sum())
    if keep.any():
        accumulator.keep_gate_sum += float(gate[keep].sum())
        accumulator.keep_gate_count += int(keep.sum())

    predicted = predicted_node_components_within_base(
        case,
        logits,
    )
    node_base = case.node_base_label.cpu().numpy()
    manual_node = case.node_manual_label.cpu().numpy()
    changed = set(case.changed_base_ids)

    for base_id in case.instance_base_ids.tolist():
        base_id = int(base_id)
        rows = np.flatnonzero(node_base == base_id)
        exact = component_partition_exact(
            rows,
            predicted,
            manual_node,
        )

        if base_id in changed:
            accumulator.changed_total += 1
            accumulator.changed_exact += int(exact)
        else:
            accumulator.unchanged_total += 1
            roots = {
                int(predicted[row])
                for row in rows.tolist()
            }
            accumulator.unchanged_split += int(len(roots) > 1)

    return predicted


# =============================================================================
# FORWARD / EVALUATION
# =============================================================================


def load_all_temporal_payloads(
    paths: Paths,
    frame_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for t in range(frame_count):
        payload = load_cache(paths.temporal_cache(t))
        rows.append(payload)
    return rows


def forward_frame(
    temporal_model: TemporalOnlyModel,
    frozen_spatial: FrozenCachedSpatialRepresentation,
    case: SpatialFrameCase,
    temporal_payload: dict[str, Any],
    *,
    device: torch.device,
    dref_um: float,
    ablation: str = "full",
):
    rag, instances = frozen_spatial(
        case,
        device=device,
        dref_um_value=dref_um,
    )

    if ablation == "empty":
        temporal = temporal_model.empty_temporal(device)
    else:
        temporal = temporal_model.encode_temporal(
            temporal_payload,
            device=device,
        )
        if ablation == "shuffled":
            temporal = shuffled_temporal_state(
                temporal,
                seed=FIXED_PROJECTION_SEED + case.t,
            )
        elif ablation != "full":
            raise ValueError(ablation)

    reasoning = temporal_model.reasoner(
        instances,
        rag,
        temporal,
        torch.tensor(
            [float(dref_um)],
            device=device,
            dtype=torch.float32,
        ),
    )
    return rag, instances, temporal, reasoning


@torch.no_grad()
def evaluate_model(
    temporal_model: TemporalOnlyModel,
    frozen_spatial: FrozenCachedSpatialRepresentation,
    cases: list[SpatialFrameCase],
    temporal_payloads: list[dict[str, Any]],
    *,
    device: torch.device,
    dref_um: float,
    output_dir: Path | None,
    save_predictions: bool,
) -> dict[str, Any]:
    temporal_model.eval()
    frozen_spatial.eval()

    results: dict[str, Any] = {}

    for ablation in ("full", "empty", "shuffled"):
        accumulator = EvalAccumulator()

        for case, temporal_payload in zip(cases, temporal_payloads):
            rag, _, _, reasoning = forward_frame(
                temporal_model,
                frozen_spatial,
                case,
                temporal_payload,
                device=device,
                dref_um=dref_um,
                ablation=ablation,
            )

            if ablation == "empty":
                error = float(
                    (
                        reasoning.final_edge_logits
                        - rag.spatial_edge_logits
                    )
                    .abs()
                    .max()
                    .item()
                    if rag.spatial_edge_logits.numel()
                    else 0.0
                )
                accumulator.max_empty_noop_error = max(
                    accumulator.max_empty_noop_error,
                    error,
                )

            predicted = accumulate_frame_metrics(
                accumulator,
                case,
                reasoning,
            )

            if (
                save_predictions
                and output_dir is not None
                and ablation == "full"
            ):
                frame_labels = rasterize_prediction(
                    case,
                    predicted,
                )
                prediction_dir = output_dir / "predictions"
                prediction_dir.mkdir(parents=True, exist_ok=True)
                np.save(
                    prediction_dir
                    / f"temporal_partition_t{case.t:03d}.npy",
                    frame_labels.astype(np.int32, copy=False),
                    allow_pickle=False,
                )

        results[ablation] = accumulator.as_dict()

    full = results["full"]
    empty = results["empty"]
    shuffled = results["shuffled"]

    empty_exact_noop = (
        float(empty["max_empty_noop_error"]) == 0.0
    )

    override_functional = bool(
        empty_exact_noop
        and full["cut_edge_recall"] >= 0.90
        and full["changed_component_exact_rate"] >= 0.75
        and full["unchanged_component_split_rate"] <= 0.02
    )

    content_gain_changed = (
        full["changed_component_exact_rate"]
        - shuffled["changed_component_exact_rate"]
    )
    content_gain_cut = (
        full["cut_edge_recall"]
        - shuffled["cut_edge_recall"]
    )
    content_dependent = bool(
        content_gain_changed >= 0.10
        or content_gain_cut >= 0.05
    )

    results["verdict"] = {
        "empty_temporal_exact_noop": empty_exact_noop,
        "temporal_override_mechanism": (
            "PASS" if override_functional else "FAIL"
        ),
        "temporal_content_dependence": (
            "PASS"
            if content_dependent
            else "INCONCLUSIVE"
        ),
        "full_minus_shuffled_changed_exact_rate": float(
            content_gain_changed
        ),
        "full_minus_shuffled_cut_recall": float(
            content_gain_cut
        ),
        "interpretation": (
            "PASS for override means the temporal path can reproduce most "
            "manual splits while preserving clean spatial components. "
            "PASS for content dependence additionally means temporally "
            "localized track content matters; INCONCLUSIVE means the model "
            "may still be exploiting static spatial features once any local "
            "temporal support is present."
        ),
    }
    return results


def print_eval(title: str, metrics: dict[str, Any]) -> None:
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)

    for name in ("full", "empty", "shuffled"):
        row = metrics[name]
        print(
            f"{name.upper():9s} | "
            f"cut recall={row['cut_edge_recall']:.4f} "
            f"keep acc={row['keep_edge_accuracy']:.4f} | "
            f"changed exact={row['changed_components_exact']}/"
            f"{row['changed_components']} "
            f"({row['changed_component_exact_rate']:.4f}) | "
            f"clean split={row['unchanged_components_accidentally_split']}/"
            f"{row['unchanged_components']} "
            f"({row['unchanged_component_split_rate']:.4f}) | "
            f"gate cut/keep={row['mean_cut_gate']:.3f}/"
            f"{row['mean_keep_gate']:.3f}"
        )

    verdict = metrics["verdict"]
    print("-" * 118)
    print(
        "EMPTY TEMPORAL EXACT NO-OP : "
        f"{'PASS' if verdict['empty_temporal_exact_noop'] else 'FAIL'}"
    )
    print(
        "TEMPORAL OVERRIDE MECHANISM: "
        f"{verdict['temporal_override_mechanism']}"
    )
    print(
        "TEMPORAL CONTENT DEPENDENCE: "
        f"{verdict['temporal_content_dependence']}"
    )
    print(
        "FULL - SHUFFLED            : "
        f"changed_exact={verdict['full_minus_shuffled_changed_exact_rate']:+.4f}, "
        f"cut_recall={verdict['full_minus_shuffled_cut_recall']:+.4f}"
    )
    print("=" * 118)


# =============================================================================
# CHECKPOINT
# =============================================================================


def checkpoint_payload(
    *,
    temporal_model: TemporalOnlyModel,
    cfg: ModelConfig,
    step: int,
    dref_um: float,
    spacing: tuple[float, float, float],
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "investigation": SCRIPT_NAME,
        "step": int(step),
        "sample_id": args.sample_id,
        "spacing_zyx_um": tuple(float(v) for v in spacing),
        "dref_um": float(dref_um),
        "model_config": dataclasses.asdict(cfg),
        "temporal_model_state_dict": temporal_model.state_dict(),
        "metrics": metrics,
        "training_args": vars(args),
        "notes": {
            "spatial_model_ran": False,
            "spatial_representation": (
                "deterministic non-trainable projections of cached RAG "
                "geometry/topology + frozen Investigation-25 partition"
            ),
            "supervision": "within-base-instance split/keep edges only",
            "manual_existence_supervision": False,
        },
    }


def load_resume_checkpoint(
    path: Path,
    temporal_model: TemporalOnlyModel,
) -> int:
    payload = torch_load(path)
    temporal_model.load_state_dict(
        payload["temporal_model_state_dict"],
        strict=True,
    )
    return int(payload.get("step", 0))


# =============================================================================
# TRAINING
# =============================================================================


def choose_training_case(
    cases: list[SpatialFrameCase],
    correction_cases: list[SpatialFrameCase],
    *,
    correction_probability: float,
    rng: random.Random,
) -> SpatialFrameCase:
    if correction_cases and rng.random() < correction_probability:
        return rng.choice(correction_cases)
    return rng.choice(cases)


def train(
    *,
    temporal_model: TemporalOnlyModel,
    frozen_spatial: FrozenCachedSpatialRepresentation,
    cases: list[SpatialFrameCase],
    temporal_payloads: list[dict[str, Any]],
    cfg: ModelConfig,
    device: torch.device,
    dref_um: float,
    spacing: tuple[float, float, float],
    paths: Paths,
    args: argparse.Namespace,
) -> dict[str, Any]:
    trainable = [
        parameter
        for parameter in temporal_model.parameters()
        if parameter.requires_grad
    ]
    frozen_parameters = sum(
        p.numel()
        for p in frozen_spatial.parameters()
    )
    if frozen_parameters != 0:
        raise RuntimeError(
            "FrozenCachedSpatialRepresentation unexpectedly has trainable parameters"
        )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume is not None:
        start_step = load_resume_checkpoint(
            resolve(args.resume),
            temporal_model,
        )
        print(f"[resume] loaded step {start_step}")

    correction_cases = [
        case
        for case in cases
        if case.cut_count > 0
    ]
    rng = random.Random(args.seed + 91)

    print()
    print("=" * 118)
    print("Investigation 30 — TEMPORAL-ONLY OVERFIT")
    print("=" * 118)
    print(f"device                     : {device}")
    print(f"frames                     : {len(cases)}")
    print(
        f"frames with manual cuts    : "
        f"{len(correction_cases)}"
    )
    print(
        f"manual changed components  : "
        f"{sum(case.changed_component_count for case in cases)}"
    )
    print(
        f"manual CUT edges           : "
        f"{sum(case.cut_count for case in cases)}"
    )
    print(
        f"supervised KEEP edges      : "
        f"{sum(case.keep_count for case in cases)}"
    )
    print(f"dref                       : {dref_um:.4f} um")
    print(f"spacing zyx                : {spacing}")
    print(f"steps                      : {args.steps}")
    print(f"lr                         : {args.lr:g}")
    print(
        "spatial CNN / watershed / RAG : NOT INSTANTIATED / NOT RUN"
    )
    print("=" * 118)

    temporal_model.to(device)
    frozen_spatial.to(device)

    initial_metrics = evaluate_model(
        temporal_model,
        frozen_spatial,
        cases,
        temporal_payloads,
        device=device,
        dref_um=dref_um,
        output_dir=None,
        save_predictions=False,
    )
    print_eval("BEFORE TEMPORAL OVERFIT", initial_metrics)

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    temporal_model.train()

    for step in range(start_step + 1, args.steps + 1):
        case = choose_training_case(
            cases,
            correction_cases,
            correction_probability=args.correction_frame_probability,
            rng=rng,
        )
        temporal_payload = temporal_payloads[case.t]

        optimizer.zero_grad(set_to_none=True)

        rag, instances, temporal, reasoning = forward_frame(
            temporal_model,
            frozen_spatial,
            case,
            temporal_payload,
            device=device,
            dref_um=dref_um,
            ablation="full",
        )

        losses = temporal_partition_loss(
            case,
            reasoning,
            device=device,
            clean_keep_edges=args.clean_keep_edges,
            keep_to_cut_ratio=args.keep_to_cut_ratio,
            split_loss_weight=args.split_loss_weight,
            keep_gate_weight=args.keep_gate_weight,
            rng=rng,
        )
        losses.total.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            temporal_model.parameters(),
            args.grad_clip,
        )
        optimizer.step()

        if step == 1 or step % args.print_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[step {step:05d}/{args.steps}] "
                f"t={case.t:02d} "
                f"loss={float(losses.total.detach()):.5f} "
                f"edge={float(losses.edge.detach()):.5f} "
                f"cut={float(losses.cut.detach()):.5f} "
                f"keep={float(losses.keep.detach()):.5f} "
                f"split={float(losses.split.detach()):.5f} "
                f"gate_keep={float(losses.keep_gate.detach()):.5f} "
                f"ncut={losses.selected_cut_edges} "
                f"nkeep={losses.selected_keep_edges} "
                f"grad={float(torch.as_tensor(grad_norm)):.3f} "
                f"elapsed={format_seconds(elapsed)}",
                flush=True,
            )

        if (
            step % args.eval_every == 0
            or step == args.steps
        ):
            metrics = evaluate_model(
                temporal_model,
                frozen_spatial,
                cases,
                temporal_payloads,
                device=device,
                dref_um=dref_um,
                output_dir=None,
                save_predictions=False,
            )
            print_eval(
                f"TEMPORAL OVERFIT EVALUATION @ STEP {step}",
                metrics,
            )
            history.append(
                {
                    "step": step,
                    "metrics": metrics,
                }
            )

            atomic_torch_save(
                paths.output / "latest.pt",
                checkpoint_payload(
                    temporal_model=temporal_model,
                    cfg=cfg,
                    step=step,
                    dref_um=dref_um,
                    spacing=spacing,
                    metrics=metrics,
                    args=args,
                ),
            )
            atomic_json(
                paths.output / "training_history.json",
                history,
            )

            temporal_model.train()

    temporal_model.eval()
    final_metrics = evaluate_model(
        temporal_model,
        frozen_spatial,
        cases,
        temporal_payloads,
        device=device,
        dref_um=dref_um,
        output_dir=paths.output,
        save_predictions=True,
    )
    print_eval("FINAL INVESTIGATION-30 EVALUATION", final_metrics)

    atomic_json(
        paths.output / "final_metrics.json",
        final_metrics,
    )
    atomic_torch_save(
        paths.output / "final.pt",
        checkpoint_payload(
            temporal_model=temporal_model,
            cfg=cfg,
            step=args.steps,
            dref_um=dref_um,
            spacing=spacing,
            metrics=final_metrics,
            args=args,
        ),
    )

    return final_metrics


# =============================================================================
# DATA AUDIT
# =============================================================================


def audit_cases(cases: list[SpatialFrameCase]) -> dict[str, Any]:
    rows = []
    for case in cases:
        rows.append(
            {
                "timepoint": case.t,
                "rag_nodes": int(case.node_sv.numel()),
                "rag_edges": int(case.edge_index.shape[1]),
                "base_instances": int(case.instance_base_ids.numel()),
                "changed_base_instances": int(
                    case.changed_component_count
                ),
                "supervised_cut_edges": int(case.cut_count),
                "supervised_keep_edges": int(case.keep_count),
            }
        )

    summary = {
        "frames": len(cases),
        "frames_with_manual_cuts": sum(
            row["supervised_cut_edges"] > 0
            for row in rows
        ),
        "changed_base_instances": sum(
            row["changed_base_instances"]
            for row in rows
        ),
        "supervised_cut_edges": sum(
            row["supervised_cut_edges"]
            for row in rows
        ),
        "supervised_keep_edges": sum(
            row["supervised_keep_edges"]
            for row in rows
        ),
        "per_frame": rows,
    }

    print()
    print("=" * 104)
    print("ANNOTATION / RAG AUDIT")
    print("=" * 104)
    print(
        f"{'t':>3s} {'nodes':>7s} {'edges':>8s} "
        f"{'base':>6s} {'changed':>8s} "
        f"{'CUT':>6s} {'KEEP':>8s}"
    )
    print("-" * 104)
    for row in rows:
        print(
            f"{row['timepoint']:3d} "
            f"{row['rag_nodes']:7d} "
            f"{row['rag_edges']:8d} "
            f"{row['base_instances']:6d} "
            f"{row['changed_base_instances']:8d} "
            f"{row['supervised_cut_edges']:6d} "
            f"{row['supervised_keep_edges']:8d}"
        )
    print("-" * 104)
    print(
        f"Frames with manual CUT edges : "
        f"{summary['frames_with_manual_cuts']}/{summary['frames']}"
    )
    print(
        f"Changed base instances       : "
        f"{summary['changed_base_instances']}"
    )
    print(
        f"Total CUT / KEEP edges       : "
        f"{summary['supervised_cut_edges']} / "
        f"{summary['supervised_keep_edges']}"
    )
    print("=" * 104)

    if summary["supervised_cut_edges"] == 0:
        raise RuntimeError(
            "No manual split boundary maps onto an Investigation-12 RAG edge. "
            "Check that Investigation-24 supervoxels, Investigation-25 base "
            "instances, and the manual annotations are aligned."
        )

    return summary


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Investigation 30: temporal-only overfit on 20 manually corrected "
            "BioHub partitions."
        )
    )

    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)

    parser.add_argument("--inv12", type=Path, default=None)
    parser.add_argument("--inv24", type=Path, default=None)
    parser.add_argument("--inv25", type=Path, default=None)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--zarr", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)

    parser.add_argument(
        "--spacing-zyx-um",
        type=float,
        nargs=3,
        default=DEFAULT_SPACING_ZYX_UM,
    )
    parser.add_argument(
        "--dref-um",
        type=float,
        default=None,
        help=(
            "Override model cell reference diameter. Default: median of "
            "estimate_model_dref_um across the 20 frozen current frames."
        ),
    )
    parser.add_argument(
        "--temporal-radius",
        type=int,
        default=DEFAULT_TEMPORAL_RADIUS,
    )

    parser.add_argument(
        "--trackastra-model",
        default="ctc",
    )
    parser.add_argument(
        "--trackastra-mode",
        default="greedy",
    )
    parser.add_argument(
        "--trackastra-device",
        default="cuda",
    )
    parser.add_argument(
        "--rebuild-trackastra",
        action="store_true",
    )
    parser.add_argument(
        "--rebuild-temporal",
        action="store_true",
    )
    parser.add_argument(
        "--rebuild-movies",
        action="store_true",
    )
    parser.add_argument(
        "--complete-candidate-graph",
        action="store_true",
        help=(
            "Use graph_builder's complete directed detection graph. Default "
            "uses accepted Trackastra links + bounded same-frame neighbours "
            "to keep the 20-frame experiment tractable."
        ),
    )

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=DEFAULT_EVAL_EVERY,
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--correction-frame-probability",
        type=float,
        default=DEFAULT_CORRECTION_FRAME_PROB,
    )
    parser.add_argument(
        "--clean-keep-edges",
        type=int,
        default=DEFAULT_CLEAN_KEEP_EDGES,
    )
    parser.add_argument(
        "--keep-to-cut-ratio",
        type=int,
        default=DEFAULT_KEEP_TO_CUT_RATIO,
    )
    parser.add_argument(
        "--split-loss-weight",
        type=float,
        default=DEFAULT_SPLIT_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--keep-gate-weight",
        type=float,
        default=DEFAULT_KEEP_GATE_WEIGHT,
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
    )
    parser.add_argument(
        "--spatial-prior-logit",
        type=float,
        default=DEFAULT_SPATIAL_PRIOR_LOGIT,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device. Default: cuda if available, else cpu.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare/audit spatial + temporal caches, then stop before training.",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not 0.0 <= args.correction_frame_probability <= 1.0:
        raise ValueError("--correction-frame-probability must be in [0,1]")
    if args.temporal_radius < 0:
        raise ValueError("--temporal-radius cannot be negative")
    if args.frame_count < 1:
        raise ValueError("--frame-count must be positive")
    if args.eval_every < 1 or args.print_every < 1:
        raise ValueError("eval/print cadence must be positive")

    paths = make_paths(args)
    paths.output.mkdir(parents=True, exist_ok=True)

    print("=" * 118)
    print("INVESTIGATION 30 — BIOHUB TEMPORAL-ONLY PARTITION OVERFIT")
    print("=" * 118)
    print(f"repository   : {ROOT}")
    print(f"sample       : {paths.sample_id}")
    print(f"Inv12 RAG    : {paths.inv12}")
    print(f"Inv24 SV     : {paths.inv24}")
    print(f"Inv25 base   : {paths.inv25}")
    print(f"annotations  : {paths.annotations}")
    print(f"raw zarr     : {paths.zarr}")
    print(f"output       : {paths.output}")
    print(
        f"temporal cache contract: v{TEMPORAL_CACHE_CONTRACT_VERSION}"
    )
    print("=" * 118)

    if TEMPORAL_CACHE_CONTRACT_VERSION < 4:
        raise RuntimeError(
            "Investigation 30 requires the finite-window availability patch "
            "(temporal cache contract v4 or newer)."
        )

    validate_required_artifacts(
        paths,
        args.frame_count,
    )

    spacing = tuple(
        float(v)
        for v in args.spacing_zyx_um
    )

    # ---------------------------------------------------------------------
    # 1) Frozen spatial cases — NO spatial model inference.
    # ---------------------------------------------------------------------
    cases = [
        load_spatial_case(
            paths,
            t,
            spacing,
            args.spatial_prior_logit,
        )
        for t in range(args.frame_count)
    ]
    audit = audit_cases(cases)
    atomic_json(paths.output / "annotation_rag_audit.json", audit)

    # ---------------------------------------------------------------------
    # 2) Assemble frozen current movie + raw movie.
    # ---------------------------------------------------------------------
    base_movie_path = assemble_base_instance_movie(
        paths,
        args.frame_count,
        rebuild=args.rebuild_movies,
    )
    base_movie = np.load(base_movie_path, mmap_mode="r")

    raw_movie_path = assemble_raw_movie(
        paths,
        args.frame_count,
        cases[0].shape_zyx,
        rebuild=args.rebuild_movies,
    )
    raw_movie = np.load(raw_movie_path, mmap_mode="r")

    dref_um = (
        float(args.dref_um)
        if args.dref_um is not None
        else resolve_movie_dref_um(
            base_movie,
            spacing,
        )
    )
    print(f"[scale] model dref = {dref_um:.5f} um")

    # ---------------------------------------------------------------------
    # 3) Trackastra temporal preprocessing.
    # ---------------------------------------------------------------------
    track_graph_path, tracked_masks_path = prepare_trackastra(
        paths,
        raw_movie_path,
        base_movie_path,
        model_name=args.trackastra_model,
        mode=args.trackastra_mode,
        device=args.trackastra_device,
        rebuild=args.rebuild_trackastra,
    )

    with track_graph_path.open("rb") as handle:
        track_graph = pickle.load(handle)
    tracked_movie = np.load(
        tracked_masks_path,
        mmap_mode="r",
    )

    if tracked_movie.shape != base_movie.shape:
        raise RuntimeError(
            "Trackastra output shape does not match frozen base movie: "
            f"{tracked_movie.shape} vs {base_movie.shape}"
        )

    # ---------------------------------------------------------------------
    # 4) Target-specific temporal-v4 caches with one-sided boundary support.
    # ---------------------------------------------------------------------
    build_temporal_caches(
        paths,
        track_graph=track_graph,
        tracked_movie=tracked_movie,
        raw_movie=raw_movie,
        frame_count=args.frame_count,
        spacing=spacing,
        dref_um=dref_um,
        temporal_radius=args.temporal_radius,
        complete_candidate_graph=args.complete_candidate_graph,
        rebuild=args.rebuild_temporal,
    )

    temporal_payloads = load_all_temporal_payloads(
        paths,
        args.frame_count,
    )

    temporal_audit = {
        "cache_contract": TEMPORAL_CACHE_CONTRACT_VERSION,
        "temporal_radius": args.temporal_radius,
        "complete_candidate_graph": bool(
            args.complete_candidate_graph
        ),
        "per_frame": [
            {
                "timepoint": t,
                "available_time_offsets": [
                    int(v)
                    for v in torch.as_tensor(
                        payload["available_time_offsets"]
                    ).tolist()
                ],
                "detections": int(
                    torch.as_tensor(payload["graph_x"]).shape[0]
                ),
                "detection_edges": int(
                    torch.as_tensor(
                        payload["graph_edge_index"]
                    ).shape[1]
                ),
                "tracklets": int(
                    torch.as_tensor(
                        payload["temporal_ref_um"]
                    ).shape[0]
                ),
                "hypothesis_edges": int(
                    torch.as_tensor(
                        payload["hypothesis_edge_index"]
                    ).shape[1]
                ),
                "history_valid": int(
                    torch.as_tensor(
                        payload["node_history_valid"]
                    ).sum()
                ),
            }
            for t, payload in enumerate(temporal_payloads)
        ],
    }
    atomic_json(
        paths.output / "temporal_cache_audit.json",
        temporal_audit,
    )

    if args.prepare_only:
        print()
        print("Preparation complete (--prepare-only).")
        print(f"Output: {paths.output}")
        return

    # ---------------------------------------------------------------------
    # 5) Temporal-only model.
    # ---------------------------------------------------------------------
    cfg = ModelConfig()
    cfg.history.dropout = 0.0
    cfg.history.activation_checkpointing = False
    cfg.temporal.dropout = 0.0
    cfg.instances.dropout = 0.0
    cfg.validate()

    temporal_model = TemporalOnlyModel(cfg)
    frozen_spatial = FrozenCachedSpatialRepresentation(cfg)

    if any(True for _ in frozen_spatial.parameters()):
        raise RuntimeError(
            "Frozen cached-spatial representation must have no parameters"
        )

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ---------------------------------------------------------------------
    # 6) Train + evaluate.
    # ---------------------------------------------------------------------
    final_metrics = train(
        temporal_model=temporal_model,
        frozen_spatial=frozen_spatial,
        cases=cases,
        temporal_payloads=temporal_payloads,
        cfg=cfg,
        device=device,
        dref_um=dref_um,
        spacing=spacing,
        paths=paths,
        args=args,
    )

    print()
    print("Investigation 30 complete.")
    print(f"Checkpoint : {paths.output / 'final.pt'}")
    print(f"Metrics    : {paths.output / 'final_metrics.json'}")
    print(f"Predictions: {paths.output / 'predictions'}")

    verdict = final_metrics["verdict"]
    print()
    print(
        "Primary result: "
        f"override={verdict['temporal_override_mechanism']}, "
        f"content={verdict['temporal_content_dependence']}"
    )


if __name__ == "__main__":
    main()
