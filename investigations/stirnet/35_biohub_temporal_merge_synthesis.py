from __future__ import annotations

r"""
Investigation 35 — concrete BioHub merge synthesis + direct temporal CUT/KEEP training.

This file is a COMPLETE REWRITE of Investigation 35.

It deliberately does not import or reuse the previous Investigation-35
implementation, Investigation-30 temporal metadata construction, DetectionRecord,
TemporalStatic, historical instance grids, or graph_builder.build_temporal_graph().

Scientific task
---------------
The current annotated BioHub movie already contains the true cell instances.

The temporal model only needs to learn:

    inside one current spatial component:
        KEEP this RAG connection
        or
        CUT this RAG connection because two real cells were merged

Therefore the synthetic dataset is simply:

    true annotated movie
        ↓
    randomly choose non-overlapping touching true-cell pairs
        ↓
    relabel each selected A|B pair as one A+B component for that frame
        ↓
    run Trackastra on the resulting concrete corrupted movie
        ↓
    use the Trackastra graph DIRECTLY as temporal input
        ↓
    train CUT/KEEP on the immutable atomic RAG

There is no other temporal-data synthesis stage.

The raw movie is shared by every virtual variant. Only the temporary label movie
changes. Each concrete variant gets an independent Trackastra graph.

Temporal model contract
-----------------------
The repository's current TemporalGraphEncoder accepts TemporalInput:

    graph_x             [N, 32]
    graph_edge_index    [2, E]
    graph_edge_attr     [E, 15]
    tracklet_id         [N]
    temporal_ref_um     [M, 3]
    temporal_status     [M, 10]
    temporal_batch      [M]

Optional fields are intentionally omitted here:

    node_history_embedding = None
    hypothesis_edge_index  = None
    hypothesis_edge_attr   = None

The 32-D and 15-D layouts follow the current graph_builder semantic contract,
but are assembled directly from Trackastra:

    node:
        time
        physical position
        component volume
        backward/forward track velocity
        track length
        boundary/start/end/division status

    edge:
        delta time
        physical displacement/distance
        volume ratio
        motion residual
        Trackastra association score when available
        relation one-hot
        accepted flag

Heavy morphology, intensity, and historical-grid fields are neutral values.
They are not recomputed for every synthetic movie.

Spatial side
------------
Temporal training still needs the mature frozen RAG and compact supervoxel
statistics used by InstanceTokenizer.

This rewrite creates the compact spatial cache itself when it is absent:

    <output>/spatial_cache/tXXX/frozen_graph.pt

The cache is rebuilt from the CURRENT repository spatial checkpoint using the
current BioHub preprocessing, source segmentation, tiled spatial STIR-Net, RAG,
and compact supervoxel-statistics path.

The file never reads any previous temporal_static.pkl, candidate_manifest.json,
old Trackastra graph, synthetic temporal graph, or old temporal checkpoint.

Initializer
-----------
The spatial checkpoint is resolved from the frozen spatial-cache metadata, or
can be supplied explicitly with --checkpoint.

The temporal modules therefore initialize from the spatial checkpoint, NOT from
the old Investigation-35 temporal model.

Split-only invariant
--------------------
The synthetic current partition is built directly from true manual IDs:

    untouched true cell A:
        all A supervoxels -> one current component

    selected touching pair A,B:
        all A and B supervoxels -> one current A+B component

Unannotated RAG nodes retain their original frozen spatial grouping.

Temporal reasoning may modify ONLY RAG edges whose endpoints already belong to
the same current synthetic component.

Edges between two separate current components remain immutable CUT.

Therefore the temporal model can:

    A+B -> A | B

but cannot:

    A | B -> A+B

Targets
-------
For editable internal RAG edges:

    true_id[src] == true_id[dst]   -> KEEP
    true_id[src] != true_id[dst]   -> CUT

CUT edges are exactly the erased boundaries between deliberately merged true
cells.

Training loss:
    CUT BCE
    + preservation_weight * KEEP BCE
    + small split-head BCE

No wrong-neighbourhood loss.
No contentless loss.
No merge objective.
No temporal metadata preprocessing.

Observer
--------
This focused CUT/KEEP experiment bypasses TemporalSpatialObserver.

The experiment first tests the direct question: can the Trackastra temporal graph
teach the reasoner where a deliberately merged current component should be cut?
No observer cache or dense temporal metadata is generated.

Validation/test
---------------
Whole Trackastra variants are held out.

Metrics:
    CUT accuracy
    KEEP accuracy
    exact merge recovery
    clean false-split rate
    split-only violations
    observer-cache hit rate

Typical commands
----------------
Prepare concrete variants only:

    python .\investigations\stirnet\35_biohub_temporal_merge_synthesis.py --prepare-only

After Trackastra finishes the final variant, preparation is DONE. There is no
"[temporal metadata]" phase.

100-step smoke test:

    python .\investigations\stirnet\35_biohub_temporal_merge_synthesis.py `
        --steps 100 `
        --eval-every 50 `
        --val-cases 12

First training run:

    python .\investigations\stirnet\35_biohub_temporal_merge_synthesis.py `
        --steps 1000 `
        --eval-every 100 `
        --val-cases 0
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
import tempfile
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


# =============================================================================
# Repository / current production modules
# =============================================================================


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents, Path.cwd().resolve()):
        if (
            (candidate / "learned" / "stirnet").is_dir()
            and (candidate / "investigations" / "stirnet" / "data").is_dir()
            and (candidate / "src").is_dir()
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


# Current checkpoint hydration compatibility layer.
INV12 = load_module(
    ROOT / "investigations" / "stirnet" / "data"
    / "12_biohub_full_volume_spatial_inference.py",
    "_inv35_clean_checkpoint_helper",
)

from src.io import load_timepoint
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


# =============================================================================
# Constants
# =============================================================================


SCRIPT_NAME = "35_biohub_temporal_merge_synthesis"
OBJECTIVE_VERSION = 1
DATASET_VERSION = 1

DEFAULT_SAMPLE = "44b6_0113de3b"
DEFAULT_FRAME_COUNT = 20
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)
DEFAULT_TEMPORAL_RADIUS = 2

# One concrete Trackastra movie already contains many simultaneous merge events.
# Six variants are enough for a first clean training experiment and keep
# --prepare-only short.
DEFAULT_TRAIN_VARIANTS = 4
DEFAULT_VAL_VARIANTS = 1
DEFAULT_TEST_VARIANTS = 1
DEFAULT_MERGE_FRACTION = 0.08
DEFAULT_MAX_MERGES_PER_FRAME = 12

DEFAULT_MIN_VOXELS = 64
DEFAULT_MAX_VOLUME_RATIO = 3.0

DEFAULT_TRACKASTRA_MODEL = "ctc"
DEFAULT_TRACKASTRA_MODE = "greedy"
DEFAULT_TRACKASTRA_DEVICE = "cuda"

DEFAULT_STEPS = 1000
DEFAULT_LR = 2.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_ACCUMULATE_CASES = 4
DEFAULT_EVAL_EVERY = 100
DEFAULT_PRINT_EVERY = 10
DEFAULT_VAL_CASES = 0

DEFAULT_SYNTHETIC_LOGIT = 3.5
DEFAULT_PRESERVE_EDGES = 512
DEFAULT_PRESERVE_RATIO = 12
DEFAULT_PRESERVATION_WEIGHT = 2.0
DEFAULT_SPLIT_WEIGHT = 0.05

DEFAULT_SEED = 20260827

NODE_DIM = 32
EDGE_DIM = 15
STATUS_DIM = 10

# Current graph_builder.py 32-D node feature layout.
GX_TIME = 0
GX_POS = slice(1, 4)
GX_LOG_VOLUME = 4
GX_BBOX = slice(5, 8)
GX_PCA = slice(8, 11)
GX_ELONGATION = 11
GX_FLATNESS = 12
GX_SOLIDITY = 13
GX_COMPACTNESS = 14
GX_INTENSITY_MEAN = 15
GX_INTENSITY_STD = 16
GX_BACK_VEL = slice(17, 20)
GX_FWD_VEL = slice(20, 23)
GX_LENGTH_BEFORE = 23
GX_LENGTH_AFTER = 24
GX_VOLUME_BOUNDARY = 25
GX_PATCH_BOUNDARY = 26
GX_IS_CURRENT = 27
GX_INTERIOR_START = 28
GX_INTERIOR_END = 29
GX_DIVISION = 30
GX_BOUNDARY = 31


# =============================================================================
# Generic helpers
# =============================================================================


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


def parse_spacing(text: str) -> tuple[float, float, float]:
    values = tuple(float(v.strip()) for v in str(text).split(","))
    if len(values) != 3 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("--spacing must be three positive finite Z,Y,X values")
    return values


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        return jsonable(tensor.item()) if tensor.ndim == 0 else jsonable(tensor.tolist())
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def torch_load(path: Path, map_location="cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


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


def map_tree(value: Any, fn):
    if torch.is_tensor(value):
        return fn(value)
    if dataclasses.is_dataclass(value):
        return type(value)(
            **{
                field.name: map_tree(getattr(value, field.name), fn)
                for field in dataclasses.fields(value)
            }
        )
    if isinstance(value, dict):
        return {k: map_tree(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [map_tree(v, fn) for v in value]
    if isinstance(value, tuple):
        return tuple(map_tree(v, fn) for v in value)
    return value


def tree_to_device_fp32(value: Any, device: torch.device) -> Any:
    def move(tensor: Tensor) -> Tensor:
        if tensor.is_floating_point():
            return tensor.to(device=device, dtype=torch.float32)
        return tensor.to(device=device)
    return map_tree(value, move)


def reference_key(ref_um: Sequence[float]) -> tuple[int, int, int]:
    values = np.asarray(ref_um, np.float64)
    return tuple(int(v) for v in np.rint(values * 10_000.0).astype(np.int64))


# =============================================================================
# Paths
# =============================================================================


@dataclass(frozen=True)
class Paths:
    sample: str
    run_root: Path
    output: Path
    annotations: Path
    zarr: Path
    spatial_cache: Path
    checkpoint: Path

    @property
    def movie_dir(self) -> Path:
        return self.output / "movies"

    @property
    def raw_movie(self) -> Path:
        return self.movie_dir / "raw.npy"

    @property
    def manual_movie(self) -> Path:
        return self.movie_dir / "manual.npy"

    @property
    def movie_meta(self) -> Path:
        return self.movie_dir / "meta.json"

    @property
    def touching_pairs(self) -> Path:
        return self.output / "touching_pairs.json"

    @property
    def dataset_manifest(self) -> Path:
        return self.output / "dataset_manifest.json"

    @property
    def variants(self) -> Path:
        return self.output / "variants"

    def variant_dir(self, index: int) -> Path:
        return self.variants / f"variant_{index:03d}"

    def merge_plan(self, index: int) -> Path:
        return self.variant_dir(index) / "merge_plan.json"

    def track_graph(self, index: int) -> Path:
        return self.variant_dir(index) / "track_graph.pkl"

    def variant_success(self, index: int) -> Path:
        return self.variant_dir(index) / "_SUCCESS.json"

    def graph_cache(self, t: int) -> Path:
        return self.spatial_cache / f"t{t:03d}" / "frozen_graph.pt"

    def spatial_meta(self, t: int) -> Path:
        return self.spatial_cache / f"t{t:03d}" / "meta.json"

    def observer_refs(self, t: int) -> Path:
        return self.spatial_cache / f"t{t:03d}" / "observer_ref_um.npy"

    def observer_d1(self, t: int) -> Path:
        return self.spatial_cache / f"t{t:03d}" / "observer_d1_raw.npy"

    def observer_d2(self, t: int) -> Path:
        return self.spatial_cache / f"t{t:03d}" / "observer_d2_raw.npy"

    def observer_hidden(self, t: int) -> Path:
        return self.spatial_cache / f"t{t:03d}" / "observer_hidden_raw.npy"

    def observer_explicit(self, t: int) -> Path:
        return self.spatial_cache / f"t{t:03d}" / "observer_explicit_raw.npy"

    @property
    def training_history(self) -> Path:
        return self.output / "training_history.json"

    @property
    def validation_history(self) -> Path:
        return self.output / "validation_history.json"

    @property
    def best_metrics(self) -> Path:
        return self.output / "best_metrics.json"

    @property
    def best(self) -> Path:
        return self.output / "best.pt"

    @property
    def latest(self) -> Path:
        return self.output / "latest.pt"

    @property
    def final(self) -> Path:
        return self.output / "final.pt"

    @property
    def test_metrics(self) -> Path:
        return self.output / "test_metrics.json"


def resolve_current_spatial_checkpoint(override: Path | None) -> Path:
    """Resolve the current spatial checkpoint without any old Inv35 cache."""
    if override is not None:
        path = resolve(override)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    # Current BioHub spatial visualization / leaderboard-side recovery.
    recovery = (
        ROOT
        / "runs"
        / "stirnet"
        / "investigations"
        / "19_morphology_rag_v2_headroom_training"
        / "recovery"
        / "drosophila_12_morphology_rag_v2_headroom_h100"
    )
    for candidate in (
        recovery / "best_checkpoint.pt",
        recovery / "checkpoint_step_000600.pt",
    ):
        if candidate.is_file():
            return candidate.resolve()
    steps = sorted(recovery.glob("checkpoint_step_*.pt"))
    if steps:
        return steps[-1].resolve()

    # Fallback used by the earlier spatial milestone.
    milestone = (
        ROOT
        / "runs"
        / "stirnet"
        / "milestones"
        / "drosophila_12_spatial_v1"
    )
    for candidate in (
        milestone / "best_checkpoint.pt",
        milestone / "checkpoint_step_000500.pt",
        milestone / "final.pt",
    ):
        if candidate.is_file():
            return candidate.resolve()
    steps = sorted(milestone.glob("checkpoint_step_*.pt"))
    if steps:
        return steps[-1].resolve()

    raise FileNotFoundError(
        "Could not resolve the current spatial checkpoint. "
        "Pass --checkpoint explicitly."
    )


def make_paths(args: argparse.Namespace) -> Paths:
    sample = str(args.sample_id)
    run_root = (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / SCRIPT_NAME
        / sample
    ).resolve()

    output = (
        resolve(args.output)
        if args.output is not None
        else (run_root / "concrete_cutkeep_v3").resolve()
    )

    # The clean rewrite owns and recreates this cache itself.
    spatial_cache = (
        resolve(args.spatial_cache_root)
        if args.spatial_cache_root is not None
        else (output / "spatial_cache").resolve()
    )

    annotations = (
        resolve(args.annotations)
        if args.annotations is not None
        else (
            ROOT
            / "evaluation"
            / "segmentation"
            / "annotations"
            / sample
        ).resolve()
    )

    zarr = (
        resolve(args.sample_zarr)
        if args.sample_zarr is not None
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

    checkpoint = resolve_current_spatial_checkpoint(args.checkpoint)

    return Paths(
        sample=sample,
        run_root=run_root,
        output=output,
        annotations=annotations,
        zarr=zarr,
        spatial_cache=spatial_cache,
        checkpoint=checkpoint,
    )


def validate_inputs(paths: Paths, frame_count: int) -> None:
    """
    No previous Investigation-35 data is required.

    Required clean inputs:
      - manual true-instance annotations
      - raw BioHub movie
      - current spatial checkpoint
    """
    missing: list[Path] = []

    for t in range(frame_count):
        annotation = paths.annotations / f"manual_instances_t{t:03d}.npy"
        if not annotation.is_file():
            missing.append(annotation)

    if not paths.zarr.exists() and not paths.raw_movie.is_file():
        missing.append(paths.zarr)

    if not paths.checkpoint.is_file():
        missing.append(paths.checkpoint)

    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:40])
        raise FileNotFoundError(
            "Required clean inputs are missing:\n"
            + preview
            + "\n\nNo previous Investigation-35 cache is required."
        )


# =============================================================================
# Movie cache
# =============================================================================


def annotation_signatures(paths: Paths, frame_count: int) -> list[dict[str, Any]]:
    rows = []
    for t in range(frame_count):
        path = paths.annotations / f"manual_instances_t{t:03d}.npy"
        stat = path.stat()
        rows.append(
            {
                "path": str(path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return rows


def build_movies(paths: Paths, frame_count: int, *, rebuild: bool) -> None:
    signatures = annotation_signatures(paths, frame_count)

    reusable = False
    if (
        not rebuild
        and paths.raw_movie.is_file()
        and paths.manual_movie.is_file()
        and paths.movie_meta.is_file()
    ):
        try:
            meta = json.loads(paths.movie_meta.read_text(encoding="utf-8"))
            manual = np.load(paths.manual_movie, mmap_mode="r", allow_pickle=False)
            raw = np.load(paths.raw_movie, mmap_mode="r", allow_pickle=False)
            reusable = (
                int(manual.shape[0]) == frame_count
                and raw.shape == manual.shape
                and meta.get("annotation_signatures") == signatures
            )
        except Exception:
            reusable = False

    if reusable:
        print("[movies] reuse raw + manual", flush=True)
        return

    paths.movie_dir.mkdir(parents=True, exist_ok=True)

    first_manual = np.load(
        paths.annotations / "manual_instances_t000.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    first_raw = np.asarray(load_timepoint(paths.zarr, 0))
    if first_raw.shape != first_manual.shape:
        raise ValueError(
            f"Raw/manual shape mismatch: {first_raw.shape} vs {first_manual.shape}"
        )

    manual_movie = np.lib.format.open_memmap(
        paths.manual_movie,
        mode="w+",
        dtype=first_manual.dtype,
        shape=(frame_count, *first_manual.shape),
    )
    raw_movie = np.lib.format.open_memmap(
        paths.raw_movie,
        mode="w+",
        dtype=first_raw.dtype,
        shape=(frame_count, *first_raw.shape),
    )

    for t in range(frame_count):
        if t:
            print(f"[movies] t={t:03d}", flush=True)
        manual_movie[t] = np.load(
            paths.annotations / f"manual_instances_t{t:03d}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        raw_movie[t] = np.asarray(load_timepoint(paths.zarr, t))

    manual_movie.flush()
    raw_movie.flush()
    del manual_movie, raw_movie

    atomic_json(
        paths.movie_meta,
        {
            "version": 1,
            "sample_id": paths.sample,
            "frame_count": int(frame_count),
            "annotation_signatures": signatures,
        },
    )


# =============================================================================
# Atomic RAG -> true-cell mapping / touching pairs
# =============================================================================


def sv_label_lookup(
    supervoxels: np.ndarray,
    labels: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    sv = np.asarray(supervoxels, np.int64).reshape(-1)
    lab = np.asarray(labels, np.int64).reshape(-1)

    max_sv = int(sv.max(initial=0))
    lo = np.full(max_sv + 1, np.iinfo(np.int64).max, np.int64)
    hi = np.full(max_sv + 1, -1, np.int64)

    positive = sv > 0
    np.minimum.at(lo, sv[positive], lab[positive])
    np.maximum.at(hi, sv[positive], lab[positive])

    present = hi >= 0
    bad = present & (lo != hi)
    if bad.any():
        raise RuntimeError(
            f"{name}: a true annotation boundary cuts through an atomic "
            f"supervoxel; examples={np.flatnonzero(bad)[:20].tolist()}"
        )

    result = np.zeros(max_sv + 1, np.int64)
    result[present] = hi[present]
    return result


def physically_touching_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    data = np.asarray(labels)
    result: set[tuple[int, int]] = set()

    for axis in range(3):
        left_slices = [slice(None)] * 3
        right_slices = [slice(None)] * 3
        left_slices[axis] = slice(0, -1)
        right_slices[axis] = slice(1, None)

        left = data[tuple(left_slices)]
        right = data[tuple(right_slices)]
        valid = (left > 0) & (right > 0) & (left != right)
        if not valid.any():
            continue

        a = left[valid].astype(np.int64, copy=False)
        b = right[valid].astype(np.int64, copy=False)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        encoded = (lo << 32) | hi

        for value in np.unique(encoded).tolist():
            result.add(
                (
                    int(value >> 32),
                    int(value & 0xFFFFFFFF),
                )
            )

    return result


@dataclass(frozen=True)
class TouchPair:
    frame: int
    a: int
    b: int
    interface_edges: int
    voxels_a: int
    voxels_b: int
    volume_ratio: float


def build_touching_catalog(
    paths: Paths,
    frame_count: int,
    *,
    min_voxels: int,
    max_volume_ratio: float,
    rebuild: bool,
) -> dict[int, list[TouchPair]]:
    """
    Build merge candidates directly from manual true instances.

    No RAG is needed to synthesize concrete merge movies. Whether a selected
    pair has an editable RAG interface is resolved later from the freshly
    reconstructed spatial cache.
    """
    if paths.touching_pairs.is_file() and not rebuild:
        payload = json.loads(paths.touching_pairs.read_text(encoding="utf-8"))
        if int(payload.get("version", -1)) == DATASET_VERSION:
            result = {
                int(t): [TouchPair(**row) for row in rows]
                for t, rows in payload["frames"].items()
            }
            print(
                f"[touching pairs] reuse total="
                f"{sum(len(rows) for rows in result.values())}",
                flush=True,
            )
            return result

    manual_movie = np.load(
        paths.manual_movie,
        mmap_mode="r",
        allow_pickle=False,
    )
    result: dict[int, list[TouchPair]] = {}
    serializable: dict[str, list[dict[str, Any]]] = {}

    for t in range(frame_count):
        manual = np.asarray(manual_movie[t])
        counts = np.bincount(
            manual.reshape(-1).astype(np.int64, copy=False)
        )

        rows: list[TouchPair] = []
        for a, b in sorted(physically_touching_pairs(manual)):
            vox_a = int(counts[a]) if a < len(counts) else 0
            vox_b = int(counts[b]) if b < len(counts) else 0
            if vox_a < int(min_voxels) or vox_b < int(min_voxels):
                continue

            ratio = max(
                vox_a / max(vox_b, 1),
                vox_b / max(vox_a, 1),
            )
            if ratio > float(max_volume_ratio):
                continue

            rows.append(
                TouchPair(
                    frame=int(t),
                    a=int(a),
                    b=int(b),
                    interface_edges=0,
                    voxels_a=vox_a,
                    voxels_b=vox_b,
                    volume_ratio=float(ratio),
                )
            )

        result[t] = rows
        serializable[str(t)] = [
            dataclasses.asdict(row)
            for row in rows
        ]
        print(
            f"[touching pairs t={t:03d}] {len(rows)}",
            flush=True,
        )

    atomic_json(
        paths.touching_pairs,
        {
            "version": DATASET_VERSION,
            "sample_id": paths.sample,
            "source": "manual_true_instances_only",
            "min_voxels": int(min_voxels),
            "max_volume_ratio": float(max_volume_ratio),
            "total_pairs": int(
                sum(len(rows) for rows in result.values())
            ),
            "frames": serializable,
        },
    )
    return result


# =============================================================================
# Variant planning
# =============================================================================


@dataclass(frozen=True)
class MergeEvent:
    frame: int
    a: int
    b: int
    representative: int
    interface_edges: int


@dataclass(frozen=True)
class VariantPlan:
    index: int
    split: str
    events_by_frame: dict[int, tuple[MergeEvent, ...]]

    @property
    def merge_count(self) -> int:
        return sum(len(rows) for rows in self.events_by_frame.values())


def build_one_variant(
    *,
    index: int,
    split: str,
    catalog: dict[int, list[TouchPair]],
    frame_count: int,
    merge_fraction: float,
    max_merges_per_frame: int,
    seed: int,
) -> VariantPlan:
    events_by_frame: dict[int, tuple[MergeEvent, ...]] = {}

    for t in range(frame_count):
        candidates = list(catalog.get(t, ()))
        rng = random.Random(
            int(seed) + 1_000_003 * int(index) + 10_007 * int(t)
        )
        rng.shuffle(candidates)

        target = min(
            int(max_merges_per_frame),
            max(
                1 if candidates else 0,
                int(round(len(candidates) * float(merge_fraction))),
            ),
        )

        selected: list[MergeEvent] = []
        used: set[int] = set()

        for pair in candidates:
            if len(selected) >= target:
                break
            if pair.a in used or pair.b in used:
                continue

            selected.append(
                MergeEvent(
                    frame=int(t),
                    a=int(pair.a),
                    b=int(pair.b),
                    representative=min(int(pair.a), int(pair.b)),
                    interface_edges=int(pair.interface_edges),
                )
            )
            used.update((int(pair.a), int(pair.b)))

        events_by_frame[t] = tuple(selected)

    return VariantPlan(
        index=int(index),
        split=str(split),
        events_by_frame=events_by_frame,
    )


def build_variant_plans(
    paths: Paths,
    *,
    catalog: dict[int, list[TouchPair]],
    frame_count: int,
    train_variants: int,
    val_variants: int,
    test_variants: int,
    merge_fraction: float,
    max_merges_per_frame: int,
    seed: int,
    rebuild: bool,
) -> list[VariantPlan]:
    total = train_variants + val_variants + test_variants
    splits = (
        ["train"] * train_variants
        + ["val"] * val_variants
        + ["test"] * test_variants
    )

    plans = [
        build_one_variant(
            index=index,
            split=splits[index],
            catalog=catalog,
            frame_count=frame_count,
            merge_fraction=merge_fraction,
            max_merges_per_frame=max_merges_per_frame,
            seed=seed,
        )
        for index in range(total)
    ]

    manifest = {
        "version": DATASET_VERSION,
        "sample_id": paths.sample,
        "frame_count": int(frame_count),
        "seed": int(seed),
        "merge_fraction": float(merge_fraction),
        "max_merges_per_frame": int(max_merges_per_frame),
        "plans": [
            {
                "index": plan.index,
                "split": plan.split,
                "merge_count": plan.merge_count,
                "events_by_frame": {
                    str(t): [
                        dataclasses.asdict(event)
                        for event in plan.events_by_frame.get(t, ())
                    ]
                    for t in range(frame_count)
                },
            }
            for plan in plans
        ],
    }

    if paths.dataset_manifest.is_file() and not rebuild:
        previous = json.loads(paths.dataset_manifest.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError(
                "Existing concrete dataset configuration differs from the "
                "requested one. Use --rebuild-dataset."
            )
    else:
        if rebuild and paths.variants.exists():
            shutil.rmtree(paths.variants)
        atomic_json(paths.dataset_manifest, manifest)

    for plan in plans:
        paths.variant_dir(plan.index).mkdir(parents=True, exist_ok=True)
        atomic_json(
            paths.merge_plan(plan.index),
            {
                "version": DATASET_VERSION,
                "variant_index": int(plan.index),
                "split": plan.split,
                "merge_count": int(plan.merge_count),
                "events_by_frame": {
                    str(t): [
                        dataclasses.asdict(event)
                        for event in plan.events_by_frame.get(t, ())
                    ]
                    for t in range(frame_count)
                },
            },
        )

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 35 — CONCRETE VARIANT PLAN", flush=True)
    print("=" * 112, flush=True)
    for split in ("train", "val", "test"):
        subset = [plan for plan in plans if plan.split == split]
        print(
            f"{split:5s} variants={len(subset):2d} "
            f"merge_events={sum(plan.merge_count for plan in subset)}",
            flush=True,
        )
    print("=" * 112, flush=True)

    return plans


# =============================================================================
# Concrete Trackastra variants
# =============================================================================


def apply_merge_events(clean: np.ndarray, events: Sequence[MergeEvent]) -> np.ndarray:
    result = np.asarray(clean).astype(np.int32, copy=True)
    for event in events:
        result[result == int(event.a)] = int(event.representative)
        result[result == int(event.b)] = int(event.representative)
    return result


def variant_ready(paths: Paths, plan: VariantPlan) -> bool:
    return (
        paths.track_graph(plan.index).is_file()
        and paths.variant_success(plan.index).is_file()
    )


def edge_score(data: dict[str, Any]) -> float | None:
    for key in ("weight", "score", "probability", "confidence"):
        value = data.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except Exception:
            continue
        if math.isfinite(value):
            return value
    return None


def annotate_graph_fast(graph, tracked_masks: np.ndarray) -> None:
    """
    Persist only cheap information needed later.

    One bincount per frame gives every detection volume. Trackastra normally
    stores detection coordinates itself. If a Trackastra version does not,
    centroids are recovered in one vectorized pass over the tracked mask.
    """
    nodes_by_time: dict[int, list[int]] = defaultdict(list)
    for node_id, data in graph.nodes(data=True):
        nodes_by_time[int(data["time"])].append(int(node_id))

    for t, node_ids in nodes_by_time.items():
        labels = np.asarray(tracked_masks[int(t)])
        flat = labels.reshape(-1).astype(np.int64, copy=False)
        counts = np.bincount(flat)

        missing_coords = [
            node_id
            for node_id in node_ids
            if not any(
                key in graph.nodes[node_id]
                for key in ("coords", "coord", "position", "centroid")
            )
        ]

        centroid_by_label: dict[int, tuple[float, float, float]] = {}
        if missing_coords:
            positive = labels > 0
            z, y, x = np.nonzero(positive)
            ids = labels[positive].astype(np.int64, copy=False)
            if len(ids):
                max_id = int(ids.max(initial=0))
                c = np.bincount(ids, minlength=max_id + 1).astype(np.float64)
                sz = np.bincount(ids, weights=z, minlength=max_id + 1)
                sy = np.bincount(ids, weights=y, minlength=max_id + 1)
                sx = np.bincount(ids, weights=x, minlength=max_id + 1)
                for label_id in np.unique(ids).tolist():
                    label_id = int(label_id)
                    if c[label_id] > 0:
                        centroid_by_label[label_id] = (
                            float(sz[label_id] / c[label_id]),
                            float(sy[label_id] / c[label_id]),
                            float(sx[label_id] / c[label_id]),
                        )

        for node_id in node_ids:
            data = graph.nodes[node_id]
            label = int(data["label"])
            data["inv35_volume_voxels"] = (
                int(counts[label])
                if 0 <= label < len(counts)
                else 1
            )
            if node_id in missing_coords:
                center = centroid_by_label.get(label)
                if center is None:
                    raise RuntimeError(
                        f"Could not recover Trackastra centroid for "
                        f"t={t}, label={label}"
                    )
                data["inv35_coords_zyx"] = center


def prepare_variants(
    paths: Paths,
    plans: Sequence[VariantPlan],
    frame_count: int,
    *,
    model_name: str,
    mode: str,
    device: str,
    rebuild: bool,
) -> None:
    todo = [
        plan
        for plan in plans
        if rebuild or not variant_ready(paths, plan)
    ]

    if not todo:
        print("[Trackastra variants] all ready", flush=True)
        return

    try:
        from trackastra.model import Trackastra
    except ImportError as exc:
        raise RuntimeError("Trackastra is required for Investigation 35") from exc

    raw = np.load(paths.raw_movie, mmap_mode="r", allow_pickle=False)
    manual = np.load(paths.manual_movie, mmap_mode="r", allow_pickle=False)

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 35 — TRACKASTRA ON CONCRETE MERGE MOVIES", flush=True)
    print("=" * 112, flush=True)
    print(f"variants : {[plan.index for plan in todo]}", flush=True)
    print(f"model    : {model_name}", flush=True)
    print(f"mode     : {mode}", flush=True)
    print(f"device   : {device}", flush=True)
    print("=" * 112, flush=True)

    tracker = Trackastra.from_pretrained(model_name, device=device)

    for ordinal, plan in enumerate(todo, 1):
        started = time.perf_counter()
        directory = paths.variant_dir(plan.index)
        directory.mkdir(parents=True, exist_ok=True)

        temporary = directory / "_labels_tmp.npy"
        temporary.unlink(missing_ok=True)
        synthetic = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.int32,
            shape=manual.shape,
        )

        for t in range(frame_count):
            synthetic[t] = apply_merge_events(
                np.asarray(manual[t]),
                plan.events_by_frame.get(t, ()),
            )
        synthetic.flush()

        print(
            f"[variant {plan.index:03d} {plan.split}] "
            f"{ordinal}/{len(todo)} merges={plan.merge_count}",
            flush=True,
        )

        graph, tracked_masks = tracker.track(raw, synthetic, mode=mode)
        annotate_graph_fast(graph, np.asarray(tracked_masks))
        atomic_pickle(paths.track_graph(plan.index), graph)

        elapsed = time.perf_counter() - started
        atomic_json(
            paths.variant_success(plan.index),
            {
                "version": DATASET_VERSION,
                "variant_index": int(plan.index),
                "split": plan.split,
                "merge_count": int(plan.merge_count),
                "trackastra_nodes": int(graph.number_of_nodes()),
                "trackastra_edges": int(graph.number_of_edges()),
                "seconds": float(elapsed),
            },
        )

        del synthetic, graph, tracked_masks
        temporary.unlink(missing_ok=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"[variant {plan.index:03d}] DONE "
            f"time={duration(elapsed)}",
            flush=True,
        )

    del tracker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# Rebuild compact spatial RAG/statistics from the CURRENT spatial model
# =============================================================================


def parse_zyx_ints(value: str, *, name: str) -> tuple[int, int, int]:
    rows = tuple(int(token.strip()) for token in str(value).split(","))
    if len(rows) != 3:
        raise ValueError(f"{name} must contain three Z,Y,X integers")
    return rows


def spatial_frame_complete(paths: Paths, t: int) -> bool:
    return paths.graph_cache(t).is_file() and paths.spatial_meta(t).is_file()


def cpu_detached_tree(value: Any) -> Any:
    return map_tree(
        value,
        lambda tensor: tensor.detach().cpu(),
    )


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def rebuild_spatial_cache(
    paths: Paths,
    frame_count: int,
    *,
    spacing: Sequence[float],
    device: torch.device,
    tile_shape_zyx: tuple[int, int, int],
    tile_overlap_zyx: tuple[int, int, int],
    tile_halo_zyx: tuple[int, int, int],
    tile_batch_size: int,
    rebuild: bool,
) -> None:
    """
    Recreate only the mature spatial state needed by temporal training.

    Per frame:
        raw
        -> canonical preprocessing
        -> current source segmentation
        -> current 5-channel STIR-Net spatial input
        -> current tiled spatial STIR-Net
        -> atomic supervoxels / RAG / compact statistics / spatial partition

    No temporal metadata is constructed.
    """
    todo = [
        t
        for t in range(frame_count)
        if rebuild or not spatial_frame_complete(paths, t)
    ]
    if not todo:
        print("[spatial cache] all frames ready", flush=True)
        return

    from dataclasses import replace as dc_replace
    from importlib import import_module

    from src.api import (
        create_binary_mask,
        preprocess_volume,
        segment_instances,
    )

    segmentation_config_module = import_module(
        "src.03_segmentation.config"
    )
    source_config = dc_replace(
        segmentation_config_module.DEFAULT_SEGMENTATION_CONFIG,
        enable_geometric_completion=False,
    )

    print("\n" + "=" * 112, flush=True)
    print(
        "INVESTIGATION 35 — REBUILD CURRENT FROZEN SPATIAL CACHE",
        flush=True,
    )
    print("=" * 112, flush=True)
    print(f"checkpoint : {paths.checkpoint}", flush=True)
    print(f"frames     : {todo}", flush=True)
    print(f"device     : {device}", flush=True)
    print(
        f"tiles      : shape={tile_shape_zyx} "
        f"overlap={tile_overlap_zyx} "
        f"halo={tile_halo_zyx} "
        f"batch={tile_batch_size}",
        flush=True,
    )
    print("=" * 112, flush=True)

    (
        checkpoint_payload,
        model,
        model_cfg,
        _train_cfg,
        _stripped_training_config,
    ) = INV12.load_checkpoint_model_for_inference(
        paths.checkpoint,
        device,
    )
    model.eval()

    inference_cfg = INV12.build_inference_config(
        model_cfg,
        tile_shape_zyx=tile_shape_zyx,
        tile_overlap_zyx=tile_overlap_zyx,
        tile_halo_zyx=tile_halo_zyx,
        tile_batch_size=int(tile_batch_size),
    )

    raw_movie = np.load(
        paths.raw_movie,
        mmap_mode="r",
        allow_pickle=False,
    )
    paths.spatial_cache.mkdir(parents=True, exist_ok=True)

    for ordinal, t in enumerate(todo, 1):
        started = time.perf_counter()
        frame_dir = paths.spatial_cache / f"t{t:03d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[spatial cache t={t:03d}] "
            f"{ordinal}/{len(todo)}",
            flush=True,
        )

        raw = np.asarray(raw_movie[t])
        preprocessed = preprocess_volume(raw)
        source_mask = create_binary_mask(preprocessed)
        source_labels = segment_instances(
            source_mask,
            config=source_config,
        )

        spatial, dref_um = INV12.build_stage6_spatial_input(
            preprocessed,
            source_labels,
            tuple(float(v) for v in spacing),
        )

        result, spatial_gpu, amp_name, inference_seconds, peak_gib = (
            INV12.run_tiled_spatial(
                model,
                spatial,
                tuple(float(v) for v in spacing),
                float(dref_um),
                device=device,
                inference_cfg=inference_cfg,
            )
        )

        if result.rag.statistics is None:
            raise RuntimeError(
                "Current tiled spatial inference returned no compact "
                "supervoxel statistics. These are required by "
                "InstanceTokenizer."
            )

        atomic_torch_save(
            paths.graph_cache(t),
            {
                "version": 1,
                "rag": cpu_detached_tree(result.rag),
                "actual_partition": cpu_detached_tree(
                    result.spatial_partition
                ),
            },
        )

        atomic_json(
            paths.spatial_meta(t),
            {
                "version": 1,
                "sample_id": paths.sample,
                "timepoint": int(t),
                "checkpoint_signature": {
                    "path": str(paths.checkpoint.resolve()),
                    "global_step": int(
                        checkpoint_payload.get("global_step", -1)
                    ),
                },
                "spacing_zyx_um": [
                    float(v)
                    for v in spacing
                ],
                "dref_um": float(dref_um),
                "source_instances": int(
                    np.count_nonzero(
                        np.unique(source_labels) > 0
                    )
                ),
                "rag_nodes": int(
                    result.rag.node_features.shape[0]
                ),
                "rag_edges": int(
                    result.rag.edge_index.shape[1]
                ),
                "spatial_components": int(
                    result.spatial_partition
                    .component_count_per_batch
                    .sum()
                    .item()
                ),
                "amp_dtype": str(amp_name),
                "spatial_inference_seconds": float(
                    inference_seconds
                ),
                "peak_allocated_vram_gib": float(peak_gib),
                "total_seconds": float(
                    time.perf_counter() - started
                ),
            },
        )

        print(
            f"[spatial cache t={t:03d}] DONE "
            f"nodes={int(result.rag.node_features.shape[0])} "
            f"edges={int(result.rag.edge_index.shape[1])} "
            f"dref={float(dref_um):.4f}um "
            f"infer={inference_seconds:.1f}s "
            f"total={duration(time.perf_counter() - started)}",
            flush=True,
        )

        del (
            result,
            spatial_gpu,
            spatial,
            preprocessed,
            source_mask,
            source_labels,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# Direct Trackastra graph -> production TemporalInput
# =============================================================================


class UnionFind:
    def __init__(self, count: int):
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> None:
        a = self.find(a)
        b = self.find(b)
        if a != b:
            self.parent[b] = a


def get_coords_zyx(node_data: dict[str, Any]) -> np.ndarray:
    for key in (
        "inv35_coords_zyx",
        "coords",
        "coord",
        "position",
        "centroid",
    ):
        value = node_data.get(key)
        if value is None:
            continue
        array = np.asarray(value, np.float32).reshape(-1)
        if array.size == 3 and np.isfinite(array).all():
            return array
    raise KeyError(
        "Trackastra node does not expose a 3-D coordinate under "
        "coords/coord/position/centroid"
    )


def physical_position_um(
    coords_zyx: Sequence[float],
    shape_zyx: Sequence[int],
    spacing: Sequence[float],
) -> np.ndarray:
    coords = np.asarray(coords_zyx, np.float32)
    spacing_np = np.asarray(spacing, np.float32)
    center = 0.5 * (np.asarray(shape_zyx, np.float32) - 1.0) * spacing_np
    return coords * spacing_np - center


def selected_nodes(
    graph,
    target_t: int,
    frame_count: int,
    radius: int,
) -> tuple[list[int], tuple[int, ...]]:
    start = max(0, int(target_t) - int(radius))
    stop = min(int(frame_count) - 1, int(target_t) + int(radius))
    offsets = tuple(t - int(target_t) for t in range(start, stop + 1))
    allowed = set(range(start, stop + 1))

    node_ids = [
        int(node_id)
        for node_id, data in graph.nodes(data=True)
        if int(data["time"]) in allowed
    ]
    node_ids.sort(key=lambda node_id: (int(graph.nodes[node_id]["time"]), node_id))
    return node_ids, offsets


def build_tracklets(
    graph,
    node_ids: Sequence[int],
) -> tuple[np.ndarray, list[list[int]], dict[int, int]]:
    """
    Match the current graph_builder one-to-one association rule:
    accepted edge joins one tracklet only when source outdegree==1 and
    destination indegree==1 inside the selected temporal window.
    """
    id_to_row = {int(node_id): row for row, node_id in enumerate(node_ids)}
    indegree = np.zeros(len(node_ids), np.int64)
    outdegree = np.zeros(len(node_ids), np.int64)
    edges: list[tuple[int, int]] = []

    for source, destination in graph.edges():
        source = int(source)
        destination = int(destination)
        if source not in id_to_row or destination not in id_to_row:
            continue
        s = id_to_row[source]
        d = id_to_row[destination]
        outdegree[s] += 1
        indegree[d] += 1
        edges.append((s, d))

    union = UnionFind(len(node_ids))
    for s, d in edges:
        if outdegree[s] == 1 and indegree[d] == 1:
            union.union(s, d)

    members: dict[int, list[int]] = defaultdict(list)
    for row in range(len(node_ids)):
        members[union.find(row)].append(row)

    roots = sorted(
        members,
        key=lambda root: min(int(node_ids[row]) for row in members[root]),
    )
    root_to_tracklet = {root: index for index, root in enumerate(roots)}
    tracklet_id = np.asarray(
        [root_to_tracklet[union.find(row)] for row in range(len(node_ids))],
        np.int64,
    )
    groups = [members[root] for root in roots]
    return tracklet_id, groups, id_to_row


def tracklet_reference(
    rows: Sequence[int],
    times: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    rows = sorted(rows, key=lambda row: float(times[row]))
    current = [row for row in rows if times[row] == 0]
    if current:
        return positions[current[0]].astype(np.float32, copy=True)

    past = [row for row in rows if times[row] < 0]
    future = [row for row in rows if times[row] > 0]

    if past and future:
        a = max(past, key=lambda row: times[row])
        b = min(future, key=lambda row: times[row])
        fraction = (0.0 - times[a]) / max(times[b] - times[a], 1e-6)
        return (
            positions[a] * (1.0 - fraction) + positions[b] * fraction
        ).astype(np.float32)

    if len(past) >= 2:
        a, b = sorted(past, key=lambda row: times[row])[-2:]
        velocity = (positions[b] - positions[a]) / max(times[b] - times[a], 1e-6)
        return (positions[b] + velocity * (0.0 - times[b])).astype(np.float32)

    if past:
        return positions[past[-1]].astype(np.float32, copy=True)

    if len(future) >= 2:
        a, b = sorted(future, key=lambda row: times[row])[:2]
        velocity = (positions[b] - positions[a]) / max(times[b] - times[a], 1e-6)
        return (positions[a] - velocity * times[a]).astype(np.float32)

    return positions[future[0]].astype(np.float32, copy=True)


def local_velocity(
    graph,
    node_id: int,
    positions_by_id: dict[int, np.ndarray],
    *,
    predecessors: bool,
) -> np.ndarray:
    neighbours = (
        list(graph.predecessors(node_id))
        if predecessors
        else list(graph.successors(node_id))
    )

    p0 = positions_by_id[node_id]
    t0 = int(graph.nodes[node_id]["time"])
    rows = []

    for other in neighbours:
        other = int(other)
        if other not in positions_by_id:
            continue
        dt = abs(int(graph.nodes[other]["time"]) - t0)
        if dt <= 0:
            continue
        if predecessors:
            delta = p0 - positions_by_id[other]
        else:
            delta = positions_by_id[other] - p0
        rows.append(delta / float(dt))

    if not rows:
        return np.zeros(3, np.float32)
    return np.mean(np.stack(rows), axis=0).astype(np.float32)


def direct_temporal_input(
    graph,
    *,
    target_t: int,
    frame_count: int,
    temporal_radius: int,
    spacing: Sequence[float],
    dref_um: float,
    shape_zyx: Sequence[int],
    device: torch.device,
    k_spatial_neighbors: int = 6,
    spatial_radius_dref: float = 2.5,
) -> TemporalInput:
    node_ids, available_offsets = selected_nodes(
        graph,
        target_t,
        frame_count,
        temporal_radius,
    )

    if not node_ids:
        return TemporalInput(
            graph_x=torch.zeros((0, NODE_DIM), device=device),
            graph_edge_index=torch.zeros((2, 0), device=device, dtype=torch.long),
            graph_edge_attr=torch.zeros((0, EDGE_DIM), device=device),
            tracklet_id=torch.zeros((0,), device=device, dtype=torch.long),
            temporal_ref_um=torch.zeros((0, 3), device=device),
            temporal_status=torch.zeros((0, STATUS_DIM), device=device),
            temporal_batch=torch.zeros((0,), device=device, dtype=torch.long),
            node_history_embedding=None,
            hypothesis_edge_index=None,
            hypothesis_edge_attr=None,
        )

    id_to_row = {node_id: row for row, node_id in enumerate(node_ids)}

    positions = np.stack(
        [
            physical_position_um(
                get_coords_zyx(graph.nodes[node_id]),
                shape_zyx,
                spacing,
            )
            for node_id in node_ids
        ],
        axis=0,
    ).astype(np.float32)

    positions_by_id = {
        node_id: positions[row]
        for row, node_id in enumerate(node_ids)
    }

    times = np.asarray(
        [int(graph.nodes[node_id]["time"]) - int(target_t) for node_id in node_ids],
        np.float32,
    )

    voxel_volume = float(np.prod(np.asarray(spacing, np.float64)))
    volumes = np.asarray(
        [
            max(int(graph.nodes[node_id].get("inv35_volume_voxels", 1)), 1)
            * voxel_volume
            for node_id in node_ids
        ],
        np.float32,
    )
    median_volume = max(float(np.median(volumes)), 1e-6)

    tracklet_id, groups, _ = build_tracklets(graph, node_ids)

    lengths_before = np.zeros(len(node_ids), np.float32)
    lengths_after = np.zeros(len(node_ids), np.float32)
    for rows in groups:
        ts = sorted(float(times[row]) for row in rows)
        for row in rows:
            lengths_before[row] = sum(t <= float(times[row]) for t in ts)
            lengths_after[row] = sum(t >= float(times[row]) for t in ts)

    graph_x = np.zeros((len(node_ids), NODE_DIM), np.float32)
    spacing_np = np.asarray(spacing, np.float32)
    shape_np = np.asarray(shape_zyx, np.float32)

    for row, node_id in enumerate(node_ids):
        data = graph.nodes[node_id]
        coords = get_coords_zyx(data)

        graph_x[row, GX_TIME] = times[row] / max(int(temporal_radius), 1)
        graph_x[row, GX_POS] = positions[row] / max(float(dref_um), 1e-6)
        graph_x[row, GX_LOG_VOLUME] = math.log(
            max(float(volumes[row]), 1e-6) / median_volume
        )

        # Neutral heavyweight fields. No morphology/intensity preprocessing.
        graph_x[row, GX_BBOX] = 0.0
        graph_x[row, GX_PCA] = 0.0
        graph_x[row, GX_ELONGATION] = 0.0
        graph_x[row, GX_FLATNESS] = 0.0
        graph_x[row, GX_SOLIDITY] = 1.0
        graph_x[row, GX_COMPACTNESS] = 1.0
        graph_x[row, GX_INTENSITY_MEAN] = 0.0
        graph_x[row, GX_INTENSITY_STD] = 0.0

        backward = local_velocity(
            graph,
            node_id,
            positions_by_id,
            predecessors=True,
        )
        forward = local_velocity(
            graph,
            node_id,
            positions_by_id,
            predecessors=False,
        )
        graph_x[row, GX_BACK_VEL] = backward / max(float(dref_um), 1e-6)
        graph_x[row, GX_FWD_VEL] = forward / max(float(dref_um), 1e-6)

        graph_x[row, GX_LENGTH_BEFORE] = lengths_before[row]
        graph_x[row, GX_LENGTH_AFTER] = lengths_after[row]

        lower_um = coords * spacing_np
        upper_um = (shape_np - 1.0 - coords) * spacing_np
        distance_boundary = float(np.min(np.concatenate([lower_um, upper_um])))
        boundary = distance_boundary <= 4.0

        graph_x[row, GX_VOLUME_BOUNDARY] = (
            distance_boundary / max(float(dref_um), 1e-6)
        )
        graph_x[row, GX_PATCH_BOUNDARY] = graph_x[row, GX_VOLUME_BOUNDARY]
        graph_x[row, GX_IS_CURRENT] = float(times[row] == 0)

        group_rows = groups[int(tracklet_id[row])]
        group_times = [float(times[r]) for r in group_rows]

        graph_x[row, GX_INTERIOR_START] = float(
            times[row] == min(group_times)
            and min(group_times) > min(available_offsets)
            and not boundary
        )
        graph_x[row, GX_INTERIOR_END] = float(
            times[row] == max(group_times)
            and max(group_times) < max(available_offsets)
            and not boundary
        )
        graph_x[row, GX_DIVISION] = float(
            graph.in_degree(node_id) > 1 or graph.out_degree(node_id) > 1
        )
        graph_x[row, GX_BOUNDARY] = float(boundary)

    # Detection graph: accepted Trackastra edges + same-frame local neighbours.
    edge_pairs: list[tuple[int, int]] = []
    edge_attrs: list[list[float]] = []
    seen: set[tuple[int, int]] = set()

    def add_edge(
        source_row: int,
        destination_row: int,
        relation: str,
        *,
        score: float | None,
        accepted: bool,
    ) -> None:
        pair = (int(source_row), int(destination_row))
        if pair in seen:
            return
        seen.add(pair)

        dt = float(times[destination_row] - times[source_row])
        delta = positions[destination_row] - positions[source_row]
        distance = float(np.linalg.norm(delta))

        source_id = node_ids[source_row]
        velocity = local_velocity(
            graph,
            source_id,
            positions_by_id,
            predecessors=False,
        )
        residual = float(
            np.linalg.norm(delta - velocity * max(dt, 1.0))
        )

        onehot = [0.0, 0.0, 0.0, 0.0]
        onehot[
            {
                "temporal_fwd": 0,
                "temporal_rev": 1,
                "division": 2,
                "spatial": 3,
            }[relation]
        ] = 1.0

        edge_attrs.append(
            [
                dt,
                *(delta / max(float(dref_um), 1e-6)).tolist(),
                distance / max(float(dref_um), 1e-6),
                math.log(
                    max(float(volumes[destination_row]), 1e-6)
                    / max(float(volumes[source_row]), 1e-6)
                ),
                0.0,  # intensity delta
                residual / max(float(dref_um), 1e-6),
                0.0 if score is None else float(score),
                float(score is not None),
                *onehot,
                float(accepted),
            ]
        )
        edge_pairs.append(pair)

    for source, destination, data in graph.edges(data=True):
        source = int(source)
        destination = int(destination)
        if source not in id_to_row or destination not in id_to_row:
            continue

        s = id_to_row[source]
        d = id_to_row[destination]
        score = edge_score(data)
        division = graph.out_degree(source) > 1 or graph.in_degree(destination) > 1

        add_edge(
            s,
            d,
            "division" if division else "temporal_fwd",
            score=score,
            accepted=True,
        )
        if not division:
            add_edge(
                d,
                s,
                "temporal_rev",
                score=score,
                accepted=True,
            )

    for time_offset in sorted(set(int(v) for v in times.tolist())):
        rows = np.flatnonzero(times == float(time_offset))
        if len(rows) < 2:
            continue

        positions_at_time = positions[rows]
        distances = np.linalg.norm(
            positions_at_time[:, None] - positions_at_time[None],
            axis=-1,
        )

        for local_source, source_row in enumerate(rows.tolist()):
            count = 0
            for local_destination in np.argsort(distances[local_source]).tolist():
                if local_destination == local_source:
                    continue
                distance = float(distances[local_source, local_destination])
                if distance > spatial_radius_dref * float(dref_um):
                    break

                destination_row = int(rows[local_destination])
                add_edge(
                    int(source_row),
                    destination_row,
                    "spatial",
                    score=None,
                    accepted=False,
                )
                count += 1
                if count >= k_spatial_neighbors:
                    break

    if edge_pairs:
        edge_index = np.asarray(edge_pairs, np.int64).T
        edge_attr = np.asarray(edge_attrs, np.float32).reshape(-1, EDGE_DIM)
    else:
        edge_index = np.zeros((2, 0), np.int64)
        edge_attr = np.zeros((0, EDGE_DIM), np.float32)

    references = np.stack(
        [tracklet_reference(rows, times, positions) for rows in groups],
        axis=0,
    ).astype(np.float32)

    status = np.zeros((len(groups), STATUS_DIM), np.float32)
    radius = max(int(temporal_radius), 1)
    past_fraction = sum(v < 0 for v in available_offsets) / float(radius)
    future_fraction = sum(v > 0 for v in available_offsets) / float(radius)

    for tracklet, rows in enumerate(groups):
        ts = sorted(int(times[row]) for row in rows)
        ts_set = set(ts)

        boundary = bool(np.any(graph_x[rows, GX_BOUNDARY] > 0.5))
        division = bool(np.any(graph_x[rows, GX_DIVISION] > 0.5))
        gaps = any(
            offset not in ts_set
            for offset in available_offsets
            if min(ts) < offset < max(ts)
        )
        interior_start = min(ts) > min(available_offsets) and not boundary
        interior_end = max(ts) < max(available_offsets) and not boundary
        complete = (
            min(ts) <= min(available_offsets)
            and max(ts) >= max(available_offsets)
            and not gaps
        )

        member_ids = {node_ids[row] for row in rows}
        scores = []
        for source, destination, data in graph.edges(data=True):
            if int(source) in member_ids and int(destination) in member_ids:
                score = edge_score(data)
                if score is not None:
                    scores.append(score)
        uncertain = bool(scores and float(np.mean(scores)) < 0.5)

        status[tracklet] = [
            float(complete),
            float(interior_start),
            float(interior_end),
            float(gaps),
            float(division),
            float(boundary),
            float(past_fraction),
            float(future_fraction),
            0.0,
            float(uncertain),
        ]

    return TemporalInput(
        graph_x=torch.from_numpy(graph_x).to(device=device, dtype=torch.float32),
        graph_edge_index=torch.from_numpy(edge_index).to(
            device=device,
            dtype=torch.long,
        ),
        graph_edge_attr=torch.from_numpy(edge_attr).to(
            device=device,
            dtype=torch.float32,
        ),
        tracklet_id=torch.from_numpy(tracklet_id).to(
            device=device,
            dtype=torch.long,
        ),
        temporal_ref_um=torch.from_numpy(references).to(
            device=device,
            dtype=torch.float32,
        ),
        temporal_status=torch.from_numpy(status).to(
            device=device,
            dtype=torch.float32,
        ),
        temporal_batch=torch.zeros(
            (len(groups),),
            device=device,
            dtype=torch.long,
        ),
        node_history_embedding=None,
        hypothesis_edge_index=None,
        hypothesis_edge_attr=None,
    )


# =============================================================================
# Existing frozen observer samples — optional, no preparation
# =============================================================================


@dataclass
class ObserverLookup:
    d1: Tensor
    d2: Tensor
    hidden: Tensor
    explicit: Tensor
    key_to_row: dict[tuple[int, int, int], int]


def load_observer_lookup(
    paths: Paths,
    t: int,
    device: torch.device,
) -> ObserverLookup | None:
    required = (
        paths.observer_refs(t),
        paths.observer_d1(t),
        paths.observer_d2(t),
        paths.observer_hidden(t),
        paths.observer_explicit(t),
    )
    if any(not path.is_file() for path in required):
        return None

    refs = np.array(
        np.load(paths.observer_refs(t), mmap_mode="r", allow_pickle=False),
        dtype=np.float32,
        order="C",
        copy=True,
    )
    d1 = np.array(
        np.load(paths.observer_d1(t), mmap_mode="r", allow_pickle=False),
        dtype=np.float32,
        order="C",
        copy=True,
    )
    d2 = np.array(
        np.load(paths.observer_d2(t), mmap_mode="r", allow_pickle=False),
        dtype=np.float32,
        order="C",
        copy=True,
    )
    hidden = np.array(
        np.load(paths.observer_hidden(t), mmap_mode="r", allow_pickle=False),
        dtype=np.float32,
        order="C",
        copy=True,
    )
    explicit = np.array(
        np.load(paths.observer_explicit(t), mmap_mode="r", allow_pickle=False),
        dtype=np.float32,
        order="C",
        copy=True,
    )

    return ObserverLookup(
        d1=torch.from_numpy(d1).to(device),
        d2=torch.from_numpy(d2).to(device),
        hidden=torch.from_numpy(hidden).to(device),
        explicit=torch.from_numpy(explicit).to(device),
        key_to_row={reference_key(ref): row for row, ref in enumerate(refs)},
    )


def observe_if_cached(
    model,
    temporal: TemporalState,
    lookup: ObserverLookup | None,
) -> tuple[TemporalState, float]:
    if temporal.is_empty or lookup is None:
        return temporal, 0.0

    refs = temporal.ref_um.detach().float().cpu().numpy()
    temporal_rows: list[int] = []
    cache_rows: list[int] = []

    for row, ref in enumerate(refs):
        cached = lookup.key_to_row.get(reference_key(ref))
        if cached is not None:
            temporal_rows.append(int(row))
            cache_rows.append(int(cached))

    if not temporal_rows:
        return temporal, 0.0

    ti = torch.tensor(
        temporal_rows,
        device=temporal.tokens.device,
        dtype=torch.long,
    )
    ci = torch.tensor(
        cache_rows,
        device=temporal.tokens.device,
        dtype=torch.long,
    )

    observer = model.temporal_observer
    p1 = observer.d1_proj(lookup.d1[ci])
    p2 = observer.d2_proj(lookup.d2[ci])
    pg = (
        observer.geometry_proj(lookup.hidden[ci])
        + observer.geometry_field_proj(lookup.explicit[ci])
    )
    message = observer.message(torch.cat([p1, p2, pg], dim=-1))
    message = message.to(temporal.tokens.dtype)

    selected = temporal.tokens[ti]
    reliability = temporal.reliability[ti]
    gate = observer.gate(
        torch.cat([selected, message, reliability], dim=-1)
    )
    observed = observer.norm(selected + gate * message)

    tokens = temporal.tokens.index_copy(0, ti, observed)
    return (
        replace(temporal, tokens=tokens),
        len(temporal_rows) / max(len(refs), 1),
    )


# =============================================================================
# Frozen spatial runtime / synthetic current partition
# =============================================================================


@dataclass
class RuntimeFrame:
    t: int
    rag: RAGState
    actual_partition: PartitionState
    manual: np.ndarray
    node_manual: Tensor
    observer: None


class RuntimeFrameLoader:
    def __init__(self, paths: Paths, device: torch.device):
        self.paths = paths
        self.device = device
        self.manual_movie = np.load(
            paths.manual_movie,
            mmap_mode="r",
            allow_pickle=False,
        )
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
        rag: RAGState = tree_to_device_fp32(payload["rag"], self.device)
        actual: PartitionState = tree_to_device_fp32(
            payload["actual_partition"],
            self.device,
        )
        if rag.statistics is None:
            raise RuntimeError(
                "Frozen spatial cache does not contain compact supervoxel "
                "statistics required by the current InstanceTokenizer. "
                "Use the full frozen_graph.pt spatial cache, not a diagnostic "
                "rag_state.npz export."
            )

        manual = np.asarray(self.manual_movie[t])
        supervoxels = (
            rag.supervoxel_labels[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        lookup = sv_label_lookup(
            supervoxels,
            manual,
            name=f"runtime t={t} manual",
        )
        node_sv = (
            rag.node_supervoxel_id
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        node_manual = torch.as_tensor(
            lookup[node_sv],
            device=self.device,
            dtype=torch.long,
        )

        self.current = RuntimeFrame(
            t=t,
            rag=rag,
            actual_partition=actual,
            manual=manual,
            node_manual=node_manual,
            observer=None,
        )
        return self.current


@dataclass
class SyntheticCase:
    rag: RAGState
    partition: PartitionState
    target_keep: Tensor
    editable: Tensor
    cut_mask: Tensor
    keep_mask: Tensor
    split_target: Tensor
    node_current_component: Tensor
    events: tuple[MergeEvent, ...]


def build_synthetic_case(
    runtime: RuntimeFrame,
    events: Sequence[MergeEvent],
    *,
    synthetic_logit: float,
) -> SyntheticCase:
    rag = runtime.rag
    node_manual = runtime.node_manual
    actual = runtime.actual_partition.node_component_global

    representative: dict[int, int] = {}
    for event in events:
        representative[int(event.a)] = int(event.representative)
        representative[int(event.b)] = int(event.representative)

    keys: list[tuple[str, int]] = []
    for node in range(int(node_manual.numel())):
        true_id = int(node_manual[node].item())
        if true_id > 0:
            keys.append(("manual", representative.get(true_id, true_id)))
        else:
            # Outside annotated supervision: preserve mature spatial grouping.
            keys.append(("extra", int(actual[node].item())))

    key_to_component: dict[tuple[str, int], int] = {}
    component_manual_ids: list[set[int]] = []
    node_component_values: list[int] = []

    for node, key in enumerate(keys):
        component = key_to_component.get(key)
        if component is None:
            component = len(key_to_component)
            key_to_component[key] = component
            component_manual_ids.append(set())

        node_component_values.append(component)

        true_id = int(node_manual[node].item())
        if true_id > 0:
            component_manual_ids[component].add(true_id)

    node_component = torch.tensor(
        node_component_values,
        device=rag.node_features.device,
        dtype=torch.long,
    )
    component_count = len(key_to_component)

    src, dst = rag.edge_index
    same_current = node_component[src] == node_component[dst]

    positive = rag.spatial_edge_logits.new_full(
        rag.spatial_edge_logits.shape,
        float(synthetic_logit),
    )
    negative = rag.spatial_edge_logits.new_full(
        rag.spatial_edge_logits.shape,
        -float(synthetic_logit),
    )
    synthetic_logits = torch.where(same_current, positive, negative)
    synthetic_rag = replace(rag, spatial_edge_logits=synthetic_logits)

    # InstanceTokenizer uses compact rag.statistics, so the native voxel map is
    # not needed here. A tiny label vector only supplies component count/device.
    if component_count:
        tiny_labels = torch.arange(
            1,
            component_count + 1,
            device=rag.node_features.device,
            dtype=torch.long,
        ).reshape(1, 1, -1)
    else:
        tiny_labels = torch.zeros(
            (1, 1, 1),
            device=rag.node_features.device,
            dtype=torch.long,
        )

    partition = PartitionState(
        labels=[tiny_labels],
        node_component=node_component,
        node_component_global=node_component.clone(),
        component_count_per_batch=torch.tensor(
            [component_count],
            device=rag.node_features.device,
            dtype=torch.long,
        ),
        edge_logits=synthetic_logits,
    )

    true_src = node_manual[src]
    true_dst = node_manual[dst]
    valid = (true_src > 0) & (true_dst > 0)

    target_keep = valid & (true_src == true_dst)
    editable = valid & same_current

    # Only deliberately erased true-cell boundaries become CUT targets.
    cut_mask = editable & ~target_keep
    keep_mask = editable & target_keep

    split_target = synthetic_logits.new_zeros((component_count,))
    for component, true_ids in enumerate(component_manual_ids):
        split_target[component] = float(len(true_ids) > 1)

    return SyntheticCase(
        rag=synthetic_rag,
        partition=partition,
        target_keep=target_keep,
        editable=editable,
        cut_mask=cut_mask,
        keep_mask=keep_mask,
        split_target=split_target,
        node_current_component=node_component,
        events=tuple(events),
    )


def dummy_geometry_and_decode(
    model,
    reference: Tensor,
) -> tuple[SpatialDecodeState, GeometryState]:
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


@dataclass
class ForwardResult:
    reasoning: Any
    final_logits: Tensor
    observer_hit_rate: float


def forward_case(
    model,
    *,
    runtime: RuntimeFrame,
    case: SyntheticCase,
    temporal_input: TemporalInput,
    spacing: Sequence[float],
    dref_um: float,
    device: torch.device,
) -> ForwardResult:
    decoded, geometry = dummy_geometry_and_decode(
        model,
        case.rag.node_features,
    )

    spacing_t = torch.tensor(
        [spacing],
        device=device,
        dtype=torch.float32,
    )
    dref_t = torch.tensor(
        [float(dref_um)],
        device=device,
        dtype=torch.float32,
    )

    instances = model.instance_tokenizer(
        case.partition,
        case.rag,
        decoded,
        geometry,
        spacing_t,
        dref_t,
        profile_prefix="inv35_concrete_tokenizer",
    )

    temporal = model.temporal_encoder(temporal_input)
    hit_rate = 0.0

    reasoning = model.instance_temporal(
        instances,
        case.rag,
        temporal,
        dref_t,
    )

    # HARD SPLIT-ONLY INVARIANT.
    final_logits = torch.where(
        case.editable,
        reasoning.final_edge_logits,
        case.rag.spatial_edge_logits,
    )

    return ForwardResult(
        reasoning=reasoning,
        final_logits=final_logits,
        observer_hit_rate=float(hit_rate),
    )


# =============================================================================
# Loss
# =============================================================================


def sampled(index: Tensor, maximum: int, rng: random.Random) -> Tensor:
    if int(index.numel()) <= int(maximum):
        return index
    rows = rng.sample(range(int(index.numel())), int(maximum))
    return index[
        torch.tensor(rows, device=index.device, dtype=torch.long)
    ]


@dataclass
class LossResult:
    total: Tensor
    cut: Tensor
    keep: Tensor
    split: Tensor
    cut_edges: int
    keep_edges: int


def training_loss(
    case: SyntheticCase,
    forward: ForwardResult,
    *,
    preserve_edges: int,
    preserve_ratio: int,
    preservation_weight: float,
    split_weight: float,
    rng: random.Random,
) -> LossResult:
    logits = forward.final_logits
    zero = logits.sum() * 0.0

    cut_index = torch.nonzero(case.cut_mask, as_tuple=False).flatten()
    keep_index = torch.nonzero(case.keep_mask, as_tuple=False).flatten()

    keep_budget = int(preserve_edges)
    if cut_index.numel():
        keep_budget = min(
            keep_budget,
            max(
                int(cut_index.numel()) * int(preserve_ratio),
                int(cut_index.numel()),
            ),
        )
    keep_index = sampled(keep_index, keep_budget, rng)

    cut = (
        F.binary_cross_entropy_with_logits(
            logits[cut_index],
            torch.zeros_like(logits[cut_index]),
        )
        if cut_index.numel()
        else zero
    )
    keep = (
        F.binary_cross_entropy_with_logits(
            logits[keep_index],
            torch.ones_like(logits[keep_index]),
        )
        if keep_index.numel()
        else zero
    )

    positive = torch.nonzero(
        case.split_target > 0.5,
        as_tuple=False,
    ).flatten()
    negative = torch.nonzero(
        case.split_target <= 0.5,
        as_tuple=False,
    ).flatten()

    if positive.numel():
        negative = sampled(
            negative,
            min(int(negative.numel()), max(16, int(positive.numel()) * 8)),
            rng,
        )
        split_index = torch.cat([positive, negative])
    else:
        split_index = sampled(
            negative,
            min(int(negative.numel()), 64),
            rng,
        )

    split = (
        F.binary_cross_entropy_with_logits(
            forward.reasoning.split_logits[split_index],
            case.split_target[split_index].to(
                forward.reasoning.split_logits.dtype
            ),
        )
        if split_index.numel()
        else zero
    )

    total = (
        cut
        + float(preservation_weight) * keep
        + float(split_weight) * split
    )

    return LossResult(
        total=total,
        cut=cut,
        keep=keep,
        split=split,
        cut_edges=int(cut_index.numel()),
        keep_edges=int(keep_index.numel()),
    )


# =============================================================================
# Case index / graph loading
# =============================================================================


@dataclass(frozen=True)
class VariantFrameCase:
    variant_index: int
    split: str
    frame: int
    events: tuple[MergeEvent, ...]


def build_case_index(
    plans: Sequence[VariantPlan],
    frame_count: int,
) -> dict[str, list[VariantFrameCase]]:
    result = {"train": [], "val": [], "test": []}

    for plan in plans:
        for t in range(frame_count):
            events = tuple(plan.events_by_frame.get(t, ()))
            if events:
                result[plan.split].append(
                    VariantFrameCase(
                        variant_index=int(plan.index),
                        split=plan.split,
                        frame=int(t),
                        events=events,
                    )
                )

    return result


def load_track_graphs(paths: Paths, plans: Sequence[VariantPlan]) -> dict[int, Any]:
    result = {}
    for plan in plans:
        with paths.track_graph(plan.index).open("rb") as handle:
            result[int(plan.index)] = pickle.load(handle)
    return result


# =============================================================================
# Metrics
# =============================================================================


def enforce_split_only_partition(
    case: SyntheticCase,
    predicted: PartitionState,
) -> PartitionState:
    """
    Intersect every predicted final component with the current synthetic
    component. This is a hard constraint, not a soft negative edge.
    """
    current = case.node_current_component.detach().cpu().numpy().astype(np.int64)
    final = (
        predicted.node_component_global
        .detach().cpu().numpy()
        .astype(np.int64)
    )

    pair_to_component: dict[tuple[int, int], int] = {}
    remapped = np.empty_like(final)

    for row, pair in enumerate(zip(current.tolist(), final.tolist())):
        key = (int(pair[0]), int(pair[1]))
        component = pair_to_component.get(key)
        if component is None:
            component = len(pair_to_component)
            pair_to_component[key] = component
        remapped[row] = component

    remapped_t = torch.as_tensor(
        remapped,
        device=predicted.node_component_global.device,
        dtype=torch.long,
    )
    count = len(pair_to_component)

    tiny = (
        torch.arange(
            1,
            count + 1,
            device=remapped_t.device,
            dtype=torch.long,
        ).reshape(1, 1, -1)
        if count
        else torch.zeros(
            (1, 1, 1),
            device=remapped_t.device,
            dtype=torch.long,
        )
    )

    return PartitionState(
        labels=[tiny],
        node_component=remapped_t,
        node_component_global=remapped_t.clone(),
        component_count_per_batch=torch.tensor(
            [count],
            device=remapped_t.device,
            dtype=torch.long,
        ),
        edge_logits=predicted.edge_logits,
    )


def pair_exact_recovery(
    runtime: RuntimeFrame,
    predicted: PartitionState,
    event: MergeEvent,
) -> bool:
    seen: set[int] = set()

    for true_id in (int(event.a), int(event.b)):
        rows = torch.nonzero(
            runtime.node_manual == true_id,
            as_tuple=False,
        ).flatten()
        if rows.numel() == 0:
            return False

        components = torch.unique(predicted.node_component_global[rows])
        if components.numel() != 1:
            return False

        component = int(components.item())
        if component in seen:
            return False
        seen.add(component)

        members = torch.nonzero(
            predicted.node_component_global == component,
            as_tuple=False,
        ).flatten()
        member_true = runtime.node_manual[members]
        positive = member_true[member_true > 0]
        if bool((positive != true_id).any()):
            return False

    return True


def clean_split_counts(
    runtime: RuntimeFrame,
    predicted: PartitionState,
    events: Sequence[MergeEvent],
) -> tuple[int, int]:
    merged_ids = {
        int(value)
        for event in events
        for value in (event.a, event.b)
    }

    total = 0
    split = 0
    for true_id in torch.unique(runtime.node_manual).tolist():
        true_id = int(true_id)
        if true_id <= 0 or true_id in merged_ids:
            continue

        rows = torch.nonzero(
            runtime.node_manual == true_id,
            as_tuple=False,
        ).flatten()
        if rows.numel() == 0:
            continue

        total += 1
        split += int(
            torch.unique(predicted.node_component_global[rows]).numel() > 1
        )

    return total, split


def split_only_violation_count(
    case: SyntheticCase,
    predicted: PartitionState,
) -> int:
    count = 0

    for final_component in torch.unique(
        predicted.node_component_global
    ).tolist():
        rows = torch.nonzero(
            predicted.node_component_global == int(final_component),
            as_tuple=False,
        ).flatten()
        if rows.numel() <= 1:
            continue

        if torch.unique(case.node_current_component[rows]).numel() > 1:
            count += 1

    return count


@torch.no_grad()
def evaluate(
    model,
    *,
    cases: Sequence[VariantFrameCase],
    graphs: dict[int, Any],
    loader: RuntimeFrameLoader,
    frame_count: int,
    temporal_radius: int,
    spacing: Sequence[float],
    dref_um: float,
    synthetic_logit: float,
    device: torch.device,
    maximum_cases: int,
    seed: int,
) -> dict[str, Any]:
    set_temporal_mode(model, False)

    ordered = list(cases)
    rng = random.Random(int(seed))
    rng.shuffle(ordered)
    if maximum_cases > 0:
        ordered = ordered[: min(maximum_cases, len(ordered))]

    if not ordered:
        raise RuntimeError("No evaluation cases")

    shape = loader.manual_movie.shape[-3:]
    threshold = float(model.cfg.partition.final_merge_threshold)

    cut_total = cut_correct = 0
    keep_total = keep_correct = 0
    pair_total = pair_exact = 0
    clean_total = clean_split = 0
    violations = 0
    observer_hits = []

    for spec in ordered:
        runtime = loader.load(spec.frame)
        case = build_synthetic_case(
            runtime,
            spec.events,
            synthetic_logit=synthetic_logit,
        )

        temporal_input = direct_temporal_input(
            graphs[spec.variant_index],
            target_t=spec.frame,
            frame_count=frame_count,
            temporal_radius=temporal_radius,
            spacing=spacing,
            dref_um=dref_um,
            shape_zyx=shape,
            device=device,
        )
        forward = forward_case(
            model,
            runtime=runtime,
            case=case,
            temporal_input=temporal_input,
            spacing=spacing,
            dref_um=dref_um,
            device=device,
        )
        observer_hits.append(forward.observer_hit_rate)

        keep_prediction = torch.sigmoid(forward.final_logits) >= threshold

        cut_total += int(case.cut_mask.sum().item())
        cut_correct += int((~keep_prediction[case.cut_mask]).sum().item())
        keep_total += int(case.keep_mask.sum().item())
        keep_correct += int(keep_prediction[case.keep_mask].sum().item())

        predicted = model.partitioner(
            case.rag,
            forward.final_logits,
            model.cfg.partition.final_merge_threshold,
            stage="final",
        )
        predicted = enforce_split_only_partition(case, predicted)

        for event in spec.events:
            pair_total += 1
            pair_exact += int(
                pair_exact_recovery(runtime, predicted, event)
            )

        total, split = clean_split_counts(
            runtime,
            predicted,
            spec.events,
        )
        clean_total += total
        clean_split += split
        violations += split_only_violation_count(case, predicted)

    cut_accuracy = cut_correct / max(cut_total, 1)
    keep_accuracy = keep_correct / max(keep_total, 1)
    exact_rate = pair_exact / max(pair_total, 1)
    false_split_rate = clean_split / max(clean_total, 1)

    strict = bool(
        cut_accuracy >= 0.90
        and keep_accuracy >= 0.98
        and exact_rate >= 0.80
        and false_split_rate <= 0.02
        and violations == 0
    )

    score = float(
        3.0 * cut_accuracy
        + 2.0 * exact_rate
        + 1.5 * keep_accuracy
        - 4.0 * false_split_rate
    )

    set_temporal_mode(model, True)

    return {
        "objective_version": OBJECTIVE_VERSION,
        "cases": int(len(ordered)),
        "merge_pairs": int(pair_total),
        "cut_edges": int(cut_total),
        "keep_edges": int(keep_total),
        "cut_accuracy": float(cut_accuracy),
        "keep_accuracy": float(keep_accuracy),
        "exact_merge_recovery": float(exact_rate),
        "clean_false_split_rate": float(false_split_rate),
        "split_only_violations": int(violations),
        "mean_observer_cache_hit_rate": float(
            np.mean(observer_hits) if observer_hits else 0.0
        ),
        "strict_pass": bool(strict),
        "checkpoint_score": float(score),
    }


def print_metrics(title: str, step: int, metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 112, flush=True)
    print(f"{title} @ STEP {step}", flush=True)
    print("=" * 112, flush=True)
    print(f"CUT accuracy          : {metrics['cut_accuracy']:.4f}", flush=True)
    print(f"KEEP accuracy         : {metrics['keep_accuracy']:.4f}", flush=True)
    print(
        f"exact merge recovery  : {metrics['exact_merge_recovery']:.4f}",
        flush=True,
    )
    print(
        f"clean false split     : {metrics['clean_false_split_rate']:.4f}",
        flush=True,
    )
    print(
        f"split-only violations : {metrics['split_only_violations']}",
        flush=True,
    )
    print(
        f"observer cache hit    : {metrics['mean_observer_cache_hit_rate']:.3f}",
        flush=True,
    )
    print(f"STRICT PASS           : {metrics['strict_pass']}", flush=True)
    print(
        f"checkpoint score      : {metrics['checkpoint_score']:.5f}",
        flush=True,
    )
    print("=" * 112, flush=True)


# =============================================================================
# Model / optimizer
# =============================================================================


def load_model(checkpoint: Path, device: torch.device):
    payload, model, _, _, _ = INV12.load_checkpoint_model_for_inference(
        checkpoint,
        device,
    )
    return payload, model


def configure_temporal_training(model) -> list[Tensor]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # history_encoder is not used in this clean experiment.
    modules = (
        model.instance_tokenizer,
        model.temporal_encoder,
        model.instance_temporal,
    )

    trainable: list[Tensor] = []
    for module in modules:
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)

    return trainable


def set_temporal_mode(model, training: bool) -> None:
    model.eval()
    for module in (
        model.instance_tokenizer,
        model.temporal_encoder,
        model.temporal_observer,
        model.instance_temporal,
    ):
        module.train(bool(training))


def parameter_audit(model) -> dict[str, int]:
    rows = {}
    total = 0
    for name, module in (
        ("instance_tokenizer", model.instance_tokenizer),
        ("temporal_encoder", model.temporal_encoder),
        ("instance_temporal", model.instance_temporal),
    ):
        count = sum(
            p.numel()
            for p in module.parameters()
            if p.requires_grad
        )
        rows[name] = int(count)
        total += int(count)
    rows["total"] = int(total)
    return rows


def training_config(args: argparse.Namespace) -> TrainingConfig:
    config = TrainingConfig()
    config.lr = float(args.lr)
    config.weight_decay = float(args.weight_decay)
    config.max_grad_norm = float(args.grad_clip)
    config.amp_dtype = str(args.amp_dtype)
    config.curriculum.fixed_stage = "instance_temporal"
    config.curriculum.instance_temporal_detached_spatial = True
    config.curriculum.instance_temporal_freeze_spatial = True
    config.loss.temporal_causal_enabled = False
    config.validate()
    return config


def save_training_state(
    path: Path,
    *,
    model,
    optimizer,
    scaler,
    step: int,
    config: TrainingConfig,
    paths: Paths,
    args: argparse.Namespace,
    dref_um: float,
    metrics: dict[str, Any] | None,
    audit: dict[str, int],
) -> None:
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=int(step),
        epoch=0,
        model_config=model.cfg,
        training_config=config,
        extra={
            "investigation": SCRIPT_NAME,
            "objective_version": OBJECTIVE_VERSION,
            "objective": "concrete_true_instance_cut_keep",
            "sample_id": paths.sample,
            "dref_um": float(dref_um),
            "initializer": str(paths.checkpoint),
            "dataset_manifest": str(paths.dataset_manifest),
            "parameter_audit": audit,
            "validation_metrics": metrics or {},
            "notes": {
                "StaticDetectionRecord": False,
                "TemporalStatic": False,
                "historical_instance_grid": False,
                "graph_builder_temporal_graph": False,
                "hypothesis_graph": False,
                "wrong_neighbourhood": False,
                "contentless_loss": False,
                "concrete_trackastra_variants": True,
                "temporal_action": "split_only",
                "cross_current_component_edges": "immutable_cut",
            },
        },
    )


def resolve_dref(paths: Paths, frame_count: int, override: float | None) -> float:
    if override is not None:
        if override <= 0:
            raise ValueError("--dref-um must be positive")
        return float(override)

    values = []
    for t in range(frame_count):
        meta = paths.spatial_meta(t)
        if not meta.is_file():
            continue
        try:
            value = float(
                json.loads(meta.read_text(encoding="utf-8"))["dref_um"]
            )
            if math.isfinite(value) and value > 0:
                values.append(value)
        except Exception:
            pass

    if not values:
        raise RuntimeError(
            "Could not resolve dref_um from spatial-cache metadata. "
            "Pass --dref-um explicitly."
        )
    return float(np.median(np.asarray(values, np.float64)))


# =============================================================================
# Training loop
# =============================================================================


def train(
    *,
    paths: Paths,
    plans: Sequence[VariantPlan],
    cases: dict[str, list[VariantFrameCase]],
    frame_count: int,
    temporal_radius: int,
    spacing: Sequence[float],
    dref_um: float,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    graphs = load_track_graphs(paths, plans)

    checkpoint_path = (
        resolve(args.resume)
        if args.resume is not None
        else paths.checkpoint
    )
    checkpoint_payload, model = load_model(checkpoint_path, device)

    trainable = configure_temporal_training(model)
    audit = parameter_audit(model)
    config = training_config(args)

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scaler = make_grad_scaler(device, str(args.amp_dtype))

    start_step = 0
    if args.resume is not None:
        start_step = int(checkpoint_payload.get("global_step", 0))
        if "optimizer" in checkpoint_payload:
            optimizer.load_state_dict(checkpoint_payload["optimizer"])
        if "scaler" in checkpoint_payload:
            try:
                scaler.load_state_dict(checkpoint_payload["scaler"])
            except Exception as exc:
                print(f"[resume] scaler not restored: {exc}", flush=True)

    train_cases = list(cases["train"])
    val_cases = list(cases["val"])
    test_cases = list(cases["test"])

    if not train_cases or not val_cases or not test_cases:
        raise RuntimeError("Train/val/test case sets must all be non-empty")

    by_frame: dict[int, list[VariantFrameCase]] = defaultdict(list)
    for case in train_cases:
        by_frame[int(case.frame)].append(case)
    train_frames = sorted(by_frame)

    loader = RuntimeFrameLoader(paths, device)
    shape_zyx = loader.manual_movie.shape[-3:]

    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_strict = False

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 35 — DIRECT TEMPORAL CUT/KEEP TRAINING", flush=True)
    print("=" * 112, flush=True)
    print(f"device                 : {device}", flush=True)
    print(f"initializer             : {checkpoint_path}", flush=True)
    print(f"optimizer steps         : {args.steps}", flush=True)
    print(f"microcases / step       : {args.accumulate_cases}", flush=True)
    print(f"train variant-frames    : {len(train_cases)}", flush=True)
    print(f"val variant-frames      : {len(val_cases)}", flush=True)
    print(f"test variant-frames     : {len(test_cases)}", flush=True)
    print(f"trainable parameters    : {audit['total']:,}", flush=True)
    print(f"preservation weight     : {args.preservation_weight:g}", flush=True)
    print("temporal action         : SPLIT ONLY", flush=True)
    print("StaticDetectionRecord   : NONE", flush=True)
    print("TemporalStatic          : NONE", flush=True)
    print("history grids           : NONE", flush=True)
    print("hypothesis graph        : NONE", flush=True)
    print("=" * 112, flush=True)

    run_started = time.perf_counter()

    for step in range(start_step + 1, int(args.steps) + 1):
        step_started = time.perf_counter()
        rng = random.Random(int(args.seed) + 15_485_863 * int(step))

        frame = rng.choice(train_frames)
        runtime = loader.load(frame)
        available = by_frame[frame]

        optimizer.zero_grad(set_to_none=True)

        sums = {
            "total": 0.0,
            "cut": 0.0,
            "keep": 0.0,
            "split": 0.0,
            "observer": 0.0,
        }
        micro_rows = []

        for micro in range(int(args.accumulate_cases)):
            spec = rng.choice(available)

            case = build_synthetic_case(
                runtime,
                spec.events,
                synthetic_logit=float(args.synthetic_spatial_logit),
            )
            if not bool(case.cut_mask.any()):
                raise RuntimeError(
                    "Synthetic merge frame contains no editable CUT RAG edge: "
                    f"variant={spec.variant_index}, frame={spec.frame}"
                )

            temporal_input = direct_temporal_input(
                graphs[spec.variant_index],
                target_t=spec.frame,
                frame_count=frame_count,
                temporal_radius=temporal_radius,
                spacing=spacing,
                dref_um=dref_um,
                shape_zyx=shape_zyx,
                device=device,
            )

            with autocast_for(device, str(args.amp_dtype)):
                forward = forward_case(
                    model,
                    runtime=runtime,
                    case=case,
                    temporal_input=temporal_input,
                    spacing=spacing,
                    dref_um=dref_um,
                    device=device,
                )
                loss = training_loss(
                    case,
                    forward,
                    preserve_edges=int(args.preserve_edges),
                    preserve_ratio=int(args.preserve_ratio),
                    preservation_weight=float(args.preservation_weight),
                    split_weight=float(args.split_weight),
                    rng=rng,
                )
                scaled = loss.total / float(args.accumulate_cases)

            if not bool(torch.isfinite(scaled.detach())):
                raise FloatingPointError(
                    f"Non-finite loss at step={step}, micro={micro}"
                )

            scaler.scale(scaled).backward()

            sums["total"] += float(loss.total.detach().float().cpu())
            sums["cut"] += float(loss.cut.detach().float().cpu())
            sums["keep"] += float(loss.keep.detach().float().cpu())
            sums["split"] += float(loss.split.detach().float().cpu())
            sums["observer"] += float(forward.observer_hit_rate)

            micro_rows.append(
                {
                    "variant": int(spec.variant_index),
                    "frame": int(spec.frame),
                    "merges": int(len(spec.events)),
                    "cut_edges": int(loss.cut_edges),
                    "keep_edges": int(loss.keep_edges),
                    "observer_hit_rate": float(forward.observer_hit_rate),
                }
            )

        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(
            trainable,
            float(args.grad_clip),
        )
        grad_norm = float(torch.as_tensor(grad).detach().cpu())
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient at step={step}")

        scaler.step(optimizer)
        scaler.update()

        denominator = max(int(args.accumulate_cases), 1)
        row = {
            "step": int(step),
            "frame": int(frame),
            "loss": sums["total"] / denominator,
            "cut_loss": sums["cut"] / denominator,
            "keep_loss": sums["keep"] / denominator,
            "split_loss": sums["split"] / denominator,
            "observer_hit_rate": sums["observer"] / denominator,
            "grad_norm": float(grad_norm),
            "step_seconds": float(time.perf_counter() - step_started),
            "microcases": micro_rows,
        }
        history.append(row)

        if step == 1 or step % int(args.print_every) == 0:
            print(
                f"[step {step:05d}/{args.steps}] "
                f"t={frame:02d} "
                f"loss={row['loss']:.5f} "
                f"cut={row['cut_loss']:.5f} "
                f"keep={row['keep_loss']:.5f} "
                f"split={row['split_loss']:.5f} "
                f"obs={row['observer_hit_rate']:.2f} "
                f"grad={row['grad_norm']:.3f} "
                f"elapsed={duration(time.perf_counter() - run_started)}",
                flush=True,
            )

        if step % 50 == 0:
            atomic_json(paths.training_history, history)

        if step % int(args.eval_every) == 0 or step == int(args.steps):
            metrics = evaluate(
                model,
                cases=val_cases,
                graphs=graphs,
                loader=loader,
                frame_count=frame_count,
                temporal_radius=temporal_radius,
                spacing=spacing,
                dref_um=dref_um,
                synthetic_logit=float(args.synthetic_spatial_logit),
                device=device,
                maximum_cases=int(args.val_cases),
                seed=int(args.seed) + step,
            )
            print_metrics("INVESTIGATION 35 VALIDATION", step, metrics)

            validation_history.append(
                {
                    "step": int(step),
                    "metrics": metrics,
                }
            )
            atomic_json(paths.validation_history, validation_history)
            atomic_json(paths.training_history, history)

            save_training_state(
                paths.latest,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                config=config,
                paths=paths,
                args=args,
                dref_um=dref_um,
                metrics=metrics,
                audit=audit,
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
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    config=config,
                    paths=paths,
                    args=args,
                    dref_um=dref_um,
                    metrics=metrics,
                    audit=audit,
                )
                atomic_json(paths.best_metrics, metrics)
                print(
                    f"[best] step={step} score={score:.5f} strict={strict}",
                    flush=True,
                )

    atomic_json(paths.training_history, history)
    atomic_json(paths.validation_history, validation_history)

    final_metrics = (
        validation_history[-1]["metrics"]
        if validation_history
        else {}
    )
    save_training_state(
        paths.final,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=max(start_step, int(args.steps)),
        config=config,
        paths=paths,
        args=args,
        dref_um=dref_um,
        metrics=final_metrics,
        audit=audit,
    )

    test_metrics = evaluate(
        model,
        cases=test_cases,
        graphs=graphs,
        loader=loader,
        frame_count=frame_count,
        temporal_radius=temporal_radius,
        spacing=spacing,
        dref_um=dref_um,
        synthetic_logit=float(args.synthetic_spatial_logit),
        device=device,
        maximum_cases=0,
        seed=int(args.seed) + 35_999_999,
    )
    print_metrics(
        "INVESTIGATION 35 HELD-OUT VARIANT TEST",
        max(start_step, int(args.steps)),
        test_metrics,
    )
    atomic_json(paths.test_metrics, test_metrics)

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 35 TRAINING COMPLETE", flush=True)
    print("=" * 112, flush=True)
    print(f"best   : {paths.best}", flush=True)
    print(f"latest : {paths.latest}", flush=True)
    print(f"final  : {paths.final}", flush=True)
    print(f"test   : {paths.test_metrics}", flush=True)
    print("=" * 112, flush=True)


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Concrete true-instance merge synthesis + direct temporal CUT/KEEP "
            "training. No temporal metadata preprocessing."
        )
    )

    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument(
        "--spacing",
        default="1.625,0.40625,0.40625",
    )

    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--sample-zarr", type=Path, default=None)
    parser.add_argument(
        "--spatial-cache-root",
        type=Path,
        default=None,
        help=(
            "Directory containing tXXX/frozen_graph.pt. Default uses the "
            "existing valid frozen spatial cache only."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--dref-um", type=float, default=None)

    parser.add_argument(
        "--temporal-radius",
        type=int,
        default=DEFAULT_TEMPORAL_RADIUS,
    )

    parser.add_argument(
        "--tile-shape-zyx",
        default="32,128,128",
    )
    parser.add_argument(
        "--tile-overlap-zyx",
        default="8,32,32",
    )
    parser.add_argument(
        "--tile-halo-zyx",
        default="4,16,16",
    )
    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--train-variants",
        type=int,
        default=DEFAULT_TRAIN_VARIANTS,
    )
    parser.add_argument(
        "--val-variants",
        type=int,
        default=DEFAULT_VAL_VARIANTS,
    )
    parser.add_argument(
        "--test-variants",
        type=int,
        default=DEFAULT_TEST_VARIANTS,
    )
    parser.add_argument(
        "--merge-fraction",
        type=float,
        default=DEFAULT_MERGE_FRACTION,
    )
    parser.add_argument(
        "--max-merges-per-frame",
        type=int,
        default=DEFAULT_MAX_MERGES_PER_FRAME,
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=DEFAULT_MIN_VOXELS,
    )
    parser.add_argument(
        "--max-volume-ratio",
        type=float,
        default=DEFAULT_MAX_VOLUME_RATIO,
    )

    parser.add_argument(
        "--trackastra-model",
        default=DEFAULT_TRACKASTRA_MODEL,
    )
    parser.add_argument(
        "--trackastra-mode",
        default=DEFAULT_TRACKASTRA_MODE,
    )
    parser.add_argument(
        "--trackastra-device",
        default=DEFAULT_TRACKASTRA_DEVICE,
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
    )

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
    )
    parser.add_argument(
        "--accumulate-cases",
        type=int,
        default=DEFAULT_ACCUMULATE_CASES,
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=DEFAULT_EVAL_EVERY,
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=DEFAULT_PRINT_EVERY,
    )
    parser.add_argument(
        "--val-cases",
        type=int,
        default=DEFAULT_VAL_CASES,
        help="0 = every held-out validation variant-frame case",
    )

    parser.add_argument(
        "--synthetic-spatial-logit",
        type=float,
        default=DEFAULT_SYNTHETIC_LOGIT,
    )
    parser.add_argument(
        "--preserve-edges",
        type=int,
        default=DEFAULT_PRESERVE_EDGES,
    )
    parser.add_argument(
        "--preserve-ratio",
        type=int,
        default=DEFAULT_PRESERVE_RATIO,
    )
    parser.add_argument(
        "--preservation-weight",
        type=float,
        default=DEFAULT_PRESERVATION_WEIGHT,
    )
    parser.add_argument(
        "--split-weight",
        type=float,
        default=DEFAULT_SPLIT_WEIGHT,
    )

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument(
        "--rebuild-movies",
        action="store_true",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Regenerate plans and rerun concrete Trackastra variants.",
    )

    parser.add_argument(
        "--rebuild-spatial-cache",
        action="store_true",
        help=(
            "Re-run the current spatial model and replace the compact "
            "per-frame RAG/statistics cache."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build concrete Trackastra variants and stop immediately.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.frame_count < 3:
        raise ValueError("--frame-count must be >= 3")
    if args.temporal_radius < 1:
        raise ValueError("--temporal-radius must be positive")
    if min(args.train_variants, args.val_variants, args.test_variants) < 1:
        raise ValueError("train/val/test variant counts must all be >= 1")
    if not (0.0 < args.merge_fraction <= 1.0):
        raise ValueError("--merge-fraction must lie in (0,1]")
    if args.max_merges_per_frame < 1:
        raise ValueError("--max-merges-per-frame must be >= 1")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.accumulate_cases < 1:
        raise ValueError("--accumulate-cases must be positive")
    if args.eval_every < 1 or args.print_every < 1:
        raise ValueError("--eval-every/--print-every must be positive")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    spacing = parse_spacing(args.spacing)

    paths = make_paths(args)
    validate_inputs(paths, int(args.frame_count))
    paths.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    tile_shape = parse_zyx_ints(
        args.tile_shape_zyx,
        name="--tile-shape-zyx",
    )
    tile_overlap = parse_zyx_ints(
        args.tile_overlap_zyx,
        name="--tile-overlap-zyx",
    )
    tile_halo = parse_zyx_ints(
        args.tile_halo_zyx,
        name="--tile-halo-zyx",
    )

    print("\n" + "=" * 112, flush=True)
    print(
        "INVESTIGATION 35 — CLEAN CONCRETE MERGE SYNTHESIS + TEMPORAL CUT/KEEP",
        flush=True,
    )
    print("=" * 112, flush=True)
    print(f"repository    : {ROOT}", flush=True)
    print(f"sample        : {paths.sample}", flush=True)
    print(f"checkpoint    : {paths.checkpoint}", flush=True)
    print(f"output        : {paths.output}", flush=True)
    print(f"spatial cache : {paths.spatial_cache}", flush=True)
    print("=" * 112, flush=True)

    # 1. Raw + true manual movie.
    build_movies(
        paths,
        int(args.frame_count),
        rebuild=bool(args.rebuild_movies),
    )

    # 2. True touching-cell candidates. No RAG needed.
    catalog = build_touching_catalog(
        paths,
        int(args.frame_count),
        min_voxels=int(args.min_voxels),
        max_volume_ratio=float(args.max_volume_ratio),
        rebuild=bool(args.rebuild_dataset),
    )

    # 3. Concrete merge plans.
    plans = build_variant_plans(
        paths,
        catalog=catalog,
        frame_count=int(args.frame_count),
        train_variants=int(args.train_variants),
        val_variants=int(args.val_variants),
        test_variants=int(args.test_variants),
        merge_fraction=float(args.merge_fraction),
        max_merges_per_frame=int(args.max_merges_per_frame),
        seed=int(args.seed),
        rebuild=bool(args.rebuild_dataset),
    )

    # 4. Trackastra on the concrete corrupted movies.
    prepare_variants(
        paths,
        plans,
        int(args.frame_count),
        model_name=str(args.trackastra_model),
        mode=str(args.trackastra_mode),
        device=str(args.trackastra_device),
        rebuild=bool(args.rebuild_dataset),
    )

    # 5. Recreate mature spatial RAG/statistics from the CURRENT spatial model.
    #    This is the only cache required by InstanceTokenizer.
    rebuild_spatial_cache(
        paths,
        int(args.frame_count),
        spacing=spacing,
        device=device,
        tile_shape_zyx=tile_shape,
        tile_overlap_zyx=tile_overlap,
        tile_halo_zyx=tile_halo,
        tile_batch_size=int(args.tile_batch_size),
        rebuild=bool(args.rebuild_spatial_cache),
    )

    dref_um = resolve_dref(
        paths,
        int(args.frame_count),
        args.dref_um,
    )

    case_index = build_case_index(
        plans,
        int(args.frame_count),
    )

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 35 — DATASET READY", flush=True)
    print("=" * 112, flush=True)
    for split in ("train", "val", "test"):
        split_cases = case_index[split]
        merge_events = sum(
            len(case.events)
            for case in split_cases
        )
        print(
            f"{split:5s} variant-frames={len(split_cases):3d} "
            f"merge_events={merge_events}",
            flush=True,
        )
    print(f"movie dref            : {dref_um:.5f} um", flush=True)
    print("StaticDetectionRecord : NONE", flush=True)
    print("TemporalStatic        : NONE", flush=True)
    print("temporal metadata pass: NONE", flush=True)
    print("history grids         : NONE", flush=True)
    print("hypothesis graph      : NONE", flush=True)
    print("temporal observer     : BYPASSED", flush=True)
    print("=" * 112, flush=True)

    if args.prepare_only:
        print(
            "Stopped after --prepare-only. Concrete Trackastra variants and "
            "the freshly reconstructed spatial cache are ready.",
            flush=True,
        )
        return 0

    train(
        paths=paths,
        plans=plans,
        cases=case_index,
        frame_count=int(args.frame_count),
        temporal_radius=int(args.temporal_radius),
        spacing=spacing,
        dref_um=dref_um,
        device=device,
        args=args,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
