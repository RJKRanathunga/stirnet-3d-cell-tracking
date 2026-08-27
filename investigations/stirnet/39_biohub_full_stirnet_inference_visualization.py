from __future__ import annotations

r"""
Investigation 39 — first full spatial + temporal STIR-Net inference on BioHub.

This investigation is intentionally aligned with the CURRENT repository and the
successful Investigation-35 temporal training contract.

Pipeline
--------
For the complete BioHub movie:

    raw BioHub
      -> canonical preprocessing
      -> binary foreground mask
      -> source segmentation
           geometric completion disabled
      -> five-channel STIR-Net input
      -> current tiled spatial STIR-Net
      -> learned watershed / RAG / signed multicut
      -> source-instance anchored split-only spatial postfilter
      -> ACCEPTED SPATIAL INSTANCES

    accepted spatial instance movie
      -> Trackastra
      -> concrete real Trackastra graph

    for each frame:
      accepted spatial instance components
      + frozen mature RAG/statistics
      + Trackastra graph converted DIRECTLY to TemporalInput
      -> trained TemporalGraphEncoder
      -> trained InstanceTemporalReasoner
      -> CUT/KEEP decisions inside accepted spatial components only
      -> HARD split-only final partition
      -> TEMPORAL FINAL INSTANCES

    temporal final instance movie
      -> Trackastra again
      -> final tracks / lineage visualization

Important training-contract match
---------------------------------
Investigation 35 trained the temporal system with:

    TemporalSpatialObserver : bypassed
    history grids           : absent
    hypothesis graph        : absent
    temporal action         : split-only

Investigation 39 preserves exactly that contract.

It does NOT call production tiled_temporal_inference(), because that function
would re-enable TemporalSpatialObserver and refinement paths that were not used
by this trained checkpoint.

The learned temporal checkpoint is still a complete STIR-Net checkpoint. Spatial
weights are frozen from the spatial initializer; temporal/tokenizer weights come
from Investigation 35.

Split-only invariant
--------------------
Let C_spatial(u) be the accepted spatial component of RAG node u.

Temporal reasoning may change only edges satisfying:

    C_spatial(src) == C_spatial(dst)

Every edge connecting already separate accepted spatial components is forced to
remain CUT, and the final graph partition is explicitly intersected with the
accepted spatial partition as a second hard guarantee.

So:

    A+B -> A | B     allowed
    A | B -> A+B     impossible

Default temporal checkpoint
---------------------------
    runs/stirnet/evaluation/
      35_biohub_temporal_merge_synthesis/
      <sample>/concrete_cutkeep_v3/best.pt

Visualization
-------------
Napari opens:

    Raw Volume
    Preprocessed Volume
    Binary Mask
    Source Instances
    Spatial Final Instances
    Evidence Trackastra Masks
    Temporal Final Instances
    Final Trackastra Masks
    Manual Ground Truth                     (when annotations exist)
    Evidence Tracks                         (hidden)
    Final Tracks                            (visible)
    Needed Split - Fixed                    (points)
    Needed Split - Missed                   (points)
    Unexpected Split                        (points)
    Spatial Overseg - Outside Objective     (points)

The diagnostic point layers are optional and are generated only when the manual
BioHub annotations are present.

Typical run
-----------
    python .\investigations\stirnet\39_biohub_full_stirnet_inference_visualization.py

Run/cache only:
    python .\investigations\stirnet\39_biohub_full_stirnet_inference_visualization.py --no-viewer

Re-open completed results:
    python .\investigations\stirnet\39_biohub_full_stirnet_inference_visualization.py --viewer-only

Force complete recomputation:
    python .\investigations\stirnet\39_biohub_full_stirnet_inference_visualization.py `
        --overwrite-spatial `
        --rebuild-evidence-trackastra `
        --rebuild-temporal `
        --rebuild-final-trackastra
"""

import argparse
import dataclasses
import gc
import importlib.util
import json
import math
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor


# =============================================================================
# Constants
# =============================================================================


SCRIPT_NAME = "39_biohub_full_stirnet_inference_visualization"
DEFAULT_SAMPLE_ID = "44b6_0113de3b"
DEFAULT_FRAME_COUNT = 20
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)

DEFAULT_TILE_SHAPE_ZYX = (32, 128, 128)
DEFAULT_TILE_OVERLAP_ZYX = (8, 32, 32)
DEFAULT_TILE_HALO_ZYX = (4, 16, 16)
DEFAULT_TILE_BATCH_SIZE = 1

DEFAULT_TRACKASTRA_MODEL = "ctc"
DEFAULT_TRACKASTRA_MODE = "greedy"
DEFAULT_TRACKASTRA_DEVICE = "cuda"

DEFAULT_TEMPORAL_RADIUS = 2
DEFAULT_ACCEPTED_INTERNAL_LOGIT = 3.5

DIAG_FIXED = 1
DIAG_MISSED = 2
DIAG_PRESERVED = 3
DIAG_UNEXPECTED_SPLIT = 4
DIAG_SPATIAL_OVERSEG = 5
DIAG_UNMAPPED = 6

DIAG_NAMES = {
    DIAG_FIXED: "Needed Split - Fixed",
    DIAG_MISSED: "Needed Split - Missed",
    DIAG_PRESERVED: "Correct Spatial - Preserved",
    DIAG_UNEXPECTED_SPLIT: "Unexpected Split",
    DIAG_SPATIAL_OVERSEG: "Spatial Overseg - Outside Objective",
    DIAG_UNMAPPED: "Unmapped / Mixed",
}


# =============================================================================
# Repository helpers
# =============================================================================


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

from learned.stirnet.model.types import PartitionState, RAGState


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


# Reuse only CURRENT, validated helper contracts.
INV35 = load_module(
    ROOT / "investigations" / "stirnet"
    / "35_biohub_temporal_merge_synthesis.py",
    "_inv39_current_inv35",
)
INV36 = load_module(
    ROOT / "investigations" / "stirnet"
    / "36_biohub_spatial_trackastra_visualization.py",
    "_inv39_current_inv36",
)


def parse_triplet(text: str, *, cast, name: str):
    values = tuple(cast(token.strip()) for token in str(text).split(","))
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return values


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


def atomic_torch_save(path: Path, payload: Any) -> None:
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
        return {key: map_tree(item, fn) for key, item in value.items()}
    if isinstance(value, list):
        return [map_tree(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(map_tree(item, fn) for item in value)
    return value


def cpu_detached_tree(value: Any) -> Any:
    return map_tree(value, lambda tensor: tensor.detach().cpu())


def to_device_fp32(value: Any, device: torch.device) -> Any:
    def move(tensor: Tensor) -> Tensor:
        if tensor.is_floating_point():
            return tensor.to(device=device, dtype=torch.float32)
        return tensor.to(device=device)
    return map_tree(value, move)


# =============================================================================
# Paths
# =============================================================================


@dataclass(frozen=True)
class OutputPaths:
    root: Path

    @property
    def movies(self) -> Path:
        return self.root / "movies"

    @property
    def spatial_state(self) -> Path:
        return self.root / "spatial_state"

    @property
    def evidence_trackastra(self) -> Path:
        return self.root / "trackastra_evidence"

    @property
    def final_trackastra(self) -> Path:
        return self.root / "trackastra_final"

    @property
    def diagnostics_dir(self) -> Path:
        return self.root / "diagnostics"

    @property
    def raw(self) -> Path:
        return self.movies / "raw.npy"

    @property
    def preprocessed(self) -> Path:
        return self.movies / "preprocessed.npy"

    @property
    def binary_mask(self) -> Path:
        return self.movies / "binary_mask.npy"

    @property
    def source_instances(self) -> Path:
        return self.movies / "source_instances.npy"

    @property
    def spatial_final(self) -> Path:
        return self.movies / "spatial_final_instances.npy"

    @property
    def temporal_final(self) -> Path:
        return self.movies / "temporal_final_instances.npy"

    @property
    def manual_gt(self) -> Path:
        return self.movies / "manual_ground_truth.npy"

    def spatial_frame(self, t: int) -> Path:
        return self.spatial_state / f"t{t:03d}.pt"

    @property
    def spatial_summary(self) -> Path:
        return self.root / "spatial_summary.json"

    @property
    def spatial_success(self) -> Path:
        return self.root / "_SPATIAL_SUCCESS.json"

    @property
    def temporal_summary(self) -> Path:
        return self.root / "temporal_summary.json"

    @property
    def temporal_success(self) -> Path:
        return self.root / "_TEMPORAL_SUCCESS.json"

    @property
    def diagnostic_components(self) -> Path:
        return self.diagnostics_dir / "components.csv"

    @property
    def diagnostic_summary(self) -> Path:
        return self.diagnostics_dir / "summary.json"

    def track_graph(self, kind: str) -> Path:
        return self._track_dir(kind) / "track_graph.pkl"

    def tracked_masks(self, kind: str) -> Path:
        return self._track_dir(kind) / "tracked_masks.npy"

    def napari_tracks(self, kind: str) -> Path:
        return self._track_dir(kind) / "napari_tracks.npy"

    def napari_graph(self, kind: str) -> Path:
        return self._track_dir(kind) / "napari_graph.json"

    def track_summary(self, kind: str) -> Path:
        return self._track_dir(kind) / "summary.json"

    def _track_dir(self, kind: str) -> Path:
        if kind == "evidence":
            return self.evidence_trackastra
        if kind == "final":
            return self.final_trackastra
        raise ValueError(kind)


def default_output(sample_id: str) -> Path:
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / SCRIPT_NAME
        / sample_id
    ).resolve()


def default_temporal_checkpoint(sample_id: str) -> Path:
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / "35_biohub_temporal_merge_synthesis"
        / sample_id
        / "concrete_cutkeep_v3"
        / "best.pt"
    ).resolve()


def resolve_temporal_checkpoint(
    sample_id: str,
    override: str | None,
) -> Path:
    path = (
        resolve(override)
        if override is not None
        else default_temporal_checkpoint(sample_id)
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Temporal checkpoint not found: {path}\n"
            "Pass --checkpoint explicitly if the Investigation-35 best.pt "
            "is stored elsewhere."
        )
    return path


def resolve_annotations(sample_id: str, override: str | None) -> Path:
    return (
        resolve(override)
        if override is not None
        else (
            ROOT
            / "evaluation"
            / "segmentation"
            / "annotations"
            / sample_id
        ).resolve()
    )


# =============================================================================
# Cache checks
# =============================================================================


def movie_shape_ok(path: Path, frame_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return int(arr.shape[0]) == int(frame_count)
    except Exception:
        return False


def spatial_cache_complete(paths: OutputPaths, frame_count: int) -> bool:
    required = (
        paths.raw,
        paths.preprocessed,
        paths.binary_mask,
        paths.source_instances,
        paths.spatial_final,
        paths.spatial_success,
    )
    if not all(path.is_file() for path in required):
        return False
    if not all(movie_shape_ok(path, frame_count) for path in required[:-1]):
        return False
    return all(paths.spatial_frame(t).is_file() for t in range(frame_count))


def trackastra_cache_complete(paths: OutputPaths, kind: str) -> bool:
    return all(
        path.is_file()
        for path in (
            paths.track_graph(kind),
            paths.tracked_masks(kind),
            paths.napari_tracks(kind),
            paths.napari_graph(kind),
            paths.track_summary(kind),
        )
    )


def temporal_cache_complete(paths: OutputPaths, frame_count: int) -> bool:
    return (
        movie_shape_ok(paths.temporal_final, frame_count)
        and paths.temporal_summary.is_file()
        and paths.temporal_success.is_file()
    )


# =============================================================================
# Spatial accepted partition
# =============================================================================


def build_accepted_partition(
    rag: RAGState,
    original_partition: PartitionState,
    final_labels: np.ndarray,
    *,
    name: str,
) -> tuple[PartitionState, Tensor]:
    """
    Convert production source-core postfiltered voxel labels back to a compact
    RAG-node partition.

    Source-core postprocessing is split-only and supervoxel-aware, so each
    atomic supervoxel must belong to at most one accepted final label.
    """
    supervoxels = (
        rag.supervoxel_labels[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    lookup = INV35.sv_label_lookup(
        supervoxels,
        np.asarray(final_labels),
        name=name,
    )
    node_sv = (
        rag.node_supervoxel_id
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    accepted_label = lookup[node_sv]

    original_component = (
        original_partition.node_component_global
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )

    key_to_component: dict[tuple[str, int], int] = {}
    node_component_values: list[int] = []

    for row, label_id in enumerate(accepted_label.tolist()):
        if int(label_id) > 0:
            key = ("accepted", int(label_id))
        else:
            # A spatial RAG node omitted by the postfilter remains isolated only
            # according to its original mature spatial component.
            key = ("extra", int(original_component[row]))

        component = key_to_component.get(key)
        if component is None:
            component = len(key_to_component)
            key_to_component[key] = component
        node_component_values.append(component)

    device = rag.node_features.device
    node_component = torch.tensor(
        node_component_values,
        device=device,
        dtype=torch.long,
    )
    component_count = len(key_to_component)

    tiny_labels = (
        torch.arange(
            1,
            component_count + 1,
            device=device,
            dtype=torch.long,
        ).reshape(1, 1, -1)
        if component_count
        else torch.zeros((1, 1, 1), device=device, dtype=torch.long)
    )

    partition = PartitionState(
        labels=[tiny_labels],
        node_component=node_component,
        node_component_global=node_component.clone(),
        component_count_per_batch=torch.tensor(
            [component_count],
            device=device,
            dtype=torch.long,
        ),
        edge_logits=rag.spatial_edge_logits,
    )
    return partition, node_component


def rag_with_accepted_logits(
    rag: RAGState,
    accepted_component: Tensor,
    *,
    magnitude: float,
) -> tuple[RAGState, Tensor]:
    src, dst = rag.edge_index
    same_current = (
        accepted_component[src] == accepted_component[dst]
    )
    positive = rag.spatial_edge_logits.new_full(
        rag.spatial_edge_logits.shape,
        float(magnitude),
    )
    negative = rag.spatial_edge_logits.new_full(
        rag.spatial_edge_logits.shape,
        -float(magnitude),
    )
    logits = torch.where(same_current, positive, negative)
    return replace(rag, spatial_edge_logits=logits), same_current


def hard_split_only_components(
    accepted_component: Tensor,
    predicted_component: Tensor,
) -> Tensor:
    """
    Intersect the predicted final partition with the accepted spatial partition.
    """
    accepted = (
        accepted_component.detach().cpu().numpy().astype(np.int64)
    )
    predicted = (
        predicted_component.detach().cpu().numpy().astype(np.int64)
    )

    pair_to_component: dict[tuple[int, int], int] = {}
    output = np.empty_like(predicted)

    for row, (a, p) in enumerate(zip(accepted.tolist(), predicted.tolist())):
        key = (int(a), int(p))
        component = pair_to_component.get(key)
        if component is None:
            component = len(pair_to_component)
            pair_to_component[key] = component
        output[row] = component

    return torch.as_tensor(
        output,
        device=accepted_component.device,
        dtype=torch.long,
    )


def materialize_node_partition(
    rag: RAGState,
    node_component: Tensor,
) -> np.ndarray:
    supervoxels = (
        rag.supervoxel_labels[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    node_sv = (
        rag.node_supervoxel_id
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    components = (
        node_component.detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )

    max_sv = int(supervoxels.max(initial=0))
    lut = np.zeros(max_sv + 1, dtype=np.int32)

    if len(node_sv) != len(components):
        raise RuntimeError(
            f"RAG node/component size mismatch: {len(node_sv)} vs {len(components)}"
        )

    lut[node_sv] = components.astype(np.int32, copy=False) + 1
    return lut[supervoxels]


# =============================================================================
# Spatial inference
# =============================================================================


def _create_movie(
    path: Path,
    *,
    dtype,
    shape: tuple[int, ...],
) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )


def run_spatial_movie(
    *,
    sample_id: str,
    sample_zarr: Path,
    checkpoint: Path,
    output: OutputPaths,
    frame_count: int,
    spacing: tuple[float, float, float],
    tile_shape: tuple[int, int, int],
    tile_overlap: tuple[int, int, int],
    tile_halo: tuple[int, int, int],
    tile_batch_size: int,
    device: torch.device,
) -> None:
    from src.io import open_sample

    runner = INV36.load_kaggle_runner()

    print("[spatial] loading trained full checkpoint ...", flush=True)
    runtime, helper = runner._load_model_runtime(
        ROOT,
        checkpoint,
        device=device,
        tile_shape=tile_shape,
        tile_overlap=tile_overlap,
        tile_halo=tile_halo,
        tile_batch_size=tile_batch_size,
    )

    image = open_sample(sample_zarr)
    if len(image.shape) != 4:
        raise ValueError(f"Expected [T,Z,Y,X], got shape={image.shape}")
    if frame_count > int(image.shape[0]):
        raise ValueError(
            f"Requested {frame_count} frames but movie has only {image.shape[0]}"
        )

    spatial_shape = tuple(int(v) for v in image.shape[-3:])
    movie_shape = (frame_count, *spatial_shape)

    output.root.mkdir(parents=True, exist_ok=True)
    output.movies.mkdir(parents=True, exist_ok=True)
    output.spatial_state.mkdir(parents=True, exist_ok=True)

    raw_movie = _create_movie(
        output.raw,
        dtype=np.dtype(image.dtype),
        shape=movie_shape,
    )
    preprocessed_movie = _create_movie(
        output.preprocessed,
        dtype=np.float16,
        shape=movie_shape,
    )
    binary_movie = _create_movie(
        output.binary_mask,
        dtype=np.uint8,
        shape=movie_shape,
    )
    source_movie = _create_movie(
        output.source_instances,
        dtype=np.int32,
        shape=movie_shape,
    )
    final_movie = _create_movie(
        output.spatial_final,
        dtype=np.int32,
        shape=movie_shape,
    )

    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()

    print("\n" + "=" * 124, flush=True)
    print("INVESTIGATION 39 — SPATIAL STIR-NET", flush=True)
    print("=" * 124, flush=True)
    print(f"checkpoint      : {checkpoint}", flush=True)
    print(f"checkpoint step : {runtime.checkpoint_step}", flush=True)
    print(f"frames          : 0..{frame_count - 1}", flush=True)
    print(f"device          : {device}", flush=True)
    print("postfilter      : source-instance anchored SPLIT ONLY", flush=True)
    print("=" * 124, flush=True)

    future: Future[Any] | None = None

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="inv39-prep",
    ) as executor:
        future = executor.submit(
            INV36.prepare_frame,
            0,
            sample_zarr=sample_zarr,
            helper=helper,
            spacing=spacing,
        )

        for index, frame in enumerate(range(frame_count), start=1):
            wait_started = time.perf_counter()
            prepared = future.result()
            wait_seconds = time.perf_counter() - wait_started

            if int(prepared.frame) != int(frame):
                raise RuntimeError(
                    f"CPU preparation ordering failure: expected {frame}, "
                    f"got {prepared.frame}"
                )

            if frame + 1 < frame_count:
                future = executor.submit(
                    INV36.prepare_frame,
                    frame + 1,
                    sample_zarr=sample_zarr,
                    helper=helper,
                    spacing=spacing,
                )

            frame_started = time.perf_counter()

            (
                result,
                spatial_gpu,
                amp_name,
                inference_seconds,
                peak_gib,
            ) = helper.run_tiled_spatial(
                runtime.model,
                prepared.spatial,
                spacing,
                prepared.dref_um,
                device=runtime.device,
                inference_cfg=runtime.inference_cfg,
            )

            before = runner._tensor_numpy(
                result.spatial_partition.labels[0],
                np.int32,
            )
            watershed = runner._tensor_numpy(
                result.supervoxel_labels[0],
                np.int64,
            )
            separator_probability = runner._tensor_numpy(
                result.dense.geometry.probabilities()["separator"][0, 0],
                np.float32,
            )

            final_labels, split_diag = runner._apply_source_core_split_only(
                before,
                watershed,
                separator_probability,
                prepared.source_mask,
                prepared.source_labels,
                spacing,
                prepared.dref_um,
            )

            accepted_partition, accepted_component = build_accepted_partition(
                result.rag,
                result.spatial_partition,
                final_labels,
                name=f"Inv39 spatial t={frame}",
            )

            atomic_torch_save(
                output.spatial_frame(frame),
                {
                    "version": 1,
                    "frame": int(frame),
                    "dref_um": float(prepared.dref_um),
                    "rag": cpu_detached_tree(result.rag),
                    "accepted_node_component": (
                        accepted_component.detach().cpu()
                    ),
                    "accepted_component_count": int(
                        accepted_partition
                        .component_count_per_batch
                        .sum()
                        .item()
                    ),
                },
            )

            raw_movie[frame] = prepared.raw
            preprocessed_movie[frame] = np.asarray(
                prepared.preprocessed,
                dtype=np.float16,
            )
            binary_movie[frame] = np.asarray(
                prepared.source_mask > 0,
                dtype=np.uint8,
            )
            source_movie[frame] = np.asarray(
                prepared.source_labels,
                dtype=np.int32,
            )
            final_movie[frame] = np.asarray(
                final_labels,
                dtype=np.int32,
            )

            spatial_components = int(
                np.count_nonzero(np.unique(final_labels) > 0)
            )
            row = {
                "frame": int(frame),
                "dref_um": float(prepared.dref_um),
                "source_instances": int(
                    np.count_nonzero(
                        np.unique(prepared.source_labels) > 0
                    )
                ),
                "multicut_instances": int(
                    np.count_nonzero(np.unique(before) > 0)
                ),
                "spatial_final_instances": spatial_components,
                "source_core_splits": int(split_diag["applied_count"]),
                "prepare_seconds": float(prepared.prepare_seconds),
                "prepare_wait_seconds": float(wait_seconds),
                "inference_seconds": float(inference_seconds),
                "frame_seconds": float(
                    time.perf_counter() - frame_started
                ),
                "amp_dtype": str(amp_name),
                "peak_vram_gib": float(peak_gib),
                "rag_nodes": int(result.rag.node_features.shape[0]),
                "rag_edges": int(result.rag.edge_index.shape[1]),
            }
            rows.append(row)

            print(
                f"[{index:02d}/{frame_count:02d} t={frame:03d}] "
                f"source={row['source_instances']} -> "
                f"multicut={row['multicut_instances']} -> "
                f"accepted={row['spatial_final_instances']} "
                f"source_splits={row['source_core_splits']} | "
                f"infer={inference_seconds:5.1f}s "
                f"total={row['frame_seconds']:5.1f}s "
                f"VRAM={peak_gib:.2f}GiB",
                flush=True,
            )

            del (
                result,
                spatial_gpu,
                accepted_partition,
                accepted_component,
                before,
                watershed,
                separator_probability,
                final_labels,
                prepared,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    for movie in (
        raw_movie,
        preprocessed_movie,
        binary_movie,
        source_movie,
        final_movie,
    ):
        movie.flush()

    del (
        raw_movie,
        preprocessed_movie,
        binary_movie,
        source_movie,
        final_movie,
    )

    elapsed = time.perf_counter() - started_all
    atomic_json(
        output.spatial_summary,
        {
            "sample_id": sample_id,
            "frame_count": int(frame_count),
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(runtime.checkpoint_step),
            "spacing_zyx_um": list(spacing),
            "elapsed_seconds": float(elapsed),
            "elapsed": format_seconds(elapsed),
            "frames": rows,
        },
    )
    atomic_json(
        output.spatial_success,
        {
            "status": "success",
            "frame_count": int(frame_count),
            "elapsed_seconds": float(elapsed),
        },
    )

    del runtime
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[spatial] complete in {format_seconds(elapsed)}",
        flush=True,
    )


# =============================================================================
# Trackastra
# =============================================================================


def run_trackastra(
    *,
    output: OutputPaths,
    kind: str,
    instance_movie: Path,
    model_name: str,
    mode: str,
    device: str,
    rebuild: bool,
) -> None:
    if trackastra_cache_complete(output, kind) and not rebuild:
        print(f"[trackastra:{kind}] reusing cached result", flush=True)
        return

    directory = output._track_dir(kind)
    directory.mkdir(parents=True, exist_ok=True)

    try:
        from trackastra.model import Trackastra
        from trackastra.tracking.utils import graph_to_napari_tracks
    except ImportError as exc:
        raise RuntimeError(
            "Trackastra is required. Activate the repository environment "
            "containing Trackastra."
        ) from exc

    raw = np.load(output.raw, mmap_mode="r", allow_pickle=False)
    labels = np.load(instance_movie, mmap_mode="r", allow_pickle=False)

    print("\n" + "=" * 112, flush=True)
    print(f"INVESTIGATION 39 — TRACKASTRA ({kind.upper()})", flush=True)
    print("=" * 112, flush=True)
    print(f"instances : {instance_movie}", flush=True)
    print(f"model     : {model_name}", flush=True)
    print(f"mode      : {mode}", flush=True)
    print(f"device    : {device}", flush=True)
    print("=" * 112, flush=True)

    started = time.perf_counter()
    tracker = Trackastra.from_pretrained(model_name, device=device)
    graph, tracked_masks = tracker.track(
        raw,
        labels,
        mode=mode,
    )

    # Same cheap direct graph contract used in Investigation 35 training.
    INV35.annotate_graph_fast(
        graph,
        np.asarray(tracked_masks),
    )

    with output.track_graph(kind).open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(
        output.tracked_masks(kind),
        np.asarray(tracked_masks),
        allow_pickle=False,
    )

    napari_tracks, napari_graph, _properties = graph_to_napari_tracks(graph)
    napari_tracks = np.asarray(napari_tracks, dtype=np.float64)
    np.save(
        output.napari_tracks(kind),
        napari_tracks,
        allow_pickle=False,
    )

    serializable_graph = {
        str(int(child)): (
            [int(v) for v in parent]
            if isinstance(parent, (list, tuple, set))
            else int(parent)
        )
        for child, parent in napari_graph.items()
    }
    atomic_json(
        output.napari_graph(kind),
        serializable_graph,
    )

    elapsed = time.perf_counter() - started
    atomic_json(
        output.track_summary(kind),
        {
            "kind": kind,
            "model": model_name,
            "mode": mode,
            "device": device,
            "seconds": float(elapsed),
            "elapsed": format_seconds(elapsed),
            "graph_nodes": int(graph.number_of_nodes()),
            "graph_edges": int(graph.number_of_edges()),
            "napari_rows": int(len(napari_tracks)),
            "tracklets": int(
                np.unique(napari_tracks[:, 0]).size
                if napari_tracks.size
                else 0
            ),
        },
    )

    print(
        f"[trackastra:{kind}] nodes={graph.number_of_nodes()} "
        f"edges={graph.number_of_edges()} "
        f"time={format_seconds(elapsed)}",
        flush=True,
    )

    del tracker, graph, tracked_masks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Temporal inference
# =============================================================================


def load_full_model(
    checkpoint: Path,
    device: torch.device,
):
    payload, model, _cfg, _train_cfg, _stripped = (
        INV35.INV12.load_checkpoint_model_for_inference(
            checkpoint,
            device,
        )
    )
    model.eval()
    return payload, model


def make_partition_from_component(
    rag: RAGState,
    node_component: Tensor,
) -> PartitionState:
    count = (
        int(node_component.max().item()) + 1
        if node_component.numel()
        else 0
    )
    tiny = (
        torch.arange(
            1,
            count + 1,
            device=node_component.device,
            dtype=torch.long,
        ).reshape(1, 1, -1)
        if count
        else torch.zeros(
            (1, 1, 1),
            device=node_component.device,
            dtype=torch.long,
        )
    )
    return PartitionState(
        labels=[tiny],
        node_component=node_component,
        node_component_global=node_component.clone(),
        component_count_per_batch=torch.tensor(
            [count],
            device=node_component.device,
            dtype=torch.long,
        ),
        edge_logits=rag.spatial_edge_logits,
    )


@torch.no_grad()
def run_temporal_movie(
    *,
    checkpoint: Path,
    output: OutputPaths,
    frame_count: int,
    spacing: tuple[float, float, float],
    temporal_radius: int,
    accepted_logit: float,
    device: torch.device,
    rebuild: bool,
) -> None:
    if temporal_cache_complete(output, frame_count) and not rebuild:
        print("[temporal] reusing cached final instance movie", flush=True)
        return

    with output.track_graph("evidence").open("rb") as handle:
        track_graph = pickle.load(handle)

    spatial_movie = np.load(
        output.spatial_final,
        mmap_mode="r",
        allow_pickle=False,
    )
    movie_shape = tuple(int(v) for v in spatial_movie.shape)

    temporal_movie = _create_movie(
        output.temporal_final,
        dtype=np.int32,
        shape=movie_shape,
    )

    payload, model = load_full_model(checkpoint, device)

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 39 — LEARNED TEMPORAL CUT/KEEP", flush=True)
    print("=" * 112, flush=True)
    print(f"checkpoint        : {checkpoint}", flush=True)
    print(f"checkpoint step   : {int(payload.get('global_step', -1))}", flush=True)
    print(f"frames            : {frame_count}", flush=True)
    print(f"temporal radius   : {temporal_radius}", flush=True)
    print(f"accepted logit    : +/-{accepted_logit:g}", flush=True)
    print("observer           : BYPASSED (matches Investigation 35 training)", flush=True)
    print("history grids      : NONE", flush=True)
    print("hypothesis graph   : NONE", flush=True)
    print("action             : SPLIT ONLY", flush=True)
    print("=" * 112, flush=True)

    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()

    shape_zyx = movie_shape[-3:]

    for index, frame in enumerate(range(frame_count), start=1):
        started = time.perf_counter()

        state = torch_load(
            output.spatial_frame(frame),
            map_location="cpu",
        )
        rag: RAGState = to_device_fp32(
            state["rag"],
            device,
        )
        accepted_component = torch.as_tensor(
            state["accepted_node_component"],
            device=device,
            dtype=torch.long,
        )
        dref_um = float(state["dref_um"])

        accepted_partition = make_partition_from_component(
            rag,
            accepted_component,
        )
        inference_rag, same_current = rag_with_accepted_logits(
            rag,
            accepted_component,
            magnitude=float(accepted_logit),
        )

        # Partition edge logits must be aligned with the synthetic +/- baseline
        # used during temporal training.
        accepted_partition = replace(
            accepted_partition,
            edge_logits=inference_rag.spatial_edge_logits,
        )

        decoded, geometry = INV35.dummy_geometry_and_decode(
            model,
            inference_rag.node_features,
        )
        spacing_t = torch.tensor(
            [spacing],
            device=device,
            dtype=torch.float32,
        )
        dref_t = torch.tensor(
            [dref_um],
            device=device,
            dtype=torch.float32,
        )

        instances = model.instance_tokenizer(
            accepted_partition,
            inference_rag,
            decoded,
            geometry,
            spacing_t,
            dref_t,
            profile_prefix="inv39_instance_tokenizer",
        )

        temporal_input = INV35.direct_temporal_input(
            track_graph,
            target_t=frame,
            frame_count=frame_count,
            temporal_radius=int(temporal_radius),
            spacing=spacing,
            dref_um=dref_um,
            shape_zyx=shape_zyx,
            device=device,
        )

        temporal = model.temporal_encoder(temporal_input)

        reasoning = model.instance_temporal(
            instances,
            inference_rag,
            temporal,
            dref_t,
        )

        # First split-only guard at the edge level.
        final_logits = torch.where(
            same_current,
            reasoning.final_edge_logits,
            inference_rag.spatial_edge_logits,
        )

        predicted = model.partitioner(
            inference_rag,
            final_logits,
            model.cfg.partition.final_merge_threshold,
            stage="final",
        )

        # Second hard split-only guard at component level.
        final_component = hard_split_only_components(
            accepted_component,
            predicted.node_component_global,
        )

        labels = materialize_node_partition(
            inference_rag,
            final_component,
        )
        temporal_movie[frame] = labels

        accepted_count = (
            int(accepted_component.max().item()) + 1
            if accepted_component.numel()
            else 0
        )
        final_count = (
            int(final_component.max().item()) + 1
            if final_component.numel()
            else 0
        )

        split_components = 0
        for component in range(accepted_count):
            node_rows = torch.nonzero(
                accepted_component == component,
                as_tuple=False,
            ).flatten()
            if node_rows.numel() == 0:
                continue
            if torch.unique(final_component[node_rows]).numel() > 1:
                split_components += 1

        gate = reasoning.edge_temporal_gate
        delta = reasoning.edge_temporal_delta
        internal = same_current

        row = {
            "frame": int(frame),
            "dref_um": float(dref_um),
            "tracklets": int(temporal.tokens.shape[0]),
            "accepted_components": int(accepted_count),
            "temporal_components": int(final_count),
            "components_split": int(split_components),
            "internal_rag_edges": int(internal.sum().item()),
            "mean_temporal_gate_internal": (
                float(gate[internal].float().mean().item())
                if bool(internal.any())
                else 0.0
            ),
            "mean_abs_temporal_delta_internal": (
                float(delta[internal].float().abs().mean().item())
                if bool(internal.any())
                else 0.0
            ),
            "seconds": float(time.perf_counter() - started),
        }
        rows.append(row)

        print(
            f"[{index:02d}/{frame_count:02d} t={frame:03d}] "
            f"spatial={accepted_count} -> temporal={final_count} "
            f"split_components={split_components} "
            f"tracklets={row['tracklets']} "
            f"gate={row['mean_temporal_gate_internal']:.3f} "
            f"|delta|={row['mean_abs_temporal_delta_internal']:.3f} "
            f"time={row['seconds']:.2f}s",
            flush=True,
        )

        del (
            state,
            rag,
            accepted_component,
            accepted_partition,
            inference_rag,
            same_current,
            decoded,
            geometry,
            spacing_t,
            dref_t,
            instances,
            temporal_input,
            temporal,
            reasoning,
            final_logits,
            predicted,
            final_component,
            labels,
            gate,
            delta,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    temporal_movie.flush()
    del temporal_movie

    elapsed = time.perf_counter() - started_all
    atomic_json(
        output.temporal_summary,
        {
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(payload.get("global_step", -1)),
            "frame_count": int(frame_count),
            "temporal_radius": int(temporal_radius),
            "accepted_internal_logit": float(accepted_logit),
            "observer": "bypassed",
            "history_grids": False,
            "hypothesis_graph": False,
            "split_only": True,
            "elapsed_seconds": float(elapsed),
            "elapsed": format_seconds(elapsed),
            "total_spatial_components_split": int(
                sum(row["components_split"] for row in rows)
            ),
            "frames": rows,
        },
    )
    atomic_json(
        output.temporal_success,
        {
            "status": "success",
            "frame_count": int(frame_count),
            "elapsed_seconds": float(elapsed),
        },
    )

    del model, payload, track_graph
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[temporal] complete in {format_seconds(elapsed)}",
        flush=True,
    )


# =============================================================================
# Manual diagnostics
# =============================================================================


def annotation_movie_available(
    annotation_root: Path,
    frame_count: int,
) -> bool:
    return all(
        (
            annotation_root
            / f"manual_instances_t{t:03d}.npy"
        ).is_file()
        for t in range(frame_count)
    )


def build_manual_movie(
    *,
    paths: OutputPaths,
    annotation_root: Path,
    frame_count: int,
) -> None:
    if movie_shape_ok(paths.manual_gt, frame_count):
        return

    first = np.load(
        annotation_root / "manual_instances_t000.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    movie = _create_movie(
        paths.manual_gt,
        dtype=first.dtype,
        shape=(frame_count, *first.shape),
    )
    for t in range(frame_count):
        movie[t] = np.load(
            annotation_root / f"manual_instances_t{t:03d}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
    movie.flush()
    del movie


def component_diagnostics_for_frame(
    *,
    frame: int,
    spatial: np.ndarray,
    temporal: np.ndarray,
    manual: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Classify accepted spatial components according to the temporal objective.
    """
    positive = spatial > 0
    if not bool(positive.any()):
        return []

    b = spatial[positive].astype(np.int64, copy=False)
    m = manual[positive].astype(np.int64, copy=False)
    t = temporal[positive].astype(np.int64, copy=False)

    # Which accepted spatial components cover each manual cell?
    manual_to_spatial: dict[int, set[int]] = {}
    valid_manual = m > 0
    if bool(valid_manual.any()):
        pairs = np.unique(
            np.stack(
                [m[valid_manual], b[valid_manual]],
                axis=1,
            ),
            axis=0,
        )
        for manual_id, baseline_id in pairs.tolist():
            manual_to_spatial.setdefault(
                int(manual_id),
                set(),
            ).add(int(baseline_id))

    order = np.argsort(b, kind="stable")
    b_sorted = b[order]
    m_sorted = m[order]
    t_sorted = t[order]

    unique_b, starts, counts = np.unique(
        b_sorted,
        return_index=True,
        return_counts=True,
    )

    # Centroids are computed once from the spatial labels.
    z, y, x = np.nonzero(spatial > 0)
    spatial_ids = spatial[spatial > 0].astype(np.int64, copy=False)
    centroid_rows: dict[int, tuple[float, float, float]] = {}
    max_id = int(spatial_ids.max(initial=0))
    if max_id > 0:
        count_by_id = np.bincount(
            spatial_ids,
            minlength=max_id + 1,
        ).astype(np.float64)
        zsum = np.bincount(
            spatial_ids,
            weights=z,
            minlength=max_id + 1,
        )
        ysum = np.bincount(
            spatial_ids,
            weights=y,
            minlength=max_id + 1,
        )
        xsum = np.bincount(
            spatial_ids,
            weights=x,
            minlength=max_id + 1,
        )
        for label_id in unique_b.tolist():
            label_id = int(label_id)
            if count_by_id[label_id] > 0:
                centroid_rows[label_id] = (
                    float(zsum[label_id] / count_by_id[label_id]),
                    float(ysum[label_id] / count_by_id[label_id]),
                    float(xsum[label_id] / count_by_id[label_id]),
                )

    rows: list[dict[str, Any]] = []

    for baseline_id, start, count in zip(
        unique_b.tolist(),
        starts.tolist(),
        counts.tolist(),
    ):
        stop = int(start + count)
        ms = m_sorted[start:stop]
        ts = t_sorted[start:stop]

        manual_ids = sorted(
            int(v)
            for v in np.unique(ms)
            if int(v) > 0
        )
        temporal_ids = sorted(
            int(v)
            for v in np.unique(ts)
            if int(v) > 0
        )

        overseg = any(
            len(manual_to_spatial.get(manual_id, ())) > 1
            for manual_id in manual_ids
        )

        if not manual_ids:
            diag = DIAG_UNMAPPED
        elif overseg:
            diag = DIAG_SPATIAL_OVERSEG
        elif len(manual_ids) >= 2:
            # Exact recovery inside this accepted component:
            # each true manual cell must map to exactly one temporal component,
            # and those temporal components must be distinct/non-mixed.
            manual_to_temporal: dict[int, set[int]] = {}
            temporal_to_manual: dict[int, set[int]] = {}

            for manual_id, temporal_id in np.unique(
                np.stack([ms, ts], axis=1),
                axis=0,
            ).tolist():
                manual_id = int(manual_id)
                temporal_id = int(temporal_id)
                if manual_id <= 0 or temporal_id <= 0:
                    continue
                manual_to_temporal.setdefault(
                    manual_id,
                    set(),
                ).add(temporal_id)
                temporal_to_manual.setdefault(
                    temporal_id,
                    set(),
                ).add(manual_id)

            exact = all(
                len(manual_to_temporal.get(manual_id, ())) == 1
                for manual_id in manual_ids
            )
            exact = exact and all(
                len(manual_ids_for_temporal) == 1
                for manual_ids_for_temporal in temporal_to_manual.values()
            )
            exact = exact and (
                len(temporal_to_manual) == len(manual_ids)
            )

            diag = DIAG_FIXED if exact else DIAG_MISSED
        else:
            diag = (
                DIAG_PRESERVED
                if len(temporal_ids) <= 1
                else DIAG_UNEXPECTED_SPLIT
            )

        center = centroid_rows.get(
            int(baseline_id),
            (float("nan"), float("nan"), float("nan")),
        )
        rows.append(
            {
                "frame": int(frame),
                "spatial_component": int(baseline_id),
                "diagnostic_id": int(diag),
                "diagnostic": DIAG_NAMES[diag],
                "manual_ids": ",".join(str(v) for v in manual_ids),
                "temporal_ids": ",".join(str(v) for v in temporal_ids),
                "manual_count": int(len(manual_ids)),
                "temporal_count": int(len(temporal_ids)),
                "z": float(center[0]),
                "y": float(center[1]),
                "x": float(center[2]),
            }
        )

    return rows


def build_manual_diagnostics(
    *,
    output: OutputPaths,
    annotation_root: Path,
    frame_count: int,
) -> None:
    if not annotation_movie_available(annotation_root, frame_count):
        print(
            "[diagnostics] manual annotations unavailable; skipping real-error "
            "classification",
            flush=True,
        )
        return

    output.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    build_manual_movie(
        paths=output,
        annotation_root=annotation_root,
        frame_count=frame_count,
    )

    spatial = np.load(
        output.spatial_final,
        mmap_mode="r",
        allow_pickle=False,
    )
    temporal = np.load(
        output.temporal_final,
        mmap_mode="r",
        allow_pickle=False,
    )
    manual = np.load(
        output.manual_gt,
        mmap_mode="r",
        allow_pickle=False,
    )

    rows: list[dict[str, Any]] = []
    for frame in range(frame_count):
        rows.extend(
            component_diagnostics_for_frame(
                frame=frame,
                spatial=np.asarray(spatial[frame]),
                temporal=np.asarray(temporal[frame]),
                manual=np.asarray(manual[frame]),
            )
        )

    table = pd.DataFrame(rows)
    atomic_csv(output.diagnostic_components, table)

    counts = (
        table["diagnostic"]
        .value_counts()
        .sort_index()
        .to_dict()
        if not table.empty
        else {}
    )

    needed = int(
        counts.get(DIAG_NAMES[DIAG_FIXED], 0)
        + counts.get(DIAG_NAMES[DIAG_MISSED], 0)
    )
    fixed = int(counts.get(DIAG_NAMES[DIAG_FIXED], 0))
    correct = int(
        counts.get(DIAG_NAMES[DIAG_PRESERVED], 0)
        + counts.get(DIAG_NAMES[DIAG_UNEXPECTED_SPLIT], 0)
    )
    unexpected = int(
        counts.get(DIAG_NAMES[DIAG_UNEXPECTED_SPLIT], 0)
    )

    summary = {
        "frame_count": int(frame_count),
        "component_counts": {
            str(key): int(value)
            for key, value in counts.items()
        },
        "needed_split_components": needed,
        "needed_split_fixed": fixed,
        "needed_split_recall": (
            fixed / needed
            if needed
            else None
        ),
        "correct_spatial_components": correct,
        "unexpected_splits": unexpected,
        "unexpected_split_rate": (
            unexpected / correct
            if correct
            else None
        ),
        "spatial_overseg_components_outside_objective": int(
            counts.get(
                DIAG_NAMES[DIAG_SPATIAL_OVERSEG],
                0,
            )
        ),
    }
    atomic_json(output.diagnostic_summary, summary)

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 39 — REAL MANUAL DIAGNOSTICS", flush=True)
    print("=" * 112, flush=True)
    print(
        f"Needed Split - Fixed  : {fixed}/{needed} "
        f"({100.0 * summary['needed_split_recall']:.1f}%)"
        if needed
        else "Needed Split - Fixed  : no eligible merge components",
        flush=True,
    )
    print(
        f"Unexpected Split      : {unexpected}/{correct} "
        f"({100.0 * summary['unexpected_split_rate']:.2f}%)"
        if correct
        else "Unexpected Split      : no eligible correct-spatial components",
        flush=True,
    )
    print(
        "Spatial Overseg outside: "
        f"{summary['spatial_overseg_components_outside_objective']}",
        flush=True,
    )
    print("=" * 112, flush=True)


# =============================================================================
# Viewer
# =============================================================================


def add_diagnostic_points(
    viewer,
    table: pd.DataFrame,
    *,
    diagnostic: str,
    layer_name: str,
    color: str,
    scale_tzyx: tuple[float, float, float, float],
    visible: bool,
) -> None:
    subset = table.loc[table["diagnostic"] == diagnostic].copy()
    if subset.empty:
        print(f"[viewer] {layer_name}: none", flush=True)
        return

    points = subset[
        ["frame", "z", "y", "x"]
    ].to_numpy(float)

    layer = viewer.add_points(
        points,
        name=layer_name,
        scale=scale_tzyx,
        size=5,
        face_color=color,
        properties={
            "spatial_component": subset[
                "spatial_component"
            ].to_numpy(np.int64),
            "manual_ids": subset["manual_ids"].astype(str).to_numpy(),
            "temporal_ids": subset["temporal_ids"].astype(str).to_numpy(),
        },
        text={
            "string": "{spatial_component}",
            "size": 8,
            "color": "white",
        },
    )
    layer.visible = bool(visible)


def open_viewer(
    *,
    output: OutputPaths,
    spacing: tuple[float, float, float],
) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is required for Investigation 39 visualization."
        ) from exc

    raw = np.load(output.raw, mmap_mode="r", allow_pickle=False)
    preprocessed = np.load(
        output.preprocessed,
        mmap_mode="r",
        allow_pickle=False,
    )
    binary = np.load(
        output.binary_mask,
        mmap_mode="r",
        allow_pickle=False,
    )
    source = np.load(
        output.source_instances,
        mmap_mode="r",
        allow_pickle=False,
    )
    spatial = np.load(
        output.spatial_final,
        mmap_mode="r",
        allow_pickle=False,
    )
    temporal = np.load(
        output.temporal_final,
        mmap_mode="r",
        allow_pickle=False,
    )
    evidence_masks = np.load(
        output.tracked_masks("evidence"),
        mmap_mode="r",
        allow_pickle=False,
    )
    final_masks = np.load(
        output.tracked_masks("final"),
        mmap_mode="r",
        allow_pickle=False,
    )

    evidence_tracks = np.load(
        output.napari_tracks("evidence"),
        allow_pickle=False,
    )
    final_tracks = np.load(
        output.napari_tracks("final"),
        allow_pickle=False,
    )

    scale_tzyx = (1.0, *spacing)

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 39 — FIRST FULL STIR-NET VISUALIZATION", flush=True)
    print("=" * 112, flush=True)
    print(f"Spatial movie  : {output.spatial_final}", flush=True)
    print(f"Temporal movie : {output.temporal_final}", flush=True)
    print(f"Evidence graph : {output.track_graph('evidence')}", flush=True)
    print(f"Final graph    : {output.track_graph('final')}", flush=True)
    print("=" * 112, flush=True)

    viewer = napari.Viewer(ndisplay=3)

    low, high = np.percentile(np.asarray(raw), [1.0, 99.8])
    viewer.add_image(
        raw,
        name="Raw Volume",
        scale=scale_tzyx,
        rendering="mip",
        colormap="gray",
        contrast_limits=[float(low), float(high)],
    )
    viewer.add_image(
        preprocessed,
        name="Preprocessed Volume",
        scale=scale_tzyx,
        rendering="mip",
        colormap="gray",
        contrast_limits=(0.0, 1.0),
        visible=False,
    )
    viewer.add_labels(
        binary,
        name="Binary Mask",
        scale=scale_tzyx,
        visible=False,
    )
    viewer.add_labels(
        source,
        name="Source Instances",
        scale=scale_tzyx,
        visible=False,
    )
    spatial_layer = viewer.add_labels(
        spatial,
        name="Spatial Final Instances",
        scale=scale_tzyx,
        opacity=1.0,
        visible=False,
    )
    evidence_layer = viewer.add_labels(
        evidence_masks,
        name="Evidence Trackastra Masks",
        scale=scale_tzyx,
        opacity=1.0,
        visible=False,
    )
    temporal_layer = viewer.add_labels(
        temporal,
        name="Temporal Final Instances",
        scale=scale_tzyx,
        opacity=1.0,
        visible=True,
    )
    final_track_masks_layer = viewer.add_labels(
        final_masks,
        name="Final Trackastra Masks",
        scale=scale_tzyx,
        opacity=1.0,
        visible=False,
    )

    if output.manual_gt.is_file():
        manual = np.load(
            output.manual_gt,
            mmap_mode="r",
            allow_pickle=False,
        )
        viewer.add_labels(
            manual,
            name="Manual Ground Truth",
            scale=scale_tzyx,
            opacity=1.0,
            visible=False,
        )

    if evidence_tracks.size:
        evidence_tracks_layer = viewer.add_tracks(
            evidence_tracks,
            name="Evidence Tracks",
            scale=scale_tzyx,
            tail_length=20,
        )
        evidence_tracks_layer.visible = False

    if final_tracks.size:
        final_tracks_layer = viewer.add_tracks(
            final_tracks,
            name="Final Tracks",
            scale=scale_tzyx,
            tail_length=20,
        )
        final_tracks_layer.visible = True

    if output.diagnostic_components.is_file():
        table = pd.read_csv(output.diagnostic_components)

        add_diagnostic_points(
            viewer,
            table,
            diagnostic=DIAG_NAMES[DIAG_FIXED],
            layer_name="Needed Split - Fixed",
            color="lime",
            scale_tzyx=scale_tzyx,
            visible=True,
        )
        add_diagnostic_points(
            viewer,
            table,
            diagnostic=DIAG_NAMES[DIAG_MISSED],
            layer_name="Needed Split - Missed",
            color="red",
            scale_tzyx=scale_tzyx,
            visible=True,
        )
        add_diagnostic_points(
            viewer,
            table,
            diagnostic=DIAG_NAMES[DIAG_UNEXPECTED_SPLIT],
            layer_name="Unexpected Split",
            color="magenta",
            scale_tzyx=scale_tzyx,
            visible=True,
        )
        add_diagnostic_points(
            viewer,
            table,
            diagnostic=DIAG_NAMES[DIAG_SPATIAL_OVERSEG],
            layer_name="Spatial Overseg - Outside Objective",
            color="orange",
            scale_tzyx=scale_tzyx,
            visible=False,
        )

    print(
        "[viewer] Temporal Final Instances + Final Tracks are enabled. "
        "Toggle Spatial Final Instances to compare before/after temporal "
        "reasoning. Red points are missed real split cases; green points are "
        "fixed real split cases.",
        flush=True,
    )

    napari.run()


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "First full BioHub spatial+temporal STIR-Net inference, final "
            "Trackastra rerun, and Napari visualization."
        )
    )

    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--sample-zarr", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument(
        "--frame-count",
        type=int,
        default=DEFAULT_FRAME_COUNT,
    )
    parser.add_argument("--output", default=None)

    parser.add_argument(
        "--spacing",
        default="1.625,0.40625,0.40625",
    )
    parser.add_argument(
        "--tile-shape",
        default="32,128,128",
    )
    parser.add_argument(
        "--tile-overlap",
        default="8,32,32",
    )
    parser.add_argument(
        "--tile-halo",
        default="4,16,16",
    )
    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=DEFAULT_TILE_BATCH_SIZE,
    )
    parser.add_argument(
        "--spatial-device",
        default="cuda",
    )

    parser.add_argument(
        "--temporal-device",
        default="cuda",
    )
    parser.add_argument(
        "--temporal-radius",
        type=int,
        default=DEFAULT_TEMPORAL_RADIUS,
    )
    parser.add_argument(
        "--accepted-internal-logit",
        type=float,
        default=DEFAULT_ACCEPTED_INTERNAL_LOGIT,
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
        "--overwrite-spatial",
        action="store_true",
        help=(
            "Delete the complete Investigation-39 cache and rerun spatial "
            "inference from scratch."
        ),
    )
    parser.add_argument(
        "--rebuild-evidence-trackastra",
        action="store_true",
    )
    parser.add_argument(
        "--rebuild-temporal",
        action="store_true",
    )
    parser.add_argument(
        "--rebuild-final-trackastra",
        action="store_true",
    )
    parser.add_argument(
        "--skip-manual-diagnostics",
        action="store_true",
    )
    parser.add_argument(
        "--viewer-only",
        action="store_true",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.frame_count < 1:
        raise ValueError("--frame-count must be positive")
    if args.temporal_radius < 1:
        raise ValueError("--temporal-radius must be positive")
    if args.accepted_internal_logit <= 0:
        raise ValueError("--accepted-internal-logit must be positive")

    spacing = tuple(
        float(v)
        for v in parse_triplet(
            args.spacing,
            cast=float,
            name="--spacing",
        )
    )
    tile_shape = tuple(
        int(v)
        for v in parse_triplet(
            args.tile_shape,
            cast=int,
            name="--tile-shape",
        )
    )
    tile_overlap = tuple(
        int(v)
        for v in parse_triplet(
            args.tile_overlap,
            cast=int,
            name="--tile-overlap",
        )
    )
    tile_halo = tuple(
        int(v)
        for v in parse_triplet(
            args.tile_halo,
            cast=int,
            name="--tile-halo",
        )
    )

    output = OutputPaths(
        default_output(args.sample_id)
        if args.output is None
        else resolve(args.output)
    )
    checkpoint = resolve_temporal_checkpoint(
        args.sample_id,
        args.checkpoint,
    )
    sample_zarr = INV36.resolve_sample_zarr(
        args.sample_id,
        args.sample_zarr,
    )
    annotation_root = resolve_annotations(
        args.sample_id,
        args.annotations,
    )

    print("\n" + "=" * 124, flush=True)
    print("INVESTIGATION 39 — FIRST FULL STIR-NET INFERENCE", flush=True)
    print("=" * 124, flush=True)
    print(f"repository          : {ROOT}", flush=True)
    print(f"sample              : {args.sample_id}", flush=True)
    print(f"frames              : {args.frame_count}", flush=True)
    print(f"trained checkpoint  : {checkpoint}", flush=True)
    print(f"output              : {output.root}", flush=True)
    print("temporal observer   : BYPASSED (training-contract match)", flush=True)
    print("temporal action     : SPLIT ONLY", flush=True)
    print("=" * 124, flush=True)

    if args.viewer_only:
        required = (
            spatial_cache_complete(output, args.frame_count)
            and trackastra_cache_complete(output, "evidence")
            and temporal_cache_complete(output, args.frame_count)
            and trackastra_cache_complete(output, "final")
        )
        if not required:
            raise FileNotFoundError(
                f"Incomplete Investigation-39 cache below {output.root}"
            )
    else:
        if args.overwrite_spatial and output.root.exists():
            shutil.rmtree(output.root)

        if spatial_cache_complete(output, args.frame_count):
            print("[spatial] reusing complete cache", flush=True)
        else:
            spatial_device = torch.device(args.spatial_device)
            if (
                spatial_device.type == "cuda"
                and not torch.cuda.is_available()
            ):
                raise RuntimeError(
                    "CUDA requested for spatial inference but unavailable"
                )
            torch.set_grad_enabled(False)
            run_spatial_movie(
                sample_id=args.sample_id,
                sample_zarr=sample_zarr,
                checkpoint=checkpoint,
                output=output,
                frame_count=args.frame_count,
                spacing=spacing,
                tile_shape=tile_shape,
                tile_overlap=tile_overlap,
                tile_halo=tile_halo,
                tile_batch_size=int(args.tile_batch_size),
                device=spatial_device,
            )

        # Trackastra graph used as temporal evidence.
        run_trackastra(
            output=output,
            kind="evidence",
            instance_movie=output.spatial_final,
            model_name=args.trackastra_model,
            mode=args.trackastra_mode,
            device=args.trackastra_device,
            rebuild=bool(
                args.rebuild_evidence_trackastra
                or args.overwrite_spatial
            ),
        )

        temporal_device = torch.device(args.temporal_device)
        if (
            temporal_device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA requested for temporal inference but unavailable"
            )

        run_temporal_movie(
            checkpoint=checkpoint,
            output=output,
            frame_count=args.frame_count,
            spacing=spacing,
            temporal_radius=int(args.temporal_radius),
            accepted_logit=float(args.accepted_internal_logit),
            device=temporal_device,
            rebuild=bool(
                args.rebuild_temporal
                or args.overwrite_spatial
                or args.rebuild_evidence_trackastra
            ),
        )

        # Final tracking after learned temporal segmentation correction.
        run_trackastra(
            output=output,
            kind="final",
            instance_movie=output.temporal_final,
            model_name=args.trackastra_model,
            mode=args.trackastra_mode,
            device=args.trackastra_device,
            rebuild=bool(
                args.rebuild_final_trackastra
                or args.rebuild_temporal
                or args.overwrite_spatial
                or args.rebuild_evidence_trackastra
            ),
        )

        if not args.skip_manual_diagnostics:
            build_manual_diagnostics(
                output=output,
                annotation_root=annotation_root,
                frame_count=args.frame_count,
            )

    print("\n" + "=" * 124, flush=True)
    print("INVESTIGATION 39 READY", flush=True)
    print("=" * 124, flush=True)
    print(f"spatial final       : {output.spatial_final}", flush=True)
    print(f"evidence graph      : {output.track_graph('evidence')}", flush=True)
    print(f"temporal final      : {output.temporal_final}", flush=True)
    print(f"final track graph   : {output.track_graph('final')}", flush=True)
    if output.diagnostic_summary.is_file():
        print(f"manual diagnostics  : {output.diagnostic_summary}", flush=True)
    print("=" * 124, flush=True)

    if not args.no_viewer:
        open_viewer(
            output=output,
            spacing=spacing,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
