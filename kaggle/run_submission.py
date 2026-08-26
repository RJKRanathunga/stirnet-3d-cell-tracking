from __future__ import annotations

"""Kaggle inference entry point for the spatial-only STIR-Net leaderboard baseline.

This runner intentionally freezes the scientific boundary at:

    canonical BioHub preprocessing + source segmentation
    -> current tiled STIR-Net spatial inference
    -> production signed multicut partition
    -> source-instance anchored split-only postfilter
    -> existing Stage 7/8/10/11 tracking, lineage, reconciliation
    -> Kaggle node/edge submission.csv

No learned temporal STIR-Net reasoning is executed here.  The first leaderboard
submission is meant to measure the current spatial branch cleanly.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)
DEFAULT_TILE_SHAPE_ZYX = (32, 128, 128)
DEFAULT_TILE_OVERLAP_ZYX = (8, 32, 32)
DEFAULT_TILE_HALO_ZYX = (4, 16, 16)
DEFAULT_TILE_BATCH_SIZE = 1
DEFAULT_GRAPH_WINDOW_SIZE = 7
DEFAULT_GRAPH_MAX_GAP = 2
DEFAULT_GRAPH_SOLVER_SECONDS = 30.0
DEFAULT_GRAPH_FALLBACK_ITERATIONS = 5
EXPECTED_SOURCE_REPO_SHA = "ac981bf4fa79f9a5f8db2aa24f11adcd87754482"
SUBMISSION_COLUMNS = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)


@dataclass(frozen=True)
class SampleSpec:
    dataset: str
    zarr_path: Path


@dataclass(frozen=True)
class ModelRuntime:
    model: Any
    model_cfg: Any
    checkpoint_step: int
    inference_cfg: Any
    device: torch.device
    checkpoint_sha256: str


def _repo_root_from(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    required = (
        root / "learned" / "stirnet",
        root / "src",
        root / "investigations" / "stirnet" / "data",
        root / "pyproject.toml",
    )
    missing = [str(value) for value in required if not value.exists()]
    if missing:
        raise FileNotFoundError(
            "Incomplete inference repository root. Missing: " + ", ".join(missing)
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(tmp, np.asarray(array), allow_pickle=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parse_zyx(text: str, *, name: str, positive: bool = True) -> tuple[int, int, int]:
    values = tuple(int(token.strip()) for token in text.split(","))
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three Z,Y,X integers")
    if positive and any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive")
    if not positive and any(value < 0 for value in values):
        raise ValueError(f"{name} values must be non-negative")
    return values


def _parse_spacing(text: str) -> tuple[float, float, float]:
    values = tuple(float(token.strip()) for token in text.split(","))
    if len(values) != 3 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("--spacing must contain three positive finite Z,Y,X values")
    return values


def _discover_samples(test_root: Path) -> list[SampleSpec]:
    root = test_root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    candidates: dict[str, Path] = {}
    patterns = ("*.zarr", "*/*.zarr")
    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.is_dir():
                continue
            dataset = path.name[:-5] if path.name.endswith(".zarr") else path.stem
            previous = candidates.get(dataset)
            if previous is not None and previous.resolve() != path.resolve():
                raise RuntimeError(
                    f"Duplicate dataset name {dataset!r}: {previous} and {path}"
                )
            candidates[dataset] = path.resolve()

    if not candidates:
        raise FileNotFoundError(f"No .zarr test datasets found below {root}")
    return [SampleSpec(name, candidates[name]) for name in sorted(candidates)]


def _resolve_device(value: str) -> torch.device:
    token = value.strip().lower()
    if token == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(token)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Inference is deterministic at the algorithm level.  We do not force
    # torch.use_deterministic_algorithms(True), which can disable otherwise-safe
    # CUDA kernels and materially increase Kaggle runtime.


def _load_model_runtime(
    repo_root: Path,
    checkpoint: Path,
    *,
    device: torch.device,
    tile_shape: tuple[int, int, int],
    tile_overlap: tuple[int, int, int],
    tile_halo: tuple[int, int, int],
    tile_batch_size: int,
) -> tuple[ModelRuntime, Any]:
    helper = _load_module(
        repo_root / "investigations" / "stirnet" / "data" / "12_biohub_full_volume_spatial_inference.py",
        "_kaggle_stirnet_spatial_helper",
    )
    (
        payload,
        model,
        model_cfg,
        _train_cfg,
        stripped_training_config,
    ) = helper.load_checkpoint_model_for_inference(checkpoint, device)
    model.eval()
    inference_cfg = helper.build_inference_config(
        model_cfg,
        tile_shape_zyx=tile_shape,
        tile_overlap_zyx=tile_overlap,
        tile_halo_zyx=tile_halo,
        tile_batch_size=tile_batch_size,
    )
    print(
        f"[checkpoint] step={int(payload['global_step'])} "
        f"training_config_compat={'stripped' if stripped_training_config else 'native'}",
        flush=True,
    )
    runtime = ModelRuntime(
        model=model,
        model_cfg=model_cfg,
        checkpoint_step=int(payload["global_step"]),
        inference_cfg=inference_cfg,
        device=device,
        checkpoint_sha256=_sha256(checkpoint),
    )
    return runtime, helper


def _tensor_numpy(value: Any, dtype=None) -> np.ndarray:
    tensor = torch.as_tensor(value).detach()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    array = tensor.cpu().numpy()
    return array.astype(dtype, copy=False) if dtype is not None else array


def _strict_refinement_check(before: np.ndarray, after: np.ndarray) -> None:
    """The split-only postfilter may split a multicut component, never merge two."""
    positive = after > 0
    if not bool(positive.any()):
        return
    pairs = np.unique(
        np.stack(
            [
                after[positive].astype(np.int64, copy=False),
                before[positive].astype(np.int64, copy=False),
            ],
            axis=1,
        ),
        axis=0,
    )
    parent_by_child: dict[int, int] = {}
    for child, parent in pairs.tolist():
        child = int(child)
        parent = int(parent)
        old = parent_by_child.get(child)
        if old is not None and old != parent:
            raise RuntimeError(
                "Source-core split-only invariant failed: output label "
                f"{child} contains multicut labels {old} and {parent}."
            )
        parent_by_child[child] = parent


def _apply_source_core_split_only(
    before: np.ndarray,
    watershed: np.ndarray,
    separator_probability: np.ndarray,
    source_mask: np.ndarray,
    source_labels: np.ndarray,
    spacing: tuple[float, float, float],
    dref_um: float,
) -> tuple[np.ndarray, dict[str, int]]:
    from learned.stirnet.model.config import InferenceConfig
    from learned.stirnet.model.postprocess.source_core_split import SourceCoreSplitOnlyFilter

    cfg = InferenceConfig()
    # This is the production default established by Investigation 25: Stage-6
    # source instance IDs are independent split-only anchors.  They do not feed
    # back into the learned RAG or signed multicut solve.
    cfg.source_core_split_anchor_mode = "prefer_source_instances"
    filt = SourceCoreSplitOnlyFilter(cfg)
    with torch.inference_mode():
        state = filt(
            [torch.as_tensor(before, dtype=torch.long)],
            torch.as_tensor((source_mask > 0)[None], dtype=torch.float32),
            torch.as_tensor(separator_probability[None], dtype=torch.float32),
            torch.tensor([spacing], dtype=torch.float32),
            torch.tensor([float(dref_um)], dtype=torch.float32),
            supervoxel_labels=[torch.as_tensor(watershed, dtype=torch.long)],
            source_instance_labels=torch.as_tensor(
                source_labels[None], dtype=torch.long
            ),
        )
    after = state.labels[0].detach().cpu().numpy().astype(np.int32, copy=False)
    _strict_refinement_check(before, after)
    diagnostics = {
        "candidate_count": int(state.candidate_count),
        "applied_count": int(state.applied_count),
        "skipped_too_many_cores": int(state.skipped_too_many_cores),
    }
    return after, diagnostics


def _count_labels(labels: np.ndarray) -> int:
    values = np.unique(labels)
    return int(np.count_nonzero(values > 0))


# STIRNET_PARALLEL_SPATIAL_PIPELINE_V1
#
# Production promotion of Investigation 34:
# keep exactly one future frame in CPU preparation while the main thread owns
# CUDA inference for the current frame. This deliberately bounds RAM and avoids
# concurrent CUDA access from worker threads.


@dataclass
class _PreparedSpatialFrame:
    frame: int
    preprocessed: np.ndarray
    source_mask: np.ndarray
    source_labels: np.ndarray
    spatial: np.ndarray
    dref_um: float
    preparation_seconds: float


def _prepare_spatial_frame(
    sample_zarr: Path,
    frame: int,
    helper: Any,
    *,
    spacing: tuple[float, float, float],
    segmentation_config: Any,
) -> _PreparedSpatialFrame:
    """Prepare one frame on CPU without touching CUDA."""
    from src.api import (
        create_binary_mask,
        preprocess_volume,
        segment_instances,
    )
    from src.io import load_timepoint

    started = time.perf_counter()

    raw = load_timepoint(sample_zarr, frame)
    preprocessed = preprocess_volume(raw)
    source_mask = create_binary_mask(preprocessed)
    source_labels = segment_instances(
        source_mask,
        config=segmentation_config,
    )
    spatial, dref_um = helper.build_stage6_spatial_input(
        preprocessed,
        source_labels,
        spacing,
    )

    preparation_seconds = time.perf_counter() - started
    del raw

    return _PreparedSpatialFrame(
        frame=int(frame),
        preprocessed=preprocessed,
        source_mask=source_mask,
        source_labels=source_labels,
        spatial=spatial,
        dref_um=float(dref_um),
        preparation_seconds=float(preparation_seconds),
    )


def _process_spatial_sample(
    sample: SampleSpec,
    sample_work: Path,
    runtime: ModelRuntime,
    helper: Any,
    *,
    spacing: tuple[float, float, float],
) -> tuple[list[pd.DataFrame], tuple[Path, ...], list[dict[str, Any]]]:
    from importlib import import_module

    from src.api import detect_cells, extract_cell_features
    from src.io import open_sample

    image = open_sample(sample.zarr_path)
    if len(image.shape) != 4:
        raise ValueError(
            f"{sample.dataset}: expected T,Z,Y,X image, got shape={image.shape}"
        )
    frame_count = int(image.shape[0])
    if frame_count <= 0:
        raise ValueError(f"{sample.dataset}: empty time axis")

    # Canonical source-instance configuration used by the Stage-6 BioHub cache
    # and validated bit-exactly before Investigation 34.
    segmentation_config_module = import_module(
        "src.03_segmentation.config"
    )
    source_segmentation_config = replace(
        segmentation_config_module.DEFAULT_SEGMENTATION_CONFIG,
        enable_geometric_completion=False,
    )

    segmentation_dir = sample_work / "segmentation"
    cells_dir = sample_work / "cells"
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)

    time_frames: list[pd.DataFrame] = []
    segmentation_files: list[Path] = []
    frame_summaries: list[dict[str, Any]] = []

    # Investigation 34 established that one producer is sufficient:
    # preparation is shorter than the CUDA spatial call and therefore remains
    # fully hidden after the first frame. Keep the lookahead bounded to one
    # prepared frame to cap host RAM.
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="stirnet-spatial-prep",
    ) as prep_executor:
        prepared_future = prep_executor.submit(
            _prepare_spatial_frame,
            sample.zarr_path,
            0,
            helper,
            spacing=spacing,
            segmentation_config=source_segmentation_config,
        )

        for frame in range(frame_count):
            started = time.perf_counter()

            wait_started = time.perf_counter()
            prepared = prepared_future.result()
            preparation_wait_seconds = time.perf_counter() - wait_started
            if prepared.frame != frame:
                raise RuntimeError(
                    f"{sample.dataset}: spatial preparation ordering failure: "
                    f"expected t={frame:03d}, got t={prepared.frame:03d}"
                )

            # Start preparing t+1 before CUDA starts processing t.
            next_frame = frame + 1
            if next_frame < frame_count:
                prepared_future = prep_executor.submit(
                    _prepare_spatial_frame,
                    sample.zarr_path,
                    next_frame,
                    helper,
                    spacing=spacing,
                    segmentation_config=source_segmentation_config,
                )

            result, spatial_gpu, amp_name, inference_seconds, peak_gib = (
                helper.run_tiled_spatial(
                    runtime.model,
                    prepared.spatial,
                    spacing,
                    prepared.dref_um,
                    device=runtime.device,
                    inference_cfg=runtime.inference_cfg,
                )
            )

            before = _tensor_numpy(
                result.spatial_partition.labels[0],
                np.int32,
            )
            watershed = _tensor_numpy(
                result.supervoxel_labels[0],
                np.int64,
            )
            separator_probability = _tensor_numpy(
                result.dense.geometry.probabilities()["separator"][0, 0],
                np.float32,
            )
            final_labels, split_diag = _apply_source_core_split_only(
                before,
                watershed,
                separator_probability,
                prepared.source_mask,
                prepared.source_labels,
                spacing,
                prepared.dref_um,
            )

            cells = detect_cells(final_labels)
            cells = extract_cell_features(
                cells,
                final_labels,
                prepared.preprocessed,
            )
            if cells.empty:
                raise RuntimeError(
                    f"{sample.dataset} t={frame}: STIR-Net produced no cells; "
                    "refusing to create a structurally valid but empty submission."
                )

            segmentation_path = segmentation_dir / f"t{frame:03d}.npy"
            cells_path = cells_dir / f"t{frame:03d}.csv"
            _atomic_npy(segmentation_path, final_labels)
            _atomic_csv(cells_path, cells)
            time_frames.append(cells)
            segmentation_files.append(segmentation_path)

            total_seconds = time.perf_counter() - started
            preparation_hidden_seconds = max(
                float(prepared.preparation_seconds)
                - float(preparation_wait_seconds),
                0.0,
            )
            preparation_hidden_fraction = (
                preparation_hidden_seconds
                / float(prepared.preparation_seconds)
                if prepared.preparation_seconds > 0
                else 0.0
            )
            summary = {
                "frame": frame,
                "source_instances": _count_labels(prepared.source_labels),
                "multicut_instances": _count_labels(before),
                "final_instances": _count_labels(final_labels),
                "split_candidates": split_diag["candidate_count"],
                "splits_applied": split_diag["applied_count"],
                "amp_dtype": amp_name,
                "inference_seconds": float(inference_seconds),
                "preparation_seconds": float(prepared.preparation_seconds),
                "preparation_wait_seconds": float(preparation_wait_seconds),
                "preparation_hidden_seconds": float(preparation_hidden_seconds),
                "preparation_hidden_fraction": float(
                    preparation_hidden_fraction
                ),
                "total_seconds": float(total_seconds),
                "peak_allocated_vram_gib": float(peak_gib),
            }
            frame_summaries.append(summary)
            print(
                f"[{sample.dataset} t={frame:03d}] "
                f"source={summary['source_instances']} -> "
                f"multicut={summary['multicut_instances']} -> "
                f"final={summary['final_instances']} | "
                f"splits={summary['splits_applied']} | "
                f"prep={summary['preparation_seconds']:.2f}s "
                f"prep_wait={summary['preparation_wait_seconds']:.2f}s "
                f"hidden={100.0 * summary['preparation_hidden_fraction']:.1f}% | "
                f"infer={inference_seconds:.2f}s total={total_seconds:.2f}s "
                f"VRAM={peak_gib:.2f}GiB",
                flush=True,
            )

            del result, spatial_gpu, prepared.spatial
            del separator_probability, watershed, before, final_labels
            del prepared, cells
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    return time_frames, tuple(segmentation_files), frame_summaries


def _run_tracking_stack(
    sample: SampleSpec,
    time_frames: list[pd.DataFrame],
    segmentation_files: tuple[Path, ...],
    *,
    graph_window_size: int,
    graph_max_gap: int,
    graph_solver_seconds: float,
    graph_fallback_iterations: int,
):
    from importlib import import_module

    from src.api import (
        FourDGraphConfig,
        GraphTrackingConfig,
        run_cell_lineage,
        run_cell_tracking,
        run_track_reconciliation,
        run_track_stitching,
    )
    from src.io import open_sample

    four_d = FourDGraphConfig(
        window_size=int(graph_window_size),
        lookback_frames=int(graph_window_size) // 2,
        lookahead_frames=int(graph_window_size) // 2,
        maximum_gap_frames=int(graph_max_gap),
        solver_time_limit_seconds=float(graph_solver_seconds),
        iterative_solver_iterations=int(graph_fallback_iterations),
        save_detailed_debug_artifacts=False,
    )
    graph_cfg = GraphTrackingConfig(
        mode="apply",
        algorithm="windowed_4d",
        four_d=four_d,
    )

    print(f"[{sample.dataset}] Stage 7 tracking ...", flush=True)
    stage7 = run_cell_tracking(
        time_frames,
        sample_id=sample.dataset,
        graph_config=graph_cfg,
    )
    print(f"[{sample.dataset}] Stage 8 stitching ...", flush=True)
    stage8 = run_track_stitching(
        stage7.tracks,
        time_frames,
        segmentation_files,
        sample_id=sample.dataset,
    )
    print(f"[{sample.dataset}] Stage 10 lineage ...", flush=True)
    raw_volume = open_sample(sample.zarr_path)
    stage10 = run_cell_lineage(
        stage8.tracks,
        time_frames,
        raw_volume,
        segmentation_files,
        segmentation_events=stage8.segmentation_events,
        sample_id=sample.dataset,
    )
    print(f"[{sample.dataset}] Stage 11 reconciliation ...", flush=True)
    reconciliation_cls = import_module(
        "src.11_track_reconciliation.step01_config"
    ).TrackReconciliationConfig
    spatial_shape = tuple(
        int(v)
        for v in np.load(segmentation_files[0], mmap_mode="r", allow_pickle=False).shape
    )
    stage11 = run_track_reconciliation(
        stage8.tracks,
        time_frames,
        segmentation_events=stage8.segmentation_events,
        division_events=stage10.division_events,
        lineage_edges=stage10.lineage_edges,
        track_lineage=stage10.track_lineage,
        protected_tracks=stage10.protected_tracks,
        global_motion=stage7.global_motion,
        association_events=stage7.association_events,
        association_candidates=stage7.association_candidates,
        spatial_shape_zyx=spatial_shape,
        sample_id=sample.dataset,
        config=reconciliation_cls(policy="submission"),
    )
    return stage7, stage8, stage10, stage11


def _valid_final_track_rows(tracks: pd.DataFrame) -> pd.DataFrame:
    required = {"track_id", "frame", "cell_id", "z", "y", "x"}
    missing = sorted(required - set(tracks.columns))
    if missing:
        raise ValueError(f"Final track table is missing columns: {missing}")
    rows = tracks.copy()
    mask = np.ones(len(rows), dtype=bool)
    if "cell" in rows.columns:
        mask &= pd.to_numeric(rows["cell"], errors="coerce").fillna(-1).to_numpy() >= 0
    mask &= pd.to_numeric(rows["cell_id"], errors="coerce").fillna(-1).to_numpy() >= 0
    for column in ("frame", "z", "y", "x"):
        mask &= np.isfinite(pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float))
    return rows.loc[mask].copy()


def _submission_rows_for_sample(
    sample: SampleSpec,
    final_tracks: pd.DataFrame,
    lineage_edges: pd.DataFrame,
    spatial_shape_zyx: tuple[int, int, int],
) -> list[dict[str, Any]]:
    tracks = _valid_final_track_rows(final_tracks)
    if tracks.empty:
        raise RuntimeError(f"{sample.dataset}: no real final track observations")

    for column in ("track_id", "frame", "cell_id"):
        tracks[column] = pd.to_numeric(tracks[column], errors="raise").astype(np.int64)
    tracks = tracks.sort_values(["frame", "track_id", "cell_id"]).reset_index(drop=True)

    # Stage 8 intentionally replaces one observed merged component with two
    # virtual parent-center trajectories.  Those virtual rows inherit the same
    # source cell/cell_id but are two legitimate predicted cell nodes, so the
    # physical-detection uniqueness guard applies only to non-virtual rows.
    if "is_virtual_merge" in tracks.columns:
        virtual_mask = tracks["is_virtual_merge"].fillna(False).astype(bool)
    else:
        virtual_mask = pd.Series(False, index=tracks.index, dtype=bool)
    real_rows = tracks.loc[~virtual_mask]
    duplicate_detection = real_rows.duplicated(subset=["frame", "cell_id"], keep=False)
    if bool(duplicate_detection.any()):
        bad = real_rows.loc[
            duplicate_detection, ["track_id", "frame", "cell_id"]
        ].head(10)
        raise RuntimeError(
            f"{sample.dataset}: one non-virtual observed cell belongs to multiple final tracks:\n{bad}"
        )
    duplicate_track_frame = tracks.duplicated(subset=["track_id", "frame"], keep=False)
    if bool(duplicate_track_frame.any()):
        bad = tracks.loc[duplicate_track_frame, ["track_id", "frame", "cell_id"]].head(10)
        raise RuntimeError(
            f"{sample.dataset}: final track has multiple observations in one frame:\n{bad}"
        )

    z_size, y_size, x_size = spatial_shape_zyx
    rows: list[dict[str, Any]] = []
    node_by_track_frame: dict[tuple[int, int], int] = {}
    node_time: dict[int, int] = {}
    next_node_id = 1

    for row in tracks.itertuples(index=False):
        t = int(row.frame)
        z = int(np.clip(np.rint(float(row.z)), 0, z_size - 1))
        y = int(np.clip(np.rint(float(row.y)), 0, y_size - 1))
        x = int(np.clip(np.rint(float(row.x)), 0, x_size - 1))
        node_id = next_node_id
        next_node_id += 1
        key = (int(row.track_id), t)
        node_by_track_frame[key] = node_id
        node_time[node_id] = t
        rows.append(
            {
                "dataset": sample.dataset,
                "row_type": "node",
                "node_id": node_id,
                "t": t,
                "z": z,
                "y": y,
                "x": x,
                "source_id": -1,
                "target_id": -1,
            }
        )

    edges: set[tuple[int, int]] = set()
    endpoints: dict[int, tuple[int, int]] = {}
    for track_id, group in tracks.groupby("track_id", sort=True):
        ordered = group.sort_values("frame")
        node_ids = [
            node_by_track_frame[(int(track_id), int(frame))]
            for frame in ordered["frame"].tolist()
        ]
        times = [int(value) for value in ordered["frame"].tolist()]
        endpoints[int(track_id)] = (node_ids[0], node_ids[-1])
        for source, target, t0, t1 in zip(node_ids[:-1], node_ids[1:], times[:-1], times[1:]):
            if t1 <= t0:
                raise RuntimeError(
                    f"{sample.dataset}: non-forward continuation on track {track_id}: {t0}->{t1}"
                )
            edges.add((int(source), int(target)))

    if lineage_edges is not None and not lineage_edges.empty:
        needed = {"parent_track_id", "child_track_id"}
        if not needed.issubset(lineage_edges.columns):
            raise ValueError(
                f"{sample.dataset}: lineage edge table missing {sorted(needed - set(lineage_edges.columns))}"
            )
        for relation in lineage_edges.itertuples(index=False):
            if pd.isna(relation.parent_track_id) or pd.isna(relation.child_track_id):
                continue
            parent = int(relation.parent_track_id)
            child = int(relation.child_track_id)
            if parent == child or parent not in endpoints or child not in endpoints:
                continue
            source = endpoints[parent][1]
            target = endpoints[child][0]
            if node_time[target] <= node_time[source]:
                raise RuntimeError(
                    f"{sample.dataset}: lineage edge is not forward in time: "
                    f"track {parent} -> {child} ({node_time[source]}->{node_time[target]})"
                )
            edges.add((source, target))

    for source, target in sorted(edges, key=lambda pair: (node_time[pair[0]], pair[0], pair[1])):
        rows.append(
            {
                "dataset": sample.dataset,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": int(source),
                "target_id": int(target),
            }
        )
    return rows


def _write_submission(path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        raise RuntimeError("No submission rows were generated")
    frame = pd.DataFrame(rows)
    frame.insert(0, "id", np.arange(len(frame), dtype=np.int64))
    frame = frame.loc[:, SUBMISSION_COLUMNS]
    for column in ("id", "node_id", "t", "z", "y", "x", "source_id", "target_id"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    _atomic_csv(path, frame)
    return frame


def _validate_submission_in_process(
    repo_root: Path,
    submission: Path,
    test_root: Path,
) -> None:
    validator = _load_module(
        repo_root / "kaggle" / "validate_submission.py",
        "_kaggle_submission_validator",
    )
    report = validator.validate_submission(
        submission,
        test_root=test_root,
        require_exact_dataset_set=True,
        check_zarr_bounds=True,
    )
    if report.errors:
        raise RuntimeError(
            "Submission validation failed:\n- " + "\n- ".join(report.errors)
        )
    for warning in report.warnings:
        print(f"[validator warning] {warning}", flush=True)
    print(
        f"[validator] OK: datasets={report.dataset_count} "
        f"nodes={report.node_count} edges={report.edge_count}",
        flush=True,
    )


def _runtime_manifest(
    *,
    args: argparse.Namespace,
    runtime: ModelRuntime,
    samples: list[SampleSpec],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_source_repo_sha": args.expected_repo_sha,
        "repo_root": str(repo_root),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "checkpoint_step": runtime.checkpoint_step,
        "device": str(runtime.device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(runtime.device)
            if runtime.device.type == "cuda"
            else None
        ),
        "spacing_zyx_um": list(args.spacing_values),
        "tile_shape_zyx": list(args.tile_shape_values),
        "tile_overlap_zyx": list(args.tile_overlap_values),
        "tile_halo_zyx": list(args.tile_halo_values),
        "tile_batch_size": args.tile_batch_size,
        "temporal_stirnet_enabled": False,
        "spatial_postfilter": "source_instance_anchors_split_only",
        "tracking": {
            "graph_mode": "apply",
            "graph_algorithm": "windowed_4d",
            "window_size": args.graph_window_size,
            "maximum_gap_frames": args.graph_max_gap,
            "solver_time_limit_seconds": args.graph_solver_seconds,
            "iterative_fallback_iterations": args.graph_fallback_iterations,
            "reconciliation_policy": "submission",
        },
        "datasets": [sample.dataset for sample in samples],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen spatial STIR-Net + current tracking stack and create submission.csv."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("/kaggle/working/stirnet_submission_work"),
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("/kaggle/working/submission.csv"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--spacing",
        default=",".join(str(v) for v in DEFAULT_SPACING_ZYX_UM),
    )
    parser.add_argument("--tile-shape-zyx", default="32,128,128")
    parser.add_argument("--tile-overlap-zyx", default="8,32,32")
    parser.add_argument("--tile-halo-zyx", default="4,16,16")
    parser.add_argument("--tile-batch-size", type=int, default=DEFAULT_TILE_BATCH_SIZE)
    parser.add_argument("--graph-window-size", type=int, default=DEFAULT_GRAPH_WINDOW_SIZE)
    parser.add_argument("--graph-max-gap", type=int, default=DEFAULT_GRAPH_MAX_GAP)
    parser.add_argument(
        "--graph-solver-seconds", type=float, default=DEFAULT_GRAPH_SOLVER_SECONDS
    )
    parser.add_argument(
        "--graph-fallback-iterations",
        type=int,
        default=DEFAULT_GRAPH_FALLBACK_ITERATIONS,
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--expected-repo-sha",
        default=EXPECTED_SOURCE_REPO_SHA,
        help="Recorded provenance SHA. The bundle verifier enforces this before execution.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep the work directory. Default is a clean run to prevent stale hidden-test artifacts.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = _repo_root_from(args.repo_root)
    args.spacing_values = _parse_spacing(args.spacing)
    args.tile_shape_values = _parse_zyx(args.tile_shape_zyx, name="--tile-shape-zyx")
    args.tile_overlap_values = _parse_zyx(
        args.tile_overlap_zyx, name="--tile-overlap-zyx", positive=False
    )
    args.tile_halo_values = _parse_zyx(
        args.tile_halo_zyx, name="--tile-halo-zyx", positive=False
    )
    if args.tile_batch_size < 1:
        raise ValueError("--tile-batch-size must be positive")
    if args.graph_window_size < 3 or args.graph_window_size % 2 == 0:
        raise ValueError("--graph-window-size must be an odd integer >= 3")
    if args.graph_max_gap < 0:
        raise ValueError("--graph-max-gap must be non-negative")

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    test_root = args.test_root.expanduser().resolve()
    samples = _discover_samples(test_root)
    device = _resolve_device(args.device)
    _seed_everything(args.seed)
    torch.set_grad_enabled(False)

    work_root = args.work_root.expanduser().resolve()
    if work_root.exists() and not args.resume:
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    submission_path = args.submission.expanduser().resolve()
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    runtime, helper = _load_model_runtime(
        repo_root,
        checkpoint,
        device=device,
        tile_shape=args.tile_shape_values,
        tile_overlap=args.tile_overlap_values,
        tile_halo=args.tile_halo_values,
        tile_batch_size=args.tile_batch_size,
    )
    manifest = _runtime_manifest(
        args=args,
        runtime=runtime,
        samples=samples,
        repo_root=repo_root,
    )
    _atomic_json(work_root / "run_manifest.json", manifest)

    print("=" * 112, flush=True)
    print("KAGGLE SPATIAL STIR-NET SUBMISSION", flush=True)
    print("=" * 112, flush=True)
    print(f"datasets       : {len(samples)}", flush=True)
    print(f"test root      : {test_root}", flush=True)
    print(f"checkpoint     : {checkpoint}", flush=True)
    print(f"checkpoint step: {runtime.checkpoint_step}", flush=True)
    print(f"device         : {device}", flush=True)
    print(f"temporal STIR  : DISABLED", flush=True)
    print(f"output         : {submission_path}", flush=True)
    print("=" * 112, flush=True)

    run_started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(samples, start=1):
        sample_started = time.perf_counter()
        sample_work = work_root / "samples" / sample.dataset
        sample_work.mkdir(parents=True, exist_ok=True)
        print(
            f"\n[{sample_index}/{len(samples)}] {sample.dataset} | {sample.zarr_path}",
            flush=True,
        )
        time_frames, segmentation_files, frame_summaries = _process_spatial_sample(
            sample,
            sample_work,
            runtime,
            helper,
            spacing=args.spacing_values,
        )
        stage7, stage8, stage10, stage11 = _run_tracking_stack(
            sample,
            time_frames,
            segmentation_files,
            graph_window_size=args.graph_window_size,
            graph_max_gap=args.graph_max_gap,
            graph_solver_seconds=args.graph_solver_seconds,
            graph_fallback_iterations=args.graph_fallback_iterations,
        )
        spatial_shape = tuple(
            int(v)
            for v in np.load(segmentation_files[0], mmap_mode="r", allow_pickle=False).shape
        )
        rows = _submission_rows_for_sample(
            sample,
            stage11.tracks,
            stage11.lineage_edges,
            spatial_shape,
        )
        all_rows.extend(rows)
        node_count = sum(row["row_type"] == "node" for row in rows)
        edge_count = len(rows) - node_count
        sample_seconds = time.perf_counter() - sample_started
        summary = {
            "dataset": sample.dataset,
            "frames": len(time_frames),
            "nodes": node_count,
            "edges": edge_count,
            "final_track_count": int(stage11.tracks["track_id"].nunique()),
            "division_lineage_edge_count": int(len(stage11.lineage_edges)),
            "seconds": float(sample_seconds),
            "spatial_frames": frame_summaries,
        }
        sample_summaries.append(summary)
        _atomic_json(sample_work / "sample_summary.json", summary)
        print(
            f"[{sample.dataset}] COMPLETE | nodes={node_count} edges={edge_count} "
            f"tracks={summary['final_track_count']} time={sample_seconds / 60:.1f} min",
            flush=True,
        )

        del stage7, stage8, stage10, stage11, time_frames
        gc.collect()
        if runtime.device.type == "cuda":
            torch.cuda.empty_cache()

    submission_frame = _write_submission(submission_path, all_rows)
    _validate_submission_in_process(repo_root, submission_path, test_root)
    total_seconds = time.perf_counter() - run_started
    final_summary = {
        "status": "success",
        "datasets": len(samples),
        "rows": int(len(submission_frame)),
        "nodes": int((submission_frame["row_type"] == "node").sum()),
        "edges": int((submission_frame["row_type"] == "edge").sum()),
        "elapsed_seconds": float(total_seconds),
        "elapsed_hours": float(total_seconds / 3600.0),
        "submission": str(submission_path),
        "submission_sha256": _sha256(submission_path),
        "sample_summaries": sample_summaries,
    }
    _atomic_json(work_root / "final_summary.json", final_summary)
    print("\n" + "=" * 112, flush=True)
    print("SUBMISSION READY", flush=True)
    print("=" * 112, flush=True)
    print(f"path    : {submission_path}", flush=True)
    print(f"rows    : {final_summary['rows']}", flush=True)
    print(f"nodes   : {final_summary['nodes']}", flush=True)
    print(f"edges   : {final_summary['edges']}", flush=True)
    print(f"elapsed : {final_summary['elapsed_hours']:.3f} h", flush=True)
    print(f"sha256  : {final_summary['submission_sha256']}", flush=True)
    print("=" * 112, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
