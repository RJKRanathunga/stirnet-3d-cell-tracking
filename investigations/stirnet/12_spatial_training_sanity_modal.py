from __future__ import annotations

"""
Stage 12 v2 — full-architecture crop-training sanity check on Modal.

Purpose
-------
After the Stage-10 bottlenecks and the later preprocessing / autograd fixes, this
stage answers one narrow question:

    Can the CURRENT production spatial-training path complete a real
    forward -> loss -> backward -> optimizer step on a representative
    BlastoSPIM frame?

It deliberately exercises the mechanisms intended for broad spatial training:

    ORIGINAL BlastoSPIM raw + GT
        -> canonical source/current segmentation
        -> source-only useful acquisition trim
        -> exact SciPy STATIC GT geometry once
        -> deterministic coverage-driven crop
        -> FAST production watershed backend
        -> full-width STIR-Net
        -> geometry + watershed + safety guard + RAG
        -> spatial_partition loss
        -> backward
        -> optimizer step

A second optimizer step reuses exactly the same STATIC GT geometry but uses a
synthetically corrupted source/current segmentation. Only the
source-conditioned corrective separator is rebuilt by the Trainer. This checks
the intended amortization path:

    expensive static GT geometry: once
    dynamic corrective separator: per changed source state
    forward/backward: per optimizer step

The second source state is created by joining two nearby source components
inside the crop selected for step B with a thin bridge. If a safe pair cannot be
constructed, the script falls back to removing one source component. Raw
microscopy and GT never change.

This is a SANITY / plumbing experiment, not a learning-quality experiment.
The GPU EDT optimization is deliberately excluded from v2 because the first
Stage-12 attempt showed a fatal CuPy CUDA illegal-address event before the
network ran. GPU EDT will be benchmarked/fixed separately so it cannot poison
the CUDA context used for PyTorch training.

Randomly initialized weights are expected. PASS means:
- static target preparation completes;
- the coverage crop planner works;
- losses are finite;
- geometry/spatial and partition gradients are nonzero;
- optimizer steps are not skipped;
- global_step advances 0 -> 1 -> 2;
- at least one trainable parameter changes;
- no CUDA OOM / autograd failure occurs.

Run
---
    modal run investigations/stirnet/12_spatial_training_sanity_modal.py

Optional frame:
    modal run investigations/stirnet/12_spatial_training_sanity_modal.py --frame 33

Outputs
-------
    stirnet-runs/stirnet/investigations/stage12_spatial_training_sanity/<timestamp>/
        summary.json
        gpu_profile.jsonl

No checkpoint is saved.
"""

import gc
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

import modal


# =============================================================================
# MODAL / REPOSITORY LAYOUT
# =============================================================================

app = modal.App("stirnet-stage12-spatial-training-sanity-v2")

data_volume = modal.Volume.from_name("stirnet-data")
runs_volume = modal.Volume.from_name("stirnet-runs")

REMOTE_REPO_ROOT = "/workspace/cell-tracking"
DATA_MOUNT = "/workspace/cell-tracking/data"
RUNS_MOUNT = "/workspace/cell-tracking/runs"

SOURCE_DIR = (
    "/workspace/cell-tracking/data/source/"
    "BlastoSPIM1_F22_030_034_source"
)
RUNS_PREFIX = (
    "/workspace/cell-tracking/runs/stirnet/investigations/"
    "stage12_spatial_training_sanity_v2"
)


def _resolve_local_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (
            (candidate / "learned").is_dir()
            and (candidate / "src").is_dir()
        ):
            return candidate

    remote = Path(REMOTE_REPO_ROOT)
    if remote.exists():
        return remote

    return Path.cwd()


LOCAL_REPO_ROOT = _resolve_local_repo_root()

# CuPy [ctk] is intentional: Modal has an NVIDIA driver but the image should
# not depend on a separately installed system CUDA toolkit just to JIT kernels.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.13.0",
        "numpy==2.4.6",
        "scipy==1.17.1",
        "scikit-image==0.26.0",
        "psutil>=6.0",
        "networkx>=3.0",
    )
    .workdir(REMOTE_REPO_ROOT)
    .add_local_dir(
        LOCAL_REPO_ROOT / "learned",
        remote_path="/workspace/cell-tracking/learned",
    )
    .add_local_dir(
        LOCAL_REPO_ROOT / "src",
        remote_path="/workspace/cell-tracking/src",
    )
)


# =============================================================================
# CONSTANTS
# =============================================================================

SPACING_ZYX_UM = (2.0, 0.208, 0.208)
DEFAULT_FRAME = 32
SEED = 230525

# Cost-aware. The full model trains only on a bounded crop here.
MODAL_GPU = "L40S"
MODAL_CPU = 8.0
MODAL_MEMORY_MB = 32_768
MODAL_TIMEOUT_SECONDS = 60 * 45

TRIM_EMPTY_ACQUISITION_BORDER = True
ROI_MARGIN_UM = 12.0

# Current production crop size. This is deliberately the crop path, not a
# full useful-ROI autograd pass.
TRAIN_CROP_SHAPE_ZYX = (32, 192, 192)
MIN_COMPLETE_CELLS = 10
CROPS_PER_STEP = 1

# Stage-12 v2 intentionally uses the exact SciPy reference target builder.
# The previous Stage-12 run showed that a large CuPy 3-D EDT can trigger a
# fatal CUDA illegal-address/Xid-31 event on L40S. Once that happens the
# CUDA context is unusable, so GPU EDT benchmarking must be isolated from
# the forward/backward sanity test.
GEOMETRY_TARGET_BACKEND = "scipy"
GEOMETRY_TARGET_GPU_MIN_VOXELS = 262_144


# =============================================================================
# SMALL HELPERS
# =============================================================================


def _sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_snapshot() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {}
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "allocated_mib": float(torch.cuda.memory_allocated() / 1024**2),
            "reserved_mib": float(torch.cuda.memory_reserved() / 1024**2),
            "peak_allocated_mib": float(
                torch.cuda.max_memory_allocated() / 1024**2
            ),
            "peak_reserved_mib": float(
                torch.cuda.max_memory_reserved() / 1024**2
            ),
            "physical_free_mib": float(free_bytes / 1024**2),
            "physical_total_mib": float(total_bytes / 1024**2),
        }
    except BaseException as error:
        return {
            "snapshot_error_type": type(error).__name__,
            "snapshot_error": str(error),
        }


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _one_physical_edt_marker_per_component(labels, spacing_um):
    import numpy as np
    from scipy import ndimage as ndi

    labels = np.asarray(labels)
    marker = np.zeros(labels.shape, dtype=np.float32)

    for instance_id, slc in enumerate(ndi.find_objects(labels), start=1):
        if slc is None:
            continue

        expanded = tuple(
            slice(
                max(0, axis.start - 1),
                min(labels.shape[d], axis.stop + 1),
            )
            for d, axis in enumerate(slc)
        )
        component = labels[expanded] == instance_id
        if not component.any():
            continue

        edt = ndi.distance_transform_edt(
            component,
            sampling=spacing_um,
        )
        local_position = np.unravel_index(np.argmax(edt), edt.shape)
        global_position = tuple(
            int(expanded[d].start) + int(local_position[d])
            for d in range(3)
        )
        marker[global_position] = 1.0

    return marker


def _source_roi(current_labels, spacing_um, margin_um: float):
    import numpy as np

    shape = np.asarray(current_labels.shape, dtype=np.int64)
    coords = np.where(np.asarray(current_labels) > 0)

    if not len(coords[0]):
        low = np.zeros(3, dtype=np.int64)
        high = shape.copy()
    else:
        low = np.asarray([axis.min() for axis in coords], dtype=np.int64)
        high = np.asarray(
            [axis.max() + 1 for axis in coords], dtype=np.int64
        )
        margin = np.ceil(
            float(margin_um)
            / np.asarray(spacing_um, dtype=np.float64)
        ).astype(np.int64)
        low = np.maximum(low - margin, 0)
        high = np.minimum(high + margin, shape)

    return tuple(
        slice(int(a), int(b)) for a, b in zip(low, high)
    )


def _gt_roi_coverage(gt_full, roi) -> dict[str, Any]:
    import numpy as np

    gt_full = np.asarray(gt_full)
    gt_crop = np.asarray(gt_full[roi])

    full_ids, full_counts = np.unique(
        gt_full[gt_full > 0], return_counts=True
    )
    crop_ids, crop_counts = np.unique(
        gt_crop[gt_crop > 0], return_counts=True
    )
    crop_map = {
        int(label_id): int(count)
        for label_id, count in zip(crop_ids, crop_counts)
    }

    complete = 0
    partial: list[int] = []
    missing: list[int] = []
    for label_id, count in zip(full_ids, full_counts):
        label_id = int(label_id)
        full_count = int(count)
        seen = crop_map.get(label_id, 0)
        if seen == full_count:
            complete += 1
        elif seen == 0:
            missing.append(label_id)
        else:
            partial.append(label_id)

    full_positive = int((gt_full > 0).sum())
    crop_positive = int((gt_crop > 0).sum())
    return {
        "full_gt_cell_count": int(len(full_ids)),
        "fully_contained_gt_cell_count": int(complete),
        "partial_gt_ids": partial,
        "missing_gt_ids": missing,
        "positive_voxel_coverage": (
            1.0
            if full_positive == 0
            else float(crop_positive / full_positive)
        ),
    }


@dataclass
class PreparedFrame:
    clean_batch: dict
    raw_norm_roi: Any
    current_roi: Any
    gt_roi: Any
    dref_um: float
    sample_report: dict[str, Any]


# =============================================================================
# FRESH SOURCE PREPARATION
# =============================================================================


def _prepare_frame(frame: int) -> PreparedFrame:
    import numpy as np
    import torch
    from scipy import ndimage as ndi

    if REMOTE_REPO_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_REPO_ROOT)

    from learned.stirnet.data.sample_builder import (
        build_spatial_channels,
        robust_normalize,
    )
    from learned.stirnet.data.targets import estimate_model_dref_um

    preprocess_volume = import_module(
        "src.01_preprocessing.pipeline"
    ).preprocess_volume
    PreprocessingConfig = import_module(
        "src.01_preprocessing.config"
    ).PreprocessingConfig
    create_binary_mask = import_module(
        "src.02_masking.pipeline"
    ).create_binary_mask
    MaskingConfig = import_module(
        "src.02_masking.config"
    ).MaskingConfig

    source_dir = Path(SOURCE_DIR)
    raw_path = source_dir / f"F22_{frame:03d}_image_0001.npy"
    gt_path = source_dir / f"F22_{frame:03d}_masks_0001.npy"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    timings: dict[str, float] = {}

    started = time.perf_counter()
    raw = np.load(raw_path)
    gt = np.load(gt_path)
    timings["load_original_arrays"] = time.perf_counter() - started

    full_shape = tuple(int(v) for v in raw.shape)
    if tuple(gt.shape) != full_shape:
        raise ValueError(
            f"raw/GT shape mismatch: raw={raw.shape}, gt={gt.shape}"
        )

    preprocessing_config = PreprocessingConfig(
        low_percentile=1.0,
        high_percentile=99.5,
        denoise_sigma_um=0.8,
        background_sigma_um=4.0,
        voxel_size_zyx_um=SPACING_ZYX_UM,
    )

    started = time.perf_counter()
    processed = preprocess_volume(
        raw,
        config=preprocessing_config,
        return_diagnostics=False,
    )
    timings["canonical_preprocessing"] = time.perf_counter() - started

    started = time.perf_counter()
    binary = create_binary_mask(
        processed,
        config=MaskingConfig(),
        return_diagnostics=False,
    )
    timings["canonical_otsu_masking"] = time.perf_counter() - started

    connectivity_6 = ndi.generate_binary_structure(3, 1)
    started = time.perf_counter()
    current_full, source_component_count = ndi.label(
        binary,
        structure=connectivity_6,
    )
    current_full = current_full.astype(np.int32, copy=False)
    timings["six_connected_components"] = (
        time.perf_counter() - started
    )

    started = time.perf_counter()
    dref_um = float(
        estimate_model_dref_um(current_full, SPACING_ZYX_UM)
    )
    timings["model_dref_from_full_current"] = (
        time.perf_counter() - started
    )

    started = time.perf_counter()
    marker_full = _one_physical_edt_marker_per_component(
        current_full,
        SPACING_ZYX_UM,
    )
    timings["physical_edt_marker"] = time.perf_counter() - started

    started = time.perf_counter()
    raw_norm_full = robust_normalize(raw)
    timings["full_raw_robust_normalization"] = (
        time.perf_counter() - started
    )

    if TRIM_EMPTY_ACQUISITION_BORDER:
        roi = _source_roi(
            current_full,
            SPACING_ZYX_UM,
            ROI_MARGIN_UM,
        )
    else:
        roi = tuple(slice(0, int(v)) for v in full_shape)

    coverage = _gt_roi_coverage(gt, roi)

    current = np.asarray(current_full[roi]).astype(
        np.int32, copy=True
    )
    gt_roi = np.asarray(gt[roi]).astype(np.int64, copy=True)
    raw_norm = np.asarray(raw_norm_full[roi]).astype(
        np.float32, copy=True
    )
    marker = np.asarray(marker_full[roi]).astype(
        np.float32, copy=True
    )

    started = time.perf_counter()
    spatial = build_spatial_channels(
        raw_norm,
        current,
        SPACING_ZYX_UM,
        dref_um,
        marker,
    )
    timings["production_build_spatial_channels"] = (
        time.perf_counter() - started
    )

    clean_batch = {
        "spatial_inputs": torch.from_numpy(spatial).unsqueeze(0),
        "instance_labels": torch.from_numpy(current).long().unsqueeze(0),
        "spacing_um": torch.tensor(
            [SPACING_ZYX_UM], dtype=torch.float32
        ),
        "dref_um": torch.tensor([dref_um], dtype=torch.float32),
        "targets": [
            {"label_map": torch.from_numpy(gt_roi).long()}
        ],
    }

    roi_shape = tuple(int(v) for v in gt_roi.shape)
    sample_report = {
        "frame": int(frame),
        "raw_path": str(raw_path),
        "gt_path": str(gt_path),
        "full_shape_zyx": list(full_shape),
        "roi_shape_zyx": list(roi_shape),
        "full_voxels": int(math.prod(full_shape)),
        "roi_voxels": int(math.prod(roi_shape)),
        "voxel_reduction_fraction": float(
            1.0 - math.prod(roi_shape) / max(math.prod(full_shape), 1)
        ),
        "spacing_zyx_um": list(SPACING_ZYX_UM),
        "dref_um": float(dref_um),
        "source_component_count": int(source_component_count),
        "roi_slices": [
            [int(axis.start), int(axis.stop)] for axis in roi
        ],
        "gt_roi_coverage": coverage,
        "timings_seconds": timings,
    }

    del (
        processed,
        binary,
        current_full,
        marker_full,
        raw_norm_full,
        marker,
        spatial,
        raw,
        gt,
    )
    gc.collect()

    return PreparedFrame(
        clean_batch=clean_batch,
        raw_norm_roi=raw_norm,
        current_roi=current,
        gt_roi=gt_roi,
        dref_um=dref_um,
        sample_report=sample_report,
    )


# =============================================================================
# COVERAGE CROP REPORTING + SYNTHETIC SOURCE CORRUPTION
# =============================================================================


def _crop_record_to_dict(record) -> dict[str, Any]:
    return {
        "slices_zyx": [
            [int(axis.start), int(axis.stop)]
            for axis in record.slices_zyx
        ],
        "shape_zyx": list(record.shape_zyx),
        "complete_cell_ids": list(record.complete_cell_ids),
        "complete_cell_count": int(len(record.complete_cell_ids)),
        "partial_cell_ids": list(record.partial_cell_ids),
        "partial_cell_count": int(len(record.partial_cell_ids)),
        "true_boundary_cell_ids": list(
            record.true_boundary_cell_ids
        ),
        "true_boundary_cell_count": int(
            len(record.true_boundary_cell_ids)
        ),
    }


def _nearest_source_pair_in_crop(
    labels,
    crop_slices,
    spacing_um,
):
    import numpy as np

    crop = np.asarray(labels[crop_slices])
    ids, counts = np.unique(crop[crop > 0], return_counts=True)
    good = [
        int(label_id)
        for label_id, count in zip(ids, counts)
        if int(count) >= 8
    ]
    if len(good) < 2:
        return None

    centers = []
    for label_id in good:
        coords = np.argwhere(crop == label_id)
        centers.append(
            coords.mean(axis=0)
            * np.asarray(spacing_um, dtype=np.float64)
        )
    centers = np.asarray(centers, dtype=np.float64)

    best = None
    for i in range(len(good)):
        for j in range(i + 1, len(good)):
            distance = float(
                np.linalg.norm(centers[i] - centers[j])
            )
            if best is None or distance < best[0]:
                best = (distance, good[i], good[j])
    return best


def _synthetic_connected_merge(
    current_labels,
    crop_slices,
    spacing_um,
):
    """Join two nearby source components with a thin 6-connected bridge."""
    import numpy as np
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree

    labels = np.asarray(current_labels)
    pair = _nearest_source_pair_in_crop(
        labels,
        crop_slices,
        spacing_um,
    )
    if pair is None:
        return None, {
            "kind": "none",
            "reason": "fewer_than_two_source_components_in_step_b_crop",
        }

    centroid_distance_um, label_a, label_b = pair
    crop = labels[crop_slices]
    coords_a_local = np.argwhere(crop == label_a)
    coords_b_local = np.argwhere(crop == label_b)
    if not len(coords_a_local) or not len(coords_b_local):
        return None, {
            "kind": "none",
            "reason": "selected_source_component_missing",
        }

    spacing = np.asarray(spacing_um, dtype=np.float64)
    tree_b = cKDTree(coords_b_local.astype(np.float64) * spacing[None])
    distances, indices = tree_b.query(
        coords_a_local.astype(np.float64) * spacing[None],
        k=1,
    )
    row_a = int(np.argmin(distances))
    row_b = int(indices[row_a])
    point_a_local = coords_a_local[row_a]
    point_b_local = coords_b_local[row_b]
    closest_distance_um = float(distances[row_a])

    crop_low = np.asarray(
        [int(axis.start) for axis in crop_slices],
        dtype=np.int64,
    )
    point_a = point_a_local + crop_low
    point_b = point_b_local + crop_low

    delta = point_b - point_a
    samples = int(np.max(np.abs(delta))) + 1
    line = np.rint(
        np.linspace(point_a, point_b, max(samples, 2))
    ).astype(np.int64)
    line = np.clip(
        line,
        np.zeros(3, dtype=np.int64),
        np.asarray(labels.shape, dtype=np.int64) - 1,
    )

    foreground = labels > 0
    bridge = np.zeros_like(foreground, dtype=bool)
    bridge[tuple(line.T)] = True

    # A one-voxel 6-connected thickening avoids diagonal-only line gaps while
    # keeping the corruption intentionally local.
    structure = ndi.generate_binary_structure(3, 1)
    bridge = ndi.binary_dilation(
        bridge,
        structure=structure,
        iterations=1,
    )

    corrupted_foreground = foreground | bridge
    corrupted, component_count = ndi.label(
        corrupted_foreground,
        structure=structure,
    )
    corrupted = corrupted.astype(np.int32, copy=False)

    before_a = int(np.unique(labels[labels == label_a]).size > 0)
    before_b = int(np.unique(labels[labels == label_b]).size > 0)
    # The two points must now belong to the same connected component.
    merged_component_a = int(corrupted[tuple(point_a)])
    merged_component_b = int(corrupted[tuple(point_b)])
    connected = (
        merged_component_a > 0
        and merged_component_a == merged_component_b
    )

    metadata = {
        "kind": "connected_merge",
        "source_label_a": int(label_a),
        "source_label_b": int(label_b),
        "centroid_distance_um": float(centroid_distance_um),
        "closest_surface_distance_um": float(closest_distance_um),
        "point_a_zyx": point_a.tolist(),
        "point_b_zyx": point_b.tolist(),
        "bridge_voxels": int(bridge.sum()),
        "result_component_count": int(component_count),
        "selected_labels_existed": bool(before_a and before_b),
        "connected_after_corruption": bool(connected),
    }
    if not connected:
        return None, {
            **metadata,
            "kind": "none",
            "reason": "bridge_failed_to_connect_selected_components",
        }
    return corrupted, metadata


def _fallback_remove_source(
    current_labels,
    crop_slices,
):
    import numpy as np
    from scipy import ndimage as ndi

    labels = np.asarray(current_labels)
    crop = labels[crop_slices]
    ids, counts = np.unique(crop[crop > 0], return_counts=True)
    if not len(ids):
        return labels.copy(), {
            "kind": "none",
            "reason": "no_source_component_in_step_b_crop",
        }

    remove_id = int(ids[int(np.argmax(counts))])
    foreground = labels > 0
    foreground[labels == remove_id] = False
    structure = ndi.generate_binary_structure(3, 1)
    corrupted, component_count = ndi.label(
        foreground,
        structure=structure,
    )
    return corrupted.astype(np.int32, copy=False), {
        "kind": "remove_source_component",
        "removed_source_label": int(remove_id),
        "removed_voxels": int(np.count_nonzero(labels == remove_id)),
        "result_component_count": int(component_count),
    }


def _make_corrupted_batch(
    prepared: PreparedFrame,
    crop_record,
):
    import numpy as np
    import torch

    from learned.stirnet.data.sample_builder import build_spatial_channels

    corrupted, metadata = _synthetic_connected_merge(
        prepared.current_roi,
        crop_record.slices_zyx,
        SPACING_ZYX_UM,
    )
    if corrupted is None:
        corrupted, metadata = _fallback_remove_source(
            prepared.current_roi,
            crop_record.slices_zyx,
        )

    started = time.perf_counter()
    marker = _one_physical_edt_marker_per_component(
        corrupted,
        SPACING_ZYX_UM,
    )
    marker_seconds = time.perf_counter() - started

    started = time.perf_counter()
    spatial = build_spatial_channels(
        prepared.raw_norm_roi,
        corrupted,
        SPACING_ZYX_UM,
        prepared.dref_um,  # fixed baseline scale by design
        marker,
    )
    channel_seconds = time.perf_counter() - started

    batch = {
        "spatial_inputs": torch.from_numpy(
            np.asarray(spatial, dtype=np.float32)
        ).unsqueeze(0),
        "instance_labels": torch.from_numpy(
            np.asarray(corrupted, dtype=np.int64)
        ).long().unsqueeze(0),
        "spacing_um": torch.tensor(
            [SPACING_ZYX_UM], dtype=torch.float32
        ),
        "dref_um": torch.tensor(
            [prepared.dref_um], dtype=torch.float32
        ),
        "targets": [
            {
                "label_map": torch.from_numpy(
                    np.asarray(prepared.gt_roi, dtype=np.int64)
                ).long()
            }
        ],
    }
    metadata = {
        **metadata,
        "marker_rebuild_seconds": float(marker_seconds),
        "spatial_channel_rebuild_seconds": float(channel_seconds),
        "dref_reused_from_clean_source": float(prepared.dref_um),
    }
    return batch, metadata


# =============================================================================
# METRIC / PARAMETER REPORTING
# =============================================================================


def _first_trainable_parameter(model):
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.numel():
            return name, parameter
    raise RuntimeError("model has no trainable parameters")


def _parameter_snapshot(parameter):
    return parameter.detach().float().cpu().clone()


def _parameter_change(before, parameter) -> dict[str, float]:
    import torch

    after = parameter.detach().float().cpu()
    difference = after - before
    return {
        "max_abs_change": float(difference.abs().max()),
        "l2_change": float(torch.linalg.vector_norm(difference)),
    }


def _finite_metric(metrics: dict[str, float], key: str) -> bool:
    import math

    return key in metrics and math.isfinite(float(metrics[key]))


def _metric_subset(metrics: dict[str, float]) -> dict[str, float]:
    wanted = (
        "loss",
        "crop_phase_a_loss",
        "crop_geometry_loss",
        "crop_spatial_rag_bce",
        "crop_count",
        "grad_geometry_spatial",
        "grad_partition",
        "grad_instances",
        "grad_temporal",
        "grad_refinement",
        "grad_norm",
        "optimizer_step_skipped",
        "forward_seconds",
        "target_seconds",
        "backward_seconds",
        "total_step_seconds",
        "peak_allocated_mb",
        "peak_reserved_mb",
        "phase_a_crop_forward_seconds",
        "phase_a_crop_target_seconds",
        "phase_a_crop_backward_seconds",
    )
    result = {
        key: float(metrics[key])
        for key in wanted
        if key in metrics
    }

    # Preserve profiler timings that matter for this investigation without
    # flooding summary.json with every internal field.
    for key, value in metrics.items():
        if (
            "geometry_targets_corrective_separator" in key
            or "optimizer_step" in key
            or "crop_select" in key
            or "crop_prepare" in key
            or "initial_watershed_marker" in key
            or "watershed" in key
            or "supervoxel" in key
            or "rag" in key
        ):
            if isinstance(value, (int, float)):
                result[key] = float(value)
    return result


# =============================================================================
# MODAL REMOTE RUN
# =============================================================================


@app.function(
    image=image,
    gpu=MODAL_GPU,
    cpu=MODAL_CPU,
    memory=MODAL_MEMORY_MB,
    timeout=MODAL_TIMEOUT_SECONDS,
    volumes={
        DATA_MOUNT: data_volume,
        RUNS_MOUNT: runs_volume,
    },
)
def run_stage12(frame: int = DEFAULT_FRAME) -> dict[str, Any]:
    import numpy as np
    import torch

    if REMOTE_REPO_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_REPO_ROOT)

    from learned.stirnet import StirNet, StirNetConfig
    from learned.stirnet.training.config import TrainingConfig
    from learned.stirnet.training.coverage_crops import (
        build_coverage_crop_manifest,
    )
    from learned.stirnet.training.trainer import Trainer

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(RUNS_PREFIX) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "gpu_profile.jsonl"
    summary_path = output_dir / "summary.json"

    summary: dict[str, Any] = {
        "stage": 12,
        "status": "running",
        "run_id": run_id,
        "frame": int(frame),
        "seed": int(SEED),
        "modal": {
            "gpu": MODAL_GPU,
            "cpu": MODAL_CPU,
            "memory_mb": MODAL_MEMORY_MB,
        },
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Stage 12 requires CUDA")

        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA device: {device_name}", flush=True)

        # ------------------------------------------------------------------
        # A. Fresh source/current sample from original data.
        # ------------------------------------------------------------------
        print("\n[A] Building fresh source sample...", flush=True)
        started = time.perf_counter()
        prepared = _prepare_frame(int(frame))
        source_prepare_seconds = time.perf_counter() - started
        summary["sample"] = prepared.sample_report
        summary["source_prepare_seconds"] = float(source_prepare_seconds)

        print(
            "  full shape : "
            f"{prepared.sample_report['full_shape_zyx']}",
            flush=True,
        )
        print(
            "  ROI shape  : "
            f"{prepared.sample_report['roi_shape_zyx']}",
            flush=True,
        )
        print(
            "  dref       : "
            f"{prepared.dref_um:.4f} um",
            flush=True,
        )

        # ------------------------------------------------------------------
        # B. Full production-width model + crop-training configuration.
        # ------------------------------------------------------------------
        print("\n[B] Initializing FULL STIR-Net...", flush=True)
        model_cfg = StirNetConfig()

        # Explicitly lock the intended production choice even though it is now
        # the repository default after Stage 11.
        model_cfg.partition.watershed_backend = "fast"
        model_cfg.validate()

        train_cfg = TrainingConfig()
        train_cfg.amp_dtype = "bf16"
        train_cfg.geometry_target_backend = GEOMETRY_TARGET_BACKEND
        train_cfg.geometry_target_gpu_min_voxels = (
            GEOMETRY_TARGET_GPU_MIN_VOXELS
        )
        train_cfg.curriculum.fixed_stage = "spatial_partition"
        train_cfg.curriculum.spatial_partition_crop_enabled = True
        train_cfg.curriculum.geometry_bootstrap_crop_enabled = True
        train_cfg.curriculum.refinement_crop_shape_zyx = (
            TRAIN_CROP_SHAPE_ZYX
        )
        train_cfg.curriculum.refinement_crops_per_step = CROPS_PER_STEP
        train_cfg.curriculum.refinement_crop_sampling = "coverage"
        train_cfg.curriculum.refinement_crop_min_complete_cells = (
            MIN_COMPLETE_CELLS
        )
        train_cfg.curriculum.refinement_crop_views_per_cell = 1
        train_cfg.profile_memory = True
        train_cfg.memory_profile_path = str(profile_path)
        train_cfg.validate()

        model = StirNet(model_cfg)
        trainer = Trainer(
            model,
            train_cfg,
            device=torch.device("cuda"),
        )

        parameter_name, tracked_parameter = _first_trainable_parameter(
            model
        )

        summary["configuration"] = {
            "model": "full_repository_default",
            "watershed_backend": model_cfg.partition.watershed_backend,
            "evidence_stem_channels": model_cfg.evidence.stem_channels,
            "spatial_channels": list(model_cfg.spatial.channels),
            "spatial_blocks_per_level": model_cfg.spatial.blocks_per_level,
            "spatial_activation_checkpointing": (
                model_cfg.spatial.activation_checkpointing
            ),
            "geometry_hidden_channels": model_cfg.geometry.hidden_channels,
            "geometry_residual_blocks": model_cfg.geometry.residual_blocks,
            "rag_hidden_dim": model_cfg.partition.rag_hidden_dim,
            "instance_d_model": model_cfg.instances.d_model,
            "crop_shape_zyx": list(TRAIN_CROP_SHAPE_ZYX),
            "crop_sampling": "coverage",
            "min_complete_cells": MIN_COMPLETE_CELLS,
            "geometry_target_backend": GEOMETRY_TARGET_BACKEND,
            "gpu_edt_deliberately_disabled": True,
            "geometry_target_gpu_min_voxels": (
                GEOMETRY_TARGET_GPU_MIN_VOXELS
            ),
            "amp_dtype": train_cfg.amp_dtype,
        }
        summary["tracked_parameter_name"] = parameter_name
        summary["gpu_edt_deliberately_disabled"] = True

        # ------------------------------------------------------------------
        # C. Build coverage manifest now so the exact step-A/B crops are
        #    visible in the report. Trainer will deterministically reconstruct
        #    the same manifest.
        # ------------------------------------------------------------------
        gt_labels = prepared.clean_batch["targets"][0]["label_map"][None]
        manifest = build_coverage_crop_manifest(
            gt_labels,
            crop_shape_zyx=TRAIN_CROP_SHAPE_ZYX,
            min_complete_cells=MIN_COMPLETE_CELLS,
            views_per_cell=1,
        )
        records = manifest.records[0]
        if not records:
            raise RuntimeError("coverage crop manifest is empty")

        step_a_record = records[0]
        step_b_record = records[1 % len(records)]
        summary["coverage_manifest"] = {
            "record_count": int(len(records)),
            "uncoverable_cell_ids": [
                list(row) for row in manifest.uncoverable_cell_ids
            ],
            "step_a_crop": _crop_record_to_dict(step_a_record),
            "step_b_crop": _crop_record_to_dict(step_b_record),
        }

        print(
            "  coverage crops : "
            f"{len(records)} | "
            f"step A complete={len(step_a_record.complete_cell_ids)} | "
            f"step B complete={len(step_b_record.complete_cell_ids)}",
            flush=True,
        )

        # ------------------------------------------------------------------
        # D. STATIC GT geometry exactly once, through the production Trainer
        #    helper. This is the expensive part we intend to amortize.
        # ------------------------------------------------------------------
        print(
            "\n[C] Building reusable STATIC GT geometry "
            "(exact SciPy reference EDT)...",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _sync_cuda()
        static_before_gpu = _gpu_snapshot()
        started = time.perf_counter()
        static_targets = trainer.prepare_static_geometry_targets(
            prepared.clean_batch
        )
        _sync_cuda()
        static_seconds = time.perf_counter() - started
        static_after_gpu = _gpu_snapshot()

        static_bytes = sum(
            int(value.numel() * value.element_size())
            for value in static_targets.__dict__.values()
        )
        summary["static_geometry"] = {
            "seconds": float(static_seconds),
            "cpu_storage_mib": float(static_bytes / 1024**2),
            "gpu_before": static_before_gpu,
            "gpu_after": static_after_gpu,
            "separator_max": float(static_targets.separator.max()),
        }
        print(
            f"  static geometry: {static_seconds:.3f}s | "
            f"stored {static_bytes / 1024**2:.1f} MiB CPU",
            flush=True,
        )

        # ------------------------------------------------------------------
        # E. Optimizer step A: clean/current source.
        # ------------------------------------------------------------------
        print(
            "\n[D] Step A: CLEAN source -> forward/backward/optimizer...",
            flush=True,
        )
        parameter_before_a = _parameter_snapshot(tracked_parameter)
        global_before_a = int(trainer.global_step)

        _sync_cuda()
        started = time.perf_counter()
        metrics_a = trainer.train_step(
            prepared.clean_batch,
            precomputed_static_geometry_targets=static_targets,
        )
        _sync_cuda()
        external_seconds_a = time.perf_counter() - started

        global_after_a = int(trainer.global_step)
        parameter_change_a = _parameter_change(
            parameter_before_a, tracked_parameter
        )

        step_a = {
            "global_step_before": global_before_a,
            "global_step_after": global_after_a,
            "external_wall_seconds": float(external_seconds_a),
            "metrics": _metric_subset(metrics_a),
            "tracked_parameter_change": parameter_change_a,
            "gpu_after_step": _gpu_snapshot(),
        }
        summary["step_a_clean"] = step_a

        print(
            f"  loss={metrics_a.get('loss', float('nan')):.6f} | "
            f"forward={metrics_a.get('forward_seconds', float('nan')):.3f}s | "
            f"backward={metrics_a.get('backward_seconds', float('nan')):.3f}s | "
            f"total={metrics_a.get('total_step_seconds', float('nan')):.3f}s",
            flush=True,
        )
        print(
            f"  grad geometry={metrics_a.get('grad_geometry_spatial', 0):.6g} | "
            f"partition={metrics_a.get('grad_partition', 0):.6g} | "
            f"parameter max Δ={parameter_change_a['max_abs_change']:.3e}",
            flush=True,
        )

        # ------------------------------------------------------------------
        # F. Build a second source state. GT + static targets are untouched.
        # ------------------------------------------------------------------
        print(
            "\n[E] Building synthetic source corruption for Step B...",
            flush=True,
        )
        started = time.perf_counter()
        corrupted_batch, corruption = _make_corrupted_batch(
            prepared,
            step_b_record,
        )
        corruption["total_build_seconds"] = float(
            time.perf_counter() - started
        )
        summary["synthetic_corruption"] = corruption
        print(
            f"  corruption={corruption.get('kind')} | "
            f"build={corruption['total_build_seconds']:.3f}s",
            flush=True,
        )

        # ------------------------------------------------------------------
        # G. Optimizer step B: same static GT geometry, changed source state.
        #    Trainer rebuilds only the corrective separator before crop training.
        # ------------------------------------------------------------------
        print(
            "\n[F] Step B: CORRUPTED source, reusing STATIC targets...",
            flush=True,
        )
        parameter_before_b = _parameter_snapshot(tracked_parameter)
        global_before_b = int(trainer.global_step)

        _sync_cuda()
        started = time.perf_counter()
        metrics_b = trainer.train_step(
            corrupted_batch,
            precomputed_static_geometry_targets=static_targets,
        )
        _sync_cuda()
        external_seconds_b = time.perf_counter() - started

        global_after_b = int(trainer.global_step)
        parameter_change_b = _parameter_change(
            parameter_before_b, tracked_parameter
        )

        step_b = {
            "global_step_before": global_before_b,
            "global_step_after": global_after_b,
            "external_wall_seconds": float(external_seconds_b),
            "metrics": _metric_subset(metrics_b),
            "tracked_parameter_change": parameter_change_b,
            "gpu_after_step": _gpu_snapshot(),
        }
        summary["step_b_corrupted"] = step_b

        print(
            f"  loss={metrics_b.get('loss', float('nan')):.6f} | "
            f"forward={metrics_b.get('forward_seconds', float('nan')):.3f}s | "
            f"backward={metrics_b.get('backward_seconds', float('nan')):.3f}s | "
            f"total={metrics_b.get('total_step_seconds', float('nan')):.3f}s",
            flush=True,
        )
        print(
            f"  grad geometry={metrics_b.get('grad_geometry_spatial', 0):.6g} | "
            f"partition={metrics_b.get('grad_partition', 0):.6g} | "
            f"parameter max Δ={parameter_change_b['max_abs_change']:.3e}",
            flush=True,
        )

        # ------------------------------------------------------------------
        # H. Acceptance.
        # ------------------------------------------------------------------
        reasons: list[str] = []

        if not _finite_metric(metrics_a, "loss"):
            reasons.append("step A loss is non-finite")
        if not _finite_metric(metrics_b, "loss"):
            reasons.append("step B loss is non-finite")

        if float(metrics_a.get("grad_geometry_spatial", 0.0)) <= 0:
            reasons.append("step A geometry/spatial gradient is zero")
        if float(metrics_b.get("grad_geometry_spatial", 0.0)) <= 0:
            reasons.append("step B geometry/spatial gradient is zero")
        if float(metrics_a.get("grad_partition", 0.0)) <= 0:
            reasons.append("step A partition gradient is zero")
        if float(metrics_b.get("grad_partition", 0.0)) <= 0:
            reasons.append("step B partition gradient is zero")

        if float(metrics_a.get("optimizer_step_skipped", 1.0)) != 0.0:
            reasons.append("step A optimizer step was skipped")
        if float(metrics_b.get("optimizer_step_skipped", 1.0)) != 0.0:
            reasons.append("step B optimizer step was skipped")

        if (global_before_a, global_after_a) != (0, 1):
            reasons.append(
                f"step A global_step was {global_before_a}->{global_after_a}"
            )
        if (global_before_b, global_after_b) != (1, 2):
            reasons.append(
                f"step B global_step was {global_before_b}->{global_after_b}"
            )

        if parameter_change_a["max_abs_change"] <= 0:
            reasons.append("tracked parameter did not change in step A")
        if parameter_change_b["max_abs_change"] <= 0:
            reasons.append("tracked parameter did not change in step B")

        if len(step_a_record.complete_cell_ids) == 0:
            reasons.append("step A coverage crop has no complete GT cells")
        if len(step_b_record.complete_cell_ids) == 0:
            reasons.append("step B coverage crop has no complete GT cells")

        summary["acceptance"] = {
            "passed": not reasons,
            "reasons": reasons,
            "expected_global_step_final": 2,
            "actual_global_step_final": int(trainer.global_step),
        }
        summary["status"] = "PASS" if not reasons else "FAIL"

        _json_dump(summary_path, summary)
        runs_volume.commit()

        print("\n" + "=" * 78, flush=True)
        print(
            f"STAGE 12 RESULT: {summary['status']}",
            flush=True,
        )
        print(
            f"Static GT preprocessing : {static_seconds:.3f}s",
            flush=True,
        )
        print(
            "Step A total             : "
            f"{metrics_a.get('total_step_seconds', float('nan')):.3f}s",
            flush=True,
        )
        print(
            "Step B total             : "
            f"{metrics_b.get('total_step_seconds', float('nan')):.3f}s",
            flush=True,
        )
        print(
            "Step A peak allocated    : "
            f"{metrics_a.get('peak_allocated_mb', float('nan')) / 1024:.2f} GiB",
            flush=True,
        )
        print(
            "Step B peak allocated    : "
            f"{metrics_b.get('peak_allocated_mb', float('nan')) / 1024:.2f} GiB",
            flush=True,
        )
        print(
            f"Summary                  : {summary_path}",
            flush=True,
        )
        if reasons:
            for reason in reasons:
                print(f"  FAIL: {reason}", flush=True)
        print("=" * 78, flush=True)

        return summary

    except BaseException as error:
        summary["status"] = "ERROR"
        summary["error_type"] = type(error).__name__
        summary["error"] = str(error)
        summary["traceback"] = traceback.format_exc()
        summary["gpu_at_error"] = _gpu_snapshot()
        try:
            _json_dump(summary_path, summary)
            runs_volume.commit()
        except Exception:
            pass
        print(summary["traceback"], flush=True)
        raise


# =============================================================================
# LOCAL ENTRYPOINT
# =============================================================================


@app.local_entrypoint()
def main(frame: int = DEFAULT_FRAME):
    result = run_stage12.remote(int(frame))

    print("\nRemote Stage-12 summary")
    print("-----------------------")
    print(f"status       : {result.get('status')}")
    print(f"frame        : {result.get('frame')}")
    print(
        "static GT    : "
        f"{result.get('static_geometry', {}).get('seconds', float('nan')):.3f}s"
    )

    for label, key in (
        ("step A", "step_a_clean"),
        ("step B", "step_b_corrupted"),
    ):
        row = result.get(key, {})
        metrics = row.get("metrics", {})
        if row:
            print(
                f"{label:<12}: "
                f"loss={metrics.get('loss', float('nan')):.6f} | "
                f"total={metrics.get('total_step_seconds', float('nan')):.3f}s | "
                f"peak={metrics.get('peak_allocated_mb', float('nan')) / 1024:.2f} GiB"
            )

    acceptance = result.get("acceptance", {})
    if acceptance and not acceptance.get("passed", False):
        print("failures:")
        for reason in acceptance.get("reasons", []):
            print(f"  - {reason}")
