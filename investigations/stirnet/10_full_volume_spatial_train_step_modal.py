from __future__ import annotations

"""
Stage 10 — fresh spatial preprocessing + Experiment-31 reduced STIR-Net
full useful-ROI forward/backward on Modal.

This diagnostic answers two separate questions:

1. Can we reconstruct one training sample from ORIGINAL Modal source data
   without using any previous STIR-Net preprocessing/debug cache?
2. With the SAME reduced model configuration used by
   experiments/stirnet/31_spatial_first_overfit.py, can one biologically useful
   all-cell ROI complete the current production spatial_partition training step
   on an L40S?

Important
---------
The network configuration below is intentionally copied from Experiment 31:

    evidence stem          12
    prior gate hidden      16
    spatial channels       (12, 24, 48, 96)
    blocks per level       1
    acquisition dim        32
    geometry hidden        32
    geometry residuals     2
    RAG node feat          16
    RAG hidden             48
    RAG layers             2
    instance d_model       64
    history hidden         16
    temporal d_model       64
    temporal graph hidden  128
    refinement hidden      32

Only the spatial curriculum stage executes here, but keeping the full reduced
configuration identical means the model architecture matches Experiment 31.

Data path
---------
Only these original source files are read for the selected frame:

    data/source/BlastoSPIM1_F22_030_034_source/
        F22_032_image_0001.npy
        F22_032_masks_0001.npy

The current/source segmentation is regenerated from the raw image:

    canonical preprocessing
      -> canonical Otsu mask
      -> 6-connected components
      -> physical EDT marker
      -> current-segmentation dref
      -> five STIR-Net spatial channels

Then an OPTIONAL deterministic acquisition trim removes empty outer space using
ONLY the regenerated current/source segmentation, never GT:

    bbox(current > 0) + ROI_MARGIN_UM physical margin

GT is used only afterward for supervision and to report whether the source-only
ROI accidentally clipped any GT content. If no empty acquisition border exists,
the ROI naturally remains close to the full frame.

Geometry targets are built fresh on CPU through the current production
StirNetCriterion/Trainer path and timed separately. The GPU train-step timing
therefore measures the same online spatial model step we care about, while the
one-time deterministic preparation cost is also visible.

No STIR-Net prepared/debug cache is read:
    no first_overfit stirnet_source channels
    no debug_crop.pt
    no temporal cache
    no Stage-01 predictions
    no Stage-08/09 controlled artifacts

Run
---
    modal run investigations/stirnet/10_full_volume_spatial_train_step_modal.py

Optional source frame:
    modal run investigations/stirnet/10_full_volume_spatial_train_step_modal.py --frame 33

Outputs
-------
Compact reports only:
    stirnet-runs/stirnet/investigations/stage10_reduced_spatial/<timestamp>/
        summary.json
        gpu_profile.jsonl

No preprocessing cache, dense prediction, or model checkpoint is persisted.
"""

import gc
import json
import os
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

import modal


# =============================================================================
# MODAL / REPOSITORY LAYOUT
# =============================================================================

app = modal.App("stirnet-stage10-reduced-spatial")

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
    "stage10_reduced_spatial"
)


def _resolve_local_repo_root() -> Path:
    """Work during Windows submission and Modal remote hydration."""
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (
            (candidate / "learned").is_dir()
            and (candidate / "src").is_dir()
        ):
            return candidate

    # During remote hydration the entrypoint itself may live under /root, while
    # the repository content is mounted at the known workdir.
    remote = Path(REMOTE_REPO_ROOT)
    if remote.exists():
        return remote

    return Path.cwd()


LOCAL_REPO_ROOT = _resolve_local_repo_root()

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
# TEST CONSTANTS
# =============================================================================

SPACING_ZYX_UM = (2.0, 0.208, 0.208)
DEFAULT_FRAME = 32
SEED = 200527

# Cost-aware first attempt, as requested.
MODAL_GPU = "L40S"
MODAL_CPU = 8.0
MODAL_MEMORY_MB = 32_768
MODAL_TIMEOUT_SECONDS = 60 * 60

# Match Experiment 31's physical all-cell margin, but define the ROI from the
# CURRENT/SOURCE segmentation only so GT does not determine network extent.
TRIM_EMPTY_ACQUISITION_BORDER = True
ROI_MARGIN_UM = 12.0


# =============================================================================
# CPU PROFILING
# =============================================================================


def _rss_gib() -> float:
    import psutil

    return (
        psutil.Process(os.getpid()).memory_info().rss
        / 1024**3
    )


def _max_rss_gib() -> float:
    # POSIX-only: import lazily because `modal run` first imports this file on
    # the user's Windows machine.
    import resource

    return (
        float(
            resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
        )
        / 1024**2
    )


@dataclass
class CpuProfileRow:
    name: str
    elapsed_seconds: float
    rss_gib_before: float
    rss_gib_after: float
    max_rss_gib: float


class CpuProfiler:
    def __init__(self) -> None:
        self.rows: list[CpuProfileRow] = []

    @contextmanager
    def profile(self, name: str):
        before = _rss_gib()
        started = time.perf_counter()
        print(
            f"[cpu-profile] enter {name}",
            flush=True,
        )
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            after = _rss_gib()
            row = CpuProfileRow(
                name=name,
                elapsed_seconds=elapsed,
                rss_gib_before=before,
                rss_gib_after=after,
                max_rss_gib=_max_rss_gib(),
            )
            self.rows.append(row)
            print(
                "[cpu-profile] "
                f"{name}: {elapsed:.3f}s | "
                f"RSS {before:.2f}->{after:.2f} GiB | "
                f"peak {row.max_rss_gib:.2f} GiB",
                flush=True,
            )

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "name": row.name,
                "elapsed_seconds": row.elapsed_seconds,
                "rss_gib_before": row.rss_gib_before,
                "rss_gib_after": row.rss_gib_after,
                "max_rss_gib": row.max_rss_gib,
            }
            for row in self.rows
        ]


# =============================================================================
# EXACT EXPERIMENT-31 REDUCED MODEL CONFIGURATION
# =============================================================================


def reduced_config():
    """Exact reduced_config() from experiments/stirnet/31_spatial_first_overfit.py."""
    from learned.stirnet import StirNetConfig

    cfg = StirNetConfig()
    cfg.evidence.stem_channels = 12
    cfg.evidence.prior_gate_hidden = 16
    cfg.spatial.channels = (12, 24, 48, 96)
    cfg.spatial.blocks_per_level = 1
    cfg.spatial.acquisition_dim = 32
    cfg.geometry.hidden_channels = 32
    cfg.geometry.residual_blocks = 2
    cfg.partition.node_feature_channels = 16
    cfg.partition.rag_hidden_dim = 48
    cfg.partition.rag_layers = 2
    cfg.partition.max_supervoxels = 4096
    cfg.instances.d_model = 64
    cfg.instances.pooled_feature_dim = 16
    cfg.history.hidden_channels = 16
    cfg.temporal.d_model = 64
    cfg.temporal.graph_hidden_dim = 128
    cfg.temporal.cross_heads = 4
    cfg.refinement.hidden_channels = 32
    cfg.refinement.query_channels = 16
    cfg.refinement.max_rois_per_batch = 8
    cfg.validate()
    return cfg


# =============================================================================
# FRESH SOURCE PREPROCESSING
# =============================================================================


def _one_physical_edt_marker_per_component(
    labels,
    spacing_um,
):
    """Match the repository's Modal data-preparation marker construction."""
    import numpy as np
    from scipy import ndimage as ndi

    labels = np.asarray(labels)
    marker = np.zeros(
        labels.shape,
        dtype=np.float32,
    )

    objects = ndi.find_objects(labels)
    for instance_id, slc in enumerate(
        objects,
        start=1,
    ):
        if slc is None:
            continue

        expanded = tuple(
            slice(
                max(0, axis.start - 1),
                min(
                    labels.shape[d],
                    axis.stop + 1,
                ),
            )
            for d, axis in enumerate(slc)
        )
        component = (
            labels[expanded]
            == instance_id
        )
        if not component.any():
            continue

        edt = ndi.distance_transform_edt(
            component,
            sampling=spacing_um,
        )
        local_position = np.unravel_index(
            np.argmax(edt),
            edt.shape,
        )
        global_position = tuple(
            int(expanded[d].start)
            + int(local_position[d])
            for d in range(3)
        )
        marker[global_position] = 1.0

    return marker


def _current_source_roi(
    current_labels,
    spacing_um,
    margin_um: float,
):
    """Bounding box of current/source foreground plus physical margin.

    GT is deliberately not used.
    """
    import numpy as np

    shape = np.asarray(
        current_labels.shape,
        dtype=np.int64,
    )
    coords = np.where(
        np.asarray(current_labels) > 0
    )

    if not len(coords[0]):
        low = np.zeros(3, dtype=np.int64)
        high = shape.copy()
    else:
        low = np.asarray(
            [axis.min() for axis in coords],
            dtype=np.int64,
        )
        high = np.asarray(
            [axis.max() + 1 for axis in coords],
            dtype=np.int64,
        )
        margin = np.ceil(
            float(margin_um)
            / np.asarray(
                spacing_um,
                dtype=np.float64,
            )
        ).astype(np.int64)
        low = np.maximum(
            low - margin,
            0,
        )
        high = np.minimum(
            high + margin,
            shape,
        )

    roi = tuple(
        slice(int(a), int(b))
        for a, b in zip(low, high)
    )
    return roi, low, high


def _gt_roi_coverage(
    gt_full,
    roi,
) -> dict[str, Any]:
    """Diagnostic only: verify that source-derived trimming did not lose GT."""
    import numpy as np

    gt_full = np.asarray(gt_full)
    gt_crop = np.asarray(gt_full[roi])

    total_positive = int(
        np.count_nonzero(gt_full > 0)
    )
    kept_positive = int(
        np.count_nonzero(gt_crop > 0)
    )

    full_ids, full_counts = np.unique(
        gt_full[gt_full > 0],
        return_counts=True,
    )
    crop_ids, crop_counts = np.unique(
        gt_crop[gt_crop > 0],
        return_counts=True,
    )
    crop_count_map = {
        int(i): int(c)
        for i, c in zip(crop_ids, crop_counts)
    }

    fully_contained = 0
    partial_ids: list[int] = []
    missing_ids: list[int] = []
    for label_id, count in zip(
        full_ids,
        full_counts,
    ):
        label_id = int(label_id)
        full_count = int(count)
        crop_count = crop_count_map.get(
            label_id,
            0,
        )
        if crop_count == full_count:
            fully_contained += 1
        elif crop_count == 0:
            missing_ids.append(label_id)
        else:
            partial_ids.append(label_id)

    return {
        "positive_voxel_coverage": (
            1.0
            if total_positive == 0
            else kept_positive
            / total_positive
        ),
        "full_gt_cell_count": int(
            len(full_ids)
        ),
        "fully_contained_gt_cells": int(
            fully_contained
        ),
        "partial_gt_ids": partial_ids,
        "missing_gt_ids": missing_ids,
    }


def build_fresh_spatial_batch(
    frame: int,
    cpu_profiler: CpuProfiler,
    work_root: Path,
):
    """Build one Experiment-31-style spatial sample from ORIGINAL source files."""
    import numpy as np
    import torch
    from scipy import ndimage as ndi

    sys.path.insert(
        0,
        REMOTE_REPO_ROOT,
    )

    from learned.stirnet.data.sample_builder import (
        build_spatial_channels,
        robust_normalize,
    )
    from learned.stirnet.data.targets import (
        estimate_model_dref_um,
    )

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
    raw_name = (
        f"F22_{frame:03d}_image_0001.npy"
    )
    gt_name = (
        f"F22_{frame:03d}_masks_0001.npy"
    )
    raw_source = source_dir / raw_name
    gt_source = source_dir / gt_name

    for path in (raw_source, gt_source):
        if not path.exists():
            raise FileNotFoundError(path)
        if "/learned/stirnet/" in str(path):
            raise RuntimeError(
                "Stage-10 source-only invariant "
                f"violated by {path}"
            )

    source_local = work_root / "source"
    source_local.mkdir(
        parents=True,
        exist_ok=True,
    )
    raw_local = source_local / raw_name
    gt_local = source_local / gt_name

    with cpu_profiler.profile(
        "source_volume_to_local_ssd"
    ):
        shutil.copy2(
            raw_source,
            raw_local,
        )
        shutil.copy2(
            gt_source,
            gt_local,
        )

    with cpu_profiler.profile(
        "load_original_raw_and_gt"
    ):
        raw = np.load(
            raw_local,
            allow_pickle=False,
        )
        gt = np.load(
            gt_local,
            allow_pickle=False,
        )

    if (
        raw.ndim != 3
        or gt.ndim != 3
        or raw.shape != gt.shape
    ):
        raise ValueError(
            "Invalid raw/GT source pair: "
            f"raw={raw.shape}, gt={gt.shape}"
        )

    full_shape = tuple(
        int(v) for v in raw.shape
    )
    full_voxels = int(
        np.prod(full_shape)
    )

    preprocessing_config = (
        PreprocessingConfig(
            low_percentile=1.0,
            high_percentile=99.5,
            denoise_sigma_um=0.8,
            background_sigma_um=4.0,
            voxel_size_zyx_um=(
                SPACING_ZYX_UM
            ),
        )
    )
    masking_config = MaskingConfig()

    # Full-frame canonical preprocessing is required because it produces the
    # current/source segmentation used to decide what outer space is irrelevant.
    with cpu_profiler.profile(
        "canonical_preprocessing"
    ):
        processed = preprocess_volume(
            raw,
            config=preprocessing_config,
            return_diagnostics=False,
        )

    with cpu_profiler.profile(
        "canonical_otsu_masking"
    ):
        binary = create_binary_mask(
            processed,
            config=masking_config,
            return_diagnostics=False,
        )

    connectivity_6 = (
        ndi.generate_binary_structure(
            3,
            1,
        )
    )
    with cpu_profiler.profile(
        "six_connected_components"
    ):
        current_full, component_count = (
            ndi.label(
                binary,
                structure=connectivity_6,
            )
        )
        current_full = (
            current_full.astype(
                np.int32,
                copy=False,
            )
        )

    # Match Experiment 31: model dref comes from the untrimmed current
    # segmentation, not GT and not the ROI.
    with cpu_profiler.profile(
        "model_dref_from_full_current"
    ):
        model_dref_um = float(
            estimate_model_dref_um(
                current_full,
                SPACING_ZYX_UM,
            )
        )

    with cpu_profiler.profile(
        "physical_edt_marker"
    ):
        marker_full = (
            _one_physical_edt_marker_per_component(
                current_full,
                SPACING_ZYX_UM,
            )
        )

    # Match Experiment 31 semantics: raw normalization is computed on the
    # original acquisition, then the model ROI is extracted.
    with cpu_profiler.profile(
        "full_raw_robust_normalization"
    ):
        raw_norm_full = robust_normalize(
            raw
        )

    if TRIM_EMPTY_ACQUISITION_BORDER:
        with cpu_profiler.profile(
            "source_only_useful_roi"
        ):
            roi, roi_low, roi_high = (
                _current_source_roi(
                    current_full,
                    SPACING_ZYX_UM,
                    ROI_MARGIN_UM,
                )
            )
    else:
        roi = tuple(
            slice(0, int(v))
            for v in full_shape
        )
        roi_low = np.zeros(
            3,
            dtype=np.int64,
        )
        roi_high = np.asarray(
            full_shape,
            dtype=np.int64,
        )

    with cpu_profiler.profile(
        "gt_roi_safety_diagnostic"
    ):
        gt_coverage = (
            _gt_roi_coverage(
                gt,
                roi,
            )
        )

    with cpu_profiler.profile(
        "crop_source_tensors_to_useful_roi"
    ):
        current = np.asarray(
            current_full[roi]
        ).astype(
            np.int64,
            copy=True,
        )
        gt_crop = np.asarray(
            gt[roi]
        ).astype(
            np.int64,
            copy=True,
        )
        raw_norm = np.asarray(
            raw_norm_full[roi]
        ).astype(
            np.float32,
            copy=True,
        )
        marker = np.asarray(
            marker_full[roi]
        ).astype(
            np.float32,
            copy=True,
        )

    # Release the largest full-frame intermediates before channel construction.
    del (
        processed,
        binary,
        raw_norm_full,
        marker_full,
        current_full,
        raw,
    )
    gc.collect()

    with cpu_profiler.profile(
        "production_build_spatial_channels"
    ):
        spatial = build_spatial_channels(
            raw_norm,
            current,
            SPACING_ZYX_UM,
            model_dref_um,
            marker,
        )

    if spatial.shape[0] != 5:
        raise RuntimeError(
            "Reduced Stage 10 does not satisfy "
            "the five-channel spatial contract."
        )

    roi_shape = tuple(
        int(v)
        for v in spatial.shape[-3:]
    )
    roi_voxels = int(
        np.prod(roi_shape)
    )

    with cpu_profiler.profile(
        "construct_cpu_training_batch"
    ):
        batch = {
            "spatial_inputs": (
                torch.from_numpy(
                    spatial
                ).unsqueeze(0)
            ),
            "instance_labels": (
                torch.from_numpy(
                    current
                ).unsqueeze(0)
            ),
            "spacing_um": torch.tensor(
                [SPACING_ZYX_UM],
                dtype=torch.float32,
            ),
            "dref_um": torch.tensor(
                [model_dref_um],
                dtype=torch.float32,
            ),
            "targets": [
                {
                    "label_map": (
                        torch.from_numpy(
                            gt_crop
                        )
                    )
                }
            ],
        }

    current_count = int(
        np.unique(
            current[current > 0]
        ).size
    )
    gt_count = int(
        np.unique(
            gt_crop[gt_crop > 0]
        ).size
    )

    sample = {
        "frame": int(frame),
        "full_shape_zyx": list(
            full_shape
        ),
        "roi_shape_zyx": list(
            roi_shape
        ),
        "full_voxel_count": int(
            full_voxels
        ),
        "roi_voxel_count": int(
            roi_voxels
        ),
        "roi_voxel_fraction": float(
            roi_voxels
            / max(full_voxels, 1)
        ),
        "voxel_reduction_fraction": float(
            1.0
            - roi_voxels
            / max(full_voxels, 1)
        ),
        "roi_low_zyx": [
            int(v) for v in roi_low
        ],
        "roi_high_zyx": [
            int(v) for v in roi_high
        ],
        "roi_margin_um": float(
            ROI_MARGIN_UM
        ),
        "roi_defined_from": (
            "current/source segmentation only"
        ),
        "current_component_count_full": int(
            component_count
        ),
        "current_component_count_roi": int(
            current_count
        ),
        "gt_cell_count_roi": int(
            gt_count
        ),
        "gt_roi_safety": gt_coverage,
        "spacing_zyx_um": [
            float(v)
            for v in SPACING_ZYX_UM
        ],
        "model_dref_um": float(
            model_dref_um
        ),
        "model_dref_source": (
            "full_current_segmentation"
        ),
        "spatial_input_shape": [
            int(v) for v in spatial.shape
        ],
        "spatial_input_gib": float(
            spatial.nbytes / 1024**3
        ),
        "cache_policy": (
            "strict source-only fresh build"
        ),
        "source_files": [
            str(raw_source),
            str(gt_source),
        ],
    }

    del (
        raw_norm,
        marker,
        spatial,
        current,
        gt_crop,
        gt,
    )
    gc.collect()

    return batch, sample


# =============================================================================
# REPORTING
# =============================================================================


def _jsonable(value: Any) -> Any:
    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            type(None),
        ),
    ):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(k): _jsonable(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(v)
            for v in value
        ]
    try:
        return float(value)
    except Exception:
        return str(value)


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            _jsonable(payload),
            indent=2,
        ),
        encoding="utf-8",
    )


def _print_cpu_summary(
    rows: list[dict[str, Any]],
) -> None:
    print(
        "\nCPU / preprocessing timing",
        flush=True,
    )
    print("-" * 108, flush=True)
    for row in rows:
        print(
            f"{row['name']:58s} "
            f"{row['elapsed_seconds']:9.3f}s  "
            f"RSS={row['rss_gib_after']:6.2f} GiB  "
            f"peak={row['max_rss_gib']:6.2f} GiB",
            flush=True,
        )
    print("-" * 108, flush=True)


def _print_gpu_profile_summary(
    trainer,
) -> None:
    summary = trainer.stage_profiler.summary()
    if not summary:
        print(
            "\nNo GPU profile records.",
            flush=True,
        )
        return

    rows = sorted(
        summary.items(),
        key=lambda item: float(
            item[1]["elapsed_seconds"]
        ),
        reverse=True,
    )

    print(
        "\nGPU stage timing / memory "
        "(CUDA-synchronized)",
        flush=True,
    )
    print("-" * 122, flush=True)
    print(
        f"{'stage':60s} "
        f"{'seconds':>9s} "
        f"{'peak alloc MiB':>15s} "
        f"{'peak reserv MiB':>16s} "
        f"{'calls':>7s}",
        flush=True,
    )
    print("-" * 122, flush=True)
    for name, row in rows:
        print(
            f"{name:60s} "
            f"{float(row['elapsed_seconds']):9.3f} "
            f"{float(row['peak_allocated_mb']):15.1f} "
            f"{float(row['peak_reserved_mb']):16.1f} "
            f"{int(row['calls']):7d}",
            flush=True,
        )
    print("-" * 122, flush=True)


# =============================================================================
# MODAL REMOTE FUNCTION
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
def run_reduced_spatial_step(
    frame: int = DEFAULT_FRAME,
) -> dict[str, Any]:
    import numpy as np
    import torch

    sys.path.insert(
        0,
        REMOTE_REPO_ROOT,
    )

    from learned.stirnet import StirNet
    from learned.stirnet.training.config import (
        TrainingConfig,
    )
    from learned.stirnet.training.trainer import (
        Trainer,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in Modal."
        )

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.set_float32_matmul_precision(
        "high"
    )
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    total_vram_gib = (
        gpu_props.total_memory / 1024**3
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(RUNS_PREFIX)
        / f"{timestamp}_F22_{frame:03d}"
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    gpu_profile_path = (
        run_dir / "gpu_profile.jsonl"
    )
    summary_path = (
        run_dir / "summary.json"
    )

    if gpu_profile_path.exists():
        gpu_profile_path.unlink()

    work_root = Path(
        f"/tmp/stirnet-stage10-{timestamp}"
    )
    shutil.rmtree(
        work_root,
        ignore_errors=True,
    )
    work_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    cpu_profiler = CpuProfiler()
    total_started = time.perf_counter()

    summary: dict[str, Any] = {
        "format_version": 2,
        "stage": (
            "10_reduced_spatial_train_step"
        ),
        "status": "running",
        "frame": int(frame),
        "seed": SEED,
        "architecture_source": (
            "experiments/stirnet/"
            "31_spatial_first_overfit.py::reduced_config"
        ),
        "cache_policy": (
            "original source only; "
            "fresh preprocessing"
        ),
        "roi_policy": {
            "trim_empty_border": bool(
                TRIM_EMPTY_ACQUISITION_BORDER
            ),
            "margin_um": float(
                ROI_MARGIN_UM
            ),
            "extent_source": (
                "current/source segmentation only"
            ),
        },
        "modal": {
            "gpu_request": MODAL_GPU,
            "gpu_name": gpu_name,
            "gpu_total_vram_gib": total_vram_gib,
            "cpu_request": MODAL_CPU,
            "memory_mb_request": MODAL_MEMORY_MB,
        },
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "bf16_supported": bool(
                torch.cuda.is_bf16_supported()
            ),
        },
        "run_dir": str(run_dir),
        "gpu_profile_path": str(
            gpu_profile_path
        ),
    }

    trainer = None
    batch = None
    geometry_targets = None

    print("\n" + "=" * 122, flush=True)
    print(
        "STIR-Net Stage 10 — fresh source preprocessing + "
        "Experiment-31 REDUCED spatial forward/backward",
        flush=True,
    )
    print("=" * 122, flush=True)
    print(
        f"GPU                      : "
        f"{gpu_name} ({total_vram_gib:.2f} GiB)",
        flush=True,
    )
    print(
        f"CPU / RAM                : "
        f"{MODAL_CPU:.0f} CPU / "
        f"{MODAL_MEMORY_MB / 1024:.1f} GiB",
        flush=True,
    )
    print(
        f"Frame                    : F22_{frame:03d}",
        flush=True,
    )
    print(
        "Architecture             : "
        "EXACT Experiment-31 reduced_config()",
        flush=True,
    )
    print(
        "ROI                      : "
        f"current/source bbox + {ROI_MARGIN_UM:g} um margin",
        flush=True,
    )
    print(
        "Cache policy             : "
        "STRICT SOURCE-ONLY",
        flush=True,
    )
    print(
        "Curriculum               : spatial_partition",
        flush=True,
    )
    print(
        "Trainable                : geometry_spatial + partition",
        flush=True,
    )
    print(
        "Temporal/refinement      : not executed",
        flush=True,
    )
    print("=" * 122, flush=True)

    try:
        # -------------------------------------------------------------
        # A. Build fresh source sample and source-only useful ROI.
        # -------------------------------------------------------------
        batch, sample = (
            build_fresh_spatial_batch(
                frame,
                cpu_profiler,
                work_root,
            )
        )
        summary["sample"] = sample

        print("\nFresh sample", flush=True)
        print("-" * 86, flush=True)
        print(
            f"Full shape               : "
            f"{tuple(sample['full_shape_zyx'])}",
            flush=True,
        )
        print(
            f"Useful ROI shape         : "
            f"{tuple(sample['roi_shape_zyx'])}",
            flush=True,
        )
        print(
            f"Voxel reduction          : "
            f"{100 * sample['voxel_reduction_fraction']:.2f}%",
            flush=True,
        )
        print(
            f"Current components       : "
            f"{sample['current_component_count_roi']}",
            flush=True,
        )
        print(
            f"GT cells in ROI          : "
            f"{sample['gt_cell_count_roi']}",
            flush=True,
        )
        print(
            f"GT voxel coverage        : "
            f"{100 * sample['gt_roi_safety']['positive_voxel_coverage']:.4f}%",
            flush=True,
        )
        print(
            f"GT fully contained       : "
            f"{sample['gt_roi_safety']['fully_contained_gt_cells']}"
            f"/{sample['gt_roi_safety']['full_gt_cell_count']}",
            flush=True,
        )
        print(
            f"dref                     : "
            f"{sample['model_dref_um']:.4f} um",
            flush=True,
        )
        print(
            f"Spatial tensor           : "
            f"{sample['spatial_input_gib']:.3f} GiB",
            flush=True,
        )

        # -------------------------------------------------------------
        # B. EXACT Experiment-31 reduced model configuration.
        # -------------------------------------------------------------
        with cpu_profiler.profile(
            "reduced_model_and_trainer_initialization"
        ):
            cfg = reduced_config()

            train_cfg = TrainingConfig()
            train_cfg.amp_dtype = "bf16"
            train_cfg.curriculum.fixed_stage = (
                "spatial_partition"
            )

            # This test is one full useful-ROI gradient pass, not the crop
            # training path. We want to reproduce the Experiment-31 model size
            # while testing the complete all-cell useful ROI.
            train_cfg.curriculum.geometry_bootstrap_crop_enabled = False
            train_cfg.curriculum.spatial_partition_crop_enabled = False

            train_cfg.profile_memory = True
            train_cfg.memory_profile_path = str(
                gpu_profile_path
            )
            train_cfg.validate()

            model = StirNet(cfg)
            trainer = Trainer(
                model,
                train_cfg,
                device=device,
            )

        total_params = sum(
            p.numel()
            for p in model.parameters()
        )
        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        summary["model_config"] = cfg.to_dict()
        summary["training_config"] = (
            train_cfg.to_dict()
        )
        summary["parameter_counts"] = {
            "total": int(total_params),
            "trainable_spatial_stage": int(
                trainable_params
            ),
        }

        print("\nReduced model", flush=True)
        print("-" * 86, flush=True)
        print(
            f"Evidence stem channels   : "
            f"{cfg.evidence.stem_channels}",
            flush=True,
        )
        print(
            f"Spatial channels         : "
            f"{cfg.spatial.channels}",
            flush=True,
        )
        print(
            f"Blocks / level           : "
            f"{cfg.spatial.blocks_per_level}",
            flush=True,
        )
        print(
            f"Geometry hidden / blocks : "
            f"{cfg.geometry.hidden_channels} / "
            f"{cfg.geometry.residual_blocks}",
            flush=True,
        )
        print(
            f"RAG hidden               : "
            f"{cfg.partition.rag_hidden_dim}",
            flush=True,
        )
        print(
            f"Trainable params         : "
            f"{trainable_params:,}",
            flush=True,
        )

        # -------------------------------------------------------------
        # C. Build fresh production geometry supervision on CPU.
        #    This is a one-time deterministic preparation cost and is excluded
        #    from the measured GPU train_step.
        # -------------------------------------------------------------
        with cpu_profiler.profile(
            "production_geometry_targets_cpu"
        ):
            geometry_targets = (
                trainer.prepare_geometry_targets(
                    batch
                )
            )

        if (
            trainer.curriculum_stage.name
            != "spatial_partition"
        ):
            raise RuntimeError(
                "Trainer did not enter "
                "spatial_partition."
            )
        if (
            trainer.curriculum_stage.execution_stage
            != "spatial"
        ):
            raise RuntimeError(
                "spatial_partition does not "
                "execute the spatial path."
            )

        # -------------------------------------------------------------
        # D. Exact current production train step on the complete useful ROI.
        # -------------------------------------------------------------
        print(
            "\nStarting production reduced-model "
            "full useful-ROI train_step() ...",
            flush=True,
        )
        train_started = time.perf_counter()

        metrics = trainer.train_step(
            batch,
            precomputed_geometry_targets=(
                geometry_targets
            ),
        )

        train_wall = (
            time.perf_counter()
            - train_started
        )

        finite_metrics = all(
            np.isfinite(float(value))
            for key, value in metrics.items()
            if (
                isinstance(
                    value,
                    (float, int),
                )
                and not key.startswith(
                    "profile_"
                )
            )
        )

        grad_geometry = float(
            metrics.get(
                "grad_geometry_spatial",
                0.0,
            )
        )
        grad_partition = float(
            metrics.get(
                "grad_partition",
                0.0,
            )
        )
        expected_gradients_nonzero = bool(
            np.isfinite(grad_geometry)
            and np.isfinite(grad_partition)
            and grad_geometry > 0
            and grad_partition > 0
        )
        optimizer_step_skipped = bool(
            metrics.get(
                "optimizer_step_skipped",
                1.0,
            )
        )

        summary.update(
            {
                "status": (
                    "pass"
                    if (
                        finite_metrics
                        and expected_gradients_nonzero
                        and not optimizer_step_skipped
                    )
                    else "failed_validation"
                ),
                "train_step_metrics": metrics,
                "finite_metrics": bool(
                    finite_metrics
                ),
                "expected_gradient_norms": {
                    "geometry_spatial": (
                        grad_geometry
                    ),
                    "partition": (
                        grad_partition
                    ),
                },
                "expected_gradients_nonzero": (
                    expected_gradients_nonzero
                ),
                "optimizer_step_skipped": (
                    optimizer_step_skipped
                ),
                "production_train_step_wall_seconds": float(
                    train_wall
                ),
                "cuda": {
                    "peak_allocated_gib": float(
                        torch.cuda.max_memory_allocated(
                            device
                        )
                        / 1024**3
                    ),
                    "peak_reserved_gib": float(
                        torch.cuda.max_memory_reserved(
                            device
                        )
                        / 1024**3
                    ),
                    "allocated_after_gib": float(
                        torch.cuda.memory_allocated(
                            device
                        )
                        / 1024**3
                    ),
                    "reserved_after_gib": float(
                        torch.cuda.memory_reserved(
                            device
                        )
                        / 1024**3
                    ),
                },
            }
        )

        _print_gpu_profile_summary(
            trainer
        )

    except BaseException as error:
        is_oom = (
            isinstance(
                error,
                torch.OutOfMemoryError,
            )
            or (
                isinstance(
                    error,
                    RuntimeError,
                )
                and "out of memory"
                in str(error).lower()
            )
        )

        try:
            torch.cuda.synchronize()
        except BaseException:
            pass

        cuda_failure = {}
        try:
            free_bytes, total_bytes = (
                torch.cuda.mem_get_info(
                    device
                )
            )
            cuda_failure = {
                "allocated_gib": float(
                    torch.cuda.memory_allocated(
                        device
                    )
                    / 1024**3
                ),
                "reserved_gib": float(
                    torch.cuda.memory_reserved(
                        device
                    )
                    / 1024**3
                ),
                "peak_allocated_gib": float(
                    torch.cuda.max_memory_allocated(
                        device
                    )
                    / 1024**3
                ),
                "peak_reserved_gib": float(
                    torch.cuda.max_memory_reserved(
                        device
                    )
                    / 1024**3
                ),
                "free_gib": float(
                    free_bytes / 1024**3
                ),
                "total_gib": float(
                    total_bytes / 1024**3
                ),
            }
        except BaseException:
            pass

        failure_stage = None
        failure_phase = None
        if trainer is not None:
            failure_stage = (
                trainer.stage_profiler.last_profile_stage
            )
            failure_phase = (
                trainer.stage_profiler.last_profile_phase
            )
            try:
                _print_gpu_profile_summary(
                    trainer
                )
            except BaseException:
                pass

        summary.update(
            {
                "status": (
                    "oom"
                    if is_oom
                    else "error"
                ),
                "error_type": type(
                    error
                ).__name__,
                "error_message": str(
                    error
                )[:4000],
                "traceback": traceback.format_exc()[
                    -12000:
                ],
                "failure_profile_stage": (
                    failure_stage
                ),
                "failure_profile_phase": (
                    failure_phase
                ),
                "cuda_failure": cuda_failure,
            }
        )

        print(
            "\n" + "!" * 122,
            flush=True,
        )
        print(
            "STAGE 10 DID NOT COMPLETE",
            flush=True,
        )
        print(
            f"Failure                  : "
            f"{summary['status']}",
            flush=True,
        )
        print(
            f"Exception                : "
            f"{type(error).__name__}: "
            f"{str(error)[:1200]}",
            flush=True,
        )
        print(
            f"Last profile phase       : "
            f"{failure_phase}",
            flush=True,
        )
        print(
            f"Last profile stage       : "
            f"{failure_stage}",
            flush=True,
        )
        if cuda_failure:
            print(
                f"Peak allocated           : "
                f"{cuda_failure.get('peak_allocated_gib', 0):.3f} GiB",
                flush=True,
            )
            print(
                f"Peak reserved            : "
                f"{cuda_failure.get('peak_reserved_gib', 0):.3f} GiB",
                flush=True,
            )
        print("!" * 122, flush=True)

        try:
            torch.cuda.empty_cache()
        except BaseException:
            pass

    finally:
        summary["cpu_profile"] = (
            cpu_profiler.to_dict()
        )
        summary["cpu_peak_rss_gib"] = (
            _max_rss_gib()
        )
        summary["total_wall_seconds"] = (
            time.perf_counter()
            - total_started
        )

        if trainer is not None:
            try:
                summary[
                    "gpu_profile_summary"
                ] = (
                    trainer.stage_profiler.summary()
                )
                summary[
                    "gpu_last_profile_stage"
                ] = (
                    trainer.stage_profiler.last_profile_stage
                )
                summary[
                    "gpu_last_profile_phase"
                ] = (
                    trainer.stage_profiler.last_profile_phase
                )
            except BaseException:
                pass

        _write_json(
            summary_path,
            summary,
        )

        try:
            runs_volume.commit()
        except BaseException as commit_error:
            print(
                "[WARN] runs volume commit "
                f"failed: {commit_error}",
                flush=True,
            )

        try:
            _print_cpu_summary(
                summary["cpu_profile"]
            )
        except BaseException:
            pass

        print(
            "\n" + "=" * 122,
            flush=True,
        )
        print(
            "Stage 10 reduced-model summary",
            flush=True,
        )
        print("=" * 122, flush=True)
        print(
            f"Status                   : "
            f"{summary.get('status')}",
            flush=True,
        )

        if "sample" in summary:
            sample = summary["sample"]
            print(
                f"Full shape               : "
                f"{sample['full_shape_zyx']}",
                flush=True,
            )
            print(
                f"Useful ROI shape         : "
                f"{sample['roi_shape_zyx']}",
                flush=True,
            )
            print(
                f"Voxel reduction          : "
                f"{100 * sample['voxel_reduction_fraction']:.2f}%",
                flush=True,
            )
            print(
                f"GT voxel coverage        : "
                f"{100 * sample['gt_roi_safety']['positive_voxel_coverage']:.4f}%",
                flush=True,
            )

        if "train_step_metrics" in summary:
            metrics = summary[
                "train_step_metrics"
            ]
            for key in (
                "loss",
                "geometry_loss",
                "spatial_rag_bce",
                "spatial_rag_accuracy",
                "rag_valid_edge_fraction",
                "grad_geometry_spatial",
                "grad_partition",
                "grad_norm",
                "forward_seconds",
                "target_seconds",
                "backward_seconds",
                "total_step_seconds",
                "peak_allocated_mb",
                "peak_reserved_mb",
            ):
                if key in metrics:
                    print(
                        f"{key:25s}: "
                        f"{metrics[key]:.6f}",
                        flush=True,
                    )

        if "cuda" in summary:
            print(
                f"Peak CUDA allocated      : "
                f"{summary['cuda']['peak_allocated_gib']:.3f} GiB",
                flush=True,
            )
            print(
                f"Peak CUDA reserved       : "
                f"{summary['cuda']['peak_reserved_gib']:.3f} GiB",
                flush=True,
            )

        print(
            f"CPU peak RSS             : "
            f"{summary['cpu_peak_rss_gib']:.3f} GiB",
            flush=True,
        )
        print(
            f"Total wall time          : "
            f"{summary['total_wall_seconds']:.3f} s",
            flush=True,
        )
        print(
            f"Saved summary            : "
            f"{summary_path}",
            flush=True,
        )
        print(
            f"Saved GPU profile        : "
            f"{gpu_profile_path}",
            flush=True,
        )
        print("=" * 122, flush=True)

        shutil.rmtree(
            work_root,
            ignore_errors=True,
        )
        batch = None
        geometry_targets = None
        gc.collect()

    return summary


# =============================================================================
# LOCAL ENTRYPOINT
# =============================================================================


@app.local_entrypoint()
def main(
    frame: int = DEFAULT_FRAME,
):
    result = run_reduced_spatial_step.remote(
        frame=frame,
    )

    print(
        "\nReturned Stage-10 result",
        flush=True,
    )
    print("-" * 86, flush=True)
    print(
        json.dumps(
            {
                "status": result.get(
                    "status"
                ),
                "frame": result.get(
                    "frame"
                ),
                "sample": result.get(
                    "sample"
                ),
                "parameter_counts": result.get(
                    "parameter_counts"
                ),
                "expected_gradient_norms": result.get(
                    "expected_gradient_norms"
                ),
                "finite_metrics": result.get(
                    "finite_metrics"
                ),
                "optimizer_step_skipped": result.get(
                    "optimizer_step_skipped"
                ),
                "cuda": result.get(
                    "cuda",
                    result.get(
                        "cuda_failure"
                    ),
                ),
                "failure_profile_phase": result.get(
                    "failure_profile_phase"
                ),
                "failure_profile_stage": result.get(
                    "failure_profile_stage"
                ),
                "production_train_step_wall_seconds": result.get(
                    "production_train_step_wall_seconds"
                ),
                "total_wall_seconds": result.get(
                    "total_wall_seconds"
                ),
                "run_dir": result.get(
                    "run_dir"
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    pass
