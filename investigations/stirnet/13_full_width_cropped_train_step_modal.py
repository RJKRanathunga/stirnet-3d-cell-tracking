from __future__ import annotations

"""
STIR-Net Stage 13 — full-width cropped forward/backward on Modal.

This is the Modal version of the Stage-13 systems validation.

Goal
----
Prove that the CURRENT FULL-WIDTH STIR-Net can complete one real
spatial_partition training step on a merge-aware crop:

    fresh original BlastoSPIM source
        -> production preprocessing/current segmentation
        -> merge-first greedy crop planner
        -> crop-only five-channel construction
        -> partial-cell ignore supervision
        -> crop-only geometry targets
        -> FULL-WIDTH StirNetConfig()
        -> fast watershed + safety guard + RAG
        -> spatial_partition loss
        -> backward + clip + optimizer step

The test deliberately uses a Modal L40S so CUDA memory accounting is not
confounded by Windows WDDM oversubscription.

Default crop:
    32 x 192 x 192

Run:
    modal run investigations/stirnet/13_full_width_cropped_train_step_modal.py

Optional:
    modal run investigations/stirnet/13_full_width_cropped_train_step_modal.py \
        --frame 32 --crop-z 32 --crop-y 192 --crop-x 192

Outputs:
    stirnet-runs/stirnet/investigations/stage13_full_width_crop_modal/<timestamp>/
        summary.json
        gpu_profile.jsonl

No checkpoint or dense prediction volume is saved.
"""

import gc
import json
import math
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

import modal


# =============================================================================
# MODAL / REPOSITORY LAYOUT
# =============================================================================

app = modal.App("stirnet-stage13-full-width-crop")

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
    "stage13_full_width_crop_modal"
)

SPACING_ZYX_UM = (2.0, 0.208, 0.208)
DEFAULT_FRAME = 32
SEED = 40266

MODAL_GPU = "L40S"
MODAL_CPU = 8.0
MODAL_MEMORY_MB = 32_768
MODAL_TIMEOUT_SECONDS = 60 * 60


def _resolve_local_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "learned").is_dir() and (candidate / "src").is_dir():
            return candidate
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
        # Exact GPU EDT path used by the current repository when available.
        "cupy-cuda13x",
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
# HELPERS
# =============================================================================


def _jsonable(value: Any) -> Any:
    import numpy as np
    import torch

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_snapshot(device) -> dict[str, float]:
    import torch

    if device.type != "cuda":
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "max_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        "free_mb": free_bytes / 1024**2,
        "total_mb": total_bytes / 1024**2,
    }


def _parameter_count(model) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return int(total), int(trainable)


def _gradient_group_norms(model) -> dict[str, float]:
    import torch
    from learned.stirnet.training.curriculum import model_parameter_groups

    rows: dict[str, float] = {}
    for name, parameters in model_parameter_groups(model).items():
        norm = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            value = float(
                torch.linalg.vector_norm(
                    parameter.grad.detach().float()
                ).cpu()
            )
            norm = math.hypot(norm, value)
        rows[name] = norm
    return rows


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
        local = np.unravel_index(np.argmax(edt), edt.shape)
        global_position = tuple(
            int(expanded[d].start) + int(local[d])
            for d in range(3)
        )
        marker[global_position] = 1.0
    return marker


def _crop_spec_from_record(record, full_shape, spacing):
    import torch
    from learned.stirnet.training.crops import CropSpec

    lower = torch.tensor(
        [int(axis.start) for axis in record.slices_zyx],
        dtype=torch.float32,
    )
    size = torch.tensor(
        [
            int(axis.stop) - int(axis.start)
            for axis in record.slices_zyx
        ],
        dtype=torch.float32,
    )
    full_center = 0.5 * (
        torch.tensor(full_shape, dtype=torch.float32) - 1
    )
    crop_center = lower + 0.5 * (size - 1)
    shift = (
        crop_center - full_center
    ) * torch.tensor(spacing, dtype=torch.float32)
    return CropSpec(
        batch_index=0,
        slices_zyx=record.slices_zyx,
        full_shape_zyx=tuple(int(v) for v in full_shape),
        center_shift_um=shift,
        candidate_type=record.candidate_type,
        complete_cell_ids=record.complete_cell_ids,
        partial_cell_ids=record.partial_cell_ids,
        true_boundary_cell_ids=record.true_boundary_cell_ids,
        merge_source_ids=record.merge_source_ids,
    )


def _full_width_config_summary(cfg) -> dict[str, Any]:
    return {
        "evidence_stem_channels": cfg.evidence.stem_channels,
        "prior_gate_hidden": cfg.evidence.prior_gate_hidden,
        "spatial_channels": list(cfg.spatial.channels),
        "blocks_per_level": cfg.spatial.blocks_per_level,
        "activation_checkpointing": cfg.spatial.activation_checkpointing,
        "acquisition_dim": cfg.spatial.acquisition_dim,
        "geometry_hidden_channels": cfg.geometry.hidden_channels,
        "geometry_residual_blocks": cfg.geometry.residual_blocks,
        "rag_node_feature_channels": cfg.partition.node_feature_channels,
        "rag_hidden_dim": cfg.partition.rag_hidden_dim,
        "rag_layers": cfg.partition.rag_layers,
        "max_supervoxels": cfg.partition.max_supervoxels,
        "watershed_backend": cfg.partition.watershed_backend,
        "supervoxel_guard_enabled": cfg.partition.supervoxel_guard_enabled,
        "instance_d_model": cfg.instances.d_model,
        "temporal_d_model": cfg.temporal.d_model,
        "temporal_graph_hidden_dim": cfg.temporal.graph_hidden_dim,
        "refinement_hidden_channels": cfg.refinement.hidden_channels,
    }


# =============================================================================
# FRESH SOURCE -> MERGE CROP
# =============================================================================


def _build_fresh_merge_crop(
    frame: int,
    crop_shape: tuple[int, int, int],
    training_cfg,
    work_root: Path,
):
    import numpy as np
    import torch
    from scipy import ndimage as ndi

    from learned.stirnet.data.sample_builder import (
        build_spatial_channels,
        robust_normalize,
    )
    from learned.stirnet.data.targets import estimate_model_dref_um
    from learned.stirnet.training.crops import CropSpec, prepare_crop_batch
    from learned.stirnet.training.merge_aware_crops import (
        build_merge_aware_crop_manifest,
    )
    from learned.stirnet.training.source_corruption import (
        apply_source_instance_dropout,
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
    raw_name = f"F22_{frame:03d}_image_0001.npy"
    gt_name = f"F22_{frame:03d}_masks_0001.npy"
    raw_source = source_dir / raw_name
    gt_source = source_dir / gt_name
    if not raw_source.exists():
        raise FileNotFoundError(raw_source)
    if not gt_source.exists():
        raise FileNotFoundError(gt_source)

    source_local = work_root / "source"
    source_local.mkdir(parents=True, exist_ok=True)
    raw_local = source_local / raw_name
    gt_local = source_local / gt_name

    timings: dict[str, float] = {}

    started = time.perf_counter()
    shutil.copy2(raw_source, raw_local)
    shutil.copy2(gt_source, gt_local)
    timings["source_copy_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    raw = np.load(raw_local, allow_pickle=False)
    gt = np.load(gt_local, allow_pickle=False)
    timings["source_load_seconds"] = time.perf_counter() - started

    if raw.ndim != 3 or gt.ndim != 3 or raw.shape != gt.shape:
        raise ValueError(
            f"invalid raw/GT source pair: raw={raw.shape}, gt={gt.shape}"
        )

    preprocessing_cfg = PreprocessingConfig(
        low_percentile=1.0,
        high_percentile=99.5,
        denoise_sigma_um=0.8,
        background_sigma_um=4.0,
        voxel_size_zyx_um=SPACING_ZYX_UM,
    )

    started = time.perf_counter()
    processed = preprocess_volume(
        raw,
        config=preprocessing_cfg,
        return_diagnostics=False,
    )
    timings["canonical_preprocessing_seconds"] = (
        time.perf_counter() - started
    )

    started = time.perf_counter()
    binary = create_binary_mask(
        processed,
        config=MaskingConfig(),
        return_diagnostics=False,
    )
    timings["canonical_masking_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    current, component_count = ndi.label(
        binary,
        structure=ndi.generate_binary_structure(3, 1),
    )
    current = current.astype(np.int32, copy=False)
    timings["connected_components_seconds"] = (
        time.perf_counter() - started
    )

    started = time.perf_counter()
    dref_um = float(
        estimate_model_dref_um(current, SPACING_ZYX_UM)
    )
    timings["dref_seconds"] = time.perf_counter() - started

    cfg = training_cfg.curriculum
    gt_tensor = torch.from_numpy(
        np.asarray(gt, dtype=np.int64)
    ).unsqueeze(0)
    current_tensor = torch.from_numpy(
        np.asarray(current, dtype=np.int64)
    ).unsqueeze(0)
    spacing_tensor = torch.tensor(
        [SPACING_ZYX_UM],
        dtype=torch.float32,
    )

    started = time.perf_counter()
    manifest = build_merge_aware_crop_manifest(
        gt_tensor,
        current_labels=current_tensor,
        spacing_um=spacing_tensor,
        crop_shape_zyx=crop_shape,
        min_complete_cells=cfg.refinement_crop_min_complete_cells,
        preferred_complete_cells=(
            cfg.refinement_crop_preferred_complete_cells
        ),
        views_per_cell=cfg.refinement_crop_views_per_cell,
        context_um=cfg.refinement_crop_context_um,
        merge_min_overlap_voxels=(
            cfg.refinement_crop_merge_min_overlap_voxels
        ),
        merge_min_gt_fraction=(
            cfg.refinement_crop_merge_min_gt_fraction
        ),
    )
    timings["crop_manifest_seconds"] = time.perf_counter() - started

    rows = list(manifest.records[0])
    if not rows:
        raise RuntimeError("merge-aware planner returned no crops")
    merge_rows = [row for row in rows if row.merge_source_ids]
    record = merge_rows[0] if merge_rows else rows[0]
    global_spec = _crop_spec_from_record(
        record,
        raw.shape,
        SPACING_ZYX_UM,
    )
    zyx = global_spec.slices_zyx

    # Compute normalization/marker in full acquisition coordinates, preserving
    # source semantics; construct expensive five-channel tensors only for crop.
    started = time.perf_counter()
    raw_norm_full = robust_normalize(raw)
    timings["raw_normalization_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    marker_full = _one_physical_edt_marker_per_component(
        current,
        SPACING_ZYX_UM,
    )
    timings["source_marker_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    current_crop = np.asarray(current[zyx], dtype=np.int64).copy()
    gt_crop = np.asarray(gt[zyx], dtype=np.int64).copy()
    raw_crop = np.asarray(raw_norm_full[zyx], dtype=np.float32)
    marker_crop = np.asarray(marker_full[zyx], dtype=np.float32)
    spatial = build_spatial_channels(
        raw_crop,
        current_crop,
        SPACING_ZYX_UM,
        dref_um,
        marker_crop,
    )
    timings["crop_channel_build_seconds"] = (
        time.perf_counter() - started
    )

    actual_shape = tuple(int(v) for v in current_crop.shape)
    base_batch = {
        "spatial_inputs": torch.from_numpy(spatial).unsqueeze(0),
        "instance_labels": torch.from_numpy(current_crop).unsqueeze(0),
        "spacing_um": spacing_tensor,
        "dref_um": torch.tensor([dref_um], dtype=torch.float32),
    }

    local_spec = CropSpec(
        batch_index=0,
        slices_zyx=tuple(slice(0, size) for size in actual_shape),
        full_shape_zyx=actual_shape,
        center_shift_um=torch.zeros(3),
        candidate_type=global_spec.candidate_type,
        complete_cell_ids=global_spec.complete_cell_ids,
        partial_cell_ids=global_spec.partial_cell_ids,
        true_boundary_cell_ids=global_spec.true_boundary_cell_ids,
        merge_source_ids=global_spec.merge_source_ids,
    )

    crop = prepare_crop_batch(
        base_batch,
        torch.from_numpy(gt_crop).unsqueeze(0),
        [local_spec],
        partial_ignore_margin_um=(
            cfg.refinement_crop_partial_ignore_margin_um
        ),
    )
    crop = apply_source_instance_dropout(
        crop,
        probability=cfg.refinement_crop_source_dropout_probability,
        max_instances=cfg.refinement_crop_source_dropout_max_instances,
        seed=cfg.refinement_crop_seed,
    )

    valid = torch.as_tensor(
        crop.batch["supervision_valid_mask"]
    ).bool()

    scene = {
        "frame": frame,
        "full_shape_zyx": list(raw.shape),
        "full_voxel_count": int(np.prod(raw.shape)),
        "spacing_zyx_um": list(SPACING_ZYX_UM),
        "current_instance_count": int(component_count),
        "gt_instance_count": int(np.unique(gt[gt > 0]).size),
        "dref_um": dref_um,
        "manifest_record_count": len(rows),
        "manifest_merge_record_count": len(merge_rows),
        "uncoverable_cell_ids": list(
            manifest.uncoverable_cell_ids[0]
        ),
        "uncoverable_merge_source_ids": list(
            manifest.uncoverable_merge_source_ids[0]
        ),
    }
    crop_info = {
        "requested_shape_zyx": list(crop_shape),
        "actual_shape_zyx": list(actual_shape),
        "voxel_count": int(np.prod(actual_shape)),
        "physical_extent_zyx_um": (
            np.asarray(actual_shape)
            * np.asarray(SPACING_ZYX_UM)
        ).tolist(),
        "candidate_type": global_spec.candidate_type,
        "global_slices_zyx": [
            [int(axis.start), int(axis.stop)]
            for axis in global_spec.slices_zyx
        ],
        "complete_cell_ids": list(global_spec.complete_cell_ids),
        "partial_cell_ids": list(global_spec.partial_cell_ids),
        "true_boundary_cell_ids": list(
            global_spec.true_boundary_cell_ids
        ),
        "merge_source_ids": list(global_spec.merge_source_ids),
        "source_dropout_ids": [
            list(row)
            for row in crop.batch.get("source_dropout_ids", ())
        ],
        "supervision_valid_fraction": float(valid.float().mean()),
    }

    del (
        raw,
        gt,
        processed,
        binary,
        current,
        raw_norm_full,
        marker_full,
        gt_tensor,
        current_tensor,
    )
    gc.collect()

    return crop, scene, crop_info, timings


# =============================================================================
# REMOTE STAGE 13
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
def run_stage13(
    frame: int = DEFAULT_FRAME,
    crop_z: int = 32,
    crop_y: int = 192,
    crop_x: int = 192,
) -> dict[str, Any]:
    import numpy as np
    import torch

    sys.path.insert(0, REMOTE_REPO_ROOT)

    from learned.stirnet import (
        SpatialForwardOutput,
        StirNet,
        StirNetConfig,
    )
    from learned.stirnet.model.geometry.edt_backend import (
        cupy_edt_available,
        release_cupy_memory,
    )
    from learned.stirnet.training import TrainingConfig
    from learned.stirnet.training.prepared_geometry import (
        build_prepared_geometry_targets,
    )
    from learned.stirnet.training.trainer import (
        Trainer,
        model_forward_from_batch,
        move_batch_to_device,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 13 Modal function did not receive a CUDA GPU")

    device = torch.device("cuda")
    crop_shape = (int(crop_z), int(crop_y), int(crop_x))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(RUNS_PREFIX) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    profile_path = run_dir / "gpu_profile.jsonl"

    work_root = Path("/tmp/stirnet_stage13")
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)

    cfg = StirNetConfig()  # FULL WIDTH: no reductions.
    cfg.validate()
    if cfg.partition.watershed_backend != "fast":
        raise RuntimeError(
            "Stage 13 expects the project production watershed backend='fast'"
        )

    train_cfg = TrainingConfig()
    train_cfg.amp_dtype = "bf16"
    train_cfg.geometry_target_backend = "auto"
    train_cfg.curriculum.fixed_stage = "spatial_partition"
    # We select one crop explicitly below. Avoid nested re-cropping.
    train_cfg.curriculum.geometry_bootstrap_crop_enabled = False
    train_cfg.curriculum.spatial_partition_crop_enabled = False
    train_cfg.profile_memory = True
    train_cfg.memory_profile_path = str(profile_path)
    train_cfg.validate()

    print("\n" + "=" * 116)
    print("STIR-Net Stage 13 — Modal FULL-WIDTH cropped forward/backward")
    print("=" * 116)
    print(f"GPU              : {torch.cuda.get_device_name(device)}")
    free, total = torch.cuda.mem_get_info(device)
    print(
        f"GPU memory       : total={total/1024**3:.2f} GiB "
        f"free={free/1024**3:.2f} GiB"
    )
    print(f"torch            : {torch.__version__}")
    print(f"CUDA             : {torch.version.cuda}")
    print(f"CuPy EDT         : {cupy_edt_available()}")
    print(f"frame            : {frame}")
    print(f"crop budget      : {crop_shape}")
    print(f"watershed        : {cfg.partition.watershed_backend}")
    print(f"output           : {run_dir}")
    print("=" * 116, flush=True)

    summary: dict[str, Any] = {
        "status": "STARTED",
        "stage": "stage13_full_width_cropped_train_step_modal",
        "frame": frame,
        "crop_shape_zyx": list(crop_shape),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cupy_edt_available": bool(cupy_edt_available()),
        "model_config": _full_width_config_summary(cfg),
    }

    try:
        # ---------------------------------------------------------------------
        # Fresh source + greedy merge crop
        # ---------------------------------------------------------------------
        crop_started = time.perf_counter()
        crop, scene, crop_info, cpu_timings = _build_fresh_merge_crop(
            frame,
            crop_shape,
            train_cfg,
            work_root,
        )
        crop_total_seconds = time.perf_counter() - crop_started

        summary["scene"] = scene
        summary["crop"] = crop_info
        summary["source_preparation_timing"] = cpu_timings
        summary["crop_total_prepare_seconds"] = crop_total_seconds

        print(
            "[crop] "
            f"type={crop_info['candidate_type']} "
            f"shape={tuple(crop_info['actual_shape_zyx'])} "
            f"voxels={crop_info['voxel_count']:,} "
            f"complete={len(crop_info['complete_cell_ids'])} "
            f"partial={len(crop_info['partial_cell_ids'])} "
            f"merge_sources={crop_info['merge_source_ids']} "
            f"valid={crop_info['supervision_valid_fraction']:.4f}",
            flush=True,
        )

        # ---------------------------------------------------------------------
        # Crop-only geometry targets
        # ---------------------------------------------------------------------
        target_started = time.perf_counter()
        geometry_targets = build_prepared_geometry_targets(
            crop.gt_labels,
            crop.batch["spacing_um"],
            crop.batch["dref_um"],
            current_labels=crop.batch.get("instance_labels"),
            geometry_config=cfg.geometry,
            backend=train_cfg.geometry_target_backend,
            gpu_min_voxels=train_cfg.geometry_target_gpu_min_voxels,
            device=torch.device("cpu"),
        )
        target_seconds = time.perf_counter() - target_started
        summary["geometry_target_prepare_seconds"] = target_seconds

        print(
            f"[targets] auto backend | CuPy={cupy_edt_available()} | "
            f"{target_seconds:.3f}s",
            flush=True,
        )

        release_cupy_memory()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # ---------------------------------------------------------------------
        # Full-width model
        # ---------------------------------------------------------------------
        model_started = time.perf_counter()
        model = StirNet(cfg)
        trainer = Trainer(model, train_cfg, device=device)
        model_setup_seconds = time.perf_counter() - model_started
        summary["model_trainer_setup_seconds"] = model_setup_seconds

        if trainer.curriculum_stage.name != "spatial_partition":
            raise RuntimeError(
                "Stage 13 expected fixed spatial_partition curriculum"
            )

        total_params, trainable_params = _parameter_count(trainer.model)
        summary["parameters"] = {
            "total": total_params,
            "trainable": trainable_params,
        }

        moved = move_batch_to_device(crop.batch, device)
        trainer.model.train()
        trainer.criterion.train()
        trainer.optimizer.zero_grad(set_to_none=True)

        trainer.stage_profiler.clear()
        trainer.stage_profiler.set_context(
            global_step=0,
            refinement_stage_step=0,
            phase="stage13",
            metadata={
                "full_width": True,
                "crop_shape_zyx": crop_info["actual_shape_zyx"],
                "candidate_type": crop_info["candidate_type"],
            },
        )

        # One OUTER profiler scope stays active across the complete train step.
        # Therefore nested forward/checkpoint/criterion/backward scopes cannot
        # reset CUDA peak statistics between phases.
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        timings: dict[str, float] = {}
        step_started = time.perf_counter()

        with trainer.stage_profiler.phase_scope("stage13"):
            with trainer.stage_profiler.profile(
                "whole_train_step",
                qualify=False,
            ):
                started = time.perf_counter()
                with trainer.stage_profiler.profile("forward"):
                    with trainer._autocast():
                        output = model_forward_from_batch(
                            trainer.model,
                            moved,
                            use_temporal=False,
                            execution_stage="spatial",
                            apply_existence_filter=False,
                            return_debug=False,
                            stage_profiler=trainer.stage_profiler,
                        )
                _sync(device)
                timings["forward_seconds"] = time.perf_counter() - started

                if not isinstance(output, SpatialForwardOutput):
                    raise TypeError(
                        "Stage 13 expected SpatialForwardOutput"
                    )

                started = time.perf_counter()
                with trainer.stage_profiler.profile("criterion"):
                    with trainer._autocast():
                        metrics = trainer.criterion(
                            output,
                            crop.gt_labels,
                            moved["spacing_um"],
                            moved["dref_um"],
                            stage="spatial_partition",
                            current_labels=crop.batch.get(
                                "instance_labels"
                            ),
                            precomputed_geometry_targets=geometry_targets,
                            supervision_valid_mask=crop.batch.get(
                                "supervision_valid_mask"
                            ),
                        )
                        loss = metrics["loss"]
                _sync(device)
                timings["criterion_seconds"] = (
                    time.perf_counter() - started
                )

                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"non-finite Stage-13 loss: {loss.detach()}"
                    )

                started = time.perf_counter()
                with trainer.stage_profiler.profile("backward"):
                    trainer.scaler.scale(loss).backward()
                _sync(device)
                timings["backward_seconds"] = (
                    time.perf_counter() - started
                )

                started = time.perf_counter()
                with trainer.stage_profiler.profile("optimizer"):
                    trainer.scaler.unscale_(trainer.optimizer)
                    gradient_groups = _gradient_group_norms(
                        trainer.model
                    )
                    total_grad = torch.nn.utils.clip_grad_norm_(
                        trainer.model.parameters(),
                        train_cfg.max_grad_norm,
                    )
                    if not bool(
                        torch.isfinite(torch.as_tensor(total_grad))
                    ):
                        raise FloatingPointError(
                            f"non-finite gradient norm: {total_grad}"
                        )
                    trainer.scaler.step(trainer.optimizer)
                    trainer.scaler.update()
                _sync(device)
                timings["optimizer_seconds"] = (
                    time.perf_counter() - started
                )

        timings["whole_train_step_seconds"] = (
            time.perf_counter() - step_started
        )

        cuda = _cuda_snapshot(device)
        cuda["profiler_overall_peak_allocated_mb"] = (
            trainer.stage_profiler.overall_peak_allocated_mb
        )
        profile_summary = trainer.stage_profiler.summary()

        supervoxel_count = sum(
            int(labels.max().item())
            for labels in output.rag.supervoxel_labels
        )
        diagnostics = {
            "output_type": type(output).__name__,
            "supervoxel_count": int(supervoxel_count),
            "rag_node_count": int(output.rag.node_features.shape[0]),
            "rag_edge_count": int(output.rag.edge_index.shape[1]),
            "partition_component_count": int(
                output.spatial_partition.component_count_per_batch
                .sum()
                .item()
            ),
        }

        loss_metrics = {
            key: float(value.detach().float().cpu())
            for key, value in metrics.items()
        }

        summary.update(
            {
                "status": "PASS",
                "timing": timings,
                "loss_metrics": loss_metrics,
                "gradient_norm_before_clip": float(
                    torch.as_tensor(total_grad).detach().cpu()
                ),
                "gradient_group_norms": gradient_groups,
                "cuda_memory": cuda,
                "diagnostics": diagnostics,
                "profile_summary": profile_summary,
                "profile_path": str(profile_path),
            }
        )
        _write_json(run_dir / "summary.json", summary)
        runs_volume.commit()

        print("\n" + "=" * 116)
        print("STAGE 13 MODAL PASS")
        print("=" * 116)
        print(
            f"Full-width spatial_partition completed on {MODAL_GPU}."
        )
        print(
            f"Crop              : {tuple(crop_info['actual_shape_zyx'])} "
            f"({crop_info['voxel_count']:,} voxels), "
            f"type={crop_info['candidate_type']}"
        )
        print(
            f"Geometry targets  : {target_seconds:.3f} s"
        )
        print(
            f"Forward           : {timings['forward_seconds']:.3f} s"
        )
        print(
            f"Criterion         : {timings['criterion_seconds']:.3f} s"
        )
        print(
            f"Backward          : {timings['backward_seconds']:.3f} s"
        )
        print(
            f"Optimizer         : {timings['optimizer_seconds']:.3f} s"
        )
        print(
            f"Whole train step  : "
            f"{timings['whole_train_step_seconds']:.3f} s"
        )
        print(
            f"Peak allocated    : "
            f"{cuda['profiler_overall_peak_allocated_mb']/1024:.3f} GiB"
        )
        print(
            f"Peak reserved     : "
            f"{cuda['max_reserved_mb']/1024:.3f} GiB"
        )
        print(
            f"Supervoxels       : {diagnostics['supervoxel_count']}"
        )
        print(
            f"RAG nodes / edges : "
            f"{diagnostics['rag_node_count']} / "
            f"{diagnostics['rag_edge_count']}"
        )
        print(f"Summary           : {run_dir / 'summary.json'}")
        print("=" * 116, flush=True)

        return summary

    except BaseException as error:
        summary.update(
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "cuda_memory_at_failure": _cuda_snapshot(
                    torch.device("cuda")
                ),
            }
        )
        try:
            _write_json(run_dir / "summary.json", summary)
            runs_volume.commit()
        except BaseException:
            pass
        print("\n" + "=" * 116)
        print("STAGE 13 MODAL FAILED")
        print("=" * 116)
        print(f"{type(error).__name__}: {error}")
        print(traceback.format_exc())
        print("=" * 116, flush=True)
        raise


@app.local_entrypoint()
def main(
    frame: int = DEFAULT_FRAME,
    crop_z: int = 32,
    crop_y: int = 192,
    crop_x: int = 192,
):
    result = run_stage13.remote(
        frame=frame,
        crop_z=crop_z,
        crop_y=crop_y,
        crop_x=crop_x,
    )
    print(
        "\nModal Stage-13 returned:\n"
        + json.dumps(
            {
                "status": result.get("status"),
                "crop": result.get("crop", {}).get("actual_shape_zyx"),
                "candidate_type": result.get("crop", {}).get(
                    "candidate_type"
                ),
                "forward_seconds": result.get("timing", {}).get(
                    "forward_seconds"
                ),
                "backward_seconds": result.get("timing", {}).get(
                    "backward_seconds"
                ),
                "peak_allocated_gib": (
                    result.get("cuda_memory", {}).get(
                        "profiler_overall_peak_allocated_mb",
                        0.0,
                    )
                    / 1024
                ),
                "peak_reserved_gib": (
                    result.get("cuda_memory", {}).get(
                        "max_reserved_mb",
                        0.0,
                    )
                    / 1024
                ),
                "supervoxels": result.get("diagnostics", {}).get(
                    "supervoxel_count"
                ),
                "rag_nodes": result.get("diagnostics", {}).get(
                    "rag_node_count"
                ),
                "rag_edges": result.get("diagnostics", {}).get(
                    "rag_edge_count"
                ),
            },
            indent=2,
        )
    )
