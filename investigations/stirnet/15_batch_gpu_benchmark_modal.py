from __future__ import annotations

"""STIR-Net Stage 15: Modal true-batch benchmark.

Benchmarks exactly:
  1) L4   + BF16 + batch=2
  2) L40S + BF16 + batch=4

Both use the current full-width STIR-Net, fresh BlastoSPIM preprocessing,
merge-aware 32x192x192 crops, crop-only geometry targets, production fast
watershed/safety-guard/RAG, and one true batched forward+loss+backward+optimizer
step. This is tensor batching, not sequential gradient accumulation.

Run:
  modal run investigations/stirnet/15_batch_gpu_benchmark_modal.py

Optional:
  modal run investigations/stirnet/15_batch_gpu_benchmark_modal.py --warmup 1 --repeats 3
"""

import gc
import json
import math
import shutil
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

import modal

app = modal.App("stirnet-stage15-batch-gpu-benchmark")
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
    "stage15_batch_gpu_benchmark"
)
SPACING_ZYX_UM = (2.0, 0.208, 0.208)
DEFAULT_FRAME = 32
DEFAULT_CROP = (32, 192, 192)
SEED = 40266


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "learned").is_dir() and (candidate / "src").is_dir():
            return candidate
    return Path.cwd()


LOCAL_REPO_ROOT = _repo_root()
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.13.0",
        "numpy==2.4.6",
        "scipy==1.17.1",
        "scikit-image==0.26.0",
        "networkx>=3.0",
        "psutil>=6.0",
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


def _jsonable(v: Any) -> Any:
    import numpy as np
    import torch
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if torch.is_tensor(v):
        return v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().tolist()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _sync(device) -> None:
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cleanup_cuda() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def _cuda(device) -> dict[str, float]:
    import torch
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        "free_mb": free / 1024**2,
        "total_mb": total / 1024**2,
    }


def _is_oom(e: BaseException) -> bool:
    import torch
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def _group_grad_norms(model) -> dict[str, float]:
    import torch
    from learned.stirnet.training.curriculum import model_parameter_groups
    out = {}
    for name, params in model_parameter_groups(model).items():
        norm = 0.0
        for p in params:
            if p.grad is not None:
                norm = math.hypot(
                    norm,
                    float(torch.linalg.vector_norm(p.grad.detach().float()).cpu()),
                )
        out[name] = norm
    return out


def _marker(labels, spacing):
    import numpy as np
    from scipy import ndimage as ndi
    marker = np.zeros(labels.shape, dtype=np.float32)
    for instance_id, slc in enumerate(ndi.find_objects(labels), start=1):
        if slc is None:
            continue
        expanded = tuple(
            slice(max(0, s.start - 1), min(labels.shape[d], s.stop + 1))
            for d, s in enumerate(slc)
        )
        component = labels[expanded] == instance_id
        if not component.any():
            continue
        edt = ndi.distance_transform_edt(component, sampling=spacing)
        local = np.unravel_index(np.argmax(edt), edt.shape)
        pos = tuple(int(expanded[d].start) + int(local[d]) for d in range(3))
        marker[pos] = 1.0
    return marker


def _global_spec(record, full_shape):
    import torch
    from learned.stirnet.training.crops import CropSpec
    lower = torch.tensor([s.start for s in record.slices_zyx], dtype=torch.float32)
    size = torch.tensor([s.stop - s.start for s in record.slices_zyx], dtype=torch.float32)
    full_center = 0.5 * (torch.tensor(full_shape, dtype=torch.float32) - 1)
    crop_center = lower + 0.5 * (size - 1)
    shift = (crop_center - full_center) * torch.tensor(SPACING_ZYX_UM)
    return CropSpec(
        batch_index=0,
        slices_zyx=record.slices_zyx,
        full_shape_zyx=tuple(full_shape),
        center_shift_um=shift,
        candidate_type=record.candidate_type,
        complete_cell_ids=record.complete_cell_ids,
        partial_cell_ids=record.partial_cell_ids,
        true_boundary_cell_ids=record.true_boundary_cell_ids,
        merge_source_ids=record.merge_source_ids,
    )


def _select_records(manifest, batch_size: int):
    rows, seen, unique = list(manifest.records[0]), set(), []
    for row in rows:
        key = tuple((s.start, s.stop) for s in row.slices_zyx)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    if not unique:
        raise RuntimeError("merge-aware manifest produced no crops")
    selected = [unique[i % len(unique)] for i in range(batch_size)]
    return selected, len(unique) < batch_size, len(unique)


def _prepare_batch(frame: int, batch_size: int, crop_shape, train_cfg, work_root: Path):
    import numpy as np
    import torch
    from scipy import ndimage as ndi
    from learned.stirnet.data.sample_builder import build_spatial_channels, robust_normalize
    from learned.stirnet.data.targets import estimate_model_dref_um
    from learned.stirnet.training.crops import CropSpec, prepare_crop_batch
    from learned.stirnet.training.merge_aware_crops import build_merge_aware_crop_manifest
    from learned.stirnet.training.source_corruption import apply_source_instance_dropout

    preprocess_volume = import_module("src.01_preprocessing.pipeline").preprocess_volume
    PreprocessingConfig = import_module("src.01_preprocessing.config").PreprocessingConfig
    create_binary_mask = import_module("src.02_masking.pipeline").create_binary_mask
    MaskingConfig = import_module("src.02_masking.config").MaskingConfig

    raw_name = f"F22_{frame:03d}_image_0001.npy"
    gt_name = f"F22_{frame:03d}_masks_0001.npy"
    raw_src, gt_src = Path(SOURCE_DIR) / raw_name, Path(SOURCE_DIR) / gt_name
    if not raw_src.exists() or not gt_src.exists():
        raise FileNotFoundError(f"missing source pair: {raw_src}, {gt_src}")
    local = work_root / "source"
    local.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_src, local / raw_name)
    shutil.copy2(gt_src, local / gt_name)
    raw = np.load(local / raw_name, allow_pickle=False)
    gt = np.load(local / gt_name, allow_pickle=False)

    prep_cfg = PreprocessingConfig(
        low_percentile=1.0,
        high_percentile=99.5,
        denoise_sigma_um=0.8,
        background_sigma_um=4.0,
        voxel_size_zyx_um=SPACING_ZYX_UM,
    )
    processed = preprocess_volume(raw, config=prep_cfg, return_diagnostics=False)
    binary = create_binary_mask(processed, config=MaskingConfig(), return_diagnostics=False)
    current, component_count = ndi.label(binary, structure=ndi.generate_binary_structure(3, 1))
    current = current.astype(np.int32, copy=False)
    dref = float(estimate_model_dref_um(current, SPACING_ZYX_UM))

    cfg = train_cfg.curriculum
    gt_full = torch.from_numpy(gt.astype(np.int64, copy=False))[None]
    current_full = torch.from_numpy(current.astype(np.int64, copy=False))[None]
    spacing1 = torch.tensor([SPACING_ZYX_UM], dtype=torch.float32)
    manifest = build_merge_aware_crop_manifest(
        gt_full,
        current_labels=current_full,
        spacing_um=spacing1,
        crop_shape_zyx=tuple(crop_shape),
        min_complete_cells=cfg.refinement_crop_min_complete_cells,
        preferred_complete_cells=cfg.refinement_crop_preferred_complete_cells,
        views_per_cell=cfg.refinement_crop_views_per_cell,
        context_um=cfg.refinement_crop_context_um,
        merge_min_overlap_voxels=cfg.refinement_crop_merge_min_overlap_voxels,
        merge_min_gt_fraction=cfg.refinement_crop_merge_min_gt_fraction,
    )
    records, duplicated, unique_count = _select_records(manifest, batch_size)
    specs = [_global_spec(r, raw.shape) for r in records]

    raw_norm, marker = robust_normalize(raw), _marker(current, SPACING_ZYX_UM)
    spatial_rows, current_rows, gt_rows, crop_info = [], [], [], []
    for b, spec in enumerate(specs):
        zyx = spec.slices_zyx
        cur = np.asarray(current[zyx], dtype=np.int64).copy()
        gtc = np.asarray(gt[zyx], dtype=np.int64).copy()
        spatial = build_spatial_channels(
            np.asarray(raw_norm[zyx], dtype=np.float32),
            cur,
            SPACING_ZYX_UM,
            dref,
            np.asarray(marker[zyx], dtype=np.float32),
        )
        if tuple(spatial.shape[-3:]) != tuple(crop_shape):
            raise RuntimeError(f"non-uniform crop shape {spatial.shape[-3:]} != {crop_shape}")
        spatial_rows.append(torch.from_numpy(spatial))
        current_rows.append(torch.from_numpy(cur))
        gt_rows.append(torch.from_numpy(gtc))
        crop_info.append({
            "batch_index": b,
            "candidate_type": spec.candidate_type,
            "merge_source_ids": list(spec.merge_source_ids),
            "complete_cell_ids": list(spec.complete_cell_ids),
            "partial_cell_ids": list(spec.partial_cell_ids),
            "global_slices_zyx": [[s.start, s.stop] for s in spec.slices_zyx],
        })

    base = {
        "spatial_inputs": torch.stack(spatial_rows),
        "instance_labels": torch.stack(current_rows),
        "spacing_um": torch.tensor([SPACING_ZYX_UM] * batch_size, dtype=torch.float32),
        "dref_um": torch.full((batch_size,), dref, dtype=torch.float32),
    }
    gt_batch = torch.stack(gt_rows)
    local_specs = [
        CropSpec(
            batch_index=b,
            slices_zyx=tuple(slice(0, n) for n in crop_shape),
            full_shape_zyx=tuple(crop_shape),
            center_shift_um=torch.zeros(3),
            candidate_type=spec.candidate_type,
            complete_cell_ids=spec.complete_cell_ids,
            partial_cell_ids=spec.partial_cell_ids,
            true_boundary_cell_ids=spec.true_boundary_cell_ids,
            merge_source_ids=spec.merge_source_ids,
        )
        for b, spec in enumerate(specs)
    ]
    crop_batch = prepare_crop_batch(
        base,
        gt_batch,
        local_specs,
        partial_ignore_margin_um=cfg.refinement_crop_partial_ignore_margin_um,
    )
    crop_batch = apply_source_instance_dropout(
        crop_batch,
        probability=cfg.refinement_crop_source_dropout_probability,
        max_instances=cfg.refinement_crop_source_dropout_max_instances,
        seed=cfg.refinement_crop_seed,
    )
    valid = crop_batch.batch["supervision_valid_mask"].bool()
    dropped = crop_batch.batch.get("source_dropout_ids", tuple(() for _ in range(batch_size)))
    for b in range(batch_size):
        crop_info[b]["supervision_valid_fraction"] = float(valid[b].float().mean())
        crop_info[b]["source_dropout_ids"] = list(dropped[b])

    scene = {
        "frame": frame,
        "full_shape_zyx": list(raw.shape),
        "current_instance_count": int(component_count),
        "gt_instance_count": int(np.unique(gt[gt > 0]).size),
        "dref_um": dref,
        "manifest_records": len(manifest.records[0]),
        "manifest_unique_records": unique_count,
        "manifest_merge_records": sum(bool(r.merge_source_ids) for r in manifest.records[0]),
        "duplicated_crops_for_batch": duplicated,
    }
    del raw, gt, processed, binary, current, raw_norm, marker, gt_full, current_full
    gc.collect()
    return crop_batch, scene, crop_info


def _run_step(trainer, moved, crop_batch, geometry_targets, label: str):
    import torch
    from learned.stirnet import SpatialForwardOutput
    from learned.stirnet.training.trainer import model_forward_from_batch

    device = trainer.device
    trainer.optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    timing = {}
    whole = time.perf_counter()
    with trainer.stage_profiler.phase_scope("stage15"):
        with trainer.stage_profiler.profile(f"whole_{label}", qualify=False):
            t = time.perf_counter()
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
            _sync(device); timing["forward_seconds"] = time.perf_counter() - t
            if not isinstance(output, SpatialForwardOutput):
                raise TypeError("expected SpatialForwardOutput")

            t = time.perf_counter()
            with trainer.stage_profiler.profile("criterion"):
                with trainer._autocast():
                    metrics = trainer.criterion(
                        output,
                        crop_batch.gt_labels,
                        moved["spacing_um"],
                        moved["dref_um"],
                        stage="spatial_partition",
                        current_labels=crop_batch.batch.get("instance_labels"),
                        precomputed_geometry_targets=geometry_targets,
                        supervision_valid_mask=crop_batch.batch.get("supervision_valid_mask"),
                    )
                    loss = metrics["loss"]
            _sync(device); timing["criterion_seconds"] = time.perf_counter() - t
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss: {loss.detach()}")

            t = time.perf_counter()
            with trainer.stage_profiler.profile("backward"):
                trainer.scaler.scale(loss).backward()
            _sync(device); timing["backward_seconds"] = time.perf_counter() - t

            t = time.perf_counter()
            with trainer.stage_profiler.profile("optimizer"):
                trainer.scaler.unscale_(trainer.optimizer)
                grad_groups = _group_grad_norms(trainer.model)
                grad = torch.nn.utils.clip_grad_norm_(
                    trainer.model.parameters(), trainer.training_config.max_grad_norm
                )
                if not bool(torch.isfinite(torch.as_tensor(grad))):
                    raise FloatingPointError(f"non-finite grad norm: {grad}")
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
            _sync(device); timing["optimizer_seconds"] = time.perf_counter() - t

    timing["whole_step_seconds"] = time.perf_counter() - whole
    sv = [int(x.max().item()) for x in output.rag.supervoxel_labels]
    edges = [
        int((output.rag.edge_batch == b).sum().item())
        for b in range(len(sv))
    ]
    row = {
        "timing": timing,
        "cuda_memory": _cuda(device),
        "loss_metrics": {k: float(v.detach().float().cpu()) for k, v in metrics.items()},
        "gradient_norm_before_clip": float(torch.as_tensor(grad).detach().cpu()),
        "gradient_group_norms": grad_groups,
        "supervoxels_per_crop": sv,
        "rag_edges_per_crop": edges,
        "rag_nodes_total": int(output.rag.node_features.shape[0]),
        "rag_edges_total": int(output.rag.edge_index.shape[1]),
    }
    del output, metrics, loss
    trainer.optimizer.zero_grad(set_to_none=True)
    return row


def _benchmark(gpu_label: str, batch_size: int, run_id: str, frame: int,
               crop_z: int, crop_y: int, crop_x: int, warmup: int, repeats: int):
    import numpy as np
    import torch
    sys.path.insert(0, REMOTE_REPO_ROOT)
    from learned.stirnet import StirNet, StirNetConfig
    from learned.stirnet.model.geometry.edt_backend import cupy_edt_available, release_cupy_memory
    from learned.stirnet.training import TrainingConfig
    from learned.stirnet.training.prepared_geometry import build_prepared_geometry_targets
    from learned.stirnet.training.trainer import Trainer, move_batch_to_device

    torch.manual_seed(SEED); np.random.seed(SEED); torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    crop_shape = (crop_z, crop_y, crop_x)
    out_dir = Path(RUNS_PREFIX) / run_id / f"{gpu_label.lower()}_b{batch_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = out_dir / "gpu_profile.jsonl"
    work = Path(f"/tmp/stirnet_stage15_{gpu_label.lower()}_b{batch_size}")
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)

    cfg = StirNetConfig(); cfg.validate()
    tc = TrainingConfig()
    tc.amp_dtype = "bf16"
    tc.geometry_target_backend = "auto"
    tc.curriculum.fixed_stage = "spatial_partition"
    tc.curriculum.geometry_bootstrap_crop_enabled = False
    tc.curriculum.spatial_partition_crop_enabled = False
    tc.profile_memory = True
    tc.memory_profile_path = str(profile_path)
    tc.validate()

    summary = {
        "status": "STARTED",
        "gpu_label": gpu_label,
        "actual_gpu_name": torch.cuda.get_device_name(device),
        "batch_size": batch_size,
        "precision": "bf16",
        "crop_shape_zyx": list(crop_shape),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cupy_edt_available": bool(cupy_edt_available()),
    }
    print("\n" + "="*110)
    print(f"STAGE 15 | {gpu_label} | TRUE BATCH={batch_size} | BF16 | crop={crop_shape}")
    print("="*110)
    free, total = torch.cuda.mem_get_info(device)
    print(f"GPU: {summary['actual_gpu_name']} | total={total/1024**3:.2f} GiB | free={free/1024**3:.2f} GiB")

    try:
        if not torch.cuda.is_bf16_supported():
            summary["status"] = "BF16_UNSUPPORTED"
            _write_json(out_dir / "summary.json", summary); runs_volume.commit(); return summary

        t = time.perf_counter()
        crop_batch, scene, crop_info = _prepare_batch(frame, batch_size, crop_shape, tc, work)
        summary["batch_prepare_seconds"] = time.perf_counter() - t
        summary["scene"] = scene; summary["crops"] = crop_info
        print("[crops] " + " | ".join(
            f"b{x['batch_index']}:{x['candidate_type']}/merge={x['merge_source_ids']}/C={len(x['complete_cell_ids'])}/P={len(x['partial_cell_ids'])}/valid={x['supervision_valid_fraction']:.3f}"
            for x in crop_info
        ), flush=True)

        t = time.perf_counter()
        geometry_targets = build_prepared_geometry_targets(
            crop_batch.gt_labels,
            crop_batch.batch["spacing_um"],
            crop_batch.batch["dref_um"],
            current_labels=crop_batch.batch.get("instance_labels"),
            geometry_config=cfg.geometry,
            backend=tc.geometry_target_backend,
            gpu_min_voxels=tc.geometry_target_gpu_min_voxels,
            device=torch.device("cpu"),
        )
        summary["geometry_target_seconds"] = time.perf_counter() - t
        print(f"[targets] {summary['geometry_target_seconds']:.3f}s", flush=True)
        release_cupy_memory(); _cleanup_cuda()

        trainer = Trainer(StirNet(cfg), tc, device=device)
        if trainer.curriculum_stage.name != "spatial_partition":
            raise RuntimeError("wrong curriculum stage")
        moved = move_batch_to_device(crop_batch.batch, device)
        trainer.model.train(); trainer.criterion.train()
        trainer.stage_profiler.set_context(
            global_step=0, refinement_stage_step=0, phase="stage15",
            metadata={"gpu": gpu_label, "batch_size": batch_size, "crop_shape": list(crop_shape)},
        )
        summary["parameters"] = {
            "total": sum(p.numel() for p in trainer.model.parameters()),
            "trainable": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        }

        warm = []
        for i in range(warmup):
            row = _run_step(trainer, moved, crop_batch, geometry_targets, f"warmup_{i+1}")
            warm.append(row)
            print(f"[warmup {i+1}] {row['timing']['whole_step_seconds']:.3f}s", flush=True)

        measured = []
        for i in range(repeats):
            row = _run_step(trainer, moved, crop_batch, geometry_targets, f"measure_{i+1}")
            measured.append(row)
            m = row["cuda_memory"]
            print(
                f"[measure {i+1}] step={row['timing']['whole_step_seconds']:.3f}s | "
                f"peak_alloc={m['peak_allocated_mb']/1024:.3f}GiB | peak_res={m['peak_reserved_mb']/1024:.3f}GiB",
                flush=True,
            )

        steps = [r["timing"]["whole_step_seconds"] for r in measured]
        forward = [r["timing"]["forward_seconds"] for r in measured]
        backward = [r["timing"]["backward_seconds"] for r in measured]
        peak_a = [r["cuda_memory"]["peak_allocated_mb"] for r in measured]
        peak_r = [r["cuda_memory"]["peak_reserved_mb"] for r in measured]
        med = float(statistics.median(steps))
        aggregate = {
            "median_whole_step_seconds": med,
            "median_forward_seconds": float(statistics.median(forward)),
            "median_backward_seconds": float(statistics.median(backward)),
            "max_peak_allocated_gib": max(peak_a)/1024,
            "max_peak_reserved_gib": max(peak_r)/1024,
            "crops_per_second": batch_size/med,
            "seconds_per_crop_equivalent": med/batch_size,
            "voxels_per_second": batch_size*math.prod(crop_shape)/med,
        }
        summary.update({
            "status": "PASS",
            "warmup_results": warm,
            "measured_results": measured,
            "aggregate": aggregate,
            "profile_path": str(profile_path),
            "profile_summary": trainer.stage_profiler.summary(),
        })
        _write_json(out_dir / "summary.json", summary); runs_volume.commit()
        print(
            f"PASS {gpu_label} B{batch_size}: step={med:.3f}s | "
            f"throughput={aggregate['crops_per_second']:.3f} crops/s | "
            f"peak_res={aggregate['max_peak_reserved_gib']:.3f}GiB",
            flush=True,
        )
        return summary

    except BaseException as e:
        summary.update({
            "status": "OOM" if _is_oom(e) else "FAILED",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        })
        try: summary["cuda_memory_at_failure"] = _cuda(device)
        except Exception: pass
        try: _write_json(out_dir / "summary.json", summary); runs_volume.commit()
        except Exception: pass
        print(f"{summary['status']} {gpu_label} B{batch_size}: {type(e).__name__}: {e}", flush=True)
        _cleanup_cuda()
        return summary


@app.function(
    image=image, gpu="L4", cpu=8.0, memory=32768, timeout=3600,
    volumes={DATA_MOUNT: data_volume, RUNS_MOUNT: runs_volume},
)
def benchmark_l4_b2(run_id: str, frame: int, crop_z: int, crop_y: int, crop_x: int,
                    warmup: int, repeats: int):
    return _benchmark("L4", 2, run_id, frame, crop_z, crop_y, crop_x, warmup, repeats)


@app.function(
    image=image, gpu="L40S", cpu=8.0, memory=32768, timeout=3600,
    volumes={DATA_MOUNT: data_volume, RUNS_MOUNT: runs_volume},
)
def benchmark_l40s_b4(run_id: str, frame: int, crop_z: int, crop_y: int, crop_x: int,
                      warmup: int, repeats: int):
    return _benchmark("L40S", 4, run_id, frame, crop_z, crop_y, crop_x, warmup, repeats)


def _compact(r):
    a = r.get("aggregate", {})
    return {
        "status": r.get("status"),
        "gpu": r.get("gpu_label"),
        "actual_gpu": r.get("actual_gpu_name"),
        "batch": r.get("batch_size"),
        "median_step_s": a.get("median_whole_step_seconds"),
        "crops_per_s": a.get("crops_per_second"),
        "equivalent_s_per_crop": a.get("seconds_per_crop_equivalent"),
        "peak_allocated_gib": a.get("max_peak_allocated_gib"),
        "peak_reserved_gib": a.get("max_peak_reserved_gib"),
    }


@app.local_entrypoint()
def main(frame: int = DEFAULT_FRAME, crop_z: int = 32, crop_y: int = 192,
         crop_x: int = 192, warmup: int = 1, repeats: int = 2):
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup >= 0 and repeats >= 1 required")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args = dict(
        run_id=run_id, frame=frame, crop_z=crop_z, crop_y=crop_y, crop_x=crop_x,
        warmup=warmup, repeats=repeats,
    )
    print(f"Stage 15 run={run_id}; order: L4 B2 -> L40S B4", flush=True)
    try:
        l4 = benchmark_l4_b2.remote(**args)
    except BaseException as e:
        l4 = {"status": "REMOTE_CALL_FAILED", "gpu_label": "L4", "batch_size": 2,
              "error_type": type(e).__name__, "error_message": str(e)}
    try:
        l40s = benchmark_l40s_b4.remote(**args)
    except BaseException as e:
        l40s = {"status": "REMOTE_CALL_FAILED", "gpu_label": "L40S", "batch_size": 4,
                "error_type": type(e).__name__, "error_message": str(e)}

    comparison = {
        "run_id": run_id,
        "crop_shape_zyx": [crop_z, crop_y, crop_x],
        "warmup": warmup,
        "repeats": repeats,
        "l4_b2": _compact(l4),
        "l40s_b4": _compact(l40s),
    }
    if l4.get("status") == l40s.get("status") == "PASS":
        a, b = l4["aggregate"], l40s["aggregate"]
        comparison["derived"] = {
            "l40s_vs_l4_crop_throughput_ratio": b["crops_per_second"] / a["crops_per_second"],
            "l40s_vs_l4_step_time_ratio": b["median_whole_step_seconds"] / a["median_whole_step_seconds"],
        }
    local = Path(f"stage15_batch_gpu_comparison_{run_id}.json")
    _write_json(local, comparison)
    print("\n" + json.dumps(comparison, indent=2), flush=True)
    print(
        f"Modal results: stirnet-runs/stirnet/investigations/"
        f"stage15_batch_gpu_benchmark/{run_id}/",
        flush=True,
    )
    print(f"Local comparison: {local}", flush=True)
