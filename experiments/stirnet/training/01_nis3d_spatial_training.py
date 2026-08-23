from __future__ import annotations

"""
STIR-Net training 01 — NIS3D spatial training locally or on Modal.

This is the first production-style spatial training script after the component
investigations.  It intentionally keeps NIS3D-specific rules here rather than
putting them into the model/trainer.

What it does
------------
* Runs on Modal L40S, 4 CPU cores, 16 GiB host RAM.
* Loads NIS3D TIFF volumes from the existing `stirnet-data` Modal volume.
* Parses physical voxel spacing strictly from Info.txt.
* Uses ConfidenceScore==1 as undefined/unreliable supervision.
* Drops any GT object touched by an undefined region so a truncated annotation
  is never treated as a complete cell.
* Adds a small physical ignore margin around undefined supervision.
* Starts from RAW image + GT only; STIR-Net's raw-source pipeline constructs the
  current/source segmentation and the five model channels.
* Uses full-width STIR-Net with true B=4 focused crops.
* Keeps natural merge/under-segmentation cases untouched.
* Uses the production missing-source-cell augmentation on eligible coverage
  crops.
* Uses synthetic X/Y reflection augmentation (p=0.5 independently per axis)
  after crop/source preparation, covering identity/X/Y/XY orientations.
* Uses exact static GT crop caching and the existing CuPy EDT path.
* Trains geometry_bootstrap -> spatial_partition only; no temporal stage.
* Writes one durable scalar record after every successful optimizer step.
* Saves a recoverable checkpoint every N successful steps (default 50) and at
  normal completion.  Checkpoints are stored in a fixed recovery directory so
  a later run can resume after a crash.
* Shows live tqdm training progress with step count, elapsed time, ETA, average
  step time, total loss, stage, sample, crop batch, merge count and dropout count.
* Keeps CUDA memory profiling and the full metric dictionary in history.jsonl
  instead of printing memory diagnostics every optimizer step.

Recommended first runs
----------------------
Unified launcher (default is Modal):
    python experiments/stirnet/training/01_nis3d_spatial_training.py \
        --max-steps 1 --samples Zebrafish_2 \
        --run-name smoke_zebra_modal_1

Local 1-step smoke: same production 32x192x192 crop, B=1
    python experiments/stirnet/training/01_nis3d_spatial_training.py \
        --execution local --max-steps 1 --samples Zebrafish_2 \
        --run-name smoke_zebra_local_1

`modal run ...` remains supported for native Modal CLI use.

5-step smoke:
    modal run experiments/stirnet/training/01_nis3d_spatial_training.py \
        --max-steps 5 --samples Zebrafish_2 --run-name smoke_zebra_5

After both are clean, a first spatial run can use:
    modal run experiments/stirnet/training/01_nis3d_spatial_training.py \
        --max-steps 1000 \
        --samples Drosophila_2,MusMusculus_2,Zebrafish_2 \
        --run-name nis3d_spatial_v1

Resume the long run:
    modal run experiments/stirnet/training/01_nis3d_spatial_training.py \
        --max-steps 1000 \
        --samples Drosophila_2,MusMusculus_2,Zebrafish_2 \
        --run-name nis3d_spatial_v1 \
        --resume
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from tqdm import tqdm
import modal


# ======================================================================================
# Modal resources
# ======================================================================================

APP_NAME = "stirnet-nis3d-spatial-training"

GPU = "L40S"
CPU = 4.0
MEMORY_MB = 8_192
TIMEOUT_SECONDS = 6 * 60 * 60

# NIS3D is stored in the dedicated Modal volume named "external".
# Keep training outputs/checkpoints in the separate "stirnet-runs" volume.
DATA_VOLUME_NAME = "external"
RUNS_VOLUME_NAME = "stirnet-runs"

# IMPORTANT: keep Modal/container paths as POSIX strings at module-import time.
# This file is imported first on the local machine (which may be Windows).
# pathlib.Path("/workspace/...") would therefore become a WindowsPath such as
# "\\workspace\\...", which Modal correctly rejects as a non-absolute POSIX
# remote_path. Convert to pathlib.Path only inside code that executes remotely.
REMOTE_REPO_ROOT = "/workspace/cell-tracking"
DATA_MOUNT = f"{REMOTE_REPO_ROOT}/data"
RUNS_MOUNT = f"{REMOTE_REPO_ROOT}/runs"

# The NIS3D download has appeared under slightly different nesting depths while
# being copied between local storage and Modal. The loader searches these roots
# and then falls back to a bounded recursive search.
NIS3D_ROOT_CANDIDATES = (
    f"{DATA_MOUNT}/external/NIS3D/NIS3D",
    f"{DATA_MOUNT}/external/NIS3D",
    f"{DATA_MOUNT}/NIS3D/NIS3D",
    f"{DATA_MOUNT}/NIS3D",
)

def _resolve_local_repo_root() -> Path:
    """Resolve the repository both during local Modal build and remote hydration.

    Locally, the script lives at experiments/stirnet/training/... and we find
    the first ancestor containing both learned/ and src/.

    Inside Modal, the entry script itself is hydrated at /root, so positional
    parent indexing is invalid. In that case the repository code has already
    been added under REMOTE_REPO_ROOT and we use that path instead.
    """
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "learned").is_dir() and (candidate / "src").is_dir():
            return candidate

    remote_repo = Path(REMOTE_REPO_ROOT)
    if (remote_repo / "learned").is_dir() and (remote_repo / "src").is_dir():
        return remote_repo

    # This fallback is primarily defensive for Modal's image-definition
    # hydration. It avoids brittle parents[N] assumptions.
    return Path.cwd()


LOCAL_REPO_ROOT = _resolve_local_repo_root()

# When this file is executed directly with:
#
#   python experiments/stirnet/training/01_nis3d_spatial_training.py
#
# Python puts the script directory on sys.path, not the repository root.
# Add the repository root explicitly so `learned.*` and `src.*` imports resolve
# exactly as they do when the project is launched through Modal.
_repo_root_text = str(LOCAL_REPO_ROOT)
if _repo_root_text not in sys.path:
    sys.path.insert(0, _repo_root_text)

LOCAL_NIS3D_ROOT_CANDIDATES = (
    LOCAL_REPO_ROOT / "data" / "external" / "NIS3D" / "NIS3D",
    LOCAL_REPO_ROOT / "data" / "external" / "NIS3D",
    LOCAL_REPO_ROOT / "data" / "NIS3D" / "NIS3D",
    LOCAL_REPO_ROOT / "data" / "NIS3D",
)

DIRECT_LOCAL_EXECUTION = __name__ == "__main__"

if DIRECT_LOCAL_EXECUTION:
    # A normal local Python smoke test must not build a Modal Image at import
    # time. In particular, Modal's add_local_dir() path validation is irrelevant
    # to local execution and previously prevented the script from reaching the
    # local training code at all.
    app = None
    data_volume = None
    runs_volume = None
    image = None

    def _modal_function_decorator(*_args, **_kwargs):
        def decorator(function):
            return function
        return decorator

    def _modal_local_entrypoint_decorator(*_args, **_kwargs):
        def decorator(function):
            return function
        return decorator
else:
    app = modal.App(APP_NAME)
    data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
    runs_volume = modal.Volume.from_name(
        RUNS_VOLUME_NAME, create_if_missing=True
    )

    # Keep the cloud image focused on the training runtime. The repository's
    # canonical requirements.txt remains the source of pinned versions;
    # visualization packages are intentionally not installed in the container.
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .env(
            {
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "PYTHONUNBUFFERED": "1",
            }
        )
        .uv_pip_install(
            "torch==2.13.0",
            "numpy==2.4.6",
            "scipy==1.17.1",
            "scikit-image==0.26.0",
            "networkx==3.6.1",
            "tifffile==2026.3.3",
            "imagecodecs==2026.3.6",
            "cupy-cuda13x[ctk]==14.2.0",
            "psutil==7.2.2",
            "tqdm==4.69.0",
        )
        .workdir(REMOTE_REPO_ROOT)
        .add_local_dir(
            LOCAL_REPO_ROOT / "learned",
            remote_path=f"{REMOTE_REPO_ROOT}/learned",
        )
        .add_local_dir(
            LOCAL_REPO_ROOT / "src",
            remote_path=f"{REMOTE_REPO_ROOT}/src",
        )
    )

    _modal_function_decorator = app.function
    _modal_local_entrypoint_decorator = app.local_entrypoint


# ======================================================================================
# Small file helpers
# ======================================================================================

def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if hasattr(value, "item"):
        try:
            return _as_jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_as_jsonable(v) for v in value]
    return str(value)


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# ======================================================================================
# NIS3D discovery / metadata
# ======================================================================================

def _discover_nis3d_root(
    sample_names: tuple[str, ...],
    *,
    root_candidates=None,
    search_root: Path | None = None,
) -> Path:
    candidates = (
        tuple(Path(value) for value in NIS3D_ROOT_CANDIDATES)
        if root_candidates is None
        else tuple(Path(value) for value in root_candidates)
    )
    for root in candidates:
        if root.is_dir() and all((root / sample).is_dir() for sample in sample_names):
            return root

    first = sample_names[0]
    fallback_root = Path(DATA_MOUNT) if search_root is None else Path(search_root)
    if fallback_root.exists():
        matches = list(fallback_root.glob(f"**/{first}"))
        for match in matches[:64]:
            parent = match.parent
            if all((parent / sample).is_dir() for sample in sample_names):
                return parent

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not locate the NIS3D dataset.\n"
        f"Requested samples: {sample_names}\n"
        f"Checked:\n{searched}\n"
        f"Fallback search root: {fallback_root}"
    )


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _parse_spacing_xyz_um(info_text: str) -> tuple[float, float, float]:
    """Parse NIS3D physical voxel size from Info.txt.

    The parser accepts common forms such as:
        Voxel size X: 0.42 um
        X Resolution = 0.42
        pixel size in x = 0.42 µm

    It deliberately does NOT silently fall back to isotropic spacing.
    """
    values: dict[str, float] = {}
    lines = info_text.replace("µ", "u").replace("μ", "u").splitlines()

    axis_patterns = {
        axis: (
            re.compile(
                rf"(?i)\b(?:voxel|pixel)?\s*(?:size|spacing|resolution)?"
                rf"\s*{axis}\b[^0-9+\-.]*({_FLOAT})"
            ),
            re.compile(
                rf"(?i)\b{axis}\s*(?:voxel|pixel)?\s*(?:size|spacing|resolution)"
                rf"\b[^0-9+\-.]*({_FLOAT})"
            ),
        )
        for axis in "xyz"
    }

    for line in lines:
        normalized = line.strip()
        for axis, patterns in axis_patterns.items():
            if axis in values:
                continue
            for pattern in patterns:
                match = pattern.search(normalized)
                if match:
                    value = float(match.group(1))
                    if value > 0:
                        values[axis] = value
                        break

    # A second conservative pass handles lines such as:
    # "VoxelSize: X=0.4 Y=0.4 Z=1.0"
    compact = " ".join(lines)
    if len(values) < 3:
        for axis in "xyz":
            if axis in values:
                continue
            match = re.search(rf"(?i)\b{axis}\s*[:=]\s*({_FLOAT})", compact)
            if match and float(match.group(1)) > 0:
                values[axis] = float(match.group(1))

    # Released NIS3D Info.txt files can use:
    #   Resolution:
    #   1 um x 1 um x 1 um
    # The dataset uses X x Y x Z order.
    if len(values) < 3:
        triple = re.search(
            rf"(?i)\bresolution\s*:\s*"
            rf"({_FLOAT})\s*(?:u?m|micron(?:s)?)?\s*[x×]\s*"
            rf"({_FLOAT})\s*(?:u?m|micron(?:s)?)?\s*[x×]\s*"
            rf"({_FLOAT})\s*(?:u?m|micron(?:s)?)?",
            compact,
        )
        if triple:
            x, y, z = (float(triple.group(i)) for i in (1, 2, 3))
            if x > 0 and y > 0 and z > 0:
                values = {"x": x, "y": y, "z": z}

    if set(values) != {"x", "y", "z"}:
        raise ValueError(
            "Could not strictly parse X/Y/Z physical voxel spacing from Info.txt. "
            "Refusing to train with an invented spacing.\n\n"
            f"Info.txt contents:\n{info_text}"
        )

    # Model order is Z,Y,X.
    return (values["z"], values["y"], values["x"])


def _load_nis3d_arrays(sample_dir: Path):
    import numpy as np
    import tifffile

    required = {
        "raw": sample_dir / "data.tif",
        "gt": sample_dir / "GroundTruth.tif",
        "confidence": sample_dir / "ConfidenceScore.tif",
        "info": sample_dir / "Info.txt",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{sample_dir.name}: required NIS3D files are missing:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    def read_volume(path: Path, label: str):
        """Prefer zero-copy TIFF memmap, fall back to normal TIFF decoding.

        NIS3D TIFFs may be compressed/tiled, in which case tifffile.memmap()
        correctly refuses them because their image bytes are not one contiguous
        memory-mappable block.  These volumes are still modest relative to the
        16 GiB host-RAM request, so decoding them into RAM is the correct robust
        fallback.
        """
        try:
            array = tifffile.memmap(path)
            mode = "memmap"
        except ValueError as exc:
            print(
                f"[data] {sample_dir.name}/{path.name}: TIFF is not directly "
                f"memory-mappable ({exc}); decoding into host RAM.",
                flush=True,
            )
            array = tifffile.imread(path)
            mode = "imread"
        array = np.asarray(array)
        print(
            f"[data] {sample_dir.name}/{label}: load={mode} "
            f"shape={tuple(int(v) for v in array.shape)} dtype={array.dtype} "
            f"RAM={array.nbytes / 2**20:.1f} MiB",
            flush=True,
        )
        return array

    raw = read_volume(required["raw"], "raw")
    gt = read_volume(required["gt"], "gt")
    confidence = read_volume(required["confidence"], "confidence")

    if raw.ndim != 3 or gt.ndim != 3 or confidence.ndim != 3:
        raise ValueError(
            f"{sample_dir.name}: expected 3-D TIFFs, got "
            f"raw={raw.shape}, gt={gt.shape}, confidence={confidence.shape}"
        )
    if raw.shape != gt.shape or raw.shape != confidence.shape:
        raise ValueError(
            f"{sample_dir.name}: TIFF shapes do not align: "
            f"raw={raw.shape}, gt={gt.shape}, confidence={confidence.shape}"
        )

    info_text = required["info"].read_text(encoding="utf-8", errors="replace")
    spacing_zyx_um = _parse_spacing_xyz_um(info_text)

    return raw, gt, confidence, spacing_zyx_um, info_text


def _prepare_nis3d_gt(
    gt,
    confidence,
    spacing_zyx_um: tuple[float, float, float],
    *,
    ignore_margin_um: float,
):
    """Return cleaned GT + conservative dataset-level supervision mask.

    ConfidenceScore==1 is undefined.  If an annotated instance touches undefined
    supervision, remove the entire object and mark all its voxels invalid so an
    annotation fragment is never trained as a complete cell.
    """
    import numpy as np
    from scipy import ndimage as ndi

    gt_np = np.asarray(gt)
    confidence_np = np.asarray(confidence)

    valid = confidence_np != 1
    raw_gt_ids = np.unique(gt_np[gt_np > 0])

    touched_ids = np.unique(gt_np[(gt_np > 0) & ~valid])
    touched_ids = touched_ids[touched_ids > 0]

    clean_gt = np.asarray(gt_np, dtype=np.int32).copy()

    if touched_ids.size:
        touched_mask = np.isin(clean_gt, touched_ids)
        valid[touched_mask] = False
        clean_gt[touched_mask] = 0

    # Set all undefined voxels to background for target construction, but the
    # validity mask below ensures they do not become supervised background.
    clean_gt[~valid] = 0

    if ignore_margin_um > 0:
        # Physical erosion of the valid region. This masks the transition zone
        # where smoothed surface/separator targets could otherwise be influenced
        # by a ConfidenceScore==1 region.
        distance_inside_valid = ndi.distance_transform_edt(
            valid, sampling=spacing_zyx_um
        )
        valid = valid & (distance_inside_valid > float(ignore_margin_um))

    kept_ids = np.unique(clean_gt[clean_gt > 0])

    report = {
        "gt_ids_original": int(raw_gt_ids.size),
        "gt_ids_removed_touching_confidence1": int(touched_ids.size),
        "gt_ids_kept": int(kept_ids.size),
        "confidence1_fraction": float((confidence_np == 1).mean()),
        "supervision_valid_fraction": float(valid.mean()),
        "confidence_values": [int(v) for v in np.unique(confidence_np).tolist()],
        "removed_gt_ids": [int(v) for v in touched_ids.tolist()],
    }
    return clean_gt, valid.astype(bool, copy=False), report


# ======================================================================================
# Dataset-level supervision patch
# ======================================================================================

def _install_dataset_validity_patch() -> None:
    """Make Trainer crop masks respect batch['supervision_valid_mask'].

    This is intentionally experiment-local.  NIS3D's confidence semantics stay
    out of the model.  The wrapper preserves the trainer's existing partial-cell
    mask and intersects it with the dataset-level validity mask.
    """
    import torch
    import learned.stirnet.training.trainer as trainer_module
    from learned.stirnet.training.crops import CropBatch

    original = trainer_module.prepare_crop_batch
    if getattr(original, "_nis3d_dataset_validity_patch", False):
        return

    def wrapped_prepare_crop_batch(
        batch,
        gt_labels,
        specs,
        *,
        geometry_targets=None,
        partial_ignore_margin_um=0.0,
    ):
        crop = original(
            batch,
            gt_labels,
            specs,
            geometry_targets=geometry_targets,
            partial_ignore_margin_um=partial_ignore_margin_um,
        )
        full_valid = batch.get("supervision_valid_mask")
        if full_valid is None:
            return crop

        full_valid = torch.as_tensor(full_valid).detach().cpu().bool()
        rows = torch.stack(
            [
                full_valid[spec.batch_index][spec.slices_zyx]
                for spec in specs
            ]
        )

        cropped_batch = dict(crop.batch)
        existing = cropped_batch.get("supervision_valid_mask")
        cropped_batch["supervision_valid_mask"] = (
            rows
            if existing is None
            else torch.as_tensor(existing).detach().cpu().bool() & rows
        )
        return CropBatch(
            batch=cropped_batch,
            gt_labels=crop.gt_labels,
            geometry_targets=crop.geometry_targets,
            specs=crop.specs,
        )

    wrapped_prepare_crop_batch._nis3d_dataset_validity_patch = True
    trainer_module.prepare_crop_batch = wrapped_prepare_crop_batch


# ======================================================================================
# Batch construction
# ======================================================================================

def _prepare_sample_batch(
    nis3d_root: Path,
    sample_name: str,
    *,
    cache_root: Path,
    confidence_ignore_margin_um: float,
):
    import torch

    from learned.stirnet.training import prepare_raw_training_batch

    sample_dir = nis3d_root / sample_name
    raw, gt, confidence, spacing, info_text = _load_nis3d_arrays(sample_dir)

    clean_gt, valid_mask, confidence_report = _prepare_nis3d_gt(
        gt,
        confidence,
        spacing,
        ignore_margin_um=confidence_ignore_margin_um,
    )

    source_cache_path = cache_root / "source" / f"{sample_name}.pt"

    started = time.perf_counter()
    batch = prepare_raw_training_batch(
        raw,
        clean_gt,
        spacing,
        source_id=f"NIS3D/{sample_name}",
        source_cache_path=source_cache_path,
    )
    prepare_seconds = time.perf_counter() - started

    batch["supervision_valid_mask"] = torch.from_numpy(valid_mask)[None]
    batch["nis3d_sample_name"] = sample_name

    metadata = dict(batch["source_preprocessing_metadata"][0])
    report = {
        "sample": sample_name,
        "sample_dir": str(sample_dir),
        "shape_zyx": [int(v) for v in raw.shape],
        "raw_dtype": str(raw.dtype),
        "gt_dtype": str(gt.dtype),
        "confidence_dtype": str(confidence.dtype),
        "spacing_zyx_um": [float(v) for v in spacing],
        "source_prepare_seconds": float(prepare_seconds),
        "source_cache_hit": bool(metadata.get("source_cache_hit", False)),
        "current_instance_count": int(
            metadata.get("source_current_instance_count", 0)
        ),
        "model_dref_um": float(metadata.get("model_dref_um", batch["dref_um"][0])),
        **confidence_report,
    }
    return batch, report


# ======================================================================================
# Training / checkpointing
# ======================================================================================

def _checkpoint_paths(recovery_dir: Path) -> list[Path]:
    return sorted(recovery_dir.glob("checkpoint_step_*.pt"))


def _latest_checkpoint(recovery_dir: Path) -> Path | None:
    paths = _checkpoint_paths(recovery_dir)
    return paths[-1] if paths else None


def _atomic_checkpoint(
    path: Path,
    *,
    trainer,
    model_config,
    training_config,
    extra: dict[str, Any],
) -> None:
    from learned.stirnet.training.checkpoint import save_checkpoint

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_checkpoint(
            temporary,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            step=trainer.global_step,
            model_config=model_config,
            training_config=training_config,
            extra=extra,
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _save_recovery_checkpoint(
    recovery_dir: Path,
    *,
    trainer,
    model_config,
    training_config,
    run_dir: Path,
    samples: tuple[str, ...],
    commit_to_modal: bool,
) -> Path:
    step = int(trainer.global_step)
    path = recovery_dir / f"checkpoint_step_{step:06d}.pt"
    extra = {
        "experiment": "01_nis3d_spatial_training",
        "run_dir": str(run_dir),
        "samples": list(samples),
        **trainer.checkpoint_metadata(),
    }
    _atomic_checkpoint(
        path,
        trainer=trainer,
        model_config=model_config,
        training_config=training_config,
        extra=extra,
    )

    _atomic_text(recovery_dir / "latest_checkpoint.txt", path.name + "\n")
    _atomic_json(
        recovery_dir / "latest_state.json",
        {
            "checkpoint": str(path),
            "global_step": step,
            "curriculum_stage": trainer.curriculum_stage.name,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "samples": list(samples),
        },
    )

    # Make the checkpoint durable in the Modal volume before more training.
    if commit_to_modal:
        runs_volume.commit()
    tqdm.write(
        f"[checkpoint] persisted step={step} path={path}"
    )
    return path


def _build_configs(
    *,
    geometry_steps: int,
    spatial_steps: int,
    crop_batch_size: int,
    crop_shape_zyx: tuple[int, int, int],
    learning_rate: float,
    static_cache_dir: Path,
    amp_dtype: str,
):
    from learned.stirnet import StirNetConfig
    from learned.stirnet.training import TrainingConfig

    model_cfg = StirNetConfig()
    # Keep full-width production architecture.
    model_cfg.validate()

    train_cfg = TrainingConfig()
    train_cfg.lr = float(learning_rate)
    train_cfg.amp_dtype = str(amp_dtype)
    train_cfg.geometry_target_backend = "auto"
    train_cfg.crop_static_target_memory_entries = 4
    train_cfg.crop_static_target_cache_dir = str(static_cache_dir)

    curriculum = train_cfg.curriculum
    curriculum.enabled = True
    curriculum.fixed_stage = None
    curriculum.geometry_bootstrap_steps = int(geometry_steps)
    curriculum.spatial_partition_steps = int(spatial_steps)
    curriculum.instance_temporal_steps = 0

    curriculum.refinement_crop_enabled = True
    curriculum.geometry_bootstrap_crop_enabled = True
    curriculum.spatial_partition_crop_enabled = True
    curriculum.refinement_crop_shape_zyx = tuple(int(v) for v in crop_shape_zyx)
    curriculum.refinement_crops_per_step = 1
    curriculum.refinement_crop_batch_size = int(crop_batch_size)
    curriculum.refinement_crop_merge_fraction = 0.50

    # Production missing-cell synthesis. Natural merge crops are protected by
    # source_corruption.py and are never synthetically altered.
    curriculum.refinement_crop_source_dropout_probability = 0.15
    curriculum.refinement_crop_source_dropout_max_instances = 1

    # Normal spatial-training augmentation, not a checkpoint-specific stage.
    curriculum.refinement_crop_xy_flip_probability = 0.50

    train_cfg.validate()
    return model_cfg, train_cfg


def _print_header(
    *,
    samples,
    nis3d_root,
    max_steps,
    checkpoint_every,
    model_cfg,
    train_cfg,
    execution_mode: str,
    results_root: Path,
):
    import torch

    props = torch.cuda.get_device_properties(0)
    print("=" * 118, flush=True)
    print("STIR-Net Training 01 — NIS3D spatial training", flush=True)
    print("=" * 118, flush=True)
    print(f"GPU                      : {props.name}", flush=True)
    print(f"GPU VRAM                 : {props.total_memory / 2**30:.2f} GiB", flush=True)
    print(f"Execution                 : {execution_mode}", flush=True)
    if execution_mode == "modal":
        print(
            f"CPU / RAM request        : {CPU:g} CPU / {MEMORY_MB / 1024:.1f} GiB",
            flush=True,
        )
        print(
            f"Dataset volume            : {DATA_VOLUME_NAME}:"
            f"{str(nis3d_root).replace(DATA_MOUNT, '')}",
            flush=True,
        )
        print(
            "Results volume            : "
            "stirnet-runs:/stirnet/training/01_nis3d_spatial_training/",
            flush=True,
        )
    else:
        print("Host memory               : local machine (paging allowed)", flush=True)
        print(f"Results directory         : {results_root}", flush=True)
    print(f"NIS3D root               : {nis3d_root}", flush=True)
    print(f"Samples                   : {list(samples)}", flush=True)
    print(f"Maximum optimizer steps   : {max_steps}", flush=True)
    print(f"Checkpoint interval       : {checkpoint_every} successful steps", flush=True)
    print(
        f"Curriculum                : geometry={train_cfg.curriculum.geometry_bootstrap_steps} "
        f"-> spatial={train_cfg.curriculum.spatial_partition_steps} -> STOP",
        flush=True,
    )
    print(
        f"Crop batch                : B={train_cfg.curriculum.refinement_crop_batch_size} "
        f"shape={train_cfg.curriculum.refinement_crop_shape_zyx}",
        flush=True,
    )
    print(f"AMP                       : {train_cfg.amp_dtype}", flush=True)
    print(f"Learning rate             : {train_cfg.lr}", flush=True)
    print(
        f"XY flip p/axis            : "
        f"{train_cfg.curriculum.refinement_crop_xy_flip_probability:.2f}",
        flush=True,
    )
    print(f"Watershed backend         : {model_cfg.partition.watershed_backend}", flush=True)
    print(f"GT backend                : {train_cfg.geometry_target_backend}", flush=True)
    print("=" * 118, flush=True)


# ======================================================================================
# Modal worker
# ======================================================================================

def _train_nis3d_impl(
    max_steps: int = 5,
    samples_csv: str = "Zebrafish_2",
    run_name: str = "smoke_zebra_5",
    resume: bool = False,
    checkpoint_every: int = 50,
    geometry_steps: int = 500,
    spatial_steps: int = 500,
    learning_rate: float = 2e-4,
    crop_batch_size: int = 4,
    confidence_ignore_margin_um: float = 1.0,
    execution_mode: str = "modal",
) -> dict[str, Any]:
    import numpy as np
    import torch

    from learned.stirnet import StirNet
    from learned.stirnet.training.trainer import Trainer
    from learned.stirnet.training.checkpoint import load_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script")
    if execution_mode not in {"local", "modal"}:
        raise ValueError("execution_mode must be local or modal")
    amp_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    if geometry_steps < 1 or spatial_steps < 1:
        raise ValueError("geometry_steps and spatial_steps must be positive")
    if max_steps > geometry_steps + spatial_steps:
        raise ValueError(
            "This script intentionally stops before temporal training. "
            f"max_steps={max_steps} exceeds geometry+spatial="
            f"{geometry_steps + spatial_steps}."
        )
    if crop_batch_size < 1:
        raise ValueError("crop_batch_size must be positive")

    samples = tuple(
        token.strip()
        for token in samples_csv.split(",")
        if token.strip()
    )
    if not samples:
        raise ValueError("At least one NIS3D sample is required")

    allowed_training_samples = {
        "Drosophila_2",
        "MusMusculus_2",
        "Zebrafish_2",
    }
    unknown = sorted(set(samples) - allowed_training_samples)
    if unknown:
        raise ValueError(
            "Training 01 uses only the official *_2 NIS3D training volumes. "
            f"Unsupported requested samples: {unknown}"
        )

    np.random.seed(230525)
    torch.manual_seed(230525)
    torch.cuda.manual_seed_all(230525)

    _install_dataset_validity_patch()

    if execution_mode == "local":
        nis3d_root = _discover_nis3d_root(
            samples,
            root_candidates=LOCAL_NIS3D_ROOT_CANDIDATES,
            search_root=LOCAL_REPO_ROOT / "data",
        )
        results_mount = LOCAL_REPO_ROOT / "runs"
    else:
        nis3d_root = _discover_nis3d_root(samples)
        results_mount = Path(RUNS_MOUNT)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    experiment_root = (
        results_mount / "stirnet" / "training" / "01_nis3d_spatial_training"
    )
    run_dir = experiment_root / "attempts" / f"{timestamp}_{run_name}"
    recovery_dir = experiment_root / "recovery" / run_name
    cache_root = experiment_root / "cache"
    static_cache_dir = cache_root / "static_gt"

    run_dir.mkdir(parents=True, exist_ok=True)
    recovery_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    model_cfg, train_cfg = _build_configs(
        geometry_steps=geometry_steps,
        spatial_steps=spatial_steps,
        crop_batch_size=crop_batch_size,
        crop_shape_zyx=(32, 192, 192),
        learning_rate=learning_rate,
        static_cache_dir=static_cache_dir,
        amp_dtype=amp_dtype,
    )

    _print_header(
        samples=samples,
        nis3d_root=nis3d_root,
        max_steps=max_steps,
        checkpoint_every=checkpoint_every,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        execution_mode=execution_mode,
        results_root=experiment_root,
    )

    # Load/prepare requested source volumes once.  With the three NIS3D *_2
    # volumes this stays within the requested 16 GiB host memory, while the
    # expensive raw->current preprocessing becomes a persistent cache hit on
    # subsequent runs.
    prepared_batches: dict[str, dict] = {}
    sample_reports: dict[str, dict] = {}
    for sample in samples:
        print(f"[data] preparing {sample} ...", flush=True)
        batch, report = _prepare_sample_batch(
            nis3d_root,
            sample,
            cache_root=cache_root,
            confidence_ignore_margin_um=confidence_ignore_margin_um,
        )
        prepared_batches[sample] = batch
        sample_reports[sample] = report
        print(
            f"[data] {sample}: shape={tuple(report['shape_zyx'])} "
            f"spacing={tuple(round(v, 6) for v in report['spacing_zyx_um'])} "
            f"GT={report['gt_ids_kept']} "
            f"valid={report['supervision_valid_fraction']:.4f} "
            f"source={report['current_instance_count']} "
            f"dref={report['model_dref_um']:.4f}um "
            f"source_cache_hit={report['source_cache_hit']} "
            f"prepare={report['source_prepare_seconds']:.2f}s",
            flush=True,
        )

    _atomic_json(run_dir / "samples.json", sample_reports)
    _atomic_json(
        run_dir / "config.json",
        {
            "model": model_cfg.to_dict(),
            "training": train_cfg.to_dict(),
            "max_steps": int(max_steps),
            "checkpoint_every": int(checkpoint_every),
            "samples": list(samples),
            "run_name": run_name,
            "confidence_ignore_margin_um": float(confidence_ignore_margin_um),
            "startup_note": (
                "Modal source/cache writes are committed at checkpoint/final "
                "persistence, not eagerly before model initialization."
            ),
        },
    )
    # Do not commit the Modal volume here. Source/cache metadata is already on
    # the mounted filesystem and will be persisted by the normal checkpoint/final
    # commit path. An eager commit before model initialization can make startup
    # appear hung and is not required for correctness.

    init_started = time.perf_counter()

    print("[init] constructing STIR-Net on CPU ...", flush=True)
    phase_started = time.perf_counter()
    model = StirNet(model_cfg)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"[init] model constructed in {time.perf_counter() - phase_started:.2f}s "
        f"({parameter_count / 1e6:.2f}M parameters, "
        f"{trainable_parameter_count / 1e6:.2f}M initially trainable)",
        flush=True,
    )

    print("[init] moving model to CUDA ...", flush=True)
    phase_started = time.perf_counter()
    model = model.to("cuda")
    torch.cuda.synchronize()
    print(
        f"[init] model on CUDA in {time.perf_counter() - phase_started:.2f}s",
        flush=True,
    )

    print("[init] constructing Trainer / criterion / optimizer ...", flush=True)
    phase_started = time.perf_counter()
    trainer = Trainer(model, train_cfg, device="cuda")
    torch.cuda.synchronize()
    print(
        f"[init] trainer ready in {time.perf_counter() - phase_started:.2f}s "
        f"(total init {time.perf_counter() - init_started:.2f}s, "
        f"stage={trainer.curriculum_stage.name})",
        flush=True,
    )

    resumed_from = None
    if resume:
        checkpoint_path = _latest_checkpoint(recovery_dir)
        if checkpoint_path is None:
            print("[resume] no previous checkpoint found; starting fresh", flush=True)
        else:
            print(f"[resume] loading {checkpoint_path}", flush=True)
            checkpoint = load_checkpoint(
                checkpoint_path,
                trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                map_location="cpu",
                strict=True,
            )
            trainer.restore_training_progress(checkpoint, resume=True)
            resumed_from = str(checkpoint_path)
            print(
                f"[resume] global_step={trainer.global_step} "
                f"stage={trainer.curriculum_stage.name}",
                flush=True,
            )

    if trainer.global_step >= max_steps:
        print(
            f"[done] checkpoint already at step {trainer.global_step} >= "
            f"requested max_steps={max_steps}",
            flush=True,
        )
        return {
            "status": "already_complete",
            "global_step": trainer.global_step,
            "run_dir": str(run_dir),
            "recovery_dir": str(recovery_dir),
            "resumed_from": resumed_from,
        }

    history_path = run_dir / "history.jsonl"
    started_all = time.perf_counter()
    last_checkpoint_step = None
    attempt_step_time_sum = 0.0
    attempt_success_steps = 0

    print(
        f"[train] starting optimizer loop at step {trainer.global_step}/{max_steps} ...",
        flush=True,
    )
    GREEN = "\033[32m"
    RESET = "\033[0m"

    progress = tqdm(
        total=max_steps,
        initial=int(trainer.global_step),
        desc=f"{GREEN}STIR-Net{RESET}",
        unit="step",
        dynamic_ncols=True,
        smoothing=0.10,
        mininterval=0.5,
        leave=True,
        colour="green",
        file=sys.stdout,
    )

    try:
        while trainer.global_step < max_steps:
            step_before = int(trainer.global_step)
            sample = samples[step_before % len(samples)]
            batch = prepared_batches[sample]

            # Keep CUDA memory profiling for persistent diagnostics, but do not
            # clutter the live terminal progress bar with it.
            torch.cuda.reset_peak_memory_stats()
            step_started = time.perf_counter()

            metrics = trainer.train_step(batch)

            # train_step must advance exactly once before we call the step
            # successful and persist its scalar record.
            if trainer.global_step != step_before + 1:
                raise RuntimeError(
                    "Trainer global_step did not advance exactly once: "
                    f"{step_before} -> {trainer.global_step}"
                )

            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - step_started

            attempt_step_time_sum += step_seconds
            attempt_success_steps += 1
            avg_step_seconds = (
                attempt_step_time_sum / max(1, attempt_success_steps)
            )
            remaining_steps = max(0, max_steps - int(trainer.global_step))
            estimated_remaining_seconds = avg_step_seconds * remaining_steps
            attempt_elapsed_seconds = time.perf_counter() - started_all

            peak_allocated = torch.cuda.max_memory_allocated() / 2**30
            peak_reserved = torch.cuda.max_memory_reserved() / 2**30
            current_allocated = torch.cuda.memory_allocated() / 2**30
            current_reserved = torch.cuda.memory_reserved() / 2**30

            record = {
                "step": int(trainer.global_step),
                "sample": sample,
                "stage": trainer.curriculum_stage.name,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "step_seconds": float(step_seconds),
                "average_step_seconds": float(avg_step_seconds),
                "attempt_elapsed_seconds": float(attempt_elapsed_seconds),
                "estimated_remaining_seconds": float(
                    estimated_remaining_seconds
                ),
                "cuda_peak_allocated_gib": float(peak_allocated),
                "cuda_peak_reserved_gib": float(peak_reserved),
                "cuda_allocated_gib": float(current_allocated),
                "cuda_reserved_gib": float(current_reserved),
                **{
                    key: _as_jsonable(value)
                    for key, value in metrics.items()
                },
            }
            _append_jsonl(history_path, record)

            total_loss = record.get(
                "loss",
                record.get("loss_total", record.get("total", "?")),
            )
            crop_b = record.get(
                "phase_a_crop_effective_batch_size",
                record.get("crop_true_batch_size", "?"),
            )
            merge_count = record.get("phase_a_crop_merge_count", "?")
            coverage_count = record.get("phase_a_crop_coverage_count", "?")
            dropout_count = record.get("phase_a_source_dropout_count", "?")
            cache_miss = record.get("phase_a_static_target_misses", "?")
            cache_mem = record.get("phase_a_static_target_memory_hits", "?")
            cache_disk = record.get("phase_a_static_target_disk_hits", "?")

            # tqdm already displays completed/total steps, elapsed time,
            # throughput and ETA. The postfix carries compact training state.
            progress.set_postfix(
                {
                    "loss": total_loss,
                    "stage": trainer.curriculum_stage.name,
                    "sample": sample,
                    "avg": f"{avg_step_seconds:.2f}s",
                    "B": crop_b,
                    "merge": merge_count,
                    "drop": dropout_count,
                },
                refresh=False,
            )
            progress.update(1)

            if (
                trainer.global_step % checkpoint_every == 0
                or trainer.global_step == max_steps
            ):
                _save_recovery_checkpoint(
                    recovery_dir,
                    trainer=trainer,
                    model_config=model_cfg,
                    training_config=train_cfg,
                    run_dir=run_dir,
                    samples=samples,
                    commit_to_modal=(execution_mode == "modal"),
                )
                last_checkpoint_step = int(trainer.global_step)

        progress.close()
        elapsed = time.perf_counter() - started_all
        summary = {
            "status": "success",
            "global_step": int(trainer.global_step),
            "max_steps": int(max_steps),
            "samples": list(samples),
            "run_dir": str(run_dir),
            "recovery_dir": str(recovery_dir),
            "resumed_from": resumed_from,
            "elapsed_seconds": float(elapsed),
            "elapsed_human": _duration(elapsed),
            "last_checkpoint_step": last_checkpoint_step,
        }
        _atomic_json(run_dir / "summary.json", summary)
        if execution_mode == "modal":
            runs_volume.commit()

        print("=" * 118, flush=True)
        print("NIS3D TRAINING ATTEMPT COMPLETE", flush=True)
        print("=" * 118, flush=True)
        print(f"Steps             : {trainer.global_step}", flush=True)
        print(f"Elapsed           : {_duration(elapsed)}", flush=True)
        print(f"Run dir           : {run_dir}", flush=True)
        print(f"Recovery dir      : {recovery_dir}", flush=True)
        print(f"Last checkpoint   : step {last_checkpoint_step}", flush=True)
        print("=" * 118, flush=True)
        return summary

    except BaseException as exc:
        progress.close()
        # Do not label a failed optimizer step as successful.  We persist the
        # traceback metadata, then commit all earlier successful-step records and
        # any previous periodic checkpoint.
        failure = {
            "status": "failed",
            "global_step": int(trainer.global_step),
            "max_steps": int(max_steps),
            "samples": list(samples),
            "run_dir": str(run_dir),
            "recovery_dir": str(recovery_dir),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "last_periodic_checkpoint": (
                None
                if last_checkpoint_step is None
                else int(last_checkpoint_step)
            ),
        }
        _atomic_json(run_dir / "failure.json", failure)
        if execution_mode == "modal":
            runs_volume.commit()
        print(
            f"[failure] step={trainer.global_step} "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        raise


# ======================================================================================
# Modal wrapper
# ======================================================================================

@_modal_function_decorator(
    image=image,
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    volumes={
        DATA_MOUNT: data_volume,
        RUNS_MOUNT: runs_volume,
    },
)
def train_nis3d(
    max_steps: int = 5,
    samples_csv: str = "Zebrafish_2",
    run_name: str = "smoke_zebra_5",
    resume: bool = False,
    checkpoint_every: int = 50,
    geometry_steps: int = 500,
    spatial_steps: int = 500,
    learning_rate: float = 2e-4,
    crop_batch_size: int = 4,
    confidence_ignore_margin_um: float = 1.0,
) -> dict[str, Any]:
    return _train_nis3d_impl(
        max_steps=max_steps,
        samples_csv=samples_csv,
        run_name=run_name,
        resume=resume,
        checkpoint_every=checkpoint_every,
        geometry_steps=geometry_steps,
        spatial_steps=spatial_steps,
        learning_rate=learning_rate,
        crop_batch_size=crop_batch_size,
        confidence_ignore_margin_um=confidence_ignore_margin_um,
        execution_mode="modal",
    )


# ======================================================================================
# Local CLI
# ======================================================================================

@_modal_local_entrypoint_decorator()
def main(
    max_steps: int = 5,
    samples: str = "Zebrafish_2",
    run_name: str = "smoke_zebra_5",
    resume: bool = False,
    checkpoint_every: int = 50,
    geometry_steps: int = 500,
    spatial_steps: int = 500,
    learning_rate: float = 2e-4,
    crop_batch_size: int = 4,
    confidence_ignore_margin_um: float = 1.0,
) -> None:
    result = train_nis3d.remote(
        max_steps=max_steps,
        samples_csv=samples,
        run_name=run_name,
        resume=resume,
        checkpoint_every=checkpoint_every,
        geometry_steps=geometry_steps,
        spatial_steps=spatial_steps,
        learning_rate=learning_rate,
        crop_batch_size=crop_batch_size,
        confidence_ignore_margin_um=confidence_ignore_margin_um,
    )
    print(json.dumps(result, indent=2))


def _python_cli_main() -> None:
    """Unified direct-Python CLI.

    `--execution modal` is the default. It re-invokes this same file through the
    Modal CLI, preserving the normal @app.local_entrypoint/@app.function path.
    `--execution local` stays entirely in the current Python process and uses
    the local GPU/data/results directories.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run NIS3D STIR-Net training locally or on Modal. "
            "Default execution target: modal."
        )
    )
    parser.add_argument(
        "--execution",
        choices=("modal", "local"),
        default="modal",
        help="Execution target. Default: modal.",
    )
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--samples", default="Zebrafish_2")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--geometry-steps", type=int, default=500)
    parser.add_argument("--spatial-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--crop-batch-size",
        type=int,
        default=None,
        help=(
            "Override true crop batch size. Defaults to 4 on Modal and 1 "
            "locally. Crop SHAPE remains 32x192x192 in both modes."
        ),
    )
    parser.add_argument(
        "--confidence-ignore-margin-um",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()

    run_name = args.run_name
    if run_name is None:
        run_name = (
            "smoke_zebra_modal_5"
            if args.execution == "modal"
            else "smoke_zebra_local_1"
        )

    if args.execution == "local":
        crop_batch_size = (
            1 if args.crop_batch_size is None else args.crop_batch_size
        )
        result = _train_nis3d_impl(
            max_steps=args.max_steps,
            samples_csv=args.samples,
            run_name=run_name,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            geometry_steps=args.geometry_steps,
            spatial_steps=args.spatial_steps,
            learning_rate=args.learning_rate,
            crop_batch_size=crop_batch_size,
            confidence_ignore_margin_um=args.confidence_ignore_margin_um,
            execution_mode="local",
        )
        print(json.dumps(result, indent=2))
        return

    # Default path: invoke the same file with `modal run`.  Keeping Modal's
    # native CLI as the launcher is deliberate: it handles app hydration,
    # mounts, logs, retries, and authentication exactly as usual.
    crop_batch_size = (
        4 if args.crop_batch_size is None else args.crop_batch_size
    )
    modal_exe = shutil.which("modal")
    if modal_exe is None:
        raise RuntimeError(
            "Could not find the `modal` executable. Activate the project "
            "virtual environment or install the repository requirements."
        )

    command = [
        modal_exe,
        "run",
        str(Path(__file__).resolve()),
        "--max-steps",
        str(args.max_steps),
        "--samples",
        args.samples,
        "--run-name",
        run_name,
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--geometry-steps",
        str(args.geometry_steps),
        "--spatial-steps",
        str(args.spatial_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--crop-batch-size",
        str(crop_batch_size),
        "--confidence-ignore-margin-um",
        str(args.confidence_ignore_margin_um),
    ]
    if args.resume:
        command.append("--resume")

    print(
        "[launcher] execution=modal (default)\n"
        "[launcher] " + " ".join(command),
        flush=True,
    )
    completed = subprocess.run(command, cwd=LOCAL_REPO_ROOT)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    _python_cli_main()
