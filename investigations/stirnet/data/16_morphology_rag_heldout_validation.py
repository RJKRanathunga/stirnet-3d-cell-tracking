from __future__ import annotations

"""
STIR-Net Investigation 16 — morphology-aware RAG held-out validation.

Goal
----
Evaluate three checkpoints on the SAME unseen Drosophila crops without any
optimization:

A) original spatial milestone
   checkpoint_step_001000.pt

B) morphology-only overfit checkpoint
   Investigation 15 checkpoint_step_000100.pt

C) joint-RAG overfit checkpoint
   Investigation 15 checkpoint_step_000300.pt

The experiment answers two questions before broad RAG training:

1. Did the new 3-D node/edge morphology representation learn anything that
   transfers beyond the single Investigation-15 overfit crop?
2. Did unfreezing the legacy RAG after step 100 help or hurt held-out
   generalization relative to the safer morphology-only checkpoint?

Default evaluation
------------------
* Drosophila_1: 24 deterministic held-out manifest crops, excluding manifest 3
  (the Investigation-15 training crop).
* Drosophila_2: 24 deterministic held-out manifest crops.
* Crop shape: 32 x 192 x 192.
* Effective Drosophila spacing:
      XYZ = 0.20312639, 0.20312639, 0.79099447 um
* No training, no augmentation, no source dropout.
* Dense geometry is computed ONCE per crop with the original milestone and is
  reused for all three RAG checkpoints. The script first verifies that every
  non-RAG checkpoint tensor is exactly identical across A/B/C.
* Crop selection is model-independent and balanced between merge-manifest and
  non-merge-manifest records where possible. We do NOT select crops based on
  checkpoint errors.

Metrics
-------
Per checkpoint, per sample and overall:
* valid-edge BCE
* classification accuracy at p=0.5
* positive/negative accuracy at p=0.5
* mean p_merge for positive and negative edges
* actual production partition threshold diagnostics (default 0.845):
    - negative edges incorrectly accepted for merge
    - positive edges retained for merge
    - separator-backed false merges
* baseline-hard-negative transfer:
    - edges where A has GT=separate and p_merge>=0.5
    - same exact edge set is measured under B and C
* baseline separator-hard-negative transfer:
    - baseline hard negatives with separator_max >= 0.5

This is an EDGE generalization test. It deliberately does not retrain the
partitioner or geometry and does not use BioHub pseudo-GT.

Recommended local run
---------------------
From repository root:

    python investigations/stirnet/data/16_morphology_rag_heldout_validation.py

A faster smoke:

    python investigations/stirnet/data/16_morphology_rag_heldout_validation.py ^
        --crops-per-sample 3 ^
        --run-name morph_rag_heldout_smoke

Useful explicit paths if auto-resolution is not appropriate:

    python investigations/stirnet/data/16_morphology_rag_heldout_validation.py ^
        --baseline-checkpoint runs/stirnet/milestones/drosophila_12_spatial_v1/checkpoint_step_001000.pt ^
        --morph-checkpoint runs/stirnet/investigations/15_morphology_rag_overfit/recovery/drosophila_1_morphology_rag_overfit/checkpoint_step_000100.pt ^
        --joint-checkpoint runs/stirnet/investigations/15_morphology_rag_overfit/recovery/drosophila_1_morphology_rag_overfit/checkpoint_step_000300.pt
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm
import modal


# ======================================================================================
# Modal / repository paths
# ======================================================================================

APP_NAME = "stirnet-morphology-rag-heldout-validation"

GPU = "L40S"
CPU = 3.0
MEMORY_MB = 16_384
TIMEOUT_SECONDS = 3 * 60 * 60

DATA_VOLUME_NAME = "external"
RUNS_VOLUME_NAME = "stirnet-runs"

REMOTE_REPO_ROOT = "/workspace/cell-tracking"
DATA_MOUNT = f"{REMOTE_REPO_ROOT}/data"
RUNS_MOUNT = f"{REMOTE_REPO_ROOT}/runs"


def _resolve_local_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "learned").is_dir() and (candidate / "src").is_dir():
            return candidate

    remote = Path(REMOTE_REPO_ROOT)
    if (remote / "learned").is_dir() and (remote / "src").is_dir():
        return remote

    cwd = Path.cwd().resolve()
    if (cwd / "learned").is_dir() and (cwd / "src").is_dir():
        return cwd
    return cwd


LOCAL_REPO_ROOT = _resolve_local_repo_root()
_repo_root_text = str(LOCAL_REPO_ROOT)
if _repo_root_text not in sys.path:
    sys.path.insert(0, _repo_root_text)

LOCAL_NIS3D_ROOT_CANDIDATES = (
    LOCAL_REPO_ROOT / "data" / "external" / "NIS3D" / "NIS3D",
    LOCAL_REPO_ROOT / "data" / "external" / "NIS3D",
    LOCAL_REPO_ROOT / "data" / "NIS3D" / "NIS3D",
    LOCAL_REPO_ROOT / "data" / "NIS3D",
)

NIS3D_ROOT_CANDIDATES = (
    f"{DATA_MOUNT}/external/NIS3D/NIS3D",
    f"{DATA_MOUNT}/external/NIS3D",
    f"{DATA_MOUNT}/NIS3D/NIS3D",
    f"{DATA_MOUNT}/NIS3D",
)

DIRECT_LOCAL_EXECUTION = __name__ == "__main__"

if DIRECT_LOCAL_EXECUTION:
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
        RUNS_VOLUME_NAME,
        create_if_missing=True,
    )
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
# Generic helpers
# ======================================================================================

MODEL_NAMES = (
    "baseline_step1000",
    "morphology_step100",
    "joint_step300",
)

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(_as_jsonable(payload), indent=2, sort_keys=True),
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_as_jsonable(payload), sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _as_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
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


def _torch_load(path: Path, *, map_location="cpu") -> dict:
    import torch

    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def _parse_triplet_text(value: str, *, name: str, cast=float):
    tokens = [
        token
        for token in re.split(r"[,\s]+", str(value).strip())
        if token
    ]
    if len(tokens) != 3:
        raise ValueError(
            f"{name} must contain exactly three values; got {value!r}"
        )
    try:
        return tuple(cast(token) for token in tokens)
    except Exception as exc:
        raise ValueError(
            f"{name} contains an invalid value: {value!r}"
        ) from exc


def _parse_spacing_xyz_override(
    value: str | None,
) -> tuple[float, float, float] | None:
    if value is None or not str(value).strip():
        return None

    x, y, z = _parse_triplet_text(
        str(value),
        name="--spacing-xyz",
        cast=float,
    )
    xyz = (float(x), float(y), float(z))
    if any(not math.isfinite(v) or v <= 0 for v in xyz):
        raise ValueError("--spacing-xyz values must be finite and positive")
    return (xyz[2], xyz[1], xyz[0])


def _parse_shape_zyx(value: str) -> tuple[int, int, int]:
    z, y, x = _parse_triplet_text(
        value,
        name="--crop-shape-zyx",
        cast=int,
    )
    shape = (int(z), int(y), int(x))
    if any(v < 8 for v in shape):
        raise ValueError("crop dimensions must each be >= 8")
    return shape


def _parse_int_set(value: str) -> set[int]:
    value = str(value).strip()
    if not value:
        return set()
    result = set()
    for token in re.split(r"[,\s]+", value):
        if token:
            result.add(int(token))
    return result


def _autocast_context(amp_dtype: str):
    import torch
    from contextlib import nullcontext

    if not torch.cuda.is_available() or amp_dtype == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================================================
# NIS3D loading — aligned with Training 01 / Investigation 15
# ======================================================================================

def _parse_spacing_xyz_um(info_text: str) -> tuple[float, float, float]:
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
                    number = float(match.group(1))
                    if number > 0:
                        values[axis] = number
                        break

    compact = " ".join(lines)
    if len(values) < 3:
        for axis in "xyz":
            if axis in values:
                continue
            match = re.search(
                rf"(?i)\b{axis}\s*[:=]\s*({_FLOAT})",
                compact,
            )
            if match and float(match.group(1)) > 0:
                values[axis] = float(match.group(1))

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
            "Could not strictly parse X/Y/Z physical spacing from Info.txt"
        )
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
            f"{sample_dir.name}: missing NIS3D files:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    def read_volume(path: Path, label: str):
        try:
            array = tifffile.memmap(path)
            mode = "memmap"
        except ValueError as exc:
            print(
                f"[data] {sample_dir.name}/{path.name}: not directly "
                f"memory-mappable ({exc}); decoding.",
                flush=True,
            )
            array = tifffile.imread(path)
            mode = "imread"

        array = np.asarray(array)
        print(
            f"[data] {sample_dir.name}/{label}: load={mode} "
            f"shape={tuple(int(v) for v in array.shape)} "
            f"dtype={array.dtype} RAM={array.nbytes / 2**20:.1f} MiB",
            flush=True,
        )
        return array

    raw = read_volume(required["raw"], "raw")
    gt = read_volume(required["gt"], "gt")
    confidence = read_volume(required["confidence"], "confidence")

    if raw.ndim != 3 or raw.shape != gt.shape or raw.shape != confidence.shape:
        raise ValueError(
            f"NIS3D arrays do not align: raw={raw.shape} gt={gt.shape} "
            f"confidence={confidence.shape}"
        )

    info_text = required["info"].read_text(
        encoding="utf-8",
        errors="replace",
    )
    spacing = _parse_spacing_xyz_um(info_text)
    return raw, gt, confidence, spacing


def _prepare_nis3d_gt(
    gt,
    confidence,
    spacing_zyx_um: tuple[float, float, float],
    *,
    ignore_margin_um: float,
):
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

    clean_gt[~valid] = 0

    if ignore_margin_um > 0:
        distance_inside_valid = ndi.distance_transform_edt(
            valid,
            sampling=spacing_zyx_um,
        )
        valid = valid & (distance_inside_valid > float(ignore_margin_um))

    kept_ids = np.unique(clean_gt[clean_gt > 0])
    return clean_gt, valid.astype(bool, copy=False), {
        "gt_ids_original": int(raw_gt_ids.size),
        "gt_ids_removed_touching_confidence1": int(touched_ids.size),
        "gt_ids_kept": int(kept_ids.size),
        "confidence1_fraction": float((confidence_np == 1).mean()),
        "supervision_valid_fraction": float(valid.mean()),
        "confidence_values": [
            int(v) for v in np.unique(confidence_np).tolist()
        ],
    }


def _discover_nis3d_root(
    samples: tuple[str, ...],
    *,
    data_dir: str,
    execution_mode: str,
) -> Path:
    explicit = str(data_dir).strip()
    if explicit:
        if execution_mode == "modal" and re.match(
            r"^[A-Za-z]:[\\/]",
            explicit,
        ):
            raise ValueError(
                "--data-dir is a Windows path but execution=modal"
            )

        root = Path(explicit).expanduser()
        if not root.is_absolute():
            base = (
                LOCAL_REPO_ROOT / "data"
                if execution_mode == "local"
                else Path(DATA_MOUNT)
            )
            root = base / root
        root = root.resolve()

        missing = [sample for sample in samples if not (root / sample).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"{root} does not contain samples: {missing}"
            )
        return root

    candidates = (
        LOCAL_NIS3D_ROOT_CANDIDATES
        if execution_mode == "local"
        else tuple(Path(value) for value in NIS3D_ROOT_CANDIDATES)
    )
    for root in candidates:
        if all((root / sample).is_dir() for sample in samples):
            return root

    search_root = (
        LOCAL_REPO_ROOT / "data"
        if execution_mode == "local"
        else Path(DATA_MOUNT)
    )
    if search_root.exists():
        first = samples[0]
        for match in list(search_root.glob(f"**/{first}"))[:64]:
            parent = match.parent
            if all((parent / sample).is_dir() for sample in samples):
                return parent

    raise FileNotFoundError(
        f"Could not locate NIS3D samples {samples}. Use --data-dir."
    )


def _data_signature(
    *,
    nis3d_root: Path,
    sample: str,
    spacing_override_zyx_um,
    confidence_ignore_margin_um: float,
) -> str:
    payload = {
        "root": str(nis3d_root),
        "samples": [sample],
        "spacing_override_zyx_um": (
            None
            if spacing_override_zyx_um is None
            else [float(v) for v in spacing_override_zyx_um]
        ),
        "confidence_ignore_margin_um": float(
            confidence_ignore_margin_um
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _prepare_sample_batch(
    *,
    nis3d_root: Path,
    sample: str,
    spacing_override_zyx_um,
    confidence_ignore_margin_um: float,
    cache_root: Path,
    cache_namespace: str,
):
    import torch
    from learned.stirnet.training import (
        prepare_raw_source_volume_cache,
        prepare_raw_training_batch,
    )

    raw, gt, confidence, native_spacing = _load_nis3d_arrays(
        nis3d_root / sample
    )
    spacing = (
        tuple(float(v) for v in native_spacing)
        if spacing_override_zyx_um is None
        else tuple(float(v) for v in spacing_override_zyx_um)
    )

    clean_gt, valid_mask, confidence_report = _prepare_nis3d_gt(
        gt,
        confidence,
        spacing,
        ignore_margin_um=confidence_ignore_margin_um,
    )

    source_cache = (
        cache_root / "source" / cache_namespace / f"{sample}.pt"
    )
    source_cache.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    batch = prepare_raw_training_batch(
        raw,
        clean_gt,
        spacing,
        source_id=f"NIS3D/{sample}@heldout/{cache_namespace}",
        source_cache_path=source_cache,
    )
    source_prepare_seconds = time.perf_counter() - started

    batch = prepare_raw_source_volume_cache(
        batch,
        release_raw_volume=True,
    )
    batch["supervision_valid_mask"] = torch.from_numpy(valid_mask)[None]
    batch["nis3d_sample_name"] = sample

    source_metadata = dict(batch["source_preprocessing_metadata"][0])
    ram_metadata = dict(batch["source_ram_cache_metadata"])

    report = {
        "sample": sample,
        "shape_zyx": [int(v) for v in clean_gt.shape],
        "native_spacing_zyx_um": [float(v) for v in native_spacing],
        "spacing_zyx_um": [float(v) for v in spacing],
        "spacing_source": (
            "Info.txt"
            if spacing_override_zyx_um is None
            else "--spacing-xyz override"
        ),
        "gt_ids_kept": int(confidence_report["gt_ids_kept"]),
        "supervision_valid_fraction": float(
            confidence_report["supervision_valid_fraction"]
        ),
        "source_instance_count": int(
            source_metadata.get("source_current_instance_count", 0)
        ),
        "model_dref_um": float(batch["dref_um"][0]),
        "source_cache_hit": bool(
            source_metadata.get("source_cache_hit", False)
        ),
        "source_prepare_seconds": float(source_prepare_seconds),
        "source_ram_cache_gross_gib": float(
            ram_metadata.get("gross_cache_bytes", 0) / 2**30
        ),
        "source_ram_cache_net_added_gib": float(
            ram_metadata.get("net_added_bytes", 0) / 2**30
        ),
        **confidence_report,
    }
    return batch, report


# ======================================================================================
# Checkpoint resolution / model loading
# ======================================================================================

def _resolve_path(
    value: str,
    *,
    execution_mode: str,
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        root = (
            LOCAL_REPO_ROOT
            if execution_mode == "local"
            else Path(REMOTE_REPO_ROOT)
        )
        path = root / path
    return path.resolve()


def _resolve_baseline_checkpoint(
    explicit: str,
    *,
    execution_mode: str,
) -> Path:
    if explicit:
        path = _resolve_path(explicit, execution_mode=execution_mode)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    runs_root = (
        LOCAL_REPO_ROOT / "runs"
        if execution_mode == "local"
        else Path(RUNS_MOUNT)
    )
    candidates = (
        runs_root
        / "stirnet"
        / "milestones"
        / "drosophila_12_spatial_v1"
        / "checkpoint_step_001000.pt",
        runs_root
        / "stirnet"
        / "training"
        / "01_nis3d_spatial_training"
        / "recovery"
        / "drosophila_12_spatial_v1"
        / "6aa4bd79ebce8248"
        / "checkpoint_step_001000.pt",
    )
    for path in candidates:
        if path.is_file():
            return path

    matches = sorted(
        runs_root.glob(
            "**/drosophila_12_spatial_v1/**/checkpoint_step_001000.pt"
        )
    )
    if matches:
        return matches[-1]

    raise FileNotFoundError(
        "Could not auto-resolve the original step-1000 milestone."
    )


def _resolve_investigation15_checkpoint(
    explicit: str,
    *,
    step: int,
    execution_mode: str,
) -> Path:
    if explicit:
        path = _resolve_path(explicit, execution_mode=execution_mode)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    runs_root = (
        LOCAL_REPO_ROOT / "runs"
        if execution_mode == "local"
        else Path(RUNS_MOUNT)
    )
    direct = (
        runs_root
        / "stirnet"
        / "investigations"
        / "15_morphology_rag_overfit"
        / "recovery"
        / "drosophila_1_morphology_rag_overfit"
        / f"checkpoint_step_{step:06d}.pt"
    )
    if direct.is_file():
        return direct

    matches = sorted(
        runs_root.glob(
            "**/15_morphology_rag_overfit/**/"
            f"checkpoint_step_{step:06d}.pt"
        )
    )
    if matches:
        return matches[-1]

    raise FileNotFoundError(
        "Could not auto-resolve Investigation-15 "
        f"checkpoint step {step}. Pass an explicit checkpoint path."
    )


def _hydrate_dataclass(instance: Any, payload: dict[str, Any]) -> Any:
    if not is_dataclass(instance):
        raise TypeError("Expected a dataclass instance")

    names = {field.name for field in fields(instance)}
    for key, value in payload.items():
        if key not in names:
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _hydrate_dataclass(current, value)
        elif isinstance(current, tuple) and isinstance(value, (tuple, list)):
            setattr(instance, key, tuple(value))
        else:
            setattr(instance, key, value)
    return instance


def _load_model_from_checkpoint(path: Path, *, device: str):
    from learned.stirnet import StirNet, StirNetConfig

    payload = _torch_load(path, map_location="cpu")
    model_cfg = StirNetConfig()
    checkpoint_cfg = payload.get("model_config")
    if isinstance(checkpoint_cfg, dict):
        _hydrate_dataclass(model_cfg, checkpoint_cfg)
    model_cfg.validate()

    model = StirNet(model_cfg)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"{path}: checkpoint has no model state")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{path}: checkpoint/model mismatch. "
            f"missing={list(missing)[:20]} unexpected={list(unexpected)[:20]}"
        )

    model = model.to(device)
    model.eval()
    return model, model_cfg, payload


def _assert_dense_checkpoint_identity(
    baseline_payload: dict,
    other_payload: dict,
    *,
    other_name: str,
) -> dict[str, Any]:
    """Prove the held-out comparison reuses exactly the same dense network."""
    import torch

    baseline = baseline_payload["model"]
    other = other_payload["model"]

    rag_prefixes = ("rag_builder.", "rag_network.")
    checked = 0
    mismatched = []

    for name, baseline_value in baseline.items():
        if name.startswith(rag_prefixes):
            continue
        other_value = other.get(name)
        if other_value is None:
            mismatched.append((name, "missing"))
            continue
        checked += 1
        if tuple(other_value.shape) != tuple(baseline_value.shape):
            mismatched.append((name, "shape"))
            continue
        if not torch.equal(other_value.cpu(), baseline_value.cpu()):
            mismatched.append((name, "value"))

    if mismatched:
        preview = "\n".join(
            f"  - {name}: {reason}"
            for name, reason in mismatched[:20]
        )
        raise RuntimeError(
            f"{other_name} does not preserve the original dense/non-RAG "
            f"checkpoint state exactly:\n{preview}"
        )

    return {
        "checked_non_rag_tensors": int(checked),
        "mismatch_count": 0,
    }


# ======================================================================================
# Manifest selection / crop materialization
# ======================================================================================

def _build_manifest(source_batch: dict, crop_shape_zyx):
    from learned.stirnet.training.merge_aware_crops import (
        build_merge_aware_crop_manifest,
    )

    return build_merge_aware_crop_manifest(
        source_batch["gt_labels"],
        current_labels=source_batch["instance_labels"],
        spacing_um=source_batch["spacing_um"],
        crop_shape_zyx=crop_shape_zyx,
        min_complete_cells=3,
        preferred_complete_cells=4,
        views_per_cell=1,
        context_um=4.0,
        merge_min_overlap_voxels=8,
        merge_min_gt_fraction=0.05,
    )


def _spread_select(rows: list[int], count: int) -> list[int]:
    import numpy as np

    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        return list(rows)

    positions = np.linspace(0, len(rows) - 1, count)
    chosen = []
    seen = set()
    for value in positions:
        index = rows[int(round(float(value)))]
        if index not in seen:
            seen.add(index)
            chosen.append(index)

    # Rounding can theoretically collapse positions. Fill deterministically.
    if len(chosen) < count:
        for index in rows:
            if index not in seen:
                seen.add(index)
                chosen.append(index)
                if len(chosen) >= count:
                    break
    return chosen


def _select_manifest_indices(
    records,
    *,
    count: int,
    exclude: set[int],
) -> list[int]:
    available = [
        index
        for index in range(len(records))
        if index not in exclude
    ]
    if not available:
        raise RuntimeError("All manifest records were excluded")

    if count >= len(available):
        return available

    merge = [
        index for index in available
        if bool(records[index].merge_source_ids)
    ]
    non_merge = [
        index for index in available
        if not bool(records[index].merge_source_ids)
    ]

    # Balanced held-out set where possible. Selection depends only on manifest
    # provenance, never on any checkpoint prediction.
    merge_target = min(len(merge), count // 2)
    non_merge_target = min(len(non_merge), count - merge_target)

    # If one stratum is small, fill from the other.
    remaining = count - merge_target - non_merge_target
    if remaining > 0:
        merge_extra = min(len(merge) - merge_target, remaining)
        merge_target += merge_extra
        remaining -= merge_extra
    if remaining > 0:
        non_merge_target += min(
            len(non_merge) - non_merge_target,
            remaining,
        )

    selected = (
        _spread_select(merge, merge_target)
        + _spread_select(non_merge, non_merge_target)
    )
    selected = sorted(set(selected))

    if len(selected) < count:
        for index in available:
            if index not in selected:
                selected.append(index)
                if len(selected) >= count:
                    break
    return selected[:count]


def _record_to_spec(record, *, full_shape, spacing_um):
    import torch
    from learned.stirnet.training.crops import CropSpec

    slices = record.slices_zyx
    lower = torch.tensor(
        [int(axis.start) for axis in slices],
        dtype=torch.float32,
    )
    size = torch.tensor(
        [
            int(axis.stop) - int(axis.start)
            for axis in slices
        ],
        dtype=torch.float32,
    )
    full_center = 0.5 * (
        torch.tensor(full_shape, dtype=torch.float32) - 1
    )
    crop_center = lower + 0.5 * (size - 1)
    shift = (crop_center - full_center) * spacing_um.detach().cpu().float()

    return CropSpec(
        batch_index=0,
        slices_zyx=slices,
        full_shape_zyx=full_shape,
        center_shift_um=shift,
        candidate_type=record.candidate_type,
        complete_cell_ids=record.complete_cell_ids,
        partial_cell_ids=record.partial_cell_ids,
        true_boundary_cell_ids=record.true_boundary_cell_ids,
        merge_source_ids=record.merge_source_ids,
    )


def _materialize_crop(
    source_batch: dict,
    spec,
    *,
    partial_ignore_margin_um: float,
):
    import torch
    from learned.stirnet.training.crops import prepare_crop_batch
    from learned.stirnet.training.raw_source import (
        materialize_raw_source_crop_batch,
    )

    gt = torch.as_tensor(source_batch["gt_labels"]).long()
    crop = prepare_crop_batch(
        source_batch,
        gt,
        [spec],
        partial_ignore_margin_um=partial_ignore_margin_um,
    )
    crop = materialize_raw_source_crop_batch(
        source_batch,
        crop,
        source_halo_um=4.0,
        dropout_probability=0.0,
        dropout_max_instances=0,
        dropout_seed=0,
        dropout_min_purity=0.80,
        dropout_min_gt_coverage=0.50,
    )

    full_valid = torch.as_tensor(
        source_batch["supervision_valid_mask"]
    ).detach().cpu().bool()
    dataset_valid = full_valid[spec.batch_index][spec.slices_zyx]

    batch = dict(crop.batch)
    existing = torch.as_tensor(
        batch["supervision_valid_mask"]
    ).detach().cpu().bool()
    batch["supervision_valid_mask"] = existing & dataset_valid[None]
    batch["gt_labels"] = crop.gt_labels
    return batch


def _move_crop_to_cuda(batch: dict) -> dict:
    import torch

    result = dict(batch)
    for key in (
        "spatial_inputs",
        "instance_labels",
        "gt_labels",
        "spacing_um",
        "dref_um",
        "supervision_valid_mask",
        "spatial_padding_mask",
    ):
        if key in result and result[key] is not None:
            result[key] = torch.as_tensor(result[key]).to(
                "cuda",
                non_blocking=True,
            )
    return result


def _record_metadata(record, index: int) -> dict[str, Any]:
    return {
        "manifest_index": int(index),
        "candidate_type": str(record.candidate_type),
        "slices_zyx": [
            [int(axis.start), int(axis.stop)]
            for axis in record.slices_zyx
        ],
        "complete_cell_count": len(record.complete_cell_ids),
        "partial_cell_count": len(record.partial_cell_ids),
        "merge_source_count": len(record.merge_source_ids),
        "merge_gt_count": len(record.merge_gt_ids),
        "merge_source_ids": [
            int(value) for value in record.merge_source_ids
        ],
        "merge_gt_ids": [
            int(value) for value in record.merge_gt_ids
        ],
    }


# ======================================================================================
# Metrics
# ======================================================================================

def _rag_targets(output, crop_batch, rag_criterion):
    return rag_criterion.build_targets(
        output.rag,
        crop_batch["gt_labels"],
        valid_mask=crop_batch.get("supervision_valid_mask"),
    )


def _zero_accumulator() -> dict[str, float]:
    return {
        "crop_count": 0.0,
        "all_edge_count": 0.0,
        "valid_edge_count": 0.0,
        "positive_edge_count": 0.0,
        "negative_edge_count": 0.0,
        "bce_sum": 0.0,
        "correct_05_count": 0.0,
        "positive_correct_05_count": 0.0,
        "negative_correct_05_count": 0.0,
        "positive_probability_sum": 0.0,
        "negative_probability_sum": 0.0,
        "negative_false_merge_count": 0.0,
        "positive_merge_accept_count": 0.0,
        "separator_negative_count": 0.0,
        "separator_false_merge_count": 0.0,
        "baseline_hard_negative_count": 0.0,
        "baseline_hard_probability_sum": 0.0,
        "baseline_hard_above_merge_count": 0.0,
        "baseline_separator_hard_count": 0.0,
        "baseline_separator_hard_probability_sum": 0.0,
        "baseline_separator_hard_above_merge_count": 0.0,
    }


def _edge_contribution(
    logits,
    targets,
    *,
    all_edge_count: int,
    merge_threshold: float,
    separator_max,
    separator_threshold: float,
    baseline_hard_negative,
    baseline_separator_hard,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    probabilities = logits.detach().sigmoid()
    valid = targets.valid.bool()
    target = targets.target > 0.5
    positive = valid & target
    negative = valid & ~target
    predicted_05 = probabilities >= 0.5

    separator_negative = (
        negative & (separator_max >= separator_threshold)
    )

    selected_logits = logits[valid]
    selected_target = targets.target[valid]
    bce_sum = (
        float(
            F.binary_cross_entropy_with_logits(
                selected_logits,
                selected_target,
                reduction="sum",
            )
            .detach()
            .float()
            .cpu()
        )
        if bool(valid.any())
        else 0.0
    )

    def count(mask) -> float:
        return float(mask.sum().item())

    def probability_sum(mask) -> float:
        if not bool(mask.any()):
            return 0.0
        return float(probabilities[mask].float().sum().cpu())

    return {
        "crop_count": 1.0,
        "all_edge_count": float(all_edge_count),
        "valid_edge_count": count(valid),
        "positive_edge_count": count(positive),
        "negative_edge_count": count(negative),
        "bce_sum": bce_sum,
        "correct_05_count": count(
            valid & (predicted_05 == target)
        ),
        "positive_correct_05_count": count(
            positive & predicted_05
        ),
        "negative_correct_05_count": count(
            negative & ~predicted_05
        ),
        "positive_probability_sum": probability_sum(positive),
        "negative_probability_sum": probability_sum(negative),
        "negative_false_merge_count": count(
            negative & (probabilities >= merge_threshold)
        ),
        "positive_merge_accept_count": count(
            positive & (probabilities >= merge_threshold)
        ),
        "separator_negative_count": count(separator_negative),
        "separator_false_merge_count": count(
            separator_negative
            & (probabilities >= merge_threshold)
        ),
        "baseline_hard_negative_count": count(
            baseline_hard_negative
        ),
        "baseline_hard_probability_sum": probability_sum(
            baseline_hard_negative
        ),
        "baseline_hard_above_merge_count": count(
            baseline_hard_negative
            & (probabilities >= merge_threshold)
        ),
        "baseline_separator_hard_count": count(
            baseline_separator_hard
        ),
        "baseline_separator_hard_probability_sum": probability_sum(
            baseline_separator_hard
        ),
        "baseline_separator_hard_above_merge_count": count(
            baseline_separator_hard
            & (probabilities >= merge_threshold)
        ),
    }


def _add_accumulator(
    accumulator: dict[str, float],
    contribution: dict[str, float],
) -> None:
    for key, value in contribution.items():
        accumulator[key] = accumulator.get(key, 0.0) + float(value)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _finalize_accumulator(
    acc: dict[str, float],
) -> dict[str, Any]:
    valid = acc["valid_edge_count"]
    positive = acc["positive_edge_count"]
    negative = acc["negative_edge_count"]
    separator_negative = acc["separator_negative_count"]
    hard = acc["baseline_hard_negative_count"]
    separator_hard = acc["baseline_separator_hard_count"]

    return {
        "crop_count": int(acc["crop_count"]),
        "all_edge_count": int(acc["all_edge_count"]),
        "valid_edge_count": int(valid),
        "positive_edge_count": int(positive),
        "negative_edge_count": int(negative),
        "bce": _safe_ratio(acc["bce_sum"], valid),
        "accuracy_05": _safe_ratio(
            acc["correct_05_count"],
            valid,
        ),
        "positive_accuracy_05": _safe_ratio(
            acc["positive_correct_05_count"],
            positive,
        ),
        "negative_accuracy_05": _safe_ratio(
            acc["negative_correct_05_count"],
            negative,
        ),
        "mean_positive_probability": _safe_ratio(
            acc["positive_probability_sum"],
            positive,
        ),
        "mean_negative_probability": _safe_ratio(
            acc["negative_probability_sum"],
            negative,
        ),
        "negative_false_merge_count": int(
            acc["negative_false_merge_count"]
        ),
        "negative_false_merge_rate": _safe_ratio(
            acc["negative_false_merge_count"],
            negative,
        ),
        "positive_merge_accept_count": int(
            acc["positive_merge_accept_count"]
        ),
        "positive_merge_accept_rate": _safe_ratio(
            acc["positive_merge_accept_count"],
            positive,
        ),
        "positive_missed_merge_count": int(
            positive - acc["positive_merge_accept_count"]
        ),
        "separator_negative_count": int(separator_negative),
        "separator_false_merge_count": int(
            acc["separator_false_merge_count"]
        ),
        "separator_false_merge_rate": _safe_ratio(
            acc["separator_false_merge_count"],
            separator_negative,
        ),
        "baseline_hard_negative_count": int(hard),
        "baseline_hard_mean_probability": _safe_ratio(
            acc["baseline_hard_probability_sum"],
            hard,
        ),
        "baseline_hard_above_merge_count": int(
            acc["baseline_hard_above_merge_count"]
        ),
        "baseline_hard_above_merge_rate": _safe_ratio(
            acc["baseline_hard_above_merge_count"],
            hard,
        ),
        "baseline_separator_hard_count": int(separator_hard),
        "baseline_separator_hard_mean_probability": _safe_ratio(
            acc["baseline_separator_hard_probability_sum"],
            separator_hard,
        ),
        "baseline_separator_hard_above_merge_count": int(
            acc["baseline_separator_hard_above_merge_count"]
        ),
        "baseline_separator_hard_above_merge_rate": _safe_ratio(
            acc["baseline_separator_hard_above_merge_count"],
            separator_hard,
        ),
    }


def _delta(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float]:
    keys = (
        "bce",
        "accuracy_05",
        "positive_accuracy_05",
        "negative_accuracy_05",
        "mean_positive_probability",
        "mean_negative_probability",
        "negative_false_merge_rate",
        "positive_merge_accept_rate",
        "separator_false_merge_rate",
        "baseline_hard_mean_probability",
        "baseline_hard_above_merge_rate",
        "baseline_separator_hard_mean_probability",
        "baseline_separator_hard_above_merge_rate",
    )
    return {
        key: float(candidate[key] - baseline[key])
        for key in keys
    }


def _per_crop_display(
    contribution: dict[str, float],
) -> tuple[int, int]:
    return (
        int(contribution["negative_false_merge_count"]),
        int(contribution["positive_merge_accept_count"]),
    )


# ======================================================================================
# Main evaluation
# ======================================================================================

def _print_header(
    *,
    execution_mode: str,
    checkpoints: dict[str, Path],
    nis3d_root: Path,
    samples: tuple[str, ...],
    crops_per_sample: int,
    exclude_by_sample: dict[str, set[int]],
    crop_shape,
    spacing_override_zyx_um,
    merge_threshold: float,
    separator_threshold: float,
    amp_dtype: str,
    results_root: Path,
):
    import torch

    props = torch.cuda.get_device_properties(0)
    print("=" * 122, flush=True)
    print(
        "STIR-Net Investigation 16 — morphology-aware RAG held-out validation",
        flush=True,
    )
    print("=" * 122, flush=True)
    print(f"GPU                      : {props.name}", flush=True)
    print(
        f"GPU VRAM                 : {props.total_memory / 2**30:.2f} GiB",
        flush=True,
    )
    print(f"Execution                : {execution_mode}", flush=True)
    print(f"Results root             : {results_root}", flush=True)
    for name in MODEL_NAMES:
        print(
            f"{name:<24}: {checkpoints[name]}",
            flush=True,
        )
    print(f"NIS3D root               : {nis3d_root}", flush=True)
    print(f"Samples                   : {list(samples)}", flush=True)
    print(f"Held-out crops/sample     : {crops_per_sample}", flush=True)
    for sample in samples:
        print(
            f"Excluded {sample:<15}: "
            f"{sorted(exclude_by_sample.get(sample, set()))}",
            flush=True,
        )
    print(f"Crop shape ZYX            : {crop_shape}", flush=True)
    if spacing_override_zyx_um is None:
        print("Effective spacing         : Info.txt", flush=True)
    else:
        z, y, x = spacing_override_zyx_um
        print(
            "Effective spacing XYZ um  : "
            f"({x:.8g}, {y:.8g}, {z:.8g})",
            flush=True,
        )
    print(
        f"Production merge threshold: {merge_threshold:.3f}",
        flush=True,
    )
    print(
        f"Separator max threshold   : {separator_threshold:.3f}",
        flush=True,
    )
    print(f"AMP                       : {amp_dtype}", flush=True)
    print(
        "Optimization              : NONE — evaluation only",
        flush=True,
    )
    print(
        "Dense geometry reuse      : original milestone once/crop, then shared A/B/C",
        flush=True,
    )
    print("=" * 122, flush=True)


def _write_comparison_csv(
    path: Path,
    summary: dict[str, Any],
) -> None:
    metrics = (
        "crop_count",
        "valid_edge_count",
        "positive_edge_count",
        "negative_edge_count",
        "bce",
        "accuracy_05",
        "positive_accuracy_05",
        "negative_accuracy_05",
        "mean_positive_probability",
        "mean_negative_probability",
        "negative_false_merge_count",
        "negative_false_merge_rate",
        "positive_merge_accept_count",
        "positive_merge_accept_rate",
        "separator_negative_count",
        "separator_false_merge_count",
        "separator_false_merge_rate",
        "baseline_hard_negative_count",
        "baseline_hard_mean_probability",
        "baseline_hard_above_merge_count",
        "baseline_hard_above_merge_rate",
        "baseline_separator_hard_count",
        "baseline_separator_hard_mean_probability",
        "baseline_separator_hard_above_merge_count",
        "baseline_separator_hard_above_merge_rate",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("scope", "model", *metrics),
        )
        writer.writeheader()

        scopes = ["overall", *summary["samples"].keys()]
        for scope in scopes:
            rows = (
                summary["overall"]
                if scope == "overall"
                else summary["samples"][scope]["models"]
            )
            for model_name in MODEL_NAMES:
                row = rows[model_name]
                writer.writerow(
                    {
                        "scope": scope,
                        "model": model_name,
                        **{key: row.get(key) for key in metrics},
                    }
                )


def _evaluation_impl(
    *,
    samples_csv: str = "Drosophila_1,Drosophila_2",
    crops_per_sample: int = 24,
    exclude_d1_indices: str = "3",
    exclude_d2_indices: str = "",
    baseline_checkpoint: str = "",
    morph_checkpoint: str = "",
    joint_checkpoint: str = "",
    data_dir: str = "",
    spacing_xyz: str = "0.20312639,0.20312639,0.79099447",
    crop_shape_zyx: str = "32,192,192",
    confidence_ignore_margin_um: float = 1.0,
    partial_ignore_margin_um: float = 1.0,
    merge_threshold: float = 0.845,
    baseline_hard_threshold: float = 0.50,
    separator_threshold: float = 0.50,
    run_name: str = "drosophila_12_morphology_rag_heldout",
    seed: int = 230525,
    execution_mode: str = "local",
) -> dict[str, Any]:
    import torch
    from learned.stirnet.model.partition.rag import RAGCriterion

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Investigation 16")
    if execution_mode not in {"local", "modal"}:
        raise ValueError("execution_mode must be local or modal")
    if crops_per_sample < 1:
        raise ValueError("crops_per_sample must be positive")
    if not 0 <= merge_threshold <= 1:
        raise ValueError("merge_threshold must be in [0,1]")
    if not 0 <= baseline_hard_threshold <= 1:
        raise ValueError("baseline_hard_threshold must be in [0,1]")
    if not 0 <= separator_threshold <= 1:
        raise ValueError("separator_threshold must be in [0,1]")

    samples = tuple(
        token.strip()
        for token in samples_csv.split(",")
        if token.strip()
    )
    if not samples:
        raise ValueError("At least one sample is required")

    crop_shape = _parse_shape_zyx(crop_shape_zyx)
    spacing_override_zyx_um = _parse_spacing_xyz_override(
        spacing_xyz
    )
    amp_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    _seed_everything(seed)

    checkpoints = {
        "baseline_step1000": _resolve_baseline_checkpoint(
            baseline_checkpoint,
            execution_mode=execution_mode,
        ),
        "morphology_step100": _resolve_investigation15_checkpoint(
            morph_checkpoint,
            step=100,
            execution_mode=execution_mode,
        ),
        "joint_step300": _resolve_investigation15_checkpoint(
            joint_checkpoint,
            step=300,
            execution_mode=execution_mode,
        ),
    }

    nis3d_root = _discover_nis3d_root(
        samples,
        data_dir=data_dir,
        execution_mode=execution_mode,
    )

    results_mount = (
        LOCAL_REPO_ROOT / "runs"
        if execution_mode == "local"
        else Path(RUNS_MOUNT)
    )
    experiment_root = (
        results_mount
        / "stirnet"
        / "investigations"
        / "16_morphology_rag_heldout_validation"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = experiment_root / f"{timestamp}_{run_name}"
    cache_root = experiment_root / "cache"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    exclude_by_sample = {
        "Drosophila_1": _parse_int_set(exclude_d1_indices),
        "Drosophila_2": _parse_int_set(exclude_d2_indices),
    }

    _print_header(
        execution_mode=execution_mode,
        checkpoints=checkpoints,
        nis3d_root=nis3d_root,
        samples=samples,
        crops_per_sample=crops_per_sample,
        exclude_by_sample=exclude_by_sample,
        crop_shape=crop_shape,
        spacing_override_zyx_um=spacing_override_zyx_um,
        merge_threshold=merge_threshold,
        separator_threshold=separator_threshold,
        amp_dtype=amp_dtype,
        results_root=experiment_root,
    )

    print("[init] loading checkpoint A/B/C models ...", flush=True)
    models = {}
    model_cfgs = {}
    payloads = {}

    for name in MODEL_NAMES:
        model, cfg, payload = _load_model_from_checkpoint(
            checkpoints[name],
            device="cuda",
        )
        models[name] = model
        model_cfgs[name] = cfg
        payloads[name] = payload
        print(
            f"[init] {name}: checkpoint_step="
            f"{payload.get('global_step', '?')} "
            f"morphology_enabled={cfg.partition.rag_morphology_enabled}",
            flush=True,
        )

    if model_cfgs["baseline_step1000"].partition.rag_morphology_enabled:
        raise RuntimeError(
            "The original milestone unexpectedly has morphology enabled"
        )
    if not model_cfgs["morphology_step100"].partition.rag_morphology_enabled:
        raise RuntimeError("Step-100 checkpoint does not enable morphology")
    if not model_cfgs["joint_step300"].partition.rag_morphology_enabled:
        raise RuntimeError("Step-300 checkpoint does not enable morphology")

    dense_identity = {
        "morphology_step100": _assert_dense_checkpoint_identity(
            payloads["baseline_step1000"],
            payloads["morphology_step100"],
            other_name="morphology_step100",
        ),
        "joint_step300": _assert_dense_checkpoint_identity(
            payloads["baseline_step1000"],
            payloads["joint_step300"],
            other_name="joint_step300",
        ),
    }
    print(
        "[init] dense/non-RAG checkpoint identity verified exactly for B/C.",
        flush=True,
    )

    # RAG target validity semantics are the same for all checkpoints.
    rag_criterion = RAGCriterion(
        model_cfgs["baseline_step1000"].partition
    ).to("cuda")
    rag_criterion.eval()

    total_requested = crops_per_sample * len(samples)
    GREEN = "\033[32m"
    RESET = "\033[0m"
    progress = tqdm(
        total=total_requested,
        desc=f"{GREEN}Heldout-RAG{RESET}",
        unit="crop",
        dynamic_ncols=True,
        smoothing=0.10,
        mininterval=0.5,
        leave=True,
        colour="green",
        file=sys.stdout,
    )

    raw_accumulators = {
        "overall": {
            name: _zero_accumulator()
            for name in MODEL_NAMES
        }
    }
    sample_summaries: dict[str, Any] = {}
    per_crop_path = run_dir / "per_crop.jsonl"

    started_all = time.perf_counter()
    evaluated_crop_count = 0
    topology_mismatch_count = 0
    peak_allocated_gib = 0.0
    peak_reserved_gib = 0.0

    try:
        for sample in samples:
            print(f"\n[data] preparing {sample} ...", flush=True)
            signature = _data_signature(
                nis3d_root=nis3d_root,
                sample=sample,
                spacing_override_zyx_um=spacing_override_zyx_um,
                confidence_ignore_margin_um=confidence_ignore_margin_um,
            )
            source_batch, sample_report = _prepare_sample_batch(
                nis3d_root=nis3d_root,
                sample=sample,
                spacing_override_zyx_um=spacing_override_zyx_um,
                confidence_ignore_margin_um=confidence_ignore_margin_um,
                cache_root=cache_root,
                cache_namespace=signature,
            )
            print(
                f"[data] {sample}: shape={tuple(sample_report['shape_zyx'])} "
                f"GT={sample_report['gt_ids_kept']} "
                f"source={sample_report['source_instance_count']} "
                f"dref={sample_report['model_dref_um']:.4f}um "
                f"RAM-cache={sample_report['source_ram_cache_gross_gib']:.2f}GiB",
                flush=True,
            )

            manifest = _build_manifest(source_batch, crop_shape)
            records = list(manifest.records[0])
            exclude = exclude_by_sample.get(sample, set())
            selected_indices = _select_manifest_indices(
                records,
                count=crops_per_sample,
                exclude=exclude,
            )

            if len(selected_indices) < crops_per_sample:
                tqdm.write(
                    f"[warning] {sample}: only {len(selected_indices)} "
                    f"eligible manifest crops are available."
                )
                progress.total -= (
                    crops_per_sample - len(selected_indices)
                )
                progress.refresh()

            merge_selected = sum(
                bool(records[index].merge_source_ids)
                for index in selected_indices
            )
            print(
                f"[select] {sample}: manifest={len(records)} "
                f"selected={len(selected_indices)} "
                f"merge={merge_selected} "
                f"nonmerge={len(selected_indices)-merge_selected} "
                f"excluded={sorted(exclude)}",
                flush=True,
            )

            sample_accumulators = {
                name: _zero_accumulator()
                for name in MODEL_NAMES
            }
            sample_metadata = {
                "sample_report": sample_report,
                "manifest_record_count": len(records),
                "selected_manifest_indices": selected_indices,
                "excluded_manifest_indices": sorted(exclude),
                "selected_merge_crop_count": int(merge_selected),
                "selected_nonmerge_crop_count": int(
                    len(selected_indices) - merge_selected
                ),
            }

            full_shape = tuple(
                int(v)
                for v in source_batch["gt_labels"].shape[-3:]
            )
            spacing = source_batch["spacing_um"][0]

            for local_ordinal, manifest_index in enumerate(
                selected_indices,
                1,
            ):
                record = records[manifest_index]
                spec = _record_to_spec(
                    record,
                    full_shape=full_shape,
                    spacing_um=spacing,
                )
                crop_cpu = _materialize_crop(
                    source_batch,
                    spec,
                    partial_ignore_margin_um=partial_ignore_margin_um,
                )
                crop = _move_crop_to_cuda(crop_cpu)

                torch.cuda.reset_peak_memory_stats()
                crop_started = time.perf_counter()

                # Dense geometry is common by exact checkpoint identity.
                baseline_model = models["baseline_step1000"]
                with torch.no_grad(), _autocast_context(amp_dtype):
                    geometry = baseline_model(
                        crop["spatial_inputs"],
                        crop["spacing_um"],
                        crop["dref_um"],
                        spatial_padding_mask=crop.get(
                            "spatial_padding_mask"
                        ),
                        execution_stage="geometry",
                    )

                outputs = {}
                with torch.no_grad(), _autocast_context(amp_dtype):
                    for name in MODEL_NAMES:
                        outputs[name] = models[name](
                            crop["spatial_inputs"],
                            crop["spacing_um"],
                            crop["dref_um"],
                            spatial_padding_mask=crop.get(
                                "spatial_padding_mask"
                            ),
                            execution_stage="spatial",
                            precomputed_geometry=geometry,
                        )

                baseline_output = outputs["baseline_step1000"]
                targets = _rag_targets(
                    baseline_output,
                    crop,
                    rag_criterion,
                )

                baseline_edge_index = baseline_output.rag.edge_index
                for name in MODEL_NAMES[1:]:
                    current_edge_index = outputs[name].rag.edge_index
                    if (
                        current_edge_index.shape != baseline_edge_index.shape
                        or not torch.equal(
                            current_edge_index,
                            baseline_edge_index,
                        )
                    ):
                        topology_mismatch_count += 1
                        raise RuntimeError(
                            f"{sample} manifest={manifest_index}: "
                            f"RAG topology differs for {name}. "
                            "Held-out edge alignment is invalid."
                        )

                probabilities_a = (
                    baseline_output.rag.spatial_edge_logits.detach().sigmoid()
                )
                valid = targets.valid.bool()
                negative = valid & (targets.target <= 0.5)
                separator_max = (
                    baseline_output.rag.edge_features[:, 1].detach()
                    if baseline_output.rag.edge_features.numel()
                    else probabilities_a.new_zeros(probabilities_a.shape)
                )
                baseline_hard_negative = (
                    negative
                    & (probabilities_a >= baseline_hard_threshold)
                )
                baseline_separator_hard = (
                    baseline_hard_negative
                    & (separator_max >= separator_threshold)
                )

                crop_result = {
                    "sample": sample,
                    "sample_crop_ordinal": int(local_ordinal),
                    **_record_metadata(record, manifest_index),
                    "baseline_hard_negative_count": int(
                        baseline_hard_negative.sum().item()
                    ),
                    "baseline_separator_hard_count": int(
                        baseline_separator_hard.sum().item()
                    ),
                    "models": {},
                }

                display_false_merges = {}
                for name in MODEL_NAMES:
                    output = outputs[name]
                    contribution = _edge_contribution(
                        output.rag.spatial_edge_logits,
                        targets,
                        all_edge_count=int(
                            output.rag.edge_index.shape[1]
                        ),
                        merge_threshold=merge_threshold,
                        separator_max=separator_max,
                        separator_threshold=separator_threshold,
                        baseline_hard_negative=baseline_hard_negative,
                        baseline_separator_hard=baseline_separator_hard,
                    )
                    crop_metrics = _finalize_accumulator(contribution)
                    crop_result["models"][name] = crop_metrics

                    _add_accumulator(
                        sample_accumulators[name],
                        contribution,
                    )
                    _add_accumulator(
                        raw_accumulators["overall"][name],
                        contribution,
                    )
                    display_false_merges[name] = int(
                        contribution["negative_false_merge_count"]
                    )

                torch.cuda.synchronize()
                crop_seconds = time.perf_counter() - crop_started
                crop_result["crop_seconds"] = float(crop_seconds)
                _append_jsonl(per_crop_path, crop_result)

                evaluated_crop_count += 1
                peak_allocated_gib = max(
                    peak_allocated_gib,
                    torch.cuda.max_memory_allocated() / 2**30,
                )
                peak_reserved_gib = max(
                    peak_reserved_gib,
                    torch.cuda.max_memory_reserved() / 2**30,
                )

                progress.set_postfix(
                    {
                        "sample": sample.replace("Drosophila_", "D"),
                        "idx": manifest_index,
                        "valid": int(valid.sum().item()),
                        "FM-A": display_false_merges[
                            "baseline_step1000"
                        ],
                        "FM-B": display_false_merges[
                            "morphology_step100"
                        ],
                        "FM-C": display_false_merges[
                            "joint_step300"
                        ],
                    },
                    refresh=False,
                )
                progress.update(1)

                del outputs, baseline_output, geometry, targets
                del crop, crop_cpu
                torch.cuda.empty_cache()

            finalized_models = {
                name: _finalize_accumulator(sample_accumulators[name])
                for name in MODEL_NAMES
            }
            sample_summaries[sample] = {
                **sample_metadata,
                "models": finalized_models,
                "delta_vs_baseline": {
                    name: _delta(
                        finalized_models[name],
                        finalized_models["baseline_step1000"],
                    )
                    for name in MODEL_NAMES[1:]
                },
            }

            # Release the large full-volume host cache before loading next sample.
            del source_batch, manifest, records
            gc.collect()
            torch.cuda.empty_cache()

        progress.close()

        overall = {
            name: _finalize_accumulator(
                raw_accumulators["overall"][name]
            )
            for name in MODEL_NAMES
        }
        elapsed = time.perf_counter() - started_all

        summary = {
            "status": "success",
            "experiment": "16_morphology_rag_heldout_validation",
            "run_dir": str(run_dir),
            "checkpoints": {
                name: {
                    "path": str(checkpoints[name]),
                    "global_step": int(
                        payloads[name].get("global_step", -1)
                    ),
                    "rag_morphology_enabled": bool(
                        model_cfgs[name].partition.rag_morphology_enabled
                    ),
                }
                for name in MODEL_NAMES
            },
            "dense_identity": dense_identity,
            "samples": sample_summaries,
            "overall": overall,
            "overall_delta_vs_baseline": {
                name: _delta(
                    overall[name],
                    overall["baseline_step1000"],
                )
                for name in MODEL_NAMES[1:]
            },
            "settings": {
                "samples": list(samples),
                "crops_per_sample": int(crops_per_sample),
                "crop_shape_zyx": list(crop_shape),
                "spacing_override_zyx_um": (
                    None
                    if spacing_override_zyx_um is None
                    else [
                        float(v)
                        for v in spacing_override_zyx_um
                    ]
                ),
                "merge_threshold": float(merge_threshold),
                "baseline_hard_threshold": float(
                    baseline_hard_threshold
                ),
                "separator_threshold": float(separator_threshold),
                "confidence_ignore_margin_um": float(
                    confidence_ignore_margin_um
                ),
                "partial_ignore_margin_um": float(
                    partial_ignore_margin_um
                ),
                "amp_dtype": amp_dtype,
                "seed": int(seed),
                "model_independent_crop_selection": True,
                "dense_geometry_computed_once_per_crop": True,
            },
            "evaluated_crop_count": int(evaluated_crop_count),
            "topology_mismatch_count": int(topology_mismatch_count),
            "elapsed_seconds": float(elapsed),
            "mean_seconds_per_crop": float(
                elapsed / max(evaluated_crop_count, 1)
            ),
            "peak_cuda_allocated_gib": float(peak_allocated_gib),
            "peak_cuda_reserved_gib": float(peak_reserved_gib),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }

        _atomic_json(run_dir / "summary.json", summary)
        _write_comparison_csv(
            run_dir / "comparison.csv",
            summary,
        )
        if execution_mode == "modal":
            runs_volume.commit()

        print("", flush=True)
        print("=" * 122, flush=True)
        print("Investigation 16 complete", flush=True)
        print("=" * 122, flush=True)
        print(
            f"Evaluated crops          : {evaluated_crop_count}",
            flush=True,
        )
        print(
            f"Elapsed                  : {_duration(elapsed)}",
            flush=True,
        )
        print(
            f"Mean sec/crop            : "
            f"{elapsed / max(evaluated_crop_count, 1):.2f}",
            flush=True,
        )
        print(
            f"Peak CUDA allocated      : {peak_allocated_gib:.2f} GiB",
            flush=True,
        )
        print("", flush=True)

        for name in MODEL_NAMES:
            row = overall[name]
            print(
                f"{name}: "
                f"BCE={row['bce']:.5f} "
                f"acc={row['accuracy_05']:.4f} "
                f"pos_p={row['mean_positive_probability']:.4f} "
                f"neg_p={row['mean_negative_probability']:.4f} "
                f"false_merge@{merge_threshold:.3f}="
                f"{row['negative_false_merge_count']}/"
                f"{row['negative_edge_count']} "
                f"pos_accept={row['positive_merge_accept_rate']:.4f} "
                f"sep_false_merge="
                f"{row['separator_false_merge_count']}/"
                f"{row['separator_negative_count']} "
                f"baseline_hard_p="
                f"{row['baseline_hard_mean_probability']:.4f}",
                flush=True,
            )

        print("", flush=True)
        for name in MODEL_NAMES[1:]:
            delta = summary["overall_delta_vs_baseline"][name]
            print(
                f"{name} vs baseline: "
                f"Δfalse_merge_rate="
                f"{delta['negative_false_merge_rate']:+.5f} "
                f"Δpos_accept="
                f"{delta['positive_merge_accept_rate']:+.5f} "
                f"Δhard_p="
                f"{delta['baseline_hard_mean_probability']:+.5f} "
                f"ΔBCE={delta['bce']:+.5f}",
                flush=True,
            )

        print(f"Run directory           : {run_dir}", flush=True)
        print("=" * 122, flush=True)
        return summary

    except BaseException as error:
        progress.close()
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "evaluated_crop_count": int(evaluated_crop_count),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(run_dir / "_FAILED.json", failure)
        if execution_mode == "modal":
            try:
                runs_volume.commit()
            except BaseException:
                pass
        raise


# ======================================================================================
# Modal wrapper
# ======================================================================================

@_modal_function_decorator(
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    image=image,
    volumes=(
        {}
        if DIRECT_LOCAL_EXECUTION
        else {
            DATA_MOUNT: data_volume,
            RUNS_MOUNT: runs_volume,
        }
    ),
)
def evaluate_morphology_rag_heldout(
    samples_csv: str = "Drosophila_1,Drosophila_2",
    crops_per_sample: int = 24,
    exclude_d1_indices: str = "3",
    exclude_d2_indices: str = "",
    baseline_checkpoint: str = "",
    morph_checkpoint: str = "",
    joint_checkpoint: str = "",
    data_dir: str = "",
    spacing_xyz: str = "0.20312639,0.20312639,0.79099447",
    crop_shape_zyx: str = "32,192,192",
    confidence_ignore_margin_um: float = 1.0,
    partial_ignore_margin_um: float = 1.0,
    merge_threshold: float = 0.845,
    baseline_hard_threshold: float = 0.50,
    separator_threshold: float = 0.50,
    run_name: str = "drosophila_12_morphology_rag_heldout",
    seed: int = 230525,
):
    return _evaluation_impl(
        samples_csv=samples_csv,
        crops_per_sample=crops_per_sample,
        exclude_d1_indices=exclude_d1_indices,
        exclude_d2_indices=exclude_d2_indices,
        baseline_checkpoint=baseline_checkpoint,
        morph_checkpoint=morph_checkpoint,
        joint_checkpoint=joint_checkpoint,
        data_dir=data_dir,
        spacing_xyz=spacing_xyz,
        crop_shape_zyx=crop_shape_zyx,
        confidence_ignore_margin_um=confidence_ignore_margin_um,
        partial_ignore_margin_um=partial_ignore_margin_um,
        merge_threshold=merge_threshold,
        baseline_hard_threshold=baseline_hard_threshold,
        separator_threshold=separator_threshold,
        run_name=run_name,
        seed=seed,
        execution_mode="modal",
    )


@_modal_local_entrypoint_decorator()
def main(
    samples_csv: str = "Drosophila_1,Drosophila_2",
    crops_per_sample: int = 24,
    exclude_d1_indices: str = "3",
    exclude_d2_indices: str = "",
    baseline_checkpoint: str = "",
    morph_checkpoint: str = "",
    joint_checkpoint: str = "",
    data_dir: str = "",
    spacing_xyz: str = "0.20312639,0.20312639,0.79099447",
    crop_shape_zyx: str = "32,192,192",
    confidence_ignore_margin_um: float = 1.0,
    partial_ignore_margin_um: float = 1.0,
    merge_threshold: float = 0.845,
    baseline_hard_threshold: float = 0.50,
    separator_threshold: float = 0.50,
    run_name: str = "drosophila_12_morphology_rag_heldout",
    seed: int = 230525,
):
    call = evaluate_morphology_rag_heldout.spawn(
        samples_csv=samples_csv,
        crops_per_sample=crops_per_sample,
        exclude_d1_indices=exclude_d1_indices,
        exclude_d2_indices=exclude_d2_indices,
        baseline_checkpoint=baseline_checkpoint,
        morph_checkpoint=morph_checkpoint,
        joint_checkpoint=joint_checkpoint,
        data_dir=data_dir,
        spacing_xyz=spacing_xyz,
        crop_shape_zyx=crop_shape_zyx,
        confidence_ignore_margin_um=confidence_ignore_margin_um,
        partial_ignore_margin_um=partial_ignore_margin_um,
        merge_threshold=merge_threshold,
        baseline_hard_threshold=baseline_hard_threshold,
        separator_threshold=separator_threshold,
        run_name=run_name,
        seed=seed,
    )
    print(
        f"[launcher] spawned held-out validation call: {call.object_id}",
        flush=True,
    )
    result = call.get()
    print(json.dumps(result, indent=2))


# ======================================================================================
# Direct Python CLI
# ======================================================================================

def _python_cli_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate baseline / morphology-only / joint-RAG checkpoints "
            "on held-out Drosophila crops."
        )
    )
    parser.add_argument(
        "--execution",
        choices=("local", "modal"),
        default="local",
    )
    parser.add_argument(
        "--samples",
        default="Drosophila_1,Drosophila_2",
    )
    parser.add_argument(
        "--crops-per-sample",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--exclude-d1-indices",
        default="3",
        help=(
            "Comma-separated Drosophila_1 manifest indices to exclude. "
            "Default 3 is the Investigation-15 training crop."
        ),
    )
    parser.add_argument(
        "--exclude-d2-indices",
        default="",
    )
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--morph-checkpoint", default="")
    parser.add_argument("--joint-checkpoint", default="")
    parser.add_argument("--data-dir", default="")
    parser.add_argument(
        "--spacing-xyz",
        default="0.20312639,0.20312639,0.79099447",
    )
    parser.add_argument(
        "--crop-shape-zyx",
        default="32,192,192",
    )
    parser.add_argument(
        "--confidence-ignore-margin-um",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--partial-ignore-margin-um",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.845,
    )
    parser.add_argument(
        "--baseline-hard-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--separator-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--run-name",
        default="drosophila_12_morphology_rag_heldout",
    )
    parser.add_argument("--seed", type=int, default=230525)
    args = parser.parse_args()

    kwargs = dict(
        samples_csv=args.samples,
        crops_per_sample=args.crops_per_sample,
        exclude_d1_indices=args.exclude_d1_indices,
        exclude_d2_indices=args.exclude_d2_indices,
        baseline_checkpoint=args.baseline_checkpoint,
        morph_checkpoint=args.morph_checkpoint,
        joint_checkpoint=args.joint_checkpoint,
        data_dir=args.data_dir,
        spacing_xyz=args.spacing_xyz,
        crop_shape_zyx=args.crop_shape_zyx,
        confidence_ignore_margin_um=args.confidence_ignore_margin_um,
        partial_ignore_margin_um=args.partial_ignore_margin_um,
        merge_threshold=args.merge_threshold,
        baseline_hard_threshold=args.baseline_hard_threshold,
        separator_threshold=args.separator_threshold,
        run_name=args.run_name,
        seed=args.seed,
    )

    if args.execution == "local":
        result = _evaluation_impl(
            **kwargs,
            execution_mode="local",
        )
        print(json.dumps(result, indent=2))
        return

    modal_exe = shutil.which("modal")
    if modal_exe is None:
        raise RuntimeError(
            "Could not find the `modal` executable in the active environment"
        )

    command = [
        modal_exe,
        "run",
        str(Path(__file__).resolve()),
        "--samples",
        args.samples,
        "--crops-per-sample",
        str(args.crops_per_sample),
        "--exclude-d1-indices",
        args.exclude_d1_indices,
        "--exclude-d2-indices",
        args.exclude_d2_indices,
        "--spacing-xyz",
        args.spacing_xyz,
        "--crop-shape-zyx",
        args.crop_shape_zyx,
        "--confidence-ignore-margin-um",
        str(args.confidence_ignore_margin_um),
        "--partial-ignore-margin-um",
        str(args.partial_ignore_margin_um),
        "--merge-threshold",
        str(args.merge_threshold),
        "--baseline-hard-threshold",
        str(args.baseline_hard_threshold),
        "--separator-threshold",
        str(args.separator_threshold),
        "--run-name",
        args.run_name,
        "--seed",
        str(args.seed),
    ]
    if args.baseline_checkpoint:
        command.extend(
            ["--baseline-checkpoint", args.baseline_checkpoint]
        )
    if args.morph_checkpoint:
        command.extend(
            ["--morph-checkpoint", args.morph_checkpoint]
        )
    if args.joint_checkpoint:
        command.extend(
            ["--joint-checkpoint", args.joint_checkpoint]
        )
    if args.data_dir:
        command.extend(["--data-dir", args.data_dir])

    print(
        "[launcher] execution=modal\n"
        "[launcher] " + " ".join(command),
        flush=True,
    )
    completed = subprocess.run(command, cwd=LOCAL_REPO_ROOT)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    _python_cli_main()
