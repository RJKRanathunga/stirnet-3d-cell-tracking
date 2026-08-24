from __future__ import annotations

"""
STIR-Net Investigation 15 — morphology-aware RAG fixed-crop overfit.

Purpose
-------
Prove that the new morphology-aware RAG can learn the current bottleneck before
launching another production training run.

This stage deliberately DOES NOT retrain dense geometry.

It:
* starts from the known-good Drosophila_1 + Drosophila_2 step-1000 milestone;
* enables partition.rag_morphology_enabled;
* transfers every exact-name / exact-shape milestone parameter;
* freezes the dense spatial / geometry network permanently;
* prepares NIS3D exactly like Training 01, including ConfidenceScore handling,
  effective spacing override, source preprocessing, and aligned source RAM cache;
* chooses one fixed Drosophila crop with useful RAG supervision;
* prefers crops containing high-probability false-merge edges from the milestone;
* caches the milestone dense geometry for the selected crop once;
* repeatedly trains only the RAG on that identical graph;
* Phase 1: train morphology CNNs + zero-init morphology residual projections;
* Phase 2: unfreeze the complete existing RAG and fine-tune it jointly;
* gives baseline hard-negative edges extra loss weight;
* shows a live green tqdm progress bar;
* writes history.jsonl, crop_selection.json, summary.json, and recoverable
  checkpoints.

The fixed crop is intentionally an overfit experiment. Success means:
    positive GT RAG edges -> p_merge toward 1
    negative GT RAG edges -> p_merge toward 0
especially the baseline high-confidence false-merge edges.

Recommended first local run
---------------------------
From repository root:

    python investigations/stirnet/data/15_morphology_rag_overfit.py ^
        --execution local ^
        --max-steps 300 ^
        --phase1-steps 100 ^
        --sample Drosophila_1

The effective Drosophila spacing defaults to the spacing used by the milestone:
    XYZ = 0.20312639, 0.20312639, 0.79099447 um

Useful smoke test:

    python investigations/stirnet/data/15_morphology_rag_overfit.py ^
        --execution local ^
        --max-steps 3 ^
        --phase1-steps 2 ^
        --candidate-limit 3 ^
        --run-name morph_rag_smoke

Modal remains available:

    python investigations/stirnet/data/15_morphology_rag_overfit.py ^
        --execution modal ^
        --max-steps 300 ^
        --phase1-steps 100 ^
        --sample Drosophila_1

Notes
-----
* No XY augmentation is applied: this is a fixed-crop proof-of-learning test.
* No source-instance dropout is applied.
* Geometry is computed once with torch.no_grad() and reused every optimizer step.
* The new morphology residual projections start at zero, so the initial
  deterministic merge logits are the transferred legacy RAG logits.
"""

import argparse
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

APP_NAME = "stirnet-morphology-rag-overfit"

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
# Small utilities
# ======================================================================================

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
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_as_jsonable(item) for item in value]
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
            f"{name} must contain exactly 3 values; got {value!r}"
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
    if any(not math.isfinite(item) or item <= 0 for item in xyz):
        raise ValueError("--spacing-xyz values must be finite and positive")
    return (xyz[2], xyz[1], xyz[0])


def _parse_shape_zyx(value: str) -> tuple[int, int, int]:
    z, y, x = _parse_triplet_text(
        value,
        name="--crop-shape-zyx",
        cast=int,
    )
    shape = (int(z), int(y), int(x))
    if any(item < 8 for item in shape):
        raise ValueError("crop dimensions must each be >= 8")
    return shape


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast_context(amp_dtype: str):
    import torch
    from contextlib import nullcontext

    if not torch.cuda.is_available() or amp_dtype == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


# ======================================================================================
# NIS3D loading — kept aligned with Training 01
# ======================================================================================

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


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
    if raw.shape != gt.shape or raw.shape != confidence.shape or raw.ndim != 3:
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
    report = {
        "gt_ids_original": int(raw_gt_ids.size),
        "gt_ids_removed_touching_confidence1": int(touched_ids.size),
        "gt_ids_kept": int(kept_ids.size),
        "confidence1_fraction": float((confidence_np == 1).mean()),
        "supervision_valid_fraction": float(valid.mean()),
        "confidence_values": [
            int(value) for value in np.unique(confidence_np).tolist()
        ],
    }
    return clean_gt, valid.astype(bool, copy=False), report


def _discover_nis3d_root(
    sample: str,
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
        if not (root / sample).is_dir():
            raise FileNotFoundError(
                f"{root} does not contain sample directory {sample}"
            )
        return root

    candidates = (
        LOCAL_NIS3D_ROOT_CANDIDATES
        if execution_mode == "local"
        else tuple(Path(value) for value in NIS3D_ROOT_CANDIDATES)
    )
    for root in candidates:
        if (root / sample).is_dir():
            return root

    search_root = (
        LOCAL_REPO_ROOT / "data"
        if execution_mode == "local"
        else Path(DATA_MOUNT)
    )
    if search_root.exists():
        for match in list(search_root.glob(f"**/{sample}"))[:64]:
            if match.is_dir():
                return match.parent

    raise FileNotFoundError(
        f"Could not locate NIS3D sample {sample}. "
        f"Use --data-dir to specify the dataset root."
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
        source_id=f"NIS3D/{sample}@morph-overfit/{cache_namespace}",
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
# Milestone checkpoint / model hydration
# ======================================================================================

def _hydrate_dataclass(instance: Any, payload: dict[str, Any]) -> Any:
    """Hydrate only fields present in the historical checkpoint config."""
    if not is_dataclass(instance):
        raise TypeError("Expected a dataclass instance")

    field_names = {field.name for field in fields(instance)}
    for key, value in payload.items():
        if key not in field_names:
            # Historical/current optional fields are allowed to differ.
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _hydrate_dataclass(current, value)
        elif isinstance(current, tuple) and isinstance(value, (list, tuple)):
            setattr(instance, key, tuple(value))
        else:
            setattr(instance, key, value)
    return instance


def _resolve_checkpoint(
    *,
    checkpoint_arg: str,
    execution_mode: str,
) -> Path:
    if checkpoint_arg:
        path = Path(checkpoint_arg).expanduser()
        if not path.is_absolute():
            path = (
                LOCAL_REPO_ROOT / path
                if execution_mode == "local"
                else Path(REMOTE_REPO_ROOT) / path
            )
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    root = (
        LOCAL_REPO_ROOT / "runs"
        if execution_mode == "local"
        else Path(RUNS_MOUNT)
    )
    candidates = (
        root
        / "stirnet"
        / "milestones"
        / "drosophila_12_spatial_v1"
        / "checkpoint_step_001000.pt",
        root
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
        root.glob(
            "**/drosophila_12_spatial_v1/**/checkpoint_step_001000.pt"
        )
    )
    if matches:
        return matches[-1]

    raise FileNotFoundError(
        "Could not auto-resolve the Drosophila step-1000 milestone. "
        "Pass --checkpoint explicitly."
    )


def _build_morphology_model(checkpoint_path: Path, *, device: str):
    import torch
    from learned.stirnet import StirNet, StirNetConfig

    payload = _torch_load(checkpoint_path, map_location="cpu")
    model_cfg = StirNetConfig()
    checkpoint_cfg = payload.get("model_config")
    if isinstance(checkpoint_cfg, dict):
        _hydrate_dataclass(model_cfg, checkpoint_cfg)

    # New architecture switch. All old input dimensions remain unchanged.
    model_cfg.partition.rag_morphology_enabled = True
    model_cfg.partition.rag_morphology_detach_geometry = True
    model_cfg.validate()

    model = StirNet(model_cfg)

    old_state = payload.get("model", {})
    current_state = model.state_dict()
    compatible = {
        name: value
        for name, value in old_state.items()
        if name in current_state
        and tuple(current_state[name].shape) == tuple(value.shape)
    }
    model.load_state_dict(compatible, strict=False)

    transferred = len(compatible)
    historical = len(old_state)
    new_names = sorted(set(current_state) - set(compatible))

    # The old model must transfer almost completely. We deliberately allow only
    # the new morphology encoder/projection parameters to be absent.
    unexpected_missing = [
        name
        for name in new_names
        if (
            "morphology_builder" not in name
            and "morphology_projection" not in name
        )
    ]
    if unexpected_missing:
        preview = "\n".join(f"  - {name}" for name in unexpected_missing[:20])
        raise RuntimeError(
            "Milestone transfer left unexpected current-model parameters "
            f"uninitialized:\n{preview}"
        )

    model = model.to(device)
    return model, model_cfg, payload, {
        "transferred_parameter_tensors": transferred,
        "historical_parameter_tensors": historical,
        "new_parameter_tensors": len(new_names),
        "new_parameter_names": new_names,
    }


# ======================================================================================
# Fixed crop construction / selection
# ======================================================================================

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


def _selection_payload(record, index: int) -> dict[str, Any]:
    return {
        "manifest_index": int(index),
        "candidate_type": str(record.candidate_type),
        "slices_zyx": [
            [int(axis.start), int(axis.stop)]
            for axis in record.slices_zyx
        ],
        "complete_cell_ids": [int(v) for v in record.complete_cell_ids],
        "partial_cell_ids": [int(v) for v in record.partial_cell_ids],
        "true_boundary_cell_ids": [
            int(v) for v in record.true_boundary_cell_ids
        ],
        "merge_source_ids": [int(v) for v in record.merge_source_ids],
        "merge_gt_ids": [int(v) for v in record.merge_gt_ids],
    }


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

    # Training 01 intersects its crop-level partial-cell mask with the
    # dataset-level NIS3D ConfidenceScore validity mask. Do the same here.
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
    ):
        if key in result and result[key] is not None:
            result[key] = torch.as_tensor(result[key]).to(
                "cuda",
                non_blocking=True,
            )
    return result


def _build_manifest(
    source_batch: dict,
    crop_shape_zyx: tuple[int, int, int],
):
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


def _rank_manifest_records(records):
    """Prioritize merge/crowded crops before expensive model scoring."""
    indexed = list(enumerate(records))
    return sorted(
        indexed,
        key=lambda item: (
            -int(bool(item[1].merge_source_ids)),
            -len(item[1].merge_gt_ids),
            -len(item[1].complete_cell_ids),
            len(item[1].partial_cell_ids),
            item[0],
        ),
    )


def _rag_targets_for_output(output, crop_batch, rag_criterion):
    return rag_criterion.build_targets(
        output.rag,
        crop_batch["gt_labels"],
        valid_mask=crop_batch.get("supervision_valid_mask"),
    )


def _edge_summary(
    output,
    targets,
    *,
    hard_negative_probability: float,
    separator_max_threshold: float,
) -> dict[str, Any]:
    import torch

    valid = targets.valid
    target = targets.target
    probabilities = output.rag.spatial_edge_logits.detach().sigmoid()

    positive = valid & (target > 0.5)
    negative = valid & (target < 0.5)
    hard_negative = (
        negative & (probabilities >= hard_negative_probability)
    )

    separator_max = (
        output.rag.edge_features[:, 1].detach()
        if output.rag.edge_features.numel()
        else probabilities.new_zeros(probabilities.shape)
    )
    separator_hard = (
        hard_negative & (separator_max >= separator_max_threshold)
    )

    def mean_or_zero(mask, values):
        if bool(mask.any()):
            return float(values[mask].float().mean().cpu())
        return 0.0

    return {
        "edge_count": int(output.rag.edge_index.shape[1]),
        "valid_edge_count": int(valid.sum().item()),
        "positive_edge_count": int(positive.sum().item()),
        "negative_edge_count": int(negative.sum().item()),
        "hard_negative_edge_count": int(hard_negative.sum().item()),
        "separator_hard_negative_edge_count": int(
            separator_hard.sum().item()
        ),
        "mean_positive_probability": mean_or_zero(
            positive,
            probabilities,
        ),
        "mean_negative_probability": mean_or_zero(
            negative,
            probabilities,
        ),
        "mean_hard_negative_probability": mean_or_zero(
            hard_negative,
            probabilities,
        ),
        "mean_separator_hard_probability": mean_or_zero(
            separator_hard,
            probabilities,
        ),
        "has_both_classes": bool(positive.any() and negative.any()),
        "_hard_negative_mask": hard_negative,
        "_separator_hard_mask": separator_hard,
    }


def _candidate_score(summary: dict[str, Any]) -> tuple:
    # Separator-backed false merges are the exact failure phenotype we want.
    return (
        int(summary["has_both_classes"]),
        int(summary["separator_hard_negative_edge_count"]),
        int(summary["hard_negative_edge_count"]),
        int(summary["negative_edge_count"]),
        int(summary["valid_edge_count"]),
        int(summary["positive_edge_count"]),
    )


def _evaluate_candidate(
    *,
    model,
    crop_batch,
    rag_criterion,
    amp_dtype: str,
    hard_negative_probability: float,
    separator_max_threshold: float,
):
    import torch

    model.eval()
    with torch.no_grad(), _autocast_context(amp_dtype):
        geometry = model(
            crop_batch["spatial_inputs"],
            crop_batch["spacing_um"],
            crop_batch["dref_um"],
            spatial_padding_mask=crop_batch.get("spatial_padding_mask"),
            execution_stage="geometry",
        )
        output = model(
            crop_batch["spatial_inputs"],
            crop_batch["spacing_um"],
            crop_batch["dref_um"],
            spatial_padding_mask=crop_batch.get("spatial_padding_mask"),
            execution_stage="spatial",
            precomputed_geometry=geometry,
        )
        targets = _rag_targets_for_output(
            output,
            crop_batch,
            rag_criterion,
        )
    summary = _edge_summary(
        output,
        targets,
        hard_negative_probability=hard_negative_probability,
        separator_max_threshold=separator_max_threshold,
    )
    return geometry, output, targets, summary


def _select_fixed_crop(
    *,
    model,
    source_batch,
    crop_shape_zyx,
    rag_criterion,
    amp_dtype: str,
    candidate_limit: int,
    crop_index: int,
    partial_ignore_margin_um: float,
    hard_negative_probability: float,
    separator_max_threshold: float,
):
    import torch

    manifest = _build_manifest(source_batch, crop_shape_zyx)
    records = list(manifest.records[0])
    if not records:
        raise RuntimeError("Crop manifest contains no records")

    full_shape = tuple(int(v) for v in source_batch["gt_labels"].shape[-3:])
    spacing = source_batch["spacing_um"][0]

    if crop_index >= 0:
        if crop_index >= len(records):
            raise IndexError(
                f"--crop-index {crop_index} >= manifest size {len(records)}"
            )
        candidates = [(crop_index, records[crop_index])]
    else:
        ranked = _rank_manifest_records(records)
        candidates = ranked[: min(candidate_limit, len(ranked))]

    best = None
    for ordinal, (manifest_index, record) in enumerate(candidates, 1):
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

        geometry, output, targets, summary = _evaluate_candidate(
            model=model,
            crop_batch=crop,
            rag_criterion=rag_criterion,
            amp_dtype=amp_dtype,
            hard_negative_probability=hard_negative_probability,
            separator_max_threshold=separator_max_threshold,
        )
        score = _candidate_score(summary)
        tqdm.write(
            "[select] "
            f"{ordinal}/{len(candidates)} manifest={manifest_index} "
            f"type={record.candidate_type} "
            f"valid={summary['valid_edge_count']} "
            f"pos={summary['positive_edge_count']} "
            f"neg={summary['negative_edge_count']} "
            f"hard={summary['hard_negative_edge_count']} "
            f"sep-hard={summary['separator_hard_negative_edge_count']}"
        )

        row = {
            "score": score,
            "manifest_index": manifest_index,
            "record": record,
            "spec": spec,
            "crop_cpu": crop_cpu,
            "crop": crop,
            "geometry": geometry,
            "output": output,
            "targets": targets,
            "summary": summary,
        }
        if best is None or score > best["score"]:
            best = row

    if best is None:
        raise RuntimeError("No fixed crop could be selected")

    # Drop GPU candidate objects other than the selected one.
    torch.cuda.empty_cache()
    return best, manifest


# ======================================================================================
# Trainability / loss / metrics
# ======================================================================================

def _set_phase_trainability(model, phase: str) -> None:
    """Freeze everything except the intended RAG subset."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    morphology_builder = model.rag_builder.morphology_builder
    if morphology_builder is None:
        raise RuntimeError(
            "rag_morphology_enabled=True but morphology_builder is absent"
        )

    if phase == "morphology_only":
        for parameter in morphology_builder.parameters():
            parameter.requires_grad_(True)
        for module in (
            model.rag_network.node_morphology_projection,
            model.rag_network.edge_morphology_projection,
        ):
            if module is None:
                raise RuntimeError("Morphology projection is missing")
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        return

    if phase == "joint_rag":
        for parameter in model.rag_builder.parameters():
            parameter.requires_grad_(True)
        for parameter in model.rag_network.parameters():
            parameter.requires_grad_(True)
        return

    raise ValueError(f"Unknown training phase: {phase}")


def _phase_for_step(step: int, phase1_steps: int) -> str:
    return "morphology_only" if step < phase1_steps else "joint_rag"


def _lr_for_phase(
    phase: str,
    *,
    phase1_lr: float,
    phase2_lr: float,
) -> float:
    return phase1_lr if phase == "morphology_only" else phase2_lr


def _rag_optimizer_parameters(model):
    parameters = []
    seen: set[int] = set()
    for module in (model.rag_builder, model.rag_network):
        for parameter in module.parameters():
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                parameters.append(parameter)
    return parameters


def _set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _parameter_count(parameters) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _gradient_norm(parameters) -> float:
    import torch

    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = float(
            torch.linalg.vector_norm(
                parameter.grad.detach().float()
            ).cpu()
        )
        total = math.hypot(total, value)
    return total


def _gradient_groups(model):
    morphology = list(
        model.rag_builder.morphology_builder.parameters()
    )
    projections = []
    for module in (
        model.rag_network.node_morphology_projection,
        model.rag_network.edge_morphology_projection,
    ):
        projections.extend(list(module.parameters()))

    morphology_ids = {id(parameter) for parameter in morphology}
    projection_ids = {id(parameter) for parameter in projections}
    legacy = [
        parameter
        for parameter in _rag_optimizer_parameters(model)
        if id(parameter) not in morphology_ids
        and id(parameter) not in projection_ids
    ]
    return morphology, projections, legacy


def _weighted_rag_loss(
    logits,
    targets,
    *,
    hard_negative_mask,
    separator_hard_mask,
    hard_negative_weight: float,
    separator_hard_weight: float,
):
    import torch
    import torch.nn.functional as F

    valid = targets.valid
    if not bool(valid.any()):
        raise RuntimeError("Selected fixed crop has no valid RAG edges")

    target = targets.target
    selected_target = target[valid]
    selected_logits = logits[valid]

    positives = selected_target.sum()
    negatives = selected_target.numel() - positives
    pos_weight = (
        negatives / positives.clamp_min(1)
    ).clamp(0.5, 20.0)

    per_edge = F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_target,
        pos_weight=pos_weight,
        reduction="none",
    )
    weights = torch.ones_like(per_edge)

    hard_selected = hard_negative_mask[valid]
    separator_selected = separator_hard_mask[valid]
    weights = torch.where(
        hard_selected,
        weights * float(hard_negative_weight),
        weights,
    )
    weights = torch.where(
        separator_selected,
        weights * float(separator_hard_weight),
        weights,
    )
    loss = (per_edge * weights).sum() / weights.sum().clamp_min(1.0)
    return loss, pos_weight.detach()


def _probability_metrics(
    logits,
    targets,
    hard_negative_mask,
    separator_hard_mask,
) -> dict[str, float]:
    import torch

    probabilities = logits.detach().sigmoid()
    valid = targets.valid
    target = targets.target.bool()
    positive = valid & target
    negative = valid & ~target

    def accuracy(mask, expected: bool) -> float:
        if not bool(mask.any()):
            return 0.0
        predicted = probabilities[mask] >= 0.5
        desired = torch.full_like(predicted, expected)
        return float((predicted == desired).float().mean().cpu())

    def mean(mask) -> float:
        if not bool(mask.any()):
            return 0.0
        return float(probabilities[mask].float().mean().cpu())

    overall = (
        float(
            (
                (probabilities[valid] >= 0.5)
                == target[valid]
            )
            .float()
            .mean()
            .cpu()
        )
        if bool(valid.any())
        else 0.0
    )
    return {
        "rag_accuracy": overall,
        "positive_accuracy": accuracy(positive, True),
        "negative_accuracy": accuracy(negative, False),
        "mean_positive_probability": mean(positive),
        "mean_negative_probability": mean(negative),
        "mean_hard_negative_probability": mean(hard_negative_mask),
        "mean_separator_hard_probability": mean(separator_hard_mask),
    }


def _unused_staticmethod_marker():
    return None


# ======================================================================================
# Checkpoint / recovery
# ======================================================================================

def _checkpoint_paths(recovery_dir: Path) -> list[Path]:
    return sorted(recovery_dir.glob("checkpoint_step_*.pt"))


def _latest_checkpoint(recovery_dir: Path) -> Path | None:
    rows = _checkpoint_paths(recovery_dir)
    return rows[-1] if rows else None


def _save_checkpoint(
    *,
    recovery_dir: Path,
    model,
    model_cfg,
    optimizer,
    scaler,
    global_step: int,
    phase: str,
    run_dir: Path,
    milestone_checkpoint: Path,
    sample: str,
    crop_selection: dict[str, Any],
    execution_mode: str,
) -> Path:
    from learned.stirnet.training.checkpoint import save_checkpoint

    path = recovery_dir / f"checkpoint_step_{global_step:06d}.pt"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_checkpoint(
            temporary,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=global_step,
            model_config=model_cfg,
            training_config={
                "experiment": "15_morphology_rag_overfit",
                "phase": phase,
            },
            extra={
                "experiment": "15_morphology_rag_overfit",
                "phase": phase,
                "run_dir": str(run_dir),
                "milestone_checkpoint": str(milestone_checkpoint),
                "sample": sample,
                "crop_selection": crop_selection,
            },
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    _atomic_text(
        recovery_dir / "latest_checkpoint.txt",
        path.name + "\n",
    )
    _atomic_json(
        recovery_dir / "latest_state.json",
        {
            "checkpoint": str(path),
            "global_step": int(global_step),
            "phase": phase,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
        },
    )
    if execution_mode == "modal":
        runs_volume.commit()
    tqdm.write(
        f"[checkpoint] persisted step={global_step} phase={phase} path={path}"
    )
    return path


def _restore_overfit_checkpoint(
    *,
    path: Path,
    model,
    optimizer,
    scaler,
):
    payload = _torch_load(path, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return payload


# ======================================================================================
# Main overfit implementation
# ======================================================================================

def _print_header(
    *,
    execution_mode: str,
    checkpoint: Path,
    sample: str,
    nis3d_root: Path,
    spacing_override_zyx_um,
    crop_shape_zyx,
    max_steps,
    phase1_steps,
    phase1_lr,
    phase2_lr,
    candidate_limit,
    hard_negative_probability,
    hard_negative_weight,
    separator_max_threshold,
    separator_hard_weight,
    amp_dtype,
    results_root,
):
    import torch

    props = torch.cuda.get_device_properties(0)
    print("=" * 118, flush=True)
    print(
        "STIR-Net Investigation 15 — morphology-aware RAG fixed-crop overfit",
        flush=True,
    )
    print("=" * 118, flush=True)
    print(f"GPU                      : {props.name}", flush=True)
    print(
        f"GPU VRAM                 : {props.total_memory / 2**30:.2f} GiB",
        flush=True,
    )
    print(f"Execution                : {execution_mode}", flush=True)
    if execution_mode == "modal":
        print(
            f"CPU / RAM request        : {CPU:g} CPU / "
            f"{MEMORY_MB / 1024:.1f} GiB",
            flush=True,
        )
    print(f"Results root             : {results_root}", flush=True)
    print(f"Milestone checkpoint     : {checkpoint}", flush=True)
    print(f"NIS3D root               : {nis3d_root}", flush=True)
    print(f"Sample                    : {sample}", flush=True)
    if spacing_override_zyx_um is None:
        print("Effective spacing         : Info.txt", flush=True)
    else:
        z, y, x = spacing_override_zyx_um
        print(
            "Effective spacing XYZ um  : "
            f"({x:.8g}, {y:.8g}, {z:.8g})",
            flush=True,
        )
    print(f"Fixed crop shape ZYX      : {crop_shape_zyx}", flush=True)
    print(f"Candidate crops scored    : <= {candidate_limit}", flush=True)
    print(f"Optimizer steps           : {max_steps}", flush=True)
    print(
        f"Phase 1                   : [0,{phase1_steps}) "
        f"morphology-only lr={phase1_lr:g}",
        flush=True,
    )
    print(
        f"Phase 2                   : [{phase1_steps},{max_steps}) "
        f"joint-RAG lr={phase2_lr:g}",
        flush=True,
    )
    print(
        f"Hard negative             : baseline p>={hard_negative_probability:g} "
        f"weight={hard_negative_weight:g}",
        flush=True,
    )
    print(
        f"Separator hard negative   : sep_max>={separator_max_threshold:g} "
        f"extra_weight={separator_hard_weight:g}",
        flush=True,
    )
    print(f"AMP                       : {amp_dtype}", flush=True)
    print(
        "Dense geometry            : FROZEN and cached once for selected crop",
        flush=True,
    )
    print(
        "Source dropout / XY flip  : OFF / OFF (fixed-crop overfit)",
        flush=True,
    )
    print("=" * 118, flush=True)


def _train_impl(
    *,
    max_steps: int = 300,
    phase1_steps: int = 100,
    phase1_lr: float = 2e-4,
    phase2_lr: float = 5e-5,
    checkpoint_every: int = 50,
    eval_every: int = 10,
    sample: str = "Drosophila_1",
    checkpoint: str = "",
    data_dir: str = "",
    spacing_xyz: str = "0.20312639,0.20312639,0.79099447",
    crop_shape_zyx: str = "32,192,192",
    candidate_limit: int = 12,
    crop_index: int = -1,
    confidence_ignore_margin_um: float = 1.0,
    partial_ignore_margin_um: float = 1.0,
    hard_negative_probability: float = 0.50,
    hard_negative_weight: float = 3.0,
    separator_max_threshold: float = 0.50,
    separator_hard_weight: float = 1.5,
    run_name: str = "drosophila_1_morphology_rag_overfit",
    resume: bool = False,
    seed: int = 230525,
    execution_mode: str = "local",
) -> dict[str, Any]:
    import torch
    from learned.stirnet.model.partition.rag import RAGCriterion

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Investigation 15")
    if execution_mode not in {"local", "modal"}:
        raise ValueError("execution_mode must be local or modal")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not 0 <= phase1_steps <= max_steps:
        raise ValueError("phase1_steps must be in [0,max_steps]")
    if phase1_lr <= 0 or phase2_lr <= 0:
        raise ValueError("learning rates must be positive")
    if checkpoint_every < 1 or eval_every < 1:
        raise ValueError("checkpoint/eval intervals must be positive")
    if candidate_limit < 1:
        raise ValueError("candidate_limit must be positive")
    if hard_negative_weight < 1 or separator_hard_weight < 1:
        raise ValueError("hard-negative weights must be >= 1")
    if not 0 <= hard_negative_probability <= 1:
        raise ValueError("hard_negative_probability must be in [0,1]")
    if not 0 <= separator_max_threshold <= 1:
        raise ValueError("separator_max_threshold must be in [0,1]")

    amp_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    _seed_everything(seed)

    spacing_override_zyx_um = _parse_spacing_xyz_override(
        spacing_xyz
    )
    crop_shape = _parse_shape_zyx(crop_shape_zyx)

    milestone = _resolve_checkpoint(
        checkpoint_arg=checkpoint,
        execution_mode=execution_mode,
    )
    nis3d_root = _discover_nis3d_root(
        sample,
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
        / "15_morphology_rag_overfit"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = experiment_root / "attempts" / f"{timestamp}_{run_name}"
    recovery_dir = experiment_root / "recovery" / run_name
    cache_root = experiment_root / "cache"
    run_dir.mkdir(parents=True, exist_ok=True)
    recovery_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    _print_header(
        execution_mode=execution_mode,
        checkpoint=milestone,
        sample=sample,
        nis3d_root=nis3d_root,
        spacing_override_zyx_um=spacing_override_zyx_um,
        crop_shape_zyx=crop_shape,
        max_steps=max_steps,
        phase1_steps=phase1_steps,
        phase1_lr=phase1_lr,
        phase2_lr=phase2_lr,
        candidate_limit=candidate_limit,
        hard_negative_probability=hard_negative_probability,
        hard_negative_weight=hard_negative_weight,
        separator_max_threshold=separator_max_threshold,
        separator_hard_weight=separator_hard_weight,
        amp_dtype=amp_dtype,
        results_root=experiment_root,
    )

    data_signature = _data_signature(
        nis3d_root=nis3d_root,
        sample=sample,
        spacing_override_zyx_um=spacing_override_zyx_um,
        confidence_ignore_margin_um=confidence_ignore_margin_um,
    )

    print(f"[data] preparing {sample} ...", flush=True)
    source_batch, sample_report = _prepare_sample_batch(
        nis3d_root=nis3d_root,
        sample=sample,
        spacing_override_zyx_um=spacing_override_zyx_um,
        confidence_ignore_margin_um=confidence_ignore_margin_um,
        cache_root=cache_root,
        cache_namespace=data_signature,
    )
    print(
        f"[data] shape={tuple(sample_report['shape_zyx'])} "
        f"spacing={tuple(round(v, 6) for v in sample_report['spacing_zyx_um'])} "
        f"GT={sample_report['gt_ids_kept']} "
        f"source={sample_report['source_instance_count']} "
        f"dref={sample_report['model_dref_um']:.4f}um "
        f"valid={sample_report['supervision_valid_fraction']:.4f} "
        f"ram_cache={sample_report['source_ram_cache_gross_gib']:.2f}GiB",
        flush=True,
    )
    _atomic_json(run_dir / "sample.json", sample_report)

    print("[init] constructing morphology-enabled STIR-Net ...", flush=True)
    model, model_cfg, milestone_payload, transfer_report = (
        _build_morphology_model(milestone, device="cuda")
    )
    torch.cuda.synchronize()
    print(
        f"[init] transferred {transfer_report['transferred_parameter_tensors']}/"
        f"{transfer_report['historical_parameter_tensors']} historical tensors; "
        f"new tensors={transfer_report['new_parameter_tensors']}",
        flush=True,
    )

    rag_criterion = RAGCriterion(model_cfg.partition).to("cuda")

    selection_file = recovery_dir / "crop_selection.json"
    if resume and selection_file.is_file():
        # Reuse the exact manifest row if possible.
        saved = json.loads(selection_file.read_text(encoding="utf-8"))
        saved_index = int(saved["manifest_index"])
        crop_index_effective = saved_index
        tqdm.write(
            f"[select] resume: reusing manifest crop index {saved_index}"
        )
    else:
        crop_index_effective = crop_index

    print("[select] building/ranking fixed-crop candidates ...", flush=True)
    selected, manifest = _select_fixed_crop(
        model=model,
        source_batch=source_batch,
        crop_shape_zyx=crop_shape,
        rag_criterion=rag_criterion,
        amp_dtype=amp_dtype,
        candidate_limit=candidate_limit,
        crop_index=crop_index_effective,
        partial_ignore_margin_um=partial_ignore_margin_um,
        hard_negative_probability=hard_negative_probability,
        separator_max_threshold=separator_max_threshold,
    )

    crop_selection = {
        **_selection_payload(
            selected["record"],
            selected["manifest_index"],
        ),
        "manifest_record_count": len(manifest.records[0]),
        "baseline": {
            key: value
            for key, value in selected["summary"].items()
            if not key.startswith("_")
        },
        "selection_score": list(selected["score"]),
    }
    _atomic_json(run_dir / "crop_selection.json", crop_selection)
    _atomic_json(selection_file, crop_selection)
    if execution_mode == "modal":
        runs_volume.commit()

    baseline_summary = crop_selection["baseline"]
    print(
        "[select] chosen "
        f"manifest={crop_selection['manifest_index']} "
        f"type={crop_selection['candidate_type']} "
        f"slices={crop_selection['slices_zyx']} "
        f"valid={baseline_summary['valid_edge_count']} "
        f"pos={baseline_summary['positive_edge_count']} "
        f"neg={baseline_summary['negative_edge_count']} "
        f"hard={baseline_summary['hard_negative_edge_count']} "
        f"sep-hard={baseline_summary['separator_hard_negative_edge_count']}",
        flush=True,
    )
    if baseline_summary["valid_edge_count"] == 0:
        raise RuntimeError("Selected crop has no valid RAG supervision")
    if not baseline_summary["has_both_classes"]:
        tqdm.write(
            "[warning] selected crop does not contain both positive and "
            "negative valid RAG edges. The run can proceed, but a different "
            "--crop-index/candidate set is preferable for the proof-of-learning."
        )

    crop = selected["crop"]
    frozen_geometry = selected["geometry"]
    baseline_output = selected["output"]
    fixed_targets = selected["targets"]

    baseline_edge_index = baseline_output.rag.edge_index.detach().clone()
    baseline_hard_negative_mask = selected["summary"][
        "_hard_negative_mask"
    ].detach().clone()
    baseline_separator_hard_mask = selected["summary"][
        "_separator_hard_mask"
    ].detach().clone()

    # Release baseline output while retaining frozen dense geometry and fixed
    # edge-level supervision/masks.
    del baseline_output
    torch.cuda.empty_cache()

    all_rag_params = _rag_optimizer_parameters(model)
    optimizer = torch.optim.AdamW(
        all_rag_params,
        lr=phase1_lr,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(amp_dtype == "fp16"),
    )

    global_step = 0
    resumed_from = None

    if resume:
        checkpoint_path = _latest_checkpoint(recovery_dir)
        if checkpoint_path is not None:
            raw_resume = _torch_load(checkpoint_path, map_location="cpu")
            global_step = int(raw_resume.get("global_step", 0))
            phase = _phase_for_step(global_step, phase1_steps)
            _set_phase_trainability(model, phase)
            _set_optimizer_lr(
                optimizer,
                _lr_for_phase(
                    phase,
                    phase1_lr=phase1_lr,
                    phase2_lr=phase2_lr,
                ),
            )
            payload = _restore_overfit_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
            )
            global_step = int(payload.get("global_step", global_step))
            resumed_from = str(checkpoint_path)
            print(
                f"[resume] loaded {checkpoint_path} "
                f"global_step={global_step} phase={phase}",
                flush=True,
            )
        else:
            print(
                "[resume] no overfit checkpoint found; starting from milestone",
                flush=True,
            )

    phase = _phase_for_step(global_step, phase1_steps)
    _set_phase_trainability(model, phase)
    current_lr = _lr_for_phase(
        phase,
        phase1_lr=phase1_lr,
        phase2_lr=phase2_lr,
    )
    _set_optimizer_lr(optimizer, current_lr)

    morphology_params, projection_params, legacy_rag_params = (
        _gradient_groups(model)
    )
    print(
        f"[trainable] phase={phase} "
        f"morphology={_parameter_count(morphology_params)/1e6:.3f}M "
        f"projections={_parameter_count(projection_params)/1e6:.3f}M "
        f"legacy_rag={_parameter_count(legacy_rag_params)/1e6:.3f}M "
        f"currently_trainable="
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}M",
        flush=True,
    )

    config_payload = {
        "experiment": "15_morphology_rag_overfit",
        "milestone_checkpoint": str(milestone),
        "milestone_global_step": int(
            milestone_payload.get("global_step", -1)
        ),
        "transfer_report": transfer_report,
        "model": model_cfg.to_dict(),
        "sample": sample,
        "data_signature": data_signature,
        "crop_shape_zyx": list(crop_shape),
        "max_steps": int(max_steps),
        "phase1_steps": int(phase1_steps),
        "phase1_lr": float(phase1_lr),
        "phase2_lr": float(phase2_lr),
        "checkpoint_every": int(checkpoint_every),
        "eval_every": int(eval_every),
        "candidate_limit": int(candidate_limit),
        "hard_negative_probability": float(hard_negative_probability),
        "hard_negative_weight": float(hard_negative_weight),
        "separator_max_threshold": float(separator_max_threshold),
        "separator_hard_weight": float(separator_hard_weight),
        "amp_dtype": amp_dtype,
        "seed": int(seed),
        "dense_geometry_frozen": True,
        "source_dropout_probability": 0.0,
        "xy_flip_probability": 0.0,
    }
    _atomic_json(run_dir / "config.json", config_payload)

    if global_step >= max_steps:
        return {
            "status": "already_complete",
            "global_step": global_step,
            "run_dir": str(run_dir),
            "recovery_dir": str(recovery_dir),
            "resumed_from": resumed_from,
        }

    history_path = run_dir / "history.jsonl"
    started_all = time.perf_counter()
    step_time_sum = 0.0
    successful_steps = 0
    last_eval = None
    last_checkpoint_step = None

    GREEN = "\033[32m"
    RESET = "\033[0m"
    progress = tqdm(
        total=max_steps,
        initial=global_step,
        desc=f"{GREEN}Morph-RAG{RESET}",
        unit="step",
        dynamic_ncols=True,
        smoothing=0.10,
        mininterval=0.5,
        leave=True,
        colour="green",
        file=sys.stdout,
    )

    try:
        while global_step < max_steps:
            expected_phase = _phase_for_step(global_step, phase1_steps)
            if expected_phase != phase:
                phase = expected_phase
                _set_phase_trainability(model, phase)
                current_lr = _lr_for_phase(
                    phase,
                    phase1_lr=phase1_lr,
                    phase2_lr=phase2_lr,
                )
                _set_optimizer_lr(optimizer, current_lr)
                tqdm.write(
                    f"[phase] step={global_step}: -> {phase}, "
                    f"lr={current_lr:g}, trainable="
                    f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}M"
                )

            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()

            step_started = time.perf_counter()
            with _autocast_context(amp_dtype):
                output = model(
                    crop["spatial_inputs"],
                    crop["spacing_um"],
                    crop["dref_um"],
                    spatial_padding_mask=crop.get("spatial_padding_mask"),
                    execution_stage="spatial",
                    precomputed_geometry=frozen_geometry,
                )

                if (
                    output.rag.edge_index.shape
                    != baseline_edge_index.shape
                    or not torch.equal(
                        output.rag.edge_index,
                        baseline_edge_index,
                    )
                ):
                    raise RuntimeError(
                        "Frozen geometry produced a different RAG topology. "
                        "Fixed targets/hard-negative masks would no longer align."
                    )

                loss, pos_weight = _weighted_rag_loss(
                    output.rag.spatial_edge_logits,
                    fixed_targets,
                    hard_negative_mask=baseline_hard_negative_mask,
                    separator_hard_mask=baseline_separator_hard_mask,
                    hard_negative_weight=hard_negative_weight,
                    separator_hard_weight=separator_hard_weight,
                )

            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite morphology-RAG loss at step {global_step}"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            morphology_grad = _gradient_norm(morphology_params)
            projection_grad = _gradient_norm(projection_params)
            legacy_grad = _gradient_norm(legacy_rag_params)

            trainable_params = [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            total_grad = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=1.0,
            )
            if not bool(torch.isfinite(torch.as_tensor(total_grad))):
                raise FloatingPointError(
                    f"Non-finite gradient norm at step {global_step}"
                )

            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - step_started
            step_time_sum += step_seconds
            successful_steps += 1
            average_step_seconds = step_time_sum / successful_steps

            train_prob_metrics = _probability_metrics(
                output.rag.spatial_edge_logits,
                fixed_targets,
                baseline_hard_negative_mask,
                baseline_separator_hard_mask,
            )

            should_eval = (
                global_step == 1
                or global_step % eval_every == 0
                or global_step == phase1_steps
                or global_step == max_steps
            )
            if should_eval:
                model.eval()
                with torch.no_grad(), _autocast_context(amp_dtype):
                    evaluated = model(
                        crop["spatial_inputs"],
                        crop["spacing_um"],
                        crop["dref_um"],
                        spatial_padding_mask=crop.get(
                            "spatial_padding_mask"
                        ),
                        execution_stage="spatial",
                        precomputed_geometry=frozen_geometry,
                    )
                    eval_loss, _ = _weighted_rag_loss(
                        evaluated.rag.spatial_edge_logits,
                        fixed_targets,
                        hard_negative_mask=baseline_hard_negative_mask,
                        separator_hard_mask=baseline_separator_hard_mask,
                        hard_negative_weight=hard_negative_weight,
                        separator_hard_weight=separator_hard_weight,
                    )
                    last_eval = {
                        "eval_loss": float(eval_loss.detach().float().cpu()),
                        **_probability_metrics(
                            evaluated.rag.spatial_edge_logits,
                            fixed_targets,
                            baseline_hard_negative_mask,
                            baseline_separator_hard_mask,
                        ),
                    }
                del evaluated

            peak_allocated = (
                torch.cuda.max_memory_allocated() / 2**30
            )
            peak_reserved = torch.cuda.max_memory_reserved() / 2**30
            elapsed = time.perf_counter() - started_all
            remaining = max_steps - global_step
            eta = average_step_seconds * remaining

            record = {
                "step": int(global_step),
                "phase": phase,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "loss": float(loss.detach().float().cpu()),
                "pos_weight": float(pos_weight.float().cpu()),
                "lr": float(current_lr),
                "step_seconds": float(step_seconds),
                "average_step_seconds": float(average_step_seconds),
                "elapsed_seconds": float(elapsed),
                "estimated_remaining_seconds": float(eta),
                "cuda_peak_allocated_gib": float(peak_allocated),
                "cuda_peak_reserved_gib": float(peak_reserved),
                "grad_norm": float(
                    torch.as_tensor(total_grad).detach().float().cpu()
                ),
                "grad_morphology_encoder": float(morphology_grad),
                "grad_morphology_projection": float(projection_grad),
                "grad_legacy_rag": float(legacy_grad),
                **{
                    f"train_{key}": value
                    for key, value in train_prob_metrics.items()
                },
            }
            if last_eval is not None and should_eval:
                record.update(last_eval)
            _append_jsonl(history_path, record)

            display = last_eval if last_eval is not None else {
                "eval_loss": record["loss"],
                **train_prob_metrics,
            }
            progress.set_postfix(
                {
                    "loss": f"{display['eval_loss']:.4f}",
                    "acc": f"{display['rag_accuracy']:.3f}",
                    "neg": f"{display['negative_accuracy']:.3f}",
                    "hard_p": f"{display['mean_hard_negative_probability']:.3f}",
                    "phase": "morph" if phase == "morphology_only" else "joint",
                    "lr": f"{current_lr:.1e}",
                    "avg": f"{average_step_seconds:.2f}s",
                },
                refresh=False,
            )
            progress.update(1)

            del output, loss

            if (
                global_step % checkpoint_every == 0
                or global_step == max_steps
            ):
                _save_checkpoint(
                    recovery_dir=recovery_dir,
                    model=model,
                    model_cfg=model_cfg,
                    optimizer=optimizer,
                    scaler=scaler,
                    global_step=global_step,
                    phase=phase,
                    run_dir=run_dir,
                    milestone_checkpoint=milestone,
                    sample=sample,
                    crop_selection=crop_selection,
                    execution_mode=execution_mode,
                )
                last_checkpoint_step = global_step

        progress.close()

        # Final deterministic evaluation.
        model.eval()
        with torch.no_grad(), _autocast_context(amp_dtype):
            final_output = model(
                crop["spatial_inputs"],
                crop["spacing_um"],
                crop["dref_um"],
                spatial_padding_mask=crop.get("spatial_padding_mask"),
                execution_stage="spatial",
                precomputed_geometry=frozen_geometry,
            )
            final_loss, _ = _weighted_rag_loss(
                final_output.rag.spatial_edge_logits,
                fixed_targets,
                hard_negative_mask=baseline_hard_negative_mask,
                separator_hard_mask=baseline_separator_hard_mask,
                hard_negative_weight=hard_negative_weight,
                separator_hard_weight=separator_hard_weight,
            )
            final_metrics = {
                "loss": float(final_loss.detach().float().cpu()),
                **_probability_metrics(
                    final_output.rag.spatial_edge_logits,
                    fixed_targets,
                    baseline_hard_negative_mask,
                    baseline_separator_hard_mask,
                ),
            }

        elapsed = time.perf_counter() - started_all
        summary = {
            "status": "success",
            "global_step": int(global_step),
            "max_steps": int(max_steps),
            "phase1_steps": int(phase1_steps),
            "sample": sample,
            "run_dir": str(run_dir),
            "recovery_dir": str(recovery_dir),
            "milestone_checkpoint": str(milestone),
            "resumed_from": resumed_from,
            "last_checkpoint_step": last_checkpoint_step,
            "elapsed_seconds": float(elapsed),
            "mean_step_seconds": float(
                step_time_sum / max(successful_steps, 1)
            ),
            "crop_selection": crop_selection,
            "baseline": baseline_summary,
            "final": final_metrics,
        }
        _atomic_json(run_dir / "summary.json", summary)
        if execution_mode == "modal":
            runs_volume.commit()

        print("", flush=True)
        print("=" * 118, flush=True)
        print("Investigation 15 complete", flush=True)
        print("=" * 118, flush=True)
        print(f"Elapsed                  : {_duration(elapsed)}", flush=True)
        print(
            f"Baseline hard-neg p       : "
            f"{baseline_summary['mean_hard_negative_probability']:.4f}",
            flush=True,
        )
        print(
            f"Final hard-neg p          : "
            f"{final_metrics['mean_hard_negative_probability']:.4f}",
            flush=True,
        )
        print(
            f"Baseline neg p            : "
            f"{baseline_summary['mean_negative_probability']:.4f}",
            flush=True,
        )
        print(
            f"Final neg p               : "
            f"{final_metrics['mean_negative_probability']:.4f}",
            flush=True,
        )
        print(
            f"Final accuracy            : {final_metrics['rag_accuracy']:.4f}",
            flush=True,
        )
        print(f"Run directory             : {run_dir}", flush=True)
        print("=" * 118, flush=True)
        return summary

    except BaseException as error:
        progress.close()
        failure = {
            "status": "failed",
            "global_step": int(global_step),
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
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
# Modal worker / launcher
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
def train_morphology_rag_overfit(
    max_steps: int = 300,
    phase1_steps: int = 100,
    phase1_lr: float = 2e-4,
    phase2_lr: float = 5e-5,
    checkpoint_every: int = 50,
    eval_every: int = 10,
    sample: str = "Drosophila_1",
    checkpoint: str = "",
    data_dir: str = "",
    spacing_xyz: str = "0.20312639,0.20312639,0.79099447",
    crop_shape_zyx: str = "32,192,192",
    candidate_limit: int = 12,
    crop_index: int = -1,
    confidence_ignore_margin_um: float = 1.0,
    partial_ignore_margin_um: float = 1.0,
    hard_negative_probability: float = 0.50,
    hard_negative_weight: float = 3.0,
    separator_max_threshold: float = 0.50,
    separator_hard_weight: float = 1.5,
    run_name: str = "drosophila_1_morphology_rag_overfit",
    resume: bool = False,
    seed: int = 230525,
):
    return _train_impl(
        max_steps=max_steps,
        phase1_steps=phase1_steps,
        phase1_lr=phase1_lr,
        phase2_lr=phase2_lr,
        checkpoint_every=checkpoint_every,
        eval_every=eval_every,
        sample=sample,
        checkpoint=checkpoint,
        data_dir=data_dir,
        spacing_xyz=spacing_xyz,
        crop_shape_zyx=crop_shape_zyx,
        candidate_limit=candidate_limit,
        crop_index=crop_index,
        confidence_ignore_margin_um=confidence_ignore_margin_um,
        partial_ignore_margin_um=partial_ignore_margin_um,
        hard_negative_probability=hard_negative_probability,
        hard_negative_weight=hard_negative_weight,
        separator_max_threshold=separator_max_threshold,
        separator_hard_weight=separator_hard_weight,
        run_name=run_name,
        resume=resume,
        seed=seed,
        execution_mode="modal",
    )


@_modal_local_entrypoint_decorator()
def main(
    max_steps: int = 300,
    phase1_steps: int = 100,
    phase1_lr: float = 2e-4,
    phase2_lr: float = 5e-5,
    checkpoint_every: int = 50,
    eval_every: int = 10,
    sample: str = "Drosophila_1",
    checkpoint: str = "",
    data_dir: str = "",
    spacing_xyz: str = "0.20312639,0.20312639,0.79099447",
    crop_shape_zyx: str = "32,192,192",
    candidate_limit: int = 12,
    crop_index: int = -1,
    confidence_ignore_margin_um: float = 1.0,
    partial_ignore_margin_um: float = 1.0,
    hard_negative_probability: float = 0.50,
    hard_negative_weight: float = 3.0,
    separator_max_threshold: float = 0.50,
    separator_hard_weight: float = 1.5,
    run_name: str = "drosophila_1_morphology_rag_overfit",
    resume: bool = False,
    seed: int = 230525,
) -> None:
    call = train_morphology_rag_overfit.spawn(
        max_steps=max_steps,
        phase1_steps=phase1_steps,
        phase1_lr=phase1_lr,
        phase2_lr=phase2_lr,
        checkpoint_every=checkpoint_every,
        eval_every=eval_every,
        sample=sample,
        checkpoint=checkpoint,
        data_dir=data_dir,
        spacing_xyz=spacing_xyz,
        crop_shape_zyx=crop_shape_zyx,
        candidate_limit=candidate_limit,
        crop_index=crop_index,
        confidence_ignore_margin_um=confidence_ignore_margin_um,
        partial_ignore_margin_um=partial_ignore_margin_um,
        hard_negative_probability=hard_negative_probability,
        hard_negative_weight=hard_negative_weight,
        separator_max_threshold=separator_max_threshold,
        separator_hard_weight=separator_hard_weight,
        run_name=run_name,
        resume=resume,
        seed=seed,
    )
    print(
        f"[launcher] spawned durable overfit call: {call.object_id}",
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
            "Morphology-aware RAG fixed-crop overfit. "
            "Default execution target is local."
        )
    )
    parser.add_argument(
        "--execution",
        choices=("local", "modal"),
        default="local",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--phase1-steps", type=int, default=100)
    parser.add_argument("--phase1-lr", type=float, default=2e-4)
    parser.add_argument("--phase2-lr", type=float, default=5e-5)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--sample", default="Drosophila_1")
    parser.add_argument(
        "--checkpoint",
        default="",
        help=(
            "Milestone checkpoint. Empty auto-resolves the Drosophila "
            "step-1000 milestone."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help=(
            "NIS3D root containing sample directories. Relative paths are "
            "resolved below <repo>/data locally and the Modal data mount remotely."
        ),
    )
    parser.add_argument(
        "--spacing-xyz",
        default="0.20312639,0.20312639,0.79099447",
        help="Effective spacing in X,Y,Z micrometres.",
    )
    parser.add_argument(
        "--crop-shape-zyx",
        default="32,192,192",
    )
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument(
        "--crop-index",
        type=int,
        default=-1,
        help=(
            "Manifest record to force. -1 auto-selects among candidate-limit "
            "ranked records."
        ),
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
        "--hard-negative-probability",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--separator-max-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--separator-hard-weight",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--run-name",
        default="drosophila_1_morphology_rag_overfit",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=230525)
    args = parser.parse_args()

    kwargs = dict(
        max_steps=args.max_steps,
        phase1_steps=args.phase1_steps,
        phase1_lr=args.phase1_lr,
        phase2_lr=args.phase2_lr,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        sample=args.sample,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        spacing_xyz=args.spacing_xyz,
        crop_shape_zyx=args.crop_shape_zyx,
        candidate_limit=args.candidate_limit,
        crop_index=args.crop_index,
        confidence_ignore_margin_um=args.confidence_ignore_margin_um,
        partial_ignore_margin_um=args.partial_ignore_margin_um,
        hard_negative_probability=args.hard_negative_probability,
        hard_negative_weight=args.hard_negative_weight,
        separator_max_threshold=args.separator_max_threshold,
        separator_hard_weight=args.separator_hard_weight,
        run_name=args.run_name,
        resume=args.resume,
        seed=args.seed,
    )

    if args.execution == "local":
        result = _train_impl(
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
        "--max-steps",
        str(args.max_steps),
        "--phase1-steps",
        str(args.phase1_steps),
        "--phase1-lr",
        str(args.phase1_lr),
        "--phase2-lr",
        str(args.phase2_lr),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--eval-every",
        str(args.eval_every),
        "--sample",
        args.sample,
        "--spacing-xyz",
        args.spacing_xyz,
        "--crop-shape-zyx",
        args.crop_shape_zyx,
        "--candidate-limit",
        str(args.candidate_limit),
        "--crop-index",
        str(args.crop_index),
        "--confidence-ignore-margin-um",
        str(args.confidence_ignore_margin_um),
        "--partial-ignore-margin-um",
        str(args.partial_ignore_margin_um),
        "--hard-negative-probability",
        str(args.hard_negative_probability),
        "--hard-negative-weight",
        str(args.hard_negative_weight),
        "--separator-max-threshold",
        str(args.separator_max_threshold),
        "--separator-hard-weight",
        str(args.separator_hard_weight),
        "--run-name",
        args.run_name,
        "--seed",
        str(args.seed),
    ]
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    if args.data_dir:
        command.extend(["--data-dir", args.data_dir])
    if args.resume:
        command.append("--resume")

    print(
        "[launcher] execution=modal\n"
        "[launcher] " + " ".join(command),
        flush=True,
    )
    completed = subprocess.run(command, cwd=LOCAL_REPO_ROOT)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    _python_cli_main()
