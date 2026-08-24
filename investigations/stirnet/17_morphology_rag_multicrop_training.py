from __future__ import annotations

"""
STIR-Net Investigation 17 — morphology-aware RAG multi-crop training.

This is the first broad training stage for the node/edge 3-D morphology branch.

Why this stage exists
---------------------
Investigation 15 proved that the new morphology representation can completely
solve a difficult separator-backed false-merge crop, but Investigation 16
showed that single-crop overfitting learned an unsafe global "separate" bias:
false merges decreased, but legitimate same-cell merges collapsed.

Investigation 17 therefore starts FRESH from the known-good step-1000 spatial
milestone and trains only the new morphology correction branch over diverse
Drosophila_1 + Drosophila_2 crops.

The legacy RAG remains frozen for the entire default run.

Training objective
------------------
For each crop:

1. Build frozen dense geometry from the step-1000 model.
2. Build the morphology-aware RAG.
3. Compute an exact legacy-RAG reference by re-running the same frozen
   rag_network with ZERO node/edge morphology embeddings.
4. Build GT RAG targets using the production RAGCriterion.
5. Supervise a balanced edge subset:
      50% positive same-GT merge edges
      50% negative different-GT separate edges
   whenever both classes exist.
6. Within the negative quota, reserve a configurable fraction for baseline
   hard negatives (GT=separate but legacy p_merge >= 0.5).
7. Add a baseline-preservation loss ONLY on edges where the legacy RAG is
   already correct and confident:
      positive legacy p >= production merge threshold
      negative legacy p <= preserve-negative threshold
   Baseline-wrong / ambiguous edges remain free to be corrected by GT.

This directly implements:
    "correct the old RAG; do not replace it."

Validation
----------
A fixed model-independent held-out manifest split is created before training.

Default:
    Drosophila_1: 6 held-out crops, forcing old Investigation-15 crop index 3
                  into validation when available.
    Drosophila_2: 6 held-out crops.

Every validation interval the script reports:
* BCE and p=0.5 edge accuracy
* false-merge rate at the actual production threshold 0.845
* positive merge acceptance at 0.845
* separator-backed false merges
* probability on baseline hard negatives
* deltas against the frozen legacy RAG

Best-checkpoint policy
----------------------
The first priority is to preserve legitimate merges.

A candidate is considered non-degraded when:
    positive_accept_rate >= baseline_positive_accept_rate
                            - allowed_positive_accept_drop

Among non-degraded checkpoints, lower false-merge rate wins, then lower BCE.
A checkpoint violating the positive-acceptance guard cannot beat one that
satisfies it.

Dense geometry / old RAG
------------------------
FROZEN:
    acquisition
    evidence stem
    spatial backbone
    geometry decoder
    all geometry heads
    RAG node_projection
    legacy RAG node encoder / message blocks / classifier
    watershed / partition behavior

TRAINABLE:
    rag_builder.morphology_builder.node_encoder
    rag_builder.morphology_builder.edge_encoder
    rag_network.node_morphology_projection
    rag_network.edge_morphology_projection

Recommended smoke
-----------------
From repository root:

    python investigations/stirnet/17_morphology_rag_multicrop_training.py ^
        --execution local ^
        --max-steps 4 ^
        --validation-every 2 ^
        --validation-crops-per-sample 2 ^
        --run-name morph_rag_multicrop_smoke

Recommended first real local run:

    python investigations/stirnet/17_morphology_rag_multicrop_training.py ^
        --execution local ^
        --max-steps 600 ^
        --run-name drosophila_12_morphology_rag_multicrop_v1

Modal is also supported:

    python investigations/stirnet/17_morphology_rag_multicrop_training.py ^
        --execution modal ^
        --max-steps 600 ^
        --run-name drosophila_12_morphology_rag_multicrop_v1

Crops with no valid supervised RAG edge are skipped without consuming an optimizer step.\n\nThe tqdm progress bar is deliberately visible and green.
"""

import argparse
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
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm
import modal


# ======================================================================================
# Modal / repository paths
# ======================================================================================

APP_NAME = "stirnet-morphology-rag-multicrop-training"

GPU = "L40S"
CPU = 3.0
MEMORY_MB = 16_384
TIMEOUT_SECONDS = 4 * 60 * 60

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
# Generic utilities
# ======================================================================================

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


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
        value,
        name="--spacing-xyz",
        cast=float,
    )
    xyz = (float(x), float(y), float(z))
    if any(not math.isfinite(v) or v <= 0 for v in xyz):
        raise ValueError("--spacing-xyz values must be finite and > 0")
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
# NIS3D loading — intentionally matched to Training 01 / Investigations 15/16
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
        else tuple(Path(v) for v in NIS3D_ROOT_CANDIDATES)
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
        "sample": sample,
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
        source_id=f"NIS3D/{sample}@morph-multicrop/{cache_namespace}",
        source_cache_path=source_cache,
    )
    source_prepare_seconds = time.perf_counter() - started

    batch = prepare_raw_source_volume_cache(
        batch,
        release_raw_volume=True,
    )

    # We never use the generic `targets` list in this investigation. Keeping an
    # int64 full-volume GT there would waste host RAM. RAG target construction
    # converts each crop to long on-device when needed.
    batch.pop("targets", None)
    batch["gt_labels"] = torch.as_tensor(
        batch["gt_labels"]
    ).to(torch.int32)
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

    # Explicitly release large temporary decoded TIFF arrays before the next
    # sample is prepared.
    del raw, gt, confidence, clean_gt, valid_mask
    gc.collect()
    return batch, report


# ======================================================================================
# Milestone hydration
# ======================================================================================

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


def _resolve_milestone(
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
        "Could not auto-resolve the Drosophila step-1000 milestone. "
        "Pass --checkpoint."
    )


def _build_morphology_model(checkpoint: Path, *, device: str):
    from learned.stirnet import StirNet, StirNetConfig

    payload = _torch_load(checkpoint, map_location="cpu")
    model_cfg = StirNetConfig()
    checkpoint_cfg = payload.get("model_config")
    if isinstance(checkpoint_cfg, dict):
        _hydrate_dataclass(model_cfg, checkpoint_cfg)

    model_cfg.partition.rag_morphology_enabled = True
    model_cfg.partition.rag_morphology_detach_geometry = True
    model_cfg.validate()

    model = StirNet(model_cfg)
    historical = payload.get("model", {})
    current = model.state_dict()

    compatible = {
        name: value
        for name, value in historical.items()
        if name in current
        and tuple(current[name].shape) == tuple(value.shape)
    }
    model.load_state_dict(compatible, strict=False)

    missing = sorted(set(current) - set(compatible))
    unexpected_missing = [
        name
        for name in missing
        if (
            "morphology_builder" not in name
            and "morphology_projection" not in name
        )
    ]
    if unexpected_missing:
        preview = "\n".join(
            f"  - {name}" for name in unexpected_missing[:20]
        )
        raise RuntimeError(
            "Unexpected milestone-transfer omissions:\n" + preview
        )

    model = model.to(device)
    return model, model_cfg, payload, {
        "historical_parameter_tensors": len(historical),
        "transferred_parameter_tensors": len(compatible),
        "new_parameter_tensors": len(missing),
        "new_parameter_names": missing,
    }


# ======================================================================================
# Frozen/trainable module policy
# ======================================================================================

def _configure_trainability(model) -> list:
    """Train ONLY the morphology CNNs + morphology residual projections."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    morphology = model.rag_builder.morphology_builder
    if morphology is None:
        raise RuntimeError(
            "Morphology-enabled model has no morphology_builder"
        )

    for parameter in morphology.parameters():
        parameter.requires_grad_(True)

    projections = (
        model.rag_network.node_morphology_projection,
        model.rag_network.edge_morphology_projection,
    )
    for module in projections:
        if module is None:
            raise RuntimeError("Morphology projection is missing")
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("No trainable morphology parameters")

    # Keep the legacy network deterministic. eval() does not disable autograd;
    # gradients still flow through the trainable morphology projections and
    # morphology CNNs.
    model.eval()
    morphology.train()
    return trainable


def _trainable_breakdown(model) -> dict[str, int]:
    morphology = model.rag_builder.morphology_builder
    projection_modules = (
        model.rag_network.node_morphology_projection,
        model.rag_network.edge_morphology_projection,
    )
    return {
        "node_edge_morphology": sum(
            p.numel() for p in morphology.parameters()
        ),
        "morphology_projections": sum(
            p.numel()
            for module in projection_modules
            for p in module.parameters()
        ),
        "legacy_rag_trainable": sum(
            p.numel()
            for name, p in model.named_parameters()
            if p.requires_grad
            and "morphology" not in name
        ),
        "total_trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }


def _gradient_norm(parameters) -> float:
    import torch

    result = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = float(
            torch.linalg.vector_norm(
                parameter.grad.detach().float()
            ).cpu()
        )
        result = math.hypot(result, value)
    return result


def _morphology_parameter_groups(model):
    morphology = list(
        model.rag_builder.morphology_builder.parameters()
    )
    projections = []
    for module in (
        model.rag_network.node_morphology_projection,
        model.rag_network.edge_morphology_projection,
    ):
        projections.extend(list(module.parameters()))
    return morphology, projections


# ======================================================================================
# Manifest / split / crop materialization
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
    result = []
    seen = set()
    for value in positions:
        index = rows[int(round(float(value)))]
        if index not in seen:
            seen.add(index)
            result.append(index)
    if len(result) < count:
        for index in rows:
            if index not in seen:
                seen.add(index)
                result.append(index)
                if len(result) >= count:
                    break
    return result


def _validation_indices(
    records,
    *,
    count: int,
    force_indices: set[int],
) -> list[int]:
    available = list(range(len(records)))
    if count >= len(available):
        return available

    forced = [
        index
        for index in sorted(force_indices)
        if 0 <= index < len(records)
    ]
    remaining_count = max(0, count - len(forced))
    blocked = set(forced)

    merge = [
        index
        for index in available
        if index not in blocked
        and bool(records[index].merge_source_ids)
    ]
    nonmerge = [
        index
        for index in available
        if index not in blocked
        and not bool(records[index].merge_source_ids)
    ]

    merge_target = min(len(merge), remaining_count // 2)
    nonmerge_target = min(
        len(nonmerge),
        remaining_count - merge_target,
    )
    leftover = remaining_count - merge_target - nonmerge_target
    if leftover:
        extra = min(len(merge) - merge_target, leftover)
        merge_target += extra
        leftover -= extra
    if leftover:
        nonmerge_target += min(
            len(nonmerge) - nonmerge_target,
            leftover,
        )

    selected = (
        forced
        + _spread_select(merge, merge_target)
        + _spread_select(nonmerge, nonmerge_target)
    )
    selected = sorted(set(selected))

    if len(selected) < count:
        for index in available:
            if index not in selected:
                selected.append(index)
                if len(selected) >= count:
                    break
    return sorted(selected[:count])


def _build_split(
    source_batches: dict[str, dict],
    *,
    crop_shape_zyx,
    validation_crops_per_sample: int,
):
    result = {}
    for sample, source_batch in source_batches.items():
        manifest = _build_manifest(source_batch, crop_shape_zyx)
        records = list(manifest.records[0])
        forced = {3} if sample == "Drosophila_1" else set()
        val = _validation_indices(
            records,
            count=validation_crops_per_sample,
            force_indices=forced,
        )
        val_set = set(val)
        train = [
            index
            for index in range(len(records))
            if index not in val_set
        ]
        if not train:
            raise RuntimeError(f"{sample}: no training crops after split")

        result[sample] = {
            "manifest": manifest,
            "records": records,
            "train_indices": train,
            "validation_indices": val,
            "train_merge_indices": [
                index
                for index in train
                if bool(records[index].merge_source_ids)
            ],
            "train_nonmerge_indices": [
                index
                for index in train
                if not bool(records[index].merge_source_ids)
            ],
        }
    return result


def _cyclic_choice(rows: list[int], ordinal: int) -> int:
    if not rows:
        raise ValueError("Cannot choose from an empty list")
    return rows[ordinal % len(rows)]


def _training_manifest_index(
    split_row: dict,
    *,
    sample_local_step: int,
) -> tuple[int, str]:
    """Alternate merge-manifest and non-merge-manifest crop provenance."""
    merge = split_row["train_merge_indices"]
    nonmerge = split_row["train_nonmerge_indices"]

    want_merge = (sample_local_step % 2) == 0
    if want_merge and merge:
        return (
            _cyclic_choice(merge, sample_local_step // 2),
            "merge",
        )
    if (not want_merge) and nonmerge:
        return (
            _cyclic_choice(nonmerge, sample_local_step // 2),
            "nonmerge",
        )

    all_rows = split_row["train_indices"]
    return (
        _cyclic_choice(all_rows, sample_local_step),
        "fallback",
    )


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
    shift = (
        crop_center - full_center
    ) * spacing_um.detach().cpu().float()

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
    record,
    *,
    partial_ignore_margin_um: float,
):
    import torch
    from learned.stirnet.training.crops import prepare_crop_batch
    from learned.stirnet.training.raw_source import (
        materialize_raw_source_crop_batch,
    )

    full_shape = tuple(
        int(v) for v in source_batch["gt_labels"].shape[-3:]
    )
    spec = _record_to_spec(
        record,
        full_shape=full_shape,
        spacing_um=source_batch["spacing_um"][0],
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
    return batch, spec


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


# ======================================================================================
# Legacy reference / edge losses
# ======================================================================================

def _legacy_rag_from_current(model, rag):
    """Exact frozen legacy-RAG result on the current graph.

    The morphology projections can be nonzero during training. Zero morphology
    embeddings make their residual exactly zero, restoring the transferred
    legacy RAG without needing a second full model.
    """
    import torch

    if (
        rag.node_morphology_embeddings is None
        or rag.edge_morphology_embeddings is None
    ):
        raise RuntimeError("Current RAG has no morphology embeddings")

    zero_rag = replace(
        rag,
        node_morphology_embeddings=torch.zeros_like(
            rag.node_morphology_embeddings
        ),
        edge_morphology_embeddings=torch.zeros_like(
            rag.edge_morphology_embeddings
        ),
    )
    return model.rag_network(zero_rag)


def _spread_tensor_indices(indices, count: int):
    import torch

    if count <= 0 or indices.numel() == 0:
        return indices[:0]
    if indices.numel() <= count:
        return indices
    positions = torch.linspace(
        0,
        indices.numel() - 1,
        count,
        device=indices.device,
    ).round().long()
    return indices[positions]


def _select_balanced_edges(
    targets,
    baseline_logits,
    *,
    max_edges_per_class: int,
    hard_negative_fraction: float,
    hard_negative_threshold: float,
):
    """Return supervised edge mask with balanced class pressure.

    Positive edges are retained up to the per-class cap.
    Negative quota matches positive count when both classes are present.
    A configurable fraction of the negative quota is reserved for baseline
    hard negatives; the remainder is spread across ordinary negatives.
    """
    import torch

    valid = targets.valid.bool()
    positive = torch.nonzero(
        valid & (targets.target > 0.5),
        as_tuple=False,
    ).flatten()
    negative = torch.nonzero(
        valid & (targets.target <= 0.5),
        as_tuple=False,
    ).flatten()

    selected_positive = _spread_tensor_indices(
        positive,
        max_edges_per_class,
    )

    if selected_positive.numel() and negative.numel():
        negative_quota = min(
            int(selected_positive.numel()),
            int(max_edges_per_class),
            int(negative.numel()),
        )
    else:
        negative_quota = min(
            int(max_edges_per_class),
            int(negative.numel()),
        )

    baseline_prob = baseline_logits.detach().sigmoid()
    hard_all = negative[
        baseline_prob[negative] >= hard_negative_threshold
    ]
    ordinary_all = negative[
        baseline_prob[negative] < hard_negative_threshold
    ]

    hard_quota = min(
        int(round(negative_quota * hard_negative_fraction)),
        int(hard_all.numel()),
    )
    if hard_quota:
        order = torch.argsort(
            baseline_prob[hard_all],
            descending=True,
        )
        selected_hard = hard_all[order[:hard_quota]]
    else:
        selected_hard = hard_all[:0]

    ordinary_quota = negative_quota - int(selected_hard.numel())
    selected_ordinary = _spread_tensor_indices(
        ordinary_all,
        ordinary_quota,
    )

    selected_negative = torch.cat(
        [selected_hard, selected_ordinary],
        dim=0,
    )

    # If ordinary negatives were insufficient, fill from unused hard rows.
    if selected_negative.numel() < negative_quota:
        used_hard = set(
            int(v)
            for v in selected_hard.detach().cpu().tolist()
        )
        hard_remaining = torch.as_tensor(
            [
                int(v)
                for v in hard_all.detach().cpu().tolist()
                if int(v) not in used_hard
            ],
            device=negative.device,
            dtype=negative.dtype,
        )
        fill = _spread_tensor_indices(
            hard_remaining,
            negative_quota - int(selected_negative.numel()),
        )
        selected_negative = torch.cat(
            [selected_negative, fill],
            dim=0,
        )

    mask = torch.zeros_like(valid)
    mask[selected_positive] = True
    mask[selected_negative] = True

    hard_mask = torch.zeros_like(valid)
    hard_mask[selected_hard] = True

    return {
        "mask": mask,
        "positive_indices": selected_positive,
        "negative_indices": selected_negative,
        "hard_negative_indices": selected_hard,
        "all_positive_count": int(positive.numel()),
        "all_negative_count": int(negative.numel()),
        "all_hard_negative_count": int(hard_all.numel()),
    }


def _class_balanced_bce(
    logits,
    targets,
    selected,
):
    import torch
    import torch.nn.functional as F

    positive_indices = selected["positive_indices"]
    negative_indices = selected["negative_indices"]

    pieces = []
    positive_loss = logits.new_zeros(())
    negative_loss = logits.new_zeros(())

    if positive_indices.numel():
        positive_loss = F.binary_cross_entropy_with_logits(
            logits[positive_indices],
            torch.ones_like(logits[positive_indices]),
        )
        pieces.append(positive_loss)

    if negative_indices.numel():
        negative_loss = F.binary_cross_entropy_with_logits(
            logits[negative_indices],
            torch.zeros_like(logits[negative_indices]),
        )
        pieces.append(negative_loss)

    if not pieces:
        return (
            logits.sum() * 0,
            positive_loss.detach(),
            negative_loss.detach(),
        )

    # Equal class contribution whenever both are present.
    loss = torch.stack(pieces).mean()
    return loss, positive_loss.detach(), negative_loss.detach()


def _balanced_mean(values, positive_mask, negative_mask):
    import torch

    pieces = []
    if bool(positive_mask.any()):
        pieces.append(values[positive_mask].mean())
    if bool(negative_mask.any()):
        pieces.append(values[negative_mask].mean())
    if not pieces:
        return values.sum() * 0
    return torch.stack(pieces).mean()


def _preservation_loss(
    new_logits,
    baseline_logits,
    targets,
    *,
    positive_threshold: float,
    negative_threshold: float,
):
    import torch.nn.functional as F

    valid = targets.valid.bool()
    target_positive = targets.target > 0.5
    baseline_prob = baseline_logits.detach().sigmoid()

    preserve_positive = (
        valid
        & target_positive
        & (baseline_prob >= positive_threshold)
    )
    preserve_negative = (
        valid
        & ~target_positive
        & (baseline_prob <= negative_threshold)
    )
    preserve = preserve_positive | preserve_negative

    if not bool(preserve.any()):
        return (
            new_logits.sum() * 0,
            preserve,
            preserve_positive,
            preserve_negative,
        )

    per_edge = F.smooth_l1_loss(
        new_logits,
        baseline_logits.detach(),
        reduction="none",
        beta=1.0,
    )
    loss = _balanced_mean(
        per_edge,
        preserve_positive,
        preserve_negative,
    )
    return loss, preserve, preserve_positive, preserve_negative


def _train_edge_metrics(
    logits,
    baseline_logits,
    targets,
    selected,
    preserve_mask,
    *,
    merge_threshold: float,
    separator_max,
    separator_threshold: float,
):
    import torch

    p = logits.detach().sigmoid()
    bp = baseline_logits.detach().sigmoid()
    valid = targets.valid.bool()
    positive = valid & (targets.target > 0.5)
    negative = valid & ~positive
    hard = negative & (bp >= 0.5)
    separator_hard = hard & (separator_max >= separator_threshold)

    def mean(mask, values):
        if not bool(mask.any()):
            return 0.0
        return float(values[mask].float().mean().cpu())

    return {
        "valid_edges": int(valid.sum().item()),
        "positive_edges": int(positive.sum().item()),
        "negative_edges": int(negative.sum().item()),
        "selected_edges": int(selected["mask"].sum().item()),
        "selected_positive_edges": int(
            selected["positive_indices"].numel()
        ),
        "selected_negative_edges": int(
            selected["negative_indices"].numel()
        ),
        "selected_hard_negative_edges": int(
            selected["hard_negative_indices"].numel()
        ),
        "all_hard_negative_edges": int(hard.sum().item()),
        "preserved_edges": int(preserve_mask.sum().item()),
        "mean_positive_probability": mean(positive, p),
        "mean_negative_probability": mean(negative, p),
        "mean_hard_negative_probability": mean(hard, p),
        "mean_separator_hard_probability": mean(
            separator_hard,
            p,
        ),
        "false_merge_count": int(
            (negative & (p >= merge_threshold)).sum().item()
        ),
        "positive_accept_count": int(
            (positive & (p >= merge_threshold)).sum().item()
        ),
    }


# ======================================================================================
# Validation aggregation
# ======================================================================================

def _zero_validation_accumulator():
    return {
        "crop_count": 0.0,
        "valid_edge_count": 0.0,
        "positive_edge_count": 0.0,
        "negative_edge_count": 0.0,
        "bce_sum": 0.0,
        "correct_count": 0.0,
        "positive_probability_sum": 0.0,
        "negative_probability_sum": 0.0,
        "false_merge_count": 0.0,
        "positive_accept_count": 0.0,
        "separator_negative_count": 0.0,
        "separator_false_merge_count": 0.0,
        "baseline_hard_count": 0.0,
        "baseline_hard_probability_sum": 0.0,
        "baseline_hard_above_merge_count": 0.0,
    }


def _validation_contribution(
    logits,
    baseline_logits,
    targets,
    *,
    merge_threshold: float,
    separator_max,
    separator_threshold: float,
):
    import torch
    import torch.nn.functional as F

    p = logits.detach().sigmoid()
    bp = baseline_logits.detach().sigmoid()
    valid = targets.valid.bool()
    positive = valid & (targets.target > 0.5)
    negative = valid & ~positive

    selected_logits = logits[valid]
    selected_target = targets.target[valid]
    bce_sum = (
        float(
            F.binary_cross_entropy_with_logits(
                selected_logits,
                selected_target,
                reduction="sum",
            ).detach().float().cpu()
        )
        if bool(valid.any())
        else 0.0
    )

    separator_negative = (
        negative & (separator_max >= separator_threshold)
    )
    baseline_hard = negative & (bp >= 0.5)

    def count(mask):
        return float(mask.sum().item())

    def probability_sum(mask):
        if not bool(mask.any()):
            return 0.0
        return float(p[mask].float().sum().cpu())

    return {
        "crop_count": 1.0,
        "valid_edge_count": count(valid),
        "positive_edge_count": count(positive),
        "negative_edge_count": count(negative),
        "bce_sum": bce_sum,
        "correct_count": count(
            valid & ((p >= 0.5) == (targets.target > 0.5))
        ),
        "positive_probability_sum": probability_sum(positive),
        "negative_probability_sum": probability_sum(negative),
        "false_merge_count": count(
            negative & (p >= merge_threshold)
        ),
        "positive_accept_count": count(
            positive & (p >= merge_threshold)
        ),
        "separator_negative_count": count(separator_negative),
        "separator_false_merge_count": count(
            separator_negative & (p >= merge_threshold)
        ),
        "baseline_hard_count": count(baseline_hard),
        "baseline_hard_probability_sum": probability_sum(
            baseline_hard
        ),
        "baseline_hard_above_merge_count": count(
            baseline_hard & (p >= merge_threshold)
        ),
    }


def _add_accumulator(acc, row):
    for key, value in row.items():
        acc[key] = acc.get(key, 0.0) + float(value)


def _safe_ratio(a, b):
    return float(a / b) if b > 0 else 0.0


def _finalize_validation(acc):
    valid = acc["valid_edge_count"]
    pos = acc["positive_edge_count"]
    neg = acc["negative_edge_count"]
    sep_neg = acc["separator_negative_count"]
    hard = acc["baseline_hard_count"]
    return {
        "crop_count": int(acc["crop_count"]),
        "valid_edge_count": int(valid),
        "positive_edge_count": int(pos),
        "negative_edge_count": int(neg),
        "bce": _safe_ratio(acc["bce_sum"], valid),
        "accuracy_05": _safe_ratio(acc["correct_count"], valid),
        "mean_positive_probability": _safe_ratio(
            acc["positive_probability_sum"],
            pos,
        ),
        "mean_negative_probability": _safe_ratio(
            acc["negative_probability_sum"],
            neg,
        ),
        "false_merge_count": int(acc["false_merge_count"]),
        "false_merge_rate": _safe_ratio(
            acc["false_merge_count"],
            neg,
        ),
        "positive_accept_count": int(
            acc["positive_accept_count"]
        ),
        "positive_accept_rate": _safe_ratio(
            acc["positive_accept_count"],
            pos,
        ),
        "positive_missed_count": int(
            pos - acc["positive_accept_count"]
        ),
        "separator_negative_count": int(sep_neg),
        "separator_false_merge_count": int(
            acc["separator_false_merge_count"]
        ),
        "separator_false_merge_rate": _safe_ratio(
            acc["separator_false_merge_count"],
            sep_neg,
        ),
        "baseline_hard_count": int(hard),
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
    }


def _validation_rank(
    metrics,
    baseline_metrics,
    *,
    allowed_positive_accept_drop: float,
):
    floor = (
        baseline_metrics["positive_accept_rate"]
        - allowed_positive_accept_drop
    )
    violation = max(
        0.0,
        floor - metrics["positive_accept_rate"],
    )
    # Lexicographic lower-is-better. Any non-degraded checkpoint beats a
    # degraded checkpoint, regardless of false-merge improvement.
    return (
        int(violation > 0),
        float(violation),
        float(metrics["false_merge_rate"]),
        float(metrics["bce"]),
    )


# ======================================================================================
# Checkpointing
# ======================================================================================

def _latest_checkpoint(recovery_dir: Path) -> Path | None:
    rows = sorted(recovery_dir.glob("checkpoint_step_*.pt"))
    return rows[-1] if rows else None


def _save_training_checkpoint(
    *,
    path: Path,
    model,
    model_cfg,
    optimizer,
    scaler,
    global_step: int,
    run_dir: Path,
    milestone: Path,
    split_summary: dict,
    validation_metrics: dict | None,
    execution_mode: str,
):
    from learned.stirnet.training.checkpoint import save_checkpoint

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_checkpoint(
            tmp,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=global_step,
            model_config=model_cfg,
            training_config={
                "experiment": "17_morphology_rag_multicrop_training",
                "legacy_rag_frozen": True,
            },
            extra={
                "experiment": "17_morphology_rag_multicrop_training",
                "run_dir": str(run_dir),
                "milestone_checkpoint": str(milestone),
                "split": split_summary,
                "validation": validation_metrics,
            },
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

    if execution_mode == "modal":
        runs_volume.commit()
    return path


def _restore_training_checkpoint(
    *,
    path: Path,
    model,
    optimizer,
    scaler,
) -> int:
    payload = _torch_load(path, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return int(payload.get("global_step", 0))


# ======================================================================================
# Forward helpers
# ======================================================================================

def _forward_crop(
    *,
    model,
    crop,
    rag_criterion,
    amp_dtype: str,
    need_grad: bool,
):
    import torch

    # The complete dense model is frozen and intentionally computed with no
    # autograd. It still supplies D0 + all learned geometry fields to the new
    # morphology path.
    with torch.no_grad(), _autocast_context(amp_dtype):
        geometry = model(
            crop["spatial_inputs"],
            crop["spacing_um"],
            crop["dref_um"],
            spatial_padding_mask=crop.get("spatial_padding_mask"),
            execution_stage="geometry",
        )

    context = (
        _autocast_context(amp_dtype)
        if need_grad
        else torch.no_grad()
    )
    if need_grad:
        with _autocast_context(amp_dtype):
            output = model(
                crop["spatial_inputs"],
                crop["spacing_um"],
                crop["dref_um"],
                spatial_padding_mask=crop.get("spatial_padding_mask"),
                execution_stage="spatial",
                precomputed_geometry=geometry,
            )
    else:
        with torch.no_grad(), _autocast_context(amp_dtype):
            output = model(
                crop["spatial_inputs"],
                crop["spacing_um"],
                crop["dref_um"],
                spatial_padding_mask=crop.get("spatial_padding_mask"),
                execution_stage="spatial",
                precomputed_geometry=geometry,
            )

    # The legacy-RAG reference is detached and deterministic because the
    # transferred RAG is frozen and the model is kept in eval mode.
    with torch.no_grad(), _autocast_context(amp_dtype):
        baseline_rag = _legacy_rag_from_current(
            model,
            output.rag,
        )

    targets = rag_criterion.build_targets(
        output.rag,
        crop["gt_labels"],
        valid_mask=crop.get("supervision_valid_mask"),
    )
    return geometry, output, baseline_rag, targets


# ======================================================================================
# Validation
# ======================================================================================

def _run_validation(
    *,
    model,
    source_batches,
    splits,
    rag_criterion,
    amp_dtype: str,
    partial_ignore_margin_um: float,
    merge_threshold: float,
    separator_threshold: float,
    progress_prefix: str,
):
    import torch

    model.eval()
    model.rag_builder.morphology_builder.eval()

    candidate_acc = _zero_validation_accumulator()
    baseline_acc = _zero_validation_accumulator()
    per_sample = {}

    total_crops = sum(
        len(splits[sample]["validation_indices"])
        for sample in source_batches
    )
    val_bar = tqdm(
        total=total_crops,
        desc=progress_prefix,
        unit="crop",
        dynamic_ncols=True,
        leave=False,
        colour="green",
        file=sys.stdout,
    )

    for sample, source_batch in source_batches.items():
        sample_candidate = _zero_validation_accumulator()
        sample_baseline = _zero_validation_accumulator()

        for manifest_index in splits[sample]["validation_indices"]:
            record = splits[sample]["records"][manifest_index]
            crop_cpu, _ = _materialize_crop(
                source_batch,
                record,
                partial_ignore_margin_um=partial_ignore_margin_um,
            )
            crop = _move_crop_to_cuda(crop_cpu)

            geometry, output, baseline_rag, targets = _forward_crop(
                model=model,
                crop=crop,
                rag_criterion=rag_criterion,
                amp_dtype=amp_dtype,
                need_grad=False,
            )

            separator_max = (
                output.rag.edge_features[:, 1].detach()
                if output.rag.edge_features.numel()
                else output.rag.spatial_edge_logits.new_zeros(
                    output.rag.spatial_edge_logits.shape
                )
            )

            candidate_row = _validation_contribution(
                output.rag.spatial_edge_logits,
                baseline_rag.spatial_edge_logits,
                targets,
                merge_threshold=merge_threshold,
                separator_max=separator_max,
                separator_threshold=separator_threshold,
            )
            # For baseline metrics, pass baseline logits as both candidate and
            # baseline reference.
            baseline_row = _validation_contribution(
                baseline_rag.spatial_edge_logits,
                baseline_rag.spatial_edge_logits,
                targets,
                merge_threshold=merge_threshold,
                separator_max=separator_max,
                separator_threshold=separator_threshold,
            )

            _add_accumulator(candidate_acc, candidate_row)
            _add_accumulator(baseline_acc, baseline_row)
            _add_accumulator(sample_candidate, candidate_row)
            _add_accumulator(sample_baseline, baseline_row)

            val_bar.set_postfix(
                {
                    "sample": sample.replace("Drosophila_", "D"),
                    "idx": manifest_index,
                    "FM": int(candidate_row["false_merge_count"]),
                    "PA": int(candidate_row["positive_accept_count"]),
                },
                refresh=False,
            )
            val_bar.update(1)

            del geometry, output, baseline_rag, targets, crop, crop_cpu
            torch.cuda.empty_cache()

        per_sample[sample] = {
            "candidate": _finalize_validation(sample_candidate),
            "baseline": _finalize_validation(sample_baseline),
        }

    val_bar.close()
    model.rag_builder.morphology_builder.train()

    return {
        "candidate": _finalize_validation(candidate_acc),
        "baseline": _finalize_validation(baseline_acc),
        "per_sample": per_sample,
    }


# ======================================================================================
# Main training implementation
# ======================================================================================

def _print_header(
    *,
    execution_mode: str,
    milestone: Path,
    nis3d_root: Path,
    samples,
    crop_shape,
    spacing_override_zyx_um,
    max_steps,
    lr,
    validation_every,
    validation_crops_per_sample,
    max_edges_per_class,
    hard_negative_fraction,
    preservation_weight,
    preserve_negative_threshold,
    merge_threshold,
    allowed_positive_accept_drop,
    amp_dtype,
    results_root,
):
    import torch

    props = torch.cuda.get_device_properties(0)
    print("=" * 122, flush=True)
    print(
        "STIR-Net Investigation 17 — morphology-aware RAG multi-crop training",
        flush=True,
    )
    print("=" * 122, flush=True)
    print(f"GPU                      : {props.name}", flush=True)
    print(
        f"GPU VRAM                 : {props.total_memory / 2**30:.2f} GiB",
        flush=True,
    )
    print(f"Execution                : {execution_mode}", flush=True)
    if execution_mode == "modal":
        print(
            f"CPU / RAM request        : {CPU:g} CPU / "
            f"{MEMORY_MB/1024:.1f} GiB",
            flush=True,
        )
    print(f"Results root             : {results_root}", flush=True)
    print(f"Milestone checkpoint     : {milestone}", flush=True)
    print(f"NIS3D root               : {nis3d_root}", flush=True)
    print(f"Samples                   : {list(samples)}", flush=True)
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
    print(f"Optimizer steps           : {max_steps}", flush=True)
    print(f"Learning rate             : {lr:g}", flush=True)
    print(
        f"Held-out validation       : every {validation_every} steps, "
        f"{validation_crops_per_sample}/sample",
        flush=True,
    )
    print(
        f"Balanced edge cap         : {max_edges_per_class}/class/crop",
        flush=True,
    )
    print(
        f"Negative hard fraction    : {hard_negative_fraction:.2f}",
        flush=True,
    )
    print(
        f"Preservation weight       : {preservation_weight:g}",
        flush=True,
    )
    print(
        "Preserve legacy correct   : "
        f"positive p>={merge_threshold:.3f}; "
        f"negative p<={preserve_negative_threshold:.3f}",
        flush=True,
    )
    print(
        f"Allowed positive drop     : {allowed_positive_accept_drop:.3f}",
        flush=True,
    )
    print(f"AMP                       : {amp_dtype}", flush=True)
    print(
        "Dense geometry            : FROZEN",
        flush=True,
    )
    print(
        "Legacy RAG                : FROZEN for entire run",
        flush=True,
    )
    print(
        "Trainable                 : node/edge morphology CNNs + residual projections",
        flush=True,
    )
    print(
        "Source dropout / XY flip  : OFF / OFF (isolate morphology learning)",
        flush=True,
    )
    print("=" * 122, flush=True)


def _training_impl(
    *,
    max_steps: int = 600,
    lr: float = 1e-4,
    checkpoint_every: int = 50,
    validation_every: int = 50,
    validation_crops_per_sample: int = 6,
    samples_csv: str = "Drosophila_1,Drosophila_2",
    checkpoint: str = "",
    data_dir: str = "",
    spacing_xyz: str = "0.20312639,0.20312639,0.79099447",
    crop_shape_zyx: str = "32,192,192",
    confidence_ignore_margin_um: float = 1.0,
    partial_ignore_margin_um: float = 1.0,
    max_edges_per_class: int = 64,
    hard_negative_fraction: float = 0.50,
    hard_negative_threshold: float = 0.50,
    preservation_weight: float = 0.25,
    preserve_negative_threshold: float = 0.10,
    merge_threshold: float = 0.845,
    separator_threshold: float = 0.50,
    allowed_positive_accept_drop: float = 0.05,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    run_name: str = "drosophila_12_morphology_rag_multicrop_v1",
    resume: bool = False,
    seed: int = 230525,
    execution_mode: str = "local",
) -> dict[str, Any]:
    import torch
    from learned.stirnet.model.partition.rag import RAGCriterion

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Investigation 17")
    if execution_mode not in {"local", "modal"}:
        raise ValueError("execution_mode must be local or modal")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if checkpoint_every < 1 or validation_every < 1:
        raise ValueError("checkpoint/validation intervals must be positive")
    if validation_crops_per_sample < 1:
        raise ValueError("validation_crops_per_sample must be positive")
    if max_edges_per_class < 1:
        raise ValueError("max_edges_per_class must be positive")
    if not 0 <= hard_negative_fraction <= 1:
        raise ValueError("hard_negative_fraction must be in [0,1]")
    if not 0 <= hard_negative_threshold <= 1:
        raise ValueError("hard_negative_threshold must be in [0,1]")
    if preservation_weight < 0:
        raise ValueError("preservation_weight cannot be negative")
    if not 0 <= preserve_negative_threshold <= 1:
        raise ValueError("preserve_negative_threshold must be in [0,1]")
    if not 0 <= merge_threshold <= 1:
        raise ValueError("merge_threshold must be in [0,1]")
    if not 0 <= separator_threshold <= 1:
        raise ValueError("separator_threshold must be in [0,1]")
    if allowed_positive_accept_drop < 0:
        raise ValueError("allowed_positive_accept_drop cannot be negative")

    samples = tuple(
        token.strip()
        for token in samples_csv.split(",")
        if token.strip()
    )
    if not samples:
        raise ValueError("At least one sample is required")

    _seed_everything(seed)
    crop_shape = _parse_shape_zyx(crop_shape_zyx)
    spacing_override_zyx_um = _parse_spacing_xyz_override(
        spacing_xyz
    )
    amp_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

    milestone = _resolve_milestone(
        checkpoint,
        execution_mode=execution_mode,
    )
    nis3d_root = _discover_nis3d_root(
        samples,
        data_dir=data_dir,
        execution_mode=execution_mode,
    )

    runs_root = (
        LOCAL_REPO_ROOT / "runs"
        if execution_mode == "local"
        else Path(RUNS_MOUNT)
    )
    experiment_root = (
        runs_root
        / "stirnet"
        / "investigations"
        / "17_morphology_rag_multicrop_training"
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
        milestone=milestone,
        nis3d_root=nis3d_root,
        samples=samples,
        crop_shape=crop_shape,
        spacing_override_zyx_um=spacing_override_zyx_um,
        max_steps=max_steps,
        lr=lr,
        validation_every=validation_every,
        validation_crops_per_sample=validation_crops_per_sample,
        max_edges_per_class=max_edges_per_class,
        hard_negative_fraction=hard_negative_fraction,
        preservation_weight=preservation_weight,
        preserve_negative_threshold=preserve_negative_threshold,
        merge_threshold=merge_threshold,
        allowed_positive_accept_drop=allowed_positive_accept_drop,
        amp_dtype=amp_dtype,
        results_root=experiment_root,
    )

    print("[init] constructing fresh morphology-enabled model from milestone ...", flush=True)
    model, model_cfg, milestone_payload, transfer_report = (
        _build_morphology_model(milestone, device="cuda")
    )
    trainable = _configure_trainability(model)
    breakdown = _trainable_breakdown(model)

    print(
        f"[init] transferred "
        f"{transfer_report['transferred_parameter_tensors']}/"
        f"{transfer_report['historical_parameter_tensors']} historical tensors; "
        f"new={transfer_report['new_parameter_tensors']}",
        flush=True,
    )
    print(
        f"[trainable] morphology={breakdown['node_edge_morphology']/1e6:.3f}M "
        f"projections={breakdown['morphology_projections']/1e6:.3f}M "
        f"legacy_rag={breakdown['legacy_rag_trainable']/1e6:.3f}M "
        f"total={breakdown['total_trainable']/1e6:.3f}M",
        flush=True,
    )
    if breakdown["legacy_rag_trainable"] != 0:
        raise RuntimeError("Legacy RAG unexpectedly has trainable parameters")

    rag_criterion = RAGCriterion(model_cfg.partition).to("cuda")
    rag_criterion.eval()

    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(amp_dtype == "fp16"),
    )

    # ------------------------------------------------------------------
    # Prepare both sources once, matching the production Training-01
    # source semantics.
    # ------------------------------------------------------------------
    source_batches = {}
    sample_reports = {}
    for sample in samples:
        print(f"[data] preparing {sample} ...", flush=True)
        signature = _data_signature(
            nis3d_root=nis3d_root,
            sample=sample,
            spacing_override_zyx_um=spacing_override_zyx_um,
            confidence_ignore_margin_um=confidence_ignore_margin_um,
        )
        source_batch, report = _prepare_sample_batch(
            nis3d_root=nis3d_root,
            sample=sample,
            spacing_override_zyx_um=spacing_override_zyx_um,
            confidence_ignore_margin_um=confidence_ignore_margin_um,
            cache_root=cache_root,
            cache_namespace=signature,
        )
        source_batches[sample] = source_batch
        sample_reports[sample] = report
        print(
            f"[data] {sample}: shape={tuple(report['shape_zyx'])} "
            f"GT={report['gt_ids_kept']} "
            f"source={report['source_instance_count']} "
            f"dref={report['model_dref_um']:.4f}um "
            f"RAM-cache={report['source_ram_cache_gross_gib']:.2f}GiB",
            flush=True,
        )

    print("[split] constructing model-independent train/validation manifests ...", flush=True)
    splits = _build_split(
        source_batches,
        crop_shape_zyx=crop_shape,
        validation_crops_per_sample=validation_crops_per_sample,
    )

    split_summary = {}
    for sample in samples:
        row = splits[sample]
        split_summary[sample] = {
            "manifest_record_count": len(row["records"]),
            "train_indices": row["train_indices"],
            "validation_indices": row["validation_indices"],
            "train_merge_count": len(row["train_merge_indices"]),
            "train_nonmerge_count": len(row["train_nonmerge_indices"]),
        }
        print(
            f"[split] {sample}: manifest={len(row['records'])} "
            f"train={len(row['train_indices'])} "
            f"(merge={len(row['train_merge_indices'])}, "
            f"nonmerge={len(row['train_nonmerge_indices'])}) "
            f"val={row['validation_indices']}",
            flush=True,
        )

    config_payload = {
        "experiment": "17_morphology_rag_multicrop_training",
        "milestone_checkpoint": str(milestone),
        "milestone_global_step": int(
            milestone_payload.get("global_step", -1)
        ),
        "transfer_report": transfer_report,
        "trainable_breakdown": breakdown,
        "model": model_cfg.to_dict(),
        "samples": list(samples),
        "sample_reports": sample_reports,
        "split": split_summary,
        "max_steps": int(max_steps),
        "lr": float(lr),
        "checkpoint_every": int(checkpoint_every),
        "validation_every": int(validation_every),
        "validation_crops_per_sample": int(
            validation_crops_per_sample
        ),
        "crop_shape_zyx": list(crop_shape),
        "max_edges_per_class": int(max_edges_per_class),
        "hard_negative_fraction": float(hard_negative_fraction),
        "hard_negative_threshold": float(hard_negative_threshold),
        "preservation_weight": float(preservation_weight),
        "preserve_positive_threshold": float(merge_threshold),
        "preserve_negative_threshold": float(
            preserve_negative_threshold
        ),
        "merge_threshold": float(merge_threshold),
        "separator_threshold": float(separator_threshold),
        "allowed_positive_accept_drop": float(
            allowed_positive_accept_drop
        ),
        "weight_decay": float(weight_decay),
        "max_grad_norm": float(max_grad_norm),
        "legacy_rag_frozen": True,
        "dense_geometry_frozen": True,
        "source_dropout_probability": 0.0,
        "xy_flip_probability": 0.0,
        "seed": int(seed),
        "amp_dtype": amp_dtype,
    }
    _atomic_json(run_dir / "config.json", config_payload)
    _atomic_json(run_dir / "samples.json", sample_reports)
    _atomic_json(run_dir / "split.json", split_summary)

    # Resume only from this run-name's recovery stream. Fresh default always
    # starts from the original milestone.
    global_step = 0
    resumed_from = None
    if resume:
        latest = _latest_checkpoint(recovery_dir)
        if latest is not None:
            global_step = _restore_training_checkpoint(
                path=latest,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
            )
            _configure_trainability(model)
            resumed_from = str(latest)
            print(
                f"[resume] loaded {latest} at step={global_step}",
                flush=True,
            )
        else:
            print(
                "[resume] no recovery checkpoint found; starting fresh",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Initial held-out reference. At step 0 the morphology residual is zero,
    # therefore candidate==legacy baseline exactly.
    # ------------------------------------------------------------------
    print("[validation] evaluating fixed held-out baseline before training ...", flush=True)
    initial_validation = _run_validation(
        model=model,
        source_batches=source_batches,
        splits=splits,
        rag_criterion=rag_criterion,
        amp_dtype=amp_dtype,
        partial_ignore_margin_um=partial_ignore_margin_um,
        merge_threshold=merge_threshold,
        separator_threshold=separator_threshold,
        progress_prefix="Val@0",
    )
    baseline_validation = initial_validation["baseline"]
    _atomic_json(
        run_dir / "validation_step_000000.json",
        initial_validation,
    )

    print(
        "[validation baseline] "
        f"BCE={baseline_validation['bce']:.5f} "
        f"FM={baseline_validation['false_merge_count']}/"
        f"{baseline_validation['negative_edge_count']} "
        f"FM-rate={baseline_validation['false_merge_rate']:.5f} "
        f"pos-accept={baseline_validation['positive_accept_rate']:.4f} "
        f"hard-p={baseline_validation['baseline_hard_mean_probability']:.4f}",
        flush=True,
    )

    best_rank = _validation_rank(
        initial_validation["candidate"],
        baseline_validation,
        allowed_positive_accept_drop=allowed_positive_accept_drop,
    )
    best_step = 0
    best_metrics = initial_validation["candidate"]

    # Step-0 best is the original behavior. Save no separate model file yet;
    # the milestone is already durable.

    history_path = run_dir / "history.jsonl"
    validation_history_path = run_dir / "validation_history.jsonl"

    morphology_parameters, projection_parameters = (
        _morphology_parameter_groups(model)
    )

    sample_local_steps = {sample: 0 for sample in samples}
    # On resume, reconstruct the deterministic alternation counters from the
    # successful global step count.
    for completed in range(global_step):
        sample = samples[completed % len(samples)]
        sample_local_steps[sample] += 1

    GREEN = "\033[32m"
    RESET = "\033[0m"
    progress = tqdm(
        total=max_steps,
        initial=global_step,
        desc=f"{GREEN}Morph-RAG-Multi{RESET}",
        unit="step",
        dynamic_ncols=True,
        smoothing=0.10,
        mininterval=0.5,
        leave=True,
        colour="green",
        file=sys.stdout,
    )

    started_all = time.perf_counter()
    step_time_sum = 0.0
    successful_steps = 0
    skipped_crops = 0
    consecutive_skipped_crops = 0
    max_consecutive_skipped_crops = 256
    skipped_crops_path = run_dir / "skipped_crops.jsonl"
    last_validation = initial_validation
    last_checkpoint_step = None

    try:
        while global_step < max_steps:
            sample = samples[global_step % len(samples)]
            local_step = sample_local_steps[sample]
            manifest_index, provenance = _training_manifest_index(
                splits[sample],
                sample_local_step=local_step,
            )
            sample_local_steps[sample] += 1

            record = splits[sample]["records"][manifest_index]
            crop_cpu, spec = _materialize_crop(
                source_batches[sample],
                record,
                partial_ignore_margin_um=partial_ignore_margin_um,
            )
            crop = _move_crop_to_cuda(crop_cpu)

            model.eval()
            model.rag_builder.morphology_builder.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()

            step_started = time.perf_counter()

            geometry, output, baseline_rag, targets = _forward_crop(
                model=model,
                crop=crop,
                rag_criterion=rag_criterion,
                amp_dtype=amp_dtype,
                need_grad=True,
            )

            if (
                output.rag.edge_index.shape
                != baseline_rag.edge_index.shape
                or not torch.equal(
                    output.rag.edge_index,
                    baseline_rag.edge_index,
                )
            ):
                raise RuntimeError(
                    "Legacy/current RAG edge topology mismatch"
                )

            selected = _select_balanced_edges(
                targets,
                baseline_rag.spatial_edge_logits,
                max_edges_per_class=max_edges_per_class,
                hard_negative_fraction=hard_negative_fraction,
                hard_negative_threshold=hard_negative_threshold,
            )

            # Some manifest crops legitimately produce no trainable RAG edge:
            # e.g. watershed yields no adjacency, or every edge is rejected by
            # the production purity/support validity guard. In that case the
            # graph network's empty-edge branch returns an empty constant logit
            # tensor, so a synthetic zero loss would have no autograd graph.
            #
            # Treat these as data skips, NOT optimizer steps. sample_local_steps
            # was already advanced above, so the next loop tries the next crop
            # from the same sample while global_step/progress remain unchanged.
            if not bool(selected["mask"].any()):
                skipped_crops += 1
                consecutive_skipped_crops += 1
                skip_record = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "optimizer_step": int(global_step),
                    "sample": sample,
                    "sample_local_step": int(local_step),
                    "manifest_index": int(manifest_index),
                    "manifest_provenance": provenance,
                    "candidate_type": str(record.candidate_type),
                    "rag_edge_count": int(output.rag.edge_index.shape[1]),
                    "valid_edge_count": int(targets.valid.sum().item()),
                    "positive_valid_edge_count": int(
                        (targets.valid & (targets.target > 0.5)).sum().item()
                    ),
                    "negative_valid_edge_count": int(
                        (targets.valid & (targets.target <= 0.5)).sum().item()
                    ),
                    "reason": "no_selected_supervised_rag_edges",
                    "skipped_crops_total": int(skipped_crops),
                }
                _append_jsonl(skipped_crops_path, skip_record)
                tqdm.write(
                    "[skip] "
                    f"step={global_step} sample={sample} "
                    f"manifest={manifest_index} "
                    f"edges={skip_record['rag_edge_count']} "
                    f"valid={skip_record['valid_edge_count']} "
                    "(no supervised RAG edges)"
                )

                del geometry, output, baseline_rag, targets
                del crop, crop_cpu
                torch.cuda.empty_cache()

                if consecutive_skipped_crops >= max_consecutive_skipped_crops:
                    raise RuntimeError(
                        "Exceeded "
                        f"{max_consecutive_skipped_crops} consecutive crops "
                        "without a supervised RAG edge. Check the manifest, "
                        "watershed topology, or RAG purity/support thresholds."
                    )
                continue

            consecutive_skipped_crops = 0

            with _autocast_context(amp_dtype):
                supervised_loss, positive_loss, negative_loss = (
                    _class_balanced_bce(
                        output.rag.spatial_edge_logits,
                        targets,
                        selected,
                    )
                )
                (
                    preserve_loss,
                    preserve_mask,
                    preserve_positive_mask,
                    preserve_negative_mask,
                ) = _preservation_loss(
                    output.rag.spatial_edge_logits,
                    baseline_rag.spatial_edge_logits,
                    targets,
                    positive_threshold=merge_threshold,
                    negative_threshold=preserve_negative_threshold,
                )
                loss = (
                    supervised_loss
                    + float(preservation_weight) * preserve_loss
                )

            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite loss at step {global_step}"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            morphology_grad = _gradient_norm(
                morphology_parameters
            )
            projection_grad = _gradient_norm(
                projection_parameters
            )
            total_grad = torch.nn.utils.clip_grad_norm_(
                trainable,
                max_norm=max_grad_norm,
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
            average_step_seconds = (
                step_time_sum / max(successful_steps, 1)
            )

            separator_max = (
                output.rag.edge_features[:, 1].detach()
                if output.rag.edge_features.numel()
                else output.rag.spatial_edge_logits.new_zeros(
                    output.rag.spatial_edge_logits.shape
                )
            )
            train_metrics = _train_edge_metrics(
                output.rag.spatial_edge_logits,
                baseline_rag.spatial_edge_logits,
                targets,
                selected,
                preserve_mask,
                merge_threshold=merge_threshold,
                separator_max=separator_max,
                separator_threshold=separator_threshold,
            )

            elapsed = time.perf_counter() - started_all
            eta = average_step_seconds * (max_steps - global_step)

            history_row = {
                "step": int(global_step),
                "sample": sample,
                "sample_local_step": int(local_step),
                "manifest_index": int(manifest_index),
                "manifest_provenance": provenance,
                "candidate_type": str(record.candidate_type),
                "merge_source_count": len(record.merge_source_ids),
                "merge_gt_count": len(record.merge_gt_ids),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "loss": float(loss.detach().float().cpu()),
                "supervised_loss": float(
                    supervised_loss.detach().float().cpu()
                ),
                "positive_loss": float(positive_loss.float().cpu()),
                "negative_loss": float(negative_loss.float().cpu()),
                "preservation_loss": float(
                    preserve_loss.detach().float().cpu()
                ),
                "preservation_weight": float(preservation_weight),
                "preserve_positive_edges": int(
                    preserve_positive_mask.sum().item()
                ),
                "preserve_negative_edges": int(
                    preserve_negative_mask.sum().item()
                ),
                "lr": float(lr),
                "grad_norm": float(
                    torch.as_tensor(total_grad).detach().float().cpu()
                ),
                "grad_morphology_encoder": float(morphology_grad),
                "grad_morphology_projection": float(projection_grad),
                "step_seconds": float(step_seconds),
                "average_step_seconds": float(average_step_seconds),
                "elapsed_seconds": float(elapsed),
                "estimated_remaining_seconds": float(eta),
                "cuda_peak_allocated_gib": float(
                    torch.cuda.max_memory_allocated() / 2**30
                ),
                "cuda_peak_reserved_gib": float(
                    torch.cuda.max_memory_reserved() / 2**30
                ),
                **train_metrics,
            }
            _append_jsonl(history_path, history_row)

            should_validate = (
                global_step % validation_every == 0
                or global_step == max_steps
            )

            if should_validate:
                progress.set_postfix(
                    {
                        "loss": f"{history_row['loss']:.4f}",
                        "sample": sample.replace("Drosophila_", "D"),
                        "idx": manifest_index,
                        "val": "...",
                        "avg": f"{average_step_seconds:.2f}s",
                    },
                    refresh=True,
                )
                tqdm.write(
                    f"[validation] step={global_step} evaluating fixed held-out split ..."
                )

                validation = _run_validation(
                    model=model,
                    source_batches=source_batches,
                    splits=splits,
                    rag_criterion=rag_criterion,
                    amp_dtype=amp_dtype,
                    partial_ignore_margin_um=partial_ignore_margin_um,
                    merge_threshold=merge_threshold,
                    separator_threshold=separator_threshold,
                    progress_prefix=f"Val@{global_step}",
                )
                last_validation = validation
                candidate = validation["candidate"]
                baseline = validation["baseline"]

                validation_record = {
                    "step": int(global_step),
                    "candidate": candidate,
                    "baseline": baseline,
                    "per_sample": validation["per_sample"],
                    "delta_vs_baseline": {
                        "bce": candidate["bce"] - baseline["bce"],
                        "false_merge_rate": (
                            candidate["false_merge_rate"]
                            - baseline["false_merge_rate"]
                        ),
                        "positive_accept_rate": (
                            candidate["positive_accept_rate"]
                            - baseline["positive_accept_rate"]
                        ),
                        "baseline_hard_mean_probability": (
                            candidate["baseline_hard_mean_probability"]
                            - baseline["baseline_hard_mean_probability"]
                        ),
                    },
                    "rank": list(
                        _validation_rank(
                            candidate,
                            baseline_validation,
                            allowed_positive_accept_drop=(
                                allowed_positive_accept_drop
                            ),
                        )
                    ),
                }
                _append_jsonl(
                    validation_history_path,
                    validation_record,
                )
                _atomic_json(
                    run_dir / f"validation_step_{global_step:06d}.json",
                    validation_record,
                )

                current_rank = tuple(validation_record["rank"])
                is_best = current_rank < tuple(best_rank)
                if is_best:
                    best_rank = current_rank
                    best_step = global_step
                    best_metrics = candidate
                    best_path = recovery_dir / "best_checkpoint.pt"
                    _save_training_checkpoint(
                        path=best_path,
                        model=model,
                        model_cfg=model_cfg,
                        optimizer=optimizer,
                        scaler=scaler,
                        global_step=global_step,
                        run_dir=run_dir,
                        milestone=milestone,
                        split_summary=split_summary,
                        validation_metrics=validation_record,
                        execution_mode=execution_mode,
                    )
                    _atomic_json(
                        recovery_dir / "best_state.json",
                        {
                            "step": int(global_step),
                            "checkpoint": str(best_path),
                            "rank": list(best_rank),
                            "metrics": best_metrics,
                        },
                    )
                    tqdm.write(
                        f"[best] step={global_step}: "
                        f"FM-rate={candidate['false_merge_rate']:.5f}, "
                        f"pos-accept={candidate['positive_accept_rate']:.4f}, "
                        f"BCE={candidate['bce']:.5f}"
                    )

                tqdm.write(
                    f"[validation] step={global_step}: "
                    f"baseline FM={baseline['false_merge_count']}/"
                    f"{baseline['negative_edge_count']} "
                    f"pos-accept={baseline['positive_accept_rate']:.4f} | "
                    f"current FM={candidate['false_merge_count']}/"
                    f"{candidate['negative_edge_count']} "
                    f"pos-accept={candidate['positive_accept_rate']:.4f} "
                    f"hard-p={candidate['baseline_hard_mean_probability']:.4f} "
                    f"BCE={candidate['bce']:.5f}"
                )

            display_val = last_validation["candidate"]
            progress.set_postfix(
                {
                    "loss": f"{history_row['loss']:.4f}",
                    "sample": sample.replace("Drosophila_", "D"),
                    "idx": manifest_index,
                    "FM": display_val["false_merge_count"],
                    "PA": f"{display_val['positive_accept_rate']:.3f}",
                    "hard_p": f"{display_val['baseline_hard_mean_probability']:.3f}",
                    "avg": f"{average_step_seconds:.2f}s",
                },
                refresh=False,
            )
            progress.update(1)

            del geometry, output, baseline_rag, targets
            del crop, crop_cpu, loss, supervised_loss, preserve_loss
            torch.cuda.empty_cache()

            if (
                global_step % checkpoint_every == 0
                or global_step == max_steps
            ):
                checkpoint_path = (
                    recovery_dir
                    / f"checkpoint_step_{global_step:06d}.pt"
                )
                _save_training_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    model_cfg=model_cfg,
                    optimizer=optimizer,
                    scaler=scaler,
                    global_step=global_step,
                    run_dir=run_dir,
                    milestone=milestone,
                    split_summary=split_summary,
                    validation_metrics=(
                        None
                        if not should_validate
                        else validation_record
                    ),
                    execution_mode=execution_mode,
                )
                last_checkpoint_step = global_step
                _atomic_text(
                    recovery_dir / "latest_checkpoint.txt",
                    checkpoint_path.name + "\n",
                )
                tqdm.write(
                    f"[checkpoint] persisted step={global_step} "
                    f"path={checkpoint_path}"
                )

        progress.close()

        elapsed = time.perf_counter() - started_all
        final_validation = last_validation["candidate"]
        summary = {
            "status": "success",
            "experiment": "17_morphology_rag_multicrop_training",
            "global_step": int(global_step),
            "max_steps": int(max_steps),
            "run_dir": str(run_dir),
            "recovery_dir": str(recovery_dir),
            "milestone_checkpoint": str(milestone),
            "resumed_from": resumed_from,
            "last_checkpoint_step": last_checkpoint_step,
            "best_step": int(best_step),
            "best_rank": list(best_rank),
            "best_metrics": best_metrics,
            "baseline_validation": baseline_validation,
            "final_validation": final_validation,
            "final_delta_vs_baseline": {
                "bce": (
                    final_validation["bce"]
                    - baseline_validation["bce"]
                ),
                "false_merge_rate": (
                    final_validation["false_merge_rate"]
                    - baseline_validation["false_merge_rate"]
                ),
                "positive_accept_rate": (
                    final_validation["positive_accept_rate"]
                    - baseline_validation["positive_accept_rate"]
                ),
                "baseline_hard_mean_probability": (
                    final_validation[
                        "baseline_hard_mean_probability"
                    ]
                    - baseline_validation[
                        "baseline_hard_mean_probability"
                    ]
                ),
            },
            "sample_reports": sample_reports,
            "split": split_summary,
            "elapsed_seconds": float(elapsed),
            "mean_training_step_seconds": float(
                step_time_sum / max(successful_steps, 1)
            ),
            "successful_optimizer_steps": int(successful_steps),
            "skipped_crops": int(skipped_crops),
            "peak_cuda_allocated_gib": float(
                torch.cuda.max_memory_allocated() / 2**30
            ),
            "peak_cuda_reserved_gib": float(
                torch.cuda.max_memory_reserved() / 2**30
            ),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(run_dir / "summary.json", summary)
        if execution_mode == "modal":
            runs_volume.commit()

        print("", flush=True)
        print("=" * 122, flush=True)
        print("Investigation 17 complete", flush=True)
        print("=" * 122, flush=True)
        print(f"Elapsed                  : {_duration(elapsed)}", flush=True)
        print(
            f"Mean train step           : "
            f"{step_time_sum / max(successful_steps, 1):.2f}s",
            flush=True,
        )
        print(
            f"Skipped empty crops       : {skipped_crops}",
            flush=True,
        )
        print(
            f"Baseline held-out         : "
            f"FM={baseline_validation['false_merge_count']}/"
            f"{baseline_validation['negative_edge_count']} "
            f"rate={baseline_validation['false_merge_rate']:.5f} "
            f"pos-accept={baseline_validation['positive_accept_rate']:.4f} "
            f"BCE={baseline_validation['bce']:.5f}",
            flush=True,
        )
        print(
            f"Final held-out            : "
            f"FM={final_validation['false_merge_count']}/"
            f"{final_validation['negative_edge_count']} "
            f"rate={final_validation['false_merge_rate']:.5f} "
            f"pos-accept={final_validation['positive_accept_rate']:.4f} "
            f"BCE={final_validation['bce']:.5f}",
            flush=True,
        )
        print(
            f"Best step                : {best_step}",
            flush=True,
        )
        print(
            f"Best metrics             : "
            f"FM-rate={best_metrics['false_merge_rate']:.5f} "
            f"pos-accept={best_metrics['positive_accept_rate']:.4f} "
            f"BCE={best_metrics['bce']:.5f}",
            flush=True,
        )
        print(f"Run directory            : {run_dir}", flush=True)
        print("=" * 122, flush=True)
        return summary

    except BaseException as error:
        progress.close()
        failure = {
            "status": "failed",
            "global_step": int(global_step),
            "skipped_crops": int(skipped_crops),
            "consecutive_skipped_crops": int(consecutive_skipped_crops),
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
def train_morphology_rag_multicrop(
    max_steps: int = 600,
    lr: float = 1e-4,
    checkpoint_every: int = 50,
    validation_every: int = 50,
    validation_crops_per_sample: int = 6,
    samples_csv: str = "Drosophila_1,Drosophila_2",
    checkpoint: str = "",
    data_dir: str = "",
    spacing_xyz: str = "0.20312639,0.20312639,0.79099447",
    crop_shape_zyx: str = "32,192,192",
    confidence_ignore_margin_um: float = 1.0,
    partial_ignore_margin_um: float = 1.0,
    max_edges_per_class: int = 64,
    hard_negative_fraction: float = 0.50,
    hard_negative_threshold: float = 0.50,
    preservation_weight: float = 0.25,
    preserve_negative_threshold: float = 0.10,
    merge_threshold: float = 0.845,
    separator_threshold: float = 0.50,
    allowed_positive_accept_drop: float = 0.05,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    run_name: str = "drosophila_12_morphology_rag_multicrop_v1",
    resume: bool = False,
    seed: int = 230525,
):
    return _training_impl(
        max_steps=max_steps,
        lr=lr,
        checkpoint_every=checkpoint_every,
        validation_every=validation_every,
        validation_crops_per_sample=validation_crops_per_sample,
        samples_csv=samples_csv,
        checkpoint=checkpoint,
        data_dir=data_dir,
        spacing_xyz=spacing_xyz,
        crop_shape_zyx=crop_shape_zyx,
        confidence_ignore_margin_um=confidence_ignore_margin_um,
        partial_ignore_margin_um=partial_ignore_margin_um,
        max_edges_per_class=max_edges_per_class,
        hard_negative_fraction=hard_negative_fraction,
        hard_negative_threshold=hard_negative_threshold,
        preservation_weight=preservation_weight,
        preserve_negative_threshold=preserve_negative_threshold,
        merge_threshold=merge_threshold,
        separator_threshold=separator_threshold,
        allowed_positive_accept_drop=allowed_positive_accept_drop,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        run_name=run_name,
        resume=resume,
        seed=seed,
        execution_mode="modal",
    )


@_modal_local_entrypoint_decorator()
def main(
    max_steps: int = 600,
    lr: float = 1e-4,
    checkpoint_every: int = 50,
    validation_every: int = 50,
    validation_crops_per_sample: int = 6,
    samples_csv: str = "Drosophila_1,Drosophila_2",
    checkpoint: str = "",
    data_dir: str = "",
    spacing_xyz: str = "0.20312639,0.20312639,0.79099447",
    crop_shape_zyx: str = "32,192,192",
    confidence_ignore_margin_um: float = 1.0,
    partial_ignore_margin_um: float = 1.0,
    max_edges_per_class: int = 64,
    hard_negative_fraction: float = 0.50,
    hard_negative_threshold: float = 0.50,
    preservation_weight: float = 0.25,
    preserve_negative_threshold: float = 0.10,
    merge_threshold: float = 0.845,
    separator_threshold: float = 0.50,
    allowed_positive_accept_drop: float = 0.05,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    run_name: str = "drosophila_12_morphology_rag_multicrop_v1",
    resume: bool = False,
    seed: int = 230525,
):
    call = train_morphology_rag_multicrop.spawn(
        max_steps=max_steps,
        lr=lr,
        checkpoint_every=checkpoint_every,
        validation_every=validation_every,
        validation_crops_per_sample=validation_crops_per_sample,
        samples_csv=samples_csv,
        checkpoint=checkpoint,
        data_dir=data_dir,
        spacing_xyz=spacing_xyz,
        crop_shape_zyx=crop_shape_zyx,
        confidence_ignore_margin_um=confidence_ignore_margin_um,
        partial_ignore_margin_um=partial_ignore_margin_um,
        max_edges_per_class=max_edges_per_class,
        hard_negative_fraction=hard_negative_fraction,
        hard_negative_threshold=hard_negative_threshold,
        preservation_weight=preservation_weight,
        preserve_negative_threshold=preserve_negative_threshold,
        merge_threshold=merge_threshold,
        separator_threshold=separator_threshold,
        allowed_positive_accept_drop=allowed_positive_accept_drop,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        run_name=run_name,
        resume=resume,
        seed=seed,
    )
    print(
        f"[launcher] spawned durable multi-crop training call: {call.object_id}",
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
            "Train the morphology correction branch over diverse D1/D2 crops "
            "while keeping dense geometry and the legacy RAG frozen."
        )
    )
    parser.add_argument(
        "--execution",
        choices=("local", "modal"),
        default="local",
    )
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument(
        "--validation-crops-per-sample",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--samples",
        default="Drosophila_1,Drosophila_2",
    )
    parser.add_argument("--checkpoint", default="")
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
        "--max-edges-per-class",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--hard-negative-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--hard-negative-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--preservation-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--preserve-negative-threshold",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.845,
    )
    parser.add_argument(
        "--separator-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--allowed-positive-accept-drop",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--run-name",
        default="drosophila_12_morphology_rag_multicrop_v1",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=230525)
    args = parser.parse_args()

    kwargs = dict(
        max_steps=args.max_steps,
        lr=args.lr,
        checkpoint_every=args.checkpoint_every,
        validation_every=args.validation_every,
        validation_crops_per_sample=args.validation_crops_per_sample,
        samples_csv=args.samples,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        spacing_xyz=args.spacing_xyz,
        crop_shape_zyx=args.crop_shape_zyx,
        confidence_ignore_margin_um=args.confidence_ignore_margin_um,
        partial_ignore_margin_um=args.partial_ignore_margin_um,
        max_edges_per_class=args.max_edges_per_class,
        hard_negative_fraction=args.hard_negative_fraction,
        hard_negative_threshold=args.hard_negative_threshold,
        preservation_weight=args.preservation_weight,
        preserve_negative_threshold=args.preserve_negative_threshold,
        merge_threshold=args.merge_threshold,
        separator_threshold=args.separator_threshold,
        allowed_positive_accept_drop=args.allowed_positive_accept_drop,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        run_name=args.run_name,
        resume=args.resume,
        seed=args.seed,
    )

    if args.execution == "local":
        result = _training_impl(
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
        "--lr",
        str(args.lr),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--validation-every",
        str(args.validation_every),
        "--validation-crops-per-sample",
        str(args.validation_crops_per_sample),
        "--samples-csv",
        args.samples,
        "--spacing-xyz",
        args.spacing_xyz,
        "--crop-shape-zyx",
        args.crop_shape_zyx,
        "--confidence-ignore-margin-um",
        str(args.confidence_ignore_margin_um),
        "--partial-ignore-margin-um",
        str(args.partial_ignore_margin_um),
        "--max-edges-per-class",
        str(args.max_edges_per_class),
        "--hard-negative-fraction",
        str(args.hard_negative_fraction),
        "--hard-negative-threshold",
        str(args.hard_negative_threshold),
        "--preservation-weight",
        str(args.preservation_weight),
        "--preserve-negative-threshold",
        str(args.preserve_negative_threshold),
        "--merge-threshold",
        str(args.merge_threshold),
        "--separator-threshold",
        str(args.separator_threshold),
        "--allowed-positive-accept-drop",
        str(args.allowed_positive_accept_drop),
        "--weight-decay",
        str(args.weight_decay),
        "--max-grad-norm",
        str(args.max_grad_norm),
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
