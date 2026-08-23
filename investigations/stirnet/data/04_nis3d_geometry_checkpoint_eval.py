from __future__ import annotations

"""
STIR-Net investigation 04 — evaluate a geometry-bootstrap NIS3D checkpoint locally.

Purpose
-------
This script evaluates a trained STIR-Net geometry checkpoint before any RAG /
spatial-partition training is trusted.  It is intentionally crop-based so the
same production 32x192x192 geometry path can run on a local 6 GiB GPU.

Default comparison:
    Zebrafish_1  -> unseen validation volume
    Zebrafish_2  -> training-reference volume

The evaluation uses the same production ingredients as training:
* strict NIS3D physical spacing from Info.txt;
* ConfidenceScore==1 exclusion and the same physical ignore margin;
* raw-source preprocessing and five STIR-Net input channels;
* merge-aware crop planning;
* no synthetic source dropout during evaluation;
* halo-aware static geometry targets;
* source-conditioned corrective separator targets;
* the checkpoint's model/loss/crop configuration;
* geometry-only forward execution (no RAG, temporal, or refinement).

Outputs
-------
For every selected crop the script writes:
* exact scalar metrics to metrics.csv / metrics.jsonl;
* a compressed .npz with raw/source/GT, predictions, targets and vector fields;
* a PNG mid-plane geometry montage.

It also writes summary.json and selected_crops.json.  With --napari it opens one
saved crop with image/label layers plus sparse flow and centroid-offset vectors.

Typical command
---------------
From the repository root:

    python investigations/stirnet/data/04_nis3d_geometry_checkpoint_eval.py

Useful overrides:

    python investigations/stirnet/data/04_nis3d_geometry_checkpoint_eval.py ^
        --checkpoint runs/stirnet/training/01_nis3d_spatial_training/recovery/nis3d_zebrafish_spatial_v1/checkpoint_step_000300.pt ^
        --samples Zebrafish_1,Zebrafish_2 ^
        --crops-per-sample 6

Open the first evaluated validation crop in Napari after evaluation:

    python investigations/stirnet/data/04_nis3d_geometry_checkpoint_eval.py --napari
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


# ======================================================================================
# Repository paths
# ======================================================================================

SCRIPT_NAME = "04_nis3d_geometry_checkpoint_eval_memorysafe_v3"
DEFAULT_CHECKPOINT_REL = Path(
    "runs/stirnet/training/01_nis3d_spatial_training/recovery/"
    "nis3d_zebrafish_spatial_v1/checkpoint_step_000300.pt"
)
DEFAULT_SAMPLES = ("Zebrafish_1", "Zebrafish_2")
DEFAULT_CONFIDENCE_IGNORE_MARGIN_UM = 1.0
LARGE_VOLUME_PROXY_VOXELS = 100_000_000
DEFAULT_LARGE_VOLUME_CONTEXT_UM = 24.0


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "learned").is_dir() and (candidate / "src").is_dir():
            return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "learned").is_dir() and (cwd / "src").is_dir():
        return cwd
    raise RuntimeError(
        "Could not resolve repository root. Run this script from the cell-tracking "
        "repository or place it under investigations/stirnet/data/."
    )


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCAL_NIS3D_ROOT_CANDIDATES = (
    REPO_ROOT / "data" / "external" / "NIS3D" / "NIS3D",
    REPO_ROOT / "data" / "external" / "NIS3D",
    REPO_ROOT / "data" / "NIS3D" / "NIS3D",
    REPO_ROOT / "data" / "NIS3D",
)


# ======================================================================================
# Small utilities
# ======================================================================================


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True))
        handle.write("\n")


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _load_torch(path: Path, *, map_location: str | torch.device = "cpu") -> dict:
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def _hydrate_dataclass(instance: Any, payload: dict[str, Any], *, path: str) -> Any:
    """Apply a checkpoint's nested config dictionary to current dataclass instances."""
    if not is_dataclass(instance):
        raise TypeError(f"{path} is not a dataclass instance")
    for key, value in payload.items():
        if not hasattr(instance, key):
            raise KeyError(
                f"Checkpoint config contains unknown field {path}.{key}. "
                "The checkpoint and current repository are not configuration-compatible."
            )
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _hydrate_dataclass(current, value, path=f"{path}.{key}")
        elif isinstance(current, tuple) and isinstance(value, (tuple, list)):
            setattr(instance, key, tuple(value))
        else:
            setattr(instance, key, value)
    return instance


def _parse_samples(text: str) -> tuple[str, ...]:
    rows = tuple(token.strip() for token in text.split(",") if token.strip())
    if not rows:
        raise ValueError("At least one sample is required")
    return rows


# ======================================================================================
# NIS3D loading / confidence semantics
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
                    value = float(match.group(1))
                    if value > 0:
                        values[axis] = value
                        break

    compact = " ".join(lines)
    if len(values) < 3:
        for axis in "xyz":
            if axis in values:
                continue
            match = re.search(rf"(?i)\b{axis}\s*[:=]\s*({_FLOAT})", compact)
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
            "Could not strictly parse X/Y/Z physical voxel spacing from Info.txt. "
            "Refusing to evaluate with invented spacing.\n\n"
            f"Info.txt contents:\n{info_text}"
        )
    return (values["z"], values["y"], values["x"])


def _first_existing(sample_dir: Path, names: Iterable[str], *, label: str) -> Path:
    for name in names:
        path = sample_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"{sample_dir.name}: could not find {label}. Tried: {', '.join(names)}"
    )


def _resolve_sample_files(sample_dir: Path) -> dict[str, Path]:
    # Keep aliases because released NIS3D folders are not fully name-consistent
    # (notably MusMusculus_2).
    return {
        "raw": _first_existing(
            sample_dir,
            ("data.tif", "Data.tif"),
            label="raw TIFF",
        ),
        "gt": _first_existing(
            sample_dir,
            ("GroundTruth.tif", "groundTruth.tif", "gt.tif", "GT.tif"),
            label="ground-truth TIFF",
        ),
        "confidence": _first_existing(
            sample_dir,
            (
                "ConfidenceScore.tif",
                "confidenceScore.tif",
                "scoreOfConfidence.tif",
                "ScoreOfConfidence.tif",
            ),
            label="confidence TIFF",
        ),
        "info": _first_existing(
            sample_dir,
            ("Info.txt", "info.txt"),
            label="Info.txt",
        ),
    }


def _read_tiff(path: Path, *, sample: str, label: str):
    import tifffile

    try:
        array = tifffile.memmap(path)
        mode = "memmap"
    except ValueError as exc:
        print(
            f"[data] {sample}/{path.name}: not directly memory-mappable "
            f"({exc}); decoding to a disk-backed memmap.",
            flush=True,
        )
        try:
            array = tifffile.imread(path, out="memmap")
            mode = "imread-memmap"
        except Exception as memmap_exc:
            print(
                f"[data] {sample}/{path.name}: disk-backed decode failed "
                f"({memmap_exc}); decoding into RAM.",
                flush=True,
            )
            array = tifffile.imread(path)
            mode = "imread"
    array = np.asarray(array)
    print(
        f"[data] {sample}/{label}: load={mode} shape={tuple(array.shape)} "
        f"dtype={array.dtype} RAM={array.nbytes / 2**20:.1f} MiB",
        flush=True,
    )
    return array


def _load_nis3d_arrays(sample_dir: Path):
    files = _resolve_sample_files(sample_dir)
    raw = _read_tiff(files["raw"], sample=sample_dir.name, label="raw")
    gt = _read_tiff(files["gt"], sample=sample_dir.name, label="gt")
    confidence = _read_tiff(
        files["confidence"],
        sample=sample_dir.name,
        label="confidence",
    )

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

    info_text = files["info"].read_text(encoding="utf-8", errors="replace")
    spacing = _parse_spacing_xyz_um(info_text)
    return raw, gt, confidence, spacing, info_text, files


def _physical_valid_erosion(
    valid: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
    margin_um: float,
) -> np.ndarray:
    """Equivalent small-radius confidence margin without a full-volume EDT.

    Training 01 keeps voxels whose distance to invalid supervision is strictly
    larger than ``margin_um``.  For the small NIS3D margin this is exactly a
    binary erosion by a physical ball, but avoids SciPy EDT's O(3*N) float64
    feature-index workspace.
    """
    from scipy import ndimage as ndi

    if margin_um <= 0:
        return np.asarray(valid, dtype=bool)
    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    radius = np.ceil(float(margin_um) / spacing).astype(np.int64)
    zz, yy, xx = np.ogrid[
        -radius[0] : radius[0] + 1,
        -radius[1] : radius[1] + 1,
        -radius[2] : radius[2] + 1,
    ]
    structure = (
        (zz * spacing[0]) ** 2
        + (yy * spacing[1]) ** 2
        + (xx * spacing[2]) ** 2
        <= float(margin_um) ** 2 + 1e-12
    )
    return ndi.binary_erosion(
        np.asarray(valid, dtype=bool),
        structure=structure,
        border_value=0,
    )


def _prepare_nis3d_gt(
    gt,
    confidence,
    spacing_zyx_um: tuple[float, float, float],
    *,
    ignore_margin_um: float,
):
    """Match Training 01 confidence handling exactly."""
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
        valid = _physical_valid_erosion(
            valid, spacing_zyx_um, float(ignore_margin_um)
        )

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


def _discover_nis3d_root(samples: tuple[str, ...], override: Path | None) -> Path:
    candidates = (
        (override,)
        if override is not None
        else LOCAL_NIS3D_ROOT_CANDIDATES
    )
    for root in candidates:
        root = Path(root)
        if root.is_dir() and all((root / sample).is_dir() for sample in samples):
            return root

    first = samples[0]
    search_root = REPO_ROOT / "data"
    if search_root.exists():
        for match in list(search_root.glob(f"**/{first}"))[:64]:
            parent = match.parent
            if all((parent / sample).is_dir() for sample in samples):
                return parent

    checked = "\n".join(f"  - {Path(path)}" for path in candidates)
    raise FileNotFoundError(
        "Could not locate requested NIS3D samples.\n"
        f"Samples: {samples}\nChecked:\n{checked}"
    )



@dataclass(frozen=True)
class _ProxyRecord:
    batch_index: int
    slices_zyx: tuple[slice, slice, slice]
    global_slices_zyx: tuple[slice, slice, slice]
    candidate_type: str
    complete_cell_ids: tuple[int, ...] = ()
    partial_cell_ids: tuple[int, ...] = ()
    true_boundary_cell_ids: tuple[int, ...] = ()
    merge_source_ids: tuple[int, ...] = ()
    merge_gt_ids: tuple[int, ...] = ()


def _scan_large_nis3d_metadata(gt, confidence, *, chunk_z: int = 4):
    """Scan a huge volume in bounded Z chunks without materializing masks."""
    gt_ids: set[int] = set()
    touched_ids: set[int] = set()
    confidence_values: set[int] = set()
    confidence1_count = 0
    total = int(np.prod(gt.shape))
    for z0 in range(0, gt.shape[0], chunk_z):
        z1 = min(gt.shape[0], z0 + chunk_z)
        gt_chunk = np.asarray(gt[z0:z1])
        conf_chunk = np.asarray(confidence[z0:z1])
        ids = np.unique(gt_chunk)
        gt_ids.update(int(v) for v in ids if int(v) > 0)
        confidence_values.update(int(v) for v in np.unique(conf_chunk))
        invalid = conf_chunk == 1
        confidence1_count += int(invalid.sum())
        if bool(invalid.any()):
            touched = np.unique(gt_chunk[invalid])
            touched_ids.update(int(v) for v in touched if int(v) > 0)
    kept_ids = gt_ids - touched_ids
    return {
        "gt_ids": gt_ids,
        "touched_ids": touched_ids,
        "kept_ids": kept_ids,
        "gt_ids_original": len(gt_ids),
        "gt_ids_removed_touching_confidence1": len(touched_ids),
        "gt_ids_kept": len(kept_ids),
        "confidence1_fraction": confidence1_count / max(total, 1),
        "confidence_values": sorted(confidence_values),
        "removed_gt_ids": sorted(touched_ids),
    }


def _clamped_crop_slices(
    center_zyx: np.ndarray,
    full_shape: tuple[int, int, int],
    crop_shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    full = np.asarray(full_shape, dtype=np.int64)
    size = np.minimum(np.asarray(crop_shape, dtype=np.int64), full)
    center = np.asarray(center_zyx, dtype=np.int64)
    start = center - size // 2
    start = np.minimum(np.maximum(start, 0), full - size)
    return tuple(
        slice(int(lo), int(lo + extent)) for lo, extent in zip(start, size)
    )


def _proxy_candidate_global_slices(
    gt,
    confidence,
    *,
    touched_ids: set[int],
    crop_shape: tuple[int, int, int],
    total: int,
    merge_fraction: float,
) -> list[tuple[tuple[slice, slice, slice], str]]:
    """Choose diverse GT-rich crops from a tiny coarse view of a huge frame."""
    from scipy import ndimage as ndi

    step = np.asarray((4, 16, 16), dtype=np.int64)
    sampled_gt = np.asarray(gt[:: step[0], :: step[1], :: step[2]])
    sampled_conf = np.asarray(confidence[:: step[0], :: step[1], :: step[2]])
    foreground = sampled_gt > 0
    if touched_ids:
        foreground &= ~np.isin(
            sampled_gt,
            np.asarray(sorted(touched_ids), dtype=sampled_gt.dtype),
        )
    foreground &= sampled_conf != 1
    if not bool(foreground.any()):
        raise RuntimeError("No valid foreground found in large-volume coarse scan")

    coarse_window = np.maximum(
        1, np.ceil(np.asarray(crop_shape) / step).astype(np.int64)
    )
    density = ndi.uniform_filter(
        foreground.astype(np.float32),
        size=tuple(int(v) for v in coarse_window),
        mode="constant",
    )
    points = np.argwhere(foreground)
    scores = density[tuple(points.T)]
    if len(points) > 4096:
        order = np.argsort(scores)[::-1][:4096]
        points = points[order]
        scores = scores[order]

    order_dense = np.argsort(scores)[::-1]
    order_sparse = np.argsort(scores)
    desired_merge = int(round(total * merge_fraction))
    desired_merge = max(0, min(total, desired_merge))
    desired_coverage = total - desired_merge
    selected: list[tuple[tuple[slice, slice, slice], str]] = []
    selected_centers: list[np.ndarray] = []
    full_shape = tuple(int(v) for v in gt.shape)
    crop = np.asarray(crop_shape, dtype=np.float64)

    def add_from(order, wanted: int, label: str) -> None:
        for idx in order.tolist():
            if sum(kind == label for _, kind in selected) >= wanted:
                return
            center = points[int(idx)].astype(np.int64) * step
            if selected_centers:
                normalized = [
                    np.max(np.abs(center - previous) / np.maximum(crop, 1.0))
                    for previous in selected_centers
                ]
                if min(normalized) < 0.75:
                    continue
            slices = _clamped_crop_slices(center, full_shape, crop_shape)
            selected.append((slices, label))
            selected_centers.append(center)

    add_from(order_dense, desired_merge, "proxy_dense")
    add_from(order_sparse, desired_coverage, "proxy_coverage")
    # Fill any shortage with the dense ordering while retaining separation.
    if len(selected) < total:
        for idx in order_dense.tolist():
            if len(selected) >= total:
                break
            center = points[int(idx)].astype(np.int64) * step
            slices = _clamped_crop_slices(center, full_shape, crop_shape)
            key = tuple((s.start, s.stop) for s in slices)
            if any(tuple((s.start, s.stop) for s in old) == key for old, _ in selected):
                continue
            selected.append((slices, "proxy_fill"))
    return selected[:total]


def _expanded_global_context(
    core: tuple[slice, slice, slice],
    full_shape: tuple[int, int, int],
    spacing_zyx_um: tuple[float, float, float],
    context_um: float,
):
    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    radius = np.ceil(float(context_um) / spacing).astype(np.int64)
    context = []
    core_relative = []
    for axis in range(3):
        lo = max(0, int(core[axis].start) - int(radius[axis]))
        hi = min(int(full_shape[axis]), int(core[axis].stop) + int(radius[axis]))
        context.append(slice(lo, hi))
        core_relative.append(
            slice(int(core[axis].start) - lo, int(core[axis].stop) - lo)
        )
    return tuple(context), tuple(core_relative)


def _classify_proxy_cells_and_merges(
    clean_gt: np.ndarray,
    current: np.ndarray,
    core: tuple[slice, slice, slice],
    *,
    min_overlap_voxels: int,
    min_gt_fraction: float,
):
    core_gt = clean_gt[core]
    core_current = current[core]
    core_ids = np.unique(core_gt)
    core_ids = core_ids[core_ids > 0]
    complete: list[int] = []
    partial: list[int] = []
    for cell_id in core_ids.tolist():
        whole = clean_gt == int(cell_id)
        inside = int((core_gt == int(cell_id)).sum())
        total = int(whole.sum())
        if total > 0 and inside == total:
            complete.append(int(cell_id))
        else:
            partial.append(int(cell_id))

    merge_sources: list[int] = []
    merge_gt_ids: set[int] = set()
    source_ids = np.unique(core_current)
    for source_id in source_ids[source_ids > 0].tolist():
        source_mask = core_current == int(source_id)
        overlap_ids, overlap_counts = np.unique(
            core_gt[source_mask], return_counts=True
        )
        useful: list[int] = []
        for gt_id, count in zip(overlap_ids.tolist(), overlap_counts.tolist()):
            if int(gt_id) <= 0 or int(count) < min_overlap_voxels:
                continue
            gt_count = int((core_gt == int(gt_id)).sum())
            if gt_count and int(count) / gt_count >= min_gt_fraction:
                useful.append(int(gt_id))
        if len(useful) >= 2:
            merge_sources.append(int(source_id))
            merge_gt_ids.update(useful)
    return (
        tuple(sorted(complete)),
        tuple(sorted(partial)),
        tuple(sorted(merge_sources)),
        tuple(sorted(merge_gt_ids)),
    )


def _prepare_large_proxy_batch(
    *,
    sample: str,
    raw,
    gt,
    confidence,
    spacing: tuple[float, float, float],
    global_core: tuple[slice, slice, slice],
    touched_ids: set[int],
    train_cfg,
    confidence_ignore_margin_um: float,
    context_um: float,
):
    from learned.stirnet.training.raw_source import prepare_raw_training_batch

    context, core_relative = _expanded_global_context(
        global_core,
        tuple(int(v) for v in gt.shape),
        spacing,
        context_um,
    )
    raw_local = np.asarray(raw[context])
    gt_local = np.asarray(gt[context])
    conf_local = np.asarray(confidence[context])

    valid = conf_local != 1
    clean_gt = np.asarray(gt_local, dtype=np.int32).copy()
    if touched_ids:
        touched_local = np.isin(
            clean_gt,
            np.asarray(sorted(touched_ids), dtype=clean_gt.dtype),
        )
        valid[touched_local] = False
        clean_gt[touched_local] = 0
    clean_gt[~valid] = 0
    if confidence_ignore_margin_um > 0:
        valid = _physical_valid_erosion(
            valid, spacing, confidence_ignore_margin_um
        )

    origin = tuple(int(s.start) for s in context)
    sid = "NIS3D/{}/proxy/z{}_y{}_x{}".format(sample, *origin)
    batch = prepare_raw_training_batch(
        raw_local,
        clean_gt,
        spacing,
        source_id=sid,
        source_cache_path=None,
    )
    batch["supervision_valid_mask"] = torch.from_numpy(valid)[None]
    batch["nis3d_sample_name"] = sample
    current = batch["instance_labels"][0].detach().cpu().numpy()
    cfg = train_cfg.curriculum
    complete, partial, merge_sources, merge_gt_ids = _classify_proxy_cells_and_merges(
        clean_gt,
        current,
        core_relative,
        min_overlap_voxels=int(cfg.refinement_crop_merge_min_overlap_voxels),
        min_gt_fraction=float(cfg.refinement_crop_merge_min_gt_fraction),
    )
    record = _ProxyRecord(
        batch_index=0,
        slices_zyx=core_relative,
        global_slices_zyx=global_core,
        candidate_type="merge" if merge_sources else "coverage",
        complete_cell_ids=complete,
        partial_cell_ids=partial,
        merge_source_ids=merge_sources,
        merge_gt_ids=merge_gt_ids,
    )
    return batch, record


# ======================================================================================
# Checkpoint / model
# ======================================================================================


def _load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    from learned.stirnet import StirNet
    from learned.stirnet.model.config import ModelConfig
    from learned.stirnet.training.checkpoint import load_checkpoint
    from learned.stirnet.training.config import TrainingConfig

    checkpoint = _load_torch(checkpoint_path, map_location="cpu")

    model_cfg = ModelConfig()
    checkpoint_model_cfg = checkpoint.get("model_config")
    if not isinstance(checkpoint_model_cfg, dict):
        raise ValueError("Checkpoint does not contain a serialized model_config")
    _hydrate_dataclass(model_cfg, checkpoint_model_cfg, path="model_config")
    model_cfg.validate()

    train_cfg = TrainingConfig()
    checkpoint_train_cfg = checkpoint.get("training_config")
    if isinstance(checkpoint_train_cfg, dict):
        _hydrate_dataclass(train_cfg, checkpoint_train_cfg, path="training_config")
        # The checkpoint contains the original Modal cache path. Evaluation uses
        # a local cache, so only semantic fields from this config are consumed.
        train_cfg.crop_static_target_cache_dir = None
        train_cfg.validate()
    else:
        print(
            "[checkpoint] training_config missing; using current defaults for "
            "evaluation-only target/crop settings.",
            flush=True,
        )

    model = StirNet(model_cfg)
    load_checkpoint(
        checkpoint_path,
        model,
        map_location="cpu",
        strict=True,
    )
    model.to(device)
    model.eval()

    return checkpoint, model, model_cfg, train_cfg


def _amp_context(
    device: torch.device,
    requested_dtype: str,
):
    if device.type != "cuda" or requested_dtype == "fp32":
        return nullcontext(), "fp32"

    if requested_dtype == "bf16" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16), "bf16"

    if requested_dtype in {"bf16", "fp16"}:
        return torch.autocast("cuda", dtype=torch.float16), "fp16"

    return nullcontext(), "fp32"


# ======================================================================================
# Source batch / crop selection
# ======================================================================================


def _prepare_source_batch(
    nis3d_root: Path,
    sample: str,
    *,
    source_cache_root: Path,
    confidence_ignore_margin_um: float,
):
    from learned.stirnet.training.raw_source import prepare_raw_training_batch

    sample_dir = nis3d_root / sample
    raw, gt, confidence, spacing, info_text, files = _load_nis3d_arrays(sample_dir)
    clean_gt, valid_mask, confidence_report = _prepare_nis3d_gt(
        gt,
        confidence,
        spacing,
        ignore_margin_um=confidence_ignore_margin_um,
    )

    source_cache = source_cache_root / f"{sample}.pt"
    started = time.perf_counter()
    batch = prepare_raw_training_batch(
        raw,
        clean_gt,
        spacing,
        source_id=f"NIS3D/{sample}",
        source_cache_path=source_cache,
    )
    prepare_seconds = time.perf_counter() - started

    batch["supervision_valid_mask"] = torch.from_numpy(valid_mask)[None]
    batch["nis3d_sample_name"] = sample

    metadata = dict(batch["source_preprocessing_metadata"][0])
    report = {
        "sample": sample,
        "sample_dir": str(sample_dir),
        "files": {k: str(v) for k, v in files.items()},
        "shape_zyx": [int(v) for v in raw.shape],
        "raw_dtype": str(raw.dtype),
        "gt_dtype": str(gt.dtype),
        "confidence_dtype": str(confidence.dtype),
        "spacing_zyx_um": [float(v) for v in spacing],
        "source_prepare_seconds": float(prepare_seconds),
        "source_cache_hit": bool(metadata.get("source_cache_hit", False)),
        "source_current_instance_count": int(
            metadata.get("source_current_instance_count", 0)
        ),
        "model_dref_um": float(metadata.get("model_dref_um", batch["dref_um"][0])),
        **confidence_report,
    }
    return batch, report


def _record_key(record) -> tuple[int, int, int]:
    return tuple(int(axis.start) for axis in record.slices_zyx)


def _evenly_spaced(rows: list[Any], count: int) -> list[Any]:
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return rows.copy()
    indices = np.linspace(0, len(rows) - 1, count)
    indices = np.rint(indices).astype(np.int64)
    # np.rint can theoretically duplicate with tiny lists; preserve requested
    # cardinality by filling deterministically from unused rows.
    chosen_indices: list[int] = []
    used: set[int] = set()
    for index in indices.tolist():
        index = int(index)
        if index not in used:
            chosen_indices.append(index)
            used.add(index)
    for index in range(len(rows)):
        if len(chosen_indices) >= count:
            break
        if index not in used:
            chosen_indices.append(index)
            used.add(index)
    return [rows[index] for index in chosen_indices[:count]]


def _select_eval_records(
    manifest,
    *,
    total: int,
    merge_fraction: float,
) -> list[Any]:
    if total < 1:
        raise ValueError("crops-per-sample must be positive")
    rows = list(manifest.records[0])
    if not rows:
        raise RuntimeError("merge-aware crop manifest is empty")

    merge_rows = sorted(
        (row for row in rows if row.merge_source_ids),
        key=_record_key,
    )
    coverage_rows = sorted(
        (row for row in rows if not row.merge_source_ids),
        key=_record_key,
    )

    target_merge = int(round(total * merge_fraction))
    target_merge = min(target_merge, total)
    target_coverage = total - target_merge

    chosen = [
        *_evenly_spaced(merge_rows, target_merge),
        *_evenly_spaced(coverage_rows, target_coverage),
    ]

    # Fill shortages from the full manifest without duplicates.
    used = {
        tuple((int(s.start), int(s.stop)) for s in row.slices_zyx)
        for row in chosen
    }
    for row in sorted(rows, key=_record_key):
        if len(chosen) >= total:
            break
        key = tuple((int(s.start), int(s.stop)) for s in row.slices_zyx)
        if key not in used:
            chosen.append(row)
            used.add(key)

    return chosen[:total]


def _record_to_spec(record, *, full_shape: tuple[int, int, int], spacing: torch.Tensor):
    from learned.stirnet.training.crops import CropSpec

    lower = torch.tensor(
        [int(axis.start) for axis in record.slices_zyx],
        dtype=torch.float32,
    )
    size = torch.tensor(
        [int(axis.stop) - int(axis.start) for axis in record.slices_zyx],
        dtype=torch.float32,
    )
    shift = (
        lower
        + 0.5 * (size - 1)
        - 0.5 * (torch.tensor(full_shape).float() - 1)
    ) * spacing.detach().cpu().float()

    return CropSpec(
        batch_index=int(record.batch_index),
        slices_zyx=record.slices_zyx,
        full_shape_zyx=full_shape,
        center_shift_um=shift,
        candidate_type=record.candidate_type,
        complete_cell_ids=record.complete_cell_ids,
        partial_cell_ids=record.partial_cell_ids,
        true_boundary_cell_ids=record.true_boundary_cell_ids,
        merge_source_ids=record.merge_source_ids,
    )


def _build_eval_manifest(source_batch: dict, train_cfg):
    from learned.stirnet.training.merge_aware_crops import (
        build_merge_aware_crop_manifest,
    )

    cfg = train_cfg.curriculum
    return build_merge_aware_crop_manifest(
        source_batch["gt_labels"],
        current_labels=source_batch.get("instance_labels"),
        spacing_um=source_batch["spacing_um"],
        crop_shape_zyx=tuple(cfg.refinement_crop_shape_zyx),
        min_complete_cells=int(cfg.refinement_crop_min_complete_cells),
        preferred_complete_cells=int(cfg.refinement_crop_preferred_complete_cells),
        views_per_cell=int(cfg.refinement_crop_views_per_cell),
        context_um=float(cfg.refinement_crop_context_um),
        merge_min_overlap_voxels=int(
            cfg.refinement_crop_merge_min_overlap_voxels
        ),
        merge_min_gt_fraction=float(
            cfg.refinement_crop_merge_min_gt_fraction
        ),
    )


def _prepare_eval_crop(
    source_batch: dict,
    spec,
    *,
    train_cfg,
):
    """Match the training crop/source path but force synthetic dropout off."""
    from learned.stirnet.training.crops import CropBatch, prepare_crop_batch
    from learned.stirnet.training.raw_source import materialize_raw_source_crop_batch

    cfg = train_cfg.curriculum
    crop = prepare_crop_batch(
        source_batch,
        source_batch["gt_labels"],
        [spec],
        partial_ignore_margin_um=float(
            cfg.refinement_crop_partial_ignore_margin_um
        ),
    )

    # Training 01 applies this as an experiment-local wrapper around
    # prepare_crop_batch. Reproduce the exact AND here without importing the
    # Modal training script.
    full_valid = torch.as_tensor(
        source_batch["supervision_valid_mask"]
    ).detach().cpu().bool()
    dataset_valid = torch.stack(
        [
            full_valid[row_spec.batch_index][row_spec.slices_zyx]
            for row_spec in crop.specs
        ]
    )
    cropped_batch = dict(crop.batch)
    existing = torch.as_tensor(
        cropped_batch["supervision_valid_mask"]
    ).detach().cpu().bool()
    cropped_batch["supervision_valid_mask"] = existing & dataset_valid
    crop = CropBatch(
        batch=cropped_batch,
        gt_labels=crop.gt_labels,
        geometry_targets=crop.geometry_targets,
        specs=crop.specs,
    )

    # Evaluation measures the natural preprocessing/source state. Do not
    # synthetically delete cells here.
    crop = materialize_raw_source_crop_batch(
        source_batch,
        crop,
        source_halo_um=float(cfg.refinement_crop_source_halo_um),
        dropout_probability=0.0,
        dropout_max_instances=0,
        dropout_seed=0,
        dropout_min_purity=float(
            cfg.refinement_crop_source_dropout_min_purity
        ),
        dropout_min_gt_coverage=float(
            cfg.refinement_crop_source_dropout_min_gt_coverage
        ),
    )
    return crop


# ======================================================================================
# Metrics
# ======================================================================================


def _masked_values(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(tensor)
    mask = torch.as_tensor(mask).bool()
    if mask.ndim == tensor.ndim - 1:
        mask = mask[:, None]
    if mask.shape[1] == 1 and tensor.shape[1] != 1:
        mask = mask.expand(
            tensor.shape[0],
            tensor.shape[1],
            *tensor.shape[-3:],
        )
    return tensor[mask]


def _binary_metrics(
    pred: torch.Tensor,
    truth: torch.Tensor,
    *,
    valid: torch.Tensor,
) -> dict[str, float]:
    pred = torch.as_tensor(pred).bool()
    truth = torch.as_tensor(truth).bool()
    valid = torch.as_tensor(valid).bool()
    if valid.ndim == pred.ndim - 1:
        valid = valid[:, None]
    valid = valid.expand_as(pred)

    pred = pred[valid]
    truth = truth[valid]
    if pred.numel() == 0:
        return {
            "dice": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "accuracy": float("nan"),
        }

    tp = int((pred & truth).sum().item())
    fp = int((pred & ~truth).sum().item())
    fn = int((~pred & truth).sum().item())
    tn = int((~pred & ~truth).sum().item())

    dice_den = 2 * tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn
    total = tp + fp + fn + tn
    return {
        "dice": 1.0 if dice_den == 0 else (2.0 * tp / dice_den),
        "precision": 1.0 if precision_den == 0 else (tp / precision_den),
        "recall": 1.0 if recall_den == 0 else (tp / recall_den),
        "accuracy": (tp + tn) / max(total, 1),
    }


def _masked_mae(
    pred: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
) -> float:
    values = _masked_values(
        (torch.as_tensor(pred).float() - torch.as_tensor(truth).float()).abs(),
        valid,
    )
    return float(values.mean().item()) if values.numel() else float("nan")


def _vector_metrics(
    pred: torch.Tensor,
    truth: torch.Tensor,
    valid_foreground: torch.Tensor,
) -> dict[str, float]:
    pred = torch.as_tensor(pred).float()
    truth = torch.as_tensor(truth).float()
    valid = torch.as_tensor(valid_foreground).bool()
    if valid.ndim == 4:
        valid = valid[:, None]

    pred_unit = F.normalize(pred, dim=1, eps=1e-6)
    truth_unit = F.normalize(truth, dim=1, eps=1e-6)
    cosine = (pred_unit * truth_unit).sum(dim=1, keepdim=True)

    epe = torch.linalg.vector_norm(pred - truth, dim=1, keepdim=True)
    valid_scalar = valid.expand_as(cosine)

    cosine_values = cosine[valid_scalar]
    epe_values = epe[valid_scalar]

    return {
        "cosine": (
            float(cosine_values.mean().item())
            if cosine_values.numel()
            else float("nan")
        ),
        "epe": (
            float(epe_values.mean().item())
            if epe_values.numel()
            else float("nan")
        ),
    }


def _sdf_metrics(
    pred: torch.Tensor,
    truth: torch.Tensor,
    sdf_valid: torch.Tensor,
    supervision_valid: torch.Tensor,
) -> dict[str, float]:
    mask = torch.as_tensor(sdf_valid).bool()
    supervision = torch.as_tensor(supervision_valid).bool()
    if supervision.ndim == 4:
        supervision = supervision[:, None]
    mask = mask & supervision.expand_as(mask)

    diff = (
        torch.as_tensor(pred).float() - torch.as_tensor(truth).float()
    )[mask]
    if diff.numel() == 0:
        return {"mae": float("nan"), "rmse": float("nan")}
    return {
        "mae": float(diff.abs().mean().item()),
        "rmse": float(diff.square().mean().sqrt().item()),
    }


def _missing_cell_metrics(
    gt_labels: torch.Tensor,
    current_labels: torch.Tensor,
    pred_foreground: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    cell_ids: Iterable[int],
    source_missing_threshold: float = 0.50,
    source_absent_threshold: float = 0.10,
    pred_recovery_threshold: float = 0.50,
) -> dict[str, float | int]:
    gt = torch.as_tensor(gt_labels)[0].long()
    current = torch.as_tensor(current_labels)[0]
    pred = torch.as_tensor(pred_foreground)[0, 0].bool()
    valid = torch.as_tensor(valid_mask)[0].bool()

    coverages: list[float] = []
    pred_recalls: list[float] = []
    missing_pred_recalls: list[float] = []
    absent_pred_recalls: list[float] = []

    for cell_id in sorted(set(int(v) for v in cell_ids)):
        cell = (gt == cell_id) & valid
        voxels = int(cell.sum().item())
        if voxels == 0:
            continue
        source_coverage = float((current[cell] > 0).float().mean().item())
        pred_recall = float(pred[cell].float().mean().item())
        coverages.append(source_coverage)
        pred_recalls.append(pred_recall)
        if source_coverage < source_missing_threshold:
            missing_pred_recalls.append(pred_recall)
        if source_coverage < source_absent_threshold:
            absent_pred_recalls.append(pred_recall)

    recovered_missing = sum(
        value >= pred_recovery_threshold for value in missing_pred_recalls
    )
    recovered_absent = sum(
        value >= pred_recovery_threshold for value in absent_pred_recalls
    )

    return {
        "complete_cell_count_evaluated": len(coverages),
        "source_missing_cell_count": len(missing_pred_recalls),
        "source_absent_cell_count": len(absent_pred_recalls),
        "mean_source_coverage_complete_cells": (
            float(np.mean(coverages)) if coverages else float("nan")
        ),
        "mean_pred_recall_complete_cells": (
            float(np.mean(pred_recalls)) if pred_recalls else float("nan")
        ),
        "mean_pred_recall_source_missing_cells": (
            float(np.mean(missing_pred_recalls))
            if missing_pred_recalls
            else float("nan")
        ),
        "mean_pred_recall_source_absent_cells": (
            float(np.mean(absent_pred_recalls))
            if absent_pred_recalls
            else float("nan")
        ),
        "recovered_source_missing_cell_fraction": (
            recovered_missing / len(missing_pred_recalls)
            if missing_pred_recalls
            else float("nan")
        ),
        "recovered_source_absent_cell_fraction": (
            recovered_absent / len(absent_pred_recalls)
            if absent_pred_recalls
            else float("nan")
        ),
    }


def _compute_metrics(
    *,
    output,
    targets,
    crop,
    record,
    model_cfg,
    criterion_metrics: dict[str, torch.Tensor],
) -> dict[str, Any]:
    geometry = output.geometry
    probs = geometry.probabilities()

    valid = crop.batch["supervision_valid_mask"].detach().cpu().bool()
    gt_fg = targets.foreground.detach().cpu() > 0.5
    current_fg = crop.batch["instance_labels"].detach().cpu()[:, None] > 0

    foreground_threshold = float(model_cfg.partition.foreground_threshold)
    pred_fg_partition = (
        probs["foreground"].detach().float().cpu() >= foreground_threshold
    )
    pred_fg_half = probs["foreground"].detach().float().cpu() >= 0.5

    source_fg_metrics = _binary_metrics(
        current_fg,
        gt_fg,
        valid=valid,
    )
    pred_fg_metrics = _binary_metrics(
        pred_fg_partition,
        gt_fg,
        valid=valid,
    )
    pred_fg_half_metrics = _binary_metrics(
        pred_fg_half,
        gt_fg,
        valid=valid,
    )

    surface_metrics = _binary_metrics(
        probs["surface"].detach().float().cpu() >= 0.5,
        targets.surface.detach().cpu() >= 0.5,
        valid=valid,
    )
    separator_metrics = _binary_metrics(
        probs["separator"].detach().float().cpu() >= 0.5,
        targets.separator.detach().cpu() >= 0.5,
        valid=valid,
    )
    seed_metrics = _binary_metrics(
        probs["seed"].detach().float().cpu() >= 0.5,
        targets.seed.detach().cpu() >= 0.5,
        valid=valid,
    )

    valid_fg = gt_fg & valid[:, None].expand_as(gt_fg)
    flow_metrics = _vector_metrics(
        geometry.flow.detach().float().cpu(),
        targets.flow.detach().cpu(),
        valid_fg,
    )
    offset_metrics = _vector_metrics(
        geometry.centroid_offset.detach().float().cpu(),
        targets.centroid_offset.detach().cpu(),
        valid_fg,
    )
    sdf_metrics = _sdf_metrics(
        geometry.sdf.detach().float().cpu(),
        targets.sdf.detach().cpu(),
        targets.sdf_valid.detach().cpu(),
        valid,
    )

    cell_metrics = _missing_cell_metrics(
        crop.gt_labels.detach().cpu(),
        crop.batch["instance_labels"].detach().cpu(),
        pred_fg_partition,
        valid,
        cell_ids=(
            tuple(record.complete_cell_ids)
            + tuple(record.true_boundary_cell_ids)
        ),
    )

    metrics: dict[str, Any] = {
        "geometry_loss": float(
            criterion_metrics["geometry_loss"].detach().float().cpu()
        ),
        "foreground_threshold": foreground_threshold,
        "source_foreground_dice": source_fg_metrics["dice"],
        "source_foreground_precision": source_fg_metrics["precision"],
        "source_foreground_recall": source_fg_metrics["recall"],
        "pred_foreground_dice": pred_fg_metrics["dice"],
        "pred_foreground_precision": pred_fg_metrics["precision"],
        "pred_foreground_recall": pred_fg_metrics["recall"],
        "pred_foreground_dice_at_0p5": pred_fg_half_metrics["dice"],
        "surface_dice_at_0p5": surface_metrics["dice"],
        "surface_precision_at_0p5": surface_metrics["precision"],
        "surface_recall_at_0p5": surface_metrics["recall"],
        "surface_soft_mae": _masked_mae(
            probs["surface"].detach().float().cpu(),
            targets.surface.detach().cpu(),
            valid,
        ),
        "separator_dice_at_0p5": separator_metrics["dice"],
        "separator_precision_at_0p5": separator_metrics["precision"],
        "separator_recall_at_0p5": separator_metrics["recall"],
        "separator_soft_mae": _masked_mae(
            probs["separator"].detach().float().cpu(),
            targets.separator.detach().cpu(),
            valid,
        ),
        "seed_dice_at_0p5": seed_metrics["dice"],
        "seed_precision_at_0p5": seed_metrics["precision"],
        "seed_recall_at_0p5": seed_metrics["recall"],
        "seed_soft_mae": _masked_mae(
            probs["seed"].detach().float().cpu(),
            targets.seed.detach().cpu(),
            valid,
        ),
        "sdf_mae": sdf_metrics["mae"],
        "sdf_rmse": sdf_metrics["rmse"],
        "flow_cosine": flow_metrics["cosine"],
        "flow_epe": flow_metrics["epe"],
        "centroid_offset_cosine": offset_metrics["cosine"],
        "centroid_offset_epe": offset_metrics["epe"],
        "valid_fraction": float(valid.float().mean().item()),
        "merge_source_count": len(record.merge_source_ids),
        "merge_gt_count": len(record.merge_gt_ids),
        "complete_cell_count": len(record.complete_cell_ids),
        "partial_cell_count": len(record.partial_cell_ids),
        "true_boundary_cell_count": len(record.true_boundary_cell_ids),
        **cell_metrics,
    }

    for key, value in criterion_metrics.items():
        if key == "loss":
            metrics["criterion_loss"] = float(
                value.detach().float().cpu()
            )
        elif key.startswith("geometry_") or key == "sdf_valid_fraction":
            metrics[f"criterion_{key}"] = float(
                value.detach().float().cpu()
            )

    return metrics


# ======================================================================================
# Output artifacts / visualization
# ======================================================================================


def _crop_bounds(record) -> list[list[int]]:
    slices = getattr(record, "global_slices_zyx", record.slices_zyx)
    return [[int(axis.start), int(axis.stop)] for axis in slices]


def _record_display_start(record) -> tuple[int, int, int]:
    slices = getattr(record, "global_slices_zyx", record.slices_zyx)
    return tuple(int(axis.start) for axis in slices)


def _save_crop_npz(
    path: Path,
    *,
    source_batch: dict,
    crop,
    output,
    targets,
    record,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    geometry = output.geometry
    probs = geometry.probabilities()

    raw_native = (
        source_batch["raw_volume"][record.batch_index][record.slices_zyx]
        .detach()
        .cpu()
        .numpy()
    )

    spatial = crop.batch["spatial_inputs"][0].detach().cpu().numpy()
    source_labels = (
        crop.batch["instance_labels"][0].detach().cpu().numpy()
    )
    gt_labels = crop.gt_labels[0].detach().cpu().numpy()
    valid = (
        crop.batch["supervision_valid_mask"][0]
        .detach()
        .cpu()
        .numpy()
    )

    np.savez_compressed(
        path,
        raw_native=np.asarray(raw_native),
        spatial_inputs=spatial.astype(np.float16, copy=False),
        normalized_raw=spatial[0].astype(np.float16, copy=False),
        source_labels=source_labels.astype(np.int32, copy=False),
        gt_labels=gt_labels.astype(np.int32, copy=False),
        supervision_valid=valid.astype(np.uint8, copy=False),
        pred_foreground=probs["foreground"][0, 0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float16),
        pred_surface=probs["surface"][0, 0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float16),
        pred_separator=probs["separator"][0, 0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float16),
        pred_seed=probs["seed"][0, 0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float16),
        pred_sdf=geometry.sdf[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float16),
        pred_flow=geometry.flow[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float16),
        pred_centroid_offset=geometry.centroid_offset[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float16),
        target_foreground=targets.foreground[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float16),
        target_surface=targets.surface[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float16),
        target_separator=targets.separator[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float16),
        target_seed=targets.seed[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float16),
        target_sdf=targets.sdf[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float16),
        target_sdf_valid=targets.sdf_valid[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8),
        target_flow=targets.flow[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float16),
        target_centroid_offset=targets.centroid_offset[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float16),
        spacing_zyx_um=crop.batch["spacing_um"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        dref_um=np.asarray(
            [float(crop.batch["dref_um"][0])],
            dtype=np.float32,
        ),
        crop_bounds_zyx=np.asarray(_crop_bounds(record), dtype=np.int32),
        metadata_json=np.asarray(json.dumps(_jsonable(metadata))),
    )


def _best_z_slice(gt: np.ndarray, valid: np.ndarray) -> int:
    score = ((gt > 0) & valid.astype(bool)).sum(axis=(1, 2))
    if score.size and int(score.max()) > 0:
        return int(score.argmax())
    return int(gt.shape[0] // 2)


def _save_montage(path: Path, npz_path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    with np.load(npz_path, allow_pickle=False) as data:
        raw = data["normalized_raw"].astype(np.float32)
        source = data["source_labels"]
        gt = data["gt_labels"]
        valid = data["supervision_valid"]
        z = _best_z_slice(gt, valid)

        pred_flow_mag = np.linalg.norm(
            data["pred_flow"].astype(np.float32),
            axis=0,
        )
        target_flow_mag = np.linalg.norm(
            data["target_flow"].astype(np.float32),
            axis=0,
        )
        pred_offset_mag = np.linalg.norm(
            data["pred_centroid_offset"].astype(np.float32),
            axis=0,
        )
        target_offset_mag = np.linalg.norm(
            data["target_centroid_offset"].astype(np.float32),
            axis=0,
        )

        panels = [
            ("normalized raw", raw[z], "gray", None, None),
            ("source labels", source[z], "nipy_spectral", None, None),
            ("GT labels", gt[z], "nipy_spectral", None, None),
            ("supervision valid", valid[z], "gray", 0, 1),
            ("pred foreground", data["pred_foreground"][z], "magma", 0, 1),
            ("target foreground", data["target_foreground"][z], "magma", 0, 1),
            ("pred surface", data["pred_surface"][z], "magma", 0, 1),
            ("target surface", data["target_surface"][z], "magma", 0, 1),
            ("pred separator", data["pred_separator"][z], "magma", 0, 1),
            ("target separator", data["target_separator"][z], "magma", 0, 1),
            ("pred seed", data["pred_seed"][z], "magma", 0, 1),
            ("target seed", data["target_seed"][z], "magma", 0, 1),
            ("pred SDF", data["pred_sdf"][z], "coolwarm", None, None),
            ("target SDF", data["target_sdf"][z], "coolwarm", None, None),
            ("pred / target flow |v|", pred_flow_mag[z] - target_flow_mag[z], "coolwarm", None, None),
            ("pred / target offset |v|", pred_offset_mag[z] - target_offset_mag[z], "coolwarm", None, None),
        ]

    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    for axis, (name, image, cmap, vmin, vmax) in zip(axes.flat, panels):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(name)
        axis.axis("off")
    fig.suptitle(f"{title} | z={z}", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _napari_vectors(
    field: np.ndarray,
    mask: np.ndarray,
    *,
    stride: tuple[int, int, int] = (2, 12, 12),
    vector_scale: float = 6.0,
) -> np.ndarray:
    z_idx = np.arange(0, field.shape[1], stride[0], dtype=np.int32)
    y_idx = np.arange(0, field.shape[2], stride[1], dtype=np.int32)
    x_idx = np.arange(0, field.shape[3], stride[2], dtype=np.int32)
    zz, yy, xx = np.meshgrid(z_idx, y_idx, x_idx, indexing="ij")
    starts = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
    keep = mask[starts[:, 0], starts[:, 1], starts[:, 2]]
    starts = starts[keep]
    if starts.size == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)

    vectors = np.stack(
        [
            field[0, starts[:, 0], starts[:, 1], starts[:, 2]],
            field[1, starts[:, 0], starts[:, 1], starts[:, 2]],
            field[2, starts[:, 0], starts[:, 1], starts[:, 2]],
        ],
        axis=-1,
    ).astype(np.float32)
    vectors *= float(vector_scale)
    return np.stack([starts.astype(np.float32), vectors], axis=1)


def _open_napari(npz_path: Path) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is not installed. Install the repository visualization "
            "requirements or rerun without --napari."
        ) from exc

    data = np.load(npz_path, allow_pickle=False)
    spacing = tuple(float(v) for v in data["spacing_zyx_um"])

    viewer = napari.Viewer(title=f"STIR-Net geometry eval — {npz_path.name}", ndisplay=3)
    viewer.add_image(
        data["normalized_raw"].astype(np.float32),
        name="normalized raw",
        scale=spacing,
    )
    viewer.add_labels(
        data["source_labels"].astype(np.int32),
        name="source labels",
        scale=spacing,
        visible=False,
    )
    viewer.add_labels(
        data["gt_labels"].astype(np.int32),
        name="GT labels",
        scale=spacing,
    )
    viewer.add_image(
        data["pred_foreground"].astype(np.float32),
        name="pred foreground",
        scale=spacing,
        opacity=0.55,
    )
    viewer.add_image(
        data["pred_surface"].astype(np.float32),
        name="pred surface",
        scale=spacing,
        opacity=0.55,
        visible=False,
    )
    viewer.add_image(
        data["pred_separator"].astype(np.float32),
        name="pred separator",
        scale=spacing,
        opacity=0.70,
    )
    viewer.add_image(
        data["target_separator"].astype(np.float32),
        name="target separator",
        scale=spacing,
        opacity=0.55,
        visible=False,
    )
    viewer.add_image(
        data["pred_seed"].astype(np.float32),
        name="pred seed",
        scale=spacing,
        opacity=0.65,
        visible=False,
    )
    viewer.add_image(
        data["pred_sdf"].astype(np.float32),
        name="pred SDF",
        scale=spacing,
        visible=False,
    )
    viewer.add_image(
        data["target_sdf"].astype(np.float32),
        name="target SDF",
        scale=spacing,
        visible=False,
    )

    fg_mask = data["target_foreground"].astype(np.float32) > 0.5
    pred_flow = _napari_vectors(
        data["pred_flow"].astype(np.float32),
        fg_mask,
    )
    target_flow = _napari_vectors(
        data["target_flow"].astype(np.float32),
        fg_mask,
    )
    pred_offset = _napari_vectors(
        data["pred_centroid_offset"].astype(np.float32),
        fg_mask,
    )
    target_offset = _napari_vectors(
        data["target_centroid_offset"].astype(np.float32),
        fg_mask,
    )

    if len(pred_flow):
        viewer.add_vectors(
            pred_flow,
            name="pred flow vectors",
            visible=False,
        )
    if len(target_flow):
        viewer.add_vectors(
            target_flow,
            name="target flow vectors",
            visible=False,
        )
    if len(pred_offset):
        viewer.add_vectors(
            pred_offset,
            name="pred centroid-offset vectors",
            visible=False,
        )
    if len(target_offset):
        viewer.add_vectors(
            target_offset,
            name="target centroid-offset vectors",
            visible=False,
        )

    napari.run()
    data.close()


# ======================================================================================
# Aggregation / CSV
# ======================================================================================


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"crop_count": 0}

    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if _finite_number(value)
        }
    )
    aggregate: dict[str, Any] = {"crop_count": len(rows)}
    for key in numeric_keys:
        values = [
            float(row[key])
            for row in rows
            if key in row and _finite_number(row[key])
        ]
        if not values:
            continue
        aggregate[f"mean_{key}"] = float(np.mean(values))
        aggregate[f"min_{key}"] = float(np.min(values))
        aggregate[f"max_{key}"] = float(np.max(values))

        if key.endswith("_count") or key.endswith("_cell_count"):
            aggregate[f"sum_{key}"] = float(np.sum(values))
    return aggregate


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    preferred = [
        "sample",
        "role",
        "crop_index",
        "candidate_type",
        "crop_bounds_zyx",
        "geometry_loss",
        "source_foreground_dice",
        "pred_foreground_dice",
        "pred_foreground_recall",
        "separator_dice_at_0p5",
        "sdf_mae",
        "flow_cosine",
        "centroid_offset_epe",
        "source_missing_cell_count",
        "mean_pred_recall_source_missing_cells",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
            seen.add(key)
    for key in sorted({key for row in rows for key in row}):
        if key not in seen:
            keys.append(key)
            seen.add(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            serialized = {}
            for key in keys:
                value = row.get(key)
                if isinstance(value, (list, tuple, dict)):
                    serialized[key] = json.dumps(_jsonable(value))
                else:
                    serialized[key] = value
            writer.writerow(serialized)


def _print_crop_line(row: dict[str, Any]) -> None:
    missing = row.get("mean_pred_recall_source_missing_cells")
    missing_text = (
        "n/a"
        if not isinstance(missing, (int, float)) or not math.isfinite(float(missing))
        else f"{float(missing):.3f}"
    )
    print(
        f"[eval] {row['sample']:11s} "
        f"crop={int(row['crop_index']):02d} "
        f"type={str(row['candidate_type']):8s} "
        f"loss={float(row['geometry_loss']):.3f} "
        f"srcFG={float(row['source_foreground_dice']):.3f} "
        f"predFG={float(row['pred_foreground_dice']):.3f} "
        f"sep={float(row['separator_dice_at_0p5']):.3f} "
        f"sdfMAE={float(row['sdf_mae']):.3f} "
        f"flowCos={float(row['flow_cosine']):.3f} "
        f"missingRecall={missing_text}",
        flush=True,
    )


# ======================================================================================
# One crop
# ======================================================================================


def _evaluate_crop(
    *,
    sample: str,
    role: str,
    crop_index: int,
    record,
    source_batch: dict,
    model,
    model_cfg,
    train_cfg,
    criterion,
    target_cache,
    device: torch.device,
    target_backend: str,
    amp_dtype: str,
    sample_output_dir: Path,
    save_png: bool,
):
    from learned.stirnet.training.prepared_geometry import (
        compose_source_conditioned_geometry_targets,
    )
    from learned.stirnet.training.trainer import (
        model_forward_from_batch,
        move_batch_to_device,
    )

    spec = _record_to_spec(
        record,
        full_shape=tuple(int(v) for v in source_batch["gt_labels"].shape[-3:]),
        spacing=source_batch["spacing_um"][record.batch_index],
    )
    crop = _prepare_eval_crop(
        source_batch,
        spec,
        train_cfg=train_cfg,
    )

    cfg = train_cfg.curriculum
    static_targets, cache_stats = target_cache.get_or_build_batch(
        source_batch,
        source_batch["gt_labels"],
        [spec],
        spacing_um=source_batch["spacing_um"],
        dref_um=source_batch["dref_um"],
        geometry_config=model_cfg.geometry,
        backend=target_backend,
        gpu_min_voxels=int(train_cfg.geometry_target_gpu_min_voxels),
        halo_um=float(cfg.refinement_crop_target_halo_um),
    )
    targets = compose_source_conditioned_geometry_targets(
        static_targets,
        crop.gt_labels,
        crop.batch.get("instance_labels"),
        crop.batch["spacing_um"],
        geometry_config=model_cfg.geometry,
        backend=target_backend,
        gpu_min_voxels=int(train_cfg.geometry_target_gpu_min_voxels),
    )

    model_batch = move_batch_to_device(crop.batch, device)
    valid_device = crop.batch["supervision_valid_mask"].to(device)
    gt_device = crop.gt_labels.to(device)

    started = time.perf_counter()
    amp_context, amp_used = _amp_context(device, amp_dtype)
    with torch.inference_mode():
        with amp_context:
            output = model_forward_from_batch(
                model,
                model_batch,
                use_temporal=False,
                execution_stage="geometry",
                apply_existence_filter=False,
            )
            criterion_metrics = criterion(
                output,
                gt_device,
                model_batch["spacing_um"],
                model_batch["dref_um"],
                stage="geometry_bootstrap",
                current_labels=model_batch.get("instance_labels"),
                precomputed_geometry_targets=targets,
                supervision_valid_mask=valid_device,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started

    metrics = _compute_metrics(
        output=output,
        targets=targets,
        crop=crop,
        record=record,
        model_cfg=model_cfg,
        criterion_metrics=criterion_metrics,
    )
    row: dict[str, Any] = {
        "sample": sample,
        "role": role,
        "crop_index": int(crop_index),
        "candidate_type": record.candidate_type,
        "crop_bounds_zyx": _crop_bounds(record),
        "complete_cell_ids": [int(v) for v in record.complete_cell_ids],
        "partial_cell_ids": [int(v) for v in record.partial_cell_ids],
        "true_boundary_cell_ids": [int(v) for v in record.true_boundary_cell_ids],
        "merge_source_ids": [int(v) for v in record.merge_source_ids],
        "merge_gt_ids": [int(v) for v in record.merge_gt_ids],
        "inference_seconds": float(inference_seconds),
        "amp_dtype": amp_used,
        "target_cache_memory_hits": int(cache_stats["memory_hits"]),
        "target_cache_disk_hits": int(cache_stats["disk_hits"]),
        "target_cache_misses": int(cache_stats["misses"]),
        **metrics,
    }

    display_start = _record_display_start(record)
    stem = (
        f"{sample}_crop_{crop_index:02d}_"
        f"{record.candidate_type}_"
        f"z{display_start[0]:03d}_"
        f"y{display_start[1]:03d}_"
        f"x{display_start[2]:03d}"
    )
    npz_path = sample_output_dir / f"{stem}.npz"
    png_path = sample_output_dir / f"{stem}.png"

    artifact_meta = {
        **row,
        "checkpoint_global_step": None,
        "spacing_zyx_um": [
            float(v) for v in crop.batch["spacing_um"][0].tolist()
        ],
        "dref_um": float(crop.batch["dref_um"][0]),
    }
    _save_crop_npz(
        npz_path,
        source_batch=source_batch,
        crop=crop,
        output=output,
        targets=targets,
        record=record,
        metadata=artifact_meta,
    )
    if save_png:
        _save_montage(
            png_path,
            npz_path,
            title=f"{sample} | {role} | {record.candidate_type} crop {crop_index}",
        )

    row["npz_path"] = str(npz_path)
    row["png_path"] = str(png_path) if save_png else None

    # Release crop-sized dense tensors before the next crop on a small local GPU.
    del output, criterion_metrics, model_batch, targets, static_targets, crop
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return row, npz_path



def _evaluate_large_volume_sample(
    *,
    sample: str,
    role: str,
    nis3d_root: Path,
    args,
    model,
    model_cfg,
    train_cfg,
    criterion,
    target_cache,
    device: torch.device,
    sample_output_dir: Path,
):
    sample_dir = nis3d_root / sample
    raw, gt, confidence, spacing, _info_text, files = _load_nis3d_arrays(sample_dir)
    scan_started = time.perf_counter()
    scan = _scan_large_nis3d_metadata(gt, confidence)
    scan_seconds = time.perf_counter() - scan_started
    crop_shape = tuple(int(v) for v in train_cfg.curriculum.refinement_crop_shape_zyx)
    selected = _proxy_candidate_global_slices(
        gt,
        confidence,
        touched_ids=scan["touched_ids"],
        crop_shape=crop_shape,
        total=int(args.crops_per_sample),
        merge_fraction=float(args.merge_fraction),
    )
    report = {
        "sample": sample,
        "sample_dir": str(sample_dir),
        "files": {k: str(v) for k, v in files.items()},
        "shape_zyx": [int(v) for v in raw.shape],
        "raw_dtype": str(raw.dtype),
        "gt_dtype": str(gt.dtype),
        "confidence_dtype": str(confidence.dtype),
        "spacing_zyx_um": [float(v) for v in spacing],
        "source_mode": "crop_proxy",
        "large_volume_scan_seconds": float(scan_seconds),
        "source_prepare_seconds": 0.0,
        "source_cache_hit": False,
        "source_current_instance_count": None,
        "model_dref_um": None,
        "supervision_valid_fraction": float("nan"),
        **{k: v for k, v in scan.items() if k not in {"gt_ids", "touched_ids", "kept_ids"}},
    }
    rows: list[dict[str, Any]] = []
    selected_report: list[dict[str, Any]] = []
    first_npz = None
    drefs: list[float] = []
    source_counts: list[int] = []
    valid_fractions: list[float] = []
    for crop_index, (global_core, planned_type) in enumerate(selected):
        print(
            f"[proxy] {sample} crop={crop_index:02d} planned={planned_type} "
            f"bounds={_crop_bounds(_ProxyRecord(0, global_core, global_core, planned_type))}",
            flush=True,
        )
        source_batch, record = _prepare_large_proxy_batch(
            sample=sample,
            raw=raw,
            gt=gt,
            confidence=confidence,
            spacing=spacing,
            global_core=global_core,
            touched_ids=scan["touched_ids"],
            train_cfg=train_cfg,
            confidence_ignore_margin_um=float(args.confidence_ignore_margin_um),
            context_um=float(args.large_volume_context_um),
        )
        meta = dict(source_batch["source_preprocessing_metadata"][0])
        drefs.append(float(source_batch["dref_um"][0]))
        source_counts.append(int(meta.get("source_current_instance_count", 0)))
        local_valid = source_batch["supervision_valid_mask"][0][record.slices_zyx]
        valid_fractions.append(float(local_valid.float().mean()))
        row, npz_path = _evaluate_crop(
            sample=sample,
            role=role,
            crop_index=crop_index,
            record=record,
            source_batch=source_batch,
            model=model,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            criterion=criterion,
            target_cache=target_cache,
            device=device,
            target_backend=args.target_backend,
            amp_dtype=train_cfg.amp_dtype,
            sample_output_dir=sample_output_dir,
            save_png=not args.no_png,
        )
        row["source_mode"] = "crop_proxy"
        row["planned_candidate_type"] = planned_type
        rows.append(row)
        selected_report.append({
            "crop_index": crop_index,
            "planned_candidate_type": planned_type,
            "candidate_type": record.candidate_type,
            "bounds_zyx": _crop_bounds(record),
            "complete_cell_count": len(record.complete_cell_ids),
            "partial_cell_count": len(record.partial_cell_ids),
            "merge_source_ids": [int(v) for v in record.merge_source_ids],
            "merge_gt_ids": [int(v) for v in record.merge_gt_ids],
        })
        _print_crop_line(row)
        if first_npz is None:
            first_npz = npz_path
        del source_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if drefs:
        report["model_dref_um"] = float(np.mean(drefs))
    if source_counts:
        report["source_current_instance_count_mean_per_context"] = float(np.mean(source_counts))
    if valid_fractions:
        report["supervision_valid_fraction"] = float(np.mean(valid_fractions))
    return rows, report, selected_report, first_npz


# ======================================================================================
# Main evaluation
# ======================================================================================


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from learned.stirnet.training.criterion import StirNetCriterion
    from learned.stirnet.training.crop_target_cache import StaticCropTargetCache

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (REPO_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    samples = _parse_samples(args.samples)
    nis3d_root = _discover_nis3d_root(
        samples,
        None if args.nis3d_root is None else Path(args.nis3d_root),
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    checkpoint, model, model_cfg, train_cfg = _load_checkpoint_model(
        checkpoint_path,
        device,
    )
    criterion = StirNetCriterion(model_cfg, train_cfg.loss).to(device)
    criterion.eval()

    checkpoint_step = int(checkpoint.get("global_step", -1))
    checkpoint_samples = tuple(
        str(value)
        for value in checkpoint.get("extra", {}).get("samples", ())
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else (
            REPO_ROOT
            / "runs"
            / "stirnet"
            / "evaluation"
            / SCRIPT_NAME
            / f"{checkpoint_path.stem}_{timestamp}"
        )
    )
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_cache_root = (
        REPO_ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / SCRIPT_NAME
        / "cache"
        / "source"
    )
    target_cache_root = (
        REPO_ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / SCRIPT_NAME
        / "cache"
        / "static_gt"
    )
    source_cache_root.mkdir(parents=True, exist_ok=True)
    target_cache_root.mkdir(parents=True, exist_ok=True)

    target_cache = StaticCropTargetCache(
        max_memory_entries=4,
        disk_dir=target_cache_root,
    )

    print("=" * 112, flush=True)
    print("STIR-Net Investigation 04 — NIS3D geometry checkpoint evaluation", flush=True)
    print("=" * 112, flush=True)
    print(f"Checkpoint       : {checkpoint_path}", flush=True)
    print(f"Checkpoint step  : {checkpoint_step}", flush=True)
    print(f"Checkpoint data  : {list(checkpoint_samples)}", flush=True)
    print(f"Samples          : {list(samples)}", flush=True)
    print(f"NIS3D root       : {nis3d_root}", flush=True)
    print(f"Device           : {device}", flush=True)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(
            f"GPU              : {props.name} ({props.total_memory / 2**30:.2f} GiB)",
            flush=True,
        )
    print(
        f"Crop shape       : {tuple(train_cfg.curriculum.refinement_crop_shape_zyx)}",
        flush=True,
    )
    print(f"Crops/sample     : {args.crops_per_sample}", flush=True)
    print(f"Merge fraction   : {args.merge_fraction:.2f}", flush=True)
    print(f"Target backend   : {args.target_backend}", flush=True)
    print(f"Output           : {output_root}", flush=True)
    print("=" * 112, flush=True)

    all_rows: list[dict[str, Any]] = []
    sample_reports: dict[str, Any] = {}
    selected_crop_report: dict[str, Any] = {}
    first_npz: Path | None = None
    started_all = time.perf_counter()

    for sample in samples:
        role = (
            "training_reference"
            if sample in checkpoint_samples
            else "validation_unseen"
        )
        print(f"\n[data] preparing {sample} ({role}) ...", flush=True)

        # Read shapes first. Huge NIS3D validation volumes are evaluated through
        # a bounded context around each production-sized crop; smaller volumes
        # retain the exact full-frame source preparation used by Training 01.
        sample_dir = nis3d_root / sample
        probe_files = _resolve_sample_files(sample_dir)
        import tifffile
        with tifffile.TiffFile(probe_files["raw"]) as tif:
            probe_shape = tuple(int(v) for v in tif.series[0].shape)
        voxel_count = int(np.prod(probe_shape))
        sample_output_dir = output_root / sample
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        if voxel_count > int(args.large_volume_proxy_voxels):
            print(
                f"[data] {sample}: voxels={voxel_count:,} -> source_mode=crop_proxy; "
                f"context={args.large_volume_context_um:.1f}um",
                flush=True,
            )
            rows, sample_report, selected_report, sample_first_npz = _evaluate_large_volume_sample(
                sample=sample,
                role=role,
                nis3d_root=nis3d_root,
                args=args,
                model=model,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                criterion=criterion,
                target_cache=target_cache,
                device=device,
                sample_output_dir=sample_output_dir,
            )
            sample_reports[sample] = sample_report
            selected_crop_report[sample] = selected_report
            for row in rows:
                row["checkpoint_global_step"] = checkpoint_step
                all_rows.append(row)
                _append_jsonl(output_root / "metrics.jsonl", row)
            if first_npz is None:
                first_npz = sample_first_npz
            continue

        source_batch, sample_report = _prepare_source_batch(
            nis3d_root,
            sample,
            source_cache_root=source_cache_root,
            confidence_ignore_margin_um=float(args.confidence_ignore_margin_um),
        )
        sample_report["source_mode"] = "full_exact"
        sample_reports[sample] = sample_report

        manifest_started = time.perf_counter()
        manifest = _build_eval_manifest(source_batch, train_cfg)
        manifest_seconds = time.perf_counter() - manifest_started
        records = _select_eval_records(
            manifest,
            total=int(args.crops_per_sample),
            merge_fraction=float(args.merge_fraction),
        )

        sample_report["manifest_seconds"] = float(manifest_seconds)
        sample_report["manifest_record_count"] = len(manifest.records[0])
        sample_report["manifest_merge_record_count"] = sum(
            bool(row.merge_source_ids) for row in manifest.records[0]
        )
        sample_report["manifest_uncoverable_cell_count"] = len(
            manifest.uncoverable_cell_ids[0]
        )
        sample_report["manifest_uncoverable_merge_source_count"] = len(
            manifest.uncoverable_merge_source_ids[0]
        )

        selected_crop_report[sample] = [
            {
                "crop_index": index,
                "candidate_type": row.candidate_type,
                "bounds_zyx": _crop_bounds(row),
                "complete_cell_count": len(row.complete_cell_ids),
                "partial_cell_count": len(row.partial_cell_ids),
                "merge_source_ids": [int(v) for v in row.merge_source_ids],
                "merge_gt_ids": [int(v) for v in row.merge_gt_ids],
            }
            for index, row in enumerate(records)
        ]

        print(
            f"[data] {sample}: GT={sample_report['gt_ids_kept']} "
            f"source={sample_report['source_current_instance_count']} "
            f"valid={sample_report['supervision_valid_fraction']:.4f} "
            f"dref={sample_report['model_dref_um']:.4f}um "
            f"manifest={len(manifest.records[0])} rows "
            f"({manifest_seconds:.2f}s)",
            flush=True,
        )

        for crop_index, record in enumerate(records):
            row, npz_path = _evaluate_crop(
                sample=sample,
                role=role,
                crop_index=crop_index,
                record=record,
                source_batch=source_batch,
                model=model,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                criterion=criterion,
                target_cache=target_cache,
                device=device,
                target_backend=args.target_backend,
                amp_dtype=train_cfg.amp_dtype,
                sample_output_dir=sample_output_dir,
                save_png=not args.no_png,
            )
            row["checkpoint_global_step"] = checkpoint_step
            row["source_mode"] = "full_exact"
            all_rows.append(row)
            _append_jsonl(output_root / "metrics.jsonl", row)
            _print_crop_line(row)
            if first_npz is None:
                first_npz = npz_path

        del source_batch, manifest, records
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_metrics_csv(output_root / "metrics.csv", all_rows)
    _write_json(output_root / "selected_crops.json", selected_crop_report)
    _write_json(output_root / "sample_reports.json", sample_reports)

    aggregate_by_sample = {
        sample: _aggregate([row for row in all_rows if row["sample"] == sample])
        for sample in samples
    }
    aggregate_by_role = {
        role: _aggregate([row for row in all_rows if row["role"] == role])
        for role in sorted(set(row["role"] for row in all_rows))
    }

    elapsed = time.perf_counter() - started_all
    summary = {
        "status": "success",
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": checkpoint_step,
        "checkpoint_extra": checkpoint.get("extra", {}),
        "checkpoint_training_samples": list(checkpoint_samples),
        "samples": list(samples),
        "nis3d_root": str(nis3d_root),
        "device": str(device),
        "crop_shape_zyx": list(
            train_cfg.curriculum.refinement_crop_shape_zyx
        ),
        "crops_per_sample": int(args.crops_per_sample),
        "merge_fraction": float(args.merge_fraction),
        "confidence_ignore_margin_um": float(
            args.confidence_ignore_margin_um
        ),
        "target_backend": args.target_backend,
        "elapsed_seconds": float(elapsed),
        "elapsed_human": _duration(elapsed),
        "output_root": str(output_root),
        "aggregate_by_sample": aggregate_by_sample,
        "aggregate_by_role": aggregate_by_role,
    }
    _write_json(output_root / "summary.json", summary)

    readme = f"""STIR-Net NIS3D geometry checkpoint evaluation

Checkpoint:
  {checkpoint_path}
Global step:
  {checkpoint_step}

Roles:
  training_reference = sample listed in checkpoint metadata
  validation_unseen   = sample not listed in checkpoint metadata

Important metrics:
  source_foreground_dice
      Binary foreground Dice of preprocessing/current segmentation vs GT.
  pred_foreground_dice
      Checkpoint foreground Dice vs GT at the production partition threshold
      ({model_cfg.partition.foreground_threshold:.3f}).
  separator_dice_at_0p5
      Thresholded source-conditioned separator target vs predicted separator.
  sdf_mae / sdf_rmse
      SDF error only where SDF supervision is valid.
  flow_cosine / flow_epe
      Direction agreement and endpoint error inside valid GT foreground.
  centroid_offset_epe
      Centroid-offset vector endpoint error inside valid GT foreground.
  mean_pred_recall_source_missing_cells
      Mean predicted foreground recall for complete GT cells whose natural
      source/current foreground coverage is below 0.50.

Files:
  summary.json          aggregate results
  metrics.csv           one row per evaluated crop
  metrics.jsonl         full per-crop metrics
  selected_crops.json   deterministic crop coordinates/metadata
  sample_reports.json   source/confidence/dataset preparation diagnostics
  <sample>/*.npz        dense predictions, targets, vectors, labels
  <sample>/*.png        quick mid-plane geometry montages

This is a geometry-only evaluation. It intentionally does not claim RAG,
instance partition, temporal, or final tracking performance.
"""
    (output_root / "README.txt").write_text(readme, encoding="utf-8")

    print("\n" + "=" * 112, flush=True)
    print("EVALUATION COMPLETE", flush=True)
    print("=" * 112, flush=True)
    print(f"Elapsed          : {_duration(elapsed)}", flush=True)
    print(f"Output           : {output_root}", flush=True)
    for sample in samples:
        aggregate = aggregate_by_sample[sample]
        print(
            f"{sample:16s} "
            f"FG={aggregate.get('mean_pred_foreground_dice', float('nan')):.3f} "
            f"sourceFG={aggregate.get('mean_source_foreground_dice', float('nan')):.3f} "
            f"SEP={aggregate.get('mean_separator_dice_at_0p5', float('nan')):.3f} "
            f"SDF_MAE={aggregate.get('mean_sdf_mae', float('nan')):.3f} "
            f"flowCos={aggregate.get('mean_flow_cosine', float('nan')):.3f}",
            flush=True,
        )
    print("=" * 112, flush=True)

    if args.napari:
        if first_npz is None:
            raise RuntimeError("No crop was generated for Napari")
        print(f"[napari] opening {first_npz}", flush=True)
        _open_napari(first_npz)

    return summary


# ======================================================================================
# CLI
# ======================================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a STIR-Net geometry-bootstrap checkpoint on deterministic "
            "merge/coverage crops from NIS3D."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_REL),
        help="Checkpoint path, absolute or relative to repository root.",
    )
    parser.add_argument(
        "--samples",
        default=",".join(DEFAULT_SAMPLES),
        help=(
            "Comma-separated NIS3D samples. Default compares unseen Zebrafish_1 "
            "with training-reference Zebrafish_2."
        ),
    )
    parser.add_argument(
        "--nis3d-root",
        default=None,
        help="Optional explicit NIS3D root containing sample directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--crops-per-sample",
        type=int,
        default=6,
        help="Number of deterministic evaluation crops per sample (default: 6).",
    )
    parser.add_argument(
        "--merge-fraction",
        type=float,
        default=0.5,
        help="Desired merge-crop share of selected evaluation crops (default: 0.5).",
    )
    parser.add_argument(
        "--confidence-ignore-margin-um",
        type=float,
        default=DEFAULT_CONFIDENCE_IGNORE_MARGIN_UM,
        help="NIS3D confidence ignore margin in micrometres (default: 1.0).",
    )
    parser.add_argument(
        "--target-backend",
        choices=("auto", "scipy", "cupy"),
        default="auto",
        help="Geometry target EDT backend (default: auto).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--large-volume-proxy-voxels",
        type=int,
        default=LARGE_VOLUME_PROXY_VOXELS,
        help=(
            "Volumes above this voxel count use memory-bounded crop-context "
            "source preprocessing (default: 100,000,000)."
        ),
    )
    parser.add_argument(
        "--large-volume-context-um",
        type=float,
        default=DEFAULT_LARGE_VOLUME_CONTEXT_UM,
        help=(
            "Physical preprocessing context around each huge-volume model crop "
            "(default: 24 um)."
        ),
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip PNG montage generation; .npz and metrics are still saved.",
    )
    parser.add_argument(
        "--napari",
        action="store_true",
        help="Open the first evaluated crop in Napari after evaluation.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.crops_per_sample < 1:
        parser.error("--crops-per-sample must be >= 1")
    if not 0.0 <= args.merge_fraction <= 1.0:
        parser.error("--merge-fraction must be in [0, 1]")
    if args.confidence_ignore_margin_um < 0:
        parser.error("--confidence-ignore-margin-um cannot be negative")
    if args.large_volume_proxy_voxels < 1:
        parser.error("--large-volume-proxy-voxels must be >= 1")
    if args.large_volume_context_um < 0:
        parser.error("--large-volume-context-um cannot be negative")

    summary = evaluate(args)
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
