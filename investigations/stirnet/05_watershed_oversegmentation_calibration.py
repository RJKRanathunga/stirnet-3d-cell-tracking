from __future__ import annotations

"""STIR-Net Stage 05: oracle watershed oversegmentation calibration.

Goal
----
Calibrate the deterministic marker/watershed proposal stage BEFORE debugging
RAG/partition learning.  The desired asymmetry is deliberate:

    one GT cell -> several supervoxels       recoverable by the RAG
    two GT cells -> one supervoxel           NOT recoverable by a merge-only RAG

This investigation therefore searches for the least aggressive proposal policy
that removes cross-GT / undersegmented supervoxels across the Stage-00 stress
cases while keeping the graph reasonably small.

Why oracle geometry
-------------------
Every sweep is run from production geometry TARGETS converted to the same oracle
GeometryState used by Stage 03.  A failure here cannot be blamed on dense-model
training; it belongs to marker generation, watershed propagation, or post-
watershed cleanup.

What the script diagnoses
-------------------------
For every candidate/configuration it records:
  * marker count per visible GT cell;
  * seed/SDF/separator evidence at each GT cell's best marker location;
  * raw-watershed and cleaned-supervoxel cross-GT contamination separately;
  * whether cleanup INTRODUCED an unsafe merge;
  * complete, artificial-crop-boundary, and true-volume-boundary cell status;
  * recoverability by unions of supervoxels;
  * likely failure mechanism: missing marker, marker bridge,
    watershed propagation, or tiny-region cleanup.

Search plan (adaptive, time-budgeted)
-------------------------------------
Phase A -- structural sweep
  - fast and reference NMS backends;
  - physical marker spacing 0.35, 0.30, 0.25, 0.20 dref;
  - current tiny cleanup vs cleanup disabled (min_supervoxel_voxels=1).

Phase B -- threshold sweep
  - lower seed thresholds on the best structural families.

Phase C -- mechanism-specific sweep
  - if missing-marker failures remain: test a production-feasible
    separator-core marker rescue (NO GT identities are used by the rescue);
  - if marker-present propagation failures remain: test stronger separator
    watershed energy;
  - if small/truncated cells still receive weak scores: test more seed-head
    weight and less globally-normalized SDF weight.

The script stops cleanly before the requested wall-clock budget, writes progress
incrementally, and caches both expensive oracle contexts and completed runs.
It can therefore be resumed after interruption.

Default outputs
---------------
  data/learned/stirnet/watershed_oversegmentation_calibration/
      context_cache/candidate_XX.pt
      run_cache/<spec-hash>/candidate_XX.json
      progress.json
      final_report.json
      recommendation.json
      before_after_unsafe_cases.pt

Typical run (about <= 115 minutes by default):

    python investigations/stirnet/05_watershed_oversegmentation_calibration.py

Allow the full two hours:

    python investigations/stirnet/05_watershed_oversegmentation_calibration.py --budget-minutes 120

Visualize a previously unsafe candidate after calibration:

    python investigations/stirnet/05_watershed_oversegmentation_calibration.py --visualize-candidate 3

Notes
-----
* No production file is modified.
* No network training occurs.
* Separator-core marker rescue is only SIMULATED here.  If it wins, Stage 05
  prints the exact production change to implement in seeds.py/config.py later.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
from scipy import ndimage as ndi
from torch import Tensor


# =============================================================================
# REPOSITORY / ADJACENT INVESTIGATION IMPORTS
# =============================================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _import_adjacent(module_name: str, filename: str):
    path = Path(__file__).with_name(filename)
    if not path.exists():
        raise FileNotFoundError(f"Missing required investigation: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


stage00 = _import_adjacent(
    "_stirnet_stage05_stage00",
    "00_GT_validation.py",
)
stage03 = _import_adjacent(
    "_stirnet_stage05_stage03",
    "03_watershed_supervoxel_debug.py",
)


# =============================================================================
# DEFAULTS / ACCEPTANCE
# =============================================================================

DEFAULT_STAGE00_SAMPLE = (
    REPOSITORY_ROOT / "data" / "learned" / "stirnet" / "debug_crop.pt"
)
DEFAULT_RESULT_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "watershed_oversegmentation_calibration"
)
DEFAULT_MAX_CANDIDATES = 12
DEFAULT_BUDGET_MINUTES = 115.0
DEFAULT_RESERVE_MINUTES = 6.0

# A proposal is considered production-safe only if no meaningful supervoxel
# spans two GT identities and all complete / true-volume-boundary visible cells
# remain recoverable by unions of supervoxels.
MIN_CELL_COVERAGE = stage03.MIN_COMPLETE_GT_COVERAGE

# Deliberate oversegmentation target.  This is a ranking preference, not a hard
# acceptance threshold: safety always dominates graph compactness.
TARGET_MEAN_SV_PER_VISIBLE_GT = 2.0
SOFT_MAX_MEANINGFUL_SV_PER_GT = 5

CONTEXT_FORMAT_VERSION = 2
RUN_FORMAT_VERSION = 2
REPORT_FORMAT_VERSION = 2


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class SweepSpec:
    name: str
    phase: str
    backend: str
    seed_min_distance_dref: float
    seed_threshold: float
    min_supervoxel_voxels: int
    seed_sdf_weight: float
    seed_head_weight: float
    watershed_separator_weight: float
    watershed_surface_weight: float
    watershed_sdf_weight: float
    marker_rescue: str = "none"
    rescue_separator_cutoff: float = 0.50
    rescue_min_component_voxels: int = 8
    rescue_min_score: float = 0.10

    def key_payload(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        payload = json.dumps(
            self.key_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:12]

    def slug(self) -> str:
        safe = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in self.name
        )
        return f"{safe}__{self.hash()}"


@dataclass
class CandidateContext:
    candidate_index: int
    crop_slices_zyx: list[list[int]]
    selection: dict[str, Any]
    gt_labels: Tensor
    foreground_mask: Tensor
    surface_prob: Tensor
    separator_prob: Tensor
    seed_prob: Tensor
    sdf_normalized: Tensor
    spacing_um: Tensor
    dref_um: Tensor
    complete_gt_ids: list[int]
    partial_gt_ids: list[int]
    gt_meta: dict[str, Any]


@dataclass
class TimeBudget:
    started: float
    budget_seconds: float
    reserve_seconds: float

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def remaining(self) -> float:
        return max(self.budget_seconds - self.elapsed, 0.0)

    @property
    def usable_remaining(self) -> float:
        return max(self.remaining - self.reserve_seconds, 0.0)


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _progress(message: str = "") -> None:
    print(message, flush=True)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _slice_pairs(slices: tuple[slice, slice, slice]) -> list[list[int]]:
    return [[int(s.start), int(s.stop)] for s in slices]


def _pairs_to_slices(pairs: Iterable[Iterable[int]]) -> tuple[slice, slice, slice]:
    rows = list(pairs)
    if len(rows) != 3:
        raise ValueError(f"Expected three slice pairs, got {rows}")
    return tuple(slice(int(row[0]), int(row[1])) for row in rows)  # type: ignore[return-value]


def _slice_shape(slices: tuple[slice, slice, slice]) -> tuple[int, int, int]:
    return tuple(int(s.stop) - int(s.start) for s in slices)


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_min(values: list[float], default: float = 1.0) -> float:
    return float(min(values)) if values else float(default)


def _jsonable_float(value: float | np.floating) -> float:
    x = float(value)
    return x if math.isfinite(x) else 0.0


# =============================================================================
# SOURCE SCENE / SAVED STAGE-00 CANDIDATES
# =============================================================================


def load_source_scene(args: argparse.Namespace):
    """Load full arrays directly and reuse candidate_summaries from debug_crop.pt.

    We intentionally DO NOT call Stage-00 load_scene(), because that would
    enumerate candidates again and can waste several minutes.
    """
    setup_start = time.perf_counter()
    stage00_sample = args.stage00_sample.resolve()
    if not stage00_sample.exists():
        raise FileNotFoundError(
            f"Missing saved Stage-00 sample: {stage00_sample}\n"
            "Run investigations/stirnet/00_GT_validation.py once first."
        )

    _progress(f"[setup] Reading saved Stage-00 candidate list: {stage00_sample}")
    saved = torch_load(stage00_sample)
    selection = saved.get("selection", {})
    summaries = list(selection.get("candidate_summaries", []))
    if not summaries:
        raise RuntimeError(
            "debug_crop.pt does not contain selection['candidate_summaries']. "
            "Re-run the current Stage-00 investigation once."
        )
    summaries = summaries[: int(args.max_candidates)]

    source = saved.get("source", {})
    data_dir = args.data_dir.resolve() if args.data_dir is not None else Path(
        source.get("data_dir", stage00.DEFAULT_DATA_DIR)
    ).resolve()
    time_index = int(
        args.time_index
        if args.time_index is not None
        else source.get("time_index", stage00.DEFAULT_TIME_INDEX)
    )

    instance_path = data_dir / "instance_movie.npy"
    gt_path = data_dir / "gt_movie.npy"
    metadata_path = data_dir / "metadata.json"
    source_dir = data_dir / "stirnet_source"
    raw_path = source_dir / "raw_norm_target.npy"
    marker_path = source_dir / "marker_heatmap_target.npy"
    required = (instance_path, gt_path, metadata_path, raw_path, marker_path)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared STIR-Net scene is incomplete. Missing:\n  "
            + "\n  ".join(str(p) for p in missing)
        )

    _progress("[setup] Memory-mapping prepared full arrays ...")
    instance_movie = np.load(instance_path, mmap_mode="r")
    gt_movie = np.load(gt_path, mmap_mode="r")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    spacing = np.asarray(metadata["spacing_zyx_um"], dtype=np.float32)

    max_t = min(len(instance_movie), len(gt_movie)) - 1
    if not (0 <= time_index <= max_t):
        raise IndexError(f"time-index {time_index} outside [0, {max_t}]")

    current_full = np.asarray(instance_movie[time_index])
    gt_full = np.asarray(gt_movie[time_index])
    raw_full = np.load(raw_path, mmap_mode="r")
    marker_full = np.load(marker_path, mmap_mode="r")

    if current_full.shape != gt_full.shape:
        raise RuntimeError(
            f"Current/GT shape mismatch: {current_full.shape} vs {gt_full.shape}"
        )

    dref_value = saved.get("dref_um")
    if isinstance(dref_value, Tensor):
        dref_um = float(dref_value.item())
    elif dref_value is not None:
        dref_um = float(dref_value)
    else:
        dref_um = stage00.estimate_model_dref_um(
            current_full, tuple(float(v) for v in spacing)
        )

    candidates: list[tuple[tuple[slice, slice, slice], dict[str, Any]]] = []
    for ordinal, row in enumerate(summaries):
        row = dict(row)
        row["candidate_index"] = int(row.get("candidate_index", ordinal))
        crop = _pairs_to_slices(row["crop_slices_zyx"])
        candidates.append((crop, row))

    # Full-volume GT metadata is shared by every crop.
    _progress("[setup] Indexing full-volume GT bounding boxes / volumes ...")
    full_boxes = stage00._box_lookup(gt_full)
    positive = gt_full[gt_full > 0]
    if positive.size:
        ids, counts = np.unique(positive, return_counts=True)
        full_counts = {
            int(i): int(c) for i, c in zip(ids.tolist(), counts.tolist())
        }
    else:
        full_counts = {}

    first_crop, first_selection = candidates[0]
    first_build = stage00._expand_build_region(
        gt_full,
        first_crop,
        spacing,
        dref_um,
        args.context_margin_dref,
    )
    base_scene = stage00.Scene(
        current_full=current_full,
        gt_full=gt_full,
        raw_full=raw_full,
        marker_full=marker_full,
        spacing_zyx_um=spacing,
        dref_um=dref_um,
        core_slices=first_crop,
        build_slices=first_build,
        selection=dict(first_selection),
    )

    _progress(
        f"[setup] Ready in {_format_duration(time.perf_counter() - setup_start)} | "
        f"shape={tuple(gt_full.shape)} | candidates={len(candidates)} | "
        f"dref={dref_um:.5f} um"
    )
    return (
        base_scene,
        candidates,
        full_boxes,
        full_counts,
        data_dir,
        time_index,
    )


def make_candidate_scene(
    base_scene,
    crop: tuple[slice, slice, slice],
    selection: dict[str, Any],
    context_margin_dref: float,
):
    build = stage00._expand_build_region(
        base_scene.gt_full,
        crop,
        base_scene.spacing_zyx_um,
        base_scene.dref_um,
        context_margin_dref,
    )
    selection = dict(selection)
    selection["build_slices_zyx"] = _slice_pairs(build)
    return stage00.Scene(
        current_full=base_scene.current_full,
        gt_full=base_scene.gt_full,
        raw_full=base_scene.raw_full,
        marker_full=base_scene.marker_full,
        spacing_zyx_um=base_scene.spacing_zyx_um,
        dref_um=base_scene.dref_um,
        core_slices=crop,
        build_slices=build,
        selection=selection,
    )


# =============================================================================
# ORACLE CONTEXT CACHE
# =============================================================================


def _current_context_signature(base_cfg) -> dict[str, Any]:
    return {
        "format_version": CONTEXT_FORMAT_VERSION,
        "geometry": asdict(base_cfg.geometry),
        "partition_oracle_derivation": {
            "foreground_threshold": float(base_cfg.partition.foreground_threshold),
            "seed_sdf_weight": float(base_cfg.partition.seed_sdf_weight),
            "seed_head_weight": float(base_cfg.partition.seed_head_weight),
        },
    }


def _gt_boundary_meta(
    *,
    gt_crop: np.ndarray,
    crop: tuple[slice, slice, slice],
    full_shape: tuple[int, int, int],
    full_boxes: dict[int, tuple[np.ndarray, np.ndarray]],
    full_counts: dict[int, int],
    complete_ids: list[int],
) -> dict[str, Any]:
    complete = set(int(v) for v in complete_ids)
    crop_low = np.asarray([int(s.start) for s in crop], dtype=np.int64)
    crop_high = np.asarray([int(s.stop) for s in crop], dtype=np.int64)
    shape_arr = np.asarray(full_shape, dtype=np.int64)

    result: dict[str, Any] = {}
    ids = sorted(int(v) for v in np.unique(gt_crop) if int(v) > 0)
    for gt_id in ids:
        mask = gt_crop == gt_id
        visible_voxels = int(mask.sum())
        full_voxels = int(full_counts.get(gt_id, visible_voxels))
        bbox = full_boxes.get(gt_id)
        if bbox is None:
            low = crop_low.copy()
            high = crop_high.copy()
        else:
            low, high = bbox
        touches_true_boundary = bool(
            np.any(low <= 0) or np.any(high >= shape_arr)
        )
        cut_by_debug_crop = bool(gt_id not in complete)
        touches_crop_face = bool(
            mask[0].any()
            or mask[-1].any()
            or mask[:, 0].any()
            or mask[:, -1].any()
            or mask[:, :, 0].any()
            or mask[:, :, -1].any()
        )
        result[str(gt_id)] = {
            "gt_id": gt_id,
            "visible_voxels": visible_voxels,
            "full_visible_instance_voxels": full_voxels,
            "visible_fraction_of_full_gt": float(
                visible_voxels / max(full_voxels, 1)
            ),
            "complete_in_debug_crop": bool(gt_id in complete),
            "cut_by_debug_crop": cut_by_debug_crop,
            "touches_debug_crop_face": touches_crop_face,
            "touches_original_volume_boundary": touches_true_boundary,
            "true_volume_boundary_fully_represented": bool(
                touches_true_boundary and gt_id in complete
            ),
            "artificial_crop_partial": bool(
                cut_by_debug_crop and not touches_true_boundary
            ),
            "full_bbox_low_zyx": [int(v) for v in low.tolist()],
            "full_bbox_high_zyx": [int(v) for v in high.tolist()],
        }
    return result


def _target_batch_for_scene(scene, targets) -> dict[str, Tensor]:
    relative = stage00._relative_slices(scene.core_slices, scene.build_slices)
    cropped = stage00._cropped_target_tensors(targets, relative)
    return {
        name: value.unsqueeze(0).float().contiguous()
        for name, value in cropped.items()
    }


def context_cache_path(result_dir: Path, candidate_index: int) -> Path:
    return result_dir / "context_cache" / f"candidate_{candidate_index:02d}.pt"


def context_from_payload(payload: dict[str, Any]) -> CandidateContext:
    return CandidateContext(
        candidate_index=int(payload["candidate_index"]),
        crop_slices_zyx=[list(v) for v in payload["crop_slices_zyx"]],
        selection=dict(payload["selection"]),
        gt_labels=payload["gt_labels"].long(),
        foreground_mask=payload["foreground_mask"].bool(),
        surface_prob=payload["surface_prob"].float(),
        separator_prob=payload["separator_prob"].float(),
        seed_prob=payload["seed_prob"].float(),
        sdf_normalized=payload["sdf_normalized"].float(),
        spacing_um=payload["spacing_um"].float(),
        dref_um=payload["dref_um"].float(),
        complete_gt_ids=[int(v) for v in payload["complete_gt_ids"]],
        partial_gt_ids=[int(v) for v in payload["partial_gt_ids"]],
        gt_meta=dict(payload["gt_meta"]),
    )


def build_or_load_context(
    *,
    args: argparse.Namespace,
    base_cfg,
    base_scene,
    crop: tuple[slice, slice, slice],
    selection: dict[str, Any],
    full_boxes,
    full_counts,
) -> CandidateContext:
    index = int(selection["candidate_index"])
    path = context_cache_path(args.result_dir, index)
    signature = _current_context_signature(base_cfg)

    if path.exists() and not args.rebuild_contexts:
        payload = torch_load(path)
        if payload.get("signature") == signature:
            return context_from_payload(payload)
        _progress(
            f"[context {index:02d}] cache signature changed; rebuilding."
        )

    scene = make_candidate_scene(
        base_scene,
        crop,
        selection,
        args.context_margin_dref,
    )
    build_start = time.perf_counter()
    _progress(
        f"[context {index:02d}] building oracle production targets | "
        f"shape={_slice_shape(crop)} ..."
    )
    targets = stage00.build_targets(scene)
    target_batch = _target_batch_for_scene(scene, targets)
    oracle = stage03.oracle_geometry_from_targets(target_batch)
    cache = stage03.build_geometry_derived_cache(
        oracle, base_cfg.partition, padding_mask=None
    )

    gt_crop = np.asarray(base_scene.gt_full[crop]).astype(np.int32, copy=True)
    complete_ids, partial_ids = stage00._complete_and_partial_ids(
        base_scene.gt_full, crop
    )
    meta = _gt_boundary_meta(
        gt_crop=gt_crop,
        crop=crop,
        full_shape=tuple(int(v) for v in base_scene.gt_full.shape),
        full_boxes=full_boxes,
        full_counts=full_counts,
        complete_ids=complete_ids,
    )

    payload = {
        "format_version": CONTEXT_FORMAT_VERSION,
        "signature": signature,
        "candidate_index": index,
        "crop_slices_zyx": _slice_pairs(crop),
        "selection": dict(selection),
        "gt_labels": torch.from_numpy(gt_crop).long(),
        "foreground_mask": cache.foreground_mask[0].cpu().bool(),
        "surface_prob": cache.surface_prob[0, 0].cpu().float(),
        "separator_prob": cache.separator_prob[0, 0].cpu().float(),
        "seed_prob": cache.seed_prob[0, 0].cpu().float(),
        "sdf_normalized": cache.sdf_normalized[0, 0].cpu().float(),
        "spacing_um": scene.spacing_zyx_um.copy()
        if isinstance(scene.spacing_zyx_um, np.ndarray)
        else np.asarray(scene.spacing_zyx_um, dtype=np.float32),
        "dref_um": float(scene.dref_um),
        "complete_gt_ids": complete_ids,
        "partial_gt_ids": partial_ids,
        "gt_meta": meta,
    }
    payload["spacing_um"] = torch.as_tensor(
        payload["spacing_um"], dtype=torch.float32
    )
    payload["dref_um"] = torch.tensor(float(scene.dref_um), dtype=torch.float32)
    atomic_torch_save(path, payload)
    _progress(
        f"[context {index:02d}] done in "
        f"{_format_duration(time.perf_counter() - build_start)} | "
        f"visible GT={len(meta)} | complete={len(complete_ids)} | "
        f"partial={len(partial_ids)}"
    )
    return context_from_payload(payload)


# =============================================================================
# MARKER RESCUE CANDIDATE (CALIBRATION-ONLY)
# =============================================================================


def separator_core_marker_rescue(
    markers: np.ndarray,
    score: np.ndarray,
    foreground: np.ndarray,
    separator_prob: np.ndarray,
    *,
    separator_cutoff: float,
    min_component_voxels: int,
    min_score: float,
    max_markers: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Ensure marker support in separator-isolated foreground cores.

    This algorithm uses only predicted/oracle dense fields available at
    inference; it does NOT use GT identities.  False splits are intentionally
    tolerated because the RAG can merge them later.
    """
    out = markers.astype(np.int32, copy=True)
    core = (
        foreground.astype(bool)
        & (separator_prob < float(separator_cutoff))
    )
    structure = ndi.generate_binary_structure(3, 1)
    components, count = ndi.label(core, structure=structure)
    objects = ndi.find_objects(components)

    next_id = int(out.max()) + 1
    added: list[dict[str, Any]] = []
    skipped_small = 0
    skipped_low_score = 0
    skipped_has_marker = 0
    exhausted = False

    for component_id, bbox in enumerate(objects, 1):
        if bbox is None:
            continue
        local_component = components[bbox] == component_id
        voxel_count = int(local_component.sum())
        if voxel_count < int(min_component_voxels):
            skipped_small += 1
            continue
        marker_view = out[bbox]
        if np.any(marker_view[local_component] > 0):
            skipped_has_marker += 1
            continue
        score_view = score[bbox]
        values = score_view[local_component]
        if values.size == 0:
            continue
        best_value = float(values.max())
        if best_value < float(min_score):
            skipped_low_score += 1
            continue
        if next_id > int(max_markers):
            exhausted = True
            break

        coords = np.argwhere(local_component)
        best_local = coords[int(np.argmax(values))]
        global_coord = tuple(
            int(best_local[axis] + int(bbox[axis].start))
            for axis in range(3)
        )
        out[global_coord] = next_id
        added.append(
            {
                "marker_id": int(next_id),
                "coord_zyx": [int(v) for v in global_coord],
                "score": best_value,
                "core_component_voxels": voxel_count,
            }
        )
        next_id += 1

    return out, {
        "mode": "separator_core",
        "separator_cutoff": float(separator_cutoff),
        "min_component_voxels": int(min_component_voxels),
        "min_score": float(min_score),
        "added_marker_count": len(added),
        "added_markers": added,
        "skipped_small_components": skipped_small,
        "skipped_low_score_components": skipped_low_score,
        "skipped_components_with_marker": skipped_has_marker,
        "max_markers_exhausted": exhausted,
    }


# =============================================================================
# ONE CONFIGURATION x ONE CANDIDATE
# =============================================================================


def cfg_for_spec(base_cfg, spec: SweepSpec):
    cfg = copy.deepcopy(base_cfg)
    p = cfg.partition
    p.watershed_backend = spec.backend
    p.seed_min_distance_dref = float(spec.seed_min_distance_dref)
    p.seed_threshold = float(spec.seed_threshold)
    p.min_supervoxel_voxels = int(spec.min_supervoxel_voxels)
    p.seed_sdf_weight = float(spec.seed_sdf_weight)
    p.seed_head_weight = float(spec.seed_head_weight)
    p.watershed_separator_weight = float(spec.watershed_separator_weight)
    p.watershed_surface_weight = float(spec.watershed_surface_weight)
    p.watershed_sdf_weight = float(spec.watershed_sdf_weight)
    return cfg


def _derive_score_energy(
    context: CandidateContext,
    spec: SweepSpec,
) -> tuple[Tensor, Tensor]:
    score = (
        float(spec.seed_sdf_weight) * context.sdf_normalized
        + float(spec.seed_head_weight) * context.seed_prob
    ) * (1.0 - context.separator_prob)
    energy = (
        float(spec.watershed_separator_weight) * context.separator_prob
        + float(spec.watershed_surface_weight) * context.surface_prob
        + float(spec.watershed_sdf_weight) * (1.0 - context.sdf_normalized)
    )
    return score.float(), energy.float()


def _marker_cross_gt_components(
    marker_summary: dict[str, Any],
    *,
    min_voxels: int,
) -> list[dict[str, Any]]:
    result = []
    for row in marker_summary["markers"]:
        positive = [
            x for x in row["gt_overlaps"] if int(x["voxels"]) >= min_voxels
        ]
        if len(positive) >= 2:
            result.append(row)
    return result


def _per_gt_score_stats(
    context: CandidateContext,
    score_np: np.ndarray,
    marker_summary: dict[str, Any],
    clean_summary: dict[str, Any],
) -> dict[str, Any]:
    gt = context.gt_labels.numpy().astype(np.int32, copy=False)
    seed_np = context.seed_prob.numpy()
    sdf_np = context.sdf_normalized.numpy()
    sep_np = context.separator_prob.numpy()
    marker_counts = marker_summary["per_gt_marker_count"]
    result: dict[str, Any] = {}

    for key, meta in context.gt_meta.items():
        gt_id = int(key)
        mask = gt == gt_id
        if not mask.any():
            continue
        scores = score_np[mask]
        best_flat_local = int(np.argmax(scores))
        coords = np.argwhere(mask)
        best_coord = coords[best_flat_local]
        idx = tuple(int(v) for v in best_coord)
        per_gt_sv = clean_summary["per_gt"].get(str(gt_id), {})
        result[str(gt_id)] = {
            **meta,
            "marker_count": int(marker_counts.get(str(gt_id), 0)),
            "max_marker_score": float(scores.max()),
            "max_seed_prob": float(seed_np[mask].max()),
            "max_sdf_normalized": float(sdf_np[mask].max()),
            "separator_at_best_score": float(sep_np[idx]),
            "best_score_coord_zyx": [int(v) for v in best_coord.tolist()],
            "coverage": float(per_gt_sv.get("coverage", 0.0)),
            "meaningful_supervoxel_count": int(
                per_gt_sv.get("meaningful_supervoxel_count", 0)
            ),
            "recoverable_by_merging": bool(
                per_gt_sv.get("recoverable_by_merging", False)
            ),
            "unsafe_overlapping_supervoxel_ids": list(
                per_gt_sv.get("unsafe_overlapping_supervoxel_ids", [])
            ),
        }
    return result


def _aggregate_cell_categories(per_gt: dict[str, Any]) -> dict[str, Any]:
    categories = {
        "all_visible": [],
        "complete_in_debug_crop": [],
        "artificial_crop_partial": [],
        "true_volume_boundary": [],
        "complete_interior": [],
    }
    for row in per_gt.values():
        categories["all_visible"].append(row)
        if row["complete_in_debug_crop"]:
            categories["complete_in_debug_crop"].append(row)
        if row["artificial_crop_partial"]:
            categories["artificial_crop_partial"].append(row)
        if row["touches_original_volume_boundary"]:
            categories["true_volume_boundary"].append(row)
        if (
            row["complete_in_debug_crop"]
            and not row["touches_original_volume_boundary"]
        ):
            categories["complete_interior"].append(row)

    out: dict[str, Any] = {}
    for name, rows in categories.items():
        recoverable = sum(bool(row["recoverable_by_merging"]) for row in rows)
        with_marker = sum(int(row["marker_count"]) > 0 for row in rows)
        coverages = [float(row["coverage"]) for row in rows]
        sv_counts = [int(row["meaningful_supervoxel_count"]) for row in rows]
        out[name] = {
            "cell_count": len(rows),
            "marker_recall": float(with_marker / len(rows)) if rows else 1.0,
            "recoverable_fraction": (
                float(recoverable / len(rows)) if rows else 1.0
            ),
            "min_coverage": _safe_min(coverages),
            "mean_meaningful_supervoxels_per_gt": (
                float(np.mean(sv_counts)) if sv_counts else 0.0
            ),
            "max_meaningful_supervoxels_per_gt": max(sv_counts) if sv_counts else 0,
        }
    return out


def _unsafe_node_details(
    clean_summary: dict[str, Any],
    context: CandidateContext,
    marker_summary: dict[str, Any],
    raw_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    marker_counts = marker_summary["per_gt_marker_count"]
    raw_unsafe_by_ids: list[set[int]] = []
    for row in raw_summary["nodes"]:
        if row["cross_gt_unsafe"]:
            raw_unsafe_by_ids.append(set(int(v) for v in row["meaningful_gt_ids"]))

    details = []
    for row in clean_summary["nodes"]:
        if not row["cross_gt_unsafe"]:
            continue
        ids = [int(v) for v in row["meaningful_gt_ids"]]
        idset = set(ids)
        meta_rows = [context.gt_meta.get(str(v), {}) for v in ids]
        missing = [
            gt_id for gt_id in ids if int(marker_counts.get(str(gt_id), 0)) <= 0
        ]
        existed_raw = any(idset.issubset(raw_ids) or raw_ids.issubset(idset) for raw_ids in raw_unsafe_by_ids)
        if missing:
            mechanism = "missing_marker"
        elif not existed_raw:
            mechanism = "tiny_region_cleanup"
        else:
            mechanism = "watershed_propagation"
        details.append(
            {
                "supervoxel_id": int(row["supervoxel_id"]),
                "meaningful_gt_ids": ids,
                "missing_marker_gt_ids": missing,
                "likely_mechanism": mechanism,
                "involves_true_volume_boundary": any(
                    bool(meta.get("touches_original_volume_boundary", False))
                    for meta in meta_rows
                ),
                "involves_artificial_crop_partial": any(
                    bool(meta.get("artificial_crop_partial", False))
                    for meta in meta_rows
                ),
                "all_complete_in_debug_crop": all(
                    bool(meta.get("complete_in_debug_crop", False))
                    for meta in meta_rows
                ),
            }
        )
    return details


def run_candidate_spec(
    *,
    context: CandidateContext,
    base_cfg,
    spec: SweepSpec,
    return_artifact: bool = False,
) -> tuple[dict[str, Any], dict[str, Tensor] | None]:
    run_start = time.perf_counter()
    cfg = cfg_for_spec(base_cfg, spec)
    score, energy = _derive_score_energy(context, spec)
    fg = context.foreground_mask

    radius_um = float(spec.seed_min_distance_dref) * float(context.dref_um.item())
    if spec.backend == "fast":
        markers = stage03.build_markers_fast(
            score,
            fg,
            context.spacing_um,
            radius_um,
            float(spec.seed_threshold),
            int(cfg.partition.max_supervoxels),
        )
    elif spec.backend == "reference":
        markers = stage03.build_markers(
            score.numpy(),
            fg.numpy(),
            context.spacing_um.numpy().astype(np.float32),
            radius_um,
            float(spec.seed_threshold),
            int(cfg.partition.max_supervoxels),
        )
    else:
        raise ValueError(f"Unknown backend: {spec.backend}")

    rescue_info: dict[str, Any] = {"mode": "none", "added_marker_count": 0}
    if spec.marker_rescue == "separator_core":
        markers, rescue_info = separator_core_marker_rescue(
            markers,
            score.numpy(),
            fg.numpy(),
            context.separator_prob.numpy(),
            separator_cutoff=spec.rescue_separator_cutoff,
            min_component_voxels=spec.rescue_min_component_voxels,
            min_score=spec.rescue_min_score,
            max_markers=int(cfg.partition.max_supervoxels),
        )
    elif spec.marker_rescue != "none":
        raise ValueError(f"Unknown marker_rescue={spec.marker_rescue!r}")

    raw, clean = stage03.run_exact_watershed(
        energy,
        markers,
        fg,
        cfg,
    )

    gt_np = context.gt_labels.numpy().astype(np.int32, copy=False)
    marker_summary = stage03.marker_diagnostics(
        markers,
        gt_np,
        context.complete_gt_ids,
    )
    raw_summary, raw_maps = stage03.supervoxel_diagnostics(
        raw,
        gt_np,
        context.complete_gt_ids,
        cfg,
    )
    clean_summary, clean_maps = stage03.supervoxel_diagnostics(
        clean,
        gt_np,
        context.complete_gt_ids,
        cfg,
    )

    per_gt = _per_gt_score_stats(
        context,
        score.numpy(),
        marker_summary,
        clean_summary,
    )
    categories = _aggregate_cell_categories(per_gt)
    unsafe_details = _unsafe_node_details(
        clean_summary,
        context,
        marker_summary,
        raw_summary,
    )
    marker_bridges = _marker_cross_gt_components(
        marker_summary,
        min_voxels=stage03.MEANINGFUL_OVERLAP_MIN_VOXELS,
    )

    mechanisms = {
        "missing_marker": 0,
        "marker_bridge": len(marker_bridges),
        "watershed_propagation": 0,
        "tiny_region_cleanup": 0,
    }
    for row in unsafe_details:
        mechanisms[row["likely_mechanism"]] += 1

    # Recoverability is only a fair hard requirement when the full visible GT
    # instance is represented inside this diagnostic crop.  Cross-GT safety,
    # however, remains strict across ALL visible voxels, including artificial
    # crop cuts, because the goal is to remove every previously reported unsafe
    # scene as a stress test.  True acquisition-boundary cells are therefore
    # hard-required whenever their full visible instance is contained in the
    # crop (complete_in_debug_crop=True).
    production_required_rows = [
        row for row in per_gt.values() if row["complete_in_debug_crop"]
    ]
    production_required_recoverable = all(
        bool(row["recoverable_by_merging"]) for row in production_required_rows
    )
    stress_all_visible_recoverable = all(
        bool(row["recoverable_by_merging"]) for row in per_gt.values()
    )

    cross_raw = int(raw_summary["cross_gt_unsafe_supervoxel_count"])
    cross_clean = int(clean_summary["cross_gt_unsafe_supervoxel_count"])
    production_safe = bool(cross_clean == 0 and production_required_recoverable)
    stress_safe = bool(cross_clean == 0 and stress_all_visible_recoverable)

    result = {
        "format_version": RUN_FORMAT_VERSION,
        "candidate_index": int(context.candidate_index),
        "spec": spec.key_payload(),
        "spec_hash": spec.hash(),
        "elapsed_seconds": float(time.perf_counter() - run_start),
        "marker_count": int(marker_summary["marker_count"]),
        "marker_bridge_count": len(marker_bridges),
        "rescue": rescue_info,
        "raw_supervoxel_count": int(raw_summary["supervoxel_count"]),
        "clean_supervoxel_count": int(clean_summary["supervoxel_count"]),
        "cleanup_removed_region_count": max(
            int(raw_summary["supervoxel_count"])
            - int(clean_summary["supervoxel_count"]),
            0,
        ),
        "raw_cross_gt_supervoxel_count": cross_raw,
        "clean_cross_gt_supervoxel_count": cross_clean,
        "cleanup_introduced_cross_gt_count": max(cross_clean - cross_raw, 0),
        "production_safe": production_safe,
        "stress_safe_all_visible": stress_safe,
        "rag_valid_fraction": float(
            clean_summary["rag_valid_supervoxel_fraction"]
        ),
        "cell_categories": categories,
        "per_gt": per_gt,
        "unsafe_supervoxels": unsafe_details,
        "failure_mechanisms": mechanisms,
    }

    artifact = None
    if return_artifact:
        artifact = {
            "markers": torch.from_numpy(markers.astype(np.int32, copy=False)),
            "raw_watershed": torch.from_numpy(raw.astype(np.int32, copy=False)),
            "supervoxels": torch.from_numpy(clean.astype(np.int32, copy=False)),
            "unsafe_mask": torch.from_numpy(
                clean_maps["unsafe_supervoxel_mask"].astype(np.uint8, copy=False)
            ),
            "seed_score": score.half(),
            "watershed_energy": energy.half(),
        }
    return result, artifact


# =============================================================================
# RUN CACHE / AGGREGATION
# =============================================================================


def run_cache_path(result_dir: Path, spec: SweepSpec, candidate_index: int) -> Path:
    return (
        result_dir
        / "run_cache"
        / spec.slug()
        / f"candidate_{candidate_index:02d}.json"
    )


def load_or_run_candidate(
    *,
    args: argparse.Namespace,
    context: CandidateContext,
    base_cfg,
    spec: SweepSpec,
) -> tuple[dict[str, Any], bool]:
    path = run_cache_path(args.result_dir, spec, context.candidate_index)
    if path.exists() and not args.rebuild_runs:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data.get("format_version") == RUN_FORMAT_VERSION
            and data.get("spec_hash") == spec.hash()
        ):
            return data, True

    data, _ = run_candidate_spec(
        context=context,
        base_cfg=base_cfg,
        spec=spec,
        return_artifact=False,
    )
    atomic_json(path, data)
    return data, False


def aggregate_spec(spec: SweepSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate an empty result list")

    visible_cells = sum(
        int(row["cell_categories"]["all_visible"]["cell_count"]) for row in rows
    )
    meaningful_sv_sum = 0.0
    per_gt_sv_counts: list[int] = []
    missing_marker_cells = 0
    true_boundary_visible_cells = 0
    true_boundary_cells = 0
    true_boundary_unrecoverable = 0
    true_boundary_missing_markers = 0
    artificial_partial_unrecoverable = 0

    for row in rows:
        for gt in row["per_gt"].values():
            count = int(gt["meaningful_supervoxel_count"])
            meaningful_sv_sum += count
            per_gt_sv_counts.append(count)
            if int(gt["marker_count"]) <= 0:
                missing_marker_cells += 1
            if gt["touches_original_volume_boundary"]:
                true_boundary_visible_cells += 1
            if gt.get("true_volume_boundary_fully_represented", False):
                true_boundary_cells += 1
                if not gt["recoverable_by_merging"]:
                    true_boundary_unrecoverable += 1
                if int(gt["marker_count"]) <= 0:
                    true_boundary_missing_markers += 1
            if gt["artificial_crop_partial"] and not gt["recoverable_by_merging"]:
                artificial_partial_unrecoverable += 1

    mean_sv_per_gt = meaningful_sv_sum / max(visible_cells, 1)
    max_sv_per_gt = max(per_gt_sv_counts) if per_gt_sv_counts else 0
    overfragmented_cells = sum(
        count > SOFT_MAX_MEANINGFUL_SV_PER_GT for count in per_gt_sv_counts
    )

    mechanism_totals = {
        key: sum(int(row["failure_mechanisms"].get(key, 0)) for row in rows)
        for key in (
            "missing_marker",
            "marker_bridge",
            "watershed_propagation",
            "tiny_region_cleanup",
        )
    }

    production_unsafe_scenes = sum(not bool(row["production_safe"]) for row in rows)
    stress_unsafe_scenes = sum(not bool(row["stress_safe_all_visible"]) for row in rows)
    cross_total = sum(int(row["clean_cross_gt_supervoxel_count"]) for row in rows)
    raw_cross_total = sum(int(row["raw_cross_gt_supervoxel_count"]) for row in rows)
    cleanup_introduced = sum(int(row["cleanup_introduced_cross_gt_count"]) for row in rows)

    return {
        "spec": spec.key_payload(),
        "spec_hash": spec.hash(),
        "candidate_count": len(rows),
        "production_safe_scene_count": len(rows) - production_unsafe_scenes,
        "production_unsafe_scene_count": production_unsafe_scenes,
        "stress_safe_scene_count": len(rows) - stress_unsafe_scenes,
        "stress_unsafe_scene_count": stress_unsafe_scenes,
        "clean_cross_gt_supervoxel_total": cross_total,
        "raw_cross_gt_supervoxel_total": raw_cross_total,
        "cleanup_introduced_cross_gt_total": cleanup_introduced,
        "missing_marker_visible_cell_total": missing_marker_cells,
        "true_volume_boundary_visible_cell_count": true_boundary_visible_cells,
        "true_volume_boundary_fully_represented_cell_count": true_boundary_cells,
        "true_volume_boundary_unrecoverable_count": true_boundary_unrecoverable,
        "true_volume_boundary_missing_marker_count": true_boundary_missing_markers,
        "artificial_crop_partial_unrecoverable_count": artificial_partial_unrecoverable,
        "mean_meaningful_supervoxels_per_visible_gt": float(mean_sv_per_gt),
        "max_meaningful_supervoxels_per_visible_gt": int(max_sv_per_gt),
        "overfragmented_visible_gt_count": int(overfragmented_cells),
        "mean_clean_supervoxel_count": float(
            np.mean([int(row["clean_supervoxel_count"]) for row in rows])
        ),
        "mean_marker_count": float(
            np.mean([int(row["marker_count"]) for row in rows])
        ),
        "mean_rag_valid_fraction": float(
            np.mean([float(row["rag_valid_fraction"]) for row in rows])
        ),
        "mean_run_seconds": float(
            np.mean([float(row["elapsed_seconds"]) for row in rows])
        ),
        "failure_mechanisms": mechanism_totals,
        "unsafe_candidate_indices": [
            int(row["candidate_index"])
            for row in rows
            if not bool(row["production_safe"])
        ],
        "stress_unsafe_candidate_indices": [
            int(row["candidate_index"])
            for row in rows
            if not bool(row["stress_safe_all_visible"])
        ],
    }


def aggregate_rank_key(agg: dict[str, Any]) -> tuple:
    """Lower tuple is better. Safety dominates compactness/runtime."""
    mean_sv = float(agg["mean_meaningful_supervoxels_per_visible_gt"])
    fragmentation_distance = abs(mean_sv - TARGET_MEAN_SV_PER_VISIBLE_GT)
    backend_penalty = 0 if agg["spec"]["backend"] == "fast" else 1
    rescue_penalty = 0 if agg["spec"]["marker_rescue"] == "none" else 1
    return (
        int(agg["production_unsafe_scene_count"]),
        int(agg["clean_cross_gt_supervoxel_total"]),
        int(agg["true_volume_boundary_unrecoverable_count"]),
        int(agg["stress_unsafe_scene_count"]),
        int(agg["missing_marker_visible_cell_total"]),
        int(agg["overfragmented_visible_gt_count"]),
        float(fragmentation_distance),
        -float(agg["mean_rag_valid_fraction"]),
        rescue_penalty,
        backend_penalty,
        float(agg["mean_run_seconds"]),
    )


def print_aggregate_table(aggregates: list[dict[str, Any]], title: str, limit: int = 20) -> None:
    ordered = sorted(aggregates, key=aggregate_rank_key)
    _progress("\n" + "=" * 174)
    _progress(title)
    _progress("=" * 174)
    _progress(
        f"{'rank':>4s} {'name':38s} {'phase':9s} {'safe':>7s} {'xGT':>4s} "
        f"{'stress':>7s} {'missMk':>6s} {'trueBU':>6s} {'clnX':>5s} "
        f"{'SV/GT':>6s} {'maxSV':>5s} {'RAGv':>6s} {'sec':>6s}"
    )
    _progress("-" * 174)
    for rank, agg in enumerate(ordered[:limit], 1):
        spec = agg["spec"]
        _progress(
            f"{rank:4d} {spec['name'][:38]:38s} {spec['phase'][:9]:9s} "
            f"{agg['production_safe_scene_count']:2d}/{agg['candidate_count']:<2d} "
            f"{agg['clean_cross_gt_supervoxel_total']:4d} "
            f"{agg['stress_safe_scene_count']:2d}/{agg['candidate_count']:<2d} "
            f"{agg['missing_marker_visible_cell_total']:6d} "
            f"{agg['true_volume_boundary_unrecoverable_count']:6d} "
            f"{agg['cleanup_introduced_cross_gt_total']:5d} "
            f"{agg['mean_meaningful_supervoxels_per_visible_gt']:6.2f} "
            f"{agg['max_meaningful_supervoxels_per_visible_gt']:5d} "
            f"{agg['mean_rag_valid_fraction']:6.3f} "
            f"{agg['mean_run_seconds']:6.1f}"
        )
    _progress("=" * 174)


# =============================================================================
# SEARCH SPACE
# =============================================================================


def spec_from_base(base_cfg, *, name: str, phase: str, **changes) -> SweepSpec:
    p = base_cfg.partition
    values = {
        "name": name,
        "phase": phase,
        "backend": p.watershed_backend,
        "seed_min_distance_dref": p.seed_min_distance_dref,
        "seed_threshold": p.seed_threshold,
        "min_supervoxel_voxels": p.min_supervoxel_voxels,
        "seed_sdf_weight": p.seed_sdf_weight,
        "seed_head_weight": p.seed_head_weight,
        "watershed_separator_weight": p.watershed_separator_weight,
        "watershed_surface_weight": p.watershed_surface_weight,
        "watershed_sdf_weight": p.watershed_sdf_weight,
        "marker_rescue": "none",
        "rescue_separator_cutoff": 0.50,
        "rescue_min_component_voxels": 8,
        "rescue_min_score": 0.10,
    }
    values.update(changes)
    return SweepSpec(**values)


def structural_specs(base_cfg) -> list[SweepSpec]:
    specs: list[SweepSpec] = []
    p = base_cfg.partition

    specs.append(spec_from_base(base_cfg, name="baseline_current", phase="A_struct"))

    # Fast backend: first isolate marker spacing under current cleanup.
    for radius in (0.30, 0.25, 0.20):
        specs.append(
            spec_from_base(
                base_cfg,
                name=f"fast_r{radius:.2f}_cleanup{p.min_supervoxel_voxels}",
                phase="A_struct",
                backend="fast",
                seed_min_distance_dref=radius,
            )
        )

    # Cleanup disabled: determines whether the current 4-voxel merge rule is
    # creating irreversible cross-cell supervoxels.
    for radius in (0.35, 0.30, 0.25, 0.20):
        specs.append(
            spec_from_base(
                base_cfg,
                name=f"fast_r{radius:.2f}_nocleanup",
                phase="A_struct",
                backend="fast",
                seed_min_distance_dref=radius,
                min_supervoxel_voxels=1,
            )
        )

    # Reference NMS uses the true physical ellipsoid rather than the fast
    # rectangular max-pool window.  Keep cleanup off so backend differences are
    # not confounded with tiny-region merging.
    specs.append(
        spec_from_base(
            base_cfg,
            name="reference_r0.35_cleanup_current",
            phase="A_struct",
            backend="reference",
            seed_min_distance_dref=0.35,
        )
    )
    for radius in (0.35, 0.30, 0.25, 0.20):
        specs.append(
            spec_from_base(
                base_cfg,
                name=f"reference_r{radius:.2f}_nocleanup",
                phase="A_struct",
                backend="reference",
                seed_min_distance_dref=radius,
                min_supervoxel_voxels=1,
            )
        )
    return _dedupe_specs(specs)


def _dedupe_specs(specs: list[SweepSpec]) -> list[SweepSpec]:
    seen = set()
    result = []
    for spec in specs:
        key = spec.hash()
        if key in seen:
            continue
        seen.add(key)
        result.append(spec)
    return result


def threshold_specs(base_cfg, top_structural: list[dict[str, Any]]) -> list[SweepSpec]:
    specs = []
    for agg in top_structural[:3]:
        base_spec = SweepSpec(**agg["spec"])
        for threshold in (0.35, 0.30, 0.25):
            specs.append(
                replace(
                    base_spec,
                    name=f"{base_spec.name}_thr{threshold:.2f}",
                    phase="B_threshold",
                    seed_threshold=threshold,
                )
            )
    return _dedupe_specs(specs)


def rescue_specs(best: SweepSpec) -> list[SweepSpec]:
    specs = []
    for cutoff in (0.35, 0.50, 0.65):
        for rescue_score in (0.10, 0.20):
            specs.append(
                replace(
                    best,
                    name=(
                        f"{best.name}_sepRescue{cutoff:.2f}_score{rescue_score:.2f}"
                    ),
                    phase="C_rescue",
                    marker_rescue="separator_core",
                    rescue_separator_cutoff=cutoff,
                    rescue_min_component_voxels=8,
                    rescue_min_score=rescue_score,
                )
            )
    return _dedupe_specs(specs)


def propagation_specs(best: SweepSpec) -> list[SweepSpec]:
    weight_sets = (
        (0.75, 0.10, 0.15),
        (0.80, 0.10, 0.10),
        (0.85, 0.05, 0.10),
    )
    return [
        replace(
            best,
            name=f"{best.name}_energy_{sep:.2f}_{surf:.2f}_{sdf:.2f}",
            phase="C_energy",
            watershed_separator_weight=sep,
            watershed_surface_weight=surf,
            watershed_sdf_weight=sdf,
        )
        for sep, surf, sdf in weight_sets
    ]


def seed_weight_specs(best: SweepSpec) -> list[SweepSpec]:
    weight_sets = (
        (0.40, 0.60),
        (0.25, 0.75),
        (0.00, 1.00),
    )
    return [
        replace(
            best,
            name=f"{best.name}_seedWeights_{sdf:.2f}_{seed:.2f}",
            phase="C_seedwt",
            seed_sdf_weight=sdf,
            seed_head_weight=seed,
        )
        for sdf, seed in weight_sets
    ]


# =============================================================================
# TIME-BUDGETED SCHEDULER
# =============================================================================


def estimate_run_seconds(completed_rows: list[dict[str, Any]]) -> float:
    if not completed_rows:
        return 24.0
    values = [float(row["elapsed_seconds"]) for row in completed_rows[-24:]]
    return max(float(np.mean(values)), 3.0)


def can_afford_runs(
    budget: TimeBudget,
    n_runs: int,
    completed_rows: list[dict[str, Any]],
    *,
    multiplier: float = 1.15,
) -> bool:
    estimate = estimate_run_seconds(completed_rows) * n_runs * multiplier
    return estimate <= budget.usable_remaining


def save_progress(
    args: argparse.Namespace,
    *,
    phase: str,
    budget: TimeBudget,
    aggregates: list[dict[str, Any]],
    completed_run_count: int,
) -> None:
    ordered = sorted(aggregates, key=aggregate_rank_key) if aggregates else []
    atomic_json(
        args.result_dir / "progress.json",
        {
            "format_version": REPORT_FORMAT_VERSION,
            "phase": phase,
            "elapsed_seconds": budget.elapsed,
            "remaining_seconds": budget.remaining,
            "completed_run_count": completed_run_count,
            "best_so_far": ordered[0] if ordered else None,
            "aggregate_count": len(aggregates),
        },
    )


def run_spec_on_indices(
    *,
    args: argparse.Namespace,
    spec: SweepSpec,
    contexts_by_index: dict[int, CandidateContext],
    indices: list[int],
    base_cfg,
    budget: TimeBudget,
    completed_rows: list[dict[str, Any]],
    phase_label: str,
) -> tuple[list[dict[str, Any]], bool]:
    results = []
    total = len(indices)
    for position, index in enumerate(indices, 1):
        # Cached runs are cheap, so check cache even if budget is nearly over.
        cache_path = run_cache_path(args.result_dir, spec, index)
        cached_exists = cache_path.exists() and not args.rebuild_runs
        if not cached_exists and not can_afford_runs(
            budget, 1, completed_rows, multiplier=1.05
        ):
            _progress(
                f"[{phase_label}] Budget guard: not starting another uncached run."
            )
            return results, False

        context = contexts_by_index[index]
        run_start = time.perf_counter()
        row, cached = load_or_run_candidate(
            args=args,
            context=context,
            base_cfg=base_cfg,
            spec=spec,
        )
        results.append(row)
        completed_rows.append(row)
        state = "cache" if cached else _format_duration(time.perf_counter() - run_start)
        _progress(
            f"[{phase_label}] {spec.name} | cand {index:02d} "
            f"[{position:02d}/{total:02d}] | {state:>7s} | "
            f"SV={row['clean_supervoxel_count']:2d} "
            f"xGT={row['clean_cross_gt_supervoxel_count']:2d} "
            f"safe={str(row['production_safe']):5s} | "
            f"remaining={_format_duration(budget.remaining)}"
        )
    return results, True


# =============================================================================
# FINAL BEFORE/AFTER ARTIFACTS + VISUALIZATION
# =============================================================================


def save_before_after_artifacts(
    *,
    args: argparse.Namespace,
    base_scene,
    contexts_by_index: dict[int, CandidateContext],
    base_cfg,
    baseline: SweepSpec,
    best: SweepSpec,
    baseline_rows: list[dict[str, Any]],
) -> Path | None:
    unsafe_indices = [
        int(row["candidate_index"])
        for row in baseline_rows
        if not bool(row["production_safe"])
        or int(row["clean_cross_gt_supervoxel_count"]) > 0
    ]
    if not unsafe_indices:
        return None

    cases: dict[str, Any] = {}
    for index in unsafe_indices:
        context = contexts_by_index[index]
        _, baseline_art = run_candidate_spec(
            context=context,
            base_cfg=base_cfg,
            spec=baseline,
            return_artifact=True,
        )
        best_row, best_art = run_candidate_spec(
            context=context,
            base_cfg=base_cfg,
            spec=best,
            return_artifact=True,
        )
        assert baseline_art is not None and best_art is not None
        crop = _pairs_to_slices(context.crop_slices_zyx)
        raw = np.asarray(base_scene.raw_full[crop]).astype(np.float32, copy=True)
        current = np.asarray(base_scene.current_full[crop]).astype(np.int32, copy=True)
        cases[str(index)] = {
            "candidate_index": index,
            "selection": context.selection,
            "spacing_um": context.spacing_um,
            "raw": torch.from_numpy(raw).float(),
            "current_labels": torch.from_numpy(current).long(),
            "gt_labels": context.gt_labels.long(),
            "baseline_supervoxels": baseline_art["supervoxels"].long(),
            "best_supervoxels": best_art["supervoxels"].long(),
            "best_markers": best_art["markers"].long(),
            "best_unsafe_mask": best_art["unsafe_mask"].to(torch.uint8),
            "best_result_summary": best_row,
        }

    path = args.result_dir / "before_after_unsafe_cases.pt"
    atomic_torch_save(
        path,
        {
            "format_version": REPORT_FORMAT_VERSION,
            "baseline_spec": baseline.key_payload(),
            "best_spec": best.key_payload(),
            "cases": cases,
        },
    )
    return path


def visualize_candidate_artifact(args: argparse.Namespace, candidate_index: int) -> None:
    path = args.result_dir / "before_after_unsafe_cases.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run Stage 05 calibration first."
        )
    payload = torch_load(path)
    key = str(int(candidate_index))
    if key not in payload["cases"]:
        available = sorted(int(v) for v in payload["cases"].keys())
        raise IndexError(
            f"Candidate {candidate_index} is not in the baseline-unsafe artifact. "
            f"Available: {available}"
        )
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError("Napari is required for --visualize-candidate") from exc

    case = payload["cases"][key]
    scale = tuple(float(v) for v in case["spacing_um"].tolist())
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(
        case["raw"].numpy(),
        name="Raw",
        scale=scale,
        colormap="gray",
        visible=True,
    )
    viewer.add_labels(
        case["gt_labels"].numpy(),
        name="GT labels",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        case["baseline_supervoxels"].numpy(),
        name="Baseline supervoxels",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        case["best_supervoxels"].numpy(),
        name="Calibrated supervoxels",
        scale=scale,
        visible=True,
    )
    viewer.add_labels(
        case["best_markers"].numpy(),
        name="Calibrated markers",
        scale=scale,
        visible=False,
    )
    _progress(
        f"Candidate {candidate_index}: baseline vs calibrated. "
        "Exactly 5 Napari layers loaded."
    )
    napari.run()


# =============================================================================
# RECOMMENDATION
# =============================================================================


def recommendation_payload(
    *,
    baseline_agg: dict[str, Any],
    best_agg: dict[str, Any],
    best_spec: SweepSpec,
) -> dict[str, Any]:
    p = best_spec
    changes: dict[str, Any] = {}
    baseline = baseline_agg["spec"]
    for field in (
        "backend",
        "seed_min_distance_dref",
        "seed_threshold",
        "min_supervoxel_voxels",
        "seed_sdf_weight",
        "seed_head_weight",
        "watershed_separator_weight",
        "watershed_surface_weight",
        "watershed_sdf_weight",
    ):
        if baseline[field] != best_agg["spec"][field]:
            changes[field] = {
                "from": baseline[field],
                "to": best_agg["spec"][field],
            }

    algorithmic = None
    if p.marker_rescue == "separator_core":
        algorithmic = {
            "required": True,
            "change": "separator-core missing-marker rescue in model/partition/seeds.py",
            "separator_cutoff": p.rescue_separator_cutoff,
            "min_component_voxels": p.rescue_min_component_voxels,
            "min_score": p.rescue_min_score,
            "gt_used_at_inference": False,
        }
    else:
        algorithmic = {"required": False}

    all_production_safe = best_agg["production_unsafe_scene_count"] == 0
    all_stress_safe = best_agg["stress_unsafe_scene_count"] == 0
    return {
        "all_scanned_scenes_production_safe": all_production_safe,
        "all_scanned_scenes_stress_safe": all_stress_safe,
        "baseline": baseline_agg,
        "recommended": best_agg,
        "parameter_changes": changes,
        "algorithmic_change": algorithmic,
        "interpretation": (
            "Parameter/rescue candidate removes every production-required unsafe scene."
            if all_production_safe
            else (
                "No tested policy removed every production-required unsafe scene. "
                "Inspect residual failure mechanisms before patching production."
            )
        ),
    }


# =============================================================================
# MAIN CALIBRATION
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Time-budgeted oracle calibration of STIR-Net marker/watershed "
            "oversegmentation before downstream RAG debugging."
        )
    )
    parser.add_argument(
        "--stage00-sample", type=Path, default=DEFAULT_STAGE00_SAMPLE
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override source data dir stored in debug_crop.pt.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=None,
        help="Override source time index stored in debug_crop.pt.",
    )
    parser.add_argument(
        "--result-dir", type=Path, default=DEFAULT_RESULT_DIR
    )
    parser.add_argument(
        "--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES
    )
    parser.add_argument(
        "--context-margin-dref",
        type=float,
        default=stage00.DEFAULT_CONTEXT_MARGIN_DREF,
    )
    parser.add_argument(
        "--budget-minutes", type=float, default=DEFAULT_BUDGET_MINUTES
    )
    parser.add_argument(
        "--reserve-minutes", type=float, default=DEFAULT_RESERVE_MINUTES
    )
    parser.add_argument(
        "--rebuild-contexts", action="store_true"
    )
    parser.add_argument(
        "--rebuild-runs", action="store_true"
    )
    parser.add_argument(
        "--visualize-candidate",
        type=int,
        default=None,
        help="After calibration, visualize baseline vs recommended supervoxels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.stage00_sample = args.stage00_sample.resolve()
    args.result_dir = args.result_dir.resolve()
    if args.data_dir is not None:
        args.data_dir = args.data_dir.resolve()

    if args.visualize_candidate is not None:
        visualize_candidate_artifact(args, int(args.visualize_candidate))
        return

    if args.max_candidates < 1:
        raise ValueError("--max-candidates must be >= 1")
    if args.budget_minutes <= 0:
        raise ValueError("--budget-minutes must be positive")
    if not 0 <= args.reserve_minutes < args.budget_minutes:
        raise ValueError("--reserve-minutes must be >=0 and smaller than budget")

    started = time.perf_counter()
    budget = TimeBudget(
        started=started,
        budget_seconds=float(args.budget_minutes) * 60.0,
        reserve_seconds=float(args.reserve_minutes) * 60.0,
    )
    args.result_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = stage03.stage01.build_debug_config()
    base_cfg.validate()
    p = base_cfg.partition

    _progress("\n" + "=" * 120)
    _progress("STIR-Net Stage 05 — oracle watershed oversegmentation calibration")
    _progress("=" * 120)
    _progress(f"Wall-clock budget       : {args.budget_minutes:.1f} min")
    _progress(f"Reserved finalization   : {args.reserve_minutes:.1f} min")
    _progress(f"Candidates              : up to {args.max_candidates}")
    _progress("Geometry                : ORACLE production targets")
    _progress("Training                : NONE")
    _progress("Primary safety rule     : zero cross-GT supervoxels")
    _progress("Recovery rule           : complete + true-volume-boundary cells recoverable")
    _progress("Stress metric           : all visible cells recoverable, including crop cuts")
    _progress(
        "Current production       : "
        f"backend={p.watershed_backend}, radius={p.seed_min_distance_dref:.2f} dref, "
        f"threshold={p.seed_threshold:.2f}, minSV={p.min_supervoxel_voxels}, "
        f"energy=({p.watershed_separator_weight:.2f},"
        f"{p.watershed_surface_weight:.2f},{p.watershed_sdf_weight:.2f})"
    )
    _progress("=" * 120)

    (
        base_scene,
        candidate_records,
        full_boxes,
        full_counts,
        data_dir,
        time_index,
    ) = load_source_scene(args)

    # ------------------------------------------------------------------
    # Context build/cache
    # ------------------------------------------------------------------
    _progress("\n[Phase 0] Build/reuse oracle candidate contexts")
    contexts_by_index: dict[int, CandidateContext] = {}
    context_times: list[float] = []
    for ordinal, (crop, selection) in enumerate(candidate_records, 1):
        t0 = time.perf_counter()
        context = build_or_load_context(
            args=args,
            base_cfg=base_cfg,
            base_scene=base_scene,
            crop=crop,
            selection=selection,
            full_boxes=full_boxes,
            full_counts=full_counts,
        )
        contexts_by_index[context.candidate_index] = context
        elapsed = time.perf_counter() - t0
        context_times.append(elapsed)
        remaining_contexts = len(candidate_records) - ordinal
        eta = _safe_mean(context_times) * remaining_contexts
        _progress(
            f"[Phase 0] contexts {ordinal:02d}/{len(candidate_records):02d} | "
            f"ETA {_format_duration(eta)} | budget remaining {_format_duration(budget.remaining)}"
        )
        if budget.usable_remaining <= 0:
            raise RuntimeError(
                "Time budget exhausted while building oracle contexts. "
                "Re-run the same command; completed contexts are cached."
            )

    candidate_indices = sorted(contexts_by_index)
    _progress(
        f"[Phase 0] Contexts ready: {candidate_indices} | "
        f"remaining {_format_duration(budget.remaining)}"
    )

    completed_rows: list[dict[str, Any]] = []
    all_aggregates: list[dict[str, Any]] = []
    rows_by_hash: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Phase A: structural sweep on ALL scenes.
    # ------------------------------------------------------------------
    _progress("\n[Phase A] Structural marker/backend/cleanup sweep")
    phase_a_specs = structural_specs(base_cfg)
    for spec_pos, spec in enumerate(phase_a_specs, 1):
        if not can_afford_runs(
            budget,
            len(candidate_indices),
            completed_rows,
            multiplier=1.10,
        ):
            _progress(
                f"[Phase A] Budget guard before spec {spec.name}; "
                "moving to adaptive phases with completed evidence."
            )
            break
        _progress(
            f"\n[Phase A] SPEC {spec_pos:02d}/{len(phase_a_specs):02d}: {spec.name}"
        )
        rows, complete = run_spec_on_indices(
            args=args,
            spec=spec,
            contexts_by_index=contexts_by_index,
            indices=candidate_indices,
            base_cfg=base_cfg,
            budget=budget,
            completed_rows=completed_rows,
            phase_label="A",
        )
        if not complete:
            break
        agg = aggregate_spec(spec, rows)
        rows_by_hash[spec.hash()] = rows
        all_aggregates.append(agg)
        save_progress(
            args,
            phase="A_structural",
            budget=budget,
            aggregates=all_aggregates,
            completed_run_count=len(completed_rows),
        )

    if not all_aggregates:
        raise RuntimeError("No complete calibration specification finished.")
    print_aggregate_table(all_aggregates, "Phase A ranking")

    baseline_spec = spec_from_base(
        base_cfg, name="baseline_current", phase="A_struct"
    )
    baseline_agg = next(
        (
            agg
            for agg in all_aggregates
            if agg["spec_hash"] == baseline_spec.hash()
        ),
        None,
    )
    if baseline_agg is None:
        # Baseline should be first, but support resume/budget edge cases.
        baseline_rows, complete = run_spec_on_indices(
            args=args,
            spec=baseline_spec,
            contexts_by_index=contexts_by_index,
            indices=candidate_indices,
            base_cfg=base_cfg,
            budget=budget,
            completed_rows=completed_rows,
            phase_label="BASE",
        )
        if not complete:
            raise RuntimeError("Could not finish baseline within budget.")
        baseline_agg = aggregate_spec(baseline_spec, baseline_rows)
        rows_by_hash[baseline_spec.hash()] = baseline_rows
        all_aggregates.append(baseline_agg)

    # ------------------------------------------------------------------
    # Phase B: thresholds on top 3 structural families.
    # ------------------------------------------------------------------
    structural_ordered = sorted(all_aggregates, key=aggregate_rank_key)
    phase_b_specs = threshold_specs(base_cfg, structural_ordered)
    if phase_b_specs:
        _progress("\n[Phase B] Lower seed-threshold sweep on strongest structural families")
    for spec_pos, spec in enumerate(phase_b_specs, 1):
        if not can_afford_runs(
            budget,
            len(candidate_indices),
            completed_rows,
            multiplier=1.12,
        ):
            _progress("[Phase B] Budget guard: ending threshold sweep.")
            break
        _progress(
            f"\n[Phase B] SPEC {spec_pos:02d}/{len(phase_b_specs):02d}: {spec.name}"
        )
        rows, complete = run_spec_on_indices(
            args=args,
            spec=spec,
            contexts_by_index=contexts_by_index,
            indices=candidate_indices,
            base_cfg=base_cfg,
            budget=budget,
            completed_rows=completed_rows,
            phase_label="B",
        )
        if not complete:
            break
        agg = aggregate_spec(spec, rows)
        rows_by_hash[spec.hash()] = rows
        all_aggregates.append(agg)
        save_progress(
            args,
            phase="B_threshold",
            budget=budget,
            aggregates=all_aggregates,
            completed_run_count=len(completed_rows),
        )

    print_aggregate_table(all_aggregates, "After Phase B ranking")

    # ------------------------------------------------------------------
    # Phase C: mechanism-specific targeted screening.
    # ------------------------------------------------------------------
    best_agg = sorted(all_aggregates, key=aggregate_rank_key)[0]
    best_spec = SweepSpec(**best_agg["spec"])
    best_rows = rows_by_hash.get(best_spec.hash(), [])
    residual_indices = list(best_agg["unsafe_candidate_indices"])

    # Include representative known-safe cases to detect pathological
    # overfragmentation while screening a mechanism-specific change.
    safe_regression = [
        idx
        for idx in candidate_indices
        if idx not in residual_indices
    ][: min(3, len(candidate_indices))]
    targeted_indices = sorted(set(residual_indices + safe_regression))
    if not targeted_indices:
        targeted_indices = candidate_indices[: min(4, len(candidate_indices))]

    mechanism_totals = best_agg["failure_mechanisms"]
    phase_c_candidates: list[SweepSpec] = []

    if (
        best_agg["production_unsafe_scene_count"] > 0
        and (
            mechanism_totals.get("missing_marker", 0) > 0
            or best_agg["missing_marker_visible_cell_total"] > 0
        )
    ):
        phase_c_candidates.extend(rescue_specs(best_spec))
        phase_c_candidates.extend(seed_weight_specs(best_spec))

    if (
        best_agg["production_unsafe_scene_count"] > 0
        and mechanism_totals.get("watershed_propagation", 0) > 0
    ):
        phase_c_candidates.extend(propagation_specs(best_spec))

    # Even if current best is production-safe, test a small rescue screen when
    # baseline had missing-marker failures; it may achieve the same safety with
    # less aggressive global fragmentation.
    if (
        best_agg["production_unsafe_scene_count"] == 0
        and baseline_agg["failure_mechanisms"].get("missing_marker", 0) > 0
    ):
        conservative = replace(
            best_spec,
            seed_min_distance_dref=max(best_spec.seed_min_distance_dref, 0.30),
        )
        phase_c_candidates.extend(rescue_specs(conservative)[:3])

    phase_c_candidates = _dedupe_specs(phase_c_candidates)
    targeted_aggs: list[dict[str, Any]] = []

    if phase_c_candidates:
        _progress(
            "\n[Phase C] Mechanism-specific targeted screen | "
            f"indices={targeted_indices} | mechanisms={mechanism_totals}"
        )
    for spec_pos, spec in enumerate(phase_c_candidates, 1):
        if not can_afford_runs(
            budget,
            len(targeted_indices),
            completed_rows,
            multiplier=1.12,
        ):
            _progress("[Phase C] Budget guard: ending targeted screen.")
            break
        _progress(
            f"\n[Phase C] SPEC {spec_pos:02d}/{len(phase_c_candidates):02d}: {spec.name}"
        )
        rows, complete = run_spec_on_indices(
            args=args,
            spec=spec,
            contexts_by_index=contexts_by_index,
            indices=targeted_indices,
            base_cfg=base_cfg,
            budget=budget,
            completed_rows=completed_rows,
            phase_label="C",
        )
        if not complete:
            break
        targeted_agg = aggregate_spec(spec, rows)
        targeted_aggs.append(targeted_agg)

    # Validate the best targeted variants on all scenes, in targeted ranking
    # order. Do at most 3 finalists, and stop as soon as a strong globally safe
    # candidate exists and the next validation would endanger finalization.
    if targeted_aggs:
        targeted_aggs.sort(key=aggregate_rank_key)
        _progress("\n[Phase C] Full-scene validation of strongest targeted variants")
        for targeted in targeted_aggs[:3]:
            spec = SweepSpec(**targeted["spec"])
            if not can_afford_runs(
                budget,
                len(candidate_indices),
                completed_rows,
                multiplier=1.12,
            ):
                _progress("[Phase C] Budget guard before full finalist validation.")
                break
            rows, complete = run_spec_on_indices(
                args=args,
                spec=spec,
                contexts_by_index=contexts_by_index,
                indices=candidate_indices,
                base_cfg=base_cfg,
                budget=budget,
                completed_rows=completed_rows,
                phase_label="C-FULL",
            )
            if not complete:
                break
            agg = aggregate_spec(spec, rows)
            rows_by_hash[spec.hash()] = rows
            all_aggregates.append(agg)
            save_progress(
                args,
                phase="C_finalists",
                budget=budget,
                aggregates=all_aggregates,
                completed_run_count=len(completed_rows),
            )

    # ------------------------------------------------------------------
    # Final ranking / recommendation
    # ------------------------------------------------------------------
    all_aggregates = list(
        {
            agg["spec_hash"]: agg for agg in all_aggregates
        }.values()
    )
    all_aggregates.sort(key=aggregate_rank_key)
    best_agg = all_aggregates[0]
    best_spec = SweepSpec(**best_agg["spec"])
    best_rows = rows_by_hash.get(best_spec.hash())
    if best_rows is None:
        # Should only occur if a targeted aggregate somehow entered the list
        # without full validation; guard anyway.
        best_rows, complete = run_spec_on_indices(
            args=args,
            spec=best_spec,
            contexts_by_index=contexts_by_index,
            indices=candidate_indices,
            base_cfg=base_cfg,
            budget=budget,
            completed_rows=completed_rows,
            phase_label="FINAL",
        )
        if not complete:
            raise RuntimeError("Best candidate was not fully validated.")
        best_agg = aggregate_spec(best_spec, best_rows)
        rows_by_hash[best_spec.hash()] = best_rows

    baseline_rows = rows_by_hash.get(baseline_spec.hash())
    if baseline_rows is None:
        raise RuntimeError("Internal error: missing baseline rows.")

    print_aggregate_table(all_aggregates, "FINAL calibration ranking", limit=30)

    recommendation = recommendation_payload(
        baseline_agg=baseline_agg,
        best_agg=best_agg,
        best_spec=best_spec,
    )

    final_report = {
        "format_version": REPORT_FORMAT_VERSION,
        "source": {
            "data_dir": str(data_dir),
            "time_index": int(time_index),
            "stage00_sample": str(args.stage00_sample),
            "candidate_indices": candidate_indices,
        },
        "budget": {
            "requested_minutes": float(args.budget_minutes),
            "reserve_minutes": float(args.reserve_minutes),
            "elapsed_seconds": float(budget.elapsed),
            "remaining_seconds": float(budget.remaining),
        },
        "acceptance": {
            "zero_meaningful_cross_gt_supervoxels": True,
            "minimum_cell_coverage": float(MIN_CELL_COVERAGE),
            "production_required_cells": (
                "complete-in-debug-crop; this includes true-volume-boundary "
                "cells when their full visible instance is represented"
            ),
            "stress_metric_includes_artificial_crop_partials": True,
            "oversegmentation_preferred_over_undersegmentation": True,
        },
        "baseline": baseline_agg,
        "best": best_agg,
        "recommendation": recommendation,
        "all_fully_validated_aggregates": all_aggregates,
        "best_candidate_rows": best_rows,
        "baseline_candidate_rows": baseline_rows,
    }
    atomic_json(args.result_dir / "final_report.json", final_report)
    atomic_json(args.result_dir / "recommendation.json", recommendation)

    # Save direct visual before/after only if the budget still has enough room.
    artifact_path = None
    baseline_unsafe_count = sum(
        (not bool(row["production_safe"]))
        or int(row["clean_cross_gt_supervoxel_count"]) > 0
        for row in baseline_rows
    )
    estimated_artifact_runs = max(2 * baseline_unsafe_count, 1)
    if can_afford_runs(
        budget,
        estimated_artifact_runs,
        completed_rows,
        multiplier=1.05,
    ):
        _progress("\n[finalize] Building baseline-vs-best visual artifacts ...")
        artifact_path = save_before_after_artifacts(
            args=args,
            base_scene=base_scene,
            contexts_by_index=contexts_by_index,
            base_cfg=base_cfg,
            baseline=baseline_spec,
            best=best_spec,
            baseline_rows=baseline_rows,
        )
    else:
        _progress(
            "\n[finalize] Skipping before/after .pt artifact to respect wall-clock budget. "
            "JSON recommendation is complete."
        )

    # ------------------------------------------------------------------
    # Human-readable conclusion
    # ------------------------------------------------------------------
    _progress("\n" + "=" * 120)
    _progress("STIR-Net Stage 05 — CALIBRATION RESULT")
    _progress("=" * 120)
    _progress(
        f"Baseline production-safe scenes : "
        f"{baseline_agg['production_safe_scene_count']}/{baseline_agg['candidate_count']}"
    )
    _progress(
        f"Baseline cross-GT SV total      : "
        f"{baseline_agg['clean_cross_gt_supervoxel_total']}"
    )
    _progress(
        f"Recommended production-safe     : "
        f"{best_agg['production_safe_scene_count']}/{best_agg['candidate_count']}"
    )
    _progress(
        f"Recommended cross-GT SV total   : "
        f"{best_agg['clean_cross_gt_supervoxel_total']}"
    )
    _progress(
        f"Stress-safe all visible         : "
        f"{best_agg['stress_safe_scene_count']}/{best_agg['candidate_count']}"
    )
    _progress(
        f"Mean meaningful SV / visible GT : "
        f"{best_agg['mean_meaningful_supervoxels_per_visible_gt']:.2f}"
    )
    _progress(f"Recommended spec                : {best_spec.name}")
    _progress(
        "  backend / radius / threshold   : "
        f"{best_spec.backend} / {best_spec.seed_min_distance_dref:.2f} / "
        f"{best_spec.seed_threshold:.2f}"
    )
    _progress(
        "  min_supervoxel_voxels          : "
        f"{best_spec.min_supervoxel_voxels}"
    )
    _progress(
        "  seed SDF/head weights          : "
        f"{best_spec.seed_sdf_weight:.2f} / {best_spec.seed_head_weight:.2f}"
    )
    _progress(
        "  energy sep/surf/SDF            : "
        f"{best_spec.watershed_separator_weight:.2f} / "
        f"{best_spec.watershed_surface_weight:.2f} / "
        f"{best_spec.watershed_sdf_weight:.2f}"
    )
    _progress(f"  marker rescue                  : {best_spec.marker_rescue}")
    if best_spec.marker_rescue == "separator_core":
        _progress(
            "    separator cutoff / min vox / min score: "
            f"{best_spec.rescue_separator_cutoff:.2f} / "
            f"{best_spec.rescue_min_component_voxels} / "
            f"{best_spec.rescue_min_score:.2f}"
        )
    _progress(f"Parameter changes               : {recommendation['parameter_changes']}")
    _progress(f"Algorithmic change              : {recommendation['algorithmic_change']}")
    _progress(f"Final report                    : {args.result_dir / 'final_report.json'}")
    _progress(f"Recommendation                  : {args.result_dir / 'recommendation.json'}")
    if artifact_path is not None:
        _progress(f"Before/after artifact            : {artifact_path}")
        unsafe_indices = [
            int(row["candidate_index"])
            for row in baseline_rows
            if (not bool(row["production_safe"]))
            or int(row["clean_cross_gt_supervoxel_count"]) > 0
        ]
        if unsafe_indices:
            _progress(
                "Visualize one baseline-unsafe scene: "
                f"python investigations/stirnet/05_watershed_oversegmentation_calibration.py "
                f"--visualize-candidate {unsafe_indices[0]}"
            )
    _progress(
        f"Elapsed                          : {_format_duration(budget.elapsed)} "
        f"(remaining {_format_duration(budget.remaining)})"
    )
    _progress("=" * 120)


if __name__ == "__main__":
    main()
