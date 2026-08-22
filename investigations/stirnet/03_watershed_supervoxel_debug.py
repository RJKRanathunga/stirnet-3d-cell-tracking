from __future__ import annotations

"""Stage 03: marker-controlled watershed and supervoxel diagnostics.

This investigation starts from the fixed Stage-00 crop and a successful
Stage-01 dense-geometry checkpoint. It does not train or modify the model.

It exposes:
    dense geometry -> foreground mask -> seed score -> markers
    -> watershed energy -> raw watershed -> tiny-region cleanup -> supervoxels

It also runs oracle substitutions to localize failures:
    predicted, oracle_all, oracle_foreground, oracle_seed, oracle_sdf,
    oracle_separator, oracle_surface, oracle_marker_score, oracle_energy

Run:
    python investigations/stirnet/03_watershed_supervoxel_debug.py

Visualize:
    python investigations/stirnet/03_watershed_supervoxel_debug.py --visualize
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from skimage.segmentation import watershed as skimage_watershed

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

STAGE01_PATH = Path(__file__).with_name("01_dense_geometry_overfit.py")
if not STAGE01_PATH.exists():
    raise FileNotFoundError(f"Missing Stage-01 investigation: {STAGE01_PATH}")

_spec = importlib.util.spec_from_file_location(
    "_stirnet_stage01_dense_geometry_overfit", STAGE01_PATH
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not import {STAGE01_PATH}")
stage01 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = stage01
_spec.loader.exec_module(stage01)

from learned.stirnet.model.geometry.derived import build_geometry_derived_cache
from learned.stirnet.model.partition.seeds import build_markers, build_markers_fast
from learned.stirnet.model.partition.watershed import (
    _component_bounded_watershed,
    _merge_tiny_regions,
)
from learned.stirnet.model.types import GeometryState

DEFAULT_SAMPLE_PATH = stage01.DEFAULT_SAMPLE_PATH
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "dense_geometry_debug"
    / "latest_success.pt"
)
DEFAULT_RESULT_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "watershed_supervoxel_debug"
)

MEANINGFUL_OVERLAP_MIN_VOXELS = 8
MEANINGFUL_OVERLAP_MIN_GT_FRACTION = 0.01
MIN_COMPLETE_GT_COVERAGE = 0.95
MIN_NODE_PURITY_FOR_SAFE = 0.80
MIN_NODE_GT_SUPPORT_FOR_SAFE = 0.50
MIN_ATOMIC_RECOVERABLE_FRACTION = 0.95
PROB_EPS = 1e-4


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def as_cpu(value: Tensor, *, half_float: bool = False) -> Tensor:
    value = value.detach().cpu()
    if half_float and value.is_floating_point():
        return value.half()
    return value


def prob_to_logit(prob: Tensor) -> Tensor:
    p = prob.float().clamp(PROB_EPS, 1.0 - PROB_EPS)
    return torch.log(p) - torch.log1p(-p)


def unique_positive_ids(labels: Tensor) -> list[int]:
    return [int(v) for v in torch.unique(labels.cpu()).tolist() if int(v) > 0]


def crop_complete_gt_ids(gt: Tensor) -> tuple[list[int], list[int]]:
    gt = gt.detach().cpu().long()
    ids = unique_positive_ids(gt)
    boundary_ids: set[int] = set()
    for face in (
        gt[0], gt[-1], gt[:, 0], gt[:, -1], gt[:, :, 0], gt[:, :, -1]
    ):
        boundary_ids.update(
            int(v) for v in torch.unique(face).tolist() if int(v) > 0
        )
    complete = [v for v in ids if v not in boundary_ids]
    partial = [v for v in ids if v in boundary_ids]
    return complete, partial


def safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    value = float(np.mean(values))
    return value if math.isfinite(value) else None


def load_dense_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing Stage-01 checkpoint: {checkpoint_path}\n"
            "Run Stage 01 first or pass --checkpoint explicitly."
        )
    checkpoint = torch_load(checkpoint_path)
    checkpoint_stage = checkpoint.get("stage")
    if checkpoint_stage not in {"joint", "full"}:
        raise RuntimeError(
            "Stage 03 needs a successful all-head dense checkpoint "
            f"(joint or full), got stage={checkpoint_stage!r}."
        )
    if "model_state" not in checkpoint:
        raise RuntimeError("Checkpoint has no model_state.")

    cfg = stage01.build_debug_config()
    model = stage01.DenseGeometryOnlyModel(cfg).to(device)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state"], strict=False
    )
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/current-model mismatch.\n"
            f"missing={missing}\nunexpected={unexpected}"
        )
    model.eval()
    return model, checkpoint, cfg


def oracle_geometry_from_targets(target: dict[str, Tensor]) -> GeometryState:
    return GeometryState(
        foreground_logits=prob_to_logit(target["foreground"]),
        surface_logits=prob_to_logit(target["surface"]),
        separator_logits=prob_to_logit(target["separator"]),
        sdf=target["sdf"].float(),
        flow=target["flow"].float(),
        centroid_offset=target["centroid_offset"].float(),
        seed_logits=prob_to_logit(target["seed"]),
        features=None,
    )


def hybrid_geometry(
    predicted: GeometryState, oracle: GeometryState, field: str
) -> GeometryState:
    attr = {
        "foreground": "foreground_logits",
        "surface": "surface_logits",
        "separator": "separator_logits",
        "sdf": "sdf",
        "seed": "seed_logits",
    }[field]
    return replace(predicted, **{attr: getattr(oracle, attr)})


def build_exact_markers(
    score: Tensor,
    foreground: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    cfg,
) -> np.ndarray:
    radius_um = (
        cfg.partition.seed_min_distance_dref * float(dref_um.item())
    )
    if cfg.partition.watershed_backend == "fast":
        return build_markers_fast(
            score,
            foreground,
            spacing_um,
            radius_um,
            cfg.partition.seed_threshold,
            cfg.partition.max_supervoxels,
        )
    return build_markers(
        score.float().cpu().numpy(),
        foreground.bool().cpu().numpy(),
        spacing_um.detach().cpu().numpy().astype(np.float32),
        radius_um,
        cfg.partition.seed_threshold,
        cfg.partition.max_supervoxels,
    )


def run_exact_watershed(
    energy: Tensor,
    markers: np.ndarray,
    foreground: Tensor,
    cfg,
) -> tuple[np.ndarray, np.ndarray]:
    fg_np = foreground.bool().cpu().numpy()
    energy_np = energy.float().cpu().numpy()

    if (
        cfg.partition.watershed_backend == "fast"
        and cfg.partition.component_bounded_watershed
    ):
        raw = _component_bounded_watershed(
            energy_np,
            markers,
            fg_np,
            halo=cfg.partition.watershed_component_halo_voxels,
        )
    else:
        raw = skimage_watershed(
            energy_np, markers=markers, mask=fg_np, connectivity=1
        ).astype(np.int32)

    clean = _merge_tiny_regions(
        raw.astype(np.int32, copy=False),
        cfg.partition.min_supervoxel_voxels,
    )
    if int(clean.max()) > cfg.partition.max_supervoxels:
        raise RuntimeError(
            f"Watershed created {int(clean.max())} supervoxels; "
            f"max_supervoxels={cfg.partition.max_supervoxels}."
        )
    return raw.astype(np.int32, copy=False), clean.astype(np.int32, copy=False)


def marker_diagnostics(
    markers: np.ndarray,
    gt: np.ndarray,
    complete_gt_ids: list[int],
) -> dict[str, Any]:
    marker_count = int(markers.max())
    gt_ids = sorted(int(v) for v in np.unique(gt) if int(v) > 0)
    per_gt = {str(gt_id): 0 for gt_id in gt_ids}
    rows: list[dict[str, Any]] = []
    markers_on_background = 0

    for marker_id in range(1, marker_count + 1):
        mask = markers == marker_id
        values, counts = np.unique(gt[mask], return_counts=True)
        positive = [
            (int(v), int(c))
            for v, c in zip(values.tolist(), counts.tolist())
            if int(v) > 0
        ]
        positive.sort(key=lambda x: (-x[1], x[0]))
        dominant_gt = positive[0][0] if positive else 0
        if dominant_gt:
            per_gt[str(dominant_gt)] += 1
        else:
            markers_on_background += 1

        coords = np.argwhere(mask)
        center = (
            coords.mean(axis=0).tolist()
            if len(coords)
            else [0.0, 0.0, 0.0]
        )
        rows.append(
            {
                "marker_id": marker_id,
                "voxel_count": int(mask.sum()),
                "center_voxel_zyx": [float(v) for v in center],
                "dominant_gt": dominant_gt,
                "gt_overlaps": [
                    {"gt_id": gt_id, "voxels": count}
                    for gt_id, count in positive
                ],
            }
        )

    complete_with_marker = sum(
        per_gt.get(str(gt_id), 0) > 0 for gt_id in complete_gt_ids
    )
    recall = (
        complete_with_marker / len(complete_gt_ids)
        if complete_gt_ids
        else 1.0
    )
    return {
        "marker_count": marker_count,
        "markers_on_background": markers_on_background,
        "per_gt_marker_count": per_gt,
        "complete_gt_with_marker": complete_with_marker,
        "complete_gt_count": len(complete_gt_ids),
        "complete_gt_marker_recall": float(recall),
        "markers": rows,
    }


def supervoxel_diagnostics(
    labels: np.ndarray,
    gt: np.ndarray,
    complete_gt_ids: list[int],
    cfg,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    labels = labels.astype(np.int64, copy=False)
    gt = gt.astype(np.int64, copy=False)
    n_sv = int(labels.max())
    gt_ids = sorted(int(v) for v in np.unique(gt) if int(v) > 0)
    gt_volumes = {gt_id: int(np.sum(gt == gt_id)) for gt_id in gt_ids}

    rows: list[dict[str, Any]] = []
    purity_lut = np.zeros(n_sv + 1, dtype=np.float32)
    gt_support_lut = np.zeros(n_sv + 1, dtype=np.float32)
    dominant_gt_lut = np.zeros(n_sv + 1, dtype=np.int32)
    unsafe_lut = np.zeros(n_sv + 1, dtype=np.uint8)
    cross_gt_count = 0
    rag_valid_count = 0
    atomic_irreducible_gt_voxels = 0

    for sv_id in range(1, n_sv + 1):
        mask = labels == sv_id
        sv_count = int(mask.sum())
        values, counts = np.unique(gt[mask], return_counts=True)
        overlap = {
            int(v): int(c)
            for v, c in zip(values.tolist(), counts.tolist())
            if int(v) > 0
        }
        positive_voxels = sum(overlap.values())
        gt_support = positive_voxels / max(sv_count, 1)

        if overlap:
            dominant_gt = max(overlap, key=lambda gt_id: (overlap[gt_id], -gt_id))
            dominant_overlap = int(overlap[dominant_gt])
            purity = dominant_overlap / max(positive_voxels, 1)
        else:
            dominant_gt = 0
            dominant_overlap = 0
            purity = 0.0

        irreducible_gt_voxels = max(positive_voxels - dominant_overlap, 0)
        atomic_irreducible_gt_voxels += irreducible_gt_voxels
        secondary = [(gt_id, count) for gt_id, count in overlap.items() if gt_id != dominant_gt]
        if secondary:
            secondary_gt, secondary_count = max(secondary, key=lambda item: (item[1], -item[0]))
            secondary_fraction = secondary_count / max(gt_volumes.get(secondary_gt, 0), 1)
        else:
            secondary_gt, secondary_count, secondary_fraction = 0, 0, 0.0

        meaningful_gt_ids = [
            gt_id
            for gt_id, count in overlap.items()
            if count >= MEANINGFUL_OVERLAP_MIN_VOXELS
            and count / max(gt_volumes[gt_id], 1) >= MEANINGFUL_OVERLAP_MIN_GT_FRACTION
        ]
        cross_gt = len(meaningful_gt_ids) >= 2
        cross_gt_count += int(cross_gt)
        rag_valid = (
            purity >= cfg.partition.rag_min_node_purity
            and gt_support >= cfg.partition.rag_min_node_gt_support
        )
        rag_valid_count += int(rag_valid)

        purity_lut[sv_id] = purity
        gt_support_lut[sv_id] = gt_support
        dominant_gt_lut[sv_id] = dominant_gt
        unsafe_lut[sv_id] = int(
            cross_gt
            or purity < MIN_NODE_PURITY_FOR_SAFE
            or gt_support < MIN_NODE_GT_SUPPORT_FOR_SAFE
        )
        rows.append({
            "supervoxel_id": sv_id,
            "voxel_count": sv_count,
            "gt_support": float(gt_support),
            "purity_among_gt": float(purity),
            "dominant_gt": dominant_gt,
            "dominant_gt_overlap_voxels": dominant_overlap,
            "atomic_irreducible_gt_voxels": int(irreducible_gt_voxels),
            "atomic_irreducible_fraction_among_gt": float(
                irreducible_gt_voxels / max(positive_voxels, 1)
            ),
            "largest_secondary_gt": int(secondary_gt),
            "largest_secondary_overlap_voxels": int(secondary_count),
            "largest_secondary_fraction_of_that_gt": float(secondary_fraction),
            "meaningful_gt_ids": sorted(meaningful_gt_ids),
            "cross_gt_unsafe": bool(cross_gt),
            "passes_current_rag_validity": bool(rag_valid),
        })

    positive = labels > 0
    gt_fg = gt > 0
    intersection = int(np.sum(positive & gt_fg))
    positive_count = int(np.sum(positive))
    gt_count = int(np.sum(gt_fg))

    per_gt: dict[str, Any] = {}
    legacy_complete_recoverable = 0
    complete_atomic: list[float] = []
    visible_atomic: list[float] = []
    total_atomic_recoverable_voxels = 0

    for gt_id in gt_ids:
        cell = gt == gt_id
        cell_volume = gt_volumes[gt_id]
        coverage = int(np.sum(cell & positive)) / max(cell_volume, 1)
        sv_ids, counts = np.unique(labels[cell], return_counts=True)
        pairs = [
            (int(sv), int(count))
            for sv, count in zip(sv_ids.tolist(), counts.tolist())
            if int(sv) > 0
        ]
        meaningful = [
            (sv, count)
            for sv, count in pairs
            if count >= MEANINGFUL_OVERLAP_MIN_VOXELS
            and count / max(cell_volume, 1) >= MEANINGFUL_OVERLAP_MIN_GT_FRACTION
        ]
        unsafe = []
        for sv_id, _ in meaningful:
            row = rows[sv_id - 1]
            if (
                row["dominant_gt"] != gt_id
                or row["purity_among_gt"] < MIN_NODE_PURITY_FOR_SAFE
                or row["cross_gt_unsafe"]
            ):
                unsafe.append(sv_id)
        legacy_recoverable = (
            coverage >= MIN_COMPLETE_GT_COVERAGE
            and len(meaningful) >= 1
            and not unsafe
        )
        if gt_id in complete_gt_ids and legacy_recoverable:
            legacy_complete_recoverable += 1

        atomic_pairs = [
            (sv_id, count)
            for sv_id, count in pairs
            if rows[sv_id - 1]["dominant_gt"] == gt_id
        ]
        atomic_voxels = int(sum(count for _, count in atomic_pairs))
        atomic_fraction = atomic_voxels / max(cell_volume, 1)
        total_atomic_recoverable_voxels += atomic_voxels
        visible_atomic.append(float(atomic_fraction))
        if gt_id in complete_gt_ids:
            complete_atomic.append(float(atomic_fraction))

        per_gt[str(gt_id)] = {
            "volume_voxels": cell_volume,
            "coverage": float(coverage),
            "overlapping_supervoxel_count": len(pairs),
            "meaningful_supervoxel_count": len(meaningful),
            "meaningful_supervoxel_ids": [sv for sv, _ in meaningful],
            "unsafe_overlapping_supervoxel_ids": unsafe,
            "recoverable_by_merging": bool(legacy_recoverable),
            "atomic_recoverable_voxels": atomic_voxels,
            "atomic_lost_voxels": max(cell_volume - atomic_voxels, 0),
            "atomic_recoverable_fraction": float(atomic_fraction),
            "atomic_assigned_supervoxel_ids": [sv for sv, _ in atomic_pairs],
            "complete_in_crop": gt_id in complete_gt_ids,
        }

    complete_coverages = [per_gt[str(gt_id)]["coverage"] for gt_id in complete_gt_ids]
    legacy_fraction = (
        legacy_complete_recoverable / len(complete_gt_ids)
        if complete_gt_ids else 1.0
    )
    total_atomic_unrecoverable = max(gt_count - total_atomic_recoverable_voxels, 0)

    atomic_assignment_error = np.zeros(labels.shape, dtype=np.uint8)
    if n_sv > 0:
        assigned = dominant_gt_lut[labels]
        wrong_owner = (
            (labels > 0) & (gt > 0) & (assigned > 0) & (assigned != gt)
        )
        atomic_assignment_error[wrong_owner] = 1
    atomic_assignment_error[(labels == 0) & (gt > 0)] = 2

    summary = {
        "supervoxel_count": n_sv,
        "foreground_precision": float(intersection / max(positive_count, 1)),
        "foreground_recall": float(intersection / max(gt_count, 1)),
        "foreground_dice": float(2.0 * intersection / max(positive_count + gt_count, 1)),
        "cross_gt_unsafe_supervoxel_count": cross_gt_count,
        "rag_valid_supervoxel_count": rag_valid_count,
        "rag_valid_supervoxel_fraction": float(rag_valid_count / max(n_sv, 1)),
        "complete_gt_count": len(complete_gt_ids),
        "complete_gt_min_coverage": float(min(complete_coverages)) if complete_coverages else 1.0,
        "complete_gt_mean_coverage": float(np.mean(complete_coverages)) if complete_coverages else 1.0,
        "complete_gt_recoverable_count": legacy_complete_recoverable,
        "complete_gt_recoverable_fraction": float(legacy_fraction),
        "atomic_irreducible_gt_voxels_inside_positive_svs": int(atomic_irreducible_gt_voxels),
        "atomic_total_unrecoverable_gt_voxels": int(total_atomic_unrecoverable),
        "atomic_global_recoverable_fraction": float(total_atomic_recoverable_voxels / max(gt_count, 1)),
        "atomic_min_complete_recoverable_fraction": float(min(complete_atomic)) if complete_atomic else 1.0,
        "atomic_mean_complete_recoverable_fraction": float(np.mean(complete_atomic)) if complete_atomic else 1.0,
        "atomic_min_visible_recoverable_fraction": float(min(visible_atomic)) if visible_atomic else 1.0,
        "atomic_mean_visible_recoverable_fraction": float(np.mean(visible_atomic)) if visible_atomic else 1.0,
        "per_gt": per_gt,
        "nodes": rows,
    }
    maps = {
        "node_purity": purity_lut[labels],
        "node_gt_support": gt_support_lut[labels],
        "dominant_gt": dominant_gt_lut[labels],
        "unsafe_supervoxel_mask": unsafe_lut[labels],
        "atomic_assignment_error": atomic_assignment_error,
    }
    return summary, maps


def scalar_field_diagnostics(
    seed_score: np.ndarray,
    energy: np.ndarray,
    target: dict[str, Tensor],
) -> dict[str, Any]:
    fg = target["foreground"][0, 0].cpu().numpy() > 0.5
    surface = target["surface"][0, 0].cpu().numpy()
    separator = target["separator"][0, 0].cpu().numpy()
    seed = target["seed"][0, 0].cpu().numpy()

    interior = fg & (surface < 0.05) & (separator < 0.05)
    separator_band = fg & (separator >= 0.25)
    surface_band = fg & (surface >= 0.25)
    seed_core = fg & (seed >= 0.75)

    e_interior = safe_mean(energy[interior])
    e_sep = safe_mean(energy[separator_band])
    e_surface = safe_mean(energy[surface_band])

    return {
        "energy_mean_interior": e_interior,
        "energy_mean_separator_band": e_sep,
        "energy_mean_surface_band": e_surface,
        "separator_energy_contrast_vs_interior": (
            None if e_sep is None or e_interior is None else e_sep - e_interior
        ),
        "surface_energy_contrast_vs_interior": (
            None
            if e_surface is None or e_interior is None
            else e_surface - e_interior
        ),
        "seed_score_mean_seed_core": safe_mean(seed_score[seed_core]),
        "seed_score_mean_foreground": safe_mean(seed_score[fg]),
    }


def run_variant(
    *,
    name: str,
    geometry: GeometryState,
    batch,
    cfg,
    complete_gt_ids: list[int],
    forced_marker_score: Tensor | None = None,
    forced_energy: Tensor | None = None,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    cache = build_geometry_derived_cache(
        geometry, cfg.partition, padding_mask=None
    )
    fg = cache.foreground_mask[0]
    score = (
        forced_marker_score
        if forced_marker_score is not None
        else cache.seed_score[0, 0]
    )
    energy = (
        forced_energy
        if forced_energy is not None
        else cache.watershed_energy[0, 0]
    )

    markers = build_exact_markers(
        score, fg, batch.spacing_um[0], batch.dref_um[0], cfg
    )
    raw, clean = run_exact_watershed(energy, markers, fg, cfg)

    gt_np = batch.gt_labels.cpu().numpy().astype(np.int32, copy=False)
    marker_summary = marker_diagnostics(
        markers, gt_np, complete_gt_ids
    )
    sv_summary, sv_maps = supervoxel_diagnostics(
        clean, gt_np, complete_gt_ids, cfg
    )
    cleanup_changed = int(np.sum(raw != clean))

    summary = {
        "name": name,
        "marker": marker_summary,
        "watershed": {
            "raw_region_count": int(raw.max()),
            "clean_supervoxel_count": int(clean.max()),
            "regions_removed_or_merged_by_cleanup": max(
                int(raw.max()) - int(clean.max()), 0
            ),
            "cleanup_changed_voxels": cleanup_changed,
            "cleanup_changed_fraction": float(
                cleanup_changed / max(raw.size, 1)
            ),
        },
        "supervoxels": sv_summary,
        "fields": scalar_field_diagnostics(
            score.float().cpu().numpy(),
            energy.float().cpu().numpy(),
            batch.targets,
        ),
    }

    artifact = {
        "foreground_prob": as_cpu(
            cache.foreground_prob[0, 0], half_float=True
        ),
        "surface_prob": as_cpu(
            cache.surface_prob[0, 0], half_float=True
        ),
        "separator_prob": as_cpu(
            cache.separator_prob[0, 0], half_float=True
        ),
        "seed_prob": as_cpu(cache.seed_prob[0, 0], half_float=True),
        "sdf": as_cpu(cache.sdf[0, 0], half_float=True),
        "sdf_normalized": as_cpu(
            cache.sdf_normalized[0, 0], half_float=True
        ),
        "foreground_mask": as_cpu(fg.to(torch.uint8)),
        "seed_score": as_cpu(score, half_float=True),
        "markers": torch.from_numpy(markers.astype(np.int32, copy=False)),
        "watershed_energy": as_cpu(energy, half_float=True),
        "raw_watershed": torch.from_numpy(
            raw.astype(np.int32, copy=False)
        ),
        "supervoxels": torch.from_numpy(
            clean.astype(np.int32, copy=False)
        ),
        "node_purity": torch.from_numpy(sv_maps["node_purity"]).half(),
        "node_gt_support": torch.from_numpy(
            sv_maps["node_gt_support"]
        ).half(),
        "dominant_gt": torch.from_numpy(
            sv_maps["dominant_gt"].astype(np.int32)
        ),
        "unsafe_supervoxel_mask": torch.from_numpy(
            sv_maps["unsafe_supervoxel_mask"].astype(np.uint8)
        ),
        "atomic_assignment_error": torch.from_numpy(
            sv_maps["atomic_assignment_error"].astype(np.uint8)
        ),
    }
    return summary, artifact


def print_variant_table(results: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 136)
    print("Stage-03 oracle substitution comparison")
    print("=" * 136)
    print(
        f"{'variant':22s} {'markers':>8s} {'mkRecall':>9s} "
        f"{'rawSV':>7s} {'SV':>6s} {'fgRec':>8s} {'minCov':>8s} "
        f"{'crossGT':>8s} {'atomicMin':>9s} {'sepΔE':>9s}"
    )
    print("-" * 136)
    for name, row in results.items():
        marker = row["marker"]
        ws = row["watershed"]
        sv = row["supervoxels"]
        sep = row["fields"]["separator_energy_contrast_vs_interior"]
        sep_text = "n/a" if sep is None else f"{sep:.3f}"
        print(
            f"{name:22s} {marker['marker_count']:8d} "
            f"{marker['complete_gt_marker_recall']:9.3f} "
            f"{ws['raw_region_count']:7d} {ws['clean_supervoxel_count']:6d} "
            f"{sv['foreground_recall']:8.3f} {sv['complete_gt_min_coverage']:8.3f} "
            f"{sv['cross_gt_unsafe_supervoxel_count']:8d} "
            f"{sv['atomic_min_complete_recoverable_fraction']:9.3f} {sep_text:>9s}"
        )
    print("=" * 136)


def predicted_acceptance(
    predicted: dict[str, Any]
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    marker = predicted["marker"]
    sv = predicted["supervoxels"]
    if marker["complete_gt_marker_recall"] < 1.0:
        failures.append(
            "not every complete GT cell has a marker "
            f"(recall={marker['complete_gt_marker_recall']:.3f})"
        )
    if sv["complete_gt_min_coverage"] < MIN_COMPLETE_GT_COVERAGE:
        failures.append(
            "complete-GT foreground coverage too low "
            f"(min={sv['complete_gt_min_coverage']:.3f})"
        )
    if sv["atomic_min_complete_recoverable_fraction"] < MIN_ATOMIC_RECOVERABLE_FRACTION:
        failures.append(
            "atomic proposal loses too much of at least one complete GT cell "
            f"(min recoverable={sv['atomic_min_complete_recoverable_fraction']:.3f}, "
            f"required={MIN_ATOMIC_RECOVERABLE_FRACTION:.3f})"
        )
    return not failures, failures


def np_array(value) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def add_image(
    viewer,
    value,
    name: str,
    scale,
    *,
    visible: bool = False,
    fixed_limits: tuple[float, float] | None = None,
) -> None:
    array = np_array(value).astype(np.float32, copy=False)
    kwargs: dict[str, Any] = {}
    if fixed_limits is not None:
        kwargs["contrast_limits"] = fixed_limits
    else:
        finite = array[np.isfinite(array)]
        if finite.size:
            lo = float(np.quantile(finite, 0.01))
            hi = float(np.quantile(finite, 0.99))
            if hi <= lo:
                hi = lo + 1.0
            kwargs["contrast_limits"] = (lo, hi)

    viewer.add_image(
        array,
        name=name,
        scale=scale,
        colormap="viridis",
        visible=visible,
        **kwargs,
    )


def visualize(sample_path: Path, artifact_path: Path) -> None:
    """Open Napari with ONLY Stage-03 inputs/reference and final outputs."""
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError("Napari is required for --visualize.") from exc

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing Stage-03 artifact: {artifact_path}\n"
            "Run Stage 03 first."
        )

    artifact = torch_load(artifact_path)
    sample = torch_load(sample_path)

    spacing = np_array(sample["spacing_um"]).astype(np.float32).reshape(3)
    scale = tuple(float(v) for v in spacing)

    raw = np_array(sample["spatial_inputs"])[0].astype(np.float32, copy=False)
    current = np_array(sample["current_labels"]).astype(np.int32, copy=False)
    gt = np_array(sample["gt_labels"]).astype(np.int32, copy=False)

    predicted = artifact["variants"]["predicted"]
    markers = np_array(predicted["markers"]).astype(np.int32, copy=False)
    watershed = np_array(predicted["raw_watershed"]).astype(np.int32, copy=False)
    supervoxels = np_array(predicted["supervoxels"]).astype(np.int32, copy=False)

    viewer = napari.Viewer(ndisplay=3)

    # Exactly six layers. No oracle variants and no intermediate diagnostic maps.
    viewer.add_image(
        raw,
        name="Raw input",
        scale=scale,
        colormap="gray",
        visible=True,
    )
    viewer.add_labels(
        current,
        name="Input current labels",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        gt,
        name="GT labels",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        markers,
        name="Predicted markers",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        watershed,
        name="Watershed result",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        supervoxels,
        name="Final supervoxels",
        scale=scale,
        visible=True,
    )

    print("\nNapari Stage-03 final-output view")
    print("  Exactly 6 layers are loaded:")
    print("    1. Raw input")
    print("    2. Input current labels")
    print("    3. GT labels")
    print("    4. Predicted markers")
    print("    5. Watershed result")
    print("    6. Final supervoxels")
    napari.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose STIR-Net marker watershed and supervoxels."
    )
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16", "fp16", "fp32"),
        default=stage01.DEFAULT_AMP_DTYPE,
    )
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_path = args.result_dir / "watershed_supervoxel_debug.pt"
    json_path = args.result_dir / "watershed_supervoxel_debug.json"

    if args.visualize:
        visualize(args.sample_path, artifact_path)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage01.set_seed(stage01.DEFAULT_SEED)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.set_float32_matmul_precision("high")

    amp_dtype, amp_enabled = stage01.resolve_amp(args.amp_dtype, device)
    batch = stage01.load_debug_batch(args.sample_path, device)
    model, checkpoint, cfg = load_dense_model(args.checkpoint, device)
    complete_gt_ids, partial_gt_ids = crop_complete_gt_ids(batch.gt_labels)

    print("\n" + "=" * 110)
    print("STIR-Net Stage 03 — marker-controlled watershed / supervoxel diagnostics")
    print("=" * 110)
    print(f"Sample                 : {args.sample_path}")
    print(f"Dense checkpoint       : {args.checkpoint}")
    print(f"Checkpoint stage       : {checkpoint.get('stage')}")
    print(f"Device                 : {device}")
    print(f"Crop shape             : {tuple(batch.gt_labels.shape)}")
    print(
        "Spacing (z,y,x) um    : "
        + ", ".join(f"{float(v):.6g}" for v in batch.spacing_um[0])
    )
    print(f"dref um                : {float(batch.dref_um[0]):.6f}")
    print(f"Complete GT IDs        : {complete_gt_ids}")
    print(f"Boundary/partial GT IDs: {partial_gt_ids}")
    print(f"Watershed backend      : {cfg.partition.watershed_backend}")
    print(f"Foreground threshold   : {cfg.partition.foreground_threshold}")
    print(f"Seed threshold         : {cfg.partition.seed_threshold}")
    print(f"Seed min dist (dref)   : {cfg.partition.seed_min_distance_dref}")
    print(
        "Energy weights        : "
        f"separator={cfg.partition.watershed_separator_weight}, "
        f"surface={cfg.partition.watershed_surface_weight}, "
        f"sdf={cfg.partition.watershed_sdf_weight}"
    )
    print(
        "Seed-score weights    : "
        f"sdf={cfg.partition.seed_sdf_weight}, "
        f"seed={cfg.partition.seed_head_weight}"
    )
    print(f"Min supervoxel voxels : {cfg.partition.min_supervoxel_voxels}")
    print("=" * 110)

    with torch.no_grad(), stage01.autocast_context(
        device, amp_dtype, amp_enabled
    ):
        predicted = model(
            batch.spatial_inputs, batch.spacing_um, batch.dref_um
        )

    predicted = replace(
        predicted,
        foreground_logits=predicted.foreground_logits.float(),
        surface_logits=predicted.surface_logits.float(),
        separator_logits=predicted.separator_logits.float(),
        sdf=predicted.sdf.float(),
        flow=predicted.flow.float(),
        centroid_offset=predicted.centroid_offset.float(),
        seed_logits=predicted.seed_logits.float(),
    )
    oracle = oracle_geometry_from_targets(batch.targets)
    oracle_cache = build_geometry_derived_cache(
        oracle, cfg.partition, padding_mask=None
    )

    variants = [
        ("predicted", predicted),
        ("oracle_all", oracle),
        ("oracle_foreground", hybrid_geometry(predicted, oracle, "foreground")),
        ("oracle_seed", hybrid_geometry(predicted, oracle, "seed")),
        ("oracle_sdf", hybrid_geometry(predicted, oracle, "sdf")),
        ("oracle_separator", hybrid_geometry(predicted, oracle, "separator")),
        ("oracle_surface", hybrid_geometry(predicted, oracle, "surface")),
    ]

    summaries: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Tensor]] = {}
    start = time.perf_counter()

    for name, geometry in variants:
        summary, artifact = run_variant(
            name=name,
            geometry=geometry,
            batch=batch,
            cfg=cfg,
            complete_gt_ids=complete_gt_ids,
        )
        summaries[name] = summary
        artifacts[name] = artifact

    summary, artifact = run_variant(
        name="oracle_marker_score",
        geometry=predicted,
        batch=batch,
        cfg=cfg,
        complete_gt_ids=complete_gt_ids,
        forced_marker_score=oracle_cache.seed_score[0, 0],
    )
    summaries["oracle_marker_score"] = summary
    artifacts["oracle_marker_score"] = artifact

    summary, artifact = run_variant(
        name="oracle_energy",
        geometry=predicted,
        batch=batch,
        cfg=cfg,
        complete_gt_ids=complete_gt_ids,
        forced_energy=oracle_cache.watershed_energy[0, 0],
    )
    summaries["oracle_energy"] = summary
    artifacts["oracle_energy"] = artifact

    elapsed = time.perf_counter() - start
    print_variant_table(summaries)

    passed, failures = predicted_acceptance(summaries["predicted"])
    print("\nPredicted proposal safety")
    print(f"  Acceptance: {'PASS' if passed else 'NOT YET PASSING'}")
    for failure in failures:
        print(f"  - {failure}")
    if not failures:
        print("  - every complete GT cell has a marker")
        print("  - complete GT coverage is adequate")
        print("  - no meaningful cross-GT supervoxel was created")
        print("  - every complete GT cell is recoverable by merging supervoxels")

    summary_payload = {
        "format_version": 1,
        "kind": "stirnet_watershed_supervoxel_debug",
        "sample_path": str(args.sample_path),
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_stage": checkpoint.get("stage"),
        "complete_gt_ids": complete_gt_ids,
        "partial_gt_ids": partial_gt_ids,
        "partition_config": {
            "foreground_threshold": cfg.partition.foreground_threshold,
            "seed_threshold": cfg.partition.seed_threshold,
            "seed_min_distance_dref": cfg.partition.seed_min_distance_dref,
            "seed_sdf_weight": cfg.partition.seed_sdf_weight,
            "seed_head_weight": cfg.partition.seed_head_weight,
            "watershed_separator_weight": cfg.partition.watershed_separator_weight,
            "watershed_surface_weight": cfg.partition.watershed_surface_weight,
            "watershed_sdf_weight": cfg.partition.watershed_sdf_weight,
            "min_supervoxel_voxels": cfg.partition.min_supervoxel_voxels,
            "max_supervoxels": cfg.partition.max_supervoxels,
            "rag_min_node_purity": cfg.partition.rag_min_node_purity,
            "rag_min_node_gt_support": cfg.partition.rag_min_node_gt_support,
            "watershed_backend": cfg.partition.watershed_backend,
            "component_bounded_watershed": (
                cfg.partition.component_bounded_watershed
            ),
        },
        "diagnostic_thresholds": {
            "meaningful_overlap_min_voxels": MEANINGFUL_OVERLAP_MIN_VOXELS,
            "meaningful_overlap_min_gt_fraction": (
                MEANINGFUL_OVERLAP_MIN_GT_FRACTION
            ),
            "min_complete_gt_coverage": MIN_COMPLETE_GT_COVERAGE,
            "min_node_purity_for_safe": MIN_NODE_PURITY_FOR_SAFE,
            "min_node_gt_support_for_safe": MIN_NODE_GT_SUPPORT_FOR_SAFE,
        },
        "predicted_acceptance": passed,
        "predicted_failures": failures,
        "variants": summaries,
        "elapsed_seconds": elapsed,
    }
    artifact_payload = {
        "format_version": 1,
        "kind": "stirnet_watershed_supervoxel_debug",
        "spacing_um": batch.spacing_um[0].cpu(),
        "dref_um": batch.dref_um[0].cpu(),
        "complete_gt_ids": complete_gt_ids,
        "partial_gt_ids": partial_gt_ids,
        "variants": artifacts,
    }

    if device.type == "cuda":
        allocated, reserved = stage01.gpu_peak_gib()
        summary_payload["peak_cuda_allocated_gib"] = allocated
        summary_payload["peak_cuda_reserved_gib"] = reserved
    else:
        allocated = reserved = 0.0

    atomic_json(json_path, summary_payload)
    atomic_torch_save(artifact_path, artifact_payload)

    print("\n" + "=" * 110)
    print("Stage-03 diagnostics complete")
    print("=" * 110)
    print(f"JSON summary            : {json_path}")
    print(f"Napari artifact         : {artifact_path}")
    print(f"Elapsed diagnostics     : {elapsed:.2f} s")
    if device.type == "cuda":
        print(
            f"Peak CUDA memory        : {allocated:.3f} GiB allocated / "
            f"{reserved:.3f} GiB reserved"
        )
    print("\nVisualize with:")
    print(
        "  python investigations/stirnet/"
        "03_watershed_supervoxel_debug.py --visualize"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
