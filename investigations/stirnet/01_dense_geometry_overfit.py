from __future__ import annotations

"""Stage 01: isolated overfit of STIR-Net's pre-watershed dense geometry path.

Training:
    python investigations/stirnet/01_dense_geometry_overfit.py --stage boundary
    python investigations/stirnet/01_dense_geometry_overfit.py --stage sdf
    python investigations/stirnet/01_dense_geometry_overfit.py --stage flow
    python investigations/stirnet/01_dense_geometry_overfit.py --stage center
    python investigations/stirnet/01_dense_geometry_overfit.py --stage joint
    python investigations/stirnet/01_dense_geometry_overfit.py --stage full

Visualization of the single latest successful stage:
    python investigations/stirnet/01_dense_geometry_overfit.py --visualize boundary
    python investigations/stirnet/01_dense_geometry_overfit.py --visualize sdf
    python investigations/stirnet/01_dense_geometry_overfit.py --visualize flow
    python investigations/stirnet/01_dense_geometry_overfit.py --visualize center
    python investigations/stirnet/01_dense_geometry_overfit.py --visualize joint
    python investigations/stirnet/01_dense_geometry_overfit.py --visualize full
    python investigations/stirnet/01_dense_geometry_overfit.py --visualize all

Change EXECUTION_BACKEND below to "local" or "modal". Visualization always runs
locally from stored final predictions, so it does not need a training GPU.

Only one large successful checkpoint is retained. A later successful stage
atomically replaces it; failed stages only overwrite a tiny last_attempt.json.

The model executed here is exactly:
    AcquisitionEmbedding -> EvidenceFusionStem -> AnisotropyAwareSpatialBackbone
    -> DenseGeometryDecoder -> 7 dense geometry outputs
and stops before watershed / supervoxels / RAG / temporal / refinement.
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

# =============================================================================
# USER-EDITABLE SETTINGS
# =============================================================================

EXECUTION_BACKEND = "local"  # "local" or "modal"

DEFAULT_SEED = 230525
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_AMP_DTYPE = "bf16"  # bf16 | fp16 | fp32
DEFAULT_EVAL_EVERY = 20
PASS_STREAK_FOR_EARLY_STOP = 2
GRAD_CLIP_NORM = 5.0
WEIGHT_DECAY = 0.0
DISABLE_PRIOR_DROPOUT = True

STAGE_STEPS = {
    "boundary": 400,
    "sdf": 400,
    "flow": 500,
    "center": 500,
    "joint": 700,
    "full": 800,
}

# Acceptance is based on the actual output fields, not an arbitrary total loss.
PASS_THRESHOLDS = {
    "boundary": {
        "foreground_dice_min": 0.97,
        "surface_dice_min": 0.82,
        "separator_dice_min": 0.78,
        "separator_recall_min": 0.85,
    },
    "sdf": {
        "sdf_mae_max": 0.10,
        "sdf_corr_min": 0.95,
        "sdf_sign_accuracy_min": 0.98,
    },
    "flow": {
        "flow_cosine_min": 0.97,
        "flow_angle_median_deg_max": 10.0,
        "flow_angle_p90_deg_max": 20.0,
        "flow_l1_max": 0.10,
        "flow_magnitude_mae_max": 0.10,
        "flow_magnitude_mae_reliable_max": 0.06,
        "flow_background_magnitude_mean_max": 0.08,
    },
    "center": {
        "centroid_endpoint_median_um_max": 1.50,
        "seed_mae_foreground_max": 0.12,
        "seed_corr_foreground_min": 0.85,
    },
    "joint": {
        "foreground_dice_min": 0.95,
        "surface_dice_min": 0.78,
        "separator_dice_min": 0.72,
        "separator_recall_min": 0.80,
        "sdf_mae_max": 0.13,
        "sdf_corr_min": 0.92,
        "flow_cosine_min": 0.85,
        "flow_angle_median_deg_max": 32.0,
        "centroid_endpoint_median_um_max": 2.00,
        "seed_mae_foreground_max": 0.15,
    },
    "full": {
        "foreground_dice_min": 0.95,
        "surface_dice_min": 0.78,
        "separator_dice_min": 0.72,
        "separator_recall_min": 0.80,
        "sdf_mae_max": 0.13,
        "sdf_corr_min": 0.92,
        "flow_cosine_min": 0.85,
        "flow_angle_median_deg_max": 32.0,
        "centroid_endpoint_median_um_max": 2.00,
        "seed_mae_foreground_max": 0.15,
    },
}

# Modal defaults. L4 has ample memory for this isolated crop while being cheaper
# than the L40S used by the broader experiment runner. Change if desired.
MODAL_GPU = "L4"
MODAL_TIMEOUT_SECONDS = 60 * 60
MODAL_CPU = 4.0
MODAL_MEMORY_MB = 12_288
MODAL_DATA_VOLUME_NAME = "stirnet-data"
MODAL_RUNS_VOLUME_NAME = "stirnet-runs"

# =============================================================================
# PATHS / REPOSITORY IMPORTS
# =============================================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

DEFAULT_SAMPLE_PATH = REPOSITORY_ROOT / "data" / "learned" / "stirnet" / "debug_crop.pt"
DEFAULT_RESULT_DIR = REPOSITORY_ROOT / "data" / "learned" / "stirnet" / "dense_geometry_debug"

REMOTE_REPO_ROOT = "/workspace/cell-tracking"
REMOTE_SAMPLE_PATH = "/data/learned/stirnet/debug_crop.pt"
REMOTE_RESULT_DIR = "/runs/stirnet/investigations/dense_geometry"
REMOTE_RESULT_VOLUME_PREFIX = "stirnet/investigations/dense_geometry"

STAGES = ("boundary", "sdf", "flow", "center", "joint", "full")
VISUALIZATIONS = (*STAGES, "all")

from learned.stirnet import StirNetConfig
from learned.stirnet.model.geometry.decoder import DenseGeometryDecoder
from learned.stirnet.model.spatial.acquisition import AcquisitionEmbedding
from learned.stirnet.model.spatial.backbone import AnisotropyAwareSpatialBackbone
from learned.stirnet.model.spatial.evidence_stem import EvidenceFusionStem
from learned.stirnet.model.utils.physical import physical_gradient3d

try:
    from tqdm.auto import trange as tqdm_trange
except ImportError:
    tqdm_trange = None

# =============================================================================
# EXACT PRE-WATERSHED MODEL
# =============================================================================

class DenseGeometryOnlyModel(nn.Module):
    """Only the modules executed by StirNet before execution_stage='geometry'."""

    def __init__(self, cfg: StirNetConfig):
        super().__init__()
        if cfg.spatial.canonical_spacing_um is not None:
            raise ValueError("Stage 01 expects native-resolution geometry.")
        self.acquisition = AcquisitionEmbedding(cfg.spatial.acquisition_dim)
        self.evidence_stem = EvidenceFusionStem(cfg.evidence, cfg.spatial)
        self.spatial_backbone = AnisotropyAwareSpatialBackbone(cfg.spatial)
        self.geometry_decoder = DenseGeometryDecoder(cfg.geometry, cfg.spatial)

    def forward(self, spatial_inputs: Tensor, spacing_um: Tensor, dref_um: Tensor):
        acquisition = self.acquisition(spacing_um, dref_um)
        stem = self.evidence_stem(spatial_inputs, acquisition)
        _, decoded = self.spatial_backbone(stem, spacing_um, acquisition, padding_mask=None)
        return self.geometry_decoder(decoded.d0, acquisition)


def build_debug_config() -> StirNetConfig:
    cfg = StirNetConfig()
    if DISABLE_PRIOR_DROPOUT:
        cfg.evidence.prior_dropout = 0.0
    cfg.spatial.activation_checkpointing = True
    cfg.validate()
    return cfg

# =============================================================================
# SAMPLE
# =============================================================================

@dataclass
class DebugBatch:
    spatial_inputs: Tensor
    current_labels: Tensor
    gt_labels: Tensor
    spacing_um: Tensor
    dref_um: Tensor
    targets: dict[str, Tensor]
    selection: dict
    source: dict

_REQUIRED_TARGETS = {
    "foreground", "surface", "separator", "sdf", "sdf_valid",
    "flow", "centroid_offset", "seed",
}


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_debug_batch(path: Path, device: torch.device) -> DebugBatch:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stage-00 sample: {path}\nRun 00_GT_validation.py first."
        )
    payload = torch_load(path)
    required = {
        "spatial_inputs", "current_labels", "gt_labels", "spacing_um",
        "dref_um", "geometry_targets",
    }
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"debug_crop.pt missing keys: {sorted(missing)}")

    target_payload = payload["geometry_targets"]
    missing_targets = _REQUIRED_TARGETS - set(target_payload)
    if missing_targets:
        raise RuntimeError(f"debug_crop.pt missing targets: {sorted(missing_targets)}")

    spatial = torch.as_tensor(payload["spatial_inputs"]).float()
    current = torch.as_tensor(payload["current_labels"]).long()
    gt = torch.as_tensor(payload["gt_labels"]).long()
    spacing = torch.as_tensor(payload["spacing_um"]).float()
    dref = torch.as_tensor(payload["dref_um"]).float().reshape(())

    if spatial.ndim != 4 or spatial.shape[0] != 5:
        raise RuntimeError(f"Expected spatial_inputs [5,Z,Y,X], got {tuple(spatial.shape)}")
    if current.shape != gt.shape or tuple(spatial.shape[-3:]) != tuple(gt.shape):
        raise RuntimeError("Spatial/current/GT shapes disagree.")
    if spacing.shape != (3,):
        raise RuntimeError(f"Expected spacing [3], got {tuple(spacing.shape)}")

    targets: dict[str, Tensor] = {}
    for name in _REQUIRED_TARGETS:
        value = torch.as_tensor(target_payload[name])
        if value.ndim != 4:
            raise RuntimeError(f"Target {name} should be [C,Z,Y,X], got {tuple(value.shape)}")
        targets[name] = value.unsqueeze(0).to(device, non_blocking=True)

    return DebugBatch(
        spatial_inputs=spatial.unsqueeze(0).to(device, non_blocking=True),
        current_labels=current,
        gt_labels=gt,
        spacing_um=spacing.unsqueeze(0).to(device, non_blocking=True),
        dref_um=dref.reshape(1).to(device, non_blocking=True),
        targets=targets,
        selection=dict(payload.get("selection", {})),
        source=dict(payload.get("source", {})),
    )

# =============================================================================
# LOSSES — SAME DIRECT/CROSS-GEOMETRY TERMS AS PRODUCTION
# =============================================================================

def soft_dice_loss(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    p = logits.sigmoid().flatten(1)
    t = target.flatten(1)
    return (1.0 - (2 * (p * t).sum(-1) + eps) / (p.sum(-1) + t.sum(-1) + eps)).mean()


def weighted_bce(logits: Tensor, target: Tensor, pos_weight: float) -> Tensor:
    voxel_weight = 1.0 + (float(pos_weight) - 1.0) * target.detach()

    return F.binary_cross_entropy_with_logits(
        logits,
        target,
        weight=voxel_weight,
    )


def geometry_loss_terms(
    pred,
    target: dict[str, Tensor],
    spacing_um: Tensor,
    dref_um: Tensor,
    cfg: StirNetConfig,
    requested: set[str],
) -> dict[str, Tensor]:
    """Compute only the terms active in the current isolation stage.

    This matters for the 6 GB local-GPU test: boundary/SDF/flow/center should
    not pay the activation/memory cost of unused consistency regularizers.
    """
    gcfg = cfg.geometry
    fg = target["foreground"]
    losses: dict[str, Tensor] = {}

    if "foreground_bce" in requested:
        losses["foreground_bce"] = F.binary_cross_entropy_with_logits(
            pred.foreground_logits, fg
        )
    if "foreground_dice" in requested:
        losses["foreground_dice"] = soft_dice_loss(pred.foreground_logits, fg)
    if "surface_bce" in requested:
        losses["surface_bce"] = weighted_bce(
            pred.surface_logits, target["surface"], gcfg.boundary_pos_weight
        )
    if "surface_dice" in requested:
        losses["surface_dice"] = gcfg.surface_dice_weight * soft_dice_loss(
            pred.surface_logits, target["surface"]
        )
    if "separator_bce" in requested:
        losses["separator_bce"] = weighted_bce(
            pred.separator_logits, target["separator"], gcfg.separator_pos_weight
        )
    if "separator_dice" in requested:
        losses["separator_dice"] = gcfg.separator_dice_weight * soft_dice_loss(
            pred.separator_logits, target["separator"]
        )

    if "sdf" in requested:
        sdf_valid = target["sdf_valid"].bool()
        losses["sdf"] = (
            F.smooth_l1_loss(pred.sdf[sdf_valid], target["sdf"][sdf_valid])
            if sdf_valid.any()
            else pred.sdf.sum() * 0
        )

    need_flow_direct = bool({"flow_direction", "flow_l1"} & requested)
    need_offset = "centroid_offset" in requested
    if need_flow_direct or need_offset:
        fg3 = fg.expand_as(pred.flow) > 0.5
        if fg3.any():
            if "flow_direction" in requested:
                pred_flow = F.normalize(pred.flow, dim=1, eps=1e-6)
                target_flow = F.normalize(target["flow"], dim=1, eps=1e-6)
                cosine = (pred_flow * target_flow).sum(1, keepdim=True)
                losses["flow_direction"] = (
                    ((1 - cosine) * fg).sum() / fg.sum().clamp_min(1)
                )
            if "flow_l1" in requested:
                losses["flow_l1"] = F.smooth_l1_loss(
                    pred.flow[fg3], target["flow"][fg3]
                )
            if need_offset:
                off_mask = fg.expand_as(pred.centroid_offset) > 0.5
                losses["centroid_offset"] = F.smooth_l1_loss(
                    pred.centroid_offset[off_mask],
                    target["centroid_offset"][off_mask],
                )
        else:
            zero = pred.sdf.sum() * 0
            if "flow_direction" in requested:
                losses["flow_direction"] = zero
            if "flow_l1" in requested:
                losses["flow_l1"] = zero
            if need_offset:
                losses["centroid_offset"] = zero

    if "flow_background" in requested:
        near_background = (
            (fg < 0.5)
            & (target["surface"] > gcfg.flow_background_surface_threshold)
        )
        near_background3 = near_background.expand_as(pred.flow)
        if near_background3.any():
            losses["flow_background"] = (
                gcfg.flow_background_weight
                * F.smooth_l1_loss(
                    pred.flow[near_background3],
                    torch.zeros_like(pred.flow[near_background3]),
                )
            )
        else:
            losses["flow_background"] = pred.flow.sum() * 0

    if "seed" in requested:
        losses["seed"] = weighted_bce(
            pred.seed_logits, target["seed"], gcfg.seed_pos_weight
        )

    if {"flow_sdf_consistency", "eikonal"} & requested:
        sdf_um = pred.sdf * dref_um[:, None, None, None, None]
        grad = physical_gradient3d(sdf_um, spacing_um)
        grad_norm = torch.linalg.vector_norm(grad.float(), dim=1, keepdim=True)

        if "flow_sdf_consistency" in requested:
            grad_dir = grad / grad_norm.clamp_min(1e-6).to(grad.dtype)
            pred_dir = F.normalize(pred.flow, dim=1, eps=1e-6)
            consistency = (1 - (grad_dir * pred_dir).sum(1, keepdim=True)) * fg
            losses["flow_sdf_consistency"] = (
                consistency.sum() / fg.sum().clamp_min(1)
            ) * gcfg.consistency_weight

        if "eikonal" in requested:
            interior = (target["sdf"] > 0.15).float()
            losses["eikonal"] = (
                ((grad_norm - 1.0).abs() * interior).sum()
                / interior.sum().clamp_min(1)
            ) * gcfg.eikonal_weight

    return losses

_STAGE_LOSS_NAMES = {
    "boundary": (
        "foreground_bce", "foreground_dice", "surface_bce", "surface_dice",
        "separator_bce", "separator_dice",
    ),
    "sdf": ("sdf",),
    "flow": ("flow_direction", "flow_l1", "flow_background"),
    "center": ("centroid_offset", "seed"),
    "joint": (
        "foreground_bce", "foreground_dice", "surface_bce", "surface_dice",
        "separator_bce", "separator_dice", "sdf", "flow_direction", "flow_l1", "flow_background",
        "centroid_offset", "seed",
    ),
    "full": (
        "foreground_bce", "foreground_dice", "surface_bce", "surface_dice",
        "separator_bce", "separator_dice", "sdf", "flow_direction", "flow_l1", "flow_background",
        "centroid_offset", "seed", "flow_sdf_consistency", "eikonal",
    ),
}


def stage_loss(stage: str, pred, target: dict[str, Tensor], spacing_um: Tensor, dref_um: Tensor, cfg: StirNetConfig):
    names = _STAGE_LOSS_NAMES[stage]
    selected = geometry_loss_terms(
        pred, target, spacing_um, dref_um, cfg, requested=set(names)
    )
    return sum(selected[name] for name in names), selected

# =============================================================================
# METRICS / PASS CRITERIA
# =============================================================================

def binary_dice(prob: Tensor, target: Tensor, threshold: float = 0.5) -> float:
    p = prob >= threshold
    t = target >= threshold
    den = p.sum().float() + t.sum().float()
    if den.item() == 0:
        return 1.0
    return float((2 * (p & t).sum().float() / den).item())


def binary_recall(prob: Tensor, target: Tensor, threshold: float = 0.5) -> float:
    p = prob >= threshold
    t = target >= threshold
    positives = t.sum().float()
    if positives.item() == 0:
        return 1.0
    return float(((p & t).sum().float() / positives).item())


def binary_precision(prob: Tensor, target: Tensor, threshold: float = 0.5) -> float:
    p = prob >= threshold
    t = target >= threshold
    positives = p.sum().float()
    if positives.item() == 0:
        return 1.0 if t.sum().item() == 0 else 0.0
    return float(((p & t).sum().float() / positives).item())


def soft_dice_score(prob: Tensor, target: Tensor, eps: float = 1e-6) -> float:
    p = prob.float().flatten(1)
    t = target.float().flatten(1)
    score = (2 * (p * t).sum(-1) + eps) / (p.sum(-1) + t.sum(-1) + eps)
    return float(score.mean().item())


def pearson(a: Tensor, b: Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    if a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    den = torch.sqrt(a.square().sum() * b.square().sum()).clamp_min(1e-12)
    return float(((a * b).sum() / den).item())


@torch.no_grad()
def collect_metrics(pred, target: dict[str, Tensor], dref_um: Tensor) -> dict[str, float]:
    fg = target["foreground"] > 0.5
    fg3 = fg.expand_as(pred.flow)
    fg_prob = pred.foreground_logits.sigmoid()
    surf_prob = pred.surface_logits.sigmoid()
    sep_prob = pred.separator_logits.sigmoid()
    seed_prob = pred.seed_logits.sigmoid()

    m: dict[str, float] = {
        "foreground_dice": binary_dice(fg_prob, target["foreground"]),
        "foreground_soft_dice": soft_dice_score(fg_prob, target["foreground"]),
        "surface_dice": binary_dice(surf_prob, target["surface"]),
        "surface_soft_dice": soft_dice_score(surf_prob, target["surface"]),
        "surface_precision": binary_precision(surf_prob, target["surface"]),
        "surface_recall": binary_recall(surf_prob, target["surface"]),
        "separator_dice": binary_dice(sep_prob, target["separator"]),
        "separator_soft_dice": soft_dice_score(sep_prob, target["separator"]),
        "separator_precision": binary_precision(sep_prob, target["separator"]),
        "separator_recall": binary_recall(sep_prob, target["separator"]),
    }

    valid = target["sdf_valid"].bool()
    ps = pred.sdf[valid].float()
    ts = target["sdf"][valid].float()
    m["sdf_mae"] = float((ps - ts).abs().mean().item())
    m["sdf_rmse"] = float(torch.sqrt((ps - ts).square().mean()).item())
    m["sdf_corr"] = pearson(ps, ts)
    m["sdf_sign_accuracy"] = float(((ps > 0) == (ts > 0)).float().mean().item())

    pred_flow = F.normalize(pred.flow.float(), dim=1, eps=1e-6)
    target_flow = F.normalize(target["flow"].float(), dim=1, eps=1e-6)
    cos = (pred_flow * target_flow).sum(1, keepdim=True)[fg].clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cos))
    m["flow_cosine"] = float(cos.mean().item())
    m["flow_angle_median_deg"] = float(torch.quantile(angle, 0.50).item())
    m["flow_angle_p90_deg"] = float(torch.quantile(angle, 0.90).item())
    m["flow_l1"] = float((pred.flow[fg3] - target["flow"][fg3]).abs().mean().item())

    pred_mag = torch.linalg.vector_norm(pred.flow.float(), dim=1, keepdim=True)
    target_mag = torch.linalg.vector_norm(target["flow"].float(), dim=1, keepdim=True)

    valid_mag = fg & (target_mag > 0.1)
    if valid_mag.any():
        m["flow_magnitude_mae"] = float(
            (pred_mag[valid_mag] - target_mag[valid_mag]).abs().mean().item()
        )
        m["flow_magnitude_mean_pred"] = float(pred_mag[valid_mag].mean().item())
        m["flow_magnitude_mean_target"] = float(target_mag[valid_mag].mean().item())
    else:
        m["flow_magnitude_mae"] = float("nan")
        m["flow_magnitude_mean_pred"] = float("nan")
        m["flow_magnitude_mean_target"] = float("nan")

    # The normalized EDT gradient is least stable at deep medial maxima.
    reliable_flow = fg & (target["seed"] < 0.85) & (target_mag > 0.1)
    if reliable_flow.any():
        m["flow_magnitude_mae_reliable"] = float(
            (pred_mag[reliable_flow] - target_mag[reliable_flow]).abs().mean().item()
        )
    else:
        m["flow_magnitude_mae_reliable"] = float("nan")

    # Mirrors GeometryConfig.flow_background_surface_threshold (=0.05 by default).
    near_background = (~fg) & (target["surface"] > 0.05)
    if near_background.any():
        m["flow_background_magnitude_mean"] = float(
            pred_mag[near_background].mean().item()
        )
    else:
        m["flow_background_magnitude_mean"] = float("nan")

    endpoint_um = torch.linalg.vector_norm(
        pred.centroid_offset.float() - target["centroid_offset"].float(),
        dim=1,
        keepdim=True,
    ) * dref_um[:, None, None, None, None]
    endpoint_fg = endpoint_um[fg]
    m["centroid_endpoint_mean_um"] = float(endpoint_fg.mean().item())
    m["centroid_endpoint_median_um"] = float(torch.quantile(endpoint_fg, 0.50).item())
    m["centroid_endpoint_p95_um"] = float(torch.quantile(endpoint_fg, 0.95).item())

    seed_pred_fg = seed_prob[fg]
    seed_target_fg = target["seed"][fg]
    m["seed_mae_foreground"] = float((seed_pred_fg - seed_target_fg).abs().mean().item())
    m["seed_corr_foreground"] = pearson(seed_pred_fg, seed_target_fg)
    return m


def pass_report(stage: str, metrics: dict[str, float]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for rule, threshold in PASS_THRESHOLDS[stage].items():
        if rule.endswith("_min"):
            metric = rule[:-4]
            value = metrics.get(metric, float("nan"))
            if not (math.isfinite(value) and value >= threshold):
                failures.append(f"{metric}={value:.5g} < {threshold:.5g}")
        elif rule.endswith("_max"):
            metric = rule[:-4]
            value = metrics.get(metric, float("nan"))
            if not (math.isfinite(value) and value <= threshold):
                failures.append(f"{metric}={value:.5g} > {threshold:.5g}")
        else:
            raise RuntimeError(f"Unknown threshold rule: {rule}")
    return len(failures) == 0, failures

_STAGE_METRICS = {
    "boundary": (
        "foreground_dice",
        "surface_dice", "surface_soft_dice", "surface_precision", "surface_recall",
        "separator_dice", "separator_soft_dice", "separator_precision", "separator_recall",
    ),
    "sdf": ("sdf_mae", "sdf_rmse", "sdf_corr", "sdf_sign_accuracy"),
    "flow": (
        "flow_cosine",
        "flow_angle_median_deg",
        "flow_angle_p90_deg",
        "flow_l1",
        "flow_magnitude_mae",
        "flow_magnitude_mean_pred",
        "flow_magnitude_mean_target",
        "flow_magnitude_mae_reliable",
        "flow_background_magnitude_mean",
    ),
    "center": (
        "centroid_endpoint_mean_um", "centroid_endpoint_median_um", "centroid_endpoint_p95_um",
        "seed_mae_foreground", "seed_corr_foreground",
    ),
    "joint": (
        "foreground_dice", "surface_dice", "separator_dice", "separator_recall",
        "sdf_mae", "sdf_corr", "flow_cosine", "flow_angle_median_deg",
        "centroid_endpoint_median_um", "seed_mae_foreground",
    ),
    "full": (
        "foreground_dice", "surface_dice", "separator_dice", "separator_recall",
        "sdf_mae", "sdf_corr", "flow_cosine", "flow_angle_median_deg",
        "centroid_endpoint_median_um", "seed_mae_foreground",
    ),
}


def print_metrics(stage: str, metrics: dict[str, float], indent: str = "  ") -> None:
    for name in _STAGE_METRICS[stage]:
        print(f"{indent}{name:34s}: {metrics.get(name, float('nan')):.6f}")

# =============================================================================
# TRAINING HELPERS
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_amp(name: str, device: torch.device) -> tuple[torch.dtype, bool]:
    if name == "fp32":
        return torch.float32, False
    if name == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, True
        print("[WARN] bf16 unsupported; falling back to fp16")
        return torch.float16, True
    if name == "fp16":
        return torch.float16, True
    raise ValueError(name)


def autocast_context(device: torch.device, dtype: torch.dtype, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=dtype) if enabled else nullcontext()


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def gpu_peak_gib() -> tuple[float, float]:
    return (
        torch.cuda.max_memory_allocated() / 1024**3,
        torch.cuda.max_memory_reserved() / 1024**3,
    )


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def state_dict_fp16(model: nn.Module) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, value in model.state_dict().items():
        value = value.detach().cpu()
        result[name] = value.half() if value.is_floating_point() else value
    return result


@torch.no_grad()
def compact_predictions(pred) -> dict[str, Tensor]:
    return {
        "foreground": pred.foreground_logits[0, 0].sigmoid().detach().cpu().half(),
        "surface": pred.surface_logits[0, 0].sigmoid().detach().cpu().half(),
        "separator": pred.separator_logits[0, 0].sigmoid().detach().cpu().half(),
        "sdf": pred.sdf[0, 0].detach().cpu().half(),
        "flow": pred.flow[0].detach().cpu().half(),
        "centroid_offset": pred.centroid_offset[0].detach().cpu().half(),
        "seed": pred.seed_logits[0, 0].sigmoid().detach().cpu().half(),
    }


def config_snapshot(cfg: StirNetConfig) -> dict:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    try:
        return asdict(cfg)
    except TypeError:
        return {"repr": repr(cfg)}


@dataclass
class TrainOptions:
    stage: str
    steps: int
    learning_rate: float
    amp_dtype: str
    seed: int
    eval_every: int
    sample_path: Path
    result_dir: Path


def train_stage(opts: TrainOptions) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Stage 01 training requires CUDA. Set EXECUTION_BACKEND='modal' if needed."
        )

    device = torch.device("cuda")
    set_seed(opts.seed)
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    amp_dtype, amp_enabled = resolve_amp(opts.amp_dtype, device)
    scaler = make_scaler(amp_enabled and amp_dtype == torch.float16)
    cfg = build_debug_config()
    batch = load_debug_batch(opts.sample_path, device)
    model = DenseGeometryOnlyModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.learning_rate, weight_decay=WEIGHT_DECAY)

    total_params = sum(p.numel() for p in model.parameters())
    gpu_name = torch.cuda.get_device_name(device)
    gpu_total = torch.cuda.get_device_properties(device).total_memory / 1024**3

    print("\n" + "=" * 96)
    print("STIR-Net Stage 01 — dense geometry isolation")
    print("=" * 96)
    print(f"Stage                  : {opts.stage}")
    print("Execution boundary     : BEFORE watershed")
    print(f"Crop shape             : {tuple(batch.spatial_inputs.shape[-3:])}")
    print(
        f"Selected merge         : source {batch.selection.get('merge_source_id', '?')} "
        f"-> GT {tuple(batch.selection.get('merge_gt_ids', []))}"
    )
    print(f"Selected free GT cell  : {batch.selection.get('free_gt_id', '?')}")
    print(f"GPU                    : {gpu_name} ({gpu_total:.2f} GiB)")
    print(f"AMP                    : {opts.amp_dtype}")
    print(f"Activation checkpoint  : {cfg.spatial.activation_checkpointing}")
    print(f"Prior dropout          : {cfg.evidence.prior_dropout}")
    print(f"Trainable parameters   : {total_params / 1e6:.3f} M")
    print(f"Learning rate          : {opts.learning_rate:.3g}")
    print(f"Maximum steps          : {opts.steps}")
    print(f"Evaluation interval    : {opts.eval_every}")
    print("Active losses          : " + ", ".join(_STAGE_LOSS_NAMES[opts.stage]))
    print("=" * 96 + "\n")

    start = time.perf_counter()
    history: list[dict] = []
    pass_streak = 0
    stopped_early = False
    last_loss = float("nan")

    iterator = (
        tqdm_trange(1, opts.steps + 1, desc=f"stirnet:{opts.stage}", unit="step", dynamic_ncols=True)
        if tqdm_trange is not None
        else range(1, opts.steps + 1)
    )
    has_tqdm = tqdm_trange is not None

    try:
        for step in iterator:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_dtype, amp_enabled):
                pred = model(batch.spatial_inputs, batch.spacing_um, batch.dref_um)
                loss, selected = stage_loss(
                    opts.stage, pred, batch.targets, batch.spacing_um, batch.dref_um, cfg
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {float(loss)}")

            use_scaler = amp_enabled and amp_dtype == torch.float16
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            last_loss = float(loss.detach().float().item())
            elapsed = time.perf_counter() - start
            if has_tqdm:
                allocated, _ = gpu_peak_gib()
                iterator.set_postfix(loss=f"{last_loss:.4f}", mem=f"{allocated:.2f}G", refresh=(step % 5 == 0))

            if step == 1 or step % opts.eval_every == 0 or step == opts.steps:
                model.eval()
                with torch.no_grad(), autocast_context(device, amp_dtype, amp_enabled):
                    eval_pred = model(batch.spatial_inputs, batch.spacing_um, batch.dref_um)
                metrics = collect_metrics(eval_pred, batch.targets, batch.dref_um)
                passed_now, failures_now = pass_report(opts.stage, metrics)
                pass_streak = pass_streak + 1 if passed_now else 0
                history.append({
                    "step": int(step),
                    "elapsed_seconds": float(elapsed),
                    "loss": last_loss,
                    "grad_norm": float(torch.as_tensor(grad_norm).detach().float().cpu().item()),
                    "passed": bool(passed_now),
                    "failures": failures_now,
                    "metrics": metrics,
                    "loss_terms": {k: float(v.detach().float().cpu().item()) for k, v in selected.items()},
                })
                if has_tqdm and passed_now:
                    iterator.write(
                        f"[PASS candidate] step={step} streak={pass_streak}/{PASS_STREAK_FOR_EARLY_STOP}"
                    )
                elif not has_tqdm:
                    print(f"step={step:04d} loss={last_loss:.5f} elapsed={elapsed/60:.1f}m pass={passed_now}")

                if passed_now and pass_streak >= PASS_STREAK_FOR_EARLY_STOP:
                    stopped_early = True
                    if has_tqdm:
                        iterator.write("[EARLY STOP] acceptance held for consecutive evaluations")
                    break

            del pred, loss, selected

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("\n[CUDA OOM] The fixed crop does not fit comfortably with the production-width pre-watershed model.")
        print("Do not shrink the scientific test merely to force a PASS. Set EXECUTION_BACKEND='modal' and rerun.")
        raise

    model.eval()
    with torch.no_grad(), autocast_context(device, amp_dtype, amp_enabled):
        final_pred = model(batch.spatial_inputs, batch.spacing_um, batch.dref_um)
    final_metrics = collect_metrics(final_pred, batch.targets, batch.dref_um)
    passed, failures = pass_report(opts.stage, final_metrics)
    elapsed = time.perf_counter() - start
    steps_completed = history[-1]["step"] if history else opts.steps
    allocated, reserved = gpu_peak_gib()

    print("\n" + "=" * 96)
    print(f"Stage {opts.stage} final evaluation")
    print("=" * 96)
    print_metrics(opts.stage, final_metrics)
    print(f"\nAcceptance             : {'PASS' if passed else 'NOT YET PASSING'}")
    for failure in failures:
        print(f"  - {failure}")
    print(f"Elapsed                : {elapsed:.2f} s ({elapsed/60:.2f} min)")
    print(f"Completed steps        : {steps_completed}")
    print(f"Average sec/step       : {elapsed/max(steps_completed, 1):.4f}")
    print(f"Peak CUDA memory       : {allocated:.3f} GiB allocated / {reserved:.3f} GiB reserved")
    print(f"Stopped early          : {stopped_early}")
    print("=" * 96)

    opts.result_dir.mkdir(parents=True, exist_ok=True)
    attempt = {
        "format_version": 1,
        "stage": opts.stage,
        "passed": bool(passed),
        "failures": failures,
        "elapsed_seconds": float(elapsed),
        "steps_completed": int(steps_completed),
        "max_steps": int(opts.steps),
        "learning_rate": float(opts.learning_rate),
        "amp_dtype": opts.amp_dtype,
        "seed": int(opts.seed),
        "eval_every": int(opts.eval_every),
        "gpu": gpu_name,
        "peak_cuda_allocated_gib": float(allocated),
        "peak_cuda_reserved_gib": float(reserved),
        "metrics": final_metrics,
        "pass_thresholds": PASS_THRESHOLDS[opts.stage],
        "active_losses": list(_STAGE_LOSS_NAMES[opts.stage]),
        "sample_path": str(opts.sample_path),
        "sample_selection": batch.selection,
        "timestamp_unix": time.time(),
    }
    atomic_json(opts.result_dir / "last_attempt.json", attempt)

    # Debugging must remain visual even when a stage FAILS. Store only the
    # final dense predictions for the latest attempt (no model/optimizer state).
    # This file is small (~11 MiB for the current crop) and is overwritten on
    # every run, so it does not create checkpoint accumulation.
    attempt_pred_path = opts.result_dir / "last_attempt_predictions.pt"
    atomic_torch_save(
        attempt_pred_path,
        {
            "format_version": 1,
            "kind": "stirnet_dense_geometry_stage01_attempt_predictions",
            "stage": opts.stage,
            "passed": bool(passed),
            "predictions": compact_predictions(final_pred),
            "metrics": final_metrics,
            "attempt": attempt,
            "sample_selection": batch.selection,
            "sample_shape_zyx": list(batch.spatial_inputs.shape[-3:]),
            "spacing_um": batch.spacing_um[0].detach().cpu(),
            "dref_um": batch.dref_um[0].detach().cpu(),
        },
    )
    print(f"Latest-attempt predictions: {attempt_pred_path}")

    ckpt_path = opts.result_dir / "latest_success.pt"
    summary_path = opts.result_dir / "latest_success.json"
    if passed:
        payload = {
            "format_version": 1,
            "kind": "stirnet_dense_geometry_stage01",
            "stage": opts.stage,
            "model_state": state_dict_fp16(model),
            "predictions": compact_predictions(final_pred),
            "metrics": final_metrics,
            "history": history,
            "attempt": attempt,
            "config": config_snapshot(cfg),
            "sample_selection": batch.selection,
            "sample_source": batch.source,
            "sample_shape_zyx": list(batch.spatial_inputs.shape[-3:]),
            "spacing_um": batch.spacing_um[0].detach().cpu(),
            "dref_um": batch.dref_um[0].detach().cpu(),
        }
        atomic_torch_save(ckpt_path, payload)
        atomic_json(summary_path, attempt)
        print("\nSaved latest successful Stage-01 state:")
        print(f"  checkpoint : {ckpt_path}")
        print(f"  summary    : {summary_path}")
        print(f"  size       : {ckpt_path.stat().st_size / 1024**2:.2f} MiB")
        print("  policy     : replaced only by a later successful stage")
    else:
        print("\nNo large model checkpoint written; previous latest_success.pt was preserved.")
        print("Failure diagnostics were written to last_attempt.json.")
        print("Compact predictions were saved so --visualize can inspect this failed attempt.")

    return {
        "passed": bool(passed),
        "stage": opts.stage,
        "checkpoint_path": str(ckpt_path) if passed else None,
        "attempt_path": str(opts.result_dir / "last_attempt.json"),
        "attempt_predictions_path": str(attempt_pred_path),
        "elapsed_seconds": float(elapsed),
        "steps_completed": int(steps_completed),
        "metrics": final_metrics,
    }

# =============================================================================
# VISUALIZATION
# =============================================================================

def np_array(x) -> np.ndarray:
    return x.detach().float().cpu().numpy() if isinstance(x, Tensor) else np.asarray(x)


def sample_vectors(
    field: np.ndarray,
    mask: np.ndarray,
    spacing: np.ndarray,
    *,
    stride=(2, 10, 10),
    max_vectors=3000,
    physical_scale_um: float | None = None,
    normalized_by_dref: float | None = None,
    normalize_direction: bool = False,
) -> np.ndarray:
    z, y, x = np.indices(mask.shape)
    select = mask & (z % stride[0] == 0) & (y % stride[1] == 0) & (x % stride[2] == 0)
    points = np.argwhere(select)
    if points.size == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)
    if len(points) > max_vectors:
        points = points[:: int(np.ceil(len(points) / max_vectors))][:max_vectors]
    values = field[:, points[:, 0], points[:, 1], points[:, 2]].T.astype(np.float32)
    if normalize_direction:
        values /= np.maximum(np.linalg.vector_norm(values, axis=1, keepdims=True), 1e-6)
    if normalized_by_dref is not None:
        vector_um = values * normalized_by_dref
    elif physical_scale_um is not None:
        vector_um = values * physical_scale_um
    else:
        vector_um = values
    return np.stack([points.astype(np.float32), (vector_um / spacing[None]).astype(np.float32)], axis=1)


def visualize_latest(mode: str, sample_path: Path, result_dir: Path) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError("Napari is required for --visualize") from exc

    attempt_path = result_dir / "last_attempt_predictions.pt"
    success_path = result_dir / "latest_success.pt"

    source_label = None
    ckpt_path = None
    ckpt = None

    if attempt_path.exists():
        candidate = torch_load(attempt_path)
        candidate_stage = str(candidate.get("stage", ""))
        if mode == "all" or mode == candidate_stage:
            ckpt_path = attempt_path
            ckpt = candidate
            source_label = (
                "latest attempt (PASS)"
                if bool(candidate.get("passed", False))
                else "latest attempt (FAILED acceptance)"
            )

    if ckpt is None and success_path.exists():
        ckpt_path = success_path
        ckpt = torch_load(success_path)
        source_label = "latest successful model"

    if ckpt is None:
        raise FileNotFoundError(
            "No visualization data found. Expected either a matching "
            f"{attempt_path.name} or {success_path.name} in {result_dir}."
        )

    sample = torch_load(sample_path)
    pred = {k: np_array(v) for k, v in ckpt["predictions"].items()}
    target = {k: np_array(v) for k, v in sample["geometry_targets"].items()}
    for k in ("foreground", "surface", "separator", "sdf", "sdf_valid", "seed"):
        target[k] = target[k][0]

    raw = np_array(sample["raw_norm"])
    current = np_array(sample["current_labels"]).astype(np.int64)
    gt = np_array(sample["gt_labels"]).astype(np.int64)
    spacing = np_array(sample["spacing_um"]).astype(np.float32)
    dref = float(torch.as_tensor(sample["dref_um"]))
    scale = tuple(float(x) for x in spacing)
    fg = gt > 0

    trained_stage = str(ckpt["stage"])
    print("=" * 88)
    print("STIR-Net Stage-01 visualization")
    print("=" * 88)
    print(f"Visualization source   : {source_label}")
    print(f"Stored stage           : {trained_stage}")
    print(f"Requested view         : {mode}")
    print(f"Data file              : {ckpt_path}")
    print_metrics(trained_stage, dict(ckpt.get("metrics", {})))

    viewer = napari.Viewer(title=f"STIR-Net Stage 01 — saved={trained_stage}, view={mode}", ndisplay=3)
    viewer.add_image(raw, name="00 Raw normalized", colormap="gray", scale=scale)
    viewer.add_labels(current, name="01 Current segmentation", scale=scale, visible=False)
    viewer.add_labels(gt, name="02 GT labels", scale=scale, visible=True)

    show_boundary = mode in ("boundary", "joint", "full", "all")
    show_sdf = mode in ("sdf", "joint", "full", "all")
    show_flow = mode in ("flow", "joint", "full", "all")
    show_center = mode in ("center", "joint", "full", "all")

    if show_boundary:
        for i, name in enumerate(("foreground", "surface", "separator"), 3):
            cmap = "gray" if name == "foreground" else "inferno"
            viewer.add_image(target[name], name=f"{i:02d} Target — {name}", colormap=cmap, contrast_limits=(0, 1), scale=scale, visible=False)
            viewer.add_image(pred[name], name=f"{i:02d} Pred — {name}", colormap=cmap, contrast_limits=(0, 1), scale=scale, visible=(name == "separator"))
            viewer.add_image(np.abs(pred[name] - target[name]), name=f"{i:02d} Error — {name}", colormap="magma", contrast_limits=(0, 1), scale=scale, visible=False)

    if show_sdf:
        limit = max(float(np.abs(target["sdf"]).max()), float(np.abs(pred["sdf"]).max()), 1e-6)
        viewer.add_image(target["sdf"], name="20 Target — SDF / dref", colormap="turbo", contrast_limits=(-limit, limit), scale=scale, visible=False)
        viewer.add_image(pred["sdf"], name="21 Pred — SDF / dref", colormap="turbo", contrast_limits=(-limit, limit), scale=scale, visible=(mode == "sdf"))
        viewer.add_image(np.abs(pred["sdf"] - target["sdf"]), name="22 Error — |SDF|", colormap="magma", scale=scale, visible=False)

    if show_flow:
        tf = target["flow"]
        pf = pred["flow"]
        tn = np.linalg.vector_norm(tf, axis=0)
        pn = np.linalg.vector_norm(pf, axis=0)
        td = tf / np.maximum(tn[None], 1e-6)
        pd = pf / np.maximum(pn[None], 1e-6)
        angle = np.rad2deg(np.arccos(np.clip(np.sum(td * pd, axis=0), -1, 1)))
        angle[~fg] = 0

        magnitude_error = np.abs(pn - tn)
        magnitude_error[~fg] = 0
        near_background = (~fg) & (target["surface"] > 0.05)
        leakage = np.zeros_like(pn)
        leakage[near_background] = pn[near_background]

        viewer.add_image(
            tn, name="30 Target — flow magnitude", colormap="viridis",
            contrast_limits=(0.0, 1.0), scale=scale, visible=False,
        )
        viewer.add_image(
            pn, name="31 Pred — flow magnitude", colormap="viridis",
            contrast_limits=(0.0, 1.0), scale=scale, visible=False,
        )
        viewer.add_image(
            magnitude_error, name="32 Error — flow magnitude", colormap="magma",
            contrast_limits=(0.0, 0.5), scale=scale, visible=False,
        )
        viewer.add_image(
            angle, name="33 Error — flow angle (deg)", colormap="magma",
            contrast_limits=(0.0, 180.0), scale=scale, visible=(mode == "flow"),
        )
        viewer.add_image(
            leakage, name="34 Pred — near-background flow leakage", colormap="magma",
            contrast_limits=(0.0, 0.25), scale=scale, visible=False,
        )

        tv = sample_vectors(tf, fg, spacing, physical_scale_um=4.0, normalize_direction=True)
        pv = sample_vectors(pf, fg, spacing, physical_scale_um=4.0, normalize_direction=True)
        if len(tv):
            viewer.add_vectors(tv, name="35 Target — flow vectors", scale=scale, visible=False)
        if len(pv):
            viewer.add_vectors(pv, name="36 Pred — flow vectors", scale=scale, visible=False)

    if show_center:
        to = target["centroid_offset"]
        po = pred["centroid_offset"]
        endpoint = np.linalg.vector_norm(po - to, axis=0) * dref
        endpoint[~fg] = 0
        viewer.add_image(target["seed"], name="40 Target — seed", colormap="inferno", contrast_limits=(0, 1), scale=scale, visible=False)
        viewer.add_image(pred["seed"], name="41 Pred — seed", colormap="inferno", contrast_limits=(0, 1), scale=scale, visible=(mode == "center"))
        viewer.add_image(np.abs(pred["seed"] - target["seed"]), name="42 Error — seed", colormap="magma", contrast_limits=(0, 1), scale=scale, visible=False)
        viewer.add_image(np.linalg.vector_norm(to, axis=0) * dref, name="43 Target — centroid offset magnitude (um)", colormap="viridis", scale=scale, visible=False)
        viewer.add_image(np.linalg.vector_norm(po, axis=0) * dref, name="44 Pred — centroid offset magnitude (um)", colormap="viridis", scale=scale, visible=False)
        viewer.add_image(endpoint, name="45 Error — centroid endpoint (um)", colormap="magma", scale=scale, visible=False)
        tv = sample_vectors(to, fg, spacing, normalized_by_dref=dref)
        pv = sample_vectors(po, fg, spacing, normalized_by_dref=dref)
        if len(tv):
            viewer.add_vectors(tv, name="46 Target — centroid offset vectors", scale=scale, visible=False)
        if len(pv):
            viewer.add_vectors(pv, name="47 Pred — centroid offset vectors", scale=scale, visible=False)

    print("Napari loaded target/prediction/error layers for the exact Stage-00 crop.")
    napari.run()

# =============================================================================
# MODAL
# =============================================================================

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    modal_app = modal.App("stirnet-dense-geometry-debug")
    modal_data_volume = modal.Volume.from_name(MODAL_DATA_VOLUME_NAME)
    modal_runs_volume = modal.Volume.from_name(MODAL_RUNS_VOLUME_NAME, create_if_missing=True)
    modal_image = (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(
            "torch==2.13.0", "numpy==2.4.6", "scipy==1.17.1",
            "scikit-image==0.26.0", "networkx>=3.0", "tqdm>=4.66",
        )
        .workdir(REMOTE_REPO_ROOT)
        .add_local_dir(REPOSITORY_ROOT / "learned", remote_path=f"{REMOTE_REPO_ROOT}/learned")
        .add_local_dir(REPOSITORY_ROOT / "investigations", remote_path=f"{REMOTE_REPO_ROOT}/investigations")
    )

    @modal_app.function(
        image=modal_image,
        gpu=MODAL_GPU,
        cpu=MODAL_CPU,
        memory=MODAL_MEMORY_MB,
        timeout=MODAL_TIMEOUT_SECONDS,
        volumes={"/data": modal_data_volume, "/runs": modal_runs_volume},
    )
    def modal_train(stage: str, steps: int, learning_rate: float, amp_dtype: str, seed: int, eval_every: int) -> dict:
        actual_steps = STAGE_STEPS[stage] if steps <= 0 else steps
        report = train_stage(TrainOptions(
            stage=stage,
            steps=actual_steps,
            learning_rate=learning_rate,
            amp_dtype=amp_dtype,
            seed=seed,
            eval_every=eval_every,
            sample_path=Path(REMOTE_SAMPLE_PATH),
            result_dir=Path(REMOTE_RESULT_DIR),
        ))
        modal_runs_volume.commit()
        return report

    def download_volume_file(volume, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = local_path.with_name(f".{local_path.name}.{os.getpid()}.download")
        with tmp.open("wb") as f:
            for chunk in volume.read_file(remote_path):
                f.write(chunk)
        os.replace(tmp, local_path)

    @modal_app.local_entrypoint()
    def modal_main(
        stage: str = "boundary",
        steps: int = 0,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        amp_dtype: str = DEFAULT_AMP_DTYPE,
        seed: int = DEFAULT_SEED,
        eval_every: int = DEFAULT_EVAL_EVERY,
    ) -> None:
        if stage not in STAGES:
            raise ValueError(stage)
        if not DEFAULT_SAMPLE_PATH.exists():
            raise FileNotFoundError(DEFAULT_SAMPLE_PATH)

        print(f"Uploading approved debug_crop.pt to Modal volume {MODAL_DATA_VOLUME_NAME}...")
        with modal_data_volume.batch_upload() as upload:
            upload.put_file(str(DEFAULT_SAMPLE_PATH), "learned/stirnet/debug_crop.pt")

        print(f"Launching {stage} on Modal {MODAL_GPU}...")
        report = modal_train.remote(stage, int(steps), float(learning_rate), amp_dtype, int(seed), int(eval_every))

        DEFAULT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
        download_volume_file(
            modal_runs_volume,
            f"{REMOTE_RESULT_VOLUME_PREFIX}/last_attempt.json",
            DEFAULT_RESULT_DIR / "last_attempt.json",
        )
        download_volume_file(
            modal_runs_volume,
            f"{REMOTE_RESULT_VOLUME_PREFIX}/last_attempt_predictions.pt",
            DEFAULT_RESULT_DIR / "last_attempt_predictions.pt",
        )
        if report["passed"]:
            download_volume_file(
                modal_runs_volume,
                f"{REMOTE_RESULT_VOLUME_PREFIX}/latest_success.pt",
                DEFAULT_RESULT_DIR / "latest_success.pt",
            )
            download_volume_file(
                modal_runs_volume,
                f"{REMOTE_RESULT_VOLUME_PREFIX}/latest_success.json",
                DEFAULT_RESULT_DIR / "latest_success.json",
            )
            print(f"Downloaded latest success to {DEFAULT_RESULT_DIR / 'latest_success.pt'}")
        else:
            print("Remote stage did not pass; previous latest_success.pt was preserved.")


def launch_modal_from_python(args: argparse.Namespace) -> None:
    if modal is None:
        raise RuntimeError("EXECUTION_BACKEND='modal' but the modal package is not installed")
    exe = shutil.which("modal")
    if exe is None:
        raise RuntimeError("Could not find the modal executable")
    command = [
        exe, "run", str(Path(__file__).resolve()),
        "--stage", args.stage,
        "--steps", str(args.steps),
        "--learning-rate", str(args.learning_rate),
        "--amp-dtype", args.amp_dtype,
        "--seed", str(args.seed),
        "--eval-every", str(args.eval_every),
    ]
    print("EXECUTION_BACKEND='modal' -> launching:")
    print("  " + " ".join(command))
    subprocess.run(command, check=True)

# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STIR-Net Stage-01 dense geometry overfit/debugger")
    parser.add_argument("--stage", choices=STAGES, default=None)
    parser.add_argument("--visualize", choices=VISUALIZATIONS, default=None)
    parser.add_argument("--steps", type=int, default=0, help="0 = use STAGE_STEPS")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default=DEFAULT_AMP_DTYPE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()
    if (args.stage is None) == (args.visualize is None):
        parser.error("Specify exactly one of --stage or --visualize")
    if args.steps < 0:
        parser.error("--steps must be >= 0")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.eval_every <= 0:
        parser.error("--eval-every must be positive")
    return args


def main() -> None:
    args = parse_args()

    if args.visualize is not None:
        visualize_latest(args.visualize, args.sample_path.resolve(), args.result_dir.resolve())
        return

    backend = EXECUTION_BACKEND.strip().lower()
    if backend == "local":
        steps = STAGE_STEPS[args.stage] if args.steps <= 0 else args.steps
        train_stage(TrainOptions(
            stage=args.stage,
            steps=steps,
            learning_rate=args.learning_rate,
            amp_dtype=args.amp_dtype,
            seed=args.seed,
            eval_every=args.eval_every,
            sample_path=args.sample_path.resolve(),
            result_dir=args.result_dir.resolve(),
        ))
    elif backend == "modal":
        launch_modal_from_python(args)
    else:
        raise ValueError(f"EXECUTION_BACKEND must be 'local' or 'modal', got {EXECUTION_BACKEND!r}")


if __name__ == "__main__":
    main()
