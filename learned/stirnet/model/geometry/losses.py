from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import GeometryConfig
from ..types import GeometryState
from ..utils.physical import physical_gradient3d
from .targets import GeometryTargets


def soft_dice_loss(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    p = logits.sigmoid().flatten(1)
    t = target.flatten(1)
    return (1.0 - (2 * (p * t).sum(-1) + eps) / (p.sum(-1) + t.sum(-1) + eps)).mean()


def weighted_bce(logits: Tensor, target: Tensor, pos_weight: float) -> Tensor:
    weight = logits.new_tensor([float(pos_weight)])
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)


class GeometryCriterion(nn.Module):
    def __init__(self, cfg: GeometryConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        pred: GeometryState,
        target: GeometryTargets,
        spacing_um: Tensor,
        dref_um: Tensor,
    ) -> Dict[str, Tensor]:
        fg = target.foreground
        losses: Dict[str, Tensor] = {}
        losses["foreground_bce"] = F.binary_cross_entropy_with_logits(
            pred.foreground_logits, fg
        )
        losses["foreground_dice"] = soft_dice_loss(pred.foreground_logits, fg)
        losses["surface"] = weighted_bce(
            pred.surface_logits, target.surface, self.cfg.boundary_pos_weight
        )
        losses["separator"] = weighted_bce(
            pred.separator_logits, target.separator, self.cfg.separator_pos_weight
        )

        near = target.sdf.abs() <= self.cfg.sdf_clip_dref
        losses["sdf"] = F.smooth_l1_loss(pred.sdf[near], target.sdf[near])

        fg3 = fg.expand_as(pred.flow) > 0.5
        if fg3.any():
            pred_flow = F.normalize(pred.flow, dim=1, eps=1e-6)
            target_flow = F.normalize(target.flow, dim=1, eps=1e-6)
            cosine = (pred_flow * target_flow).sum(dim=1, keepdim=True)
            losses["flow_direction"] = ((1 - cosine) * fg).sum() / fg.sum().clamp_min(1)
            losses["flow_l1"] = F.smooth_l1_loss(
                pred.flow[fg3], target.flow[fg3]
            )
            off_mask = fg.expand_as(pred.centroid_offset) > 0.5
            losses["centroid_offset"] = F.smooth_l1_loss(
                pred.centroid_offset[off_mask], target.centroid_offset[off_mask]
            )
        else:
            zero = pred.sdf.sum() * 0
            losses["flow_direction"] = zero
            losses["flow_l1"] = zero
            losses["centroid_offset"] = zero

        # A soft marker map is regression-supervised; it should track the
        # medial structure rather than only a single chosen centroid voxel.
        losses["seed"] = F.binary_cross_entropy_with_logits(
            pred.seed_logits, target.seed,
            pos_weight=pred.seed_logits.new_tensor([self.cfg.seed_pos_weight]),
        )

        # End-to-end geometric consistency: the predicted flow should agree
        # with the physical gradient of the predicted SDF. Because pred.sdf is
        # expressed in dref units, convert to micrometres before differentiating.
        if spacing_um.ndim == 1:
            spacing_um = spacing_um[None]
        sdf_um = pred.sdf * dref_um[:, None, None, None, None]
        grad = physical_gradient3d(sdf_um, spacing_um)
        grad_norm = torch.linalg.vector_norm(grad.float(), dim=1, keepdim=True)
        grad_dir = grad / grad_norm.clamp_min(1e-6).to(grad.dtype)
        pred_dir = F.normalize(pred.flow, dim=1, eps=1e-6)
        consistency = (1 - (grad_dir * pred_dir).sum(1, keepdim=True)) * fg
        losses["flow_sdf_consistency"] = (
            consistency.sum() / fg.sum().clamp_min(1)
        ) * self.cfg.consistency_weight

        # Eikonal regularization is applied only to sufficiently interior GT
        # voxels where the physical distance gradient should have unit norm.
        interior = (target.sdf > 0.15).float()
        losses["eikonal"] = (
            ((grad_norm - 1.0).abs() * interior).sum()
            / interior.sum().clamp_min(1)
        ) * self.cfg.eikonal_weight
        return losses
