from __future__ import annotations
from typing import Dict
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from ..config import GeometryConfig
from ..types import GeometryState
from ..utils.physical import physical_gradient3d
from .targets import GeometryTargets


def _valid(mask: Tensor | None, ref: Tensor) -> Tensor | None:
    if mask is None: return None
    v = torch.as_tensor(mask, device=ref.device).bool()
    if v.ndim == ref.ndim - 1: v = v[:,None]
    if v.ndim != ref.ndim or v.shape[0] != ref.shape[0] or v.shape[-3:] != ref.shape[-3:]: raise ValueError('geometry valid_mask must align [B,Z,Y,X]')
    if v.shape[1] not in {1, ref.shape[1]}: raise ValueError('invalid valid_mask channel count')
    return v.expand_as(ref)


def soft_dice_loss(logits: Tensor, target: Tensor, eps: float = 1e-6, *, valid_mask: Tensor | None = None) -> Tensor:
    p, t = logits.sigmoid(), target
    v = _valid(valid_mask, logits)
    if v is not None:
        w = v.to(p.dtype); p = p * w; t = t * w
    p = p.flatten(1); t = t.flatten(1)
    return (1.0 - (2 * (p*t).sum(-1) + eps) / (p.sum(-1) + t.sum(-1) + eps)).mean()


def weighted_bce(logits: Tensor, target: Tensor, pos_weight: float, *, valid_mask: Tensor | None = None) -> Tensor:
    weight = 1.0 + (float(pos_weight) - 1.0) * target.detach()
    loss = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction='none')
    v = _valid(valid_mask, loss)
    if v is None: return loss.mean()
    w = v.to(loss.dtype); return (loss * w).sum() / w.sum().clamp_min(1)


class GeometryCriterion(nn.Module):
    def __init__(self, cfg: GeometryConfig): super().__init__(); self.cfg = cfg

    def forward(self, pred: GeometryState, target: GeometryTargets, spacing_um: Tensor, dref_um: Tensor, *, valid_mask: Tensor | None = None) -> Dict[str, Tensor]:
        fg = target.foreground; v1 = _valid(valid_mask, fg); losses: Dict[str, Tensor] = {}
        losses['foreground_bce'] = weighted_bce(pred.foreground_logits, fg, 1.0, valid_mask=valid_mask)
        losses['foreground_dice'] = soft_dice_loss(pred.foreground_logits, fg, valid_mask=valid_mask)
        losses['surface_bce'] = weighted_bce(pred.surface_logits, target.surface, self.cfg.boundary_pos_weight, valid_mask=valid_mask)
        losses['surface_dice'] = self.cfg.surface_dice_weight * soft_dice_loss(pred.surface_logits, target.surface, valid_mask=valid_mask)
        losses['separator_bce'] = weighted_bce(pred.separator_logits, target.separator, self.cfg.separator_pos_weight, valid_mask=valid_mask)
        losses['separator_dice'] = self.cfg.separator_dice_weight * soft_dice_loss(pred.separator_logits, target.separator, valid_mask=valid_mask)
        sdf_valid = target.sdf_valid.bool() & (v1 if v1 is not None else torch.ones_like(target.sdf_valid, dtype=torch.bool))
        losses['sdf'] = F.smooth_l1_loss(pred.sdf[sdf_valid], target.sdf[sdf_valid]) if sdf_valid.any() else pred.sdf.sum() * 0
        fgv = (fg > .5) & (v1 if v1 is not None else torch.ones_like(fg, dtype=torch.bool)); fg3 = fgv.expand_as(pred.flow)
        if fg3.any():
            pf = F.normalize(pred.flow, dim=1, eps=1e-6); tf = F.normalize(target.flow, dim=1, eps=1e-6); cosine = (pf*tf).sum(1, keepdim=True)
            losses['flow_direction'] = (1-cosine)[fgv].mean(); losses['flow_l1'] = F.smooth_l1_loss(pred.flow[fg3], target.flow[fg3])
            pm = torch.linalg.vector_norm(pred.flow.float(), dim=1, keepdim=True); tm = torch.linalg.vector_norm(target.flow.float(), dim=1, keepdim=True); mv = fgv & (tm > .1)
            losses['flow_magnitude'] = self.cfg.flow_magnitude_weight * F.l1_loss(pm[mv], tm[mv]) if mv.any() else pred.flow.sum()*0
            off = fgv.expand_as(pred.centroid_offset); losses['centroid_offset'] = F.smooth_l1_loss(pred.centroid_offset[off], target.centroid_offset[off])
        else:
            zero = pred.sdf.sum()*0; losses.update(flow_direction=zero, flow_l1=zero, flow_magnitude=zero, centroid_offset=zero)
        near = (fg < .5) & (target.surface > self.cfg.flow_background_surface_threshold)
        if v1 is not None: near &= v1
        losses['flow_background'] = self.cfg.flow_background_weight * torch.linalg.vector_norm(pred.flow.float(), dim=1, keepdim=True)[near].mean() if near.any() else pred.flow.sum()*0
        losses['seed'] = weighted_bce(pred.seed_logits, target.seed, self.cfg.seed_pos_weight, valid_mask=valid_mask)
        if spacing_um.ndim == 1: spacing_um = spacing_um[None]
        sdf_um = pred.sdf * dref_um[:,None,None,None,None]; grad = physical_gradient3d(sdf_um, spacing_um); grad_norm = torch.linalg.vector_norm(grad.float(), dim=1, keepdim=True)
        grad_dir = grad / grad_norm.clamp_min(1e-6).to(grad.dtype); pred_dir = F.normalize(pred.flow, dim=1, eps=1e-6)
        cw = fg * (v1.to(fg.dtype) if v1 is not None else 1.0); consistency = (1 - (grad_dir*pred_dir).sum(1, keepdim=True)) * cw
        losses['flow_sdf_consistency'] = consistency.sum() / cw.sum().clamp_min(1) * self.cfg.consistency_weight
        interior = target.sdf > .15
        if v1 is not None: interior &= v1
        losses['eikonal'] = ((grad_norm-1).abs()[interior].mean() if interior.any() else pred.sdf.sum()*0) * self.cfg.eikonal_weight
        return losses
