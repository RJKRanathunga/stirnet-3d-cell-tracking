from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import LossConfig, QueryConfig
from .heads import render_native_masks
from .matcher import HungarianMatcher3D, MatchResult
from .types import StirNetOutput


def binary_focal_loss_with_logits(logits: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1-p) * (1-targets)
    alpha_t = alpha * targets + (1-alpha) * (1-targets)
    loss = alpha_t * (1-pt).pow(gamma) * ce
    if reduction == "sum": return loss.sum()
    if reduction == "none": return loss
    return loss.mean()


def dice_loss(logits: Tensor, targets: Tensor, eps: float = 1e-6) -> Tensor:
    p = logits.sigmoid().flatten(1)
    t = targets.float().flatten(1)
    score = (2*(p*t).sum(-1)+eps)/(p.sum(-1)+t.sum(-1)+eps)
    return (1-score).mean() if score.numel() else logits.sum()*0


def _matched_native_targets(targets: list[dict], matches: list[MatchResult], device) -> list[Tensor]:
    out=[]
    for b,m in enumerate(matches):
        out.append(targets[b]["masks"].to(device)[m.target_indices])
    return out


class RefinementCriterion(nn.Module):
    def __init__(self, loss_cfg: LossConfig, query_cfg: QueryConfig):
        super().__init__()
        self.cfg = loss_cfg
        self.query_cfg = query_cfg
        self.matcher = HungarianMatcher3D()

    def _existence_loss(self, logits: Tensor, padding: Tensor, matches: list[MatchResult]) -> Tensor:
        target = torch.zeros_like(logits)
        valid = ~padding
        for b,m in enumerate(matches):
            target[b,m.pred_indices] = 1
        loss = binary_focal_loss_with_logits(logits[valid], target[valid],
                                             alpha=self.cfg.exist_focal_alpha_pos,
                                             gamma=self.cfg.exist_focal_gamma)
        return loss

    def _coarse_losses(self, out: dict[str,Tensor], matches: list[MatchResult], targets: list[dict]):
        masks_p=[]; masks_t=[]; centers_p=[]; centers_t=[]
        for b,m in enumerate(matches):
            if m.pred_indices.numel()==0: continue
            pm=out["coarse_mask_logits"][b,m.pred_indices]
            gt=targets[b]["masks"].float().to(pm.device)[m.target_indices]
            gt=F.interpolate(gt[:,None], size=pm.shape[-3:], mode="nearest").squeeze(1)
            masks_p.append(pm); masks_t.append(gt)
            centers_p.append(out["centers_cellscale"][b,m.pred_indices])
            centers_t.append(targets[b]["centers_cellscale"].to(pm.device)[m.target_indices])
        zero=out["exist_logits"].sum()*0
        if not masks_p: return zero,zero,zero
        p=torch.cat(masks_p); t=torch.cat(masks_t)
        cp=torch.cat(centers_p); ct=torch.cat(centers_t)
        return dice_loss(p,t), binary_focal_loss_with_logits(p,t), F.smooth_l1_loss(cp,ct)

    def _count_loss(self, logits: Tensor, padding: Tensor, targets: list[dict]) -> Tensor:
        p=torch.sigmoid(logits).masked_fill(padding,0)
        pred=p.sum(dim=-1)
        gt=torch.tensor([len(t["masks"]) for t in targets],device=logits.device,dtype=logits.dtype)
        return (F.smooth_l1_loss(pred,gt,reduction="none")/gt.clamp_min(1)).mean()

    def _overlap_loss(self, coarse: Tensor, matches: list[MatchResult]) -> Tensor:
        vals=[]
        for b,m in enumerate(matches):
            if m.pred_indices.numel()<2: continue
            s=coarse[b,m.pred_indices].sigmoid().sum(dim=0)
            vals.append(F.relu(s-1).pow(2).mean())
        return torch.stack(vals).mean() if vals else coarse.sum()*0

    def _dense_losses(self, dense: dict[str,Tensor], targets: list[dict]):
        B=dense["foreground_logits"].shape[0]
        fg=[]; ch=[]; bd=[]
        for b,t in enumerate(targets):
            if "foreground" in t:
                fg_t=t["foreground"].to(dense["foreground_logits"].device).float()
            else:
                fg_t=t["masks"].to(dense["foreground_logits"].device).any(dim=0).float()
            fg.append(fg_t)
            if "center_heatmap" in t: ch.append(t["center_heatmap"].to(fg_t.device).float())
            if "boundary" in t: bd.append(t["boundary"].to(fg_t.device).float())
        fg_t=torch.stack(fg)[:,None]
        fg_loss=F.binary_cross_entropy_with_logits(dense["foreground_logits"],fg_t)+dice_loss(dense["foreground_logits"],fg_t)
        zero=fg_loss*0
        center_loss=zero
        if len(ch)==B:
            center_loss=binary_focal_loss_with_logits(dense["center_heatmap_logits"],torch.stack(ch)[:,None],alpha=0.25,gamma=2)
        boundary_loss=zero
        if len(bd)==B:
            bt=torch.stack(bd)[:,None]
            pos_weight=torch.tensor(self.cfg.boundary_pos_weight,device=bt.device,dtype=bt.dtype)
            boundary_loss=F.binary_cross_entropy_with_logits(dense["boundary_logits"],bt,pos_weight=pos_weight)+dice_loss(dense["boundary_logits"],bt)
        return fg_loss,center_loss,boundary_loss

    def forward(self, outputs: StirNetOutput, targets: list[dict]) -> dict[str,Tensor]:
        final_dict={
            "exist_logits":outputs.exist_logits,
            "centers_cellscale":outputs.centers_cellscale,
            "coarse_mask_logits":outputs.coarse_mask_logits,
        }
        matches=self.matcher(final_dict,outputs.query_padding_mask,targets)
        l_exist=self._existence_loss(outputs.exist_logits,outputs.query_padding_mask,matches)
        l_cdice,l_cfocal,l_center=self._coarse_losses(final_dict,matches,targets)
        l_count=self._count_loss(outputs.exist_logits,outputs.query_padding_mask,targets)
        l_overlap=self._overlap_loss(outputs.coarse_mask_logits,matches)

        selected=[m.pred_indices for m in matches]
        native_pred=render_native_masks(
            outputs.mask_features,outputs.native_mask_embeddings,selected,
            outputs.query_types,outputs.source_instance_ids,outputs.centers_cellscale,
            outputs.instance_labels,outputs.spacing_um,outputs.dref_um,
            prior_inside_logit=self.query_cfg.prior_inside_logit,
            prior_outside_logit=self.query_cfg.prior_outside_logit,
            temporal_sigma_dref=self.query_cfg.temporal_gaussian_sigma_dref,
        )
        hi_p=[];hi_t=[]
        for b,m in enumerate(matches):
            if m.pred_indices.numel():
                hi_p.append(native_pred[b])
                hi_t.append(targets[b]["masks"].to(outputs.mask_features.device).float()[m.target_indices])
        zero=outputs.exist_logits.sum()*0
        if hi_p:
            hp=torch.cat(hi_p); ht=torch.cat(hi_t)
            l_hdice=dice_loss(hp,ht)
            l_hfocal=binary_focal_loss_with_logits(hp,ht)
        else:
            l_hdice=l_hfocal=zero

        l_fg,l_heat,l_boundary=self._dense_losses(outputs.dense_outputs,targets)
        total=(self.cfg.exist*l_exist+self.cfg.dice_hi*l_hdice+self.cfg.focal_hi*l_hfocal+
               self.cfg.dice_coarse*l_cdice+self.cfg.focal_coarse*l_cfocal+
               self.cfg.center*l_center+self.cfg.count*l_count+self.cfg.overlap*l_overlap+
               self.cfg.foreground*l_fg+self.cfg.center_heatmap*l_heat+self.cfg.boundary*l_boundary)

        aux_total=zero
        for aux in outputs.aux_outputs:
            am=self.matcher(aux,outputs.query_padding_mask,targets)
            ae=self._existence_loss(aux["exist_logits"],outputs.query_padding_mask,am)
            ad,af,ac=self._coarse_losses(aux,am,targets)
            aux_total=aux_total+self.cfg.aux_layer*(self.cfg.exist*ae+self.cfg.dice_coarse*ad+self.cfg.focal_coarse*af+self.cfg.center*ac)
        total=total+aux_total
        return {
            "loss":total,
            "exist":l_exist,"dice_hi":l_hdice,"focal_hi":l_hfocal,
            "dice_coarse":l_cdice,"focal_coarse":l_cfocal,"center":l_center,
            "count":l_count,"overlap":l_overlap,"foreground":l_fg,
            "center_heatmap":l_heat,"boundary":l_boundary,"aux":aux_total,
        }
