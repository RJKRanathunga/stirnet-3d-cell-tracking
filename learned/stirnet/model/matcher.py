from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _pairwise_dice_cost(pred_logits: Tensor, gt: Tensor, eps: float = 1e-6) -> Tensor:
    # pred [Q,V], gt [K,V]
    p = pred_logits.sigmoid()
    inter = 2 * torch.einsum("qv,kv->qk", p, gt)
    denom = p.sum(-1)[:,None] + gt.sum(-1)[None,:]
    return 1 - (inter + eps) / (denom + eps)


def _pairwise_focal_cost(pred_logits: Tensor, gt: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    p = pred_logits.sigmoid().clamp(1e-6, 1-1e-6)
    pos = -alpha * ((1-p)**gamma) * torch.log(p)
    neg = -(1-alpha) * (p**gamma) * torch.log(1-p)
    # average cost against each GT without materializing QxKxV at once
    costs = []
    for k in range(gt.shape[0]):
        g = gt[k][None]
        costs.append((pos * g + neg * (1-g)).mean(dim=-1))
    return torch.stack(costs, dim=1) if costs else pred_logits.new_zeros((pred_logits.shape[0],0))


@dataclass
class MatchResult:
    pred_indices: Tensor
    target_indices: Tensor


class HungarianMatcher3D(nn.Module):
    def __init__(self, w_exist: float = 2.0, w_dice: float = 5.0, w_focal: float = 2.0, w_center: float = 2.0):
        super().__init__()
        self.w_exist = w_exist
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_center = w_center

    @torch.no_grad()
    def forward(self, output: dict[str,Tensor], query_padding_mask: Tensor, targets: list[dict]) -> list[MatchResult]:
        B,Q = output["exist_logits"].shape
        results = []
        coarse = output["coarse_mask_logits"]
        for b in range(B):
            valid_q = torch.nonzero(~query_padding_mask[b], as_tuple=False).flatten()
            gt_masks = targets[b]["masks"].float().to(coarse.device)
            gt_centers = targets[b]["centers_cellscale"].to(coarse.device)
            if gt_masks.shape[0] == 0 or valid_q.numel() == 0:
                results.append(MatchResult(valid_q[:0], torch.empty(0,device=coarse.device,dtype=torch.long)))
                continue
            gm = F.interpolate(gt_masks[:,None], size=coarse.shape[-3:], mode="nearest").squeeze(1)
            pl = coarse[b, valid_q].flatten(1)
            gf = gm.flatten(1)
            dice = _pairwise_dice_cost(pl, gf)
            focal = _pairwise_focal_cost(pl, gf)
            exist = -F.logsigmoid(output["exist_logits"][b, valid_q])[:,None].expand(-1, gt_masks.shape[0])
            center = torch.cdist(output["centers_cellscale"][b, valid_q], gt_centers, p=1)
            cost = self.w_exist*exist + self.w_dice*dice + self.w_focal*focal + self.w_center*center
            row,col = linear_sum_assignment(cost.detach().cpu().numpy())
            results.append(MatchResult(valid_q[torch.as_tensor(row,device=coarse.device)], torch.as_tensor(col,device=coarse.device,dtype=torch.long)))
        return results
