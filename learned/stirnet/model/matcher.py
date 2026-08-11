from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .coordinates import resize_label_map_nearest


def target_ids(target: dict) -> Tensor:
    if "ids" in target:
        return torch.as_tensor(target["ids"], dtype=torch.long)
    if "masks" in target:
        masks = torch.as_tensor(target["masks"])
        return torch.arange(len(masks), dtype=torch.long, device=masks.device)
    raise KeyError("A STIR-Net target requires 'ids' with 'label_map', or legacy 'masks'")


def valid_target_indices(target: dict) -> Tensor:
    ids = target_ids(target)
    valid = target.get("valid", target.get("gt_valid"))
    if valid is None:
        return torch.arange(len(ids), dtype=torch.long, device=ids.device)
    return torch.nonzero(torch.as_tensor(valid, dtype=torch.bool), as_tuple=False).flatten()


def target_masks_at_shape(
    target: dict,
    spatial_shape: tuple[int, int, int],
    device: torch.device,
    *,
    target_indices: Tensor | None = None,
) -> Tensor:
    """Materialize only coarse target masks, preferably from one label map."""
    if target_indices is None:
        target_indices = valid_target_indices(target)
    target_indices_cpu = target_indices.detach().cpu().long()
    if "label_map" in target:
        labels = torch.as_tensor(target["label_map"])
        labels = resize_label_map_nearest(labels, spatial_shape)
        all_ids = target_ids(target)
        ids = all_ids[target_indices_cpu.to(all_ids.device)].to(labels.device)
        masks = labels.unsqueeze(0) == ids[:, None, None, None]
    elif "masks" in target:
        masks = torch.as_tensor(target["masks"])
        masks = masks[target_indices_cpu.to(masks.device)]
        masks = resize_label_map_nearest(masks, spatial_shape)
    else:
        raise KeyError("Target needs either 'label_map' plus 'ids', or legacy 'masks'")
    return masks.to(device=device, dtype=torch.float32, non_blocking=True)


def _pairwise_dice_cost(pred_logits: Tensor, gt: Tensor, eps: float = 1e-6) -> Tensor:
    """FP32 pairwise Dice cost for pred [Q,V] and gt [K,V]."""
    pred_logits = pred_logits.float()
    gt = gt.float()
    p = pred_logits.sigmoid()
    inter = 2 * torch.einsum("qv,kv->qk", p, gt)
    denom = p.sum(-1)[:, None] + gt.sum(-1)[None, :]
    return 1 - (inter + eps) / (denom + eps)


def _pairwise_focal_cost(
    pred_logits: Tensor,
    gt: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Stable FP32 focal matching cost using logit-space log probabilities."""
    logits = pred_logits.float()
    gt = gt.float()
    probability = logits.sigmoid()
    positive = -alpha * (1 - probability).pow(gamma) * F.logsigmoid(logits)
    negative = -(1 - alpha) * probability.pow(gamma) * F.logsigmoid(-logits)
    costs = []
    for k in range(gt.shape[0]):
        target = gt[k][None]
        costs.append((positive * target + negative * (1 - target)).mean(dim=-1))
    return torch.stack(costs, dim=1) if costs else logits.new_zeros((logits.shape[0], 0))


def build_cost_matrix(
    exist_logits: Tensor,
    coarse_mask_logits: Tensor,
    centers_cellscale: Tensor,
    gt_masks: Tensor,
    gt_centers_cellscale: Tensor,
    *,
    w_exist: float = 2.0,
    w_dice: float = 5.0,
    w_focal: float = 2.0,
    w_center: float = 2.0,
) -> Tensor:
    """Construct the complete Hungarian cost explicitly in FP32."""
    with torch.autocast(device_type=coarse_mask_logits.device.type, enabled=False):
        masks = coarse_mask_logits.float().flatten(1)
        gt = gt_masks.float().flatten(1)
        dice = _pairwise_dice_cost(masks, gt)
        focal = _pairwise_focal_cost(masks, gt)
        exist = -F.logsigmoid(exist_logits.float())[:, None].expand(-1, gt.shape[0])
        center = torch.cdist(
            centers_cellscale.float(), gt_centers_cellscale.float(), p=1
        )
        cost = w_exist * exist + w_dice * dice + w_focal * focal + w_center * center
    if not torch.isfinite(cost).all():
        bad = ~torch.isfinite(cost)
        raise ValueError(
            "Hungarian cost contains non-finite entries after FP32 construction: "
            f"shape={tuple(cost.shape)}, nonfinite={int(bad.sum())}, "
            f"exist_finite={bool(torch.isfinite(exist).all())}, "
            f"dice_finite={bool(torch.isfinite(dice).all())}, "
            f"focal_finite={bool(torch.isfinite(focal).all())}, "
            f"center_finite={bool(torch.isfinite(center).all())}"
        )
    return cost


@dataclass
class MatchResult:
    pred_indices: Tensor
    target_indices: Tensor


class HungarianMatcher3D(nn.Module):
    def __init__(
        self,
        w_exist: float = 2.0,
        w_dice: float = 5.0,
        w_focal: float = 2.0,
        w_center: float = 2.0,
    ):
        super().__init__()
        self.w_exist = w_exist
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_center = w_center

    @torch.no_grad()
    def forward(
        self,
        output: dict[str, Tensor],
        query_padding_mask: Tensor,
        targets: list[dict],
        coarse_target_masks: list[Tensor] | None = None,
    ) -> list[MatchResult]:
        batch_size, _ = output["exist_logits"].shape
        results = []
        coarse = output["coarse_mask_logits"]
        for b in range(batch_size):
            valid_q = torch.nonzero(~query_padding_mask[b], as_tuple=False).flatten()
            valid_gt = valid_target_indices(targets[b])
            if valid_gt.numel() == 0 or valid_q.numel() == 0:
                results.append(
                    MatchResult(
                        valid_q[:0],
                        torch.empty(0, device=coarse.device, dtype=torch.long),
                    )
                )
                continue
            gt_masks = (
                coarse_target_masks[b]
                if coarse_target_masks is not None
                else target_masks_at_shape(
                    targets[b], coarse.shape[-3:], coarse.device, target_indices=valid_gt
                )
            )
            gt_centers = torch.as_tensor(targets[b]["centers_cellscale"])[valid_gt]
            gt_centers = gt_centers.to(coarse.device, dtype=torch.float32, non_blocking=True)
            cost = build_cost_matrix(
                output["exist_logits"][b, valid_q],
                coarse[b, valid_q],
                output["centers_cellscale"][b, valid_q],
                gt_masks,
                gt_centers,
                w_exist=self.w_exist,
                w_dice=self.w_dice,
                w_focal=self.w_focal,
                w_center=self.w_center,
            )
            row, col = linear_sum_assignment(cost.cpu().numpy())
            row_t = torch.as_tensor(row, device=coarse.device, dtype=torch.long)
            col_t = torch.as_tensor(col, device=valid_gt.device, dtype=torch.long)
            results.append(
                MatchResult(
                    valid_q[row_t],
                    valid_gt[col_t].to(coarse.device),
                )
            )
        return results
