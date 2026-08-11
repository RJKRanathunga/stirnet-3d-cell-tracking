from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ...model.matcher import HungarianMatcher3D, MatchResult, target_ids, target_masks_at_shape


@dataclass
class MatchingProbeResult:
    matches: list[MatchResult]
    rows: list[dict[str, Any]]
    query_to_target: dict[tuple[int, int], int]


@torch.no_grad()
def run_matching_probe(outputs, targets: list[dict]) -> MatchingProbeResult:
    """Run the same final-layer Hungarian matcher used by the criterion."""

    matcher = HungarianMatcher3D()
    output_dict = {
        "exist_logits": outputs.exist_logits,
        "coarse_mask_logits": outputs.coarse_mask_logits,
        "centers_cellscale": outputs.centers_cellscale,
    }
    matches = matcher(output_dict, outputs.query_padding_mask, targets)

    rows: list[dict[str, Any]] = []
    mapping: dict[tuple[int, int], int] = {}

    for b, match in enumerate(matches):
        ids = target_ids(targets[b]).detach().cpu().long()
        gt_centers = torch.as_tensor(targets[b]["centers_cellscale"], dtype=torch.float32)
        dref = float(outputs.dref_um[b].detach().cpu())
        exist = torch.sigmoid(outputs.exist_logits[b]).detach().float().cpu()
        centers = outputs.centers_cellscale[b].detach().float().cpu()

        for pred_idx, target_idx in zip(
            match.pred_indices.detach().cpu().tolist(),
            match.target_indices.detach().cpu().tolist(),
        ):
            pred_idx = int(pred_idx)
            target_idx = int(target_idx)
            mapping[(b, pred_idx)] = target_idx
            delta_um = (centers[pred_idx] - gt_centers[target_idx]).numpy() * dref
            rows.append(
                {
                    "batch": b,
                    "query": pred_idx,
                    "target_index": target_idx,
                    "gt_id": int(ids[target_idx]),
                    "exist_prob": float(exist[pred_idx]),
                    "center_error_um": float(np.linalg.norm(delta_um)),
                }
            )

    return MatchingProbeResult(matches=matches, rows=rows, query_to_target=mapping)


@torch.no_grad()
def coarse_dice_for_matches(logits, target: dict, match: MatchResult) -> dict[int, float]:
    if match.pred_indices.numel() == 0:
        return {}

    pred_indices = match.pred_indices.to(logits.device)
    target_indices = match.target_indices.to(logits.device)
    target_masks = target_masks_at_shape(
        target,
        tuple(int(v) for v in logits.shape[-3:]),
        logits.device,
        target_indices=target_indices,
    ).float()
    pred = logits[pred_indices].float().sigmoid().flatten(1)
    gt = target_masks.flatten(1)
    dice = (2.0 * (pred * gt).sum(-1) + 1e-6) / (
        pred.sum(-1) + gt.sum(-1) + 1e-6
    )
    return {
        int(q): float(v)
        for q, v in zip(
            match.pred_indices.detach().cpu().tolist(),
            dice.detach().cpu().tolist(),
        )
    }
