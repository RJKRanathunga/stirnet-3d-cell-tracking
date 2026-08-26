"""Multi-task training losses for sparse reconciliation errors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from ..contracts import ReconciliationOutput


@dataclass(frozen=True)
class LossWeights:
    continuation: float = 1.0
    division: float = 1.5
    division_prior: float = 0.5
    appearance: float = 0.35
    termination: float = 0.35
    fingerprint_metric: float = 0.15


def focal_binary_probability_loss(
    probability: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    gamma: float = 2.0,
    positive_weight: float = 1.0,
) -> Tensor:
    p = probability.clamp(1e-6, 1.0 - 1e-6)
    y = target.to(p.dtype)
    ce = -(positive_weight * y * torch.log(p) + (1.0 - y) * torch.log1p(-p))
    pt = torch.where(y > 0.5, p, 1.0 - p)
    loss = ((1.0 - pt) ** gamma) * ce
    selected = loss[mask]
    return selected.mean() if selected.numel() else loss.sum() * 0.0


def masked_bce(logits: Tensor, target: Tensor, mask: Tensor, pos_weight: float | None = None) -> Tensor:
    weight = None
    if pos_weight is not None:
        weight = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), reduction="none", pos_weight=weight)
    selected = loss[mask]
    return selected.mean() if selected.numel() else loss.sum() * 0.0


def fingerprint_pair_loss(
    first: Tensor,
    second: Tensor,
    same_identity: Tensor,
    *,
    margin: float = 0.35,
) -> Tensor:
    """Cosine metric loss usable with abundant high-confidence tracklet pairs."""

    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    similarity = (first * second).sum(dim=-1)
    positive = (1.0 - similarity) * same_identity.to(similarity.dtype)
    negative = F.relu(similarity - margin) * (~same_identity).to(similarity.dtype)
    return (positive + negative).mean()


def reconciliation_loss(
    output: ReconciliationOutput,
    *,
    edge_target: Tensor,
    edge_mask: Tensor,
    tracklet_mask: Tensor,
    division_target: Tensor | None = None,
    division_mask: Tensor | None = None,
    division_prior_target: Tensor | None = None,
    appearance_target: Tensor | None = None,
    termination_target: Tensor | None = None,
    weights: LossWeights | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Primary event loss; trivial links should be controlled by the sampler."""

    w = weights or LossWeights()
    pieces: dict[str, Tensor] = {}
    pieces["continuation"] = focal_binary_probability_loss(
        output.parental_probabilities,
        edge_target,
        edge_mask,
        gamma=2.0,
    )
    if output.division_logits is not None and division_target is not None and division_mask is not None:
        pieces["division"] = masked_bce(output.division_logits, division_target, division_mask, pos_weight=2.0)
    else:
        pieces["division"] = output.continuation_logits.sum() * 0.0
    if division_prior_target is not None:
        pieces["division_prior"] = masked_bce(
            output.division_prior_logits, division_prior_target, tracklet_mask, pos_weight=2.0
        )
    else:
        pieces["division_prior"] = output.division_prior_logits.sum() * 0.0
    pieces["appearance"] = (
        masked_bce(output.appearance_logits, appearance_target, tracklet_mask)
        if appearance_target is not None
        else output.appearance_logits.sum() * 0.0
    )
    pieces["termination"] = (
        masked_bce(output.termination_logits, termination_target, tracklet_mask)
        if termination_target is not None
        else output.termination_logits.sum() * 0.0
    )
    total = (
        w.continuation * pieces["continuation"]
        + w.division * pieces["division"]
        + w.division_prior * pieces["division_prior"]
        + w.appearance * pieces["appearance"]
        + w.termination * pieces["termination"]
    )
    return total, pieces
