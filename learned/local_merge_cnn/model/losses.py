"""Multi-task losses for the canonical-coordinate vector instance CNN."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn
from torch.nn import functional as F

from .vector_cnn import VectorCNNOutput


@dataclass(frozen=True)
class VectorCNNTargets:
    """Aligned training targets.

    ``vectors_normalized`` are canonical axis-fraction displacements. They are
    independent of the original biological scale of the source component.
    """

    foreground: torch.Tensor
    vectors_normalized: torch.Tensor
    boundary: torch.Tensor
    center: torch.Tensor
    valid_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class VectorCNNLossWeights:
    foreground: float = 0.5
    vector: float = 2.0
    direction: float = 0.5
    boundary: float = 1.0
    center: float = 1.0

    def __post_init__(self) -> None:
        for name in ("foreground", "vector", "direction", "boundary", "center"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} loss weight must be finite and nonnegative")


@dataclass(frozen=True)
class VectorCNNLossBreakdown:
    total: torch.Tensor
    foreground: torch.Tensor
    vector: torch.Tensor
    direction: torch.Tensor
    boundary: torch.Tensor
    center: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "total": self.total,
            "foreground": self.foreground,
            "vector": self.vector,
            "direction": self.direction,
            "boundary": self.boundary,
            "center": self.center,
        }


def _as_single_channel_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.ndim == reference.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.ndim != reference.ndim:
        raise ValueError("mask dimensionality does not match reference")
    if mask.shape[0] != reference.shape[0] or mask.shape[-3:] != reference.shape[-3:]:
        raise ValueError("mask batch/spatial dimensions must match reference")
    if mask.shape[1] != 1:
        raise ValueError("mask must have exactly one channel")
    return mask.to(dtype=reference.dtype, device=reference.device)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def soft_dice_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError("logits and target must have identical shapes")
    probability = torch.sigmoid(logits)
    target = target.to(dtype=probability.dtype, device=probability.device)
    mask = torch.ones_like(probability)
    if valid_mask is not None:
        mask = _as_single_channel_mask(valid_mask, probability)
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target * mask).sum(dim=dims)
    prediction_mass = (probability * mask).sum(dim=dims)
    target_mass = (target * mask).sum(dim=dims)
    dice = (2 * intersection + eps) / (prediction_mass + target_mass + eps)
    return 1.0 - dice.mean()


def focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError("logits and target must have identical shapes")
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    target = target.to(dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * target + (1 - probability) * (1 - target)
    focal = (1 - p_t).pow(gamma) * bce
    if alpha is not None:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0,1]")
        focal *= alpha * target + (1 - alpha) * (1 - target)
    mask = torch.ones_like(focal)
    if valid_mask is not None:
        mask = _as_single_channel_mask(valid_mask, focal)
    return _masked_mean(focal, mask)


class VectorCNNLoss(nn.Module):
    def __init__(
        self,
        *,
        weights: VectorCNNLossWeights | None = None,
        direction_min_target_norm: float = 0.02,
        smooth_l1_beta: float = 0.05,
        boundary_focal_gamma: float = 2.0,
        center_focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.weights = weights or VectorCNNLossWeights()
        if direction_min_target_norm < 0:
            raise ValueError("direction_min_target_norm cannot be negative")
        if smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        self.direction_min_target_norm = float(direction_min_target_norm)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.boundary_focal_gamma = float(boundary_focal_gamma)
        self.center_focal_gamma = float(center_focal_gamma)

    @staticmethod
    def _validate_shapes(output: VectorCNNOutput, target: VectorCNNTargets) -> None:
        scalar_shapes = (
            output.foreground_logits.shape,
            output.boundary_logits.shape,
            output.center_logits.shape,
            target.foreground.shape,
            target.boundary.shape,
            target.center.shape,
        )
        if len(set(scalar_shapes)) != 1:
            raise ValueError("all scalar outputs/targets must have identical shapes")
        if output.vectors_normalized.shape != target.vectors_normalized.shape:
            raise ValueError("vector output/target shapes must match")
        if output.vectors_normalized.shape[1] != 3:
            raise ValueError("vector tensors must contain three channels")

    def _foreground_loss(self, logits, target, valid_mask):
        target = target.to(dtype=logits.dtype, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        mask = torch.ones_like(bce)
        if valid_mask is not None:
            mask = _as_single_channel_mask(valid_mask, bce)
        return _masked_mean(bce, mask) + soft_dice_loss_from_logits(
            logits, target, valid_mask=valid_mask
        )

    def _vector_losses(self, prediction, target_vectors, foreground, valid_mask):
        target_vectors = target_vectors.to(dtype=prediction.dtype, device=prediction.device)
        foreground_mask = (_as_single_channel_mask(foreground, prediction[:, :1]) > 0.5).to(prediction.dtype)
        if valid_mask is not None:
            valid = (_as_single_channel_mask(valid_mask, prediction[:, :1]) > 0.5).to(prediction.dtype)
            foreground_mask *= valid
        component_mask = foreground_mask.expand_as(prediction)
        per_component = F.smooth_l1_loss(
            prediction, target_vectors, reduction="none", beta=self.smooth_l1_beta
        )
        vector_loss = _masked_mean(per_component, component_mask)
        target_norm = torch.linalg.vector_norm(target_vectors, dim=1, keepdim=True)
        direction_mask = foreground_mask * (target_norm > self.direction_min_target_norm).to(prediction.dtype)
        cosine = F.cosine_similarity(prediction, target_vectors, dim=1, eps=1e-6).unsqueeze(1)
        direction_loss = _masked_mean(1 - cosine, direction_mask)
        return vector_loss, direction_loss

    def forward(self, output: VectorCNNOutput, target: VectorCNNTargets) -> VectorCNNLossBreakdown:
        self._validate_shapes(output, target)
        foreground = self._foreground_loss(output.foreground_logits, target.foreground, target.valid_mask)
        vector, direction = self._vector_losses(
            output.vectors_normalized, target.vectors_normalized, target.foreground, target.valid_mask
        )
        boundary = focal_bce_with_logits(
            output.boundary_logits, target.boundary,
            valid_mask=target.valid_mask, gamma=self.boundary_focal_gamma
        ) + soft_dice_loss_from_logits(
            output.boundary_logits, target.boundary, valid_mask=target.valid_mask
        )
        center = focal_bce_with_logits(
            output.center_logits, target.center,
            valid_mask=target.valid_mask, gamma=self.center_focal_gamma
        )
        total = (
            self.weights.foreground * foreground
            + self.weights.vector * vector
            + self.weights.direction * direction
            + self.weights.boundary * boundary
            + self.weights.center * center
        )
        return VectorCNNLossBreakdown(total, foreground, vector, direction, boundary, center)
