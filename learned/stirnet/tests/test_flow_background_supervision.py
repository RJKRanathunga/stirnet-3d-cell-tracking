from __future__ import annotations

import math
import torch

from learned.stirnet.model.config import GeometryConfig
from learned.stirnet.model.geometry.losses import GeometryCriterion, weighted_bce
from learned.stirnet.model.geometry.targets import GeometryTargets
from learned.stirnet.model.types import GeometryState


def _state(flow: torch.Tensor) -> GeometryState:
    shape = flow.shape[-3:]
    scalar = torch.zeros((1, 1, *shape), dtype=torch.float32)
    vector = torch.zeros((1, 3, *shape), dtype=torch.float32)
    return GeometryState(
        foreground_logits=scalar.clone(),
        surface_logits=scalar.clone(),
        separator_logits=scalar.clone(),
        sdf=scalar.clone(),
        flow=flow,
        centroid_offset=vector,
        seed_logits=scalar.clone(),
        features=None,
    )


def _targets() -> GeometryTargets:
    shape = (5, 5, 5)
    fg = torch.zeros((1, 1, *shape), dtype=torch.float32)
    fg[:, :, 2, 2, 2] = 1.0
    surface = torch.zeros_like(fg)
    surface[:, :, 2, 2, 3] = 0.5  # near-background voxel
    scalar = torch.zeros_like(fg)
    vector = torch.zeros((1, 3, *shape), dtype=torch.float32)
    return GeometryTargets(
        foreground=fg,
        surface=surface,
        separator=scalar.clone(),
        sdf=scalar.clone(),
        sdf_valid=torch.ones_like(fg),
        flow=vector.clone(),
        centroid_offset=vector.clone(),
        seed=scalar.clone(),
    )


def test_flow_background_penalizes_near_exterior_leakage() -> None:
    criterion = GeometryCriterion(
        GeometryConfig(flow_background_weight=1.0, flow_background_surface_threshold=0.05)
    )
    target = _targets()
    leaked = torch.zeros((1, 3, 5, 5, 5), dtype=torch.float32)
    leaked[:, :, 2, 2, 3] = 0.5
    losses = criterion(
        _state(leaked), target,
        torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([1.0]),
    )
    assert float(losses["flow_background"]) > 0.0

    zero_losses = criterion(
        _state(torch.zeros_like(leaked)), target,
        torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([1.0]),
    )
    assert math.isclose(float(zero_losses["flow_background"]), 0.0, abs_tol=1e-8)


def test_weighted_bce_preserves_soft_target_optimum() -> None:
    target = torch.tensor([[[[[0.2]]]]], dtype=torch.float32)
    logits = torch.logit(target.clone()).requires_grad_(True)
    weighted_bce(logits, target, pos_weight=6.0).backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().max()) < 1e-6
