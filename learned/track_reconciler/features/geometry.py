"""Differentiable geometric primitives in physical z,y,x coordinates."""

from __future__ import annotations

import torch
from torch import Tensor


def point_segment_distance(points: Tensor, start: Tensor, end: Tensor, eps: float = 1e-8) -> Tensor:
    """Euclidean distance from points to finite line segments.

    Inputs broadcast over all leading dimensions; final dimension must be 3.
    """

    direction = end - start
    denom = (direction * direction).sum(dim=-1, keepdim=True).clamp_min(eps)
    t = ((points - start) * direction).sum(dim=-1, keepdim=True) / denom
    t = t.clamp(0.0, 1.0)
    closest = start + t * direction
    return torch.linalg.vector_norm(points - closest, dim=-1)


def segment_segment_distance(
    p0: Tensor,
    p1: Tensor,
    q0: Tensor,
    q1: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Minimum Euclidean distance between two finite 3-D segments.

    This is the clamped analytical closest-point construction used in classic
    segment-distance algorithms.  Inputs broadcast over leading dimensions.
    Geometry is normally detached before attention because coordinates are
    fixed observations rather than trainable quantities.
    """

    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = (u * u).sum(-1)
    b = (u * v).sum(-1)
    c = (v * v).sum(-1)
    d = (u * w).sum(-1)
    e = (v * w).sum(-1)
    det = a * c - b * b

    # Initial unconstrained closest points.
    parallel = det.abs() < eps
    s_num = b * e - c * d
    t_num = a * e - b * d
    s_den = det.clone()
    t_den = det.clone()

    s_num = torch.where(parallel, torch.zeros_like(s_num), s_num)
    s_den = torch.where(parallel, torch.ones_like(s_den), s_den)
    t_num = torch.where(parallel, e, t_num)
    t_den = torch.where(parallel, c.clamp_min(eps), t_den)

    # Clamp s to [0, 1], updating t for the selected boundary.
    low_s = (~parallel) & (s_num < 0)
    high_s = (~parallel) & (s_num > s_den)
    s_num = torch.where(low_s, torch.zeros_like(s_num), s_num)
    t_num = torch.where(low_s, e, t_num)
    t_den = torch.where(low_s, c.clamp_min(eps), t_den)
    s_num = torch.where(high_s, s_den, s_num)
    t_num = torch.where(high_s, e + b, t_num)
    t_den = torch.where(high_s, c.clamp_min(eps), t_den)

    # Clamp t to [0, 1], then re-solve s on that boundary.
    low_t = t_num < 0
    high_t = t_num > t_den

    minus_d = -d
    s_num_low_t = minus_d.clamp(min=0.0)
    s_num_low_t = torch.minimum(s_num_low_t, a)
    s_den_low_t = a.clamp_min(eps)
    s_num = torch.where(low_t, s_num_low_t, s_num)
    s_den = torch.where(low_t, s_den_low_t, s_den)
    t_num = torch.where(low_t, torch.zeros_like(t_num), t_num)
    t_den = torch.where(low_t, torch.ones_like(t_den), t_den)

    minus_d_plus_b = -d + b
    s_num_high_t = minus_d_plus_b.clamp(min=0.0)
    s_num_high_t = torch.minimum(s_num_high_t, a)
    s_den_high_t = a.clamp_min(eps)
    s_num = torch.where(high_t, s_num_high_t, s_num)
    s_den = torch.where(high_t, s_den_high_t, s_den)
    t_num = torch.where(high_t, torch.ones_like(t_num), t_num)
    t_den = torch.where(high_t, torch.ones_like(t_den), t_den)

    s = torch.where(s_num.abs() < eps, torch.zeros_like(s_num), s_num / s_den.clamp_min(eps))
    t = torch.where(t_num.abs() < eps, torch.zeros_like(t_num), t_num / t_den.clamp_min(eps))
    delta = w + s.unsqueeze(-1) * u - t.unsqueeze(-1) * v
    return torch.linalg.vector_norm(delta, dim=-1)


def pairwise_segment_distance(start: Tensor, end: Tensor) -> Tensor:
    """All-pairs finite-segment distance for `[B, E, 3]` segments."""

    if start.ndim != 3 or start.shape != end.shape or start.shape[-1] != 3:
        raise ValueError("start and end must both be [B, E, 3]")
    p0 = start[:, :, None, :]
    p1 = end[:, :, None, :]
    q0 = start[:, None, :, :]
    q1 = end[:, None, :, :]
    distance = segment_segment_distance(p0, p1, q0, q1)
    # Numerical symmetry is useful for deterministic attention/tests.
    return 0.5 * (distance + distance.transpose(1, 2))
