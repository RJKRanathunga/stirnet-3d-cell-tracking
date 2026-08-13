from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def target_to_history_grid(
    target_points_um: Tensor,
    historical_center_um: Tensor,
    projected_center_um: Tensor,
    extent_um: Tensor,
) -> Tensor:
    """Map target-frame zyx points to normalized canonical support coordinates.

    Projection is translational: the historical center is shifted onto the
    immutable target-frame temporal reference. Returned coordinates remain zyx
    in ``[-1,1]``; :func:`sample_projected_history_support` performs the xyz
    reorder required by ``grid_sample``.
    """

    translation = projected_center_um - historical_center_um
    historical_points = target_points_um - translation[..., None, :]
    local = historical_points - historical_center_um[..., None, :]
    return 2.0 * local / extent_um[..., None, None].clamp_min(1e-8)


def sample_projected_history_support(
    support: Tensor,
    valid: Tensor,
    historical_center_um: Tensor,
    projected_center_um: Tensor,
    extent_um: Tensor,
    target_points_um: Tensor,
) -> Tensor:
    """Sample compact past/future support without creating a dense target volume.

    Args:
        support: ``[Q,2,C,G,G,G]``.
        valid: ``[Q,2]``.
        target_points_um: ``[Q,K,3]`` or ``[K,3]`` target-frame zyx points.
    Returns:
        ``[Q,K,2,C]`` sampled features, exactly zero for invalid supports.
    """

    if target_points_um.ndim == 2:
        target_points_um = target_points_um[None].expand(support.shape[0], -1, -1)
    q, sides, channels = support.shape[:3]
    if sides != 2:
        raise ValueError("history_support side dimension must be past/future=2")
    values = []
    for side in range(2):
        normalized_zyx = target_to_history_grid(
            target_points_um,
            historical_center_um[:, side],
            projected_center_um,
            extent_um[:, side],
        )
        grid_xyz = normalized_zyx[..., [2, 1, 0]][:, None, None]
        sampled = F.grid_sample(
            support[:, side].float(),
            grid_xyz.float(),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled[:, :, 0, 0].transpose(1, 2)
        sampled = sampled * valid[:, side, None, None].to(sampled.dtype)
        values.append(sampled)
    return torch.stack(values, dim=2).to(support.dtype)


class HistorySupportBias(nn.Module):
    """Map sampled occupancy/SDF and validity metadata to per-head logit bias."""

    def __init__(self, heads: int, hidden_dim: int = 32, dt_normalizer: float = 2.0):
        super().__init__()
        self.dt_normalizer = float(dt_normalizer)
        self.net = nn.Sequential(nn.Linear(8, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, heads))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, samples: Tensor, valid: Tensor, dt: Tensor) -> Tensor:
        if samples.shape[-2:] != (2, 2):
            raise ValueError("history attention requires past/future occupancy+SDF")
        q, key_count = samples.shape[:2]
        validity = valid[:, None, :].expand(q, key_count, 2).to(samples.dtype)
        dt_norm = (
            dt.abs() / max(self.dt_normalizer, 1e-8)
        )[:, None, :].expand(q, key_count, 2).to(samples.dtype)
        features = torch.cat(
            [samples[:, :, 0], samples[:, :, 1], validity, dt_norm], dim=-1
        ).float()
        any_valid = valid.any(dim=-1)[:, None, None]
        return self.net(features) * any_valid.to(features.dtype)
