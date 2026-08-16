from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import GeometryConfig, SpatialConfig, TemporalConfig
from ..types import GeometryState, SpatialDecodeState, TemporalState


def _sample_local_grid(
    feature: Tensor,
    refs_um: Tensor,
    spacing_um: Tensor,
    radius_um: Tensor,
) -> Tensor:
    """Sample a 3x3x3 physical neighborhood for each reference; return [N,C]."""
    if refs_um.shape[0] == 0:
        return feature.new_zeros((0, feature.shape[0]))
    shape = feature.shape[-3:]
    extent = refs_um.new_tensor(
        [(shape[0] - 1), (shape[1] - 1), (shape[2] - 1)]
    ) * spacing_um.float()
    unit = torch.tensor([-1.0, 0.0, 1.0], device=refs_um.device)
    base = torch.stack(torch.meshgrid(unit, unit, unit, indexing="ij"), dim=-1).reshape(-1, 3)
    offsets = base[None] * radius_um.reshape(-1, 1, 1)
    points = refs_um[:, None] + offsets
    normalized_zyx = points / (0.5 * extent[None, None]).clamp_min(1e-6)
    grid_xyz = normalized_zyx[..., [2, 1, 0]]
    sampled = F.grid_sample(
        feature[None],
        grid_xyz.reshape(1, refs_um.shape[0], 27, 1, 3),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, :, :, :, 0].permute(1, 0, 2)
    return sampled.mean(dim=-1)


class TemporalSpatialObserver(nn.Module):
    """One-way spatial -> temporal observation.

    This intentionally replaces the old early bidirectional co-reasoning. The
    dense current-frame geometry is completed independently; temporal tokens may
    inspect it, but cannot globally write back into D0/D1/D2.
    """

    def __init__(
        self,
        temporal_cfg: TemporalConfig,
        spatial_cfg: SpatialConfig,
        geometry_cfg: GeometryConfig,
    ):
        super().__init__()
        d = temporal_cfg.d_model
        self.cfg = temporal_cfg
        self.d1_proj = nn.Conv3d(spatial_cfg.channels[1], d, 1, bias=False)
        self.d2_proj = nn.Conv3d(spatial_cfg.channels[2], d, 1, bias=False)
        self.geometry_proj = nn.Conv3d(geometry_cfg.hidden_channels, d, 1, bias=False)
        self.geometry_field_proj = nn.Conv3d(11, d, 1, bias=False)
        self.message = nn.Sequential(
            nn.Linear(3 * d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * d + 1, d), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        temporal: TemporalState,
        decoded: SpatialDecodeState,
        geometry: GeometryState,
        spatial_spacings_um: list[Tensor],
        spacing_um: Tensor,
        dref_um: Tensor,
    ) -> TemporalState:
        if temporal.is_empty:
            return temporal
        d1 = self.d1_proj(decoded.d1)
        d2 = self.d2_proj(decoded.d2)
        probabilities = geometry.probabilities()
        explicit_geometry = torch.cat(
            [
                probabilities["foreground"],
                probabilities["surface"],
                probabilities["separator"],
                geometry.sdf,
                geometry.flow,
                geometry.centroid_offset,
                probabilities["seed"],
            ],
            dim=1,
        ).to(geometry.features.dtype)
        geo = self.geometry_proj(geometry.features) + self.geometry_field_proj(
            explicit_geometry
        )
        messages = torch.zeros_like(temporal.tokens)
        for b in range(decoded.d0.shape[0]):
            idx = torch.nonzero(temporal.batch_index == b, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            refs = temporal.ref_um[idx]
            radius = dref_um[b] * self.cfg.observation_radius_dref
            radius_vec = radius.expand(len(idx))
            p1 = _sample_local_grid(
                d1[b], refs, spatial_spacings_um[1][b], radius_vec
            )
            p2 = _sample_local_grid(
                d2[b], refs, spatial_spacings_um[2][b], radius_vec
            )
            pg = _sample_local_grid(
                geo[b], refs, spacing_um[b], radius_vec
            )
            messages[idx] = self.message(torch.cat([p1, p2, pg], dim=-1))
        gate = self.gate(
            torch.cat([temporal.tokens, messages, temporal.reliability], dim=-1)
        )
        tokens = temporal.tokens + gate * messages
        return replace(temporal, tokens=self.norm(tokens))
