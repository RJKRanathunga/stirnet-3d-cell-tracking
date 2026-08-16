from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from ..config import GeometryConfig, SpatialConfig, TemporalConfig
from ..types import GeometryState, SpatialDecodeState, TemporalState


def _sample_local_grid(
    feature: Tensor,
    refs_um: Tensor,
    spacing_um: Tensor,
    radius_um: Tensor,
) -> Tensor:
    """Sample a 3x3x3 physical neighborhood for each reference; return [N,C].

    This is the align_corners=True, border-padded trilinear interpolation used
    by grid_sample, expressed as bounded gathers. CUDA autocast otherwise
    promotes the complete source volume to FP32 before sampling. Here only the
    gathered corner values and interpolation accumulator use FP32.
    """
    if refs_um.shape[0] == 0:
        return feature.new_zeros((0, feature.shape[0]))
    shape = feature.shape[-3:]
    coordinate_dtype = torch.float32
    unit = torch.tensor(
        [-1.0, 0.0, 1.0],
        device=refs_um.device,
        dtype=coordinate_dtype,
    )
    base = torch.stack(
        torch.meshgrid(unit, unit, unit, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    points_um = refs_um.float()[:, None] + (
        base[None] * radius_um.float().reshape(-1, 1, 1)
    )
    center_vox = points_um.new_tensor(
        [(shape[0] - 1) * 0.5, (shape[1] - 1) * 0.5, (shape[2] - 1) * 0.5]
    )
    points_vox = points_um / spacing_um.float().reshape(1, 1, 3) + center_vox
    maximum = points_um.new_tensor(
        [shape[0] - 1, shape[1] - 1, shape[2] - 1]
    )
    points_vox = torch.minimum(
        points_vox.clamp_min(0), maximum.reshape(1, 1, 3)
    ).reshape(-1, 3)
    lower = points_vox.floor().long()
    upper = torch.minimum(lower + 1, maximum.long())
    fraction = points_vox - lower.to(points_vox.dtype)
    accumulation_dtype = (
        torch.float32
        if feature.dtype in {torch.float16, torch.bfloat16}
        else feature.dtype
    )
    flattened = feature.reshape(feature.shape[0], -1)
    corner_indices = []
    corner_weights = []
    for z_index, z_weight in (
        (lower[:, 0], 1 - fraction[:, 0]),
        (upper[:, 0], fraction[:, 0]),
    ):
        for y_index, y_weight in (
            (lower[:, 1], 1 - fraction[:, 1]),
            (upper[:, 1], fraction[:, 1]),
        ):
            for x_index, x_weight in (
                (lower[:, 2], 1 - fraction[:, 2]),
                (upper[:, 2], fraction[:, 2]),
            ):
                linear_index = (
                    z_index * shape[1] * shape[2]
                    + y_index * shape[2]
                    + x_index
                )
                corner_indices.append(linear_index)
                corner_weights.append(z_weight * y_weight * x_weight)
    all_indices = torch.stack(corner_indices)
    all_weights = torch.stack(corner_weights).to(accumulation_dtype)
    sampled = flattened.index_select(1, all_indices.reshape(-1)).to(
        accumulation_dtype
    ).reshape(feature.shape[0], 8, -1)
    sampled = (sampled * all_weights[None]).sum(dim=1)
    return sampled.reshape(feature.shape[0], refs_um.shape[0], 27).permute(
        1, 0, 2
    ).mean(dim=-1)


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
        # A 1x1x1 convolution commutes with trilinear sampling and local mean
        # aggregation when it has no bias. Applying the equivalent Linear only
        # to sampled vectors avoids dense d_model-channel feature volumes.
        self.d1_proj = nn.Linear(spatial_cfg.channels[1], d, bias=False)
        self.d2_proj = nn.Linear(spatial_cfg.channels[2], d, bias=False)
        self.geometry_proj = nn.Linear(
            geometry_cfg.hidden_channels, d, bias=False
        )
        self.geometry_field_proj = nn.Linear(11, d, bias=False)
        self.message = nn.Sequential(
            nn.Linear(3 * d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * d + 1, d), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # V2 checkpoints written before sample-first observation store these
        # algebraically equivalent projections as [out,in,1,1,1] Conv3d
        # kernels. Accept them under strict loading for warm-start continuity.
        for name in (
            "d1_proj",
            "d2_proj",
            "geometry_proj",
            "geometry_field_proj",
        ):
            key = f"{prefix}{name}.weight"
            weight = state_dict.get(key)
            if weight is not None and weight.ndim == 5 and weight.shape[-3:] == (
                1,
                1,
                1,
            ):
                state_dict[key] = weight[..., 0, 0, 0]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
        messages = torch.zeros_like(temporal.tokens)
        for b in range(decoded.d0.shape[0]):
            idx = torch.nonzero(temporal.batch_index == b, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            refs = temporal.ref_um[idx]
            radius = dref_um[b] * self.cfg.observation_radius_dref
            radius_vec = radius.expand(len(idx))
            d1_local = _sample_local_grid(
                decoded.d1[b], refs, spatial_spacings_um[1][b], radius_vec
            )
            d2_local = _sample_local_grid(
                decoded.d2[b], refs, spatial_spacings_um[2][b], radius_vec
            )
            hidden_geometry_local = _sample_local_grid(
                geometry.features[b], refs, spacing_um[b], radius_vec
            )
            explicit_geometry_local = torch.cat(
                [
                    _sample_local_grid(
                        geometry.foreground_logits[b].sigmoid(),
                        refs,
                        spacing_um[b],
                        radius_vec,
                    ),
                    _sample_local_grid(
                        geometry.surface_logits[b].sigmoid(),
                        refs,
                        spacing_um[b],
                        radius_vec,
                    ),
                    _sample_local_grid(
                        geometry.separator_logits[b].sigmoid(),
                        refs,
                        spacing_um[b],
                        radius_vec,
                    ),
                    _sample_local_grid(
                        geometry.sdf[b], refs, spacing_um[b], radius_vec
                    ),
                    _sample_local_grid(
                        geometry.flow[b], refs, spacing_um[b], radius_vec
                    ),
                    _sample_local_grid(
                        geometry.centroid_offset[b],
                        refs,
                        spacing_um[b],
                        radius_vec,
                    ),
                    _sample_local_grid(
                        geometry.seed_logits[b].sigmoid(),
                        refs,
                        spacing_um[b],
                        radius_vec,
                    ),
                ],
                dim=-1,
            ).to(hidden_geometry_local.dtype)
            p1 = self.d1_proj(d1_local)
            p2 = self.d2_proj(d2_local)
            pg = self.geometry_proj(
                hidden_geometry_local
            ) + self.geometry_field_proj(explicit_geometry_local)
            local_message = self.message(torch.cat([p1, p2, pg], dim=-1))
            messages[idx] = local_message.to(messages.dtype)
        gate = self.gate(
            torch.cat([temporal.tokens, messages, temporal.reliability], dim=-1)
        )
        tokens = temporal.tokens + gate * messages
        return replace(temporal, tokens=self.norm(tokens))
