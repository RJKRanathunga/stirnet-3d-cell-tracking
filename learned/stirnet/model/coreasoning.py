from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from .attention import LocalPhysicalCrossAttention
from .blocks import FeedForward, PhysicalAwareResBlock
from .config import CoReasoningConfig, SpatialConfig, TemporalConfig
from .coordinates import feature_grid_coordinates_um
from .graph_encoder import HypothesisGraphBlock
from .types import TemporalState


class CoReasoningBlock(nn.Module):
    def __init__(self, in_channels: int, spatial_cfg: SpatialConfig, temporal_cfg: TemporalConfig, cfg: CoReasoningConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.to_model = nn.Conv3d(in_channels, d, 1) if in_channels != d else nn.Identity()
        self.from_model = nn.Conv3d(d, in_channels, 1) if in_channels != d else nn.Identity()
        self.cross = LocalPhysicalCrossAttention(cfg)
        self.t_norm1 = nn.LayerNorm(d)
        self.t_norm2 = nn.LayerNorm(d)
        self.t_gate = nn.Sequential(nn.Linear(2 * d + 2, d), nn.Sigmoid())
        self.t_ffn = FeedForward(d, 2 * d, cfg.dropout)
        self.hyp_graph = HypothesisGraphBlock(temporal_cfg)
        self.s_norm = nn.LayerNorm(d)
        self.s_gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.spatial_blocks = nn.ModuleList([
            PhysicalAwareResBlock(d, d, spatial_cfg.acquisition_dim, spatial_cfg.group_norm_max_groups),
            PhysicalAwareResBlock(d, d, spatial_cfg.acquisition_dim, spatial_cfg.group_norm_max_groups),
        ])

    def forward(
        self,
        spatial_feature: Tensor,
        spacing_um: Tensor,
        temporal: TemporalState,
        dref_um: Tensor,
        acquisition_embedding: Tensor,
        spatial_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, TemporalState]:
        base = spatial_feature
        s3d = self.to_model(spatial_feature)
        B, D, Z, Y, X = s3d.shape
        spatial_tokens = s3d.flatten(2).transpose(1, 2)
        pos_um = feature_grid_coordinates_um((Z, Y, X), spacing_um, relative_to_center=True)

        if not temporal.is_empty:
            t_norm = self.t_norm1(temporal.tokens)
            t_msg = self.cross.temporal_reads_spatial(
                t_norm, temporal.ref_um, temporal.salience,
                self.s_norm(spatial_tokens), pos_um, temporal.batch_index,
                dref_um, spatial_padding_mask, self.cfg.base_radius_dref,
            )
            gate_in = torch.cat([temporal.tokens, t_msg, temporal.salience, temporal.reliability], dim=-1)
            t = temporal.tokens + self.t_gate(gate_in) * t_msg
            t = t + self.t_ffn(self.t_norm2(t))
            t = self.hyp_graph(t, temporal.edge_index, temporal.edge_attr)
            temporal = replace(temporal, tokens=t)

            s_msg = self.cross.spatial_reads_temporal(
                self.s_norm(spatial_tokens), pos_um,
                temporal.tokens, temporal.ref_um, temporal.salience, temporal.reliability,
                temporal.batch_index, dref_um, spatial_padding_mask, self.cfg.base_radius_dref,
            )
            s_gate = self.s_gate(torch.cat([spatial_tokens, s_msg], dim=-1))
            spatial_tokens = spatial_tokens + s_gate * s_msg

        s3d = spatial_tokens.transpose(1, 2).reshape(B, D, Z, Y, X)
        for block in self.spatial_blocks:
            s3d = block(s3d, acquisition_embedding)
        updated = base + self.from_model(s3d)
        return updated, temporal
