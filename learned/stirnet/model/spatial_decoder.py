from __future__ import annotations

import torch
from torch import Tensor, nn

from .blocks import PhysicalAwareResBlock, UpsampleBlock
from .checkpointing import checkpoint_if_enabled
from .config import SpatialConfig
from .types import SpatialPyramid


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        cfg: SpatialConfig,
        *,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.up = UpsampleBlock(in_channels, out_channels)
        self.fuse = nn.Conv3d(out_channels + skip_channels, out_channels, 1, bias=False)
        self.blocks = nn.ModuleList([
            PhysicalAwareResBlock(out_channels, out_channels, cfg.acquisition_dim, cfg.group_norm_max_groups)
            for _ in range(cfg.blocks_per_level)
        ])

    def _forward_impl(self, x: Tensor, skip: Tensor, acquisition_embedding: Tensor) -> Tensor:
        x = self.up(x, skip.shape[-3:])
        x = self.fuse(torch.cat([x, skip], dim=1))
        for block in self.blocks:
            x = block(x, acquisition_embedding)
        return x

    def forward(self, x: Tensor, skip: Tensor, acquisition_embedding: Tensor) -> Tensor:
        return checkpoint_if_enabled(
            self._forward_impl,
            x,
            skip,
            acquisition_embedding,
            enabled=self.activation_checkpointing and self.training,
        )


class SpatialDecoder(nn.Module):
    """Mirror decoder with an explicit E2-scale hook for the second co-reasoning block."""
    def __init__(self, cfg: SpatialConfig, *, activation_checkpointing: bool = False):
        super().__init__()
        ch = cfg.channels
        self.stage_e2 = DecoderStage(ch[3], ch[2], ch[2], cfg, activation_checkpointing=activation_checkpointing)
        self.stage_e1 = DecoderStage(ch[2], ch[1], ch[1], cfg, activation_checkpointing=activation_checkpointing)
        self.stage_e0 = DecoderStage(ch[1], ch[0], ch[0], cfg, activation_checkpointing=activation_checkpointing)
        self.mask_proj = nn.Conv3d(ch[0], cfg.mask_dim, 1)

    def decode_to_e2(self, deepest: Tensor, pyramid: SpatialPyramid, acquisition_embedding: Tensor) -> Tensor:
        return self.stage_e2(deepest, pyramid.features[2], acquisition_embedding)

    def decode_from_e2(self, e2_feature: Tensor, pyramid: SpatialPyramid, acquisition_embedding: Tensor):
        d1 = self.stage_e1(e2_feature, pyramid.features[1], acquisition_embedding)
        d0 = self.stage_e0(d1, pyramid.features[0], acquisition_embedding)
        mask_features = self.mask_proj(d0)
        return d1, d0, mask_features
