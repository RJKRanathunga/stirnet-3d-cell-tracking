from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

from .blocks import DownsampleBlock, PhysicalAwareResBlock
from .checkpointing import checkpoint_if_enabled
from .config import SpatialConfig
from .spacing import choose_downsample_stride, propagate_spacing
from .types import SpatialPyramid


class SpatialEncoder(nn.Module):
    def __init__(self, cfg: SpatialConfig, *, activation_checkpointing: bool = False):
        super().__init__()
        self.cfg = cfg
        self.activation_checkpointing = activation_checkpointing
        ch = cfg.channels
        self.stem = nn.Conv3d(cfg.in_channels, ch[0], 1, bias=False)
        self.levels = nn.ModuleList()
        for c in ch:
            self.levels.append(nn.ModuleList([
                PhysicalAwareResBlock(c, c, cfg.acquisition_dim, cfg.group_norm_max_groups)
                for _ in range(cfg.blocks_per_level)
            ]))
        # Strides are selected at runtime; use a bank for the only 7 useful stride combinations.
        combos = [(1,1,2),(1,2,1),(2,1,1),(1,2,2),(2,1,2),(2,2,1),(2,2,2)]
        self.downs = nn.ModuleList()
        for i in range(len(ch)-1):
            bank = nn.ModuleDict({
                "".join(map(str, s)): DownsampleBlock(ch[i], ch[i+1], s) for s in combos
            })
            self.downs.append(bank)

    def forward(
        self,
        x: Tensor,
        spacing_um: Tensor,
        acquisition_embedding: Tensor,
        padding_mask: Tensor | None = None,
    ) -> SpatialPyramid:
        features: List[Tensor] = []
        spacings: List[Tensor] = []
        strides: List[tuple[int,int,int]] = []
        masks: List[Tensor] = []

        x = self.stem(x)
        current_spacing = spacing_um
        current_mask = padding_mask
        for level_idx, blocks in enumerate(self.levels):
            def run_level(
                level_input: Tensor,
                embedding: Tensor,
                level_blocks: nn.ModuleList = blocks,
            ) -> Tensor:
                result = level_input
                for block in level_blocks:
                    result = block(result, embedding)
                return result

            x = checkpoint_if_enabled(
                run_level,
                x,
                acquisition_embedding,
                enabled=self.activation_checkpointing and self.training,
            )
            features.append(x)
            spacings.append(current_spacing)
            if current_mask is not None:
                masks.append(current_mask)
            if level_idx < len(self.levels)-1:
                stride = choose_downsample_stride(current_spacing, self.cfg.anisotropy_threshold)
                strides.append(stride)
                key = "".join(map(str, stride))
                x = self.downs[level_idx][key](x)
                current_spacing = propagate_spacing(current_spacing, stride)
                if current_mask is not None:
                    current_mask = torch.nn.functional.max_pool3d(
                        current_mask.float().unsqueeze(1), kernel_size=stride, stride=stride
                    ).squeeze(1).bool()
        return SpatialPyramid(
            features=features,
            spacings_um=spacings,
            strides=strides,
            padding_masks=masks if padding_mask is not None else None,
        )
