from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import SpatialConfig
from ..types import SpatialDecodeState, SpatialPyramid
from .acquisition import choose_downsample_stride, propagate_spacing
from .blocks import DownsampleBlock, PhysicalAwareResBlock, UpsampleFuse


class AnisotropyAwareSpatialBackbone(nn.Module):
    """Native-resolution residual 3D U-Net with runtime physical downsampling."""

    _STRIDES = (
        (1, 1, 2),
        (1, 2, 1),
        (2, 1, 1),
        (1, 2, 2),
        (2, 1, 2),
        (2, 2, 1),
        (2, 2, 2),
    )

    def __init__(self, cfg: SpatialConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.channels
        self.levels = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PhysicalAwareResBlock(
                            c,
                            c,
                            cfg.acquisition_dim,
                            cfg.group_norm_max_groups,
                            cfg.axis_conv_variant,
                            cfg.axis_conv_bottleneck_ratio,
                        )
                        for _ in range(cfg.blocks_per_level)
                    ]
                )
                for c in ch
            ]
        )
        self.downs = nn.ModuleList()
        for i in range(3):
            self.downs.append(
                nn.ModuleDict(
                    {
                        "".join(map(str, stride)): DownsampleBlock(
                            ch[i], ch[i + 1], stride
                        )
                        for stride in self._STRIDES
                    }
                )
            )
        self.up2 = UpsampleFuse(
            ch[3], ch[2], ch[2], cfg.acquisition_dim, cfg.blocks_per_level, cfg.group_norm_max_groups,
            cfg.axis_conv_variant, cfg.axis_conv_bottleneck_ratio,
        )
        self.up1 = UpsampleFuse(
            ch[2], ch[1], ch[1], cfg.acquisition_dim, cfg.blocks_per_level, cfg.group_norm_max_groups,
            cfg.axis_conv_variant, cfg.axis_conv_bottleneck_ratio,
        )
        self.up0 = UpsampleFuse(
            ch[1], ch[0], ch[0], cfg.acquisition_dim, cfg.blocks_per_level, cfg.group_norm_max_groups,
            cfg.axis_conv_variant, cfg.axis_conv_bottleneck_ratio,
        )

    def _run(self, module: nn.Module, *args: Tensor) -> Tensor:
        if (
            self.cfg.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and any(arg.requires_grad for arg in args)
        ):
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(
        self,
        x0: Tensor,
        spacing_um: Tensor,
        acquisition_embedding: Tensor,
        padding_mask: Tensor | None = None,
        stage_profiler=None,
    ) -> tuple[SpatialPyramid, SpatialDecodeState]:
        def profiled(name: str):
            return (
                nullcontext()
                if stage_profiler is None
                else stage_profiler.profile(name)
            )

        features: list[Tensor] = []
        spacings: list[Tensor] = []
        strides: list[tuple[int, int, int]] = []
        masks: list[Tensor] = []
        x = x0
        current_spacing = spacing_um
        current_mask = padding_mask
        for level_idx, blocks in enumerate(self.levels):
            with profiled(f"encoder_level{level_idx}"):
                for block in blocks:
                    x = self._run(block, x, acquisition_embedding)
            features.append(x)
            spacings.append(current_spacing)
            if current_mask is not None:
                masks.append(current_mask)
            if level_idx < 3:
                with profiled(f"down{level_idx}"):
                    stride = choose_downsample_stride(
                        current_spacing, self.cfg.anisotropy_threshold
                    )
                    strides.append(stride)
                    x = self.downs[level_idx]["".join(map(str, stride))](x)
                    current_spacing = propagate_spacing(current_spacing, stride)
                    if current_mask is not None:
                        current_mask = F.max_pool3d(
                            current_mask.float().unsqueeze(1),
                            kernel_size=stride,
                            stride=stride,
                        ).squeeze(1).bool()
        pyramid = SpatialPyramid(
            features=features,
            spacings_um=spacings,
            strides=strides,
            padding_masks=masks if padding_mask is not None else None,
        )
        with profiled("decoder_up2"):
            d2 = self._run(self.up2, features[3], features[2], acquisition_embedding)
        with profiled("decoder_up1"):
            d1 = self._run(self.up1, d2, features[1], acquisition_embedding)
        with profiled("decoder_up0"):
            d0 = self._run(self.up0, d1, features[0], acquisition_embedding)
        return pyramid, SpatialDecodeState(d2=d2, d1=d1, d0=d0)
