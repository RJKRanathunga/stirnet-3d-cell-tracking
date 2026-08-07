"""Anisotropy-aware encoder for the vector instance CNN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .blocks import (
    Downsample3D,
    ResidualAnisotropicBlock,
    ResidualIsotropicBlock,
)


@dataclass(frozen=True)
class EncoderFeatures:
    """Skip features plus the bottleneck feature map."""

    level0: torch.Tensor
    level1: torch.Tensor
    level2: torch.Tensor
    level3: torch.Tensor
    bottleneck: torch.Tensor


class VectorCNNEncoder(nn.Module):
    """Four-level encoder with delayed Z downsampling.

    For the canonical Biohub crop ``16x64x64`` and channel tuple
    ``(24, 48, 96, 160, 256)``, the feature shapes are:

    - level0: ``16x64x64``
    - level1: ``16x32x32``
    - level2: ``16x16x16``
    - level3: ``8x8x8``
    - bottleneck: ``4x4x4``
    """

    def __init__(
        self,
        *,
        in_channels: int,
        channels: tuple[int, int, int, int, int],
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(channels) != 5 or any(value <= 0 for value in channels):
            raise ValueError("channels must contain five positive values")

        c0, c1, c2, c3, cb = channels

        self.level0 = ResidualAnisotropicBlock(
            in_channels, c0, groups=groups, dropout=dropout
        )

        self.down1 = Downsample3D(c0, c1, stride=(1, 2, 2), groups=groups)
        self.level1 = ResidualAnisotropicBlock(
            c1, c1, groups=groups, dropout=dropout
        )

        self.down2 = Downsample3D(c1, c2, stride=(1, 2, 2), groups=groups)
        self.level2 = ResidualIsotropicBlock(
            c2, c2, groups=groups, dropout=dropout
        )

        self.down3 = Downsample3D(c2, c3, stride=(2, 2, 2), groups=groups)
        self.level3 = ResidualIsotropicBlock(
            c3, c3, groups=groups, dropout=dropout
        )

        self.down4 = Downsample3D(c3, cb, stride=(2, 2, 2), groups=groups)
        self.bottleneck = ResidualIsotropicBlock(
            cb, cb, groups=groups, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> EncoderFeatures:
        level0 = self.level0(x)
        level1 = self.level1(self.down1(level0))
        level2 = self.level2(self.down2(level1))
        level3 = self.level3(self.down3(level2))
        bottleneck = self.bottleneck(self.down4(level3))
        return EncoderFeatures(level0, level1, level2, level3, bottleneck)
