"""Fully isotropic encoder for the cubic vector instance CNN."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from .blocks import Downsample3D, ResidualIsotropicBlock


@dataclass(frozen=True)
class EncoderFeatures:
    level0: torch.Tensor
    level1: torch.Tensor
    level2: torch.Tensor
    level3: torch.Tensor
    bottleneck: torch.Tensor


class VectorCNNEncoder(nn.Module):
    """Five-level isotropic encoder: 64 -> 32 -> 16 -> 8 -> 4 for 64^3 inputs."""

    def __init__(
        self,
        *,
        in_channels: int,
        channels: tuple[int, int, int, int, int],
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(channels) != 5 or any(v <= 0 for v in channels):
            raise ValueError("channels must contain five positive values")
        c0, c1, c2, c3, cb = channels
        self.level0 = ResidualIsotropicBlock(in_channels, c0, groups=groups, dropout=dropout)
        self.down1 = Downsample3D(c0, c1, stride=(2, 2, 2), groups=groups)
        self.level1 = ResidualIsotropicBlock(c1, c1, groups=groups, dropout=dropout)
        self.down2 = Downsample3D(c1, c2, stride=(2, 2, 2), groups=groups)
        self.level2 = ResidualIsotropicBlock(c2, c2, groups=groups, dropout=dropout)
        self.down3 = Downsample3D(c2, c3, stride=(2, 2, 2), groups=groups)
        self.level3 = ResidualIsotropicBlock(c3, c3, groups=groups, dropout=dropout)
        self.down4 = Downsample3D(c3, cb, stride=(2, 2, 2), groups=groups)
        self.bottleneck = ResidualIsotropicBlock(cb, cb, groups=groups, dropout=dropout)

    def forward(self, x: torch.Tensor) -> EncoderFeatures:
        level0 = self.level0(x)
        level1 = self.level1(self.down1(level0))
        level2 = self.level2(self.down2(level1))
        level3 = self.level3(self.down3(level2))
        bottleneck = self.bottleneck(self.down4(level3))
        return EncoderFeatures(level0, level1, level2, level3, bottleneck)
