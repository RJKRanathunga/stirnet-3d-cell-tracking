from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def groups_for(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class AxisFactorizedConv(nn.Module):
    """Spacing-conditioned z/y/x factorized convolution from the current model."""

    def __init__(self, channels: int, acquisition_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.conv_z = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), bias=False)
        self.conv_y = nn.Conv3d(channels, channels, (1, 3, 1), padding=(0, 1, 0), bias=False)
        self.conv_x = nn.Conv3d(channels, channels, (1, 1, 3), padding=(0, 0, 1), bias=False)
        self.gate = nn.Linear(acquisition_dim, 3 * channels)
        self.fuse = nn.Conv3d(channels, channels, 1, bias=False)

    def forward(self, x: Tensor, acquisition_embedding: Tensor) -> Tensor:
        gates = torch.sigmoid(self.gate(acquisition_embedding)).view(
            x.shape[0], 3, self.channels, 1, 1, 1
        )
        out = self.conv_z(x) * gates[:, 0]
        out = out + self.conv_y(x) * gates[:, 1]
        out = out + self.conv_x(x) * gates[:, 2]
        return self.fuse(out)


class PhysicalAwareResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        acquisition_dim: int = 64,
        max_groups: int = 8,
    ):
        super().__init__()
        self.pre = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.norm1 = nn.GroupNorm(groups_for(out_channels, max_groups), out_channels)
        self.norm2 = nn.GroupNorm(groups_for(out_channels, max_groups), out_channels)
        self.conv1 = AxisFactorizedConv(out_channels, acquisition_dim)
        self.conv2 = AxisFactorizedConv(out_channels, acquisition_dim)

    def forward(self, x: Tensor, acquisition_embedding: Tensor) -> Tensor:
        residual = self.pre(x)
        x = self.conv1(F.silu(self.norm1(residual)), acquisition_embedding)
        x = self.conv2(F.silu(self.norm2(x)), acquisition_embedding)
        return residual + x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: Tuple[int, int, int]):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=stride, stride=stride, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UpsampleFuse(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        acquisition_dim: int,
        blocks: int,
        max_groups: int,
    ):
        super().__init__()
        self.up = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.fuse = nn.Conv3d(out_channels + skip_channels, out_channels, 1, bias=False)
        self.blocks = nn.ModuleList(
            [
                PhysicalAwareResBlock(
                    out_channels, out_channels, acquisition_dim, max_groups
                )
                for _ in range(blocks)
            ]
        )

    def forward(self, x: Tensor, skip: Tensor, acquisition_embedding: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = self.up(x)
        x = self.fuse(torch.cat([x, skip], dim=1))
        for block in self.blocks:
            x = block(x, acquisition_embedding)
        return x
