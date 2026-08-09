from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _groups(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class AxisFactorizedConv(nn.Module):
    def __init__(self, channels: int, acquisition_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.conv_z = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), bias=False)
        self.conv_y = nn.Conv3d(channels, channels, (1, 3, 1), padding=(0, 1, 0), bias=False)
        self.conv_x = nn.Conv3d(channels, channels, (1, 1, 3), padding=(0, 0, 1), bias=False)
        self.gate = nn.Linear(acquisition_dim, 3 * channels)
        self.fuse = nn.Conv3d(channels, channels, 1, bias=False)

    def forward(self, x: Tensor, acquisition_embedding: Tensor) -> Tensor:
        gates = torch.sigmoid(self.gate(acquisition_embedding)).view(x.shape[0], 3, self.channels, 1, 1, 1)
        z = self.conv_z(x) * gates[:, 0]
        y = self.conv_y(x) * gates[:, 1]
        xx = self.conv_x(x) * gates[:, 2]
        return self.fuse(z + y + xx)


class PhysicalAwareResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, acquisition_dim: int = 64, max_groups: int = 8):
        super().__init__()
        self.pre = nn.Conv3d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        self.norm1 = nn.GroupNorm(_groups(out_channels, max_groups), out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels, max_groups), out_channels)
        self.conv1 = AxisFactorizedConv(out_channels, acquisition_dim)
        self.conv2 = AxisFactorizedConv(out_channels, acquisition_dim)

    def forward(self, x: Tensor, acquisition_embedding: Tensor) -> Tensor:
        residual = self.pre(x)
        x = residual
        x = self.conv1(F.silu(self.norm1(x)), acquisition_embedding)
        x = self.conv2(F.silu(self.norm2(x)), acquisition_embedding)
        return x + residual


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: Tuple[int, int, int]):
        super().__init__()
        self.stride = tuple(stride)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=self.stride, stride=self.stride, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: Tensor, target_shape: tuple[int, int, int]) -> Tensor:
        x = F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)
        return self.proj(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
