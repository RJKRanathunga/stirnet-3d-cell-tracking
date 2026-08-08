"""Reusable 3-D convolutional blocks for the vector instance CNN."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

Stride3D = tuple[int, int, int]


def _group_count(channels: int, requested_groups: int) -> int:
    if channels <= 0 or requested_groups <= 0:
        raise ValueError("channels and requested_groups must be positive")
    groups = min(channels, requested_groups)
    while channels % groups != 0:
        groups -= 1
    return groups


class ConvNormAct3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | None = None,
        groups: int = 8,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                kernel_tuple = tuple(int(v) for v in kernel_size)
                padding = tuple(v // 2 for v in kernel_tuple)
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.norm = nn.GroupNorm(_group_count(out_channels, groups), out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualAnisotropicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.conv_xy = ConvNormAct3D(
            in_channels, out_channels, kernel_size=(1, 3, 3),
            padding=(0, 1, 1), groups=groups
        )
        self.conv_z = nn.Conv3d(
            out_channels, out_channels, kernel_size=(3, 1, 1),
            padding=(1, 0, 0), bias=False
        )
        self.norm = nn.GroupNorm(_group_count(out_channels, groups), out_channels)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        y = self.conv_xy(x)
        y = self.norm(self.conv_z(y))
        y = self.dropout(y)
        return self.act(y + residual)


class ResidualIsotropicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.conv1 = ConvNormAct3D(in_channels, out_channels, groups=groups)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels, groups), out_channels)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        y = self.conv1(x)
        y = self.norm(self.conv2(y))
        y = self.dropout(y)
        return self.act(y + residual)


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: Stride3D, groups: int = 8) -> None:
        super().__init__()
        if any(v not in (1, 2) for v in stride):
            raise ValueError("downsample stride values must be 1 or 2")
        self.proj = ConvNormAct3D(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, groups=groups
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, groups: int = 8) -> None:
        super().__init__()
        self.proj = ConvNormAct3D(in_channels, out_channels, kernel_size=1, padding=0, groups=groups)

    def forward(self, x: torch.Tensor, *, target_shape: tuple[int, int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)
        return self.proj(x)
