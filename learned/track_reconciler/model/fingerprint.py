"""Lightweight 3-D cell fingerprint encoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import FingerprintConfig


def _groups(channels: int, requested: int) -> int:
    value = min(int(requested), int(channels))
    while value > 1 and channels % value:
        value -= 1
    return value


class Residual3DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, groups: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_ch, groups), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_ch, groups), out_ch)
        self.drop = nn.Dropout3d(dropout)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_ch == out_ch
            else nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False)
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.drop(x)
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class CellFingerprintEncoder(nn.Module):
    """Encode raw/mask/EDT/boundary/context crops into a compact embedding.

    The network is deliberately small: segmentation is *not* its task.  It
    provides richer morphology/texture/state evidence to the reconciler.
    GroupNorm is used because reconciliation batches may be very small.
    """

    def __init__(self, config: FingerprintConfig) -> None:
        super().__init__()
        c = config.base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(config.in_channels, c, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(c, config.group_norm_groups), c),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            Residual3DBlock(c, c, 1, config.group_norm_groups, config.dropout),
            Residual3DBlock(c, 2 * c, 2, config.group_norm_groups, config.dropout),
            Residual3DBlock(2 * c, 2 * c, 1, config.group_norm_groups, config.dropout),
            Residual3DBlock(2 * c, 4 * c, 2, config.group_norm_groups, config.dropout),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(4 * c),
            nn.Linear(4 * c, config.embedding_dim),
        )

    def forward(self, crops: Tensor) -> Tensor:
        if crops.ndim != 5:
            raise ValueError("fingerprint encoder expects [M,C,D,H,W]")
        return self.head(self.pool(self.body(self.stem(crops))))
