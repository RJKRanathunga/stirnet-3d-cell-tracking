"""Prediction heads for foreground, canonical vectors, boundary, and centers."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvNormAct3D


class ScalarPredictionHead(nn.Module):
    def __init__(self, in_channels: int, *, groups: int = 8) -> None:
        super().__init__()
        hidden = max(8, in_channels // 2)
        self.features = ConvNormAct3D(in_channels, hidden, kernel_size=3, groups=groups)
        self.output = nn.Conv3d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x))


class VectorPredictionHead(nn.Module):
    def __init__(self, in_channels: int, *, groups: int = 8) -> None:
        super().__init__()
        hidden = max(12, in_channels // 2)
        self.features = ConvNormAct3D(in_channels, hidden, kernel_size=3, groups=groups)
        self.output = nn.Conv3d(hidden, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.output(self.features(x)))


class VectorCNNHeads(nn.Module):
    def __init__(self, in_channels: int, *, groups: int = 8) -> None:
        super().__init__()
        self.foreground = ScalarPredictionHead(in_channels, groups=groups)
        self.vectors = VectorPredictionHead(in_channels, groups=groups)
        self.boundary = ScalarPredictionHead(in_channels, groups=groups)
        self.center = ScalarPredictionHead(in_channels, groups=groups)

    def forward(self, x: torch.Tensor):
        return (
            self.foreground(x),
            self.vectors(x),
            self.boundary(x),
            self.center(x),
        )
