"""Task heads for foreground, center vectors, boundaries, and centers."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import ResidualAnisotropicBlock


class ScalarPredictionHead(nn.Module):
    """Small anisotropy-aware head returning one unnormalized logit channel."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int = 16,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.features = ResidualAnisotropicBlock(
            in_channels,
            hidden_channels,
            groups=groups,
            dropout=0.0,
        )
        self.output = nn.Conv3d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x))


class VectorPredictionHead(nn.Module):
    """Predict normalized Z/Y/X offsets in the closed interval ``[-1, 1]``."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int = 32,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.features = ResidualAnisotropicBlock(
            in_channels,
            hidden_channels,
            groups=groups,
            dropout=0.0,
        )
        self.output = nn.Conv3d(hidden_channels, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.output(self.features(x)))


class VectorCNNHeads(nn.Module):
    """Independent prediction heads attached to the full-resolution decoder map."""

    def __init__(self, in_channels: int, *, groups: int = 8) -> None:
        super().__init__()
        self.foreground = ScalarPredictionHead(in_channels, groups=groups)
        self.vectors = VectorPredictionHead(in_channels, groups=groups)
        self.boundary = ScalarPredictionHead(in_channels, groups=groups)
        self.center = ScalarPredictionHead(in_channels, groups=groups)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.foreground(x),
            self.vectors(x),
            self.boundary(x),
            self.center(x),
        )
