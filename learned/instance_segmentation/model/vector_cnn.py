"""Compact isotropic residual 3-D U-Net for cubic canonical instance correction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .decoder import VectorCNNDecoder
from .encoder import VectorCNNEncoder
from .heads import VectorCNNHeads


@dataclass(frozen=True)
class VectorCNNConfig:
    """Architecture configuration for the normalized cubic CNN."""

    input_channels: int = 4
    channels: tuple[int, int, int, int, int] = (16, 32, 64, 128, 192)
    group_norm_groups: int = 8
    dropout: float = 0.0
    input_shape_zyx: tuple[int, int, int] = (64, 64, 64)
    enforce_input_shape: bool = True

    def __post_init__(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if len(self.channels) != 5 or any(v <= 0 for v in self.channels):
            raise ValueError("channels must contain five positive values")
        if self.group_norm_groups <= 0:
            raise ValueError("group_norm_groups must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if len(self.input_shape_zyx) != 3 or any(int(v) < 16 for v in self.input_shape_zyx):
            raise ValueError("input_shape_zyx must contain three values >= 16")
        if any(int(v) % 16 != 0 for v in self.input_shape_zyx):
            raise ValueError("each input spatial dimension must be divisible by 16")


@dataclass(frozen=True)
class VectorCNNOutput:
    foreground_logits: torch.Tensor
    vectors_normalized: torch.Tensor
    boundary_logits: torch.Tensor
    center_logits: torch.Tensor

    @property
    def foreground_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.foreground_logits)

    @property
    def boundary_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.boundary_logits)

    @property
    def center_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.center_logits)

    def vectors_to_canonical_displacement(self) -> torch.Tensor:
        """Convert axis-fraction vectors to canonical voxel displacement."""
        spatial = torch.tensor(
            [max(int(v) - 1, 1) for v in self.vectors_normalized.shape[-3:]],
            dtype=self.vectors_normalized.dtype,
            device=self.vectors_normalized.device,
        ).view(1, 3, 1, 1, 1)
        return self.vectors_normalized * spatial


class VectorInstanceCNN(nn.Module):
    """Multi-task CNN operating on object-normalized cubic 3-D ROIs.

    Input channels:
      0 normalized fluorescence
      1 Stage-2-like component mask
      2 normalized cubic-voxel EDT
      3 effective-marker heatmap

    Output vectors are canonical axis fractions.  For the default 64^3 input,
    multiplying by 63 converts each vector channel to canonical voxel offsets.
    """

    def __init__(self, config: VectorCNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or VectorCNNConfig()
        self.encoder = VectorCNNEncoder(
            in_channels=self.config.input_channels,
            channels=self.config.channels,
            groups=self.config.group_norm_groups,
            dropout=self.config.dropout,
        )
        self.decoder = VectorCNNDecoder(
            channels=self.config.channels,
            groups=self.config.group_norm_groups,
            dropout=self.config.dropout,
        )
        self.heads = VectorCNNHeads(
            self.config.channels[0], groups=self.config.group_norm_groups
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final_layers = (
            self.heads.foreground.output,
            self.heads.vectors.output,
            self.heads.boundary.output,
            self.heads.center.output,
        )
        for layer in final_layers:
            nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(f"VectorInstanceCNN expects [B,C,Z,Y,X], got {tuple(x.shape)}")
        if x.shape[1] != self.config.input_channels:
            raise ValueError(
                f"expected {self.config.input_channels} input channels, got {x.shape[1]}"
            )
        spatial = tuple(int(v) for v in x.shape[-3:])
        if any(v < 16 or v % 16 != 0 for v in spatial):
            raise ValueError("spatial dimensions must each be >=16 and divisible by 16")
        if self.config.enforce_input_shape and spatial != tuple(self.config.input_shape_zyx):
            raise ValueError(
                f"expected spatial shape {self.config.input_shape_zyx}, got {spatial}"
            )

    def forward(self, x: torch.Tensor) -> VectorCNNOutput:
        self._validate_input(x)
        features = self.encoder(x)
        decoded = self.decoder(features)
        foreground, vectors, boundary, center = self.heads(decoded)
        return VectorCNNOutput(foreground, vectors, boundary, center)
