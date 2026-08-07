"""Compact anisotropic residual 3D U-Net for instance-segmentation correction."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn

from .decoder import VectorCNNDecoder
from .encoder import VectorCNNEncoder
from .heads import VectorCNNHeads


@dataclass(frozen=True)
class VectorCNNConfig:
    """Architecture and physical-vector configuration for the V1 model."""

    input_channels: int = 4
    channels: tuple[int, int, int, int, int] = (24, 48, 96, 160, 256)
    group_norm_groups: int = 8
    dropout: float = 0.0
    vector_max_distance_um: float = 16.0

    def __post_init__(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if len(self.channels) != 5 or any(value <= 0 for value in self.channels):
            raise ValueError("channels must contain five positive values")
        if self.group_norm_groups <= 0:
            raise ValueError("group_norm_groups must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if (
            not isfinite(float(self.vector_max_distance_um))
            or self.vector_max_distance_um <= 0
        ):
            raise ValueError("vector_max_distance_um must be positive")


@dataclass(frozen=True)
class VectorCNNOutput:
    """Dense predictions produced at the same spatial resolution as the input."""

    foreground_logits: torch.Tensor
    vectors_normalized: torch.Tensor
    boundary_logits: torch.Tensor
    center_logits: torch.Tensor

    def vectors_um(self, max_distance_um: float) -> torch.Tensor:
        """Convert normalized offsets to physical micrometre offsets."""
        if max_distance_um <= 0:
            raise ValueError("max_distance_um must be positive")
        return self.vectors_normalized * float(max_distance_um)

    @property
    def foreground_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.foreground_logits)

    @property
    def boundary_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.boundary_logits)

    @property
    def center_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.center_logits)


class VectorInstanceCNN(nn.Module):
    """V1 learned correction model for merged-cell instance segmentation.

    Expected input channels are, by convention:

    0. normalized fluorescence,
    1. Stage-2-like foreground/component mask,
    2. normalized physical EDT,
    3. effective-marker Gaussian heatmap.

    The architecture is intentionally component-centric and preserves full
    resolution in Z through the first two encoder reductions. It does not
    produce instance IDs directly; its vector, center, and boundary outputs are
    intended for deterministic voting/marker reconciliation/watershed.
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
            self.config.channels[0],
            groups=self.config.group_norm_groups,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize residual features strongly and task outputs near zero.

        Small final-head weights avoid saturating the vector ``tanh`` before
        learning starts while still allowing gradients to reach the shared
        decoder on the first optimization step.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
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
            raise ValueError(
                "VectorInstanceCNN expects [B, C, Z, Y, X], "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.config.input_channels:
            raise ValueError(
                f"expected {self.config.input_channels} input channels, "
                f"got {x.shape[1]}"
            )
        z, y, x_size = (int(value) for value in x.shape[-3:])
        if z < 4 or y < 16 or x_size < 16:
            raise ValueError(
                "spatial input is too small for the four-level encoder; "
                "require at least Z>=4, Y>=16, X>=16"
            )

    def forward(self, x: torch.Tensor) -> VectorCNNOutput:
        self._validate_input(x)
        features = self.encoder(x)
        decoded = self.decoder(features)
        foreground, vectors, boundary, center = self.heads(decoded)
        return VectorCNNOutput(
            foreground_logits=foreground,
            vectors_normalized=vectors,
            boundary_logits=boundary,
            center_logits=center,
        )

    def vectors_to_um(self, vectors_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized vector predictions to physical micrometres."""
        return vectors_normalized * self.config.vector_max_distance_um
