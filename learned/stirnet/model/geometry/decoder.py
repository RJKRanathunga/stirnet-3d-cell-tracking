from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import GeometryConfig, SpatialConfig
from ..spatial.blocks import PhysicalAwareResBlock
from ..types import GeometryState


class DenseGeometryDecoder(nn.Module):
    """Native-resolution geometric representation.

    Research basis:
      * Omnipose: distance field + distance-gradient flow + boundary evidence.
      * NucMM: foreground + contour/separator + signed-distance hybrid representation.
      * NISNet3D: learned 3-D vector evidence for marker/boundary reasoning.

    The heads intentionally share a non-trivial 3x3x3 residual trunk instead of
    using independent 1x1 auxiliary projections from D0.
    """

    def __init__(self, cfg: GeometryConfig, spatial_cfg: SpatialConfig):
        super().__init__()
        self.cfg = cfg
        self.activation_checkpointing = spatial_cfg.activation_checkpointing
        in_ch = spatial_cfg.channels[0]
        hidden = cfg.hidden_channels
        self.input_proj = nn.Conv3d(in_ch, hidden, 3, padding=1, bias=False)
        self.blocks = nn.ModuleList(
            [
                PhysicalAwareResBlock(
                    hidden,
                    hidden,
                    spatial_cfg.acquisition_dim,
                    spatial_cfg.group_norm_max_groups,
                )
                for _ in range(cfg.residual_blocks)
            ]
        )
        self.foreground = nn.Conv3d(hidden, 1, 1)
        self.surface = nn.Conv3d(hidden, 1, 1)
        self.separator = nn.Conv3d(hidden, 1, 1)
        self.sdf = nn.Conv3d(hidden, 1, 1)
        self.flow = nn.Conv3d(hidden, 3, 1)
        self.centroid_offset = nn.Conv3d(hidden, 3, 1)
        self.seed = nn.Conv3d(hidden, 1, 1)

    def forward(self, d0: Tensor, acquisition_embedding: Tensor) -> GeometryState:
        x = self.input_proj(d0)
        for block in self.blocks:
            if (
                self.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
                and (x.requires_grad or acquisition_embedding.requires_grad)
            ):
                x = checkpoint(
                    block, x, acquisition_embedding, use_reentrant=False
                )
            else:
                x = block(x, acquisition_embedding)
        # SDF is bounded in cell-reference units, preventing large outliers from
        # dominating watershed energy while retaining signed geometry.
        sdf = self.cfg.sdf_clip_dref * F.tanh(self.sdf(x))
        # Omnipose-style flow is a direction field; tanh keeps each component
        # bounded and the loss later normalizes it before angular comparison.
        flow = torch.tanh(self.flow(x))
        centroid_offset = self.centroid_offset(x)
        return GeometryState(
            foreground_logits=self.foreground(x),
            surface_logits=self.surface(x),
            separator_logits=self.separator(x),
            sdf=sdf,
            flow=flow,
            centroid_offset=centroid_offset,
            seed_logits=self.seed(x),
            features=x,
        )
