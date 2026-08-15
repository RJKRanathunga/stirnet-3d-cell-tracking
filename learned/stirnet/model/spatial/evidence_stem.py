from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import EvidenceStemConfig, SpatialConfig
from .blocks import PhysicalAwareResBlock, groups_for


class EvidenceFusionStem(nn.Module):
    """Keep raw evidence authoritative while treating current segmentation as a prior.

    The old model concatenated all inputs symmetrically. Here raw microscopy and
    segmentation-derived channels receive independent feature stems. A learned,
    acquisition-conditioned gate controls how much prior evidence may enter.
    During training, prior dropout makes missing/noisy masks non-catastrophic.
    """

    def __init__(self, cfg: EvidenceStemConfig, spatial_cfg: SpatialConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.stem_channels
        self.raw_stem = nn.Sequential(
            nn.Conv3d(len(cfg.raw_channels), c, 3, padding=1, bias=False),
            nn.GroupNorm(groups_for(c), c),
            nn.SiLU(),
            nn.Conv3d(c, c, 3, padding=1, bias=False),
        )
        self.prior_stem = nn.Sequential(
            nn.Conv3d(len(cfg.prior_channels), c, 3, padding=1, bias=False),
            nn.GroupNorm(groups_for(c), c),
            nn.SiLU(),
            nn.Conv3d(c, c, 3, padding=1, bias=False),
        )
        self.prior_gate = nn.Sequential(
            nn.Linear(spatial_cfg.acquisition_dim, cfg.prior_gate_hidden),
            nn.SiLU(),
            nn.Linear(cfg.prior_gate_hidden, c),
        )
        # Conservative initial prior trust: sigmoid(-1) ~= 0.27.
        nn.init.constant_(self.prior_gate[-1].bias, -1.0)
        self.out = nn.Conv3d(c, spatial_cfg.channels[0], 1, bias=False)

    def forward(
        self,
        spatial_inputs: Tensor,
        acquisition_embedding: Tensor,
        *,
        prior_keep_mask: Tensor | None = None,
    ) -> Tensor:
        if spatial_inputs.shape[1] <= max(self.cfg.raw_channels + self.cfg.prior_channels):
            raise ValueError(
                "spatial_inputs does not contain all configured raw/prior channels"
            )
        raw = spatial_inputs[:, self.cfg.raw_channels]
        prior = spatial_inputs[:, self.cfg.prior_channels]
        if self.training and self.cfg.prior_dropout > 0:
            if prior_keep_mask is None:
                keep = torch.rand(
                    (prior.shape[0], 1, 1, 1, 1), device=prior.device
                ) >= self.cfg.prior_dropout
            else:
                keep = prior_keep_mask.reshape(prior.shape[0], 1, 1, 1, 1).bool()
            prior = prior * keep.to(prior.dtype)
        raw_feat = self.raw_stem(raw)
        prior_feat = self.prior_stem(prior)
        gate = torch.sigmoid(self.prior_gate(acquisition_embedding))[:, :, None, None, None]
        fused = raw_feat + gate * prior_feat
        return self.out(F.silu(fused))
