from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import HistoryConfig, TemporalConfig
from ..spatial.blocks import groups_for


class HistoricalInstanceEncoder(nn.Module):
    """Encode compact per-detection historical instance grids.

    This preserves the useful historical shape/occupancy evidence from the old
    STIR-Net, but it feeds the temporal memory rather than the dense current-
    frame spatial trunk.
    """

    def __init__(self, cfg: HistoryConfig, temporal_cfg: TemporalConfig):
        super().__init__()
        h = cfg.hidden_channels
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Conv3d(cfg.input_channels, h, 3, padding=1, bias=False),
            nn.GroupNorm(groups_for(h), h),
            nn.SiLU(),
            nn.Conv3d(h, h, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups_for(h), h),
            nn.SiLU(),
            nn.Conv3d(h, 2 * h, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups_for(2 * h), 2 * h),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(4 * h, temporal_cfg.d_model),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(temporal_cfg.d_model, temporal_cfg.d_model),
        )
        self.invalid_token = nn.Parameter(torch.zeros(temporal_cfg.d_model))

    def forward(
        self, grids: Tensor, valid: Tensor | None = None
    ) -> Tensor:
        if grids.ndim != 5 or grids.shape[1] != self.cfg.input_channels:
            raise ValueError(
                f"history grids must have shape [N,{self.cfg.input_channels},G,G,G]"
            )
        if grids.shape[0] == 0:
            return grids.new_zeros((0, self.invalid_token.numel()))
        x = self.net(grids)
        mean = F.adaptive_avg_pool3d(x, 1).flatten(1)
        maximum = F.adaptive_max_pool3d(x, 1).flatten(1)
        token = self.proj(torch.cat([mean, maximum], dim=-1))
        if valid is not None:
            valid = valid.bool()
            token = torch.where(
                valid[:, None], token, self.invalid_token[None].to(token.dtype)
            )
        return token
