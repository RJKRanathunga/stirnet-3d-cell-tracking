from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
        self.output_dim = temporal_cfg.d_model

    def _encode(self, grids: Tensor) -> Tensor:
        x = self.net(grids)
        mean = F.adaptive_avg_pool3d(x, 1).flatten(1)
        maximum = F.adaptive_max_pool3d(x, 1).flatten(1)
        return self.proj(torch.cat([mean, maximum], dim=-1))

    def forward(
        self, grids: Tensor, valid: Tensor | None = None
    ) -> Tensor:
        if grids.ndim != 5 or grids.shape[1] != self.cfg.input_channels:
            raise ValueError(
                f"history grids must have shape [N,{self.cfg.input_channels},G,G,G]"
            )
        if grids.shape[0] == 0:
            return grids.new_zeros((0, self.output_dim))
        valid_mask = (
            torch.ones(grids.shape[0], device=grids.device, dtype=torch.bool)
            if valid is None
            else valid.to(device=grids.device, dtype=torch.bool)
        )
        if valid_mask.shape != (grids.shape[0],):
            raise ValueError("node_history_valid must have shape [N]")
        parameter_dtype = self.net[0].weight.dtype
        output = torch.zeros(
            (grids.shape[0], self.output_dim),
            device=grids.device,
            dtype=parameter_dtype,
        )
        rows = torch.nonzero(valid_mask, as_tuple=False).flatten()
        for start in range(0, rows.numel(), self.cfg.node_chunk_size):
            selected = rows[start : start + self.cfg.node_chunk_size]
            chunk = grids[selected].to(dtype=parameter_dtype)
            encoded = (
                checkpoint(self._encode, chunk, use_reentrant=False)
                if self.cfg.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
                else self._encode(chunk)
            )
            output = output.index_copy(0, selected, encoded.to(output.dtype))
        return output
