from __future__ import annotations

import torch
from torch import Tensor, nn

from .blocks import _groups
from .config import HistoryConfig
from torch.utils.checkpoint import checkpoint


class HistoricalInstanceEncoder(nn.Module):
    """Encode bounded per-detection ``[4,G,G,G]`` history descriptors.

    Invalid rows are never evaluated by the CNN and are explicitly written as
    zeros, preventing missing history from becoming a morphology shortcut.
    """

    def __init__(
        self,
        cfg: HistoryConfig,
        d_model: int = 128,
        *,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.activation_checkpointing = activation_checkpointing
        self.features = nn.Sequential(
            nn.Conv3d(cfg.input_channels, 16, 3, padding=1),
            nn.GroupNorm(_groups(16), 16),
            nn.SiLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(_groups(32), 32),
            nn.SiLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(_groups(64), 64),
            nn.SiLU(),
        )
        self.projection = nn.Identity() if d_model == 128 else nn.Linear(128, d_model)
        if cfg.node_chunk_size <= 0:
            raise ValueError("history.node_chunk_size must be positive")

    def _encode_chunk(self, grid: Tensor) -> Tensor:
        feature = self.features(grid)
        pooled = torch.cat(
            [feature.mean(dim=(-3, -2, -1)), feature.amax(dim=(-3, -2, -1))],
            dim=-1,
        )
        return self.projection(pooled)

    def forward(self, grid: Tensor, valid: Tensor, *, chunk_size: int | None = None) -> Tensor:
        if grid.ndim != 5:
            raise ValueError(f"node_instance_grid must be [N,C,G,G,G], got {tuple(grid.shape)}")
        if grid.shape[1] != self.cfg.input_channels:
            raise ValueError(
                f"history input has {grid.shape[1]} channels; expected {self.cfg.input_channels}"
            )
        valid = valid.to(device=grid.device, dtype=torch.bool)
        if valid.shape != (grid.shape[0],):
            raise ValueError("node_history_valid must have shape [N]")
        compute_dtype = self.features[0].weight.dtype
        output = torch.zeros(
            (grid.shape[0], self.d_model), device=grid.device, dtype=compute_dtype
        )
        ids = torch.nonzero(valid, as_tuple=False).flatten()
        step = int(chunk_size or self.cfg.node_chunk_size)
        for start in range(0, ids.numel(), step):
            chunk_ids = ids[start : start + step]
            chunk=grid[chunk_ids].to(compute_dtype)
            encoded=(
                checkpoint(self._encode_chunk,chunk,use_reentrant=False)
                if self.activation_checkpointing and self.training and torch.is_grad_enabled()
                else self._encode_chunk(chunk)
            )
            output = output.index_copy(0, chunk_ids, encoded.to(output.dtype))
        return output


class HistoryFusion(nn.Module):
    """Conservative validity-aware residual fusion before detection GNN layers."""

    def __init__(self, d_model: int, gate_init_bias: float = -2.0) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model + 1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_init_bias)

    def forward(
        self, scalar_embedding: Tensor, history_embedding: Tensor, history_valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        valid = history_valid.to(scalar_embedding.dtype)[:, None]
        safe_history = history_embedding * valid
        gate = torch.sigmoid(
            self.gate(torch.cat([scalar_embedding, safe_history, valid], dim=-1))
        ) * valid
        return scalar_embedding + gate * safe_history, gate
