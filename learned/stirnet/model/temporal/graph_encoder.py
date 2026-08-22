from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import TemporalConfig
from ..types import TemporalInput, TemporalState


class DetectionMessageBlock(nn.Module):
    def __init__(self, d_model: int, edge_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * d_model + edge_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_index.shape[1] == 0:
            return x
        src, dst = edge_index
        msg = self.message(torch.cat([x[src], x[dst], edge_attr], dim=-1))
        agg = x.new_zeros(x.shape)
        count = x.new_zeros((x.shape[0], 1))
        agg.index_add_(0, dst, msg)
        count.index_add_(0, dst, x.new_ones((len(dst), 1)))
        # Detection preprocessing is not required to duplicate accepted edges.
        agg.index_add_(0, src, msg)
        count.index_add_(0, src, x.new_ones((len(src), 1)))
        agg = agg / count.clamp_min(1)
        return x + self.update(torch.cat([x, agg], dim=-1))


class HypothesisMessageBlock(nn.Module):
    """Directed message passing over pooled tracklet hypotheses.

    graph_builder emits both directions for every hypothesis pair and reverses
    directional attributes for the reverse row. Therefore this block aggregates
    source -> destination only.
    """

    def __init__(self, d_model: int, edge_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * d_model + edge_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_index.shape[1] == 0:
            return x
        src, dst = edge_index
        msg = self.message(torch.cat([x[src], x[dst], edge_attr], dim=-1))
        agg = x.new_zeros(x.shape)
        count = x.new_zeros((x.shape[0], 1))
        agg.index_add_(0, dst, msg)
        count.index_add_(0, dst, x.new_ones((len(dst), 1)))
        agg = agg / count.clamp_min(1)
        return x + self.update(torch.cat([x, agg], dim=-1))


class TemporalGraphEncoder(nn.Module):
    """Encode detections, pool tracklets, then reason over tracklet pairs."""

    def __init__(self, cfg: TemporalConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.node_proj = nn.Sequential(
            nn.Linear(cfg.node_dim, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.history_gate = nn.Linear(2 * d, d)
        nn.init.constant_(self.history_gate.bias, -1.5)
        self.blocks = nn.ModuleList(
            [
                DetectionMessageBlock(
                    d, cfg.edge_dim, cfg.graph_hidden_dim, cfg.dropout
                )
                for _ in range(cfg.graph_layers)
            ]
        )
        self.status_proj = nn.Sequential(
            nn.Linear(cfg.status_dim, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.pool = nn.Sequential(
            nn.Linear(3 * d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)
        )
        self.hypothesis_blocks = nn.ModuleList(
            [
                HypothesisMessageBlock(
                    d,
                    cfg.hypothesis_edge_dim,
                    cfg.graph_hidden_dim,
                    cfg.dropout,
                )
                for _ in range(cfg.hypothesis_layers)
            ]
        )
        self.salience = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.reliability = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def empty(self, device: torch.device, dtype: torch.dtype) -> TemporalState:
        d = self.cfg.d_model
        return TemporalState(
            tokens=torch.zeros((0, d), device=device, dtype=dtype),
            ref_um=torch.zeros((0, 3), device=device, dtype=torch.float32),
            batch_index=torch.zeros((0,), device=device, dtype=torch.long),
            salience=torch.zeros((0, 1), device=device, dtype=dtype),
            reliability=torch.zeros((0, 1), device=device, dtype=dtype),
            status=torch.zeros(
                (0, self.cfg.status_dim), device=device, dtype=dtype
            ),
            node_tokens=torch.zeros((0, d), device=device, dtype=dtype),
        )

    def _hypothesis_graph(
        self,
        data: TemporalInput,
        *,
        tracklet_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        edge_index = data.hypothesis_edge_index
        edge_attr = data.hypothesis_edge_attr
        if edge_index is None and edge_attr is None:
            return (
                torch.zeros((2, 0), device=device, dtype=torch.long),
                torch.zeros(
                    (0, self.cfg.hypothesis_edge_dim),
                    device=device,
                    dtype=dtype,
                ),
            )
        if edge_index is None or edge_attr is None:
            raise ValueError(
                "hypothesis_edge_index and hypothesis_edge_attr must be "
                "provided together"
            )
        edge_index = edge_index.to(device=device, dtype=torch.long)
        edge_attr = edge_attr.to(device=device, dtype=dtype)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("hypothesis_edge_index must have shape [2,H]")
        if edge_attr.ndim != 2 or edge_attr.shape != (
            edge_index.shape[1],
            self.cfg.hypothesis_edge_dim,
        ):
            raise ValueError(
                "hypothesis_edge_attr must have shape "
                f"[H,{self.cfg.hypothesis_edge_dim}]"
            )
        if edge_index.numel():
            if int(edge_index.min().item()) < 0:
                raise ValueError("hypothesis_edge_index cannot be negative")
            if int(edge_index.max().item()) >= tracklet_count:
                raise ValueError(
                    "hypothesis_edge_index references a tracklet outside "
                    "temporal_ref_um/temporal_status"
                )
        return edge_index, edge_attr

    def forward(self, data: TemporalInput | None) -> TemporalState:
        if data is None:
            device = next(self.parameters()).device
            return self.empty(device, next(self.parameters()).dtype)
        if data.graph_x.shape[-1] != self.cfg.node_dim:
            raise ValueError(
                f"Temporal graph_x must have width {self.cfg.node_dim}; "
                f"got {data.graph_x.shape[-1]}"
            )
        if data.graph_edge_attr.shape[-1] != self.cfg.edge_dim:
            raise ValueError(
                f"Temporal graph_edge_attr must have width {self.cfg.edge_dim}; "
                f"got {data.graph_edge_attr.shape[-1]}"
            )
        m = data.temporal_ref_um.shape[0]
        if m == 0:
            return self.empty(data.graph_x.device, data.graph_x.dtype)

        x = self.node_proj(data.graph_x)
        if data.node_history_embedding is not None:
            history = data.node_history_embedding.to(x.dtype)
            if history.shape != x.shape:
                raise ValueError("node_history_embedding must have shape [N,d_model]")
            gate = torch.sigmoid(
                self.history_gate(torch.cat([x, history], dim=-1))
            )
            x = x + gate * history
        for block in self.blocks:
            x = block(x, data.graph_edge_index, data.graph_edge_attr)

        if data.tracklet_id.shape[0] != x.shape[0]:
            raise ValueError("tracklet_id must align one-to-one with graph nodes")
        if data.tracklet_id.numel():
            if int(data.tracklet_id.min().item()) < 0:
                raise ValueError("tracklet_id cannot be negative")
            if int(data.tracklet_id.max().item()) >= m:
                raise ValueError(
                    "tracklet_id references a row outside temporal_ref_um"
                )

        sums = x.new_zeros((m, x.shape[-1]))
        counts = x.new_zeros((m, 1))
        sums.index_add_(0, data.tracklet_id, x)
        counts.index_add_(
            0, data.tracklet_id, x.new_ones((x.shape[0], 1))
        )
        mean = sums / counts.clamp_min(1)
        maxima = x.new_full((m, x.shape[-1]), -torch.inf)
        if x.shape[0]:
            maxima.scatter_reduce_(
                0,
                data.tracklet_id[:, None].expand(-1, x.shape[-1]),
                x,
                reduce="amax",
                include_self=True,
            )
        maxima = torch.where(
            torch.isfinite(maxima), maxima, torch.zeros_like(maxima)
        )
        status = data.temporal_status.to(x.dtype)
        if status.shape != (m, self.cfg.status_dim):
            raise ValueError(
                f"temporal_status must have shape [M,{self.cfg.status_dim}]"
            )
        status_emb = self.status_proj(status)
        tokens = self.pool(torch.cat([mean, maxima, status_emb], dim=-1))

        hypothesis_edge_index, hypothesis_edge_attr = self._hypothesis_graph(
            data,
            tracklet_count=m,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        for block in self.hypothesis_blocks:
            tokens = block(
                tokens, hypothesis_edge_index, hypothesis_edge_attr
            )

        return TemporalState(
            tokens=tokens,
            ref_um=data.temporal_ref_um.float(),
            batch_index=data.temporal_batch.long(),
            salience=torch.sigmoid(self.salience(tokens)),
            reliability=torch.sigmoid(self.reliability(tokens)),
            status=status,
            node_tokens=x,
        )
