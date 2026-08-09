from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import TemporalConfig
from .graph_encoder import segment_softmax
from .types import TemporalState


class TrackletPooler(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, d_model))
        self.score = nn.Sequential(nn.Linear(d_model, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, node_embeddings: Tensor, tracklet_id: Tensor, node_time: Tensor, n_tracklets: int | None = None) -> Tensor:
        if node_embeddings.shape[0] == 0:
            n = int(n_tracklets or 0)
            return node_embeddings.new_zeros((n, node_embeddings.shape[-1]))
        if n_tracklets is None:
            n_tracklets = int(tracklet_id.max().item()) + 1
        h = node_embeddings + self.time_embed(node_time[:, None])
        raw = self.score(h).squeeze(-1)[:, None]
        alpha = segment_softmax(raw, tracklet_id, n_tracklets).squeeze(-1)
        out = node_embeddings.new_zeros((n_tracklets, node_embeddings.shape[-1]))
        out.index_add_(0, tracklet_id, node_embeddings * alpha[:, None])
        return out


class TemporalStateBuilder(nn.Module):
    def __init__(self, cfg: TemporalConfig):
        super().__init__()
        self.cfg = cfg
        self.status_proj = nn.Sequential(
            nn.Linear(cfg.status_dim, 64), nn.SiLU(), nn.Linear(64, cfg.d_model)
        )
        self.salience = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.status_dim, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        self.status_salience_prior = nn.Parameter(torch.tensor(
            [0.0, 0.6, 0.6, 0.8, 0.2, -0.3, -0.2, -0.2, 0.4, 0.3], dtype=torch.float32
        ))
        self.reliability = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.status_dim, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        # Mild prior: interior start/end/gap status can push salience upward through learning;
        # no hard-coded output values are imposed.

    def forward(
        self,
        pooled_tokens: Tensor,
        ref_um: Tensor,
        status: Tensor,
        hyp_edge_index: Tensor,
        hyp_edge_attr: Tensor,
        batch_index: Tensor,
        dref_um_per_hyp: Tensor,
    ) -> TemporalState:
        if pooled_tokens.shape[0] == 0:
            z1 = pooled_tokens.new_zeros((0, 1))
            return TemporalState(
                tokens=pooled_tokens,
                ref_um=ref_um,
                ref_cellscale=ref_um,
                salience=z1,
                reliability=z1,
                status=status,
                edge_index=hyp_edge_index,
                edge_attr=hyp_edge_attr,
                batch_index=batch_index,
            )
        tokens = pooled_tokens + self.status_proj(status)
        combined = torch.cat([tokens, status], dim=-1)
        salience_logits = self.salience(combined) + (status * self.status_salience_prior[None]).sum(dim=-1, keepdim=True)
        salience = torch.sigmoid(salience_logits)
        reliability = torch.sigmoid(self.reliability(combined))
        ref_cellscale = ref_um / dref_um_per_hyp[:, None].clamp_min(1e-8)
        return TemporalState(
            tokens=tokens,
            ref_um=ref_um,
            ref_cellscale=ref_cellscale,
            salience=salience,
            reliability=reliability,
            status=status,
            edge_index=hyp_edge_index,
            edge_attr=hyp_edge_attr,
            batch_index=batch_index,
        )
