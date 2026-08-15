from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import PartitionConfig
from ..types import RAGState


class RAGMessageBlock(nn.Module):
    def __init__(self, hidden: int, edge_raw_dim: int, dropout: float = 0.05):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + edge_raw_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.node_norm = nn.LayerNorm(hidden)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
        )

    def forward(
        self, nodes: Tensor, edge_index: Tensor, edge_raw: Tensor
    ) -> tuple[Tensor, Tensor]:
        if edge_index.shape[1] == 0:
            return nodes, edge_raw.new_zeros((0, nodes.shape[-1]))
        src, dst = edge_index
        edge_emb = self.edge_mlp(torch.cat([nodes[src], nodes[dst], edge_raw], dim=-1))
        agg = nodes.new_zeros(nodes.shape)
        degree = nodes.new_zeros((nodes.shape[0], 1))
        agg.index_add_(0, src, edge_emb)
        agg.index_add_(0, dst, edge_emb)
        ones = nodes.new_ones((edge_emb.shape[0], 1))
        degree.index_add_(0, src, ones)
        degree.index_add_(0, dst, ones)
        agg = agg / degree.clamp_min(1)
        update = self.node_mlp(torch.cat([self.node_norm(nodes), agg], dim=-1))
        return nodes + update, edge_emb


class SpatialRAGNetwork(nn.Module):
    """Learned adjacency reasoning before any instance token is created."""

    def __init__(self, cfg: PartitionConfig, node_in_dim: int, edge_in_dim: int):
        super().__init__()
        self.cfg = cfg
        h = cfg.rag_hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, h), nn.SiLU(), nn.Linear(h, h)
        )
        self.blocks = nn.ModuleList(
            [RAGMessageBlock(h, edge_in_dim) for _ in range(cfg.rag_layers)]
        )
        self.final_edge = nn.Sequential(
            nn.Linear(2 * h + edge_in_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.classifier = nn.Linear(h, 1)

    def forward(self, rag: RAGState) -> RAGState:
        nodes = self.node_encoder(rag.node_features)
        edge_emb = rag.edge_features.new_zeros(
            (rag.edge_features.shape[0], self.cfg.rag_hidden_dim)
        )
        for block in self.blocks:
            nodes, edge_emb = block(nodes, rag.edge_index, rag.edge_features)
        if rag.edge_index.shape[1]:
            src, dst = rag.edge_index
            edge_emb = self.final_edge(
                torch.cat([nodes[src], nodes[dst], rag.edge_features], dim=-1)
            )
            logits = self.classifier(edge_emb).squeeze(-1)
        else:
            logits = rag.node_features.new_zeros((0,))
        return replace(
            rag,
            node_embeddings=nodes,
            edge_embeddings=edge_emb,
            spatial_edge_logits=logits,
        )
