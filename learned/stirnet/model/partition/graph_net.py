# STIRNET_MORPHOLOGY_AWARE_RAG_V1
from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from ..config import PartitionConfig
from ..types import RAGState
from .separator_barrier import SeparatorAwareBarrier


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
        self,
        nodes: Tensor,
        edge_index: Tensor,
        edge_raw: Tensor,
        edge_residual: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if edge_index.shape[1] == 0:
            return nodes, edge_raw.new_zeros((0, nodes.shape[-1]))
        src, dst = edge_index
        edge_emb = self.edge_mlp(
            torch.cat([nodes[src], nodes[dst], edge_raw], dim=-1)
        )
        if edge_residual is not None:
            edge_emb = edge_emb + edge_residual
        agg = nodes.new_zeros(nodes.shape)
        degree = nodes.new_zeros((nodes.shape[0], 1))
        agg.index_add_(0, src, edge_emb)
        agg.index_add_(0, dst, edge_emb)
        ones = nodes.new_ones((edge_emb.shape[0], 1))
        degree.index_add_(0, src, ones)
        degree.index_add_(0, dst, ones)
        agg = agg / degree.clamp_min(1)
        update = self.node_mlp(
            torch.cat([self.node_norm(nodes), agg], dim=-1)
        )
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

        if cfg.rag_separator_barrier_enabled:
            morphology_dim = (
                cfg.rag_edge_morphology_dim
                if cfg.rag_separator_barrier_use_morphology
                else 0
            )
            self.separator_barrier = SeparatorAwareBarrier(
                morphology_dim=morphology_dim,
                hidden_dim=cfg.rag_separator_barrier_hidden_dim,
                max_barrier_logit=cfg.rag_separator_barrier_max_logit,
                initial_gate_bias=cfg.rag_separator_barrier_initial_gate_bias,
                score_scale=cfg.rag_separator_barrier_score_scale,
            )
        else:
            self.separator_barrier = None

        if cfg.rag_morphology_enabled:
            self.node_morphology_projection = nn.Linear(
                cfg.rag_node_morphology_dim, h, bias=False
            )
            self.edge_morphology_projection = nn.Linear(
                cfg.rag_edge_morphology_dim, h, bias=False
            )
            # Exact legacy behavior at transfer initialization.
            nn.init.zeros_(self.node_morphology_projection.weight)
            nn.init.zeros_(self.edge_morphology_projection.weight)
        else:
            self.node_morphology_projection = None
            self.edge_morphology_projection = None

    def _morphology_residuals(
        self, rag: RAGState
    ) -> tuple[Tensor | None, Tensor | None]:
        if not self.cfg.rag_morphology_enabled:
            return None, None
        if (
            rag.node_morphology_embeddings is None
            or rag.edge_morphology_embeddings is None
        ):
            raise ValueError(
                "Morphology-aware RAG is enabled but RAGState does not contain "
                "node/edge morphology embeddings"
            )
        if rag.node_morphology_embeddings.shape[0] != rag.node_features.shape[0]:
            raise ValueError("Node morphology rows must align with RAG nodes")
        if rag.edge_morphology_embeddings.shape[0] != rag.edge_features.shape[0]:
            raise ValueError("Edge morphology rows must align with RAG edges")
        return (
            self.node_morphology_projection(rag.node_morphology_embeddings),
            self.edge_morphology_projection(rag.edge_morphology_embeddings),
        )

    def forward(self, rag: RAGState) -> RAGState:
        nodes = self.node_encoder(rag.node_features)
        node_residual, edge_residual = self._morphology_residuals(rag)
        if node_residual is not None:
            nodes = nodes + node_residual

        edge_emb = rag.edge_features.new_zeros(
            (rag.edge_features.shape[0], self.cfg.rag_hidden_dim)
        )
        for block in self.blocks:
            nodes, edge_emb = block(
                nodes,
                rag.edge_index,
                rag.edge_features,
                edge_residual=edge_residual,
            )
        barrier_score: Tensor | None = None
        barrier_correction: Tensor | None = None
        if rag.edge_index.shape[1]:
            src, dst = rag.edge_index
            edge_emb = self.final_edge(
                torch.cat([nodes[src], nodes[dst], rag.edge_features], dim=-1)
            )
            if edge_residual is not None:
                edge_emb = edge_emb + edge_residual
            base_logits = self.classifier(edge_emb).squeeze(-1)
            if self.separator_barrier is not None:
                if rag.separator_barrier_features is None:
                    raise ValueError("separator barrier enabled without features")
                morphology = (
                    rag.edge_morphology_embeddings
                    if self.cfg.rag_separator_barrier_use_morphology
                    else None
                )
                barrier_score, barrier_correction = self.separator_barrier(
                    rag.separator_barrier_features,
                    morphology,
                )
                logits = base_logits - barrier_correction
            else:
                logits = base_logits
        else:
            base_logits = rag.node_features.new_zeros((0,))
            logits = base_logits
            if self.separator_barrier is not None:
                barrier_score = base_logits
                barrier_correction = base_logits
        return replace(
            rag,
            node_embeddings=nodes,
            edge_embeddings=edge_emb,
            spatial_edge_logits=logits,
            base_spatial_edge_logits=base_logits,
            separator_barrier_score=barrier_score,
            separator_barrier_correction=barrier_correction,
        )
