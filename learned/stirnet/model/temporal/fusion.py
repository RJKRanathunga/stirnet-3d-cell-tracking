from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import (
    InstanceConfig,
    PartitionConfig,
    RefinementConfig,
    TemporalConfig,
)
from ..types import InstanceState, RAGState, ReasoningState, TemporalState


class PhysicalLocalCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        radius_dref: float,
        dropout: float,
        reliability_floor: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.radius_dref = radius_dref
        self.reliability_floor = reliability_floor
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias = nn.Sequential(
            nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, heads)
        )
        self.salience_scale = nn.Parameter(torch.full((heads,), 0.25))
        self.reliability_scale = nn.Parameter(torch.full((heads,), 0.25))
        self.dropout = dropout

    def _split(self, x: Tensor) -> Tensor:
        return x.reshape(*x.shape[:-1], self.heads, self.head_dim)

    def forward(
        self,
        query: Tensor,
        query_ref_um: Tensor,
        query_batch: Tensor,
        temporal: TemporalState,
        dref_um: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return message, reliability-weighted support, attention entropy."""
        if query.shape[0] == 0 or temporal.is_empty:
            return (
                torch.zeros_like(query),
                query.new_zeros((query.shape[0], 1)),
                query.new_zeros((query.shape[0], 1)),
            )
        output = torch.zeros_like(query)
        support_out = query.new_zeros((query.shape[0], 1))
        entropy_out = query.new_zeros((query.shape[0], 1))
        for b in torch.unique(query_batch).tolist():
            qi = torch.nonzero(query_batch == b, as_tuple=False).flatten()
            ti = torch.nonzero(temporal.batch_index == b, as_tuple=False).flatten()
            if qi.numel() == 0 or ti.numel() == 0:
                continue
            q = self._split(self.q(query[qi])).permute(1, 0, 2)
            k = self._split(self.k(temporal.tokens[ti])).permute(1, 0, 2)
            v = self._split(self.v(temporal.tokens[ti])).permute(1, 0, 2)
            delta = temporal.ref_um[ti][None] - query_ref_um[qi][:, None]
            dist = torch.linalg.vector_norm(delta, dim=-1)
            dref = dref_um[b].float().clamp_min(1e-6)
            delta_norm = delta / dref
            rel = torch.cat([delta_norm, dist[..., None] / dref], dim=-1)
            logits = torch.einsum("hqd,hkd->hqk", q, k) / math.sqrt(self.head_dim)
            logits = logits + self.pos_bias(rel).permute(2, 0, 1)
            salience = temporal.salience[ti, 0].float().clamp_min(1e-6)
            reliability = temporal.reliability[ti, 0].float().clamp_min(
                self.reliability_floor
            )
            logits = logits + (
                self.salience_scale[:, None, None] * salience.log()[None, None]
                + self.reliability_scale[:, None, None]
                * reliability.log()[None, None]
            ).to(logits.dtype)
            allowed = dist <= self.radius_dref * dref
            # Queries with no local temporal evidence receive exactly zero
            # message rather than being forced to attend a far-away track.
            valid_rows = allowed.any(dim=-1)
            if not valid_rows.any():
                continue
            logits = logits.masked_fill(~allowed[None], -1e4)
            weights = torch.softmax(logits, dim=-1)
            weights = weights * allowed[None].to(weights.dtype)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            weights = F.dropout(weights, self.dropout, self.training)
            msg = torch.einsum("hqk,hkd->hqd", weights, v).permute(1, 0, 2).reshape(len(qi), -1)
            msg = self.out(msg)
            msg[~valid_rows] = 0
            output[qi] = msg
            mean_weight = weights.mean(0)
            reliability = temporal.reliability[ti, 0]
            support = (mean_weight * reliability[None]).sum(-1, keepdim=True)
            support[~valid_rows] = 0
            entropy = -(mean_weight.clamp_min(1e-8) * mean_weight.clamp_min(1e-8).log()).sum(-1, keepdim=True)
            entropy[~valid_rows] = 0
            support_out[qi] = support.to(support_out.dtype)
            entropy_out[qi] = entropy.to(entropy_out.dtype)
        return output, support_out, entropy_out


class InstanceTemporalReasoner(nn.Module):
    """Object-level temporal reasoning plus gated residuals on spatial RAG edges."""

    def __init__(
        self,
        temporal_cfg: TemporalConfig,
        instance_cfg: InstanceConfig,
        partition_cfg: PartitionConfig,
        refinement_cfg: RefinementConfig,
    ):
        super().__init__()
        d = temporal_cfg.d_model
        self.temporal_cfg = temporal_cfg
        self.partition_cfg = partition_cfg
        self.refinement_cfg = refinement_cfg
        self.instance_attention = PhysicalLocalCrossAttention(
            d,
            temporal_cfg.cross_heads,
            temporal_cfg.instance_match_radius_dref,
            temporal_cfg.dropout,
            temporal_cfg.reliability_floor,
        )
        self.node_to_model = nn.Linear(partition_cfg.rag_hidden_dim, d)
        self.node_attention = PhysicalLocalCrossAttention(
            d,
            temporal_cfg.cross_heads,
            temporal_cfg.instance_match_radius_dref,
            temporal_cfg.dropout,
            temporal_cfg.reliability_floor,
        )
        self.instance_gate = nn.Sequential(
            nn.Linear(2 * d + 1, d), nn.Sigmoid()
        )
        self.instance_norm = nn.LayerNorm(d)
        self.exist = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.split = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.edge_to_model = nn.Linear(partition_cfg.rag_hidden_dim, d)
        self.edge_delta = nn.Sequential(
            nn.Linear(4 * d + 1, 2 * d),
            nn.SiLU(),
            nn.Linear(2 * d, 1),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.recovery = nn.Sequential(
            nn.Linear(d + 2, d), nn.SiLU(), nn.Linear(d, 1)
        )

    def _recovery_outputs(
        self,
        temporal: TemporalState,
        instances: InstanceState,
        dref_um: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if temporal.is_empty:
            return (
                torch.zeros(0, device=instances.tokens.device, dtype=torch.long),
                instances.tokens.new_zeros((0,)),
                instances.tokens.new_zeros((0,)),
            )
        distances = temporal.tokens.new_full((temporal.tokens.shape[0],), 10.0)
        for b in torch.unique(temporal.batch_index).tolist():
            ti = torch.nonzero(temporal.batch_index == b, as_tuple=False).flatten()
            ii = torch.nonzero(instances.batch_index == b, as_tuple=False).flatten()
            if ii.numel() == 0:
                distances[ti] = 10.0
                continue
            dist = torch.cdist(temporal.ref_um[ti].float(), instances.ref_um[ii].float())
            distances[ti] = dist.min(dim=1).values / dref_um[b].float().clamp_min(1e-6)
        logits = self.recovery(
            torch.cat(
                [temporal.tokens, temporal.reliability, distances[:, None]], dim=-1
            )
        ).squeeze(-1)
        score = torch.sigmoid(logits)
        idx = torch.nonzero(
            score >= self.refinement_cfg.recovery_threshold, as_tuple=False
        ).flatten()
        return idx, logits, score

    def forward(
        self,
        instances: InstanceState,
        rag: RAGState,
        temporal: TemporalState,
        dref_um: Tensor,
    ) -> ReasoningState:
        d = self.temporal_cfg.d_model
        if instances.is_empty:
            inst_tokens = instances.tokens
            inst_msg = inst_tokens
            inst_support = inst_tokens.new_zeros((0, 1))
            inst_entropy = inst_tokens.new_zeros((0, 1))
        else:
            inst_msg, inst_support, inst_entropy = self.instance_attention(
                instances.tokens,
                instances.ref_um,
                instances.batch_index,
                temporal,
                dref_um,
            )
            gate = self.instance_gate(
                torch.cat([instances.tokens, inst_msg, inst_support], dim=-1)
            )
            inst_tokens = self.instance_norm(instances.tokens + gate * inst_msg)

        if rag.node_embeddings.shape[0]:
            node_base = self.node_to_model(rag.node_embeddings)
            node_msg, node_support, _ = self.node_attention(
                node_base,
                rag.node_centroid_um,
                rag.node_batch,
                temporal,
                dref_um,
            )
            node_ctx = node_base + node_msg
        else:
            node_ctx = rag.node_embeddings.new_zeros((0, d))
            node_support = rag.node_embeddings.new_zeros((0, 1))

        if rag.edge_index.shape[1]:
            src, dst = rag.edge_index
            edge_emb = self.edge_to_model(rag.edge_embeddings)
            spatial_logit = rag.spatial_edge_logits[:, None]
            raw_delta = self.edge_delta(
                torch.cat(
                    [
                        edge_emb,
                        node_ctx[src],
                        node_ctx[dst],
                        (node_ctx[src] - node_ctx[dst]).abs(),
                        spatial_logit,
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            # Predict a bounded temporal CANDIDATE logit and interpolate from
            # the spatial decision toward it. Unlike a bounded additive
            # residual, reliable temporal evidence can therefore overturn an
            # arbitrarily confident wrong spatial edge. With zero temporal
            # support the gate below is exactly zero and spatial logits are
            # preserved exactly.
            temporal_candidate = (
                self.temporal_cfg.temporal_residual_scale
                * torch.tanh(raw_delta)
            )
            delta = temporal_candidate - rag.spatial_edge_logits
            uncertainty = torch.exp(-rag.spatial_edge_logits.abs())
            s_left = node_support[src, 0]
            s_right = node_support[dst, 0]
            same_provisional = (
                instances.node_to_instance[src] == instances.node_to_instance[dst]
            ).float()
            gate = self.edge_gate(
                torch.stack(
                    [uncertainty, s_left, s_right, same_provisional], dim=-1
                )
            ).squeeze(-1)
            # No temporal support => no temporal write, by construction.
            gate = gate * torch.maximum(s_left, s_right).clamp(0, 1)
            final_logits = rag.spatial_edge_logits + gate * delta
        else:
            delta = rag.spatial_edge_logits.new_zeros((0,))
            gate = rag.spatial_edge_logits.new_zeros((0,))
            final_logits = rag.spatial_edge_logits

        recovery_idx, recovery_logits, recovery_scores = self._recovery_outputs(
            temporal, instances, dref_um
        )
        base_quality = instances.quality_logits
        exist_logits = (
            base_quality + self.exist(inst_tokens).squeeze(-1)
            if inst_tokens.shape[0]
            else base_quality
        )
        split_logits = (
            self.split(inst_tokens).squeeze(-1)
            if inst_tokens.shape[0]
            else inst_tokens.new_zeros((0,))
        )
        return ReasoningState(
            instance_tokens=inst_tokens,
            instance_exist_logits=exist_logits,
            split_logits=split_logits,
            temporal_support=inst_support,
            temporal_attention_entropy=inst_entropy,
            edge_temporal_delta=delta,
            edge_temporal_gate=gate,
            final_edge_logits=final_logits,
            recovery_track_indices=recovery_idx,
            recovery_logits=recovery_logits,
            recovery_scores=recovery_scores,
        )
