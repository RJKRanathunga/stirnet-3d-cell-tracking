from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .blocks import FeedForward
from .config import TemporalConfig
from .types import TemporalNodeMemory, TemporalState


MEMORY_ABLATIONS = {
    "full",
    "zero_node",
    "shuffle_node",
    "tracklet_only",
    "node_only",
}


class TemporalRelationBias(nn.Module):
    """Learn an attention bias from raw physical and temporal relations.

    The ten inputs are observed and projected zyx displacement (six values),
    their two Euclidean distances, signed time, and history validity. Spatial
    values are normalized by the logical sample's dref; time is normalized by
    the bounded temporal radius.
    """

    input_dim = 10

    def __init__(self, heads: int, hidden_dim: int, temporal_radius: int):
        super().__init__()
        self.temporal_radius = max(int(temporal_radius), 1)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, heads),
        )

    def forward(
        self,
        query_ref_um: Tensor,
        observed_ref_um: Tensor,
        projected_ref_um: Tensor,
        time_offset: Tensor,
        history_valid: Tensor,
        dref_um: Tensor,
    ) -> Tensor:
        # Physical calculations and the learned logit bias stay in FP32 under
        # AMP. Coordinates are relative physical zyx; no voxel resampling is
        # performed here.
        device_type = query_ref_um.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            scale = dref_um.float().clamp_min(1e-8)
            observed_delta = (
                observed_ref_um.float()[None] - query_ref_um.float()[:, None]
            ) / scale
            projected_delta = (
                projected_ref_um.float()[None] - query_ref_um.float()[:, None]
            ) / scale
            observed_distance = torch.linalg.vector_norm(
                observed_delta, dim=-1, keepdim=True
            )
            projected_distance = torch.linalg.vector_norm(
                projected_delta, dim=-1, keepdim=True
            )
            time = (
                time_offset.float()[None, :, None] / float(self.temporal_radius)
            ).expand(query_ref_um.shape[0], -1, -1)
            valid = history_valid.float()[None, :, None].expand_as(time)
            relation = torch.cat(
                [
                    observed_delta,
                    projected_delta,
                    observed_distance,
                    projected_distance,
                    time,
                    valid,
                ],
                dim=-1,
            )
            return self.net(relation.float()).float()


class TemporalMemoryAttention(nn.Module):
    """Batch-isolated attention over a flat temporal memory bank."""

    def __init__(self, cfg: TemporalConfig):
        super().__init__()
        d_model = cfg.d_model
        if d_model % cfg.memory_heads:
            raise ValueError("temporal d_model must be divisible by memory_heads")
        self.d_model = d_model
        self.heads = cfg.memory_heads
        self.head_dim = d_model // cfg.memory_heads
        self.debug_topk = cfg.memory_debug_topk
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.relation_bias = TemporalRelationBias(
            cfg.memory_heads, cfg.relation_bias_hidden, cfg.temporal_radius
        )

    def _split(self, value: Tensor) -> Tensor:
        return value.view(value.shape[0], self.heads, self.head_dim)

    @staticmethod
    def _empty_debug(query_tokens: Tensor, memory_tokens: Tensor) -> dict[str, Tensor]:
        query_count, memory_count = query_tokens.shape[0], memory_tokens.shape[0]
        return {
            "entropy": query_tokens.new_zeros((query_count,), dtype=torch.float32),
            "max_weight": query_tokens.new_zeros((query_count,), dtype=torch.float32),
            "top_indices": torch.full(
                (query_count, 0), -1, device=query_tokens.device, dtype=torch.long
            ),
            "top_weights": query_tokens.new_zeros((query_count, 0), dtype=torch.float32),
            "query_batch_index": torch.zeros(
                query_count, device=query_tokens.device, dtype=torch.long
            ),
            "memory_count": torch.tensor(memory_count, device=query_tokens.device),
        }

    def forward(
        self,
        query_tokens: Tensor,
        query_ref_um: Tensor,
        query_batch_index: Tensor,
        memory_tokens: Tensor,
        observed_ref_um: Tensor,
        projected_ref_um: Tensor,
        time_offset: Tensor,
        memory_batch_index: Tensor,
        history_valid: Tensor,
        dref_um: Tensor,
        *,
        return_debug: bool = False,
        full_attention: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        query_count = query_tokens.shape[0]
        memory_count = memory_tokens.shape[0]
        output = torch.zeros_like(query_tokens)
        if query_count == 0 or memory_count == 0:
            debug = self._empty_debug(query_tokens, memory_tokens) if return_debug else None
            if debug is not None:
                debug["query_batch_index"] = query_batch_index.detach()
                if full_attention:
                    debug["full_weights"] = query_tokens.new_zeros(
                        (query_count, memory_count), dtype=torch.float32
                    )
            return output, debug

        if query_ref_um.shape != (query_count, 3):
            raise ValueError("query_ref_um must have shape [Q,3]")
        if observed_ref_um.shape != (memory_count, 3) or projected_ref_um.shape != (
            memory_count,
            3,
        ):
            raise ValueError("temporal memory references must have shape [K,3]")

        entropy = query_tokens.new_zeros((query_count,), dtype=torch.float32)
        max_weight = query_tokens.new_zeros((query_count,), dtype=torch.float32)
        top_count = min(max(int(self.debug_topk), 0), memory_count)
        top_indices = torch.full(
            (query_count, top_count), -1, device=query_tokens.device, dtype=torch.long
        )
        top_weights = query_tokens.new_zeros(
            (query_count, top_count), dtype=torch.float32
        )
        full_weights = (
            query_tokens.new_zeros((query_count, memory_count), dtype=torch.float32)
            if return_debug and full_attention
            else None
        )

        for batch_index in torch.unique(query_batch_index).tolist():
            query_ids = torch.nonzero(
                query_batch_index == batch_index, as_tuple=False
            ).flatten()
            memory_ids = torch.nonzero(
                memory_batch_index == batch_index, as_tuple=False
            ).flatten()
            if query_ids.numel() == 0 or memory_ids.numel() == 0:
                continue
            query = self._split(self.q(query_tokens[query_ids])).permute(1, 0, 2)
            key = self._split(self.k(memory_tokens[memory_ids])).permute(1, 0, 2)
            value = self._split(self.v(memory_tokens[memory_ids])).permute(1, 0, 2)
            logits = torch.einsum("hqd,hkd->hqk", query, key).float()
            logits = logits / math.sqrt(self.head_dim)
            relation = self.relation_bias(
                query_ref_um[query_ids],
                observed_ref_um[memory_ids],
                projected_ref_um[memory_ids],
                time_offset[memory_ids],
                history_valid[memory_ids],
                dref_um[int(batch_index)],
            ).permute(2, 0, 1)
            weights = torch.softmax(logits + relation, dim=-1)
            message = torch.einsum("hqk,hkd->hqd", weights, value.float())
            message = message.permute(1, 0, 2).reshape(len(query_ids), self.d_model)
            output[query_ids] = self.out(message.to(query_tokens.dtype)).to(output.dtype)

            if return_debug:
                mean_weights = weights.mean(dim=0)
                entropy[query_ids] = -(
                    mean_weights * mean_weights.clamp_min(1e-12).log()
                ).sum(dim=-1)
                max_weight[query_ids] = mean_weights.max(dim=-1).values
                local_top_count = min(top_count, len(memory_ids))
                if local_top_count:
                    local_weights, local_indices = torch.topk(
                        mean_weights, local_top_count, dim=-1
                    )
                    top_indices[query_ids, :local_top_count] = memory_ids[local_indices]
                    top_weights[query_ids, :local_top_count] = local_weights
                if full_weights is not None:
                    full_weights[query_ids[:, None], memory_ids[None, :]] = mean_weights

        debug = None
        if return_debug:
            debug = {
                "entropy": entropy.detach(),
                "max_weight": max_weight.detach(),
                "top_indices": top_indices.detach(),
                "top_weights": top_weights.detach(),
                "query_batch_index": query_batch_index.detach(),
                "memory_count": torch.tensor(memory_count, device=query_tokens.device),
            }
            if full_weights is not None:
                debug["full_weights"] = full_weights.detach()
        return output, debug


class HierarchicalTemporalFusion(nn.Module):
    """Shared fine-node/coarse-tracklet residual memory fusion."""

    def __init__(self, cfg: TemporalConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)
        self.node_attention = TemporalMemoryAttention(cfg)
        self.tracklet_attention = TemporalMemoryAttention(cfg)
        self.node_gate = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.tracklet_gate = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.ffn_norm = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.memory_ffn_dim, 0.1)
        self.ffn_gate = nn.Parameter(torch.tensor(float(cfg.memory_gate_init_bias)))
        nn.init.constant_(self.node_gate.bias, cfg.memory_gate_init_bias)
        nn.init.constant_(self.tracklet_gate.bias, cfg.memory_gate_init_bias)

    @staticmethod
    def _shuffle_node_tokens(memory: TemporalNodeMemory) -> Tensor:
        shuffled = memory.tokens.clone()
        for batch_index in torch.unique(memory.batch_index).tolist():
            ids = torch.nonzero(
                memory.batch_index == batch_index, as_tuple=False
            ).flatten()
            if ids.numel() > 1:
                shuffled[ids] = memory.tokens[ids.roll(1)]
        return shuffled

    def forward(
        self,
        query_tokens: Tensor,
        query_ref_um: Tensor,
        query_batch_index: Tensor,
        temporal: TemporalState,
        dref_um: Tensor,
        *,
        memory_ablation: str = "full",
        return_debug: bool = False,
        full_attention: bool = False,
    ) -> tuple[Tensor, dict[str, Any] | None]:
        if memory_ablation not in MEMORY_ABLATIONS:
            raise ValueError(
                f"Unknown temporal-memory ablation {memory_ablation!r}; "
                f"expected one of {sorted(MEMORY_ABLATIONS)}"
            )
        normalized = self.norm(query_tokens)
        output = query_tokens
        diagnostics: dict[str, Any] = {}
        used_memory = False

        use_nodes = (
            self.cfg.component_memory_enabled or self.cfg.query_memory_enabled
        ) and memory_ablation not in {"tracklet_only"}
        node_memory = temporal.node_memory
        if use_nodes and node_memory is not None and not node_memory.is_empty:
            node_tokens = node_memory.tokens
            if memory_ablation == "zero_node":
                node_tokens = torch.zeros_like(node_tokens)
            elif memory_ablation == "shuffle_node":
                node_tokens = self._shuffle_node_tokens(node_memory)
            node_message, node_debug = self.node_attention(
                normalized,
                query_ref_um,
                query_batch_index,
                node_tokens,
                node_memory.observed_ref_um,
                node_memory.projected_ref_um,
                node_memory.time_offset,
                node_memory.batch_index,
                node_memory.history_valid,
                dref_um,
                return_debug=return_debug,
                full_attention=full_attention,
            )
            node_gate = torch.sigmoid(
                self.node_gate(torch.cat([normalized, node_message], dim=-1))
            )
            output = output + node_gate * node_message
            used_memory = True
            if node_debug is not None:
                diagnostics["node"] = node_debug

        use_tracklets = memory_ablation != "node_only"
        if use_tracklets and not temporal.is_empty:
            if (
                temporal.history_support_valid is not None
                and temporal.history_support_valid.shape[0] == temporal.tokens.shape[0]
            ):
                tracklet_history_valid = temporal.history_support_valid.any(dim=-1)
            else:
                tracklet_history_valid = torch.ones(
                    temporal.tokens.shape[0],
                    device=temporal.tokens.device,
                    dtype=torch.bool,
                )
            tracklet_message, tracklet_debug = self.tracklet_attention(
                normalized,
                query_ref_um,
                query_batch_index,
                temporal.tokens,
                temporal.ref_um,
                temporal.ref_um,
                temporal.ref_um.new_zeros((temporal.tokens.shape[0],)),
                temporal.batch_index,
                tracklet_history_valid,
                dref_um,
                return_debug=return_debug,
                full_attention=full_attention,
            )
            tracklet_gate = torch.sigmoid(
                self.tracklet_gate(
                    torch.cat([normalized, tracklet_message], dim=-1)
                )
            )
            output = output + tracklet_gate * tracklet_message
            used_memory = True
            if tracklet_debug is not None:
                diagnostics["tracklet"] = tracklet_debug

        if used_memory:
            output = output + torch.sigmoid(self.ffn_gate) * self.ffn(
                self.ffn_norm(output)
            )
        return output, diagnostics if return_debug else None
