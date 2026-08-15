from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .blocks import FeedForward
from .config import TemporalConfig
from .types import TemporalNodeMemory, TemporalState
from ..temporal_events import TEMPORAL_NODE_EVENT_FEATURE_DIM


MEMORY_ABLATIONS = {
    "full",
    "zero_node",
    "shuffle_node",
    "tracklet_only",
    "node_only",
}

ROUTING_ABLATIONS = {
    "full",
    "event_only",
    "competition_only",
    "baseline",
    "shuffled_event",
}

_ROUTING_ABLATION_ALIASES = {
    "no_event": "competition_only",
    "no_competition": "event_only",
}


class TemporalEventRelevance(nn.Module):
    """Score fine temporal nodes from explicit event metadata.

    Correction events are an inductive prior rather than ground truth. The
    zero-initialized learned residual can override that prior, boundary
    suppression prevents ordinary field-of-view entries/exits from being
    confused with corrections, and continuous evidence is never masked out.
    """

    def __init__(self, cfg: TemporalConfig):
        super().__init__()
        if cfg.event_feature_dim != TEMPORAL_NODE_EVENT_FEATURE_DIM:
            raise ValueError(
                "temporal event_feature_dim must match the explicit [N,8] contract"
            )
        self.feature_dim = int(cfg.event_feature_dim)
        self.logit_clip = float(cfg.event_logit_clip)
        prior_weights = torch.zeros(self.feature_dim, dtype=torch.float32)
        prior_weights[4] = float(cfg.event_start_weight)
        prior_weights[5] = float(cfg.event_end_weight)
        prior_weights[6] = float(cfg.event_division_weight)
        prior_weights[7] = float(cfg.event_boundary_weight)
        self.register_buffer("prior_weights", prior_weights, persistent=False)
        self.register_buffer(
            "prior_bias",
            torch.tensor(float(cfg.event_prior_bias), dtype=torch.float32),
            persistent=False,
        )
        self.residual = nn.Sequential(
            nn.Linear(self.feature_dim, cfg.event_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.event_hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        event_features: Tensor | None,
        *,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        node_count = reference.shape[0]
        if event_features is None:
            neutral = reference.new_zeros((node_count,), dtype=torch.float32)
            return neutral, torch.sigmoid(neutral), neutral
        if event_features.shape != (node_count, self.feature_dim):
            raise ValueError(
                "event_features must align with temporal nodes and have shape [N,8]"
            )
        if event_features.device != reference.device:
            raise ValueError(
                "event_features and temporal node memory must be on the same device"
            )
        device_type = event_features.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            features = event_features.float()
            prior = self.prior_bias + features @ self.prior_weights
            residual = self.residual(features).squeeze(-1)
            event_logit = (prior + residual).clamp(
                -self.logit_clip, self.logit_clip
            )
        return event_logit, torch.sigmoid(event_logit), residual


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

    def __init__(self, cfg: TemporalConfig, *, fine_routing: bool = False):
        super().__init__()
        d_model = cfg.d_model
        if d_model % cfg.memory_heads:
            raise ValueError("temporal d_model must be divisible by memory_heads")
        self.d_model = d_model
        self.heads = cfg.memory_heads
        self.head_dim = d_model // cfg.memory_heads
        self.debug_topk = cfg.memory_debug_topk
        self.fine_routing = bool(fine_routing)
        self.event_routing_enabled = bool(cfg.event_routing_enabled)
        self.source_competition_enabled = bool(cfg.source_competition_enabled)
        self.event_strength_max = float(cfg.event_strength_max)
        self.competition_temperature_min = float(
            cfg.competition_temperature_min
        )
        self.competition_temperature_max = float(
            cfg.competition_temperature_max
        )
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.relation_bias = TemporalRelationBias(
            cfg.memory_heads, cfg.relation_bias_hidden, cfg.temporal_radius
        )
        if self.fine_routing:
            self.event_relevance = TemporalEventRelevance(cfg)
            initial_strength_raw = math.log(math.expm1(1.0))
            self.event_strength_raw = nn.Parameter(
                torch.full(
                    (self.heads,), initial_strength_raw, dtype=torch.float32
                )
            )
            self.log_competition_temperature = nn.Parameter(
                torch.tensor(
                    math.log(float(cfg.competition_temperature_init)),
                    dtype=torch.float32,
                )
            )

    def _split(self, value: Tensor) -> Tensor:
        return value.view(value.shape[0], self.heads, self.head_dim)

    @property
    def event_strength(self) -> Tensor:
        if not self.fine_routing:
            raise AttributeError("coarse temporal attention has no event strength")
        return F.softplus(self.event_strength_raw).clamp(0.0, self.event_strength_max)

    @property
    def competition_temperature(self) -> Tensor:
        if not self.fine_routing:
            raise AttributeError("coarse temporal attention has no competition temperature")
        return self.log_competition_temperature.exp().clamp(
            self.competition_temperature_min,
            self.competition_temperature_max,
        )

    @staticmethod
    def _canonical_routing_ablation(routing_ablation: str) -> str:
        canonical = _ROUTING_ABLATION_ALIASES.get(
            routing_ablation, routing_ablation
        )
        if canonical not in ROUTING_ABLATIONS:
            raise ValueError(
                f"Unknown temporal-routing ablation {routing_ablation!r}; "
                f"expected one of {sorted(ROUTING_ABLATIONS)}"
            )
        return canonical

    def _routing_flags(self, routing_ablation: str) -> tuple[str, bool, bool]:
        canonical = self._canonical_routing_ablation(routing_ablation)
        if not self.fine_routing or canonical == "baseline":
            return canonical, False, False
        if canonical == "event_only":
            return canonical, self.event_routing_enabled, False
        if canonical == "competition_only":
            return canonical, False, self.source_competition_enabled
        return (
            canonical,
            self.event_routing_enabled,
            self.source_competition_enabled,
        )

    @staticmethod
    def _shuffle_within_batches(values: Tensor, batch_index: Tensor) -> Tensor:
        shuffled = values
        for batch_id in torch.unique(batch_index).tolist():
            ids = torch.nonzero(
                batch_index == batch_id, as_tuple=False
            ).flatten()
            if ids.numel() > 1:
                shuffled = shuffled.index_copy(0, ids, values[ids.roll(1)])
        return shuffled

    def _empty_debug(
        self,
        query_tokens: Tensor,
        memory_tokens: Tensor,
        query_batch_index: Tensor,
        competition_group_ids: Tensor | None,
        routing_ablation: str,
    ) -> dict[str, Any]:
        query_count, memory_count = query_tokens.shape[0], memory_tokens.shape[0]
        debug: dict[str, Any] = {
            "entropy": query_tokens.new_zeros((query_count,), dtype=torch.float32),
            "max_weight": query_tokens.new_zeros((query_count,), dtype=torch.float32),
            "top_indices": torch.full(
                (query_count, 0), -1, device=query_tokens.device, dtype=torch.long
            ),
            "top_weights": query_tokens.new_zeros((query_count, 0), dtype=torch.float32),
            "query_batch_index": query_batch_index.detach(),
            "memory_count": torch.tensor(memory_count, device=query_tokens.device),
        }
        if self.fine_routing:
            canonical, _, _ = self._routing_flags(routing_ablation)
            debug.update(
                {
                    "event_logit": query_tokens.new_zeros(
                        (memory_count,), dtype=torch.float32
                    ),
                    "event_probability": query_tokens.new_zeros(
                        (memory_count,), dtype=torch.float32
                    ),
                    "event_residual_logit": query_tokens.new_zeros(
                        (memory_count,), dtype=torch.float32
                    ),
                    "event_strength": self.event_strength.detach(),
                    "competition_temperature": (
                        self.competition_temperature.detach()
                    ),
                    "competition_group_size": torch.ones(
                        query_count,
                        device=query_tokens.device,
                        dtype=torch.long,
                    ),
                    "competition_group_id": (
                        competition_group_ids.detach()
                        if competition_group_ids is not None
                        else torch.full(
                            (query_count,),
                            -1,
                            device=query_tokens.device,
                            dtype=torch.long,
                        )
                    ),
                    "correction_event_mass": query_tokens.new_zeros(
                        (query_count,), dtype=torch.float32
                    ),
                    "continuous_track_mass": query_tokens.new_zeros(
                        (query_count,), dtype=torch.float32
                    ),
                    "boundary_event_mass": query_tokens.new_zeros(
                        (query_count,), dtype=torch.float32
                    ),
                    "routing_mode": canonical,
                    "routing_ablation": canonical,
                }
            )
        return debug

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
        event_features: Tensor | None = None,
        competition_group_ids: Tensor | None = None,
        routing_ablation: str = "full",
        return_debug: bool = False,
        full_attention: bool = False,
    ) -> tuple[Tensor, dict[str, Any] | None]:
        query_count = query_tokens.shape[0]
        memory_count = memory_tokens.shape[0]
        output = torch.zeros_like(query_tokens)
        canonical, use_event, use_competition = self._routing_flags(
            routing_ablation
        )
        if competition_group_ids is not None and competition_group_ids.shape != (
            query_count,
        ):
            raise ValueError("competition_group_ids must have shape [Q]")
        if event_features is not None and event_features.shape != (
            memory_count,
            TEMPORAL_NODE_EVENT_FEATURE_DIM,
        ):
            raise ValueError("event_features must have shape [K,8]")
        if query_count == 0 or memory_count == 0:
            debug = (
                self._empty_debug(
                    query_tokens,
                    memory_tokens,
                    query_batch_index,
                    competition_group_ids,
                    canonical,
                )
                if return_debug
                else None
            )
            if debug is not None:
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

        entropy = max_weight = top_indices = top_weights = None
        correction_mass = continuous_mass = boundary_mass = None
        competition_group_size = None
        top_count = min(max(int(self.debug_topk), 0), memory_count)
        if return_debug:
            entropy = query_tokens.new_zeros((query_count,), dtype=torch.float32)
            max_weight = query_tokens.new_zeros((query_count,), dtype=torch.float32)
            top_indices = torch.full(
                (query_count, top_count),
                -1,
                device=query_tokens.device,
                dtype=torch.long,
            )
            top_weights = query_tokens.new_zeros(
                (query_count, top_count), dtype=torch.float32
            )
            if self.fine_routing:
                competition_group_size = torch.ones(
                    query_count, device=query_tokens.device, dtype=torch.long
                )
                correction_mass = query_tokens.new_zeros(
                    (query_count,), dtype=torch.float32
                )
                continuous_mass = query_tokens.new_zeros(
                    (query_count,), dtype=torch.float32
                )
                boundary_mass = query_tokens.new_zeros(
                    (query_count,), dtype=torch.float32
                )
        full_weights = (
            query_tokens.new_zeros((query_count, memory_count), dtype=torch.float32)
            if return_debug and full_attention
            else None
        )

        event_logit = memory_tokens.new_zeros(
            (memory_count,), dtype=torch.float32
        )
        event_probability = torch.sigmoid(event_logit)
        event_residual_logit = event_logit
        if self.fine_routing and (use_event or return_debug):
            event_logit, event_probability, event_residual_logit = (
                self.event_relevance(event_features, reference=memory_tokens)
            )
            if canonical == "shuffled_event":
                event_logit = self._shuffle_within_batches(
                    event_logit, memory_batch_index
                )
                event_probability = self._shuffle_within_batches(
                    event_probability, memory_batch_index
                )
                event_residual_logit = self._shuffle_within_batches(
                    event_residual_logit, memory_batch_index
                )

        if event_features is None:
            correction_event = torch.zeros(
                memory_count, device=memory_tokens.device, dtype=torch.bool
            )
            boundary_event = torch.zeros_like(correction_event)
            continuous_track = torch.zeros_like(correction_event)
        else:
            correction_event = (event_features[:, 4:7] > 0.5).any(dim=-1)
            boundary_event = event_features[:, 7] > 0.5
            continuous_track = (~correction_event) & (~boundary_event)

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
            base_logits = logits + relation
            selection_logits = base_logits
            if use_event:
                selection_logits = selection_logits + (
                    self.event_strength[:, None, None]
                    * event_logit[memory_ids][None, None, :]
                )
            weights = torch.softmax(selection_logits, dim=-1)

            local_group_ids = (
                competition_group_ids[query_ids]
                if competition_group_ids is not None
                else None
            )
            if self.fine_routing and local_group_ids is not None and (
                use_competition or return_debug
            ):
                valid_groups = local_group_ids >= 0
                for group_id in torch.unique(local_group_ids[valid_groups]).tolist():
                    group = torch.nonzero(
                        valid_groups & (local_group_ids == group_id),
                        as_tuple=False,
                    ).flatten()
                    if return_debug:
                        assert competition_group_size is not None
                        competition_group_size[query_ids[group]] = int(group.numel())
                    if not use_competition or group.numel() <= 1:
                        continue
                    ownership = torch.softmax(
                        base_logits[:, group, :]
                        / self.competition_temperature,
                        dim=1,
                    )
                    competitive_weights = weights[:, group, :] * ownership
                    competitive_weights = competitive_weights / (
                        competitive_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    )
                    # The softmax result is saved for backward. Replace sibling
                    # rows functionally; never mutate those saved values in-place.
                    weights = weights.index_copy(1, group, competitive_weights)
            message = torch.einsum("hqk,hkd->hqd", weights, value.float())
            message = message.permute(1, 0, 2).reshape(len(query_ids), self.d_model)
            output = output.index_copy(
                0,
                query_ids,
                self.out(message.to(query_tokens.dtype)).to(output.dtype),
            )

            if return_debug:
                mean_weights = weights.detach().mean(dim=0)
                assert entropy is not None
                assert max_weight is not None
                assert top_indices is not None
                assert top_weights is not None
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
                if self.fine_routing:
                    assert correction_mass is not None
                    assert continuous_mass is not None
                    assert boundary_mass is not None
                    correction_mass[query_ids] = mean_weights[
                        :, correction_event[memory_ids]
                    ].sum(dim=-1)
                    continuous_mass[query_ids] = mean_weights[
                        :, continuous_track[memory_ids]
                    ].sum(dim=-1)
                    boundary_mass[query_ids] = mean_weights[
                        :, boundary_event[memory_ids]
                    ].sum(dim=-1)

        debug = None
        if return_debug:
            assert entropy is not None
            assert max_weight is not None
            assert top_indices is not None
            assert top_weights is not None
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
            if self.fine_routing:
                assert competition_group_size is not None
                assert correction_mass is not None
                assert continuous_mass is not None
                assert boundary_mass is not None
                debug.update(
                    {
                        "event_logit": event_logit.detach(),
                        "event_probability": event_probability.detach(),
                        "event_residual_logit": event_residual_logit.detach(),
                        "event_strength": self.event_strength.detach(),
                        "competition_temperature": (
                            self.competition_temperature.detach()
                        ),
                        "competition_group_size": competition_group_size.detach(),
                        "competition_group_id": (
                            competition_group_ids.detach()
                            if competition_group_ids is not None
                            else torch.full(
                                (query_count,),
                                -1,
                                device=query_tokens.device,
                                dtype=torch.long,
                            )
                        ),
                        "correction_event_mass": correction_mass.detach(),
                        "continuous_track_mass": continuous_mass.detach(),
                        "boundary_event_mass": boundary_mass.detach(),
                        "routing_mode": canonical,
                        "routing_ablation": canonical,
                    }
                )
                if full_weights is not None and competition_group_ids is not None:
                    cosine_values = []
                    unique_top_count = 0
                    for batch_id in torch.unique(query_batch_index).tolist():
                        batch_queries = torch.nonzero(
                            query_batch_index == batch_id, as_tuple=False
                        ).flatten()
                        batch_groups = competition_group_ids[batch_queries]
                        for group_id in torch.unique(
                            batch_groups[batch_groups >= 0]
                        ).tolist():
                            group = batch_queries[batch_groups == group_id]
                            active = full_weights[group].sum(dim=-1) > 0
                            group = group[active]
                            if group.numel() <= 1:
                                continue
                            normalized_weights = F.normalize(
                                full_weights[group], dim=-1, eps=1e-12
                            )
                            cosine = normalized_weights @ normalized_weights.t()
                            pair_rows, pair_columns = torch.triu_indices(
                                group.numel(),
                                group.numel(),
                                offset=1,
                                device=query_tokens.device,
                            )
                            cosine_values.append(cosine[pair_rows, pair_columns])
                            unique_top_count += int(
                                torch.unique(full_weights[group].argmax(dim=-1)).numel()
                            )
                    debug["same_source_sibling_attention_cosine"] = (
                        torch.cat(cosine_values).mean().detach()
                        if cosine_values
                        else query_tokens.new_zeros((), dtype=torch.float32)
                    )
                    debug["unique_top_temporal_node_explanations"] = torch.tensor(
                        unique_top_count,
                        device=query_tokens.device,
                        dtype=torch.long,
                    )
        return output, debug


class HierarchicalTemporalFusion(nn.Module):
    """Shared fine-node/coarse-tracklet residual memory fusion."""

    def __init__(self, cfg: TemporalConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)
        self.node_attention = TemporalMemoryAttention(cfg, fine_routing=True)
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
        routing_ablation: str = "full",
        competition_group_ids: Tensor | None = None,
        return_debug: bool = False,
        full_attention: bool = False,
    ) -> tuple[Tensor, dict[str, Any] | None]:
        if memory_ablation not in MEMORY_ABLATIONS:
            raise ValueError(
                f"Unknown temporal-memory ablation {memory_ablation!r}; "
                f"expected one of {sorted(MEMORY_ABLATIONS)}"
            )
        canonical_routing = self.node_attention._canonical_routing_ablation(
            routing_ablation
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
                event_features=node_memory.event_features,
                competition_group_ids=competition_group_ids,
                routing_ablation=canonical_routing,
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
        if return_debug:
            diagnostics["routing_mode"] = canonical_routing
            diagnostics["routing_ablation"] = canonical_routing
        return output, diagnostics if return_debug else None
