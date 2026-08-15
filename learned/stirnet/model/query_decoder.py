from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .attention import PhysicalPositionBias
from .blocks import FeedForward
from .config import DecoderConfig, ProposalConfig, QueryConfig, TemporalConfig
from .coordinates import feature_grid_coordinates_um, resize_label_map_nearest
from .heads import CenterHead, ExistenceHead, MaskEmbeddingHead, dot_mask_logits
from .query_builder import (
    QUERY_DISCOVERY,
    QUERY_PRIMARY,
    QUERY_SPATIAL_PROPOSAL,
    QUERY_SPLIT,
    QUERY_TEMPORAL,
    build_competition_group_ids,
)
from .temporal_memory import HierarchicalTemporalFusion
from .types import QueryState, TemporalState


def _cap_feature_tokens(feature: Tensor, spacing_um: Tensor, max_tokens: int) -> tuple[Tensor, Tensor]:
    z, y, x = feature.shape[-3:]
    n = z * y * x
    if n <= max_tokens:
        return feature, spacing_um
    scale = (n / max_tokens) ** (1.0 / 3.0)
    target = tuple(max(1, int(round(v / scale))) for v in (z, y, x))
    while target[0] * target[1] * target[2] > max_tokens:
        k = max(range(3), key=lambda i: target[i])
        target = tuple(v - 1 if i == k and v > 1 else v for i, v in enumerate(target))
    pooled = F.adaptive_avg_pool3d(feature, target)
    ratio = torch.tensor(
        [
            (size - 1) / (pooled_size - 1) if pooled_size > 1 else 1.0
            for size, pooled_size in zip((z, y, x), target)
        ],
        device=feature.device,
        dtype=spacing_um.dtype,
    )
    return pooled, spacing_um * ratio[None]


class QueryCrossAttention(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        d, h = cfg.d_model, cfg.heads
        self.d_model, self.heads, self.head_dim = d, h, d // h
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.pos_bias = PhysicalPositionBias(h, 32)
        self.dropout = cfg.dropout

    def _split(self, x: Tensor) -> Tensor:
        return x.view(*x.shape[:-1], self.heads, self.head_dim)

    def forward(
        self,
        q_in: Tensor,
        spatial: Tensor,
        spatial_pos_um: Tensor,
        refs_um: Tensor,
        support: Tensor,
        query_padding_mask: Tensor,
        dref_um: Tensor,
    ) -> Tensor:
        B, Q, _ = q_in.shape
        out = torch.zeros_like(q_in)
        for b in range(B):
            valid_q = ~query_padding_mask[b]
            if not valid_q.any():
                continue
            q = self._split(self.q(q_in[b, valid_q])).permute(1,0,2)
            k = self._split(self.k(spatial[b])).permute(1,0,2)
            v = self._split(self.v(spatial[b])).permute(1,0,2)
            logits = torch.einsum("hqd,hnd->hqn", q, k) / math.sqrt(self.head_dim)
            delta = spatial_pos_um[b][None] - refs_um[b, valid_q][:, None]
            logits = logits + self.pos_bias(delta, dref_um[b]).permute(2,0,1)
            sup = support[b, valid_q].clone()
            # Ensure every query has at least one key.
            empty = ~sup.any(dim=-1)
            if empty.any():
                nearest = torch.linalg.vector_norm(delta[empty], dim=-1).argmin(dim=-1)
                rows = torch.nonzero(empty, as_tuple=False).flatten()
                sup[rows, nearest] = True
            logits = logits.masked_fill(~sup[None], -1e4)
            weights = torch.softmax(logits, dim=-1)
            weights = F.dropout(weights, self.dropout, self.training)
            msg = torch.einsum("hqn,hnd->hqd", weights, v).permute(1,0,2).reshape(int(valid_q.sum()), self.d_model)
            out[b, valid_q] = self.out(msg).to(out.dtype)
        return out


class QueryDecoderLayer(nn.Module):
    def __init__(
        self, cfg: DecoderConfig, temporal_cfg: TemporalConfig | None = None
    ):
        super().__init__()
        self.self_norm = nn.LayerNorm(cfg.d_model)
        self.self_attn = nn.MultiheadAttention(cfg.d_model, cfg.heads, cfg.dropout, batch_first=True)
        temporal_cfg = temporal_cfg or TemporalConfig(d_model=cfg.d_model)
        self.temporal_fusion = HierarchicalTemporalFusion(temporal_cfg)
        self.query_memory_enabled = temporal_cfg.query_memory_enabled
        self.last_temporal_debug: dict | None = None
        self.cross_norm = nn.LayerNorm(cfg.d_model)
        self.cross_attn = QueryCrossAttention(cfg)
        self.ffn_norm = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.ffn_dim, cfg.dropout)
        self.exist = ExistenceHead(cfg.d_model)
        self.center = CenterHead(cfg.d_model)
        self.mask_embed = MaskEmbeddingHead(cfg.d_model, cfg.mask_dim)

        self.center_step_by_type = (
            cfg.primary_center_step_dref,
            cfg.split_center_step_dref,
            cfg.temporal_center_step_dref,
            cfg.discovery_center_step_dref,
            (
                cfg.proposal_center_max_offset_dref
                if cfg.proposal_center_step_dref is None
                else cfg.proposal_center_step_dref
            ),
        )
        self.proposal_center_max_offset_dref = float(
            self.center_step_by_type[QUERY_SPATIAL_PROPOSAL]
        )

    def _bounded_center_delta(
            self,
            raw_delta: Tensor,
            q: QueryState,
    ) -> Tensor:
        limits = raw_delta.new_tensor(self.center_step_by_type)
        safe_types = q.query_types.clamp(
            0,
            len(self.center_step_by_type) - 1,
        )
        max_step = limits[safe_types]

        bounded = torch.tanh(raw_delta)
        delta = bounded * max_step[..., None]

        proposal = q.query_types == QUERY_SPATIAL_PROPOSAL

        # Normalize proposal displacement by Euclidean norm in FP32 for
        # numerical stability under autocast, then cast back.
        direction_fp32 = bounded.float()
        norm_fp32 = torch.linalg.vector_norm(
            direction_fp32,
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)

        proposal_delta = (
                direction_fp32
                / norm_fp32
                * max_step[..., None].float()
        ).to(dtype=delta.dtype)

        delta = torch.where(
            proposal[..., None],
            proposal_delta,
            delta,
        )

        return delta.masked_fill(
            q.padding_mask[..., None],
            0,
        )

    def forward(
        self,
        q: QueryState,
        spatial_tokens: Tensor,
        spatial_pos_um: Tensor,
        support: Tensor,
        dref_um: Tensor,
        spatial_mask_features: Tensor,
        temporal: TemporalState | None = None,
        *,
        memory_ablation: str = "full",
        routing_ablation: str = "full",
        return_debug: bool = False,
        full_attention: bool = False,
    ):
        x = q.embeddings
        n = self.self_norm(x)
        a, _ = self.self_attn(n, n, n, key_padding_mask=q.padding_mask, need_weights=False)
        x = x + a
        self.last_temporal_debug = None
        if self.query_memory_enabled and temporal is not None:
            valid = ~q.padding_mask
            flat_batch = torch.arange(
                x.shape[0], device=x.device, dtype=torch.long
            )[:, None].expand_as(valid)
            competition_group_ids = (
                q.competition_group_ids
                if q.competition_group_ids is not None
                else build_competition_group_ids(
                    q.query_types, q.source_instance_ids
                )
            )
            temporal_message, self.last_temporal_debug = self.temporal_fusion(
                x[valid],
                (q.references_cellscale * dref_um[:, None, None])[valid],
                flat_batch[valid],
                temporal,
                dref_um,
                memory_ablation=memory_ablation,
                routing_ablation=routing_ablation,
                competition_group_ids=competition_group_ids[valid],
                return_debug=return_debug,
                full_attention=full_attention,
            )
            if self.last_temporal_debug is not None:
                query_slots = torch.arange(
                    x.shape[1], device=x.device, dtype=torch.long
                )[None].expand_as(valid)
                self.last_temporal_debug["query_slot_index"] = query_slots[valid].detach()
                self.last_temporal_debug["query_type"] = q.query_types[valid].detach()
                self.last_temporal_debug["source_instance_id"] = (
                    q.source_instance_ids[valid].detach()
                )
                self.last_temporal_debug["competition_group_id"] = (
                    competition_group_ids[valid].detach()
                )
            updated = x.clone()
            updated[valid] = temporal_message
            x = updated
        c = self.cross_attn(self.cross_norm(x), spatial_tokens, spatial_pos_um,
                            q.references_cellscale * dref_um[:, None, None], support, q.padding_mask, dref_um)
        x = x + c
        x = x + self.ffn(self.ffn_norm(x))
        x = x.masked_fill(q.padding_mask[..., None], 0)
        reference_before_update = q.references_cellscale
        raw_delta = self.center(x)
        delta = self._bounded_center_delta(raw_delta, q)
        proposal = q.query_types == QUERY_SPATIAL_PROPOSAL
        initial_references = (
            q.initial_references_cellscale
            if q.initial_references_cellscale is not None
            else reference_before_update
        )
        refinement_origin = torch.where(
            proposal[..., None], initial_references, reference_before_update
        )
        refs = refinement_origin + delta
        emb = self.mask_embed(x)
        masks = dot_mask_logits(emb, spatial_mask_features)
        masks = masks.masked_fill(q.padding_mask[..., None, None, None], -20.0)
        return replace(q, embeddings=x, references_cellscale=refs), {
            "exist_logits": self.exist(x).masked_fill(q.padding_mask, -20.0),
            "centers_cellscale": refs,
            "coarse_mask_logits": masks,
            "query_embeddings": x,
            "center_delta_cellscale": delta,
            "reference_before_update_cellscale": reference_before_update,
        }


class InstanceQueryDecoder(nn.Module):
    def __init__(
        self,
        input_channels: tuple[int,int,int],
        cfg: DecoderConfig,
        query_cfg: QueryConfig,
        temporal_cfg: TemporalConfig | None = None,
        proposal_cfg: ProposalConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.query_cfg = query_cfg
        self.proposal_cfg = proposal_cfg or ProposalConfig()
        self.feature_proj = nn.ModuleList([nn.Conv3d(c, cfg.d_model, 1) for c in input_channels])
        self.mask_feature_proj = nn.ModuleList([nn.Conv3d(c, cfg.mask_dim, 1) for c in input_channels])
        self.layers = nn.ModuleList([
            QueryDecoderLayer(cfg, temporal_cfg) for _ in range(cfg.layers)
        ])

    def _reference_support(self, q: QueryState, pos_um: Tensor, dref_um: Tensor, layer_idx: int) -> Tensor:
        B, Q = q.query_types.shape
        proposal = q.query_types == QUERY_SPATIAL_PROPOSAL
        anchor_references = (
            q.initial_references_cellscale
            if q.initial_references_cellscale is not None
            else q.references_cellscale
        )
        support_references = torch.where(
            proposal[..., None], anchor_references, q.references_cellscale
        )
        refs_um = support_references * dref_um[:, None, None]
        dist = torch.linalg.vector_norm(pos_um[:, None] - refs_um[:, :, None], dim=-1)
        base = self.query_cfg.temporal_gaussian_sigma_dref * 2.0
        radius = torch.full((B,Q), base, device=dist.device, dtype=dist.dtype)
        temporal = q.query_types == QUERY_TEMPORAL
        radius = torch.where(temporal, 1.5 + q.temporal_salience[...,0], radius)
        discovery = q.query_types == QUERY_DISCOVERY
        if layer_idx == 0:
            radius = torch.where(discovery, torch.full_like(radius, 1e6), radius)
        else:
            radius = torch.where(discovery, torch.full_like(radius, 2.5), radius)
        proposal_radii = (
            self.proposal_cfg.attention_radius_layer0_dref,
            self.proposal_cfg.attention_radius_layer1_dref,
            self.proposal_cfg.attention_radius_layer2_dref,
        )
        radius = torch.where(
            proposal,
            torch.full_like(radius, float(proposal_radii[layer_idx])),
            radius,
        )
        support = dist <= radius[...,None] * dref_um[:,None,None]
        support = support & (~q.padding_mask[...,None])
        return support

    def _source_instance_support(self, q: QueryState, instance_labels: Tensor, spatial_shape: tuple[int,int,int]) -> Tensor:
        labels = resize_label_map_nearest(instance_labels, spatial_shape)
        B,Q = q.source_instance_ids.shape
        support = torch.zeros((B,Q,*spatial_shape), device=instance_labels.device, dtype=torch.bool)
        for b in range(B):
            ids = q.source_instance_ids[b]
            seeded = (ids >= 0) & (
                (q.query_types[b] == QUERY_PRIMARY)
                | (q.query_types[b] == QUERY_SPLIT)
            )
            if seeded.any():
                support[b,seeded] = labels[b][None] == ids[seeded,None,None,None]
        return support

    def _dilate(self, masks: Tensor, spacing_um: Tensor, dref_um: Tensor) -> Tensor:
        out = torch.zeros_like(masks)
        B,Q = masks.shape[:2]
        for b in range(B):
            radius_um = self.cfg.support_dilation_dref * dref_um[b]
            rv = torch.ceil(radius_um / spacing_um[b].clamp_min(1e-8)).long().clamp(0,3)
            k = tuple(int(2*v.item()+1) for v in rv)
            if k == (1,1,1):
                out[b] = masks[b]
            else:
                out[b] = F.max_pool3d(masks[b].float().unsqueeze(1), kernel_size=k, stride=1,
                                      padding=tuple(int(v.item()) for v in rv)).squeeze(1).bool()
        return out

    def forward(
        self,
        query_state: QueryState,
        spatial_features: list[Tensor],
        spatial_spacings_um: list[Tensor],
        instance_labels: Tensor,
        dref_um: Tensor,
        temporal: TemporalState | None = None,
        *,
        memory_ablation: str = "full",
        routing_ablation: str = "full",
        return_debug: bool = False,
        full_attention: bool = False,
    ) -> tuple[QueryState, list[dict[str,Tensor]]]:
        q = query_state
        outputs = []
        previous_mask = None
        for li, layer in enumerate(self.layers):
            feat, spacing = _cap_feature_tokens(spatial_features[li], spatial_spacings_um[li], self.cfg.max_spatial_tokens)
            proj = self.feature_proj[li](feat)
            mask_feat = self.mask_feature_proj[li](feat)
            B,_,Z,Y,X = proj.shape
            spatial_tokens = proj.flatten(2).transpose(1,2)
            pos_um = feature_grid_coordinates_um((Z,Y,X), spacing, relative_to_center=True)
            support = self._reference_support(q, pos_um, dref_um, li).reshape(B, q.embeddings.shape[1], Z, Y, X)
            source_support = self._source_instance_support(q, instance_labels, (Z,Y,X))
            seeded = (q.query_types == QUERY_PRIMARY) | (q.query_types == QUERY_SPLIT)
            support = support | (source_support & seeded[...,None,None,None])
            if previous_mask is not None:
                prev = F.interpolate(previous_mask.sigmoid(), size=(Z,Y,X), mode="trilinear", align_corners=False)
                prev_support = prev > self.cfg.mask_attention_threshold
                prev_support = self._dilate(prev_support, spacing, dref_um)
                proposal = q.query_types == QUERY_SPATIAL_PROPOSAL
                if proposal.any():
                    proposal_cap = self._reference_support(
                        q, pos_um, dref_um, li
                    ).reshape(B, q.embeddings.shape[1], Z, Y, X)
                    prev_support = torch.where(
                        proposal[..., None, None, None],
                        prev_support & proposal_cap,
                        prev_support,
                    )
                support = support | prev_support
            support_flat = support.flatten(2)
            q, out = layer(
                q,
                spatial_tokens,
                pos_um,
                support_flat,
                dref_um,
                mask_feat,
                temporal,
                memory_ablation=memory_ablation,
                routing_ablation=routing_ablation,
                return_debug=return_debug,
                full_attention=full_attention,
            )
            out["coarse_spacing_um"] = spacing
            outputs.append(out)
            previous_mask = out["coarse_mask_logits"]
        return q, outputs
