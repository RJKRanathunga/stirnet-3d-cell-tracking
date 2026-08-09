from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .attention import PhysicalPositionBias
from .blocks import FeedForward
from .config import DecoderConfig, QueryConfig
from .coordinates import feature_grid_coordinates_um
from .heads import CenterHead, ExistenceHead, MaskEmbeddingHead, dot_mask_logits
from .query_builder import QUERY_DISCOVERY, QUERY_PRIMARY, QUERY_SPLIT, QUERY_TEMPORAL
from .types import QueryState


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
    ratio = torch.tensor([z / target[0], y / target[1], x / target[2]], device=feature.device, dtype=feature.dtype)
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
            sup = support[b, valid_q]
            # Ensure every query has at least one key.
            empty = ~sup.any(dim=-1)
            if empty.any():
                nearest = torch.linalg.vector_norm(delta[empty], dim=-1).argmin(dim=-1)
                rows = torch.nonzero(empty, as_tuple=False).flatten()
                sup[rows, nearest] = True
            logits = logits.masked_fill(~sup[None], -1e4)
            weights = torch.softmax(logits, dim=-1)
            weights = F.dropout(weights, self.dropout, self.training)
            msg = torch.einsum("hqn,hnd->hqd", weights, v).permute(1,0,2).reshape(valid_q.sum(), self.d_model)
            out[b, valid_q] = self.out(msg)
        return out


class QueryDecoderLayer(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.self_norm = nn.LayerNorm(cfg.d_model)
        self.self_attn = nn.MultiheadAttention(cfg.d_model, cfg.heads, cfg.dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(cfg.d_model)
        self.cross_attn = QueryCrossAttention(cfg)
        self.ffn_norm = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.ffn_dim, cfg.dropout)
        self.exist = ExistenceHead(cfg.d_model)
        self.center = CenterHead(cfg.d_model)
        self.mask_embed = MaskEmbeddingHead(cfg.d_model, cfg.mask_dim)

    def forward(self, q: QueryState, spatial_tokens: Tensor, spatial_pos_um: Tensor, support: Tensor, dref_um: Tensor, spatial_mask_features: Tensor):
        x = q.embeddings
        n = self.self_norm(x)
        a, _ = self.self_attn(n, n, n, key_padding_mask=q.padding_mask, need_weights=False)
        x = x + a
        c = self.cross_attn(self.cross_norm(x), spatial_tokens, spatial_pos_um,
                            q.references_cellscale * dref_um[:, None, None], support, q.padding_mask, dref_um)
        x = x + c
        x = x + self.ffn(self.ffn_norm(x))
        x = x.masked_fill(q.padding_mask[..., None], 0)
        delta = self.center(x)
        refs = q.references_cellscale + delta.masked_fill(q.padding_mask[..., None], 0)
        emb = self.mask_embed(x)
        masks = dot_mask_logits(emb, spatial_mask_features)
        masks = masks.masked_fill(q.padding_mask[..., None, None, None], -20.0)
        return replace(q, embeddings=x, references_cellscale=refs), {
            "exist_logits": self.exist(x).masked_fill(q.padding_mask, -20.0),
            "centers_cellscale": refs,
            "coarse_mask_logits": masks,
            "query_embeddings": x,
        }


class InstanceQueryDecoder(nn.Module):
    def __init__(self, input_channels: tuple[int,int,int], cfg: DecoderConfig, query_cfg: QueryConfig):
        super().__init__()
        self.cfg = cfg
        self.query_cfg = query_cfg
        self.feature_proj = nn.ModuleList([nn.Conv3d(c, cfg.d_model, 1) for c in input_channels])
        self.mask_feature_proj = nn.ModuleList([nn.Conv3d(c, cfg.mask_dim, 1) for c in input_channels])
        self.layers = nn.ModuleList([QueryDecoderLayer(cfg) for _ in range(cfg.layers)])

    def _reference_support(self, q: QueryState, pos_um: Tensor, dref_um: Tensor, layer_idx: int) -> Tensor:
        B, Q = q.query_types.shape
        refs_um = q.references_cellscale * dref_um[:, None, None]
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
        support = dist <= radius[...,None] * dref_um[:,None,None]
        support = support & (~q.padding_mask[...,None])
        return support

    def _source_instance_support(self, q: QueryState, instance_labels: Tensor, spatial_shape: tuple[int,int,int]) -> Tensor:
        labels = F.interpolate(instance_labels[:,None].float(), size=spatial_shape, mode="nearest").squeeze(1).long()
        B,Q = q.source_instance_ids.shape
        support = torch.zeros((B,Q,*spatial_shape), device=instance_labels.device, dtype=torch.bool)
        for b in range(B):
            ids = q.source_instance_ids[b]
            seeded = ids >= 0
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
                support = support | prev_support
            support_flat = support.flatten(2)
            q, out = layer(q, spatial_tokens, pos_um, support_flat, dref_um, mask_feat)
            outputs.append(out)
            previous_mask = out["coarse_mask_logits"]
        return q, outputs
