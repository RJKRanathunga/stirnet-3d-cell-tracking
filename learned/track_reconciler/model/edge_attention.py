"""Geometry-biased edge-centric Transformer reasoning."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import ReconcilerConfig
from ..features.geometry import pairwise_segment_distance


class Rotary3D(nn.Module):
    """Apply axis-factorized 3-D rotary position encoding to Q/K tensors."""

    def __init__(self, head_dim: int, position_scale_um: float) -> None:
        super().__init__()
        if head_dim % 6:
            raise ValueError("head_dim must be divisible by 6 for 3-D RoPE")
        axis_dim = head_dim // 3
        inv = 1.0 / (10000 ** (torch.arange(0, axis_dim, 2).float() / axis_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.axis_dim = axis_dim
        self.scale = float(position_scale_um)

    @staticmethod
    def _rotate(x: Tensor, phase: Tensor) -> Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        cos = phase.cos()
        sin = phase.sin()
        out_even = even * cos - odd * sin
        out_odd = even * sin + odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

    def forward(self, q: Tensor, k: Tensor, position_zyx_um: Tensor) -> tuple[Tensor, Tensor]:
        # q/k [B,H,E,D], positions [B,E,3]
        chunks_q = q.split(self.axis_dim, dim=-1)
        chunks_k = k.split(self.axis_dim, dim=-1)
        out_q, out_k = [], []
        scaled = position_zyx_um / max(self.scale, 1e-6)
        for axis in range(3):
            phase = scaled[..., axis, None] * self.inv_freq
            phase = phase[:, None, :, :]  # [B,1,E,axis_dim/2]
            out_q.append(self._rotate(chunks_q[axis], phase))
            out_k.append(self._rotate(chunks_k[axis], phase))
        return torch.cat(out_q, dim=-1), torch.cat(out_k, dim=-1)


class GeometryBiasedEdgeAttention(nn.Module):
    """Global edge attention with HOCT-style attractive/repulsive distance heads.

    The module supports both self-attention and the first raw-edge -> contextual
    edge cross-attention used to ground primitive relation evidence in the
    source/target tracklet representations.
    """

    def __init__(self, config: ReconcilerConfig) -> None:
        super().__init__()
        ec = config.edge
        self.dim = ec.edge_dim
        self.heads = ec.num_heads
        self.head_dim = self.dim // self.heads
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(ec.dropout)
        self.rope = Rotary3D(self.head_dim, ec.rope_position_scale_um)
        self.distance_scale = float(ec.physical_distance_scale_um)
        self.alpha = nn.Parameter(torch.full((self.heads,), float(ec.geometry_alpha_init)))
        signs = torch.where(torch.arange(self.heads) % 2 == 0, -1.0, 1.0)
        self.register_buffer("signs", signs, persistent=True)

    def _heads(self, x: Tensor, projection: nn.Linear) -> Tensor:
        b, e, _ = x.shape
        return projection(x).view(b, e, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        edge_mask: Tensor,
        segment_start_um: Tensor,
        segment_end_um: Tensor,
        *,
        key_value: Tensor | None = None,
    ) -> Tensor:
        kv = query if key_value is None else key_value
        b, e, d = query.shape
        q = self._heads(query, self.q_proj)
        k = self._heads(kv, self.k_proj)
        v = self._heads(kv, self.v_proj)
        midpoint = 0.5 * (segment_start_um + segment_end_um)
        q, k = self.rope(q, k, midpoint)
        logits = torch.einsum("bhed,bhfd->bhef", q, k) / math.sqrt(self.head_dim)

        # Exact finite segment distance; detached because observed geometry is fixed.
        distance = pairwise_segment_distance(segment_start_um, segment_end_um).detach()
        distance = distance / max(self.distance_scale, 1e-6)
        strength = F.softplus(self.alpha) * self.signs
        logits = logits + strength[None, :, None, None] * distance[:, None]

        key_valid = edge_mask[:, None, None, :]
        logits = logits.masked_fill(~key_valid, torch.finfo(logits.dtype).min)
        empty = ~edge_mask.any(dim=-1)
        if empty.any():
            logits[empty, :, 0, 0] = 0.0
        attention = torch.softmax(logits, dim=-1)
        attention = self.dropout(attention)
        y = torch.einsum("bhef,bhfd->bhed", attention, v)
        y = y.transpose(1, 2).contiguous().view(b, e, d)
        return self.out(y) * edge_mask.unsqueeze(-1).to(y.dtype)


class EdgeTransformerLayer(nn.Module):
    def __init__(self, config: ReconcilerConfig) -> None:
        super().__init__()
        ec = config.edge
        self.norm_q = nn.LayerNorm(ec.edge_dim)
        self.norm_kv = nn.LayerNorm(ec.edge_dim)
        self.attention = GeometryBiasedEdgeAttention(config)
        self.norm2 = nn.LayerNorm(ec.edge_dim)
        self.ff = nn.Sequential(
            nn.Linear(ec.edge_dim, ec.feedforward_dim),
            nn.GELU(),
            nn.Dropout(ec.dropout),
            nn.Linear(ec.feedforward_dim, ec.edge_dim),
            nn.Dropout(ec.dropout),
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        start: Tensor,
        end: Tensor,
        *,
        key_value: Tensor | None = None,
    ) -> Tensor:
        kv = None if key_value is None else self.norm_kv(key_value)
        x = x + self.attention(self.norm_q(x), mask, start, end, key_value=kv)
        x = x + self.ff(self.norm2(x)) * mask.unsqueeze(-1).to(x.dtype)
        return x * mask.unsqueeze(-1).to(x.dtype)


class EdgeReasoner(nn.Module):
    """Ground primitive edge evidence, then perform edge-edge self-attention."""

    def __init__(self, config: ReconcilerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            EdgeTransformerLayer(config) for _ in range(config.edge.num_layers)
        )
        self.norm = nn.LayerNorm(config.edge.edge_dim)

    def forward(
        self,
        primitive_tokens: Tensor,
        contextual_tokens: Tensor,
        mask: Tensor,
        start: Tensor,
        end: Tensor,
    ) -> Tensor:
        if primitive_tokens.shape[1] == 0 or not self.layers:
            return self.norm(primitive_tokens)
        # Research-aligned grounding: raw edge relations query contextual A/B
        # tracklet-pair tokens in the first geometry-biased layer.
        x = self.layers[0](
            primitive_tokens,
            mask,
            start,
            end,
            key_value=contextual_tokens,
        )
        for layer in self.layers[1:]:
            x = layer(x, mask, start, end)
        return self.norm(x) * mask.unsqueeze(-1).to(x.dtype)
