from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .blocks import FeedForward
from .config import TemporalConfig


def segment_softmax(scores: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Softmax over edges grouped by destination index. scores=[E,H]."""
    if scores.numel() == 0:
        return scores
    original_dtype = scores.dtype
    work = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
    h = work.shape[1]
    idx = index[:, None].expand(-1, h)
    max_buf = torch.full((num_segments, h), -torch.inf, device=work.device, dtype=work.dtype)
    max_buf.scatter_reduce_(0, idx, work, reduce="amax", include_self=True)
    stable = work - max_buf[index]
    ex = torch.exp(stable)
    denom = torch.zeros((num_segments, h), device=work.device, dtype=work.dtype)
    denom.index_add_(0, index, ex)
    return (ex / denom[index].clamp_min(1e-8)).to(original_dtype)


class EdgeGATv2Conv(nn.Module):
    """Small pure-PyTorch edge-aware GATv2-style message passing layer."""
    def __init__(self, d_model: int, heads: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.edge = nn.Linear(edge_dim, d_model, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, self.head_dim))
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        n = x.shape[0]
        if n == 0:
            return x
        # Always provide self-loops.
        loops = torch.arange(n, device=x.device)
        loop_index = torch.stack([loops, loops], dim=0)
        loop_attr = torch.zeros((n, edge_attr.shape[-1]), device=x.device, dtype=edge_attr.dtype)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

        src, dst = edge_index[0], edge_index[1]
        q = self.q(x).view(n, self.heads, self.head_dim)
        k = self.k(x).view(n, self.heads, self.head_dim)
        v = self.v(x).view(n, self.heads, self.head_dim)
        e = self.edge(edge_attr).view(-1, self.heads, self.head_dim)
        joint = F.leaky_relu(q[dst] + k[src] + e, negative_slope=0.2)
        scores = (joint * self.attn[None]).sum(dim=-1) / math.sqrt(self.head_dim)
        alpha = segment_softmax(scores, dst, n)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = v[src] * alpha[..., None]
        out = torch.zeros((n, self.heads, self.head_dim), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg.to(out.dtype))
        return self.out(out.reshape(n, self.d_model))


class EdgeGATv2Block(nn.Module):
    def __init__(self, d_model: int, heads: int, edge_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = EdgeGATv2Conv(d_model, heads, edge_dim, dropout)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), edge_index, edge_attr)
        x = x + self.ffn(self.norm2(x))
        return x


class DetectionGraphEncoder(nn.Module):
    def __init__(self, cfg: TemporalConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.node_dim, 64), nn.SiLU(), nn.Linear(64, cfg.d_model)
        )
        self.layers = nn.ModuleList([
            EdgeGATv2Block(cfg.d_model, cfg.graph_heads, cfg.edge_dim, cfg.graph_ffn_dim, 0.1)
            for _ in range(cfg.graph_layers)
        ])

    def forward(self, graph_x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        x = self.input_proj(graph_x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x


class HypothesisGraphBlock(nn.Module):
    def __init__(self, cfg: TemporalConfig):
        super().__init__()
        self.block = EdgeGATv2Block(
            cfg.d_model, cfg.graph_heads, cfg.hypothesis_edge_dim, cfg.graph_ffn_dim, 0.1
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if x.shape[0] == 0:
            return x
        return self.block(x, edge_index, edge_attr)
