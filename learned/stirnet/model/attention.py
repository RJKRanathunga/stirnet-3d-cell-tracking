from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import CoReasoningConfig


class PhysicalPositionBias(nn.Module):
    def __init__(self, heads: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, heads))

    def forward(self, delta_um: Tensor, dref_um: Tensor) -> Tensor:
        # delta_um [...,3], dref scalar/tensor broadcastable to [...,1]
        while dref_um.ndim < delta_um.ndim:
            dref_um = dref_um.unsqueeze(-1)
        scaled = delta_um / dref_um.clamp_min(1e-8)
        dist = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
        return self.net(torch.cat([scaled, dist], dim=-1))


class LocalPhysicalCrossAttention(nn.Module):
    """Bidirectional sparse-by-radius cross-attention implemented batch-by-batch.

    The implementation intentionally favours correctness/debuggability over fused-kernel
    complexity. V2 can replace this with deformable sampling without changing interfaces.
    """
    def __init__(self, cfg: CoReasoningConfig):
        super().__init__()
        d, h = cfg.d_model, cfg.heads
        if d % h:
            raise ValueError("d_model must be divisible by heads")
        self.d_model, self.heads, self.head_dim = d, h, d // h
        self.dropout = cfg.dropout
        self.pos_bias = PhysicalPositionBias(h, cfg.position_bias_hidden)

        self.t_q = nn.Linear(d, d, bias=False)
        self.s_k = nn.Linear(d, d, bias=False)
        self.s_v = nn.Linear(d, d, bias=False)
        self.t_out = nn.Linear(d, d, bias=False)

        self.s_q = nn.Linear(d, d, bias=False)
        self.t_k = nn.Linear(d, d, bias=False)
        self.t_v = nn.Linear(d, d, bias=False)
        self.s_out = nn.Linear(d, d, bias=False)

        self.salience_scale = nn.Parameter(torch.zeros(h))
        self.reliability_scale = nn.Parameter(torch.zeros(h))

    def _split(self, x: Tensor) -> Tensor:
        return x.view(*x.shape[:-1], self.heads, self.head_dim)

    def temporal_reads_spatial(
        self,
        temporal: Tensor,
        temporal_ref_um: Tensor,
        salience: Tensor,
        spatial: Tensor,
        spatial_pos_um: Tensor,
        temporal_batch: Tensor,
        dref_um: Tensor,
        spatial_padding_mask: Tensor | None = None,
        base_radius_dref: float = 1.5,
    ) -> Tensor:
        out = torch.zeros_like(temporal)
        B = spatial.shape[0]
        for b in range(B):
            ids = torch.nonzero(temporal_batch == b, as_tuple=False).flatten()
            if ids.numel() == 0:
                continue
            t = temporal[ids]
            tref = temporal_ref_um[ids]
            sal = salience[ids, 0]
            s = spatial[b]
            spos = spatial_pos_um[b]
            q = self._split(self.t_q(t)).permute(1, 0, 2)        # H,M,Dh
            k = self._split(self.s_k(s)).permute(1, 0, 2)        # H,N,Dh
            v = self._split(self.s_v(s)).permute(1, 0, 2)
            logits = torch.einsum("hmd,hnd->hmn", q, k) / math.sqrt(self.head_dim)
            delta = spos[None, :, :] - tref[:, None, :]           # M,N,3
            bias = self.pos_bias(delta, dref_um[b]).permute(2, 0, 1)
            logits = logits + bias
            dist = torch.linalg.vector_norm(delta, dim=-1)
            radius = (base_radius_dref + sal).clamp_max(base_radius_dref + 1.0) * dref_um[b]
            invalid = dist > radius[:, None]
            if spatial_padding_mask is not None:
                invalid = invalid | spatial_padding_mask[b].reshape(1, -1)
            logits = logits.masked_fill(invalid[None], -1e4)
            weights = torch.softmax(logits, dim=-1)
            weights = F.dropout(weights, self.dropout, self.training)
            msg = torch.einsum("hmn,hnd->hmd", weights, v).permute(1, 0, 2).reshape(ids.numel(), self.d_model)
            out[ids] = self.t_out(msg)
        return out

    def spatial_reads_temporal(
        self,
        spatial: Tensor,
        spatial_pos_um: Tensor,
        temporal: Tensor,
        temporal_ref_um: Tensor,
        salience: Tensor,
        reliability: Tensor,
        temporal_batch: Tensor,
        dref_um: Tensor,
        spatial_padding_mask: Tensor | None = None,
        base_radius_dref: float = 1.5,
    ) -> Tensor:
        out = torch.zeros_like(spatial)
        B = spatial.shape[0]
        sal_scale = F.softplus(self.salience_scale)[:, None, None]
        rel_scale = self.reliability_scale[:, None, None]
        for b in range(B):
            ids = torch.nonzero(temporal_batch == b, as_tuple=False).flatten()
            if ids.numel() == 0:
                continue
            t = temporal[ids]
            tref = temporal_ref_um[ids]
            sal = salience[ids, 0]
            rel = reliability[ids, 0].clamp_min(1e-4)
            s = spatial[b]
            spos = spatial_pos_um[b]
            q = self._split(self.s_q(s)).permute(1, 0, 2)        # H,N,Dh
            k = self._split(self.t_k(t)).permute(1, 0, 2)        # H,M,Dh
            v = self._split(self.t_v(t)).permute(1, 0, 2)
            logits = torch.einsum("hnd,hmd->hnm", q, k) / math.sqrt(self.head_dim)
            delta = tref[None, :, :] - spos[:, None, :]           # N,M,3
            bias = self.pos_bias(delta, dref_um[b]).permute(2, 0, 1)
            logits = logits + bias
            logits = logits + sal_scale * sal[None, None, :]
            logits = logits + rel_scale * torch.log(rel)[None, None, :]
            dist = torch.linalg.vector_norm(delta, dim=-1)
            radius = (base_radius_dref + sal).clamp_max(base_radius_dref + 1.0) * dref_um[b]
            invalid = dist > radius[None, :]
            logits = logits.masked_fill(invalid[None], -1e4)
            valid_any = (~invalid).any(dim=-1)                   # N
            weights = torch.softmax(logits, dim=-1)
            weights = weights * valid_any[None, :, None]
            weights = F.dropout(weights, self.dropout, self.training)
            msg = torch.einsum("hnm,hmd->hnd", weights, v).permute(1, 0, 2).reshape(s.shape[0], self.d_model)
            msg = self.s_out(msg)
            if spatial_padding_mask is not None:
                msg = msg.masked_fill(spatial_padding_mask[b].reshape(-1, 1), 0)
            out[b] = msg
        return out
