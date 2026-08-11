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
    """Bounded-memory physical-radius cross-attention.

    Query/key chunking is purely an internal execution detail: all spatial and
    temporal tokens still participate in one logical sample. Temporal-to-spatial
    attention uses an online softmax over spatial key chunks, so its result is
    mathematically equivalent to attending over every local key at once.
    """
    def __init__(self, cfg: CoReasoningConfig):
        super().__init__()
        d, h = cfg.d_model, cfg.heads
        if d % h:
            raise ValueError("d_model must be divisible by heads")
        self.d_model, self.heads, self.head_dim = d, h, d // h
        self.dropout = cfg.dropout
        self.temporal_query_chunk_size = cfg.temporal_query_chunk_size
        self.spatial_query_chunk_size = cfg.spatial_query_chunk_size
        self.spatial_key_chunk_size = cfg.spatial_key_chunk_size
        for name in (
            "temporal_query_chunk_size",
            "spatial_query_chunk_size",
            "spatial_key_chunk_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
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
            radius = (base_radius_dref + sal).clamp_max(base_radius_dref + 1.0) * dref_um[b]
            padding = spatial_padding_mask[b].reshape(-1) if spatial_padding_mask is not None else None
            messages = []
            for q_start in range(0, ids.numel(), self.temporal_query_chunk_size):
                q_end = min(q_start + self.temporal_query_chunk_size, ids.numel())
                q = self._split(self.t_q(t[q_start:q_end])).permute(1, 0, 2)
                q_count = q_end - q_start

                # FP32 online softmax state. Only H x query-chunk state survives
                # between spatial-key chunks.
                running_max = torch.full(
                    (self.heads, q_count), -torch.inf, device=s.device, dtype=torch.float32
                )
                denominator = torch.zeros_like(running_max)
                weighted_value = torch.zeros(
                    (self.heads, q_count, self.head_dim),
                    device=s.device,
                    dtype=torch.float32,
                )
                qref = tref[q_start:q_end]
                qradius = radius[q_start:q_end]

                for k_start in range(0, s.shape[0], self.spatial_key_chunk_size):
                    k_end = min(k_start + self.spatial_key_chunk_size, s.shape[0])
                    s_chunk = s[k_start:k_end]
                    k = self._split(self.s_k(s_chunk)).permute(1, 0, 2)
                    v = self._split(self.s_v(s_chunk)).permute(1, 0, 2)
                    logits = torch.einsum("hmd,hnd->hmn", q, k) / math.sqrt(self.head_dim)
                    delta = spos[k_start:k_end][None, :, :] - qref[:, None, :]
                    bias = self.pos_bias(delta, dref_um[b]).permute(2, 0, 1)
                    logits = (logits + bias).float()
                    valid = torch.linalg.vector_norm(delta, dim=-1) <= qradius[:, None]
                    if padding is not None:
                        valid = valid & (~padding[k_start:k_end][None, :])
                    logits = logits.masked_fill(~valid[None], -torch.inf)

                    chunk_max = logits.amax(dim=-1)
                    new_max = torch.maximum(running_max, chunk_max)
                    safe_new_max = torch.where(torch.isfinite(new_max), new_max, torch.zeros_like(new_max))
                    old_scale = torch.where(
                        torch.isfinite(running_max),
                        torch.exp(running_max - safe_new_max),
                        torch.zeros_like(running_max),
                    )
                    exp_logits = torch.where(
                        valid[None],
                        torch.exp(logits - safe_new_max[..., None]),
                        torch.zeros_like(logits),
                    )
                    dropped_exp = F.dropout(exp_logits, self.dropout, self.training)
                    weighted_value = (
                        weighted_value * old_scale[..., None]
                        + torch.einsum("hmn,hnd->hmd", dropped_exp, v.float())
                    )
                    denominator = denominator * old_scale + exp_logits.sum(dim=-1)
                    running_max = new_max

                msg = weighted_value / denominator.clamp_min(1e-12)[..., None]
                msg = torch.where(denominator[..., None] > 0, msg, torch.zeros_like(msg))
                msg = msg.permute(1, 0, 2).reshape(q_count, self.d_model)
                messages.append(self.t_out(msg).to(out.dtype))

            out = out.index_copy(0, ids, torch.cat(messages, dim=0))
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
            k = self._split(self.t_k(t)).permute(1, 0, 2)        # H,M,Dh
            v = self._split(self.t_v(t)).permute(1, 0, 2)
            radius = (base_radius_dref + sal).clamp_max(base_radius_dref + 1.0) * dref_um[b]
            padding = spatial_padding_mask[b].reshape(-1) if spatial_padding_mask is not None else None
            for q_start in range(0, s.shape[0], self.spatial_query_chunk_size):
                q_end = min(q_start + self.spatial_query_chunk_size, s.shape[0])
                q = self._split(self.s_q(s[q_start:q_end])).permute(1, 0, 2)
                delta = tref[None, :, :] - spos[q_start:q_end, None, :]
                bias = self.pos_bias(delta, dref_um[b]).permute(2, 0, 1)
                logits = torch.einsum("hnd,hmd->hnm", q, k) / math.sqrt(self.head_dim)
                logits = (
                    logits.float()
                    + bias.float()
                    + sal_scale.float() * sal.float()[None, None, :]
                    + rel_scale.float() * torch.log(rel.float())[None, None, :]
                )
                valid = torch.linalg.vector_norm(delta, dim=-1) <= radius[None, :]
                valid_any = valid.any(dim=-1)
                # Softmax receives finite values for empty rows, after which
                # their messages are explicitly zeroed.
                safe_logits = logits.masked_fill(~valid[None], -torch.inf)
                safe_logits = torch.where(
                    valid_any[None, :, None], safe_logits, torch.zeros_like(safe_logits)
                )
                weights = torch.softmax(safe_logits, dim=-1)
                weights = F.dropout(weights, self.dropout, self.training)
                weights = weights * valid_any[None, :, None]
                msg = torch.einsum("hnm,hmd->hnd", weights, v.float())
                msg = msg.permute(1, 0, 2).reshape(q_end - q_start, self.d_model)
                projected = self.s_out(msg).to(out.dtype)
                if padding is not None:
                    projected = projected.masked_fill(padding[q_start:q_end, None], 0)
                out[b, q_start:q_end] = projected
        return out
