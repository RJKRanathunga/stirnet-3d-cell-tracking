from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .checkpointing import checkpoint_if_enabled
from .config import CoReasoningConfig, HistoryConfig
from .history_support import HistorySupportBias, sample_projected_history_support


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
    def __init__(
        self,
        cfg: CoReasoningConfig,
        *,
        history_cfg: HistoryConfig | None = None,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        d, h = cfg.d_model, cfg.heads
        if d % h:
            raise ValueError("d_model must be divisible by heads")
        self.d_model, self.heads, self.head_dim = d, h, d // h
        self.dropout = cfg.dropout
        self.activation_checkpointing = activation_checkpointing
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
        self.history_bias = (
            HistorySupportBias(
                h, history_cfg.attention_bias_hidden, history_cfg.dt_normalizer
            )
            if history_cfg is not None
            and history_cfg.enabled
            and history_cfg.attention_bias_enabled
            else None
        )
        self.last_history_bias_stats: dict[str, Tensor] = {}

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

    def _temporal_key_update(
        self,
        q: Tensor,
        query_ref_um: Tensor,
        query_radius_um: Tensor,
        running_max: Tensor,
        denominator: Tensor,
        weighted_value: Tensor,
        spatial_chunk: Tensor,
        spatial_pos_chunk_um: Tensor,
        dref_um: Tensor,
        padding_chunk: Tensor,
        history_support: Tensor,
        history_valid: Tensor,
        history_dt: Tensor,
        history_center_um: Tensor,
        history_extent_um: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply one online-softmax spatial key chunk."""
        key = self._split(self.s_k(spatial_chunk)).permute(1, 0, 2)
        value = self._split(self.s_v(spatial_chunk)).permute(1, 0, 2)
        logits = torch.einsum("hmd,hnd->hmn", q, key) / math.sqrt(self.head_dim)
        delta = spatial_pos_chunk_um[None, :, :] - query_ref_um[:, None, :]
        bias = self.pos_bias(delta, dref_um).permute(2, 0, 1)
        logits = (logits + bias).float()
        if self.history_bias is not None and history_support.numel():
            samples = sample_projected_history_support(
                history_support,
                history_valid,
                history_center_um,
                query_ref_um,
                history_extent_um,
                spatial_pos_chunk_um,
            )
            support_bias = self.history_bias(samples, history_valid, history_dt)
            logits = logits + support_bias.permute(2, 0, 1).float()
            self.last_history_bias_stats = {
                "mean": support_bias.detach().mean(),
                "mean_abs": support_bias.detach().abs().mean(),
                "max_abs": support_bias.detach().abs().amax(),
                "valid_fraction": history_valid.detach().float().mean(),
            }
        valid = torch.linalg.vector_norm(delta, dim=-1) <= query_radius_um[:, None]
        valid = valid & (~padding_chunk[None, :])
        logits = logits.masked_fill(~valid[None], -torch.inf)

        chunk_max = logits.amax(dim=-1)
        new_max = torch.maximum(running_max, chunk_max)
        safe_new_max = torch.where(
            torch.isfinite(new_max), new_max, torch.zeros_like(new_max)
        )
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
        new_weighted_value = (
            weighted_value * old_scale[..., None]
            + torch.einsum("hmn,hnd->hmd", dropped_exp, value.float())
        )
        new_denominator = denominator * old_scale + exp_logits.sum(dim=-1)
        return new_max, new_denominator, new_weighted_value

    def _spatial_query_chunk(
        self,
        spatial_chunk: Tensor,
        spatial_pos_chunk_um: Tensor,
        temporal_key: Tensor,
        temporal_value: Tensor,
        temporal_ref_um: Tensor,
        salience: Tensor,
        reliability: Tensor,
        radius_um: Tensor,
        dref_um: Tensor,
        salience_scale: Tensor,
        reliability_scale: Tensor,
        padding_chunk: Tensor,
    ) -> Tensor:
        """Compute one bounded spatial-query chunk."""
        query = self._split(self.s_q(spatial_chunk)).permute(1, 0, 2)
        delta = temporal_ref_um[None, :, :] - spatial_pos_chunk_um[:, None, :]
        bias = self.pos_bias(delta, dref_um).permute(2, 0, 1)
        logits = torch.einsum("hnd,hmd->hnm", query, temporal_key)
        logits = logits / math.sqrt(self.head_dim)
        logits = (
            logits.float()
            + bias.float()
            + salience_scale.float() * salience.float()[None, None, :]
            + reliability_scale.float()
            * torch.log(reliability.float())[None, None, :]
        )
        valid = torch.linalg.vector_norm(delta, dim=-1) <= radius_um[None, :]
        valid_any = valid.any(dim=-1)
        safe_logits = logits.masked_fill(~valid[None], -torch.inf)
        safe_logits = torch.where(
            valid_any[None, :, None], safe_logits, torch.zeros_like(safe_logits)
        )
        weights = torch.softmax(safe_logits, dim=-1)
        weights = F.dropout(weights, self.dropout, self.training)
        weights = weights * valid_any[None, :, None]
        message = torch.einsum("hnm,hmd->hnd", weights, temporal_value.float())
        message = message.permute(1, 0, 2).reshape(
            spatial_chunk.shape[0], self.d_model
        )
        projected = self.s_out(message).to(spatial_chunk.dtype)
        return projected.masked_fill(padding_chunk[:, None], 0)

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
        history_support: Tensor | None = None,
        history_support_valid: Tensor | None = None,
        history_support_dt: Tensor | None = None,
        history_support_center_um: Tensor | None = None,
        history_support_extent_um: Tensor | None = None,
    ) -> Tensor:
        out = torch.zeros_like(temporal)
        self.last_history_bias_stats = {}
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
                local_ids = ids[q_start:q_end]
                empty = s.new_zeros((q_count, 0))
                q_support = history_support[local_ids] if history_support is not None else empty
                q_valid = history_support_valid[local_ids] if history_support_valid is not None else empty.bool()
                q_dt = history_support_dt[local_ids] if history_support_dt is not None else empty
                q_center = history_support_center_um[local_ids] if history_support_center_um is not None else empty
                q_extent = history_support_extent_um[local_ids] if history_support_extent_um is not None else empty

                for k_start in range(0, s.shape[0], self.spatial_key_chunk_size):
                    k_end = min(k_start + self.spatial_key_chunk_size, s.shape[0])
                    s_chunk = s[k_start:k_end]
                    padding_chunk = (
                        padding[k_start:k_end]
                        if padding is not None
                        else torch.zeros(
                            k_end - k_start, device=s.device, dtype=torch.bool
                        )
                    )
                    running_max, denominator, weighted_value = checkpoint_if_enabled(
                        self._temporal_key_update,
                        q,
                        qref,
                        qradius,
                        running_max,
                        denominator,
                        weighted_value,
                        s_chunk,
                        spos[k_start:k_end],
                        dref_um[b],
                        padding_chunk,
                        q_support,
                        q_valid,
                        q_dt,
                        q_center,
                        q_extent,
                        enabled=self.activation_checkpointing and self.training,
                    )

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
                padding_chunk = (
                    padding[q_start:q_end]
                    if padding is not None
                    else torch.zeros(
                        q_end - q_start, device=s.device, dtype=torch.bool
                    )
                )
                projected = checkpoint_if_enabled(
                    self._spatial_query_chunk,
                    s[q_start:q_end],
                    spos[q_start:q_end],
                    k,
                    v,
                    tref,
                    sal,
                    rel,
                    radius,
                    dref_um[b],
                    sal_scale,
                    rel_scale,
                    padding_chunk,
                    enabled=self.activation_checkpointing and self.training,
                ).to(out.dtype)
                out[b, q_start:q_end] = projected
        return out
