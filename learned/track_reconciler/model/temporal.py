"""Temporal tracklet encoding with separate appearance and structured streams."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import ReconcilerConfig
from ..contracts import TrackletBatch, TrackletEncoding
from .blocks import BranchDropout, GatedFusion
from .fingerprint import CellFingerprintEncoder


class FourierTimeEncoding(nn.Module):
    def __init__(self, bands: int, out_dim: int) -> None:
        super().__init__()
        self.bands = int(bands)
        frequencies = 2.0 ** torch.arange(self.bands, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.proj = nn.Linear(2 * self.bands + 1, out_dim)

    def forward(self, times: Tensor, mask: Tensor) -> Tensor:
        # Remove sequence-specific absolute frame offset while retaining gaps.
        safe = torch.where(mask, times, torch.full_like(times, float("inf")))
        valid_reference = safe.min(dim=-1, keepdim=True).values
        valid_reference = torch.where(torch.isfinite(valid_reference), valid_reference, torch.zeros_like(valid_reference))
        rel = times - valid_reference
        phase = rel.unsqueeze(-1) * self.frequencies * (math.pi / 4.0)
        encoded = torch.cat((rel.unsqueeze(-1), phase.sin(), phase.cos()), dim=-1)
        return self.proj(encoded)


class TemporalStream(nn.Module):
    def __init__(self, in_dim: int, config: ReconcilerConfig) -> None:
        super().__init__()
        tc = config.temporal
        self.input = nn.Linear(in_dim, tc.model_dim)
        self.time = FourierTimeEncoding(tc.time_fourier_bands, tc.model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=tc.model_dim,
            nhead=tc.num_heads,
            dim_feedforward=tc.feedforward_dim,
            dropout=tc.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=tc.num_layers,
            norm=nn.LayerNorm(tc.model_dim),
            enable_nested_tensor=False,
        )

    def forward(self, x: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        # x [M,K,D], mask True=valid. Transformer uses True=padding.
        token = self.input(x) + self.time(times, mask)
        # All-masked sequences are prevented at the data-contract level for
        # active tracklets; padded tracklets get a temporary first valid token
        # to avoid NaNs and are masked out again downstream.
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=-1)
        if empty.any():
            safe_mask[empty, 0] = True
        out = self.encoder(token, src_key_padding_mask=~safe_mask)
        return out * mask.unsqueeze(-1).to(out.dtype)


def _first_last_pool(sequence: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    m, k, d = sequence.shape
    lengths = mask.sum(dim=-1).clamp_min(1)
    first_idx = torch.argmax(mask.to(torch.int64), dim=-1)
    reversed_idx = torch.argmax(mask.flip(-1).to(torch.int64), dim=-1)
    last_idx = (k - 1) - reversed_idx
    rows = torch.arange(m, device=sequence.device)
    first = sequence[rows, first_idx]
    last = sequence[rows, last_idx]
    pooled = (sequence * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
    return first, last, pooled


class TrackletEncoder(nn.Module):
    """Encode a high-purity tracklet into role-specific head/tail states."""

    def __init__(self, config: ReconcilerConfig) -> None:
        super().__init__()
        self.config = config
        self.fingerprint = CellFingerprintEncoder(config.fingerprint)
        self.appearance_stream = TemporalStream(config.fingerprint.embedding_dim, config)
        self.structured_stream = TemporalStream(config.temporal.structured_dim, config)
        tc = config.temporal
        self.fuse_head = GatedFusion(tc.model_dim, tc.model_dim, tc.reliability_dim, tc.fused_tracklet_dim)
        self.fuse_tail = GatedFusion(tc.model_dim, tc.model_dim, tc.reliability_dim, tc.fused_tracklet_dim)
        self.fuse_pool = GatedFusion(tc.model_dim, tc.model_dim, tc.reliability_dim, tc.fused_tracklet_dim)
        self.appearance_drop = BranchDropout(config.appearance_modality_dropout, shared_dims=(-2,))
        self.structured_drop = BranchDropout(config.structured_modality_dropout, shared_dims=(-2,))

    def _fingerprints(self, batch: TrackletBatch) -> Tensor:
        if batch.fingerprints is not None:
            return batch.fingerprints
        assert batch.crops is not None
        b, n, k, c, d, h, w = batch.crops.shape
        flat = batch.crops.reshape(b * n * k, c, d, h, w)
        active = batch.observation_mask.reshape(-1)
        encoded = torch.zeros(
            b * n * k,
            self.config.fingerprint.embedding_dim,
            device=flat.device,
            dtype=flat.dtype,
        )
        if active.any():
            encoded[active] = self.fingerprint(flat[active])
        return encoded.reshape(b, n, k, -1)

    def forward(self, batch: TrackletBatch) -> TrackletEncoding:
        b, n, k, _ = batch.structured.shape
        fingerprints = self._fingerprints(batch)
        if fingerprints.shape[-1] != self.config.fingerprint.embedding_dim:
            raise ValueError(
                f"fingerprint dim must be {self.config.fingerprint.embedding_dim}, "
                f"got {fingerprints.shape[-1]}"
            )
        if batch.structured.shape[-1] != self.config.temporal.structured_dim:
            raise ValueError(
                f"structured dim must be {self.config.temporal.structured_dim}, "
                f"got {batch.structured.shape[-1]}"
            )
        if batch.reliability.shape[-1] != self.config.temporal.reliability_dim:
            raise ValueError("tracklet reliability feature dimension does not match config")

        m = b * n
        mask = batch.observation_mask.reshape(m, k)
        times = batch.times.reshape(m, k)
        app = self.appearance_drop(fingerprints.reshape(m, k, -1))
        structured = self.structured_drop(batch.structured.reshape(m, k, -1))
        app_seq = self.appearance_stream(app, times, mask)
        struct_seq = self.structured_stream(structured, times, mask)
        ah, at, ap = _first_last_pool(app_seq, mask)
        sh, st, sp = _first_last_pool(struct_seq, mask)
        reliability = batch.reliability.reshape(m, -1)

        head = self.fuse_head(ah, sh, reliability)
        tail = self.fuse_tail(at, st, reliability)
        pooled = self.fuse_pool(ap, sp, reliability)
        track_mask = batch.tracklet_mask.reshape(m, 1).to(head.dtype)
        head, tail, pooled = head * track_mask, tail * track_mask, pooled * track_mask
        return TrackletEncoding(
            head=head.reshape(b, n, -1),
            tail=tail.reshape(b, n, -1),
            pooled=pooled.reshape(b, n, -1),
            fingerprint_sequence=fingerprints,
        )
