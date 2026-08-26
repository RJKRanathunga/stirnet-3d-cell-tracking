"""Explicit, sparse and daughter-order-invariant division reasoning."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import ReconcilerConfig
from ..contracts import DivisionHypothesisBatch
from .blocks import MLP


def _gather(values: Tensor, indices: Tensor) -> Tensor:
    b = torch.arange(values.shape[0], device=values.device)[:, None]
    return values[b, indices]


def _gather_scalar(values: Tensor, indices: Tensor) -> Tensor:
    b = torch.arange(values.shape[0], device=values.device)[:, None]
    return values[b, indices]


class DivisionHead(nn.Module):
    """Score parent -> {child A, child B} after edge reasoning.

    Symmetric operations guarantee that swapping daughter order cannot alter the
    prediction.  This keeps lineage semantics out of arbitrary tensor ordering.
    """

    def __init__(self, config: ReconcilerConfig) -> None:
        super().__init__()
        h = config.edge.edge_dim
        q = config.division.pair_feature_dim
        in_dim = config.temporal.fused_tracklet_dim + 3 * h + q + 1
        self.net = MLP(
            in_dim,
            1,
            hidden_dim=config.division.hidden_dim,
            dropout=config.division.dropout,
            layers=3,
        )

    def forward(
        self,
        edge_embeddings: Tensor,
        edge_index: Tensor,
        tracklet_tail: Tensor,
        division_prior_logits: Tensor,
        batch: DivisionHypothesisBatch,
    ) -> Tensor:
        first = batch.edge_pair_index[..., 0]
        second = batch.edge_pair_index[..., 1]
        e1 = _gather(edge_embeddings, first)
        e2 = _gather(edge_embeddings, second)

        source_for_edge = edge_index[..., 0]
        target_for_edge = edge_index[..., 1]
        source1 = _gather_scalar(source_for_edge, first)
        source2 = _gather_scalar(source_for_edge, second)
        target1 = _gather_scalar(target_for_edge, first)
        target2 = _gather_scalar(target_for_edge, second)
        active = batch.hypothesis_mask
        if active.any() and not torch.equal(source1[active], source2[active]):
            raise ValueError("both edges of a division hypothesis must share the same parent")
        if active.any() and torch.any(target1[active] == target2[active]):
            raise ValueError("division daughters must be distinct target tracklets")
        parent = _gather(tracklet_tail, source1)
        prior = _gather(division_prior_logits.unsqueeze(-1), source1)

        symmetric = torch.cat((e1 + e2, torch.abs(e1 - e2), e1 * e2), dim=-1)
        x = torch.cat((parent, symmetric, batch.features, prior), dim=-1)
        logits = self.net(x).squeeze(-1)
        return logits.masked_fill(~batch.hypothesis_mask, 0.0)
