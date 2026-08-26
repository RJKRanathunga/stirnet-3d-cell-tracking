"""End-to-end learned tracklet reconciliation network."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import ReconcilerConfig
from ..contracts import ReconciliationBatch, ReconciliationOutput
from ..features.motion import MOTION_RELATION_DIM, build_motion_relation_features, gather_edge_endpoints
from ..features.parental import parental_softmax
from .blocks import BranchDropout, MLP
from .division import DivisionHead
from .edge_attention import EdgeReasoner
from .temporal import TrackletEncoder


def _gather(values: Tensor, indices: Tensor) -> Tensor:
    b = torch.arange(values.shape[0], device=values.device)[:, None]
    return values[b, indices]


class TrackletReconciliationNetwork(nn.Module):
    """Edge-centric learned reconciler operating on high-purity tracklets.

    Information flow:
      cell crop -> fingerprint -> separate temporal streams -> head/tail states
      -> candidate transition edge tokens -> geometry-biased edge attention
      -> continuation/division/birth/termination event logits.

    Graph legality is intentionally *not* learned; use `graph.MILPDecoder` on
    these event logits/probabilities during inference.
    """

    def __init__(self, config: ReconcilerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ReconcilerConfig()
        if self.config.edge.relation_dim != MOTION_RELATION_DIM:
            raise ValueError(
                f"edge.relation_dim must equal implemented MOTION_RELATION_DIM={MOTION_RELATION_DIM}"
            )
        self.tracklets = TrackletEncoder(self.config)
        h = self.config.temporal.fused_tracklet_dim
        pair_in = self.config.edge.pair_feature_dim + self.config.edge.relation_dim
        self.pair_drop = BranchDropout(self.config.pair_modality_dropout)
        self.gather = MLP(2 * h, self.config.edge.edge_dim, hidden_dim=self.config.edge.edge_dim, dropout=0.1)
        self.relation = MLP(pair_in, self.config.edge.edge_dim, hidden_dim=self.config.edge.edge_dim, dropout=0.1)
        self.edge_reasoner = EdgeReasoner(self.config)
        self.continuation_head = nn.Sequential(
            nn.LayerNorm(self.config.edge.edge_dim), nn.Linear(self.config.edge.edge_dim, 1)
        )
        self.division_prior_head = MLP(2 * h, 1, hidden_dim=h, dropout=0.1)
        self.appearance_head = MLP(2 * h, 1, hidden_dim=h, dropout=0.1)
        self.termination_head = MLP(2 * h, 1, hidden_dim=h, dropout=0.1)
        self.division_head = DivisionHead(self.config)

    def forward(self, batch: ReconciliationBatch) -> ReconciliationOutput:
        batch.validate()
        encoded = self.tracklets(batch.tracklets)
        edge = batch.edges
        src = edge.edge_index[..., 0]
        dst = edge.edge_index[..., 1]
        src_tail = _gather(encoded.tail, src)
        dst_head = _gather(encoded.head, dst)
        source_end, target_start = gather_edge_endpoints(
            batch.tracklets.start_xyz_um,
            batch.tracklets.end_xyz_um,
            edge.edge_index,
        )
        relation = build_motion_relation_features(
            source_end_xyz_um=source_end,
            target_start_xyz_um=target_start,
            expected_global_xyz_um=edge.expected_global_xyz_um,
            expected_global_relative_xyz_um=edge.expected_global_relative_xyz_um,
            expected_local_xyz_um=edge.expected_local_xyz_um,
            expected_backward_source_xyz_um=edge.expected_backward_source_xyz_um,
            prediction_valid=edge.prediction_valid,
            gap_frames=edge.gap_frames,
            distance_scale_um=self.config.edge.physical_distance_scale_um,
        )
        if edge.pair_features.shape[-1] != self.config.edge.pair_feature_dim:
            raise ValueError(
                f"pair feature dim must be {self.config.edge.pair_feature_dim}, "
                f"got {edge.pair_features.shape[-1]}"
            )
        raw_pair = self.pair_drop(torch.cat((edge.pair_features, relation), dim=-1))
        contextual = self.gather(torch.cat((src_tail, dst_head), dim=-1))
        primitive = self.relation(raw_pair)
        contextual = contextual * edge.edge_mask.unsqueeze(-1).to(contextual.dtype)
        primitive = primitive * edge.edge_mask.unsqueeze(-1).to(primitive.dtype)

        refined = self.edge_reasoner(
            primitive, contextual, edge.edge_mask, source_end, target_start
        )
        continuation_logits = self.continuation_head(refined).squeeze(-1)
        continuation_logits = continuation_logits.masked_fill(~edge.edge_mask, 0.0)
        parental_p, no_parent_p = parental_softmax(
            continuation_logits,
            dst,
            edge.gap_frames,
            edge.edge_mask,
        )

        # Per-tracklet event priors use both endpoint-local and pooled history.
        division_prior = self.division_prior_head(
            torch.cat((encoded.tail, encoded.pooled), dim=-1)
        ).squeeze(-1)
        appearance = self.appearance_head(
            torch.cat((encoded.head, encoded.pooled), dim=-1)
        ).squeeze(-1)
        termination = self.termination_head(
            torch.cat((encoded.tail, encoded.pooled), dim=-1)
        ).squeeze(-1)
        track_mask = batch.tracklets.tracklet_mask
        division_prior = division_prior.masked_fill(~track_mask, 0.0)
        appearance = appearance.masked_fill(~track_mask, 0.0)
        termination = termination.masked_fill(~track_mask, 0.0)

        division_logits = None
        if batch.divisions is not None:
            division_logits = self.division_head(
                refined,
                edge.edge_index,
                encoded.tail,
                division_prior,
                batch.divisions,
            )

        return ReconciliationOutput(
            tracklets=encoded,
            edge_embeddings=refined,
            continuation_logits=continuation_logits,
            parental_probabilities=parental_p,
            no_parent_probability=no_parent_p,
            division_prior_logits=division_prior,
            division_logits=division_logits,
            appearance_logits=appearance,
            termination_logits=termination,
        )
