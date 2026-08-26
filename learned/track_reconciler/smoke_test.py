"""Small CPU/GPU forward smoke test. Run from repository root."""

from __future__ import annotations

import torch

from learned.track_reconciler import (
    CandidateEdgeBatch,
    DivisionHypothesisBatch,
    ReconciliationBatch,
    ReconcilerConfig,
    TrackletBatch,
    TrackletReconciliationNetwork,
)


def make_batch(device: str = "cpu") -> ReconciliationBatch:
    cfg = ReconcilerConfig()
    b, n, k, e, d = 1, 5, 4, 6, 2
    fingerprints = torch.randn(b, n, k, cfg.fingerprint.embedding_dim, device=device)
    structured = torch.randn(b, n, k, cfg.temporal.structured_dim, device=device)
    obs_mask = torch.ones(b, n, k, dtype=torch.bool, device=device)
    track_mask = torch.ones(b, n, dtype=torch.bool, device=device)
    times = torch.arange(k, device=device).float()[None, None].expand(b, n, -1)
    start = torch.randn(b, n, 3, device=device) * 5
    end = start + torch.randn(b, n, 3, device=device)
    reliability = torch.randn(b, n, cfg.temporal.reliability_dim, device=device)
    tracklets = TrackletBatch(
        structured, obs_mask, track_mask, times, start, end, reliability,
        fingerprints=fingerprints,
    )
    edge_index = torch.tensor([[[0,1],[0,2],[1,3],[2,3],[3,4],[1,4]]], device=device)
    source_end = end[:, edge_index[0,:,0]]
    target_start = start[:, edge_index[0,:,1]]
    edges = CandidateEdgeBatch(
        edge_index=edge_index,
        edge_mask=torch.ones(b,e,dtype=torch.bool,device=device),
        gap_frames=torch.ones(b,e,dtype=torch.long,device=device),
        pair_features=torch.randn(b,e,cfg.edge.pair_feature_dim,device=device),
        expected_global_xyz_um=source_end + 0.5,
        expected_global_relative_xyz_um=source_end + 0.8,
        expected_local_xyz_um=source_end + 0.7,
        expected_backward_source_xyz_um=target_start - 0.8,
        prediction_valid=torch.ones(b,e,4,dtype=torch.bool,device=device),
    )
    divisions = DivisionHypothesisBatch(
        edge_pair_index=torch.tensor([[[0,1],[2,5]]],device=device),
        hypothesis_mask=torch.ones(b,d,dtype=torch.bool,device=device),
        features=torch.randn(b,d,cfg.division.pair_feature_dim,device=device),
    )
    return ReconciliationBatch(tracklets, edges, divisions)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrackletReconciliationNetwork().to(device)
    output = model(make_batch(device))
    print("device:", device)
    print("continuation:", tuple(output.continuation_logits.shape))
    print("division:", tuple(output.division_logits.shape) if output.division_logits is not None else None)
    print("appearance:", tuple(output.appearance_logits.shape))
