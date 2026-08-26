import torch

from learned.track_reconciler import (
    CandidateEdgeBatch, ReconciliationBatch, ReconcilerConfig,
    TrackletBatch, TrackletReconciliationNetwork,
)


def test_model_handles_component_with_no_candidate_edges():
    cfg = ReconcilerConfig()
    b, n, k = 1, 2, 2
    tracklets = TrackletBatch(
        structured=torch.randn(b,n,k,cfg.temporal.structured_dim),
        observation_mask=torch.ones(b,n,k,dtype=torch.bool),
        tracklet_mask=torch.ones(b,n,dtype=torch.bool),
        times=torch.arange(k).float()[None,None].expand(b,n,-1),
        start_xyz_um=torch.randn(b,n,3),
        end_xyz_um=torch.randn(b,n,3),
        reliability=torch.randn(b,n,cfg.temporal.reliability_dim),
        fingerprints=torch.randn(b,n,k,cfg.fingerprint.embedding_dim),
    )
    edges = CandidateEdgeBatch(
        edge_index=torch.empty(b,0,2,dtype=torch.long),
        edge_mask=torch.empty(b,0,dtype=torch.bool),
        gap_frames=torch.empty(b,0,dtype=torch.long),
        pair_features=torch.empty(b,0,cfg.edge.pair_feature_dim),
        expected_global_xyz_um=torch.empty(b,0,3),
        expected_global_relative_xyz_um=torch.empty(b,0,3),
        expected_local_xyz_um=torch.empty(b,0,3),
        expected_backward_source_xyz_um=torch.empty(b,0,3),
        prediction_valid=torch.empty(b,0,4,dtype=torch.bool),
    )
    out = TrackletReconciliationNetwork(cfg)(ReconciliationBatch(tracklets, edges))
    assert out.continuation_logits.shape == (1,0)
    assert out.appearance_logits.shape == (1,2)
    assert out.termination_logits.shape == (1,2)
