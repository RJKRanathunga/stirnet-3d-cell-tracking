from __future__ import annotations

import numpy as np
import torch

from learned.stirnet.data.augmentation import corrupt_temporal_clues, random_flip_sample
from learned.stirnet.data.graph_builder import AssociationRecord, DetectionRecord, build_temporal_graph
from learned.stirnet.data.historical_instances import (
    build_historical_instance_grid,
    component_overlap_from_projected_support,
    pairwise_convergence_statistics,
    select_nearest_history_support,
)
from learned.stirnet.model.config import HistoryConfig, StirNetConfig
from learned.stirnet.model.history_encoder import HistoricalInstanceEncoder, HistoryFusion
from learned.stirnet.model.history_support import sample_projected_history_support
from learned.stirnet.model.stir_net import StirNet
from learned.stirnet.training.checkpoint import migrate_history_checkpoint_state_dict
from learned.stirnet.training.curriculum import model_parameter_groups


def _sphere(spacing: tuple[float, float, float], extent_um: float = 32.0):
    shape = tuple(int(round(extent_um / value)) + 1 for value in spacing)
    axes = [np.arange(size) * step for size, step in zip(shape, spacing)]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    center = np.array([extent_um / 2] * 3)
    mask = (zz-center[0])**2+(yy-center[1])**2+(xx-center[2])**2 <= 5.0**2
    labels = mask.astype(np.int32)
    raw = np.exp(-((zz-center[0])**2+(yy-center[1])**2+(xx-center[2])**2)/50).astype(np.float32)
    return raw, labels, center


def _support(centers=(-1.0, 1.0), grid_size=12):
    support = torch.zeros((2, 2, 2, grid_size, grid_size, grid_size), dtype=torch.float32)
    support[:, 0, 0, 3:9, 3:9, 3:9] = 1
    support[:, 0, 1] = support[:, 0, 0] * 0.5
    return {
        "history_support": support,
        "history_support_valid": torch.tensor([[True, False], [True, False]]),
        "history_support_dt": torch.tensor([[-1.0, 0.0], [-1.0, 0.0]]),
        "history_support_center_um": torch.tensor([[[0.0, 0.0, centers[0]], [0.0]*3], [[0.0, 0.0, centers[1]], [0.0]*3]]),
        "history_support_extent_um": torch.full((2, 2), 10.0),
    }


def test_historical_grid_shape_channels_and_physical_invariance() -> None:
    grids = []
    for spacing in ((1.0, 1.0, 1.0), (0.5, 0.5, 0.5)):
        raw, labels, center = _sphere(spacing)
        grid, valid = build_historical_instance_grid(
            raw, labels, 1, spacing, 8.0, center_um=center
        )
        assert valid and grid.shape == (4, 12, 12, 12)
        assert grid[1].min() >= -1 and grid[1].max() <= 1
        assert grid[1, 6, 6, 6] > 0
        grids.append(grid)
    # Boundary voxels differ under native rasterization, but the physical
    # descriptor as a whole remains stable.
    assert (grids[0][0]-grids[1][0]).abs().mean() < 0.03


def test_invalid_history_is_exactly_zero_and_gate_has_finite_gradients() -> None:
    torch.manual_seed(2)
    encoder = HistoricalInstanceEncoder(HistoryConfig(), d_model=32)
    fusion = HistoryFusion(32, -2.0)
    grid = torch.randn(3, 4, 12, 12, 12, requires_grad=True)
    valid = torch.tensor([True, False, True])
    history = encoder(grid, valid)
    assert torch.count_nonzero(history[1]) == 0
    scalar = torch.randn(3, 32, requires_grad=True)
    fused, gate = fusion(scalar, history, valid)
    torch.testing.assert_close(fused[1], scalar[1])
    assert gate.shape == (3, 1) and gate[1].item() == 0
    fused.square().mean().backward()
    assert grid.grad is not None and torch.isfinite(grid.grad).all()


def test_history_encoder_chunked_and_unchunked_eval_agree() -> None:
    torch.manual_seed(3)
    encoder = HistoricalInstanceEncoder(HistoryConfig(node_chunk_size=2), d_model=32).eval()
    grid = torch.randn(5, 4, 12, 12, 12)
    valid = torch.tensor([True, True, False, True, True])
    chunked = encoder(grid, valid, chunk_size=2)
    unchunked = encoder(grid, valid, chunk_size=32)
    torch.testing.assert_close(chunked, unchunked)


def test_translational_support_projection_samples_projected_center() -> None:
    support = torch.zeros((1, 2, 2, 5, 5, 5))
    support[0, 0, 0, 2, 2, 2] = 1
    sampled = sample_projected_history_support(
        support,
        torch.tensor([[True, False]]),
        torch.tensor([[[2.0, 3.0, 4.0], [0.0, 0.0, 0.0]]]),
        torch.tensor([[7.0, 8.0, 9.0]]),
        torch.full((1, 2), 4.0),
        torch.tensor([[7.0, 8.0, 9.0]]),
    )
    torch.testing.assert_close(sampled[0, 0, 0, 0], torch.tensor(1.0))
    assert torch.count_nonzero(sampled[0, 0, 1]) == 0


def test_nearest_past_and_future_support_are_selected() -> None:
    records = [
        DetectionRecord(i, t, (float(t), 0, 0), 1.0)
        for i, t in enumerate((-2, -1, 1, 2))
    ]
    grid = torch.stack([torch.full((4, 4, 4, 4), float(i)) for i in range(4)])
    selected = select_nearest_history_support(
        records, np.zeros(4, np.int64), grid, torch.ones(4, dtype=torch.bool),
        n_tracklets=1, dref_um=2.0,
    )
    assert selected["history_support_dt"].tolist() == [[-1.0, 1.0]]
    assert selected["history_support"][0, 0, 0, 0, 0, 0].item() == 1
    assert selected["history_support"][0, 1, 0, 0, 0, 0].item() == 2


def test_projected_mask_assigns_component_when_reference_is_outside() -> None:
    labels = np.zeros((21, 21, 21), np.int32)
    labels[12:14, 9:12, 9:12] = 7
    support = torch.zeros((1, 2, 2, 5, 5, 5))
    support[0, 0, 0, 4, 1:4, 1:4] = 1
    best_id, best, second, available = component_overlap_from_projected_support(
        support, torch.tensor([[True, False]]), np.zeros((1, 3), np.float32),
        torch.full((1, 2), 4.0), labels, (1.0, 1.0, 1.0),
    )
    assert available[0] and best_id[0] == 7 and best[0] > 0
    assert second[0] == 0 and labels[10, 10, 10] == 0


def test_hypothesis_edge_schema_direction_and_convergence() -> None:
    records = [
        DetectionRecord(0, -2, (0, 0, -3), 10),
        DetectionRecord(1, -1, (0, 0, -1), 10),
        DetectionRecord(2, -2, (0, 0, 3), 10),
        DetectionRecord(3, -1, (0, 0, 1), 10),
    ]
    associations = [AssociationRecord(0, 1), AssociationRecord(2, 3)]
    grid = torch.zeros((4, 4, 12, 12, 12)); grid[:, 0, 3:9, 3:9, 3:9] = 1
    graph = build_temporal_graph(
        records, associations, dref_um=2.0,
        node_instance_grid=grid, node_history_valid=torch.ones(4, dtype=torch.bool),
    )
    edge = graph["hypothesis_edge_attr"]
    assert edge.shape == (2, 22)
    torch.testing.assert_close(edge[0, :3], -edge[1, :3])
    torch.testing.assert_close(edge[0, 12:15], -edge[1, 12:15])
    torch.testing.assert_close(edge[0, 3:12], edge[1, 3:12])
    assert edge[0, 10] < 0 and edge[0, 11] > 0
    assert edge[0, 19:21].tolist() == [1.0, 1.0]
    direct = pairwise_convergence_statistics(records[:2], records[2:], 2.0)
    assert direct["closing_speed"] > 0


def test_flip_and_hypothesis_corruption_keep_history_aligned(monkeypatch) -> None:
    sample = {
        "spatial_inputs":torch.zeros((5,4,4,4)),"instance_labels":torch.zeros((4,4,4),dtype=torch.long),
        "dref_um":torch.tensor(2.0),"temporal_ref_um":torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]]),
        "temporal_status":torch.zeros((2,10)),"tracklet_id":torch.tensor([0,1]),
        "graph_x":torch.zeros((2,32)),"graph_edge_index":torch.zeros((2,0),dtype=torch.long),"graph_edge_attr":torch.zeros((0,14)),
        "hypothesis_edge_index":torch.tensor([[0,1],[1,0]]),"hypothesis_edge_attr":torch.zeros((2,22)),
        "node_instance_grid":torch.arange(2*4*3*3*3).reshape(2,4,3,3,3).float(),"node_history_valid":torch.ones(2,dtype=torch.bool),
        **_support(grid_size=3),
        "best_current_component_id":torch.tensor([1,1]),"best_component_overlap":torch.ones(2),"second_best_component_overlap":torch.zeros(2),
    }
    monkeypatch.setattr("random.random",lambda:0.0)
    flipped=random_flip_sample(sample,p=1.0)
    assert flipped["temporal_ref_um"][0].tolist()==[-1.0,-2.0,-3.0]
    assert torch.equal(flipped["node_instance_grid"],torch.flip(sample["node_instance_grid"],[2,3,4]))
    assert torch.equal(flipped["history_support"],torch.flip(sample["history_support"],[3,4,5]))
    corrupted=corrupt_temporal_clues(
        sample,hypothesis_dropout=0,edge_dropout=0,position_jitter_dref=0,
        large_jitter_dref=0,false_clue_prob=1,
    )
    assert len(corrupted["temporal_ref_um"])==3
    assert not corrupted["history_support_valid"][-1].any()
    assert torch.count_nonzero(corrupted["history_support"][-1])==0


def test_history_disabled_model_ignores_history_tensors() -> None:
    cfg = StirNetConfig()
    cfg.history.enabled = False
    cfg.spatial.channels=(4,8,8,16);cfg.spatial.blocks_per_level=1;cfg.spatial.mask_dim=4
    cfg.temporal.d_model=16;cfg.temporal.graph_ffn_dim=32
    cfg.coreasoning.d_model=16;cfg.coreasoning.dropout=0
    cfg.queries.d_model=16;cfg.queries.discovery_queries=1
    cfg.decoder.d_model=16;cfg.decoder.ffn_dim=32;cfg.decoder.mask_dim=4;cfg.decoder.dropout=0;cfg.decoder.max_spatial_tokens=64
    cfg.training.activation_checkpointing=False
    model=StirNet(cfg).eval()
    common=(torch.randn(1,5,8,16,16),torch.zeros((1,8,16,16),dtype=torch.long),torch.tensor([[1.,1.,1.]]),torch.tensor([4.]),
            torch.zeros((0,14)),torch.zeros((0,),dtype=torch.long),torch.zeros((0,),dtype=torch.long),torch.zeros((0,3)),
            torch.zeros((1,32)),torch.zeros((2,0),dtype=torch.long),torch.zeros((0,14)),torch.tensor([0]),torch.zeros((1,3)),torch.zeros((1,10)),
            torch.zeros((2,0),dtype=torch.long),torch.zeros((0,22)),torch.tensor([0]))
    with torch.no_grad():
        a=model(*common,node_instance_grid=torch.zeros((1,4,12,12,12)),node_history_valid=torch.tensor([True]))
        b=model(*common,node_instance_grid=torch.randn((1,4,12,12,12)),node_history_valid=torch.tensor([True]))
    torch.testing.assert_close(a.exist_logits,b.exist_logits)


def test_history_parameters_are_temporal_and_checkpoint_edges_migrate() -> None:
    model=StirNet()
    groups=model_parameter_groups(model)
    temporal_ids={id(p) for p in groups["temporal"]}
    assert all(id(p) in temporal_ids for p in model.history_encoder.parameters())
    assert all(id(p) in temporal_ids for p in model.history_fusion.parameters())
    state=model.state_dict()
    legacy={k:v.clone() for k,v in state.items() if not k.startswith(("history_encoder.","history_fusion.")) and ".cross.history_bias." not in k}
    edge_keys=[k for k in legacy if k.endswith("hyp_graph.block.attn.edge.weight")]
    for key in edge_keys: legacy[key]=legacy[key][:,:8].clone()
    migrated,notes=migrate_history_checkpoint_state_dict(model,legacy)
    assert notes
    for key in edge_keys:
        torch.testing.assert_close(migrated[key][:,:8],legacy[key])
        assert torch.count_nonzero(migrated[key][:,8:])==0
    model.load_state_dict(migrated,strict=True)


def test_full_history_forward_and_backward_are_finite() -> None:
    torch.manual_seed(8)
    cfg=StirNetConfig()
    cfg.spatial.channels=(4,8,8,16);cfg.spatial.blocks_per_level=1;cfg.spatial.mask_dim=4
    cfg.temporal.d_model=16;cfg.temporal.graph_ffn_dim=32
    cfg.coreasoning.d_model=16;cfg.coreasoning.dropout=0
    cfg.coreasoning.temporal_query_chunk_size=1;cfg.coreasoning.spatial_key_chunk_size=64
    cfg.queries.d_model=16;cfg.queries.discovery_queries=1
    cfg.decoder.d_model=16;cfg.decoder.ffn_dim=32;cfg.decoder.mask_dim=4;cfg.decoder.dropout=0;cfg.decoder.max_spatial_tokens=64
    cfg.training.activation_checkpointing=False
    model=StirNet(cfg).train()
    grid=torch.randn((1,4,12,12,12),requires_grad=True)
    support=torch.zeros((1,2,2,12,12,12));support[0,0,0,3:9,3:9,3:9]=1;support[0,0,1]=support[0,0,0]
    output=model(
        torch.randn(1,5,8,16,16),torch.zeros((1,8,16,16),dtype=torch.long),torch.ones((1,3)),torch.tensor([4.]),
        torch.zeros((0,14)),torch.zeros((0,),dtype=torch.long),torch.zeros((0,),dtype=torch.long),torch.zeros((0,3)),
        torch.zeros((1,32)),torch.zeros((2,0),dtype=torch.long),torch.zeros((0,14)),torch.tensor([0]),
        torch.zeros((1,3)),torch.zeros((1,10)),torch.zeros((2,0),dtype=torch.long),torch.zeros((0,22)),torch.tensor([0]),
        node_instance_grid=grid,node_history_valid=torch.tensor([True]),
        history_support=support,history_support_valid=torch.tensor([[True,False]]),
        history_support_dt=torch.tensor([[-1.,0.]]),history_support_center_um=torch.zeros((1,2,3)),
        history_support_extent_um=torch.full((1,2),10.),
        return_debug=True,
    )
    loss=output.exist_logits.square().mean()+output.coarse_mask_logits.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert grid.grad is not None and torch.isfinite(grid.grad).all()
    assert output.debug is not None and "history_statistics" in output.debug
    bias_parameters=list(model.cr1.cross.history_bias.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in bias_parameters)
