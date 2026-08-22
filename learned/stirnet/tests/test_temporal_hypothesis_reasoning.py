from __future__ import annotations

import pytest
import torch

from learned.stirnet.model.config import (
    InstanceConfig,
    PartitionConfig,
    RefinementConfig,
    TemporalConfig,
)
from learned.stirnet.model.temporal.fusion import InstanceTemporalReasoner
from learned.stirnet.model.temporal.graph_encoder import TemporalGraphEncoder
from learned.stirnet.model.types import (
    InstanceState,
    RAGState,
    TemporalInput,
    TemporalState,
)
from learned.stirnet.training.trainer import (
    MODEL_INPUT_KEYS,
    model_forward_from_batch,
)


def _temporal_input(
    *,
    hypothesis: bool = True,
    wrong_hypothesis_width: bool = False,
) -> TemporalInput:
    torch.manual_seed(7)
    width = 21 if wrong_hypothesis_width else 22
    return TemporalInput(
        graph_x=torch.randn(4, 32),
        graph_edge_index=torch.zeros((2, 0), dtype=torch.long),
        graph_edge_attr=torch.zeros((0, 15)),
        tracklet_id=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        temporal_ref_um=torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
        ),
        temporal_status=torch.zeros((2, 10)),
        temporal_batch=torch.zeros(2, dtype=torch.long),
        hypothesis_edge_index=(
            torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            if hypothesis
            else torch.zeros((2, 0), dtype=torch.long)
        ),
        hypothesis_edge_attr=(
            torch.randn(2, width)
            if hypothesis
            else torch.zeros((0, width))
        ),
    )


def test_hypothesis_graph_reaches_trainable_temporal_parameters():
    cfg = TemporalConfig(
        d_model=32,
        cross_heads=4,
        graph_hidden_dim=64,
        graph_layers=1,
        hypothesis_layers=1,
        dropout=0.0,
    )
    encoder = TemporalGraphEncoder(cfg)
    output = encoder(_temporal_input(hypothesis=True))
    assert output.tokens.shape == (2, 32)
    output.tokens.square().sum().backward()
    grads = [
        p.grad
        for p in encoder.hypothesis_blocks.parameters()
        if p.grad is not None
    ]
    assert grads
    assert any(torch.count_nonzero(g).item() > 0 for g in grads)


def test_hypothesis_graph_contract_rejects_wrong_feature_width():
    cfg = TemporalConfig(
        d_model=32,
        cross_heads=4,
        graph_hidden_dim=64,
        graph_layers=1,
        hypothesis_layers=1,
        dropout=0.0,
    )
    with pytest.raises(ValueError, match="hypothesis_edge_attr"):
        TemporalGraphEncoder(cfg)(
            _temporal_input(
                hypothesis=True,
                wrong_hypothesis_width=True,
            )
        )


def test_trainer_threads_hypothesis_graph_into_model_call():
    assert {
        "hypothesis_edge_index",
        "hypothesis_edge_attr",
    }.issubset(MODEL_INPUT_KEYS)

    class Recorder:
        def __call__(self, *args, **kwargs):
            return kwargs

    hidx = torch.tensor([[0], [1]], dtype=torch.long)
    hattr = torch.randn(1, 22)
    batch = {
        "spatial_inputs": torch.zeros((1, 5, 2, 2, 2)),
        "spacing_um": torch.ones((1, 3)),
        "dref_um": torch.ones((1,)),
        "hypothesis_edge_index": hidx,
        "hypothesis_edge_attr": hattr,
    }
    forwarded = model_forward_from_batch(
        Recorder(),
        batch,
        use_temporal=True,
        execution_stage="temporal",
    )
    assert forwarded["hypothesis_edge_index"] is hidx
    assert forwarded["hypothesis_edge_attr"] is hattr


def _reasoning_fixture():
    temporal_cfg = TemporalConfig(
        d_model=16,
        cross_heads=4,
        dropout=0.0,
        temporal_residual_scale=4.0,
    )
    reasoner = InstanceTemporalReasoner(
        temporal_cfg,
        InstanceConfig(d_model=16, dropout=0.0),
        PartitionConfig(rag_hidden_dim=8),
        RefinementConfig(),
    ).eval()
    with torch.no_grad():
        for parameter in reasoner.parameters():
            parameter.zero_()
        reasoner.edge_delta[-1].bias.fill_(-10.0)
        reasoner.edge_gate[2].bias.fill_(10.0)

    sv_labels = torch.tensor([[[1, 2]]], dtype=torch.long)
    rag = RAGState(
        node_features=torch.zeros((2, 4)),
        node_embeddings=torch.zeros((2, 8)),
        node_batch=torch.zeros(2, dtype=torch.long),
        node_supervoxel_id=torch.tensor([1, 2], dtype=torch.long),
        node_centroid_um=torch.tensor(
            [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]
        ),
        node_volume_voxels=torch.ones(2),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_features=torch.zeros((1, 2)),
        edge_embeddings=torch.zeros((1, 8)),
        spatial_edge_logits=torch.tensor([8.0]),
        edge_batch=torch.zeros(1, dtype=torch.long),
        supervoxel_labels=[sv_labels],
        node_offsets=torch.tensor([0, 2], dtype=torch.long),
    )
    instances = InstanceState(
        tokens=torch.zeros((1, 16)),
        ref_um=torch.zeros((1, 3)),
        batch_index=torch.zeros(1, dtype=torch.long),
        local_ids=torch.ones(1, dtype=torch.long),
        quality_logits=torch.zeros(1),
        labels=[torch.ones_like(sv_labels)],
        token_offsets=torch.tensor([0, 1], dtype=torch.long),
        node_to_instance=torch.zeros(2, dtype=torch.long),
        spatial_tokens=torch.zeros((1, 16)),
    )
    temporal = TemporalState(
        tokens=torch.zeros((1, 16)),
        ref_um=torch.zeros((1, 3)),
        batch_index=torch.zeros(1, dtype=torch.long),
        salience=torch.ones((1, 1)),
        reliability=torch.ones((1, 1)),
        status=torch.zeros((1, 10)),
        node_tokens=torch.zeros((1, 16)),
    )
    empty_temporal = TemporalState(
        tokens=torch.zeros((0, 16)),
        ref_um=torch.zeros((0, 3)),
        batch_index=torch.zeros(0, dtype=torch.long),
        salience=torch.zeros((0, 1)),
        reliability=torch.zeros((0, 1)),
        status=torch.zeros((0, 10)),
        node_tokens=torch.zeros((0, 16)),
    )
    return reasoner, instances, rag, temporal, empty_temporal


def test_supported_temporal_candidate_can_override_confident_spatial_edge():
    reasoner, instances, rag, temporal, _ = _reasoning_fixture()
    output = reasoner(
        instances, rag, temporal, torch.tensor([1.0])
    )
    assert float(rag.spatial_edge_logits[0]) == 8.0
    assert float(output.edge_temporal_gate[0]) > 0.99
    assert float(output.final_edge_logits[0]) < -3.0


def test_zero_temporal_support_is_exact_spatial_noop():
    reasoner, instances, rag, _, empty_temporal = _reasoning_fixture()
    output = reasoner(
        instances, rag, empty_temporal, torch.tensor([1.0])
    )
    torch.testing.assert_close(
        output.final_edge_logits,
        rag.spatial_edge_logits,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.edge_temporal_gate,
        torch.zeros_like(output.edge_temporal_gate),
        rtol=0.0,
        atol=0.0,
    )
