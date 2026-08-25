from __future__ import annotations

import torch

from learned.stirnet.model.config import PartitionConfig
from learned.stirnet.model.partition.graph_net import SpatialRAGNetwork
from learned.stirnet.model.partition.rag import RAGTargets
from learned.stirnet.model.partition.separator_barrier import (
    SEPARATOR_BARRIER_FEATURE_DIM,
)
from learned.stirnet.model.types import RAGState
from learned.stirnet.training.config import LossConfig
from learned.stirnet.training.criterion import _separator_barrier_auxiliary


def _rag(features=None):
    return RAGState(
        node_features=torch.randn(2, 4),
        node_embeddings=torch.zeros(2, 8),
        node_batch=torch.zeros(2, dtype=torch.long),
        node_supervoxel_id=torch.tensor([1, 2]),
        node_centroid_um=torch.zeros(2, 3),
        node_volume_voxels=torch.ones(2),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_features=torch.zeros(1, 8),
        edge_embeddings=torch.zeros(1, 8),
        spatial_edge_logits=torch.zeros(1),
        edge_batch=torch.zeros(1, dtype=torch.long),
        supervoxel_labels=[torch.tensor([[[1, 1, 2, 2]]])],
        node_offsets=torch.tensor([0, 2], dtype=torch.long),
        separator_barrier_features=features,
    )


def test_barrier_can_only_lower_merge_logit():
    cfg = PartitionConfig()
    cfg.rag_hidden_dim = 8
    cfg.rag_layers = 1
    cfg.rag_separator_barrier_enabled = True
    cfg.rag_separator_barrier_use_morphology = False
    features = torch.zeros(1, SEPARATOR_BARRIER_FEATURE_DIM)
    features[:, 0] = 0.95
    features[:, 1] = 0.99
    features[:, 4] = 1.0
    out = SpatialRAGNetwork(
        cfg, node_in_dim=4, edge_in_dim=8
    )(_rag(features))
    assert out.base_spatial_edge_logits is not None
    assert out.separator_barrier_correction is not None
    assert bool((out.separator_barrier_correction >= 0).all())
    assert bool(
        (
            out.spatial_edge_logits
            <= out.base_spatial_edge_logits + 1e-7
        ).all()
    )


def test_disabled_barrier_is_historical_path():
    cfg = PartitionConfig()
    cfg.rag_hidden_dim = 8
    cfg.rag_layers = 1
    out = SpatialRAGNetwork(
        cfg, node_in_dim=4, edge_in_dim=8
    )(_rag())
    assert torch.allclose(
        out.spatial_edge_logits,
        out.base_spatial_edge_logits,
    )
    assert out.separator_barrier_correction is None


def test_strong_separator_false_merge_gets_margin_loss():
    features = torch.zeros(1, SEPARATOR_BARRIER_FEATURE_DIM)
    features[:, 0] = 0.90
    features[:, 1] = 0.98
    features[:, 4] = 0.90
    rag = _rag(features)
    rag.spatial_edge_logits = torch.tensor([4.0], requires_grad=True)
    rag.separator_barrier_score = torch.tensor([0.0], requires_grad=True)
    rag.separator_barrier_correction = torch.tensor([0.1], requires_grad=True)
    targets = RAGTargets(
        target=torch.tensor([0.0]),
        valid=torch.tensor([True]),
        weight=torch.ones(1),
        node_purity=torch.ones(2),
        node_gt_support=torch.ones(2),
        dominant_gt=torch.tensor([1, 2]),
    )
    losses = _separator_barrier_auxiliary(
        rag,
        targets,
        LossConfig(),
        neutral_probability=0.845,
    )
    assert float(losses["separator_barrier_margin"]) > 0
    assert int(losses["separator_barrier_strong_negative_count"]) == 1
