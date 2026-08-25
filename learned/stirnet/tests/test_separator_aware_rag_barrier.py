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

def test_rag_builder_attaches_separator_features_before_graph_network():
    """Regression test for the real production RAGBuilder -> graph-net path."""
    from learned.stirnet.model.config import PartitionConfig, SpatialConfig
    from learned.stirnet.model.partition.rag import RAGBuilder
    from learned.stirnet.model.types import GeometryState, RAGState

    cfg = PartitionConfig()
    cfg.rag_hidden_dim = 8
    cfg.rag_layers = 1
    cfg.rag_morphology_enabled = False
    cfg.rag_separator_barrier_enabled = True
    cfg.rag_separator_barrier_use_morphology = False

    spatial_cfg = SpatialConfig()
    builder = RAGBuilder(cfg, spatial_cfg)

    labels = torch.tensor(
        [[[1, 1, 2, 2], [1, 1, 2, 2]]],
        dtype=torch.long,
    )
    rag = RAGState(
        node_features=torch.zeros(2, builder.node_feature_dim),
        node_embeddings=torch.zeros(2, cfg.rag_hidden_dim),
        node_batch=torch.zeros(2, dtype=torch.long),
        node_supervoxel_id=torch.tensor([1, 2], dtype=torch.long),
        node_centroid_um=torch.zeros(2, 3),
        node_volume_voxels=torch.ones(2),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_features=torch.zeros(1, builder.edge_feature_dim),
        edge_embeddings=torch.zeros(1, cfg.rag_hidden_dim),
        spatial_edge_logits=torch.zeros(1),
        edge_batch=torch.zeros(1, dtype=torch.long),
        supervoxel_labels=[labels],
        node_offsets=torch.tensor([0, 2], dtype=torch.long),
    )

    shape = (1, 1, 1, 2, 4)
    separator_probability = torch.zeros(shape)
    separator_probability[..., 1:3] = 0.95
    surface_probability = torch.zeros(shape)
    foreground_probability = torch.ones(shape)
    seed_probability = torch.zeros(shape)

    geometry = GeometryState(
        foreground_logits=torch.logit(
            foreground_probability.clamp(1e-4, 1 - 1e-4)
        ),
        surface_logits=torch.logit(
            surface_probability.clamp(1e-4, 1 - 1e-4)
        ),
        separator_logits=torch.logit(
            separator_probability.clamp(1e-4, 1 - 1e-4)
        ),
        sdf=torch.zeros(shape),
        flow=torch.zeros(1, 3, 1, 2, 4),
        centroid_offset=torch.zeros(1, 3, 1, 2, 4),
        seed_logits=torch.logit(
            seed_probability.clamp(1e-4, 1 - 1e-4)
        ),
        features=None,
    )

    attached = builder._attach_morphology(
        rag,
        torch.zeros(1, spatial_cfg.in_channels, 1, 2, 4),
        geometry,
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([2.0]),
    )

    assert attached.separator_barrier_features is not None
    assert attached.separator_barrier_features.shape == (
        1,
        SEPARATOR_BARRIER_FEATURE_DIM,
    )
    assert float(attached.separator_barrier_features[0, 0]) > 0.9
