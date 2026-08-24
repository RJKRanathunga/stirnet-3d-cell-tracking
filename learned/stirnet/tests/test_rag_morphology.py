# STIRNET_MORPHOLOGY_AWARE_RAG_V1
from __future__ import annotations

from dataclasses import replace

import torch

from learned.stirnet.model.config import PartitionConfig
from learned.stirnet.model.partition.graph_net import SpatialRAGNetwork
from learned.stirnet.model.partition.morphology import RAGMorphologyEmbeddingBuilder
from learned.stirnet.model.types import GeometryState, RAGState


def _synthetic_case():
    shape = (10, 16, 16)
    labels = torch.zeros(shape, dtype=torch.long)
    labels[2:8, 3:13, 2:8] = 1
    labels[2:8, 3:13, 8:14] = 2

    spatial = torch.zeros((1, 5, *shape), dtype=torch.float32)
    spatial[:, 0] = 0.15
    spatial[:, 0, 2:8, 3:13, 2:14] = 0.8

    foreground = torch.full((1, 1, *shape), -3.0)
    foreground[:, :, 2:8, 3:13, 2:14] = 4.0
    surface = torch.full_like(foreground, -2.0)
    separator = torch.full_like(foreground, -4.0)
    separator[:, :, 2:8, 3:13, 7:9] = 5.0
    sdf = torch.zeros_like(foreground)
    sdf[:, :, 2:8, 3:13, 2:14] = 0.8
    flow = torch.zeros((1, 3, *shape))
    flow[:, 2, 2:8, 3:13, 2:8] = 1.0
    flow[:, 2, 2:8, 3:13, 8:14] = -1.0
    offset = torch.zeros_like(flow)
    offset[:, 2, 2:8, 3:13, 2:8] = 0.5
    offset[:, 2, 2:8, 3:13, 8:14] = -0.5
    seed = torch.full_like(foreground, -3.0)
    seed[:, :, 4:6, 7:9, 4:6] = 4.0
    seed[:, :, 4:6, 7:9, 10:12] = 4.0

    geometry = GeometryState(
        foreground_logits=foreground,
        surface_logits=surface,
        separator_logits=separator,
        sdf=sdf,
        flow=flow,
        centroid_offset=offset,
        seed_logits=seed,
    )
    return labels, spatial, geometry


def _cfg():
    return PartitionConfig(
        rag_morphology_enabled=True,
        rag_node_morphology_dim=16,
        rag_edge_morphology_dim=16,
        rag_node_patch_shape_zyx=(8, 8, 8),
        rag_edge_patch_shape_zyx=(8, 8, 8),
        rag_morphology_chunk_size=2,
        rag_node_context_dref=0.25,
        rag_edge_radius_dref=1.25,
    )


def _rag(labels):
    return RAGState(
        node_features=torch.randn(2, 12),
        node_embeddings=torch.zeros(2, 96),
        node_batch=torch.zeros(2, dtype=torch.long),
        node_supervoxel_id=torch.tensor([1, 2]),
        node_centroid_um=torch.zeros(2, 3),
        node_volume_voxels=torch.tensor([360.0, 360.0]),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_features=torch.randn(1, 8),
        edge_embeddings=torch.zeros(1, 96),
        spatial_edge_logits=torch.zeros(1),
        edge_batch=torch.zeros(1, dtype=torch.long),
        supervoxel_labels=[labels],
        node_offsets=torch.tensor([0, 2], dtype=torch.long),
    )


def test_morphology_builder_produces_finite_aligned_embeddings():
    labels, spatial, geometry = _synthetic_case()
    cfg = _cfg()
    rag = _rag(labels)
    builder = RAGMorphologyEmbeddingBuilder(cfg)
    node, edge = builder(
        rag,
        spatial,
        geometry,
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([6.0]),
    )
    assert node.shape == (2, 16)
    assert edge.shape == (1, 16)
    assert torch.isfinite(node).all()
    assert torch.isfinite(edge).all()


def test_zero_initialized_morphology_residual_preserves_legacy_logits():
    labels, spatial, geometry = _synthetic_case()
    cfg = _cfg()
    rag = _rag(labels)
    builder = RAGMorphologyEmbeddingBuilder(cfg)
    node, edge = builder(
        rag,
        spatial,
        geometry,
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([6.0]),
    )
    rag = replace(
        rag,
        node_morphology_embeddings=node,
        edge_morphology_embeddings=edge,
    )
    network = SpatialRAGNetwork(cfg, node_in_dim=12, edge_in_dim=8)
    # The legacy RAG contains dropout in its message block.  This test checks
    # architectural identity, not stochastic-training equivalence, so evaluate
    # deterministically.  With zero morphology projections, both paths must then
    # be bit-identical.
    network.eval()
    assert torch.count_nonzero(network.node_morphology_projection.weight) == 0
    assert torch.count_nonzero(network.edge_morphology_projection.weight) == 0

    with_morph = network(rag)
    without_morph = network(
        replace(
            rag,
            node_morphology_embeddings=torch.zeros_like(node),
            edge_morphology_embeddings=torch.zeros_like(edge),
        )
    )
    torch.testing.assert_close(
        with_morph.spatial_edge_logits,
        without_morph.spatial_edge_logits,
        rtol=0,
        atol=0,
    )


def test_morphology_branch_receives_gradient_after_residual_projection_opens():
    labels, spatial, geometry = _synthetic_case()
    cfg = _cfg()
    rag = _rag(labels)
    builder = RAGMorphologyEmbeddingBuilder(cfg)
    node, edge = builder(
        rag,
        spatial,
        geometry,
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([6.0]),
    )
    rag = replace(
        rag,
        node_morphology_embeddings=node,
        edge_morphology_embeddings=edge,
    )
    network = SpatialRAGNetwork(cfg, node_in_dim=12, edge_in_dim=8)
    with torch.no_grad():
        network.node_morphology_projection.weight.fill_(0.01)
        network.edge_morphology_projection.weight.fill_(0.01)

    loss = network(rag).spatial_edge_logits.square().mean()
    loss.backward()

    node_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in builder.node_encoder.parameters()
        if parameter.grad is not None
    )
    edge_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in builder.edge_encoder.parameters()
        if parameter.grad is not None
    )
    assert node_grad > 0
    assert edge_grad > 0
