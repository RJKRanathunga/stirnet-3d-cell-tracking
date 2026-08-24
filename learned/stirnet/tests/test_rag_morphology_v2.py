from __future__ import annotations

from dataclasses import replace

import torch

from learned.stirnet.model.config import PartitionConfig
from learned.stirnet.model.partition.graph_net import SpatialRAGNetwork
from learned.stirnet.model.partition.morphology import (
    RAGMorphologyEmbeddingBuilder,
)
from learned.stirnet.model.types import GeometryState, RAGState


def _case():
    shape = (12, 20, 22)
    labels = torch.zeros(shape, dtype=torch.long)
    labels[2:10, 4:16, 2:11] = 1
    labels[2:10, 4:16, 11:20] = 2

    spatial = torch.zeros((1, 5, *shape), dtype=torch.float32)
    spatial[:, 0] = 0.1
    spatial[:, 0, 2:10, 4:16, 2:20] = 0.8

    foreground = torch.full((1, 1, *shape), -3.0)
    foreground[:, :, 2:10, 4:16, 2:20] = 4.0
    surface = torch.full_like(foreground, -2.0)
    separator = torch.full_like(foreground, -4.0)
    separator[:, :, 2:10, 4:16, 10:12] = 5.0
    sdf = torch.zeros_like(foreground)
    sdf[:, :, 2:10, 4:16, 2:20] = 0.8

    flow = torch.zeros((1, 3, *shape), dtype=torch.float32)
    flow[:, 2, 2:10, 4:16, 2:11] = 1.0
    flow[:, 2, 2:10, 4:16, 11:20] = -1.0

    offset = torch.zeros_like(flow)
    offset[:, 2, 2:10, 4:16, 2:11] = 0.5
    offset[:, 2, 2:10, 4:16, 11:20] = -0.5

    seed = torch.full_like(foreground, -3.0)
    seed[:, :, 5:7, 8:10, 5:7] = 4.0
    seed[:, :, 5:7, 8:10, 15:17] = 4.0

    geometry = GeometryState(
        foreground_logits=foreground,
        surface_logits=surface,
        separator_logits=separator,
        sdf=sdf,
        flow=flow,
        centroid_offset=offset,
        seed_logits=seed,
    )

    rag = RAGState(
        node_features=torch.randn(2, 12),
        node_embeddings=torch.zeros(2, 96),
        node_batch=torch.zeros(2, dtype=torch.long),
        node_supervoxel_id=torch.tensor([1, 2]),
        node_centroid_um=torch.zeros(2, 3),
        node_volume_voxels=torch.tensor([864.0, 864.0]),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_features=torch.randn(1, 8),
        edge_embeddings=torch.zeros(1, 96),
        spatial_edge_logits=torch.zeros(1),
        edge_batch=torch.zeros(1, dtype=torch.long),
        supervoxel_labels=[labels],
        node_offsets=torch.tensor([0, 2], dtype=torch.long),
    )
    return labels, spatial, geometry, rag


def _cfg():
    return PartitionConfig(
        rag_morphology_enabled=True,
        rag_node_morphology_dim=16,
        rag_edge_morphology_dim=16,
        rag_node_patch_shape_zyx=(12, 12, 12),
        rag_edge_patch_shape_zyx=(12, 12, 12),
        rag_node_context_dref=0.25,
        rag_edge_contact_headroom_fraction=1.0,
        rag_edge_local_headroom_fraction=0.5,
        rag_edge_min_headroom_dref=0.5,
        rag_morphology_chunk_size=2,
    )


def test_contact_bbox_contains_complete_touch_set_and_large_headroom():
    labels, spatial, geometry, rag = _case()
    builder = RAGMorphologyEmbeddingBuilder(_cfg())

    patch = builder.edge_debug_patch(
        rag,
        0,
        spatial,
        geometry,
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([4.0]),
    )

    assert torch.equal(
        patch.contact_lower_zyx,
        torch.tensor([2, 4, 10]),
    )
    assert torch.equal(
        patch.contact_upper_zyx,
        torch.tensor([9, 15, 11]),
    )

    contact_extent = (
        patch.contact_upper_zyx
        - patch.contact_lower_zyx
        + 1
    ).float()

    broad_low = (
        patch.contact_lower_zyx
        - patch.broad.requested_start_zyx
    ).float()
    broad_high = (
        patch.broad.requested_stop_zyx
        - 1
        - patch.contact_upper_zyx
    ).float()
    local_low = (
        patch.contact_lower_zyx
        - patch.local.requested_start_zyx
    ).float()
    local_high = (
        patch.local.requested_stop_zyx
        - 1
        - patch.contact_upper_zyx
    ).float()

    assert torch.all(
        broad_low >= contact_extent
    )
    assert torch.all(
        broad_high >= contact_extent
    )
    assert torch.all(
        local_low >= 0.5 * contact_extent
    )
    assert torch.all(
        local_high >= 0.5 * contact_extent
    )


def test_joint_edge_topology_keeps_A_and_B_separate_in_same_roi():
    labels, spatial, geometry, rag = _case()
    builder = RAGMorphologyEmbeddingBuilder(_cfg())

    patch = builder.edge_debug_patch(
        rag,
        0,
        spatial,
        geometry,
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([4.0]),
    )

    topology = patch.broad.topology
    assert topology.shape == (5, 12, 12, 12)

    mask_a = topology[0] > 0.5
    mask_b = topology[1] > 0.5
    union = topology[2] > 0.5
    interface = topology[3] > 0.0
    proximity = topology[4] > 0.0

    assert mask_a.any()
    assert mask_b.any()
    assert not torch.equal(mask_a, mask_b)
    assert torch.equal(union, mask_a | mask_b)
    assert interface.any()
    assert proximity.any()
    assert torch.all(proximity | ~interface)


def test_edge_embedding_is_invariant_to_swapping_A_and_B():
    labels, spatial, geometry, rag = _case()
    builder = RAGMorphologyEmbeddingBuilder(_cfg())
    builder.eval()

    spacing = torch.tensor([[1.0, 1.0, 1.0]])
    dref = torch.tensor([4.0])

    _, edge_ab = builder(
        rag,
        spatial,
        geometry,
        spacing,
        dref,
    )

    swapped_rag = replace(
        rag,
        edge_index=torch.tensor(
            [[1], [0]],
            dtype=torch.long,
        ),
    )
    _, edge_ba = builder(
        swapped_rag,
        spatial,
        geometry,
        spacing,
        dref,
    )

    torch.testing.assert_close(
        edge_ab,
        edge_ba,
        rtol=1e-6,
        atol=1e-6,
    )


def test_topology_and_multiscale_fusion_receive_gradient():
    labels, spatial, geometry, rag = _case()
    cfg = _cfg()
    builder = RAGMorphologyEmbeddingBuilder(cfg)

    node, edge = builder(
        rag,
        spatial,
        geometry,
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([4.0]),
    )
    rag = replace(
        rag,
        node_morphology_embeddings=node,
        edge_morphology_embeddings=edge,
    )

    network = SpatialRAGNetwork(
        cfg,
        node_in_dim=12,
        edge_in_dim=8,
    )
    with torch.no_grad():
        network.node_morphology_projection.weight.fill_(0.01)
        network.edge_morphology_projection.weight.fill_(0.01)

    loss = network(rag).spatial_edge_logits.square().mean()
    loss.backward()

    member_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in builder.edge_encoder.member.parameters()
        if parameter.grad is not None
    )
    relation_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in builder.edge_encoder.relation.parameters()
        if parameter.grad is not None
    )
    gate_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in builder.edge_encoder.topology_gate.parameters()
        if parameter.grad is not None
    )
    scale_fusion_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in builder.edge_scale_fusion.parameters()
        if parameter.grad is not None
    )

    assert member_grad > 0
    assert relation_grad > 0
    assert gate_grad > 0
    assert scale_fusion_grad > 0


def test_zero_morphology_projection_still_preserves_legacy_rag_identity():
    labels, spatial, geometry, rag = _case()
    cfg = _cfg()
    builder = RAGMorphologyEmbeddingBuilder(cfg)

    node, edge = builder(
        rag,
        spatial,
        geometry,
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([4.0]),
    )
    rag = replace(
        rag,
        node_morphology_embeddings=node,
        edge_morphology_embeddings=edge,
    )

    network = SpatialRAGNetwork(
        cfg,
        node_in_dim=12,
        edge_in_dim=8,
    )
    network.eval()

    assert torch.count_nonzero(
        network.node_morphology_projection.weight
    ) == 0
    assert torch.count_nonzero(
        network.edge_morphology_projection.weight
    ) == 0

    with_morph = network(rag)
    zero_morph = network(
        replace(
            rag,
            node_morphology_embeddings=torch.zeros_like(node),
            edge_morphology_embeddings=torch.zeros_like(edge),
        )
    )
    torch.testing.assert_close(
        with_morph.spatial_edge_logits,
        zero_morph.spatial_edge_logits,
        rtol=0,
        atol=0,
    )
