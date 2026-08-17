from __future__ import annotations

import torch

import learned.stirnet.model.partition.rag as rag_module
import learned.stirnet.model.partition.statistics as statistics_module
from learned.stirnet.model.partition.rag import RAGBuilder
from learned.stirnet.model.partition.statistics import (
    NATIVE_FIELD_ORDER,
    build_supervoxel_statistics,
    update_supervoxel_statistics_local,
)
from learned.stirnet.model.types import GeometryState, SpatialDecodeState

from .conftest import small_model_config


def _field(shape, channels):
    return torch.randn((1, channels, *shape))


def _sort_edges(rag):
    n = max(int(rag.node_supervoxel_id.numel()), 1)
    if rag.edge_index.shape[1] == 0:
        return rag.edge_index, rag.edge_features
    packed = rag.edge_index[0] * n + rag.edge_index[1]
    order = torch.argsort(packed)
    return rag.edge_index[:, order], rag.edge_features[order]


def _assert_statistics_close(local, full):
    torch.testing.assert_close(local.counts, full.counts)
    torch.testing.assert_close(local.coordinate_sums, full.coordinate_sums)
    torch.testing.assert_close(
        local.coordinate_square_sums,
        full.coordinate_square_sums,
    )
    torch.testing.assert_close(local.min_voxel, full.min_voxel)
    torch.testing.assert_close(local.max_voxel, full.max_voxel)
    torch.testing.assert_close(
        local.sdf_argmax_flat_index,
        full.sdf_argmax_flat_index,
    )
    for name in NATIVE_FIELD_ORDER:
        torch.testing.assert_close(
            local.field_sums[name],
            full.field_sums[name],
        )
        torch.testing.assert_close(
            local.field_maxima[name],
            full.field_maxima[name],
        )
    for local_scale, full_scale in zip(local.scales, full.scales):
        torch.testing.assert_close(local_scale.counts, full_scale.counts)
        torch.testing.assert_close(local_scale.sums, full_scale.sums)
        torch.testing.assert_close(local_scale.maxima, full_scale.maxima)


def test_cached_local_rag_refresh_keeps_distant_clusters_independent(monkeypatch):
    torch.manual_seed(23)
    cfg = small_model_config()
    builder = RAGBuilder(cfg.partition, cfg.spatial)
    shape = (4, 16, 16)

    initial_labels = torch.zeros(shape, dtype=torch.long)
    initial_labels[:, 1:7, 1:7] = 1
    initial_labels[:, 9:15, 9:15] = 2

    updated_labels = initial_labels.clone()
    updated_labels[:, 1:7, 1:4] = 3
    updated_labels[:, 9:15, 9:12] = 4
    box_left = (slice(0, 4), slice(1, 7), slice(1, 4))
    box_right = (slice(0, 4), slice(9, 15), slice(9, 12))
    updated_boxes = [(0, box_left), (0, box_right)]

    geometry = GeometryState(
        foreground_logits=_field(shape, 1),
        surface_logits=_field(shape, 1),
        separator_logits=_field(shape, 1),
        sdf=_field(shape, 1),
        flow=_field(shape, 3),
        centroid_offset=_field(shape, 3),
        seed_logits=_field(shape, 1),
        features=None,
    )
    spatial_inputs = torch.randn(
        (1, cfg.spatial.in_channels, *shape)
    )
    d0 = torch.randn((1, cfg.spatial.channels[0], *shape))
    d1 = torch.randn((1, cfg.spatial.channels[1], 2, 8, 8))
    d2 = torch.randn((1, cfg.spatial.channels[2], 1, 4, 4))
    decoded = SpatialDecodeState(d2=d2, d1=d1, d0=d0)
    spacing = torch.tensor([[1.5, 0.4, 0.4]])
    dref = torch.tensor([3.0])

    initial_stats = build_supervoxel_statistics(
        [initial_labels],
        spatial_inputs,
        geometry,
        spacing,
        (d0, d1, d2),
    )
    initial_rag = builder(
        [initial_labels],
        d0,
        spatial_inputs,
        geometry,
        spacing,
        dref,
        statistics_by_batch=initial_stats,
    )
    assert initial_rag.statistics is not None

    reduction_shapes = []
    original_reduce = statistics_module.reduce_labeled_voxels

    def recording_reduce(labels, *args, **kwargs):
        reduction_shapes.append(tuple(labels.shape))
        return original_reduce(labels, *args, **kwargs)

    monkeypatch.setattr(
        statistics_module,
        "reduce_labeled_voxels",
        recording_reduce,
    )

    edge_shapes = []
    original_edges = rag_module._adjacent_pairs_and_stats

    def recording_edges(labels, *args, **kwargs):
        edge_shapes.append(tuple(labels.shape))
        return original_edges(labels, *args, **kwargs)

    monkeypatch.setattr(
        rag_module,
        "_adjacent_pairs_and_stats",
        recording_edges,
    )

    local = builder.update_local(
        initial_rag,
        [updated_labels],
        updated_boxes,
        decoded,
        spatial_inputs,
        geometry,
        spacing,
        dref,
    )

    # One native reduction and one incident-edge scan per distant cluster.
    assert len(reduction_shapes) == 2
    assert len(edge_shapes) == 2
    assert all(shape[1] < 15 or shape[2] < 15 for shape in reduction_shapes)
    assert all(shape[1] < 16 or shape[2] < 16 for shape in edge_shapes)

    local_stats = local.statistics
    assert local_stats is not None

    # Stop recording before the full reference rebuild.
    monkeypatch.setattr(
        statistics_module,
        "reduce_labeled_voxels",
        original_reduce,
    )
    monkeypatch.setattr(
        rag_module,
        "_adjacent_pairs_and_stats",
        original_edges,
    )

    full_stats = build_supervoxel_statistics(
        [updated_labels],
        spatial_inputs,
        geometry,
        spacing,
        (d0, d1, d2),
    )
    full = builder(
        [updated_labels],
        d0,
        spatial_inputs,
        geometry,
        spacing,
        dref,
        statistics_by_batch=full_stats,
    )

    _assert_statistics_close(local_stats[0], full_stats[0])
    torch.testing.assert_close(local.node_features, full.node_features)
    torch.testing.assert_close(
        local.node_centroid_um,
        full.node_centroid_um,
    )
    torch.testing.assert_close(
        local.node_volume_voxels,
        full.node_volume_voxels,
    )
    local_edges, local_features = _sort_edges(local)
    full_edges, full_features = _sort_edges(full)
    assert torch.equal(local_edges, full_edges)
    torch.testing.assert_close(local_features, full_features)


def test_local_statistics_clear_disappeared_highest_label():
    torch.manual_seed(29)
    cfg = small_model_config()
    shape = (3, 8, 8)
    initial_labels = torch.zeros(shape, dtype=torch.long)
    initial_labels[:, 1:4, 1:4] = 1
    initial_labels[:, 4:7, 4:7] = 2
    updated_labels = initial_labels.clone()
    updated_labels[updated_labels == 2] = 1
    box = (slice(0, 3), slice(4, 7), slice(4, 7))

    geometry = GeometryState(
        foreground_logits=_field(shape, 1),
        surface_logits=_field(shape, 1),
        separator_logits=_field(shape, 1),
        sdf=_field(shape, 1),
        flow=_field(shape, 3),
        centroid_offset=_field(shape, 3),
        seed_logits=_field(shape, 1),
        features=None,
    )
    spatial_inputs = torch.randn(
        (1, cfg.spatial.in_channels, *shape)
    )
    d0 = torch.randn((1, cfg.spatial.channels[0], *shape))
    d1 = torch.randn((1, cfg.spatial.channels[1], 2, 4, 4))
    d2 = torch.randn((1, cfg.spatial.channels[2], 1, 2, 2))
    spacing = torch.tensor([[1.5, 0.4, 0.4]])

    initial_stats = build_supervoxel_statistics(
        [initial_labels],
        spatial_inputs,
        geometry,
        spacing,
        (d0, d1, d2),
    )
    local_stats, _ = update_supervoxel_statistics_local(
        initial_stats,
        [initial_labels],
        [updated_labels],
        [(0, box)],
        spatial_inputs,
        geometry,
        spacing,
        (d0, d1, d2),
    )
    full_stats = build_supervoxel_statistics(
        [updated_labels],
        spatial_inputs,
        geometry,
        spacing,
        (d0, d1, d2),
    )
    _assert_statistics_close(local_stats[0], full_stats[0])
