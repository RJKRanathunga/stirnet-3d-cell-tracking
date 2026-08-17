from __future__ import annotations

import torch

import learned.stirnet.model.partition.statistics as statistics_module
from learned.stirnet.model.partition.local_update import _actual_change_metadata
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


def _sort_edges(rag):
    n = max(int(rag.node_supervoxel_id.numel()), 1)
    if rag.edge_index.shape[1] == 0:
        return rag.edge_index, rag.edge_features
    packed = rag.edge_index[0] * n + rag.edge_index[1]
    order = torch.argsort(packed)
    return rag.edge_index[:, order], rag.edge_features[order]


def test_actual_change_metadata_reports_only_changed_assignments():
    parent = (slice(4, 7), slice(10, 14), slice(20, 26))
    writable = torch.zeros((3, 4, 6), dtype=torch.bool)
    writable[1, 1, 1] = True
    writable[1, 1, 4] = True
    writable[2, 2, 3] = True

    before = torch.tensor([0, 0, 9], dtype=torch.long)
    after = torch.zeros((3, 4, 6), dtype=torch.long)
    after[1, 1, 1] = 7
    after[1, 1, 4] = 0
    after[2, 2, 3] = 9

    box, ids, count = _actual_change_metadata(
        before,
        after,
        writable,
        parent,
    )
    assert count == 1
    assert box == (
        slice(5, 6),
        slice(11, 12),
        slice(21, 22),
    )
    assert ids.tolist() == [7]


def test_exact_change_ids_keep_distant_statistics_refreshes_bounded(monkeypatch):
    torch.manual_seed(31)
    cfg = small_model_config()
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
    updated_ids = [
        torch.tensor([1, 3], dtype=torch.long),
        torch.tensor([2, 4], dtype=torch.long),
    ]

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
    spacing = torch.tensor([[1.5, 0.4, 0.4]])
    dref = torch.tensor([3.0])

    initial_stats = build_supervoxel_statistics(
        [initial_labels],
        spatial_inputs,
        geometry,
        spacing,
        (d0, d1, d2),
    )

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
    local_stats, affected = update_supervoxel_statistics_local(
        initial_stats,
        [initial_labels],
        [updated_labels],
        updated_boxes,
        spatial_inputs,
        geometry,
        spacing,
        (d0, d1, d2),
        updated_label_ids=updated_ids,
    )

    # Exactly one bounded native reduction per independent changed region.
    assert len(reduction_shapes) == 2
    assert all(spatial[1] <= 6 and spatial[2] <= 6 for spatial in reduction_shapes)
    assert affected[0].tolist() == [1, 2, 3, 4]

    monkeypatch.setattr(
        statistics_module,
        "reduce_labeled_voxels",
        original_reduce,
    )
    full_stats = build_supervoxel_statistics(
        [updated_labels],
        spatial_inputs,
        geometry,
        spacing,
        (d0, d1, d2),
    )
    _assert_statistics_close(local_stats[0], full_stats[0])

    builder = RAGBuilder(cfg.partition, cfg.spatial)
    initial_rag = builder(
        [initial_labels],
        d0,
        spatial_inputs,
        geometry,
        spacing,
        dref,
        statistics_by_batch=initial_stats,
    )
    decoded = SpatialDecodeState(d2=d2, d1=d1, d0=d0)
    local_rag = builder.update_local(
        initial_rag,
        [updated_labels],
        updated_boxes,
        decoded,
        spatial_inputs,
        geometry,
        spacing,
        dref,
        updated_label_ids=updated_ids,
    )
    full_rag = builder(
        [updated_labels],
        d0,
        spatial_inputs,
        geometry,
        spacing,
        dref,
        statistics_by_batch=full_stats,
    )

    torch.testing.assert_close(local_rag.node_features, full_rag.node_features)
    torch.testing.assert_close(
        local_rag.node_centroid_um,
        full_rag.node_centroid_um,
    )
    torch.testing.assert_close(
        local_rag.node_volume_voxels,
        full_rag.node_volume_voxels,
    )
    local_edges, local_features = _sort_edges(local_rag)
    full_edges, full_features = _sort_edges(full_rag)
    assert torch.equal(local_edges, full_edges)
    torch.testing.assert_close(local_features, full_features)


def test_disappeared_affected_row_is_cleared():
    torch.manual_seed(37)
    cfg = small_model_config()
    shape = (3, 8, 8)

    initial_labels = torch.zeros(shape, dtype=torch.long)
    initial_labels[:, 1:4, 1:4] = 1
    initial_labels[:, 4:7, 4:7] = 2
    initial_labels[:, 1:4, 4:7] = 3

    updated_labels = initial_labels.clone()
    updated_labels[updated_labels == 2] = 1
    changed_box = (slice(0, 3), slice(4, 7), slice(4, 7))

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
        [(0, changed_box)],
        spatial_inputs,
        geometry,
        spacing,
        (d0, d1, d2),
        updated_label_ids=[torch.tensor([1, 2])],
    )
    full_stats = build_supervoxel_statistics(
        [updated_labels],
        spatial_inputs,
        geometry,
        spacing,
        (d0, d1, d2),
    )
    _assert_statistics_close(local_stats[0], full_stats[0])
