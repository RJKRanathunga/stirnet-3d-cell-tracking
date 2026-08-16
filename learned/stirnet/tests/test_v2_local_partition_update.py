from __future__ import annotations

import torch
from torch import nn

from learned.stirnet.model.config import RefinementConfig
from learned.stirnet.model.partition.local_update import (
    LocalPartitionUpdater,
    _merge_boxes,
    reconcile_local_labels,
)
from learned.stirnet.model.partition.rag import RAGBuilder
from learned.stirnet.model.types import (
    GeometryState,
    RefinedGeometryView,
    SparseGeometryDelta,
    SparseGeometryROI,
)

from .conftest import small_model_config


def _reconcile(
    base: torch.Tensor,
    local: torch.Tensor,
    box: tuple[slice, slice, slice],
    core: tuple[slice, slice, slice],
):
    mask = torch.zeros_like(local, dtype=torch.bool)
    mask[core] = True
    return reconcile_local_labels(base, local, box, mask)


def test_local_reconcile_boundary_shift_preserves_external_id():
    base = torch.zeros((5, 7, 7), dtype=torch.long)
    base[1:4, 2:5, 2:5] = 4
    box = (slice(0, 5), slice(1, 6), slice(1, 6))
    local = base[box].clone()
    local[:, 1:4, 1:4] = 1
    updated, reason = _reconcile(
        base, local, box, (slice(1, 4), slice(1, 4), slice(1, 4))
    )
    assert reason == ""
    assert updated is not None
    assert 4 in torch.unique(updated).tolist()
    assert torch.equal(updated[:, 0], base[:, 0])


def test_local_reconcile_split_and_recovery_allocate_new_ids():
    base = torch.zeros((7, 9, 9), dtype=torch.long)
    base[2:5, 3:6, 2:7] = 1
    box = (slice(1, 6), slice(2, 7), slice(1, 8))
    local = torch.zeros((5, 5, 7), dtype=torch.long)
    local[1:4, 1:4, 1:3] = 1
    local[1:4, 1:4, 4:6] = 2
    updated, reason = _reconcile(
        base, local, box, (slice(1, 4), slice(1, 4), slice(1, 6))
    )
    assert reason == ""
    assert updated is not None
    assert torch.unique(updated[updated > 0]).numel() == 2

    recovery_base = torch.zeros_like(base)
    recovered, reason = _reconcile(
        recovery_base, local, box, (slice(1, 4), slice(1, 4), slice(1, 6))
    )
    assert reason == ""
    assert recovered is not None
    assert torch.unique(recovered[recovered > 0]).numel() == 2


def test_local_reconcile_merge_is_ambiguous_and_border_roi_is_supported():
    base = torch.zeros((5, 7, 7), dtype=torch.long)
    base[1:4, 2:5, 1:3] = 1
    base[1:4, 2:5, 4:6] = 2
    box = (slice(0, 5), slice(1, 6), slice(0, 7))
    local = torch.zeros((5, 5, 7), dtype=torch.long)
    local[1:4, 1:4, 1:6] = 1
    updated, reason = _reconcile(
        base, local, box, (slice(1, 4), slice(1, 4), slice(1, 6))
    )
    assert updated is None
    assert "merge" in reason or "multiple" in reason

    border_base = torch.zeros((4, 5, 5), dtype=torch.long)
    border_local = torch.zeros_like(border_base)
    border_local[0:3, 1:4, 1:4] = 1
    border, reason = _reconcile(
        border_base,
        border_local,
        (slice(0, 4), slice(0, 5), slice(0, 5)),
        (slice(0, 3), slice(1, 4), slice(1, 4)),
    )
    assert reason == ""
    assert border is not None and int(border.max()) == 1


def test_overlapping_refinement_boxes_are_merged_once():
    boxes = [
        (slice(1, 5), slice(1, 5), slice(1, 5)),
        (slice(4, 8), slice(3, 7), slice(2, 6)),
        (slice(10, 12), slice(10, 12), slice(10, 12)),
    ]
    merged = _merge_boxes(boxes)
    assert len(merged) == 2
    assert any(
        box[0].start == 1 and box[0].stop == 8 for box in merged
    )


def test_ambiguous_local_update_uses_verified_full_recompute_fallback():
    shape = (5, 7, 7)
    base_labels = torch.zeros(shape, dtype=torch.long)
    base_labels[1:4, 2:5, 1:3] = 1
    base_labels[1:4, 2:5, 4:6] = 2
    full_labels = torch.full(shape, 9, dtype=torch.long)
    roi = (slice(1, 4), slice(2, 5), slice(1, 6))

    class FakeWatershed(nn.Module):
        def forward(self, geometry, spacing, dref, padding=None):
            if geometry.sdf.shape[-3:] == shape:
                return [full_labels]
            return [torch.ones(geometry.sdf.shape[-3:], dtype=torch.long)]

    def field(channels):
        return torch.zeros((1, channels, *shape))

    base_geometry = GeometryState(
        foreground_logits=field(1),
        surface_logits=field(1),
        separator_logits=field(1),
        sdf=field(1),
        flow=field(3),
        centroid_offset=field(3),
        seed_logits=field(1),
        features=None,
    )
    refined = RefinedGeometryView(
        base_geometry,
        SparseGeometryDelta([SparseGeometryROI(0, roi, torch.zeros((11, 3, 3, 5)))]),
    )
    config = RefinementConfig(partition_halo_dref=0.0)
    result = LocalPartitionUpdater(FakeWatershed(), config)(
        [base_labels],
        refined,
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([1.0]),
    )
    assert result.used_fallback
    assert result.fallback_reason
    assert torch.equal(result.supervoxel_labels[0], full_labels)


def test_local_rag_refresh_matches_full_rebuild_for_affected_nodes_and_edges():
    torch.manual_seed(17)
    cfg = small_model_config()
    builder = RAGBuilder(cfg.partition, cfg.spatial)
    shape = (4, 8, 8)
    initial_labels = torch.zeros(shape, dtype=torch.long)
    initial_labels[:, :4, :4] = 1
    initial_labels[:, :4, 4:] = 2
    initial_labels[:, 4:, :4] = 3
    initial_labels[:, 4:, 4:] = 4
    updated_labels = initial_labels.clone()
    updated_labels[:, :4, :2] = 5
    box = (slice(0, 4), slice(0, 4), slice(0, 4))

    def field(channels: int) -> torch.Tensor:
        return torch.randn((1, channels, *shape))

    geometry = GeometryState(
        foreground_logits=field(1),
        surface_logits=field(1),
        separator_logits=field(1),
        sdf=field(1),
        flow=field(3),
        centroid_offset=field(3),
        seed_logits=field(1),
        features=None,
    )
    d0 = torch.randn((1, cfg.spatial.channels[0], *shape))
    spatial_inputs = torch.randn((1, cfg.spatial.in_channels, *shape))
    spacing = torch.tensor([[1.5, 0.4, 0.4]])
    dref = torch.tensor([3.0])
    initial = builder(
        [initial_labels], d0, spatial_inputs, geometry, spacing, dref
    )
    local = builder.update_local(
        initial,
        [updated_labels],
        [(0, box)],
        d0,
        spatial_inputs,
        geometry,
        spacing,
        dref,
    )
    full = builder([updated_labels], d0, spatial_inputs, geometry, spacing, dref)

    torch.testing.assert_close(local.node_features, full.node_features)
    torch.testing.assert_close(local.node_centroid_um, full.node_centroid_um)
    torch.testing.assert_close(local.node_volume_voxels, full.node_volume_voxels)
    local_order = torch.argsort(local.edge_index[0] * 5 + local.edge_index[1])
    full_order = torch.argsort(full.edge_index[0] * 5 + full.edge_index[1])
    assert torch.equal(
        local.edge_index[:, local_order], full.edge_index[:, full_order]
    )
    torch.testing.assert_close(
        local.edge_features[local_order], full.edge_features[full_order]
    )
