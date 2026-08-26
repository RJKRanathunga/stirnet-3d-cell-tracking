from __future__ import annotations

import torch

from learned.stirnet.inference.tiled_dense import (
    assign_reference_rows_to_dense_tiles,
    generate_dense_tiles,
)
from learned.stirnet.model.config import InferenceConfig
from learned.stirnet.model.temporal.observer import _sample_explicit_geometry
from learned.stirnet.model.types import GeometryState, RefinedGeometryView, SparseGeometryDelta


def _geometry(shape=(12, 16, 18)):
    torch.manual_seed(7)
    return GeometryState(
        foreground_logits=torch.randn(1, 1, *shape),
        surface_logits=torch.randn(1, 1, *shape),
        separator_logits=torch.randn(1, 1, *shape),
        sdf=torch.randn(1, 1, *shape),
        flow=torch.randn(1, 3, *shape),
        centroid_offset=torch.randn(1, 3, *shape),
        seed_logits=torch.randn(1, 1, *shape),
        features=torch.randn(1, 4, *shape),
    )


def test_bounded_dense_explicit_fastpath_matches_crop_reference():
    geometry = _geometry()
    slow_view = RefinedGeometryView(base=geometry, delta=SparseGeometryDelta())
    refs = torch.tensor(
        [[0.0, 0.0, 0.0], [-4.0, 1.5, 2.0], [20.0, -30.0, 5.0]],
        dtype=torch.float32,
    )
    spacing = torch.tensor([1.0, 0.8, 0.7])
    radius = torch.tensor([2.0, 2.0, 2.0])

    fast = _sample_explicit_geometry(geometry, 0, refs, spacing, radius)
    reference = _sample_explicit_geometry(slow_view, 0, refs, spacing, radius)

    assert fast.shape == (3, 11)
    assert torch.allclose(fast, reference, atol=1e-5, rtol=1e-5)


def test_reference_tile_assignment_covers_inside_and_outside_refs_once():
    config = InferenceConfig(
        mode="tiled",
        tiled_dense_enabled=True,
        tile_shape_zyx=(8, 12, 12),
        tile_overlap_zyx=(2, 4, 4),
        tile_halo_zyx=(1, 2, 2),
        tile_batch_size=1,
    )
    shape = (16, 24, 24)
    specs = generate_dense_tiles(1, shape, config)
    refs = torch.tensor(
        [[0.0, 0.0, 0.0], [-100.0, -100.0, -100.0], [100.0, 100.0, 100.0], [0.0, 3.0, -2.0]],
        dtype=torch.float32,
    )
    assignments = assign_reference_rows_to_dense_tiles(
        refs,
        torch.zeros(len(refs), dtype=torch.long),
        torch.tensor([[1.0, 1.0, 1.0]]),
        shape,
        specs,
        config.tile_halo_zyx,
    )
    routed = sorted(row for rows in assignments.values() for row in rows)
    assert routed == list(range(len(refs)))
    assert sum(len(rows) for rows in assignments.values()) == len(refs)
