from __future__ import annotations

import torch

from learned.stirnet.training.coverage_crops import (
    build_coverage_crop_manifest,
    sample_coverage_crop_specs,
)


def _scene():
    labels = torch.zeros((1, 8, 48, 48), dtype=torch.long)
    cell_id = 1
    for y in (4, 14, 24, 34):
        for x in (4, 14, 24, 34):
            labels[0, 2:5, y:y+4, x:x+4] = cell_id
            cell_id += 1
    labels[0, 1:4, 0:4, 20:24] = cell_id  # true acquisition boundary
    return labels


def test_manifest_covers_every_coverable_interior_cell():
    labels = _scene()
    manifest = build_coverage_crop_manifest(
        labels,
        crop_shape_zyx=(8, 28, 28),
        min_complete_cells=4,
        views_per_cell=1,
    )
    assert manifest.uncoverable_cell_ids == ((),)
    covered = {
        cell_id
        for row in manifest.records[0]
        for cell_id in row.complete_cell_ids
    }
    assert covered == set(range(1, 17))
    assert any(row.true_boundary_cell_ids for row in manifest.records[0])


def test_sampling_is_deterministic():
    labels = _scene()
    manifest = build_coverage_crop_manifest(
        labels,
        crop_shape_zyx=(8, 28, 28),
        min_complete_cells=4,
        views_per_cell=1,
    )
    spacing = torch.tensor([[2.0, 0.4, 0.4]])
    first = sample_coverage_crop_specs(
        labels, spacing, manifest, crops_per_step=2, global_step=3
    )
    second = sample_coverage_crop_specs(
        labels, spacing, manifest, crops_per_step=2, global_step=3
    )
    assert [r[0].slices_zyx for r in first] == [r[0].slices_zyx for r in second]
    assert all(r[0].shape_zyx == (8, 28, 28) for r in first)
