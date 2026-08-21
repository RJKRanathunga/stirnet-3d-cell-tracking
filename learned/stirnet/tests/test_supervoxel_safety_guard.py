from __future__ import annotations

import numpy as np

from learned.stirnet.model.config import PartitionConfig
from learned.stirnet.model.partition.supervoxel_guard import (
    split_preliminary_supervoxels,
)


def _base_scene():
    shape = (3, 9, 11)
    labels = np.ones(shape, dtype=np.int32)
    separator = np.zeros(shape, dtype=np.float32)
    centroid = np.zeros((3, *shape), dtype=np.float32)
    flow = np.zeros((3, *shape), dtype=np.float32)
    seed = np.ones(shape, dtype=np.float32)
    sdf = np.ones(shape, dtype=np.float32)
    spacing = np.ones(3, dtype=np.float32)
    return labels, separator, centroid, flow, seed, sdf, spacing


def _cfg(**updates):
    cfg = PartitionConfig(
        supervoxel_guard_enabled=True,
        supervoxel_guard_separator_high=0.60,
        supervoxel_guard_separator_low=0.30,
        supervoxel_guard_centroid_disagreement_dref=0.30,
        supervoxel_guard_centroid_strong_disagreement_dref=0.70,
        supervoxel_guard_flow_disagreement=0.70,
        supervoxel_guard_flow_min_norm=0.10,
        supervoxel_guard_seed_valley_max=0.55,
        supervoxel_guard_sdf_valley_max=0.45,
        supervoxel_guard_min_fragment_voxels=4,
        supervoxel_guard_min_fragment_fraction=0.0,
        supervoxel_guard_max_fragments=8,
        supervoxel_guard_closing_iterations=0,
    )
    for key, value in updates.items():
        setattr(cfg, key, value)
    return cfg


def test_strong_separator_splits_preliminary_supervoxel_without_losing_foreground():
    labels, separator, centroid, flow, seed, sdf, spacing = _base_scene()
    separator[:, :, 5] = 0.95

    guarded, diagnostics = split_preliminary_supervoxels(
        labels, separator, centroid, flow, seed, sdf, spacing, 4.0, _cfg()
    )

    assert int(guarded.max()) == 2
    assert np.all(guarded > 0)
    assert guarded[1, 4, 2] != guarded[1, 4, 8]
    assert diagnostics.split_supervoxel_count == 1


def test_no_geometric_barrier_preserves_preliminary_supervoxel():
    labels, separator, centroid, flow, seed, sdf, spacing = _base_scene()
    guarded, diagnostics = split_preliminary_supervoxels(
        labels, separator, centroid, flow, seed, sdf, spacing, 4.0, _cfg()
    )
    assert int(guarded.max()) == 1
    assert np.array_equal(guarded, labels)
    assert diagnostics.added_supervoxel_count == 0


def test_medium_separator_can_be_hardened_by_centroid_vote_and_valley_support():
    labels, separator, centroid, flow, seed, sdf, spacing = _base_scene()
    separator[:, :, 5] = 0.40
    seed[:, :, 5] = 0.10
    sdf[:, :, 5] = 0.10

    x = np.arange(labels.shape[2], dtype=np.float32)
    centroid[2] = np.where(
        x[None, None, :] <= 5,
        (2.0 - x[None, None, :]) / 4.0,
        (8.0 - x[None, None, :]) / 4.0,
    )

    guarded, _ = split_preliminary_supervoxels(
        labels, separator, centroid, flow, seed, sdf, spacing, 4.0, _cfg()
    )
    assert int(guarded.max()) == 2
    assert guarded[1, 4, 2] != guarded[1, 4, 8]


def test_medium_separator_without_corroboration_remains_soft():
    labels, separator, centroid, flow, seed, sdf, spacing = _base_scene()
    separator[:, :, 5] = 0.40
    guarded, _ = split_preliminary_supervoxels(
        labels, separator, centroid, flow, seed, sdf, spacing, 4.0, _cfg()
    )
    assert int(guarded.max()) == 1
    assert np.array_equal(guarded, labels)


def test_guard_never_merges_preexisting_supervoxels():
    labels, separator, centroid, flow, seed, sdf, spacing = _base_scene()
    labels[:, :, 6:] = 2
    guarded, _ = split_preliminary_supervoxels(
        labels, separator, centroid, flow, seed, sdf, spacing, 4.0, _cfg()
    )
    left = np.unique(guarded[:, :, :6])
    right = np.unique(guarded[:, :, 6:])
    assert set(left.tolist()).isdisjoint(set(right.tolist()))
    assert int(guarded.max()) >= 2
