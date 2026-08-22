from __future__ import annotations

import numpy as np

from learned.stirnet.model.config import PartitionConfig
from learned.stirnet.model.partition.supervoxel_guard import (
    build_supervoxel_face_cuts,
    split_preliminary_supervoxels,
)


def _cfg(**updates):
    cfg = PartitionConfig(
        supervoxel_guard_enabled=True,
        supervoxel_guard_face_separator_high=0.80,
        supervoxel_guard_face_separator_low=0.55,
        supervoxel_guard_face_ridge_tolerance=0.02,
        supervoxel_guard_centroid_disagreement_dref=0.30,
        supervoxel_guard_centroid_strong_disagreement_dref=0.70,
        supervoxel_guard_flow_disagreement=0.70,
        supervoxel_guard_flow_min_norm=0.10,
        supervoxel_guard_seed_valley_max=0.55,
        supervoxel_guard_sdf_valley_max=0.45,
        supervoxel_guard_geometry_only_enabled=True,
        supervoxel_guard_min_fragment_voxels=4,
        supervoxel_guard_min_fragment_fraction=0.0,
        supervoxel_guard_max_fragments=16,
    )
    for key, value in updates.items():
        setattr(cfg, key, value)
    return cfg


def _base_scene(shape=(3, 9, 11)):
    labels = np.ones(shape, dtype=np.int32)
    separator = np.zeros(shape, dtype=np.float32)
    centroid = np.zeros((3, *shape), dtype=np.float32)
    flow = np.zeros((3, *shape), dtype=np.float32)
    seed = np.ones(shape, dtype=np.float32)
    sdf = np.ones(shape, dtype=np.float32)
    spacing = np.ones(3, dtype=np.float32)
    return labels, separator, centroid, flow, seed, sdf, spacing


def test_anisotropic_z_face_is_normalized_before_thresholding():
    shape = (5, 3, 3)
    labels, separator, centroid, flow, seed, sdf, spacing = _base_scene(shape)
    spacing[:] = (2.0, 1.0, 1.0)
    sigma = 0.50

    z_um = np.arange(shape[0], dtype=np.float32) * spacing[0]
    distance = np.abs(z_um - 3.0)
    profile = np.exp(-0.5 * (distance / sigma) ** 2).astype(np.float32)
    separator[:] = profile[:, None, None]

    cuts, evidence = build_supervoxel_face_cuts(
        separator, centroid, flow, seed, sdf, spacing, 4.0, _cfg(),
        separator_sigma_um=sigma,
    )

    assert float(separator[1, 1, 1]) < 0.20
    assert np.all(cuts[0][1])
    assert not np.any(cuts[0][0])
    assert float(evidence["separator_face_score"][0][1, 1, 1]) > 0.99


def test_ridge_thinning_avoids_parallel_xy_cut_band():
    shape = (3, 3, 11)
    labels, separator, centroid, flow, seed, sdf, spacing = _base_scene(shape)
    spacing[:] = (2.0, 0.2, 0.2)
    sigma = 0.50

    x_um = np.arange(shape[2], dtype=np.float32) * spacing[2]
    face_um = 0.5 * (x_um[4] + x_um[5])
    distance = np.abs(x_um - face_um)
    profile = np.exp(-0.5 * (distance / sigma) ** 2).astype(np.float32)
    separator[:] = profile[None, None, :]

    cuts, _ = build_supervoxel_face_cuts(
        separator, centroid, flow, seed, sdf, spacing, 4.0, _cfg(),
        separator_sigma_um=sigma,
    )

    x_cut_indices = np.flatnonzero(np.any(cuts[2], axis=(0, 1)))
    assert x_cut_indices.tolist() == [4]


def test_strong_face_separator_splits_without_losing_foreground():
    labels, separator, centroid, flow, seed, sdf, spacing = _base_scene()
    sigma = 0.50
    expected = np.exp(-0.5 * ((0.5 * spacing[2]) / sigma) ** 2)
    separator[:, :, 5] = expected
    separator[:, :, 6] = expected

    guarded, diagnostics = split_preliminary_supervoxels(
        labels, separator, centroid, flow, seed, sdf, spacing, 4.0, _cfg(),
        separator_sigma_um=sigma,
    )

    assert int(guarded.max()) == 2
    assert np.all(guarded > 0)
    assert guarded[1, 4, 2] != guarded[1, 4, 8]
    assert diagnostics.split_supervoxel_count == 1
    assert diagnostics.cut_face_count > 0


def test_medium_separator_can_be_hardened_by_centroid_vote_and_valley_support():
    labels, separator, centroid, flow, seed, sdf, spacing = _base_scene()
    sigma = 0.50
    expected = np.exp(-0.5 * ((0.5 * spacing[2]) / sigma) ** 2)
    separator[:, :, 5] = 0.60 * expected
    separator[:, :, 6] = 0.60 * expected
    seed[:, :, 5:7] = 0.10
    sdf[:, :, 5:7] = 0.10

    x = np.arange(labels.shape[2], dtype=np.float32)
    centroid[2] = np.where(
        x[None, None, :] <= 5,
        (2.0 - x[None, None, :]) / 4.0,
        (8.0 - x[None, None, :]) / 4.0,
    )

    guarded, _ = split_preliminary_supervoxels(
        labels, separator, centroid, flow, seed, sdf, spacing, 4.0, _cfg(),
        separator_sigma_um=sigma,
    )

    assert int(guarded.max()) == 2
    assert guarded[1, 4, 2] != guarded[1, 4, 8]


def test_medium_separator_without_corroboration_remains_soft():
    labels, separator, centroid, flow, seed, sdf, spacing = _base_scene()
    sigma = 0.50
    expected = np.exp(-0.5 * ((0.5 * spacing[2]) / sigma) ** 2)
    separator[:, :, 5] = 0.60 * expected
    separator[:, :, 6] = 0.60 * expected

    guarded, diagnostics = split_preliminary_supervoxels(
        labels, separator, centroid, flow, seed, sdf, spacing, 4.0, _cfg(),
        separator_sigma_um=sigma,
    )

    assert int(guarded.max()) == 1
    assert np.array_equal(guarded, labels)
    assert diagnostics.added_supervoxel_count == 0


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
