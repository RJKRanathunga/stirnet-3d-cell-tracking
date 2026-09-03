from __future__ import annotations

# STIRNET_TINY_SUPERVOXEL_AGGLOMERATION_V1

import numpy as np

from learned.stirnet.model.partition.tiny_agglomeration import (
    agglomerate_tiny_supervoxels,
)


def _sizes(labels: np.ndarray) -> list[int]:
    counts = np.bincount(np.asarray(labels).ravel())
    return [
        int(counts[label])
        for label in range(1, len(counts))
        if counts[label] > 0
    ]


def test_connected_tiny_region_merges_into_largest_physical_interface():
    labels = np.zeros((3, 3, 4), dtype=np.int32)

    # Large A. Tiny region touches A through THREE Z-normal faces.
    labels[0, :, 0:2] = 1

    # Large B. Tiny region touches B through ONE X-normal face.
    labels[1, 0, 2:4] = 2
    labels[2, 0, 2:4] = 2

    # Three-voxel tiny region.
    labels[1, :, 1] = 3

    # spacing (z,y,x)=(4,1,1):
    #   Z-normal area = 1, so A contact = 3
    #   X-normal area = 4, so B contact = 4
    out, diag = agglomerate_tiny_supervoxels(
        labels,
        (4.0, 1.0, 1.0),
        max_voxels=3,
    )

    tiny_output_label = int(out[1, 1, 1])
    b_output_label = int(out[1, 0, 2])
    a_output_label = int(out[0, 1, 1])

    assert tiny_output_label == b_output_label
    assert tiny_output_label != a_output_label
    assert diag.merge_operation_count == 1
    assert diag.deleted_region_count == 0


def test_isolated_tiny_region_is_deleted_to_background():
    labels = np.zeros((3, 7, 7), dtype=np.int32)
    labels[0:2, 0:3, 0:3] = 1
    labels[2, 6, 6] = 2

    out, diag = agglomerate_tiny_supervoxels(
        labels,
        (1.0, 1.0, 1.0),
        max_voxels=2,
    )

    assert int(out[2, 6, 6]) == 0
    assert np.all(out[0:2, 0:3, 0:3] > 0)
    assert diag.deleted_region_count == 1
    assert diag.deleted_voxel_count == 1


def test_tiny_chain_is_resolved_iteratively():
    labels = np.zeros((1, 3, 8), dtype=np.int32)
    labels[0, :, 0:3] = 1
    labels[0, 1, 3] = 2
    labels[0, 1, 4] = 3

    out, diag = agglomerate_tiny_supervoxels(
        labels,
        (1.0, 1.0, 1.0),
        max_voxels=2,
    )

    receiver = int(out[0, 1, 2])
    assert receiver > 0
    assert int(out[0, 1, 3]) == receiver
    assert int(out[0, 1, 4]) == receiver
    assert diag.merge_operation_count == 2
    assert diag.remaining_tiny_supervoxel_count == 0


def test_threshold_is_inclusive():
    labels = np.zeros((1, 2, 8), dtype=np.int32)
    labels[:, :, :4] = 1
    labels[0, 0, 4:6] = 2  # exactly two voxels

    out, diag = agglomerate_tiny_supervoxels(
        labels,
        (1.0, 1.0, 1.0),
        max_voxels=2,
    )

    assert int(out[0, 0, 4]) == int(out[0, 0, 3])
    assert diag.input_tiny_supervoxel_count == 1
    assert diag.merge_operation_count == 1


def test_surviving_labels_are_dense_and_all_above_threshold():
    labels = np.zeros((2, 5, 9), dtype=np.int32)
    labels[:, :, 0:3] = 3
    labels[:, :, 6:9] = 11
    labels[0, 2, 3] = 7
    labels[0, 2, 5] = 19

    out, diag = agglomerate_tiny_supervoxels(
        labels,
        (1.625, 0.40625, 0.40625),
        max_voxels=2,
    )

    positive = np.unique(out)
    positive = positive[positive > 0]
    assert positive.tolist() == list(range(1, len(positive) + 1))
    assert all(size > 2 for size in _sizes(out))
    assert diag.remaining_tiny_supervoxel_count == 0
