from __future__ import annotations

import numpy as np

from learned.stirnet.model.partition.watershed import _merge_tiny_regions


def _reference_merge_tiny_regions(
    labels: np.ndarray,
    min_voxels: int,
) -> np.ndarray:
    """Copy of the pre-optimization implementation for equivalence testing."""
    if min_voxels <= 1 or labels.max() <= 1:
        return labels

    labels = labels.copy()
    counts = np.bincount(labels.ravel())
    tiny = np.flatnonzero((counts > 0) & (counts < min_voxels))
    tiny = tiny[tiny > 0]

    if tiny.size:
        pair_chunks: list[np.ndarray] = []
        for axis in range(3):
            left_slice = [slice(None)] * 3
            right_slice = [slice(None)] * 3
            left_slice[axis] = slice(0, -1)
            right_slice[axis] = slice(1, None)
            left = labels[tuple(left_slice)]
            right = labels[tuple(right_slice)]
            valid = (left > 0) & (right > 0) & (left != right)
            if valid.any():
                pair_chunks.append(
                    np.stack([left[valid], right[valid]], axis=-1)
                )
                pair_chunks.append(
                    np.stack([right[valid], left[valid]], axis=-1)
                )

        if pair_chunks:
            directed = np.concatenate(pair_chunks, axis=0)
            base = int(labels.max()) + 1
            packed = (
                directed[:, 0].astype(np.int64) * base
                + directed[:, 1]
            )
            keys, interface_counts = np.unique(
                packed,
                return_counts=True,
            )
            source = keys // base
            target = keys % base
            remap = np.arange(base, dtype=np.int32)

            for region_id in tiny.tolist():
                candidates = np.flatnonzero(source == region_id)
                if candidates.size:
                    best = candidates[
                        np.lexsort(
                            (
                                target[candidates],
                                -interface_counts[candidates],
                            )
                        )[0]
                    ]
                    remap[region_id] = int(target[best])

            labels = remap[labels]

    unique = np.unique(labels)
    unique = unique[unique > 0]
    out = np.zeros_like(labels, dtype=np.int32)
    for new_id, old_id in enumerate(unique, 1):
        out[labels == old_id] = new_id
    return out


def test_tiny_region_cleanup_matches_reference_handcrafted():
    labels = np.zeros((5, 12, 14), dtype=np.int32)
    labels[:, 1:11, 1:6] = 3
    labels[:, 1:11, 8:13] = 9

    # Tiny bridge labels with different and tied interface situations.
    labels[2, 5, 6] = 5
    labels[2, 5, 7] = 7
    labels[1, 3, 6] = 11

    expected = _reference_merge_tiny_regions(labels, min_voxels=4)
    actual = _merge_tiny_regions(labels, min_voxels=4)

    assert actual.dtype == np.int32
    assert np.array_equal(actual, expected)


def test_tiny_region_cleanup_matches_reference_randomized():
    rng = np.random.default_rng(20260817)

    for _ in range(30):
        shape = (
            int(rng.integers(3, 8)),
            int(rng.integers(8, 18)),
            int(rng.integers(8, 18)),
        )
        labels = rng.integers(
            0,
            16,
            size=shape,
            dtype=np.int32,
        )

        # Create holes in the label ID set so final compact relabeling is
        # explicitly exercised.
        labels[np.isin(labels, [2, 6, 10, 14])] = 0

        min_voxels = int(rng.integers(2, 10))
        expected = _reference_merge_tiny_regions(
            labels,
            min_voxels,
        )
        actual = _merge_tiny_regions(
            labels,
            min_voxels,
        )

        assert np.array_equal(actual, expected)


def test_tiny_region_cleanup_preserves_early_return_behavior():
    labels = np.array(
        [[[0, 1], [1, 1]]],
        dtype=np.int32,
    )
    actual = _merge_tiny_regions(labels, min_voxels=1)
    assert actual is labels
