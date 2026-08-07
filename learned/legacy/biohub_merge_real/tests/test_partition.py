import numpy as np

from learned.instance_segmentation.datasets.merge_real.review.partition_generation import generate_partition


def test_center_seeded_partition_splits_connected_3d_mask():
    zz, yy, xx = np.indices((17, 33, 33))
    first = ((zz - 8) / 5) ** 2 + ((yy - 16) / 8) ** 2 + ((xx - 12) / 7) ** 2 <= 1
    second = ((zz - 8) / 5) ** 2 + ((yy - 16) / 8) ** 2 + ((xx - 20) / 7) ** 2 <= 1
    mask = first | second
    labels = generate_partition(mask, np.array([[8, 16, 11], [8, 16, 21]]), (1.625, 0.40625, 0.40625))
    assert labels.max() == 2
    assert np.all(labels[~mask] == 0)
    assert np.count_nonzero(labels == 1) > 0
    assert np.count_nonzero(labels == 2) > 0
