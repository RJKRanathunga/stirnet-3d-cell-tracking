import numpy as np
from scipy import ndimage

from learned.instance_segmentation.datasets.core.mask_corruption import build_stage2_like_component


def test_pair_is_bridged_into_one_component() -> None:
    labels = np.zeros((12, 32, 32), dtype=np.int32)
    labels[3:9, 8:14, 6:12] = 1
    labels[3:9, 8:14, 18:24] = 2
    mask = build_stage2_like_component(
        labels,
        (1.0, 1.0, 1.0),
        bridge_radius_um=1.0,
    )
    _, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    assert count == 1
    assert np.all(mask[labels > 0])
