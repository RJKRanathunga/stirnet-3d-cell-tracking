import numpy as np
from scipy import ndimage

from ..core.mask_corruption import build_stage2_like_component


def test_synthetic_pair_becomes_one_component_with_cubic_bridge():
    labels = np.zeros((32, 32, 32), np.int32)
    labels[8:16, 5:12, 5:12] = 1
    labels[16:24, 16:23, 16:23] = 2
    mask = build_stage2_like_component(
        labels, (1.0, 1.0, 1.0), bridge_radius_vox=1.5
    )
    _, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    assert count == 1
    assert np.all(mask[labels > 0])
