import numpy as np
from scipy import ndimage

from ..core.mask_corruption import build_stage2_like_component


def test_synthetic_pair_becomes_one_component():
    labels = np.zeros((12, 32, 32), np.int32)
    labels[3:8, 5:12, 5:12] = 1
    labels[3:8, 16:23, 16:23] = 2
    mask = build_stage2_like_component(
        labels, (1.625, 0.40625, 0.40625), bridge_radius_um=0.45
    )
    _, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    assert count == 1
    assert np.all(mask[labels > 0])
