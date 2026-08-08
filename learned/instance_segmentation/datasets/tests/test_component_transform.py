import numpy as np
from scipy import ndimage

from ..core.component_transform import bbox_from_instance_slices, build_canonical_transform


def _transform(labels, spacing=(1.0, 1.0, 1.0)):
    boxes = tuple(ndimage.find_objects(labels))
    bbox = bbox_from_instance_slices((1,), boxes, spacing)
    return build_canonical_transform(
        bbox,
        native_spacing_zyx_um=spacing,
        canonical_shape_zyx=(16, 64, 64),
        canonical_spacing_zyx=(1.625, 0.40625, 0.40625),
        component_occupancy=0.78,
        border_margin_voxels=1,
    )


def test_large_and_small_components_receive_different_isotropic_scales():
    small = np.zeros((80, 100, 100), np.int32)
    small[20:30, 20:40, 20:40] = 1
    large = np.zeros_like(small)
    large[20:40, 10:70, 10:70] = 1
    ts = _transform(small)
    tl = _transform(large)
    assert ts.normalization_scale > tl.normalization_scale
    assert np.isclose(ts.normalization_scale * 20, tl.normalization_scale * 60, rtol=0.15)


def test_transform_round_trip():
    labels = np.zeros((60, 80, 90), np.int32)
    labels[10:24, 20:50, 30:70] = 1
    t = _transform(labels, spacing=(2.0, 0.5, 0.5))
    native = np.array([[12.2, 25.5, 35.1], [20.0, 45.0, 65.0]])
    restored = t.canonical_to_native(t.native_to_canonical(native))
    assert np.allclose(restored, native)
