import numpy as np
from scipy import ndimage

from ..core.component_transform import (
    bbox_from_instance_slices,
    build_canonical_transform,
    transformed_bbox_extent_vox,
)


def _transform(labels, spacing=(1.0, 1.0, 1.0), shape=(64, 64, 64)):
    boxes = tuple(ndimage.find_objects(labels))
    bbox = bbox_from_instance_slices((1,), boxes, spacing)
    return build_canonical_transform(
        bbox,
        native_spacing_zyx_um=spacing,
        canonical_shape_zyx=shape,
        canonical_spacing_zyx=(1.0, 1.0, 1.0),
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
    assert ts.scale_vox_per_um > tl.scale_vox_per_um
    assert np.isclose(ts.scale_vox_per_um * 20, tl.scale_vox_per_um * 60, rtol=0.15)


def test_transform_round_trip():
    labels = np.zeros((60, 80, 90), np.int32)
    labels[10:24, 20:50, 30:70] = 1
    t = _transform(labels, spacing=(2.0, 0.5, 0.5))
    native = np.array([[12.2, 25.5, 35.1], [20.0, 45.0, 65.0]])
    restored = t.canonical_to_native(t.native_to_canonical(native))
    assert np.allclose(restored, native)


def test_biohub_like_anisotropic_voxels_become_nearly_spherical_in_canonical_space():
    labels = np.zeros((30, 80, 80), np.int32)
    # Native voxel bbox 8 x 30 x 32, but Biohub physical spacing makes it
    # approximately 13.0 x 12.19 x 13.0 um: nearly spherical physically.
    labels[8:16, 20:50, 20:52] = 1
    transform = _transform(labels, spacing=(1.625, 0.40625, 0.40625))
    extent = np.asarray(transformed_bbox_extent_vox(transform))
    assert extent.max() - extent.min() < 4.0
    assert extent.min() > 40.0


def test_canonical_spacing_must_be_isotropic():
    labels = np.zeros((30, 40, 40), np.int32)
    labels[5:15, 10:20, 10:20] = 1
    boxes = tuple(ndimage.find_objects(labels))
    bbox = bbox_from_instance_slices((1,), boxes, (1, 1, 1))
    try:
        build_canonical_transform(
            bbox,
            native_spacing_zyx_um=(1, 1, 1),
            canonical_shape_zyx=(64, 64, 64),
            canonical_spacing_zyx=(2, 1, 1),
            component_occupancy=0.78,
        )
    except ValueError as error:
        assert "isotropic" in str(error)
    else:
        raise AssertionError("anisotropic canonical spacing should be rejected")
