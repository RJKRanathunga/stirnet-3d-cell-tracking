import numpy as np
from scipy import ndimage

from ..core.component_transform import bbox_from_instance_slices, build_canonical_transform
from ..core.models import AnnotatedVolume
from ..core.resampling import resample_with_transform


def test_object_centric_resampling_produces_64_cube():
    labels = np.zeros((40, 100, 100), np.int32)
    labels[10:20, 20:70, 25:75] = 7
    image = labels.astype(np.float32) * 3
    volume = AnnotatedVolume(image, labels, (1.0, 1.0, 1.0), "test", "s")
    bbox = bbox_from_instance_slices((7,), tuple(ndimage.find_objects(labels)), volume.spacing_zyx_um)
    transform = build_canonical_transform(
        bbox,
        native_spacing_zyx_um=volume.spacing_zyx_um,
        canonical_shape_zyx=(64, 64, 64),
        canonical_spacing_zyx=(1.0, 1.0, 1.0),
        component_occupancy=0.78,
    )
    crop = resample_with_transform(volume, transform)
    assert crop.image.shape == (64, 64, 64)
    assert crop.labels.shape == (64, 64, 64)
    assert np.any(crop.labels == 7)
