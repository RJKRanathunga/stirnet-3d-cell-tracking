import numpy as np
import pytest

from ..config import SampleBuildConfig
from ..core.marker_heatmap import deepest_point_marker
from ..core.models import AnnotatedVolume, InstanceGroup
from ..core.sample_builder import SampleBuildError, SampleBuilder


def _test_config(**kwargs):
    return SampleBuildConfig(
        min_instance_voxels_after_resampling=4,
        min_instance_bbox_zyx_vox=(2, 2, 2),
        **kwargs,
    )


def test_large_pair_is_scaled_into_64_cube():
    shape = (40, 160, 160)
    image = np.zeros(shape, np.float32)
    labels = np.zeros(shape, np.int32)
    labels[10:20, 20:65, 30:75] = 1
    labels[12:22, 68:113, 35:80] = 2
    image[labels > 0] = 1.0
    volume = AnnotatedVolume(image, labels, (1.0, 1.0, 1.0), "synthetic", "large")
    sample = SampleBuilder(_test_config(), marker_detector=deepest_point_marker).build(
        volume, InstanceGroup((1, 2), "pair_merge")
    )
    assert sample.inputs.shape == (4, 64, 64, 64)
    assert sample.targets.instance_labels.max() == 2
    assert sample.metadata["vector_coordinate_system"] == "canonical_axis_fraction_cubic"
    extent = sample.metadata["canonical_group_bbox_extent_vox_zyx"]
    assert max(extent) < 50.0
    assert max(extent) > 45.0


def test_small_single_is_scaled_up():
    shape = (50, 100, 100)
    image = np.zeros(shape, np.float32)
    labels = np.zeros(shape, np.int32)
    labels[20:24, 45:53, 45:53] = 1
    image[labels > 0] = 1
    volume = AnnotatedVolume(image, labels, (1.0, 1.0, 1.0), "synthetic", "small")
    sample = SampleBuilder(_test_config(), marker_detector=deepest_point_marker).build(
        volume, InstanceGroup((1,), "single")
    )
    assert sample.transform.scale_vox_per_um > 1.0


def test_native_boundary_instance_is_rejected_for_training():
    image = np.zeros((30, 80, 80), np.float32)
    labels = np.zeros_like(image, np.int32)
    labels[0:8, 20:40, 20:40] = 1
    image[labels > 0] = 1
    volume = AnnotatedVolume(image, labels, (1, 1, 1), "synthetic", "boundary")
    with pytest.raises(SampleBuildError, match="native source boundary"):
        SampleBuilder(_test_config(), marker_detector=deepest_point_marker).build(
            volume, InstanceGroup((1,), "single")
        )
