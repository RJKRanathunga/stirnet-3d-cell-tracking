import numpy as np

from ..config import SampleBuildConfig
from ..core.marker_heatmap import deepest_point_marker
from ..core.models import AnnotatedVolume, InstanceGroup
from ..core.sample_builder import SampleBuilder


def test_large_pair_is_scaled_instead_of_rejected_for_absolute_size():
    shape = (40, 160, 160)
    image = np.zeros(shape, np.float32)
    labels = np.zeros(shape, np.int32)
    labels[10:20, 20:65, 30:75] = 1
    labels[12:22, 68:113, 35:80] = 2
    image[labels > 0] = 1.0
    volume = AnnotatedVolume(image, labels, (1.0, 1.0, 1.0), "synthetic", "large")
    config = SampleBuildConfig(
        min_instance_voxels_after_resampling=4,
        min_instance_bbox_zyx_vox=(1, 2, 2),
    )
    sample = SampleBuilder(config, marker_detector=deepest_point_marker).build(
        volume, InstanceGroup((1, 2), "pair_merge")
    )
    assert sample.inputs.shape == (4, 16, 64, 64)
    assert sample.transform.normalization_scale < 1.0
    assert sample.targets.instance_labels.max() == 2
    assert sample.metadata["vector_coordinate_system"] == "canonical_axis_fraction"


def test_small_single_is_scaled_up():
    shape = (50, 100, 100)
    image = np.zeros(shape, np.float32)
    labels = np.zeros(shape, np.int32)
    labels[20:24, 45:53, 45:53] = 1
    image[labels > 0] = 1
    volume = AnnotatedVolume(image, labels, (1.0, 1.0, 1.0), "synthetic", "small")
    config = SampleBuildConfig(
        min_instance_voxels_after_resampling=4,
        min_instance_bbox_zyx_vox=(1, 2, 2),
    )
    sample = SampleBuilder(config, marker_detector=deepest_point_marker).build(
        volume, InstanceGroup((1,), "single")
    )
    assert sample.transform.normalization_scale > 1.0
