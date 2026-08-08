import numpy as np

from ..config import SampleBuildConfig
from ..core.marker_heatmap import deepest_point_marker
from ..core.models import AnnotatedVolume, InstanceGroup
from ..core.sample_builder import SampleBuilder
from ..core.sample_selection import select_valid_sample


def test_valid_index_skips_invalid_group():
    image = np.zeros((30, 80, 80), np.float32)
    labels = np.zeros_like(image, np.int32)
    labels[10:17, 15:35, 15:35] = 1
    labels[10:17, 38:58, 15:35] = 2
    image[labels > 0] = 1
    volume = AnnotatedVolume(image, labels, (1,1,1), "x", "x")
    groups = [InstanceGroup((99,), "bad"), InstanceGroup((1,2), "good")]
    builder = SampleBuilder(
        SampleBuildConfig(min_instance_voxels_after_resampling=4, min_instance_bbox_zyx_vox=(2,2,2)),
        marker_detector=deepest_point_marker,
    )
    result = select_valid_sample(volume, groups, builder, valid_index=0)
    assert result.raw_index == 1
    assert len(result.rejections_before) == 1
