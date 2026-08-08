import numpy as np

from learned.instance_segmentation.datasets.config import SampleBuildConfig
from learned.instance_segmentation.datasets.core.models import AnnotatedVolume, InstanceGroup
from learned.instance_segmentation.datasets.core.sample_builder import SampleBuilder


def _dummy_marker_detector(mask, spacing):
    coords = np.argwhere(mask)
    center = coords[len(coords) // 2]
    return (tuple(int(v) for v in center),)


def test_sample_builder_matches_vector_cnn_contract() -> None:
    shape = (24, 96, 96)
    zz, yy, xx = np.indices(shape)
    image = np.exp(-((yy - 44) ** 2 + (xx - 38) ** 2) / 100.0).astype(np.float32)
    labels = np.zeros(shape, dtype=np.int32)
    labels[8:16, 36:52, 28:40] = 10
    labels[8:16, 36:52, 46:58] = 20
    volume = AnnotatedVolume(
        image=image,
        instance_labels=labels,
        spacing_zyx_um=(1.0, 1.0, 1.0),
        dataset_name="synthetic",
        sample_id="synthetic_0",
        split="train",
        intensity_bounds=(0.0, 1.0),
    )
    config = SampleBuildConfig(
        target_spacing_zyx_um=(1.0, 1.0, 1.0),
        crop_shape_zyx=(16, 64, 64),
        edt_clip_um=8.0,
        marker_sigma_um=1.0,
        vector_max_distance_um=16.0,
        center_sigma_um=1.0,
        center_interior_fraction=0.7,
        boundary_radius_um=1.0,
        bridge_radius_um=0.5,
        adjacency_max_distance_um=4.0,
    )
    sample = SampleBuilder(config, marker_detector=_dummy_marker_detector).build(
        volume,
        InstanceGroup(instance_ids=(10, 20), kind="pair_merge"),
    )
    assert sample.inputs.shape == (4, 16, 64, 64)
    assert sample.inputs.dtype == np.float32
    assert sample.targets.foreground.shape == (1, 16, 64, 64)
    assert sample.targets.vectors_normalized.shape == (3, 16, 64, 64)
    assert sample.targets.boundary.shape == (1, 16, 64, 64)
    assert sample.targets.center.shape == (1, 16, 64, 64)
    assert sample.valid_mask.shape == (1, 16, 64, 64)
    assert sample.targets.instance_labels.max() == 2
    assert sample.targets.boundary.sum() > 0
    assert 0.0 <= sample.inputs.min() <= sample.inputs.max() <= 1.0
