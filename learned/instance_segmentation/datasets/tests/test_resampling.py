import numpy as np

from learned.instance_segmentation.datasets.core.models import AnnotatedVolume
from learned.instance_segmentation.datasets.core.resampling import resample_centered_crop


def test_resampling_preserves_discrete_labels() -> None:
    image = np.zeros((20, 40, 40), dtype=np.float32)
    labels = np.zeros_like(image, dtype=np.int32)
    labels[6:12, 12:20, 12:20] = 7
    volume = AnnotatedVolume(
        image=image,
        instance_labels=labels,
        spacing_zyx_um=(1.0, 1.0, 1.0),
        dataset_name="synthetic",
        sample_id="a",
    )
    crop = resample_centered_crop(
        volume,
        (9.0, 16.0, 16.0),
        output_shape_zyx=(12, 24, 24),
        target_spacing_zyx_um=(1.0, 1.0, 1.0),
        anti_alias_image=False,
    )
    assert crop.labels.shape == (12, 24, 24)
    assert set(np.unique(crop.labels)).issubset({0, 7})
