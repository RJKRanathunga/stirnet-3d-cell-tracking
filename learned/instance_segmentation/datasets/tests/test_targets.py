import numpy as np

from learned.instance_segmentation.datasets.core.mask_corruption import build_stage2_like_component
from learned.instance_segmentation.datasets.core.targets import build_targets


def test_vectors_point_to_owning_centers_and_boundary_exists() -> None:
    labels = np.zeros((12, 32, 32), dtype=np.int32)
    labels[3:9, 8:14, 7:13] = 1
    labels[3:9, 8:14, 18:24] = 2
    spacing = (1.0, 1.0, 1.0)
    input_mask = build_stage2_like_component(labels, spacing, bridge_radius_um=0.0)

    targets = build_targets(
        labels,
        input_mask,
        spacing,
        vector_max_distance_um=16.0,
        center_sigma_um=1.0,
        center_interior_fraction=0.7,
        boundary_radius_um=1.0,
    )
    assert len(targets.centers_zyx) == 2
    assert targets.boundary.sum() > 0
    assert targets.center.max() == 1.0

    for instance_id, center in enumerate(targets.centers_zyx, start=1):
        point = np.argwhere(labels == instance_id)[0]
        z, y, x = (int(v) for v in point)
        predicted_delta = targets.vectors_normalized[:, z, y, x] * 16.0
        expected_delta = np.asarray(center, dtype=np.float32) - point.astype(np.float32)
        assert np.allclose(predicted_delta, expected_delta)
