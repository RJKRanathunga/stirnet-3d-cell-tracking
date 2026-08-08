import numpy as np

from ..core.targets import build_targets, vectors_to_canonical_displacement


def test_vectors_point_exactly_to_centers_in_canonical_voxels():
    labels = np.zeros((16, 32, 32), np.int32)
    labels[4:10, 5:14, 5:14] = 1
    labels[5:11, 17:27, 17:27] = 2
    component = labels > 0
    # connect component for boundary ownership
    component[:, :, 13:18] |= np.any(component[:, :, 13:18], axis=2, keepdims=True)
    targets = build_targets(
        labels,
        component,
        (1.625, 0.40625, 0.40625),
        center_sigma_um=1.0,
        center_interior_fraction=0.7,
        boundary_radius_um=0.75,
    )
    displacement = vectors_to_canonical_displacement(targets.vectors_normalized, labels.shape)
    for local_id, center in enumerate(targets.centers_zyx, start=1):
        point = np.argwhere(labels == local_id)[0]
        z, y, x = point
        predicted = displacement[:, z, y, x]
        expected = np.asarray(center, dtype=np.float32) - point.astype(np.float32)
        assert np.allclose(predicted, expected, atol=1e-5)
