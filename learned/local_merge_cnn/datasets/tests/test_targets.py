import numpy as np

from ..core.targets import build_targets, vectors_to_canonical_displacement


def test_vectors_point_exactly_to_centers_in_canonical_voxels():
    labels = np.zeros((32, 32, 32), np.int32)
    labels[4:12, 5:14, 5:14] = 1
    labels[17:27, 17:27, 17:27] = 2
    component = labels > 0
    component[11:18, 13:18, 13:18] = True
    targets = build_targets(
        labels,
        component,
        (1.0, 1.0, 1.0),
        center_sigma_vox=2.0,
        center_interior_fraction=0.7,
        boundary_radius_vox=1.5,
    )
    displacement = vectors_to_canonical_displacement(targets.vectors_normalized, labels.shape)
    for local_id, center in enumerate(targets.centers_zyx, start=1):
        point = np.argwhere(labels == local_id)[0]
        z, y, x = point
        predicted = displacement[:, z, y, x]
        expected = np.asarray(center, dtype=np.float32) - point.astype(np.float32)
        assert np.allclose(predicted, expected, atol=1e-5)
