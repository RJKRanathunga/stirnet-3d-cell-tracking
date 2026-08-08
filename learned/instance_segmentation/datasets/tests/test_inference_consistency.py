import numpy as np

from ...inference.component_crop import build_inference_roi
from ..config import SampleBuildConfig
from ..core.marker_heatmap import deepest_point_marker


def test_inference_component_uses_same_64_cube_normalization():
    image = np.zeros((40, 120, 120), np.float32)
    mask = np.zeros_like(image, dtype=bool)
    mask[8:20, 20:90, 25:85] = True
    image[mask] = 2.0
    config = SampleBuildConfig()
    roi = build_inference_roi(
        image,
        mask,
        (1.0, 1.0, 1.0),
        config=config,
        marker_detector=deepest_point_marker,
    )
    assert roi.inputs.shape == (4, 64, 64, 64)
    point = np.array([[31.5, 31.5, 31.5]])
    native = roi.canonical_centers_to_native(point)
    assert native.shape == (1, 3)
    restored = roi.transform.native_to_canonical(native)
    assert np.allclose(restored, point)
