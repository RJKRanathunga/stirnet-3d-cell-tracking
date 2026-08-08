import numpy as np

from ...inference.component_crop import build_inference_roi
from ..config import SampleBuildConfig
from ..core.marker_heatmap import deepest_point_marker


def test_inference_component_uses_same_object_centric_normalization():
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
    assert roi.inputs.shape == (4, 16, 64, 64)
    assert roi.transform.normalization_scale < 1.0
    point = np.array([[7.5, 31.5, 31.5]])
    native = roi.canonical_centers_to_native(point)
    assert native.shape == (1, 3)
