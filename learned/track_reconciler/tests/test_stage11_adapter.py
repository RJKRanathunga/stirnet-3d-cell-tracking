import numpy as np
import pandas as pd

from learned.track_reconciler.integration.stage11 import (
    STAGE11_PAIR_FEATURES,
    add_global_only_predictions,
    tensorize_stage11_pair_features,
)


def test_pair_adapter_keeps_missingness_separate_from_zero():
    frame = pd.DataFrame({"gap_frames": [1, 2], "forward_error_um": [0.0, np.nan]})
    values, validity = tensorize_stage11_pair_features(frame)
    assert values.shape == (2, 80)
    assert validity.shape == (2, 40)
    index = STAGE11_PAIR_FEATURES.index("forward_error_um")
    assert values[0, index] == 0.0 and values[1, index] == 0.0
    assert values[0, 40 + index] == 1.0 and values[1, 40 + index] == 0.0


def test_global_only_prediction_uses_cumulative_shift():
    candidates = pd.DataFrame({
        "source_track_id": [4], "source_end_frame": [1], "target_start_frame": [3]
    })
    observations = pd.DataFrame({
        "track_id": [4], "frame": [1], "z": [2.0], "y": [10.0], "x": [20.0]
    })
    motion = pd.DataFrame({
        "from_frame": [1, 2], "to_frame": [2, 3],
        "global_shift_z_um": [1.0, 2.0],
        "global_shift_y_um": [0.5, 0.5],
        "global_shift_x_um": [-1.0, -1.0],
    })
    out = add_global_only_predictions(candidates, observations, motion, voxel_size_zyx_um=(1.0, 1.0, 1.0))
    expected = np.array([2.0, 10.0, 20.0]) + np.array([3.0, 1.0, -2.0])
    assert np.allclose(out[["global_predicted_z_um", "global_predicted_y_um", "global_predicted_x_um"]].iloc[0], expected)
