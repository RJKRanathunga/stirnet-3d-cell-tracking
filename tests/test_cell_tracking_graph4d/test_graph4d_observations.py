from __future__ import annotations

import numpy as np
from importlib import import_module

build_observations = import_module(
    "src.07_cell_tracking.graph_tracking.four_d.observations"
).build_observations

from .helpers import FourDGraphConfig, moving_frames, run_cell_tracking


def test_observations_are_columnar_physical_and_unique() -> None:
    frames = moving_frames(2)
    provisional = run_cell_tracking(frames, sample_id="obs")
    observations = build_observations(
        time_frames=frames,
        provisional_tracks=provisional.tracks,
        spatial_shape_zyx=np.asarray([64, 256, 256]),
        voxel_size_zyx_um=np.asarray([1.625, 0.40625, 0.40625]),
        config=FourDGraphConfig(),
    )
    assert observations.node_count == 6
    assert len(observations.node_by_frame_detection) == 6
    np.testing.assert_allclose(
        observations.positions_zyx_um[0], np.asarray([32.5, 36.5625, 36.5625])
    )
