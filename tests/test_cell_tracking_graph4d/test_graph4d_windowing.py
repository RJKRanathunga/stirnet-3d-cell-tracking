from __future__ import annotations

import numpy as np
from importlib import import_module

AmbiguityComponent = import_module(
    "legacy.classical_pipeline.tracking.graph_tracking.four_d.types"
).AmbiguityComponent
build_windows = import_module(
    "legacy.classical_pipeline.tracking.graph_tracking.four_d.windowing"
).build_windows
build_observations = import_module(
    "legacy.classical_pipeline.tracking.graph_tracking.four_d.observations"
).build_observations

from .helpers import FourDGraphConfig, moving_frames, run_cell_tracking


def test_overlapping_windows_commit_every_frame_once() -> None:
    frames = moving_frames(10)
    provisional = run_cell_tracking(frames, sample_id="windows")
    config = FourDGraphConfig()
    observations = build_observations(
        time_frames=frames,
        provisional_tracks=provisional.tracks,
        spatial_shape_zyx=np.asarray([64, 256, 256]),
        voxel_size_zyx_um=np.asarray([1.625, 0.40625, 0.40625]),
        config=config,
    )
    component = AmbiguityComponent(
        component_id=0,
        node_indices=np.arange(observations.node_count, dtype=np.int32),
        edge_indices=np.empty(0, dtype=np.int32),
        seed_nodes=np.asarray([0], dtype=np.int32),
        minimum_frame=0,
        maximum_frame=9,
    )
    windows = build_windows(component, observations=observations, edges=[], config=config)
    committed = [window.commit_start_frame for window in windows]
    assert committed == list(range(10))
    assert all(window.end_frame - window.start_frame + 1 <= config.window_size for window in windows)
