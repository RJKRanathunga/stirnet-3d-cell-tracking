from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd

from .helpers import FourDGraphConfig, GraphTrackingConfig, detection, run_cell_tracking


build_observations = import_module(
    "src.07_cell_tracking.graph_tracking.four_d.observations"
).build_observations
start_event_options = import_module(
    "src.07_cell_tracking.graph_tracking.four_d.boundary_events"
).start_event_options
end_event_options = import_module(
    "src.07_cell_tracking.graph_tracking.four_d.boundary_events"
).end_event_options


def test_face_specific_entry_and_exit_options_are_plausible() -> None:
    frames = [
        pd.DataFrame([detection(1, 20, 100, 1)]),
        pd.DataFrame([detection(1, 20, 100, 3)]),
    ]
    provisional = run_cell_tracking(frames, sample_id="boundary-options")
    config = FourDGraphConfig()
    observations = build_observations(
        time_frames=frames,
        provisional_tracks=provisional.tracks,
        spatial_shape_zyx=np.asarray([64, 256, 256]),
        voxel_size_zyx_um=np.asarray([1.625, 0.40625, 0.40625]),
        config=config,
    )
    entries = start_event_options(
        0, edges=[], observations=observations, config=config
    )
    exits = end_event_options(
        0, edges=[], observations=observations, config=config
    )
    assert any(event == "boundary_entry" and face == "x_min" for event, face, _, _ in entries)
    assert any(event == "boundary_exit" and face == "x_min" for event, face, _, _ in exits)
    assert all(face in {"", "x_min"} for _, face, _, _ in entries)


def test_middle_sequence_boundary_track_selects_entry_and_exit() -> None:
    frames = [
        pd.DataFrame([
            detection(1, 20, 80, 20),
            detection(2, 20, 120, 20),
        ]),
        pd.DataFrame([
            detection(1, 20, 81, 20),
            detection(2, 20, 121, 20),
            detection(3, 20, 100, 1),
        ]),
        pd.DataFrame([
            detection(1, 20, 82, 20),
            detection(2, 20, 122, 20),
        ]),
    ]
    result = run_cell_tracking(
        frames,
        sample_id="boundary-events",
        graph_config=GraphTrackingConfig(mode="apply", algorithm="windowed_4d"),
    )
    event_types = set(result.graph4d_boundary_events["event_type"])
    assert {"boundary_entry", "boundary_exit"}.issubset(event_types)
