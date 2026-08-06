from __future__ import annotations

import pandas as pd

from .helpers import GraphTrackingConfig, moving_frames, run_cell_tracking


def test_disabled_is_exact_and_shadow_preserves_tracks() -> None:
    frames = moving_frames()
    disabled = run_cell_tracking(frames, sample_id="disabled")
    explicit = run_cell_tracking(
        frames,
        sample_id="disabled",
        graph_config=GraphTrackingConfig(mode="disabled", algorithm="windowed_4d"),
    )
    pd.testing.assert_frame_equal(disabled.tracks, explicit.tracks)
    shadow = run_cell_tracking(
        frames,
        sample_id="shadow",
        graph_config=GraphTrackingConfig(mode="shadow", algorithm="windowed_4d"),
    )
    pd.testing.assert_frame_equal(disabled.tracks, shadow.tracks)
    pd.testing.assert_frame_equal(disabled.association_events, shadow.association_events)
    pd.testing.assert_frame_equal(
        disabled.association_candidates, shadow.association_candidates
    )
    assert not shadow.graph4d_temporal_edges.empty
    assert shadow.metadata["graph_tracking"]["algorithm"] == "windowed_4d"


def test_apply_is_unique_deterministic_and_traceable() -> None:
    frames = moving_frames()
    config = GraphTrackingConfig(mode="apply", algorithm="windowed_4d")
    first, trace = run_cell_tracking(
        frames, sample_id="apply", graph_config=config, return_diagnostics=True
    )
    second = run_cell_tracking(frames, sample_id="apply", graph_config=config)
    pd.testing.assert_frame_equal(first.tracks, second.tracks)
    assert not first.tracks.duplicated(["frame", "cell"]).any()
    assert not first.tracks.duplicated(["track_id", "frame"]).any()
    assert "graph4d_temporal_edges" in trace.intermediates
    assert first.metadata["graph_tracking"]["validation_status"] == "valid"
