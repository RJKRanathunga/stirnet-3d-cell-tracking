from __future__ import annotations

import pandas as pd

from .helpers import GraphTrackingConfig, detection, run_cell_tracking


def test_one_frame_missing_detection_is_a_direct_gap_edge() -> None:
    frames = [
        pd.DataFrame([
            detection(1, 20, 100, 100),
            detection(2, 20, 80, 100),
            detection(3, 20, 120, 100),
        ]),
        pd.DataFrame([
            detection(2, 20, 81, 100),
            detection(3, 20, 121, 100),
        ]),
        pd.DataFrame([
            detection(1, 20, 102, 100),
            detection(2, 20, 82, 100),
            detection(3, 20, 122, 100),
        ]),
    ]
    result = run_cell_tracking(
        frames,
        sample_id="gap",
        graph_config=GraphTrackingConfig(mode="apply", algorithm="windowed_4d"),
    )
    gaps = result.graph4d_temporal_edges.loc[
        result.graph4d_temporal_edges["optimized_selected"].astype(bool)
        & (result.graph4d_temporal_edges["frame_gap"] == 2)
    ]
    assert not gaps.empty
    assert "graph4d_gap_reacquired" in set(result.tracks["match_type"])


def test_unrelated_far_birth_is_not_forced_into_gap() -> None:
    frames = [
        pd.DataFrame([detection(1, 20, 20, 20)]),
        pd.DataFrame(),
        pd.DataFrame([detection(2, 50, 220, 220)]),
    ]
    # Empty frames still need the Stage 7 detection schema.
    frames[1] = pd.DataFrame(columns=frames[0].columns)
    result = run_cell_tracking(
        frames,
        sample_id="unrelated-birth",
        graph_config=GraphTrackingConfig(mode="apply", algorithm="windowed_4d"),
    )
    selected_gaps = result.graph4d_temporal_edges.loc[
        result.graph4d_temporal_edges["optimized_selected"].astype(bool)
        & (result.graph4d_temporal_edges["frame_gap"] > 1)
    ]
    assert selected_gaps.empty
    assert result.tracks["track_id"].nunique() == 2
