from __future__ import annotations

from .helpers import GraphTrackingConfig, moving_frames, run_cell_tracking


def test_track_ids_follow_start_frame_and_detection_order() -> None:
    result = run_cell_tracking(
        moving_frames(),
        sample_id="ids",
        graph_config=GraphTrackingConfig(mode="apply", algorithm="windowed_4d"),
    )
    starts = (
        result.tracks.sort_values(["frame", "cell"], kind="mergesort")
        .groupby("track_id", sort=True)
        .head(1)
        .sort_values(["frame", "cell"], kind="mergesort")
    )
    assert starts["track_id"].astype(int).tolist() == list(range(len(starts)))
    assert not result.graph4d_track_id_map.empty

