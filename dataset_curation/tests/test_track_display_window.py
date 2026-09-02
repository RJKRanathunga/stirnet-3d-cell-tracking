from __future__ import annotations

from pathlib import Path

import ast
import pandas as pd

from dataset_curation.annotation.tracks.current import (
    filter_diagnostic_rows_for_frame,
)


def _track(track_id: int, start: int, end: int) -> pd.DataFrame:
    frames = list(range(start, end + 1))
    return pd.DataFrame(
        {
            "track_id": [track_id] * len(frames),
            "frame": frames,
            "cell_id": [100 + track_id] * len(frames),
            "z": [1.0] * len(frames),
            "y": [2.0] * len(frames),
            "x": [3.0] * len(frames),
        }
    )


def test_broken_track_appears_only_near_future_break():
    broken = _track(1, 0, 10)

    assert filter_diagnostic_rows_for_frame(
        broken,
        category="broken",
        current_frame=4,
        horizon_frames=5,
    ).empty

    visible = filter_diagnostic_rows_for_frame(
        broken,
        category="broken",
        current_frame=5,
        horizon_frames=5,
    )
    assert not visible.empty
    assert visible["frame"].min() == 0
    assert visible["frame"].max() == 5

    at_break = filter_diagnostic_rows_for_frame(
        broken,
        category="broken",
        current_frame=10,
        horizon_frames=5,
    )
    assert not at_break.empty
    assert at_break["frame"].min() == 5
    assert at_break["frame"].max() == 10

    assert filter_diagnostic_rows_for_frame(
        broken,
        category="broken",
        current_frame=11,
        horizon_frames=5,
    ).empty


def test_new_track_is_highlighted_only_near_recent_start():
    new = _track(2, 10, 30)

    assert filter_diagnostic_rows_for_frame(
        new,
        category="new",
        current_frame=9,
        horizon_frames=5,
    ).empty

    visible = filter_diagnostic_rows_for_frame(
        new,
        category="new",
        current_frame=14,
        horizon_frames=5,
    )
    assert not visible.empty
    assert visible["frame"].min() == 10
    assert visible["frame"].max() == 14

    edge = filter_diagnostic_rows_for_frame(
        new,
        category="new",
        current_frame=15,
        horizon_frames=5,
    )
    assert not edge.empty

    assert filter_diagnostic_rows_for_frame(
        new,
        category="new",
        current_frame=16,
        horizon_frames=5,
    ).empty


def test_viewer_uses_bounded_track_history_and_lightweight_refresh():
    root = Path(__file__).resolve().parents[2]
    viewer = (
        root
        / "dataset_curation"
        / "annotation"
        / "viewer.py"
    ).read_text(encoding="utf-8")

    assert "TRACK_HISTORY_FRAMES = 5" in viewer
    assert "DIAGNOSTIC_HORIZON_FRAMES = 5" in viewer
    assert "tail_length=frame_count" not in viewer

    # Async publication adds more legitimate track-layer update sites. Do not
    # assert a brittle exact count; instead verify that every Tracks-layer call
    # inside make_viewer uses the bounded five-frame constant.
    tree = ast.parse(
        viewer,
        filename="dataset_curation/annotation/viewer.py",
    )
    make_viewer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "make_viewer"
    )

    bounded_tail_values = []
    for call in ast.walk(make_viewer):
        if not isinstance(call, ast.Call):
            continue
        if not (
            isinstance(call.func, ast.Name)
            and call.func.id
            in {
                "_sync_tracks_layer",
                "_track_group_layers",
            }
        ):
            continue
        for keyword in call.keywords:
            if keyword.arg == "tail_length":
                bounded_tail_values.append(
                    keyword.value
                )

    assert bounded_tail_values
    assert all(
        isinstance(value, ast.Name)
        and value.id == "TRACK_HISTORY_FRAMES"
        for value in bounded_tail_values
    )

    marker = "def on_dims_change(_event=None) -> None:"
    block = viewer[viewer.index(marker):]
    block = block[
        : block.index("viewer.dims.events.current_step.connect")
    ]

    assert "refresh_tracks=False" in block
    assert "refresh_diagnostic_layers(" in block
