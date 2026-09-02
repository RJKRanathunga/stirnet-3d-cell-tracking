from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_curation.annotation.background_refresh import (
    build_track_refresh_result,
    capture_track_refresh_snapshot,
)
from dataset_curation.annotation.tracks.session import (
    TrackAnnotationSession,
)
from dataset_curation.annotation.tracks.storage import (
    OutputPaths,
)


def _session(tmp_path: Path) -> TrackAnnotationSession:
    nodes = {
        (0, 1),
        (1, 1),
        (3, 2),
        (4, 2),
    }
    return TrackAnnotationSession(
        sample_id="async-refresh-test",
        source_root=tmp_path / "source",
        output=OutputPaths(
            tmp_path / "tracks"
        ),
        valid_nodes=nodes,
        base_edges={
            ((0, 1), (1, 1)),
            ((3, 2), (4, 2)),
        },
        frame_count=5,
        boundary_entry_nodes=set(),
        boundary_exit_nodes=set(),
        resume=False,
    )


def _centers() -> dict[tuple[int, int], np.ndarray]:
    return {
        (0, 1): np.asarray(
            [1.0, 1.0, 0.0]
        ),
        (1, 1): np.asarray(
            [1.0, 1.0, 1.0]
        ),
        (3, 2): np.asarray(
            [1.0, 2.0, 3.0]
        ),
        (4, 2): np.asarray(
            [1.0, 2.0, 4.0]
        ),
    }


def test_deferred_track_edit_keeps_canonical_json_immediate(
    tmp_path: Path,
):
    session = _session(tmp_path)
    initial_rebuilds = session._analysis_rebuilds

    session.set_deferred_derived_persistence(
        True
    )

    session.add_selection((1, 1))
    session.add_selection((3, 2))
    edge = session.connect_selected()

    # The GUI path no longer forces full graph analysis/export merely to
    # accept the annotation.
    assert session._analysis_rebuilds == initial_rebuilds
    assert edge in session.forced_edges

    payload = json.loads(
        session.output.state_json.read_text(
            encoding="utf-8"
        )
    )
    assert payload["forced_edges"]
    assert "automatic_hidden_nodes" not in payload
    assert "unresolved_start_nodes" not in payload
    assert "unresolved_end_nodes" not in payload


def test_background_snapshot_rebuilds_and_exports_latest_state(
    tmp_path: Path,
):
    session = _session(tmp_path)
    session.set_deferred_derived_persistence(
        True
    )

    session.add_selection((1, 1))
    session.add_selection((3, 2))
    session.connect_selected()

    snapshot = capture_track_refresh_snapshot(
        generation=7,
        reason="continue",
        track_session=session,
        track_centers=_centers(),
    )
    result = build_track_refresh_result(
        snapshot
    )

    assert result.generation == 7
    assert result.reason == "continue"

    # The two original fragments now form a complete component.
    assert result.visible_edge_count == 0
    assert result.hidden_node_count == 4
    assert result.unresolved_start_count == 0
    assert result.unresolved_end_count == 0

    assert session.output.current_tracks_csv.is_file()
    assert session.output.corrected_edges_csv.is_file()
    assert session.output.overrides_csv.is_file()

    overrides = pd.read_csv(
        session.output.overrides_csv
    )
    assert "CONTINUE" in set(
        overrides["action"].astype(str)
    )


def test_viewer_actions_schedule_background_refresh():
    root = Path(__file__).resolve().parents[2]
    viewer = (
        root
        / "dataset_curation"
        / "annotation"
        / "viewer.py"
    ).read_text(encoding="utf-8")

    assert "Background: idle" in viewer
    assert "TrackRefreshCoordinator" in viewer
    assert "QTimer" in viewer

    for function_name in (
        "ignore_selected",
        "save_split",
        "save_merge",
        "mark_hallucination",
        "undo_spatial",
        "continue_track",
        "break_track",
        "mark_birth",
        "undo_track",
    ):
        start = viewer.index(
            f"def {function_name}("
        )
        tail = viewer[start:]
        next_def = tail.find(
            "\n    def ",
            1,
        )
        block = (
            tail
            if next_def < 0
            else tail[:next_def]
        )
        assert (
            "request_background_track_refresh("
            in block
        ), function_name

    for function_name in (
        "save_split",
        "save_merge",
        "mark_hallucination",
        "undo_spatial",
    ):
        start = viewer.index(
            f"def {function_name}("
        )
        tail = viewer[start:]
        next_def = tail.find(
            "\n    def ",
            1,
        )
        block = (
            tail
            if next_def < 0
            else tail[:next_def]
        )
        assert "refresh_tracks=False" in block
        assert "refresh_tracks=True" not in block
