from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset_curation.annotation.tracks.current import (
    diagnostic_events_for_node,
    prepare_current_endpoint_groups,
)
from dataset_curation.annotation.tracks.session import (
    TrackAnnotationSession,
)
from dataset_curation.annotation.tracks.storage import (
    OutputPaths,
)


def _tracks(track_id: int, start: int, end: int) -> pd.DataFrame:
    frames = list(range(start, end + 1))
    return pd.DataFrame(
        {
            "track_id": [track_id] * len(frames),
            "frame": frames,
            "cell_id": [track_id] * len(frames),
            "z": [1.0] * len(frames),
            "y": [2.0] * len(frames),
            "x": [3.0] * len(frames),
        }
    )


def test_ignore_is_persisted_as_pending_review(tmp_path: Path):
    nodes = {
        (0, 1),
        (1, 1),
        (3, 2),
        (4, 2),
    }
    session = TrackAnnotationSession(
        sample_id="ignore-test",
        source_root=tmp_path / "source",
        output=OutputPaths(tmp_path / "tracks"),
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

    added = session.ignore_events(
        selected_node=(1, 1),
        source_mode="tracking",
        events=[("broken", (1, 1))],
    )

    assert len(added) == 1
    assert (1, 1) in session.ignored_end_nodes
    assert session.output.ignored_events_csv.is_file()

    exported = pd.read_csv(session.output.ignored_events_csv)
    assert exported.loc[0, "event_type"] == "broken"
    assert exported.loc[0, "source_mode"] == "tracking"
    assert exported.loc[0, "review_status"] == "pending"


def test_ignored_endpoints_are_hidden_from_diagnostics():
    broken_tracks = _tracks(1, 0, 4)
    broken = prepare_current_endpoint_groups(
        broken_tracks,
        unresolved_start_nodes=set(),
        unresolved_end_nodes={(4, 1)},
        boundary_entry_nodes=set(),
        boundary_exit_nodes=set(),
        hidden_nodes=set(),
        ignored_start_nodes=set(),
        ignored_end_nodes={(4, 1)},
    )
    assert broken.ended_failure_tracks.empty

    new_tracks = _tracks(2, 3, 8)
    new = prepare_current_endpoint_groups(
        new_tracks,
        unresolved_start_nodes={(3, 2)},
        unresolved_end_nodes=set(),
        boundary_entry_nodes=set(),
        boundary_exit_nodes=set(),
        hidden_nodes=set(),
        ignored_start_nodes={(3, 2)},
        ignored_end_nodes=set(),
    )
    assert new.new_failure_tracks.empty


def test_diagnostic_event_identity_uses_endpoint():
    tracks = _tracks(7, 2, 8)
    groups = prepare_current_endpoint_groups(
        tracks,
        unresolved_start_nodes={(2, 7)},
        unresolved_end_nodes={(8, 7)},
        boundary_entry_nodes=set(),
        boundary_exit_nodes=set(),
        hidden_nodes=set(),
    )

    events = diagnostic_events_for_node(
        groups,
        (5, 7),
        current_frame=5,
        horizon_frames=5,
    )

    assert ("new", (2, 7)) in events
    assert ("broken", (8, 7)) in events


def test_viewer_has_one_common_ignore_button():
    root = Path(__file__).resolve().parents[2]
    viewer = (
        root / "dataset_curation" / "annotation" / "viewer.py"
    ).read_text(encoding="utf-8")

    assert viewer.count('text="Ignore"') == 1
    assert "def ignore_selected() -> None:" in viewer
    assert "ignore_button.changed.connect(" in viewer
    assert "current_instance_for_supervoxel(" in viewer
