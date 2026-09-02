from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataset_curation.annotation.layers import (
    ray_pick_label_from_raw,
)
from dataset_curation.annotation.tracks.graph import (
    AnnotationError,
)
from dataset_curation.annotation.tracks.session import (
    TrackAnnotationSession,
)
from dataset_curation.annotation.tracks.storage import (
    OutputPaths,
)


def _session(
    tmp_path: Path,
    *,
    valid_nodes,
    base_edges,
    frame_count,
    boundary_entry_nodes=(),
    boundary_exit_nodes=(),
) -> TrackAnnotationSession:
    return TrackAnnotationSession(
        sample_id="sample",
        source_root=tmp_path / "source",
        output=OutputPaths(
            tmp_path / "tracks"
        ),
        valid_nodes=set(valid_nodes),
        base_edges=set(base_edges),
        frame_count=int(frame_count),
        boundary_entry_nodes=set(
            boundary_entry_nodes
        ),
        boundary_exit_nodes=set(
            boundary_exit_nodes
        ),
        resume=False,
    )


def test_continue_auto_hides_complete_track_and_break_reopens_it(
    tmp_path: Path,
):
    a0 = (0, 1)
    a1 = (1, 1)
    b3 = (3, 2)
    b4 = (4, 2)

    session = _session(
        tmp_path,
        valid_nodes={
            a0,
            a1,
            b3,
            b4,
        },
        base_edges={
            (a0, a1),
            (b3, b4),
        },
        frame_count=5,
    )

    assert not session.hidden_nodes
    assert a1 in session.unresolved_end_nodes
    assert b3 in session.unresolved_start_nodes

    session.add_selection(a1)
    session.add_selection(b3)
    edge = session.connect_selected()

    assert edge == (a1, b3)
    assert session.hidden_nodes == {
        a0,
        a1,
        b3,
        b4,
    }
    assert not session.visible_edges

    session.add_selection(a1)
    session.add_selection(b3)
    session.break_selected()

    assert not session.hidden_nodes
    assert a1 in session.unresolved_end_nodes
    assert b3 in session.unresolved_start_nodes


def test_birth_is_one_parent_to_two_daughters_in_next_frame(
    tmp_path: Path,
):
    parent = (0, 10)
    d1 = (1, 20)
    d2 = (1, 21)
    d1_end = (2, 20)
    d2_end = (2, 21)

    session = _session(
        tmp_path,
        valid_nodes={
            parent,
            d1,
            d2,
            d1_end,
            d2_end,
        },
        base_edges={
            (d1, d1_end),
            (d2, d2_end),
        },
        frame_count=3,
    )

    session.add_selection(d2)
    session.add_selection(parent)
    session.add_selection(d1)
    result = session.birth_selected()

    assert result["parent"] == parent
    assert set(result["daughters"]) == {
        d1,
        d2,
    }
    assert len(session.birth_events) == 1
    assert parent in session.hidden_nodes
    assert d1_end in session.hidden_nodes

    birth_csv = (
        tmp_path
        / "tracks"
        / "birth_events.csv"
    )
    assert birth_csv.is_file()

    session.undo()
    assert not session.birth_events
    assert not session.hidden_nodes


def test_birth_rejects_two_cells_in_earlier_frame(
    tmp_path: Path,
):
    a = (0, 1)
    b = (0, 2)
    c = (1, 3)

    session = _session(
        tmp_path,
        valid_nodes={a, b, c},
        base_edges=set(),
        frame_count=2,
    )
    session.add_selection(a)
    session.add_selection(b)
    session.add_selection(c)

    with pytest.raises(
        AnnotationError,
        match="parent -> two daughters",
    ):
        session.birth_selected()


class _RawLayer:
    def __init__(self):
        self.called = False

    def get_ray_intersections(
        self,
        *,
        position,
        view_direction,
        dims_displayed,
        world,
    ):
        self.called = True
        return (
            np.asarray(
                [7.0, 0.0, 1.0, 1.0]
            ),
            np.asarray(
                [7.0, 4.0, 1.0, 1.0]
            ),
        )

    def world_to_data(self, position):
        return np.asarray(position)


class _Event:
    position = (7.0, 0.0, 1.0, 1.0)
    view_direction = (0.0, 1.0, 0.0, 0.0)
    dims_displayed = (1, 2, 3)


def test_label_pick_uses_raw_volume_ray():
    labels = np.zeros(
        (5, 3, 3),
        dtype=np.uint16,
    )
    labels[2, 1, 1] = 84

    raw_layer = _RawLayer()
    picked = ray_pick_label_from_raw(
        raw_layer,
        labels,
        _Event(),
    )

    assert raw_layer.called
    assert picked == 84


def test_unified_viewer_has_no_manual_complete_or_3d_contour():
    root = Path(__file__).resolve().parents[2]
    viewer = (
        root
        / "dataset_curation"
        / "annotation"
        / "viewer.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "Complete Track" not in viewer
    assert "complete_track_button" not in viewer
    assert ".contour =" not in viewer
    assert "ray_pick_label_from_raw" in viewer
    assert "Track selections were preserved." in viewer
    assert 'text="Birth"' in viewer
