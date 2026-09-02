from __future__ import annotations

from pathlib import Path

from dataset_curation.annotation.tracks.session import TrackAnnotationSession
from dataset_curation.annotation.tracks.storage import OutputPaths


def test_spatial_node_sync_invalidates_old_edge_without_deleting_history(
    tmp_path: Path,
):
    old_a = (0, 1)
    old_b = (1, 2)
    session = TrackAnnotationSession(
        sample_id="sample",
        source_root=tmp_path,
        output=OutputPaths(
            tmp_path
            / "tracks"
        ),
        valid_nodes={
            old_a,
            old_b,
        },
        base_edges={
            (
                old_a,
                old_b,
            )
        },
        resume=False,
    )

    removed, added = (
        session.set_frame_nodes(
            0,
            [3],
        )
    )
    assert removed == {
        old_a
    }
    assert added == {
        (0, 3)
    }
    assert session.active_edges == set()

    session.add_selection(
        (0, 3)
    )
    session.add_selection(
        old_b
    )
    edge = session.connect_selected()

    assert edge in session.active_edges
    assert edge in session.visible_edges


def test_complete_moves_component_to_hidden_edges(
    tmp_path: Path,
):
    a = (0, 1)
    b = (1, 2)
    session = TrackAnnotationSession(
        sample_id="sample",
        source_root=tmp_path,
        output=OutputPaths(
            tmp_path
            / "tracks"
        ),
        valid_nodes={
            a,
            b,
        },
        base_edges={
            (
                a,
                b,
            )
        },
        resume=False,
    )
    session.add_selection(a)
    completed = (
        session.complete_selected_components()
    )

    assert completed == {
        a,
        b,
    }
    assert session.visible_edges == set()
    assert session.hidden_edges == {
        (
            a,
            b,
        )
    }
