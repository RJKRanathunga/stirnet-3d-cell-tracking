from __future__ import annotations

from pathlib import Path

from dataset_curation.annotation.tracks.session import TrackAnnotationSession
from dataset_curation.annotation.tracks.storage import OutputPaths


def _session(tmp_path: Path) -> TrackAnnotationSession:
    valid_nodes = set()
    base_edges = set()
    for cell_id in range(1, 301):
        a = (0, cell_id)
        b = (4, cell_id)
        valid_nodes.update((a, b))
        base_edges.add((a, b))

    return TrackAnnotationSession(
        sample_id='cache-test',
        source_root=tmp_path / 'source',
        output=OutputPaths(tmp_path / 'tracks'),
        valid_nodes=valid_nodes,
        base_edges=base_edges,
        frame_count=5,
        boundary_entry_nodes=set(),
        boundary_exit_nodes=set(),
        resume=False,
    )


def test_analysis_cache_reused(tmp_path: Path):
    session = _session(tmp_path)
    initial = session._analysis_rebuilds
    assert initial == 1
    for _ in range(10):
        assert session.hidden_nodes
        assert session.hidden_edges
        assert not session.visible_edges
        assert not session.unresolved_start_nodes
        assert not session.unresolved_end_nodes
    assert session._analysis_rebuilds == initial


def test_selection_does_not_rebuild_graph(tmp_path: Path):
    session = _session(tmp_path)
    initial = session._analysis_rebuilds
    session.add_selection((0, 1))
    session.reset_selections()
    assert session._analysis_rebuilds == initial


def test_graph_edit_invalidates_once(tmp_path: Path):
    session = _session(tmp_path)
    initial = session._analysis_rebuilds
    session.add_selection((0, 1))
    session.add_selection((4, 1))
    session.break_selected()
    assert session._analysis_rebuilds == initial + 1
    assert (0, 1) in session.unresolved_end_nodes
    assert (4, 1) in session.unresolved_start_nodes
    assert session._analysis_rebuilds == initial + 1
