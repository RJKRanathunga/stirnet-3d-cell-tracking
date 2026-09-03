from __future__ import annotations

# DATASET_CURATION_EXACT_BOUNDARY_TOUCH_V1

from pathlib import Path

import numpy as np

from dataset_curation.annotation.tracks.diagnostics import (
    boundary_touching_instance_ids,
)
from dataset_curation.annotation.tracks.session import TrackAnnotationSession
from dataset_curation.annotation.tracks.storage import OutputPaths


def _session(
    tmp_path: Path,
    *,
    valid_nodes,
    base_edges,
    frame_count=4,
    boundary_entry_nodes=(),
    boundary_exit_nodes=(),
    exact_boundary_touch_nodes=(),
):
    return TrackAnnotationSession(
        sample_id="sample",
        source_root=tmp_path / "source",
        output=OutputPaths(tmp_path / "tracks"),
        valid_nodes=set(valid_nodes),
        base_edges=set(base_edges),
        frame_count=int(frame_count),
        boundary_entry_nodes=set(boundary_entry_nodes),
        boundary_exit_nodes=set(boundary_exit_nodes),
        resume=False,
        exact_boundary_touch_nodes=set(exact_boundary_touch_nodes),
    )


def test_boundary_touching_instance_ids_checks_all_six_faces():
    labels = np.zeros((5, 7, 9), dtype=np.uint16)

    labels[0, 1, 1] = 1
    labels[-1, 1, 2] = 2
    labels[2, 0, 3] = 3
    labels[2, -1, 4] = 4
    labels[3, 3, 0] = 5
    labels[3, 4, -1] = 6
    labels[2, 3, 4] = 7  # interior only

    assert boundary_touching_instance_ids(labels) == {1, 2, 3, 4, 5, 6}


def test_exact_boundary_contact_is_additive_to_existing_heuristic(tmp_path: Path):
    start = (1, 10)
    end = (2, 10)

    session = _session(
        tmp_path,
        valid_nodes={start, end},
        base_edges={(start, end)},
        frame_count=4,
        boundary_exit_nodes={end},
        exact_boundary_touch_nodes={start},
    )

    assert start in session.boundary_entry_nodes
    assert start in session.boundary_exit_nodes
    assert end in session.boundary_exit_nodes
    assert not session.unresolved_start_nodes
    assert not session.unresolved_end_nodes
    assert session.hidden_nodes == {start, end}


def test_dynamic_exact_boundary_contact_updates_after_spatial_edit(tmp_path: Path):
    start = (1, 10)
    end = (2, 10)

    session = _session(
        tmp_path,
        valid_nodes={start, end},
        base_edges={(start, end)},
        frame_count=4,
        boundary_exit_nodes={end},
    )

    assert start in session.unresolved_start_nodes
    assert end not in session.unresolved_end_nodes
    assert not session.hidden_nodes

    removed, added = session.set_frame_exact_boundary_touch_nodes(1, [10])
    assert removed == set()
    assert added == {start}
    assert not session.unresolved_start_nodes
    assert session.hidden_nodes == {start, end}

    removed, added = session.set_frame_exact_boundary_touch_nodes(1, [])
    assert removed == {start}
    assert added == set()
    assert start in session.unresolved_start_nodes
    assert end in session.boundary_exit_nodes
    assert end not in session.unresolved_end_nodes


def test_viewer_recomputes_exact_boundary_after_spatial_authority_change():
    root = Path(__file__).resolve().parents[2]
    viewer = (
        root / "dataset_curation" / "annotation" / "viewer.py"
    ).read_text(encoding="utf-8")

    assert "boundary_touching_instance_ids" in viewer
    assert "set_frame_exact_boundary_touch_nodes" in viewer


def test_runner_initializes_exact_boundary_nodes_from_current_labels():
    root = Path(__file__).resolve().parents[2]
    runner = (
        root / "dataset_curation" / "annotation" / "curation_runner.py"
    ).read_text(encoding="utf-8")

    assert "boundary_touching_instance_ids" in runner
    assert "exact_boundary_touch_nodes" in runner
    assert "exact_boundary_touch_nodes=exact_boundary_touch_nodes" in runner
