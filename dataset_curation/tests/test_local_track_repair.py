from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_curation.annotation.background_refresh import (
    build_track_refresh_result,
    capture_track_refresh_snapshot,
)
from dataset_curation.annotation.tracks.local_repair import (
    repair_split_tracks,
)
from dataset_curation.annotation.tracks.session import (
    TrackAnnotationSession,
)
from dataset_curation.annotation.tracks.storage import OutputPaths


def _label_frame(ids: list[int], voxels_each: int = 20) -> np.ndarray:
    result = np.zeros((1, 10, 20), dtype=np.uint16)
    flat = result.reshape(-1)
    cursor = 0
    for instance_id in ids:
        flat[cursor : cursor + voxels_each] = int(instance_id)
        cursor += voxels_each
    return result


def _two_path_session(tmp_path: Path):
    nodes = {
        (0, 10), (1, 10),
        (0, 20), (1, 20),
        (2, 101), (2, 102),
        (3, 30), (4, 30),
        (3, 40), (4, 40),
    }
    session = TrackAnnotationSession(
        sample_id="local-repair",
        source_root=tmp_path / "source",
        output=OutputPaths(tmp_path / "tracks"),
        valid_nodes=nodes,
        base_edges={
            ((0, 10), (1, 10)),
            ((0, 20), (1, 20)),
            ((3, 30), (4, 30)),
            ((3, 40), (4, 40)),
        },
        frame_count=5,
        boundary_entry_nodes=set(),
        boundary_exit_nodes=set(),
        resume=False,
    )

    centers = {
        (0, 10): np.asarray([0.0, 0.0, 0.0]),
        (1, 10): np.asarray([0.0, 0.0, 1.0]),
        (2, 101): np.asarray([0.0, 0.0, 2.0]),
        (3, 30): np.asarray([0.0, 0.0, 3.0]),
        (4, 30): np.asarray([0.0, 0.0, 4.0]),

        (0, 20): np.asarray([0.0, 20.0, 0.0]),
        (1, 20): np.asarray([0.0, 20.0, 1.0]),
        (2, 102): np.asarray([0.0, 20.0, 2.0]),
        (3, 40): np.asarray([0.0, 20.0, 3.0]),
        (4, 40): np.asarray([0.0, 20.0, 4.0]),
    }

    labels = {
        0: _label_frame([10, 20]),
        1: _label_frame([10, 20]),
        2: _label_frame([101, 102]),
        3: _label_frame([30, 40]),
        4: _label_frame([30, 40]),
    }
    return session, centers, labels


def test_local_hungarian_repairs_both_sides_without_global_analysis(
    tmp_path: Path,
):
    session, centers, labels = _two_path_session(tmp_path)
    session.set_deferred_derived_persistence(True)
    rebuilds_before = session._analysis_rebuilds

    result = repair_split_tracks(
        frame=2,
        new_instance_ids=(101, 102),
        track_session=session,
        track_centers=centers,
        labels_for_frame=lambda frame: labels[int(frame)],
        spacing_zyx_um=(1.0, 1.0, 1.0),
    )

    expected = {
        ((1, 10), (2, 101)),
        ((1, 20), (2, 102)),
        ((2, 101), (3, 30)),
        ((2, 102), (3, 40)),
    }

    assert set(result.applied_edges) == expected
    assert expected <= session.forced_edges
    assert session._analysis_rebuilds == rebuilds_before

    payload = json.loads(
        session.output.state_json.read_text(encoding="utf-8")
    )
    assert payload["history"][-1]["type"] == "auto_repair"
    assert payload["history"][-1]["details"]["algorithm"] == (
        "local_hungarian_conservative_v2"
    )


def test_ambiguous_assignment_remains_manual(tmp_path: Path):
    nodes = {
        (0, 10),
        (1, 10),
        (2, 101),
        (2, 102),
    }
    session = TrackAnnotationSession(
        sample_id="ambiguous-repair",
        source_root=tmp_path / "source",
        output=OutputPaths(tmp_path / "tracks"),
        valid_nodes=nodes,
        base_edges={
            ((0, 10), (1, 10)),
        },
        frame_count=3,
        boundary_entry_nodes=set(),
        boundary_exit_nodes=set(),
        resume=False,
    )
    session.set_deferred_derived_persistence(True)

    centers = {
        (0, 10): np.asarray([0.0, 0.0, 0.0]),
        (1, 10): np.asarray([0.0, 0.0, 1.0]),
        (2, 101): np.asarray([0.0, -0.2, 2.0]),
        (2, 102): np.asarray([0.0, 0.2, 2.0]),
    }
    labels = {
        0: _label_frame([10]),
        1: _label_frame([10]),
        2: _label_frame([101, 102]),
    }

    result = repair_split_tracks(
        frame=2,
        new_instance_ids=(101, 102),
        track_session=session,
        track_centers=centers,
        labels_for_frame=lambda frame: labels[int(frame)],
        spacing_zyx_um=(1.0, 1.0, 1.0),
    )

    assert result.applied_edges == ()
    assert result.incoming.candidate_count == 2
    assert result.incoming.ambiguous_assignments == 1
    assert not session.forced_edges


def test_auto_repair_undo_and_background_provenance(tmp_path: Path):
    a = (0, 1)
    b = (1, 2)

    session = TrackAnnotationSession(
        sample_id="repair-provenance",
        source_root=tmp_path / "source",
        output=OutputPaths(tmp_path / "tracks"),
        valid_nodes={a, b},
        base_edges=set(),
        frame_count=2,
        boundary_entry_nodes=set(),
        boundary_exit_nodes=set(),
        resume=False,
    )
    session.set_deferred_derived_persistence(True)

    assert session.apply_auto_repair(
        [(a, b)],
        focus_frame=1,
        details={"algorithm": "test"},
    ) == ((a, b),)
    assert (a, b) in session.auto_repair_edges

    centers = {
        a: np.asarray([0.0, 0.0, 0.0]),
        b: np.asarray([0.0, 0.0, 1.0]),
    }

    snapshot = capture_track_refresh_snapshot(
        generation=1,
        reason="auto-repair-test",
        track_session=session,
        track_centers=centers,
    )
    build_track_refresh_result(snapshot)

    corrected = pd.read_csv(
        session.output.corrected_edges_csv
    )
    assert corrected.loc[0, "origin"] == "auto_repair"

    operation = session.undo()
    assert operation["type"] == "auto_repair"
    assert (a, b) not in session.forced_edges
    assert (a, b) not in session.auto_repair_edges
