from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataset_curation.annotation.instances.session import (
    AnnotationSession,
)
from dataset_curation.annotation.instances.split import (
    AnnotationError,
)


def _session(tmp_path: Path) -> tuple[AnnotationSession, np.ndarray]:
    supervoxels = np.zeros(
        (1, 4, 6, 8),
        dtype=np.uint16,
    )

    supervoxels[0, :, 0:3, 0:2] = 1
    supervoxels[0, :, 3:6, 0:2] = 2
    supervoxels[0, :, 0:3, 3:5] = 3
    supervoxels[0, :, 3:6, 3:5] = 4
    supervoxels[0, :, 1:5, 6:8] = 5

    base = np.zeros_like(
        supervoxels,
        dtype=np.int32,
    )
    base[supervoxels == 1] = 10
    base[supervoxels == 2] = 10
    base[supervoxels == 3] = 20
    base[supervoxels == 4] = 20
    base[supervoxels == 5] = 30

    session = AnnotationSession(
        sample_id="merge-test",
        timepoints=(0,),
        supervoxels=supervoxels,
        base_instances=base,
        output_dir=tmp_path / "instances",
        resume=False,
    )
    return session, base.copy()


def test_save_merge_merges_entire_parent_instances(tmp_path: Path):
    session, base = _session(tmp_path)

    result = session.apply_merge(
        0,
        [1, 3],
    )

    assert result.source_instance_ids == (10, 20)
    assert result.selected_supervoxel_ids == (1, 3)

    corrected = session.frame(0)
    merged_id = int(result.output_instance_id)

    assert np.all(
        corrected[
            np.isin(
                session.supervoxels[0],
                [1, 2, 3, 4],
            )
        ]
        == merged_id
    )
    assert np.all(
        corrected[
            session.supervoxels[0] == 5
        ]
        == 30
    )
    assert not np.any(corrected == 10)
    assert not np.any(corrected == 20)

    undo = session.undo()
    assert undo.operation_type == "merge"
    np.testing.assert_array_equal(
        session.frame(0),
        base[0],
    )


def test_save_merge_requires_two_distinct_instances(tmp_path: Path):
    session, _ = _session(tmp_path)

    with pytest.raises(
        AnnotationError,
        match="at least two different",
    ):
        session.apply_merge(
            0,
            [1, 2],
        )


def test_merge_is_logged_and_uses_fresh_id(tmp_path: Path):
    session, _ = _session(tmp_path)

    result = session.apply_merge(
        0,
        [2, 4],
    )

    assert result.output_instance_id > 30
    assert session.merge_corrections_in_frame(0) == 1
    op = session.operations[-1]
    assert op["type"] == "merge"
    assert op["source_instance_ids"] == [10, 20]
    assert op["output_instance_id"] == result.output_instance_id
    assert len(op["source_groups"]) == 2


def test_viewer_has_save_merge_and_robust_escape_reset():
    root = Path(__file__).resolve().parents[2]
    viewer = (
        root
        / "dataset_curation"
        / "annotation"
        / "viewer.py"
    ).read_text(encoding="utf-8")

    assert 'text="Save Merge"' in viewer
    assert "spatial_session.apply_merge(" in viewer
    assert "save_merge_button.changed.connect(" in viewer

    start = viewer.index("def save_merge() -> None:")
    end = viewer.index("def mark_hallucination() -> None:", start)
    merge_block = viewer[start:end]
    assert "spatial_authority_changed=True" in merge_block
    assert "refresh_tracks=True" in merge_block

    assert '"Escape"' in viewer
    assert "viewer.bind_key(" in viewer
    assert "clear_spatial_selection()" in viewer
    assert "clear_track_selection()" in viewer
    assert "layer.bind_key(" in viewer
    assert '"Escape"' in viewer
