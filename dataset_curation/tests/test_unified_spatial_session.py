from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dataset_curation.annotation.instances.session import AnnotationSession


def _session(tmp_path: Path) -> AnnotationSession:
    supervoxels = np.zeros(
        (1, 2, 4, 4),
        dtype=np.uint16,
    )
    supervoxels[
        0,
        :,
        :,
        :2,
    ] = 1
    supervoxels[
        0,
        :,
        :,
        2:,
    ] = 2

    instances = np.zeros_like(
        supervoxels
    )
    instances[
        supervoxels > 0
    ] = 7

    return AnnotationSession(
        sample_id="sample",
        timepoints=(0,),
        supervoxels=supervoxels,
        base_instances=instances,
        output_dir=(
            tmp_path
            / "instances"
        ),
        resume=False,
    )


def test_hallucination_removes_selected_sv_and_persists_csv(
    tmp_path: Path,
):
    session = _session(
        tmp_path
    )
    record = (
        session.apply_hallucination(
            0,
            1,
        )
    )

    frame = session.frame(0)
    assert np.all(
        frame[
            session.supervoxels[0]
            == 1
        ]
        == 0
    )
    assert np.all(
        frame[
            session.supervoxels[0]
            == 2
        ]
        == 7
    )
    assert record[
        "previous_instance_id"
    ] == 7
    assert (
        session.hallucinated_supervoxels(
            0
        )
        == {1}
    )

    table = pd.read_csv(
        session.hallucinations_path
    )
    assert table[
        "supervoxel_id"
    ].tolist() == [1]


def test_hallucination_undo_restores_supervoxel(
    tmp_path: Path,
):
    session = _session(
        tmp_path
    )
    session.apply_hallucination(
        0,
        1,
    )
    result = session.undo()

    assert (
        result.operation_type
        == "hallucination"
    )
    frame = session.frame(0)
    assert np.all(
        frame[
            session.supervoxels[0]
            == 1
        ]
        == 7
    )
    assert (
        session.hallucinated_supervoxels(
            0
        )
        == set()
    )


def test_split_then_hallucination_is_lifo_safe(
    tmp_path: Path,
):
    session = _session(
        tmp_path
    )
    split = session.apply_split(
        0,
        [
            [1],
            [2],
        ],
    )
    assert len(
        split.output_instance_ids
    ) == 2

    session.apply_hallucination(
        0,
        1,
    )
    first = session.undo()
    second = session.undo()

    assert (
        first.operation_type
        == "hallucination"
    )
    assert (
        second.operation_type
        == "split"
    )
    assert set(
        np.unique(
            session.frame(0)
        ).tolist()
    ) == {7}
