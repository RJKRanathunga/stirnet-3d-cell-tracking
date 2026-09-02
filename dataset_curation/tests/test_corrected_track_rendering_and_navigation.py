from __future__ import annotations

from pathlib import Path

import numpy as np

from dataset_curation.annotation.instances.centers import (
    supervoxel_interior_points,
)
from dataset_curation.annotation.tracks.current import (
    build_current_track_table,
    prepare_current_endpoint_groups,
)


def _centers(nodes):
    return {
        node: np.asarray([1.0, float(node[1]), float(node[0])])
        for node in nodes
    }


def test_continue_materializes_as_normal_merged_tracklet():
    a0, a1 = (0, 1), (1, 1)
    b3, b4 = (3, 2), (4, 2)
    c6, c7 = (6, 3), (7, 3)
    nodes = {a0, a1, b3, b4, c6, c7}

    edges_after_first_continue = {
        (a0, a1),
        (a1, b3),
        (b3, b4),
        (c6, c7),
    }
    tracks = build_current_track_table(
        valid_nodes=nodes,
        active_edges=edges_after_first_continue,
        centers=_centers(nodes),
    )

    first_ids = tracks.loc[
        tracks["cell_id"].isin([1, 2]),
        "track_id",
    ].unique()
    assert len(first_ids) == 1

    groups = prepare_current_endpoint_groups(
        tracks,
        unresolved_start_nodes={c6},
        unresolved_end_nodes={b4},
        boundary_entry_nodes=set(),
        boundary_exit_nodes=set(),
        hidden_nodes=set(),
    )

    broken_ids = set(
        groups.ended_failure_tracks["track_id"].astype(int).tolist()
    )
    assert int(first_ids[0]) in broken_ids

    broken_rows = groups.ended_failure_tracks[
        groups.ended_failure_tracks["track_id"] == first_ids[0]
    ]
    assert set(broken_rows["frame"].astype(int)) == {0, 1, 3, 4}


def test_no_broken_layer_after_all_gaps_are_connected():
    a0, a1 = (0, 1), (1, 1)
    b3, b4 = (3, 2), (4, 2)
    c6, c7 = (6, 3), (7, 3)
    nodes = {a0, a1, b3, b4, c6, c7}
    edges = {
        (a0, a1),
        (a1, b3),
        (b3, b4),
        (b4, c6),
        (c6, c7),
    }
    tracks = build_current_track_table(
        valid_nodes=nodes,
        active_edges=edges,
        centers=_centers(nodes),
    )

    groups = prepare_current_endpoint_groups(
        tracks,
        unresolved_start_nodes=set(),
        unresolved_end_nodes=set(),
        boundary_entry_nodes=set(),
        boundary_exit_nodes=set(),
        hidden_nodes=set(nodes),
    )

    assert groups.new_failure_tracks.empty
    assert groups.ended_failure_tracks.empty


def test_supervoxel_text_point_is_inside_supervoxel():
    labels = np.zeros((8, 12, 12), dtype=np.uint16)
    labels[1:7, 2:10, 2:5] = 7
    labels[4:7, 5:10, 5:10] = 7

    points, properties = supervoxel_interior_points(labels)
    assert properties["sv_id"].tolist() == [7]
    point = np.rint(points[0]).astype(int)
    assert int(labels[tuple(point.tolist())]) == 7


def test_viewer_navigation_does_not_rebuild_global_tracks():
    root = Path(__file__).resolve().parents[2]
    viewer = (
        root / "dataset_curation" / "annotation" / "viewer.py"
    ).read_text(encoding="utf-8")

    marker = "def on_dims_change(_event=None) -> None:"
    block = viewer[viewer.index(marker):]
    block = block[: block.index("viewer.dims.events.current_step.connect")]
    assert "refresh_current_frame_layers(refresh_tracks=False)" in block
