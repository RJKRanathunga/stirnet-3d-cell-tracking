# DATASET_CURATION_ANNOTATION_PROGRESS_V1
from __future__ import annotations

import json
from pathlib import Path

from dataset_curation.annotation.progress import (
    compute_annotation_progress,
    format_annotation_progress,
)
from dataset_curation.cli import build_parser


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_progress_counts_unique_saved_activity_frames(tmp_path: Path):
    root = tmp_path / "main"
    _write_json(
        root / "instances" / "spatial_operations.json",
        {"operations": [
            {"type": "split", "timepoint": 3},
            {"type": "merge", "timepoint": 3},
            {"type": "hallucination", "timepoint": 7},
        ]},
    )
    (root / "instances" / "manual_instances_t003.npy").write_bytes(b"x")
    (root / "instances" / "manual_instances_t007.npy").write_bytes(b"x")
    _write_json(
        root / "tracks" / "track_annotations.json",
        {
            "forced_edges": [
                [[7, 10], [8, 20]],
                [[9, 30], [10, 40]],
                [[9, 30], [10, 41]],
            ],
            "broken_edges": [[[12, 50], [13, 60]]],
            "birth_events": [{
                "parent": [9, 30],
                "daughters": [[10, 40], [10, 41]],
            }],
            "ignored_events": [{
                "event_type": "instance",
                "endpoint": [15, 70],
                "selected_node": [15, 70],
            }],
        },
    )

    progress = compute_annotation_progress(root, frame_count=20)
    assert progress.spatial_frames == (3, 7)
    assert progress.spatial_operation_count == 3
    assert progress.manual_frame_file_count == 2
    assert progress.manual_continue_edge_count == 1
    assert progress.manual_break_edge_count == 1
    assert progress.birth_event_count == 1
    assert progress.ignored_event_count == 1
    assert progress.track_frames == (7, 8, 9, 10, 12, 13, 15)
    assert progress.activity_frames == (3, 7, 8, 9, 10, 12, 13, 15)


def test_progress_handles_missing_state_and_explains_semantics(tmp_path: Path):
    progress = compute_annotation_progress(tmp_path / "main", frame_count=100)
    text = format_annotation_progress(
        progress,
        split="train",
        volume_id="volume",
        annotation_set="main",
    )
    assert progress.activity_frame_count == 0
    assert "not reviewed-frame coverage" in text


def test_cli_exposes_progress_command():
    args = build_parser().parse_args(
        ["progress", "--split", "train", "--id", "abc"]
    )
    assert args.command == "progress"
    assert args.id == "abc"
