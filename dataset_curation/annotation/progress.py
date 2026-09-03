from __future__ import annotations

# DATASET_CURATION_ANNOTATION_PROGRESS_V1

"""Read-only annotation progress derived from canonical persisted state."""

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class AnnotationProgress:
    frame_count: int
    spatial_frames: tuple[int, ...]
    spatial_operation_count: int
    spatial_operation_counts: dict[str, int]
    manual_frame_file_count: int
    track_frames: tuple[int, ...]
    manual_continue_edge_count: int
    manual_break_edge_count: int
    birth_event_count: int
    ignored_event_count: int
    activity_frames: tuple[int, ...]

    @property
    def spatial_frame_count(self) -> int:
        return len(self.spatial_frames)

    @property
    def track_frame_count(self) -> int:
        return len(self.track_frames)

    @property
    def activity_frame_count(self) -> int:
        return len(self.activity_frames)

    def percent(self, count: int) -> float:
        return 0.0 if self.frame_count <= 0 else 100.0 * float(count) / float(self.frame_count)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _node(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Invalid serialized node: {value!r}")
    return int(value[0]), int(value[1])


def _edge(value: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Invalid serialized edge: {value!r}")
    a = _node(value[0])
    b = _node(value[1])
    return (a, b) if a <= b else (b, a)


def _valid_frame(frame: int, frame_count: int) -> bool:
    return 0 <= int(frame) < int(frame_count)


def _compact_ranges(frames: Iterable[int]) -> str:
    values = sorted({int(v) for v in frames})
    if not values:
        return "-"
    out: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        out.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    out.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(out)


def compute_annotation_progress(
    annotation_root: str | Path,
    *,
    frame_count: int,
) -> AnnotationProgress:
    root = Path(annotation_root)
    frame_count = int(frame_count)
    if frame_count < 0:
        raise ValueError("frame_count cannot be negative")

    spatial_root = root / "instances"
    track_root = root / "tracks"

    spatial_payload = _read_json(spatial_root / "spatial_operations.json")
    operations = spatial_payload.get("operations", [])
    if not isinstance(operations, list):
        raise ValueError("spatial_operations.json 'operations' must be a list")

    spatial_frames: set[int] = set()
    counts: Counter[str] = Counter()
    for op in operations:
        if not isinstance(op, dict):
            continue
        op_type = str(op.get("type", "unknown")).strip() or "unknown"
        counts[op_type] += 1
        try:
            frame = int(op["timepoint"])
        except (KeyError, TypeError, ValueError):
            continue
        if _valid_frame(frame, frame_count):
            spatial_frames.add(frame)

    manual_file_count = sum(
        1 for path in spatial_root.glob("manual_instances_t*.npy") if path.is_file()
    )

    track_payload = _read_json(track_root / "track_annotations.json")
    forced_edges = {_edge(v) for v in track_payload.get("forced_edges", [])}
    broken_edges = {_edge(v) for v in track_payload.get("broken_edges", [])}
    birth_events = track_payload.get("birth_events", [])
    ignored_events = track_payload.get("ignored_events", [])
    if not isinstance(birth_events, list):
        raise ValueError("track_annotations.json 'birth_events' must be a list")
    if not isinstance(ignored_events, list):
        raise ValueError("track_annotations.json 'ignored_events' must be a list")

    birth_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    track_frames: set[int] = set()

    for event in birth_events:
        if not isinstance(event, dict):
            continue
        try:
            parent = _node(event["parent"])
        except (KeyError, TypeError, ValueError):
            continue
        if _valid_frame(parent[0], frame_count):
            track_frames.add(parent[0])
        daughters = event.get("daughters", [])
        if not isinstance(daughters, list):
            continue
        for raw in daughters:
            try:
                daughter = _node(raw)
            except (TypeError, ValueError):
                continue
            birth_edges.add((parent, daughter) if parent <= daughter else (daughter, parent))
            if _valid_frame(daughter[0], frame_count):
                track_frames.add(daughter[0])

    manual_continue_edges = forced_edges - birth_edges
    for edge in manual_continue_edges | broken_edges:
        for node in edge:
            if _valid_frame(node[0], frame_count):
                track_frames.add(node[0])

    for event in ignored_events:
        if not isinstance(event, dict):
            continue
        for key in ("endpoint", "selected_node"):
            if key not in event:
                continue
            try:
                node = _node(event[key])
            except (TypeError, ValueError):
                continue
            if _valid_frame(node[0], frame_count):
                track_frames.add(node[0])

    activity_frames = spatial_frames | track_frames

    return AnnotationProgress(
        frame_count=frame_count,
        spatial_frames=tuple(sorted(spatial_frames)),
        spatial_operation_count=int(sum(counts.values())),
        spatial_operation_counts=dict(sorted(counts.items())),
        manual_frame_file_count=int(manual_file_count),
        track_frames=tuple(sorted(track_frames)),
        manual_continue_edge_count=int(len(manual_continue_edges)),
        manual_break_edge_count=int(len(broken_edges)),
        birth_event_count=int(len(birth_events)),
        ignored_event_count=int(len(ignored_events)),
        activity_frames=tuple(sorted(activity_frames)),
    )


def format_annotation_progress(
    progress: AnnotationProgress,
    *,
    split: str,
    volume_id: str,
    annotation_set: str,
) -> str:
    width = 96
    lines = [
        "=" * width,
        "DATASET CURATION — ANNOTATION PROGRESS",
        "=" * width,
        f"split                : {split}",
        f"volume               : {volume_id}",
        f"annotation set       : {annotation_set}",
        f"frames               : {progress.frame_count}",
        "=" * width,
        "",
        "SPATIAL CORRECTIONS",
        "-" * width,
        (
            "frames with actions  : "
            f"{progress.spatial_frame_count}/{progress.frame_count} "
            f"({progress.percent(progress.spatial_frame_count):.1f}%)"
        ),
        f"saved operations     : {progress.spatial_operation_count}",
    ]

    preferred = ("split", "merge", "hallucination")
    emitted: set[str] = set()
    for key in preferred:
        if key in progress.spatial_operation_counts:
            lines.append(f"  {key:<18}: {progress.spatial_operation_counts[key]}")
            emitted.add(key)
    for key, value in progress.spatial_operation_counts.items():
        if key not in emitted:
            lines.append(f"  {key:<18}: {value}")

    lines.extend(
        [
            f"manual frame files   : {progress.manual_frame_file_count}",
            f"spatial frame IDs    : {_compact_ranges(progress.spatial_frames)}",
            "",
            "TRACK ANNOTATIONS",
            "-" * width,
            (
                "frames with actions  : "
                f"{progress.track_frame_count}/{progress.frame_count} "
                f"({progress.percent(progress.track_frame_count):.1f}%)"
            ),
            f"manual Continue      : {progress.manual_continue_edge_count}",
            f"manual Break         : {progress.manual_break_edge_count}",
            f"Birth events         : {progress.birth_event_count}",
            f"ignored/deferred     : {progress.ignored_event_count}",
            f"track frame IDs      : {_compact_ranges(progress.track_frames)}",
            "",
            "OVERALL SAVED ANNOTATION ACTIVITY",
            "-" * width,
            (
                "unique frames        : "
                f"{progress.activity_frame_count}/{progress.frame_count} "
                f"({progress.percent(progress.activity_frame_count):.1f}%)"
            ),
            f"frame IDs            : {_compact_ranges(progress.activity_frames)}",
            "",
            (
                "NOTE: this is saved annotation activity, not reviewed-frame coverage. "
                "The current annotator does not persist a marker when a frame is "
                "inspected and accepted with no correction."
            ),
            "=" * width,
        ]
    )
    return "\n".join(lines)
