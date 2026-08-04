"""Discovery and validation of manually extracted division scenes."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.io import load_tracking_scene

from .models import DivisionCase


class DivisionSceneError(ValueError):
    """Raised when a scene cannot represent one unambiguous 1-to-2 division."""


def discover_scene_paths(root: str | Path) -> tuple[Path, ...]:
    """Return every directory below *root* containing ``scene.json``."""
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"Division scene directory does not exist: {directory}")
    paths = tuple(sorted({path.parent for path in directory.rglob("scene.json")}))
    if not paths:
        raise FileNotFoundError(f"No scene.json files were found below: {directory}")
    return paths


def _case_id(scene_path: Path, root: Path) -> str:
    relative = scene_path.relative_to(root)
    text = "__".join(relative.parts)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or scene_path.name


def parse_division_scene(scene_path: str | Path, *, root: str | Path) -> DivisionCase:
    """Parse one scene and infer the first saved 1-to-2 transition."""
    path = Path(scene_path)
    scene = load_tracking_scene(path)
    metadata = scene.metadata

    sample_id = str(metadata.get("sample_id", "")).strip()
    if not sample_id:
        raise DivisionSceneError("scene.json has no non-empty sample_id")

    raw_selection = metadata.get("selected_cells")
    if not isinstance(raw_selection, dict) or not raw_selection:
        raise DivisionSceneError("scene.json has no selected_cells mapping")

    selected: dict[int, tuple[int, ...]] = {}
    for frame_text, values in raw_selection.items():
        frame = int(frame_text)
        if not isinstance(values, list):
            raise DivisionSceneError(f"selected_cells[{frame}] must be a list")
        ids = tuple(dict.fromkeys(int(value) for value in values))
        if len(ids) not in {1, 2}:
            raise DivisionSceneError(
                f"frame {frame} has {len(ids)} selected cells; expected one or two"
            )
        selected[frame] = ids

    selected_frames = tuple(sorted(selected))
    counts = [len(selected[frame]) for frame in selected_frames]
    first_two_index = next((i for i, count in enumerate(counts) if count == 2), None)
    if first_two_index is None:
        raise DivisionSceneError("no saved frame contains two selected child cells")
    if first_two_index == 0:
        raise DivisionSceneError("the scene begins with two cells and has no parent frame")

    if any(count != 1 for count in counts[:first_two_index]):
        raise DivisionSceneError("frames before the inferred event are not all one-cell frames")
    if any(count != 2 for count in counts[first_two_index:]):
        raise DivisionSceneError("frames after the inferred event are not all two-cell frames")

    event_frame = selected_frames[first_two_index]
    previous_parent_frame = selected_frames[first_two_index - 1]
    warnings: list[str] = []
    transition_gap = event_frame - previous_parent_frame
    if transition_gap != 1:
        warnings.append(
            "the first two-cell frame is not immediately after the last saved parent frame "
            f"(gap={transition_gap})"
        )

    scene_frames = tuple(int(value) for value in scene.frames.tolist())
    missing_selection_frames = tuple(
        frame for frame in scene_frames if frame not in selected
    )
    if missing_selection_frames:
        warnings.append(
            "scene contains frames without manual cell selections: "
            + ", ".join(map(str, missing_selection_frames))
        )

    if event_frame not in scene_frames:
        raise DivisionSceneError(
            f"inferred event frame {event_frame} is absent from frames.npy"
        )

    root_path = Path(root)
    return DivisionCase(
        case_id=_case_id(path, root_path),
        scene_path=path,
        sample_id=sample_id,
        frame_numbers=selected_frames,
        selected_cells=selected,
        event_frame=event_frame,
        previous_parent_frame=previous_parent_frame,
        transition_gap_frames=transition_gap,
        warnings=tuple(warnings),
    )


def scan_division_scenes(
    root: str | Path,
    *,
    strict: bool = False,
) -> tuple[list[DivisionCase], pd.DataFrame]:
    """Discover scenes, returning valid cases and a complete validation table."""
    root_path = Path(root)
    cases: list[DivisionCase] = []
    records: list[dict[str, object]] = []

    for scene_path in discover_scene_paths(root_path):
        try:
            case = parse_division_scene(scene_path, root=root_path)
        except Exception as error:
            records.append(
                {
                    "scene_path": str(scene_path),
                    "case_id": _case_id(scene_path, root_path),
                    "valid": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "warnings": "",
                }
            )
            if strict:
                raise
            continue

        cases.append(case)
        records.append(
            {
                "scene_path": str(scene_path),
                "case_id": case.case_id,
                "valid": True,
                "sample_id": case.sample_id,
                "event_frame": case.event_frame,
                "previous_parent_frame": case.previous_parent_frame,
                "transition_gap_frames": case.transition_gap_frames,
                "selected_frame_count": len(case.frame_numbers),
                "parent_frame_count": len(case.parent_frames),
                "child_frame_count": len(case.child_frames),
                "error_type": "",
                "error": "",
                "warnings": " | ".join(case.warnings),
            }
        )

    if not cases:
        raise DivisionSceneError(
            f"No valid one-to-two division scenes were found below {root_path}"
        )
    return cases, pd.DataFrame(records)
