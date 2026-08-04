"""Discover division scenes and convert them into single-frame target labels."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.io import load_tracking_scene

from .models import PhenotypeCase


class PhenotypeSceneError(ValueError):
    pass


def discover_scene_paths(root: str | Path) -> tuple[Path, ...]:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"Division scene directory does not exist: {directory}")
    paths = tuple(sorted({path.parent for path in directory.rglob("scene.json")}))
    if not paths:
        raise FileNotFoundError(f"No scene.json files were found below: {directory}")
    return paths


def _case_id(scene_path: Path, root: Path) -> str:
    text = "__".join(scene_path.relative_to(root).parts)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or scene_path.name


def parse_scene(scene_path: str | Path, *, root: str | Path) -> PhenotypeCase:
    path = Path(scene_path)
    scene = load_tracking_scene(path)
    metadata = scene.metadata
    sample_id = str(metadata.get("sample_id", "")).strip()
    if not sample_id:
        raise PhenotypeSceneError("scene.json has no non-empty sample_id")
    raw_selection = metadata.get("selected_cells")
    if not isinstance(raw_selection, dict) or not raw_selection:
        raise PhenotypeSceneError("scene.json has no selected_cells mapping")

    selected: dict[int, tuple[int, ...]] = {}
    for frame_text, values in raw_selection.items():
        frame = int(frame_text)
        if not isinstance(values, list):
            raise PhenotypeSceneError(f"selected_cells[{frame}] must be a list")
        ids = tuple(dict.fromkeys(int(value) for value in values))
        if len(ids) not in {1, 2}:
            raise PhenotypeSceneError(f"frame {frame} has {len(ids)} selected cells; expected one or two")
        selected[frame] = ids

    frames = tuple(sorted(selected))
    counts = [len(selected[frame]) for frame in frames]
    first_two = next((i for i, count in enumerate(counts) if count == 2), None)
    if first_two is None or first_two == 0:
        raise PhenotypeSceneError("scene must contain one-cell parent frames followed by two-cell daughter frames")
    if any(count != 1 for count in counts[:first_two]):
        raise PhenotypeSceneError("all saved frames before the event must contain one selected cell")
    if any(count != 2 for count in counts[first_two:]):
        raise PhenotypeSceneError("all saved frames from the event onward must contain two selected cells")

    event_frame = frames[first_two]
    previous = frames[first_two - 1]
    warnings: list[str] = []
    if event_frame - previous != 1:
        warnings.append(f"last parent to first daughter selection has a gap of {event_frame - previous} frames")
    scene_frames = {int(value) for value in scene.frames.tolist()}
    if event_frame not in scene_frames:
        raise PhenotypeSceneError(f"event frame {event_frame} is absent from frames.npy")

    return PhenotypeCase(
        case_id=_case_id(path, Path(root)),
        scene_path=path,
        sample_id=sample_id,
        selected_cells=selected,
        frame_numbers=frames,
        event_frame=event_frame,
        previous_parent_frame=previous,
        warnings=tuple(warnings),
    )


def scan_scenes(root: str | Path, *, strict: bool = False) -> tuple[list[PhenotypeCase], pd.DataFrame]:
    root_path = Path(root)
    cases: list[PhenotypeCase] = []
    records: list[dict[str, object]] = []
    for scene_path in discover_scene_paths(root_path):
        try:
            case = parse_scene(scene_path, root=root_path)
        except Exception as error:
            records.append({
                "scene_path": str(scene_path),
                "case_id": _case_id(scene_path, root_path),
                "valid": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "warnings": "",
            })
            if strict:
                raise
            continue
        cases.append(case)
        records.append({
            "scene_path": str(scene_path),
            "case_id": case.case_id,
            "valid": True,
            "sample_id": case.sample_id,
            "event_frame": case.event_frame,
            "previous_parent_frame": case.previous_parent_frame,
            "selected_frame_count": len(case.frame_numbers),
            "parent_frame_count": len(case.parent_frames),
            "daughter_frame_count": len(case.daughter_frames),
            "warnings": " | ".join(case.warnings),
            "error_type": "",
            "error": "",
        })
    if not cases:
        raise PhenotypeSceneError(f"No valid division scenes were found below {root_path}")
    return cases, pd.DataFrame(records)


def build_target_labels(cases: list[PhenotypeCase]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for case in cases:
        final_parent = max(case.parent_frames)
        for frame in case.frame_numbers:
            ids = case.selected_cells[frame]
            if frame < case.event_frame:
                subtype = "parent_final" if frame == final_parent else "parent_earlier"
                role = "parent"
            else:
                subtype = "daughter_birth" if frame == case.event_frame else "daughter_later"
                role = "daughter"
            for ordinal, cell_id in enumerate(ids):
                records.append({
                    "case_id": case.case_id,
                    "scene_path": str(case.scene_path),
                    "sample_id": case.sample_id,
                    "frame": int(frame),
                    "cell_id": int(cell_id),
                    "phenotype_role": role,
                    "phenotype_subtype": subtype,
                    "daughter_ordinal": ordinal if role == "daughter" else -1,
                    "event_frame": int(case.event_frame),
                    "relative_frame": int(frame - case.event_frame),
                    "is_primary_target": subtype in {"parent_final", "daughter_birth"},
                })
    result = pd.DataFrame(records)
    duplicates = result.duplicated(["case_id", "sample_id", "frame", "cell_id"], keep=False)
    if duplicates.any():
        raise PhenotypeSceneError("duplicate target labels were generated within a scene")
    return result.sort_values(["case_id", "frame", "cell_id"]).reset_index(drop=True)
