"""Discovery and validation of manually curated small-cell scenes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import pandas as pd


class SmallCellSceneError(ValueError):
    """Raised when a curated scene cannot be interpreted safely."""


@dataclass(frozen=True)
class SmallCellScene:
    case_id: str
    scene_path: Path
    sample_id: str
    selected_cells: dict[int, tuple[int, ...]]
    source_tracks_csv: Path | None
    frame_start: int
    frame_end: int
    warnings: tuple[str, ...]


def discover_scene_paths(root: str | Path) -> tuple[Path, ...]:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"Small-cell scene directory does not exist: {directory}")
    paths = tuple(sorted({path.parent for path in directory.rglob("scene.json")}))
    if not paths:
        raise FileNotFoundError(f"No scene.json files were found below: {directory}")
    return paths


def _case_id(scene_path: Path, root: Path) -> str:
    text = "__".join(scene_path.relative_to(root).parts)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or scene_path.name


def _source_tracks_path(metadata: dict[str, object]) -> Path | None:
    source = metadata.get("source")
    if not isinstance(source, dict):
        return None
    raw = source.get("tracks_csv")
    if raw is None or not str(raw).strip():
        return None
    return Path(str(raw))


def parse_scene(scene_path: str | Path, *, root: str | Path) -> SmallCellScene:
    path = Path(scene_path)
    with (path / "scene.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    sample_id = str(metadata.get("sample_id", "")).strip()
    if not sample_id:
        raise SmallCellSceneError("scene.json has no non-empty sample_id")
    raw_selection = metadata.get("selected_cells")
    if not isinstance(raw_selection, dict) or not raw_selection:
        raise SmallCellSceneError("scene.json has no selected_cells mapping")

    selected: dict[int, tuple[int, ...]] = {}
    for frame_text, values in raw_selection.items():
        frame = int(frame_text)
        if frame < 0:
            raise SmallCellSceneError(f"selected frame cannot be negative: {frame}")
        if not isinstance(values, list):
            raise SmallCellSceneError(f"selected_cells[{frame}] must be a list")
        ids = tuple(dict.fromkeys(int(value) for value in values))
        if not ids:
            raise SmallCellSceneError(f"selected_cells[{frame}] is empty")
        if any(value <= 0 for value in ids):
            raise SmallCellSceneError(f"selected_cells[{frame}] contains a non-positive cell ID")
        selected[frame] = ids

    frames = sorted(selected)
    warnings: list[str] = []
    gaps = [current - previous for previous, current in zip(frames[:-1], frames[1:])]
    if any(gap > 1 for gap in gaps):
        warnings.append("manual selections contain one or more missing frames")
    counts = {len(values) for values in selected.values()}
    if len(counts) > 1:
        warnings.append("the number of selected cells changes between frames; trajectories will be linked geometrically")

    return SmallCellScene(
        case_id=_case_id(path, Path(root)),
        scene_path=path,
        sample_id=sample_id,
        selected_cells=selected,
        source_tracks_csv=_source_tracks_path(metadata),
        frame_start=min(frames),
        frame_end=max(frames),
        warnings=tuple(warnings),
    )


def scan_scenes(root: str | Path, *, strict: bool = False) -> tuple[list[SmallCellScene], pd.DataFrame]:
    root_path = Path(root)
    cases: list[SmallCellScene] = []
    records: list[dict[str, object]] = []
    for scene_path in discover_scene_paths(root_path):
        try:
            case = parse_scene(scene_path, root=root_path)
        except Exception as error:
            records.append({
                "scene_path": str(scene_path),
                "case_id": _case_id(scene_path, root_path),
                "valid": False,
                "sample_id": "",
                "selected_frame_count": 0,
                "selected_observation_count": 0,
                "source_tracks_csv": "",
                "warnings": "",
                "error_type": type(error).__name__,
                "error": str(error),
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
            "selected_frame_count": len(case.selected_cells),
            "selected_observation_count": sum(len(values) for values in case.selected_cells.values()),
            "frame_start": case.frame_start,
            "frame_end": case.frame_end,
            "source_tracks_csv": str(case.source_tracks_csv or ""),
            "warnings": " | ".join(case.warnings),
            "error_type": "",
            "error": "",
        })
    if not cases:
        raise SmallCellSceneError(f"No valid small-cell scenes were found below {root_path}")
    return cases, pd.DataFrame(records)


def build_selected_observations(cases: list[SmallCellScene]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for case in cases:
        for frame, cell_ids in sorted(case.selected_cells.items()):
            for ordinal, cell_id in enumerate(cell_ids):
                records.append({
                    "case_id": case.case_id,
                    "scene_path": str(case.scene_path),
                    "sample_id": case.sample_id,
                    "frame": int(frame),
                    "cell_id": int(cell_id),
                    "selection_ordinal": int(ordinal),
                    "source_tracks_csv": str(case.source_tracks_csv or ""),
                })
    result = pd.DataFrame(records)
    duplicates = result.duplicated(["case_id", "frame", "cell_id"], keep=False)
    if duplicates.any():
        raise SmallCellSceneError("duplicate selected observations were generated within a scene")
    return result.sort_values(["sample_id", "case_id", "frame", "cell_id"]).reset_index(drop=True)
