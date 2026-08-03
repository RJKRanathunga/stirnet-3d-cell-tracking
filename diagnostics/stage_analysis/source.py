"""Scene, frame, and complete Stage 2 component loading for Stage 3 analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Collection

import numpy as np
from scipy import ndimage

from diagnostics.pipeline_replay.source import PipelineReplaySource
from src.io import PipelinePaths

from .models import DisplayScope, Stage3FrameSelection, TargetComponentResolution


CONNECTIVITY_3D_6 = ndimage.generate_binary_structure(3, 1)


def discover_categories(root: str | Path) -> list[str]:
    """Return scene-category directories in deterministic order."""

    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def discover_scenes(root: str | Path, category: str) -> list[str]:
    """Return scene directories containing valid scene metadata files."""

    directory = Path(root) / str(category)
    if not directory.is_dir():
        return []
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.is_dir() and (path / "scene.json").is_file()
    )


def _padded_union_crop(
    bboxes: Collection[tuple[slice, slice, slice]],
    shape: tuple[int, int, int],
    padding: int,
) -> tuple[slice, slice, slice] | None:
    if not bboxes:
        return None
    boxes = tuple(bboxes)
    padding = max(int(padding), 0)
    return tuple(
        slice(
            max(min(box[axis].start for box in boxes) - padding, 0),
            min(max(box[axis].stop for box in boxes) + padding, shape[axis]),
        )
        for axis in range(3)
    )  # type: ignore[return-value]


def resolve_target_components(
    production_labels: np.ndarray,
    binary_mask: np.ndarray,
    selected_ids: Collection[int],
    *,
    display_padding: int = 4,
) -> TargetComponentResolution:
    """Resolve production IDs to complete full-frame 6-connected Stage 2 components."""

    production = np.asarray(production_labels)
    binary = np.asarray(binary_mask, dtype=bool)
    if production.ndim != 3 or binary.ndim != 3:
        raise ValueError("production_labels and binary_mask must both be 3-D")
    if production.shape != binary.shape:
        raise ValueError("production_labels and binary_mask must have equal shapes")

    selected = tuple(dict.fromkeys(int(value) for value in selected_ids))
    if any(value <= 0 for value in selected):
        raise ValueError("selected production IDs must be positive")
    present = {int(value) for value in np.unique(production) if int(value) > 0}
    missing = tuple(value for value in selected if value not in present)
    selected_mask = np.isin(production, selected)

    component_labels, component_count = ndimage.label(
        binary, structure=CONNECTIVITY_3D_6
    )
    target_ids = tuple(
        int(value)
        for value in np.unique(component_labels[selected_mask])
        if int(value) > 0
    )
    objects = ndimage.find_objects(component_labels, max_label=int(component_count))
    bboxes = {
        component_id: objects[component_id - 1]
        for component_id in target_ids
        if objects[component_id - 1] is not None
    }
    component_selected_ids = {
        component_id: tuple(
            selected_id
            for selected_id in selected
            if np.any(
                (production == selected_id) & (component_labels == component_id)
            )
        )
        for component_id in target_ids
    }
    no_foreground = tuple(
        selected_id
        for selected_id in selected
        if selected_id not in missing
        and not np.any((production == selected_id) & binary)
    )
    crop = _padded_union_crop(
        bboxes.values(), tuple(int(value) for value in binary.shape), display_padding
    )
    return TargetComponentResolution(
        selected,
        selected_mask,
        component_labels,
        target_ids,
        bboxes,  # type: ignore[arg-type]
        component_selected_ids,
        crop,
        missing,
        no_foreground,
    )


def build_display_scope(
    production_labels: np.ndarray,
    binary_mask: np.ndarray,
    resolution: TargetComponentResolution,
    *,
    selected_only: bool,
    crop: tuple[slice, slice, slice] | None = None,
) -> DisplayScope:
    """Build visualization data without changing or rerunning Stage 3."""

    display_crop = crop or resolution.diagnostic_crop
    if display_crop is None:
        raise ValueError("no diagnostic display crop is available")
    production_crop = np.asarray(production_labels)[display_crop]
    binary_crop = np.asarray(binary_mask, dtype=bool)[display_crop]
    if selected_only:
        production_crop = np.where(
            np.isin(production_crop, resolution.selected_ids), production_crop, 0
        )
        binary_crop = resolution.target_mask()[display_crop]
    return DisplayScope(production_crop, binary_crop)


@dataclass(frozen=True)
class Stage3AnalysisSource:
    """Maps native Napari time indices to original saved scene frames."""

    replay_source: PipelineReplaySource
    display_padding: int = 4

    @classmethod
    def from_scene(
        cls,
        scene_path: str | Path,
        paths: PipelinePaths,
        *,
        display_padding: int = 4,
    ) -> "Stage3AnalysisSource":
        return cls(
            PipelineReplaySource.from_scene(scene_path, paths),
            int(display_padding),
        )

    @property
    def frames(self) -> tuple[int, ...]:
        return self.replay_source.frames

    @property
    def scene_time_to_original_frame(self) -> tuple[int, ...]:
        return self.frames

    @property
    def voxel_size_zyx_um(self) -> tuple[float, float, float]:
        return self.replay_source.voxel_size_zyx_um

    @property
    def time_count(self) -> int:
        return len(self.frames)

    def original_frame_number(self, scene_time_index: int) -> int:
        index = int(scene_time_index)
        if index < 0 or index >= self.time_count:
            raise IndexError(
                f"scene time index {index} is outside 0..{self.time_count - 1}"
            )
        return int(self.frames[index])

    def load_time_index(self, scene_time_index: int) -> Stage3FrameSelection:
        """Load exactly one original frame and resolve its saved selected IDs."""

        index = int(scene_time_index)
        frame_number = self.original_frame_number(index)
        selected_ids = tuple(
            self.replay_source.selected_cells.get(frame_number, ())
        )
        production_frame = self.replay_source.load_frame(frame_number)
        resolution = resolve_target_components(
            production_frame.instance_labels,
            production_frame.binary_mask,
            selected_ids,
            display_padding=self.display_padding,
        )
        display_crop = resolution.diagnostic_crop or self.replay_source.crop_slices
        return Stage3FrameSelection(
            index,
            frame_number,
            selected_ids,
            production_frame,
            resolution,
            display_crop,
        )


__all__ = [
    "Stage3AnalysisSource",
    "build_display_scope",
    "discover_categories",
    "discover_scenes",
    "resolve_target_components",
]
