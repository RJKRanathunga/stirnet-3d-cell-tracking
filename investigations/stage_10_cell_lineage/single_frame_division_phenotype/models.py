"""Data models for single-frame parent/daughter phenotype analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhenotypeCase:
    case_id: str
    scene_path: Path
    sample_id: str
    selected_cells: dict[int, tuple[int, ...]]
    frame_numbers: tuple[int, ...]
    event_frame: int
    previous_parent_frame: int
    warnings: tuple[str, ...] = ()

    @property
    def parent_frames(self) -> tuple[int, ...]:
        return tuple(frame for frame in self.frame_numbers if frame < self.event_frame)

    @property
    def daughter_frames(self) -> tuple[int, ...]:
        return tuple(frame for frame in self.frame_numbers if frame >= self.event_frame)
