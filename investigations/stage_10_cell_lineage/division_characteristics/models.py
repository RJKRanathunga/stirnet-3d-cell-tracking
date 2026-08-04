"""Data models used by the division-characteristics investigation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DivisionCase:
    """One validated manually extracted 1-to-2 division scene."""

    case_id: str
    scene_path: Path
    sample_id: str
    frame_numbers: tuple[int, ...]
    selected_cells: dict[int, tuple[int, ...]]
    event_frame: int
    previous_parent_frame: int
    transition_gap_frames: int
    warnings: tuple[str, ...] = ()

    @property
    def parent_frames(self) -> tuple[int, ...]:
        return tuple(frame for frame in self.frame_numbers if frame < self.event_frame)

    @property
    def child_frames(self) -> tuple[int, ...]:
        return tuple(frame for frame in self.frame_numbers if frame >= self.event_frame)
