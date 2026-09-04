"""Per-sample output paths for the full-dataset batch runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .stage_registry import get_stage_spec


@dataclass(frozen=True)
class SampleOutputPaths:
    output_root: Path
    sample_id: str

    @property
    def train_root(self) -> Path:
        return self.output_root / "train"

    @property
    def sample_root(self) -> Path:
        return self.train_root / self.sample_id

    def stage_directory(self, stage: int) -> Path:
        return self.sample_root / get_stage_spec(stage).directory_name

    def success_marker(self, stage: int) -> Path:
        return self.stage_directory(stage) / "_SUCCESS.json"

    def failed_marker(self, stage: int) -> Path:
        return self.stage_directory(stage) / "_FAILED.json"

    @property
    def source_metadata(self) -> Path:
        return self.sample_root / "source.json"

    @property
    def status_path(self) -> Path:
        return self.sample_root / "pipeline_status.json"


__all__ = ["SampleOutputPaths"]
