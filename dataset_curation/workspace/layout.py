
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class CurationLayout:
    root: Path
    dataset: str
    sample_id: str

    @property
    def sample_root(self) -> Path:
        return self.root / self.dataset / self.sample_id

    @property
    def sample_manifest(self) -> Path:
        return self.sample_root / "sample.json"

    @property
    def prepared(self) -> Path:
        return self.sample_root / "prepared"

    @property
    def inference(self) -> Path:
        return self.sample_root / "inference"

    def inference_run(self, run_id: str) -> Path:
        return self.inference / run_id

    def inference_manifest(self, run_id: str) -> Path:
        return self.inference_run(run_id) / "manifest.json"

    @property
    def annotations(self) -> Path:
        return self.sample_root / "annotations"

    def annotation_set(self, name: str) -> Path:
        return self.annotations / name

    def annotation_manifest(self, name: str) -> Path:
        return self.annotation_set(name) / "manifest.json"

    def instance_annotations(self, name: str) -> Path:
        return self.annotation_set(name) / "instances"

    def track_annotations(self, name: str) -> Path:
        return self.annotation_set(name) / "tracks"

    def point_annotations(self, name: str) -> Path:
        return self.annotation_set(name) / "points"
