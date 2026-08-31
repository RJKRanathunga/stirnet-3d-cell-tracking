
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from dataset_curation._repo import repo_root

@dataclass(frozen=True)
class LegacyEvaluationPaths:
    sample_id: str

    @property
    def instance_annotations(self) -> Path:
        return repo_root() / "evaluation" / "segmentation" / "annotations" / self.sample_id

    @property
    def suspect_scores(self) -> Path:
        return repo_root() / "evaluation" / "segmentation" / "suspects" / self.sample_id

    @property
    def track_annotations(self) -> Path:
        return repo_root() / "evaluation" / "segmentation" / "track_annotations" / self.sample_id
