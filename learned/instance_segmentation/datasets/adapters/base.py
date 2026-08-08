"""Minimal adapter protocol for external annotated 3-D datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..core.models import AnnotatedVolume, VolumeRecord


class AnnotatedDatasetAdapter(Protocol):
    """Contract implemented by each external dataset loader."""

    root: Path

    def discover_records(self) -> tuple[VolumeRecord, ...]: ...

    def load(self, record: VolumeRecord) -> AnnotatedVolume: ...
