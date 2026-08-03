"""Typed state shared by the pipeline replay workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from src.diagnostics import StageTrace


class ReplayMode(str, Enum):
    EXACT = "Exact full-frame"
    LOCAL = "Local approximation"


@dataclass(frozen=True)
class ProductionFrame:
    frame: int
    raw: np.ndarray
    preprocessed: np.ndarray
    binary_mask: np.ndarray
    instance_labels: np.ndarray
    cells: pd.DataFrame


@dataclass
class PipelineReplayState:
    preprocessing_config: object
    masking_config: object
    segmentation_config: object
    preprocessing_trace: StageTrace | None = None
    masking_trace: StageTrace | None = None
    component_results: tuple[object, ...] = ()
    trial_labels: np.ndarray | None = None
    trial_markers: np.ndarray | None = None
    trial_cells: pd.DataFrame | None = None
    trial_features: pd.DataFrame | None = None
    connected_components: np.ndarray | None = None
    relevant_component_ids: tuple[int, ...] = ()
    dirty_from_stage: int = 1
    mode: ReplayMode = ReplayMode.EXACT
    frame: int | None = None
    work_origin_zyx: tuple[int, int, int] = (0, 0, 0)
    work_crop_global: tuple[slice, slice, slice] | None = None
    warnings: list[str] = field(default_factory=list)

    def invalidate_from(self, stage: int) -> None:
        """Discard stage ``stage`` and all dependent cached outputs."""

        if not 1 <= stage <= 5:
            raise ValueError("stage must be between 1 and 5")
        self.dirty_from_stage = min(self.dirty_from_stage, stage)
        if stage <= 1:
            self.preprocessing_trace = None
        if stage <= 2:
            self.masking_trace = None
        if stage <= 3:
            self.component_results = ()
            self.trial_labels = None
            self.trial_markers = None
            self.connected_components = None
            self.relevant_component_ids = ()
        if stage <= 4:
            self.trial_cells = None
        if stage <= 5:
            self.trial_features = None


@dataclass(frozen=True)
class ReplayResult:
    frame: int
    mode: ReplayMode
    display_raw: np.ndarray
    display_preprocessed: np.ndarray | None
    display_binary_mask: np.ndarray | None
    display_connected_components: np.ndarray | None
    display_instance_labels: np.ndarray | None
    display_markers: np.ndarray | None
    cells: pd.DataFrame | None
    features: pd.DataFrame | None
    component_results: tuple[object, ...]
    work_origin_zyx: tuple[int, int, int]
    display_origin_zyx: tuple[int, int, int]
    warning: str | None = None


@dataclass(frozen=True)
class TrackingContext:
    stage7: dict[str, pd.DataFrame]
    stage8: dict[str, pd.DataFrame]
    metadata: dict[str, object]


def frame_artifact_path(root: Path, frame: int, suffix: str) -> Path:
    return root / f"t{int(frame):03d}.{suffix}"
