"""Data and result models for the Stage 3 diagnostic package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diagnostics.pipeline_replay.models import ProductionFrame


SpatialSlices = tuple[slice, slice, slice]


class CanonicalMismatchError(RuntimeError):
    """Raised when explicit Stage 3 orchestration differs from production."""


@dataclass(frozen=True)
class TargetComponentResolution:
    """Full-frame Stage 2 components implicated by saved production IDs."""

    selected_ids: tuple[int, ...]
    selected_production_mask: np.ndarray
    component_labels: np.ndarray
    target_component_ids: tuple[int, ...]
    component_bboxes: dict[int, SpatialSlices]
    component_selected_ids: dict[int, tuple[int, ...]]
    diagnostic_crop: SpatialSlices | None
    missing_production_ids: tuple[int, ...]
    selected_ids_without_stage2_foreground: tuple[int, ...]

    def complete_component_mask(self, component_id: int) -> np.ndarray:
        """Return one complete component in its tight full-frame bounding box."""

        component_id = int(component_id)
        try:
            bbox = self.component_bboxes[component_id]
        except KeyError as error:
            raise KeyError(f"unknown target component {component_id}") from error
        return self.component_labels[bbox] == component_id

    def target_mask(self) -> np.ndarray:
        """Return the union of all implicated complete Stage 2 components."""

        return np.isin(self.component_labels, self.target_component_ids)


@dataclass(frozen=True)
class DisplayScope:
    """Crop-local production labels and binary mask for one display scope."""

    production_labels: np.ndarray
    binary_mask: np.ndarray


@dataclass(frozen=True)
class Stage3FrameSelection:
    """The one saved Stage 6 frame currently selected by Napari time index."""

    scene_time_index: int
    original_frame_number: int
    selected_ids: tuple[int, ...]
    production_frame: ProductionFrame
    resolution: TargetComponentResolution
    display_crop: SpatialSlices


@dataclass(frozen=True)
class Stage3ComponentRun:
    """Retained canonical intermediates for one target Stage 2 component."""

    component_id: int
    component_bbox: SpatialSlices
    component_mask: np.ndarray
    padded_mask: np.ndarray
    peak_detail: object
    pair_evidence: tuple
    collapse_result: object
    merged_description: object
    evaluation: object
    decision: object
    canonical: object
    config: object

    @property
    def peak_analysis(self):
        return self.peak_detail.analysis


__all__ = [
    "CanonicalMismatchError",
    "DisplayScope",
    "SpatialSlices",
    "Stage3ComponentRun",
    "Stage3FrameSelection",
    "TargetComponentResolution",
]
