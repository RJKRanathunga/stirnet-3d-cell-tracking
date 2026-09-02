from __future__ import annotations

# DATASET_CURATION_CANONICAL_SKIP_V1

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from dataset_curation.errors import CurationError


PATHOLOGICAL_CONNECTED_FOREGROUND = "pathological_connected_foreground"


@dataclass(frozen=True)
class SourceQualityPolicy:
    """Conservative guardrail for source masks outside the production regime."""

    min_pathological_component_voxels: int = 100_000
    min_pathological_foreground_fraction: float = 0.50
    min_pathological_volume_fraction: float = 0.05

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class SourceMaskMetrics:
    frame_voxels: int
    foreground_voxels: int
    connected_components: int
    largest_component_id: int
    largest_component_voxels: int
    largest_component_fraction_of_foreground: float
    largest_component_fraction_of_volume: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


DEFAULT_SOURCE_QUALITY_POLICY = SourceQualityPolicy()


class SourceQualityRejected(CurationError):
    """Expected terminal skip: this source mask is outside the safe regime."""

    def __init__(
        self,
        *,
        frame: int,
        metrics: SourceMaskMetrics,
        policy: SourceQualityPolicy,
        reason_code: str = PATHOLOGICAL_CONNECTED_FOREGROUND,
    ) -> None:
        self.frame = int(frame)
        self.metrics = metrics
        self.policy = policy
        self.reason_code = str(reason_code)
        super().__init__(
            f"t={self.frame:03d} {self.reason_code}: "
            f"largest_component={metrics.largest_component_voxels}, "
            f"foreground_fraction="
            f"{metrics.largest_component_fraction_of_foreground:.3f}, "
            f"volume_fraction="
            f"{metrics.largest_component_fraction_of_volume:.3f}"
        )


def evaluate_source_mask(source_mask: np.ndarray) -> SourceMaskMetrics:
    """Measure 6-connected foreground using a single cheap CPU labeling pass."""
    mask = np.asarray(source_mask) > 0
    if mask.ndim != 3:
        raise ValueError(
            f"Source quality mask must be 3-D, got {mask.shape}"
        )

    frame_voxels = int(mask.size)
    foreground_voxels = int(np.count_nonzero(mask))

    if foreground_voxels == 0:
        return SourceMaskMetrics(
            frame_voxels=frame_voxels,
            foreground_voxels=0,
            connected_components=0,
            largest_component_id=0,
            largest_component_voxels=0,
            largest_component_fraction_of_foreground=0.0,
            largest_component_fraction_of_volume=0.0,
        )

    structure = ndimage.generate_binary_structure(3, 1)
    labels, component_count = ndimage.label(mask, structure=structure)
    counts = np.bincount(labels.reshape(-1))
    if counts.size <= 1:
        largest_id = 0
        largest_voxels = 0
    else:
        largest_id = int(np.argmax(counts[1:]) + 1)
        largest_voxels = int(counts[largest_id])

    return SourceMaskMetrics(
        frame_voxels=frame_voxels,
        foreground_voxels=foreground_voxels,
        connected_components=int(component_count),
        largest_component_id=largest_id,
        largest_component_voxels=largest_voxels,
        largest_component_fraction_of_foreground=(
            float(largest_voxels) / float(foreground_voxels)
        ),
        largest_component_fraction_of_volume=(
            float(largest_voxels) / float(frame_voxels)
            if frame_voxels > 0
            else 0.0
        ),
    )


def validate_source_mask(
    frame: int,
    source_mask: np.ndarray,
    *,
    policy: SourceQualityPolicy = DEFAULT_SOURCE_QUALITY_POLICY,
) -> SourceMaskMetrics:
    """Raise SourceQualityRejected before expensive source segmentation."""
    metrics = evaluate_source_mask(source_mask)
    pathological = (
        metrics.largest_component_voxels
        >= int(policy.min_pathological_component_voxels)
        and metrics.largest_component_fraction_of_foreground
        >= float(policy.min_pathological_foreground_fraction)
        and metrics.largest_component_fraction_of_volume
        >= float(policy.min_pathological_volume_fraction)
    )
    if pathological:
        raise SourceQualityRejected(
            frame=int(frame),
            metrics=metrics,
            policy=policy,
        )
    return metrics
