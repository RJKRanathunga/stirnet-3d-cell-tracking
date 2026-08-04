"""Neutral marker and immutable geometric-completion models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


Position3D = tuple[int, int, int]
Float3 = tuple[float, float, float]
MarkerSource = Literal["effective_edt", "geometric_completion"]


def _readonly_matrix(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"array must have shape {shape} and contain finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class InstanceMarker:
    """One source-neutral marker whose tuple order defines its instance ID."""

    position_zyx: Position3D
    source: MarkerSource
    source_reference_id: int
    confidence: float

    def __post_init__(self) -> None:
        position = tuple(int(value) for value in self.position_zyx)
        if len(position) != 3:
            raise ValueError("position_zyx must contain z, y, and x")
        if self.source not in ("effective_edt", "geometric_completion"):
            raise ValueError("unsupported marker source")
        if int(self.source_reference_id) <= 0:
            raise ValueError("source_reference_id must be positive")
        confidence = float(self.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("marker confidence must be finite and in [0, 1]")
        object.__setattr__(self, "position_zyx", position)
        object.__setattr__(self, "source_reference_id", int(self.source_reference_id))
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class SurfaceSample:
    """One boundary observation at one physical smoothing level."""

    position_zyx: Position3D
    position_um: Float3
    normal_zyx: Float3
    principal_curvatures: tuple[float, float]
    prominence_um: float
    normal_coherence: float
    scale_index: int


@dataclass(frozen=True)
class SurfaceCap:
    """A consolidated, coherent convex surface patch."""

    cap_id: int
    center_zyx: Float3
    center_um: Float3
    mean_normal: Float3
    area_proxy_um2: float
    prominence_um: float
    normal_coherence: float
    curvature_score: float
    scale_support: float
    sample_indices: tuple[int, ...]


@dataclass(frozen=True)
class CapPairEvidence:
    """Cheap, explicit evidence for one possible cap-supported long axis."""

    pair_id: int
    cap_ids: tuple[int, int]
    midpoint_zyx: Float3
    midpoint_um: Float3
    axis_unit_um: Float3
    separation_um: float
    normal_opposition: float
    axis_alignment: float
    scale_support: float
    axis_occupancy: float
    consecutive_axis_occupancy: float
    valid: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CrossSectionEvidence:
    """Ellipse and centerline evidence for one perpendicular slab."""

    t_um: float
    area_um2: float
    centroid_offset_um: float
    radius_major_um: float
    radius_minor_um: float
    ellipse_iou: float
    boundary_error_um: float
    sample_count: int
    valid: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class BodyEvidence:
    """Normalized hard-gate measurements for a cap-pair body candidate."""

    cap_opposition: float
    cap_axis_alignment: float
    cap_scale_support: float
    axis_occupancy: float
    consecutive_axis_occupancy: float
    valid_cross_section_fraction: float
    median_ellipse_iou: float
    area_profile_score: float
    centerline_score: float
    ellipsoid_surface_score: float
    ellipsoid_occupancy: float
    unique_volume_fraction: float
    unique_surface_fraction: float


@dataclass(frozen=True)
class GeometricBody:
    """One fitted complete body, valid or rejected, with explicit evidence."""

    body_id: int
    cap_ids: tuple[int, int]
    center_zyx: Float3
    center_um: Float3
    rotation_matrix: np.ndarray = field(compare=False, repr=False)
    semi_axes_um: Float3
    axis_endpoints_zyx: tuple[Float3, Float3]
    evidence: BodyEvidence
    score: float
    valid: bool
    rejection_reasons: tuple[str, ...]
    represented_by_effective_peak_ids: tuple[int, ...] = ()
    cross_sections: tuple[CrossSectionEvidence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rotation_matrix",
            _readonly_matrix(self.rotation_matrix, (3, 3)),
        )


@dataclass(frozen=True)
class GeometricDebugArtifacts:
    """Large geometry collections retained only on an explicit debug request."""

    boundary_positions_zyx: np.ndarray = field(compare=False, repr=False)
    boundary_normals_zyx: np.ndarray = field(compare=False, repr=False)
    surface_samples: tuple[SurfaceSample, ...]
    cap_pair_evidence: tuple[CapPairEvidence, ...]
    ellipsoid_support_zyx: np.ndarray = field(compare=False, repr=False)
    unique_support_zyx: np.ndarray = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        boundary = np.array(self.boundary_positions_zyx, dtype=float, copy=True)
        normals = np.array(self.boundary_normals_zyx, dtype=float, copy=True)
        ellipsoid = np.array(self.ellipsoid_support_zyx, dtype=float, copy=True)
        unique = np.array(self.unique_support_zyx, dtype=float, copy=True)
        for name, array in (
            ("boundary_positions_zyx", boundary),
            ("boundary_normals_zyx", normals),
            ("ellipsoid_support_zyx", ellipsoid),
            ("unique_support_zyx", unique),
        ):
            if array.ndim != 2 or array.shape[1:] != (3,):
                raise ValueError(f"{name} must have shape (N, 3)")
            array.setflags(write=False)
            object.__setattr__(self, name, array)


@dataclass(frozen=True)
class GeometricCompletionResult:
    """Compact production result plus optional retained diagnostic geometry."""

    surface_caps: tuple[SurfaceCap, ...]
    body_candidates: tuple[GeometricBody, ...]
    selected_bodies: tuple[GeometricBody, ...]
    supplemental_markers: tuple[InstanceMarker, ...]
    processing_status: str
    error: str | None
    debug_artifacts: GeometricDebugArtifacts | None = field(
        default=None, compare=False, repr=False
    )

    @classmethod
    def failed(cls, error: BaseException) -> "GeometricCompletionResult":
        return cls(
            (),
            (),
            (),
            (),
            "failed",
            f"{type(error).__name__}: {error}",
        )


__all__ = [
    "BodyEvidence",
    "CapPairEvidence",
    "CrossSectionEvidence",
    "GeometricBody",
    "GeometricCompletionResult",
    "GeometricDebugArtifacts",
    "InstanceMarker",
    "SurfaceCap",
    "SurfaceSample",
]
