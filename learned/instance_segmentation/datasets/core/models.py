"""Canonical data models shared by annotated 3-D datasets and CNN samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

Spacing3D = tuple[float, float, float]
Shape3D = tuple[int, int, int]
Index3D = tuple[int, int, int]
Float3D = tuple[float, float, float]


@dataclass(frozen=True)
class VolumeRecord:
    sample_id: str
    split: str
    image_path: Path
    labels_path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnnotatedVolume:
    image: np.ndarray
    instance_labels: np.ndarray
    spacing_zyx_um: Spacing3D
    dataset_name: str
    sample_id: str
    split: str = "unspecified"
    valid_mask: np.ndarray | None = None
    intensity_bounds: tuple[float, float] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        image = np.asarray(self.image)
        labels = np.asarray(self.instance_labels)
        if image.ndim != 3 or labels.ndim != 3:
            raise ValueError("image and instance_labels must both be 3-D [Z,Y,X]")
        if image.shape != labels.shape:
            raise ValueError(f"image/label shape mismatch: {image.shape} versus {labels.shape}")
        if len(self.spacing_zyx_um) != 3 or any(float(v) <= 0 for v in self.spacing_zyx_um):
            raise ValueError("spacing_zyx_um must contain three positive values")
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError("instance_labels must use an integer dtype")
        if np.any(labels < 0):
            raise ValueError("instance_labels cannot contain negative IDs")
        if self.valid_mask is not None and np.asarray(self.valid_mask).shape != image.shape:
            raise ValueError("valid_mask must match image shape")
        if self.intensity_bounds is not None:
            low, high = map(float, self.intensity_bounds)
            if not low < high:
                raise ValueError("intensity_bounds must satisfy low < high")

    @property
    def shape_zyx(self) -> Shape3D:
        return tuple(int(v) for v in self.image.shape)  # type: ignore[return-value]

    @property
    def instance_ids(self) -> np.ndarray:
        ids = np.unique(self.instance_labels)
        return ids[ids > 0]

    @property
    def effective_valid_mask(self) -> np.ndarray:
        if self.valid_mask is None:
            return np.ones(self.image.shape, dtype=bool)
        return np.asarray(self.valid_mask, dtype=bool)


@dataclass(frozen=True)
class AdjacencyEdge:
    instance_a: int
    instance_b: int
    separation_um: float
    centroid_distance_um: float

    def __post_init__(self) -> None:
        if self.instance_a <= 0 or self.instance_b <= 0:
            raise ValueError("instance IDs must be positive")
        if self.instance_a >= self.instance_b:
            raise ValueError("AdjacencyEdge requires instance_a < instance_b")
        if self.separation_um < 0 or self.centroid_distance_um < 0:
            raise ValueError("distances cannot be negative")


@dataclass(frozen=True)
class InstanceGroup:
    instance_ids: tuple[int, ...]
    kind: str
    score: float = 0.0

    def __post_init__(self) -> None:
        if not self.instance_ids or any(int(v) <= 0 for v in self.instance_ids):
            raise ValueError("instance_ids must contain positive IDs")
        if len(set(self.instance_ids)) != len(self.instance_ids):
            raise ValueError("instance_ids must be unique")


@dataclass(frozen=True)
class ComponentBBox:
    """Tight selected-group bounding box in native index and physical coordinates."""

    start_zyx: Index3D
    stop_zyx: Index3D  # exclusive
    extent_vox_zyx: Shape3D
    extent_um_zyx: Float3D
    center_native_zyx: Float3D


@dataclass(frozen=True)
class CanonicalTransform:
    """Invertible object-centric mapping between native and canonical indices.

    ``normalization_scale`` is isotropic in physical source coordinates:
    canonical_physical_offset = source_physical_offset * normalization_scale.
    """

    native_center_zyx: Float3D
    normalization_scale: float
    native_spacing_zyx_um: Spacing3D
    canonical_spacing_zyx: Spacing3D
    canonical_shape_zyx: Shape3D
    component_bbox: ComponentBBox

    def __post_init__(self) -> None:
        if self.normalization_scale <= 0:
            raise ValueError("normalization_scale must be positive")

    @property
    def canonical_center_zyx(self) -> Float3D:
        return tuple((int(v) - 1) / 2.0 for v in self.canonical_shape_zyx)  # type: ignore[return-value]

    @property
    def effective_native_spacing_canonical(self) -> Spacing3D:
        return tuple(
            float(v) * float(self.normalization_scale)
            for v in self.native_spacing_zyx_um
        )  # type: ignore[return-value]

    def native_to_canonical(self, coordinates_zyx: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        coords = np.asarray(coordinates_zyx, dtype=np.float64)
        native_center = np.asarray(self.native_center_zyx, dtype=np.float64)
        source_spacing = np.asarray(self.native_spacing_zyx_um, dtype=np.float64)
        canonical_spacing = np.asarray(self.canonical_spacing_zyx, dtype=np.float64)
        canonical_center = np.asarray(self.canonical_center_zyx, dtype=np.float64)
        return canonical_center + (
            (coords - native_center) * source_spacing * self.normalization_scale
        ) / canonical_spacing

    def canonical_to_native(self, coordinates_zyx: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        coords = np.asarray(coordinates_zyx, dtype=np.float64)
        native_center = np.asarray(self.native_center_zyx, dtype=np.float64)
        source_spacing = np.asarray(self.native_spacing_zyx_um, dtype=np.float64)
        canonical_spacing = np.asarray(self.canonical_spacing_zyx, dtype=np.float64)
        canonical_center = np.asarray(self.canonical_center_zyx, dtype=np.float64)
        return native_center + (
            (coords - canonical_center) * canonical_spacing
        ) / (source_spacing * self.normalization_scale)


@dataclass(frozen=True)
class TargetBundle:
    foreground: np.ndarray
    vectors_normalized: np.ndarray
    boundary: np.ndarray
    center: np.ndarray
    instance_labels: np.ndarray
    centers_zyx: tuple[Index3D, ...]

    def __post_init__(self) -> None:
        spatial = self.instance_labels.shape
        if self.instance_labels.ndim != 3:
            raise ValueError("instance_labels must be 3-D")
        expected_scalar = (1, *spatial)
        if self.foreground.shape != expected_scalar:
            raise ValueError("foreground shape does not match instance labels")
        if self.boundary.shape != expected_scalar:
            raise ValueError("boundary shape does not match instance labels")
        if self.center.shape != expected_scalar:
            raise ValueError("center shape does not match instance labels")
        if self.vectors_normalized.shape != (3, *spatial):
            raise ValueError("vectors_normalized must have shape [3,Z,Y,X]")


@dataclass(frozen=True)
class TrainingSample:
    inputs: np.ndarray
    targets: TargetBundle
    valid_mask: np.ndarray
    input_component_mask: np.ndarray
    edt_normalized: np.ndarray
    marker_heatmap: np.ndarray
    marker_positions_zyx: tuple[Index3D, ...]
    transform: CanonicalTransform
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        spatial = self.targets.instance_labels.shape
        if self.inputs.shape != (4, *spatial):
            raise ValueError(f"inputs must have shape [4,Z,Y,X], got {self.inputs.shape}")
        if self.valid_mask.shape != (1, *spatial):
            raise ValueError("valid_mask must have shape [1,Z,Y,X]")
        if self.input_component_mask.shape != spatial:
            raise ValueError("input_component_mask shape mismatch")
        if self.edt_normalized.shape != spatial or self.marker_heatmap.shape != spatial:
            raise ValueError("EDT/marker heatmap shape mismatch")
        arrays = (
            self.inputs,
            self.targets.foreground,
            self.targets.vectors_normalized,
            self.targets.boundary,
            self.targets.center,
            self.valid_mask,
            self.edt_normalized,
            self.marker_heatmap,
        )
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("training sample contains non-finite values")
