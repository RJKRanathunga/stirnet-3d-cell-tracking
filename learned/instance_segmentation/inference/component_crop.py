"""Inference-time construction of the same cubic canonical ROI used in training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..datasets.config import DEFAULT_SAMPLE_BUILD_CONFIG, SampleBuildConfig
from ..datasets.core.component_transform import bbox_from_binary_mask, build_canonical_transform
from ..datasets.core.marker_heatmap import (
    MarkerDetector,
    detect_effective_markers_stage3,
    markers_to_heatmap,
    normalized_canonical_edt,
)
from ..datasets.core.models import AnnotatedVolume, CanonicalTransform
from ..datasets.core.normalization import normalize_intensity, robust_intensity_bounds
from ..datasets.core.resampling import resample_with_transform


@dataclass(frozen=True)
class InferenceROI:
    inputs: np.ndarray  # [4,Z,Y,X]
    component_mask: np.ndarray
    valid_mask: np.ndarray
    marker_positions_zyx: tuple[tuple[int, int, int], ...]
    transform: CanonicalTransform

    def canonical_centers_to_native(self, centers_zyx: np.ndarray) -> np.ndarray:
        return self.transform.canonical_to_native(centers_zyx)


def build_inference_roi(
    image: np.ndarray,
    component_mask: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
    *,
    config: SampleBuildConfig = DEFAULT_SAMPLE_BUILD_CONFIG,
    valid_mask: np.ndarray | None = None,
    intensity_bounds: tuple[float, float] | None = None,
    marker_detector: MarkerDetector = detect_effective_markers_stage3,
) -> InferenceROI:
    """Normalize one Stage-2 component into the fixed cubic CNN coordinate system."""
    image = np.asarray(image)
    component = np.asarray(component_mask, dtype=bool)
    if image.shape != component.shape or image.ndim != 3:
        raise ValueError("image and component_mask must be matching 3-D arrays")
    if valid_mask is not None and np.asarray(valid_mask).shape != image.shape:
        raise ValueError("valid_mask must match image")

    bbox = bbox_from_binary_mask(component, spacing_zyx_um)
    extent = np.asarray(bbox.extent_um_zyx, dtype=np.float64)
    aspect = float(extent.max() / max(extent.min(), 1e-9))
    if aspect > config.max_group_aspect_ratio:
        raise ValueError(
            f"component physical bbox aspect ratio {aspect:.2f} exceeds "
            f"{config.max_group_aspect_ratio:.2f}"
        )
    transform = build_canonical_transform(
        bbox,
        native_spacing_zyx_um=spacing_zyx_um,
        canonical_shape_zyx=config.crop_shape_zyx,
        canonical_spacing_zyx=config.canonical_spacing_zyx,
        component_occupancy=config.component_occupancy,
        border_margin_voxels=config.border_margin_voxels,
        min_scale=config.min_normalization_scale,
        max_scale=config.max_normalization_scale,
    )

    # At inference labels carry only the observed Stage-2 component; there is no GT.
    volume = AnnotatedVolume(
        image=image,
        instance_labels=component.astype(np.int32),
        spacing_zyx_um=spacing_zyx_um,
        dataset_name="inference",
        sample_id="component",
        valid_mask=valid_mask,
        intensity_bounds=intensity_bounds,
    )
    crop = resample_with_transform(volume, transform, anti_alias_image=config.anti_alias_image)
    canonical_component = crop.labels > 0
    if not canonical_component.any():
        raise ValueError("component vanished during canonical normalization")

    bounds = intensity_bounds
    if bounds is None:
        bounds = robust_intensity_bounds(
            crop.image,
            valid_mask=crop.valid_mask,
            low_percentile=config.image_percentile_low,
            high_percentile=config.image_percentile_high,
        )
    image_normalized = normalize_intensity(crop.image, bounds)
    edt = normalized_canonical_edt(
        canonical_component,
        config.canonical_spacing_zyx,
        clip_distance_vox=config.edt_clip_vox,
    )
    markers = marker_detector(canonical_component, config.canonical_spacing_zyx)
    marker_heatmap = markers_to_heatmap(
        canonical_component.shape,
        markers,
        config.canonical_spacing_zyx,
        sigma_vox=config.marker_sigma_vox,
    )
    inputs = np.stack(
        [
            image_normalized,
            canonical_component.astype(np.float32),
            edt,
            marker_heatmap,
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    return InferenceROI(
        inputs=inputs,
        component_mask=canonical_component,
        valid_mask=crop.valid_mask,
        marker_positions_zyx=markers,
        transform=transform,
    )
