"""End-to-end object-centric conversion from annotated group to CNN tensors."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from scipy import ndimage

from ..config import DEFAULT_SAMPLE_BUILD_CONFIG, SampleBuildConfig
from .component_transform import bbox_from_instance_slices, build_canonical_transform
from .marker_heatmap import (
    MarkerDetector,
    detect_effective_markers_stage3,
    markers_to_heatmap,
    normalized_canonical_edt,
)
from .mask_corruption import build_stage2_like_component
from .models import AnnotatedVolume, InstanceGroup, TrainingSample
from .normalization import normalize_intensity, robust_intensity_bounds
from .resampling import resample_with_transform
from .targets import build_targets, relabel_selected_instances


class SampleBuildError(RuntimeError):
    """A selected instance group cannot safely produce a canonical sample."""


class SampleBuilder:
    """Build scale-normalized component-centric training examples.

    The selected union bbox determines one isotropic physical scale.  All raw
    fluorescence, labels, and validity information in the local scene use the
    same transform.  GT labels are used only for selection and supervision, not
    for EDT/marker input evidence.
    """

    def __init__(
        self,
        config: SampleBuildConfig = DEFAULT_SAMPLE_BUILD_CONFIG,
        *,
        marker_detector: MarkerDetector = detect_effective_markers_stage3,
    ) -> None:
        self.config = config
        self.marker_detector = marker_detector
        self._slice_cache_key: tuple[int, tuple[int, ...], str] | None = None
        self._slice_cache: tuple[tuple[slice, slice, slice] | None, ...] | None = None

    def _object_slices(self, labels: np.ndarray):
        key = (id(labels), tuple(labels.shape), labels.dtype.str)
        if self._slice_cache_key != key or self._slice_cache is None:
            self._slice_cache = tuple(ndimage.find_objects(labels))
            self._slice_cache_key = key
        return self._slice_cache

    def _transform_for_group(self, volume: AnnotatedVolume, group: InstanceGroup):
        try:
            bbox = bbox_from_instance_slices(
                group.instance_ids,
                self._object_slices(volume.instance_labels),
                volume.spacing_zyx_um,
            )
        except ValueError as error:
            raise SampleBuildError(str(error)) from error

        extent = np.asarray(bbox.extent_um_zyx, dtype=np.float64)
        aspect = float(extent.max() / max(extent.min(), 1e-9))
        if aspect > self.config.max_group_aspect_ratio:
            raise SampleBuildError(
                f"group physical bbox aspect ratio {aspect:.2f} exceeds "
                f"{self.config.max_group_aspect_ratio:.2f}"
            )

        return build_canonical_transform(
            bbox,
            native_spacing_zyx_um=volume.spacing_zyx_um,
            canonical_shape_zyx=self.config.crop_shape_zyx,
            canonical_spacing_zyx=self.config.canonical_spacing_zyx,
            component_occupancy=self.config.component_occupancy,
            border_margin_voxels=self.config.border_margin_voxels,
            min_scale=self.config.min_normalization_scale,
            max_scale=self.config.max_normalization_scale,
        )

    def _validate_selected_labels(
        self,
        local_labels: np.ndarray,
        group: InstanceGroup,
    ) -> None:
        margin = self.config.border_margin_voxels
        minimum_bbox = np.asarray(self.config.min_instance_bbox_zyx_vox, dtype=int)
        for source_id in group.instance_ids:
            mask = local_labels == int(source_id)
            voxel_count = int(np.count_nonzero(mask))
            if voxel_count < self.config.min_instance_voxels_after_resampling:
                raise SampleBuildError(
                    f"instance {source_id} has only {voxel_count} voxels after normalization"
                )
            coords = np.argwhere(mask)
            if coords.size == 0:
                raise SampleBuildError(f"instance {source_id} vanished during normalization")
            bbox_extent = coords.max(axis=0) - coords.min(axis=0) + 1
            if np.any(bbox_extent < minimum_bbox):
                raise SampleBuildError(
                    f"instance {source_id} canonical bbox {tuple(int(v) for v in bbox_extent)} "
                    f"is below minimum {tuple(int(v) for v in minimum_bbox)}"
                )
            if margin > 0:
                shape = np.asarray(mask.shape)
                if np.any(coords < margin) or np.any(coords >= (shape - margin)):
                    raise SampleBuildError(
                        f"instance {source_id} touches the canonical crop border"
                    )

    def _native_group_diagnostics(self, volume: AnnotatedVolume, group: InstanceGroup) -> str:
        spacing = np.asarray(volume.spacing_zyx_um, dtype=np.float64)
        voxel_volume = float(np.prod(spacing))
        slices = self._object_slices(volume.instance_labels)
        parts: list[str] = []
        for source_id in group.instance_ids:
            index = int(source_id) - 1
            box = slices[index] if 0 <= index < len(slices) else None
            if box is None:
                parts.append(f"id={source_id}: missing")
                continue
            shape_vox = np.asarray([sl.stop - sl.start for sl in box], dtype=int)
            bbox_um = shape_vox * spacing
            local = volume.instance_labels[box] == int(source_id)
            count = int(np.count_nonzero(local))
            parts.append(
                "id={}: voxels={}, bbox_vox={}, bbox_um=({:.3f},{:.3f},{:.3f}), volume_um3={:.3f}".format(
                    source_id,
                    count,
                    tuple(int(v) for v in shape_vox),
                    *[float(v) for v in bbox_um],
                    count * voxel_volume,
                )
            )
        return "; ".join(parts)

    def build(self, volume: AnnotatedVolume, group: InstanceGroup) -> TrainingSample:
        transform = self._transform_for_group(volume, group)
        crop = resample_with_transform(
            volume,
            transform,
            anti_alias_image=self.config.anti_alias_image,
        )
        try:
            self._validate_selected_labels(crop.labels, group)
        except SampleBuildError as error:
            native = self._native_group_diagnostics(volume, group)
            raise SampleBuildError(
                f"{error}; scale={transform.normalization_scale:.5f}; native: {native}"
            ) from error

        local_gt = relabel_selected_instances(crop.labels, group.instance_ids)
        if int(local_gt.max(initial=0)) != len(group.instance_ids):
            raise SampleBuildError("one or more selected instances vanished during normalization")

        canonical_spacing = self.config.canonical_spacing_zyx
        input_mask = build_stage2_like_component(
            local_gt,
            canonical_spacing,
            bridge_radius_um=self.config.bridge_radius_canonical,
            closing_radius_um=self.config.closing_radius_canonical,
        )
        if self.config.require_single_input_component:
            _, count = ndimage.label(
                input_mask, structure=ndimage.generate_binary_structure(3, 1)
            )
            if int(count) != 1:
                raise SampleBuildError(f"expected one Stage-2-like component, found {count}")

        intensity_bounds = volume.intensity_bounds
        if intensity_bounds is None:
            intensity_bounds = robust_intensity_bounds(
                crop.image,
                valid_mask=crop.valid_mask,
                low_percentile=self.config.image_percentile_low,
                high_percentile=self.config.image_percentile_high,
            )
        image_normalized = normalize_intensity(crop.image, intensity_bounds)

        edt = normalized_canonical_edt(
            input_mask,
            canonical_spacing,
            clip_distance=self.config.edt_clip_canonical,
        )
        try:
            markers = self.marker_detector(input_mask, canonical_spacing)
        except Exception as error:  # noqa: BLE001
            raise SampleBuildError(
                f"effective-marker generation failed for {volume.sample_id} "
                f"{group.instance_ids}: {error}"
            ) from error
        if not markers:
            raise SampleBuildError("effective-marker detector returned no markers")
        marker_heatmap = markers_to_heatmap(
            input_mask.shape,
            markers,
            canonical_spacing,
            sigma_um=self.config.marker_sigma_canonical,
        )

        targets = build_targets(
            local_gt,
            input_mask,
            canonical_spacing,
            center_sigma_um=self.config.center_sigma_canonical,
            center_interior_fraction=self.config.center_interior_fraction,
            boundary_radius_um=self.config.boundary_radius_canonical,
        )

        inputs = np.stack(
            [
                image_normalized,
                input_mask.astype(np.float32),
                edt,
                marker_heatmap,
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        valid = crop.valid_mask.astype(np.float32, copy=False)[None, ...]

        metadata: dict[str, Any] = {
            "dataset_name": volume.dataset_name,
            "sample_id": volume.sample_id,
            "split": volume.split,
            "group_kind": group.kind,
            "source_instance_ids": tuple(int(v) for v in group.instance_ids),
            "target_instance_count": len(group.instance_ids),
            "source_spacing_zyx_um": tuple(float(v) for v in volume.spacing_zyx_um),
            "canonical_spacing_zyx": tuple(float(v) for v in canonical_spacing),
            "canonical_shape_zyx": tuple(int(v) for v in self.config.crop_shape_zyx),
            "normalization_scale": float(transform.normalization_scale),
            "native_center_zyx": tuple(float(v) for v in transform.native_center_zyx),
            "source_group_bbox_extent_um_zyx": tuple(
                float(v) for v in transform.component_bbox.extent_um_zyx
            ),
            "canonical_group_bbox_extent_zyx": tuple(
                float(v) * float(transform.normalization_scale)
                for v in transform.component_bbox.extent_um_zyx
            ),
            "component_occupancy": float(self.config.component_occupancy),
            "marker_count": len(markers),
            "vector_coordinate_system": "canonical_axis_fraction",
            "config": asdict(self.config),
        }

        return TrainingSample(
            inputs=inputs,
            targets=targets,
            valid_mask=valid,
            input_component_mask=input_mask,
            edt_normalized=edt,
            marker_heatmap=marker_heatmap,
            marker_positions_zyx=markers,
            transform=transform,
            metadata=metadata,
        )
