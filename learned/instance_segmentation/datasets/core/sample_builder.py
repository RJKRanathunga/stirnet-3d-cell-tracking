"""End-to-end conversion from annotated volume + instance group to CNN tensors."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from scipy import ndimage

from ..config import DEFAULT_SAMPLE_BUILD_CONFIG, SampleBuildConfig
from .marker_heatmap import (
    MarkerDetector,
    detect_effective_markers_stage3,
    markers_to_heatmap,
    normalized_physical_edt,
)
from .mask_corruption import build_stage2_like_component
from .models import AnnotatedVolume, InstanceGroup, TrainingSample
from .normalization import normalize_intensity, robust_intensity_bounds
from .resampling import resample_centered_crop, selected_group_center_native_zyx
from .targets import build_targets, relabel_selected_instances


class SampleBuildError(RuntimeError):
    """A selected instance group cannot safely produce a canonical sample."""


class SampleBuilder:
    """Build component-centric training examples on the Biohub target grid."""

    def __init__(
        self,
        config: SampleBuildConfig = DEFAULT_SAMPLE_BUILD_CONFIG,
        *,
        marker_detector: MarkerDetector = detect_effective_markers_stage3,
    ) -> None:
        self.config = config
        self.marker_detector = marker_detector

    def _validate_selected_labels(
        self,
        local_labels: np.ndarray,
        group: InstanceGroup,
    ) -> None:
        margin = self.config.border_margin_voxels
        for source_id in group.instance_ids:
            mask = local_labels == int(source_id)
            voxel_count = int(np.count_nonzero(mask))
            if voxel_count < self.config.min_instance_voxels_after_resampling:
                raise SampleBuildError(
                    f"instance {source_id} has only {voxel_count} voxels after resampling"
                )
            if margin > 0:
                coords = np.argwhere(mask)
                shape = np.asarray(mask.shape)
                if np.any(coords < margin) or np.any(coords >= (shape - margin)):
                    raise SampleBuildError(
                        f"instance {source_id} touches the canonical crop border"
                    )


    @staticmethod
    def _native_group_diagnostics(
        volume: AnnotatedVolume,
        group: InstanceGroup,
    ) -> str:
        spacing = np.asarray(volume.spacing_zyx_um, dtype=np.float64)
        voxel_volume_um3 = float(np.prod(spacing))
        parts: list[str] = []
        for source_id in group.instance_ids:
            coords = np.argwhere(volume.instance_labels == int(source_id))
            if coords.size == 0:
                parts.append(f"id={source_id}: native_voxels=0")
                continue
            minimum = coords.min(axis=0)
            maximum = coords.max(axis=0)
            bbox_vox = maximum - minimum + 1
            bbox_um = bbox_vox.astype(np.float64) * spacing
            parts.append(
                "id={}: native_voxels={}, bbox_vox_zyx={}, bbox_um_zyx=({:.3f},{:.3f},{:.3f}), volume_um3={:.3f}".format(
                    int(source_id),
                    int(len(coords)),
                    tuple(int(v) for v in bbox_vox),
                    float(bbox_um[0]),
                    float(bbox_um[1]),
                    float(bbox_um[2]),
                    float(len(coords) * voxel_volume_um3),
                )
            )
        return "; ".join(parts)

    def build(self, volume: AnnotatedVolume, group: InstanceGroup) -> TrainingSample:
        center_native = selected_group_center_native_zyx(
            volume.instance_labels,
            group.instance_ids,
        )
        crop = resample_centered_crop(
            volume,
            center_native,
            output_shape_zyx=self.config.crop_shape_zyx,
            target_spacing_zyx_um=self.config.target_spacing_zyx_um,
            anti_alias_image=self.config.anti_alias_image,
        )
        try:
            self._validate_selected_labels(crop.labels, group)
        except SampleBuildError as error:
            native = self._native_group_diagnostics(volume, group)
            raise SampleBuildError(f"{error}; native: {native}") from error

        local_gt = relabel_selected_instances(crop.labels, group.instance_ids)
        if int(local_gt.max(initial=0)) != len(group.instance_ids):
            raise SampleBuildError("one or more selected instances vanished during resampling")

        input_mask = build_stage2_like_component(
            local_gt,
            self.config.target_spacing_zyx_um,
            bridge_radius_um=self.config.bridge_radius_um,
            closing_radius_um=self.config.closing_radius_um,
        )
        if self.config.require_single_input_component:
            _, component_count = ndimage.label(
                input_mask,
                structure=ndimage.generate_binary_structure(3, 1),
            )
            if int(component_count) != 1:
                raise SampleBuildError(
                    f"expected one Stage-2-like component, found {component_count}"
                )

        intensity_bounds = volume.intensity_bounds
        if intensity_bounds is None:
            intensity_bounds = robust_intensity_bounds(
                crop.image,
                valid_mask=crop.valid_mask,
                low_percentile=self.config.image_percentile_low,
                high_percentile=self.config.image_percentile_high,
            )
        image_normalized = normalize_intensity(crop.image, intensity_bounds)

        edt = normalized_physical_edt(
            input_mask,
            self.config.target_spacing_zyx_um,
            clip_um=self.config.edt_clip_um,
        )
        try:
            markers = self.marker_detector(
                input_mask,
                self.config.target_spacing_zyx_um,
            )
        except Exception as error:  # noqa: BLE001 - preserve source failure context.
            raise SampleBuildError(
                f"effective-marker generation failed for {volume.sample_id} {group.instance_ids}: {error}"
            ) from error
        if not markers:
            raise SampleBuildError("effective-marker detector returned no markers")
        marker_heatmap = markers_to_heatmap(
            input_mask.shape,
            markers,
            self.config.target_spacing_zyx_um,
            sigma_um=self.config.marker_sigma_um,
        )

        targets = build_targets(
            local_gt,
            input_mask,
            self.config.target_spacing_zyx_um,
            vector_max_distance_um=self.config.vector_max_distance_um,
            center_sigma_um=self.config.center_sigma_um,
            center_interior_fraction=self.config.center_interior_fraction,
            boundary_radius_um=self.config.boundary_radius_um,
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
            "target_spacing_zyx_um": tuple(float(v) for v in self.config.target_spacing_zyx_um),
            "center_native_zyx": tuple(float(v) for v in center_native),
            "marker_count": len(markers),
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
            metadata=metadata,
        )
