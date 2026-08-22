from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class SpatialPyramid:
    features: List[Tensor]
    spacings_um: List[Tensor]
    strides: List[tuple[int, int, int]]
    padding_masks: Optional[List[Tensor]] = None

    @property
    def e0(self) -> Tensor:
        return self.features[0]

    @property
    def e1(self) -> Tensor:
        return self.features[1]

    @property
    def e2(self) -> Tensor:
        return self.features[2]

    @property
    def e3(self) -> Tensor:
        return self.features[3]


@dataclass
class SpatialDecodeState:
    d2: Tensor
    d1: Tensor
    d0: Tensor


@dataclass
class GeometryState:
    foreground_logits: Tensor
    surface_logits: Tensor
    separator_logits: Tensor
    sdf: Tensor
    flow: Tensor
    centroid_offset: Tensor
    seed_logits: Tensor
    features: Optional[Tensor] = None
    feature_spacing_um: Optional[Tensor] = None

    def probabilities(self) -> Dict[str, Tensor]:
        return {
            "foreground": self.foreground_logits.sigmoid(),
            "surface": self.surface_logits.sigmoid(),
            "separator": self.separator_logits.sigmoid(),
            "seed": self.seed_logits.sigmoid(),
        }


@dataclass(frozen=True)
class GeometryDerivedCache:
    """Forward-scoped, non-parameter geometry used by spatial partitioning."""

    foreground_prob: Tensor
    surface_prob: Tensor
    separator_prob: Tensor
    seed_prob: Tensor
    sdf: Tensor
    sdf_normalized: Tensor
    foreground_mask: Tensor
    seed_score: Tensor
    watershed_energy: Tensor


@dataclass(frozen=True)
class ScaleFeatureStatistics:
    counts: Tensor
    sums: Tensor
    maxima: Tensor

    @property
    def means(self) -> Tensor:
        return self.sums / self.counts.clamp_min(1)[:, None]

    @property
    def mean_max(self) -> Tensor:
        return torch.cat([self.means, self.maxima], dim=-1)


@dataclass(frozen=True)
class SupervoxelStatistics:
    """Sufficient statistics for one batch item's positive supervoxels."""

    volume_shape_zyx: tuple[int, int, int]
    spacing_um: Tensor
    counts: Tensor
    coordinate_sums: Tensor
    coordinate_square_sums: Tensor
    min_voxel: Tensor
    max_voxel: Tensor
    field_sums: Dict[str, Tensor]
    field_maxima: Dict[str, Tensor]
    sdf_argmax_flat_index: Tensor
    scales: tuple[ScaleFeatureStatistics, ScaleFeatureStatistics, ScaleFeatureStatistics]

    @property
    def centroid_voxel(self) -> Tensor:
        return self.coordinate_sums / self.counts.clamp_min(1)[:, None]

    @property
    def centroid_um(self) -> Tensor:
        center = 0.5 * (
            torch.as_tensor(
                self.volume_shape_zyx,
                device=self.counts.device,
                dtype=torch.float32,
            )
            - 1
        )
        return (self.centroid_voxel - center) * self.spacing_um.float()[None]

    @property
    def variance_um2(self) -> Tensor:
        mean = self.centroid_voxel
        second = self.coordinate_square_sums / self.counts.clamp_min(1)[:, None]
        return (second - mean.square()).clamp_min(0) * self.spacing_um.float().square()[None]

    def field_means(self, name: str) -> Tensor:
        return self.field_sums[name] / self.counts.to(self.field_sums[name].dtype).clamp_min(1)


@dataclass(frozen=True)
class AggregatedRegionStatistics:
    counts: Tensor
    coordinate_sums: Tensor
    coordinate_square_sums: Tensor
    min_voxel: Tensor
    max_voxel: Tensor
    field_sums: Dict[str, Tensor]
    field_maxima: Dict[str, Tensor]
    sdf_argmax_flat_index: Tensor
    scales: tuple[ScaleFeatureStatistics, ScaleFeatureStatistics, ScaleFeatureStatistics]
    volume_shape_zyx: tuple[int, int, int]
    spacing_um: Tensor

    @property
    def centroid_voxel(self) -> Tensor:
        return self.coordinate_sums / self.counts.clamp_min(1)[:, None]

    @property
    def centroid_um(self) -> Tensor:
        center = 0.5 * (
            torch.as_tensor(self.volume_shape_zyx, device=self.counts.device, dtype=torch.float32) - 1
        )
        return (self.centroid_voxel - center) * self.spacing_um.float()[None]

    @property
    def variance_um2(self) -> Tensor:
        second = self.coordinate_square_sums / self.counts.clamp_min(1)[:, None]
        return (second - self.centroid_voxel.square()).clamp_min(0) * self.spacing_um.float().square()[None]

    def field_means(self, name: str) -> Tensor:
        return self.field_sums[name] / self.counts.to(self.field_sums[name].dtype).clamp_min(1)


_GEOMETRY_DELTA_CHANNELS = {
    "foreground_logits": slice(0, 1),
    "surface_logits": slice(1, 2),
    "separator_logits": slice(2, 3),
    "sdf": slice(3, 4),
    "flow": slice(4, 7),
    "centroid_offset": slice(7, 10),
    "seed_logits": slice(10, 11),
}


@dataclass(frozen=True)
class SparseGeometryROI:
    batch_index: int
    slices_zyx: tuple[slice, slice, slice]
    delta: Tensor


@dataclass(frozen=True)
class SparseGeometryDelta:
    rois: List[SparseGeometryROI] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.rois


@dataclass(frozen=True)
class RefinedGeometryView:
    """Lazy base-plus-sparse-delta view of corrected dense geometry."""

    base: GeometryState
    delta: SparseGeometryDelta

    def materialize_field(self, name: str) -> Tensor:
        if name not in _GEOMETRY_DELTA_CHANNELS:
            raise KeyError(f"Unknown geometry field: {name}")
        base = getattr(self.base, name)
        channel_slice = _GEOMETRY_DELTA_CHANNELS[name]
        shape = base.shape[-3:]
        batches: list[Tensor] = []
        for batch_index in range(base.shape[0]):
            overlay: Tensor | None = None
            for roi in self.delta.rois:
                if roi.batch_index != batch_index:
                    continue
                zyx = roi.slices_zyx
                padding = (
                    int(zyx[2].start),
                    shape[2] - int(zyx[2].stop),
                    int(zyx[1].start),
                    shape[1] - int(zyx[1].stop),
                    int(zyx[0].start),
                    shape[0] - int(zyx[0].stop),
                )
                padded = F.pad(roi.delta[channel_slice][None], padding)
                overlay = padded if overlay is None else overlay + padded
            row = base[batch_index : batch_index + 1]
            batches.append(row if overlay is None else row + overlay.to(row.dtype))
        return torch.cat(batches, dim=0)

    def field_crop(
        self,
        name: str,
        batch_index: int,
        crop: tuple[slice, slice, slice],
    ) -> Tensor:
        base = getattr(self.base, name)
        result = base[batch_index, :, crop[0], crop[1], crop[2]]
        channel_slice = _GEOMETRY_DELTA_CHANNELS[name]
        crop_shape = tuple(int(axis.stop) - int(axis.start) for axis in crop)
        for roi in self.delta.rois:
            if roi.batch_index != batch_index:
                continue
            overlap_start = [
                max(int(crop[axis].start), int(roi.slices_zyx[axis].start))
                for axis in range(3)
            ]
            overlap_stop = [
                min(int(crop[axis].stop), int(roi.slices_zyx[axis].stop))
                for axis in range(3)
            ]
            if any(stop <= start for start, stop in zip(overlap_start, overlap_stop)):
                continue
            source = tuple(
                slice(
                    overlap_start[axis] - int(roi.slices_zyx[axis].start),
                    overlap_stop[axis] - int(roi.slices_zyx[axis].start),
                )
                for axis in range(3)
            )
            padding = (
                overlap_start[2] - int(crop[2].start),
                int(crop[2].stop) - overlap_stop[2],
                overlap_start[1] - int(crop[1].start),
                int(crop[1].stop) - overlap_stop[1],
                overlap_start[0] - int(crop[0].start),
                int(crop[0].stop) - overlap_stop[0],
            )
            local = roi.delta[channel_slice, source[0], source[1], source[2]]
            result = result + F.pad(local, padding).to(result.dtype)
        if result.shape[-3:] != crop_shape:
            raise RuntimeError("sparse geometry crop shape mismatch")
        return result

    @property
    def foreground_logits(self) -> Tensor:
        return self.materialize_field("foreground_logits")

    @property
    def surface_logits(self) -> Tensor:
        return self.materialize_field("surface_logits")

    @property
    def separator_logits(self) -> Tensor:
        return self.materialize_field("separator_logits")

    @property
    def sdf(self) -> Tensor:
        return self.materialize_field("sdf")

    @property
    def flow(self) -> Tensor:
        return self.materialize_field("flow")

    @property
    def centroid_offset(self) -> Tensor:
        return self.materialize_field("centroid_offset")

    @property
    def seed_logits(self) -> Tensor:
        return self.materialize_field("seed_logits")

    @property
    def features(self) -> Optional[Tensor]:
        return self.base.features

    @property
    def feature_spacing_um(self) -> Optional[Tensor]:
        return self.base.feature_spacing_um

    def probabilities(self) -> Dict[str, Tensor]:
        return {
            "foreground": self.foreground_logits.sigmoid(),
            "surface": self.surface_logits.sigmoid(),
            "separator": self.separator_logits.sigmoid(),
            "seed": self.seed_logits.sigmoid(),
        }


GeometryLike = GeometryState | RefinedGeometryView


def geometry_field(geometry: GeometryLike, name: str) -> Tensor:
    return (
        geometry.materialize_field(name)
        if isinstance(geometry, RefinedGeometryView)
        else getattr(geometry, name)
    )


def geometry_field_crop(
    geometry: GeometryLike,
    name: str,
    batch_index: int,
    crop: tuple[slice, slice, slice],
) -> Tensor:
    if isinstance(geometry, RefinedGeometryView):
        return geometry.field_crop(name, batch_index, crop)
    field = getattr(geometry, name)
    return field[batch_index, :, crop[0], crop[1], crop[2]]


def geometry_probability(geometry: GeometryLike, name: str) -> Tensor:
    field_name = {
        "foreground": "foreground_logits",
        "surface": "surface_logits",
        "separator": "separator_logits",
        "seed": "seed_logits",
    }[name]
    return geometry_field(geometry, field_name).sigmoid()


@dataclass
class RAGState:
    node_features: Tensor
    node_embeddings: Tensor
    node_batch: Tensor
    node_supervoxel_id: Tensor
    node_centroid_um: Tensor
    node_volume_voxels: Tensor
    edge_index: Tensor
    edge_features: Tensor
    edge_embeddings: Tensor
    spatial_edge_logits: Tensor
    edge_batch: Tensor
    supervoxel_labels: List[Tensor]
    node_offsets: Tensor
    statistics: Optional[List[SupervoxelStatistics]] = None

    @property
    def is_empty(self) -> bool:
        return self.node_features.shape[0] == 0


@dataclass
class PartitionState:
    labels: List[Tensor]
    node_component: Tensor
    node_component_global: Tensor
    component_count_per_batch: Tensor
    edge_logits: Tensor


@dataclass
class InstanceState:
    tokens: Tensor
    ref_um: Tensor
    batch_index: Tensor
    local_ids: Tensor
    quality_logits: Tensor
    labels: List[Tensor]
    token_offsets: Tensor
    node_to_instance: Tensor
    spatial_tokens: Optional[Tensor] = None

    @property
    def is_empty(self) -> bool:
        return self.tokens.shape[0] == 0


@dataclass
class TemporalInput:
    graph_x: Tensor
    graph_edge_index: Tensor
    graph_edge_attr: Tensor
    tracklet_id: Tensor
    temporal_ref_um: Tensor
    temporal_status: Tensor
    temporal_batch: Tensor
    node_history_embedding: Optional[Tensor] = None
    # Directed tracklet-level pair evidence generated by data/graph_builder.py.
    hypothesis_edge_index: Optional[Tensor] = None
    hypothesis_edge_attr: Optional[Tensor] = None


@dataclass
class TemporalState:
    tokens: Tensor
    ref_um: Tensor
    batch_index: Tensor
    salience: Tensor
    reliability: Tensor
    status: Tensor
    node_tokens: Optional[Tensor] = None

    @property
    def is_empty(self) -> bool:
        return self.tokens.shape[0] == 0


@dataclass
class SpatialObservationCache:
    """Compact sample-first spatial evidence aligned with temporal rows."""

    d1_projected: Tensor
    d2_projected: Tensor
    hidden_geometry_projected: Tensor
    explicit_geometry_projected: Optional[Tensor] = None


@dataclass
class ReasoningState:
    instance_tokens: Tensor
    instance_exist_logits: Tensor
    split_logits: Tensor
    temporal_support: Tensor
    temporal_attention_entropy: Tensor
    edge_temporal_delta: Tensor
    edge_temporal_gate: Tensor
    final_edge_logits: Tensor
    recovery_track_indices: Tensor
    recovery_logits: Tensor
    recovery_scores: Tensor


@dataclass
class RefinementRequest:
    batch_index: int
    center_um: Tensor
    query_token: Tensor
    kind: str
    source_index: int
    score: float
    selection_source: str = "model"


@dataclass
class RefinementState:
    geometry: GeometryLike
    requests: List[RefinementRequest] = field(default_factory=list)
    applied_count: int = 0
    model_request_count: int = 0
    teacher_request_count: int = 0
    partition_update: str = "none"
    partition_fallback: bool = False
    partition_fallback_reason: str = ""
    partition_fallback_reason_code: int = 0
    partition_fallback_batch_index: int = -1
    partition_fallback_box_index: int = -1
    partition_fallback_box_shape_zyx: tuple[int, int, int] | None = None
    partition_fallback_box_voxel_count: int = 0
    partition_fallback_core_voxel_count: int = 0
    partition_fallback_local_component_count: int = 0
    partition_fallback_old_core_label_count: int = 0
    partition_fallback_old_shell_label_count: int = 0
    partition_fallback_conflicting_old_label_ids: List[int] = field(
        default_factory=list
    )
    local_update_box_count: int = 0
    local_update_voxel_fraction: float = 0.0


@dataclass
class GeometryForwardOutput:
    geometry: GeometryState
    spatial_pyramid: SpatialPyramid
    decoded_spatial: SpatialDecodeState


@dataclass
class SpatialForwardOutput(GeometryForwardOutput):
    rag: RAGState
    spatial_partition: PartitionState


@dataclass
class StirNetOutput:
    final_labels: List[Tensor]
    centers_um: List[Tensor]
    geometry: GeometryLike
    spatial_pyramid: SpatialPyramid
    decoded_spatial: SpatialDecodeState
    rag: RAGState
    spatial_partition: PartitionState
    provisional_instances: InstanceState
    temporal: TemporalState
    reasoning: ReasoningState
    final_partition: PartitionState
    initial_geometry: GeometryState
    initial_rag: RAGState
    initial_spatial_partition: PartitionState
    initial_provisional_instances: InstanceState
    initial_reasoning: ReasoningState
    refinement: Optional[RefinementState] = None
    debug: Optional[Dict[str, Any]] = None

    @property
    def dense_outputs(self) -> Dict[str, Tensor]:
        # Compatibility helper for existing diagnostics/training code.
        return {
            "foreground_logits": self.geometry.foreground_logits,
            "surface_boundary_logits": self.geometry.surface_logits,
            "boundary_logits": self.geometry.separator_logits,
            "separator_logits": self.geometry.separator_logits,
            "sdf": self.geometry.sdf,
            "flow": self.geometry.flow,
            "centroid_offset": self.geometry.centroid_offset,
            "seed_logits": self.geometry.seed_logits,
        }
