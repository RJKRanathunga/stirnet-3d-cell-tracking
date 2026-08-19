from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple


@dataclass
class EvidenceStemConfig:
    """Separate raw-image evidence from fallible segmentation priors."""

    raw_channels: Tuple[int, ...] = (0,)
    prior_channels: Tuple[int, ...] = (1, 2, 3, 4)
    stem_channels: int = 24
    prior_gate_hidden: int = 32
    prior_dropout: float = 0.20


@dataclass
class SpatialConfig:
    in_channels: int = 5
    channels: Tuple[int, ...] = (24, 48, 96, 192)
    blocks_per_level: int = 2
    anisotropy_threshold: float = 1.5
    acquisition_dim: int = 64
    group_norm_max_groups: int = 8
    activation_checkpointing: bool = True
    canonical_spacing_um: Optional[Tuple[float, float, float]] = None
    axis_conv_variant: str = "dense"
    axis_conv_bottleneck_ratio: float = 0.5


@dataclass
class GeometryConfig:
    """Native-resolution dense geometry, following Omnipose/NucMM/NISNet3D ideas."""

    hidden_channels: int = 48
    residual_blocks: int = 3
    sdf_clip_dref: float = 2.5
    sdf_supervision_radius_dref: float = 2.5
    boundary_pos_weight: float = 6.0
    separator_pos_weight: float = 10.0
    surface_target_sigma_um: float = 0.75
    separator_target_sigma_um: float = 0.50
    # GT defines object identity; current/noisy instances define where an
    # under-segmentation correction sheet is required.
    separator_source_conditioned: bool = True
    separator_source_min_overlap_voxels: int = 8
    separator_source_min_gt_fraction: float = 0.05
    surface_dice_weight: float = 1.0
    separator_dice_weight: float = 1.0
    seed_pos_weight: float = 4.0

    # Suppress learned flow only in the narrow exterior surface band.
    flow_background_weight: float = 1.0
    flow_background_surface_threshold: float = 0.05

    # Explicitly supervise the norm of the foreground EDT-gradient flow.
    flow_magnitude_weight: float = 1.0

    consistency_weight: float = 0.0
    eikonal_weight: float = 0.10


@dataclass
class PartitionConfig:
    """Learned-geometry watershed followed by a learned RAG partition."""

    foreground_threshold: float = 0.45
    seed_threshold: float = 0.40
    seed_min_distance_dref: float = 0.35
    seed_sdf_weight: float = 0.55
    seed_head_weight: float = 0.45
    watershed_separator_weight: float = 0.65
    watershed_surface_weight: float = 0.15
    watershed_sdf_weight: float = 0.20
    min_supervoxel_voxels: int = 4
    node_feature_channels: int = 24
    rag_hidden_dim: int = 96
    rag_layers: int = 2
    spatial_merge_threshold: float = 0.50
    final_merge_threshold: float = 0.50
    max_supervoxels: int = 4096
    rag_min_node_purity: float = 0.80
    # Require at least half of a proposed node to be backed by GT foreground.
    # This keeps mostly-background supervoxels out of edge supervision while
    # still tolerating imperfect proposal boundaries during early training.
    rag_min_node_gt_support: float = 0.50
    watershed_backend: str = "fast"
    region_stats_backend: str = "auto"
    component_bounded_watershed: bool = True
    watershed_component_halo_voxels: int = 1


@dataclass
class InstanceConfig:
    d_model: int = 128
    pooled_feature_dim: int = 32
    shape_feature_dim: int = 12
    dropout: float = 0.10
    exist_threshold: float = 0.40
    apply_existence_filter: bool = True


@dataclass
class HistoryConfig:
    enabled: bool = True
    input_channels: int = 4
    hidden_channels: int = 32
    dropout: float = 0.10
    node_chunk_size: int = 128
    activation_checkpointing: bool = True



@dataclass
class TemporalConfig:
    """Compatibility-oriented temporal encoder for the existing STIR-Net preprocessing."""

    node_dim: int = 32
    edge_dim: int = 15
    status_dim: int = 10
    d_model: int = 128
    graph_layers: int = 2
    graph_hidden_dim: int = 256
    dropout: float = 0.10
    observation_radius_dref: float = 1.75
    instance_match_radius_dref: float = 2.50
    cross_heads: int = 4
    temporal_residual_scale: float = 2.0
    reliability_floor: float = 0.05


@dataclass
class RefinementConfig:
    enabled: bool = True
    roi_radius_dref: float = 1.75
    hidden_channels: int = 48
    query_channels: int = 32
    max_rois_per_batch: int = 16
    max_roi_voxels: int = 262_144
    request_nms_radius_dref: float = 0.50
    split_threshold: float = 0.55
    recovery_threshold: float = 0.60
    ambiguity_logit_abs_max: float = 0.85
    residual_scale: float = 0.75
    partition_update: str = "local"
    partition_halo_dref: float = 1.0


@dataclass
class InferenceConfig:
    mode: str = "full"
    tiled_dense_enabled: bool = False
    tile_shape_zyx: Tuple[int, int, int] = (32, 128, 128)
    tile_overlap_zyx: Tuple[int, int, int] = (8, 32, 32)
    tile_halo_zyx: Tuple[int, int, int] = (4, 16, 16)
    tile_batch_size: int = 1


@dataclass
class ModelConfig:
    evidence: EvidenceStemConfig = field(default_factory=EvidenceStemConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    instances: InstanceConfig = field(default_factory=InstanceConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def validate(self) -> None:
        if len(self.spatial.channels) != 4:
            raise ValueError("SpatialConfig.channels must contain four levels E0..E3")
        if any(c <= 0 for c in self.spatial.channels):
            raise ValueError("All spatial channel widths must be positive")
        if self.spatial.canonical_spacing_um is not None and any(
            value <= 0 for value in self.spatial.canonical_spacing_um
        ):
            raise ValueError("canonical_spacing_um values must be positive")
        if self.spatial.axis_conv_variant not in {"dense", "depthwise"}:
            raise ValueError("axis_conv_variant must be 'dense' or 'depthwise'")
        if not 0 < self.spatial.axis_conv_bottleneck_ratio <= 1:
            raise ValueError("axis_conv_bottleneck_ratio must be in (0, 1]")
        if self.instances.d_model != self.temporal.d_model:
            raise ValueError("Instance and temporal d_model must match")
        if self.temporal.d_model % self.temporal.cross_heads:
            raise ValueError("temporal.d_model must be divisible by temporal.cross_heads")
        if not 0.0 < self.partition.foreground_threshold < 1.0:
            raise ValueError("foreground_threshold must be in (0, 1)")
        if self.partition.max_supervoxels < 1:
            raise ValueError("max_supervoxels must be positive")
        if self.partition.watershed_backend not in {"reference", "fast"}:
            raise ValueError("partition.watershed_backend must be 'reference' or 'fast'")
        if self.partition.region_stats_backend not in {"torch", "auto"}:
            raise ValueError("partition.region_stats_backend must be 'torch' or 'auto'")
        if self.partition.watershed_component_halo_voxels < 0:
            raise ValueError("watershed_component_halo_voxels cannot be negative")
        if self.geometry.sdf_clip_dref <= 0:
            raise ValueError("sdf_clip_dref must be positive")
        if self.geometry.sdf_supervision_radius_dref <= 0:
            raise ValueError("sdf_supervision_radius_dref must be positive")
        if self.geometry.surface_target_sigma_um <= 0:
            raise ValueError("surface_target_sigma_um must be positive")
        if self.geometry.separator_target_sigma_um <= 0:
            raise ValueError("separator_target_sigma_um must be positive")
        if self.geometry.separator_source_min_overlap_voxels < 1:
            raise ValueError(
                "separator_source_min_overlap_voxels must be positive"
            )
        if not 0.0 <= self.geometry.separator_source_min_gt_fraction <= 1.0:
            raise ValueError(
                "separator_source_min_gt_fraction must be in [0, 1]"
            )
        if self.geometry.flow_background_weight < 0:
            raise ValueError("flow_background_weight cannot be negative")
        if self.geometry.flow_magnitude_weight < 0:
            raise ValueError("flow_magnitude_weight cannot be negative")
        if not 0.0 <= self.geometry.flow_background_surface_threshold <= 1.0:
            raise ValueError(
                "flow_background_surface_threshold must be in [0, 1]"
            )
        if not 0.0 <= self.partition.rag_min_node_purity <= 1.0:
            raise ValueError("rag_min_node_purity must be in [0, 1]")
        if not 0.0 <= self.partition.rag_min_node_gt_support <= 1.0:
            raise ValueError("rag_min_node_gt_support must be in [0, 1]")
        channels = self.evidence.raw_channels + self.evidence.prior_channels
        if sorted(channels) != list(range(self.spatial.in_channels)):
            raise ValueError(
                "Evidence raw/prior channels must form an exact, non-overlapping "
                "partition of spatial input channels"
            )
        if self.history.node_chunk_size < 1:
            raise ValueError("history.node_chunk_size must be positive")
        if self.refinement.max_rois_per_batch < 1:
            raise ValueError("refinement.max_rois_per_batch must be positive")
        if self.refinement.max_roi_voxels < 1:
            raise ValueError("refinement.max_roi_voxels must be positive")
        if self.refinement.request_nms_radius_dref < 0:
            raise ValueError("refinement.request_nms_radius_dref cannot be negative")
        if self.refinement.partition_update not in {"local", "full"}:
            raise ValueError("refinement.partition_update must be 'local' or 'full'")
        if self.refinement.partition_halo_dref < 0:
            raise ValueError("refinement.partition_halo_dref cannot be negative")
        if self.inference.mode not in {"full", "tiled"}:
            raise ValueError("inference.mode must be 'full' or 'tiled'")
        if self.inference.tile_batch_size < 1:
            raise ValueError("inference.tile_batch_size must be positive")
        for size, overlap, halo in zip(
            self.inference.tile_shape_zyx,
            self.inference.tile_overlap_zyx,
            self.inference.tile_halo_zyx,
        ):
            if size < 1 or overlap < 0 or halo < 0:
                raise ValueError("tile sizes must be positive; overlap/halo non-negative")
            if overlap >= size:
                raise ValueError("tile overlap must be smaller than tile shape")
            if halo * 2 >= size:
                raise ValueError("tile halo must leave a non-empty reliable interior")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


# Backward-friendly alias for Codex integration with the existing package.
StirNetConfig = ModelConfig
