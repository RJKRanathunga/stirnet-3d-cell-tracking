from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass
class SpatialConfig:
    in_channels: int = 5
    channels: Tuple[int, ...] = (16, 32, 64, 128)
    blocks_per_level: int = 2
    anisotropy_threshold: float = 1.5
    acquisition_dim: int = 64
    group_norm_max_groups: int = 8
    mask_dim: int = 32


@dataclass
class TemporalConfig:
    node_dim: int = 32
    edge_dim: int = 14
    hypothesis_edge_dim: int = 8
    d_model: int = 128
    graph_layers: int = 2
    graph_heads: int = 4
    graph_ffn_dim: int = 256
    temporal_radius: int = 2
    k_spatial_neighbors: int = 6
    spatial_neighbor_radius_dref: float = 2.5
    status_dim: int = 10


@dataclass
class CoReasoningConfig:
    d_model: int = 128
    heads: int = 4
    blocks: int = 2
    base_radius_dref: float = 1.5
    max_radius_dref: float = 2.5
    position_bias_hidden: int = 32
    dropout: float = 0.10


@dataclass
class QueryConfig:
    d_model: int = 128
    split_companions_per_instance: int = 1
    discovery_queries: int = 8
    max_queries: int = 128
    instance_feature_dim: int = 14
    temporal_gaussian_sigma_dref: float = 0.75
    prior_inside_logit: float = 1.5
    prior_outside_logit: float = -1.5


@dataclass
class DecoderConfig:
    d_model: int = 128
    heads: int = 4
    layers: int = 3
    ffn_dim: int = 512
    dropout: float = 0.10
    mask_dim: int = 32
    mask_attention_threshold: float = 0.20
    max_spatial_tokens: int = 16384
    support_dilation_dref: float = 0.5


@dataclass
class LossConfig:
    exist: float = 2.0
    dice_hi: float = 5.0
    focal_hi: float = 2.0
    dice_coarse: float = 1.0
    focal_coarse: float = 0.5
    center: float = 2.0
    count: float = 0.25
    overlap: float = 0.10
    foreground: float = 0.50
    center_heatmap: float = 1.00
    boundary: float = 0.50
    aux_layer: float = 0.50
    exist_focal_gamma: float = 2.0
    exist_focal_alpha_pos: float = 0.75
    exist_focal_alpha_neg: float = 0.25
    boundary_pos_weight: float = 4.0


@dataclass
class TrainingConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    clean_sample_prob: float = 0.40
    corrupted_sample_prob: float = 0.60
    temporal_hypothesis_dropout: float = 0.10
    temporal_edge_dropout: float = 0.10
    temporal_position_jitter_dref: float = 0.10
    temporal_large_jitter_dref: float = 0.50
    temporal_false_clue_prob: float = 0.05


@dataclass
class InferenceConfig:
    patch_context_diameters: float = 8.0
    patch_valid_diameters: float = 6.0
    render_exist_threshold: float = 0.30
    final_exist_threshold: float = 0.50
    mask_threshold: float = 0.50
    min_mask_voxels: int = 8


@dataclass
class StirNetConfig:
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    coreasoning: CoReasoningConfig = field(default_factory=CoReasoningConfig)
    queries: QueryConfig = field(default_factory=QueryConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_dict(self) -> dict:
        return asdict(self)
