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
    # Detection edges retain the legacy 14-D schema as an exact prefix and
    # append the accepted-Trackastra indicator at column 14.
    edge_dim: int = 15
    hypothesis_edge_dim: int = 22
    d_model: int = 128
    graph_layers: int = 2
    graph_heads: int = 4
    graph_ffn_dim: int = 256
    temporal_radius: int = 2
    k_spatial_neighbors: int = 6
    spatial_neighbor_radius_dref: float = 2.5
    status_dim: int = 10
    candidate_graph_enabled: bool = True
    max_candidate_edges: int | None = None
    memory_heads: int = 4
    memory_ffn_dim: int = 256
    relation_bias_hidden: int = 32
    memory_gate_init_bias: float = -2.0
    component_memory_enabled: bool = True
    query_memory_enabled: bool = True
    memory_debug_topk: int = 5


@dataclass
class HistoryConfig:
    """Compact, cell-scale historical-instance evidence."""

    enabled: bool = True
    grid_size: int = 12
    extent_dref: float = 2.5
    input_channels: int = 4
    support_channels: int = 2
    node_chunk_size: int = 128
    gate_init_bias: float = -2.0
    attention_bias_enabled: bool = True
    attention_bias_hidden: int = 32
    dt_normalizer: float = 2.0


@dataclass
class CoReasoningConfig:
    d_model: int = 128
    heads: int = 4
    blocks: int = 2
    base_radius_dref: float = 1.5
    max_radius_dref: float = 2.5
    position_bias_hidden: int = 32
    dropout: float = 0.10
    temporal_query_chunk_size: int = 8
    spatial_query_chunk_size: int = 8192
    spatial_key_chunk_size: int = 65536


@dataclass
class QueryConfig:
    d_model: int = 128
    split_companions_per_instance: int = 1
    max_split_companions_per_instance: int = 8
    split_volume_ratio_per_hypothesis: float = 1.0
    discovery_queries: int = 8
    max_queries: int | None = None
    instance_feature_dim: int = 14
    temporal_gaussian_sigma_dref: float = 0.75
    prior_inside_logit: float = 1.5
    prior_outside_logit: float = -1.5
    native_support_radius_dref: float = 1.5
    native_source_dilation_dref: float = 0.5
    native_background_logit: float = -20.0
    temporal_match_radius_dref: float = 1.0
    discovery_match_radius_dref: float = 1.5


@dataclass
class ProposalConfig:
    enabled: bool = True
    query_mode: str = "spatial_proposals"

    # Candidate generation.
    max_proposals: int = 128
    candidate_pool_size: int = 512
    nms_radius_dref: float = 0.35
    inference_score_threshold: float = 0.05

    # Proposal-local representation.
    local_grid_size: int = 3
    local_extent_dref: float = 0.50
    local_dim: int = 128

    # Safe coverage for existing components.
    ensure_source_fallback: bool = True
    source_fallback_match_radius_dref: float = 0.50

    # Structured matching.
    match_radius_dref: float = 1.0

    # Query-decoder spatial support.
    attention_radius_layer0_dref: float = 1.0
    attention_radius_layer1_dref: float = 1.5
    attention_radius_layer2_dref: float = 2.0

    # Native rendering support.
    native_support_radius_dref: float = 1.5

    # Component context must remain secondary to local proposal identity.
    component_context_gate_init_bias: float = -1.5


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
    primary_center_step_dref: float = 0.50
    split_center_step_dref: float = 0.75
    temporal_center_step_dref: float = 0.25
    discovery_center_step_dref: float = 1.00
    proposal_center_max_offset_dref: float = 0.50
    # Deprecated compatibility override.  New configurations should leave
    # this unset so proposal corrections are clearly total anchor offsets.
    proposal_center_step_dref: float | None = None


@dataclass
class LocalMaskConfig:
    enabled: bool = True
    support_radius_dref: float = 1.5
    hidden_channels: int = 32
    query_channels: int = 32
    query_chunk_size: int = 1
    detach_dense_evidence: bool = True


@dataclass
class LossConfig:
    exist: float = 2.0
    dice_hi: float = 5.0
    focal_hi: float = 2.0
    dice_coarse: float = 1.0
    focal_coarse: float = 0.5
    center: float = 2.0
    count: float = 0.25
    overlap: float = 0.0
    foreground: float = 0.50
    center_heatmap: float = 1.00
    boundary: float = 0.50
    internal_boundary: float = 0.25
    proposal_center: float = 0.75
    aux_layer: float = 0.50
    exist_focal_gamma: float = 2.0
    exist_focal_alpha_pos: float = 0.75
    exist_focal_alpha_neg: float = 0.25
    mask_supervision_radius_dref: float = 1.5
    mask_focal_alpha_pos: float = 0.75
    mask_focal_gamma: float = 2.0
    boundary_pos_weight: float = 4.0
    native_chunk_voxels: int = 262_144
    dense_chunk_voxels: int = 524_288


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
    activation_checkpointing: bool = True
    checkpoint_spatial: bool = True
    checkpoint_coreasoning: bool = True
    checkpoint_history: bool = True
    checkpoint_losses: bool = True


@dataclass
class CurriculumConfig:
    enabled: bool = False
    spatial_dense_steps: int = 0
    temporal_dense_steps: int = 0
    query_bootstrap_steps: int = 0
    native_bootstrap_steps: int = 0
    joint_spatial_lr_scale: float = 0.10
    joint_dense_lr_scale: float = 0.50


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
    history: HistoryConfig = field(default_factory=HistoryConfig)
    coreasoning: CoReasoningConfig = field(default_factory=CoReasoningConfig)
    queries: QueryConfig = field(default_factory=QueryConfig)
    proposals: ProposalConfig = field(default_factory=ProposalConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    local_masks: LocalMaskConfig = field(default_factory=LocalMaskConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_dict(self) -> dict:
        return asdict(self)
