"""Configuration for the learned tracklet reconciliation network."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FingerprintConfig:
    """Small 3-D CNN used to encode one cell and its immediate context."""

    in_channels: int = 5
    base_channels: int = 24
    embedding_dim: int = 96
    dropout: float = 0.05
    group_norm_groups: int = 8


@dataclass(frozen=True)
class TemporalConfig:
    """Temporal encoding for per-observation appearance and structured streams."""

    structured_dim: int = 48
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.10
    time_fourier_bands: int = 8
    fused_tracklet_dim: int = 192
    reliability_dim: int = 12


@dataclass(frozen=True)
class EdgeReasonerConfig:
    """Candidate-edge encoder and geometry-biased edge Transformer."""

    pair_feature_dim: int = 80
    relation_dim: int = 34
    edge_dim: int = 192
    num_heads: int = 4
    num_layers: int = 4
    feedforward_dim: int = 384
    dropout: float = 0.10
    physical_distance_scale_um: float = 10.0
    rope_position_scale_um: float = 10.0
    geometry_alpha_init: float = -5.0


@dataclass(frozen=True)
class DivisionConfig:
    """Sparse parent + daughter-pair hypothesis head."""

    pair_feature_dim: int = 16
    hidden_dim: int = 256
    dropout: float = 0.10


@dataclass(frozen=True)
class CandidateConfig:
    """High-recall deterministic candidate gating defaults."""

    maximum_gap_frames: int = 4
    radius_base_um: float = 12.0
    radius_per_extra_gap_um: float = 4.0
    radius_max_um: float = 25.0
    maximum_targets_per_source: int = 12
    maximum_division_children_per_parent: int = 4


@dataclass(frozen=True)
class DecoderConfig:
    """Global event decoder settings."""

    milp_time_limit_seconds: float | None = None
    mip_relative_gap: float | None = None


@dataclass(frozen=True)
class ReconcilerConfig:
    """Top-level architecture configuration.

    Defaults are deliberately modest so local reconciliation components fit
    comfortably on the project's 6-GB development GPU.  The edge reasoner is
    the main learned relational component; the CNN and temporal encoders are
    intentionally lightweight.
    """

    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    edge: EdgeReasonerConfig = field(default_factory=EdgeReasonerConfig)
    division: DivisionConfig = field(default_factory=DivisionConfig)
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    # Branch-level modality dropout makes Trackastra / image / handcrafted
    # evidence less likely to become a shortcut during training.
    appearance_modality_dropout: float = 0.08
    structured_modality_dropout: float = 0.05
    pair_modality_dropout: float = 0.08

    def __post_init__(self) -> None:
        if self.edge.edge_dim % self.edge.num_heads:
            raise ValueError("edge_dim must be divisible by edge.num_heads")
        head_dim = self.edge.edge_dim // self.edge.num_heads
        if head_dim % 6:
            raise ValueError(
                "edge attention head dimension must be divisible by 6 for 3-D RoPE"
            )
        for name in (
            "appearance_modality_dropout",
            "structured_modality_dropout",
            "pair_modality_dropout",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
