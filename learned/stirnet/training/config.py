from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class LossConfig:
    geometry_weight: float = 1.0
    spatial_rag_weight: float = 1.0
    final_rag_weight: float = 1.0
    existence_weight: float = 0.50
    split_weight: float = 0.50
    recovery_weight: float = 0.25

    existence_min_precision: float = 0.25
    existence_min_gt_coverage: float = 0.10
    split_min_pred_fraction: float = 0.10
    split_min_gt_coverage: float = 0.20
    recovery_search_radius_dref: float = 0.50
    recovery_missing_gt_coverage: float = 0.25


@dataclass
class CurriculumConfig:
    enabled: bool = True
    geometry_bootstrap_steps: int = 500
    spatial_partition_steps: int = 500
    instance_temporal_steps: int = 500
    fixed_stage: str | None = None
    spatial_lr_scale_temporal: float = 0.25
    spatial_lr_scale_refinement: float = 0.10


@dataclass
class TrainingConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    amp_dtype: str = "bf16"
    loss: LossConfig = field(default_factory=LossConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    def validate(self) -> None:
        if self.lr <= 0:
            raise ValueError("training lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.amp_dtype not in {"fp16", "bf16", "fp32"}:
            raise ValueError("amp_dtype must be fp16, bf16, or fp32")
        durations = (
            self.curriculum.geometry_bootstrap_steps,
            self.curriculum.spatial_partition_steps,
            self.curriculum.instance_temporal_steps,
        )
        if any(value < 0 for value in durations):
            raise ValueError("curriculum durations cannot be negative")
        weights = (
            self.loss.geometry_weight,
            self.loss.spatial_rag_weight,
            self.loss.final_rag_weight,
            self.loss.existence_weight,
            self.loss.split_weight,
            self.loss.recovery_weight,
        )
        if any(value < 0 for value in weights):
            raise ValueError("loss weights cannot be negative")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)
