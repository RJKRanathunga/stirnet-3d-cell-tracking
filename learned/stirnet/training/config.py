from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Tuple


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
    recovery_min_pred_precision: float = 0.10


@dataclass
class CurriculumConfig:
    enabled: bool = True
    geometry_bootstrap_steps: int = 500
    spatial_partition_steps: int = 500
    instance_temporal_steps: int = 500
    fixed_stage: str | None = None
    spatial_lr_scale_temporal: float = 0.25
    spatial_lr_scale_refinement: float = 0.10
    refinement_crop_enabled: bool = True
    refinement_crop_shape_zyx: Tuple[int, int, int] = (32, 192, 192)
    refinement_crops_per_step: int = 1
    refinement_crop_sampling: str = "coverage"
    refinement_crop_min_foreground_fraction: float = 0.001
    refinement_crop_min_complete_cells: int = 3
    refinement_crop_preferred_complete_cells: int = 4
    refinement_crop_views_per_cell: int = 1
    refinement_crop_context_um: float = 4.0
    refinement_crop_partial_ignore_margin_um: float = 1.0
    refinement_crop_merge_min_overlap_voxels: int = 8
    refinement_crop_merge_min_gt_fraction: float = 0.05
    refinement_crop_source_dropout_probability: float = 0.15
    refinement_crop_source_dropout_max_instances: int = 1
    refinement_crop_geometry_weight: float = 1.0
    refinement_crop_rag_weight: float = 0.0
    refinement_crop_seed: int = 40_266
    full_frame_spatial_grad: bool = False
    geometry_bootstrap_crop_enabled: bool = True
    spatial_partition_crop_enabled: bool = True
    instance_temporal_detached_spatial: bool = False


@dataclass
class TrainingConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    amp_dtype: str = "bf16"
    profile_memory: bool = False
    memory_profile_path: str | None = None
    geometry_target_backend: str = "auto"
    geometry_target_gpu_min_voxels: int = 262_144
    refinement_teacher_forcing_start: float = 1.0
    refinement_teacher_forcing_end: float = 0.0
    refinement_teacher_forcing_decay_steps: int = 2_000
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
        if self.geometry_target_backend not in {"auto", "scipy", "cupy"}:
            raise ValueError("geometry_target_backend must be auto, scipy, or cupy")
        if self.geometry_target_gpu_min_voxels < 1:
            raise ValueError("geometry_target_gpu_min_voxels must be positive")
        if not 0.0 <= self.refinement_teacher_forcing_start <= 1.0:
            raise ValueError("refinement_teacher_forcing_start must be in [0, 1]")
        if not 0.0 <= self.refinement_teacher_forcing_end <= 1.0:
            raise ValueError("refinement_teacher_forcing_end must be in [0, 1]")
        if self.refinement_teacher_forcing_decay_steps < 0:
            raise ValueError("refinement_teacher_forcing_decay_steps cannot be negative")
        durations = (
            self.curriculum.geometry_bootstrap_steps,
            self.curriculum.spatial_partition_steps,
            self.curriculum.instance_temporal_steps,
        )
        if any(value < 0 for value in durations):
            raise ValueError("curriculum durations cannot be negative")
        if any(value < 1 for value in self.curriculum.refinement_crop_shape_zyx):
            raise ValueError("refinement crop dimensions must be positive")
        if self.curriculum.refinement_crops_per_step < 1:
            raise ValueError("refinement_crops_per_step must be positive")
        if self.curriculum.refinement_crop_sampling not in {"mixed", "coverage"}:
            raise ValueError("refinement_crop_sampling must be mixed or coverage")
        if self.curriculum.refinement_crop_min_complete_cells < 1:
            raise ValueError("refinement_crop_min_complete_cells must be positive")
        if self.curriculum.refinement_crop_preferred_complete_cells < self.curriculum.refinement_crop_min_complete_cells:
            raise ValueError("refinement_crop_preferred_complete_cells must be >= minimum")
        if self.curriculum.refinement_crop_views_per_cell < 1:
            raise ValueError("refinement_crop_views_per_cell must be positive")
        if self.curriculum.refinement_crop_context_um < 0 or self.curriculum.refinement_crop_partial_ignore_margin_um < 0:
            raise ValueError("crop physical margins cannot be negative")
        if self.curriculum.refinement_crop_merge_min_overlap_voxels < 1:
            raise ValueError("refinement_crop_merge_min_overlap_voxels must be positive")
        if not 0.0 <= self.curriculum.refinement_crop_merge_min_gt_fraction <= 1.0:
            raise ValueError("refinement_crop_merge_min_gt_fraction must be in [0,1]")
        if not 0.0 <= self.curriculum.refinement_crop_source_dropout_probability <= 1.0:
            raise ValueError("refinement_crop_source_dropout_probability must be in [0,1]")
        if self.curriculum.refinement_crop_source_dropout_max_instances < 0:
            raise ValueError("refinement_crop_source_dropout_max_instances cannot be negative")
        if not 0.0 <= self.curriculum.refinement_crop_min_foreground_fraction <= 1.0:
            raise ValueError(
                "refinement_crop_min_foreground_fraction must be in [0, 1]"
            )
        if self.curriculum.refinement_crop_geometry_weight < 0:
            raise ValueError("refinement_crop_geometry_weight cannot be negative")
        if self.curriculum.refinement_crop_rag_weight < 0:
            raise ValueError("refinement_crop_rag_weight cannot be negative")
        if self.curriculum.refinement_crop_seed < 0:
            raise ValueError("refinement_crop_seed cannot be negative")
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
