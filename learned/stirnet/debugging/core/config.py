from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True)
class DebugConfig:
    """Configuration for one STIR-Net inspection pass."""

    device: str | None = None
    amp_dtype: str = "fp16"  # fp16 | bf16 | none

    capture_module_stats: bool = True
    capture_spatial: bool = True
    capture_temporal: bool = True
    capture_queries: bool = True
    capture_matching: bool = True
    capture_dense_outputs: bool = False
    capture_scene_arrays: bool = False
    capture_native_masks: bool = False

    selected_query_indices: tuple[int, ...] = ()
    max_selected_queries: int = 6
    select_worst_center: int = 2
    select_worst_coarse_dice: int = 2
    select_top_surviving_temporal: int = 1
    select_top_split: int = 1

    final_exist_threshold: float | None = None
    mask_threshold: float | None = None
    native_chunk_voxels: int = 262_144
    mask_crop_margin_dref: float = 1.5
    unmatched_mask_crop_radius_dref: float = 2.0

    scene_float_dtype: str = "float16"
    dense_float_dtype: str = "float16"
    mask_float_dtype: str = "float16"

    capture_feature_norm_volumes: bool = False
    capture_channel_stats: bool = True

    @classmethod
    def light(cls, **overrides) -> "DebugConfig":
        return replace(cls(), **overrides)

    @classmethod
    def deep(cls, **overrides) -> "DebugConfig":
        return replace(
            cls(
                capture_dense_outputs=True,
                capture_scene_arrays=True,
                capture_native_masks=True,
            ),
            **overrides,
        )

    def with_queries(self, indices: Iterable[int]) -> "DebugConfig":
        return replace(
            self,
            selected_query_indices=tuple(int(v) for v in indices),
        )

    def validate(self) -> None:
        if self.amp_dtype not in {"fp16", "bf16", "none"}:
            raise ValueError("amp_dtype must be one of {'fp16', 'bf16', 'none'}")
        if self.max_selected_queries < 0:
            raise ValueError("max_selected_queries must be non-negative")
        if self.native_chunk_voxels <= 0:
            raise ValueError("native_chunk_voxels must be positive")
        if self.mask_crop_margin_dref < 0:
            raise ValueError("mask_crop_margin_dref must be non-negative")
        if self.unmatched_mask_crop_radius_dref <= 0:
            raise ValueError("unmatched_mask_crop_radius_dref must be positive")
