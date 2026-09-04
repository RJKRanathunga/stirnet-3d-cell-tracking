"""Configuration and result contracts for production Trackastra tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GlobalMotionConfig:
    enabled: bool = True
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0)
    minimum_pairs: int = 10
    mad_scale: float = 4.0
    minimum_residual_gate_physical: float = 2.0

    def __post_init__(self) -> None:
        spacing = tuple(float(v) for v in self.voxel_size_zyx)
        if len(spacing) != 3 or any(not np.isfinite(v) or v <= 0.0 for v in spacing):
            raise ValueError("voxel_size_zyx must contain three positive finite values")
        if int(self.minimum_pairs) < 1:
            raise ValueError("minimum_pairs must be >= 1")
        if not np.isfinite(float(self.mad_scale)) or float(self.mad_scale) < 0.0:
            raise ValueError("mad_scale must be finite and >= 0")
        gate = float(self.minimum_residual_gate_physical)
        if not np.isfinite(gate) or gate < 0.0:
            raise ValueError("minimum_residual_gate_physical must be finite and >= 0")


@dataclass(frozen=True)
class TrackastraConfig:
    model_name: str = "ctc"
    mode: str = "greedy"
    device: str = "cuda"
    batch_size: int | None = None
    global_motion: GlobalMotionConfig = field(default_factory=GlobalMotionConfig)

    def __post_init__(self) -> None:
        if str(self.mode) not in {"greedy", "greedy_nodiv", "ilp"}:
            raise ValueError("mode must be one of {'greedy', 'greedy_nodiv', 'ilp'}")
        if self.batch_size is not None and int(self.batch_size) < 1:
            raise ValueError("batch_size must be >= 1 when provided")


@dataclass(frozen=True)
class GlobalMotionEstimate:
    pairwise_float_zyx: np.ndarray
    cumulative_float_zyx: np.ndarray
    align_int_zyx: np.ndarray
    placement_zyx: np.ndarray
    canvas_shape_zyx: tuple[int, int, int]
    pair_counts: np.ndarray
    inlier_counts: np.ndarray
    gate_physical: np.ndarray
    median_residual_physical: np.ndarray
    p90_residual_physical: np.ndarray

    def records(self) -> list[dict[str, float | int]]:
        rows: list[dict[str, float | int]] = []
        for t in range(int(self.pairwise_float_zyx.shape[0])):
            to = t + 1
            pair = self.pairwise_float_zyx[t]
            cumulative = self.cumulative_float_zyx[to]
            align = self.align_int_zyx[to]
            placement = self.placement_zyx[to]
            rows.append({
                "frame_from": int(t),
                "frame_to": int(to),
                "pair_dz": float(pair[0]),
                "pair_dy": float(pair[1]),
                "pair_dx": float(pair[2]),
                "pairs_total": int(self.pair_counts[t]),
                "pairs_inlier": int(self.inlier_counts[t]),
                "gate_physical": float(self.gate_physical[t]),
                "median_residual_physical": float(self.median_residual_physical[t]),
                "p90_residual_physical": float(self.p90_residual_physical[t]),
                "cumulative_dz_at_to": float(cumulative[0]),
                "cumulative_dy_at_to": float(cumulative[1]),
                "cumulative_dx_at_to": float(cumulative[2]),
                "align_z_at_to": int(align[0]),
                "align_y_at_to": int(align[1]),
                "align_x_at_to": int(align[2]),
                "placement_z_at_to": int(placement[0]),
                "placement_y_at_to": int(placement[1]),
                "placement_x_at_to": int(placement[2]),
            })
        return rows

    def summary(self) -> dict[str, object]:
        max_abs = np.max(np.abs(self.align_int_zyx), axis=0)
        return {
            "canvas_shape_zyx": [int(v) for v in self.canvas_shape_zyx],
            "maximum_absolute_alignment_zyx": [int(v) for v in max_abs.tolist()],
            "minimum_pairs_observed": int(np.min(self.pair_counts)) if self.pair_counts.size else 0,
            "minimum_inlier_pairs": int(np.min(self.inlier_counts)) if self.inlier_counts.size else 0,
            "integer_translation": True,
            "wraparound": False,
        }


@dataclass
class TrackastraResult:
    graph: Any
    napari_tracks: np.ndarray
    napari_graph: dict[int, int | list[int]]
    seconds: float
    summary: dict[str, object]
    global_motion: GlobalMotionEstimate | None = None
    pass_summaries: tuple[dict[str, object], ...] = ()


__all__ = [
    "GlobalMotionConfig",
    "GlobalMotionEstimate",
    "TrackastraConfig",
    "TrackastraResult",
]
