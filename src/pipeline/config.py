"""Configuration for processing every sample in a full Biohub dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .stage_registry import BATCH_STAGE_ORDER


@dataclass(frozen=True)
class FullPipelineConfig:
    dataset_root: Path
    output_root: Path
    train_subdirectory: str = "train"
    sample_ids: tuple[str, ...] = ()

    start_stage: int = 6
    end_stage: int = 11
    resume: bool = False
    continue_on_error: bool = True

    # These defaults intentionally reproduce the current Stage 7 notebook's
    # active apply-mode windowed 4D research configuration.
    graph_mode: str = "apply"
    graph_algorithm: str = "windowed_4d"
    graph_window_size: int = 7
    graph_maximum_gap_frames: int = 2
    graph_solver_time_limit_seconds: float = 30.0
    graph_iterative_fallback_iterations: int = 5
    graph_save_debug_npz: bool = False

    # Current Stage 11 notebook policy.
    reconciliation_policy: str = "submission"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_root",
            Path(self.dataset_root).expanduser().resolve(),
        )
        object.__setattr__(
            self,
            "output_root",
            Path(self.output_root).expanduser().resolve(),
        )
        object.__setattr__(
            self,
            "sample_ids",
            tuple(str(value) for value in self.sample_ids),
        )

        if not self.train_subdirectory.strip():
            raise ValueError("train_subdirectory must be nonempty")
        if not 1 <= int(self.start_stage) <= 11:
            raise ValueError("start_stage must be between 1 and 11")
        if not 1 <= int(self.end_stage) <= 11:
            raise ValueError("end_stage must be between 1 and 11")
        if int(self.start_stage) > int(self.end_stage):
            raise ValueError("start_stage cannot be greater than end_stage")
        if self.graph_mode not in {"disabled", "shadow", "apply"}:
            raise ValueError("graph_mode must be disabled, shadow, or apply")
        if self.graph_algorithm not in {"pairwise", "windowed_4d"}:
            raise ValueError("graph_algorithm must be pairwise or windowed_4d")
        if int(self.graph_window_size) < 3 or int(self.graph_window_size) % 2 == 0:
            raise ValueError("graph_window_size must be an odd integer of at least 3")
        if int(self.graph_maximum_gap_frames) < 1:
            raise ValueError("graph_maximum_gap_frames must be at least 1")
        if float(self.graph_solver_time_limit_seconds) <= 0:
            raise ValueError("graph_solver_time_limit_seconds must be positive")
        if int(self.graph_iterative_fallback_iterations) < 1:
            raise ValueError("graph_iterative_fallback_iterations must be at least 1")

        if not self.selected_stages:
            if self.start_stage <= 9 <= self.end_stage:
                raise ValueError(
                    "The requested range contains no batch processing stage. "
                    "Stage 9 is visualization-only and is intentionally skipped."
                )
            raise ValueError(
                "The requested range contains no executable batch stage. "
                f"Batch stages are {BATCH_STAGE_ORDER}."
            )

    @property
    def selected_stages(self) -> tuple[int, ...]:
        return tuple(
            stage
            for stage in BATCH_STAGE_ORDER
            if int(self.start_stage) <= stage <= int(self.end_stage)
        )


__all__ = ["FullPipelineConfig"]
