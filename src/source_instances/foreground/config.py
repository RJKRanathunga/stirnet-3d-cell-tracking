"""Configuration for canonical foreground masking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaskingConfig:
    threshold_mode: str = "otsu"
    threshold_multiplier: float = 1.0
    threshold_offset: float = 0.0

    def __post_init__(self) -> None:
        if self.threshold_mode != "otsu":
            raise ValueError("threshold_mode must currently be 'otsu'")


DEFAULT_MASKING_CONFIG = MaskingConfig()
