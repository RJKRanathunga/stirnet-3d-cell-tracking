"""Combined-volume evidence for candidate merges."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..config import MergeRealConfig


@dataclass(frozen=True)
class VolumeEvidence:
    candidate_volume: float
    volume_a: float
    volume_b: float
    ratio: float
    log_error: float
    broad_match: bool
    strong_match: bool
    score: float


def volume_sum_evidence(
    candidate_volume: float,
    volume_a: float,
    volume_b: float,
    config: MergeRealConfig,
) -> VolumeEvidence:
    values = (float(candidate_volume), float(volume_a), float(volume_b))
    if not all(math.isfinite(v) and v > 0 for v in values):
        return VolumeEvidence(*values, math.nan, math.nan, False, False, 0.0)
    expected = volume_a + volume_b
    ratio = candidate_volume / expected
    log_error = abs(math.log(ratio))
    broad = config.volume_sum_ratio_min <= ratio <= config.volume_sum_ratio_max
    strong = config.strong_volume_sum_ratio_min <= ratio <= config.strong_volume_sum_ratio_max
    # Smoothly favors ratios near one while retaining broad high-recall candidates.
    scale = max(abs(math.log(config.volume_sum_ratio_min)), abs(math.log(config.volume_sum_ratio_max)), 1e-6)
    score = max(0.0, 1.0 - log_error / scale)
    return VolumeEvidence(
        candidate_volume=float(candidate_volume),
        volume_a=float(volume_a),
        volume_b=float(volume_b),
        ratio=float(ratio),
        log_error=float(log_error),
        broad_match=bool(broad),
        strong_match=bool(strong),
        score=float(score),
    )


__all__ = ["VolumeEvidence", "volume_sum_evidence"]
