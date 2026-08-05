"""Deterministic candidate-local reliability policy for small cells in Stage 11."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Mapping

from .step01_config import TrackReconciliationConfig


EVIDENCE_NAMES = (
    "forward_position",
    "backward_position",
    "bidirectional",
    "anchor_position",
    "neighborhood",
    "temporal_gap",
    "uniqueness",
    "volume",
    "shape",
    "intensity",
    "candidate_quality",
    "stage7_alternative",
)


def _finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _log_interpolate(value: float, left: float, right: float, a: float, b: float) -> float:
    """Interpolate in log-volume space, including deterministic endpoint handling."""

    if value <= left:
        return float(a)
    if value >= right:
        return float(b)
    fraction = (math.log(value) - math.log(left)) / (math.log(right) - math.log(left))
    return float(a + fraction * (b - a))


def effective_pair_volume(source_volume: object, target_volume: object) -> float:
    """Return the smaller positive finite endpoint reference volume."""

    source = _finite(source_volume)
    target = _finite(target_volume)
    if source <= 0 or target <= 0:
        return math.nan
    return min(source, target)


def classify_small_cell_regime(
    volume: object,
    config: TrackReconciliationConfig,
) -> str:
    """Classify a pair for diagnostics without independently accepting it."""

    value = _finite(volume)
    if not math.isfinite(value):
        return "unknown"
    if value <= config.extremely_small_volume_threshold:
        return "extremely_small"
    if value <= config.small_cell_volume_threshold:
        return "small"
    if value <= config.small_cell_transition_volume_threshold:
        return "transition"
    return "normal"


def volume_log_score_scale(
    volume: object,
    config: TrackReconciliationConfig,
) -> float:
    """Return the calibrated log-error scale, smoothly interpolated by log volume."""

    value = _finite(volume)
    if (
        not config.small_cell_mode_enabled
        or not math.isfinite(value)
        or value >= config.small_cell_normal_volume_threshold
    ):
        return float(config.volume_log_score_scale)
    knots = (
        (75.0, config.small_cell_volume_log_scale_at_75),
        (150.0, config.small_cell_volume_log_scale_at_150),
        (250.0, config.small_cell_volume_log_scale_at_250),
        (400.0, config.small_cell_volume_log_scale_at_400),
        (config.small_cell_normal_volume_threshold, config.volume_log_score_scale),
    )
    if value <= knots[0][0]:
        return float(knots[0][1])
    for (left_volume, left_scale), (right_volume, right_scale) in zip(knots, knots[1:]):
        if value <= right_volume:
            return _log_interpolate(
                value, left_volume, right_volume, left_scale, right_scale
            )
    return float(config.volume_log_score_scale)


def evidence_weight_multipliers(
    volume: object,
    config: TrackReconciliationConfig,
) -> dict[str, float]:
    """Interpolate local weight multipliers without mutating global configuration."""

    value = _finite(volume)
    normal = {name: 1.0 for name in EVIDENCE_NAMES}
    if (
        not config.small_cell_mode_enabled
        or not math.isfinite(value)
        or value >= config.small_cell_normal_volume_threshold
    ):
        return normal
    extremely_small = asdict(config.extremely_small_weight_multipliers)
    small = asdict(config.small_cell_weight_multipliers)
    if value <= config.extremely_small_volume_threshold:
        return {name: float(extremely_small[name]) for name in EVIDENCE_NAMES}
    if value <= config.small_cell_volume_threshold:
        return {
            name: _log_interpolate(
                value,
                config.extremely_small_volume_threshold,
                config.small_cell_volume_threshold,
                float(extremely_small[name]),
                float(small[name]),
            )
            for name in EVIDENCE_NAMES
        }
    return {
        name: _log_interpolate(
            value,
            config.small_cell_volume_threshold,
            config.small_cell_normal_volume_threshold,
            float(small[name]),
            1.0,
        )
        for name in EVIDENCE_NAMES
    }


def small_cell_mode_applied(volume: object, config: TrackReconciliationConfig) -> bool:
    value = _finite(volume)
    return bool(
        config.small_cell_mode_enabled
        and math.isfinite(value)
        and value < config.small_cell_normal_volume_threshold
    )


def _weighted_mean(values: Mapping[str, object], weights: Mapping[str, float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for name, weight in weights.items():
        value = _finite(values.get(name))
        if math.isfinite(value) and weight > 0:
            numerator += float(weight) * value
            denominator += float(weight)
    return numerator / denominator if denominator > 0 else math.nan


def size_aware_intensity_error(
    component_errors: Mapping[str, object],
    normal_error: object,
    volume: object,
    config: TrackReconciliationConfig,
) -> float:
    """Prefer intensity location and suppress dispersion/sum for small cells."""

    value = _finite(volume)
    baseline = _finite(normal_error)
    if (
        not config.small_cell_mode_enabled
        or not math.isfinite(value)
        or value >= config.small_cell_normal_volume_threshold
    ):
        return baseline
    small_weights = {
        "intensity_mean_error": config.small_cell_intensity_mean_weight,
        "intensity_median_error": config.small_cell_intensity_median_weight,
        "intensity_std_error": config.small_cell_intensity_std_weight,
        "intensity_iqr_error": config.small_cell_intensity_iqr_weight,
        "intensity_cv_error": config.small_cell_intensity_cv_weight,
        "intensity_sum_error": config.small_cell_intensity_sum_weight,
    }
    small_error = _weighted_mean(component_errors, small_weights)
    if not math.isfinite(small_error):
        return (
            math.nan
            if value <= config.small_cell_volume_threshold
            else baseline
        )
    if value <= config.small_cell_volume_threshold or not math.isfinite(baseline):
        return small_error
    return _log_interpolate(
        value,
        config.small_cell_volume_threshold,
        config.small_cell_normal_volume_threshold,
        small_error,
        baseline,
    )


def support_diagnostics(
    row: Mapping[str, object],
    config: TrackReconciliationConfig,
) -> dict[str, object]:
    """Calculate the auditable evidence-supported small-cell acceptance predicate."""

    volume = _finite(row.get("effective_pair_volume"))
    forward = _finite(row.get("forward_error_um"))
    backward = _finite(row.get("backward_error_um"))
    anchor = _finite(row.get("anchor_prediction_error_um"))
    neighborhood = _finite(row.get("neighborhood_distance_error_um"))
    source_margin = _finite(row.get("source_score_margin"))
    target_margin = _finite(row.get("target_score_margin"))
    strong_forward = (
        math.isfinite(forward) and forward <= config.small_cell_forward_error_um
    )
    strong_backward = bool(row.get("backward_prediction_available", False)) and (
        math.isfinite(backward) and backward <= config.small_cell_backward_error_um
    )
    strong_anchor = (
        int(row.get("anchor_count", 0)) >= config.small_cell_minimum_anchor_count
        and math.isfinite(anchor)
        and anchor <= config.small_cell_anchor_error_um
    )
    strong_neighborhood = (
        math.isfinite(neighborhood)
        and neighborhood <= config.small_cell_neighborhood_error_um
    )
    support_count = sum(
        (strong_forward, strong_backward, strong_anchor, strong_neighborhood)
    )
    unique_enough = bool(row.get("mutual_best", False)) and (
        math.isfinite(source_margin)
        and source_margin >= config.small_cell_minimum_source_margin
        and math.isfinite(target_margin)
        and target_margin >= config.small_cell_minimum_target_margin
    )
    special_acceptance = bool(
        config.small_cell_mode_enabled
        and math.isfinite(volume)
        and volume <= config.small_cell_volume_threshold
        and support_count >= config.small_cell_minimum_support_count
        and strong_forward
        and unique_enough
    )
    return {
        "small_cell_strong_forward": strong_forward,
        "small_cell_strong_backward": strong_backward,
        "small_cell_strong_anchor": strong_anchor,
        "small_cell_strong_neighborhood": strong_neighborhood,
        "small_cell_support_count": int(support_count),
        "small_cell_unique_enough": unique_enough,
        "small_cell_special_acceptance": special_acceptance,
    }
