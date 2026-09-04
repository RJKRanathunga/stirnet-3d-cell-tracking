"""Pure numeric scoring and classification helpers for Stage 10."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
import pandas as pd

from .step01_config import (
    CONFIRMED,
    PROBABLE,
    REJECTED,
    CellLineageConfig,
)


def safe_ratio(numerator, denominator) -> float:
    """Return a finite ratio or NaN for invalid division."""

    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        return math.nan
    result = numerator / denominator
    return float(result) if math.isfinite(result) else math.nan


def relative_error(value, reference) -> float:
    """Return absolute relative error with safe invalid handling."""

    try:
        difference = float(value) - float(reference)
    except (TypeError, ValueError):
        return math.nan
    ratio = safe_ratio(difference, reference)
    return abs(ratio) if math.isfinite(ratio) else math.nan


def exponential_score(error, scale) -> float:
    """Map a nonnegative error to a clipped exponential score."""

    try:
        error = float(error)
        scale = float(scale)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(error) or not math.isfinite(scale) or scale <= 0:
        return math.nan
    return float(np.clip(np.exp(-max(error, 0.0) / scale), 0.0, 1.0))


def weighted_score(
    components: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """Average finite components while omitting missing evidence."""

    numerator = 0.0
    denominator = 0.0
    for name, value in components.items():
        weight = float(weights.get(name, 0.0))
        if weight < 0 or not math.isfinite(weight):
            raise ValueError(f"Invalid weight for component {name!r}: {weight}")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if weight == 0 or not math.isfinite(numeric):
            continue
        numerator += weight * float(np.clip(numeric, 0.0, 1.0))
        denominator += weight
    if denominator <= 0:
        raise ValueError("No finite score component with positive weight is available")
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def spatial_score(record: Mapping[str, object], config: CellLineageConfig) -> float:
    individual = np.nanmean([
        exponential_score(
            record["child_a_distance_um"],
            config.individual_child_distance_score_scale_um,
        ),
        exponential_score(
            record["child_b_distance_um"],
            config.individual_child_distance_score_scale_um,
        ),
    ])
    centroid = exponential_score(
        record["weighted_centroid_error_um"],
        config.weighted_centroid_score_scale_um,
    )
    return weighted_score(
        {"individual": float(individual), "centroid": centroid},
        {"individual": 0.40, "centroid": 0.60},
    )


def divergence_score(record: Mapping[str, object], config: CellLineageConfig) -> float:
    gain = record.get("separation_gain_um", math.nan)
    slope = record.get("separation_slope_um_per_frame", math.nan)
    try:
        gain = float(gain)
        slope = float(slope)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(gain) or not math.isfinite(slope):
        return math.nan

    def logistic(value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0))))

    gain_support = logistic(
        (gain + config.divergence_negative_tolerance_um)
        / config.divergence_gain_score_scale_um
    )
    slope_support = logistic(
        (slope + config.divergence_negative_tolerance_um / max(config.future_child_horizon, 1))
        / config.divergence_slope_score_scale_um_per_frame
    )
    return float(np.clip(0.65 * gain_support + 0.35 * slope_support, 0.0, 1.0))


def _support_from_ratio(ratio: object, scale: float) -> float:
    try:
        numeric = float(ratio)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(numeric) or numeric <= 0:
        return math.nan
    transformed = math.log(numeric) / scale
    return float(np.clip(0.5 + 0.5 * np.tanh(transformed), 0.0, 1.0))


def parent_intensity_support(
    record: Mapping[str, object],
    config: CellLineageConfig,
) -> float:
    values = [
        _support_from_ratio(
            record.get("parent_final_integrated_intensity_ratio", math.nan),
            config.parent_intensity_log_ratio_scale,
        ),
        _support_from_ratio(
            record.get("parent_final_core_intensity_ratio", math.nan),
            config.parent_intensity_log_ratio_scale,
        ),
        _support_from_ratio(
            record.get("parent_final_background_corrected_ratio", math.nan),
            config.parent_intensity_log_ratio_scale,
        ),
    ]
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def child_brightening_support(
    record: Mapping[str, object],
    config: CellLineageConfig,
) -> float:
    return _support_from_ratio(
        record.get("child_delayed_max_intensity_ratio", math.nan),
        config.child_brightening_log_ratio_scale,
    )


def _shape_similarity(
    parent: pd.Series,
    child: pd.Series,
    config: CellLineageConfig,
) -> float:
    errors = []
    for feature in ("equivalent_radius", "axis_major", "axis_middle", "axis_minor"):
        if feature not in parent.index or feature not in child.index:
            continue
        try:
            error = relative_error(float(child[feature]), float(parent[feature]))
        except (TypeError, ValueError):
            continue
        if math.isfinite(error):
            errors.append(error)
    if not errors:
        return math.nan
    return exponential_score(float(np.mean(errors)), config.continuation_shape_score_scale)


def continuation_scores(
    parent: pd.Series,
    child: pd.Series,
    distance_um: float,
    config: CellLineageConfig,
) -> dict[str, float]:
    """Score the ordinary parent-to-one-child continuation alternative."""

    position = exponential_score(distance_um, config.continuation_position_score_scale_um)
    volume = exponential_score(
        relative_error(float(child["volume"]), float(parent["volume"])),
        config.continuation_volume_score_scale,
    )
    shape = _shape_similarity(parent, child, config)
    total = weighted_score(
        {"position": position, "volume": volume, "shape": shape},
        {"position": 0.55, "volume": 0.35, "shape": 0.10},
    )
    return {
        "position": position,
        "volume": volume,
        "shape": shape,
        "score": total,
    }


def hard_gate_reason(record: Mapping[str, object], config: CellLineageConfig) -> str:
    """Return the first deterministic structural rejection reason."""

    if record.get("_hard_rejection_reason"):
        return str(record["_hard_rejection_reason"])
    if bool(record.get("transition_overlaps_virtual_merge", False)):
        return "virtual_observation_overlap"
    volume_error = float(record.get("combined_volume_relative_error", math.nan))
    if not math.isfinite(volume_error) or volume_error > config.maximum_combined_volume_relative_error:
        return "combined_volume_mismatch"
    centroid_error = float(record.get("weighted_centroid_error_um", math.nan))
    if not math.isfinite(centroid_error) or centroid_error > config.maximum_weighted_centroid_error_um:
        return "weighted_centroid_mismatch"
    separation = float(record.get("birth_separation_um", math.nan))
    if not math.isfinite(separation) or separation > config.maximum_birth_separation_um:
        return "birth_separation_too_large"
    if not bool(record.get("birth_masks_available", False)):
        return "missing_birth_mask"
    if record.get("tiny_fragment_child", ""):
        return "tiny_child_fragment"
    return ""


def score_candidate(
    record: dict[str, object],
    parent_endpoint: pd.Series,
    child_a_start: pd.Series,
    child_b_start: pd.Series,
    config: CellLineageConfig,
) -> dict[str, object]:
    """Attach component, continuation, overall, and preliminary decision scores."""

    record = dict(record)
    record["spatial_score"] = spatial_score(record, config)
    record["combined_volume_score"] = exponential_score(
        record["combined_volume_relative_error"], config.combined_volume_score_scale
    )
    record["persistence_score"] = float(np.clip(
        float(record["minimum_child_observation_count"])
        / config.minimum_child_observations,
        0.0,
        1.0,
    ))
    record["divergence_score"] = divergence_score(record, config)
    record["parent_intensity_score"] = parent_intensity_support(record, config)
    record["child_brightening_score"] = child_brightening_support(record, config)

    preferred_a = bool(
        float(record["child_a_volume_voxels"]) >= config.preferred_minimum_child_voxels
        and float(record["child_a_volume_fraction"]) >= config.preferred_minimum_child_volume_fraction
    )
    preferred_b = bool(
        float(record["child_b_volume_voxels"]) >= config.preferred_minimum_child_voxels
        and float(record["child_b_volume_fraction"]) >= config.preferred_minimum_child_volume_fraction
    )
    record["child_a_preferred_quality_pass"] = preferred_a
    record["child_b_preferred_quality_pass"] = preferred_b
    record["artifact_penalty"] = float(np.clip(
        config.preferred_fragment_penalty_per_child * ((not preferred_a) + (not preferred_b)),
        0.0,
        1.0,
    ))

    continuation_a = continuation_scores(
        parent_endpoint, child_a_start, float(record["child_a_distance_um"]), config
    )
    continuation_b = continuation_scores(
        parent_endpoint, child_b_start, float(record["child_b_distance_um"]), config
    )
    for suffix, values in (("a", continuation_a), ("b", continuation_b)):
        record[f"continuation_position_score_{suffix}"] = values["position"]
        record[f"continuation_volume_score_{suffix}"] = values["volume"]
        record[f"continuation_shape_score_{suffix}"] = values["shape"]
        record[f"continuation_score_{suffix}"] = values["score"]
    record["best_continuation_score"] = max(
        float(continuation_a["score"]), float(continuation_b["score"])
    )

    primary_names = ("spatial_score", "combined_volume_score", "persistence_score")
    if not all(math.isfinite(float(record[name])) for name in primary_names):
        raise ValueError("A primary division score component is unavailable")
    components = {
        "spatial": float(record["spatial_score"]),
        "combined_volume": float(record["combined_volume_score"]),
        "persistence": float(record["persistence_score"]),
        "divergence": float(record["divergence_score"]),
        "parent_intensity": float(record["parent_intensity_score"]),
        "child_brightening": float(record["child_brightening_score"]),
    }
    weights = {
        "spatial": config.spatial_weight,
        "combined_volume": config.combined_volume_weight,
        "persistence": config.persistence_weight,
        "divergence": config.divergence_weight,
        "parent_intensity": config.parent_intensity_weight,
        "child_brightening": config.child_brightening_weight,
    }
    base = weighted_score(components, weights)
    record["division_score"] = float(np.clip(
        base - float(record["artifact_penalty"]), 0.0, 1.0
    ))
    record["division_margin"] = (
        float(record["division_score"]) - float(record["best_continuation_score"])
    )

    secondary: list[str] = []
    if not preferred_a or not preferred_b:
        secondary.append("preferred_fragment_limit")
    if bool(record["future_window_truncated"]):
        secondary.append("future_window_truncated")
    if not bool(record["both_children_persist"]):
        secondary.append("insufficient_persistence")
    if not math.isfinite(float(record["divergence_score"])):
        secondary.append("missing_divergence_evidence")
    if (
        not math.isfinite(float(record["parent_intensity_score"]))
        or not math.isfinite(float(record["child_brightening_score"]))
    ):
        secondary.append("missing_optional_intensity")

    reason = hard_gate_reason(record, config)
    if reason:
        decision = REJECTED
    elif not bool(record["both_children_persist"]) and not bool(record["future_window_truncated"]):
        decision = REJECTED
        reason = "insufficient_persistence"
    elif (
        bool(record["both_children_persist"])
        and float(record["division_score"]) >= config.confirmed_minimum_score
        and float(record["division_margin"]) >= config.confirmed_minimum_margin
    ):
        decision = CONFIRMED
    elif (
        float(record["division_score"]) >= config.probable_minimum_score
        and float(record["division_margin"]) >= config.probable_minimum_margin
    ):
        decision = PROBABLE
    else:
        decision = REJECTED
        reason = (
            "continuation_hypothesis_stronger"
            if float(record["division_margin"]) < config.probable_minimum_margin
            else "score_below_threshold"
        )
    record["preliminary_decision"] = decision
    record["decision"] = decision
    record["rejection_reason"] = reason
    record["secondary_reasons"] = ";".join(dict.fromkeys(secondary))
    record["division_event_id"] = pd.NA
    return record
