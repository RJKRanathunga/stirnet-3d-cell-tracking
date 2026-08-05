"""Smooth normalized evidence scoring and deterministic mutual rankings."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .step01_config import CONTINUATION_CANDIDATE_COLUMNS, TrackReconciliationConfig
from .small_cell_reliability import (
    evidence_weight_multipliers,
    support_diagnostics,
    volume_log_score_scale,
)


WEIGHTED_COMPONENTS = (
    ("forward_position_score", "forward_position", "forward_position_weight"),
    ("backward_position_score", "backward_position", "backward_position_weight"),
    ("bidirectional_score", "bidirectional", "bidirectional_weight"),
    ("anchor_position_score", "anchor_position", "anchor_position_weight"),
    ("neighborhood_score", "neighborhood", "neighborhood_weight"),
    ("temporal_gap_score", "temporal_gap", "temporal_gap_weight"),
    ("volume_score", "volume", "volume_weight"),
    ("shape_score", "shape", "shape_weight"),
    ("candidate_quality_score", "candidate_quality", "candidate_quality_weight"),
    ("intensity_score", "intensity", "intensity_weight"),
    ("stage7_alternative_score", "stage7_alternative", "stage7_alternative_weight"),
)

MULTIPLIER_COLUMNS = {
    "forward_position": "forward_weight_multiplier",
    "backward_position": "backward_weight_multiplier",
    "bidirectional": "bidirectional_weight_multiplier",
    "anchor_position": "anchor_weight_multiplier",
    "neighborhood": "neighborhood_weight_multiplier",
    "temporal_gap": "temporal_gap_weight_multiplier",
    "uniqueness": "uniqueness_weight_multiplier",
    "volume": "volume_weight_multiplier",
    "shape": "shape_weight_multiplier",
    "intensity": "intensity_weight_multiplier",
    "candidate_quality": "candidate_quality_weight_multiplier",
    "stage7_alternative": "stage7_alternative_weight_multiplier",
}


def _score(error: object, scale: float) -> float:
    try:
        value = float(error)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(value):
        return math.nan
    return float(math.exp(-max(value, 0.0) / scale))


def _initial_component_scores(
    record: dict[str, object],
    config: TrackReconciliationConfig,
    *,
    size_aware: bool,
) -> dict[str, object]:
    forward_scale = config.position_score_scale_um
    try:
        uncertainty = float(record["forward_uncertainty_um"])
        if math.isfinite(uncertainty):
            forward_scale += 0.25 * uncertainty
    except (TypeError, ValueError, KeyError):
        pass
    stage7_score = math.nan
    try:
        probability = float(record["stage7_candidate_probability"])
        if math.isfinite(probability):
            stage7_score = float(np.clip(probability, 0.0, 1.0))
    except (TypeError, ValueError, KeyError):
        pass
    if not math.isfinite(stage7_score):
        try:
            distance = float(record["stage7_candidate_distance_um"])
            if math.isfinite(distance):
                stage7_score = _score(distance, config.position_score_scale_um)
        except (TypeError, ValueError, KeyError):
            pass
    volume_scale = (
        volume_log_score_scale(record.get("effective_pair_volume"), config)
        if size_aware else config.volume_log_score_scale
    )
    intensity_error = record.get(
        "intensity_error" if size_aware else "normal_intensity_error",
        record.get("intensity_error"),
    )
    return {
        "forward_position_score": _score(record.get("forward_error_um"), forward_scale),
        "backward_position_score": _score(
            record.get("backward_error_um"), config.backward_score_scale_um
        ),
        "bidirectional_score": _score(
            record.get("bidirectional_disagreement_um"),
            config.backward_score_scale_um,
        ),
        "anchor_position_score": _score(
            record.get("anchor_prediction_error_um"), config.anchor_score_scale_um
        ),
        "neighborhood_score": _score(
            record.get("neighborhood_distance_error_um"),
            config.neighborhood_score_scale_um,
        ),
        "temporal_gap_score": float(
            config.gap_decay ** max(int(record["gap_frames"]) - 1, 0)
        ),
        "volume_score": _score(
            record.get("volume_log_error"), volume_scale
        ),
        "shape_score": _score(record.get("shape_error"), config.shape_score_scale),
        "intensity_score": _score(
            intensity_error, config.intensity_score_scale
        ),
        "stage7_alternative_score": stage7_score,
    }


def _weighted_score(
    row: pd.Series | dict[str, object],
    config: TrackReconciliationConfig,
    *,
    include_uniqueness: bool,
    multipliers: dict[str, float] | None = None,
    component_overrides: dict[str, object] | None = None,
) -> float:
    local_multipliers = multipliers or {
        name: 1.0 for _, name, _ in WEIGHTED_COMPONENTS
    }
    weighted = tuple(
        (
            column,
            float(getattr(config, weight_name)) * float(local_multipliers[name]),
        )
        for column, name, weight_name in WEIGHTED_COMPONENTS
    )
    if include_uniqueness:
        weighted += ((
            "uniqueness_score",
            config.uniqueness_weight * float(local_multipliers.get("uniqueness", 1.0)),
        ),)
    numerator = 0.0
    denominator = 0.0
    for column, weight in weighted:
        try:
            value = float(
                component_overrides[column]
                if component_overrides is not None and column in component_overrides
                else row[column]
            )
        except (TypeError, ValueError, KeyError):
            continue
        if math.isfinite(value):
            numerator += weight * float(np.clip(value, 0.0, 1.0))
            denominator += weight
    return numerator / denominator if denominator > 0 else 0.0


def _uniqueness_values(frame: pd.DataFrame) -> list[float]:
    values: list[float] = []
    for row in frame.itertuples(index=False):
        if int(row.source_candidate_count) == 1 and int(row.target_predecessor_count) == 1:
            values.append(1.0)
        elif bool(row.mutual_best):
            margin = min(float(row.source_score_margin), float(row.target_score_margin))
            values.append(float(1.0 - math.exp(-margin / 0.10)))
        else:
            values.append(0.0)
    return values


def _rank_and_margin(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    result = frame.copy()
    source_counts = result.groupby("source_track_id")["candidate_id"].transform("count")
    target_counts = result.groupby("target_track_id")["candidate_id"].transform("count")
    result["source_candidate_count"] = source_counts.astype(int)
    result["target_predecessor_count"] = target_counts.astype(int)

    result["source_rank"] = 0
    result["target_rank"] = 0
    result["source_runner_up_score"] = 0.0
    result["target_runner_up_score"] = 0.0
    for _, indices in result.groupby("source_track_id", sort=True).groups.items():
        ordered = result.loc[list(indices)].sort_values(
            [score_column, "target_start_frame", "target_track_id", "candidate_id"],
            ascending=[False, True, True, True], kind="mergesort",
        )
        scores = ordered[score_column].astype(float).tolist()
        runner_up = scores[1] if len(scores) > 1 else 0.0
        for rank, index in enumerate(ordered.index, start=1):
            result.at[index, "source_rank"] = rank
            result.at[index, "source_runner_up_score"] = runner_up
    for _, indices in result.groupby("target_track_id", sort=True).groups.items():
        ordered = result.loc[list(indices)].sort_values(
            [score_column, "source_end_frame", "source_track_id", "candidate_id"],
            ascending=[False, True, True, True], kind="mergesort",
        )
        scores = ordered[score_column].astype(float).tolist()
        runner_up = scores[1] if len(scores) > 1 else 0.0
        for rank, index in enumerate(ordered.index, start=1):
            result.at[index, "target_rank"] = rank
            result.at[index, "target_runner_up_score"] = runner_up
    result["source_score_margin"] = (
        result[score_column] - result["source_runner_up_score"]
    ).clip(lower=0.0)
    result["target_score_margin"] = (
        result[score_column] - result["target_runner_up_score"]
    ).clip(lower=0.0)
    result["mutual_best"] = (
        (result["source_rank"] == 1) & (result["target_rank"] == 1)
    )
    return result


def score_candidates(
    records: list[dict[str, object]],
    config: TrackReconciliationConfig,
) -> pd.DataFrame:
    """Score candidate evidence while omitting unavailable components."""

    if not records:
        return pd.DataFrame(columns=CONTINUATION_CANDIDATE_COLUMNS)
    enriched = []
    for record in records:
        size_components = _initial_component_scores(record, config, size_aware=True)
        normal_components = _initial_component_scores(record, config, size_aware=False)
        multipliers = evidence_weight_multipliers(
            record.get("effective_pair_volume"), config
        )
        row = {**record, **size_components}
        row["volume_log_score_scale_used"] = volume_log_score_scale(
            record.get("effective_pair_volume"), config
        )
        for name, column in MULTIPLIER_COLUMNS.items():
            row[column] = float(multipliers[name])
        row["uniqueness_score"] = math.nan
        row["_size_base_score"] = _weighted_score(
            row,
            config,
            include_uniqueness=False,
            multipliers=multipliers,
        )
        row["_normal_base_score"] = _weighted_score(
            row,
            config,
            include_uniqueness=False,
            component_overrides=normal_components,
        )
        row["_normal_volume_score"] = normal_components["volume_score"]
        row["_normal_intensity_score"] = normal_components["intensity_score"]
        enriched.append(row)
    frame = pd.DataFrame(enriched)
    admissible = frame.loc[frame["admissible"].astype(bool)].copy()
    inadmissible = frame.loc[~frame["admissible"].astype(bool)].copy()
    if not admissible.empty:
        normal_ranked = _rank_and_margin(admissible, "_normal_base_score")
        normal_ranked["_normal_uniqueness_score"] = _uniqueness_values(normal_ranked)
        admissible["_normal_uniqueness_score"] = normal_ranked[
            "_normal_uniqueness_score"
        ]
        admissible["normal_weighted_score"] = admissible.apply(
            lambda row: _weighted_score(
                row,
                config,
                include_uniqueness=True,
                component_overrides={
                    "volume_score": row["_normal_volume_score"],
                    "intensity_score": row["_normal_intensity_score"],
                    "uniqueness_score": row["_normal_uniqueness_score"],
                },
            ),
            axis=1,
        )
        admissible = _rank_and_margin(admissible, "_size_base_score")
        admissible["uniqueness_score"] = _uniqueness_values(admissible)
        admissible["continuation_score"] = admissible.apply(
            lambda row: _weighted_score(
                row,
                config,
                include_uniqueness=True,
                multipliers=evidence_weight_multipliers(
                    row["effective_pair_volume"], config
                ),
            ),
            axis=1,
        )
        admissible = _rank_and_margin(admissible, "continuation_score")
        support = admissible.apply(
            lambda row: pd.Series(support_diagnostics(row, config)), axis=1
        )
        for column in support.columns:
            admissible[column] = support[column]
        admissible["size_aware_weighted_score"] = admissible["continuation_score"]
        admissible["size_aware_score_delta"] = (
            admissible["size_aware_weighted_score"]
            - admissible["normal_weighted_score"]
        )
        admissible["assignment_cost"] = -np.log(
            admissible["continuation_score"].clip(lower=config.probability_floor)
        )
    if not inadmissible.empty:
        inadmissible["continuation_score"] = inadmissible["_size_base_score"]
        inadmissible["normal_weighted_score"] = inadmissible["_normal_base_score"]
        inadmissible["size_aware_weighted_score"] = inadmissible["continuation_score"]
        inadmissible["size_aware_score_delta"] = (
            inadmissible["size_aware_weighted_score"]
            - inadmissible["normal_weighted_score"]
        )
        inadmissible["assignment_cost"] = config.invalid_assignment_cost
        inadmissible["source_candidate_count"] = 0
        inadmissible["target_predecessor_count"] = 0
        inadmissible["source_rank"] = 0
        inadmissible["target_rank"] = 0
        inadmissible["source_runner_up_score"] = math.nan
        inadmissible["target_runner_up_score"] = math.nan
        inadmissible["source_score_margin"] = math.nan
        inadmissible["target_score_margin"] = math.nan
        inadmissible["mutual_best"] = False
        support = inadmissible.apply(
            lambda row: pd.Series(support_diagnostics(row, config)), axis=1
        )
        for column in support.columns:
            inadmissible[column] = support[column]
    frame = pd.concat([admissible, inadmissible], ignore_index=True)
    frame = frame.sort_values(
        ["source_end_frame", "source_track_id", "target_start_frame", "target_track_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return frame.reindex(columns=CONTINUATION_CANDIDATE_COLUMNS)
