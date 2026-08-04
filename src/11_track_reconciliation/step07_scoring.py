"""Smooth normalized evidence scoring and deterministic mutual rankings."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .step01_config import CONTINUATION_CANDIDATE_COLUMNS, TrackReconciliationConfig


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
            record.get("volume_log_error"), config.volume_log_score_scale
        ),
        "shape_score": _score(record.get("shape_error"), config.shape_score_scale),
        "intensity_score": _score(
            record.get("intensity_error"), config.intensity_score_scale
        ),
        "stage7_alternative_score": stage7_score,
    }


def _weighted_score(
    row: pd.Series | dict[str, object],
    config: TrackReconciliationConfig,
    *,
    include_uniqueness: bool,
) -> float:
    weighted = (
        ("forward_position_score", config.forward_position_weight),
        ("backward_position_score", config.backward_position_weight),
        ("bidirectional_score", config.bidirectional_weight),
        ("anchor_position_score", config.anchor_position_weight),
        ("neighborhood_score", config.neighborhood_weight),
        ("temporal_gap_score", config.temporal_gap_weight),
        ("volume_score", config.volume_weight),
        ("shape_score", config.shape_weight),
        ("candidate_quality_score", config.candidate_quality_weight),
        ("intensity_score", config.intensity_weight),
        ("stage7_alternative_score", config.stage7_alternative_weight),
    )
    if include_uniqueness:
        weighted += (("uniqueness_score", config.uniqueness_weight),)
    numerator = 0.0
    denominator = 0.0
    for column, weight in weighted:
        try:
            value = float(row[column])
        except (TypeError, ValueError, KeyError):
            continue
        if math.isfinite(value):
            numerator += weight * float(np.clip(value, 0.0, 1.0))
            denominator += weight
    return numerator / denominator if denominator > 0 else 0.0


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
        row = {**record, **_initial_component_scores(record, config)}
        row["uniqueness_score"] = math.nan
        row["_base_score"] = _weighted_score(row, config, include_uniqueness=False)
        enriched.append(row)
    frame = pd.DataFrame(enriched)
    admissible = frame.loc[frame["admissible"].astype(bool)].copy()
    inadmissible = frame.loc[~frame["admissible"].astype(bool)].copy()
    if not admissible.empty:
        admissible = _rank_and_margin(admissible, "_base_score")
        uniqueness = []
        for row in admissible.itertuples(index=False):
            if int(row.source_candidate_count) == 1 and int(row.target_predecessor_count) == 1:
                uniqueness.append(1.0)
            elif bool(row.mutual_best):
                # Mutual rank alone is not uniqueness: a tied or near-tied
                # source must not manufacture its own conservative margin.
                margin = min(float(row.source_score_margin), float(row.target_score_margin))
                uniqueness.append(float(1.0 - math.exp(-margin / 0.10)))
            else:
                uniqueness.append(0.0)
        admissible["uniqueness_score"] = uniqueness
        admissible["continuation_score"] = admissible.apply(
            lambda row: _weighted_score(row, config, include_uniqueness=True), axis=1
        )
        admissible = _rank_and_margin(admissible, "continuation_score")
        admissible["assignment_cost"] = -np.log(
            admissible["continuation_score"].clip(lower=config.probability_floor)
        )
    if not inadmissible.empty:
        inadmissible["continuation_score"] = inadmissible["_base_score"]
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
    frame = pd.concat([admissible, inadmissible], ignore_index=True)
    frame = frame.sort_values(
        ["source_end_frame", "source_track_id", "target_start_frame", "target_track_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return frame.reindex(columns=CONTINUATION_CANDIDATE_COLUMNS)
