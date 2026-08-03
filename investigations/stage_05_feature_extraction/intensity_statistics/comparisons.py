"""Paired method agreement and association-discrimination analysis."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .statistics import normalized_relative_difference, paired_summary


def compare_methods_to_raw(
    cell_stats: pd.DataFrame,
    *,
    features: tuple[str, ...] = (
        "mean",
        "median",
        "variance",
        "std",
        "max",
        "p95",
        "iqr",
        "mad",
        "cv",
        "p95_p05_range",
    ),
) -> pd.DataFrame:
    """Compare every method with raw on identical frame/cell masks."""
    keys = ["sample_id", "frame", "cell_id"]
    raw = cell_stats[cell_stats["method"] == "raw"]
    rows: list[dict[str, object]] = []
    for method in sorted(set(cell_stats["method"]) - {"raw"}):
        candidate = cell_stats[cell_stats["method"] == method]
        merged = raw.merge(candidate, on=keys, suffixes=("_raw", "_candidate"))
        for feature in features:
            left = f"{feature}_raw"
            right = f"{feature}_candidate"
            if left not in merged.columns or right not in merged.columns:
                continue
            rows.append(
                {
                    "reference_method": "raw",
                    "candidate_method": method,
                    "feature": feature,
                    **paired_summary(
                        merged[left].to_numpy(),
                        merged[right].to_numpy(),
                    ),
                }
            )
    return pd.DataFrame(rows)


def _feature_cost(first: pd.Series, second: pd.Series, features: tuple[str, ...]) -> float:
    costs = [
        normalized_relative_difference(first[name], second[name])
        for name in features
        if name in first.index and name in second.index
    ]
    return float(np.mean(costs)) if costs else math.nan


def compute_association_pairs(
    tracked_cell_stats: pd.DataFrame,
    tracks: pd.DataFrame,
    selected_track_ids: Iterable[int],
    *,
    voxel_size_zyx_um: tuple[float, float, float],
    radius_um: float,
    negatives_per_positive: int,
    association_features: tuple[str, ...],
) -> pd.DataFrame:
    """Build true consecutive pairs and nearby plausible wrong candidates."""
    selected = set(int(value) for value in selected_track_ids)
    track_rows = tracks[tracks["track_id"].isin(selected)].copy()
    if track_rows.empty:
        return pd.DataFrame()

    voxel_size = np.asarray(voxel_size_zyx_um, dtype=float)
    rows: list[dict[str, object]] = []

    for method, method_data in tracked_cell_stats.groupby("method", sort=True):
        indexed = method_data.set_index(["frame", "cell_id"], drop=False)
        by_frame = {
            int(frame): frame_data
            for frame, frame_data in method_data.groupby("frame", sort=True)
        }

        for track_id, path in track_rows.groupby("track_id", sort=True):
            path = path.sort_values("frame")
            records = list(path.itertuples(index=False))
            for previous, current in zip(records, records[1:]):
                if int(current.frame) != int(previous.frame) + 1:
                    continue
                previous_key = (int(previous.frame), int(previous.cell_id))
                current_key = (int(current.frame), int(current.cell_id))
                if previous_key not in indexed.index or current_key not in indexed.index:
                    continue

                first = indexed.loc[previous_key]
                correct = indexed.loc[current_key]
                if isinstance(first, pd.DataFrame) or isinstance(correct, pd.DataFrame):
                    continue

                first_position = first[
                    ["centroid_z", "centroid_y", "centroid_x"]
                ].to_numpy(dtype=float)
                correct_position = correct[
                    ["centroid_z", "centroid_y", "centroid_x"]
                ].to_numpy(dtype=float)
                correct_distance = float(
                    np.linalg.norm((correct_position - first_position) * voxel_size)
                )
                rows.append(
                    {
                        "method": method,
                        "pair_type": "correct",
                        "track_id": int(track_id),
                        "from_frame": int(previous.frame),
                        "to_frame": int(current.frame),
                        "from_cell_id": int(previous.cell_id),
                        "to_cell_id": int(current.cell_id),
                        "distance_um": correct_distance,
                        "feature_cost": _feature_cost(
                            first, correct, association_features
                        ),
                    }
                )

                candidates = by_frame.get(int(current.frame))
                if candidates is None or candidates.empty:
                    continue
                candidate_positions = candidates[
                    ["centroid_z", "centroid_y", "centroid_x"]
                ].to_numpy(dtype=float)
                distances = np.linalg.norm(
                    (candidate_positions - first_position[None, :]) * voxel_size[None, :],
                    axis=1,
                )
                candidate_table = candidates.copy()
                candidate_table["_distance_um"] = distances
                candidate_table = candidate_table[
                    (candidate_table["cell_id"] != int(current.cell_id))
                    & (candidate_table["_distance_um"] <= float(radius_um))
                ].sort_values("_distance_um").head(int(negatives_per_positive))

                for _, wrong in candidate_table.iterrows():
                    rows.append(
                        {
                            "method": method,
                            "pair_type": "wrong_nearby",
                            "track_id": int(track_id),
                            "from_frame": int(previous.frame),
                            "to_frame": int(current.frame),
                            "from_cell_id": int(previous.cell_id),
                            "to_cell_id": int(wrong["cell_id"]),
                            "distance_um": float(wrong["_distance_um"]),
                            "feature_cost": _feature_cost(
                                first, wrong, association_features
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def summarize_association_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Summarize correct/wrong costs and their rank-separation AUC."""
    if pairs.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for method, group in pairs.groupby("method", sort=True):
        correct = group.loc[
            group["pair_type"] == "correct", "feature_cost"
        ].dropna().to_numpy(dtype=float)
        wrong = group.loc[
            group["pair_type"] == "wrong_nearby", "feature_cost"
        ].dropna().to_numpy(dtype=float)

        for pair_type, values in (("correct", correct), ("wrong_nearby", wrong)):
            if values.size == 0:
                continue
            rows.append(
                {
                    "method": method,
                    "pair_type": pair_type,
                    "pair_count": int(values.size),
                    "mean_cost": float(np.mean(values)),
                    "median_cost": float(np.median(values)),
                    "std_cost": float(np.std(values, ddof=0)),
                    "q25_cost": float(np.percentile(values, 25)),
                    "q75_cost": float(np.percentile(values, 75)),
                    "wrong_greater_than_correct_auc": math.nan,
                    "median_separation": math.nan,
                }
            )

        if correct.size and wrong.size:
            # AUC = P(cost_wrong > cost_correct), with half credit for ties.
            statistic = stats.mannwhitneyu(
                wrong, correct, alternative="two-sided", method="auto"
            ).statistic
            auc = float(statistic / (wrong.size * correct.size))
            rows.append(
                {
                    "method": method,
                    "pair_type": "separation",
                    "pair_count": int(correct.size + wrong.size),
                    "mean_cost": math.nan,
                    "median_cost": math.nan,
                    "std_cost": math.nan,
                    "q25_cost": math.nan,
                    "q75_cost": math.nan,
                    "wrong_greater_than_correct_auc": auc,
                    "median_separation": float(
                        np.median(wrong) - np.median(correct)
                    ),
                }
            )
    return pd.DataFrame(rows)
