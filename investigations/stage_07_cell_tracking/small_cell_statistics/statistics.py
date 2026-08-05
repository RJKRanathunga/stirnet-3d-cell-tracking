"""Pure statistical helpers for small-cell population and track comparisons."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu


IDENTIFIER_COLUMNS = {
    "sample_id", "case_id", "scene_path", "source_tracks_csv", "tracks_source",
    "frame", "cell", "cell_id", "track_id", "production_track_id",
    "manual_track_id", "selection_ordinal",
}
POSITION_COLUMNS = {
    "centroid_z", "centroid_y", "centroid_x", "z", "y", "x",
    "z_min", "y_min", "x_min", "z_max", "y_max", "x_max",
}


def infer_population_features(frame: pd.DataFrame) -> tuple[str, ...]:
    preferred = [
        "analysis_volume_voxels", "volume_voxels", "mask_volume_voxels",
        "equivalent_radius", "extent", "bbox_volume", "bbox_depth",
        "bbox_height", "bbox_width", "axis_major", "axis_middle", "axis_minor",
        "elongation", "flatness", "anisotropy", "solidity", "compactness",
        "intensity_mean", "intensity_median", "intensity_std", "intensity_min",
        "intensity_max", "intensity_q25", "intensity_q75", "intensity_iqr",
        "intensity_cv", "intensity_sum", "intensity_range",
    ]
    numeric = set(frame.select_dtypes(include=[np.number]).columns)
    ordered = [name for name in preferred if name in numeric]
    extras = sorted(numeric - set(ordered) - IDENTIFIER_COLUMNS - POSITION_COLUMNS)
    return tuple(dict.fromkeys([*ordered, *extras]))


def link_manual_trajectories(
    observations: pd.DataFrame,
    *,
    voxel_size_zyx_um: tuple[float, float, float],
    maximum_distance_um: float,
) -> pd.DataFrame:
    """Link manually selected cells across frames using deterministic physical-distance assignment."""
    result_parts: list[pd.DataFrame] = []
    spacing = np.asarray(voxel_size_zyx_um, dtype=float)
    coordinate_columns = next(
        (columns for columns in (("centroid_z", "centroid_y", "centroid_x"), ("z", "y", "x"))
         if set(columns).issubset(observations.columns)),
        None,
    )
    if coordinate_columns is None:
        raise KeyError("Selected observations need centroid_z/y/x or z/y/x coordinates")

    for case_id, case in observations.groupby("case_id", sort=True):
        case = case.sort_values(["frame", "cell_id"], kind="mergesort").copy()
        next_id = 0
        active: dict[int, tuple[int, np.ndarray]] = {}
        assignments: dict[int, int] = {}
        for frame, current in case.groupby("frame", sort=True):
            current_indices = current.index.to_numpy()
            current_positions = current[list(coordinate_columns)].to_numpy(dtype=float) * spacing
            eligible_ids = [
                manual_id for manual_id, (previous_frame, _) in sorted(active.items())
                if int(frame) > previous_frame
            ]
            matched_current: set[int] = set()
            if eligible_ids and len(current_indices):
                previous_positions = np.vstack([active[manual_id][1] for manual_id in eligible_ids])
                distances = np.linalg.norm(
                    previous_positions[:, None, :] - current_positions[None, :, :], axis=2
                )
                rows, cols = linear_sum_assignment(distances)
                for row, col in zip(rows, cols):
                    if float(distances[row, col]) > maximum_distance_um:
                        continue
                    manual_id = eligible_ids[int(row)]
                    index = int(current_indices[int(col)])
                    assignments[index] = manual_id
                    active[manual_id] = (int(frame), current_positions[int(col)])
                    matched_current.add(int(col))
            for position, index in enumerate(current_indices):
                if position in matched_current:
                    continue
                manual_id = next_id
                next_id += 1
                assignments[int(index)] = manual_id
                active[manual_id] = (int(frame), current_positions[position])
        case["manual_track_number"] = [assignments[int(index)] for index in case.index]
        case["manual_track_id"] = [f"{case_id}:manual:{number:03d}" for number in case["manual_track_number"]]
        result_parts.append(case)
    return pd.concat(result_parts, ignore_index=True, sort=False).sort_values(
        ["sample_id", "case_id", "manual_track_id", "frame"], kind="mergesort"
    ).reset_index(drop=True)


def attach_frame_population_comparisons(
    targets: pd.DataFrame,
    cells: pd.DataFrame,
    features: Iterable[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    target_keys_by_case_frame = {
        (str(case_id), str(sample_id), int(frame)): set(group["cell_id"].astype(int))
        for (case_id, sample_id, frame), group in targets.groupby(["case_id", "sample_id", "frame"], sort=False)
    }
    grouped_cells = {
        (str(sample_id), int(frame)): group
        for (sample_id, frame), group in cells.groupby(["sample_id", "frame"], sort=False)
    }
    for target in targets.itertuples(index=False):
        population = grouped_cells[(str(target.sample_id), int(target.frame))]
        excluded = target_keys_by_case_frame[(str(target.case_id), str(target.sample_id), int(target.frame))]
        controls = population.loc[~population["cell_id"].astype(int).isin(excluded)]
        for feature in features:
            if feature not in targets.columns or feature not in controls.columns:
                continue
            try:
                target_value = float(getattr(target, feature))
            except (AttributeError, TypeError, ValueError):
                continue
            values = pd.to_numeric(controls[feature], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not math.isfinite(target_value) or values.size == 0:
                continue
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            robust_scale = 1.4826 * mad
            records.append({
                "case_id": target.case_id,
                "sample_id": target.sample_id,
                "manual_track_id": target.manual_track_id,
                "production_track_id": getattr(target, "production_track_id", np.nan),
                "frame": int(target.frame),
                "cell_id": int(target.cell_id),
                "feature": feature,
                "target_value": target_value,
                "population_count": int(values.size),
                "population_mean": float(np.mean(values)),
                "population_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "population_median": median,
                "population_mad": mad,
                "population_q05": float(np.quantile(values, 0.05)),
                "population_q10": float(np.quantile(values, 0.10)),
                "population_q25": float(np.quantile(values, 0.25)),
                "population_q75": float(np.quantile(values, 0.75)),
                "population_q90": float(np.quantile(values, 0.90)),
                "population_q95": float(np.quantile(values, 0.95)),
                "percentile_rank": float(100.0 * np.mean(values <= target_value)),
                "robust_z": (
                    float((target_value - median) / robust_scale)
                    if robust_scale > 1e-12 else np.nan
                ),
                "ratio_to_population_median": (
                    float(target_value / median) if abs(median) > 1e-12 else np.nan
                ),
            })
    return pd.DataFrame(records)


def _safe_slope(frames: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2 or np.all(frames == frames[0]):
        return np.nan
    return float(np.polyfit(frames.astype(float), values.astype(float), 1)[0])


def track_transition_table(
    observations: pd.DataFrame,
    *,
    track_column: str,
    cohort: str,
    feature: str = "analysis_volume_voxels",
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    group_columns = ["sample_id", track_column]
    for keys, group in observations.groupby(group_columns, sort=True):
        sample_id, track_id = keys
        ordered = group.sort_values(["frame", "cell_id"], kind="mergesort")
        frames = ordered["frame"].to_numpy(dtype=int)
        values = pd.to_numeric(ordered[feature], errors="coerce").to_numpy(dtype=float)
        for index in range(1, len(ordered)):
            first, second = values[index - 1], values[index]
            if not np.isfinite(first) or not np.isfinite(second) or first <= 0 or second <= 0:
                continue
            denominator = 0.5 * (abs(first) + abs(second))
            records.append({
                "cohort": cohort,
                "sample_id": sample_id,
                "track_identity": str(track_id),
                "from_frame": int(frames[index - 1]),
                "to_frame": int(frames[index]),
                "frame_gap": int(frames[index] - frames[index - 1]),
                "adjacent_frames": int(frames[index] - frames[index - 1]) == 1,
                "from_value": float(first),
                "to_value": float(second),
                "signed_change": float(second - first),
                "absolute_change": float(abs(second - first)),
                "symmetric_relative_change": float(abs(second - first) / max(denominator, 1e-12)),
                "log_change": float(abs(math.log(second / first))),
                "ratio": float(max(first, second) / min(first, second)),
            })
    return pd.DataFrame(records)


def summarize_tracks(
    observations: pd.DataFrame,
    *,
    track_column: str,
    cohort: str,
    feature: str = "analysis_volume_voxels",
) -> pd.DataFrame:
    transition = track_transition_table(
        observations, track_column=track_column, cohort=cohort, feature=feature
    )
    transition_groups = {
        (str(sample_id), str(track_id)): group.loc[group["adjacent_frames"].astype(bool)]
        for (sample_id, track_id), group in transition.groupby(["sample_id", "track_identity"], sort=False)
    } if not transition.empty else {}
    records: list[dict[str, object]] = []
    for (sample_id, track_id), group in observations.groupby(["sample_id", track_column], sort=True):
        ordered = group.sort_values(["frame", "cell_id"], kind="mergesort")
        frames = ordered["frame"].to_numpy(dtype=int)
        values = pd.to_numeric(ordered[feature], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        q10, q25, q75, q90 = np.quantile(values, [0.10, 0.25, 0.75, 0.90])
        transitions = transition_groups.get((str(sample_id), str(track_id)), pd.DataFrame())
        duplicate_frames = bool(pd.Series(frames).duplicated().any())
        frame_span = int(frames.max() - frames.min() + 1)
        contiguous = not duplicate_frames and len(frames) == frame_span
        record = {
            "cohort": cohort,
            "sample_id": sample_id,
            "track_identity": str(track_id),
            "observation_count": int(len(ordered)),
            "first_frame": int(frames.min()),
            "last_frame": int(frames.max()),
            "frame_span": frame_span,
            "duplicate_frames": duplicate_frames,
            "contiguous": contiguous,
            "missing_frame_count": int(max(frame_span - len(np.unique(frames)), 0)),
            "volume_min": float(np.min(values)),
            "volume_q10": float(q10),
            "volume_q25": float(q25),
            "volume_median": median,
            "volume_mean": float(np.mean(values)),
            "volume_q75": float(q75),
            "volume_q90": float(q90),
            "volume_max": float(np.max(values)),
            "volume_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "volume_cv": float(np.std(values, ddof=1) / median) if values.size > 1 and median > 0 else 0.0,
            "volume_mad": mad,
            "volume_robust_cv": float(1.4826 * mad / median) if median > 0 else np.nan,
            "volume_iqr_relative": float((q75 - q25) / median) if median > 0 else np.nan,
            "volume_range_relative": float((np.max(values) - np.min(values)) / median) if median > 0 else np.nan,
            "volume_slope_voxels_per_frame": _safe_slope(frames, pd.to_numeric(ordered[feature], errors="coerce").to_numpy(dtype=float)),
        }
        record["volume_slope_fraction_per_frame"] = (
            float(record["volume_slope_voxels_per_frame"] / median)
            if median > 0 and np.isfinite(record["volume_slope_voxels_per_frame"]) else np.nan
        )
        if transitions.empty:
            for name in (
                "adjacent_absolute_change_median", "adjacent_absolute_change_max",
                "adjacent_relative_change_median", "adjacent_relative_change_p90",
                "adjacent_relative_change_max", "adjacent_log_change_median",
                "adjacent_log_change_p90", "adjacent_log_change_max",
            ):
                record[name] = np.nan
            record["adjacent_transition_count"] = 0
        else:
            record.update({
                "adjacent_transition_count": int(len(transitions)),
                "adjacent_absolute_change_median": float(transitions["absolute_change"].median()),
                "adjacent_absolute_change_max": float(transitions["absolute_change"].max()),
                "adjacent_relative_change_median": float(transitions["symmetric_relative_change"].median()),
                "adjacent_relative_change_p90": float(transitions["symmetric_relative_change"].quantile(0.90)),
                "adjacent_relative_change_max": float(transitions["symmetric_relative_change"].max()),
                "adjacent_log_change_median": float(transitions["log_change"].median()),
                "adjacent_log_change_p90": float(transitions["log_change"].quantile(0.90)),
                "adjacent_log_change_max": float(transitions["log_change"].max()),
            })
        records.append(record)
    return pd.DataFrame(records)


def classify_control_tracks(
    track_observations: pd.DataFrame,
    *,
    target_track_ids: set[int],
    event_track_ids: set[int],
    minimum_observations: int,
    require_contiguous: bool,
    exclude_boundary: bool,
    exclude_virtual: bool,
    exclude_events: bool,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (sample_id, track_id), group in track_observations.groupby(["sample_id", "track_id"], sort=True):
        frames = group["frame"].astype(int)
        duplicate = bool(frames.duplicated().any())
        span = int(frames.max() - frames.min() + 1)
        contiguous = not duplicate and frames.nunique() == span
        boundary = bool(group.get("touches_boundary", pd.Series(False, index=group.index)).fillna(False).astype(bool).any())
        virtual = bool(group.get("is_virtual_merge", pd.Series(False, index=group.index)).fillna(False).astype(bool).any())
        target_related = int(track_id) in target_track_ids
        event_related = int(track_id) in event_track_ids
        reasons: list[str] = []
        if len(group) < minimum_observations:
            reasons.append("too_short")
        if duplicate:
            reasons.append("duplicate_frame")
        if require_contiguous and not contiguous:
            reasons.append("temporal_gap")
        if exclude_boundary and boundary:
            reasons.append("boundary_related")
        if exclude_virtual and virtual:
            reasons.append("virtual_merge_related")
        if exclude_events and event_related:
            reasons.append("merge_or_division_related")
        if target_related:
            reasons.append("selected_small_cell_track")
        records.append({
            "sample_id": sample_id,
            "track_id": int(track_id),
            "first_frame": int(frames.min()),
            "last_frame": int(frames.max()),
            "observation_count": int(len(group)),
            "frame_span": span,
            "contiguous": contiguous,
            "duplicate_frames": duplicate,
            "boundary_related": boundary,
            "virtual_related": virtual,
            "event_related": event_related,
            "target_related": target_related,
            "stable_control": not reasons,
            "exclusion_reasons": "|".join(reasons),
        })
    result = pd.DataFrame(records)
    if result.empty:
        return result
    sequence = result.groupby("sample_id").agg(sequence_first=("first_frame", "min"), sequence_last=("last_frame", "max"))
    result = result.merge(sequence, on="sample_id", how="left")
    result["full_span_control"] = (
        result["stable_control"]
        & result["first_frame"].eq(result["sequence_first"])
        & result["last_frame"].eq(result["sequence_last"])
    )
    return result.sort_values(["sample_id", "track_id"]).reset_index(drop=True)


def summarize_feature_variation(
    observations: pd.DataFrame,
    *,
    track_column: str,
    cohort: str,
    features: Iterable[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (sample_id, track_id), group in observations.groupby(["sample_id", track_column], sort=True):
        ordered = group.sort_values("frame", kind="mergesort")
        for feature in features:
            if feature not in ordered.columns:
                continue
            values = pd.to_numeric(ordered[feature], errors="coerce").to_numpy(dtype=float)
            frames = ordered["frame"].to_numpy(dtype=int)
            valid = np.isfinite(values)
            values = values[valid]
            valid_frames = frames[valid]
            if values.size < 2:
                continue
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            adjacent = []
            for index in range(1, len(values)):
                if valid_frames[index] - valid_frames[index - 1] != 1:
                    continue
                denominator = 0.5 * (abs(values[index]) + abs(values[index - 1]))
                adjacent.append(abs(values[index] - values[index - 1]) / max(denominator, 1e-12))
            records.append({
                "cohort": cohort,
                "sample_id": sample_id,
                "track_identity": str(track_id),
                "feature": feature,
                "observation_count": int(values.size),
                "median": median,
                "standard_deviation": float(np.std(values, ddof=1)),
                "coefficient_of_variation": float(np.std(values, ddof=1) / max(abs(median), 1e-12)),
                "mad": mad,
                "robust_cv": float(1.4826 * mad / max(abs(median), 1e-12)),
                "range_relative": float((np.max(values) - np.min(values)) / max(abs(median), 1e-12)),
                "adjacent_relative_change_median": float(np.median(adjacent)) if adjacent else np.nan,
                "adjacent_relative_change_p90": float(np.quantile(adjacent, 0.90)) if adjacent else np.nan,
                "adjacent_relative_change_max": float(np.max(adjacent)) if adjacent else np.nan,
            })
    return pd.DataFrame(records)


def match_size_controls(
    target_summary: pd.DataFrame,
    control_summary: pd.DataFrame,
    *,
    controls_per_target: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if controls_per_target <= 0 or target_summary.empty or control_summary.empty:
        return pd.DataFrame()
    for target in target_summary.itertuples(index=False):
        candidates = control_summary.loc[control_summary["sample_id"].eq(target.sample_id)].copy()
        if candidates.empty or target.volume_median <= 0:
            continue
        candidates["size_distance"] = np.abs(
            np.log(candidates["volume_median"].clip(lower=1e-12) / float(target.volume_median))
        )
        candidates["length_distance"] = np.abs(
            candidates["observation_count"] - int(target.observation_count)
        ) / max(int(target.observation_count), 1)
        candidates["match_score"] = candidates["size_distance"] + 0.10 * candidates["length_distance"]
        chosen = candidates.sort_values(
            ["match_score", "size_distance", "track_identity"], kind="mergesort"
        ).head(controls_per_target)
        for rank, row in enumerate(chosen.itertuples(index=False), start=1):
            records.append({
                "sample_id": target.sample_id,
                "target_track_identity": target.track_identity,
                "target_volume_median": float(target.volume_median),
                "target_observation_count": int(target.observation_count),
                "control_track_identity": row.track_identity,
                "control_volume_median": float(row.volume_median),
                "control_observation_count": int(row.observation_count),
                "size_distance_log": float(row.size_distance),
                "length_distance_fraction": float(row.length_distance),
                "match_score": float(row.match_score),
                "match_rank": rank,
            })
    return pd.DataFrame(records)


def distribution_summary(named_values: dict[str, np.ndarray]) -> pd.DataFrame:
    quantiles = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    records: list[dict[str, object]] = []
    for cohort, values in named_values.items():
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        if array.size == 0:
            continue
        record: dict[str, object] = {
            "cohort": cohort,
            "count": int(array.size),
            "mean": float(np.mean(array)),
            "standard_deviation": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        }
        for quantile in quantiles:
            label = f"q{int(round(100 * quantile)):02d}"
            record[label] = float(np.quantile(array, quantile))
        records.append(record)
    return pd.DataFrame(records)


def threshold_candidates(
    target_values: np.ndarray,
    population_values: np.ndarray,
    control_values: np.ndarray,
) -> pd.DataFrame:
    target = np.asarray(target_values, dtype=float)
    target = target[np.isfinite(target)]
    population = np.asarray(population_values, dtype=float)
    population = population[np.isfinite(population)]
    controls = np.asarray(control_values, dtype=float)
    controls = controls[np.isfinite(controls)]
    candidates: list[tuple[str, float]] = []
    if target.size:
        for quantile in (0.75, 0.90, 0.95, 1.0):
            candidates.append((f"selected_small_cells_q{int(quantile * 100):02d}", float(np.quantile(target, quantile))))
    if population.size:
        for quantile in (0.01, 0.05, 0.10, 0.15, 0.20, 0.25):
            candidates.append((f"all_other_cells_p{int(quantile * 100):02d}", float(np.quantile(population, quantile))))
    records = []
    for source, threshold in candidates:
        records.append({
            "threshold_source": source,
            "threshold_volume_voxels": threshold,
            "selected_target_recall": float(np.mean(target <= threshold)) if target.size else np.nan,
            "all_other_cell_prevalence": float(np.mean(population <= threshold)) if population.size else np.nan,
            "stable_control_prevalence": float(np.mean(controls <= threshold)) if controls.size else np.nan,
        })
    return pd.DataFrame(records).sort_values("threshold_volume_voxels").reset_index(drop=True)


def _cliffs_delta(first: np.ndarray, second: np.ndarray) -> float:
    if first.size == 0 or second.size == 0:
        return np.nan
    greater = sum(np.sum(value > second) for value in first)
    less = sum(np.sum(value < second) for value in first)
    return float((greater - less) / (first.size * second.size))


def compare_track_variation(
    target_summary: pd.DataFrame,
    control_summary: pd.DataFrame,
    *,
    control_cohort_name: str,
) -> pd.DataFrame:
    metrics = (
        "volume_cv", "volume_robust_cv", "volume_iqr_relative", "volume_range_relative",
        "adjacent_relative_change_median", "adjacent_relative_change_p90",
        "adjacent_relative_change_max", "adjacent_log_change_median",
        "adjacent_log_change_p90", "adjacent_log_change_max",
    )
    records: list[dict[str, object]] = []
    for metric in metrics:
        first = pd.to_numeric(target_summary.get(metric), errors="coerce").dropna().to_numpy(dtype=float)
        second = pd.to_numeric(control_summary.get(metric), errors="coerce").dropna().to_numpy(dtype=float)
        if first.size == 0 or second.size == 0:
            continue
        try:
            p_value = float(mannwhitneyu(first, second, alternative="two-sided").pvalue)
        except ValueError:
            p_value = np.nan
        control_median = float(np.median(second))
        records.append({
            "metric": metric,
            "control_cohort": control_cohort_name,
            "target_track_count": int(first.size),
            "control_track_count": int(second.size),
            "target_median": float(np.median(first)),
            "target_q25": float(np.quantile(first, 0.25)),
            "target_q75": float(np.quantile(first, 0.75)),
            "control_median": control_median,
            "control_q25": float(np.quantile(second, 0.25)),
            "control_q75": float(np.quantile(second, 0.75)),
            "median_difference": float(np.median(first) - control_median),
            "median_ratio": float(np.median(first) / control_median) if abs(control_median) > 1e-12 else np.nan,
            "target_median_control_percentile": float(100.0 * np.mean(second <= np.median(first))),
            "mann_whitney_p_value": p_value,
            "cliffs_delta": _cliffs_delta(first, second),
        })
    return pd.DataFrame(records)


def summarize_population_feature_effects(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-observation same-frame feature comparisons."""
    if comparisons.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    for feature, group in comparisons.groupby("feature", sort=True):
        percentile = pd.to_numeric(group["percentile_rank"], errors="coerce").dropna().to_numpy(dtype=float)
        robust_z = pd.to_numeric(group["robust_z"], errors="coerce").dropna().to_numpy(dtype=float)
        ratio = pd.to_numeric(group["ratio_to_population_median"], errors="coerce").dropna().to_numpy(dtype=float)
        records.append({
            "feature": feature,
            "target_observation_count": int(len(group)),
            "median_percentile": float(np.median(percentile)) if percentile.size else np.nan,
            "percentile_q25": float(np.quantile(percentile, 0.25)) if percentile.size else np.nan,
            "percentile_q75": float(np.quantile(percentile, 0.75)) if percentile.size else np.nan,
            "fraction_at_or_below_population_p05": float(np.mean(percentile <= 5.0)) if percentile.size else np.nan,
            "fraction_at_or_below_population_p10": float(np.mean(percentile <= 10.0)) if percentile.size else np.nan,
            "fraction_at_or_below_population_p25": float(np.mean(percentile <= 25.0)) if percentile.size else np.nan,
            "fraction_at_or_above_population_p75": float(np.mean(percentile >= 75.0)) if percentile.size else np.nan,
            "fraction_at_or_above_population_p90": float(np.mean(percentile >= 90.0)) if percentile.size else np.nan,
            "median_robust_z": float(np.median(robust_z)) if robust_z.size else np.nan,
            "median_ratio_to_population_median": float(np.median(ratio)) if ratio.size else np.nan,
        })
    result = pd.DataFrame(records)
    result["distance_from_population_median_percentile"] = (result["median_percentile"] - 50.0).abs()
    return result.sort_values(
        ["distance_from_population_median_percentile", "feature"],
        ascending=[False, True], kind="mergesort",
    ).reset_index(drop=True)


def compare_feature_variation(feature_variation: pd.DataFrame) -> pd.DataFrame:
    """Compare within-track feature variability between selected and stable cohorts."""
    if feature_variation.empty:
        return pd.DataFrame()
    metrics = (
        "coefficient_of_variation", "robust_cv", "range_relative",
        "adjacent_relative_change_median", "adjacent_relative_change_p90",
        "adjacent_relative_change_max",
    )
    target = feature_variation.loc[feature_variation["cohort"].eq("selected_small_cell")]
    control = feature_variation.loc[feature_variation["cohort"].eq("stable_control")]
    records: list[dict[str, object]] = []
    for feature in sorted(set(target["feature"]) & set(control["feature"])):
        first_group = target.loc[target["feature"].eq(feature)]
        second_group = control.loc[control["feature"].eq(feature)]
        for metric in metrics:
            first = pd.to_numeric(first_group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            second = pd.to_numeric(second_group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if first.size == 0 or second.size == 0:
                continue
            try:
                p_value = float(mannwhitneyu(first, second, alternative="two-sided").pvalue)
            except ValueError:
                p_value = np.nan
            control_median = float(np.median(second))
            records.append({
                "feature": feature,
                "variation_metric": metric,
                "target_track_count": int(first.size),
                "control_track_count": int(second.size),
                "target_median": float(np.median(first)),
                "target_q25": float(np.quantile(first, 0.25)),
                "target_q75": float(np.quantile(first, 0.75)),
                "control_median": control_median,
                "control_q25": float(np.quantile(second, 0.25)),
                "control_q75": float(np.quantile(second, 0.75)),
                "median_difference": float(np.median(first) - control_median),
                "median_ratio": float(np.median(first) / control_median) if abs(control_median) > 1e-12 else np.nan,
                "target_median_control_percentile": float(100.0 * np.mean(second <= np.median(first))),
                "mann_whitney_p_value": p_value,
                "cliffs_delta": _cliffs_delta(first, second),
            })
    result = pd.DataFrame(records)
    if result.empty:
        return result
    return result.sort_values(
        ["variation_metric", "target_median_control_percentile", "feature"],
        ascending=[True, False, True], kind="mergesort",
    ).reset_index(drop=True)


def variation_by_size_bins(control_summary: pd.DataFrame, maximum_bins: int = 10) -> pd.DataFrame:
    """Estimate how normal track variability changes with median cell size."""
    if control_summary.empty:
        return pd.DataFrame()
    frame = control_summary.copy()
    valid = pd.to_numeric(frame["volume_median"], errors="coerce").notna()
    frame = frame.loc[valid].copy()
    if frame.empty:
        return pd.DataFrame()
    unique = frame["volume_median"].nunique()
    bin_count = max(1, min(maximum_bins, int(unique), max(1, len(frame) // 5)))
    if bin_count == 1:
        frame["size_bin"] = "all"
    else:
        frame["size_bin"] = pd.qcut(
            frame["volume_median"], q=bin_count, duplicates="drop"
        ).astype(str)
    metrics = (
        "volume_cv", "volume_robust_cv", "volume_range_relative",
        "adjacent_relative_change_median", "adjacent_relative_change_p90",
        "adjacent_relative_change_max", "adjacent_log_change_median",
        "adjacent_log_change_p90",
    )
    records: list[dict[str, object]] = []
    for size_bin, group in frame.groupby("size_bin", sort=False, observed=True):
        record: dict[str, object] = {
            "size_bin": str(size_bin),
            "track_count": int(len(group)),
            "volume_median_min": float(group["volume_median"].min()),
            "volume_median_q25": float(group["volume_median"].quantile(0.25)),
            "volume_median_center": float(group["volume_median"].median()),
            "volume_median_q75": float(group["volume_median"].quantile(0.75)),
            "volume_median_max": float(group["volume_median"].max()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            record[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
            record[f"{metric}_q90"] = float(values.quantile(0.90)) if len(values) else np.nan
            record[f"{metric}_q95"] = float(values.quantile(0.95)) if len(values) else np.nan
        records.append(record)
    return pd.DataFrame(records).sort_values("volume_median_center").reset_index(drop=True)
