"""Same-frame population construction and target-versus-population contrasts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import PhenotypeInvestigationConfig


IDENTITY_AND_CONTEXT_COLUMNS = {
    "frame", "cell_id", "track_id", "track_observation_count",
    "centroid_z", "centroid_y", "centroid_x",
    "centroid_z_um", "centroid_y_um", "centroid_x_um",
    "boundary_distance_um", "frame_raw_foreground_median",
    "frame_preprocessed_foreground_median",
}


def eligible_feature_columns(features: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for name in features.select_dtypes(include=[np.number]).columns:
        if name in IDENTITY_AND_CONTEXT_COLUMNS:
            continue
        if name.startswith("saved_"):
            continue
        if name.endswith("_voxel_count"):
            continue
        if "centroid" in name:
            continue
        columns.append(name)
    return sorted(columns)


def attach_target_features(targets: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    merged = targets.merge(features, on=["sample_id", "frame", "cell_id"], how="left", validate="many_to_one")
    missing = merged["volume_voxels"].isna() if "volume_voxels" in merged.columns else pd.Series(True, index=merged.index)
    if missing.any():
        keys = merged.loc[missing, ["case_id", "sample_id", "frame", "cell_id"]].to_dict("records")
        raise ValueError(f"Target cells are missing from full-frame segmentation features: {keys}")
    return merged


def _clean_control_mask(frame_rows: pd.DataFrame, all_manual_keys: set[tuple[str, int, int]]) -> pd.Series:
    manual = pd.Series(
        [
            (str(row.sample_id), int(row.frame), int(row.cell_id)) in all_manual_keys
            for row in frame_rows.itertuples(index=False)
        ],
        index=frame_rows.index,
    )
    return (
        ~manual
        & ~frame_rows["is_boundary"].fillna(False).astype(bool)
        & ~frame_rows["is_virtual_merge"].fillna(False).astype(bool)
        & ~frame_rows["is_known_lineage"].fillna(False).astype(bool)
        & ~frame_rows["overlaps_segmentation_event"].fillna(False).astype(bool)
    )


def _contrast(target: float, controls: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(controls, dtype=float)
    data = data[np.isfinite(data)]
    if not np.isfinite(target) or not data.size:
        return {
            "control_count": int(data.size), "control_mean": math.nan,
            "control_median": math.nan, "control_std": math.nan,
            "control_mad": math.nan, "delta_from_median": math.nan,
            "robust_z": math.nan, "percentile_rank": math.nan,
            "cliffs_delta": math.nan,
        }
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    less = int(np.count_nonzero(data < target))
    equal = int(np.count_nonzero(data == target))
    greater = int(np.count_nonzero(data > target))
    return {
        "control_count": int(data.size),
        "control_mean": float(np.mean(data)),
        "control_median": median,
        "control_std": float(np.std(data, ddof=0)),
        "control_mad": mad,
        "delta_from_median": float(target - median),
        "robust_z": float((target - median) / (1.4826 * mad)) if mad > 1e-12 else math.nan,
        "percentile_rank": float((less + 0.5 * equal) / data.size),
        "cliffs_delta": float((less - greater) / data.size),
    }


def _size_adjustment(target_row: pd.Series, controls: pd.DataFrame, feature: str, config: PhenotypeInvestigationConfig) -> dict[str, float | int]:
    if feature == "volume_um3" or "volume" in feature and feature.startswith("volume"):
        return {"size_model_count": 0, "size_expected_value": math.nan, "size_adjusted_residual": math.nan, "size_adjusted_scale": math.nan, "size_adjusted_z": math.nan}
    x = pd.to_numeric(controls["volume_um3"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(controls[feature], errors="coerce").to_numpy(float)
    valid = np.isfinite(x) & (x > 0) & np.isfinite(y)
    x, y = np.log(x[valid]), y[valid]
    target_volume = float(target_row["volume_um3"])
    target_value = float(target_row[feature])
    if len(x) < config.minimum_size_model_controls or target_volume <= 0 or not np.isfinite(target_value) or np.unique(x).size <= config.size_model_degree:
        return {"size_model_count": int(len(x)), "size_expected_value": math.nan, "size_adjusted_residual": math.nan, "size_adjusted_scale": math.nan, "size_adjusted_z": math.nan}
    coefficients = np.polyfit(x, y, deg=config.size_model_degree)
    expected = float(np.polyval(coefficients, math.log(target_volume)))
    residuals = y - np.polyval(coefficients, x)
    center = float(np.median(residuals))
    scale = float(1.4826 * np.median(np.abs(residuals - center)))
    residual = float(target_value - expected)
    return {
        "size_model_count": int(len(x)),
        "size_expected_value": expected,
        "size_adjusted_residual": residual,
        "size_adjusted_scale": scale,
        "size_adjusted_z": float((residual - center) / scale) if scale > 1e-12 else math.nan,
    }


def compare_targets_to_populations(targets: pd.DataFrame, features: pd.DataFrame, config: PhenotypeInvestigationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = eligible_feature_columns(features)
    manual_keys = {
        (str(row.sample_id), int(row.frame), int(row.cell_id))
        for row in targets.itertuples(index=False)
    }
    records: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []

    frame_groups = {
        (str(sample_id), int(frame)): group.copy()
        for (sample_id, frame), group in features.groupby(["sample_id", "frame"], sort=False)
    }
    for target in targets.itertuples(index=False):
        key = (str(target.sample_id), int(target.frame))
        frame_rows = frame_groups[key]
        target_rows = frame_rows.loc[frame_rows["cell_id"].astype(int) == int(target.cell_id)]
        if len(target_rows) != 1:
            raise ValueError(f"Expected one feature row for target {key}, cell {target.cell_id}")
        target_row = target_rows.iloc[0]
        other_mask = frame_rows["cell_id"].astype(int) != int(target.cell_id)
        non_manual_mask = pd.Series(
            [
                (str(row.sample_id), int(row.frame), int(row.cell_id)) not in manual_keys
                for row in frame_rows.itertuples(index=False)
            ],
            index=frame_rows.index,
        )
        others = frame_rows.loc[other_mask & non_manual_mask].copy()
        clean = others.loc[_clean_control_mask(others, manual_keys)].copy()
        target_position = target_row[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy(float)
        positions = others[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy(float)
        distances = np.linalg.norm(positions - target_position[None, :], axis=1)
        local = others.loc[distances <= config.local_population_radius_um].copy()
        clean_local = local.loc[_clean_control_mask(local, manual_keys)].copy()
        manifest.append({
            "case_id": target.case_id,
            "sample_id": target.sample_id,
            "frame": int(target.frame),
            "target_cell_id": int(target.cell_id),
            "phenotype_role": target.phenotype_role,
            "phenotype_subtype": target.phenotype_subtype,
            "all_other_count": int(len(others)),
            "clean_normal_count": int(len(clean)),
            "local_other_count": int(len(local)),
            "clean_local_count": int(len(clean_local)),
        })
        populations = {
            "all_other_cells": others,
            "clean_normal_cells": clean,
            "local_other_cells": local,
            "clean_local_cells": clean_local,
        }
        for feature in feature_columns:
            target_value = float(target_row[feature]) if pd.notna(target_row[feature]) else math.nan
            for population_name, population in populations.items():
                contrast = _contrast(target_value, pd.to_numeric(population[feature], errors="coerce").to_numpy(float))
                record = {
                    "case_id": target.case_id,
                    "sample_id": target.sample_id,
                    "frame": int(target.frame),
                    "target_cell_id": int(target.cell_id),
                    "phenotype_role": target.phenotype_role,
                    "phenotype_subtype": target.phenotype_subtype,
                    "is_primary_target": bool(target.is_primary_target),
                    "feature": feature,
                    "target_value": target_value,
                    "population": population_name,
                    **contrast,
                }
                if population_name == "clean_normal_cells":
                    record.update(_size_adjustment(target_row, population, feature, config))
                else:
                    record.update({"size_model_count": 0, "size_expected_value": math.nan, "size_adjusted_residual": math.nan, "size_adjusted_scale": math.nan, "size_adjusted_z": math.nan})
                records.append(record)
    return pd.DataFrame(records), pd.DataFrame(manifest)


def summarize_feature_effects(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Summarize each feature using the scene as the independent unit."""
    if contrasts.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    grouped = contrasts.groupby(["phenotype_subtype", "population", "feature"], sort=True)
    for (subtype, population, feature), rows in grouped:
        case_values = (
            rows.groupby("case_id", as_index=False)
            .agg(
                percentile_rank=("percentile_rank", "median"),
                robust_z=("robust_z", "median"),
                cliffs_delta=("cliffs_delta", "median"),
                size_adjusted_z=("size_adjusted_z", "median"),
            )
        )
        values = case_values["percentile_rank"].to_numpy(float)
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        upper = float(np.mean(values >= 0.80))
        lower = float(np.mean(values <= 0.20))
        direction = "high" if np.median(values) >= 0.5 else "low"
        consistency = upper if direction == "high" else lower
        records.append({
            "phenotype_subtype": subtype,
            "population": population,
            "feature": feature,
            "case_count": int(values.size),
            "median_percentile": float(np.median(values)),
            "minimum_percentile": float(np.min(values)),
            "maximum_percentile": float(np.max(values)),
            "fraction_at_or_above_0p80": upper,
            "fraction_at_or_below_0p20": lower,
            "dominant_direction": direction,
            "extreme_direction_consistency": consistency,
            "median_absolute_extremeness": float(np.median(np.abs(values - 0.5)) * 2),
            "median_robust_z": (
                float(np.nanmedian(case_values["robust_z"]))
                if np.isfinite(case_values["robust_z"].to_numpy(float)).any()
                else math.nan
            ),
            "median_cliffs_delta": (
                float(np.nanmedian(case_values["cliffs_delta"]))
                if np.isfinite(case_values["cliffs_delta"].to_numpy(float)).any()
                else math.nan
            ),
            "median_size_adjusted_z": (
                float(np.nanmedian(case_values["size_adjusted_z"]))
                if np.isfinite(case_values["size_adjusted_z"].to_numpy(float)).any()
                else math.nan
            ),
        })
    result = pd.DataFrame(records)
    if result.empty:
        return result
    return result.sort_values(
        ["phenotype_subtype", "population", "extreme_direction_consistency", "median_absolute_extremeness"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
