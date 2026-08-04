"""Parent-baseline normalization for variable-length division scenes."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import InvestigationConfig


EPS = 1e-12


DEFAULT_NORMALIZED_FEATURES = (
    "volume_voxels",
    "volume_um3",
    "equivalent_radius_um",
    "axis_major_um",
    "axis_middle_um",
    "axis_minor_um",
    "elongation",
    "anisotropy",
    "surface_area_um2",
    "sphericity",
    "solidity",
    "raw_mask_mean",
    "raw_mask_median",
    "raw_mask_sum",
    "raw_core_mean",
    "raw_sphere_clean_mean",
    "raw_background_corrected_mean",
    "raw_background_corrected_sum",
    "raw_mask_mean_frame_ratio",
    "raw_mask_median_frame_ratio",
    "raw_mask_sum_frame_ratio",
    "raw_background_corrected_mean_frame_ratio",
    "preprocessed_mask_mean",
    "preprocessed_mask_median",
    "preprocessed_mask_sum",
    "preprocessed_core_mean",
    "preprocessed_sphere_clean_mean",
    "preprocessed_background_corrected_mean",
    "preprocessed_mask_mean_frame_ratio",
)


def _baseline_rows(case_rows: pd.DataFrame, config: InvestigationConfig) -> tuple[pd.DataFrame, str]:
    parent = case_rows[case_rows["role"] == "parent"].sort_values("frame")
    if parent.empty:
        raise ValueError("case has no parent observations")

    exclude = int(config.baseline_exclude_last_parent_frames)
    preferred = parent.iloc[:-exclude] if exclude and len(parent) > exclude else parent.iloc[0:0]
    if len(preferred) >= config.minimum_baseline_frames:
        return preferred, "parent_history_excluding_last"
    if len(parent) >= config.minimum_baseline_frames:
        return parent, "all_available_parent_frames"
    return parent, "limited_parent_history"


def build_normalized_trajectories(
    observations: pd.DataFrame,
    config: InvestigationConfig,
    *,
    features: tuple[str, ...] = DEFAULT_NORMALIZED_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a long trajectory table and one baseline table per case/feature."""
    trajectory_records: list[dict[str, object]] = []
    baseline_records: list[dict[str, object]] = []

    for case_id, case_rows in observations.groupby("case_id", sort=True):
        baseline_rows, source = _baseline_rows(case_rows, config)
        for feature in features:
            if feature not in case_rows.columns:
                continue
            baseline_values = pd.to_numeric(
                baseline_rows[feature], errors="coerce"
            ).to_numpy(dtype=float)
            baseline_values = baseline_values[np.isfinite(baseline_values)]
            baseline = float(np.median(baseline_values)) if baseline_values.size else math.nan
            baseline_mad = (
                float(np.median(np.abs(baseline_values - baseline)))
                if baseline_values.size
                else math.nan
            )
            baseline_records.append(
                {
                    "case_id": case_id,
                    "feature": feature,
                    "baseline_value": baseline,
                    "baseline_mad": baseline_mad,
                    "baseline_frame_count": int(baseline_values.size),
                    "baseline_source": source,
                    "baseline_frames": ",".join(
                        str(int(value)) for value in baseline_rows["frame"].tolist()
                    ),
                }
            )

            for row in case_rows.itertuples(index=False):
                value = getattr(row, feature, math.nan)
                value = float(value) if pd.notna(value) else math.nan
                ratio = (
                    value / baseline
                    if np.isfinite(value) and np.isfinite(baseline) and abs(baseline) > EPS
                    else math.nan
                )
                log2_ratio = (
                    float(np.log2(ratio))
                    if np.isfinite(ratio) and ratio > 0
                    else math.nan
                )
                robust_z = (
                    (value - baseline) / (1.4826 * baseline_mad)
                    if np.isfinite(value)
                    and np.isfinite(baseline)
                    and np.isfinite(baseline_mad)
                    and baseline_mad > EPS
                    else math.nan
                )
                trajectory_records.append(
                    {
                        "case_id": case_id,
                        "sample_id": row.sample_id,
                        "frame": int(row.frame),
                        "relative_frame": int(row.relative_frame),
                        "role": row.role,
                        "cell_id": int(row.cell_id),
                        "track_id": row.track_id,
                        "feature": feature,
                        "value": value,
                        "baseline_value": baseline,
                        "baseline_source": source,
                        "ratio_to_parent_baseline": ratio,
                        "delta_from_parent_baseline": (
                            value - baseline
                            if np.isfinite(value) and np.isfinite(baseline)
                            else math.nan
                        ),
                        "log2_ratio_to_parent_baseline": log2_ratio,
                        "robust_z_to_parent_baseline": robust_z,
                    }
                )

    return pd.DataFrame(trajectory_records), pd.DataFrame(baseline_records)
