"""Combined-child and per-event summaries."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


EPS = 1e-12


def _safe_log_asymmetry(first: float, second: float) -> float:
    if not np.isfinite(first) or not np.isfinite(second) or first <= 0 or second <= 0:
        return math.nan
    return float(abs(np.log(first / second)))


def build_combined_children(observations: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    sum_features = (
        "volume_voxels",
        "volume_um3",
        "raw_mask_sum",
        "raw_background_corrected_sum",
        "preprocessed_mask_sum",
        "preprocessed_background_corrected_sum",
    )
    asymmetry_features = (
        "volume_um3",
        "raw_mask_mean",
        "raw_mask_sum",
        "raw_background_corrected_mean",
        "preprocessed_mask_mean",
    )

    children = observations[observations["role"].isin(["daughter_a", "daughter_b"])]
    for (case_id, frame), group in children.groupby(["case_id", "frame"], sort=True):
        roles = set(group["role"])
        if roles != {"daughter_a", "daughter_b"} or len(group) != 2:
            continue
        a = group[group["role"] == "daughter_a"].iloc[0]
        b = group[group["role"] == "daughter_b"].iloc[0]
        delta = np.asarray(
            [
                float(a["centroid_z_um"]) - float(b["centroid_z_um"]),
                float(a["centroid_y_um"]) - float(b["centroid_y_um"]),
                float(a["centroid_x_um"]) - float(b["centroid_x_um"]),
            ]
        )
        record: dict[str, object] = {
            "case_id": case_id,
            "sample_id": a["sample_id"],
            "frame": int(frame),
            "relative_frame": int(a["relative_frame"]),
            "event_frame": int(a["event_frame"]),
            "daughter_a_cell_id": int(a["cell_id"]),
            "daughter_b_cell_id": int(b["cell_id"]),
            "daughter_a_track_id": a["track_id"],
            "daughter_b_track_id": b["track_id"],
            "daughter_separation_um": float(np.linalg.norm(delta)),
        }
        for feature in sum_features:
            if feature in group.columns:
                record[f"combined_{feature}"] = float(
                    pd.to_numeric(group[feature], errors="coerce").sum(min_count=1)
                )
        total_volume = float(record.get("combined_volume_voxels", math.nan))
        if np.isfinite(total_volume) and total_volume > EPS:
            for representation in ("raw", "preprocessed"):
                combined_sum = record.get(f"combined_{representation}_mask_sum", math.nan)
                if np.isfinite(combined_sum):
                    record[f"combined_{representation}_mask_mean"] = (
                        float(combined_sum) / total_volume
                    )
        for feature in asymmetry_features:
            if feature in group.columns:
                record[f"asymmetry_{feature}"] = _safe_log_asymmetry(
                    float(a[feature]), float(b[feature])
                )
        records.append(record)
    return pd.DataFrame(records)


def _baseline_map(baselines: pd.DataFrame, case_id: str) -> dict[str, float]:
    current = baselines[baselines["case_id"] == case_id]
    return {
        str(row.feature): float(row.baseline_value)
        for row in current.itertuples(index=False)
    }


def _ratio(value: float, baseline: float) -> float:
    if not np.isfinite(value) or not np.isfinite(baseline) or abs(baseline) <= EPS:
        return math.nan
    return float(value / baseline)


def build_event_summaries(
    observations: pd.DataFrame,
    combined_children: pd.DataFrame,
    baselines: pd.DataFrame,
    case_manifest: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for case_id, case_rows in observations.groupby("case_id", sort=True):
        parent = case_rows[case_rows["role"] == "parent"].sort_values("frame")
        event_children = case_rows[
            (case_rows["relative_frame"] == 0)
            & case_rows["role"].isin(["daughter_a", "daughter_b"])
        ].sort_values("role")
        combined_event = combined_children[
            (combined_children["case_id"] == case_id)
            & (combined_children["relative_frame"] == 0)
        ]
        if parent.empty or len(event_children) != 2 or len(combined_event) != 1:
            continue

        final_parent = parent.iloc[-1]
        daughter_a = event_children[event_children["role"] == "daughter_a"].iloc[0]
        daughter_b = event_children[event_children["role"] == "daughter_b"].iloc[0]
        combined = combined_event.iloc[0]
        baseline = _baseline_map(baselines, case_id)
        manifest_row = case_manifest[case_manifest["case_id"] == case_id].iloc[0]

        record: dict[str, object] = {
            "case_id": case_id,
            "sample_id": final_parent["sample_id"],
            "event_frame": int(final_parent["event_frame"]),
            "parent_frame_count": int(len(parent)),
            "child_frame_count": int(
                case_rows[case_rows["role"].isin(["daughter_a", "daughter_b"])][
                    "frame"
                ].nunique()
            ),
            "transition_gap_frames": int(manifest_row["transition_gap_frames"]),
            "final_parent_frame": int(final_parent["frame"]),
            "final_parent_cell_id": int(final_parent["cell_id"]),
            "daughter_a_cell_id": int(daughter_a["cell_id"]),
            "daughter_b_cell_id": int(daughter_b["cell_id"]),
            "daughter_separation_um": float(combined["daughter_separation_um"]),
        }

        scalar_features = (
            "volume_um3",
            "equivalent_radius_um",
            "axis_major_um",
            "elongation",
            "sphericity",
            "raw_mask_mean",
            "raw_mask_sum",
            "raw_core_mean",
            "raw_sphere_clean_mean",
            "raw_background_corrected_mean",
            "raw_background_corrected_sum",
            "raw_mask_mean_frame_ratio",
            "preprocessed_mask_mean",
            "preprocessed_mask_sum",
        )
        for feature in scalar_features:
            if feature not in case_rows.columns:
                continue
            parent_value = float(final_parent[feature])
            a_value = float(daughter_a[feature])
            b_value = float(daughter_b[feature])
            base = baseline.get(feature, math.nan)
            record[f"parent_final_{feature}"] = parent_value
            record[f"parent_final_{feature}_ratio"] = _ratio(parent_value, base)
            record[f"daughter_a_{feature}"] = a_value
            record[f"daughter_a_{feature}_ratio"] = _ratio(a_value, base)
            record[f"daughter_b_{feature}"] = b_value
            record[f"daughter_b_{feature}_ratio"] = _ratio(b_value, base)
            record[f"daughter_mean_{feature}_ratio"] = _ratio(
                float(np.nanmean([a_value, b_value])), base
            )

        combined_pairs = {
            "volume_um3": "combined_volume_um3",
            "raw_mask_sum": "combined_raw_mask_sum",
            "raw_background_corrected_sum": "combined_raw_background_corrected_sum",
            "preprocessed_mask_sum": "combined_preprocessed_mask_sum",
        }
        for baseline_feature, combined_feature in combined_pairs.items():
            if combined_feature in combined.index:
                value = float(combined[combined_feature])
                record[combined_feature] = value
                record[f"{combined_feature}_ratio"] = _ratio(
                    value, baseline.get(baseline_feature, math.nan)
                )

        parent_volume = parent["volume_um3"].astype(float).to_numpy()
        base_volume = baseline.get("volume_um3", math.nan)
        record["parent_peak_volume_um3"] = float(np.nanmax(parent_volume))
        record["parent_peak_volume_ratio"] = _ratio(
            float(np.nanmax(parent_volume)), base_volume
        )
        records.append(record)
    return pd.DataFrame(records)


def summarize_feature_consistency(event_summaries: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for column in event_summaries.columns:
        if not column.endswith("_ratio"):
            continue
        values = pd.to_numeric(event_summaries[column], errors="coerce").to_numpy(float)
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        records.append(
            {
                "feature_ratio": column,
                "case_count": int(values.size),
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "fraction_above_1": float(np.mean(values > 1.0)),
                "fraction_above_1p10": float(np.mean(values > 1.10)),
                "fraction_below_0p90": float(np.mean(values < 0.90)),
                "consistent_increase": bool(np.mean(values > 1.0) >= 0.8),
                "consistent_decrease": bool(np.mean(values < 1.0) >= 0.8),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["consistent_increase", "consistent_decrease", "fraction_above_1p10"],
        ascending=[False, False, False],
    ) if records else pd.DataFrame()
