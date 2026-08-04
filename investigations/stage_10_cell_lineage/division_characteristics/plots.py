"""Plots for inspecting individual and aligned division characteristics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


def _plot_case_feature(
    case_rows: pd.DataFrame,
    feature: str,
    ylabel: str,
    output: Path,
) -> None:
    if feature not in case_rows.columns:
        return
    plt.figure(figsize=(8, 5))
    plotted = False
    for role, group in case_rows.groupby("role", sort=False):
        values = pd.to_numeric(group[feature], errors="coerce")
        valid = np.isfinite(values.to_numpy(dtype=float))
        if not np.any(valid):
            continue
        ordered = group.loc[valid].sort_values("relative_frame")
        plt.plot(
            ordered["relative_frame"],
            ordered[feature],
            marker="o",
            label=role,
        )
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Frame relative to first two-child frame")
    plt.ylabel(ylabel)
    plt.title(f"{case_rows['case_id'].iloc[0]}: {feature}")
    plt.legend()
    _save(output)


def plot_per_case(observations: pd.DataFrame, output_directory: Path) -> None:
    specifications = (
        ("volume_um3", "Volume (µm³)"),
        ("raw_mask_mean_frame_ratio", "Raw mask mean / frame foreground median"),
        ("raw_background_corrected_mean", "Raw local-background-corrected mean"),
        ("raw_sphere_clean_mean", "Raw fixed-radius clean-sphere mean"),
        ("preprocessed_mask_mean", "Preprocessed mask mean"),
        ("axis_major_um", "Physical PCA major axis (µm)"),
        ("sphericity", "Sphericity"),
    )
    for case_id, case_rows in observations.groupby("case_id", sort=True):
        case_dir = output_directory / "per_case" / str(case_id)
        for feature, ylabel in specifications:
            _plot_case_feature(
                case_rows,
                feature,
                ylabel,
                case_dir / f"{feature}.png",
            )


def plot_combined_children(combined: pd.DataFrame, output_directory: Path) -> None:
    if combined.empty:
        return
    specifications = (
        ("combined_volume_um3", "Combined child volume (µm³)"),
        ("combined_raw_mask_sum", "Combined child raw integrated intensity"),
        ("daughter_separation_um", "Daughter separation (µm)"),
    )
    for feature, ylabel in specifications:
        if feature not in combined.columns:
            continue
        plt.figure(figsize=(8, 5))
        plotted = False
        for case_id, group in combined.groupby("case_id", sort=True):
            group = group.sort_values("relative_frame")
            values = pd.to_numeric(group[feature], errors="coerce")
            valid = np.isfinite(values.to_numpy(dtype=float))
            if not np.any(valid):
                continue
            plt.plot(
                group.loc[valid, "relative_frame"],
                group.loc[valid, feature],
                marker="o",
                label=case_id,
            )
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.xlabel("Frame relative to first two-child frame")
        plt.ylabel(ylabel)
        plt.title(feature)
        plt.legend(fontsize=8)
        _save(output_directory / "combined_children" / f"{feature}.png")


def plot_aligned_normalized(
    trajectories: pd.DataFrame,
    output_directory: Path,
) -> None:
    selected = (
        "volume_um3",
        "raw_mask_mean_frame_ratio",
        "raw_background_corrected_mean",
        "raw_sphere_clean_mean",
        "preprocessed_mask_mean",
    )
    for feature in selected:
        data = trajectories[trajectories["feature"] == feature].copy()
        if data.empty:
            continue
        lineage = (
            data.groupby(["case_id", "relative_frame"], as_index=False)[
                "ratio_to_parent_baseline"
            ]
            .mean()
            .dropna()
        )
        if lineage.empty:
            continue
        plt.figure(figsize=(8, 5))
        for case_id, group in lineage.groupby("case_id", sort=True):
            group = group.sort_values("relative_frame")
            plt.plot(
                group["relative_frame"],
                group["ratio_to_parent_baseline"],
                marker="o",
                alpha=0.55,
                label=case_id,
            )
        median = (
            lineage.groupby("relative_frame", as_index=False)[
                "ratio_to_parent_baseline"
            ]
            .median()
            .sort_values("relative_frame")
        )
        plt.plot(
            median["relative_frame"],
            median["ratio_to_parent_baseline"],
            marker="o",
            linewidth=3,
            label="cross-case median",
        )
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.axhline(1, linestyle=":", linewidth=1)
        plt.xlabel("Frame relative to first two-child frame")
        plt.ylabel("Ratio to parent baseline")
        plt.title(f"Event-aligned {feature}")
        plt.legend(fontsize=8)
        _save(output_directory / "aligned" / f"{feature}.png")


def create_all_plots(
    observations: pd.DataFrame,
    trajectories: pd.DataFrame,
    combined_children: pd.DataFrame,
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    plot_per_case(observations, output_directory)
    plot_combined_children(combined_children, output_directory)
    plot_aligned_normalized(trajectories, output_directory)
