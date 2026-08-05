"""Diagnostic plots for the small-cell statistics investigation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_plots(
    targets: pd.DataFrame,
    all_other_cells: pd.DataFrame,
    target_summary: pd.DataFrame,
    control_summary: pd.DataFrame,
    frame_comparisons: pd.DataFrame,
    thresholds: pd.DataFrame,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    population = pd.to_numeric(all_other_cells["analysis_volume_voxels"], errors="coerce").dropna()
    selected = pd.to_numeric(targets["analysis_volume_voxels"], errors="coerce").dropna()
    bins = np.geomspace(max(min(population.min(), selected.min()), 1), max(population.max(), selected.max()), 35)
    ax.hist(population, bins=bins, alpha=0.55, label="All other cells")
    ax.hist(selected, bins=bins, alpha=0.75, label="Selected small cells")
    ax.set_xscale("log")
    ax.set_xlabel("Volume (voxels, logarithmic axis)")
    ax.set_ylabel("Observation count")
    ax.set_title("Selected small-cell volumes versus the frame population")
    ax.legend()
    _save(fig, output / "01_volume_distribution.png")

    volume_comparisons = frame_comparisons.loc[
        frame_comparisons["feature"].eq("analysis_volume_voxels")
    ]
    if not volume_comparisons.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(volume_comparisons["percentile_rank"].dropna(), bins=np.arange(0, 105, 5))
        ax.axvline(10, linestyle="--", linewidth=1)
        ax.axvline(25, linestyle="--", linewidth=1)
        ax.set_xlabel("Selected-cell percentile within the same frame")
        ax.set_ylabel("Observation count")
        ax.set_title("Where manually selected cells lie in each frame's size distribution")
        _save(fig, output / "02_frame_volume_percentiles.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, group in targets.groupby("manual_track_id", sort=True):
        ax.plot(group["frame"], group["analysis_volume_voxels"], marker="o", alpha=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Volume (voxels)")
    ax.set_title("Manual small-cell trajectories")
    _save(fig, output / "03_selected_track_volume_trajectories.png")

    if not target_summary.empty and not control_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            control_summary["volume_median"],
            control_summary["adjacent_relative_change_p90"],
            alpha=0.45,
            label="Stable controls",
        )
        ax.scatter(
            target_summary["volume_median"],
            target_summary["adjacent_relative_change_p90"],
            marker="x",
            s=60,
            label="Selected small-cell tracks",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Median track volume (voxels)")
        ax.set_ylabel("90th percentile adjacent relative volume change")
        ax.set_title("Volume instability as a function of cell size")
        ax.legend()
        _save(fig, output / "04_variation_vs_track_size.png")

        metrics = [
            "volume_robust_cv", "volume_range_relative",
            "adjacent_relative_change_median", "adjacent_relative_change_p90",
        ]
        data = []
        labels = []
        positions = []
        position = 1
        for metric in metrics:
            target_values = pd.to_numeric(target_summary[metric], errors="coerce").dropna()
            control_values = pd.to_numeric(control_summary[metric], errors="coerce").dropna()
            data.extend([control_values, target_values])
            positions.extend([position, position + 1])
            labels.extend([f"{metric}\ncontrol", f"{metric}\ntarget"])
            position += 3
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.boxplot(data, positions=positions, widths=0.7, showfliers=False)
        ax.set_xticks(positions, labels, rotation=25, ha="right")
        ax.set_ylabel("Normalized variation")
        ax.set_title("Selected tracks versus stable successful tracks")
        _save(fig, output / "05_track_variation_boxplots.png")

    if not thresholds.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ordered = thresholds.sort_values("threshold_volume_voxels")
        ax.plot(ordered["threshold_volume_voxels"], ordered["selected_target_recall"], marker="o", label="Selected-cell recall")
        ax.plot(ordered["threshold_volume_voxels"], ordered["all_other_cell_prevalence"], marker="o", label="Other-cell prevalence")
        ax.plot(ordered["threshold_volume_voxels"], ordered["stable_control_prevalence"], marker="o", label="Stable-control prevalence")
        ax.set_xscale("log")
        ax.set_xlabel("Candidate small-cell threshold (voxels)")
        ax.set_ylabel("Fraction at or below threshold")
        ax.set_title("Trade-off for empirical small-cell size thresholds")
        ax.legend()
        _save(fig, output / "06_threshold_tradeoff.png")
