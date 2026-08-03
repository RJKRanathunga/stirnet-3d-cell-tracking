"""Focused plots generated from saved investigation tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_temporal_cv(track_stats: pd.DataFrame, output: Path) -> None:
    data = track_stats[
        (track_stats["feature"] == "median")
        & np.isfinite(track_stats["temporal_cv"])
    ]
    if data.empty:
        return
    methods = sorted(data["method"].unique())
    values = [
        data.loc[data["method"] == method, "temporal_cv"].to_numpy()
        for method in methods
    ]
    plt.figure(figsize=(max(8, len(methods) * 1.4), 5))
    plt.boxplot(values, tick_labels=methods, showfliers=False)
    plt.ylabel("Temporal CV of per-cell median")
    plt.xlabel("Intensity representation")
    plt.xticks(rotation=30, ha="right")
    plt.title("Within-track temporal variation")
    _save(output)


def plot_association_costs(pairs: pd.DataFrame, output: Path) -> None:
    if pairs.empty:
        return
    labels = []
    values = []
    for method in sorted(pairs["method"].unique()):
        for pair_type in ("correct", "wrong_nearby"):
            current = pairs.loc[
                (pairs["method"] == method)
                & (pairs["pair_type"] == pair_type),
                "feature_cost",
            ].dropna()
            if current.empty:
                continue
            labels.append(f"{method}\n{pair_type}")
            values.append(current.to_numpy())
    if not values:
        return
    plt.figure(figsize=(max(10, len(values) * 1.1), 5))
    plt.boxplot(values, tick_labels=labels, showfliers=False)
    plt.ylabel("Mean normalized intensity-feature cost")
    plt.xticks(rotation=35, ha="right")
    plt.title("Correct vs nearby incorrect association costs")
    _save(output)


def plot_method_scatter(
    cell_stats: pd.DataFrame,
    output_directory: Path,
    *,
    feature: str = "median",
) -> None:
    keys = ["sample_id", "frame", "cell_id"]
    raw = cell_stats[cell_stats["method"] == "raw"][keys + [feature]]
    raw = raw.rename(columns={feature: "raw_value"})
    for method in sorted(set(cell_stats["method"]) - {"raw"}):
        current = cell_stats[cell_stats["method"] == method][keys + [feature]]
        current = current.rename(columns={feature: "candidate_value"})
        merged = raw.merge(current, on=keys)
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
        if merged.empty:
            continue
        plt.figure(figsize=(6, 6))
        plt.scatter(
            merged["raw_value"],
            merged["candidate_value"],
            s=8,
            alpha=0.35,
        )
        plt.xlabel(f"Raw {feature}")
        plt.ylabel(f"{method} {feature}")
        plt.title(f"Paired per-cell {feature}: raw vs {method}")
        _save(output_directory / f"raw_vs_{method}_{feature}.png")


def plot_track_trajectories(
    tracked_stats: pd.DataFrame,
    selected_track_ids: list[int],
    output_directory: Path,
    *,
    feature: str = "median",
    maximum_tracks: int = 5,
) -> None:
    for track_id in selected_track_ids[:maximum_tracks]:
        data = tracked_stats[tracked_stats["track_id"] == track_id]
        if data.empty:
            continue
        plt.figure(figsize=(8, 5))
        plotted = False
        for method, group in data.groupby("method", sort=True):
            group = group.sort_values("frame")
            values = group[feature].to_numpy(dtype=float)
            if values.size == 0 or not np.isfinite(values[0]) or abs(values[0]) < 1e-12:
                continue
            relative = values / values[0]
            plt.plot(group["frame"], relative, marker="o", label=method)
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xlabel("Frame")
        plt.ylabel(f"{feature} relative to first frame")
        plt.title(f"Track {track_id}: relative intensity trajectory")
        plt.legend()
        _save(output_directory / f"track_{track_id}_{feature}_trajectory.png")


def create_all_plots(
    *,
    cell_stats: pd.DataFrame,
    tracked_cell_stats: pd.DataFrame,
    track_stats: pd.DataFrame,
    association_pairs: pd.DataFrame,
    selected_track_ids: list[int],
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    plot_temporal_cv(track_stats, output_directory / "temporal_cv_median.png")
    plot_association_costs(
        association_pairs, output_directory / "association_costs.png"
    )
    plot_method_scatter(cell_stats, output_directory, feature="median")
    plot_track_trajectories(
        tracked_cell_stats,
        selected_track_ids,
        output_directory,
        feature="median",
    )
