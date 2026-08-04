"""Plots and standardized same-frame visual galleries."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PhenotypeInvestigationConfig
from .repository_io import RepositoryData


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


def plot_feature_rankings(summary: pd.DataFrame, output_directory: Path) -> None:
    if summary.empty:
        return
    selected = summary.loc[
        summary["phenotype_subtype"].isin(["parent_final", "daughter_birth"])
        & summary["population"].eq("all_other_cells")
    ]
    for subtype, rows in selected.groupby("phenotype_subtype", sort=True):
        top = rows.sort_values(
            ["extreme_direction_consistency", "median_absolute_extremeness"],
            ascending=[False, False],
        ).head(25).copy()
        if top.empty:
            continue
        signed = np.where(
            top["dominant_direction"].eq("high"),
            top["median_absolute_extremeness"],
            -top["median_absolute_extremeness"],
        )
        plt.figure(figsize=(10, max(6, 0.32 * len(top))))
        y = np.arange(len(top))
        plt.barh(y, signed)
        plt.yticks(y, top["feature"])
        plt.axvline(0, linewidth=1)
        plt.xlabel("Signed median percentile extremeness")
        plt.title(f"{subtype}: strongest same-frame phenotype features")
        plt.gca().invert_yaxis()
        _save(output_directory / "rankings" / f"{subtype}_all_other_cells.png")


def plot_case_heatmaps(contrasts: pd.DataFrame, summary: pd.DataFrame, output_directory: Path) -> None:
    for subtype in ("parent_final", "daughter_birth"):
        ranked = summary.loc[
            summary["phenotype_subtype"].eq(subtype)
            & summary["population"].eq("all_other_cells")
        ].head(20)
        features = ranked["feature"].tolist()
        if not features:
            continue
        rows = contrasts.loc[
            contrasts["phenotype_subtype"].eq(subtype)
            & contrasts["population"].eq("all_other_cells")
            & contrasts["feature"].isin(features)
        ]
        matrix = rows.pivot_table(index="feature", columns="case_id", values="percentile_rank", aggfunc="median").reindex(features)
        if matrix.empty:
            continue
        plt.figure(figsize=(max(7, 1.1 * matrix.shape[1]), max(6, 0.35 * matrix.shape[0])))
        image = plt.imshow(matrix.to_numpy(float), aspect="auto", vmin=0, vmax=1)
        plt.colorbar(image, label="Same-frame percentile")
        plt.xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=35, ha="right")
        plt.yticks(np.arange(matrix.shape[0]), matrix.index)
        plt.title(f"{subtype}: case consistency of top features")
        _save(output_directory / "heatmaps" / f"{subtype}_case_percentiles.png")


def plot_size_relationships(target_features: pd.DataFrame, cell_features: pd.DataFrame, summary: pd.DataFrame, output_directory: Path) -> None:
    top = summary.loc[
        summary["phenotype_subtype"].isin(["parent_final", "daughter_birth"])
        & summary["population"].eq("clean_normal_cells")
    ].sort_values("median_absolute_extremeness", ascending=False)
    for subtype in ("parent_final", "daughter_birth"):
        features = top.loc[top["phenotype_subtype"].eq(subtype), "feature"].head(6).tolist()
        targets = target_features.loc[target_features["phenotype_subtype"].eq(subtype)]
        for feature in features:
            if feature not in cell_features.columns:
                continue
            plt.figure(figsize=(8, 5))
            normal = cell_features[["volume_um3", feature]].replace([np.inf, -np.inf], np.nan).dropna()
            if normal.empty:
                plt.close()
                continue
            plt.scatter(normal["volume_um3"], normal[feature], s=8, alpha=0.20, label="same selected frames")
            plt.scatter(targets["volume_um3"], targets[feature], s=55, marker="x", label=subtype)
            plt.xscale("log")
            plt.xlabel("Cell volume (µm³, log scale)")
            plt.ylabel(feature)
            plt.title(f"{subtype}: {feature} versus cell size")
            plt.legend()
            _save(output_directory / "size_relationships" / subtype / f"{feature}.png")


def _fixed_crop(volume: np.ndarray, center_zyx: np.ndarray, half_voxels: np.ndarray) -> np.ndarray:
    center = np.rint(center_zyx).astype(int)
    shape = 2 * half_voxels + 1
    output = np.zeros(tuple(int(v) for v in shape), dtype=volume.dtype)
    source_start = np.maximum(center - half_voxels, 0)
    source_stop = np.minimum(center + half_voxels + 1, np.asarray(volume.shape))
    target_start = source_start - (center - half_voxels)
    target_stop = target_start + (source_stop - source_start)
    source = tuple(slice(int(a), int(b)) for a, b in zip(source_start, source_stop))
    target = tuple(slice(int(a), int(b)) for a, b in zip(target_start, target_stop))
    output[target] = volume[source]
    return output


def create_galleries(target_features: pd.DataFrame, cell_features: pd.DataFrame, repository: RepositoryData, config: PhenotypeInvestigationConfig, output_directory: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    spacing = np.asarray(config.voxel_size_zyx_um, dtype=float)
    half_voxels = np.ceil(np.asarray(config.gallery_half_size_um) / spacing).astype(int)
    rng = np.random.default_rng(config.random_seed)

    for target in target_features.itertuples(index=False):
        manual_ids = set(
            target_features.loc[
                target_features["sample_id"].eq(target.sample_id)
                & target_features["frame"].eq(int(target.frame)),
                "cell_id",
            ].astype(int)
        )
        frame_rows = cell_features.loc[
            cell_features["sample_id"].eq(target.sample_id)
            & cell_features["frame"].eq(int(target.frame))
            & ~cell_features["cell_id"].astype(int).isin(manual_ids)
        ].copy()
        if frame_rows.empty:
            continue
        target_position = np.asarray([target.centroid_z_um, target.centroid_y_um, target.centroid_x_um], dtype=float)
        positions = frame_rows[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy(float)
        frame_rows["gallery_distance_um"] = np.linalg.norm(positions - target_position[None, :], axis=1)
        local = frame_rows.sort_values(["gallery_distance_um", "cell_id"]).head(config.gallery_local_control_count)
        remaining = frame_rows.loc[~frame_rows.index.isin(local.index)]
        random_count = min(config.gallery_control_count - len(local), len(remaining))
        random_rows = remaining.iloc[rng.choice(len(remaining), size=random_count, replace=False)] if random_count > 0 else remaining.iloc[0:0]
        controls = pd.concat([local, random_rows], ignore_index=False).drop_duplicates("cell_id")
        selected = pd.concat([
            pd.DataFrame([target._asdict()]),
            controls,
        ], ignore_index=True, sort=False).head(config.gallery_control_count + 1)

        artifacts = repository.load_frame(str(target.sample_id), int(target.frame))
        foreground = artifacts.instance_labels > 0
        display_values = artifacts.raw[foreground]
        if display_values.size:
            vmin, vmax = np.percentile(display_values, [1, 99.5])
        else:
            vmin, vmax = float(np.min(artifacts.raw)), float(np.max(artifacts.raw))
        columns = 4
        rows_count = int(math.ceil(len(selected) / columns))
        plt.figure(figsize=(4 * columns, 3.6 * rows_count))
        for index, row in enumerate(selected.itertuples(index=False), start=1):
            center = np.asarray([row.centroid_z, row.centroid_y, row.centroid_x], dtype=float)
            crop = _fixed_crop(artifacts.raw, center, half_voxels)
            label_crop = _fixed_crop(artifacts.instance_labels, center, half_voxels)
            isolated = np.where(label_crop == int(row.cell_id), crop, vmin)
            projection = np.max(isolated, axis=0)
            axis = plt.subplot(rows_count, columns, index)
            axis.imshow(projection, vmin=vmin, vmax=vmax)
            is_target = int(row.cell_id) == int(target.cell_id)
            title = "TARGET" if is_target else ("local" if float(getattr(row, "gallery_distance_um", np.inf)) <= config.local_population_radius_um else "frame")
            axis.set_title(f"{title} C{int(row.cell_id)}\nV={float(row.volume_um3):.1f} μm³")
            axis.axis("off")
        plt.suptitle(f"{target.case_id} | {target.phenotype_subtype} | frame {int(target.frame)} | shared intensity scale")
        filename = f"{target.case_id}__f{int(target.frame):03d}__c{int(target.cell_id)}__{target.phenotype_subtype}.png"
        path = output_directory / "galleries" / filename
        _save(path)
        records.append({
            "case_id": target.case_id,
            "sample_id": target.sample_id,
            "frame": int(target.frame),
            "target_cell_id": int(target.cell_id),
            "phenotype_subtype": target.phenotype_subtype,
            "gallery_path": str(path),
            "display_vmin": float(vmin),
            "display_vmax": float(vmax),
            "control_count": int(len(selected) - 1),
        })
    return pd.DataFrame(records)


def create_all_plots(target_features: pd.DataFrame, cell_features: pd.DataFrame, contrasts: pd.DataFrame, summary: pd.DataFrame, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    plot_feature_rankings(summary, output_directory)
    plot_case_heatmaps(contrasts, summary, output_directory)
    plot_size_relationships(target_features, cell_features, summary, output_directory)
