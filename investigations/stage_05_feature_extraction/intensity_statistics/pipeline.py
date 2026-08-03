"""End-to-end orchestration for the paired intensity investigation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .comparisons import (
    compare_methods_to_raw,
    compute_association_pairs,
    summarize_association_pairs,
)
from .config import InvestigationConfig
from .frame_analysis import analyze_frame
from .methods import build_intensity_images
from .plots import create_all_plots
from .repository_io import (
    discover_available_frames,
    load_frame_artifacts,
    load_tracks,
    validate_frame_artifacts,
)
from .tracks import (
    attach_track_ids,
    compute_between_within_ratio,
    compute_track_temporal_statistics,
    select_track_candidates,
)


@dataclass(frozen=True)
class InvestigationResult:
    output_directory: Path
    cell_statistics: pd.DataFrame
    foreground_statistics: pd.DataFrame
    track_candidates: pd.DataFrame
    track_temporal_statistics: pd.DataFrame
    method_comparison: pd.DataFrame
    between_within_statistics: pd.DataFrame
    association_pairs: pd.DataFrame
    association_summary: pd.DataFrame
    metadata: dict[str, object]


def _default_output_directory(paths, sample_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        paths.project_root
        / "data"
        / "investigations"
        / "stage_05_feature_extraction"
        / "intensity_statistics"
        / sample_id
        / timestamp
    )


def _write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def run_investigation(
    config: InvestigationConfig,
    *,
    tracks_csv: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> InvestigationResult:
    """Run the complete investigation without modifying production artifacts."""
    from src.io import PipelinePaths

    paths = PipelinePaths.discover()
    available = discover_available_frames(paths, config.sample_id)
    missing = sorted(set(config.frame_ids).difference(available))
    if missing:
        raise ValueError(f"configured frames are unavailable: {missing}")

    output = (
        Path(output_directory)
        if output_directory is not None
        else _default_output_directory(paths, config.sample_id)
    )
    tables_directory = output / "tables"
    figures_directory = output / "figures"
    output.mkdir(parents=True, exist_ok=True)

    cell_tables: list[pd.DataFrame] = []
    foreground_tables: list[pd.DataFrame] = []
    warnings: list[str] = []
    frame_method_metadata: dict[str, object] = {}

    for frame in config.frame_ids:
        artifacts = load_frame_artifacts(paths, config.sample_id, frame)
        warnings.extend(validate_frame_artifacts(artifacts))
        images, method_metadata = build_intensity_images(
            artifacts.raw,
            artifacts.preprocessed,
            weak_sigmas_um=config.weak_sigmas_um,
            voxel_size_zyx_um=config.voxel_size_zyx_um,
            preprocessing_rtol=config.preprocessing_rtol,
            preprocessing_atol=config.preprocessing_atol,
            allow_preprocessing_mismatch=config.allow_preprocessing_mismatch,
        )
        frame_method_metadata[str(frame)] = method_metadata
        cells, foreground = analyze_frame(
            sample_id=config.sample_id,
            frame=frame,
            binary_mask=artifacts.binary_mask,
            instance_labels=artifacts.instance_labels,
            images=images,
        )
        cell_tables.append(cells)
        foreground_tables.append(foreground)

    cell_statistics = pd.concat(cell_tables, ignore_index=True)
    foreground_statistics = pd.concat(foreground_tables, ignore_index=True)

    tracks = load_tracks(paths, tracks_csv)
    tracks = tracks[tracks["frame"].isin(config.frame_ids)].copy()
    tracked_cell_statistics = attach_track_ids(cell_statistics, tracks)
    track_candidates = select_track_candidates(
        tracks,
        config.frame_ids,
        manual_track_ids=config.manual_track_ids,
    )
    selected_track_ids = (
        track_candidates.loc[track_candidates["selected"], "track_id"]
        .astype(int)
        .tolist()
    )

    track_temporal = compute_track_temporal_statistics(
        tracked_cell_statistics,
        selected_track_ids,
        feature_names=config.temporal_features,
    )
    between_within = compute_between_within_ratio(
        tracked_cell_statistics,
        selected_track_ids,
        feature_names=config.temporal_features,
    )
    method_comparison = compare_methods_to_raw(cell_statistics)
    association_pairs = compute_association_pairs(
        tracked_cell_statistics,
        tracks,
        selected_track_ids,
        voxel_size_zyx_um=config.voxel_size_zyx_um,
        radius_um=config.association_radius_um,
        negatives_per_positive=config.negatives_per_positive,
        association_features=config.association_features,
    )
    association_summary = summarize_association_pairs(association_pairs)

    tables = {
        "cell_intensity_statistics.csv": cell_statistics,
        "foreground_statistics.csv": foreground_statistics,
        "tracked_cell_intensity_statistics.csv": tracked_cell_statistics,
        "track_candidates.csv": track_candidates,
        "track_temporal_statistics.csv": track_temporal,
        "method_comparison.csv": method_comparison,
        "between_within_statistics.csv": between_within,
        "association_pairs.csv": association_pairs,
        "association_summary.csv": association_summary,
    }
    for filename, table in tables.items():
        _write_table(table, tables_directory / filename)

    metadata: dict[str, object] = {
        "schema_version": 1,
        "purpose": (
            "Compare raw, weakly denoised, and current preprocessing intensity "
            "statistics using identical saved production masks."
        ),
        "config": config.as_dict(),
        "available_frames": list(available),
        "selected_track_ids": selected_track_ids,
        "selected_track_count": len(selected_track_ids),
        "automatic_track_selection_is_ground_truth": False,
        "warnings": warnings,
        "frame_method_metadata": frame_method_metadata,
        "output_tables": list(tables),
    }
    with (output / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)

    if config.create_plots:
        create_all_plots(
            cell_stats=cell_statistics,
            tracked_cell_stats=tracked_cell_statistics,
            track_stats=track_temporal,
            association_pairs=association_pairs,
            selected_track_ids=selected_track_ids,
            output_directory=figures_directory,
        )

    return InvestigationResult(
        output,
        cell_statistics,
        foreground_statistics,
        track_candidates,
        track_temporal,
        method_comparison,
        between_within,
        association_pairs,
        association_summary,
        metadata,
    )
