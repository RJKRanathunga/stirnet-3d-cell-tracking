"""End-to-end small-cell size and volume-variation investigation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SmallCellStatisticsConfig
from .plots import create_plots
from .repository_io import (
    enrich_tracks_with_cells,
    load_event_track_ids,
    load_sample_cells,
    load_tracks,
    resolve_tracks_path,
)
from .scene_io import build_selected_observations, scan_scenes
from .statistics import (
    attach_frame_population_comparisons,
    classify_control_tracks,
    compare_track_variation,
    distribution_summary,
    infer_population_features,
    link_manual_trajectories,
    match_size_controls,
    summarize_feature_variation,
    summarize_population_feature_effects,
    compare_feature_variation,
    variation_by_size_bins,
    summarize_tracks,
    threshold_candidates,
    track_transition_table,
)


@dataclass(frozen=True)
class SmallCellStatisticsResult:
    output_directory: Path
    scene_validation: pd.DataFrame
    selected_observations: pd.DataFrame
    all_cell_observations: pd.DataFrame
    track_observations: pd.DataFrame
    control_track_manifest: pd.DataFrame
    target_track_summary: pd.DataFrame
    control_track_summary: pd.DataFrame
    frame_population_comparisons: pd.DataFrame
    volume_variation_comparison: pd.DataFrame
    size_distribution_summary: pd.DataFrame
    threshold_candidates: pd.DataFrame
    metadata: dict[str, object]


def _default_output(paths) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        paths.project_root
        / "data"
        / "investigations"
        / "stage_07_cell_tracking"
        / "small_cell_statistics"
        / timestamp
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _choose_tracks_path(paths, case_rows: pd.DataFrame) -> Path:
    hints = [value for value in case_rows["source_tracks_csv"].tolist() if str(value).strip()]
    return resolve_tracks_path(paths, hints)


def run_investigation(
    config: SmallCellStatisticsConfig | None = None,
    *,
    output_directory: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> SmallCellStatisticsResult:
    """Measure selected small-cell size and temporal variability against full-volume controls."""
    from src.io import PipelinePaths

    resolved = config or SmallCellStatisticsConfig()
    paths = PipelinePaths.discover()
    scenes_root = resolved.resolved_scenes_root(paths)
    cases, scene_validation = scan_scenes(scenes_root, strict=resolved.strict)
    selected_labels = build_selected_observations(cases)
    output = Path(output_directory) if output_directory is not None else _default_output(paths)
    tables_directory = output / "tables"
    figures_directory = output / "figures"
    output.mkdir(parents=True, exist_ok=True)

    all_cells_parts: list[pd.DataFrame] = []
    track_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    control_manifest_parts: list[pd.DataFrame] = []
    event_ids_by_sample: dict[str, set[int]] = {}
    tracks_paths: dict[str, str] = {}
    warnings: list[str] = []

    samples = sorted(selected_labels["sample_id"].unique())
    for sample_index, sample_id in enumerate(samples, start=1):
        if progress is not None:
            progress(f"[{sample_index}/{len(samples)}] loading full sample statistics for {sample_id}")
        sample_selected = selected_labels.loc[selected_labels["sample_id"].eq(sample_id)].copy()
        try:
            cells = load_sample_cells(
                paths, str(sample_id), validate_mask_volumes=resolved.validate_mask_volumes
            )
            tracks_path = _choose_tracks_path(paths, sample_selected)
            tracks = load_tracks(tracks_path, str(sample_id))
            track_observations = enrich_tracks_with_cells(tracks, cells)
            event_ids = load_event_track_ids(paths, tracks_path)
        except Exception as error:
            message = f"{sample_id}: {type(error).__name__}: {error}"
            warnings.append(message)
            if resolved.strict:
                raise
            continue

        tracks_paths[str(sample_id)] = str(tracks_path)
        event_ids_by_sample[str(sample_id)] = event_ids
        selected = sample_selected.merge(
            cells,
            on=["sample_id", "frame", "cell_id"],
            how="left",
            validate="many_to_one",
            suffixes=("", "_cell"),
        )
        missing_features = selected["analysis_volume_voxels"].isna()
        if missing_features.any():
            message = (
                f"{sample_id}: {int(missing_features.sum())} selected observations were absent "
                "from the Stage 6 cell tables"
            )
            warnings.append(message)
            if resolved.strict:
                raise ValueError(message)
            selected = selected.loc[~missing_features].copy()
        production_lookup = track_observations[
            ["sample_id", "frame", "cell_id", "track_id"]
        ].drop_duplicates(["sample_id", "frame", "cell_id"])
        selected = selected.merge(
            production_lookup.rename(columns={"track_id": "production_track_id"}),
            on=["sample_id", "frame", "cell_id"],
            how="left",
            validate="many_to_one",
        )
        selected = link_manual_trajectories(
            selected,
            voxel_size_zyx_um=resolved.voxel_size_zyx_um,
            maximum_distance_um=resolved.target_link_max_distance_um,
        )
        unmapped = selected["production_track_id"].isna()
        if unmapped.any():
            message = f"{sample_id}: {int(unmapped.sum())} selected observations have no production track mapping"
            warnings.append(message)
            if resolved.strict:
                raise ValueError(message)

        target_track_ids = set(
            pd.to_numeric(selected["production_track_id"], errors="coerce").dropna().astype(int)
        )
        manifest = classify_control_tracks(
            track_observations,
            target_track_ids=target_track_ids,
            event_track_ids=event_ids,
            minimum_observations=resolved.control_minimum_observations,
            require_contiguous=resolved.control_require_contiguous,
            exclude_boundary=resolved.control_exclude_boundary,
            exclude_virtual=resolved.control_exclude_virtual,
            exclude_events=resolved.control_exclude_event_tracks,
        )
        all_cells_parts.append(cells)
        track_parts.append(track_observations)
        selected_parts.append(selected)
        control_manifest_parts.append(manifest)

    if not selected_parts:
        details = "\n".join(f"  - {message}" for message in warnings)
        raise RuntimeError("No sample completed small-cell statistical extraction" + (f"\n{details}" if details else ""))

    all_cells = pd.concat(all_cells_parts, ignore_index=True, sort=False)
    track_observations = pd.concat(track_parts, ignore_index=True, sort=False)
    selected = pd.concat(selected_parts, ignore_index=True, sort=False)
    control_manifest = pd.concat(control_manifest_parts, ignore_index=True, sort=False)

    feature_columns = infer_population_features(all_cells)
    frame_comparisons = attach_frame_population_comparisons(selected, all_cells, feature_columns)
    population_feature_summary = summarize_population_feature_effects(frame_comparisons)

    target_summary = summarize_tracks(
        selected, track_column="manual_track_id", cohort="selected_small_cell"
    )
    target_identity = (
        selected.groupby(["sample_id", "manual_track_id"], as_index=False)
        .agg(
            case_id=("case_id", "first"),
            production_track_id_count=("production_track_id", lambda values: int(pd.to_numeric(values, errors="coerce").dropna().nunique())),
            production_track_ids=("production_track_id", lambda values: "|".join(str(value) for value in sorted(set(pd.to_numeric(values, errors="coerce").dropna().astype(int))))),
            unmapped_production_observation_count=("production_track_id", lambda values: int(pd.to_numeric(values, errors="coerce").isna().sum())),
        )
        .rename(columns={"manual_track_id": "track_identity"})
    )
    production_changes = []
    for (sample_id, manual_track_id), group in selected.groupby(["sample_id", "manual_track_id"], sort=True):
        values = pd.to_numeric(group.sort_values("frame")["production_track_id"], errors="coerce").to_numpy(dtype=float)
        changes = sum(
            int(np.isfinite(values[index - 1]) and np.isfinite(values[index]) and values[index - 1] != values[index])
            for index in range(1, len(values))
        )
        production_changes.append({
            "sample_id": sample_id, "track_identity": manual_track_id,
            "production_track_id_change_count": int(changes),
        })
    target_summary = target_summary.merge(target_identity, on=["sample_id", "track_identity"], how="left")
    target_summary = target_summary.merge(pd.DataFrame(production_changes), on=["sample_id", "track_identity"], how="left")
    stable_ids = control_manifest.loc[control_manifest["stable_control"].astype(bool), ["sample_id", "track_id"]]
    stable_observations = track_observations.merge(
        stable_ids.assign(_stable=True), on=["sample_id", "track_id"], how="inner"
    ).drop(columns=["_stable"])
    control_summary = summarize_tracks(
        stable_observations, track_column="track_id", cohort="stable_control"
    )

    target_transitions = track_transition_table(
        selected, track_column="manual_track_id", cohort="selected_small_cell"
    )
    control_transitions = track_transition_table(
        stable_observations, track_column="track_id", cohort="stable_control"
    )
    transitions = pd.concat([target_transitions, control_transitions], ignore_index=True, sort=False)

    target_feature_variation = summarize_feature_variation(
        selected,
        track_column="manual_track_id",
        cohort="selected_small_cell",
        features=resolved.variation_features,
    )
    control_feature_variation = summarize_feature_variation(
        stable_observations,
        track_column="track_id",
        cohort="stable_control",
        features=resolved.variation_features,
    )
    feature_variation = pd.concat(
        [target_feature_variation, control_feature_variation], ignore_index=True, sort=False
    )
    feature_variation_comparison = compare_feature_variation(feature_variation)

    matched_controls = match_size_controls(
        target_summary, control_summary,
        controls_per_target=resolved.matched_controls_per_target,
    )
    matched_control_ids = matched_controls[["sample_id", "control_track_identity"]].drop_duplicates() if not matched_controls.empty else pd.DataFrame()
    if not matched_control_ids.empty:
        matched_control_summary = control_summary.merge(
            matched_control_ids,
            left_on=["sample_id", "track_identity"],
            right_on=["sample_id", "control_track_identity"],
            how="inner",
        )
    else:
        matched_control_summary = control_summary.iloc[0:0].copy()

    comparisons = [
        compare_track_variation(target_summary, control_summary, control_cohort_name="all_stable_controls")
    ]
    if not matched_control_summary.empty:
        comparisons.append(
            compare_track_variation(
                target_summary, matched_control_summary,
                control_cohort_name="size_matched_stable_controls",
            )
        )
    variation_comparison = pd.concat(comparisons, ignore_index=True, sort=False)

    target_keys = selected[["sample_id", "frame", "cell_id"]].drop_duplicates()
    other_cells = all_cells.merge(
        target_keys.assign(_selected=True),
        on=["sample_id", "frame", "cell_id"],
        how="left",
    )
    other_cells = other_cells.loc[~other_cells["_selected"].eq(True)].drop(columns=["_selected"])

    size_summary = distribution_summary({
        "selected_small_cell_observations": selected["analysis_volume_voxels"].to_numpy(),
        "selected_small_cell_track_medians": target_summary["volume_median"].to_numpy(),
        "all_other_cell_observations": other_cells["analysis_volume_voxels"].to_numpy(),
        "stable_control_observations": stable_observations["analysis_volume_voxels"].to_numpy(),
        "stable_control_track_medians": control_summary["volume_median"].to_numpy(),
    })
    size_variation_curve = variation_by_size_bins(control_summary)

    thresholds = threshold_candidates(
        selected["analysis_volume_voxels"].to_numpy(),
        other_cells["analysis_volume_voxels"].to_numpy(),
        stable_observations["analysis_volume_voxels"].to_numpy(),
    )

    frame_volume = frame_comparisons.loc[
        frame_comparisons["feature"].eq("analysis_volume_voxels")
    ].copy()
    frame_size_summary = (
        frame_volume.groupby(["sample_id", "case_id", "frame"], as_index=False)
        .agg(
            selected_count=("cell_id", "count"),
            selected_volume_median=("target_value", "median"),
            selected_volume_min=("target_value", "min"),
            selected_volume_max=("target_value", "max"),
            selected_percentile_median=("percentile_rank", "median"),
            population_count=("population_count", "max"),
            population_volume_median=("population_median", "median"),
            population_volume_q10=("population_q10", "median"),
            population_volume_q25=("population_q25", "median"),
        )
        if not frame_volume.empty else pd.DataFrame()
    )

    if resolved.create_plots:
        create_plots(
            selected, other_cells, target_summary, control_summary,
            frame_comparisons, thresholds, figures_directory,
        )

    tables = {
        "scene_validation.csv": scene_validation,
        "selected_small_cell_observations.csv": selected,
        "all_cell_observations.csv": all_cells,
        "all_other_cell_observations.csv": other_cells,
        "track_observations_with_features.csv": track_observations,
        "stable_control_observations.csv": stable_observations,
        "control_track_manifest.csv": control_manifest,
        "selected_small_track_summary.csv": target_summary,
        "stable_control_track_summary.csv": control_summary,
        "track_volume_transitions.csv": transitions,
        "frame_population_feature_comparisons.csv": frame_comparisons,
        "selected_feature_population_summary.csv": population_feature_summary,
        "frame_size_summary.csv": frame_size_summary,
        "feature_within_track_variation.csv": feature_variation,
        "feature_variation_comparison.csv": feature_variation_comparison,
        "stable_control_variation_by_size_bin.csv": size_variation_curve,
        "volume_variation_comparison.csv": variation_comparison,
        "size_matched_control_pairs.csv": matched_controls,
        "size_distribution_summary.csv": size_summary,
        "small_size_threshold_candidates.csv": thresholds,
    }
    for filename, table in tables.items():
        _write_csv(table, tables_directory / filename)

    source_counts = Counter(tracks_paths.values())
    metadata: dict[str, object] = {
        "schema_version": 1,
        "purpose": (
            "Measure the empirical size range and temporal feature variation of manually curated "
            "small-cell failure scenes against full-frame populations and stable successful tracks."
        ),
        "scenes_root": str(scenes_root),
        "config": resolved.as_dict(),
        "valid_scene_count": int(len(cases)),
        "sample_count": int(selected["sample_id"].nunique()),
        "selected_observation_count": int(len(selected)),
        "manual_small_track_count": int(selected["manual_track_id"].nunique()),
        "mapped_production_track_count": int(pd.to_numeric(selected["production_track_id"], errors="coerce").dropna().nunique()),
        "all_cell_observation_count": int(len(all_cells)),
        "stable_control_track_count": int(control_manifest["stable_control"].sum()),
        "full_span_control_track_count": int(control_manifest["full_span_control"].sum()),
        "stable_control_observation_count": int(len(stable_observations)),
        "tracks_sources": tracks_paths,
        "tracks_source_usage": dict(source_counts),
        "population_feature_count": int(len(feature_columns)),
        "population_features": list(feature_columns),
        "mask_volume_validation_enabled": bool(resolved.validate_mask_volumes),
        "production_artifacts_modified": False,
        "warnings": warnings,
        "output_tables": list(tables),
    }
    with (output / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)
        file.write("\n")

    return SmallCellStatisticsResult(
        output,
        scene_validation,
        selected,
        all_cells,
        track_observations,
        control_manifest,
        target_summary,
        control_summary,
        frame_comparisons,
        variation_comparison,
        size_summary,
        thresholds,
        metadata,
    )
