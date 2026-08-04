"""End-to-end single-frame division phenotype investigation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from collections.abc import Callable

import pandas as pd

from .config import PhenotypeInvestigationConfig
from .feature_extraction import extract_frame_features
from .plots import create_all_plots, create_galleries
from .population import attach_target_features, compare_targets_to_populations, summarize_feature_effects
from .repository_io import RepositoryData
from .scene_labels import build_target_labels, scan_scenes


@dataclass(frozen=True)
class PhenotypeInvestigationResult:
    output_directory: Path
    scene_validation: pd.DataFrame
    target_labels: pd.DataFrame
    frame_manifest: pd.DataFrame
    cell_features: pd.DataFrame
    target_features: pd.DataFrame
    population_manifest: pd.DataFrame
    population_contrasts: pd.DataFrame
    feature_effect_summary: pd.DataFrame
    gallery_manifest: pd.DataFrame
    metadata: dict[str, object]


def _default_output(paths) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        paths.project_root
        / "data"
        / "investigations"
        / "stage_10_cell_lineage"
        / "single_frame_division_phenotype"
        / timestamp
    )


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def run_investigation(
    config: PhenotypeInvestigationConfig | None = None,
    *,
    output_directory: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> PhenotypeInvestigationResult:
    """Compare labelled parent/daughter cells with every other cell in the same frames."""
    from src.io import PipelinePaths

    resolved = config or PhenotypeInvestigationConfig()
    paths = PipelinePaths.discover()
    scenes_root = resolved.resolved_scenes_root(paths)
    cases, scene_validation = scan_scenes(scenes_root, strict=resolved.strict)
    target_labels = build_target_labels(cases)
    output = Path(output_directory) if output_directory is not None else _default_output(paths)
    tables_directory = output / "tables"
    figures_directory = output / "figures"
    output.mkdir(parents=True, exist_ok=True)

    repository = RepositoryData(paths)
    feature_tables: list[pd.DataFrame] = []
    frame_records: list[dict[str, object]] = []
    warnings: list[str] = []
    frames = target_labels[["sample_id", "frame"]].drop_duplicates().sort_values(["sample_id", "frame"])
    total_frames = len(frames)
    for frame_index, (sample_id, frame) in enumerate(frames.itertuples(index=False), start=1):
        if progress is not None:
            progress(f"[{frame_index}/{total_frames}] extracting {sample_id} frame {int(frame)}")
        try:
            artifacts = repository.load_frame(str(sample_id), int(frame))
            table = extract_frame_features(artifacts, repository, resolved)
        except Exception as error:
            message = f"{sample_id} frame {frame}: {type(error).__name__}: {error}"
            frame_records.append({
                "sample_id": sample_id,
                "frame": int(frame),
                "processed": False,
                "cell_count": 0,
                "error": message,
            })
            warnings.append(message)
            if resolved.strict:
                raise
            continue
        feature_tables.append(table)
        frame_records.append({
            "sample_id": sample_id,
            "frame": int(frame),
            "processed": True,
            "cell_count": int(len(table)),
            "error": "",
        })

    if not feature_tables:
        details = "\n".join(f"  - {message}" for message in warnings[:10])
        if len(warnings) > 10:
            details += f"\n  - ... and {len(warnings) - 10} more frame errors"
        suffix = f"\nFrame errors:\n{details}" if details else ""
        raise RuntimeError(
            "No selected frame completed full-population feature extraction" + suffix
        )
    cell_features = pd.concat(feature_tables, ignore_index=True, sort=False)
    cell_features = cell_features.sort_values(["sample_id", "frame", "cell_id"]).reset_index(drop=True)
    frame_manifest = pd.DataFrame(frame_records)

    available_keys = cell_features[["sample_id", "frame", "cell_id"]].drop_duplicates()
    labelled = target_labels.merge(
        available_keys.assign(feature_available=True),
        on=["sample_id", "frame", "cell_id"],
        how="left",
    )
    labelled["feature_available"] = labelled["feature_available"].fillna(False).astype(bool)
    missing_targets = labelled.loc[~labelled["feature_available"]]
    if not missing_targets.empty:
        message = (
            f"{len(missing_targets)} manually labelled observations are absent from "
            "the loaded production segmentation and were excluded from comparisons"
        )
        warnings.append(message)
        if resolved.strict:
            raise ValueError(message + ": " + str(
                missing_targets[["case_id", "sample_id", "frame", "cell_id"]].to_dict("records")
            ))
    target_labels = labelled
    available_targets = target_labels.loc[target_labels["feature_available"]].drop(columns=["feature_available"])
    if available_targets.empty:
        raise RuntimeError("No manually labelled target exists in the loaded production segmentations")
    target_features = attach_target_features(available_targets, cell_features)
    population_contrasts, population_manifest = compare_targets_to_populations(
        target_features, cell_features, resolved
    )
    feature_effect_summary = summarize_feature_effects(population_contrasts)

    gallery_manifest = pd.DataFrame()
    if resolved.create_plots:
        create_all_plots(
            target_features,
            cell_features,
            population_contrasts,
            feature_effect_summary,
            figures_directory,
        )
    if resolved.create_galleries:
        gallery_manifest = create_galleries(
            target_features,
            cell_features,
            repository,
            resolved,
            figures_directory,
        )

    manual_keys = target_features[["sample_id", "frame", "cell_id"]].drop_duplicates()
    normal_cell_features = cell_features.merge(
        manual_keys.assign(_manual_target=True),
        on=["sample_id", "frame", "cell_id"],
        how="left",
    )
    normal_cell_features = normal_cell_features.loc[
        ~normal_cell_features["_manual_target"].eq(True)
    ].drop(columns=["_manual_target"])
    parent_features = target_features.loc[target_features["phenotype_role"].eq("parent")].copy()
    daughter_features = target_features.loc[target_features["phenotype_role"].eq("daughter")].copy()

    tables = {
        "scene_validation.csv": scene_validation,
        "target_labels.csv": target_labels,
        "frame_manifest.csv": frame_manifest,
        "static_cell_features.csv": cell_features,
        "normal_cell_features.csv": normal_cell_features,
        "target_features.csv": target_features,
        "parent_features.csv": parent_features,
        "daughter_features.csv": daughter_features,
        "primary_parent_features.csv": parent_features.loc[parent_features["phenotype_subtype"].eq("parent_final")],
        "birth_daughter_features.csv": daughter_features.loc[daughter_features["phenotype_subtype"].eq("daughter_birth")],
        "population_manifest.csv": population_manifest,
        "population_contrasts.csv": population_contrasts,
        "feature_effect_summary.csv": feature_effect_summary,
        "gallery_manifest.csv": gallery_manifest,
    }
    for filename, table in tables.items():
        _write_csv(table, tables_directory / filename)

    metadata: dict[str, object] = {
        "schema_version": 1,
        "purpose": "Identify parent and daughter visual phenotypes from one frame without temporal predictor features.",
        "scenes_root": str(scenes_root),
        "config": resolved.as_dict(),
        "valid_scene_count": int(len(cases)),
        "target_observation_count": int(target_labels["feature_available"].sum()),
        "unavailable_target_observation_count": int((~target_labels["feature_available"]).sum()),
        "primary_parent_count": int((target_labels["phenotype_subtype"].eq("parent_final") & target_labels["feature_available"]).sum()),
        "primary_daughter_count": int((target_labels["phenotype_subtype"].eq("daughter_birth") & target_labels["feature_available"]).sum()),
        "processed_frame_count": int(frame_manifest["processed"].sum()),
        "full_population_cell_observation_count": int(len(cell_features)),
        "temporal_features_used": False,
        "size_retained_as_primary_feature": True,
        "size_adjusted_analysis_also_generated": True,
        "production_artifacts_modified": False,
        "warnings": warnings,
        "output_tables": list(tables),
    }
    with (output / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)
        file.write("\n")

    return PhenotypeInvestigationResult(
        output,
        scene_validation,
        target_labels,
        frame_manifest,
        cell_features,
        target_features,
        population_manifest,
        population_contrasts,
        feature_effect_summary,
        gallery_manifest,
        metadata,
    )
