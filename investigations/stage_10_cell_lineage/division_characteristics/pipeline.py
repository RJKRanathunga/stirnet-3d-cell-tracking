"""End-to-end Stage 10 division-characteristics investigation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import InvestigationConfig
from .feature_extraction import (
    add_frame_corrected_features,
    extract_cell_observation,
    extract_frame_reference,
)
from .normalization import build_normalized_trajectories
from .plots import create_all_plots
from .repository_io import RepositoryData
from .scene_cases import scan_division_scenes
from .summaries import (
    build_combined_children,
    build_event_summaries,
    summarize_feature_consistency,
)


@dataclass(frozen=True)
class InvestigationResult:
    output_directory: Path
    scene_validation: pd.DataFrame
    case_manifest: pd.DataFrame
    frame_reference: pd.DataFrame
    observations: pd.DataFrame
    normalized_trajectories: pd.DataFrame
    parent_baselines: pd.DataFrame
    combined_children: pd.DataFrame
    event_summaries: pd.DataFrame
    feature_consistency: pd.DataFrame
    metadata: dict[str, object]


def _default_output_directory(paths) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        paths.project_root
        / "data"
        / "investigations"
        / "stage_10_cell_lineage"
        / "division_characteristics"
        / timestamp
    )


def _write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def _daughter_assignment(
    frame_rows: list[dict[str, object]],
    previous_positions: dict[str, np.ndarray] | None,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    if len(frame_rows) != 2:
        raise ValueError("daughter assignment requires exactly two selected cells")

    def position(row: dict[str, object]) -> np.ndarray:
        return np.asarray(
            [row["centroid_z_um"], row["centroid_y_um"], row["centroid_x_um"]],
            dtype=float,
        )

    if previous_positions is None:
        ordered = sorted(
            frame_rows,
            key=lambda row: (
                float(row["centroid_x_um"]),
                float(row["centroid_y_um"]),
                float(row["centroid_z_um"]),
                int(row["cell_id"]),
            ),
        )
        ordered[0]["role"] = "daughter_a"
        ordered[1]["role"] = "daughter_b"
    else:
        first, second = frame_rows
        direct = (
            np.linalg.norm(position(first) - previous_positions["daughter_a"])
            + np.linalg.norm(position(second) - previous_positions["daughter_b"])
        )
        swapped = (
            np.linalg.norm(position(first) - previous_positions["daughter_b"])
            + np.linalg.norm(position(second) - previous_positions["daughter_a"])
        )
        if direct <= swapped:
            first["role"] = "daughter_a"
            second["role"] = "daughter_b"
            ordered = [first, second]
        else:
            first["role"] = "daughter_b"
            second["role"] = "daughter_a"
            ordered = [second, first]

    positions = {
        str(row["role"]): position(row)
        for row in ordered
    }
    return ordered, positions


def _case_manifest_record(case, *, processed: bool, error: str = "") -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "scene_path": str(case.scene_path),
        "sample_id": case.sample_id,
        "event_frame": case.event_frame,
        "previous_parent_frame": case.previous_parent_frame,
        "transition_gap_frames": case.transition_gap_frames,
        "first_selected_frame": min(case.frame_numbers),
        "last_selected_frame": max(case.frame_numbers),
        "selected_frame_count": len(case.frame_numbers),
        "parent_frame_count": len(case.parent_frames),
        "child_frame_count": len(case.child_frames),
        "parent_frames": ",".join(map(str, case.parent_frames)),
        "child_frames": ",".join(map(str, case.child_frames)),
        "processed": processed,
        "processing_error": error,
        "warnings": " | ".join(case.warnings),
    }


def _extract_case(
    case,
    repository: RepositoryData,
    config: InvestigationConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    observations: list[dict[str, object]] = []
    frame_references: list[dict[str, object]] = []
    previous_daughters: dict[str, np.ndarray] | None = None

    for frame in case.frame_numbers:
        artifacts = repository.load_frame(case.sample_id, frame)
        reference = extract_frame_reference(artifacts)
        frame_references.append(reference)

        selected_ids = case.selected_cells[frame]
        frame_rows: list[dict[str, object]] = []
        for cell_id in selected_ids:
            row = extract_cell_observation(artifacts, cell_id, config=config)
            row.update(reference)
            row.update(
                {
                    "case_id": case.case_id,
                    "scene_path": str(case.scene_path),
                    "event_frame": case.event_frame,
                    "relative_frame": int(frame - case.event_frame),
                    "selection_count": len(selected_ids),
                    "is_event_frame": bool(frame == case.event_frame),
                    "track_id": repository.lookup_track_id(frame, cell_id),
                }
            )
            frame_rows.append(row)

        if frame < case.event_frame:
            if len(frame_rows) != 1:
                raise ValueError(f"frame {frame}: expected one parent observation")
            frame_rows[0]["role"] = "parent"
            observations.extend(frame_rows)
        else:
            assigned, previous_daughters = _daughter_assignment(
                frame_rows, previous_daughters
            )
            observations.extend(assigned)

    return observations, frame_references


def run_investigation(
    config: InvestigationConfig | None = None,
    *,
    output_directory: str | Path | None = None,
) -> InvestigationResult:
    """Analyze all valid scenes without modifying production artifacts."""
    from src.io import PipelinePaths

    resolved_config = config or InvestigationConfig()
    paths = PipelinePaths.discover()
    scenes_root = resolved_config.resolved_scenes_root(paths)
    cases, scene_validation = scan_division_scenes(
        scenes_root,
        strict=resolved_config.strict,
    )

    output = (
        Path(output_directory)
        if output_directory is not None
        else _default_output_directory(paths)
    )
    tables_directory = output / "tables"
    figures_directory = output / "figures"
    output.mkdir(parents=True, exist_ok=True)

    repository = RepositoryData(paths)
    observation_records: list[dict[str, object]] = []
    reference_records: list[dict[str, object]] = []
    manifest_records: list[dict[str, object]] = []
    processing_warnings: list[str] = []

    for case in cases:
        try:
            case_observations, case_references = _extract_case(
                case, repository, resolved_config
            )
        except Exception as error:
            message = f"{case.case_id}: {type(error).__name__}: {error}"
            manifest_records.append(
                _case_manifest_record(case, processed=False, error=message)
            )
            processing_warnings.append(message)
            if resolved_config.strict:
                raise
            continue

        observation_records.extend(case_observations)
        reference_records.extend(case_references)
        manifest_records.append(_case_manifest_record(case, processed=True))

    observations = pd.DataFrame(observation_records)
    if observations.empty:
        raise RuntimeError("No division scene completed feature extraction")

    frame_reference = (
        pd.DataFrame(reference_records)
        .drop_duplicates(["sample_id", "frame"])
        .sort_values(["sample_id", "frame"])
        .reset_index(drop=True)
    )
    case_manifest = pd.DataFrame(manifest_records)
    observations = (
        add_frame_corrected_features(observations)
        .sort_values(["case_id", "frame", "role"])
        .reset_index(drop=True)
    )

    normalized, baselines = build_normalized_trajectories(
        observations, resolved_config
    )
    combined_children = build_combined_children(observations)
    event_summaries = build_event_summaries(
        observations,
        combined_children,
        baselines,
        case_manifest,
    )
    feature_consistency = summarize_feature_consistency(event_summaries)

    tables = {
        "scene_validation.csv": scene_validation,
        "case_manifest.csv": case_manifest,
        "frame_reference.csv": frame_reference,
        "observation_features.csv": observations,
        "normalized_trajectories.csv": normalized,
        "parent_baselines.csv": baselines,
        "combined_children.csv": combined_children,
        "event_summaries.csv": event_summaries,
        "feature_consistency.csv": feature_consistency,
    }
    for filename, table in tables.items():
        _write_table(table, tables_directory / filename)

    metadata: dict[str, object] = {
        "schema_version": 1,
        "purpose": (
            "Characterize manually selected parent and child cells around the first "
            "saved one-to-two division transition."
        ),
        "scenes_root": str(scenes_root),
        "config": resolved_config.as_dict(),
        "valid_scene_count": len(cases),
        "processed_case_count": int(case_manifest["processed"].sum()),
        "failed_case_count": int((~case_manifest["processed"]).sum()),
        "event_frame_definition": (
            "The first manually selected frame containing two cells after one-cell frames."
        ),
        "variable_time_windows_supported": True,
        "production_artifacts_modified": False,
        "processing_warnings": processing_warnings,
        "output_tables": list(tables),
    }
    with (output / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)
        file.write("\n")

    if resolved_config.create_plots:
        create_all_plots(
            observations,
            normalized,
            combined_children,
            figures_directory,
        )

    return InvestigationResult(
        output,
        scene_validation,
        case_manifest,
        frame_reference,
        observations,
        normalized,
        baselines,
        combined_children,
        event_summaries,
        feature_consistency,
        metadata,
    )
