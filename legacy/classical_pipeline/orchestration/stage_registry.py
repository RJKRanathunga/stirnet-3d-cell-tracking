"""Stage registry for full-dataset batch execution.

Stage 9 is visualization-only and Stage 12 is intentionally outside this runner.
Stages 1-6 are executed through the repository's existing Stage 6 orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpec:
    number: int
    name: str
    directory_name: str
    required_outputs: tuple[str, ...]


STAGE_SPECS: dict[int, StageSpec] = {
    6: StageSpec(
        number=6,
        name="processed_dataset",
        directory_name="stage_6_processed_dataset",
        required_outputs=(
            "preprocessing/t000.npy",
            "masking/t000.npy",
            "segmentation/t000.npy",
            "cells/t000.csv",
        ),
    ),
    7: StageSpec(
        number=7,
        name="cell_tracking",
        directory_name="stage_7_cell_tracking",
        required_outputs=(
            "tracks.csv",
            "global_motion.csv",
            "association_events.csv",
            "association_candidates.csv",
            "metadata.json",
        ),
    ),
    8: StageSpec(
        number=8,
        name="track_stitching",
        directory_name="stage_8_track_stitching",
        required_outputs=(
            "detections.csv",
            "tracks.csv",
            "segmentation_events.csv",
            "metadata.json",
        ),
    ),
    10: StageSpec(
        number=10,
        name="cell_lineage",
        directory_name="stage_10_cell_lineage",
        required_outputs=(
            "division_candidates.csv",
            "division_events.csv",
            "lineage_edges.csv",
            "track_lineage.csv",
            "protected_tracks.csv",
            "metadata.json",
        ),
    ),
    11: StageSpec(
        number=11,
        name="track_reconciliation",
        directory_name="stage_11_track_reconciliation",
        required_outputs=(
            "tracks.csv",
            "endpoint_classifications.csv",
            "continuation_candidates.csv",
            "continuation_decisions.csv",
            "track_id_remap.csv",
            "unresolved_endings.csv",
            "validation_results.csv",
            "metadata.json",
        ),
    ),
}

BATCH_STAGE_ORDER: tuple[int, ...] = tuple(STAGE_SPECS)


def get_stage_spec(stage: int) -> StageSpec:
    try:
        return STAGE_SPECS[int(stage)]
    except KeyError as error:
        raise ValueError(
            f"Stage {stage} is not a batch-executed stage. "
            f"Available batch stages are {BATCH_STAGE_ORDER}. "
            "Stage 9 is visualization-only."
        ) from error


__all__ = [
    "BATCH_STAGE_ORDER",
    "STAGE_SPECS",
    "StageSpec",
    "get_stage_spec",
]
