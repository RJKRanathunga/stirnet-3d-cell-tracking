"""Integrated source/output checks used during real batch execution."""

from __future__ import annotations

import json
from pathlib import Path

from src.io import open_sample
from src.io.arrays import list_timepoint_files

from .dataset import FullDatasetSample
from .paths import SampleOutputPaths
from .stage_registry import get_stage_spec


def validate_source_sample(sample: FullDatasetSample) -> tuple[int, int, int, int]:
    array = open_sample(sample.zarr_path)
    shape = tuple(int(value) for value in array.shape)
    if len(shape) != 4:
        raise ValueError(
            f"{sample.sample_id}: expected raw image shape (T, Z, Y, X), found {shape}"
        )
    if shape[0] <= 0 or any(value <= 0 for value in shape[1:]):
        raise ValueError(f"{sample.sample_id}: invalid raw image shape {shape}")
    return shape


def _validate_stage6(directory: Path) -> None:
    preprocessing = list_timepoint_files(directory / "preprocessing", "npy")
    masking = list_timepoint_files(directory / "masking", "npy")
    segmentation = list_timepoint_files(directory / "segmentation", "npy")
    cells = list_timepoint_files(directory / "cells", "csv")
    counts = {len(preprocessing), len(masking), len(segmentation), len(cells)}
    if len(counts) != 1:
        raise ValueError(
            f"Stage 6 series have inconsistent frame counts in {directory}: "
            f"preprocessing={len(preprocessing)}, masking={len(masking)}, "
            f"segmentation={len(segmentation)}, cells={len(cells)}"
        )


def validate_stage_output(stage: int, directory: str | Path) -> None:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Stage {stage} output directory is missing: {root}")

    if int(stage) == 6:
        _validate_stage6(root)

    spec = get_stage_spec(stage)
    missing = [
        relative
        for relative in spec.required_outputs
        if not (root / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Stage {stage} output is incomplete in {root}. Missing: {missing}"
        )


def stage_is_complete(paths: SampleOutputPaths, stage: int) -> bool:
    marker = paths.success_marker(stage)
    if not marker.is_file():
        return False
    try:
        with marker.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if int(payload.get("stage", -1)) != int(stage):
            return False
        if str(payload.get("sample_id", "")) != paths.sample_id:
            return False
        validate_stage_output(stage, paths.stage_directory(stage))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


__all__ = [
    "stage_is_complete",
    "validate_source_sample",
    "validate_stage_output",
]
