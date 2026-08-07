"""Crop extraction and permanent accepted-case materialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MergeRealConfig, MergeRealPaths
from .repository_io import FullProcessedRepository, SampleArtifacts


def candidate_crop_slices(
    sample: SampleArtifacts,
    frame: int,
    cell_id: int,
    config: MergeRealConfig,
) -> tuple[slice, slice, slice]:
    cells = sample.cells(frame)
    match = cells.loc[pd.to_numeric(cells["cell_id"], errors="coerce") == int(cell_id)]
    shape = np.asarray(sample.spatial_shape_zyx, dtype=int)
    spacing = np.asarray(config.voxel_size_zyx_um, dtype=float)
    margin = np.ceil(config.crop_margin_um / spacing).astype(int)

    if not match.empty and all(
        column in match.columns
        for column in ("z_min", "y_min", "x_min", "z_max", "y_max", "x_max")
    ):
        row = match.iloc[0]
        lo = np.floor([row.z_min, row.y_min, row.x_min]).astype(int) - margin
        hi = np.ceil([row.z_max, row.y_max, row.x_max]).astype(int) + margin
    else:
        labels = sample.labels(frame, mmap=True)
        coords = np.argwhere(labels == int(cell_id))
        if coords.size == 0:
            raise ValueError(f"component {cell_id} not present in frame {frame}")
        lo = coords.min(axis=0) - margin
        hi = coords.max(axis=0) + 1 + margin
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]


def crop_origin(slices: tuple[slice, slice, slice]) -> tuple[int, int, int]:
    return tuple(int(s.start or 0) for s in slices)


def load_review_volume(
    sample: SampleArtifacts,
    frame: int,
    cell_id: int,
    config: MergeRealConfig,
) -> dict[str, np.ndarray | tuple[slice, slice, slice]]:
    slices = candidate_crop_slices(sample, frame, cell_id, config)
    labels = np.asarray(sample.labels(frame, mmap=True)[slices])
    result: dict[str, np.ndarray | tuple[slice, slice, slice]] = {
        "slices": slices,
        "preprocessed": np.asarray(sample.preprocessed(frame, mmap=True)[slices]),
        "binary_mask": np.asarray(sample.binary_mask(frame, mmap=True)[slices]),
        "production_labels": labels,
        "candidate_mask": np.asarray(labels == int(cell_id), dtype=np.uint8),
    }
    raw = sample.raw_frame(frame)
    if raw is not None:
        result["raw"] = np.asarray(raw[slices])
    return result


def materialize_case(
    paths: MergeRealPaths,
    config: MergeRealConfig,
    candidate: pd.Series,
    instance_labels: np.ndarray,
    uncertain_mask: np.ndarray | None,
    *,
    partition_status: str,
) -> Path:
    repository = FullProcessedRepository(paths)
    sample = repository.sample(str(candidate.sample_id))
    frame = int(candidate.frame)
    cell_id = int(candidate.cell_id)
    volumes = load_review_volume(sample, frame, cell_id, config)
    slices = volumes.pop("slices")
    assert isinstance(slices, tuple)
    case_dir = paths.cases_dir / str(candidate.candidate_id)
    case_dir.mkdir(parents=True, exist_ok=True)

    for name, array in volumes.items():
        np.save(case_dir / f"{name}.npy", np.asarray(array), allow_pickle=False)
    np.save(case_dir / "instance_labels.npy", np.asarray(instance_labels, dtype=np.int32), allow_pickle=False)
    if uncertain_mask is not None and np.any(uncertain_mask):
        np.save(case_dir / "uncertain_mask.npy", np.asarray(uncertain_mask, dtype=np.uint8), allow_pickle=False)

    metadata = {
        "candidate_id": str(candidate.candidate_id),
        "sample_id": str(candidate.sample_id),
        "frame": frame,
        "source_cell_id": cell_id,
        "sources": str(candidate.get("sources", "")),
        "involved_track_ids": str(candidate.get("involved_track_ids", "")),
        "crop_origin_zyx": list(crop_origin(slices)),
        "crop_shape_zyx": list(np.asarray(instance_labels).shape),
        "voxel_size_zyx_um": list(config.voxel_size_zyx_um),
        "partition_status": str(partition_status),
        "expected_instance_count": (
            int(candidate.expected_cell_count)
            if "expected_cell_count" in candidate.index and pd.notna(candidate.expected_cell_count)
            else None
        ),
    }
    with (case_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return case_dir


__all__ = [
    "candidate_crop_slices",
    "crop_origin",
    "load_review_volume",
    "materialize_case",
]
