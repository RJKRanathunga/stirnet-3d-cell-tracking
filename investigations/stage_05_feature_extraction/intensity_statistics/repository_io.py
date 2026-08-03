"""Repository-aware loading with explicit Stage 2/3/6 alignment checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


_FRAME_PATTERN = re.compile(r"^t(\d+)\.(?:npy|csv)$")


@dataclass(frozen=True)
class FrameArtifacts:
    frame: int
    raw: np.ndarray
    preprocessed: np.ndarray
    binary_mask: np.ndarray
    instance_labels: np.ndarray
    cells: pd.DataFrame


def _frame_index(path: Path) -> int | None:
    match = _FRAME_PATTERN.match(path.name)
    return int(match.group(1)) if match else None


def discover_available_frames(paths, sample_id: str) -> tuple[int, ...]:
    """Return frames present in preprocessing, masking, segmentation, and cells."""
    series = {}
    for name, suffix in (
        ("preprocessing", "npy"),
        ("masking", "npy"),
        ("segmentation", "npy"),
        ("cells", "csv"),
    ):
        directory = paths.processed_series(sample_id, name)
        indices = {
            index
            for path in directory.glob(f"t*.{suffix}")
            if (index := _frame_index(path)) is not None
        }
        series[name] = indices

    common = set.intersection(*series.values()) if series else set()
    if not common:
        details = ", ".join(f"{name}={len(values)}" for name, values in series.items())
        raise FileNotFoundError(
            f"No aligned processed frames were found for {sample_id}: {details}"
        )
    return tuple(sorted(common))


def load_frame_artifacts(paths, sample_id: str, frame: int) -> FrameArtifacts:
    """Load one raw frame and its saved production artifacts."""
    from src.io import load_timepoint

    stem = f"t{int(frame):03d}"
    raw = np.asarray(load_timepoint(paths.sample_zarr(sample_id), int(frame)))
    preprocessed = np.load(
        paths.processed_series(sample_id, "preprocessing") / f"{stem}.npy",
        allow_pickle=False,
    )
    binary_mask = np.load(
        paths.processed_series(sample_id, "masking") / f"{stem}.npy",
        allow_pickle=False,
    ).astype(bool, copy=False)
    labels = np.load(
        paths.processed_series(sample_id, "segmentation") / f"{stem}.npy",
        allow_pickle=False,
    )
    cells = pd.read_csv(
        paths.processed_series(sample_id, "cells") / f"{stem}.csv"
    )

    shapes = {
        "raw": raw.shape,
        "preprocessed": preprocessed.shape,
        "binary_mask": binary_mask.shape,
        "instance_labels": labels.shape,
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(f"frame {frame} artifact shapes differ: {shapes}")

    return FrameArtifacts(
        int(frame),
        raw,
        np.asarray(preprocessed),
        binary_mask,
        np.asarray(labels),
        cells,
    )


def validate_frame_artifacts(artifacts: FrameArtifacts) -> list[str]:
    """Return nonfatal alignment warnings for one production frame."""
    warnings: list[str] = []
    labels = artifacts.instance_labels
    binary = artifacts.binary_mask
    if np.any((labels > 0) & ~binary):
        warnings.append(
            f"frame {artifacts.frame}: some labeled voxels lie outside the binary mask"
        )

    label_ids = {int(value) for value in np.unique(labels) if value > 0}
    if "cell_id" in artifacts.cells.columns:
        table_ids = {
            int(value)
            for value in artifacts.cells["cell_id"].dropna().astype(int).tolist()
        }
        if label_ids != table_ids:
            missing_table = sorted(label_ids - table_ids)
            missing_labels = sorted(table_ids - label_ids)
            warnings.append(
                f"frame {artifacts.frame}: cell table/label IDs differ; "
                f"missing from table={missing_table[:10]}, "
                f"missing from labels={missing_labels[:10]}"
            )
    else:
        warnings.append(
            f"frame {artifacts.frame}: saved cell table has no cell_id column"
        )
    return warnings


def load_tracks(paths, tracks_csv: str | Path | None = None) -> pd.DataFrame:
    """Load the current Stage 7 tracking table, with an optional explicit path."""
    path = (
        Path(tracks_csv)
        if tracks_csv is not None
        else paths.stage7_tracking / "tracks.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Stage 7 tracks table was not found at {path}. "
            "Pass --tracks-csv to use another table."
        )
    tracks = pd.read_csv(path)
    required = {"track_id", "frame", "cell_id"}
    missing = required.difference(tracks.columns)
    if missing:
        raise KeyError(f"tracks table is missing columns: {sorted(missing)}")
    return tracks
