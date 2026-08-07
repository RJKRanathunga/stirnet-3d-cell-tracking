"""Physical-space point-to-component association."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from ..config import MergeRealConfig
from ..repository_io import SampleArtifacts


@dataclass(frozen=True)
class PointComponentMatch:
    cell_id: int | None
    distance_um: float
    inside: bool


def associate_point_to_component(
    sample: SampleArtifacts,
    frame: int,
    point_zyx: np.ndarray | None,
    config: MergeRealConfig,
) -> PointComponentMatch:
    if point_zyx is None:
        return PointComponentMatch(None, math.nan, False)
    point = np.asarray(point_zyx, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return PointComponentMatch(None, math.nan, False)

    labels = sample.labels(frame, mmap=True)
    rounded = np.rint(point).astype(int)
    clipped = np.clip(rounded, [0, 0, 0], np.asarray(labels.shape) - 1)
    if np.all(rounded == clipped):
        label = int(labels[tuple(clipped)])
        if label > 0:
            return PointComponentMatch(label, 0.0, True)

    cells = sample.cells(frame)
    if cells.empty or not {"cell_id", "centroid_z", "centroid_y", "centroid_x"}.issubset(cells.columns):
        return PointComponentMatch(None, math.inf, False)
    centroids = cells[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
    spacing = np.asarray(config.voxel_size_zyx_um, dtype=float)
    distances = np.linalg.norm((centroids - point[None, :]) * spacing[None, :], axis=1)
    index = int(np.argmin(distances))
    distance = float(distances[index])
    if distance > config.prediction_component_radius_um:
        return PointComponentMatch(None, distance, False)
    return PointComponentMatch(int(cells.iloc[index]["cell_id"]), distance, False)


def cell_row(sample: SampleArtifacts, frame: int, cell_id: int) -> pd.Series | None:
    cells = sample.cells(frame)
    match = cells.loc[pd.to_numeric(cells["cell_id"], errors="coerce") == int(cell_id)]
    if match.empty:
        return None
    return match.iloc[0]


__all__ = ["PointComponentMatch", "associate_point_to_component", "cell_row"]
