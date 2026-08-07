"""Component-level shape and boundary evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from ..config import MergeRealConfig
from ..repository_io import SampleArtifacts


def component_is_boundary(
    sample: SampleArtifacts,
    frame: int,
    cell_id: int,
    config: MergeRealConfig,
) -> bool:
    if not config.exclude_boundary_components:
        return False
    cells = sample.cells(frame)
    match = cells.loc[pd.to_numeric(cells["cell_id"], errors="coerce") == int(cell_id)]
    if match.empty:
        return False
    row = match.iloc[0]
    if "touches_boundary" in row.index and pd.notna(row["touches_boundary"]):
        value = row["touches_boundary"]
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return bool(value)
    required = ("z_min", "y_min", "x_min", "z_max", "y_max", "x_max")
    if not all(name in row.index and pd.notna(row[name]) for name in required):
        return False
    shape = np.asarray(sample.spatial_shape_zyx, dtype=int)
    lower = np.asarray([row.z_min, row.y_min, row.x_min], dtype=float)
    upper = np.asarray([row.z_max, row.y_max, row.x_max], dtype=float)
    return bool(np.any(lower <= 0) or np.any(upper >= shape))


def count_edt_peaks(
    sample: SampleArtifacts,
    frame: int,
    cell_id: int,
    config: MergeRealConfig,
) -> int:
    labels = sample.labels(frame, mmap=True)
    mask = np.asarray(labels == int(cell_id), dtype=bool)
    if not mask.any():
        return 0
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(axis=0) - 1, 0)
    hi = np.minimum(coords.max(axis=0) + 2, np.asarray(mask.shape))
    crop = mask[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]
    distance = ndi.distance_transform_edt(crop, sampling=config.voxel_size_zyx_um)
    maximum = float(distance.max())
    if maximum <= 0:
        return 0
    radius_voxels = np.maximum(
        1,
        np.ceil(config.anomaly_edt_peak_separation_um / np.asarray(config.voxel_size_zyx_um)).astype(int),
    )
    footprint_shape = tuple(int(2 * r + 1) for r in radius_voxels)
    local_max = distance == ndi.maximum_filter(distance, size=footprint_shape, mode="constant")
    local_max &= distance >= config.anomaly_edt_peak_relative_height * maximum
    labeled, count = ndi.label(local_max)
    if count == 0:
        return 0
    # One connected plateau is one center candidate.
    return int(count)


__all__ = ["component_is_boundary", "count_edt_peaks"]
