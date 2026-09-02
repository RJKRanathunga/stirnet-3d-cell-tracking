from __future__ import annotations

"""Fast center extraction for corrected cells and supervoxel ID placement."""

from typing import Iterable

import numpy as np
from scipy import ndimage

from dataset_curation.config import DEFAULT_SPACING_ZYX_UM


def frame_instance_centers(
    labels_zyx: np.ndarray,
) -> dict[int, np.ndarray]:
    labels = np.asarray(labels_zyx)
    if labels.ndim != 3:
        raise ValueError(
            f"Expected a 3-D label frame, got {labels.shape}."
        )

    ids = np.unique(labels)
    ids = ids[ids > 0]
    if ids.size == 0:
        return {}

    centers = ndimage.center_of_mass(
        np.ones(labels.shape, dtype=np.uint8),
        labels=labels,
        index=ids.tolist(),
    )

    result: dict[int, np.ndarray] = {}
    for label_id, center in zip(ids.tolist(), centers):
        point = np.asarray(center, dtype=np.float64)
        if point.shape == (3,) and np.all(np.isfinite(point)):
            result[int(label_id)] = point
    return result


def supervoxel_interior_points(
    supervoxels_zyx: np.ndarray,
    *,
    hidden_ids: Iterable[int] = (),
    spacing_zyx=DEFAULT_SPACING_ZYX_UM,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Put each SV ID near its geometric middle while guaranteeing it is inside.

    This uses center-of-mass followed by a nearest-interior-voxel snap. The
    previous per-supervoxel physical EDT was accurate but far too expensive to
    repeat while scrubbing through a 3-D time series.
    """
    labels = np.asarray(supervoxels_zyx)
    if labels.ndim != 3:
        raise ValueError(
            f"Expected a 3-D supervoxel frame, got {labels.shape}."
        )

    hidden = {
        int(value)
        for value in hidden_ids
        if int(value) > 0
    }
    ids = np.unique(labels)
    ids = ids[ids > 0]

    if ids.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            {"sv_id": np.zeros((0,), dtype=np.int64)},
        )

    centers = ndimage.center_of_mass(
        np.ones(labels.shape, dtype=np.uint8),
        labels=labels,
        index=ids.tolist(),
    )
    objects = ndimage.find_objects(
        labels,
        max_label=int(ids.max()),
    )
    spacing = np.asarray(
        tuple(float(v) for v in spacing_zyx),
        dtype=np.float64,
    )

    points: list[np.ndarray] = []
    kept_ids: list[int] = []

    for raw_id, center in zip(ids.tolist(), centers):
        sv_id = int(raw_id)
        if sv_id in hidden:
            continue

        point = np.asarray(center, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            continue

        rounded = np.rint(point).astype(np.int64)
        rounded = np.clip(
            rounded,
            0,
            np.asarray(labels.shape, dtype=np.int64) - 1,
        )

        if int(labels[tuple(rounded.tolist())]) == sv_id:
            chosen = rounded.astype(np.float64)
        else:
            sl = (
                objects[sv_id - 1]
                if 0 <= sv_id - 1 < len(objects)
                else None
            )
            if sl is None:
                continue

            crop = labels[sl] == sv_id
            coordinates = np.argwhere(crop)
            if coordinates.size == 0:
                continue

            starts = np.asarray(
                [
                    int(axis_slice.start or 0)
                    for axis_slice in sl
                ],
                dtype=np.float64,
            )
            target_local = point - starts
            physical_delta = (
                coordinates.astype(np.float64) - target_local
            ) * spacing
            nearest = int(
                np.argmin(
                    np.einsum(
                        "ij,ij->i",
                        physical_delta,
                        physical_delta,
                    )
                )
            )
            chosen = (
                coordinates[nearest].astype(np.float64)
                + starts
            )

        points.append(chosen)
        kept_ids.append(sv_id)

    point_array = (
        np.asarray(points, dtype=np.float32)
        if points
        else np.zeros((0, 3), dtype=np.float32)
    )
    return (
        point_array,
        {"sv_id": np.asarray(kept_ids, dtype=np.int64)},
    )
