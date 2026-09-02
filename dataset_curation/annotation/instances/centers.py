from __future__ import annotations

"""Center extraction for corrected cells and interior supervoxel ID placement."""

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

    weights = np.ones(
        labels.shape,
        dtype=np.uint8,
    )
    centers = ndimage.center_of_mass(
        weights,
        labels=labels,
        index=ids.tolist(),
    )

    result: dict[int, np.ndarray] = {}
    for label_id, center in zip(
        ids.tolist(),
        centers,
    ):
        point = np.asarray(
            center,
            dtype=np.float64,
        )
        if point.shape == (3,) and np.all(np.isfinite(point)):
            result[int(label_id)] = point
    return result


def supervoxel_interior_points(
    supervoxels_zyx: np.ndarray,
    *,
    hidden_ids: Iterable[int] = (),
    spacing_zyx=DEFAULT_SPACING_ZYX_UM,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
]:
    """
    Put each SV ID at an interior "middle" voxel.

    The selected point is the voxel with maximum physical EDT inside the
    supervoxel bounding box. Unlike a raw centroid, this is guaranteed to lie
    inside concave/irregular supervoxels.
    """
    labels = np.asarray(supervoxels_zyx)
    if labels.ndim != 3:
        raise ValueError(
            f"Expected a 3-D supervoxel frame, got {labels.shape}."
        )

    hidden = {
        int(v)
        for v in hidden_ids
        if int(v) > 0
    }
    ids = np.unique(labels)
    ids = ids[ids > 0]

    if ids.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            {
                "sv_id": np.zeros((0,), dtype=np.int64),
            },
        )

    max_id = int(ids.max())
    objects = ndimage.find_objects(
        labels,
        max_label=max_id,
    )

    points: list[np.ndarray] = []
    kept_ids: list[int] = []

    for raw_id in ids.tolist():
        sv_id = int(raw_id)
        if sv_id in hidden:
            continue

        sl = (
            objects[sv_id - 1]
            if 0 <= sv_id - 1 < len(objects)
            else None
        )
        if sl is None:
            continue

        crop = labels[sl] == sv_id
        if not np.any(crop):
            continue

        distance = ndimage.distance_transform_edt(
            crop,
            sampling=tuple(
                float(v)
                for v in spacing_zyx
            ),
        )
        local = np.asarray(
            np.unravel_index(
                int(np.argmax(distance)),
                distance.shape,
            ),
            dtype=np.float64,
        )
        starts = np.asarray(
            [
                int(axis_slice.start or 0)
                for axis_slice in sl
            ],
            dtype=np.float64,
        )
        points.append(local + starts)
        kept_ids.append(sv_id)

    if not points:
        point_array = np.zeros(
            (0, 3),
            dtype=np.float32,
        )
    else:
        point_array = np.asarray(
            points,
            dtype=np.float32,
        )

    return (
        point_array,
        {
            "sv_id": np.asarray(
                kept_ids,
                dtype=np.int64,
            ),
        },
    )
