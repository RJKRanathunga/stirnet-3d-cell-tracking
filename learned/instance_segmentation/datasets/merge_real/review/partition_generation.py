"""Generate a 3D instance proposal from human center markers."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed


def snap_centers_to_mask(mask: np.ndarray, centers_zyx: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    centers = np.asarray(centers_zyx, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers_zyx must have shape (N, 3)")
    foreground = np.argwhere(mask)
    if foreground.size == 0:
        raise ValueError("candidate mask is empty")
    result = []
    shape = np.asarray(mask.shape)
    for point in centers:
        rounded = np.clip(np.rint(point).astype(int), 0, shape - 1)
        if mask[tuple(rounded)]:
            result.append(rounded)
            continue
        distances = np.sum((foreground - point[None, :]) ** 2, axis=1)
        result.append(foreground[int(np.argmin(distances))])
    return np.asarray(result, dtype=int)


def generate_partition(
    candidate_mask: np.ndarray,
    centers_zyx: np.ndarray,
    voxel_size_zyx_um: tuple[float, float, float],
) -> np.ndarray:
    mask = np.asarray(candidate_mask, dtype=bool)
    centers = snap_centers_to_mask(mask, centers_zyx)
    markers = np.zeros(mask.shape, dtype=np.int32)
    for marker_id, center in enumerate(centers, start=1):
        position = tuple(int(v) for v in center)
        if markers[position] != 0:
            # Find the nearest unoccupied foreground voxel when two markers round together.
            coords = np.argwhere(mask & (markers == 0))
            if coords.size == 0:
                raise ValueError("not enough foreground voxels for distinct markers")
            distance = np.sum((coords - center[None, :]) ** 2, axis=1)
            position = tuple(int(v) for v in coords[int(np.argmin(distance))])
        markers[position] = marker_id
    distance = ndi.distance_transform_edt(mask, sampling=voxel_size_zyx_um)
    return watershed(-distance, markers=markers, mask=mask, watershed_line=False).astype(np.int32)


__all__ = ["generate_partition", "snap_centers_to_mask"]
