from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def ellipsoid_footprint(spacing_um: np.ndarray, radius_um: float) -> np.ndarray:
    radii = np.maximum(1, np.ceil(radius_um / np.maximum(spacing_um, 1e-6)).astype(int))
    axes = [np.arange(-r, r + 1, dtype=np.float32) * spacing_um[i] for i, r in enumerate(radii)]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    return (zz**2 + yy**2 + xx**2) <= (radius_um + 1e-6) ** 2


def build_markers(
    score: np.ndarray,
    foreground: np.ndarray,
    spacing_um: np.ndarray,
    radius_um: float,
    threshold: float,
    max_markers: int,
) -> np.ndarray:
    footprint = ellipsoid_footprint(spacing_um, radius_um)
    local_max = score >= ndi.maximum_filter(score, footprint=footprint, mode="nearest") - 1e-7
    candidates = local_max & foreground & (score >= threshold)

    # Plateau maxima become one marker. This is intentionally more tolerant
    # than assigning every local-maximum voxel its own object identity.
    markers, count = ndi.label(candidates)
    if count > max_markers:
        values = []
        for marker_id in range(1, count + 1):
            mask = markers == marker_id
            values.append((float(score[mask].max()), marker_id))
        keep = {m for _, m in sorted(values, reverse=True)[:max_markers]}
        markers = np.where(np.isin(markers, list(keep)), markers, 0)
        markers, count = ndi.label(markers > 0)

    # Every disconnected foreground component needs at least one seed, but a
    # component can have many candidate seeds; the later RAG decides whether
    # those watershed basins should merge.
    fg_cc, fg_count = ndi.label(foreground)
    next_id = int(markers.max()) + 1
    for cc_id in range(1, fg_count + 1):
        region = fg_cc == cc_id
        if np.any(markers[region] > 0):
            continue
        points = np.argwhere(region)
        if len(points) == 0:
            continue
        best = points[np.argmax(score[region])]
        markers[tuple(best)] = next_id
        next_id += 1
    return markers.astype(np.int32, copy=False)
