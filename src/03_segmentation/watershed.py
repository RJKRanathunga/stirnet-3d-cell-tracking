"""Marker-controlled watershed used by production Stage 3."""

from __future__ import annotations

import numpy as np
from skimage.segmentation import watershed

from .peaks import PeakCandidate


def build_marker_watershed(
    mask: np.ndarray,
    watershed_distance: np.ndarray,
    effective_peaks: tuple[PeakCandidate, ...],
) -> np.ndarray:
    """Partition ``mask`` with labels assigned in effective-peak order."""

    mask = np.asarray(mask, dtype=bool)
    watershed_distance = np.asarray(watershed_distance)
    if mask.ndim != 3 or watershed_distance.shape != mask.shape:
        raise ValueError("mask and watershed_distance must be aligned 3-D arrays")
    if not effective_peaks:
        raise ValueError("at least one effective peak is required")

    marker_image = np.zeros(mask.shape, dtype=np.int32)
    for marker_label, peak in enumerate(effective_peaks, start=1):
        position = peak.position_zyx
        if not mask[position]:
            raise ValueError(f"marker {position} is outside the component")
        if marker_image[position] != 0:
            raise ValueError(f"duplicate marker position {position}")
        marker_image[position] = marker_label

    return watershed(
        -watershed_distance,
        markers=marker_image,
        mask=mask,
        watershed_line=False,
    ).astype(np.int32)
