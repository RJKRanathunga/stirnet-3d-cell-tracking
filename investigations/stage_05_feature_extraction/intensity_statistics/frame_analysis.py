"""Fixed-mask foreground and per-cell intensity measurements."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .methods import IntensityImage
from .statistics import summarize_values


def _geometry(labels: np.ndarray, cell_id: int) -> dict[str, float | int | bool]:
    coordinates = np.argwhere(labels == int(cell_id))
    if coordinates.size == 0:
        raise ValueError(f"cell {cell_id} has no voxels")
    centroid = coordinates.mean(axis=0)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    shape = np.asarray(labels.shape)
    touches_boundary = bool(np.any(lower == 0) or np.any(upper == shape - 1))
    return {
        "volume_voxels": int(coordinates.shape[0]),
        "centroid_z": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_x": float(centroid[2]),
        "z_min": int(lower[0]),
        "y_min": int(lower[1]),
        "x_min": int(lower[2]),
        "z_max": int(upper[0]),
        "y_max": int(upper[1]),
        "x_max": int(upper[2]),
        "touches_image_boundary": touches_boundary,
    }


def analyze_frame(
    *,
    sample_id: str,
    frame: int,
    binary_mask: np.ndarray,
    instance_labels: np.ndarray,
    images: dict[str, IntensityImage],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure every image representation using identical production masks."""
    mask = np.asarray(binary_mask, dtype=bool)
    labels = np.asarray(instance_labels)
    if mask.shape != labels.shape:
        raise ValueError("binary mask and instance labels must have the same shape")
    for method in images.values():
        if method.image.shape != labels.shape:
            raise ValueError(
                f"method {method.name} has shape {method.image.shape}; "
                f"expected {labels.shape}"
            )

    foreground_rows: list[dict[str, object]] = []
    for method in images.values():
        foreground_rows.append(
            {
                "sample_id": sample_id,
                "frame": int(frame),
                "method": method.name,
                "sigma_um": method.sigma_um,
                "mask_source": "production_binary_mask",
                **summarize_values(method.image[mask]),
            }
        )

    cell_rows: list[dict[str, object]] = []
    cell_ids = [int(value) for value in np.unique(labels) if value > 0]
    for cell_id in cell_ids:
        cell_mask = labels == cell_id
        geometry = _geometry(labels, cell_id)
        for method in images.values():
            cell_rows.append(
                {
                    "sample_id": sample_id,
                    "frame": int(frame),
                    "cell_id": int(cell_id),
                    "method": method.name,
                    "sigma_um": method.sigma_um,
                    "mask_source": "production_instance_labels",
                    **geometry,
                    **summarize_values(method.image[cell_mask]),
                }
            )

    return pd.DataFrame(cell_rows), pd.DataFrame(foreground_rows)
