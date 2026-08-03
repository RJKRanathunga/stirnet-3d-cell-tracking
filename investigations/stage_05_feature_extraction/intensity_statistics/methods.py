"""Construct paired image representations without changing production masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import sigma_method_name


@dataclass(frozen=True)
class IntensityImage:
    name: str
    image: np.ndarray
    sigma_um: float | None
    description: str


def weak_gaussian(
    raw: np.ndarray,
    *,
    sigma_um: float,
    voxel_size_zyx_um: tuple[float, float, float],
) -> np.ndarray:
    """Apply a physical Gaussian directly to raw-scale floating-point data."""
    values = np.asarray(raw, dtype=np.float32)
    sigma_zyx = float(sigma_um) / np.asarray(voxel_size_zyx_um, dtype=float)
    if np.all(sigma_zyx == 0):
        return values.copy()
    return gaussian_filter(values, sigma=sigma_zyx)


def build_intensity_images(
    raw: np.ndarray,
    saved_preprocessed: np.ndarray,
    *,
    weak_sigmas_um: tuple[float, ...],
    voxel_size_zyx_um: tuple[float, float, float],
    preprocessing_rtol: float,
    preprocessing_atol: float,
    allow_preprocessing_mismatch: bool,
) -> tuple[dict[str, IntensityImage], dict[str, object]]:
    """Build all paired methods and validate replayed preprocessing."""
    # Delayed import keeps synthetic unit tests independent from repository imports.
    from src.api import preprocess_volume

    raw_values = np.asarray(raw)
    saved = np.asarray(saved_preprocessed, dtype=np.float32)
    if raw_values.shape != saved.shape:
        raise ValueError(
            f"raw and saved preprocessed shapes differ: {raw_values.shape} vs {saved.shape}"
        )

    replayed, trace = preprocess_volume(raw_values, return_diagnostics=True)
    replayed = np.asarray(replayed, dtype=np.float32)
    absolute_error = np.abs(replayed - saved)
    max_abs_error = float(np.max(absolute_error))
    mean_abs_error = float(np.mean(absolute_error))
    matches = bool(
        np.allclose(
            replayed,
            saved,
            rtol=preprocessing_rtol,
            atol=preprocessing_atol,
            equal_nan=True,
        )
    )
    if not matches and not allow_preprocessing_mismatch:
        raise RuntimeError(
            "Canonical preprocessing replay does not match the saved Stage 6 "
            f"preprocessed frame (max abs error={max_abs_error:.6g}). "
            "Use --allow-preprocessing-mismatch only when this difference is expected."
        )

    images: dict[str, IntensityImage] = {
        "raw": IntensityImage(
            "raw",
            raw_values.astype(np.float32, copy=False),
            None,
            "Original raw fluorescence values converted to float32.",
        )
    }
    for sigma_um in weak_sigmas_um:
        name = sigma_method_name(sigma_um)
        images[name] = IntensityImage(
            name,
            weak_gaussian(
                raw_values,
                sigma_um=sigma_um,
                voxel_size_zyx_um=voxel_size_zyx_um,
            ),
            float(sigma_um),
            "Weak physical Gaussian applied directly to raw-scale values.",
        )

    normalized = np.asarray(trace.intermediates["normalized"], dtype=np.float32)
    current_denoised = np.asarray(trace.intermediates["denoised"], dtype=np.float32)
    images["normalized"] = IntensityImage(
        "normalized",
        normalized,
        None,
        "Current robust percentile-normalized intermediate before denoising.",
    )
    images["current_denoised"] = IntensityImage(
        "current_denoised",
        current_denoised,
        None,
        "Current normalized image after the production 0.8 um denoising step.",
    )
    images["production_preprocessed"] = IntensityImage(
        "production_preprocessed",
        saved,
        None,
        "Saved Stage 6 fully preprocessed image used by the current feature extractor.",
    )

    metadata = {
        "preprocessing_replay_matches_saved": matches,
        "preprocessing_replay_max_abs_error": max_abs_error,
        "preprocessing_replay_mean_abs_error": mean_abs_error,
        "preprocessing_trace_metrics": dict(trace.metrics),
        "method_descriptions": {
            name: item.description for name, item in images.items()
        },
    }
    return images, metadata
