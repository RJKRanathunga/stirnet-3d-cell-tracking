"""Mask, intensity, morphology, and local-context feature extraction."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import ConvexHull

from .config import InvestigationConfig
from .repository_io import FrameArtifacts, find_cell_row


EPS = 1e-12


def _finite(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    return data[np.isfinite(data)]


def summarize_values(values: np.ndarray) -> dict[str, float | int]:
    data = _finite(values)
    names = (
        "mean", "median", "std", "mad", "min", "max", "sum",
        "p10", "p25", "p75", "p90", "p95", "iqr", "cv", "range",
        "top10_mean", "top25_mean",
    )
    if data.size == 0:
        return {"voxel_count": 0, **{name: math.nan for name in names}}

    p10, p25, p75, p90, p95 = np.percentile(data, [10, 25, 75, 90, 95])
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=0))
    center = float(np.median(data))
    top10_count = max(1, int(math.ceil(0.10 * data.size)))
    top25_count = max(1, int(math.ceil(0.25 * data.size)))
    ordered = np.partition(data, data.size - top25_count)
    top25 = ordered[-top25_count:]
    top10 = np.partition(top25, top25.size - min(top10_count, top25.size))[
        -min(top10_count, top25.size):
    ]
    return {
        "voxel_count": int(data.size),
        "mean": mean,
        "median": center,
        "std": std,
        "mad": float(np.median(np.abs(data - center))),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "sum": float(np.sum(data)),
        "p10": float(p10),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
        "p95": float(p95),
        "iqr": float(p75 - p25),
        "cv": float(std / abs(mean)) if abs(mean) > EPS else math.nan,
        "range": float(np.max(data) - np.min(data)),
        "top10_mean": float(np.mean(top10)),
        "top25_mean": float(np.mean(top25)),
    }


def _prefixed(prefix: str, values: Mapping[str, object]) -> dict[str, object]:
    return {f"{prefix}_{name}": value for name, value in values.items()}


def _physical_surface_area(mask: np.ndarray, spacing: np.ndarray) -> float:
    padded = np.pad(mask.astype(np.int8), 1)
    area = 0.0
    for axis in range(3):
        face_count = int(np.count_nonzero(np.diff(padded, axis=axis)))
        face_area = float(np.prod(np.delete(spacing, axis)))
        area += face_count * face_area
    return float(area)


def _shape_features(mask: np.ndarray, spacing: np.ndarray) -> dict[str, float | int]:
    coordinates = np.argwhere(mask)
    voxel_count = int(coordinates.shape[0])
    voxel_volume = float(np.prod(spacing))
    volume_um3 = voxel_count * voxel_volume
    physical = coordinates.astype(np.float64) * spacing[None, :]
    centroid_voxel = coordinates.mean(axis=0)
    centroid_um = physical.mean(axis=0)

    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    bbox_voxels = maximum - minimum + 1
    bbox_um = bbox_voxels * spacing
    bbox_volume_voxels = int(np.prod(bbox_voxels))
    extent = voxel_count / max(bbox_volume_voxels, 1)

    centered = physical - centroid_um[None, :]
    if voxel_count >= 3:
        covariance = np.cov(centered.T)
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    else:
        eigenvalues = np.zeros(3, dtype=float)
    axes = np.sqrt(np.maximum(eigenvalues, 0.0))

    surface_area = _physical_surface_area(mask, spacing)
    equivalent_radius = (
        (3.0 * volume_um3) / (4.0 * math.pi)
    ) ** (1.0 / 3.0) if volume_um3 > 0 else math.nan
    sphericity = (
        (math.pi ** (1.0 / 3.0)) * ((6.0 * volume_um3) ** (2.0 / 3.0))
        / surface_area
        if surface_area > EPS and volume_um3 > 0
        else math.nan
    )

    convex_volume = math.nan
    solidity = math.nan
    if voxel_count >= 4:
        try:
            convex_volume = float(ConvexHull(physical).volume)
            solidity = (
                float(volume_um3 / convex_volume)
                if convex_volume > EPS
                else math.nan
            )
        except Exception:
            pass

    return {
        "volume_voxels": voxel_count,
        "volume_um3": volume_um3,
        "centroid_z": float(centroid_voxel[0]),
        "centroid_y": float(centroid_voxel[1]),
        "centroid_x": float(centroid_voxel[2]),
        "centroid_z_um": float(centroid_um[0]),
        "centroid_y_um": float(centroid_um[1]),
        "centroid_x_um": float(centroid_um[2]),
        "bbox_depth_voxels": int(bbox_voxels[0]),
        "bbox_height_voxels": int(bbox_voxels[1]),
        "bbox_width_voxels": int(bbox_voxels[2]),
        "bbox_depth_um": float(bbox_um[0]),
        "bbox_height_um": float(bbox_um[1]),
        "bbox_width_um": float(bbox_um[2]),
        "extent": float(extent),
        "equivalent_radius_um": float(equivalent_radius),
        "axis_major_um": float(axes[0]),
        "axis_middle_um": float(axes[1]),
        "axis_minor_um": float(axes[2]),
        "elongation": float(axes[0] / axes[1]) if axes[1] > EPS else math.nan,
        "flatness": float(axes[1] / axes[2]) if axes[2] > EPS else math.nan,
        "anisotropy": float(axes[0] / axes[2]) if axes[2] > EPS else math.nan,
        "surface_area_um2": surface_area,
        "sphericity": float(sphericity),
        "convex_volume_um3": convex_volume,
        "solidity": solidity,
    }


def _crop_slices(mask: np.ndarray, spacing: np.ndarray, margin_um: float) -> tuple[slice, ...]:
    coordinates = np.argwhere(mask)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0) + 1
    margin = np.ceil(float(margin_um) / spacing).astype(int) + 1
    start = np.maximum(minimum - margin, 0)
    stop = np.minimum(maximum + margin, np.asarray(mask.shape, dtype=int))
    return tuple(slice(int(a), int(b)) for a, b in zip(start, stop))


def _local_intensity_features(
    image: np.ndarray,
    labels: np.ndarray,
    target_mask: np.ndarray,
    cell_id: int,
    spacing: np.ndarray,
    config: InvestigationConfig,
) -> dict[str, object]:
    margin_um = max(config.fixed_radius_um, config.background_shell_outer_um)
    slices = _crop_slices(target_mask, spacing, margin_um)
    image_crop = np.asarray(image[slices])
    labels_crop = np.asarray(labels[slices])
    target = np.asarray(target_mask[slices], dtype=bool)

    coordinates = np.argwhere(target)
    centroid_local = coordinates.mean(axis=0)

    core_distance = ndimage.distance_transform_edt(target, sampling=spacing)
    core = core_distance >= float(config.core_erosion_um)
    core_fallback = False
    if not np.any(core):
        core = target.copy()
        core_fallback = True

    outside_distance = ndimage.distance_transform_edt(~target, sampling=spacing)
    shell = (
        (outside_distance >= float(config.background_shell_inner_um))
        & (outside_distance <= float(config.background_shell_outer_um))
        & (labels_crop == 0)
    )

    grid = np.indices(target.shape, dtype=np.float64)
    squared_distance = np.zeros(target.shape, dtype=np.float64)
    for axis in range(3):
        squared_distance += (
            (grid[axis] - centroid_local[axis]) * spacing[axis]
        ) ** 2
    sphere = squared_distance <= float(config.fixed_radius_um) ** 2
    sphere_clean = sphere & ((labels_crop == 0) | (labels_crop == int(cell_id)))

    mask_stats = summarize_values(image_crop[target])
    core_stats = summarize_values(image_crop[core])
    sphere_stats = summarize_values(image_crop[sphere])
    sphere_clean_stats = summarize_values(image_crop[sphere_clean])
    background_stats = summarize_values(image_crop[shell])

    background_median = float(background_stats["median"])
    target_values = np.asarray(image_crop[target], dtype=np.float64)
    corrected_mean = math.nan
    corrected_sum = math.nan
    if np.isfinite(background_median) and target_values.size:
        corrected = target_values - background_median
        corrected_mean = float(np.mean(corrected))
        corrected_sum = float(np.sum(corrected))

    return {
        **_prefixed("mask", mask_stats),
        **_prefixed("core", core_stats),
        **_prefixed("sphere", sphere_stats),
        **_prefixed("sphere_clean", sphere_clean_stats),
        **_prefixed("background", background_stats),
        "background_corrected_mean": corrected_mean,
        "background_corrected_sum": corrected_sum,
        "core_fallback_to_full_mask": core_fallback,
        "shell_available": bool(np.any(shell)),
    }


def extract_frame_reference(artifacts: FrameArtifacts) -> dict[str, object]:
    foreground = artifacts.binary_mask.astype(bool, copy=False)
    background = ~foreground
    raw_foreground = summarize_values(artifacts.raw[foreground])
    raw_background = summarize_values(artifacts.raw[background])
    prep_foreground = summarize_values(artifacts.preprocessed[foreground])
    return {
        "sample_id": artifacts.sample_id,
        "frame": artifacts.frame,
        "frame_foreground_voxels": int(np.count_nonzero(foreground)),
        "frame_raw_foreground_mean": raw_foreground["mean"],
        "frame_raw_foreground_median": raw_foreground["median"],
        "frame_raw_foreground_p90": raw_foreground["p90"],
        "frame_raw_background_median": raw_background["median"],
        "frame_preprocessed_foreground_mean": prep_foreground["mean"],
        "frame_preprocessed_foreground_median": prep_foreground["median"],
    }


def extract_cell_observation(
    artifacts: FrameArtifacts,
    cell_id: int,
    *,
    config: InvestigationConfig,
) -> dict[str, object]:
    labels = artifacts.instance_labels
    target_mask = labels == int(cell_id)
    if not np.any(target_mask):
        raise ValueError(
            f"Sample {artifacts.sample_id} frame {artifacts.frame}: "
            f"cell_id {cell_id} has no instance voxels"
        )

    spacing = np.asarray(config.voxel_size_zyx_um, dtype=np.float64)
    shape = _shape_features(target_mask, spacing)
    raw = _local_intensity_features(
        artifacts.raw,
        labels,
        target_mask,
        int(cell_id),
        spacing,
        config,
    )
    preprocessed = _local_intensity_features(
        artifacts.preprocessed,
        labels,
        target_mask,
        int(cell_id),
        spacing,
        config,
    )

    row: dict[str, object] = {
        "sample_id": artifacts.sample_id,
        "frame": artifacts.frame,
        "cell_id": int(cell_id),
        **shape,
        **_prefixed("raw", raw),
        **_prefixed("preprocessed", preprocessed),
    }

    saved = find_cell_row(artifacts.cells, int(cell_id))
    if saved is not None:
        for name, value in saved.items():
            if name == "cell_id" or isinstance(value, (str, bytes)):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            row[f"saved_{name}"] = numeric
    return row


def add_frame_corrected_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Add multiplicative frame-reference ratios to raw/preprocessed features."""
    result = observations.copy()
    references = {
        "raw": "frame_raw_foreground_median",
        "preprocessed": "frame_preprocessed_foreground_median",
    }
    suffixes = (
        "mask_mean", "mask_median", "mask_sum", "core_mean",
        "sphere_clean_mean", "background_corrected_mean",
        "background_corrected_sum",
    )
    for representation, reference_column in references.items():
        denominator = result[reference_column].astype(float)
        safe = denominator.where(denominator.abs() > EPS)
        for suffix in suffixes:
            source = f"{representation}_{suffix}"
            if source in result.columns:
                result[f"{source}_frame_ratio"] = result[source].astype(float) / safe
    return result
