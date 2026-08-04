"""Identical single-frame feature extraction for targets and population cells."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.stats import kurtosis, skew

from .config import PhenotypeInvestigationConfig
from .repository_io import FrameArtifacts, RepositoryData, saved_cell_row


EPS = 1e-12


def _finite(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    return data[np.isfinite(data)]


def _gini(values: np.ndarray) -> float:
    data = _finite(values)
    if not data.size:
        return math.nan
    data = data - np.min(data)
    total = float(np.sum(data))
    if total <= EPS:
        return 0.0
    ordered = np.sort(data)
    n = ordered.size
    return float((2.0 * np.sum((np.arange(n) + 1) * ordered) / (n * total)) - (n + 1) / n)


def _entropy(values: np.ndarray, bins: int = 64) -> float:
    data = _finite(values)
    if data.size < 2 or float(np.max(data) - np.min(data)) <= EPS:
        return 0.0 if data.size else math.nan
    histogram, _ = np.histogram(data, bins=min(bins, max(8, int(np.sqrt(data.size)))))
    probability = histogram[histogram > 0].astype(float)
    probability /= probability.sum()
    return float(-np.sum(probability * np.log2(probability)))


def summarize_values(values: np.ndarray) -> dict[str, float | int]:
    data = _finite(values)
    names = (
        "mean", "median", "std", "mad", "min", "max", "sum", "p01", "p05",
        "p10", "p25", "p75", "p90", "p95", "p99", "iqr", "cv", "range",
        "skew", "kurtosis", "entropy", "gini", "top01_mean", "top05_mean",
        "top10_mean", "top20_mean", "top10_sum_fraction",
    )
    if data.size == 0:
        return {"voxel_count": 0, **{name: math.nan for name in names}}
    percentiles = np.percentile(data, [1, 5, 10, 25, 75, 90, 95, 99])
    p01, p05, p10, p25, p75, p90, p95, p99 = map(float, percentiles)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=0))
    median = float(np.median(data))

    def top_mean(fraction: float) -> float:
        count = max(1, int(math.ceil(fraction * data.size)))
        return float(np.mean(np.partition(data, data.size - count)[-count:]))

    top10_count = max(1, int(math.ceil(0.10 * data.size)))
    top10 = np.partition(data, data.size - top10_count)[-top10_count:]
    total = float(np.sum(data))
    return {
        "voxel_count": int(data.size),
        "mean": mean,
        "median": median,
        "std": std,
        "mad": float(np.median(np.abs(data - median))),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "sum": total,
        "p01": p01,
        "p05": p05,
        "p10": p10,
        "p25": p25,
        "p75": p75,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "iqr": float(p75 - p25),
        "cv": float(std / abs(mean)) if abs(mean) > EPS else math.nan,
        "range": float(np.max(data) - np.min(data)),
        "skew": float(skew(data, bias=False)) if data.size >= 3 and std > EPS else 0.0,
        "kurtosis": float(kurtosis(data, fisher=True, bias=False)) if data.size >= 4 and std > EPS else 0.0,
        "entropy": _entropy(data),
        "gini": _gini(data),
        "top01_mean": top_mean(0.01),
        "top05_mean": top_mean(0.05),
        "top10_mean": float(np.mean(top10)),
        "top20_mean": top_mean(0.20),
        "top10_sum_fraction": float(np.sum(top10) / total) if abs(total) > EPS else math.nan,
    }


def _prefixed(prefix: str, values: Mapping[str, object]) -> dict[str, object]:
    return {f"{prefix}_{name}": value for name, value in values.items()}


def _surface_area(mask: np.ndarray, spacing: np.ndarray) -> float:
    padded = np.pad(mask.astype(np.int8), 1)
    area = 0.0
    for axis in range(3):
        area += int(np.count_nonzero(np.diff(padded, axis=axis))) * float(np.prod(np.delete(spacing, axis)))
    return float(area)


def _local_maxima_count(values: np.ndarray, mask: np.ndarray, spacing: np.ndarray, minimum_distance_um: float, threshold: float) -> tuple[int, float, float]:
    size = tuple(max(1, int(2 * math.ceil(minimum_distance_um / s) + 1)) for s in spacing)
    smooth_sigma = tuple(max(0.35, 0.35 / s) for s in spacing)
    smoothed = ndimage.gaussian_filter(values.astype(np.float64), sigma=smooth_sigma)
    maxima = mask & (smoothed == ndimage.maximum_filter(smoothed, size=size, mode="nearest")) & (smoothed >= threshold)
    labels, count = ndimage.label(maxima)
    if count == 0:
        return 0, math.nan, math.nan
    centers = np.asarray(ndimage.center_of_mass(smoothed, labels, range(1, count + 1)), dtype=float)
    strengths = np.asarray(ndimage.maximum(smoothed, labels, range(1, count + 1)), dtype=float)
    order = np.argsort(strengths)[::-1]
    strongest = float(strengths[order[0]])
    separation = math.nan
    if count >= 2:
        separation = float(np.linalg.norm((centers[order[0]] - centers[order[1]]) * spacing))
    return int(count), strongest, separation


def _shape_features(mask: np.ndarray, offset: np.ndarray, spacing: np.ndarray, config: PhenotypeInvestigationConfig) -> dict[str, object]:
    local = np.argwhere(mask)
    count = int(len(local))
    global_voxel = local + offset[None, :]
    physical = global_voxel.astype(float) * spacing[None, :]
    centroid_voxel = global_voxel.mean(axis=0)
    centroid_um = physical.mean(axis=0)
    minimum = global_voxel.min(axis=0)
    maximum = global_voxel.max(axis=0)
    bbox_voxels = maximum - minimum + 1
    bbox_um = bbox_voxels * spacing
    volume_um3 = count * float(np.prod(spacing))
    bbox_volume = int(np.prod(bbox_voxels))
    centered = physical - centroid_um
    if count >= 3:
        eigvals = np.sort(np.linalg.eigvalsh(np.cov(centered.T)))[::-1]
    else:
        eigvals = np.zeros(3)
    axes = np.sqrt(np.maximum(eigvals, 0))
    area = _surface_area(mask, spacing)
    equivalent_radius = ((3 * volume_um3) / (4 * math.pi)) ** (1 / 3) if volume_um3 > 0 else math.nan
    sphericity = ((math.pi ** (1 / 3)) * ((6 * volume_um3) ** (2 / 3)) / area) if area > EPS else math.nan
    convex_volume = solidity = math.nan
    if count >= 4:
        try:
            convex_volume = float(ConvexHull(physical).volume)
            solidity = float(volume_um3 / convex_volume) if convex_volume > EPS else math.nan
        except Exception:
            pass
    distance = ndimage.distance_transform_edt(mask, sampling=spacing)
    max_distance = float(distance.max())
    peak_threshold = config.geometric_peak_threshold_fraction * max_distance
    geometric_peak_count, _, geometric_peak_separation = _local_maxima_count(
        distance, mask, spacing, config.internal_peak_minimum_distance_um, peak_threshold
    )
    return {
        "volume_voxels": count,
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
        "bbox_volume_voxels": bbox_volume,
        "extent": float(count / max(bbox_volume, 1)),
        "equivalent_radius_um": float(equivalent_radius),
        "axis_major_um": float(axes[0]),
        "axis_middle_um": float(axes[1]),
        "axis_minor_um": float(axes[2]),
        "elongation": float(axes[0] / axes[1]) if axes[1] > EPS else math.nan,
        "flatness": float(axes[1] / axes[2]) if axes[2] > EPS else math.nan,
        "anisotropy": float(axes[0] / axes[2]) if axes[2] > EPS else math.nan,
        "surface_area_um2": area,
        "surface_to_volume_ratio": float(area / volume_um3) if volume_um3 > EPS else math.nan,
        "sphericity": float(sphericity),
        "convex_volume_um3": convex_volume,
        "solidity": solidity,
        "compactness": float((volume_um3 ** (2 / 3)) / area) if area > EPS else math.nan,
        "edt_mean_um": float(distance[mask].mean()),
        "edt_max_um": max_distance,
        "geometric_peak_count": geometric_peak_count,
        "geometric_peak_separation_um": geometric_peak_separation,
    }


def _intensity_features(image: np.ndarray, labels: np.ndarray, mask: np.ndarray, cell_id: int, spacing: np.ndarray, config: PhenotypeInvestigationConfig, frame_foreground_median: float) -> dict[str, object]:
    values = image[mask].astype(np.float64, copy=False)
    stats = summarize_values(values)
    distance = ndimage.distance_transform_edt(mask, sampling=spacing)
    max_distance = float(distance.max())
    normalized = distance / max_distance if max_distance > EPS else np.zeros_like(distance)
    core = mask & (normalized >= config.intensity_core_fraction)
    middle = mask & (normalized >= config.intensity_middle_fraction) & (normalized < config.intensity_core_fraction)
    outer = mask & (normalized < config.intensity_middle_fraction)
    if not np.any(core):
        core = mask.copy()
    if not np.any(middle):
        middle = mask.copy()
    if not np.any(outer):
        outer = mask.copy()
    core_stats = summarize_values(image[core])
    middle_stats = summarize_values(image[middle])
    outer_stats = summarize_values(image[outer])

    outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
    background_shell = (
        (outside >= config.background_shell_inner_um)
        & (outside <= config.background_shell_outer_um)
        & (labels == 0)
    )
    background = float(np.median(image[background_shell])) if np.any(background_shell) else math.nan
    corrected = values - background if math.isfinite(background) else np.full_like(values, np.nan)

    local_coords = np.argwhere(mask).astype(float)
    geometric_centroid = local_coords.mean(axis=0)
    shifted = values - float(np.percentile(values, 10))
    weights = np.maximum(shifted, 0)
    if float(weights.sum()) > EPS:
        intensity_centroid = np.average(local_coords, axis=0, weights=weights)
        centroid_offset = float(np.linalg.norm((intensity_centroid - geometric_centroid) * spacing))
    else:
        intensity_centroid = geometric_centroid
        centroid_offset = 0.0
    equivalent_radius = ((3 * mask.sum() * np.prod(spacing)) / (4 * math.pi)) ** (1 / 3)
    brightest_local = np.asarray(np.unravel_index(int(np.argmax(np.where(mask, image, -np.inf))), image.shape), dtype=float)
    brightest_offset = float(np.linalg.norm((brightest_local - geometric_centroid) * spacing))

    threshold = float(np.percentile(values, config.internal_peak_threshold_percentile))
    bright = mask & (image >= threshold)
    bright_labels, bright_count = ndimage.label(bright)
    bright_sizes = np.asarray(ndimage.sum(bright, bright_labels, range(1, bright_count + 1)), dtype=float) if bright_count else np.asarray([])
    peak_count, strongest_peak, peak_separation = _local_maxima_count(
        image, mask, spacing, config.internal_peak_minimum_distance_um, threshold
    )

    smooth = ndimage.gaussian_filter(image.astype(np.float64), sigma=tuple(max(0.35, 0.5 / s) for s in spacing))
    gradients = np.gradient(smooth, *spacing)
    gradient_magnitude = np.sqrt(sum(component ** 2 for component in gradients))
    laplacian = ndimage.gaussian_laplace(image.astype(np.float64), sigma=tuple(max(0.35, 0.5 / s) for s in spacing))
    gradient_stats = summarize_values(gradient_magnitude[mask])
    laplace_values = laplacian[mask]

    return {
        **_prefixed("mask", stats),
        **_prefixed("core", core_stats),
        **_prefixed("middle", middle_stats),
        **_prefixed("outer", outer_stats),
        "core_to_outer_mean_ratio": float(core_stats["mean"] / outer_stats["mean"]) if abs(float(outer_stats["mean"])) > EPS else math.nan,
        "core_to_middle_mean_ratio": float(core_stats["mean"] / middle_stats["mean"]) if abs(float(middle_stats["mean"])) > EPS else math.nan,
        "radial_mean_range": float(max(core_stats["mean"], middle_stats["mean"], outer_stats["mean"]) - min(core_stats["mean"], middle_stats["mean"], outer_stats["mean"])),
        "background_median": background,
        "background_corrected_mean": float(np.nanmean(corrected)) if np.isfinite(corrected).any() else math.nan,
        "background_corrected_sum": float(np.nansum(corrected)) if np.isfinite(corrected).any() else math.nan,
        "mask_mean_frame_ratio": float(stats["mean"] / frame_foreground_median) if abs(frame_foreground_median) > EPS else math.nan,
        "core_mean_frame_ratio": float(core_stats["mean"] / frame_foreground_median) if abs(frame_foreground_median) > EPS else math.nan,
        "intensity_centroid_offset_um": centroid_offset,
        "intensity_centroid_offset_radius_ratio": float(centroid_offset / equivalent_radius) if equivalent_radius > EPS else math.nan,
        "brightest_voxel_offset_um": brightest_offset,
        "bright_region_count": int(bright_count),
        "largest_bright_region_fraction": float(bright_sizes.max() / mask.sum()) if bright_sizes.size else 0.0,
        "second_bright_region_fraction": float(np.sort(bright_sizes)[-2] / mask.sum()) if bright_sizes.size >= 2 else 0.0,
        "internal_peak_count": peak_count,
        "strongest_internal_peak": strongest_peak,
        "two_strongest_peak_separation_um": peak_separation,
        "gradient_mean": float(gradient_stats["mean"]),
        "gradient_p90": float(gradient_stats["p90"]),
        "gradient_p95": float(gradient_stats["p95"]),
        "laplacian_abs_mean": float(np.mean(np.abs(laplace_values))),
        "laplacian_energy": float(np.mean(laplace_values ** 2)),
    }


def _frame_reference(image: np.ndarray, labels: np.ndarray) -> float:
    foreground = labels > 0
    return float(np.median(image[foreground])) if np.any(foreground) else math.nan


def extract_frame_features(artifacts: FrameArtifacts, repository: RepositoryData, config: PhenotypeInvestigationConfig) -> pd.DataFrame:
    """Extract a wide feature row for every segmented cell in one frame."""
    labels = artifacts.instance_labels
    spacing = np.asarray(config.voxel_size_zyx_um, dtype=float)
    object_slices = ndimage.find_objects(labels)
    raw_reference = _frame_reference(artifacts.raw, labels)
    prep_reference = _frame_reference(artifacts.preprocessed, labels)
    records: list[dict[str, object]] = []

    ids = np.unique(labels)
    ids = ids[ids > 0].astype(int)
    for cell_id in ids:
        base_slice = object_slices[cell_id - 1] if cell_id - 1 < len(object_slices) else None
        if base_slice is None:
            continue
        margin = np.ceil(config.background_shell_outer_um / spacing).astype(int) + 1
        start = np.asarray([part.start for part in base_slice], dtype=int)
        stop = np.asarray([part.stop for part in base_slice], dtype=int)
        expanded_start = np.maximum(start - margin, 0)
        expanded_stop = np.minimum(stop + margin, np.asarray(labels.shape))
        expanded = tuple(slice(int(a), int(b)) for a, b in zip(expanded_start, expanded_stop))
        label_crop = np.asarray(labels[expanded])
        mask = label_crop == cell_id
        raw_crop = np.asarray(artifacts.raw[expanded], dtype=np.float64)
        prep_crop = np.asarray(artifacts.preprocessed[expanded], dtype=np.float64)

        shape = _shape_features(mask, expanded_start, spacing, config)
        raw = _intensity_features(raw_crop, label_crop, mask, cell_id, spacing, config, raw_reference)
        prep = _intensity_features(prep_crop, label_crop, mask, cell_id, spacing, config, prep_reference)
        track = repository.track_metadata(artifacts.frame, cell_id)
        centroid_um = np.asarray([shape["centroid_z_um"], shape["centroid_y_um"], shape["centroid_x_um"]])
        frame_extent_um = np.asarray(labels.shape, dtype=float) * spacing
        boundary_distance = float(np.min(np.concatenate([centroid_um, frame_extent_um - centroid_um])))
        lineage_role = repository.known_lineage_role(track["track_id"])
        event_overlap = repository.overlaps_segmentation_event(artifacts.frame, cell_id, track["track_id"])

        record: dict[str, object] = {
            "sample_id": artifacts.sample_id,
            "frame": artifacts.frame,
            "cell_id": int(cell_id),
            **track,
            "boundary_distance_um": boundary_distance,
            "is_boundary": boundary_distance <= config.boundary_margin_um,
            "known_lineage_role": lineage_role,
            "is_known_lineage": bool(lineage_role),
            "overlaps_segmentation_event": event_overlap,
            **shape,
            **_prefixed("raw", raw),
            **_prefixed("preprocessed", prep),
            "frame_raw_foreground_median": raw_reference,
            "frame_preprocessed_foreground_median": prep_reference,
        }
        saved = saved_cell_row(artifacts.cells, cell_id)
        if saved is not None:
            for name, value in saved.items():
                if name == "cell_id" or isinstance(value, (str, bytes)):
                    continue
                try:
                    record[f"saved_{name}"] = float(value)
                except (TypeError, ValueError):
                    pass
        records.append(record)
    return pd.DataFrame(records).sort_values("cell_id").reset_index(drop=True)
