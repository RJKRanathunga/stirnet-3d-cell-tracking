"""Statistical primitives used by the investigation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats


EPS = 1e-12


def _finite_1d(values: np.ndarray | Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    return result[np.isfinite(result)]


def median_absolute_deviation(values: np.ndarray | Sequence[float]) -> float:
    data = _finite_1d(values)
    if data.size == 0:
        return math.nan
    center = np.median(data)
    return float(np.median(np.abs(data - center)))


def summarize_values(values: np.ndarray | Sequence[float]) -> dict[str, float | int]:
    """Return ordinary and robust distribution statistics for one voxel set."""
    data = _finite_1d(values)
    if data.size == 0:
        return {
            "voxel_count": 0,
            **{
                name: math.nan
                for name in (
                    "mean", "median", "variance", "std", "min", "max", "sum",
                    "p01", "p05", "p10", "p25", "p75", "p90", "p95", "p99",
                    "iqr", "mad", "cv", "range", "p95_p05_range",
                    "skewness", "kurtosis",
                )
            },
        }

    q01, q05, q10, q25, q75, q90, q95, q99 = np.percentile(
        data, [1, 5, 10, 25, 75, 90, 95, 99]
    )
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=0))
    minimum = float(np.min(data))
    maximum = float(np.max(data))

    if data.size >= 3 and std > EPS:
        skewness = float(stats.skew(data, bias=False))
    else:
        skewness = 0.0

    if data.size >= 4 and std > EPS:
        kurtosis = float(stats.kurtosis(data, fisher=True, bias=False))
    else:
        kurtosis = 0.0

    return {
        "voxel_count": int(data.size),
        "mean": mean,
        "median": float(np.median(data)),
        "variance": float(np.var(data, ddof=0)),
        "std": std,
        "min": minimum,
        "max": maximum,
        "sum": float(np.sum(data)),
        "p01": float(q01),
        "p05": float(q05),
        "p10": float(q10),
        "p25": float(q25),
        "p75": float(q75),
        "p90": float(q90),
        "p95": float(q95),
        "p99": float(q99),
        "iqr": float(q75 - q25),
        "mad": median_absolute_deviation(data),
        "cv": float(std / abs(mean)) if abs(mean) > EPS else math.nan,
        "range": float(maximum - minimum),
        "p95_p05_range": float(q95 - q05),
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def normalized_relative_difference(first: float, second: float) -> float:
    """Symmetric relative difference used by the current Stage 7 feature cost."""
    if not np.isfinite(first) or not np.isfinite(second):
        return 1.0
    denominator = max(abs(float(first)), abs(float(second)), EPS)
    return float(abs(float(first) - float(second)) / denominator)


def temporal_summary(
    frames: np.ndarray | Sequence[int],
    values: np.ndarray | Sequence[float],
) -> dict[str, float | int]:
    """Summarize temporal stability, trend, and detrended residual variation."""
    x = _finite_1d(frames)
    y = _finite_1d(values)
    if x.size != y.size:
        raise ValueError("frames and values must have equal finite lengths")
    if y.size == 0:
        return {
            "frame_count": 0,
            "temporal_mean": math.nan,
            "temporal_std": math.nan,
            "temporal_cv": math.nan,
            "median_absolute_delta": math.nan,
            "maximum_absolute_delta": math.nan,
            "linear_slope": math.nan,
            "linear_intercept": math.nan,
            "detrended_std": math.nan,
            "detrended_cv": math.nan,
        }

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    mean = float(np.mean(y))
    std = float(np.std(y, ddof=0))
    deltas = np.abs(np.diff(y))

    if y.size >= 2 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
    else:
        slope = 0.0
        intercept = mean
        residual = y - mean

    detrended_std = float(np.std(residual, ddof=0))
    return {
        "frame_count": int(y.size),
        "temporal_mean": mean,
        "temporal_std": std,
        "temporal_cv": float(std / abs(mean)) if abs(mean) > EPS else math.nan,
        "median_absolute_delta": (
            float(np.median(deltas)) if deltas.size else 0.0
        ),
        "maximum_absolute_delta": (
            float(np.max(deltas)) if deltas.size else 0.0
        ),
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
        "detrended_std": detrended_std,
        "detrended_cv": (
            float(detrended_std / abs(mean)) if abs(mean) > EPS else math.nan
        ),
    }


def paired_summary(
    reference: np.ndarray | Sequence[float],
    candidate: np.ndarray | Sequence[float],
) -> dict[str, float | int]:
    """Return paired agreement statistics for two measurements of the same cells."""
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size == 0:
        return {
            "pair_count": 0,
            "pearson_r": math.nan,
            "spearman_r": math.nan,
            "mean_bias": math.nan,
            "difference_std": math.nan,
            "loa_lower": math.nan,
            "loa_upper": math.nan,
            "mean_absolute_difference": math.nan,
            "median_absolute_difference": math.nan,
            "median_relative_difference": math.nan,
        }

    difference = right - left
    bias = float(np.mean(difference))
    difference_std = float(np.std(difference, ddof=0))

    if left.size >= 2 and np.std(left) > EPS and np.std(right) > EPS:
        pearson = float(stats.pearsonr(left, right).statistic)
        spearman = float(stats.spearmanr(left, right).statistic)
    else:
        pearson = math.nan
        spearman = math.nan

    relative = np.abs(difference) / np.maximum(
        np.maximum(np.abs(left), np.abs(right)), EPS
    )
    return {
        "pair_count": int(left.size),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "mean_bias": bias,
        "difference_std": difference_std,
        "loa_lower": float(bias - 1.96 * difference_std),
        "loa_upper": float(bias + 1.96 * difference_std),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "median_absolute_difference": float(np.median(np.abs(difference))),
        "median_relative_difference": float(np.median(relative)),
    }
