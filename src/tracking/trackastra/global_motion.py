"""Bootstrap global-motion estimation from first-pass Trackastra links."""

from __future__ import annotations

import math

import numpy as np

from .config import GlobalMotionConfig, GlobalMotionEstimate


class GlobalMotionEstimationError(RuntimeError):
    """Raised when one frame transition has too few reliable pass-1 links."""


def _robust_median_displacement(
    vectors_zyx: np.ndarray,
    *,
    config: GlobalMotionConfig,
) -> tuple[np.ndarray, int, float, float, float]:
    vectors = np.asarray(vectors_zyx, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"Expected displacement array (N,3), got {vectors.shape}")

    minimum_pairs = int(config.minimum_pairs)
    if int(vectors.shape[0]) < minimum_pairs:
        raise GlobalMotionEstimationError(
            f"Only {vectors.shape[0]} unambiguous predicted continuation pairs "
            f"are available; need at least {minimum_pairs}."
        )

    spacing = np.asarray(config.voxel_size_zyx, dtype=np.float64)
    center0 = np.median(vectors, axis=0)
    residual = np.linalg.norm((vectors - center0) * spacing[None, :], axis=1)

    residual_median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - residual_median)))
    robust_sigma = 1.4826 * mad
    gate = max(
        float(config.minimum_residual_gate_physical),
        residual_median + float(config.mad_scale) * robust_sigma,
    )

    keep = residual <= gate
    if int(np.count_nonzero(keep)) < minimum_pairs:
        order = np.argsort(residual)
        keep = np.zeros(len(vectors), dtype=bool)
        keep[order[:minimum_pairs]] = True

    center = np.median(vectors[keep], axis=0)
    final_residual = np.linalg.norm(
        (vectors[keep] - center) * spacing[None, :],
        axis=1,
    )
    median_final = float(np.median(final_residual)) if final_residual.size else math.nan
    p90_final = (
        float(np.percentile(final_residual, 90))
        if final_residual.size
        else math.nan
    )
    return center, int(np.count_nonzero(keep)), float(gate), median_final, p90_final


def estimate_global_motion(
    graph,
    *,
    frame_count: int,
    spatial_shape_zyx: tuple[int, int, int],
    config: GlobalMotionConfig,
) -> GlobalMotionEstimate:
    """Estimate common adjacent-frame translation from pass-1 Trackastra paths.

    Only edges satisfying all of the following are used:
      * t -> t+1
      * source out-degree == 1
      * target in-degree == 1

    Division parents are therefore excluded.
    """
    if int(frame_count) < 2:
        raise GlobalMotionEstimationError(
            "At least two frames are required for global-motion estimation."
        )

    vectors_by_frame: dict[int, list[np.ndarray]] = {
        t: [] for t in range(int(frame_count) - 1)
    }

    for source_id, target_id in graph.edges:
        source = graph.nodes[source_id]
        target = graph.nodes[target_id]
        t0 = int(source["time"])
        t1 = int(target["time"])

        if t1 != t0 + 1:
            continue
        if not (0 <= t0 < int(frame_count) - 1):
            continue
        if int(graph.out_degree(source_id)) != 1:
            continue
        if int(graph.in_degree(target_id)) != 1:
            continue

        left = np.asarray(source["coords"], dtype=np.float64)
        right = np.asarray(target["coords"], dtype=np.float64)
        if left.shape != (3,) or right.shape != (3,):
            continue
        if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
            continue
        vectors_by_frame[t0].append(right - left)

    transitions = int(frame_count) - 1
    pairwise = np.zeros((transitions, 3), dtype=np.float64)
    pair_counts = np.zeros(transitions, dtype=np.int64)
    inlier_counts = np.zeros(transitions, dtype=np.int64)
    gates = np.zeros(transitions, dtype=np.float64)
    medians = np.zeros(transitions, dtype=np.float64)
    p90s = np.zeros(transitions, dtype=np.float64)

    for t in range(transitions):
        vectors = np.asarray(vectors_by_frame[t], dtype=np.float64)
        if vectors.size == 0:
            vectors = np.empty((0, 3), dtype=np.float64)
        pair_counts[t] = int(len(vectors))
        try:
            center, inliers, gate, median_residual, p90_residual = (
                _robust_median_displacement(vectors, config=config)
            )
        except Exception as exc:
            raise GlobalMotionEstimationError(
                f"Could not estimate bootstrap global motion for "
                f"t={t:03d}->{t + 1:03d}: {exc}"
            ) from exc

        pairwise[t] = center
        inlier_counts[t] = int(inliers)
        gates[t] = float(gate)
        medians[t] = float(median_residual)
        p90s[t] = float(p90_residual)

    cumulative = np.zeros((int(frame_count), 3), dtype=np.float64)
    for t in range(1, int(frame_count)):
        cumulative[t] = cumulative[t - 1] + pairwise[t - 1]

    align_int = -np.rint(cumulative).astype(np.int64)
    minimum = align_int.min(axis=0)
    maximum = align_int.max(axis=0)
    placement = align_int - minimum[None, :]

    spatial_shape = np.asarray(spatial_shape_zyx, dtype=np.int64)
    canvas_shape = tuple(
        int(v)
        for v in (spatial_shape + (maximum - minimum)).tolist()
    )

    return GlobalMotionEstimate(
        pairwise_float_zyx=pairwise,
        cumulative_float_zyx=cumulative,
        align_int_zyx=align_int,
        placement_zyx=placement,
        canvas_shape_zyx=canvas_shape,
        pair_counts=pair_counts,
        inlier_counts=inlier_counts,
        gate_physical=gates,
        median_residual_physical=medians,
        p90_residual_physical=p90s,
    )


__all__ = ["GlobalMotionEstimationError", "estimate_global_motion"]
