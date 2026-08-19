from __future__ import annotations

"""Validate, visualize, and persist the fixed STIR-Net geometry-debug sample.

Stage 00 now chooses the exact scene that Stage 01 will train on. By default
the saved scene is 32 x 128 x 128 = 524,288 voxels (~0.52 M) and must contain:

* one visibly connected current/noisy merge involving at least two GT cells;
* one clean one-to-one "free" GT cell;
* all three required GT cells fully inside the crop.

Production geometry targets and the production five-channel spatial input are
built with a larger CPU-only context and then cropped back to this fixed scene.
The resulting sample is saved by default to:

    data/learned/stirnet/debug_crop.pt

Run from the repository root:

    python investigations/stirnet/00_GT_validation.py

Useful options:

    python investigations/stirnet/00_GT_validation.py --list-candidates
    python investigations/stirnet/00_GT_validation.py --candidate-index 1
    python investigations/stirnet/00_GT_validation.py --exclude-merge-source 4
    python investigations/stirnet/00_GT_validation.py --no-napari
    python investigations/stirnet/00_GT_validation.py --no-save
    python investigations/stirnet/00_GT_validation.py --voxel-budget 524288
    python investigations/stirnet/00_GT_validation.py --crop-shape 32 128 128
    python investigations/stirnet/00_GT_validation.py --data-dir <path>
    python investigations/stirnet/00_GT_validation.py --output <path>
"""

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------------

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from learned.stirnet.data.sample_builder import (
    SPATIAL_CHANNEL_NAMES,
    build_spatial_channels,
)
from learned.stirnet.data.targets import estimate_model_dref_um
from learned.stirnet.model.config import GeometryConfig
from learned.stirnet.model.geometry.targets import (
    GeometryTargets,
    build_geometry_targets,
)


# ---------------------------------------------------------------------------
# Defaults: intentionally match the current first-overfit investigation scene
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "first_overfit"
    / "BlastoSPIM1_F22_030_034"
)
DEFAULT_TIME_INDEX = 2
# Stage 1 should stay near 0.52 M voxels, but the Z/Y/X aspect ratio is chosen
# automatically so the required merge + free cell can actually fit.
DEFAULT_VOXEL_BUDGET = 524_288
DEFAULT_SHAPE_MULTIPLE = 8
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "debug_crop.pt"
)
DEFAULT_CONTEXT_MARGIN_DREF = 3.0
DEFAULT_VECTOR_STRIDE = (2, 10, 10)
DEFAULT_MAX_VECTORS = 4000
DEFAULT_FLOW_VECTOR_LENGTH_UM = 4.0
MAX_FREE_CANDIDATES_PER_MERGE_PAIR = 24
DEFAULT_MAX_CANDIDATES = 12


@dataclass(frozen=True)
class SourceComponent:
    source_id: int
    component_id: int
    low_zyx: tuple[int, int, int]
    high_zyx: tuple[int, int, int]
    meaningful_gt_ids: tuple[int, ...]
    overlap_voxels: tuple[tuple[int, int], ...]
    overlap_gt_fraction: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class Scene:
    current_full: np.ndarray
    gt_full: np.ndarray
    raw_full: np.ndarray
    marker_full: np.ndarray
    spacing_zyx_um: np.ndarray
    dref_um: float
    core_slices: tuple[slice, slice, slice]
    build_slices: tuple[slice, slice, slice]
    selection: dict


# ---------------------------------------------------------------------------
# Loading / crop selection
# ---------------------------------------------------------------------------


def _slice_shape(
    slices: tuple[slice, slice, slice],
) -> tuple[int, int, int]:
    return tuple(int(s.stop) - int(s.start) for s in slices)


def _slice_pairs(
    slices: tuple[slice, slice, slice],
) -> list[list[int]]:
    return [[int(s.start), int(s.stop)] for s in slices]


def _bounded_crop_from_lower(
    shape: tuple[int, int, int],
    lower_zyx: np.ndarray,
    crop_shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    shape_arr = np.asarray(shape, dtype=np.int64)
    size = np.asarray(crop_shape, dtype=np.int64)
    if np.any(size > shape_arr):
        raise ValueError(
            f"Requested crop {tuple(size)} exceeds scene shape {tuple(shape_arr)}"
        )
    lower = np.asarray(lower_zyx, dtype=np.int64)
    lower = np.maximum(lower, 0)
    lower = np.minimum(lower, shape_arr - size)
    upper = lower + size
    return tuple(
        slice(int(lo), int(hi)) for lo, hi in zip(lower, upper)
    )


def _hard_interfaces(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Hard GT cell-background and direct GT cell-cell interfaces."""
    surface = np.zeros_like(labels, dtype=bool)
    separator = np.zeros_like(labels, dtype=bool)
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        a = labels[tuple(lo)]
        b = labels[tuple(hi)]
        different = a != b
        surf = different & ((a == 0) ^ (b == 0))
        sep = different & (a > 0) & (b > 0)
        surface[tuple(lo)] |= surf
        surface[tuple(hi)] |= surf
        separator[tuple(lo)] |= sep
        separator[tuple(hi)] |= sep
    return surface, separator


def _instance_boxes(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_slices = ndi.find_objects(labels)
    ids: list[int] = []
    lows: list[list[int]] = []
    highs: list[list[int]] = []
    for instance_id, bbox in enumerate(object_slices, 1):
        if bbox is None:
            continue
        ids.append(instance_id)
        lows.append([int(s.start) for s in bbox])
        highs.append([int(s.stop) for s in bbox])
    if not ids:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0, 3), dtype=np.int64),
        )
    return (
        np.asarray(ids, dtype=np.int64),
        np.asarray(lows, dtype=np.int64),
        np.asarray(highs, dtype=np.int64),
    )


def _box_lookup(
    labels: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    ids, lows, highs = _instance_boxes(labels)
    return {
        int(instance_id): (low.copy(), high.copy())
        for instance_id, low, high in zip(ids, lows, highs)
    }


def _complete_and_partial_ids(
    labels: np.ndarray,
    crop: tuple[slice, slice, slice],
) -> tuple[list[int], list[int]]:
    ids, lows, highs = _instance_boxes(labels)
    crop_low = np.asarray([s.start for s in crop], dtype=np.int64)
    crop_high = np.asarray([s.stop for s in crop], dtype=np.int64)

    intersects = np.all(highs > crop_low[None], axis=1) & np.all(
        lows < crop_high[None], axis=1
    )
    complete = (
        intersects
        & np.all(lows >= crop_low[None], axis=1)
        & np.all(highs <= crop_high[None], axis=1)
    )
    partial = intersects & ~complete
    return (
        [int(x) for x in ids[complete].tolist()],
        [int(x) for x in ids[partial].tolist()],
    )


def _source_components(
    gt: np.ndarray,
    current: np.ndarray,
    geometry_cfg: GeometryConfig,
) -> list[SourceComponent]:
    """Describe connected current-source components and meaningful GT overlaps.

    The overlap rule mirrors the source-conditioned separator defaults.
    """
    gt_positive = gt[gt > 0]
    if gt_positive.size == 0:
        return []

    gt_ids, gt_counts = np.unique(gt_positive, return_counts=True)
    gt_count_lookup = {
        int(instance_id): int(count)
        for instance_id, count in zip(gt_ids.tolist(), gt_counts.tolist())
    }

    source_slices = ndi.find_objects(current)
    connectivity = ndi.generate_binary_structure(3, 1)
    records: list[SourceComponent] = []

    for source_id, source_bbox in enumerate(source_slices, 1):
        if source_bbox is None:
            continue

        local_source = current[source_bbox] == source_id
        if not local_source.any():
            continue

        components, count = ndi.label(
            local_source, structure=connectivity
        )
        component_slices = ndi.find_objects(components)
        source_origin = np.asarray(
            [int(s.start) for s in source_bbox], dtype=np.int64
        )

        for component_id in range(1, count + 1):
            local_bbox = component_slices[component_id - 1]
            if local_bbox is None:
                continue

            component_mask = components[local_bbox] == component_id
            low = source_origin + np.asarray(
                [int(s.start) for s in local_bbox], dtype=np.int64
            )
            high = source_origin + np.asarray(
                [int(s.stop) for s in local_bbox], dtype=np.int64
            )
            global_bbox = tuple(
                slice(int(lo), int(hi))
                for lo, hi in zip(low, high)
            )

            local_gt = gt[global_bbox]
            overlaps = local_gt[component_mask]
            overlaps = overlaps[overlaps > 0]

            overlap_voxels: list[tuple[int, int]] = []
            overlap_fractions: list[tuple[int, float]] = []
            meaningful: list[int] = []

            if overlaps.size:
                candidate_ids, counts = np.unique(
                    overlaps, return_counts=True
                )
                for candidate_id, overlap_count in zip(
                    candidate_ids.tolist(), counts.tolist()
                ):
                    gt_count = gt_count_lookup.get(int(candidate_id), 0)
                    fraction = (
                        float(overlap_count) / float(gt_count)
                        if gt_count > 0
                        else 0.0
                    )
                    overlap_voxels.append(
                        (int(candidate_id), int(overlap_count))
                    )
                    overlap_fractions.append(
                        (int(candidate_id), float(fraction))
                    )
                    if (
                        int(overlap_count)
                        >= geometry_cfg.separator_source_min_overlap_voxels
                        and fraction
                        >= geometry_cfg.separator_source_min_gt_fraction
                    ):
                        meaningful.append(int(candidate_id))

            records.append(
                SourceComponent(
                    source_id=int(source_id),
                    component_id=int(component_id),
                    low_zyx=tuple(int(x) for x in low),
                    high_zyx=tuple(int(x) for x in high),
                    meaningful_gt_ids=tuple(sorted(meaningful)),
                    overlap_voxels=tuple(overlap_voxels),
                    overlap_gt_fraction=tuple(overlap_fractions),
                )
            )

    return records


def _clean_free_gt_ids(
    components: list[SourceComponent],
) -> list[int]:
    """GT IDs with exactly one meaningful one-to-one current component."""
    memberships: dict[int, list[int]] = {}
    for row_index, component in enumerate(components):
        for gt_id in component.meaningful_gt_ids:
            memberships.setdefault(int(gt_id), []).append(row_index)

    result: list[int] = []
    for gt_id, rows in memberships.items():
        if len(rows) != 1:
            continue
        component = components[rows[0]]
        if component.meaningful_gt_ids == (gt_id,):
            result.append(int(gt_id))
    return sorted(result)


def _visible_merge_component_mask(
    gt_crop: np.ndarray,
    current_crop: np.ndarray,
    source_id: int,
    merge_gt_ids: tuple[int, int],
) -> np.ndarray | None:
    """Find a crop-local connected current component overlapping both GT cells."""
    source = current_crop == int(source_id)
    if not source.any():
        return None

    labeled, count = ndi.label(
        source, structure=ndi.generate_binary_structure(3, 1)
    )
    required = set(int(x) for x in merge_gt_ids)
    for component_id in range(1, count + 1):
        mask = labeled == component_id
        gt_ids = set(
            int(x) for x in np.unique(gt_crop[mask]) if int(x) > 0
        )
        if required.issubset(gt_ids):
            return mask
    return None


def _candidate_lowers_for_union(
    union_low: np.ndarray,
    union_high: np.ndarray,
    scene_shape: np.ndarray,
    crop_shape: np.ndarray,
) -> list[np.ndarray]:
    """Feasible crop positions that fully contain the required GT union."""
    lower_min = np.maximum(union_high - crop_shape, 0)
    lower_max = np.minimum(union_low, scene_shape - crop_shape)
    if np.any(lower_min > lower_max):
        return []

    axis_options: list[list[int]] = []
    for lo, hi in zip(lower_min.tolist(), lower_max.tolist()):
        mid = int(round(0.5 * (lo + hi)))
        axis_options.append(sorted(set((int(lo), mid, int(hi)))))

    return [
        np.asarray(values, dtype=np.int64)
        for values in itertools.product(*axis_options)
    ]


def _minimum_required_margin_fraction(
    crop: tuple[slice, slice, slice],
    required_ids: tuple[int, int, int],
    gt_boxes: dict[int, tuple[np.ndarray, np.ndarray]],
) -> float:
    crop_low = np.asarray(
        [s.start for s in crop], dtype=np.float32
    )
    crop_high = np.asarray(
        [s.stop for s in crop], dtype=np.float32
    )
    crop_size = crop_high - crop_low

    margins: list[float] = []
    for gt_id in required_ids:
        low, high = gt_boxes[int(gt_id)]
        left = low.astype(np.float32) - crop_low
        right = crop_high - high.astype(np.float32)
        normalized = np.minimum(left, right) / np.maximum(
            crop_size, 1
        )
        margins.append(float(normalized.min()))
    return min(margins) if margins else 0.0



def _candidate_crop_shapes(
    required_extent: np.ndarray,
    scene_shape: np.ndarray,
    spacing_um: np.ndarray,
    voxel_budget: int,
    *,
    multiple: int = DEFAULT_SHAPE_MULTIPLE,
    maximum: int = 8,
) -> list[np.ndarray]:
    """Return high-value crop shapes that fit the required union under budget.

    Shapes are multiples of 8 so the three-level spatial encoder/decoder has a
    clean tensor geometry.  We maximize use of the voxel budget, preserve margin
    around the required objects, and mildly prefer physically balanced fields of
    view.  Y/X are allowed to be asymmetric.
    """
    required_extent = np.asarray(required_extent, dtype=np.int64)
    scene_shape = np.asarray(scene_shape, dtype=np.int64)
    spacing_um = np.asarray(spacing_um, dtype=np.float64)

    if voxel_budget < 1:
        raise ValueError("voxel_budget must be positive")
    if multiple < 1:
        raise ValueError("multiple must be positive")

    minimum = (
        np.ceil(required_extent / float(multiple)).astype(np.int64) * multiple
    )
    maximum_shape = (scene_shape // multiple) * multiple

    if np.any(minimum > maximum_shape):
        return []
    if int(np.prod(minimum)) > voxel_budget:
        return []

    rows: list[tuple[float, np.ndarray]] = []
    z_values = range(int(minimum[0]), int(maximum_shape[0]) + 1, multiple)
    y_values = range(int(minimum[1]), int(maximum_shape[1]) + 1, multiple)

    for z in z_values:
        for y in y_values:
            base = z * y
            if base <= 0:
                continue
            max_x_budget = (voxel_budget // base // multiple) * multiple
            x = min(int(maximum_shape[2]), int(max_x_budget))
            if x < int(minimum[2]):
                continue

            # Also evaluate the largest legal X and a slightly smaller one. The
            # smaller option can reduce partial peripheral cells without wasting
            # much of the memory budget.
            x_options = {x}
            if x - multiple >= int(minimum[2]):
                x_options.add(x - multiple)

            for x_value in x_options:
                shape = np.asarray((z, y, x_value), dtype=np.int64)
                voxels = int(np.prod(shape))
                if voxels > voxel_budget:
                    continue

                spare = np.maximum(shape - required_extent, 0).astype(np.float64)
                normalized_margin = float(
                    np.mean(spare / np.maximum(shape, 1))
                )
                utilization = voxels / float(voxel_budget)

                physical = shape.astype(np.float64) * spacing_um
                # Mild preference only: do not override topology/containment.
                physical_log = np.log(np.maximum(physical, 1e-6))
                aspect_penalty = float(np.std(physical_log))

                score = (
                    8.0 * utilization
                    + 2.5 * normalized_margin
                    - 0.35 * aspect_penalty
                )
                rows.append((score, shape))

    rows.sort(key=lambda row: row[0], reverse=True)

    result: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()
    for _, shape in rows:
        key = tuple(int(x) for x in shape)
        if key in seen:
            continue
        seen.add(key)
        result.append(shape)
        if len(result) >= maximum:
            break
    return result


def _preferred_merge_pairs(
    merge_component: SourceComponent,
    gt_centroids: dict[int, np.ndarray],
    spacing_um: np.ndarray,
    *,
    maximum: int = 8,
) -> list[tuple[int, int]]:
    """Prefer simple two-cell merges, otherwise nearby GT pairs within a merge."""
    ids = list(merge_component.meaningful_gt_ids)
    if len(ids) == 2:
        return [(int(ids[0]), int(ids[1]))]

    rows: list[tuple[float, tuple[int, int]]] = []
    for a, b in itertools.combinations(ids, 2):
        if a not in gt_centroids or b not in gt_centroids:
            continue
        distance = float(
            np.linalg.vector_norm(
                (gt_centroids[a] - gt_centroids[b]) * spacing_um
            )
        )
        rows.append((distance, (int(a), int(b))))
    rows.sort(key=lambda row: row[0])
    return [pair for _, pair in rows[:maximum]]


def _preferred_candidate_lowers(
    union_low: np.ndarray,
    union_high: np.ndarray,
    scene_shape: np.ndarray,
    crop_shape: np.ndarray,
) -> list[np.ndarray]:
    """Try a centered feasible placement first, then axis-wise edge alternatives."""
    lower_min = np.maximum(union_high - crop_shape, 0)
    lower_max = np.minimum(union_low, scene_shape - crop_shape)
    if np.any(lower_min > lower_max):
        return []

    center = np.rint(0.5 * (lower_min + lower_max)).astype(np.int64)
    rows = [center]
    for axis in range(3):
        for value in (lower_min[axis], lower_max[axis]):
            candidate = center.copy()
            candidate[axis] = int(value)
            rows.append(candidate)

    result: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()
    for row in rows:
        key = tuple(int(x) for x in row)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _enumerate_debug_crop_candidates(
    gt: np.ndarray,
    current: np.ndarray,
    spacing_um: np.ndarray,
    voxel_budget: int,
    geometry_cfg: GeometryConfig,
    *,
    fixed_crop_shape: tuple[int, int, int] | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    exclude_merge_sources: set[int] | None = None,
) -> list[tuple[tuple[slice, slice, slice], dict]]:
    """Enumerate reusable Stage-1 crop candidates under a fixed voxel budget.

    Each candidate must contain:
      - two complete GT cells belonging to one visible merged current component;
      - one complete clean one-to-one free GT cell.

    By default the crop SHAPE is adaptive while its voxel count stays <= the
    requested budget. ``fixed_crop_shape`` is only an explicit override.
    """
    shape = np.asarray(gt.shape, dtype=np.int64)
    gt_boxes = _box_lookup(gt)
    components = _source_components(gt, current, geometry_cfg)
    exclude_merge_sources = set() if exclude_merge_sources is None else set(int(x) for x in exclude_merge_sources)
    merge_components = [
        row
        for row in components
        if len(row.meaningful_gt_ids) >= 2 and row.source_id not in exclude_merge_sources
    ]
    free_ids = _clean_free_gt_ids(components)

    if not merge_components:
        raise RuntimeError(
            "No current connected source component meaningfully overlaps "
            "2+ GT cells; cannot build the required merge-containing crop."
        )
    if not free_ids:
        raise RuntimeError(
            "No clean one-to-one free GT cell was found in this scene."
        )

    gt_centroids = {
        gt_id: 0.5 * (low.astype(np.float32) + high.astype(np.float32) - 1.0)
        for gt_id, (low, high) in gt_boxes.items()
    }

    fixed_shape_arr: np.ndarray | None = None
    if fixed_crop_shape is not None:
        fixed_shape_arr = np.asarray(fixed_crop_shape, dtype=np.int64)
        if np.any(fixed_shape_arr > shape):
            raise ValueError(
                f"Fixed crop shape {tuple(fixed_shape_arr)} exceeds "
                f"scene shape {tuple(shape)}"
            )
        fixed_voxels = int(np.prod(fixed_shape_arr))
        if fixed_voxels > voxel_budget:
            raise ValueError(
                f"Fixed crop shape {tuple(fixed_shape_arr)} uses "
                f"{fixed_voxels:,} voxels, above budget {voxel_budget:,}"
            )

    rows: list[tuple[float, tuple[slice, slice, slice], dict]] = []
    bbox_feasible_triplets = 0
    minimum_required_voxels_seen: int | None = None
    minimum_required_extent_seen: tuple[int, int, int] | None = None

    for merge_component in merge_components:
        merge_all = merge_component.meaningful_gt_ids
        merge_pairs = _preferred_merge_pairs(
            merge_component,
            gt_centroids,
            spacing_um,
        )

        for a, b in merge_pairs:
            if a not in gt_boxes or b not in gt_boxes:
                continue

            pair_center = 0.5 * (gt_centroids[a] + gt_centroids[b])
            ranked_free = sorted(
                (
                    (
                        float(
                            np.linalg.vector_norm(
                                (gt_centroids[free_id] - pair_center) * spacing_um
                            )
                        ),
                        int(free_id),
                    )
                    for free_id in free_ids
                    if free_id not in (a, b) and free_id in gt_boxes
                ),
                key=lambda row: row[0],
            )[:MAX_FREE_CANDIDATES_PER_MERGE_PAIR]

            for free_distance_um, free_id in ranked_free:
                required = (int(a), int(b), int(free_id))
                lows = np.stack([gt_boxes[x][0] for x in required])
                highs = np.stack([gt_boxes[x][1] for x in required])
                union_low = lows.min(axis=0)
                union_high = highs.max(axis=0)
                required_extent = union_high - union_low

                rounded_minimum = (
                    np.ceil(required_extent / float(DEFAULT_SHAPE_MULTIPLE)).astype(np.int64)
                    * DEFAULT_SHAPE_MULTIPLE
                )
                minimum_voxels = int(np.prod(rounded_minimum))
                if (
                    minimum_required_voxels_seen is None
                    or minimum_voxels < minimum_required_voxels_seen
                ):
                    minimum_required_voxels_seen = minimum_voxels
                    minimum_required_extent_seen = tuple(int(x) for x in required_extent)

                if fixed_shape_arr is not None:
                    crop_shapes = [fixed_shape_arr] if np.all(required_extent <= fixed_shape_arr) else []
                else:
                    crop_shapes = _candidate_crop_shapes(
                        required_extent,
                        shape,
                        spacing_um,
                        voxel_budget,
                    )

                if not crop_shapes:
                    continue
                bbox_feasible_triplets += 1

                for crop_shape in crop_shapes:
                    candidate_lowers = _preferred_candidate_lowers(
                        union_low,
                        union_high,
                        shape,
                        crop_shape,
                    )
                    if not candidate_lowers:
                        continue

                    for lower in candidate_lowers:
                        crop = _bounded_crop_from_lower(
                            tuple(int(x) for x in shape),
                            lower,
                            tuple(int(x) for x in crop_shape),
                        )
                        gt_crop = gt[crop]
                        current_crop = current[crop]

                        merge_mask = _visible_merge_component_mask(
                            gt_crop,
                            current_crop,
                            merge_component.source_id,
                            (int(a), int(b)),
                        )
                        if merge_mask is None:
                            continue

                        complete_ids, partial_ids = _complete_and_partial_ids(gt, crop)
                        complete_set = set(complete_ids)
                        if not set(required).issubset(complete_set):
                            continue

                        margin_fraction = _minimum_required_margin_fraction(
                            crop, required, gt_boxes
                        )
                        merge_fraction = float(merge_mask.mean())
                        crop_voxels = int(np.prod(_slice_shape(crop)))
                        utilization = crop_voxels / float(voxel_budget)

                        score = (
                            -30.0 * len(partial_ids)
                            + 80.0 * margin_fraction
                            + 2.0 * len(complete_ids)
                            + 8.0 * utilization
                            + 4.0 * min(merge_fraction / 0.05, 1.0)
                            - 0.03 * free_distance_um
                            - 0.80 * max(len(merge_all) - 2, 0)
                        )
                        if len(merge_all) == 2:
                            score += 4.0

                        selection = {
                            "score": float(score),
                            "voxel_budget": int(voxel_budget),
                            "budget_utilization": float(utilization),
                            "shape_mode": (
                                "fixed_override" if fixed_shape_arr is not None else "adaptive_under_budget"
                            ),
                            "merge_source_id": int(merge_component.source_id),
                            "merge_source_component_id": int(merge_component.component_id),
                            "merge_gt_ids": [int(a), int(b)],
                            "merge_all_meaningful_gt_ids": [int(x) for x in merge_all],
                            "free_gt_id": int(free_id),
                            "complete_gt_ids": complete_ids,
                            "partial_gt_ids": partial_ids,
                            "required_margin_fraction": float(margin_fraction),
                            "free_distance_to_merge_um": float(free_distance_um),
                            "visible_merge_component_fraction": float(merge_fraction),
                            "required_union_extent_zyx": [int(x) for x in required_extent],
                            "meaningful_overlap_rule": {
                                "min_overlap_voxels": int(
                                    geometry_cfg.separator_source_min_overlap_voxels
                                ),
                                "min_gt_fraction": float(
                                    geometry_cfg.separator_source_min_gt_fraction
                                ),
                            },
                        }
                        rows.append((score, crop, selection))

    if not rows:
        examples = [
            {
                "source_id": row.source_id,
                "component_id": row.component_id,
                "gt_ids": list(row.meaningful_gt_ids),
            }
            for row in merge_components[:10]
        ]
        minimum_note = (
            "unknown"
            if minimum_required_voxels_seen is None
            else (
                f"{minimum_required_voxels_seen:,} voxels for rounded minimum "
                f"extent near {minimum_required_extent_seen}"
            )
        )
        mode = (
            f"fixed shape {tuple(int(x) for x in fixed_shape_arr)}"
            if fixed_shape_arr is not None
            else f"adaptive shapes under {voxel_budget:,} voxels"
        )
        raise RuntimeError(
            "Could not find one visible current-source merge plus one clean "
            f"free cell using {mode}. "
            f"Found {len(merge_components)} merge components, "
            f"{len(free_ids)} clean free cells, and "
            f"{bbox_feasible_triplets} bbox-feasible triplets. "
            f"Smallest rounded required union seen: {minimum_note}. "
            f"Example merges: {examples}."
        )

    # Deduplicate by semantic case: same merge source, same GT pair, same free GT.
    rows.sort(key=lambda row: row[0], reverse=True)
    deduped: list[tuple[tuple[slice, slice, slice], dict]] = []
    seen: set[tuple[int, tuple[int, int], int]] = set()
    for _, crop, selection in rows:
        key = (
            int(selection["merge_source_id"]),
            tuple(int(x) for x in selection["merge_gt_ids"]),
            int(selection["free_gt_id"]),
        )
        if key in seen:
            continue
        seen.add(key)
        selection = dict(selection)
        selection.update(
            {
                "crop_slices_zyx": _slice_pairs(crop),
                "crop_shape_zyx": list(_slice_shape(crop)),
                "voxel_count": int(np.prod(_slice_shape(crop))),
            }
        )
        deduped.append((crop, selection))
        if len(deduped) >= max_candidates:
            break
    
    for index, (_, selection) in enumerate(deduped):
        selection["candidate_index"] = int(index)
    return deduped


def _select_debug_crop(
    gt: np.ndarray,
    current: np.ndarray,
    spacing_um: np.ndarray,
    voxel_budget: int,
    geometry_cfg: GeometryConfig,
    *,
    fixed_crop_shape: tuple[int, int, int] | None = None,
    candidate_index: int = 0,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    exclude_merge_sources: set[int] | None = None,
) -> tuple[tuple[slice, slice, slice], dict, list[dict]]:
    candidates = _enumerate_debug_crop_candidates(
        gt,
        current,
        spacing_um,
        voxel_budget,
        geometry_cfg,
        fixed_crop_shape=fixed_crop_shape,
        max_candidates=max_candidates,
        exclude_merge_sources=exclude_merge_sources,
    )
    if not (0 <= candidate_index < len(candidates)):
        raise IndexError(
            f"candidate-index {candidate_index} is outside [0, {len(candidates) - 1}]"
        )
    crop, selection = candidates[candidate_index]
    selection = dict(selection)
    selection["candidate_count"] = len(candidates)
    summaries = [dict(candidate_selection) for _, candidate_selection in candidates]
    return crop, selection, summaries

def _expand_build_region(
    gt_full: np.ndarray,
    core: tuple[slice, slice, slice],
    spacing_um: np.ndarray,
    dref_um: float,
    context_margin_dref: float,
) -> tuple[slice, slice, slice]:
    """Build targets with context and complete every cell visible in the core.

    This avoids creating artificial target geometry at the displayed crop edge.
    All GT objects that occur in the visible core are included completely in the
    larger build ROI, then the generated targets are cropped back to the core.
    """
    shape = np.asarray(gt_full.shape, dtype=np.int64)
    lower = np.asarray([s.start for s in core], dtype=np.int64)
    upper = np.asarray([s.stop for s in core], dtype=np.int64)

    selected_ids = np.unique(gt_full[core])
    selected_ids = selected_ids[selected_ids > 0]

    object_slices = ndi.find_objects(gt_full)
    for instance_id in selected_ids.tolist():
        index = int(instance_id) - 1
        if index < 0 or index >= len(object_slices):
            continue
        bbox = object_slices[index]
        if bbox is None:
            continue
        lower = np.minimum(lower, np.asarray([s.start for s in bbox], dtype=np.int64))
        upper = np.maximum(upper, np.asarray([s.stop for s in bbox], dtype=np.int64))

    margin_um = max(float(context_margin_dref) * float(dref_um), float(spacing_um.max()))
    margin_vox = np.ceil(margin_um / spacing_um).astype(np.int64)
    lower = np.maximum(lower - margin_vox, 0)
    upper = np.minimum(upper + margin_vox, shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))


def _relative_slices(
    inner: tuple[slice, slice, slice],
    outer: tuple[slice, slice, slice],
) -> tuple[slice, slice, slice]:
    return tuple(
        slice(int(i.start - o.start), int(i.stop - o.start))
        for i, o in zip(inner, outer)
    )



def print_candidate_summaries(candidate_summaries: list[dict]) -> None:
    print("\nCandidate crops (inspect these with --candidate-index)")
    for row in candidate_summaries:
        merge_gt_ids = tuple(int(x) for x in row["merge_gt_ids"])
        print(
            f"  [{int(row['candidate_index'])}] "
            f"source {int(row['merge_source_id'])} GT {merge_gt_ids} "
            f"free {int(row['free_gt_id'])} | "
            f"shape {tuple(int(x) for x in row['crop_shape_zyx'])} | "
            f"partials {len(row['partial_gt_ids'])} | "
            f"score {float(row['score']):.2f}"
        )

def load_scene(args: argparse.Namespace) -> Scene:
    data_dir = args.data_dir.resolve()
    instance_path = data_dir / "instance_movie.npy"
    gt_path = data_dir / "gt_movie.npy"
    metadata_path = data_dir / "metadata.json"
    source_dir = data_dir / "stirnet_source"
    raw_path = source_dir / "raw_norm_target.npy"
    marker_path = source_dir / "marker_heatmap_target.npy"

    required = (
        instance_path,
        gt_path,
        metadata_path,
        raw_path,
        marker_path,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared STIR-Net data is incomplete. Missing:\n  "
            + "\n  ".join(str(path) for path in missing)
        )

    instance_movie = np.load(instance_path, mmap_mode="r")
    gt_movie = np.load(gt_path, mmap_mode="r")
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    spacing = np.asarray(
        metadata["spacing_zyx_um"], dtype=np.float32
    )

    t = int(args.time_index)
    maximum_t = min(len(instance_movie), len(gt_movie)) - 1
    if not (0 <= t <= maximum_t):
        raise IndexError(
            f"time-index {t} is outside [0, {maximum_t}]"
        )

    current_full = np.asarray(instance_movie[t])
    gt_full = np.asarray(gt_movie[t])
    if current_full.shape != gt_full.shape:
        raise RuntimeError(
            f"Current/GT shape mismatch: "
            f"{current_full.shape} vs {gt_full.shape}"
        )

    raw_full = np.load(raw_path, mmap_mode="r")
    marker_full = np.load(marker_path, mmap_mode="r")
    if raw_full.shape != gt_full.shape:
        raise RuntimeError(
            f"Raw/GT shape mismatch: "
            f"{raw_full.shape} vs {gt_full.shape}"
        )
    if marker_full.shape != gt_full.shape:
        raise RuntimeError(
            f"Marker/GT shape mismatch: "
            f"{marker_full.shape} vs {gt_full.shape}"
        )

    if t != DEFAULT_TIME_INDEX:
        print(
            "[WARN] raw_norm_target.npy and marker_heatmap_target.npy "
            "are the prepared target-frame inputs. The selected time "
            f"index is not the default t={DEFAULT_TIME_INDEX}; verify "
            "that the prepared source files belong to this frame."
        )

    dref_um = estimate_model_dref_um(
        current_full,
        tuple(float(value) for value in spacing),
    )

    geometry_cfg = GeometryConfig()
    core, selection, candidate_summaries = _select_debug_crop(
        gt_full,
        current_full,
        spacing,
        int(args.voxel_budget),
        geometry_cfg,
        fixed_crop_shape=(
            None
            if args.crop_shape is None
            else tuple(int(x) for x in args.crop_shape)
        ),
        candidate_index=int(args.candidate_index),
        max_candidates=int(args.max_candidates),
        exclude_merge_sources=set(
            int(x) for x in args.exclude_merge_source
        ),
    )
    build = _expand_build_region(
        gt_full,
        core,
        spacing,
        dref_um,
        args.context_margin_dref,
    )
    selection["build_slices_zyx"] = _slice_pairs(build)
    selection["candidate_summaries"] = candidate_summaries

    return Scene(
        current_full=current_full,
        gt_full=gt_full,
        raw_full=raw_full,
        marker_full=marker_full,
        spacing_zyx_um=spacing,
        dref_um=float(dref_um),
        core_slices=core,
        build_slices=build,
        selection=selection,
    )


# ---------------------------------------------------------------------------
# Production target generation
# ---------------------------------------------------------------------------


def build_targets(scene: Scene) -> GeometryTargets:
    geometry_cfg = GeometryConfig()

    gt_build = np.asarray(
        scene.gt_full[scene.build_slices]
    ).astype(np.int64, copy=True)
    current_build = np.asarray(
        scene.current_full[scene.build_slices]
    ).astype(np.int64, copy=True)

    return build_geometry_targets(
        torch.from_numpy(gt_build),
        torch.from_numpy(scene.spacing_zyx_um.copy()),
        torch.tensor(scene.dref_um, dtype=torch.float32),
        current_labels=torch.from_numpy(current_build),
        sdf_clip_dref=geometry_cfg.sdf_clip_dref,
        sdf_supervision_radius_dref=(
            geometry_cfg.sdf_supervision_radius_dref
        ),
        surface_target_sigma_um=(
            geometry_cfg.surface_target_sigma_um
        ),
        separator_target_sigma_um=(
            geometry_cfg.separator_target_sigma_um
        ),
        separator_source_conditioned=(
            geometry_cfg.separator_source_conditioned
        ),
        separator_source_min_overlap_voxels=(
            geometry_cfg.separator_source_min_overlap_voxels
        ),
        separator_source_min_gt_fraction=(
            geometry_cfg.separator_source_min_gt_fraction
        ),
        device=torch.device("cpu"),
    )


def build_saved_spatial_inputs(scene: Scene) -> np.ndarray:
    """Build production 5-channel input with context, then crop to core."""
    current_build = np.asarray(
        scene.current_full[scene.build_slices]
    ).astype(np.int64, copy=True)
    raw_build = np.asarray(
        scene.raw_full[scene.build_slices]
    ).astype(np.float32, copy=False)
    marker_build = np.asarray(
        scene.marker_full[scene.build_slices]
    ).astype(np.float32, copy=False)

    spatial_build = build_spatial_channels(
        raw_build,
        current_build,
        tuple(float(x) for x in scene.spacing_zyx_um),
        scene.dref_um,
        marker_build,
    )
    relative_core = _relative_slices(
        scene.core_slices, scene.build_slices
    )
    return np.asarray(
        spatial_build[(slice(None), *relative_core)],
        dtype=np.float32,
    ).copy()


def _target_arrays(
    targets: GeometryTargets,
    relative_core: tuple[slice, slice, slice],
) -> dict[str, np.ndarray]:
    s = relative_core
    return {
        "foreground": targets.foreground[0, 0][s].numpy(),
        "surface": targets.surface[0, 0][s].numpy(),
        "separator": targets.separator[0, 0][s].numpy(),
        "sdf": targets.sdf[0, 0][s].numpy(),
        "sdf_valid": targets.sdf_valid[0, 0][s].numpy(),
        "flow": targets.flow[0][(slice(None), *s)].numpy(),
        "centroid_offset": targets.centroid_offset[0][
            (slice(None), *s)
        ].numpy(),
        "seed": targets.seed[0, 0][s].numpy(),
    }


def _cropped_target_tensors(
    targets: GeometryTargets,
    relative_core: tuple[slice, slice, slice],
) -> dict[str, torch.Tensor]:
    """Per-sample targets [C,Z,Y,X], intentionally without batch dim."""
    result: dict[str, torch.Tensor] = {}
    for name, value in targets.__dict__.items():
        result[name] = value[
            0,
            :,
            relative_core[0],
            relative_core[1],
            relative_core[2],
        ].detach().cpu().contiguous()
    return result


# ---------------------------------------------------------------------------
# Numerical validation
# ---------------------------------------------------------------------------


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _print_scalar_stats(name: str, array: np.ndarray) -> None:
    arr = np.asarray(array)
    finite = np.isfinite(arr)
    if arr.size == 0:
        print(f"  {name:18s} empty")
        return
    if not finite.any():
        print(f"  {name:18s} all non-finite")
        return
    values = arr[finite]
    print(
        f"  {name:18s} shape={str(arr.shape):18s} "
        f"min={float(values.min()):9.5f} "
        f"mean={float(values.mean()):9.5f} "
        f"max={float(values.max()):9.5f}"
    )


def validate_targets(
    scene: Scene,
    targets: GeometryTargets,
    arrays: dict[str, np.ndarray],
    spatial_inputs: np.ndarray,
) -> None:
    core = scene.core_slices
    gt = np.asarray(scene.gt_full[core])
    current = np.asarray(scene.current_full[core])
    foreground = gt > 0

    crop_shape = tuple(int(x) for x in gt.shape)
    voxel_count = int(np.prod(crop_shape))

    selection = scene.selection
    merge_gt_ids = tuple(
        int(x) for x in selection["merge_gt_ids"]
    )
    free_gt_id = int(selection["free_gt_id"])
    merge_source_id = int(selection["merge_source_id"])

    complete_ids, partial_ids = _complete_and_partial_ids(
        scene.gt_full, scene.core_slices
    )
    complete_set = set(complete_ids)

    merge_mask = _visible_merge_component_mask(
        gt,
        current,
        merge_source_id,
        merge_gt_ids,
    )

    print("\n" + "=" * 92)
    print("STIR-Net Stage 00 — fixed geometry-debug sample")
    print("=" * 92)
    print(f"Full scene shape       : {tuple(scene.gt_full.shape)}")
    print(f"Saved crop slices      : {scene.core_slices}")
    print(f"Saved crop shape       : {crop_shape}")
    print(
        f"Saved crop voxels      : {voxel_count:,} "
        f"({voxel_count / 1e6:.3f} M)"
    )
    print(f"Target/input build ROI : {scene.build_slices}")
    print(
        "Spacing z,y,x (um)    : "
        f"{tuple(float(x) for x in scene.spacing_zyx_um)}"
    )
    print(f"Model dref (um)        : {scene.dref_um:.5f}")
    print()
    print(
        "Required merge        : "
        f"current source {merge_source_id}, "
        f"GT cells {merge_gt_ids}"
    )
    print(f"Required free cell     : GT cell {free_gt_id}")
    print(f"Complete GT cells      : {complete_ids}")
    print(f"Partially cut GT cells : {partial_ids}")
    print(
        f"Current FG fraction   : "
        f"{float((current > 0).mean()):.4f}"
    )
    print(
        f"GT FG fraction        : "
        f"{float(foreground.mean()):.4f}"
    )

    print("\nSpatial input contract")
    print(f"  shape                 : {tuple(spatial_inputs.shape)}")
    for index, name in enumerate(SPATIAL_CHANNEL_NAMES):
        channel = spatial_inputs[index]
        print(
            f"  [{index}] {name:24s} "
            f"min={float(channel.min()):.5f} "
            f"mean={float(channel.mean()):.5f} "
            f"max={float(channel.max()):.5f}"
        )

    print("\nTarget scalar statistics")
    for name in (
        "foreground",
        "surface",
        "separator",
        "sdf",
        "sdf_valid",
        "seed",
    ):
        _print_scalar_stats(name, arrays[name])
    _print_scalar_stats(
        "flow magnitude",
        np.linalg.vector_norm(arrays["flow"], axis=0),
    )
    _print_scalar_stats(
        "offset magnitude um",
        np.linalg.vector_norm(
            arrays["centroid_offset"] * scene.dref_um,
            axis=0,
        ),
    )

    checks: list[tuple[str, bool, str]] = []

    checks.append(
        (
            "spatial input has 5 channels",
            spatial_inputs.shape == (5, *crop_shape),
            str(spatial_inputs.shape),
        )
    )
    checks.append(
        (
            "required merge GT cells complete",
            set(merge_gt_ids).issubset(complete_set),
            str(merge_gt_ids),
        )
    )
    checks.append(
        (
            "required free GT cell complete",
            free_gt_id in complete_set,
            str(free_gt_id),
        )
    )
    checks.append(
        (
            "merge visibly connected in crop",
            merge_mask is not None,
            f"source={merge_source_id}",
        )
    )

    fg_exact = np.array_equal(
        arrays["foreground"] >= 0.5, foreground
    )
    checks.append(
        (
            "foreground == (GT > 0)",
            fg_exact,
            "exact binary agreement",
        )
    )

    for name in (
        "foreground",
        "surface",
        "separator",
        "sdf",
        "flow",
        "centroid_offset",
        "seed",
    ):
        checks.append(
            (
                f"{name} finite",
                bool(np.isfinite(arrays[name]).all()),
                "no NaN/Inf",
            )
        )

    for name in ("surface", "separator", "seed"):
        arr = arrays[name]
        in_range = bool(
            (arr >= -1e-6).all()
            and (arr <= 1.0 + 1e-6).all()
        )
        checks.append(
            (
                f"{name} in [0,1]",
                in_range,
                f"range=({arr.min():.5f},{arr.max():.5f})",
            )
        )

    sdf = arrays["sdf"]
    if foreground.any():
        checks.append(
            (
                "SDF positive in foreground",
                bool((sdf[foreground] > 0).all()),
                f"min={float(sdf[foreground].min()):.6f}",
            )
        )
        checks.append(
            (
                "SDF-valid covers foreground",
                bool(arrays["sdf_valid"][foreground].all()),
                "all GT voxels supervised",
            )
        )
    if (~foreground).any():
        checks.append(
            (
                "SDF negative in background",
                bool((sdf[~foreground] < 0).all()),
                f"max={float(sdf[~foreground].max()):.6f}",
            )
        )

    hard_surface, hard_gt_separator = _hard_interfaces(gt)
    if hard_surface.any():
        minimum = float(
            arrays["surface"][hard_surface].min()
        )
        checks.append(
            (
                "surface covers hard GT interface",
                minimum > 0.0,
                f"min={minimum:.6f}",
            )
        )
    if hard_gt_separator.any():
        minimum = float(
            arrays["separator"][hard_gt_separator].min()
        )
        checks.append(
            (
                "separator covers direct GT interface",
                minimum > 0.0,
                f"min={minimum:.6f}",
            )
        )

    if merge_mask is not None:
        values = arrays["separator"][merge_mask]
        maximum = float(values.max()) if values.size else 0.0
        checks.append(
            (
                "source-conditioned merge separator exists",
                maximum > 0.0,
                f"max inside selected merge={maximum:.6f}",
            )
        )

    # Seed maxima: acceptance applies to complete cells only.
    gt_build = np.asarray(
        scene.gt_full[scene.build_slices]
    )
    seed_build = targets.seed[0, 0].numpy()
    seed_failures: list[int] = []
    for instance_id in complete_ids:
        values = seed_build[gt_build == instance_id]
        if (
            values.size == 0
            or not np.isclose(
                float(values.max()), 1.0, atol=1e-5
            )
        ):
            seed_failures.append(int(instance_id))
    checks.append(
        (
            "complete cells have seed max 1",
            len(seed_failures) == 0,
            "all complete cells"
            if not seed_failures
            else f"failed IDs={seed_failures}",
        )
    )

    # Exact centroid-offset reconstruction on sparse complete-cell samples.
    build_origin = np.asarray(
        [s.start for s in scene.build_slices],
        dtype=np.float32,
    )
    core_origin = np.asarray(
        [s.start for s in scene.core_slices],
        dtype=np.float32,
    )
    offset_build = targets.centroid_offset[0].numpy()
    max_centroid_error_um = 0.0
    sampled = 0

    for instance_id in complete_ids:
        full_coords = np.argwhere(
            scene.gt_full == instance_id
        )
        local_points = np.argwhere(gt == instance_id)
        if full_coords.size == 0 or local_points.size == 0:
            continue

        expected_centroid_um = (
            full_coords.astype(np.float32).mean(axis=0)
            * scene.spacing_zyx_um
        )
        if len(local_points) > 192:
            step = max(1, len(local_points) // 192)
            local_points = local_points[::step][:192]

        global_points = (
            local_points.astype(np.float32) + core_origin
        )
        build_points = (
            global_points - build_origin
        ).astype(np.int64)
        offset_values = offset_build[
            :,
            build_points[:, 0],
            build_points[:, 1],
            build_points[:, 2],
        ].T
        reconstructed = (
            global_points * scene.spacing_zyx_um[None]
            + offset_values * scene.dref_um
        )
        error = np.linalg.vector_norm(
            reconstructed - expected_centroid_um[None],
            axis=1,
        )
        max_centroid_error_um = max(
            max_centroid_error_um,
            float(error.max()),
        )
        sampled += len(error)

    checks.append(
        (
            "centroid-offset reconstruction",
            max_centroid_error_um <= 1e-3,
            f"max error={max_centroid_error_um:.6g} um "
            f"over {sampled} voxels",
        )
    )

    print("\nContract checks")
    for name, ok, detail in checks:
        print(
            f"  [{_status(ok):4s}] "
            f"{name:43s} {detail}"
        )

    failed = [name for name, ok, _ in checks if not ok]
    print("\nResult")
    if failed:
        print(
            f"  FAIL: {len(failed)} contract check(s) failed."
        )
        for name in failed:
            print(f"    - {name}")
        print(
            "  The crop can still be visualized, but do not "
            "accept it for Stage 1 until the failure is understood."
        )
    else:
        print(
            "  PASS: fixed debug crop and production geometry "
            "targets satisfy all Stage-00 contracts."
        )
    print("=" * 92 + "\n")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_debug_sample(
    output_path: Path,
    scene: Scene,
    targets: GeometryTargets,
    spatial_inputs: np.ndarray,
    relative_core: tuple[slice, slice, slice],
    *,
    source_data_dir: Path,
    time_index: int,
) -> None:
    """Persist the exact scene Stage 01+ should reuse."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    current = np.asarray(
        scene.current_full[scene.core_slices]
    ).astype(np.int64, copy=True)
    gt = np.asarray(
        scene.gt_full[scene.core_slices]
    ).astype(np.int64, copy=True)

    target_tensors = _cropped_target_tensors(
        targets, relative_core
    )

    payload = {
        "format_version": 1,
        "purpose": "stirnet_component_debug_fixed_crop",
        "spatial_channel_names": tuple(
            SPATIAL_CHANNEL_NAMES
        ),
        # Per-sample tensors intentionally omit batch dimension.
        "spatial_inputs": torch.from_numpy(
            spatial_inputs
        ).float(),
        "raw_norm": torch.from_numpy(
            spatial_inputs[0].copy()
        ).float(),
        "marker_heatmap": torch.from_numpy(
            spatial_inputs[4].copy()
        ).float(),
        "current_labels": torch.from_numpy(
            current
        ).long(),
        "gt_labels": torch.from_numpy(gt).long(),
        "spacing_um": torch.from_numpy(
            scene.spacing_zyx_um.copy()
        ).float(),
        "dref_um": torch.tensor(
            scene.dref_um, dtype=torch.float32
        ),
        "geometry_targets": target_tensors,
        "selection": scene.selection,
        "source": {
            "data_dir": str(source_data_dir.resolve()),
            "time_index": int(time_index),
            "core_slices_zyx": _slice_pairs(
                scene.core_slices
            ),
            "build_slices_zyx": _slice_pairs(
                scene.build_slices
            ),
        },
    }

    temporary = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )
    torch.save(payload, temporary)
    temporary.replace(output_path)

    metadata_path = output_path.with_name(
        output_path.stem + "_metadata.json"
    )
    metadata = {
        "format_version": 1,
        "purpose": payload["purpose"],
        "saved_file": str(output_path),
        "source": payload["source"],
        "spatial_channel_names": list(
            SPATIAL_CHANNEL_NAMES
        ),
        "shape_zyx": list(current.shape),
        "voxel_count": int(current.size),
        "spacing_zyx_um": [
            float(x)
            for x in scene.spacing_zyx_um.tolist()
        ],
        "dref_um": float(scene.dref_um),
        "selection": scene.selection,
        "tensor_contract": {
            "spatial_inputs": "[5,Z,Y,X]",
            "current_labels": "[Z,Y,X]",
            "gt_labels": "[Z,Y,X]",
            "geometry_targets": "[C,Z,Y,X] per field",
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    size_mb = output_path.stat().st_size / (1024**2)
    print("Saved reusable debug sample:")
    print(f"  sample   : {output_path}")
    print(f"  metadata : {metadata_path}")
    print(f"  size     : {size_mb:.2f} MiB")


# ---------------------------------------------------------------------------
# Napari helpers
# ---------------------------------------------------------------------------


def _sample_vectors(
    field: np.ndarray,
    mask: np.ndarray,
    spacing_um: np.ndarray,
    stride: tuple[int, int, int],
    max_vectors: int,
    *,
    physical_scale_um: float | None = None,
    normalized_by_dref: float | None = None,
) -> np.ndarray:
    """Convert dense z/y/x vector field to Napari ``(N,2,3)`` vectors."""
    z, y, x = np.indices(mask.shape)
    select = (
        mask
        & (z % max(1, stride[0]) == 0)
        & (y % max(1, stride[1]) == 0)
        & (x % max(1, stride[2]) == 0)
    )
    points = np.argwhere(select)
    if points.size == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)

    if len(points) > max_vectors:
        step = int(np.ceil(len(points) / max_vectors))
        points = points[::step][:max_vectors]

    values = field[:, points[:, 0], points[:, 1], points[:, 2]].T.astype(np.float32)
    if normalized_by_dref is not None:
        vector_um = values * float(normalized_by_dref)
    elif physical_scale_um is not None:
        vector_um = values * float(physical_scale_um)
    else:
        vector_um = values

    # Napari vector coordinates are in voxels; convert physical displacement to
    # voxel displacement. Layer ``scale`` then restores physical z/y/x spacing.
    vector_vox = vector_um / spacing_um[None]
    return np.stack([points.astype(np.float32), vector_vox.astype(np.float32)], axis=1)


def _points_for_seed_maxima(
    gt: np.ndarray,
    seed: np.ndarray,
) -> np.ndarray:
    points: list[np.ndarray] = []
    for instance_id in np.unique(gt):
        if instance_id <= 0:
            continue
        mask = gt == instance_id
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        values = seed[mask]
        best = coords[int(np.argmax(values))]
        points.append(best.astype(np.float32))
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(points)


def _points_for_gt_centroids(gt: np.ndarray) -> np.ndarray:
    points: list[np.ndarray] = []
    shape = np.asarray(gt.shape)
    for instance_id in np.unique(gt):
        if instance_id <= 0:
            continue
        coords = np.argwhere(gt == instance_id)
        if coords.size == 0:
            continue
        centroid = coords.astype(np.float32).mean(axis=0)
        if np.all(centroid >= 0) and np.all(centroid <= shape - 1):
            points.append(centroid)
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(points)


def open_napari(
    scene: Scene,
    arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is required for this investigation. Install it in the project "
            "environment, for example: pip install 'napari[all]'"
        ) from exc

    core = scene.core_slices
    gt = np.asarray(scene.gt_full[core]).astype(np.int64, copy=False)
    current = np.asarray(scene.current_full[core]).astype(np.int64, copy=False)
    raw = np.asarray(scene.raw_full[core])
    scale = tuple(float(v) for v in scene.spacing_zyx_um)

    flow_magnitude = np.linalg.vector_norm(arrays["flow"], axis=0)
    offset_um = arrays["centroid_offset"] * scene.dref_um
    offset_magnitude_um = np.linalg.vector_norm(offset_um, axis=0)

    flow_vectors = _sample_vectors(
        arrays["flow"],
        gt > 0,
        scene.spacing_zyx_um,
        tuple(args.vector_stride),
        args.max_vectors,
        physical_scale_um=args.flow_vector_length_um,
    )
    offset_vectors = _sample_vectors(
        arrays["centroid_offset"],
        gt > 0,
        scene.spacing_zyx_um,
        tuple(args.vector_stride),
        args.max_vectors,
        normalized_by_dref=scene.dref_um,
    )

    seed_points = _points_for_seed_maxima(gt, arrays["seed"])
    centroid_points = _points_for_gt_centroids(gt)

    viewer = napari.Viewer(
        title="STIR-Net — Stage 00 fixed debug crop",
        ndisplay=3,
    )

    viewer.add_image(
        raw,
        name="00 Raw normalized",
        colormap="gray",
        scale=scale,
    )

    viewer.add_labels(
        current,
        name="01 Current segmentation (context)",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        gt,
        name="02 GT instance labels",
        scale=scale,
    )

    merge_gt_ids = tuple(
        int(x) for x in scene.selection["merge_gt_ids"]
    )
    free_gt_id = int(scene.selection["free_gt_id"])
    merge_source_id = int(scene.selection["merge_source_id"])
    merge_component_mask = _visible_merge_component_mask(
        gt,
        current,
        merge_source_id,
        merge_gt_ids,
    )
    merge_component_layer = (
        np.zeros_like(gt, dtype=np.uint8)
        if merge_component_mask is None
        else merge_component_mask.astype(np.uint8)
    )
    merge_gt_layer = np.where(
        np.isin(gt, np.asarray(merge_gt_ids, dtype=gt.dtype)),
        gt,
        0,
    )
    free_gt_layer = np.where(gt == free_gt_id, gt, 0)

    viewer.add_labels(
        merge_component_layer,
        name=(
            "02a Required merged current component "
            f"(source {merge_source_id})"
        ),
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        merge_gt_layer,
        name=f"02b Required merge GT cells {merge_gt_ids}",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        free_gt_layer,
        name=f"02c Required free GT cell {free_gt_id}",
        scale=scale,
        visible=False,
    )
    viewer.add_image(
        arrays["foreground"],
        name="03 Target — foreground",
        colormap="gray",
        contrast_limits=(0.0, 1.0),
        scale=scale,
        visible=False,
    )
    viewer.add_image(
        arrays["surface"],
        name="04 Target — surface",
        colormap="inferno",
        contrast_limits=(0.0, 1.0),
        scale=scale,
        blending="additive",
        visible=False,
    )
    viewer.add_image(
        arrays["separator"],
        name="05 Target — separator",
        colormap="magenta",
        contrast_limits=(0.0, 1.0),
        scale=scale,
        blending="additive",
        visible=True,
    )

    sdf_abs = float(max(abs(float(arrays["sdf"].min())), abs(float(arrays["sdf"].max())), 1e-6))
    viewer.add_image(
        arrays["sdf"],
        name="06 Target — signed distance / dref",
        colormap="turbo",
        contrast_limits=(-sdf_abs, sdf_abs),
        scale=scale,
        visible=False,
    )
    viewer.add_image(
        arrays["sdf_valid"].astype(np.float32),
        name="07 Target — SDF valid",
        colormap="gray",
        contrast_limits=(0.0, 1.0),
        scale=scale,
        visible=False,
    )
    viewer.add_image(
        arrays["seed"],
        name="08 Target — seed",
        colormap="inferno",
        contrast_limits=(0.0, 1.0),
        scale=scale,
        visible=False,
    )
    viewer.add_image(
        flow_magnitude,
        name="09 Target — flow magnitude",
        colormap="viridis",
        contrast_limits=(0.0, max(1.0, float(flow_magnitude.max()))),
        scale=scale,
        visible=False,
    )
    viewer.add_image(
        offset_magnitude_um,
        name="10 Target — centroid offset magnitude (um)",
        colormap="viridis",
        scale=scale,
        visible=False,
    )

    if len(flow_vectors):
        viewer.add_vectors(
            flow_vectors,
            name="11 Target — SDF flow vectors",
            scale=scale,
            visible=False,
        )
    if len(offset_vectors):
        viewer.add_vectors(
            offset_vectors,
            name="12 Target — centroid offset vectors",
            scale=scale,
            visible=False,
        )
    if len(seed_points):
        viewer.add_points(
            seed_points,
            name="13 Seed maxima (one chosen per visible cell)",
            scale=scale,
            size=2.0,
            visible=False,
        )
    if len(centroid_points):
        viewer.add_points(
            centroid_points,
            name="14 GT voxel centroids",
            scale=scale,
            size=2.0,
            visible=False,
        )

    # Start near the center of the visible diagnostic crop.
    viewer.dims.current_step = tuple(int(n // 2) for n in gt.shape)

    print("Napari layers loaded.")
    print(
        "This is the exact crop saved for Stage 01. "
        "Use layers 02a/02b/02c to verify the required visible merge "
        "and the fully-contained free cell."
    )

    napari.run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the fixed ~0.52M-voxel STIR-Net geometry-debug scene, "
            "validate its targets, persist it, and inspect it in Napari."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Prepared source scene (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Saved reusable debug sample (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=DEFAULT_TIME_INDEX,
        help=f"Frame to use (default: {DEFAULT_TIME_INDEX})",
    )
    parser.add_argument(
        "--voxel-budget",
        type=int,
        default=DEFAULT_VOXEL_BUDGET,
        help=(
            "Maximum saved crop voxels. Default 524288 (~0.52 M). "
            "The crop shape is chosen automatically under this budget."
        ),
    )
    parser.add_argument(
        "--crop-shape",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=None,
        help=(
            "Optional fixed Z Y X override. By default Stage 00 searches for "
            "an 8-aligned crop shape under --voxel-budget that can contain the "
            "required visible merge and free cell."
        ),
    )
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=0,
        help="Which ranked candidate crop to open/save (default: 0).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help="How many semantic candidate cases to enumerate.",
    )
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="Print ranked candidate summaries and exit without saving/opening Napari.",
    )
    parser.add_argument(
        "--exclude-merge-source",
        type=int,
        action="append",
        default=[],
        help="Merge source-id(s) to exclude from the candidate search. Repeat as needed.",
    )
    parser.add_argument(
        "--context-margin-dref",
        type=float,
        default=DEFAULT_CONTEXT_MARGIN_DREF,
        help=(
            "CPU-only context used when building targets/spatial priors "
            "before cropping back to the saved scene."
        ),
    )
    parser.add_argument(
        "--vector-stride",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=DEFAULT_VECTOR_STRIDE,
        help="Napari sampling stride for vector layers.",
    )
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=DEFAULT_MAX_VECTORS,
        help="Maximum vectors in each Napari vector layer.",
    )
    parser.add_argument(
        "--flow-vector-length-um",
        type=float,
        default=DEFAULT_FLOW_VECTOR_LENGTH_UM,
        help="Display length for unit flow arrows.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Validate/visualize but do not write debug_crop.pt.",
    )
    parser.add_argument(
        "--no-napari",
        action="store_true",
        help="Build, validate, and save without opening Napari.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 92)
    print("STIR-Net Stage 00 — selecting fixed reusable debug crop")
    print("=" * 92)
    print(f"Source data      : {args.data_dir}")
    print(
        f"Voxel budget     : {int(args.voxel_budget):,} "
        f"({int(args.voxel_budget) / 1e6:.3f} M)"
    )
    print(
        "Shape mode       : "
        + (
            "adaptive under budget"
            if args.crop_shape is None
            else f"fixed override {tuple(args.crop_shape)}"
        )
    )

    scene = load_scene(args)
    candidate_summaries = scene.selection.get("candidate_summaries", [])
    print_candidate_summaries(candidate_summaries)

    if args.list_candidates:
        print("\nListed candidates only; exiting without saving or opening Napari.")
        return

    print("\nSelected topology")
    print(f"  candidate index : {scene.selection['candidate_index']}")
    print(
        "  merge          : "
        f"current source {scene.selection['merge_source_id']} "
        f"contains GT {tuple(scene.selection['merge_gt_ids'])}"
    )
    print(f"  free GT cell   : {scene.selection['free_gt_id']}")
    print(
        f"  selected shape : {tuple(scene.selection['crop_shape_zyx'])} "
        f"= {scene.selection['voxel_count']:,} voxels "
        f"({100.0 * scene.selection['budget_utilization']:.1f}% of budget)"
    )
    print(f"  core slices    : {scene.core_slices}")
    print(f"  build slices   : {scene.build_slices}")

    print("\nBuilding production geometry targets on CPU...")
    targets = build_targets(scene)
    relative_core = _relative_slices(
        scene.core_slices, scene.build_slices
    )
    arrays = _target_arrays(targets, relative_core)

    print("Building production 5-channel spatial input on CPU...")
    spatial_inputs = build_saved_spatial_inputs(scene)

    validate_targets(
        scene,
        targets,
        arrays,
        spatial_inputs,
    )

    if not args.no_save:
        save_debug_sample(
            args.output,
            scene,
            targets,
            spatial_inputs,
            relative_core,
            source_data_dir=args.data_dir,
            time_index=args.time_index,
        )

    if not args.no_napari:
        open_napari(scene, arrays, args)


if __name__ == "__main__":
    main()
