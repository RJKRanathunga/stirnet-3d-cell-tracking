"""Instance matching, center diagnostics, and Stage 5 comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage


@dataclass(frozen=True)
class CenterDiagnostic:
    instance_id: int
    geometric_centroid_zyx: tuple[float, float, float]
    in_body_center_zyx: tuple[int, int, int]
    centroid_inside_mask: bool
    connected_part_count: int
    in_body_distance_voxels: float


@dataclass(frozen=True)
class InstanceMatch:
    production_instance_id: int
    trial_instance_id: int | None
    iou: float
    intersection_voxels: int
    union_voxels: int
    production_volume: int
    trial_volume: int
    centroid_displacement_voxels: float | None
    centroid_displacement_um: float | None
    trial_centroid_inside_mask: bool | None
    trial_disconnected_parts: int | None
    status: str
    overlapping_trial_ids: tuple[int, ...]
    trial_overlapping_production_ids: tuple[int, ...]


def diagnose_instance_centers(labels: np.ndarray) -> tuple[CenterDiagnostic, ...]:
    result: list[CenterDiagnostic] = []
    structure = ndimage.generate_binary_structure(3, 1)
    for instance_id in (int(value) for value in np.unique(labels) if value > 0):
        coordinates = np.argwhere(labels == instance_id)
        centroid = coordinates.mean(axis=0)
        rounded = np.rint(centroid).astype(int)
        inside = bool(
            np.all(rounded >= 0)
            and np.all(rounded < labels.shape)
            and labels[tuple(rounded)] == instance_id
        )
        squared = np.sum((coordinates - centroid[None, :]) ** 2, axis=1)
        nearest = coordinates[int(np.argmin(squared))]
        _, parts = ndimage.label(labels == instance_id, structure=structure)
        result.append(CenterDiagnostic(
            instance_id,
            tuple(float(value) for value in centroid),
            tuple(int(value) for value in nearest),
            inside,
            int(parts),
            float(np.sqrt(np.min(squared))),
        ))
    return tuple(result)


def match_instance_by_iou(
    production_labels: np.ndarray,
    trial_labels: np.ndarray,
    production_instance_id: int,
    voxel_size_zyx_um: tuple[float, float, float],
) -> InstanceMatch:
    if production_labels.shape != trial_labels.shape:
        raise ValueError("production and trial labels must have the same shape")
    production_mask = production_labels == int(production_instance_id)
    production_volume = int(np.count_nonzero(production_mask))
    if production_volume == 0:
        raise ValueError(f"production cell {production_instance_id} is absent")
    overlap_ids, overlap_counts = np.unique(
        trial_labels[production_mask & (trial_labels > 0)], return_counts=True
    )
    overlaps = tuple(int(value) for value in overlap_ids)
    if not overlaps:
        return InstanceMatch(
            int(production_instance_id), None, 0.0, 0, production_volume,
            production_volume, 0, None, None, None, None,
            "no_overlapping_trial_instance", (), (),
        )

    candidates: list[tuple[float, int, int, int, int]] = []
    for trial_id, intersection in zip(overlap_ids, overlap_counts):
        trial_volume = int(np.count_nonzero(trial_labels == trial_id))
        union = production_volume + trial_volume - int(intersection)
        candidates.append((int(intersection) / union, int(trial_id), int(intersection), union, trial_volume))
    iou, trial_id, intersection, union, trial_volume = max(candidates)
    trial_mask = trial_labels == trial_id
    production_centroid = np.argwhere(production_mask).mean(axis=0)
    trial_centroid = np.argwhere(trial_mask).mean(axis=0)
    delta = trial_centroid - production_centroid
    center = next(item for item in diagnose_instance_centers(trial_labels) if item.instance_id == trial_id)
    production_ids = tuple(
        int(value) for value in np.unique(production_labels[trial_mask]) if value > 0
    )
    if len(overlaps) > 1:
        status = "production_overlaps_multiple_trial_instances"
    elif len(production_ids) > 1:
        status = "multiple_production_instances_overlap_trial"
    else:
        status = "clear_overlap"
    return InstanceMatch(
        int(production_instance_id), trial_id, float(iou), intersection, union,
        production_volume, trial_volume, float(np.linalg.norm(delta)),
        float(np.linalg.norm(delta * np.asarray(voxel_size_zyx_um))),
        center.centroid_inside_mask, center.connected_part_count, status,
        overlaps, production_ids,
    )


def compare_feature_rows(
    production: pd.DataFrame,
    trial: pd.DataFrame,
    production_id: int,
    trial_id: int | None,
) -> pd.DataFrame:
    """Return available shared Stage 5 values without changing either schema."""

    if trial_id is None:
        return pd.DataFrame(columns=("feature", "production", "trial", "delta"))
    left = production[production["cell_id"] == production_id]
    right = trial[trial["cell_id"] == trial_id]
    if left.empty or right.empty:
        return pd.DataFrame(columns=("feature", "production", "trial", "delta"))
    preferred = (
        "volume", "volume_voxels", "centroid_z", "centroid_y", "centroid_x",
        "intensity_mean", "intensity_sum", "intensity_std", "equivalent_radius",
        "axis_major", "axis_middle", "axis_minor", "elongation", "flatness",
        "anisotropy", "solidity", "compactness", "bbox_depth", "bbox_height",
        "bbox_width",
    )
    common = [name for name in preferred if name in left.columns and name in right.columns]
    rows = []
    for name in common:
        a, b = left.iloc[0][name], right.iloc[0][name]
        try:
            delta = float(b) - float(a)
        except (TypeError, ValueError):
            delta = np.nan
        rows.append({"feature": name, "production": a, "trial": b, "delta": delta})
    return pd.DataFrame(rows)
