"""Physical-volume geometry and boundary observability helpers."""

from __future__ import annotations

import numpy as np

from .types import VolumePointClassification


FACE_NAMES = ("z_min", "z_max", "y_min", "y_max", "x_min", "x_max")
FACE_NORMALS = {
    "z_min": np.asarray([-1.0, 0.0, 0.0]),
    "z_max": np.asarray([1.0, 0.0, 0.0]),
    "y_min": np.asarray([0.0, -1.0, 0.0]),
    "y_max": np.asarray([0.0, 1.0, 0.0]),
    "x_min": np.asarray([0.0, 0.0, -1.0]),
    "x_max": np.asarray([0.0, 0.0, 1.0]),
}


def physical_volume_maximum(
    volume_shape_zyx: np.ndarray,
    voxel_size_zyx_um: np.ndarray,
) -> np.ndarray:
    shape = np.asarray(volume_shape_zyx, dtype=float)
    spacing = np.asarray(voxel_size_zyx_um, dtype=float)
    if shape.shape != (3,) or spacing.shape != (3,):
        raise ValueError("volume shape and voxel size must have shape (3,)")
    if np.any(shape <= 0) or np.any(spacing <= 0):
        raise ValueError("volume shape and voxel size must be positive")
    return (shape - 1.0) * spacing


def distances_to_volume_faces(
    point_zyx_um: np.ndarray,
    volume_shape_zyx: np.ndarray,
    voxel_size_zyx_um: np.ndarray,
) -> np.ndarray:
    point = np.asarray(point_zyx_um, dtype=float)
    maximum = physical_volume_maximum(volume_shape_zyx, voxel_size_zyx_um)
    return np.asarray(
        [point[0], maximum[0] - point[0], point[1], maximum[1] - point[1], point[2], maximum[2] - point[2]],
        dtype=float,
    )


def classify_point_against_volume(
    point_zyx_um: np.ndarray,
    *,
    volume_shape_zyx: np.ndarray,
    voxel_size_zyx_um: np.ndarray,
) -> VolumePointClassification:
    point = np.asarray(point_zyx_um, dtype=float)
    maximum = physical_volume_maximum(volume_shape_zyx, voxel_size_zyx_um)
    low_violation = np.maximum(-point, 0.0)
    high_violation = np.maximum(point - maximum, 0.0)
    violation = low_violation + high_violation
    inside = bool(np.all(violation <= 0.0))
    outside_distance = float(np.linalg.norm(violation))
    face_distances = distances_to_volume_faces(point, volume_shape_zyx, voxel_size_zyx_um)

    if inside:
        face_index = int(np.argmin(face_distances))
        nearest_face = FACE_NAMES[face_index]
        signed_distance = float(face_distances[face_index])
        outside_axes: tuple[int, ...] = ()
    else:
        candidates: list[tuple[float, str, int]] = []
        for axis, axis_name in enumerate(("z", "y", "x")):
            if point[axis] < 0.0:
                candidates.append((float(-point[axis]), f"{axis_name}_min", axis))
            elif point[axis] > maximum[axis]:
                candidates.append((float(point[axis] - maximum[axis]), f"{axis_name}_max", axis))
        candidates.sort(reverse=True)
        nearest_face = candidates[0][1]
        outside_axes = tuple(sorted(item[2] for item in candidates))
        signed_distance = -outside_distance

    return VolumePointClassification(
        inside=inside,
        nearest_face=nearest_face,
        signed_distance_to_volume_um=signed_distance,
        outside_distance_um=outside_distance,
        distances_to_faces_um=face_distances,
        outside_axes=outside_axes,
    )


def boundary_coverage_score(
    point_zyx_um: np.ndarray,
    *,
    search_radius_um: float,
    volume_shape_zyx: np.ndarray,
    voxel_size_zyx_um: np.ndarray,
) -> float:
    """Return a conservative visibility proxy in [0, 1]."""
    if search_radius_um <= 0:
        raise ValueError("search_radius_um must be positive")
    distances = distances_to_volume_faces(point_zyx_um, volume_shape_zyx, voxel_size_zyx_um)
    clipped = np.clip(distances / search_radius_um, 0.0, 1.0)
    return float(np.mean(clipped))


def is_near_boundary(
    point_zyx_um: np.ndarray,
    *,
    margin_um: float,
    volume_shape_zyx: np.ndarray,
    voxel_size_zyx_um: np.ndarray,
) -> bool:
    classification = classify_point_against_volume(
        point_zyx_um,
        volume_shape_zyx=volume_shape_zyx,
        voxel_size_zyx_um=voxel_size_zyx_um,
    )
    return classification.inside and classification.signed_distance_to_volume_um <= margin_um


def directional_agreement(
    origins_zyx_um: np.ndarray,
    destinations_zyx_um: np.ndarray,
    face: str,
) -> float:
    if face not in FACE_NORMALS:
        return 0.0
    origins = np.asarray(origins_zyx_um, dtype=float)
    destinations = np.asarray(destinations_zyx_um, dtype=float)
    if origins.size == 0:
        return 0.0
    displacement = destinations - origins
    norm = FACE_NORMALS[face]
    return float(np.mean((displacement @ norm) > 0.0))
