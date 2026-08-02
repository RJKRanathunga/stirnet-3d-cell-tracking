"""Physical shape and watershed-interface measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distance import validate_voxel_size


@dataclass(frozen=True)
class CellShape:
    """Translation-independent physical description of a binary cell mask."""

    voxel_count: int
    volume_um3: float
    equivalent_radius_um: float
    elongation: float
    flatness: float
    extent: float


@dataclass(frozen=True)
class InterfaceStatistics:
    """Measurements of a face-connected interface between two regions."""

    contact: bool
    interface_area_um2: float
    distance_median_um: float
    distance_mean_um: float


def describe_cell_mask(
    mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float] | np.ndarray,
) -> CellShape:
    """Describe a 3-D mask using physical PCA extents and volume."""

    region = np.asarray(mask, dtype=bool)
    voxel_size = validate_voxel_size(voxel_size_zyx)
    coordinates = np.argwhere(region)
    if coordinates.size == 0:
        raise ValueError("cell mask is empty")

    physical_coordinates = coordinates.astype(float) * voxel_size[None, :]
    centered = physical_coordinates - physical_coordinates.mean(axis=0)[None, :]

    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    bbox_shape = maximum - minimum + 1
    bbox_dimensions_um = bbox_shape * voxel_size

    if len(coordinates) >= 4:
        covariance = np.cov(centered, rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        projected = centered @ eigenvectors[:, order]
        principal_extents = np.percentile(projected, 97.5, axis=0) - np.percentile(
            projected, 2.5, axis=0
        )
    else:
        principal_extents = np.sort(bbox_dimensions_um)[::-1]

    major, middle, minor = (float(value) for value in principal_extents)
    denominator_floor = 0.5 * float(voxel_size.min())
    safe_minor = max(minor, denominator_floor)

    voxel_count = int(region.sum())
    volume_um3 = float(voxel_count * np.prod(voxel_size))
    bbox_voxels = int(np.prod(bbox_shape))

    return CellShape(
        voxel_count=voxel_count,
        volume_um3=volume_um3,
        equivalent_radius_um=float(
            (3.0 * volume_um3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        ),
        elongation=major / safe_minor,
        flatness=middle / safe_minor,
        extent=float(voxel_count / bbox_voxels),
    )


def region_surface_area_um2(
    region_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float] | np.ndarray,
) -> float:
    """Compute voxel-face surface area with anisotropic face dimensions."""

    region = np.asarray(region_mask, dtype=bool)
    voxel_size = validate_voxel_size(voxel_size_zyx)
    total_area = 0.0
    for axis in range(3):
        padded = np.pad(
            region.astype(np.int8),
            [(1, 1) if index == axis else (0, 0) for index in range(3)],
            mode="constant",
        )
        transitions = np.diff(padded, axis=axis) != 0
        total_area += int(np.count_nonzero(transitions)) * float(
            np.prod(np.delete(voxel_size, axis))
        )
    return float(total_area)


def pair_interface_statistics(
    labels: np.ndarray,
    label_a: int,
    label_b: int,
    distance_um: np.ndarray,
    voxel_size_zyx: tuple[float, float, float] | np.ndarray,
) -> InterfaceStatistics:
    """Measure the face interface and EDT depth between two watershed children."""

    labels = np.asarray(labels)
    distance = np.asarray(distance_um, dtype=float)
    voxel_size = validate_voxel_size(voxel_size_zyx)
    interface_values: list[np.ndarray] = []
    interface_area = 0.0

    for axis in range(3):
        first_slice = [slice(None)] * 3
        second_slice = [slice(None)] * 3
        first_slice[axis] = slice(0, -1)
        second_slice[axis] = slice(1, None)
        first = labels[tuple(first_slice)]
        second = labels[tuple(second_slice)]
        touching = ((first == label_a) & (second == label_b)) | (
            (first == label_b) & (second == label_a)
        )
        if not np.any(touching):
            continue

        first_coordinates = np.argwhere(touching)
        second_coordinates = first_coordinates.copy()
        second_coordinates[:, axis] += 1
        interface_values.append(
            np.minimum(
                distance[tuple(first_coordinates.T)],
                distance[tuple(second_coordinates.T)],
            )
        )
        interface_area += int(np.count_nonzero(touching)) * float(
            np.prod(np.delete(voxel_size, axis))
        )

    if not interface_values:
        return InterfaceStatistics(False, 0.0, 0.0, 0.0)
    values = np.concatenate(interface_values)
    return InterfaceStatistics(
        True,
        float(interface_area),
        float(np.median(values)),
        float(np.mean(values)),
    )
