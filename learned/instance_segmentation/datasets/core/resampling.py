"""Physical crop extraction and resampling without whole-volume up/downsampling."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy import ndimage

from .models import AnnotatedVolume, Shape3D, Spacing3D


@dataclass(frozen=True)
class ResampledCrop:
    image: np.ndarray
    labels: np.ndarray
    valid_mask: np.ndarray
    center_native_zyx: tuple[float, float, float]
    target_spacing_zyx_um: Spacing3D


def selected_group_center_native_zyx(
    labels: np.ndarray,
    instance_ids: tuple[int, ...],
) -> tuple[float, float, float]:
    """Use the selected union's bounding-box center as a stable crop center."""

    selected = np.isin(labels, np.asarray(instance_ids, dtype=labels.dtype))
    coordinates = np.argwhere(selected)
    if coordinates.size == 0:
        raise ValueError("selected instances do not occur in the label volume")
    lo = coordinates.min(axis=0).astype(np.float64)
    hi = coordinates.max(axis=0).astype(np.float64)
    center = (lo + hi) / 2.0
    return tuple(float(v) for v in center)


def _anti_alias_sigma_native(
    native_spacing: Spacing3D,
    target_spacing: Spacing3D,
) -> tuple[float, float, float]:
    """Conservative Gaussian prefilter for axes that are downsampled."""

    sigma: list[float] = []
    for native, target in zip(native_spacing, target_spacing):
        ratio = float(target) / float(native)
        if ratio <= 1.0:
            sigma.append(0.0)
        else:
            # A common anti-alias approximation for decimation. This is a
            # numerical prefilter, not a model of the microscope PSF.
            sigma.append(0.5 * float(np.sqrt(max(ratio * ratio - 1.0, 0.0))))
    return tuple(sigma)  # type: ignore[return-value]


def _native_window(
    shape: Shape3D,
    center_native: tuple[float, float, float],
    native_spacing: Spacing3D,
    target_spacing: Spacing3D,
    output_shape: Shape3D,
    extra_margin_native: tuple[int, int, int],
) -> tuple[tuple[slice, slice, slice], tuple[float, float, float]]:
    slices: list[slice] = []
    local_center: list[float] = []
    for axis in range(3):
        half_output_um = (output_shape[axis] - 1) * target_spacing[axis] / 2.0
        half_native = half_output_um / native_spacing[axis]
        radius = int(ceil(half_native)) + int(extra_margin_native[axis]) + 2
        center = float(center_native[axis])
        start = max(0, int(np.floor(center)) - radius)
        stop = min(int(shape[axis]), int(np.ceil(center)) + radius + 1)
        slices.append(slice(start, stop))
        local_center.append(center - start)
    return tuple(slices), tuple(local_center)  # type: ignore[return-value]


def _coordinate_grid(
    center_local: tuple[float, float, float],
    native_spacing: Spacing3D,
    target_spacing: Spacing3D,
    output_shape: Shape3D,
) -> np.ndarray:
    axes: list[np.ndarray] = []
    for axis in range(3):
        output_index = np.arange(output_shape[axis], dtype=np.float64)
        offset_um = (
            output_index - (output_shape[axis] - 1) / 2.0
        ) * target_spacing[axis]
        axes.append(center_local[axis] + offset_um / native_spacing[axis])
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=0)


def resample_centered_crop(
    volume: AnnotatedVolume,
    center_native_zyx: tuple[float, float, float],
    *,
    output_shape_zyx: Shape3D,
    target_spacing_zyx_um: Spacing3D,
    anti_alias_image: bool = True,
) -> ResampledCrop:
    """Extract a physical field of view and map it directly to a fixed grid.

    Only the local native window is filtered/resampled, which keeps this method
    suitable for much larger datasets such as NIS3D.
    """

    native_spacing = tuple(float(v) for v in volume.spacing_zyx_um)
    target_spacing = tuple(float(v) for v in target_spacing_zyx_um)
    output_shape = tuple(int(v) for v in output_shape_zyx)

    sigma_native = _anti_alias_sigma_native(native_spacing, target_spacing)
    margin = tuple(int(ceil(3.0 * sigma)) for sigma in sigma_native)
    window, center_local = _native_window(
        volume.shape_zyx,
        center_native_zyx,
        native_spacing,
        target_spacing,
        output_shape,
        margin,
    )

    image_local = np.asarray(volume.image[window], dtype=np.float32)
    labels_local = np.asarray(volume.instance_labels[window])
    valid_local = np.asarray(volume.effective_valid_mask[window], dtype=np.uint8)

    if anti_alias_image and any(sigma > 0 for sigma in sigma_native):
        image_local = ndimage.gaussian_filter(
            image_local,
            sigma=sigma_native,
            mode="nearest",
        )

    coords = _coordinate_grid(
        center_local,
        native_spacing,
        target_spacing,
        output_shape,
    )
    image = ndimage.map_coordinates(
        image_local,
        coords,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)
    labels = ndimage.map_coordinates(
        labels_local,
        coords,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(np.int32, copy=False)
    valid = ndimage.map_coordinates(
        valid_local,
        coords,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(bool, copy=False)

    return ResampledCrop(
        image=image,
        labels=labels,
        valid_mask=valid,
        center_native_zyx=center_native_zyx,
        target_spacing_zyx_um=target_spacing,
    )
