"""Local object-centric resampling into the fixed canonical CNN grid."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy import ndimage

from .models import AnnotatedVolume, CanonicalTransform, Shape3D, Spacing3D


@dataclass(frozen=True)
class ResampledCrop:
    image: np.ndarray
    labels: np.ndarray
    valid_mask: np.ndarray
    transform: CanonicalTransform


def _anti_alias_sigma_native(
    effective_native_spacing_canonical: Spacing3D,
    canonical_spacing: Spacing3D,
) -> tuple[float, float, float]:
    sigma: list[float] = []
    for native_effective, target in zip(effective_native_spacing_canonical, canonical_spacing):
        ratio = float(target) / float(native_effective)
        if ratio <= 1.0:
            sigma.append(0.0)
        else:
            sigma.append(0.5 * float(np.sqrt(max(ratio * ratio - 1.0, 0.0))))
    return tuple(sigma)  # type: ignore[return-value]


def _native_window(
    shape: Shape3D,
    transform: CanonicalTransform,
    extra_margin_native: tuple[int, int, int],
) -> tuple[tuple[slice, slice, slice], tuple[float, float, float]]:
    canonical_center = np.asarray(transform.canonical_center_zyx, dtype=np.float64)
    corners = np.asarray(
        [
            [0.0, 0.0, 0.0],
            np.asarray(transform.canonical_shape_zyx, dtype=np.float64) - 1.0,
        ]
    )
    native_corners = transform.canonical_to_native(corners)
    lo_native = np.floor(np.min(native_corners, axis=0)).astype(int)
    hi_native = np.ceil(np.max(native_corners, axis=0)).astype(int)

    slices: list[slice] = []
    local_center: list[float] = []
    for axis in range(3):
        start = max(0, int(lo_native[axis]) - int(extra_margin_native[axis]) - 2)
        stop = min(
            int(shape[axis]),
            int(hi_native[axis]) + int(extra_margin_native[axis]) + 3,
        )
        slices.append(slice(start, stop))
        local_center.append(float(transform.native_center_zyx[axis]) - start)
    return tuple(slices), tuple(local_center)  # type: ignore[return-value]


def _coordinate_grid_local(
    transform: CanonicalTransform,
    window: tuple[slice, slice, slice],
) -> np.ndarray:
    axes = [np.arange(int(n), dtype=np.float64) for n in transform.canonical_shape_zyx]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    canonical = np.stack([zz, yy, xx], axis=-1)
    native = transform.canonical_to_native(canonical)
    starts = np.asarray([sl.start or 0 for sl in window], dtype=np.float64)
    local = native - starts
    return np.moveaxis(local, -1, 0)


def resample_with_transform(
    volume: AnnotatedVolume,
    transform: CanonicalTransform,
    *,
    anti_alias_image: bool = True,
) -> ResampledCrop:
    """Resample only the required local source window using an invertible transform."""

    sigma_native = _anti_alias_sigma_native(
        transform.effective_native_spacing_canonical,
        transform.canonical_spacing_zyx,
    )
    margin = tuple(int(ceil(3.0 * sigma)) for sigma in sigma_native)
    window, _ = _native_window(volume.shape_zyx, transform, margin)

    image_local = np.asarray(volume.image[window], dtype=np.float32)
    labels_local = np.asarray(volume.instance_labels[window])
    valid_local = np.asarray(volume.effective_valid_mask[window], dtype=np.uint8)

    if anti_alias_image and any(sigma > 0 for sigma in sigma_native):
        image_local = ndimage.gaussian_filter(image_local, sigma=sigma_native, mode="nearest")

    coords = _coordinate_grid_local(transform, window)
    image = ndimage.map_coordinates(
        image_local, coords, order=1, mode="constant", cval=0.0, prefilter=False
    ).astype(np.float32, copy=False)
    labels = ndimage.map_coordinates(
        labels_local, coords, order=0, mode="constant", cval=0, prefilter=False
    ).astype(np.int32, copy=False)
    valid = ndimage.map_coordinates(
        valid_local, coords, order=0, mode="constant", cval=0, prefilter=False
    ).astype(bool, copy=False)

    return ResampledCrop(image=image, labels=labels, valid_mask=valid, transform=transform)


# Kept only as an explicit compatibility helper.  New training code should use
# build_canonical_transform + resample_with_transform.
def resample_centered_crop(
    volume: AnnotatedVolume,
    center_native_zyx: tuple[float, float, float],
    *,
    output_shape_zyx: Shape3D,
    target_spacing_zyx_um: Spacing3D,
    anti_alias_image: bool = True,
) -> ResampledCrop:
    raise RuntimeError(
        "resample_centered_crop is obsolete in the object-centric architecture. "
        "Build a CanonicalTransform from the selected component/group bbox and "
        "call resample_with_transform instead."
    )
