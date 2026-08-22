from __future__ import annotations

"""Exact EDT backend selection for STIR-Net geometry target generation."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Iterator

import numpy as np
from scipy import ndimage as scipy_ndi

_BACKEND: ContextVar[str] = ContextVar("stirnet_geometry_edt_backend", default="scipy")
_GPU_MIN_VOXELS: ContextVar[int] = ContextVar(
    "stirnet_geometry_edt_gpu_min_voxels", default=262_144
)


@lru_cache(maxsize=1)
def _load_cupy():
    try:
        import cupy as cp
        from cupyx.scipy import ndimage as cupy_ndi
    except Exception:
        return None, None
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            return None, None
    except Exception:
        return None, None
    return cp, cupy_ndi


def cupy_edt_available() -> bool:
    cp, cupy_ndi = _load_cupy()
    return cp is not None and cupy_ndi is not None


@contextmanager
def geometry_edt_backend(
    backend: str = "auto", *, gpu_min_voxels: int = 262_144
) -> Iterator[None]:
    backend = str(backend).lower()
    if backend not in {"auto", "scipy", "cupy"}:
        raise ValueError("EDT backend must be auto, scipy, or cupy")
    if gpu_min_voxels < 1:
        raise ValueError("gpu_min_voxels must be positive")
    token_backend = _BACKEND.set(backend)
    token_threshold = _GPU_MIN_VOXELS.set(int(gpu_min_voxels))
    try:
        yield
    finally:
        _GPU_MIN_VOXELS.reset(token_threshold)
        _BACKEND.reset(token_backend)


def _scipy_edt(
    image: np.ndarray,
    *,
    sampling=None,
    return_distances: bool = True,
    return_indices: bool = False,
):
    return scipy_ndi.distance_transform_edt(
        image,
        sampling=sampling,
        return_distances=return_distances,
        return_indices=return_indices,
    )


def _cupy_edt(
    image: np.ndarray,
    *,
    sampling=None,
    return_distances: bool = True,
    return_indices: bool = False,
):
    cp, cupy_ndi = _load_cupy()
    if cp is None or cupy_ndi is None:
        raise RuntimeError(
            "CuPy EDT requested but CuPy/CUDA is unavailable. "
            "Install the matching CuPy CUDA wheel or use scipy."
        )
    result = cupy_ndi.distance_transform_edt(
        cp.asarray(np.asarray(image)),
        sampling=sampling,
        return_distances=return_distances,
        return_indices=return_indices,
    )
    cp.cuda.get_current_stream().synchronize()
    if isinstance(result, tuple):
        return tuple(cp.asnumpy(value) for value in result)
    return cp.asnumpy(result)


def distance_transform_edt(
    image: Any,
    sampling=None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances=None,
    indices=None,
):
    """SciPy-compatible EDT with optional exact CuPy acceleration.

    In ``auto`` mode only sufficiently large transforms go to CuPy.  Small
    per-cell EDTs stay on SciPy to avoid PCIe/kernel-launch overhead.  If CuPy
    fails in ``auto`` mode, correctness wins and the call falls back to SciPy.
    """
    array = np.asarray(image)
    if distances is not None or indices is not None:
        return scipy_ndi.distance_transform_edt(
            array,
            sampling=sampling,
            return_distances=return_distances,
            return_indices=return_indices,
            distances=distances,
            indices=indices,
        )

    backend = _BACKEND.get()
    use_cupy = (
        backend == "cupy"
        or (
            backend == "auto"
            and array.size >= int(_GPU_MIN_VOXELS.get())
            and cupy_edt_available()
        )
    )
    if not use_cupy:
        return _scipy_edt(
            array,
            sampling=sampling,
            return_distances=return_distances,
            return_indices=return_indices,
        )
    try:
        return _cupy_edt(
            array,
            sampling=sampling,
            return_distances=return_distances,
            return_indices=return_indices,
        )
    except Exception:
        if backend == "cupy":
            raise
        return _scipy_edt(
            array,
            sampling=sampling,
            return_distances=return_distances,
            return_indices=return_indices,
        )


def release_cupy_memory() -> None:
    """Return CuPy allocator blocks before PyTorch starts using the GPU."""
    cp, _ = _load_cupy()
    if cp is None:
        return
    try:
        cp.cuda.get_current_stream().synchronize()
    except Exception:
        pass
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


__all__ = [
    "cupy_edt_available",
    "distance_transform_edt",
    "geometry_edt_backend",
    "release_cupy_memory",
]
