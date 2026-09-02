from __future__ import annotations

"""Lazy source-Zarr access and RAM-only binary-mask reconstruction."""

from collections import OrderedDict
from pathlib import Path

import numpy as np


def open_source_movie(
    sample_zarr: str | Path,
):
    """Open the canonical raw BioHub movie lazily from source Zarr."""
    from src.io import open_sample

    return open_sample(
        Path(sample_zarr)
    )


def _binary_mask_for_raw(
    raw: np.ndarray,
) -> np.ndarray:
    """Reconstruct the exact production Stage-6 foreground mask."""
    from src.api import (
        create_binary_mask,
        preprocess_volume,
    )

    preprocessed = preprocess_volume(
        np.asarray(raw)
    )
    mask = create_binary_mask(
        preprocessed
    )
    return np.asarray(
        mask > 0,
        dtype=np.uint8,
    )


class BinaryMaskFrameCache:
    """
    Small RAM-only LRU cache for exact Stage-6 masks.

    Binary masks are never persisted as another full 4-D movie.
    """

    def __init__(
        self,
        sample_zarr: str | Path,
        *,
        max_frames: int = 3,
    ) -> None:
        from src.io import open_sample

        self.sample_zarr = Path(
            sample_zarr
        )
        self._source = open_sample(
            self.sample_zarr
        )
        self.max_frames = max(
            int(max_frames),
            1,
        )
        self._cache: OrderedDict[
            int,
            np.ndarray,
        ] = OrderedDict()

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(
            int(v)
            for v in self._source.shape
        )

    def frame(
        self,
        t: int,
    ) -> np.ndarray:
        frame = int(t)
        if not (
            0 <= frame < self.shape[0]
        ):
            raise IndexError(
                f"Frame {frame} is outside "
                f"0..{self.shape[0] - 1}"
            )

        cached = self._cache.pop(
            frame,
            None,
        )
        if cached is not None:
            self._cache[
                frame
            ] = cached
            return cached

        print(
            f"[binary mask] reconstructing t={frame:03d}",
            flush=True,
        )
        raw = np.asarray(
            self._source[frame]
        )
        mask = _binary_mask_for_raw(
            raw
        )
        self._cache[
            frame
        ] = mask

        while (
            len(self._cache)
            > self.max_frames
        ):
            self._cache.popitem(
                last=False
            )
        return mask


def estimate_contrast_limits(
    movie,
    *,
    max_frames: int = 5,
    samples_per_frame: int = 200_000,
) -> tuple[float, float]:
    """Estimate display contrast without materializing the full source movie."""
    frame_count = int(
        movie.shape[0]
    )
    if frame_count <= 0:
        return 0.0, 1.0

    count = min(
        max(int(max_frames), 1),
        frame_count,
    )
    indices = np.unique(
        np.linspace(
            0,
            frame_count - 1,
            count,
            dtype=np.int64,
        )
    )

    samples: list[np.ndarray] = []
    for frame in indices.tolist():
        values = np.asarray(
            movie[int(frame)]
        ).reshape(-1)
        if values.size == 0:
            continue
        stride = max(
            int(
                np.ceil(
                    values.size
                    / max(
                        int(samples_per_frame),
                        1,
                    )
                )
            ),
            1,
        )
        samples.append(
            values[::stride]
        )

    if not samples:
        return 0.0, 1.0

    merged = np.concatenate(
        samples
    )
    low, high = np.percentile(
        merged,
        [1.0, 99.8],
    )
    if not np.isfinite(low):
        low = 0.0
    if not np.isfinite(high):
        high = low + 1.0
    if high <= low:
        high = low + 1.0
    return (
        float(low),
        float(high),
    )
