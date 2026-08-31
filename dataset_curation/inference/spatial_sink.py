from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_curation.io.atomic import atomic_json
from learned.stirnet.inference import (
    SpatialFrameResult,
    SpatialInferenceConfig,
    SpatialModelRuntime,
    SpatialVolumeResult,
)


PERSISTED_LABEL_DTYPE = np.dtype(np.uint16)
PERSISTED_LABEL_MAX = int(np.iinfo(PERSISTED_LABEL_DTYPE).max)

# Files produced by the previous full-volume cache contract. They are deleted
# when a volume is regenerated under the compact contract.
_OBSOLETE_FULL_VOLUME_MOVIES = (
    "raw.npy",
    "preprocessed.npy",
    "binary_mask.npy",
    "source_instances.npy",
)


def _atomic_csv(
    path: Path,
    frame: pd.DataFrame,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    try:
        frame.to_csv(
            tmp,
            index=False,
        )
        os.replace(
            tmp,
            path,
        )
    finally:
        tmp.unlink(
            missing_ok=True
        )


def _as_uint16_labels(
    labels: np.ndarray,
    *,
    name: str,
    frame: int,
) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 3:
        raise ValueError(
            f"{name} at t={frame} must be 3-D, got {array.shape}"
        )

    if array.size:
        minimum = int(array.min())
        maximum = int(array.max())
    else:
        minimum = 0
        maximum = 0

    if minimum < 0:
        raise RuntimeError(
            f"{name} at t={frame} contains a negative label ID: {minimum}"
        )
    if maximum > PERSISTED_LABEL_MAX:
        raise RuntimeError(
            f"{name} at t={frame} has label ID {maximum}, which exceeds "
            f"uint16 capacity ({PERSISTED_LABEL_MAX}). "
            "Do not silently truncate label IDs."
        )

    return np.asarray(
        array,
        dtype=PERSISTED_LABEL_DTYPE,
    )


class CurationSpatialSink:
    """
    Persist only the annotation-critical spatial label movies.

    The production spatial engine still computes raw/preprocessed/binary/source
    data and the 5-channel STIR-Net tensor frame-by-frame, but those objects are
    ephemeral. Persistent full-volume arrays are exactly:

        movies/supervoxels.npy      uint16
        movies/final_instances.npy  uint16

    Raw imagery is always read from the canonical source Zarr when needed.
    """

    def __init__(
        self,
        paths,
        *,
        run_id: str,
        frame_count: int,
    ) -> None:
        self.paths = paths
        self.run_id = str(run_id)
        self.frame_count = int(frame_count)

        self._supervoxels = None
        self._final = None

        self._cells: list[pd.DataFrame] = []
        self._written_frames: set[int] = set()

        output = self.paths.inference_run(self.run_id)
        output.mkdir(parents=True, exist_ok=True)

        # A regenerated run is incomplete until finish() writes a fresh marker.
        self.paths.spatial_success(self.run_id).unlink(missing_ok=True)
        self.paths.spatial_summary(self.run_id).unlink(missing_ok=True)
        self.paths.cells_csv(self.run_id).unlink(missing_ok=True)

        movies = self.paths.movies(self.run_id)
        movies.mkdir(parents=True, exist_ok=True)
        for name in _OBSOLETE_FULL_VOLUME_MOVIES:
            stale = movies / name
            if stale.is_file():
                stale.unlink()
                print(
                    f"[cache] removed obsolete full-volume artifact: {stale}",
                    flush=True,
                )

        self.cells_dir = output / "cells"
        self.cells_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.cells_dir.glob("t*.csv"):
            stale.unlink(missing_ok=True)

    def _initialize(
        self,
        result: SpatialFrameResult,
    ) -> None:
        shape = (
            self.frame_count,
            *tuple(
                int(v)
                for v in result.final_labels.shape
            ),
        )

        self._supervoxels = np.lib.format.open_memmap(
            self.paths.supervoxels(self.run_id),
            mode="w+",
            dtype=PERSISTED_LABEL_DTYPE,
            shape=shape,
        )
        self._final = np.lib.format.open_memmap(
            self.paths.final_instances(self.run_id),
            mode="w+",
            dtype=PERSISTED_LABEL_DTYPE,
            shape=shape,
        )

    def write_frame(
        self,
        result: SpatialFrameResult,
    ) -> None:
        if self._supervoxels is None:
            self._initialize(result)

        frame = int(result.frame)
        if not (0 <= frame < self.frame_count):
            raise IndexError(
                f"Frame {frame} is outside 0..{self.frame_count - 1}"
            )
        if frame in self._written_frames:
            raise RuntimeError(
                f"Frame {frame} was written twice"
            )

        supervoxels = _as_uint16_labels(
            result.supervoxel_labels,
            name="supervoxels",
            frame=frame,
        )
        final_instances = _as_uint16_labels(
            result.final_labels,
            name="final_instances",
            frame=frame,
        )

        self._supervoxels[frame] = supervoxels
        self._final[frame] = final_instances

        cells = result.cells.copy()
        cells["frame"] = frame
        self._cells.append(cells)
        _atomic_csv(
            self.cells_dir / f"t{frame:03d}.csv",
            cells,
        )
        self._written_frames.add(frame)

    def _flush_movies(self) -> None:
        for movie in (
            self._supervoxels,
            self._final,
        ):
            if movie is not None:
                movie.flush()

    def finish(
        self,
        volume_result: SpatialVolumeResult,
        *,
        runtime: SpatialModelRuntime,
        config: SpatialInferenceConfig,
        sample_id: str,
    ) -> None:
        expected = set(range(self.frame_count))
        if self._written_frames != expected:
            missing = sorted(expected - self._written_frames)
            raise RuntimeError(
                "Cannot finalize incomplete spatial cache; "
                f"missing={missing}"
            )

        self._flush_movies()

        if not self._cells:
            raise RuntimeError(
                "No cell tables were produced"
            )

        combined = pd.concat(
            self._cells,
            ignore_index=True,
        )
        _atomic_csv(
            self.paths.cells_csv(self.run_id),
            combined,
        )

        summary = {
            "sample_id": str(sample_id),
            "frame_count": int(volume_result.frame_count),
            "spatial_shape_zyx": list(
                volume_result.spatial_shape_zyx
            ),
            "spacing_zyx_um": list(
                config.spacing_zyx_um
            ),
            "checkpoint": str(
                runtime.checkpoint_path
            ),
            "checkpoint_step": int(
                runtime.checkpoint_step
            ),
            "checkpoint_sha256": str(
                runtime.checkpoint_sha256
            ),
            "total_seconds": float(
                volume_result.total_seconds
            ),
            "frames": list(
                volume_result.frame_summaries
            ),
            "storage_contract": {
                "version": 2,
                "persistent_full_volume_arrays": {
                    "supervoxels": {
                        "path": "movies/supervoxels.npy",
                        "dtype": "uint16",
                    },
                    "final_instances": {
                        "path": "movies/final_instances.npy",
                        "dtype": "uint16",
                    },
                },
                "ephemeral_only": [
                    "raw",
                    "preprocessed",
                    "binary_mask",
                    "source_instances",
                    "stirnet_spatial_channels",
                    "dense_model_outputs",
                ],
            },
            "scientific_path": {
                "source_geometric_completion": False,
                "parallel_preparation_workers": 1,
                "parallel_prefetch_depth": 1,
                "temporal_stirnet": False,
                "source_core_split_only": True,
                "supervoxels_persisted": True,
            },
        }

        atomic_json(
            self.paths.spatial_summary(self.run_id),
            summary,
        )
        atomic_json(
            self.paths.spatial_success(self.run_id),
            {
                "status": "success",
                "sample_id": str(sample_id),
                "frame_count": int(
                    volume_result.frame_count
                ),
                "checkpoint_sha256": str(
                    runtime.checkpoint_sha256
                ),
                "storage_contract_version": 2,
                "label_dtype": "uint16",
            },
        )

    def close(self) -> None:
        self._flush_movies()
        self._supervoxels = None
        self._final = None
