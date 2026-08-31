from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_curation.io.atomic import atomic_json
from learned.stirnet.inference import (
    SpatialFrameResult,
    SpatialModelRuntime,
    SpatialVolumeResult,
    SpatialInferenceConfig,
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


class CurationSpatialSink:
    """
    Persist the complete annotation-ready spatial contract.

    Arrays are written incrementally with NPY memmaps, so a 100-frame volume
    never needs to be materialized as another full 4-D copy in RAM.
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

        self._raw = None
        self._preprocessed = None
        self._binary = None
        self._source = None
        self._supervoxels = None
        self._final = None

        self._cells: list[
            pd.DataFrame
        ] = []
        self._written_frames: set[
            int
        ] = set()

        self.cells_dir = (
            self.paths.inference_run(
                self.run_id
            )
            / "cells"
        )
        self.cells_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

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

        movies = self.paths.movies(
            self.run_id
        )
        movies.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._raw = np.lib.format.open_memmap(
            self.paths.raw(self.run_id),
            mode="w+",
            dtype=result.raw.dtype,
            shape=shape,
        )
        self._preprocessed = np.lib.format.open_memmap(
            self.paths.preprocessed(
                self.run_id
            ),
            mode="w+",
            dtype=np.float16,
            shape=shape,
        )
        self._binary = np.lib.format.open_memmap(
            self.paths.binary_mask(
                self.run_id
            ),
            mode="w+",
            dtype=np.uint8,
            shape=shape,
        )
        self._source = np.lib.format.open_memmap(
            self.paths.source_instances(
                self.run_id
            ),
            mode="w+",
            dtype=np.int32,
            shape=shape,
        )
        self._supervoxels = np.lib.format.open_memmap(
            self.paths.supervoxels(
                self.run_id
            ),
            mode="w+",
            dtype=np.int32,
            shape=shape,
        )
        self._final = np.lib.format.open_memmap(
            self.paths.final_instances(
                self.run_id
            ),
            mode="w+",
            dtype=np.int32,
            shape=shape,
        )

    def write_frame(
        self,
        result: SpatialFrameResult,
    ) -> None:
        if self._raw is None:
            self._initialize(
                result
            )

        frame = int(
            result.frame
        )
        if not (
            0 <= frame < self.frame_count
        ):
            raise IndexError(
                f"Frame {frame} is outside "
                f"0..{self.frame_count - 1}"
            )
        if frame in self._written_frames:
            raise RuntimeError(
                f"Frame {frame} was written twice"
            )

        self._raw[frame] = np.asarray(
            result.raw
        )
        self._preprocessed[frame] = np.asarray(
            result.preprocessed,
            dtype=np.float16,
        )
        self._binary[frame] = np.asarray(
            result.source_mask > 0,
            dtype=np.uint8,
        )
        self._source[frame] = np.asarray(
            result.source_labels,
            dtype=np.int32,
        )
        self._supervoxels[frame] = np.asarray(
            result.supervoxel_labels,
            dtype=np.int32,
        )
        self._final[frame] = np.asarray(
            result.final_labels,
            dtype=np.int32,
        )

        cells = result.cells.copy()
        cells["frame"] = frame
        self._cells.append(
            cells
        )
        _atomic_csv(
            self.cells_dir
            / f"t{frame:03d}.csv",
            cells,
        )
        self._written_frames.add(
            frame
        )

    def _flush_movies(
        self,
    ) -> None:
        for movie in (
            self._raw,
            self._preprocessed,
            self._binary,
            self._source,
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
        expected = set(
            range(self.frame_count)
        )
        if self._written_frames != expected:
            missing = sorted(
                expected
                - self._written_frames
            )
            raise RuntimeError(
                "Cannot finalize incomplete "
                f"spatial cache; missing={missing}"
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
            self.paths.cells_csv(
                self.run_id
            ),
            combined,
        )

        summary = {
            "sample_id": str(
                sample_id
            ),
            "frame_count": int(
                volume_result.frame_count
            ),
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
            self.paths.spatial_summary(
                self.run_id
            ),
            summary,
        )
        atomic_json(
            self.paths.spatial_success(
                self.run_id
            ),
            {
                "status": "success",
                "sample_id": str(
                    sample_id
                ),
                "frame_count": int(
                    volume_result.frame_count
                ),
                "checkpoint_sha256": str(
                    runtime.checkpoint_sha256
                ),
            },
        )

    def close(
        self,
    ) -> None:
        self._flush_movies()
        self._raw = None
        self._preprocessed = None
        self._binary = None
        self._source = None
        self._supervoxels = None
        self._final = None
