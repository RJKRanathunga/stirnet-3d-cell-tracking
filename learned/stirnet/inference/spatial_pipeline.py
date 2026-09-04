from __future__ import annotations

# DATASET_CURATION_CANONICAL_SKIP_V1

from dataclasses import dataclass
import gc
import multiprocessing
from pathlib import Path
import queue
import time
import traceback
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from learned.stirnet.inference.runtime import (
    SpatialInferenceConfig,
    SpatialModelRuntime,
    run_tiled_spatial,
    tensor_numpy,
)
from learned.stirnet.inference.spatial_input import (
    PreparedSpatialFrame,
    canonical_source_segmentation_config,
    prepare_spatial_frame,
)


@dataclass
class SpatialFrameResult:
    frame: int
    raw: np.ndarray
    preprocessed: np.ndarray
    source_mask: np.ndarray
    source_labels: np.ndarray
    supervoxel_labels: np.ndarray
    multicut_labels: np.ndarray
    final_labels: np.ndarray
    cells: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class SpatialVolumeResult:
    frame_count: int
    spatial_shape_zyx: tuple[int, int, int]
    frame_summaries: tuple[dict[str, Any], ...]
    total_seconds: float


def _strict_refinement_check(
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
) -> None:
    parent = np.asarray(parent_labels)
    child = np.asarray(child_labels)
    if parent.shape != child.shape:
        raise RuntimeError(
            "Split-only refinement changed volume shape: "
            f"{parent.shape} -> {child.shape}"
        )

    parent_by_child: dict[int, int] = {}
    for child_id, parent_id in zip(
        child.reshape(-1),
        parent.reshape(-1),
    ):
        child_value = int(child_id)
        if child_value <= 0:
            continue
        parent_value = int(parent_id)
        old = parent_by_child.get(child_value)
        if old is not None and old != parent_value:
            raise RuntimeError(
                "Source-core split-only invariant failed: "
                f"output label {child_value} contains "
                f"multicut labels {old} and {parent_value}."
            )
        parent_by_child[child_value] = parent_value


def apply_source_core_split_only(
    before: np.ndarray,
    watershed: np.ndarray,
    separator_probability: np.ndarray,
    source_mask: np.ndarray,
    source_labels: np.ndarray,
    spacing: tuple[float, float, float],
    dref_um: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Production split-only final filter promoted from the validated path."""
    from learned.stirnet.model.config import (
        InferenceConfig,
    )
    from learned.stirnet.model.postprocess.source_core_split import (
        SourceCoreSplitOnlyFilter,
    )

    cfg = InferenceConfig()
    cfg.source_core_split_anchor_mode = (
        "prefer_source_instances"
    )
    filt = SourceCoreSplitOnlyFilter(cfg)

    with torch.inference_mode():
        state = filt(
            [
                torch.as_tensor(
                    before,
                    dtype=torch.long,
                )
            ],
            torch.as_tensor(
                (source_mask > 0)[None],
                dtype=torch.float32,
            ),
            torch.as_tensor(
                separator_probability[None],
                dtype=torch.float32,
            ),
            torch.tensor(
                [spacing],
                dtype=torch.float32,
            ),
            torch.tensor(
                [float(dref_um)],
                dtype=torch.float32,
            ),
            supervoxel_labels=[
                torch.as_tensor(
                    watershed,
                    dtype=torch.long,
                )
            ],
            source_instance_labels=torch.as_tensor(
                source_labels[None],
                dtype=torch.long,
            ),
        )

    after = (
        state.labels[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int32, copy=False)
    )
    _strict_refinement_check(
        before,
        after,
    )

    diagnostics = {
        "candidate_count": int(
            state.candidate_count
        ),
        "applied_count": int(
            state.applied_count
        ),
        "skipped_too_many_cores": int(
            state.skipped_too_many_cores
        ),
    }
    return after, diagnostics


def _count_labels(
    labels: np.ndarray,
) -> int:
    values = np.unique(labels)
    return int(
        np.count_nonzero(values > 0)
    )


class SpatialPreparationTimeout(RuntimeError):
    """Hard wall-clock timeout for one CPU frame-preparation task."""

    def __init__(
        self,
        *,
        frame: int,
        timeout_seconds: float,
        elapsed_seconds: float,
    ) -> None:
        self.frame = int(frame)
        self.timeout_seconds = float(timeout_seconds)
        self.elapsed_seconds = float(elapsed_seconds)
        super().__init__(
            f"t={self.frame:03d} preparation exceeded "
            f"{self.timeout_seconds:.1f}s "
            f"(elapsed={self.elapsed_seconds:.1f}s)"
        )


class SpatialPreparationWorkerError(RuntimeError):
    """Error raised in the isolated CPU preparation process."""

    def __init__(
        self,
        *,
        frame: int,
        error_type: str,
        message: str,
        traceback_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.frame = int(frame)
        self.error_type = str(error_type)
        self.remote_message = str(message)
        self.traceback_text = str(traceback_text)
        self.metadata = dict(metadata or {})
        super().__init__(
            f"t={self.frame:03d} preparation worker raised "
            f"{self.error_type}: {self.remote_message}\n"
            f"{self.traceback_text}"
        )


def _serialize_preparation_error(
    frame: int,
    error: BaseException,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    for name in (
        "frame",
        "reason_code",
    ):
        if hasattr(error, name):
            value = getattr(error, name)
            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                    type(None),
                ),
            ):
                metadata[name] = value

    for name in (
        "metrics",
        "policy",
    ):
        value = getattr(
            error,
            name,
            None,
        )
        as_dict = getattr(
            value,
            "as_dict",
            None,
        )
        if callable(as_dict):
            metadata[name] = as_dict()

    return {
        "frame": int(frame),
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback_text": traceback.format_exc(),
        "metadata": metadata,
    }


def _spatial_preparation_worker_main(
    task_queue,
    result_queue,
    sample_zarr: str,
    config: SpatialInferenceConfig,
    segmentation_config,
    source_mask_validator,
) -> None:
    """Persistent CPU-only worker. It must never initialize or touch CUDA."""
    while True:
        frame = task_queue.get()
        if frame is None:
            return

        frame = int(frame)
        try:
            prepared = prepare_spatial_frame(
                sample_zarr,
                frame,
                config=config,
                segmentation_config=segmentation_config,
                source_mask_validator=source_mask_validator,
            )
        except BaseException as error:
            result_queue.put(
                (
                    "error",
                    frame,
                    _serialize_preparation_error(
                        frame,
                        error,
                    ),
                )
            )
            return

        result_queue.put(
            (
                "ok",
                frame,
                prepared,
            )
        )


class _SpatialPreparationProcess:
    """
    One persistent spawned process for CPU frame preparation.

    The main process owns CUDA. A single task is in flight, preserving the
    existing one-frame lookahead while making preparation force-terminable.
    """

    def __init__(
        self,
        *,
        sample_zarr: Path,
        config: SpatialInferenceConfig,
        segmentation_config,
        source_mask_validator,
        timeout_seconds: float,
    ) -> None:
        timeout_seconds = float(
            timeout_seconds
        )
        if not np.isfinite(
            timeout_seconds
        ) or timeout_seconds <= 0:
            raise ValueError(
                "preparation_timeout_seconds must be positive"
            )

        self.timeout_seconds = timeout_seconds
        self._ctx = multiprocessing.get_context(
            "spawn"
        )
        self._tasks = self._ctx.Queue(
            maxsize=1
        )
        self._results = self._ctx.Queue(
            maxsize=1
        )
        self._process = self._ctx.Process(
            target=_spatial_preparation_worker_main,
            args=(
                self._tasks,
                self._results,
                str(sample_zarr),
                config,
                segmentation_config,
                source_mask_validator,
            ),
            name="stirnet-spatial-prep",
            daemon=True,
        )
        self._pending_frame: int | None = None
        self._submitted_at: float | None = None
        self._closed = False
        self._process.start()

    def submit(
        self,
        frame: int,
    ) -> None:
        if self._closed:
            raise RuntimeError(
                "Preparation process is closed"
            )
        if self._pending_frame is not None:
            raise RuntimeError(
                "Only one preparation frame may be in flight"
            )
        if not self._process.is_alive():
            raise SpatialPreparationWorkerError(
                frame=int(frame),
                error_type="WorkerExited",
                message=(
                    "CPU preparation process exited "
                    f"with code {self._process.exitcode}"
                ),
                traceback_text="",
            )

        self._tasks.put(
            int(frame)
        )
        self._pending_frame = int(frame)
        self._submitted_at = (
            time.perf_counter()
        )

    def result(
        self,
    ) -> PreparedSpatialFrame:
        if self._pending_frame is None:
            raise RuntimeError(
                "No preparation frame is pending"
            )
        if self._submitted_at is None:
            raise RuntimeError(
                "Pending preparation has no submit timestamp"
            )

        frame = int(
            self._pending_frame
        )
        deadline = (
            self._submitted_at
            + self.timeout_seconds
        )

        while True:
            now = time.perf_counter()
            remaining = deadline - now
            if remaining <= 0:
                elapsed = (
                    now
                    - self._submitted_at
                )
                self.terminate()
                raise SpatialPreparationTimeout(
                    frame=frame,
                    timeout_seconds=(
                        self.timeout_seconds
                    ),
                    elapsed_seconds=elapsed,
                )

            try:
                message = self._results.get(
                    timeout=min(
                        0.5,
                        remaining,
                    )
                )
            except queue.Empty:
                if not self._process.is_alive():
                    raise SpatialPreparationWorkerError(
                        frame=frame,
                        error_type="WorkerExited",
                        message=(
                            "CPU preparation process exited "
                            f"with code {self._process.exitcode}"
                        ),
                        traceback_text="",
                    )
                continue

            status = str(
                message[0]
            )
            result_frame = int(
                message[1]
            )
            payload = message[2]

            self._pending_frame = None
            self._submitted_at = None

            if result_frame != frame:
                raise RuntimeError(
                    "Preparation ordering failure: "
                    f"expected t={frame:03d}, "
                    f"got t={result_frame:03d}"
                )

            if status == "ok":
                if not isinstance(
                    payload,
                    PreparedSpatialFrame,
                ):
                    raise TypeError(
                        "Preparation worker returned "
                        f"{type(payload).__name__}, "
                        "expected PreparedSpatialFrame"
                    )
                return payload

            if status == "error":
                raise SpatialPreparationWorkerError(
                    frame=int(
                        payload.get(
                            "frame",
                            frame,
                        )
                    ),
                    error_type=str(
                        payload.get(
                            "error_type",
                            "RemoteError",
                        )
                    ),
                    message=str(
                        payload.get(
                            "message",
                            "",
                        )
                    ),
                    traceback_text=str(
                        payload.get(
                            "traceback_text",
                            "",
                        )
                    ),
                    metadata=dict(
                        payload.get(
                            "metadata",
                            {},
                        )
                    ),
                )

            raise RuntimeError(
                f"Unknown preparation worker status: {status!r}"
            )

    def terminate(
        self,
    ) -> None:
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(
                timeout=2.0
            )

        if self._process.is_alive():
            kill = getattr(
                self._process,
                "kill",
                None,
            )
            if callable(kill):
                kill()
                self._process.join(
                    timeout=2.0
                )

    def close(
        self,
        *,
        graceful: bool,
    ) -> None:
        if self._closed:
            return
        self._closed = True

        if (
            graceful
            and self._process.is_alive()
        ):
            try:
                self._tasks.put(
                    None,
                    timeout=0.5,
                )
            except Exception:
                pass
            self._process.join(
                timeout=2.0
            )

        if self._process.is_alive():
            self.terminate()

        for channel in (
            self._tasks,
            self._results,
        ):
            try:
                channel.cancel_join_thread()
            except Exception:
                pass
            try:
                channel.close()
            except Exception:
                pass


# STIRNET_PARALLEL_SPATIAL_PIPELINE_V3
#
# Canonical production owner of the one-frame CPU lookahead scheduler.
# Exactly one spawned CPU process prepares t+1 while the main process owns
# CUDA inference for t. The process can be force-terminated on timeout or
# KeyboardInterrupt. Host-memory lookahead remains bounded to one frame.
def run_parallel_spatial_volume(
    sample_zarr: str | Path,
    runtime: SpatialModelRuntime,
    *,
    config: SpatialInferenceConfig,
    sample_id: str | None = None,
    on_frame: Callable[
        [SpatialFrameResult],
        None,
    ]
    | None = None,
    require_nonempty_cells: bool = True,
    source_mask_validator: Callable[[int, np.ndarray], None] | None = None,
    preparation_timeout_seconds: float = 300.0,
) -> SpatialVolumeResult:
    from src.source_instances import (
        detect_cells,
        extract_cell_features,
    )
    from src.io import open_sample

    config.validate()
    sample_zarr = Path(
        sample_zarr
    ).expanduser().resolve()

    image = open_sample(sample_zarr)
    if len(image.shape) != 4:
        raise ValueError(
            "Expected T,Z,Y,X image, "
            f"got shape={image.shape}"
        )

    frame_count = int(
        image.shape[0]
    )
    spatial_shape = tuple(
        int(v)
        for v in image.shape[1:]
    )
    del image

    if frame_count <= 0:
        raise ValueError(
            "Sample has an empty time axis"
        )

    name = (
        str(sample_id)
        if sample_id is not None
        else sample_zarr.stem
    )
    segmentation_config = (
        canonical_source_segmentation_config()
    )

    summaries: list[
        dict[str, Any]
    ] = []
    run_started = time.perf_counter()

    prep_worker = _SpatialPreparationProcess(
        sample_zarr=sample_zarr,
        config=config,
        segmentation_config=segmentation_config,
        source_mask_validator=source_mask_validator,
        timeout_seconds=float(
            preparation_timeout_seconds
        ),
    )
    completed = False

    try:
        prep_worker.submit(0)
        print(
            f"[{name} t=000] prep started "
            f"(timeout={float(preparation_timeout_seconds):.0f}s)",
            flush=True,
        )

        for frame in range(
            frame_count
        ):
            frame_started = (
                time.perf_counter()
            )

            wait_started = (
                time.perf_counter()
            )
            prepared = (
                prep_worker.result()
            )
            preparation_wait_seconds = (
                time.perf_counter()
                - wait_started
            )

            if prepared.frame != frame:
                raise RuntimeError(
                    f"{name}: preparation ordering failure: "
                    f"expected t={frame:03d}, "
                    f"got t={prepared.frame:03d}"
                )

            # Critical scheduling invariant:
            # submit t+1 BEFORE CUDA starts for t.
            next_frame = frame + 1
            if next_frame < frame_count:
                prep_worker.submit(
                    next_frame
                )
                print(
                    f"[{name} t={next_frame:03d}] prep started "
                    f"(timeout={float(preparation_timeout_seconds):.0f}s)",
                    flush=True,
                )

            (
                result,
                spatial_gpu,
                amp_name,
                inference_seconds,
                peak_gib,
            ) = run_tiled_spatial(
                runtime,
                prepared.spatial,
                config.spacing_zyx_um,
                prepared.dref_um,
            )

            before = tensor_numpy(
                result.spatial_partition.labels[0],
                np.int32,
            )
            watershed = tensor_numpy(
                result.supervoxel_labels[0],
                np.int32,
            )
            separator_probability = (
                tensor_numpy(
                    result.dense.geometry.probabilities()[
                        "separator"
                    ][0, 0],
                    np.float32,
                )
            )

            post_started = (
                time.perf_counter()
            )
            (
                final_labels,
                split_diag,
            ) = apply_source_core_split_only(
                before,
                watershed,
                separator_probability,
                prepared.source_mask,
                prepared.source_labels,
                config.spacing_zyx_um,
                prepared.dref_um,
            )

            cells = detect_cells(
                final_labels
            )
            cells = extract_cell_features(
                cells,
                final_labels,
                prepared.preprocessed,
            )

            if (
                require_nonempty_cells
                and cells.empty
            ):
                raise RuntimeError(
                    f"{name} t={frame}: "
                    "STIR-Net produced no cells"
                )

            postprocess_seconds = (
                time.perf_counter()
                - post_started
            )
            total_seconds = (
                time.perf_counter()
                - frame_started
            )
            hidden_seconds = max(
                float(
                    prepared.preparation_seconds
                )
                - float(
                    preparation_wait_seconds
                ),
                0.0,
            )
            hidden_fraction = (
                hidden_seconds
                / float(
                    prepared.preparation_seconds
                )
                if prepared.preparation_seconds
                > 0
                else 0.0
            )

            summary = {
                "frame": int(frame),
                "source_instances": _count_labels(
                    prepared.source_labels
                ),
                "multicut_instances": _count_labels(
                    before
                ),
                "final_instances": _count_labels(
                    final_labels
                ),
                "split_candidates": int(
                    split_diag[
                        "candidate_count"
                    ]
                ),
                "splits_applied": int(
                    split_diag[
                        "applied_count"
                    ]
                ),
                "amp_dtype": str(
                    amp_name
                ),
                "inference_seconds": float(
                    inference_seconds
                ),
                "preparation_seconds": float(
                    prepared.preparation_seconds
                ),
                "preparation_wait_seconds": float(
                    preparation_wait_seconds
                ),
                "preparation_hidden_seconds": float(
                    hidden_seconds
                ),
                "preparation_hidden_fraction": float(
                    hidden_fraction
                ),
                "postprocess_seconds": float(
                    postprocess_seconds
                ),
                "total_seconds": float(
                    total_seconds
                ),
                "peak_allocated_vram_gib": float(
                    peak_gib
                ),
            }

            frame_result = (
                SpatialFrameResult(
                    frame=int(frame),
                    raw=prepared.raw,
                    preprocessed=(
                        prepared.preprocessed
                    ),
                    source_mask=(
                        prepared.source_mask
                    ),
                    source_labels=(
                        prepared.source_labels
                    ),
                    supervoxel_labels=(
                        watershed
                    ),
                    multicut_labels=before,
                    final_labels=final_labels,
                    cells=cells,
                    summary=summary,
                )
            )

            if on_frame is not None:
                on_frame(frame_result)

            summaries.append(summary)

            print(
                f"[{name} t={frame:03d}] "
                f"source={summary['source_instances']} -> "
                f"multicut={summary['multicut_instances']} -> "
                f"final={summary['final_instances']} | "
                f"splits={summary['splits_applied']} | "
                f"prep={summary['preparation_seconds']:.2f}s "
                f"prep_wait={summary['preparation_wait_seconds']:.2f}s "
                f"hidden={100.0 * summary['preparation_hidden_fraction']:.1f}% | "
                f"infer={summary['inference_seconds']:.2f}s "
                f"post={summary['postprocess_seconds']:.2f}s "
                f"total={summary['total_seconds']:.2f}s "
                f"VRAM={summary['peak_allocated_vram_gib']:.2f}GiB",
                flush=True,
            )

            del frame_result
            del result, spatial_gpu
            del separator_probability
            del watershed, before
            del final_labels, cells
            del prepared

            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        completed = True
    finally:
        prep_worker.close(
            graceful=completed
        )

    return SpatialVolumeResult(
        frame_count=frame_count,
        spatial_shape_zyx=spatial_shape,
        frame_summaries=tuple(
            summaries
        ),
        total_seconds=float(
            time.perf_counter()
            - run_started
        ),
    )
