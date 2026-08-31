from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
from pathlib import Path
import time
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


# STIRNET_PARALLEL_SPATIAL_PIPELINE_V2
#
# Canonical production owner of the one-frame CPU lookahead scheduler.
# Exactly one worker prepares t+1 while the main thread owns CUDA inference for
# t. No worker thread is allowed to touch CUDA. Host-memory lookahead is bounded
# to one prepared frame.
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
) -> SpatialVolumeResult:
    from src.api import (
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

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="stirnet-spatial-prep",
    ) as prep_executor:
        prepared_future = (
            prep_executor.submit(
                prepare_spatial_frame,
                sample_zarr,
                0,
                config=config,
                segmentation_config=(
                    segmentation_config
                ),
            )
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
                prepared_future.result()
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
                prepared_future = (
                    prep_executor.submit(
                        prepare_spatial_frame,
                        sample_zarr,
                        next_frame,
                        config=config,
                        segmentation_config=(
                            segmentation_config
                        ),
                    )
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
