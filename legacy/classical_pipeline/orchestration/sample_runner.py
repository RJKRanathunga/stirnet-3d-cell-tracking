"""Run the complete non-visual pipeline for one full sample."""

from __future__ import annotations

import json
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from .config import FullPipelineConfig
from .dataset import FullDatasetSample
from .manifest import PipelineManifest, utc_now
from .paths import SampleOutputPaths
from .stage_registry import get_stage_spec
from .stages import (
    PipelineState,
    load_processed_state,
    load_stage10_state,
    load_stage7_state,
    load_stage8_state,
    run_stage10,
    run_stage11,
    run_stage6,
    run_stage7,
    run_stage8,
)
from .validation import (
    stage_is_complete,
    validate_source_sample,
    validate_stage_output,
)


@dataclass(frozen=True)
class SampleRunResult:
    sample_id: str
    status: str
    completed_stages: tuple[int, ...]
    skipped_stages: tuple[int, ...]
    failed_stage: int | None = None
    error: str = ""


class SampleStageError(RuntimeError):
    def __init__(self, sample_id: str, stage: int, error: BaseException) -> None:
        self.sample_id = sample_id
        self.stage = int(stage)
        self.original_error = error
        super().__init__(
            f"{sample_id}: Stage {stage} failed: {type(error).__name__}: {error}"
        )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)
        file.write("\n")
    temporary.replace(path)


def _write_source_metadata(
    paths: SampleOutputPaths,
    sample: FullDatasetSample,
    raw_shape: tuple[int, int, int, int],
) -> None:
    _write_json(
        paths.source_metadata,
        {
            "sample_id": sample.sample_id,
            "sample_directory": str(sample.sample_directory),
            "zarr_path": str(sample.zarr_path),
            "zarr_array_path": str(sample.zarr_array_path),
            "raw_shape_tzyx": list(raw_shape),
            "frame_count": int(raw_shape[0]),
            "ground_truth_nodes_path": (
                str(sample.ground_truth_nodes_path)
                if sample.ground_truth_nodes_path is not None
                else None
            ),
            "ground_truth_edges_path": (
                str(sample.ground_truth_edges_path)
                if sample.ground_truth_edges_path is not None
                else None
            ),
            "ground_truth_available": bool(sample.has_ground_truth),
        },
    )


def _clean_for_execution(paths: SampleOutputPaths, stage: int) -> Path:
    directory = paths.stage_directory(stage)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _load_completed_stage(
    state: PipelineState,
    paths: SampleOutputPaths,
    stage: int,
) -> None:
    directory = paths.stage_directory(stage)
    if stage == 6:
        state.processed = load_processed_state(directory)
    elif stage == 7:
        state.stage7 = load_stage7_state(directory)
    elif stage == 8:
        state.stage8 = load_stage8_state(directory)
    elif stage == 10:
        state.stage10 = load_stage10_state(directory)
    elif stage == 11:
        # No later stage in this runner consumes Stage 11.
        return
    else:
        raise ValueError(f"Unsupported stage: {stage}")


def _require_prerequisites(
    stage: int,
    state: PipelineState,
    paths: SampleOutputPaths,
) -> None:
    if stage >= 7 and state.processed is None:
        validate_stage_output(6, paths.stage_directory(6))
        state.processed = load_processed_state(paths.stage_directory(6))

    if stage in {8, 11} and state.stage7 is None:
        validate_stage_output(7, paths.stage_directory(7))
        state.stage7 = load_stage7_state(paths.stage_directory(7))

    if stage in {10, 11} and state.stage8 is None:
        validate_stage_output(8, paths.stage_directory(8))
        state.stage8 = load_stage8_state(paths.stage_directory(8))

    if stage == 11 and state.stage10 is None:
        validate_stage_output(10, paths.stage_directory(10))
        state.stage10 = load_stage10_state(paths.stage_directory(10))


def _execute_stage(
    stage: int,
    *,
    sample: FullDatasetSample,
    paths: SampleOutputPaths,
    config: FullPipelineConfig,
    state: PipelineState,
) -> None:
    if stage == 6:
        state.processed = run_stage6(sample, paths, config)
        return

    assert state.processed is not None
    if stage == 7:
        state.stage7 = run_stage7(
            sample,
            paths,
            config,
            state.processed,
        )
    elif stage == 8:
        assert state.stage7 is not None
        state.stage8 = run_stage8(
            sample,
            paths,
            state.processed,
            state.stage7,
        )
    elif stage == 10:
        assert state.stage8 is not None
        state.stage10 = run_stage10(
            sample,
            paths,
            state.processed,
            state.stage8,
        )
    elif stage == 11:
        assert state.stage7 is not None
        assert state.stage8 is not None
        assert state.stage10 is not None
        state.stage11 = run_stage11(
            sample,
            paths,
            config,
            state.processed,
            state.stage7,
            state.stage8,
            state.stage10,
        )
    else:
        raise ValueError(f"Unsupported stage: {stage}")


def run_sample(
    sample: FullDatasetSample,
    config: FullPipelineConfig,
    manifest: PipelineManifest,
) -> SampleRunResult:
    paths = SampleOutputPaths(config.output_root, sample.sample_id)
    paths.sample_root.mkdir(parents=True, exist_ok=True)

    raw_shape = validate_source_sample(sample)
    _write_source_metadata(paths, sample, raw_shape)

    print(
        f"\n[{sample.sample_id}] {raw_shape[0]} frames | "
        f"shape={raw_shape} | ground_truth={'yes' if sample.has_ground_truth else 'no'}",
        flush=True,
    )

    state = PipelineState()
    completed: list[int] = []
    skipped: list[int] = []

    for stage in config.selected_stages:
        spec = get_stage_spec(stage)
        _require_prerequisites(stage, state, paths)

        if config.resume and stage_is_complete(paths, stage):
            print(
                f"  Stage {stage} ({spec.name}): already complete, skipping",
                flush=True,
            )
            _load_completed_stage(state, paths, stage)
            skipped.append(stage)
            manifest.update(
                sample_id=sample.sample_id,
                stage=stage,
                status="skipped_resume",
                output_directory=paths.stage_directory(stage),
                completed_at=utc_now(),
            )
            continue

        output_directory = _clean_for_execution(paths, stage)
        started_at = utc_now()
        started = time.perf_counter()
        manifest.update(
            sample_id=sample.sample_id,
            stage=stage,
            status="running",
            output_directory=output_directory,
            started_at=started_at,
        )
        print(
            f"  Stage {stage} ({spec.name}): running -> {output_directory}",
            flush=True,
        )

        try:
            _execute_stage(
                stage,
                sample=sample,
                paths=paths,
                config=config,
                state=state,
            )
            validate_stage_output(stage, output_directory)
            runtime = time.perf_counter() - started
            completed_at = utc_now()
            _write_json(
                paths.success_marker(stage),
                {
                    "sample_id": sample.sample_id,
                    "stage": int(stage),
                    "stage_name": spec.name,
                    "status": "completed",
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "runtime_seconds": runtime,
                    "output_directory": str(output_directory),
                    "source_zarr": str(sample.zarr_path),
                },
            )
            failed_marker = paths.failed_marker(stage)
            if failed_marker.exists():
                failed_marker.unlink()
            completed.append(stage)
            manifest.update(
                sample_id=sample.sample_id,
                stage=stage,
                status="completed",
                output_directory=output_directory,
                started_at=started_at,
                completed_at=completed_at,
                runtime_seconds=runtime,
            )
            print(
                f"  Stage {stage}: completed in {runtime:.1f} s",
                flush=True,
            )
        except Exception as error:
            runtime = time.perf_counter() - started
            completed_at = utc_now()
            trace = traceback.format_exc()
            _write_json(
                paths.failed_marker(stage),
                {
                    "sample_id": sample.sample_id,
                    "stage": int(stage),
                    "stage_name": spec.name,
                    "status": "failed",
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "runtime_seconds": runtime,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": trace,
                },
            )
            manifest.update(
                sample_id=sample.sample_id,
                stage=stage,
                status="failed",
                output_directory=output_directory,
                started_at=started_at,
                completed_at=completed_at,
                runtime_seconds=runtime,
                error=error,
            )
            _write_json(
                paths.status_path,
                {
                    "sample_id": sample.sample_id,
                    "status": "failed",
                    "completed_stages": completed,
                    "skipped_stages": skipped,
                    "failed_stage": int(stage),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise SampleStageError(sample.sample_id, stage, error) from error

    result = SampleRunResult(
        sample_id=sample.sample_id,
        status="completed",
        completed_stages=tuple(completed),
        skipped_stages=tuple(skipped),
    )
    _write_json(
        paths.status_path,
        {
            "sample_id": sample.sample_id,
            "status": result.status,
            "completed_stages": list(result.completed_stages),
            "skipped_stages": list(result.skipped_stages),
            "failed_stage": None,
        },
    )
    return result


__all__ = [
    "SampleRunResult",
    "SampleStageError",
    "run_sample",
]
