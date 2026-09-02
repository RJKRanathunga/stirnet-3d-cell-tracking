from __future__ import annotations

# DATASET_CURATION_CANONICAL_SKIP_V1

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
from pathlib import Path
import shutil
from typing import Literal, Sequence
from uuid import uuid4

import torch

from dataset_curation._repo import repo_root
from dataset_curation.catalog import VolumeRecord
from dataset_curation.errors import ArtifactError
from dataset_curation.io.atomic import atomic_json
from dataset_curation.inference.quality import (
    SourceMaskMetrics,
    SourceQualityPolicy,
    SourceQualityRejected,
    validate_source_mask,
)
from dataset_curation.inference.spatial_sink import CurationSpatialSink
from dataset_curation.inference.trackastra import run_trackastra
from learned.stirnet.inference import (
    SpatialInferenceConfig,
    load_spatial_runtime,
    run_parallel_spatial_volume,
)
from learned.stirnet.inference.spatial_pipeline import (
    SpatialPreparationTimeout,
    SpatialPreparationWorkerError,
)


DEFAULT_TILE_SHAPE_ZYX = (32, 128, 128)
DEFAULT_TILE_OVERLAP_ZYX = (8, 32, 32)
DEFAULT_TILE_HALO_ZYX = (4, 16, 16)
DEFAULT_TILE_BATCH_SIZE = 1

DEFAULT_TRACKASTRA_MODEL = "ctc"
DEFAULT_TRACKASTRA_MODE = "greedy"
DEFAULT_TRACKASTRA_DEVICE = "cuda"


@dataclass(frozen=True)
class InferenceOutcome:
    status: Literal["complete", "reused", "skipped"]
    output: Path
    reason_code: str | None = None
    trigger_frame: int | None = None
    existing_skip: bool = False


def _parse_triplet(
    value: str,
    *,
    name: str,
    positive: bool,
) -> tuple[int, int, int]:
    rows = tuple(int(token.strip()) for token in str(value).split(","))
    if len(rows) != 3:
        raise ValueError(
            f"{name} must contain exactly three comma-separated integers"
        )
    if positive:
        if any(v <= 0 for v in rows):
            raise ValueError(f"{name} values must be positive")
    elif any(v < 0 for v in rows):
        raise ValueError(f"{name} values must be non-negative")
    return rows


def _checkpoint_step(path: Path) -> int:
    name = path.name
    prefix = "checkpoint_step_"
    suffix = ".pt"
    if name.startswith(prefix) and name.endswith(suffix):
        token = name[len(prefix) : -len(suffix)]
        if token.isdigit():
            return int(token)
    return -1


def _latest_checkpoint(directory: Path) -> Path:
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    best = directory / "best_checkpoint.pt"
    if best.is_file():
        return best.resolve()

    candidates = [
        path
        for path in directory.glob("checkpoint_step_*.pt")
        if path.is_file() and _checkpoint_step(path) >= 0
    ]
    if not candidates:
        candidates = [
            path
            for path in directory.rglob("checkpoint_step_*.pt")
            if path.is_file() and _checkpoint_step(path) >= 0
        ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found below {directory}")

    return max(
        candidates,
        key=lambda path: (_checkpoint_step(path), str(path)),
    ).resolve()


def resolve_default_checkpoint() -> Path:
    """
    Resolve the current spatial checkpoint.

    The historical run directory name may contain `investigations`, but this is
    an artifact location only; no Python code is imported from investigations.
    """
    root = repo_root()
    recovery = (
        root
        / "runs"
        / "stirnet"
        / "investigations"
        / "19_morphology_rag_v2_headroom_training"
        / "recovery"
        / "drosophila_12_morphology_rag_v2_headroom_h100"
    )

    for name in ("best_checkpoint.pt", "checkpoint_step_000600.pt"):
        candidate = recovery / name
        if candidate.is_file():
            return candidate.resolve()

    if recovery.is_dir():
        return _latest_checkpoint(recovery)

    raise FileNotFoundError(
        "Could not auto-discover the current STIR-Net spatial checkpoint. "
        "Pass --checkpoint <path> after `--`."
    )


def _resolve_checkpoint(
    checkpoint: str | None,
    checkpoint_dir: str | None,
) -> Path:
    if checkpoint is not None and checkpoint_dir is not None:
        raise ValueError("--checkpoint and --checkpoint-dir are mutually exclusive")

    if checkpoint_dir is not None:
        path = Path(checkpoint_dir).expanduser()
        if not path.is_absolute():
            path = repo_root() / path
        return _latest_checkpoint(path.resolve())

    if checkpoint is not None:
        path = Path(checkpoint).expanduser()
        if not path.is_absolute():
            path = repo_root() / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    return resolve_default_checkpoint()


def _parse_extra(values: Sequence[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--spatial-device", default="cuda")
    parser.add_argument("--tile-shape-zyx", default="32,128,128")
    parser.add_argument("--tile-overlap-zyx", default="8,32,32")
    parser.add_argument("--tile-halo-zyx", default="4,16,16")
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument(
        "--frame-prep-timeout",
        type=float,
        default=300.0,
        help=(
            "Hard wall-clock timeout in seconds for one CPU frame "
            "preparation task. Timed-out volumes are skipped."
        ),
    )
    parser.add_argument("--trackastra-model", default=DEFAULT_TRACKASTRA_MODEL)
    parser.add_argument("--trackastra-mode", default=DEFAULT_TRACKASTRA_MODE)
    parser.add_argument("--trackastra-device", default=DEFAULT_TRACKASTRA_DEVICE)
    parser.add_argument("--rebuild-trackastra", action="store_true")
    parser.add_argument("--overwrite-spatial", action="store_true")

    parsed, unknown = parser.parse_known_args(list(values))
    if unknown:
        raise ValueError(
            "Unsupported dataset-curation inference option(s): "
            + " ".join(unknown)
        )
    if not float(parsed.frame_prep_timeout) > 0.0:
        raise ValueError(
            "--frame-prep-timeout must be positive"
        )
    return parsed


def _cleanup_inference_artifacts(paths) -> None:
    """Remove only artifacts owned by canonical dataset-curation inference."""
    for directory in (
        paths.movies,
        paths.cells_dir,
        paths.trackastra_root,
        paths.suspect_scores,
    ):
        if directory.exists():
            shutil.rmtree(directory)

    for path in (
        paths.cells_csv,
        paths.spatial_summary,
        paths.spatial_success,
        paths.inference_manifest,
        paths.skip_marker,
    ):
        path.unlink(missing_ok=True)


def _existing_skip_outcome(record: VolumeRecord) -> InferenceOutcome:
    paths = record.paths
    try:
        payload = paths.read_skip_record()
        reason = str(payload.get("reason_code", "recorded_skip"))
        raw_frame = payload.get("trigger_frame")
        frame = int(raw_frame) if raw_frame is not None else None
    except Exception:
        reason = "recorded_skip"
        frame = None

    frame_text = f" t={frame:03d}" if frame is not None else ""
    print(
        f"[skip] {record.split}/{record.volume_id}: "
        f"recorded {reason}{frame_text}",
        flush=True,
    )
    return InferenceOutcome(
        status="skipped",
        output=paths.preprocessed_root,
        reason_code=reason,
        trigger_frame=frame,
        existing_skip=True,
    )


def _record_quality_skip(
    record: VolumeRecord,
    rejection: SourceQualityRejected,
) -> InferenceOutcome:
    paths = record.paths
    _cleanup_inference_artifacts(paths)
    paths.preprocessed_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "kind": "biohub_inference_skip",
        "status": "skipped",
        "volume_id": record.volume_id,
        "split": record.split,
        "reason_code": rejection.reason_code,
        "trigger_frame": int(rejection.frame),
        "metrics": rejection.metrics.as_dict(),
        "policy": rejection.policy.as_dict(),
        "source_zarr": str(paths.zarr),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(paths.skip_marker, payload)

    print(
        f"[SKIPPED] {record.split}/{record.volume_id}: "
        f"{rejection.reason_code} at t={rejection.frame:03d} | "
        f"foreground={rejection.metrics.foreground_voxels} "
        f"largest_component={rejection.metrics.largest_component_voxels} "
        f"foreground_fraction="
        f"{rejection.metrics.largest_component_fraction_of_foreground:.1%} "
        f"volume_fraction="
        f"{rejection.metrics.largest_component_fraction_of_volume:.1%}",
        flush=True,
    )
    return InferenceOutcome(
        status="skipped",
        output=paths.preprocessed_root,
        reason_code=rejection.reason_code,
        trigger_frame=int(rejection.frame),
        existing_skip=False,
    )


def _quality_rejection_from_worker(
    error: SpatialPreparationWorkerError,
) -> SourceQualityRejected | None:
    if error.error_type != "SourceQualityRejected":
        return None

    metadata = error.metadata
    try:
        return SourceQualityRejected(
            frame=int(
                metadata["frame"]
            ),
            metrics=SourceMaskMetrics(
                **dict(
                    metadata["metrics"]
                )
            ),
            policy=SourceQualityPolicy(
                **dict(
                    metadata["policy"]
                )
            ),
            reason_code=str(
                metadata.get(
                    "reason_code",
                    "pathological_connected_foreground",
                )
            ),
        )
    except Exception:
        return None


def _record_preparation_timeout(
    record: VolumeRecord,
    timeout: SpatialPreparationTimeout,
) -> InferenceOutcome:
    paths = record.paths
    _cleanup_inference_artifacts(paths)
    paths.preprocessed_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    reason_code = "preparation_timeout"
    payload = {
        "schema_version": 1,
        "kind": "biohub_inference_skip",
        "status": "skipped",
        "volume_id": record.volume_id,
        "split": record.split,
        "reason_code": reason_code,
        "trigger_frame": int(
            timeout.frame
        ),
        "timeout_seconds": float(
            timeout.timeout_seconds
        ),
        "elapsed_seconds": float(
            timeout.elapsed_seconds
        ),
        "stage": "prepare_spatial_frame",
        "source_zarr": str(
            paths.zarr
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }
    atomic_json(
        paths.skip_marker,
        payload,
    )

    print(
        f"[SKIPPED] {record.split}/{record.volume_id}: "
        f"{reason_code} at t={timeout.frame:03d} after "
        f"{timeout.elapsed_seconds:.1f}s "
        f"(limit={timeout.timeout_seconds:.1f}s)",
        flush=True,
    )
    return InferenceOutcome(
        status="skipped",
        output=paths.preprocessed_root,
        reason_code=reason_code,
        trigger_frame=int(
            timeout.frame
        ),
        existing_skip=False,
    )

class StirNetTrackastraBackend:
    """
    Production curation backend.

    Spatial inference is owned by learned.stirnet.inference. Trackastra remains
    a curation-level downstream adapter. No runtime code is loaded from
    investigations/.
    """

    name = "production_stirnet_trackastra"

    def run_volume(
        self,
        record: VolumeRecord,
        *,
        force: bool = False,
        retry_skipped: bool = False,
        extra_args: Sequence[str] = (),
    ) -> InferenceOutcome:
        paths = record.paths
        paths.ensure_output_roots()
        options = _parse_extra(extra_args)

        if paths.inference_skipped() and not retry_skipped:
            return _existing_skip_outcome(record)
        if retry_skipped:
            paths.skip_marker.unlink(missing_ok=True)

        force_spatial = bool(force or options.overwrite_spatial)

        if record.frame_count is None:
            raise ArtifactError(
                f"Could not determine frame count for {record.volume_id}."
            )
        frame_count = int(record.frame_count)

        if (
            not force_spatial
            and not options.rebuild_trackastra
            and paths.inference_complete(frame_count=frame_count)
        ):
            print(
                f"[reuse] {record.split}/{record.volume_id}: "
                "canonical inference is complete.",
                flush=True,
            )
            return InferenceOutcome(
                status="reused",
                output=paths.preprocessed_root,
            )

        output = paths.preprocessed_root
        output.mkdir(parents=True, exist_ok=True)
        spatial_was_run = False

        if force_spatial or not paths.spatial_complete(frame_count=frame_count):
            # Any newly generated spatial labels invalidate a previous Trackastra graph.
            if paths.trackastra_root.exists():
                shutil.rmtree(paths.trackastra_root)
            paths.inference_manifest.unlink(missing_ok=True)

            checkpoint = _resolve_checkpoint(
                options.checkpoint,
                options.checkpoint_dir,
            )
            config = SpatialInferenceConfig(
                spacing_zyx_um=(1.625, 0.40625, 0.40625),
                tile_shape_zyx=_parse_triplet(
                    options.tile_shape_zyx,
                    name="--tile-shape-zyx",
                    positive=True,
                ),
                tile_overlap_zyx=_parse_triplet(
                    options.tile_overlap_zyx,
                    name="--tile-overlap-zyx",
                    positive=False,
                ),
                tile_halo_zyx=_parse_triplet(
                    options.tile_halo_zyx,
                    name="--tile-halo-zyx",
                    positive=False,
                ),
                tile_batch_size=int(options.tile_batch_size),
            )

            runtime = load_spatial_runtime(
                checkpoint,
                device=options.spatial_device,
                config=config,
            )
            sink = CurationSpatialSink(
                paths,
                frame_count=frame_count,
            )

            print("", flush=True)
            print("=" * 104, flush=True)
            print(
                "DATASET CURATION — PRODUCTION STIR-NET SPATIAL",
                flush=True,
            )
            print("=" * 104, flush=True)
            print(f"volume     : {record.volume_id}", flush=True)
            print(f"split      : {record.split}", flush=True)
            print(f"frames     : {frame_count}", flush=True)
            print(f"source     : {paths.zarr}", flush=True)
            print(f"output     : {output}", flush=True)
            print(f"checkpoint : {checkpoint}", flush=True)
            print(
                f"prep timeout: {float(options.frame_prep_timeout):.0f}s",
                flush=True,
            )
            print("=" * 104, flush=True)

            rejection: SourceQualityRejected | None = None
            preparation_timeout: SpatialPreparationTimeout | None = None
            try:
                volume_result = run_parallel_spatial_volume(
                    paths.zarr,
                    runtime,
                    config=config,
                    sample_id=record.volume_id,
                    on_frame=sink.write_frame,
                    source_mask_validator=validate_source_mask,
                    preparation_timeout_seconds=float(
                        options.frame_prep_timeout
                    ),
                )
                sink.finish(
                    volume_result,
                    runtime=runtime,
                    config=config,
                    sample_id=record.volume_id,
                )
            except SourceQualityRejected as exc:
                rejection = exc
            except SpatialPreparationTimeout as exc:
                preparation_timeout = exc
            except SpatialPreparationWorkerError as exc:
                rejection = _quality_rejection_from_worker(
                    exc
                )
                if rejection is None:
                    raise
            finally:
                sink.close()

            # Release STIR-Net before either Trackastra or a skip cleanup.
            del runtime
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if preparation_timeout is not None:
                return _record_preparation_timeout(
                    record,
                    preparation_timeout,
                )

            if rejection is not None:
                return _record_quality_skip(record, rejection)

            spatial_was_run = True

        rebuild_trackastra = bool(
            force or spatial_was_run or options.rebuild_trackastra
        )
        run_trackastra(
            paths,
            model_name=options.trackastra_model,
            mode=options.trackastra_mode,
            device=options.trackastra_device,
            rebuild=rebuild_trackastra,
        )

        inference_id = uuid4().hex
        atomic_json(
            paths.inference_manifest,
            {
                "schema_version": 3,
                "kind": "biohub_curation_inference",
                "inference_id": inference_id,
                "backend": self.name,
                "volume_id": record.volume_id,
                "split": record.split,
                "frame_count": frame_count,
                "source_zarr": str(paths.zarr),
                "ground_truth_present": bool(record.has_ground_truth),
                "ground_truth_used_for_inference": False,
                "spatial_engine": (
                    "learned.stirnet.inference.run_parallel_spatial_volume"
                ),
                "parallel_preparation_workers": 1,
                "parallel_preparation_worker_kind": "spawn_process",
                "parallel_prefetch_depth": 1,
                "frame_preparation_timeout_seconds": float(
                    options.frame_prep_timeout
                ),
                "source_quality_gate": {
                    "enabled": True,
                    "location": "after_binary_mask_before_source_segmentation",
                    "reason_code": "pathological_connected_foreground",
                },
                "trackastra": {
                    "model": str(options.trackastra_model),
                    "mode": str(options.trackastra_mode),
                    "device": str(options.trackastra_device),
                },
            },
        )

        if not paths.inference_complete(frame_count=frame_count):
            raise ArtifactError(
                "Production inference returned but the canonical curation "
                f"cache is incomplete below {output}."
            )

        return InferenceOutcome(status="complete", output=output)
