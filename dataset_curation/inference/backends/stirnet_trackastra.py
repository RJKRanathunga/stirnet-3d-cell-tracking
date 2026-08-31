from __future__ import annotations

import argparse
import gc
from pathlib import Path
import shutil
from typing import Sequence

import torch

from dataset_curation._repo import repo_root
from dataset_curation.catalog import (
    VolumeRecord,
)
from dataset_curation.errors import (
    ArtifactError,
)
from dataset_curation.io.atomic import (
    atomic_json,
)
from dataset_curation.inference.spatial_sink import (
    CurationSpatialSink,
)
from dataset_curation.inference.trackastra import (
    run_trackastra,
)
from learned.stirnet.inference import (
    SpatialInferenceConfig,
    load_spatial_runtime,
    run_parallel_spatial_volume,
)


DEFAULT_TILE_SHAPE_ZYX = (
    32,
    128,
    128,
)
DEFAULT_TILE_OVERLAP_ZYX = (
    8,
    32,
    32,
)
DEFAULT_TILE_HALO_ZYX = (
    4,
    16,
    16,
)
DEFAULT_TILE_BATCH_SIZE = 1

DEFAULT_TRACKASTRA_MODEL = "ctc"
DEFAULT_TRACKASTRA_MODE = "greedy"
DEFAULT_TRACKASTRA_DEVICE = "cuda"


def _parse_triplet(
    value: str,
    *,
    name: str,
    positive: bool,
) -> tuple[int, int, int]:
    rows = tuple(
        int(token.strip())
        for token in str(value).split(",")
    )
    if len(rows) != 3:
        raise ValueError(
            f"{name} must contain exactly "
            "three comma-separated integers"
        )
    if positive:
        if any(v <= 0 for v in rows):
            raise ValueError(
                f"{name} values must be positive"
            )
    elif any(v < 0 for v in rows):
        raise ValueError(
            f"{name} values must be non-negative"
        )
    return rows


def _checkpoint_step(
    path: Path,
) -> int:
    name = path.name
    prefix = "checkpoint_step_"
    suffix = ".pt"
    if (
        name.startswith(prefix)
        and name.endswith(suffix)
    ):
        token = name[
            len(prefix):
            -len(suffix)
        ]
        if token.isdigit():
            return int(token)
    return -1


def _latest_checkpoint(
    directory: Path,
) -> Path:
    if not directory.is_dir():
        raise NotADirectoryError(
            directory
        )

    best = directory / "best_checkpoint.pt"
    if best.is_file():
        return best.resolve()

    candidates = [
        path
        for path in directory.glob(
            "checkpoint_step_*.pt"
        )
        if path.is_file()
        and _checkpoint_step(path) >= 0
    ]
    if not candidates:
        candidates = [
            path
            for path in directory.rglob(
                "checkpoint_step_*.pt"
            )
            if path.is_file()
            and _checkpoint_step(path) >= 0
        ]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found below {directory}"
        )

    return max(
        candidates,
        key=lambda path: (
            _checkpoint_step(path),
            str(path),
        ),
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

    for name in (
        "best_checkpoint.pt",
        "checkpoint_step_000600.pt",
    ):
        candidate = recovery / name
        if candidate.is_file():
            return candidate.resolve()

    if recovery.is_dir():
        return _latest_checkpoint(
            recovery
        )

    raise FileNotFoundError(
        "Could not auto-discover the current "
        "STIR-Net spatial checkpoint. "
        "Pass --checkpoint <path> after `--`."
    )


def _resolve_checkpoint(
    checkpoint: str | None,
    checkpoint_dir: str | None,
) -> Path:
    if (
        checkpoint is not None
        and checkpoint_dir is not None
    ):
        raise ValueError(
            "--checkpoint and --checkpoint-dir "
            "are mutually exclusive"
        )

    if checkpoint_dir is not None:
        path = Path(
            checkpoint_dir
        ).expanduser()
        if not path.is_absolute():
            path = (
                repo_root()
                / path
            )
        return _latest_checkpoint(
            path.resolve()
        )

    if checkpoint is not None:
        path = Path(
            checkpoint
        ).expanduser()
        if not path.is_absolute():
            path = (
                repo_root()
                / path
            )
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                path
            )
        return path

    return resolve_default_checkpoint()


def _parse_extra(
    values: Sequence[str],
):
    parser = argparse.ArgumentParser(
        add_help=False,
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
    )
    parser.add_argument(
        "--spatial-device",
        default="cuda",
    )
    parser.add_argument(
        "--tile-shape-zyx",
        default="32,128,128",
    )
    parser.add_argument(
        "--tile-overlap-zyx",
        default="8,32,32",
    )
    parser.add_argument(
        "--tile-halo-zyx",
        default="4,16,16",
    )
    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--trackastra-model",
        default=DEFAULT_TRACKASTRA_MODEL,
    )
    parser.add_argument(
        "--trackastra-mode",
        default=DEFAULT_TRACKASTRA_MODE,
    )
    parser.add_argument(
        "--trackastra-device",
        default=DEFAULT_TRACKASTRA_DEVICE,
    )
    parser.add_argument(
        "--rebuild-trackastra",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite-spatial",
        action="store_true",
    )

    parsed, unknown = (
        parser.parse_known_args(
            list(values)
        )
    )
    if unknown:
        raise ValueError(
            "Unsupported dataset-curation "
            "inference option(s): "
            + " ".join(unknown)
        )
    return parsed


class StirNetTrackastraBackend:
    """
    Production curation backend.

    Spatial inference is owned by learned.stirnet.inference.
    Trackastra remains a curation-level downstream adapter.
    No runtime code is loaded from investigations/.
    """

    name = "production_stirnet_trackastra"

    def run_volume(
        self,
        record: VolumeRecord,
        *,
        run_id: str = "current",
        force: bool = False,
        extra_args: Sequence[str] = (),
    ) -> Path:
        paths = record.paths
        paths.ensure_output_roots()

        options = _parse_extra(
            extra_args
        )
        force_spatial = bool(
            force
            or options.overwrite_spatial
        )

        if record.frame_count is None:
            raise ArtifactError(
                f"Could not determine frame count "
                f"for {record.volume_id}."
            )

        frame_count = int(
            record.frame_count
        )
        if (
            not force_spatial
            and paths.inference_complete(
                run_id,
                frame_count=frame_count,
            )
        ):
            print(
                f"[skip] {record.split}/"
                f"{record.volume_id}: "
                f"run {run_id!r} is complete.",
                flush=True,
            )
            return paths.inference_run(
                run_id
            )

        output = paths.inference_run(
            run_id
        )
        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        spatial_was_run = False

        if (
            force_spatial
            or not paths.spatial_complete(
                run_id,
                frame_count=frame_count,
            )
        ):
            # Any newly generated spatial labels invalidate a previous
            # Trackastra graph.
            trackastra_root = (
                paths.trackastra_root(
                    run_id
                )
            )
            if trackastra_root.exists():
                shutil.rmtree(
                    trackastra_root
                )

            checkpoint = _resolve_checkpoint(
                options.checkpoint,
                options.checkpoint_dir,
            )

            config = SpatialInferenceConfig(
                spacing_zyx_um=(
                    1.625,
                    0.40625,
                    0.40625,
                ),
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
                tile_batch_size=int(
                    options.tile_batch_size
                ),
            )

            runtime = load_spatial_runtime(
                checkpoint,
                device=options.spatial_device,
                config=config,
            )
            sink = CurationSpatialSink(
                paths,
                run_id=run_id,
                frame_count=frame_count,
            )

            print("", flush=True)
            print("=" * 104, flush=True)
            print(
                "DATASET CURATION — "
                "PRODUCTION STIR-NET SPATIAL",
                flush=True,
            )
            print("=" * 104, flush=True)
            print(
                f"volume     : {record.volume_id}",
                flush=True,
            )
            print(
                f"split      : {record.split}",
                flush=True,
            )
            print(
                f"frames     : {frame_count}",
                flush=True,
            )
            print(
                f"source     : {paths.zarr}",
                flush=True,
            )
            print(
                f"output     : {output}",
                flush=True,
            )
            print(
                f"checkpoint : {checkpoint}",
                flush=True,
            )
            print("=" * 104, flush=True)

            try:
                volume_result = (
                    run_parallel_spatial_volume(
                        paths.zarr,
                        runtime,
                        config=config,
                        sample_id=record.volume_id,
                        on_frame=sink.write_frame,
                    )
                )
                sink.finish(
                    volume_result,
                    runtime=runtime,
                    config=config,
                    sample_id=record.volume_id,
                )
            finally:
                sink.close()

            spatial_was_run = True

            # Release STIR-Net before Trackastra claims the GPU.
            del runtime
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rebuild_trackastra = bool(
            force
            or spatial_was_run
            or options.rebuild_trackastra
        )
        run_trackastra(
            paths,
            run_id=run_id,
            model_name=options.trackastra_model,
            mode=options.trackastra_mode,
            device=options.trackastra_device,
            rebuild=rebuild_trackastra,
        )

        if not paths.inference_complete(
            run_id,
            frame_count=frame_count,
        ):
            raise ArtifactError(
                "Production inference returned but "
                "the curation cache is incomplete "
                f"below {output}."
            )

        atomic_json(
            paths.inference_manifest(
                run_id
            ),
            {
                "schema_version": 2,
                "kind": "biohub_curation_inference",
                "backend": self.name,
                "volume_id": record.volume_id,
                "split": record.split,
                "frame_count": frame_count,
                "run_id": str(run_id),
                "source_zarr": str(
                    paths.zarr
                ),
                "ground_truth_present": bool(
                    record.has_ground_truth
                ),
                "ground_truth_used_for_inference": False,
                "spatial_engine": (
                    "learned.stirnet.inference."
                    "run_parallel_spatial_volume"
                ),
                "parallel_preparation_workers": 1,
                "parallel_prefetch_depth": 1,
                "trackastra": {
                    "model": str(
                        options.trackastra_model
                    ),
                    "mode": str(
                        options.trackastra_mode
                    ),
                    "device": str(
                        options.trackastra_device
                    ),
                },
            },
        )

        return output
