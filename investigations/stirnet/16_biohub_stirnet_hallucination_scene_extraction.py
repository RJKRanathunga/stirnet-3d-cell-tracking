from __future__ import annotations

"""
Investigation 16 — curate STIR-Net hallucinated-cell scenes from saved BioHub inference.

This script reuses the existing advanced tracking-scene extraction GUI/model
infrastructure from:

    diagnostics/tracking_scene_extraction/napari_scene_extractor.py

but changes the scene-saving semantics for STIR-Net hard-negative curation:

* the selected STIR-Net instance is NOT intersected with the Stage-6 binary mask;
* surrounding raw / preprocessed / prediction context is preserved unmasked;
* the selected predicted instance is saved separately as the hard-negative mask;
* the full local STIR-Net spatial partition is also saved;
* scene provenance records the Investigation-12 output/checkpoint.

No STIR-Net inference is run here. The script consumes the already-persisted
Investigation-12 outputs.

Typical usage
-------------
From the repository root:

    python investigations/stirnet/16_biohub_stirnet_hallucination_scene_extraction.py

Workflow in Napari
------------------
1. Inspect Raw / Stage-6 / STIR-Net layers.
2. Turn on "STIR-Net instance centers" to see frame-local instance IDs.
3. Find an obvious hallucinated cell.
4. In the existing "Tracking Scene Extractor" dock:
     - enter the current frame and STIR-Net cell ID;
     - record the frame;
     - adjust Z/Y/X padding if desired;
     - save under the "stirnet_hallucinations" category.
5. The scene is written under:

       data/tracking_scenes/stirnet_hallucinations/001/
       data/tracking_scenes/stirnet_hallucinations/002/
       ...

The saved binary_mask.npy is the selected STIR-Net hallucinated-instance mask,
NOT the Stage-6 binary mask. The Stage-6 binary mask is preserved separately as
an unmasked contextual array.
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_NAME = "16_biohub_stirnet_hallucination_scene_extraction"
DEFAULT_SAMPLE = "44b6_0113de3b"
DEFAULT_CATEGORY = "stirnet_hallucinations"
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)
DEFAULT_PADDING_ZYX = (4, 24, 24)


# ======================================================================================
# Repository / path helpers
# ======================================================================================


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (
            (candidate / "learned").is_dir()
            and (candidate / "src").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    cwd = Path.cwd().resolve()
    if (
        (cwd / "learned").is_dir()
        and (cwd / "src").is_dir()
        and (cwd / "pyproject.toml").is_file()
    ):
        return cwd
    raise RuntimeError("Could not find the cell-tracking repository root.")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _step_from_dir_name(name: str) -> int:
    match = re.fullmatch(r"step(\d+)", name)
    return int(match.group(1)) if match else -1


def latest_inference_directory(sample_id: str) -> Path:
    base = (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / "12_biohub_full_volume_spatial_inference"
        / sample_id
    )
    if not base.is_dir():
        raise FileNotFoundError(
            f"Investigation-12 output directory does not exist: {base}"
        )

    candidates = [
        path
        for path in base.iterdir()
        if path.is_dir()
        and _step_from_dir_name(path.name) >= 0
        and (path / "manifest.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No complete stepXXXXXX Investigation-12 directory found under {base}"
        )
    return max(
        candidates,
        key=lambda path: (_step_from_dir_name(path.name), path.name),
    ).resolve()


def resolve_inference_directory(
    sample_id: str,
    override: str | None,
) -> Path:
    root = latest_inference_directory(sample_id) if override is None else resolve(override)
    required = ("manifest.json", "summary.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Inference directory is incomplete: {root}; missing={missing}"
        )
    return root.resolve()


def resolve_stage6_root(sample_id: str, override: str | None) -> Path:
    from src.io import PipelinePaths

    if override is None:
        root = PipelinePaths.discover(ROOT).processed_dataset(sample_id)
    else:
        supplied = resolve(override)
        if (supplied / "preprocessing").is_dir():
            root = supplied
        elif (supplied / sample_id / "preprocessing").is_dir():
            root = supplied / sample_id
        else:
            root = supplied

    required = ("preprocessing", "masking", "segmentation")
    missing = [name for name in required if not (root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Stage-6 directory is incomplete: {root}; missing={missing}"
        )
    return root.resolve()


def resolve_sample_zarr(sample_id: str, override: str | None) -> Path:
    from src.io import PipelinePaths

    if override is None:
        root = PipelinePaths.discover(ROOT).sample_zarr(sample_id)
    else:
        supplied = resolve(override)
        root = supplied.parent if supplied.name == "0" else supplied

    if not (root / "0").exists():
        raise FileNotFoundError(f"BioHub Zarr array not found: {root / '0'}")
    return root.resolve()


def completed_timepoints(inference_root: Path) -> list[int]:
    frames: list[int] = []
    for path in inference_root.glob("t[0-9][0-9][0-9]"):
        if path.is_dir() and (path / "_SUCCESS.json").is_file():
            frames.append(int(path.name[1:]))
    if not frames:
        raise FileNotFoundError(
            f"No completed Investigation-12 frames found under {inference_root}"
        )
    frames.sort()

    # The existing tracking-scene extractor indexes the first axis directly with
    # the real frame number. Keep that invariant rather than silently remapping T.
    expected = list(range(frames[-1] + 1))
    if frames != expected:
        raise RuntimeError(
            "Investigation-16 currently requires contiguous completed frames "
            f"starting at t000. Found {frames}."
        )
    return frames


# ======================================================================================
# Lazy 4-D data loading
# ======================================================================================


def _stack_npy(paths: Sequence[Path], *, name: str):
    paths = list(paths)
    if not paths:
        raise ValueError(f"No files supplied for {name}")

    arrays = [
        np.load(path, mmap_mode="r", allow_pickle=False)
        for path in paths
    ]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError(
            f"{name}: frame shapes differ: {[array.shape for array in arrays]}"
        )

    try:
        import dask.array as da

        chunks = (
            min(8, shape[0]),
            min(128, shape[1]),
            min(128, shape[2]),
        )
        result = da.stack(
            [
                da.from_array(array, chunks=chunks, asarray=False)
                for array in arrays
            ],
            axis=0,
        )
        backend = "dask-memmap"
    except ImportError:
        print(
            f"[data] Dask unavailable; eagerly stacking {name}.",
            flush=True,
        )
        result = np.stack([np.asarray(array) for array in arrays], axis=0)
        backend = "numpy"

    return result, backend


def stage6_stack(stage6_root: Path, frames: Sequence[int], series: str):
    return _stack_npy(
        [stage6_root / series / f"t{frame:03d}.npy" for frame in frames],
        name=f"Stage-6 {series}",
    )


def inference_stack(
    inference_root: Path,
    frames: Sequence[int],
    relative_path: str,
    *,
    name: str,
):
    return _stack_npy(
        [
            inference_root / f"t{frame:03d}" / relative_path
            for frame in frames
        ],
        name=name,
    )


def open_raw_stack(sample_zarr: Path, frames: Sequence[int]):
    from src.io import open_sample

    raw_source = open_sample(sample_zarr)
    if raw_source.ndim != 4:
        raise ValueError(
            f"Expected raw sample [T,Z,Y,X], got shape {raw_source.shape}"
        )
    if max(frames) >= int(raw_source.shape[0]):
        raise IndexError(
            f"Raw sample only has {raw_source.shape[0]} frames, requested {frames}"
        )

    try:
        import dask.array as da

        raw = da.from_zarr(str(sample_zarr / "0"))[list(frames)]
        backend = "dask-zarr"
    except ImportError:
        print("[data] Dask unavailable; eagerly loading raw frames.", flush=True)
        raw = np.stack(
            [np.asarray(raw_source[int(frame)]) for frame in frames],
            axis=0,
        )
        backend = "numpy-zarr"

    return raw, backend


def materialize(value: Any) -> np.ndarray:
    if hasattr(value, "compute"):
        value = value.compute()
    return np.asarray(value)


# ======================================================================================
# Build a STIR-Net cells table from the final partition itself
# ======================================================================================


def build_stirnet_cells_table(
    spatial_partition: Any,
    *,
    frames: Sequence[int],
    inference_root: Path,
) -> pd.DataFrame:
    """Build the frame/cell table expected by the existing scene extractor.

    We deliberately derive IDs, centroids and bounding boxes from the final
    `spatial_partition.npy` label image itself. This guarantees that the points
    shown in Napari and the IDs accepted by the extractor exactly match the
    persisted STIR-Net output.
    """
    from scipy import ndimage

    rows: list[dict[str, Any]] = []

    for frame in frames:
        labels = materialize(spatial_partition[int(frame)])
        if labels.ndim != 3:
            raise ValueError(
                f"Spatial partition t{frame:03d} must be 3-D, got {labels.shape}"
            )

        max_id = int(labels.max())
        if max_id <= 0:
            continue

        ids = np.arange(1, max_id + 1, dtype=np.int32)
        positive = labels > 0

        centers = ndimage.center_of_mass(
            positive,
            labels,
            ids.tolist(),
        )
        boxes = ndimage.find_objects(labels, max_label=max_id)
        counts = np.bincount(labels.reshape(-1), minlength=max_id + 1)

        quality_by_id: dict[int, float] = {}
        rag_path = inference_root / f"t{frame:03d}" / "rag" / "rag_state.npz"
        if rag_path.is_file():
            with np.load(rag_path, allow_pickle=False) as rag:
                if (
                    "provisional_local_ids" in rag
                    and "provisional_quality_logits" in rag
                ):
                    local_ids = np.asarray(
                        rag["provisional_local_ids"]
                    ).reshape(-1)
                    quality = np.asarray(
                        rag["provisional_quality_logits"],
                        dtype=np.float32,
                    ).reshape(-1)
                    if len(local_ids) == len(quality):
                        quality_by_id = {
                            int(cell_id): float(logit)
                            for cell_id, logit in zip(local_ids, quality)
                        }

        for cell_id in range(1, max_id + 1):
            bbox = boxes[cell_id - 1] if cell_id - 1 < len(boxes) else None
            if bbox is None or int(counts[cell_id]) <= 0:
                continue

            center = centers[cell_id - 1]
            if not all(math.isfinite(float(v)) for v in center):
                continue

            rows.append(
                {
                    "frame": int(frame),
                    "cell_id": int(cell_id),
                    "centroid_z": float(center[0]),
                    "centroid_y": float(center[1]),
                    "centroid_x": float(center[2]),
                    "z_min": int(bbox[0].start),
                    "y_min": int(bbox[1].start),
                    "x_min": int(bbox[2].start),
                    "z_max": int(bbox[0].stop),
                    "y_max": int(bbox[1].stop),
                    "x_max": int(bbox[2].stop),
                    "voxel_count": int(counts[cell_id]),
                    "quality_logit": quality_by_id.get(cell_id, float("nan")),
                }
            )

    if not rows:
        raise RuntimeError("No STIR-Net spatial instances were found.")

    cells = pd.DataFrame(rows)
    cells.sort_values(["frame", "cell_id"], inplace=True, ignore_index=True)
    return cells


# ======================================================================================
# Context-preserving specialization of the existing tracking-scene extractor
# ======================================================================================


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _safe_array_name(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip().lower())
    return token.strip("_-") or "volume"


from diagnostics.tracking_scene_extraction import (  # noqa: E402
    TrackingSceneCaptureModel,
    TrackingSceneExtractorWidget,
)


class StirNetHallucinationCaptureModel(TrackingSceneCaptureModel):
    """Reuse selection/crop/category logic but save full unmasked local context."""

    def save_scene(
        self,
        *,
        category: str,
        padding_zyx: Sequence[int],
    ) -> Path:
        if not self.selections:
            raise ValueError("No frame selections have been recorded.")

        category_path = self._category_path(category)
        safe_name = self._next_scene_name(category_path)
        final_path = category_path / safe_name

        crop = self.compute_crop_bounds(padding_zyx)
        frame_start = min(self.selections)
        frame_end = max(self.selections)
        frames = np.arange(frame_start, frame_end + 1, dtype=np.int32)
        crop_slices = crop.slices

        selected_mask_frames: list[np.ndarray] = []
        selected_label_frames: list[np.ndarray] = []
        full_partition_frames: list[np.ndarray] = []
        context_frames: dict[str, list[np.ndarray]] = {
            name: [] for name in self.image_volumes
        }

        label_dtype = np.dtype(self.instance_labels_volume.dtype)

        for frame_value in frames:
            frame = int(frame_value)
            selected_ids = self.selections.get(frame, ())

            labels_crop = materialize(
                self.instance_labels_volume[(frame, *crop_slices)]
            )
            full_partition_frames.append(
                labels_crop.astype(label_dtype, copy=False)
            )

            if selected_ids:
                selected_mask = np.isin(labels_crop, selected_ids)
                missing = [
                    cell_id
                    for cell_id in selected_ids
                    if not np.any(labels_crop == int(cell_id))
                ]
                if missing:
                    raise ValueError(
                        f"Selected STIR-Net ID(s) disappeared from t{frame:03d}: "
                        + ", ".join(str(value) for value in missing)
                    )
                selected_labels = np.where(
                    selected_mask,
                    labels_crop,
                    0,
                ).astype(label_dtype, copy=False)
            else:
                selected_mask = np.zeros(crop.shape_zyx, dtype=bool)
                selected_labels = np.zeros(crop.shape_zyx, dtype=label_dtype)

            selected_mask_frames.append(
                selected_mask.astype(bool, copy=False)
            )
            selected_label_frames.append(selected_labels)

            # Critical difference from the legacy tracking-scene saver:
            # preserve the COMPLETE local crop, not only voxels under selected_mask.
            for name, volume in self.image_volumes.items():
                context_crop = materialize(volume[(frame, *crop_slices)])
                context_frames[name].append(context_crop)

        temporary_path = (
            category_path / f".{safe_name}.tmp-{uuid.uuid4().hex}"
        )
        temporary_path.mkdir(parents=False, exist_ok=False)

        try:
            np.save(
                temporary_path / "frames.npy",
                frames,
                allow_pickle=False,
            )

            # Compatibility with the existing tracking-scene schema:
            # binary_mask.npy is the selected predicted hallucinated cell.
            np.save(
                temporary_path / "binary_mask.npy",
                np.stack(selected_mask_frames, axis=0),
                allow_pickle=False,
            )
            np.save(
                temporary_path / "instance_labels.npy",
                np.stack(selected_label_frames, axis=0),
                allow_pickle=False,
            )

            # Extra explicit names remove any ambiguity for future training code.
            np.save(
                temporary_path / "selected_hallucination_mask.npy",
                np.stack(selected_mask_frames, axis=0),
                allow_pickle=False,
            )
            np.save(
                temporary_path / "stirnet_spatial_partition.npy",
                np.stack(full_partition_frames, axis=0),
                allow_pickle=False,
            )

            context_files: dict[str, str] = {}
            used_names: set[str] = {
                "frames",
                "binary_mask",
                "instance_labels",
                "selected_hallucination_mask",
                "stirnet_spatial_partition",
            }

            for display_name, arrays in context_frames.items():
                base = _safe_array_name(display_name)
                candidate = base
                suffix = 2
                while candidate in used_names:
                    candidate = f"{base}_{suffix}"
                    suffix += 1
                used_names.add(candidate)

                filename = f"{candidate}.npy"
                np.save(
                    temporary_path / filename,
                    np.stack(arrays, axis=0),
                    allow_pickle=False,
                )
                context_files[display_name] = filename

            selected_rows: list[dict[str, Any]] = []
            for frame, cell_ids in sorted(self.selections.items()):
                for cell_id in cell_ids:
                    match = self.cells[
                        (self.cells[self.frame_column].astype(int) == int(frame))
                        & (
                            self.cells[self.cell_id_column].astype(int)
                            == int(cell_id)
                        )
                    ]
                    if not match.empty:
                        selected_rows.append(
                            {
                                key: _json_ready(value)
                                for key, value in match.iloc[0].to_dict().items()
                            }
                        )

            metadata = {
                "schema_version": 1,
                "annotation_schema": "stirnet_hard_negative_v1",
                "annotation_type": "confirmed_hallucinated_cell",
                "scene_name": safe_name,
                "category": category,
                "sample_id": self.sample_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "selected_cells": {
                    str(int(frame)): [int(value) for value in ids]
                    for frame, ids in sorted(self.selections.items())
                },
                "selected_cell_rows": selected_rows,
                "frame_start": int(frame_start),
                "frame_end": int(frame_end),
                "frames": [int(value) for value in frames],
                "missing_selected_frames": [
                    int(frame)
                    for frame in frames
                    if int(frame) not in self.selections
                ],
                "crop": {
                    "start_zyx": list(crop.start_zyx),
                    "stop_zyx": list(crop.stop_zyx),
                    "shape_zyx": list(crop.shape_zyx),
                    "padding_zyx": [int(value) for value in padding_zyx],
                    "voxel_size_zyx_um": list(self.voxel_size_zyx),
                },
                "files": {
                    "frames": "frames.npy",
                    "binary_mask": "binary_mask.npy",
                    "instance_labels": "instance_labels.npy",
                    "selected_hallucination_mask": (
                        "selected_hallucination_mask.npy"
                    ),
                    "stirnet_spatial_partition": (
                        "stirnet_spatial_partition.npy"
                    ),
                    # Kept for compatibility with the existing scene visualizer.
                    "masked_images": context_files,
                    # Semantically accurate name for new training/debug code.
                    "context_images": context_files,
                },
                "source": self.source_metadata,
            }

            with (temporary_path / "scene.json").open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(_json_ready(metadata), handle, indent=2)
                handle.write("\n")

            os.replace(temporary_path, final_path)

        except Exception:
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise

        return final_path


# ======================================================================================
# Napari viewer + reused tracking-scene extractor widget
# ======================================================================================


def parse_zyx(text: str, *, name: str) -> tuple[int, int, int]:
    values = tuple(int(token.strip()) for token in text.split(","))
    if len(values) != 3 or any(value < 0 for value in values):
        raise ValueError(f"{name} must contain three non-negative Z,Y,X integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Curate confirmed STIR-Net hallucinated-cell scenes from persisted "
            "Investigation-12 BioHub full-volume inference."
        )
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    parser.add_argument(
        "--inference-dir",
        default=None,
        help=(
            "Investigation-12 stepXXXXXX directory. Default selects the latest "
            "completed step for this sample."
        ),
    )
    parser.add_argument(
        "--stage6-root",
        default=None,
        help="Optional Stage-6 sample directory.",
    )
    parser.add_argument(
        "--sample-zarr",
        default=None,
        help="Optional original BioHub sample .zarr path.",
    )
    parser.add_argument(
        "--save-root",
        default=None,
        help=(
            "Tracking-scene root. Default uses PipelinePaths.tracking_scenes."
        ),
    )
    parser.add_argument(
        "--category",
        default=DEFAULT_CATEGORY,
        help=(
            "Scene category directory created below save-root. "
            f"Default: {DEFAULT_CATEGORY}"
        ),
    )
    parser.add_argument(
        "--padding-zyx",
        default=",".join(str(value) for value in DEFAULT_PADDING_ZYX),
        help="Default scene padding Z,Y,X in voxels.",
    )
    parser.add_argument(
        "--center-point-size",
        type=float,
        default=4.0,
    )
    args = parser.parse_args()

    from src.io import PipelinePaths

    paths = PipelinePaths.discover(ROOT)
    inference_root = resolve_inference_directory(
        args.sample_id,
        args.inference_dir,
    )
    stage6_root = resolve_stage6_root(args.sample_id, args.stage6_root)
    sample_zarr = resolve_sample_zarr(args.sample_id, args.sample_zarr)
    frames = completed_timepoints(inference_root)

    manifest = json.loads(
        (inference_root / "manifest.json").read_text(encoding="utf-8")
    )
    spacing = tuple(
        float(value)
        for value in manifest.get(
            "spacing_zyx_um",
            DEFAULT_SPACING_ZYX_UM,
        )
    )
    scale_tzyx = (1.0, *spacing)
    padding_zyx = parse_zyx(args.padding_zyx, name="--padding-zyx")

    save_root = (
        resolve(args.save_root)
        if args.save_root is not None
        else Path(paths.tracking_scenes).resolve()
    )
    save_root.mkdir(parents=True, exist_ok=True)
    category_dir = save_root / args.category
    category_dir.mkdir(parents=True, exist_ok=True)

    print("[data] loading persisted 4-D stacks ...", flush=True)

    raw, raw_backend = open_raw_stack(sample_zarr, frames)
    preprocessed, saved_backend = stage6_stack(
        stage6_root,
        frames,
        "preprocessing",
    )
    stage6_mask, _ = stage6_stack(stage6_root, frames, "masking")
    stage6_segmentation, _ = stage6_stack(
        stage6_root,
        frames,
        "segmentation",
    )

    spatial_partition, _ = inference_stack(
        inference_root,
        frames,
        "partition/spatial_partition.npy",
        name="STIR-Net spatial partition",
    )
    watershed, _ = inference_stack(
        inference_root,
        frames,
        "partition/watershed_supervoxels.npy",
        name="STIR-Net watershed supervoxels",
    )
    foreground, _ = inference_stack(
        inference_root,
        frames,
        "geometry/foreground_probability.npy",
        name="STIR-Net foreground probability",
    )
    surface, _ = inference_stack(
        inference_root,
        frames,
        "geometry/surface_probability.npy",
        name="STIR-Net surface probability",
    )
    separator, _ = inference_stack(
        inference_root,
        frames,
        "geometry/separator_probability.npy",
        name="STIR-Net separator probability",
    )
    seed, _ = inference_stack(
        inference_root,
        frames,
        "geometry/seed_probability.npy",
        name="STIR-Net seed probability",
    )
    sdf, _ = inference_stack(
        inference_root,
        frames,
        "geometry/sdf.npy",
        name="STIR-Net SDF",
    )

    expected_shape = tuple(spatial_partition.shape)
    for name, volume in {
        "raw": raw,
        "preprocessed": preprocessed,
        "stage6_mask": stage6_mask,
        "stage6_segmentation": stage6_segmentation,
        "watershed": watershed,
        "foreground": foreground,
        "surface": surface,
        "separator": separator,
        "seed": seed,
        "sdf": sdf,
    }.items():
        if tuple(volume.shape) != expected_shape:
            raise ValueError(
                f"4-D volume '{name}' has shape {tuple(volume.shape)}, "
                f"expected {expected_shape}"
            )

    print("[data] deriving exact STIR-Net IDs / centers / boxes ...", flush=True)
    cells = build_stirnet_cells_table(
        spatial_partition,
        frames=frames,
        inference_root=inference_root,
    )

    checkpoint_step = int(
        manifest.get("checkpoint_step", _step_from_dir_name(inference_root.name))
    )

    print("=" * 118)
    print("STIR-Net Investigation 16 — hallucinated-cell scene curation")
    print("=" * 118)
    print("sample                    :", args.sample_id)
    print("raw sample                :", sample_zarr)
    print("Stage-6                   :", stage6_root)
    print("Investigation-12          :", inference_root)
    print("checkpoint step           :", checkpoint_step)
    print("frames                    :", frames)
    print("STIR-Net instances        :", len(cells))
    print("spacing ZYX um            :", spacing)
    print("raw backend               :", raw_backend)
    print("saved-result backend      :", saved_backend)
    print("save root                 :", save_root)
    print("category                  :", args.category)
    print("default padding ZYX       :", padding_zyx)
    print("=" * 118)

    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is required for Investigation 16."
        ) from exc

    viewer = napari.Viewer(
        ndisplay=3,
        title=(
            f"STIR-Net hallucination curation | {args.sample_id} | "
            f"step {checkpoint_step}"
        ),
    )

    # Raw is the primary evidence for deciding whether a predicted instance is
    # truly a hallucination.
    viewer.add_image(
        raw,
        name="Raw BioHub volume",
        scale=scale_tzyx,
        rendering="mip",
        colormap="gray",
        visible=True,
    )
    viewer.add_image(
        preprocessed,
        name="Stage-6 preprocessing",
        scale=scale_tzyx,
        rendering="mip",
        colormap="gray",
        contrast_limits=(0.0, 1.0),
        visible=False,
    )
    viewer.add_labels(
        stage6_mask,
        name="Stage-6 binary mask",
        scale=scale_tzyx,
        opacity=0.35,
        visible=False,
    )
    viewer.add_labels(
        stage6_segmentation,
        name="Stage-6 segmentation",
        scale=scale_tzyx,
        opacity=0.45,
        visible=False,
    )
    viewer.add_labels(
        watershed,
        name="STIR-Net watershed supervoxels",
        scale=scale_tzyx,
        opacity=0.45,
        visible=False,
    )
    viewer.add_labels(
        spatial_partition,
        name="STIR-Net spatial partition",
        scale=scale_tzyx,
        opacity=0.55,
        visible=True,
    )
    viewer.add_image(
        foreground,
        name="STIR-Net foreground probability",
        scale=scale_tzyx,
        colormap="magenta",
        contrast_limits=(0.0, 1.0),
        blending="additive",
        visible=True,
    )
    viewer.add_image(
        surface,
        name="STIR-Net surface probability",
        scale=scale_tzyx,
        colormap="cyan",
        contrast_limits=(0.0, 1.0),
        blending="additive",
        visible=False,
    )
    viewer.add_image(
        separator,
        name="STIR-Net separator probability",
        scale=scale_tzyx,
        colormap="yellow",
        contrast_limits=(0.0, 1.0),
        blending="additive",
        visible=False,
    )
    viewer.add_image(
        seed,
        name="STIR-Net seed probability",
        scale=scale_tzyx,
        colormap="green",
        contrast_limits=(0.0, 1.0),
        blending="additive",
        visible=False,
    )
    viewer.add_image(
        sdf,
        name="STIR-Net SDF",
        scale=scale_tzyx,
        colormap="turbo",
        blending="additive",
        visible=False,
    )

    points = cells[
        ["frame", "centroid_z", "centroid_y", "centroid_x"]
    ].to_numpy(dtype=np.float32)

    quality = cells["quality_logit"].to_numpy(dtype=np.float32)
    quality_text = np.where(
        np.isfinite(quality),
        np.char.mod("%.2f", quality),
        "n/a",
    )

    centers_layer = viewer.add_points(
        points,
        name="STIR-Net instance centers",
        scale=scale_tzyx,
        size=float(args.center_point_size),
        face_color="red",
        properties={
            "cell_id": cells["cell_id"].to_numpy(dtype=np.int32),
            "quality_logit": quality,
            "quality_text": quality_text,
        },
        text={
            "string": "{cell_id}",
            "size": 8,
            "color": "white",
            "anchor": "center",
        },
    )
    centers_layer.visible = True

    # Reuse the original advanced selection/crop/category/widget logic, while
    # swapping only the save semantics through our model subclass.
    context_volumes = {
        "raw": raw,
        "stage6_preprocessed": preprocessed,
        "stage6_binary_mask": stage6_mask,
        "stage6_segmentation": stage6_segmentation,
        "stirnet_foreground_probability": foreground,
        "stirnet_surface_probability": surface,
        "stirnet_separator_probability": separator,
        "stirnet_seed_probability": seed,
        "stirnet_sdf": sdf,
        "stirnet_watershed_supervoxels": watershed,
    }

    capture_model = StirNetHallucinationCaptureModel(
        cells=cells,
        instance_labels_volume=spatial_partition,
        # Deliberately None: never clip the hallucinated prediction to Stage 6.
        binary_mask_volume=None,
        image_volumes=context_volumes,
        sample_id=args.sample_id,
        save_root=save_root,
        voxel_size_zyx=spacing,
        cell_id_column="cell_id",
        frame_column="frame",
        source_metadata={
            "annotation_type": "confirmed_hallucinated_cell",
            "script": SCRIPT_NAME,
            "inference_dir": inference_root,
            "checkpoint_step": checkpoint_step,
            "checkpoint_path": manifest.get("checkpoint_path"),
            "stage6_root": stage6_root,
            "source_zarr_array": sample_zarr / "0",
            "spacing_zyx_um": spacing,
            "important_semantics": {
                "binary_mask.npy": (
                    "selected STIR-Net predicted hallucination mask; "
                    "NOT the Stage-6 binary mask"
                ),
                "context_images": (
                    "full unmasked local crops retained for hard-negative training"
                ),
            },
        },
    )

    extractor_widget = TrackingSceneExtractorWidget(
        viewer=viewer,
        model=capture_model,
        default_padding_zyx=padding_zyx,
    )
    viewer.window.add_dock_widget(
        extractor_widget,
        name="Tracking Scene Extractor",
        area="right",
    )

    viewer.dims.set_current_step(0, 0)

    print()
    print(
        "CURATION WORKFLOW: find an obvious fake cell in Raw BioHub volume, "
        "read its red STIR-Net ID, enter frame + ID in the Tracking Scene "
        "Extractor, record it, then save."
    )
    print(
        f"Scenes will be stored under: {category_dir}"
    )
    print(
        "Saved context is UNMASKED. selected_hallucination_mask.npy is the "
        "negative ROI corresponding to the selected STIR-Net prediction."
    )

    napari.run()


if __name__ == "__main__":
    main()
