from __future__ import annotations

"""
Investigation 13 — Napari viewer for Investigation-12 full-volume BioHub results.

This script does NOT run STIR-Net again.  It opens the persistent artifacts
written by:

    investigations/stirnet/data/12_biohub_full_volume_spatial_inference.py

The default view is a 4-D T,Z,Y,X stack over all completed BioHub timepoints,
so the Napari time slider can be used to inspect the complete sequence.

Core layers
-----------
* Raw BioHub volume from the original sample Zarr
* Stage-6 preprocessed intensity
* Stage-6 binary mask
* Stage-6 source segmentation
* STIR-Net watershed supervoxels
* STIR-Net spatial partition

Dense diagnostic layers
-----------------------
* foreground probability
* surface probability
* separator probability
* seed probability
* SDF

Optional vector layers
----------------------
Flow and centroid-offset vectors can be loaded for one chosen timepoint using
`--vectors`.  They are sampled sparsely so Napari remains responsive.

Typical command
---------------
From the repository root:

    python investigations/stirnet/data/13_biohub_full_volume_spatial_results_viewer.py

Show sparse vectors for timepoint 0:

    python investigations/stirnet/data/13_biohub_full_volume_spatial_results_viewer.py ^
        --vectors --vector-timepoint 0

Open only selected timepoints:

    python investigations/stirnet/data/13_biohub_full_volume_spatial_results_viewer.py ^
        --timepoints 0,5,10,15,19

Memory behavior
---------------
When Dask is available, .npy files are memory-mapped and stacked lazily.  Napari
then reads only the chunks needed for display.  If Dask is unavailable, the
script falls back to NumPy stacking.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_NAME = "13_biohub_full_volume_spatial_results_viewer"
DEFAULT_SAMPLE = "44b6_0113de3b"
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)
DEFAULT_VECTOR_STRIDE_ZYX = (4, 12, 12)


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
    raise RuntimeError("Could not resolve the cell-tracking repository root.")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def parse_step_directory_name(name: str) -> int:
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
            "Investigation-12 output directory does not exist: "
            f"{base}"
        )
    candidates = [
        path
        for path in base.iterdir()
        if path.is_dir() and parse_step_directory_name(path.name) >= 0
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No stepXXXXXX Investigation-12 directory found below {base}"
        )
    return max(
        candidates,
        key=lambda path: (parse_step_directory_name(path.name), path.name),
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
    """Resolve the original BioHub sample Zarr used to create Stage 6."""
    from src.io import PipelinePaths

    if override is None:
        sample_path = PipelinePaths.discover(ROOT).sample_zarr(sample_id)
    else:
        supplied = resolve(override)
        sample_path = supplied.parent if supplied.name == "0" else supplied

    array_path = sample_path / "0"
    if not array_path.exists():
        raise FileNotFoundError(
            f"Original BioHub Zarr array does not exist: {array_path}"
        )
    return sample_path.resolve()


def load_raw_time_series(sample_zarr: Path, frames: list[int]):
    """Load selected original raw frames lazily when Dask is available."""
    from src.io import open_sample

    sample = open_sample(sample_zarr)
    if sample.ndim != 4:
        raise ValueError(f"Expected raw BioHub sample [T,Z,Y,X], got {sample.shape}")

    missing = [frame for frame in frames if not 0 <= frame < int(sample.shape[0])]
    if missing:
        raise IndexError(
            f"Raw Zarr does not contain requested frames {missing}; shape={sample.shape}"
        )

    try:
        import dask.array as da
        raw_all = da.from_zarr(str(sample_zarr / "0"))
        raw = raw_all[frames]
        backend = "dask-zarr"
    except ImportError:
        print(
            "[viewer] Dask unavailable; eagerly loading selected raw Zarr frames.",
            flush=True,
        )
        raw = np.stack([np.asarray(sample[frame]) for frame in frames], axis=0)
        backend = "numpy-zarr"

    return raw, backend


def completed_timepoints(inference_root: Path) -> list[int]:
    frames = []
    for path in inference_root.glob("t[0-9][0-9][0-9]"):
        if path.is_dir() and (path / "_SUCCESS.json").is_file():
            frames.append(int(path.name[1:]))
    if not frames:
        raise FileNotFoundError(
            f"No completed tXXX/_SUCCESS.json frames found below {inference_root}"
        )
    return sorted(frames)


def parse_timepoints(text: str, available: list[int]) -> list[int]:
    token = text.strip().lower()
    if token in {"all", "*"}:
        return list(available)

    selected: set[int] = set()
    for item in token.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"Invalid timepoint range: {item}")
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(item))

    missing = sorted(selected.difference(available))
    if missing:
        raise ValueError(
            f"Requested frames are not completed: {missing}; available={available}"
        )
    if not selected:
        raise ValueError("No timepoints selected")
    return sorted(selected)


def parse_zyx(text: str, *, name: str) -> tuple[int, int, int]:
    values = tuple(int(token.strip()) for token in text.split(","))
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain three positive Z,Y,X integers")
    return values


# ======================================================================================
# Lazy stack loading
# ======================================================================================


def _load_memmap(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def stack_npy(paths: Iterable[Path], *, name: str):
    paths = list(paths)
    arrays = [_load_memmap(path) for path in paths]
    if not arrays:
        raise ValueError(f"No arrays supplied for {name}")

    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError(
            f"{name}: frame shapes differ: {[array.shape for array in arrays]}"
        )

    try:
        import dask.array as da

        # One full Z plane block per chunk is a good compromise for interactive
        # 3-D viewing while avoiding eager loading of every timepoint.
        chunks = (
            min(8, shape[0]),
            min(128, shape[1]),
            min(128, shape[2]),
        )
        rows = [da.from_array(array, chunks=chunks, asarray=False) for array in arrays]
        result = da.stack(rows, axis=0)
        backend = "dask-memmap"
    except ImportError:
        print(
            f"[viewer] Dask unavailable; eagerly stacking {name} with NumPy.",
            flush=True,
        )
        result = np.stack([np.asarray(array) for array in arrays], axis=0)
        backend = "numpy"

    return result, backend


def stage6_paths(stage6_root: Path, frames: list[int], series: str) -> list[Path]:
    return [
        stage6_root / series / f"t{frame:03d}.npy"
        for frame in frames
    ]


def inference_paths(
    inference_root: Path,
    frames: list[int],
    relative: str,
) -> list[Path]:
    return [
        inference_root / f"t{frame:03d}" / relative
        for frame in frames
    ]


# ======================================================================================
# Vector visualization
# ======================================================================================


def build_vector_layer(
    vector_path: Path,
    foreground_path: Path,
    *,
    displayed_time_index: int,
    stride_zyx: tuple[int, int, int],
    minimum_magnitude: float,
    vector_scale: float,
) -> np.ndarray:
    """Return Napari 4-D vectors [N, 2, (T,Z,Y,X)] for one timepoint."""
    field = np.asarray(_load_memmap(vector_path), dtype=np.float32)
    foreground = np.asarray(_load_memmap(foreground_path), dtype=np.float32)

    if field.ndim != 4 or field.shape[0] != 3:
        raise ValueError(
            f"Expected vector field [3,Z,Y,X], got {field.shape}: {vector_path}"
        )
    if foreground.shape != field.shape[1:]:
        raise ValueError(
            f"Vector/foreground shapes disagree: {field.shape} vs {foreground.shape}"
        )

    z_idx = np.arange(0, field.shape[1], stride_zyx[0], dtype=np.int32)
    y_idx = np.arange(0, field.shape[2], stride_zyx[1], dtype=np.int32)
    x_idx = np.arange(0, field.shape[3], stride_zyx[2], dtype=np.int32)
    zz, yy, xx = np.meshgrid(z_idx, y_idx, x_idx, indexing="ij")

    z = zz.reshape(-1)
    y = yy.reshape(-1)
    x = xx.reshape(-1)

    vectors = field[:, z, y, x].T
    magnitude = np.linalg.norm(vectors, axis=1)
    keep = (
        (foreground[z, y, x] >= 0.5)
        & np.isfinite(magnitude)
        & (magnitude >= float(minimum_magnitude))
    )

    z = z[keep]
    y = y[keep]
    x = x[keep]
    vectors = vectors[keep] * float(vector_scale)

    starts = np.stack(
        [
            np.full(len(z), displayed_time_index, dtype=np.float32),
            z.astype(np.float32),
            y.astype(np.float32),
            x.astype(np.float32),
        ],
        axis=1,
    )
    directions = np.concatenate(
        [
            np.zeros((len(vectors), 1), dtype=np.float32),
            vectors.astype(np.float32, copy=False),
        ],
        axis=1,
    )
    return np.stack([starts, directions], axis=1)


# ======================================================================================
# Napari
# ======================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open Investigation-12 full-volume BioHub inference in Napari."
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    parser.add_argument(
        "--inference-dir",
        default=None,
        help=(
            "Investigation-12 stepXXXXXX directory. Default selects the highest "
            "available step for the sample."
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
        help=(
            "Optional original BioHub sample .zarr path. "
            "Default comes from src.io.PipelinePaths."
        ),
    )
    parser.add_argument(
        "--timepoints",
        default="all",
        help='Completed timepoints: "all", list ("0,5,10"), or range ("0-19").',
    )
    parser.add_argument(
        "--no-geometry",
        action="store_true",
        help="Open only image/mask/source/watershed/partition layers.",
    )
    parser.add_argument(
        "--vectors",
        action="store_true",
        help="Add sparse flow and centroid-offset vectors for one timepoint.",
    )
    parser.add_argument(
        "--vector-timepoint",
        type=int,
        default=0,
        help="Original BioHub timepoint whose vectors should be shown.",
    )
    parser.add_argument(
        "--vector-stride-zyx",
        default=",".join(str(v) for v in DEFAULT_VECTOR_STRIDE_ZYX),
        help="Sparse vector sampling stride Z,Y,X.",
    )
    parser.add_argument(
        "--vector-min-magnitude",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--flow-scale",
        type=float,
        default=3.0,
        help="Purely visual multiplier applied to flow vectors.",
    )
    parser.add_argument(
        "--centroid-offset-scale",
        type=float,
        default=1.0,
        help="Purely visual multiplier applied to centroid-offset vectors.",
    )
    args = parser.parse_args()

    inference_root = resolve_inference_directory(
        args.sample_id,
        args.inference_dir,
    )
    stage6_root = resolve_stage6_root(args.sample_id, args.stage6_root)
    sample_zarr = resolve_sample_zarr(args.sample_id, args.sample_zarr)
    available = completed_timepoints(inference_root)
    frames = parse_timepoints(args.timepoints, available)

    manifest = json.loads(
        (inference_root / "manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (inference_root / "summary.json").read_text(encoding="utf-8")
    )
    spacing = tuple(
        float(value)
        for value in manifest.get(
            "spacing_zyx_um",
            DEFAULT_SPACING_ZYX_UM,
        )
    )
    scale_4d = (1.0, *spacing)

    print("=" * 110)
    print("STIR-Net Investigation 13 — BioHub full-volume result viewer")
    print("=" * 110)
    print("sample             :", args.sample_id)
    print("Raw sample Zarr    :", sample_zarr)
    print("Stage-6            :", stage6_root)
    print("Inference          :", inference_root)
    print("checkpoint step    :", manifest.get("checkpoint_step"))
    print("frames             :", frames)
    print("spacing ZYX um     :", spacing)
    print("completed          :", summary.get("completed_count"))
    print("=" * 110)

    raw, raw_backend = load_raw_time_series(sample_zarr, frames)
    preprocessed, backend = stack_npy(
        stage6_paths(stage6_root, frames, "preprocessing"),
        name="Stage-6 preprocessing",
    )
    source_mask, _ = stack_npy(
        stage6_paths(stage6_root, frames, "masking"),
        name="Stage-6 mask",
    )
    source_labels, _ = stack_npy(
        stage6_paths(stage6_root, frames, "segmentation"),
        name="Stage-6 source segmentation",
    )
    watershed, _ = stack_npy(
        inference_paths(
            inference_root,
            frames,
            "partition/watershed_supervoxels.npy",
        ),
        name="watershed supervoxels",
    )
    spatial_partition, _ = stack_npy(
        inference_paths(
            inference_root,
            frames,
            "partition/spatial_partition.npy",
        ),
        name="STIR-Net spatial partition",
    )

    print(
        f"[viewer] array backend: raw={raw_backend}, saved-results={backend}",
        flush=True,
    )

    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is not installed in this environment. Install the project's "
            "visualization dependencies and rerun Investigation 13."
        ) from exc

    viewer = napari.Viewer(
        title=(
            f"STIR-Net BioHub full volume | {args.sample_id} | "
            f"step {manifest.get('checkpoint_step', '?')}"
        )
    )

    # Core comparison layers.
    viewer.add_image(
        raw,
        name="Raw BioHub volume",
        scale=scale_4d,
        colormap="gray",
        visible=False,
    )
    viewer.add_image(
        preprocessed,
        name="Stage-6 preprocessing",
        scale=scale_4d,
        colormap="gray",
        contrast_limits=(0.0, 1.0),
        visible=True,
    )
    viewer.add_labels(
        source_mask,
        name="Stage-6 binary mask",
        scale=scale_4d,
        opacity=0.28,
        visible=False,
    )
    viewer.add_labels(
        source_labels,
        name="Stage-6 source segmentation",
        scale=scale_4d,
        opacity=0.45,
        visible=True,
    )
    viewer.add_labels(
        watershed,
        name="STIR-Net watershed supervoxels",
        scale=scale_4d,
        opacity=0.45,
        visible=False,
    )
    viewer.add_labels(
        spatial_partition,
        name="STIR-Net spatial partition",
        scale=scale_4d,
        opacity=0.55,
        visible=True,
    )

    if not args.no_geometry:
        geometry_specs = (
            (
                "foreground probability",
                "geometry/foreground_probability.npy",
                "magenta",
                True,
                (0.0, 1.0),
            ),
            (
                "separator probability",
                "geometry/separator_probability.npy",
                "yellow",
                False,
                (0.0, 1.0),
            ),
            (
                "surface probability",
                "geometry/surface_probability.npy",
                "cyan",
                False,
                (0.0, 1.0),
            ),
            (
                "seed probability",
                "geometry/seed_probability.npy",
                "green",
                False,
                (0.0, 1.0),
            ),
            (
                "SDF",
                "geometry/sdf.npy",
                "turbo",
                False,
                None,
            ),
        )

        for layer_name, relative, colormap, visible, limits in geometry_specs:
            data, _ = stack_npy(
                inference_paths(inference_root, frames, relative),
                name=layer_name,
            )
            kwargs = {
                "name": f"STIR-Net {layer_name}",
                "scale": scale_4d,
                "colormap": colormap,
                "visible": visible,
                "blending": "additive",
            }
            if limits is not None:
                kwargs["contrast_limits"] = limits
            viewer.add_image(data, **kwargs)

    if args.vectors:
        if args.vector_timepoint not in frames:
            raise ValueError(
                f"--vector-timepoint {args.vector_timepoint} is not in the "
                f"displayed frames {frames}"
            )
        displayed_index = frames.index(args.vector_timepoint)
        frame_dir = inference_root / f"t{args.vector_timepoint:03d}"
        stride = parse_zyx(
            args.vector_stride_zyx,
            name="--vector-stride-zyx",
        )

        flow_vectors = build_vector_layer(
            frame_dir / "geometry/flow_zyx.npy",
            frame_dir / "geometry/foreground_probability.npy",
            displayed_time_index=displayed_index,
            stride_zyx=stride,
            minimum_magnitude=args.vector_min_magnitude,
            vector_scale=args.flow_scale,
        )
        centroid_vectors = build_vector_layer(
            frame_dir / "geometry/centroid_offset_zyx.npy",
            frame_dir / "geometry/foreground_probability.npy",
            displayed_time_index=displayed_index,
            stride_zyx=stride,
            minimum_magnitude=args.vector_min_magnitude,
            vector_scale=args.centroid_offset_scale,
        )

        viewer.add_vectors(
            flow_vectors,
            name=f"flow vectors t{args.vector_timepoint:03d}",
            scale=scale_4d,
            edge_width=0.7,
            opacity=0.8,
            visible=True,
        )
        viewer.add_vectors(
            centroid_vectors,
            name=f"centroid-offset vectors t{args.vector_timepoint:03d}",
            scale=scale_4d,
            edge_width=0.7,
            opacity=0.8,
            visible=False,
        )
        print(
            f"[vectors] t{args.vector_timepoint:03d}: "
            f"flow={len(flow_vectors):,}, centroid={len(centroid_vectors):,}",
            flush=True,
        )

    # Time axis is dimension 0.  Start at the first displayed frame.
    viewer.dims.set_current_step(0, 0)

    # Make the Z/Y/X volume immediately obvious rather than opening in a 2-D
    # projection with an ambiguous time dimension.
    viewer.dims.ndisplay = 3

    # The displayed Napari time index maps to these original BioHub frames.
    print(
        "[viewer] Napari T index -> BioHub frame: "
        + ", ".join(f"{i}->t{frame:03d}" for i, frame in enumerate(frames)),
        flush=True,
    )
    print(
        "[viewer] Toggle Stage-6 source segmentation and STIR-Net spatial "
        "partition to inspect corrected merges.",
        flush=True,
    )

    napari.run()


if __name__ == "__main__":
    main()
