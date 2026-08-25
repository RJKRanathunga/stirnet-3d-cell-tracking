# STIRNET_NAPARI_DIRECT_RAG_DECISIONS_V1
# STIRNET_GENERIC_BIOHUB_VIEWER_V2
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


def result_label(
    inference_root: Path,
    manifest: dict,
    override: str | None,
) -> str:
    if override is not None:
        label = override.strip()
        if not label:
            raise ValueError("Result label cannot be empty")
        return label

    run_label = manifest.get("run_label")
    if isinstance(run_label, str) and run_label.strip():
        return run_label.strip()

    checkpoint_path = manifest.get("checkpoint_path")
    if isinstance(checkpoint_path, str) and checkpoint_path:
        parent = Path(checkpoint_path).parent.name
        if parent:
            return parent

    if inference_root.name.startswith("step"):
        return inference_root.parent.name
    return inference_root.name


def load_result_metadata(inference_root: Path) -> tuple[dict, dict]:
    manifest = json.loads(
        (inference_root / "manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (inference_root / "summary.json").read_text(encoding="utf-8")
    )
    return manifest, summary


def watershed_mismatch_frames(
    primary_root: Path,
    comparison_root: Path,
    frames: list[int],
) -> list[int]:
    mismatches: list[int] = []
    relative = Path("partition") / "watershed_supervoxels.npy"

    for frame in frames:
        primary = np.load(
            primary_root / f"t{frame:03d}" / relative,
            mmap_mode="r",
            allow_pickle=False,
        )
        comparison = np.load(
            comparison_root / f"t{frame:03d}" / relative,
            mmap_mode="r",
            allow_pickle=False,
        )
        if (
            primary.shape != comparison.shape
            or primary.dtype != comparison.dtype
            or not np.array_equal(primary, comparison)
        ):
            mismatches.append(int(frame))
    return mismatches


def internal_partition_boundary_4d(labels):
    """Return internal boundaries between positive labels.

    Works lazily for Dask arrays and eagerly for NumPy arrays. Raw label IDs are
    never compared across runs; only each run's own boundary geometry is used.
    """

    try:
        import dask.array as da
        is_dask = isinstance(labels, da.Array)
    except ImportError:
        da = None
        is_dask = False

    pad = da.pad if is_dask else np.pad
    boundary = None

    # labels are [T,Z,Y,X]; never compare across the T axis.
    for axis in (1, 2, 3):
        lower_slice = [slice(None)] * 4
        upper_slice = [slice(None)] * 4
        lower_slice[axis] = slice(0, -1)
        upper_slice[axis] = slice(1, None)

        lower = labels[tuple(lower_slice)]
        upper = labels[tuple(upper_slice)]
        changed = (
            (lower > 0)
            & (upper > 0)
            & (lower != upper)
        )

        low_padding = [(0, 0)] * 4
        high_padding = [(0, 0)] * 4
        low_padding[axis] = (0, 1)
        high_padding[axis] = (1, 0)

        axis_boundary = (
            pad(changed, low_padding, mode="constant")
            | pad(changed, high_padding, mode="constant")
        )
        boundary = (
            axis_boundary
            if boundary is None
            else (boundary | axis_boundary)
        )

    return boundary


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
# Direct RAG edge-decision visualization
# ======================================================================================


def build_direct_rag_interface_layers(
    inference_root: Path,
    frame: int,
    *,
    threshold: float,
):
    """Rasterize pairwise RAG decisions onto exact watershed contact faces."""
    frame_dir = inference_root / f"t{frame:03d}"
    watershed_path = frame_dir / "partition" / "watershed_supervoxels.npy"
    rag_path = frame_dir / "rag" / "rag_state.npz"

    watershed = np.load(watershed_path, mmap_mode="r", allow_pickle=False)
    if watershed.ndim != 3:
        raise ValueError(
            f"Expected 3-D watershed at t{frame:03d}, got {watershed.shape}"
        )

    with np.load(rag_path, allow_pickle=False) as rag:
        required = (
            "node_supervoxel_id",
            "edge_index",
            "spatial_edge_probability",
            "partition_node_component",
        )
        missing = [name for name in required if name not in rag]
        if missing:
            raise KeyError(
                f"{rag_path} is missing direct-decision fields: {missing}"
            )

        node_supervoxel_id = np.asarray(
            rag["node_supervoxel_id"], dtype=np.int64
        )
        edge_index = np.asarray(rag["edge_index"], dtype=np.int64)
        probability = np.asarray(
            rag["spatial_edge_probability"], dtype=np.float32
        ).reshape(-1)
        component = np.asarray(
            rag["partition_node_component"], dtype=np.int64
        ).reshape(-1)

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must be [2,E], got {edge_index.shape}: {rag_path}"
        )

    edge_count = int(edge_index.shape[1])
    if probability.shape != (edge_count,):
        raise ValueError(
            "spatial_edge_probability does not align with edge_index: "
            f"{probability.shape} vs E={edge_count}"
        )

    merge_mask = np.zeros(watershed.shape, dtype=np.uint8)
    separate_mask = np.zeros(watershed.shape, dtype=np.uint8)
    transitive_mask = np.zeros(watershed.shape, dtype=np.uint8)

    if edge_count == 0:
        return (
            merge_mask,
            separate_mask,
            transitive_mask,
            {
                "edge_count": 0,
                "direct_merge_edges": 0,
                "direct_separate_edges": 0,
                "transitive_join_edges": 0,
                "unmatched_contact_faces": 0,
            },
        )

    if edge_index.min() < 0 or edge_index.max() >= len(node_supervoxel_id):
        raise IndexError(f"edge_index references invalid node rows in {rag_path}")

    node_a = edge_index[0]
    node_b = edge_index[1]
    sv_a = node_supervoxel_id[node_a]
    sv_b = node_supervoxel_id[node_b]
    pair_lo = np.minimum(sv_a, sv_b)
    pair_hi = np.maximum(sv_a, sv_b)

    maximum_label = int(
        max(
            int(np.max(watershed)) if watershed.size else 0,
            int(pair_hi.max()) if pair_hi.size else 0,
        )
    )
    base = np.int64(maximum_label + 1)
    edge_keys = pair_lo * base + pair_hi
    order = np.argsort(edge_keys, kind="stable")
    edge_keys = edge_keys[order]
    probability = probability[order]

    final_same_component = (
        component[node_a] == component[node_b]
    )[order]

    direct_merge_edge = probability >= float(threshold)
    direct_separate_edge = ~direct_merge_edge
    transitive_edge = direct_separate_edge & final_same_component

    unmatched_contact_faces = 0

    for axis in range(3):
        lower_slice = [slice(None)] * 3
        upper_slice = [slice(None)] * 3
        lower_slice[axis] = slice(0, -1)
        upper_slice[axis] = slice(1, None)

        lower = np.asarray(watershed[tuple(lower_slice)])
        upper = np.asarray(watershed[tuple(upper_slice)])
        valid = (lower > 0) & (upper > 0) & (lower != upper)
        if not bool(valid.any()):
            continue

        lo = np.minimum(lower[valid], upper[valid]).astype(np.int64, copy=False)
        hi = np.maximum(lower[valid], upper[valid]).astype(np.int64, copy=False)
        face_keys = lo * base + hi

        positions = np.searchsorted(edge_keys, face_keys)
        in_range = positions < edge_count
        matched = np.zeros(face_keys.shape, dtype=bool)
        if bool(in_range.any()):
            matched[in_range] = (
                edge_keys[positions[in_range]] == face_keys[in_range]
            )

        unmatched_contact_faces += int(np.count_nonzero(~matched))

        face_merge_values = np.zeros(face_keys.shape, dtype=bool)
        face_separate_values = np.zeros(face_keys.shape, dtype=bool)
        face_transitive_values = np.zeros(face_keys.shape, dtype=bool)

        if bool(matched.any()):
            edge_rows = positions[matched]
            face_merge_values[matched] = direct_merge_edge[edge_rows]
            face_separate_values[matched] = direct_separate_edge[edge_rows]
            face_transitive_values[matched] = transitive_edge[edge_rows]

        face_merge = np.zeros(valid.shape, dtype=bool)
        face_separate = np.zeros(valid.shape, dtype=bool)
        face_transitive = np.zeros(valid.shape, dtype=bool)
        face_merge[valid] = face_merge_values
        face_separate[valid] = face_separate_values
        face_transitive[valid] = face_transitive_values

        merge_mask[tuple(lower_slice)] |= face_merge
        merge_mask[tuple(upper_slice)] |= face_merge
        separate_mask[tuple(lower_slice)] |= face_separate
        separate_mask[tuple(upper_slice)] |= face_separate
        transitive_mask[tuple(lower_slice)] |= face_transitive
        transitive_mask[tuple(upper_slice)] |= face_transitive

    stats = {
        "edge_count": edge_count,
        "direct_merge_edges": int(np.count_nonzero(direct_merge_edge)),
        "direct_separate_edges": int(np.count_nonzero(direct_separate_edge)),
        "transitive_join_edges": int(np.count_nonzero(transitive_edge)),
        "unmatched_contact_faces": int(unmatched_contact_faces),
    }
    return merge_mask, separate_mask, transitive_mask, stats


def stack_direct_rag_interface_layers(
    inference_root: Path,
    frames: list[int],
    *,
    threshold: float,
):
    merge_rows = []
    separate_rows = []
    transitive_rows = []
    frame_stats = {}

    for frame in frames:
        merge_mask, separate_mask, transitive_mask, stats = (
            build_direct_rag_interface_layers(
                inference_root,
                frame,
                threshold=threshold,
            )
        )
        merge_rows.append(merge_mask)
        separate_rows.append(separate_mask)
        transitive_rows.append(transitive_mask)
        frame_stats[int(frame)] = stats

    return (
        np.stack(merge_rows, axis=0),
        np.stack(separate_rows, axis=0),
        np.stack(transitive_rows, axis=0),
        frame_stats,
    )


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
            "Primary Investigation-12-compatible result directory. Default "
            "selects the highest legacy step for the sample."
        ),
    )
    parser.add_argument(
        "--inference-label",
        default=None,
        help="Optional display label for the primary result.",
    )
    parser.add_argument(
        "--compare-dir",
        action="append",
        default=[],
        help=(
            "Additional Investigation-12-compatible result directory to compare "
            "against the primary. Repeat this argument for multiple models."
        ),
    )
    parser.add_argument(
        "--compare-label",
        action="append",
        default=[],
        help=(
            "Optional display label corresponding to each --compare-dir, in "
            "the same order. Omit to infer labels from manifests."
        ),
    )
    parser.add_argument(
        "--no-difference-overlays",
        action="store_true",
        help=(
            "Do not add primary-only/comparison-only internal-boundary overlays."
        ),
    )
    parser.add_argument(
        "--no-watershed-check",
        action="store_true",
        help="Skip exact watershed identity checks between result directories.",
    )
    parser.add_argument(
        "--direct-decisions",
        action="store_true",
        help=(
            "Add primary-run direct RAG MERGE/SEPARATE interface overlays and "
            "a transitive-join diagnostic layer. Uses saved rag_state.npz."
        ),
    )
    parser.add_argument(
        "--direct-threshold",
        type=float,
        default=None,
        help=(
            "Optional direct RAG merge threshold. Default uses the manifest "
            "spatial_merge_threshold (normally 0.845)."
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

    manifest, summary = load_result_metadata(inference_root)
    spacing = tuple(
        float(value)
        for value in manifest.get(
            "spacing_zyx_um",
            DEFAULT_SPACING_ZYX_UM,
        )
    )
    scale_4d = (1.0, *spacing)
    primary_label = result_label(
        inference_root,
        manifest,
        args.inference_label,
    )

    if len(args.compare_label) > len(args.compare_dir):
        raise ValueError(
            "More --compare-label values were provided than --compare-dir values"
        )

    comparisons = []
    for index, directory_text in enumerate(args.compare_dir):
        comparison_root = resolve_inference_directory(
            args.sample_id,
            directory_text,
        )
        if comparison_root == inference_root:
            raise ValueError(
                f"Comparison directory equals primary directory: {comparison_root}"
            )

        comparison_available = set(
            completed_timepoints(comparison_root)
        )
        missing_frames = [
            frame
            for frame in frames
            if frame not in comparison_available
        ]
        if missing_frames:
            raise FileNotFoundError(
                f"Comparison directory {comparison_root} is missing displayed "
                f"frames: {missing_frames}"
            )

        comparison_manifest, comparison_summary = load_result_metadata(
            comparison_root
        )
        comparison_spacing = tuple(
            float(value)
            for value in comparison_manifest.get(
                "spacing_zyx_um",
                DEFAULT_SPACING_ZYX_UM,
            )
        )
        if not np.allclose(
            np.asarray(comparison_spacing, dtype=np.float64),
            np.asarray(spacing, dtype=np.float64),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                "Primary/comparison physical spacing differs: "
                f"{spacing} vs {comparison_spacing} ({comparison_root})"
            )

        explicit_label = (
            args.compare_label[index]
            if index < len(args.compare_label)
            else None
        )
        label = result_label(
            comparison_root,
            comparison_manifest,
            explicit_label,
        )
        comparisons.append(
            {
                "root": comparison_root,
                "manifest": comparison_manifest,
                "summary": comparison_summary,
                "label": label,
            }
        )

    print("=" * 110)
    print("STIR-Net Investigation 13 — BioHub full-volume result viewer")
    print("=" * 110)
    print("sample             :", args.sample_id)
    print("Raw sample Zarr    :", sample_zarr)
    print("Stage-6            :", stage6_root)
    print("Primary inference  :", inference_root)
    print("Primary label      :", primary_label)
    print("checkpoint step    :", manifest.get("checkpoint_step"))
    for index, comparison in enumerate(comparisons, 1):
        print(
            f"Compare {index:<11}:",
            comparison["label"],
            "->",
            comparison["root"],
        )
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
        name=f"STIR-Net spatial partition [{primary_label}]",
    )

    comparison_partitions = []
    for comparison in comparisons:
        comparison_partition, _ = stack_npy(
            inference_paths(
                comparison["root"],
                frames,
                "partition/spatial_partition.npy",
            ),
            name=(
                "STIR-Net spatial partition "
                f"[{comparison['label']}]"
            ),
        )
        comparison_partitions.append(
            (comparison, comparison_partition)
        )

        if not args.no_watershed_check:
            mismatches = watershed_mismatch_frames(
                inference_root,
                comparison["root"],
                frames,
            )
            print(
                f"[compare] watershed {primary_label} vs "
                f"{comparison['label']}: "
                + (
                    "IDENTICAL"
                    if not mismatches
                    else f"DIFFERS at frames {mismatches}"
                ),
                flush=True,
            )

    direct_decision_layers = None
    if args.direct_decisions:
        direct_threshold = (
            float(args.direct_threshold)
            if args.direct_threshold is not None
            else float(manifest.get("spatial_merge_threshold", 0.845))
        )
        if not 0.0 <= direct_threshold <= 1.0:
            raise ValueError("--direct-threshold must be in [0,1]")

        print(
            f"[direct RAG] rasterizing pairwise decisions at "
            f"threshold={direct_threshold:.4f} ...",
            flush=True,
        )
        direct_merge, direct_separate, transitive_join, direct_stats = (
            stack_direct_rag_interface_layers(
                inference_root,
                frames,
                threshold=direct_threshold,
            )
        )
        direct_decision_layers = (
            direct_merge,
            direct_separate,
            transitive_join,
            direct_threshold,
        )

        print(
            "[direct RAG] per-frame: "
            + ", ".join(
                (
                    f"t{frame:03d}: "
                    f"M={row['direct_merge_edges']} "
                    f"S={row['direct_separate_edges']} "
                    f"T={row['transitive_join_edges']}"
                )
                for frame, row in direct_stats.items()
            ),
            flush=True,
        )
        print(
            "[direct RAG] M=direct merge, S=direct separate, "
            "T=direct separate but same final component",
            flush=True,
        )
        unmatched = sum(
            row["unmatched_contact_faces"]
            for row in direct_stats.values()
        )
        if unmatched:
            print(
                f"[direct RAG] WARNING: {unmatched} contact faces did not map "
                "to a saved RAG edge.",
                flush=True,
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
            f"{primary_label} | step {manifest.get('checkpoint_step', '?')}"
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
        name=f"STIR-Net spatial partition [{primary_label}]",
        scale=scale_4d,
        opacity=0.55,
        visible=True,
    )

    if direct_decision_layers is not None:
        direct_merge, direct_separate, transitive_join, direct_threshold = (
            direct_decision_layers
        )
        viewer.add_image(
            direct_merge,
            name=(
                f"DIRECT RAG says MERGE [{primary_label}] "
                f"p>={direct_threshold:.3f}"
            ),
            scale=scale_4d,
            colormap="red",
            contrast_limits=(0.0, 1.0),
            opacity=0.95,
            blending="additive",
            visible=False,
        )
        viewer.add_image(
            direct_separate,
            name=(
                f"DIRECT RAG says SEPARATE [{primary_label}] "
                f"p<{direct_threshold:.3f}"
            ),
            scale=scale_4d,
            colormap="green",
            contrast_limits=(0.0, 1.0),
            opacity=0.95,
            blending="additive",
            visible=False,
        )
        viewer.add_image(
            transitive_join,
            name=(
                "TRANSITIVE join despite DIRECT SEPARATE "
                f"[{primary_label}]"
            ),
            scale=scale_4d,
            colormap="magenta",
            contrast_limits=(0.0, 1.0),
            opacity=1.0,
            blending="additive",
            visible=True,
        )


    for comparison, comparison_partition in comparison_partitions:
        viewer.add_labels(
            comparison_partition,
            name=(
                "STIR-Net spatial partition "
                f"[{comparison['label']}]"
            ),
            scale=scale_4d,
            opacity=0.55,
            visible=False,
        )

    if comparison_partitions and not args.no_difference_overlays:
        primary_boundary = internal_partition_boundary_4d(
            spatial_partition
        )
        for index, (
            comparison,
            comparison_partition,
        ) in enumerate(comparison_partitions):
            comparison_boundary = internal_partition_boundary_4d(
                comparison_partition
            )
            primary_only = (
                primary_boundary & ~comparison_boundary
            )
            comparison_only = (
                comparison_boundary & ~primary_boundary
            )

            viewer.add_image(
                primary_only,
                name=(
                    f"DIFF boundary [{primary_label}] only "
                    f"vs [{comparison['label']}]"
                ),
                scale=scale_4d,
                colormap="green",
                contrast_limits=(0.0, 1.0),
                opacity=1.0,
                blending="additive",
                visible=(index == 0),
            )
            viewer.add_image(
                comparison_only,
                name=(
                    f"DIFF boundary [{comparison['label']}] only "
                    f"vs [{primary_label}]"
                ),
                scale=scale_4d,
                colormap="red",
                contrast_limits=(0.0, 1.0),
                opacity=1.0,
                blending="additive",
                visible=False,
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
    if comparisons:
        print(
            "[viewer] Toggle the named spatial-partition layers for direct "
            "model comparison.",
            flush=True,
        )
        if not args.no_difference_overlays:
            print(
                "[viewer] GREEN = internal boundary present only in PRIMARY; "
                "RED = boundary present only in COMPARISON.",
                flush=True,
            )
            print(
                "[viewer] Difference overlays compare boundary geometry, not "
                "raw label IDs, so relabeling alone does not create a change.",
                flush=True,
            )
    else:
        print(
            "[viewer] Toggle Stage-6 source segmentation and STIR-Net spatial "
            "partition to inspect corrected merges.",
            flush=True,
        )

    napari.run()


if __name__ == "__main__":
    main()
