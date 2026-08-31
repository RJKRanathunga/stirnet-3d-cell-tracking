from __future__ import annotations

# DATASET_CURATION_REFACTOR_CURRENT_V1: migrated current instance annotator

r"""
Interactive BioHub supervoxel split annotator — v11.

# STIRNET_ANNOTATOR_V11_DEFAULT_INV25_V1

Purpose
-------
Use the current spatial segmentation as pseudo-ground-truth and manually correct
the comparatively rare merged cells using 2-4 groups of representative atomic
supervoxels as split seeds.

You do NOT need to enumerate every supervoxel belonging to each true cell.
The entered IDs are only seed/hint supervoxels. The script expands those seeds
over the current merged instance using its 6-neighbour supervoxel contact graph.
Broad supervoxel contacts are cheaper to traverse than narrow contacts, so the
automatic split naturally prefers narrow necks / weak geometric connections.

The viewer shows:
    1. raw BioHub image,
    2. Stage-6 binary foreground mask,
    3. current/corrected instance labels,
    4. atomic supervoxel boundaries,
    5. supervoxel ID text placed just outside foreground surfaces.

Workflow
--------
For the current timepoint:
    - Find a merged predicted instance.
    - Read the supervoxel IDs around that instance.
    - Put comma-separated IDs for true cell #1 in "Instance 1".
    - Put comma-separated IDs for true cell #2 in "Instance 2".
    - Optionally use Instance 3 and Instance 4.
    - Press Save.

Rules enforced by the tool:
    - At least 2 boxes must be non-empty.
    - A supervoxel cannot appear in more than one box.
    - Every entered supervoxel must exist in the current frame.
    - All entered seed supervoxels must belong to the same CURRENT predicted
      instance, because the operation is a split of one merged instance.
    - The remaining supervoxels do NOT need to be entered.

On Save:
    - the seed groups are expanded over the merged instance with a weighted
      supervoxel-contact graph,
    - the selected merged instance is replaced by 2-4 new instance IDs,
    - the Napari label colors refresh immediately,
    - the correction remains in memory while moving through time,
    - corrected full-frame labels and a JSON correction log are written to disk.

Undo:
    - "Undo last Save" reverses the most recent split operation,
    - it works even after moving to another timepoint,
    - the viewer jumps back to the affected frame,
    - the previous instance label is restored,
    - the persisted .npy volumes and JSON correction log are updated immediately.

3-D ray picking:
    - freely rotate the volume,
    - single-click a visible atomic supervoxel anywhere on the canvas,
    - the click defines a camera ray through the 3-D supervoxel labels,
    - the FIRST non-background supervoxel hit by that ray is selected,
    - click #1 automatically fills Instance 1,
    - click #2 automatically fills Instance 2,
    - click #3/#4 fill Instance 3/#4 if needed,
    - selected seed supervoxels get distinct highlight colors,
    - click-drag camera navigation is left unchanged,
    - "Reset selections" clears the boxes and seed highlights,
    - Escape is the keyboard shortcut for Reset selections,
    - Save automatically resets the selections after a successful split.

Unique label display coloring:
    - every positive atomic-supervoxel label value gets its own display color,
    - every positive segmented/current-instance label value gets its own display color,
    - two DIFFERENT label IDs therefore never intentionally reuse the same color,
    - touching labels are automatically different because all distinct IDs differ,
    - the mapping is deterministic across all loaded timepoints,
    - after Save/Undo the changed instance frame is rescanned and new output IDs
      receive new unique colors.

Why this is stronger than the previous graph coloring:
    - graph coloring only guaranteed different colors when two labels physically
      shared a voxel face,
    - two separate cells with a thin background gap could legally reuse a color,
    - that was visually misleading in 3-D projection,
    - this annotator therefore uses unique-per-label colors instead.

Default production input
------------------------
By default this script now uses Investigation 25 as the CURRENT instance
segmentation:

    runs/stirnet/evaluation/
        25_source_core_split_biohub_visualization/
        <sample-id>/
        source_instance_anchors_supervoxel_graph_defaults/
            t000/partition/after_split_only.npy
            ...
            t019/partition/after_split_only.npy

Investigation 25 deliberately reuses the Investigation-24 atomic watershed
instead of duplicating it. Therefore the annotator loads atomic SV IDs from:

    runs/stirnet/evaluation/
        24_multicut_biohub_full_volume_visualization/
        <sample-id>/h100_q0p845/
            t000/partition/watershed_supervoxels.npy
            ...
            t019/partition/watershed_supervoxels.npy

`after_split_only.npy` supplies the final Investigation-25 instances.
`watershed_supervoxels.npy` supplies the same atomic SV IDs used by
Investigation 25.

Timepoint selection examples:
    --timepoints all
    --timepoints 0-19
    --timepoints 4,5,8-12

Legacy --supervoxels/--instances overrides are still supported, but normally
you should omit them.

The BioHub physical scale is unchanged across Investigations 24/25:
    Z,Y,X = 1.625, 0.40625, 0.40625 um

BINARY_MASK_PATH is optional. If omitted, after_split_only > 0 is used as the
foreground mask for placing supervoxel text outside cell surfaces.

Run this file from the repository root.

    python .\apply_biohub_merge_suspect_layer_patch.py

Then you can run the scorer separately:

    python .\evaluation\segmentation\scripts\03_biohub_merge_suspect_export.py `
        --sample-id 44b6_0113de3b `
        --timepoints all

and later open the annotator with any threshold:

    python .\evaluation\segmentation\scripts\02_supervoxel_instance_annotator.py `
        --timepoints all `
        --suspect-threshold 0.70

Or use the one-command version:

    python .\evaluation\segmentation\scripts\02_supervoxel_instance_annotator.py `
        --timepoints all `
        --build-suspects `
        --suspect-threshold 0.70
"""

import argparse
import heapq
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import napari
import numpy as np
from scipy import ndimage

try:
    from magicgui.widgets import Container, Label, LineEdit, PushButton
except ImportError as exc:
    raise ImportError(
        "magicgui is required for the annotation panel. It normally comes with "
        "Napari. Install it with: pip install magicgui"
    ) from exc

try:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QSizePolicy
except ImportError:
    Qt = None
    QSizePolicy = None


# ============================================================
# REPOSITORY / DEFAULT CONFIGURATION
# ============================================================

from dataset_curation._repo import repo_root as _dataset_curation_repo_root

REPO_ROOT = _dataset_curation_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io import load_timepoint  # noqa: E402


DEFAULT_SAMPLE_ID = "44b6_0113de3b"

# Change these to the 2-3 consecutive BioHub frames you want to annotate.
DEFAULT_TIMEPOINTS = (0, 1, 2)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "evaluation" / "segmentation" / "annotations"

# STIRNET_ANNOTATOR_SUSPECT_LAYER_V1
DEFAULT_SUSPECT_ROOT = REPO_ROOT / "evaluation" / "segmentation" / "suspects"
DEFAULT_SUSPECT_THRESHOLD = 0.70

DEFAULT_INV25_ROOT = (
    REPO_ROOT
    / "runs"
    / "stirnet"
    / "evaluation"
    / "25_source_core_split_biohub_visualization"
)

DEFAULT_INV24_MULTICUT_ROOT = (
    REPO_ROOT
    / "runs"
    / "stirnet"
    / "evaluation"
    / "24_multicut_biohub_full_volume_visualization"
)

DEFAULT_INV25_VARIANT = "source_instance_anchors_supervoxel_graph_defaults"

DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)

# Text is placed only slightly beyond the first foreground -> background
# crossing in the selected z-slice. A 0.5 px offset lets the number visually
# touch the supervoxel/cell surface while keeping it outside the foreground.
LABEL_SURFACE_OFFSET_PX = 0.5

# Try to keep two text anchors in the same t/z plane at least this far apart.
LABEL_MIN_SEPARATION_PX = 10.0

# Maximum local search radius when two labels would overlap.
LABEL_REPOSITION_RADIUS_PX = 24

# The supervoxel boundary layer uses contour mode where supported.
SUPERVOXEL_CONTOUR_WIDTH = 1


# ============================================================
# ERRORS / DATA CLASSES
# ============================================================


class AnnotationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SplitResult:
    timepoint: int
    original_instance_id: int
    output_instance_ids: tuple[int, ...]
    seed_groups: tuple[tuple[int, ...], ...]
    groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class UndoResult:
    timepoint: int
    original_instance_id: int
    removed_instance_ids: tuple[int, ...]


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive supervoxel-based merged-cell annotator."
    )

    parser.add_argument(
        "--sample-id",
        default=DEFAULT_SAMPLE_ID,
        help="BioHub sample ID.",
    )
    parser.add_argument(
        "--timepoints",
        default="all",
        help=(
            "BioHub timepoints: 'all', a range such as '0-19', or a "
            "comma-separated selection such as '0,1,4-7'."
        ),
    )
    parser.add_argument(
        "--spatial-root",
        type=Path,
        default=None,
        help=(
            "Investigation-25 output root containing "
            "t###/partition/after_split_only.npy. If omitted, the standard "
            "source_instance_anchors_supervoxel_graph_defaults root is used."
        ),
    )
    parser.add_argument(
        "--supervoxel-root",
        type=Path,
        default=None,
        help=(
            "Investigation-24 h100_q0p845 root containing "
            "t###/partition/watershed_supervoxels.npy. Investigation 25 "
            "reuses these atomic supervoxels. Normally omit this option."
        ),
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=None,
        help="Raw BioHub .zarr path. Default is derived from --sample-id.",
    )
    parser.add_argument(
        "--stage6-root",
        type=Path,
        default=None,
        help=(
            "Optional Stage-6 sample directory containing preprocessing/, "
            "masking/, and segmentation/. Default is resolved through "
            "src.io.PipelinePaths for --sample-id."
        ),
    )
    parser.add_argument(
        "--supervoxels",
        type=Path,
        default=None,
        help=(
            "Optional legacy override for atomic supervoxels. Normally omit "
            "this and use the default Investigation-25/24 roots."
        ),
    )
    parser.add_argument(
        "--instances",
        type=Path,
        default=None,
        help=(
            "Optional legacy override for spatial instances. Normally omit "
            "this and use the default Investigation-25/24 roots."
        ),
    )
    parser.add_argument(
        "--binary-mask",
        type=Path,
        default=None,
        help="Optional binary-mask .npy/.npz stack or per-timepoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Annotation output directory.",
    )
    parser.add_argument(
        "--suspect-root",
        type=Path,
        default=None,
        help=(
            "Directory containing 03_ suspect t###.npz outputs. Default: "
            "evaluation/segmentation/suspects/<sample-id>."
        ),
    )
    parser.add_argument(
        "--suspect-threshold",
        type=float,
        default=DEFAULT_SUSPECT_THRESHOLD,
        help=(
            "Display an original predicted instance when suspect_score is >= "
            "this threshold. Changing it does not rerun inference."
        ),
    )
    parser.add_argument(
        "--build-suspects",
        action="store_true",
        help=(
            "Run sibling 03_biohub_merge_suspect_export.py first, then consume "
            "its score files."
        ),
    )
    parser.add_argument(
        "--suspect-checkpoint",
        type=Path,
        default=None,
        help="Optional causal temporal checkpoint passed to the 03_ exporter.",
    )
    parser.add_argument(
        "--suspect-device",
        default="auto",
        help="Device passed to 03_: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--rebuild-suspects",
        action="store_true",
        help="Recompute score NPZ files when --build-suspects is used.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing manual-instance outputs in the output directory.",
    )
    return parser.parse_args()


def default_spatial_root(sample_id: str) -> Path:
    """Final Investigation-25 split-only instance root."""
    return (
        DEFAULT_INV25_ROOT
        / sample_id
        / DEFAULT_INV25_VARIANT
    )


def default_supervoxel_root(sample_id: str) -> Path:
    """Investigation-24 atomic watershed inherited by Investigation 25."""
    return (
        DEFAULT_INV24_MULTICUT_ROOT
        / sample_id
        / "h100_q0p845"
    )


def completed_spatial_frames(root: Path) -> list[int]:
    frames: list[int] = []

    if not root.is_dir():
        return frames

    for frame_dir in root.glob("t[0-9][0-9][0-9]"):
        if not frame_dir.is_dir():
            continue

        partition_dir = frame_dir / "partition"
        if (partition_dir / "after_split_only.npy").is_file():
            frames.append(int(frame_dir.name[1:]))

    return sorted(frames)


def parse_timepoint_selection(
    text: str,
    available: list[int],
) -> tuple[int, ...]:
    token = str(text).strip().lower()

    if not available:
        raise AnnotationError(
            "No completed Investigation-25 frames were found."
        )

    if token in {"all", "*"}:
        return tuple(available)

    selected: set[int] = set()

    for item in token.split(","):
        item = item.strip()
        if not item:
            continue

        if "-" in item:
            left, right = item.split("-", 1)
            first = int(left)
            last = int(right)

            if last < first:
                raise AnnotationError(
                    f"Invalid timepoint range: {item}"
                )

            selected.update(range(first, last + 1))
        else:
            selected.add(int(item))

    if not selected:
        raise AnnotationError("No timepoints selected.")

    missing = sorted(selected - set(available))
    if missing:
        raise AnnotationError(
            f"Requested timepoints are unavailable: {missing}. "
            f"Available frames: {available}"
        )

    return tuple(sorted(selected))


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    sample_id = args.sample_id

    if args.zarr_path is None:
        args.zarr_path = (
            REPO_ROOT
            / "data"
            / "sample"
            / "biohub_5samples_20timepoints"
            / "train"
            / sample_id
            / f"{sample_id}.zarr"
        )

    if args.spatial_root is None:
        args.spatial_root = default_spatial_root(sample_id)

    if args.supervoxel_root is None:
        args.supervoxel_root = default_supervoxel_root(sample_id)

    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / sample_id

    if args.suspect_root is None:
        args.suspect_root = DEFAULT_SUSPECT_ROOT / sample_id

    if not 0.0 <= float(args.suspect_threshold) <= 1.0:
        raise AnnotationError("--suspect-threshold must lie in [0, 1].")

    return args


def resolve_stage6_root(
    sample_id: str,
    override: Path | None,
) -> Path:
    """Resolve the same Stage-6 sample root used by Investigation 13/24."""
    from src.io import PipelinePaths

    if override is None:
        root = (
            PipelinePaths.discover(REPO_ROOT)
            .processed_dataset(sample_id)
        )
    else:
        supplied = Path(override).expanduser()
        if not supplied.is_absolute():
            supplied = REPO_ROOT / supplied
        supplied = supplied.resolve()

        if (supplied / "masking").is_dir():
            root = supplied
        elif (supplied / sample_id / "masking").is_dir():
            root = supplied / sample_id
        else:
            root = supplied

    required = ("preprocessing", "masking", "segmentation")
    missing = [
        name for name in required
        if not (root / name).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            f"Stage-6 directory is incomplete: {root}; missing={missing}"
        )

    return root.resolve()


def load_stage6_binary_mask_frames(
    stage6_root: Path,
    timepoints: tuple[int, ...],
) -> np.ndarray:
    """Stack Stage-6 masking/t###.npy into [T,Z,Y,X] uint8."""
    frames: list[np.ndarray] = []

    for dataset_t in timepoints:
        path = stage6_root / "masking" / f"t{dataset_t:03d}.npy"

        if not path.is_file():
            raise FileNotFoundError(
                "Missing Stage-6 binary mask:\n"
                f"  {path}"
            )

        mask = np.asarray(
            np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )
        )

        if mask.ndim != 3:
            raise AnnotationError(
                f"Stage-6 mask at t={dataset_t} must be 3-D; "
                f"got {mask.shape}."
            )

        frames.append(
            (mask > 0).astype(
                np.uint8,
                copy=False,
            )
        )

    return np.stack(frames, axis=0)


def load_investigation25_frames(
    spatial_root: Path,
    supervoxel_root: Path,
    timepoints: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Load Inv25 final instances with the exact Inv24 atomic SVs they refine."""
    supervoxel_frames: list[np.ndarray] = []
    instance_frames: list[np.ndarray] = []

    for t in timepoints:
        inv25_partition = spatial_root / f"t{t:03d}" / "partition"
        inv24_partition = supervoxel_root / f"t{t:03d}" / "partition"

        sv_path = inv24_partition / "watershed_supervoxels.npy"
        instance_path = inv25_partition / "after_split_only.npy"

        if not sv_path.is_file():
            raise FileNotFoundError(
                "Missing Investigation-24 atomic watershed supervoxels "
                "required by Investigation 25:\n"
                f"  {sv_path}"
            )

        if not instance_path.is_file():
            raise FileNotFoundError(
                "Missing Investigation-25 final split-only instances:\n"
                f"  {instance_path}"
            )

        print(
            f"[spatial] loading t={t}:\n"
            f"  supervoxels : {sv_path}\n"
            f"  instances   : {instance_path}"
        )

        sv = np.asarray(np.load(sv_path, mmap_mode="r", allow_pickle=False))
        instances = np.asarray(
            np.load(instance_path, mmap_mode="r", allow_pickle=False)
        )

        if sv.ndim != 3 or instances.ndim != 3:
            raise AnnotationError(
                f"Investigation-24/25 arrays must be 3-D at t={t}; "
                f"got supervoxels={sv.shape}, instances={instances.shape}."
            )

        if sv.shape != instances.shape:
            raise AnnotationError(
                f"Investigation-24/25 shape mismatch at t={t}: "
                f"supervoxels={sv.shape}, instances={instances.shape}."
            )

        supervoxel_frames.append(sv)
        instance_frames.append(instances)

    return (
        np.stack(supervoxel_frames, axis=0),
        np.stack(instance_frames, axis=0),
    )


# ============================================================
# OPTIONAL MERGE-SUSPECT DISPLAY ARTIFACT
# ============================================================


def _suspect_score_path(suspect_root: Path, dataset_t: int) -> Path:
    return suspect_root / f"t{int(dataset_t):03d}.npz"


def run_suspect_exporter(
    *,
    args: argparse.Namespace,
    timepoints: tuple[int, ...],
    spatial_root: Path,
    supervoxel_root: Path,
) -> None:
    """Convenience wrapper; all scoring remains in sibling 03_."""
    script = Path(__file__).resolve().parent / "merge_suspect_exporter.py"
    if not script.is_file():
        raise FileNotFoundError(f"Merge-suspect exporter is missing:\n  {script}")

    selection = ",".join(str(int(t)) for t in timepoints)
    command = [
        sys.executable,
        str(script),
        "--sample-id", str(args.sample_id),
        "--timepoints", selection,
        "--inv25", str(spatial_root),
        "--inv24", str(supervoxel_root),
        "--zarr", str(args.zarr_path),
        "--output-dir", str(args.suspect_root),
        "--device", str(args.suspect_device),
    ]
    if args.suspect_checkpoint is not None:
        command.extend(["--checkpoint", str(args.suspect_checkpoint)])
    if args.rebuild_suspects:
        command.append("--rebuild-scores")

    print()
    print("=" * 72)
    print("Building merge-suspect scores with sibling 03_ exporter")
    print("=" * 72)
    print(" ".join(command))
    print("=" * 72)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def load_suspect_instance_frames(
    *,
    suspect_root: Path,
    timepoints: tuple[int, ...],
    instances: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Rasterize threshold-passing ORIGINAL instance IDs only."""
    if instances.ndim != 4:
        raise AnnotationError(
            f"Suspect display expects (T,Z,Y,X); got {instances.shape}."
        )
    if len(timepoints) != instances.shape[0]:
        raise AnnotationError("Suspect timepoint/instance-stack length mismatch.")

    output = np.zeros_like(instances)
    total_rows = 0
    total_displayed = 0

    for local_t, dataset_t in enumerate(timepoints):
        path = _suspect_score_path(suspect_root, dataset_t)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing merge-suspect score file:\n  {path}\n"
                "Run 03_ first or pass --build-suspects."
            )
        with np.load(path, allow_pickle=False) as payload:
            missing = [
                key for key in ("instance_id", "suspect_score")
                if key not in payload.files
            ]
            if missing:
                raise AnnotationError(f"{path} is missing arrays: {missing}")
            ids = np.asarray(payload["instance_id"], dtype=np.int64).reshape(-1)
            scores = np.asarray(payload["suspect_score"], dtype=np.float32).reshape(-1)

        if ids.shape != scores.shape:
            raise AnnotationError(f"ID/score mismatch in {path}")
        if ids.size and len(np.unique(ids)) != len(ids):
            raise AnnotationError(f"Duplicate instance IDs in {path}")
        if np.any(ids <= 0) or np.any(~np.isfinite(scores)):
            raise AnnotationError(f"Invalid suspect rows in {path}")

        frame = instances[local_t]
        max_label = int(frame.max(initial=0))
        lookup = np.zeros(max_label + 1, dtype=bool)
        passing_ids = ids[scores >= float(threshold)]
        passing_ids = passing_ids[passing_ids <= max_label]
        if passing_ids.size:
            lookup[passing_ids] = True
            frame_index = frame.astype(np.int64, copy=False)
            output[local_t] = np.where(lookup[frame_index], frame, 0).astype(
                instances.dtype, copy=False
            )

        total_rows += int(len(ids))
        total_displayed += int(len(passing_ids))
        print(
            f"[suspects] t={dataset_t}: {len(passing_ids)}/{len(ids)} "
            f"instances >= {float(threshold):.3f}"
        )

    return output, {
        "score_rows": int(total_rows),
        "displayed_instances": int(total_displayed),
    }


# ============================================================
# INPUT LOADING
# ============================================================


def _pick_npz_array(npz: np.lib.npyio.NpzFile, role: str) -> np.ndarray:
    preferred = {
        "supervoxels": ("supervoxels", "atomic_labels", "labels", "arr_0"),
        "instances": ("instances", "instance_labels", "labels", "arr_0"),
        "binary_mask": ("binary_mask", "mask", "foreground", "arr_0"),
    }[role]

    for key in preferred:
        if key in npz.files:
            return np.asarray(npz[key])

    if len(npz.files) == 1:
        return np.asarray(npz[npz.files[0]])

    raise AnnotationError(
        f"Could not decide which array to use from NPZ for role={role!r}. "
        f"Available keys: {npz.files}"
    )


def _load_array_file(path: Path, role: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, mmap_mode="r")

    if path.suffix.lower() == ".npz":
        with np.load(path) as npz:
            return _pick_npz_array(npz, role)

    raise AnnotationError(
        f"Unsupported {role} file type: {path}. Use .npy or .npz."
    )


def _find_frame_file(directory: Path, role: str, timepoint: int) -> Path:
    candidates = [
        directory / f"{role}_t{timepoint:03d}.npy",
        directory / f"{role}_t{timepoint:04d}.npy",
        directory / f"t{timepoint:03d}_{role}.npy",
        directory / f"t{timepoint:04d}_{role}.npy",
        directory / f"t{timepoint:03d}.npy",
        directory / f"t{timepoint:04d}.npy",
        directory / f"{timepoint:03d}.npy",
        directory / f"{timepoint:04d}.npy",
    ]

    # Common singular aliases.
    aliases = {
        "supervoxels": ("supervoxel", "atomic_labels", "atomic"),
        "instances": ("instance", "instance_labels", "segmentation"),
        "binary_mask": ("mask", "foreground"),
    }[role]

    for alias in aliases:
        candidates.extend(
            [
                directory / f"{alias}_t{timepoint:03d}.npy",
                directory / f"{alias}_t{timepoint:04d}.npy",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Last-resort unambiguous glob.
    matches = sorted(directory.glob(f"*{timepoint:03d}*.npy"))
    if len(matches) == 1:
        return matches[0]

    raise AnnotationError(
        f"Could not find a unique {role} file for t={timepoint} in {directory}."
    )


def load_selected_label_frames(
    path: Path,
    timepoints: tuple[int, ...],
    *,
    role: str,
) -> np.ndarray:
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"{role} input does not exist:\n  {path}\n\n"
            "Either edit the defaults at the top of this script or pass the "
            f"corresponding --{role.replace('_', '-')} argument."
        )

    if path.is_dir():
        frames = []
        for t in timepoints:
            frame_path = _find_frame_file(path, role, t)
            frame = np.asarray(np.load(frame_path))
            if frame.ndim != 3:
                raise AnnotationError(
                    f"{frame_path} must be 3-D (Z,Y,X); got {frame.shape}."
                )
            frames.append(frame)
        return np.stack(frames, axis=0)

    array = _load_array_file(path, role)

    if array.ndim == 3:
        if len(timepoints) != 1:
            raise AnnotationError(
                f"{path} is one 3-D frame but {len(timepoints)} timepoints were "
                "requested."
            )
        return np.asarray(array)[None, ...]

    if array.ndim != 4:
        raise AnnotationError(
            f"{path} must have shape (T,Z,Y,X) or (Z,Y,X); got {array.shape}."
        )

    if max(timepoints) >= array.shape[0] or min(timepoints) < 0:
        raise AnnotationError(
            f"Requested timepoints {timepoints}, but {path} has T={array.shape[0]}."
        )

    # Copy only the selected 3-D volumes into RAM. We intentionally avoid
    # materializing the entire source stack.
    return np.stack([np.asarray(array[t]) for t in timepoints], axis=0)


def load_raw_frames(zarr_path: Path, timepoints: tuple[int, ...]) -> np.ndarray:
    if not zarr_path.exists():
        raise FileNotFoundError(f"Raw Zarr does not exist:\n  {zarr_path}")

    frames = []
    for t in timepoints:
        print(f"[raw] loading t={t}")
        frame = np.asarray(load_timepoint(zarr_path, t))
        if frame.ndim != 3:
            raise AnnotationError(
                f"load_timepoint(..., {t}) returned {frame.shape}; expected Z,Y,X."
            )
        frames.append(frame)

    return np.stack(frames, axis=0)


def validate_stacks(
    raw: np.ndarray,
    supervoxels: np.ndarray,
    instances: np.ndarray,
    foreground: np.ndarray,
) -> None:
    expected = raw.shape

    for name, array in (
        ("supervoxels", supervoxels),
        ("instances", instances),
        ("foreground", foreground),
    ):
        if array.shape != expected:
            raise AnnotationError(
                f"Shape mismatch: raw={expected}, {name}={array.shape}."
            )

    if np.any(supervoxels < 0):
        raise AnnotationError("Supervoxel IDs must be non-negative.")

    if np.any(instances < 0):
        raise AnnotationError("Instance IDs must be non-negative.")


# ============================================================
# SUPERVOXEL -> LABEL ANCHOR PLACEMENT
# ============================================================


def _dominant_parent_instance(
    sv_frame: np.ndarray,
    instance_frame: np.ndarray,
    sv_id: int,
) -> int:
    mask = sv_frame == sv_id
    values, counts = np.unique(instance_frame[mask], return_counts=True)

    nonzero = values > 0
    if not np.any(nonzero):
        return 0

    values = values[nonzero]
    counts = counts[nonzero]
    return int(values[np.argmax(counts)])


def _centroid_2d(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise AnnotationError("Cannot compute centroid of an empty mask.")
    return coords.mean(axis=0).astype(np.float64)


def _fallback_nearest_background(
    foreground_2d: np.ndarray,
    source_yx: np.ndarray,
) -> np.ndarray:
    source = np.rint(source_yx).astype(int)
    source[0] = np.clip(source[0], 0, foreground_2d.shape[0] - 1)
    source[1] = np.clip(source[1], 0, foreground_2d.shape[1] - 1)

    if not foreground_2d[tuple(source)]:
        return source.astype(np.float64)

    # For every foreground pixel, scipy returns the coordinate of its nearest
    # zero/background pixel.
    _, nearest = ndimage.distance_transform_edt(
        foreground_2d,
        return_indices=True,
    )
    y = int(nearest[0, source[0], source[1]])
    x = int(nearest[1, source[0], source[1]])
    return np.array([y, x], dtype=np.float64)


def _ray_to_background(
    foreground_2d: np.ndarray,
    start_yx: np.ndarray,
    direction_yx: np.ndarray,
) -> np.ndarray:
    norm = float(np.linalg.norm(direction_yx))
    if norm < 1e-6:
        return _fallback_nearest_background(foreground_2d, start_yx)

    unit = direction_yx / norm
    h, w = foreground_2d.shape
    max_steps = int(math.ceil(math.hypot(h, w))) + 2

    last_inside = np.asarray(start_yx, dtype=np.float64)

    for step in range(max_steps):
        p = start_yx + unit * float(step)
        y, x = np.rint(p).astype(int)

        if y < 0 or y >= h or x < 0 or x >= w:
            break

        last_inside = p
        if not foreground_2d[y, x]:
            candidate = p + unit * LABEL_SURFACE_OFFSET_PX
            candidate[0] = np.clip(candidate[0], 0, h - 1)
            candidate[1] = np.clip(candidate[1], 0, w - 1)

            cy, cx = np.rint(candidate).astype(int)
            if not foreground_2d[cy, cx]:
                return candidate

            return p

    return _fallback_nearest_background(foreground_2d, last_inside)


def _reposition_to_avoid_overlap(
    candidate_yx: np.ndarray,
    foreground_2d: np.ndarray,
    used_yx: list[np.ndarray],
) -> np.ndarray:
    if not used_yx:
        return candidate_yx

    def valid(point: np.ndarray) -> bool:
        y, x = np.rint(point).astype(int)
        if y < 0 or y >= foreground_2d.shape[0]:
            return False
        if x < 0 or x >= foreground_2d.shape[1]:
            return False
        if foreground_2d[y, x]:
            return False
        return all(
            np.linalg.norm(point - previous) >= LABEL_MIN_SEPARATION_PX
            for previous in used_yx
        )

    if valid(candidate_yx):
        return candidate_yx

    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)

    for radius in range(2, LABEL_REPOSITION_RADIUS_PX + 1, 2):
        for angle in angles:
            offset = radius * np.array(
                [math.sin(angle), math.cos(angle)],
                dtype=np.float64,
            )
            point = candidate_yx + offset
            if valid(point):
                return point

    return candidate_yx


def compute_supervoxel_label_anchor(
    sv_frame: np.ndarray,
    instance_frame: np.ndarray,
    foreground_frame: np.ndarray,
    sv_id: int,
    used_by_z: dict[int, list[np.ndarray]],
) -> tuple[float, float, float, float, float]:
    """Return z, outside-label-y/x, and same-slice SV-target-y/x."""
    mask_3d = sv_frame == sv_id
    z_counts = mask_3d.reshape(mask_3d.shape[0], -1).sum(axis=1)

    if z_counts.max() <= 0:
        raise AnnotationError(f"Supervoxel {sv_id} is empty.")

    # Put both the text and leader line on the z slice where this supervoxel
    # has its largest visible cross-section.
    z = int(np.argmax(z_counts))
    sv_2d = mask_3d[z]
    sv_center = _centroid_2d(sv_2d)

    parent_id = _dominant_parent_instance(
        sv_frame,
        instance_frame,
        sv_id,
    )

    if parent_id > 0:
        parent_2d = instance_frame[z] == parent_id
        if np.any(parent_2d):
            parent_center = _centroid_2d(parent_2d)
        else:
            parent_center = sv_center
    else:
        parent_center = sv_center

    direction = sv_center - parent_center

    # A central SV can have an almost-zero radial direction. Give it a stable
    # direction based on its ID, then cast toward background.
    if np.linalg.norm(direction) < 1e-4:
        golden_angle = 2.399963229728653
        angle = float(sv_id) * golden_angle
        direction = np.array(
            [math.sin(angle), math.cos(angle)],
            dtype=np.float64,
        )

    candidate = _ray_to_background(
        foreground_frame[z],
        sv_center,
        direction,
    )

    used = used_by_z.setdefault(z, [])
    candidate = _reposition_to_avoid_overlap(
        candidate,
        foreground_frame[z],
        used,
    )
    used.append(candidate.copy())

    return (
        float(z),
        float(candidate[0]),
        float(candidate[1]),
        float(sv_center[0]),
        float(sv_center[1]),
    )


def build_supervoxel_text_points(
    supervoxels: np.ndarray,
    instances: np.ndarray,
    foreground: np.ndarray,
    timepoints: tuple[int, ...],
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[np.ndarray],
]:
    points: list[tuple[float, float, float, float]] = []
    leader_lines: list[np.ndarray] = []
    sv_ids: list[int] = []
    dataset_times: list[int] = []

    for local_t, dataset_t in enumerate(timepoints):
        frame = supervoxels[local_t]
        ids = np.unique(frame)
        ids = ids[ids > 0]

        print(
            f"[labels] t={dataset_t}: computing outside-surface positions for "
            f"{len(ids)} supervoxels"
        )

        used_by_z: dict[int, list[np.ndarray]] = {}

        for sv_id in ids.tolist():
            (
                z,
                label_y,
                label_x,
                target_y,
                target_x,
            ) = compute_supervoxel_label_anchor(
                frame,
                instances[local_t],
                foreground[local_t],
                int(sv_id),
                used_by_z,
            )

            points.append(
                (
                    float(local_t),
                    z,
                    label_y,
                    label_x,
                )
            )

            # Each leader line stays inside one t/z plane:
            #
            # outside number --------> actual supervoxel center
            #
            # This makes it visually unambiguous which number belongs to which
            # atomic region.
            leader_lines.append(
                np.asarray(
                    [
                        [
                            float(local_t),
                            z,
                            label_y,
                            label_x,
                        ],
                        [
                            float(local_t),
                            z,
                            target_y,
                            target_x,
                        ],
                    ],
                    dtype=np.float32,
                )
            )

            sv_ids.append(int(sv_id))
            dataset_times.append(int(dataset_t))

    point_array = np.asarray(points, dtype=np.float32)
    if point_array.size == 0:
        point_array = np.zeros((0, 4), dtype=np.float32)

    properties = {
        "sv_id": np.asarray(sv_ids, dtype=np.int64),
        "dataset_t": np.asarray(dataset_times, dtype=np.int64),
    }

    return point_array, properties, leader_lines


def _supervoxel_contact_graph(
    sv_frame: np.ndarray,
    allowed_ids: set[int],
) -> dict[int, dict[int, float]]:
    """
    Build a 6-neighbour graph for the allowed supervoxels.

    Edge weight stored in the graph is physical contact area in um^2.
    """
    graph: dict[int, dict[int, float]] = {
        int(sv_id): {} for sv_id in allowed_ids
    }

    if len(allowed_ids) <= 1:
        return graph

    max_sv = int(np.max(sv_frame)) if sv_frame.size else 0
    allowed_lookup = np.zeros(max_sv + 1, dtype=bool)

    valid_ids = np.asarray(
        sorted(sv_id for sv_id in allowed_ids if 0 < sv_id <= max_sv),
        dtype=np.int64,
    )
    allowed_lookup[valid_ids] = True

    sz, sy, sx = DEFAULT_SPACING_ZYX_UM
    face_area_by_axis = (
        sy * sx,  # z-neighbour face has Y*X area
        sz * sx,  # y-neighbour face has Z*X area
        sz * sy,  # x-neighbour face has Z*Y area
    )

    for axis, face_area in enumerate(face_area_by_axis):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)

        a = sv_frame[tuple(left)].astype(np.int64, copy=False)
        b = sv_frame[tuple(right)].astype(np.int64, copy=False)

        valid = (
            (a > 0)
            & (b > 0)
            & (a != b)
            & allowed_lookup[a]
            & allowed_lookup[b]
        )

        if not np.any(valid):
            continue

        aa = a[valid]
        bb = b[valid]

        low = np.minimum(aa, bb)
        high = np.maximum(aa, bb)
        pairs = np.stack([low, high], axis=1)

        unique_pairs, counts = np.unique(
            pairs,
            axis=0,
            return_counts=True,
        )

        for (sv_a, sv_b), count in zip(
            unique_pairs.tolist(),
            counts.tolist(),
        ):
            sv_a = int(sv_a)
            sv_b = int(sv_b)
            area = float(count) * float(face_area)

            graph[sv_a][sv_b] = graph[sv_a].get(sv_b, 0.0) + area
            graph[sv_b][sv_a] = graph[sv_b].get(sv_a, 0.0) + area

    return graph


def _expand_seed_groups_by_contact_graph(
    *,
    sv_frame: np.ndarray,
    parent_supervoxels: set[int],
    seed_groups: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """
    Assign every SV in one merged instance to one seed group.

    We perform multi-source Dijkstra on the SV contact graph. Traversing a broad
    contact is cheap and traversing a narrow contact is expensive:

        traversal_cost = 1 / physical_contact_area

    Therefore the eventual group boundary tends to fall on narrow contact necks.
    """
    graph = _supervoxel_contact_graph(
        sv_frame,
        parent_supervoxels,
    )

    # best[sv] = (distance, group_index)
    best: dict[int, tuple[float, int]] = {}
    queue: list[tuple[float, int, int]] = []

    for group_index, seeds in enumerate(seed_groups):
        for sv_id in seeds:
            best[int(sv_id)] = (0.0, group_index)
            heapq.heappush(
                queue,
                (0.0, group_index, int(sv_id)),
            )

    while queue:
        distance, group_index, sv_id = heapq.heappop(queue)

        current = best.get(sv_id)
        if current is None:
            continue

        current_distance, current_group = current
        if (
            distance > current_distance + 1e-12
            or group_index != current_group
        ):
            continue

        for neighbour, contact_area in graph[sv_id].items():
            # Broad intra-cell contacts are preferred; narrow necks are costly.
            edge_cost = 1.0 / max(float(contact_area), 1e-12)
            candidate_distance = distance + edge_cost

            previous = best.get(neighbour)

            should_update = (
                previous is None
                or candidate_distance < previous[0] - 1e-12
                or (
                    abs(candidate_distance - previous[0]) <= 1e-12
                    and group_index < previous[1]
                )
            )

            if should_update:
                best[neighbour] = (
                    candidate_distance,
                    group_index,
                )
                heapq.heappush(
                    queue,
                    (
                        candidate_distance,
                        group_index,
                        neighbour,
                    ),
                )

    unreachable = sorted(
        int(sv_id)
        for sv_id in parent_supervoxels
        if int(sv_id) not in best
    )

    if unreachable:
        preview = ", ".join(str(v) for v in unreachable[:30])
        suffix = " ..." if len(unreachable) > 30 else ""
        raise AnnotationError(
            "The current predicted instance is not fully connected in the "
            "6-neighbour supervoxel contact graph. Unreachable SVs: "
            f"{preview}{suffix}"
        )

    expanded: list[list[int]] = [
        [] for _ in seed_groups
    ]

    for sv_id in sorted(parent_supervoxels):
        _, group_index = best[int(sv_id)]
        expanded[group_index].append(int(sv_id))

    # Every seed group necessarily remains non-empty, but validate anyway.
    if any(len(group) == 0 for group in expanded):
        raise AnnotationError(
            "Internal error: automatic seed expansion produced an empty cell."
        )

    return tuple(
        tuple(group)
        for group in expanded
    )


# ============================================================
# ANNOTATION SESSION
# ============================================================


class AnnotationSession:
    def __init__(
        self,
        *,
        sample_id: str,
        timepoints: tuple[int, ...],
        supervoxels: np.ndarray,
        base_instances: np.ndarray,
        output_dir: Path,
        resume: bool,
    ) -> None:
        self.sample_id = sample_id
        self.timepoints = timepoints
        self.supervoxels = np.asarray(supervoxels)
        self.base_instances = np.asarray(base_instances).astype(
            np.int32,
            copy=False,
        )
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.corrected = self.base_instances.copy()
        self.corrections: list[dict] = []
        self.corrected_original_ids: dict[int, set[int]] = {
            i: set() for i in range(len(timepoints))
        }

        self.log_path = self.output_dir / "supervoxel_split_corrections.json"

        if resume:
            self._resume_existing()

        max_label = int(self.corrected.max(initial=0))
        self.next_label = max_label + 1

    def output_path_for_timepoint(self, dataset_t: int) -> Path:
        return self.output_dir / f"manual_instances_t{dataset_t:03d}.npy"

    def _resume_existing(self) -> None:
        for local_t, dataset_t in enumerate(self.timepoints):
            path = self.output_path_for_timepoint(dataset_t)
            if not path.exists():
                continue

            existing = np.load(path)
            if existing.shape != self.corrected[local_t].shape:
                raise AnnotationError(
                    f"Cannot resume {path}: expected "
                    f"{self.corrected[local_t].shape}, got {existing.shape}."
                )
            self.corrected[local_t] = existing.astype(np.int32, copy=False)
            print(f"[resume] loaded {path}")

        if not self.log_path.exists():
            return

        with self.log_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if payload.get("sample_id") != self.sample_id:
            raise AnnotationError(
                f"Existing log belongs to sample {payload.get('sample_id')!r}, "
                f"not {self.sample_id!r}."
            )

        existing_timepoints = tuple(int(v) for v in payload.get("timepoints", []))
        if existing_timepoints and existing_timepoints != self.timepoints:
            print(
                "[resume] existing log uses a different timepoint selection; "
                "loading only corrections whose timepoints are in this session."
            )

        self.corrections = list(payload.get("corrections", []))

        time_to_local = {t: i for i, t in enumerate(self.timepoints)}
        for correction in self.corrections:
            dataset_t = int(correction["timepoint"])
            if dataset_t not in time_to_local:
                continue
            local_t = time_to_local[dataset_t]
            self.corrected_original_ids[local_t].add(
                int(correction["original_instance_id"])
            )

    def _parent_instance_for_sv(self, local_t: int, sv_id: int) -> int:
        # Use the CURRENT corrected partition. This permits another split of a
        # previously generated child instance later in the same session.
        return _dominant_parent_instance(
            self.supervoxels[local_t],
            self.corrected[local_t],
            sv_id,
        )

    def _expected_supervoxels(
        self,
        local_t: int,
        original_instance_id: int,
    ) -> set[int]:
        mask = self.corrected[local_t] == original_instance_id

        if np.any(mask & (self.supervoxels[local_t] == 0)):
            missing_voxels = int(
                np.count_nonzero(mask & (self.supervoxels[local_t] == 0))
            )
            raise AnnotationError(
                f"Original instance {original_instance_id} contains "
                f"{missing_voxels} voxels with supervoxel ID 0. The split cannot "
                "be made losslessly from supervoxels; inspect the spatial export."
            )

        ids = np.unique(self.supervoxels[local_t][mask])
        return {int(v) for v in ids.tolist() if int(v) > 0}

    def apply_split(
        self,
        local_t: int,
        groups: list[list[int]],
    ) -> SplitResult:
        if not (0 <= local_t < len(self.timepoints)):
            raise AnnotationError(f"Invalid local frame index: {local_t}")

        seed_groups = tuple(
            tuple(int(v) for v in group)
            for group in groups
            if group
        )

        if len(seed_groups) < 2:
            raise AnnotationError(
                "At least two instance boxes must be filled before Save."
            )

        if len(seed_groups) > 4:
            raise AnnotationError(
                "At most four output instances are supported."
            )

        flat = [
            sv_id
            for group in seed_groups
            for sv_id in group
        ]

        if len(flat) != len(set(flat)):
            duplicates = sorted(
                sv_id
                for sv_id in set(flat)
                if flat.count(sv_id) > 1
            )
            raise AnnotationError(
                "A supervoxel cannot be used as a seed for two cells. "
                f"Duplicates: {duplicates}"
            )

        frame_ids = {
            int(v)
            for v in np.unique(self.supervoxels[local_t]).tolist()
            if int(v) > 0
        }

        missing = sorted(set(flat) - frame_ids)
        if missing:
            raise AnnotationError(
                "These supervoxels do not exist in the current frame: "
                f"{missing}"
            )

        parent_ids = {
            self._parent_instance_for_sv(
                local_t,
                sv_id,
            )
            for sv_id in flat
        }

        if 0 in parent_ids:
            raise AnnotationError(
                "At least one selected seed supervoxel is currently background "
                "rather than part of a spatial instance."
            )

        if len(parent_ids) != 1:
            raise AnnotationError(
                "The selected seed supervoxels are already in DIFFERENT current "
                "spatial instances, so there is no single merged instance to "
                "split between them. Current instance IDs: "
                f"{sorted(parent_ids)}. "
                "Use the red leader lines to choose seed SVs from the same "
                "merged colored instance."
            )

        original_instance_id = next(iter(parent_ids))

        parent_supervoxels = self._expected_supervoxels(
            local_t,
            original_instance_id,
        )

        # No completeness requirement: the user's entries are ONLY split seeds.
        # Automatically assign every remaining supervoxel in the current merged
        # instance using the weighted contact graph.
        expanded_groups = _expand_seed_groups_by_contact_graph(
            sv_frame=self.supervoxels[local_t],
            parent_supervoxels=parent_supervoxels,
            seed_groups=seed_groups,
        )

        original_mask = (
            self.corrected[local_t] == original_instance_id
        )

        # Clear only this current merged component, then fill it from the
        # automatically expanded groups.
        self.corrected[local_t][original_mask] = 0

        output_ids: list[int] = []
        group_records: list[dict] = []

        for seed_group, expanded_group in zip(
            seed_groups,
            expanded_groups,
        ):
            new_instance_id = int(self.next_label)
            self.next_label += 1

            group_mask = np.isin(
                self.supervoxels[local_t],
                np.asarray(
                    expanded_group,
                    dtype=self.supervoxels.dtype,
                ),
            ) & original_mask

            self.corrected[local_t][group_mask] = new_instance_id
            output_ids.append(new_instance_id)

            group_records.append(
                {
                    "output_instance_id": new_instance_id,
                    "seed_supervoxels": [
                        int(v) for v in seed_group
                    ],
                    "assigned_supervoxels": [
                        int(v) for v in expanded_group
                    ],
                }
            )

        if np.any(
            self.corrected[local_t][original_mask] == 0
        ):
            raise AnnotationError(
                "Internal error: automatic graph split left part of the "
                "original instance unassigned."
            )

        dataset_t = int(self.timepoints[local_t])

        record = {
            "timepoint": dataset_t,
            "original_instance_id": int(original_instance_id),
            "split_method": (
                "multi_source_dijkstra_inverse_physical_contact_area"
            ),
            "groups": group_records,
        }

        self.corrections.append(record)
        self.corrected_original_ids[local_t].add(
            int(original_instance_id)
        )

        self.persist()

        return SplitResult(
            timepoint=dataset_t,
            original_instance_id=int(original_instance_id),
            output_instance_ids=tuple(output_ids),
            seed_groups=seed_groups,
            groups=expanded_groups,
        )

    def _local_index_for_dataset_timepoint(
        self,
        dataset_t: int,
    ) -> int | None:
        try:
            return self.timepoints.index(int(dataset_t))
        except ValueError:
            return None

    def can_undo(self) -> bool:
        """
        True when this session contains at least one persisted correction for
        one of the currently loaded timepoints.
        """
        for correction in reversed(self.corrections):
            dataset_t = int(correction["timepoint"])
            if self._local_index_for_dataset_timepoint(dataset_t) is not None:
                return True
        return False

    def undo_last_split(self) -> UndoResult:
        """
        Reverse the newest correction belonging to a loaded timepoint.

        This is safe for nested edits because undo is LIFO. If an output of an
        earlier split was itself split later, that later split must be undone
        first, after which the earlier output label exists again.
        """
        correction_index: int | None = None
        local_t: int | None = None

        for index in range(len(self.corrections) - 1, -1, -1):
            correction = self.corrections[index]
            candidate_local_t = self._local_index_for_dataset_timepoint(
                int(correction["timepoint"])
            )
            if candidate_local_t is not None:
                correction_index = index
                local_t = candidate_local_t
                break

        if correction_index is None or local_t is None:
            raise AnnotationError(
                "There is no saved split operation to undo."
            )

        correction = self.corrections[correction_index]
        dataset_t = int(correction["timepoint"])
        original_instance_id = int(
            correction["original_instance_id"]
        )

        groups = list(correction.get("groups", []))
        output_ids = tuple(
            int(group["output_instance_id"])
            for group in groups
            if "output_instance_id" in group
        )

        if len(output_ids) < 2:
            raise AnnotationError(
                "The newest correction log entry does not contain enough "
                "output instance IDs to undo safely."
            )

        frame = self.corrected[local_t]

        # Since this is the newest correction affecting the loaded data, these
        # labels should still exist. Restore their entire union back to the
        # pre-split instance ID.
        changed_mask = np.isin(
            frame,
            np.asarray(output_ids, dtype=frame.dtype),
        )

        changed_voxels = int(np.count_nonzero(changed_mask))
        if changed_voxels == 0:
            raise AnnotationError(
                "Cannot undo the newest correction because none of its output "
                f"instance IDs {output_ids} are present in t={dataset_t}. "
                "The annotation files may have been modified outside this tool."
            )

        frame[changed_mask] = original_instance_id

        # Remove exactly the operation we reversed.
        self.corrections.pop(correction_index)

        # This set is only a UI/count bookkeeping structure. IDs created by a
        # parent split are globally unique, so discarding the restored parent
        # operation is safe in the normal LIFO workflow.
        self.corrected_original_ids[local_t].discard(
            original_instance_id
        )

        # Persist both raster labels and the shortened correction history.
        self.persist()

        return UndoResult(
            timepoint=dataset_t,
            original_instance_id=original_instance_id,
            removed_instance_ids=output_ids,
        )

    def persist(self) -> None:
        # Save a complete pseudo-GT instance volume for every selected frame.
        # Frames not manually changed remain equal to the strong spatial output.
        for local_t, dataset_t in enumerate(self.timepoints):
            path = self.output_path_for_timepoint(dataset_t)
            np.save(path, self.corrected[local_t].astype(np.int32, copy=False))

        payload = {
            "format_version": 1,
            "sample_id": self.sample_id,
            "timepoints": [int(t) for t in self.timepoints],
            "description": (
                "Base spatial instance labels with manually corrected merged "
                "instances using atomic-supervoxel grouping."
            ),
            "corrections": self.corrections,
        }

        with self.log_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def corrections_in_frame(self, local_t: int) -> int:
        return len(self.corrected_original_ids[local_t])


# ============================================================
# INPUT PARSING
# ============================================================


def parse_supervoxel_group(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []

    tokens = [part.strip() for part in text.split(",")]
    if any(token == "" for token in tokens):
        raise AnnotationError(
            f"Invalid comma-separated list: {text!r}. "
            "Example: 12, 15, 19"
        )

    values: list[int] = []
    for token in tokens:
        try:
            value = int(token)
        except ValueError as exc:
            raise AnnotationError(
                f"Supervoxel ID {token!r} is not an integer."
            ) from exc

        if value <= 0:
            raise AnnotationError(
                f"Supervoxel IDs must be positive; received {value}."
            )
        values.append(value)

    if len(values) != len(set(values)):
        raise AnnotationError(
            f"The same supervoxel appears twice in one box: {text!r}"
        )

    return values




# ============================================================
# ADJACENCY-AWARE LABEL GRAPH COLORING
# ============================================================

# A restrained but visually distinct starting palette. The graph-color index,
# not the raw label ID, selects the display color.
_GRAPH_COLOR_BASE_RGBA = (
    (0.90, 0.12, 0.12, 1.0),  # red
    (0.12, 0.36, 0.95, 1.0),  # blue
    (0.10, 0.74, 0.20, 1.0),  # green
    (0.92, 0.12, 0.72, 1.0),  # magenta
    (0.00, 0.74, 0.80, 1.0),  # cyan
    (0.96, 0.70, 0.05, 1.0),  # amber
    (0.54, 0.22, 0.86, 1.0),  # purple
    (0.98, 0.42, 0.05, 1.0),  # orange
    (0.42, 0.84, 0.06, 1.0),  # lime
    (0.96, 0.34, 0.55, 1.0),  # pink
    (0.18, 0.72, 0.54, 1.0),  # teal-green
    (0.43, 0.47, 0.96, 1.0),  # periwinkle
)


def _hsv_to_rgba(
    hue: float,
    saturation: float,
    value: float,
) -> tuple[float, float, float, float]:
    """Dependency-free HSV -> RGBA conversion."""
    hue = float(hue) % 1.0
    saturation = float(np.clip(saturation, 0.0, 1.0))
    value = float(np.clip(value, 0.0, 1.0))

    h6 = hue * 6.0
    sector = int(np.floor(h6)) % 6
    fraction = h6 - np.floor(h6)

    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))

    if sector == 0:
        r, g, b = value, t, p
    elif sector == 1:
        r, g, b = q, value, p
    elif sector == 2:
        r, g, b = p, value, t
    elif sector == 3:
        r, g, b = p, q, value
    elif sector == 4:
        r, g, b = t, p, value
    else:
        r, g, b = value, p, q

    return float(r), float(g), float(b), 1.0


def _display_color_for_unique_index(
    unique_index: int,
) -> tuple[float, float, float, float]:
    """
    Return a deterministic, non-repeating display color for one label rank.

    Consecutive indices are deliberately far apart in hue using golden-ratio
    stepping. That is particularly useful here because neighbouring watershed
    IDs are often numerically close.

    For the few hundred labels in these BioHub frames this produces a large
    practical palette without exact color reuse.
    """
    unique_index = int(unique_index)

    if unique_index < 0:
        raise ValueError(
            f"unique_index must be non-negative, got {unique_index}"
        )

    # Keep the first few colors maximally obvious.
    if unique_index < len(_GRAPH_COLOR_BASE_RGBA):
        return _GRAPH_COLOR_BASE_RGBA[unique_index]

    extra = unique_index - len(_GRAPH_COLOR_BASE_RGBA)

    golden_ratio_conjugate = 0.6180339887498949
    hue = (
        0.03
        + (extra + 1) * golden_ratio_conjugate
    ) % 1.0

    # Cycle saturation/value independently from hue. This increases separation
    # between colors whose hue eventually comes close after many labels.
    saturation_cycle = (
        0.82,
        0.68,
        0.92,
        0.74,
    )
    value_cycle = (
        0.98,
        0.86,
        0.94,
    )

    saturation = saturation_cycle[
        extra % len(saturation_cycle)
    ]
    value = value_cycle[
        (extra // len(saturation_cycle))
        % len(value_cycle)
    ]

    return _hsv_to_rgba(
        hue,
        saturation,
        value,
    )


def _build_frame_touch_adjacency(
    labels_zyx: np.ndarray,
) -> tuple[
    dict[int, set[int]],
    set[int],
]:
    """
    Build the 6-neighbour face-contact graph for ONE 3-D label volume.

    Two positive labels are adjacent iff at least one z/y/x voxel face separates
    them. Background 0 is excluded.
    """
    frame = np.asarray(labels_zyx)

    if frame.ndim != 3:
        raise ValueError(
            "Frame adjacency expects (Z,Y,X), got "
            f"{frame.shape}."
        )

    positive_ids = np.unique(frame)
    positive_ids = positive_ids[positive_ids > 0]

    all_labels = {
        int(value)
        for value in positive_ids.tolist()
    }
    adjacency: dict[int, set[int]] = {
        label_id: set()
        for label_id in all_labels
    }

    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)

        a = frame[tuple(left)]
        b = frame[tuple(right)]

        touching = (
            (a > 0)
            & (b > 0)
            & (a != b)
        )

        if not np.any(touching):
            continue

        aa = a[touching].astype(
            np.int64,
            copy=False,
        )
        bb = b[touching].astype(
            np.int64,
            copy=False,
        )

        pairs = np.stack(
            [
                np.minimum(aa, bb),
                np.maximum(aa, bb),
            ],
            axis=1,
        )

        # Thousands of voxel faces can represent the same graph edge.
        pairs = np.unique(
            pairs,
            axis=0,
        )

        for u, v in pairs.tolist():
            u = int(u)
            v = int(v)

            adjacency.setdefault(u, set()).add(v)
            adjacency.setdefault(v, set()).add(u)

            all_labels.add(u)
            all_labels.add(v)

    return adjacency, all_labels


def _build_frame_graph_cache(
    labels_tzyx: np.ndarray,
) -> list[
    tuple[
        dict[int, set[int]],
        set[int],
    ]
]:
    """
    Cache one contact graph per selected timepoint.

    This intentionally does NOT connect labels between t and t+1.
    """
    data = np.asarray(labels_tzyx)

    if data.ndim == 3:
        return [
            _build_frame_touch_adjacency(data)
        ]

    if data.ndim != 4:
        raise ValueError(
            "Graph-coloring expects (Z,Y,X) or (T,Z,Y,X), got "
            f"{data.shape}."
        )

    return [
        _build_frame_touch_adjacency(
            data[t]
        )
        for t in range(data.shape[0])
    ]


def _merge_frame_graph_cache(
    frame_graphs: list[
        tuple[
            dict[int, set[int]],
            set[int],
        ]
    ],
) -> tuple[
    dict[int, set[int]],
    set[int],
]:
    """
    Merge per-frame graphs by LABEL VALUE.

    Napari Labels colors are keyed by label value, not by (time,label), so if
    values 12 and 19 touch in any selected frame they must receive different
    global display colors. This union graph guarantees that.
    """
    merged_adjacency: dict[int, set[int]] = {}
    all_labels: set[int] = set()

    for adjacency, labels in frame_graphs:
        all_labels.update(
            int(value)
            for value in labels
        )

        for node, neighbours in adjacency.items():
            target = merged_adjacency.setdefault(
                int(node),
                set(),
            )
            target.update(
                int(value)
                for value in neighbours
            )

    for label_id in all_labels:
        merged_adjacency.setdefault(
            int(label_id),
            set(),
        )

    return merged_adjacency, all_labels


def _greedy_graph_coloring(
    adjacency: dict[int, set[int]],
    all_labels: set[int],
) -> dict[int, int]:
    """
    Deterministic largest-degree-first greedy graph coloring.

    Guarantee:
        if u-v is an adjacency edge, color[u] != color[v]
    """
    assigned: dict[int, int] = {}

    nodes = sorted(
        all_labels,
        key=lambda node: (
            -len(
                adjacency.get(
                    int(node),
                    set(),
                )
            ),
            int(node),
        ),
    )

    for node in nodes:
        used = {
            assigned[neighbour]
            for neighbour in adjacency.get(
                int(node),
                set(),
            )
            if neighbour in assigned
        }

        candidate = 0
        while candidate in used:
            candidate += 1

        assigned[int(node)] = int(candidate)

    return assigned


def _color_dict_from_frame_graph_cache(
    frame_graphs: list[
        tuple[
            dict[int, set[int]],
            set[int],
        ]
    ],
) -> tuple[
    dict[int, tuple[float, float, float, float]],
    dict[str, int],
]:
    """
    Build a UNIQUE label-value -> RGBA mapping.

    The frame contact graphs are still merged for diagnostics, but unlike the
    previous graph-coloring implementation we never reuse a display color
    between two different positive label IDs.
    """
    adjacency, all_labels = (
        _merge_frame_graph_cache(
            frame_graphs
        )
    )

    ordered_labels = sorted(
        int(label_id)
        for label_id in all_labels
    )

    color_dict: dict[
        int,
        tuple[float, float, float, float],
    ] = {
        0: (0.0, 0.0, 0.0, 0.0),
    }

    for unique_index, label_id in enumerate(
        ordered_labels
    ):
        color_dict[int(label_id)] = (
            _display_color_for_unique_index(
                int(unique_index)
            )
        )

    edge_count = (
        sum(
            len(neighbours)
            for neighbours in adjacency.values()
        )
        // 2
    )

    # Defensive invariant: every touching pair must have different RGBA values.
    # This is now implied by unique-per-label coloring, but checking it here
    # protects future modifications to the color generator.
    for node, neighbours in adjacency.items():
        node_color = color_dict[int(node)]

        for neighbour in neighbours:
            if node_color == color_dict[int(neighbour)]:
                raise RuntimeError(
                    "Unique display-color invariant failed for touching "
                    f"labels {node} and {neighbour}."
                )

    stats = {
        "label_count": int(len(ordered_labels)),
        "touch_edge_count": int(edge_count),
        "color_count": int(len(ordered_labels)),
    }

    return color_dict, stats


def _apply_label_color_dict(
    layer,
    color_dict: dict[
        int,
        tuple[float, float, float, float],
    ],
) -> None:
    """
    Update a Napari Labels layer's explicit label -> RGBA mapping.

    Napari's API differs across versions:
        newer: layer.color = mapping
        older: layer.color_mode / internal direct-color machinery may be needed

    Try public APIs first and only then fall back to the layer's direct-colormap
    interface if exposed.
    """
    first_error = None

    try:
        layer.color = color_dict
        layer.refresh()
        return
    except Exception as exc:
        first_error = exc

    # Some versions expose a setter through the property but require direct
    # color mode before assigning the mapping.
    try:
        if hasattr(layer, "color_mode"):
            try:
                layer.color_mode = "direct"
            except Exception:
                pass

        layer.color = color_dict
        layer.refresh()
        return
    except Exception:
        pass

    # Older Labels implementations may expose `_direct_colormap` or
    # `direct_colormap`. We avoid assuming one exact class/API shape.
    for attr_name in (
        "direct_colormap",
        "_direct_colormap",
    ):
        if not hasattr(layer, attr_name):
            continue

        try:
            colormap = getattr(layer, attr_name)

            if hasattr(colormap, "color_dict"):
                colormap.color_dict = color_dict
                layer.refresh()
                return

            if hasattr(colormap, "colors"):
                colormap.colors = color_dict
                layer.refresh()
                return
        except Exception:
            continue

    raise RuntimeError(
        "This Napari version did not accept the adjacency-aware Labels color "
        "mapping after layer creation either. "
        f"Original error: {first_error}"
    )


# ============================================================
# 3-D RAY PICKING
# ============================================================


def _coerce_positive_label_value(value) -> int:
    """
    Convert Napari Labels.get_value() output to one positive integer label.

    Most Napari versions return a scalar label for Labels. We deliberately
    reject ambiguous multi-value outputs and fall back to explicit ray
    traversal instead of guessing.
    """
    if value is None:
        return 0

    array = np.asarray(value)

    if array.ndim == 0:
        try:
            result = int(array.item())
        except (TypeError, ValueError, OverflowError):
            return 0
        return result if result > 0 else 0

    if array.size == 1:
        try:
            result = int(array.reshape(-1)[0])
        except (TypeError, ValueError, OverflowError):
            return 0
        return result if result > 0 else 0

    return 0


def _first_nonzero_label_along_data_ray(
    labels: np.ndarray,
    start_point: np.ndarray,
    end_point: np.ndarray,
    *,
    samples_per_voxel: float = 4.0,
) -> int:
    """
    Return the first positive label encountered from start_point -> end_point.

    start_point is the camera-near intersection supplied by Napari, so traversal
    order directly implements "frontmost visible cell".

    The points are in full n-D layer data coordinates. Sampling at 4 samples per
    voxel along the largest-changing dimension is intentionally conservative for
    label picking while still requiring only ~10^3 samples for these volumes.
    """
    data = np.asarray(labels)
    start = np.asarray(start_point, dtype=np.float64).reshape(-1)
    end = np.asarray(end_point, dtype=np.float64).reshape(-1)

    if start.shape != end.shape:
        return 0
    if start.size != data.ndim:
        return 0
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        return 0

    delta = end - start
    max_axis_distance = float(np.max(np.abs(delta)))

    sample_count = max(
        2,
        int(np.ceil(max_axis_distance * float(samples_per_voxel))) + 1,
    )

    shape = np.asarray(data.shape, dtype=np.int64)

    for alpha in np.linspace(
        0.0,
        1.0,
        sample_count,
        endpoint=True,
        dtype=np.float64,
    ):
        point = start + alpha * delta
        index = np.rint(point).astype(np.int64)

        if np.any(index < 0) or np.any(index >= shape):
            continue

        value = int(data[tuple(index.tolist())])
        if value > 0:
            return value

    return 0


def _ray_pick_frontmost_label(
    layer,
    event,
) -> int:
    """
    Pick the frontmost positive Labels value under a Napari mouse event.

    Preferred path:
        Labels.get_value(... view_direction ...)

    Fallback:
        get_ray_intersections() + explicit front-to-back label traversal.
    """
    view_direction = getattr(event, "view_direction", None)
    dims_displayed = getattr(event, "dims_displayed", None)

    # In true 3-D, ask Napari for its native ray-aware top value first.
    if view_direction is not None and dims_displayed is not None:
        try:
            value = layer.get_value(
                event.position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=True,
            )
            label_id = _coerce_positive_label_value(value)
            if label_id > 0:
                return label_id
        except Exception:
            # Fall through to explicit ray traversal.
            pass

        try:
            start_point, end_point = layer.get_ray_intersections(
                position=event.position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=True,
            )
        except TypeError:
            # Compatibility with versions accepting positional parameters.
            try:
                start_point, end_point = layer.get_ray_intersections(
                    event.position,
                    view_direction,
                    dims_displayed,
                    world=True,
                )
            except Exception:
                start_point, end_point = None, None
        except Exception:
            start_point, end_point = None, None

        if start_point is not None and end_point is not None:
            return _first_nonzero_label_along_data_ray(
                np.asarray(layer.data),
                np.asarray(start_point),
                np.asarray(end_point),
            )

    # 2-D compatibility path. It is not the main workflow, but clicking still
    # selects the label directly under the cursor if the user switches ndisplay.
    try:
        value = layer.get_value(
            event.position,
            world=True,
        )
        return _coerce_positive_label_value(value)
    except Exception:
        return 0


# ============================================================
# NAPARI UI
# ============================================================


def _make_magicgui_label_horizontally_shrinkable(widget) -> None:
    """Prevent long QLabel text from forcing the entire dock width."""
    native = getattr(widget, "native", None)
    if native is None:
        return

    try:
        native.setWordWrap(True)
    except Exception:
        pass

    try:
        native.setMinimumWidth(0)
    except Exception:
        pass

    if QSizePolicy is not None:
        try:
            native.setSizePolicy(
                QSizePolicy.Ignored,
                QSizePolicy.Preferred,
            )
        except Exception:
            pass


def _configure_resizable_annotation_dock(dock_widget, panel) -> None:
    """
    Keep the annotation panel horizontally resizable.

    The critical part is allowing child widgets, especially long status/error
    labels, to shrink. Otherwise Qt's sizeHint can make the dock appear locked
    at a huge width after one long error message.
    """
    panel_native = getattr(panel, "native", None)

    if panel_native is not None:
        try:
            panel_native.setMinimumWidth(260)
        except Exception:
            pass

        if QSizePolicy is not None:
            try:
                panel_native.setSizePolicy(
                    QSizePolicy.Preferred,
                    QSizePolicy.Expanding,
                )
            except Exception:
                pass

    # Napari returns a Qt dock widget in current versions. Keep only a modest
    # minimum width and no artificial maximum width. The divider between the
    # canvas and dock can then be dragged left/right normally.
    if dock_widget is not None:
        try:
            dock_widget.setMinimumWidth(280)
        except Exception:
            pass

        try:
            dock_widget.setMaximumWidth(16_777_215)
        except Exception:
            pass

        try:
            if Qt is not None and hasattr(dock_widget, "setFeatures"):
                features = dock_widget.features()
                dock_widget.setFeatures(features)
        except Exception:
            pass


def make_viewer(
    *,
    sample_id: str,
    timepoints: tuple[int, ...],
    raw: np.ndarray,
    stage6_binary_mask: np.ndarray,
    supervoxels: np.ndarray,
    foreground: np.ndarray,
    session: AnnotationSession,
    suspect_instances: np.ndarray | None = None,
) -> napari.Viewer:
    print()
    print("=" * 72)
    print("Building supervoxel text positions")
    print("=" * 72)

    (
        text_points,
        text_properties,
        leader_lines,
    ) = build_supervoxel_text_points(
        supervoxels,
        session.base_instances,
        foreground,
        timepoints,
    )

    viewer = napari.Viewer(ndisplay=2)
    scale_4d = (1.0, *DEFAULT_SPACING_ZYX_UM)

    print()
    print("=" * 72)
    print("Computing unique per-label display colors")
    print("=" * 72)

    supervoxel_frame_graphs = (
        _build_frame_graph_cache(
            supervoxels
        )
    )
    instance_frame_graphs = (
        _build_frame_graph_cache(
            session.corrected
        )
    )

    (
        supervoxel_color_dict,
        supervoxel_color_stats,
    ) = _color_dict_from_frame_graph_cache(
        supervoxel_frame_graphs
    )

    (
        instance_color_dict,
        instance_color_stats,
    ) = _color_dict_from_frame_graph_cache(
        instance_frame_graphs
    )

    print(
        "[display colors] supervoxels: "
        f"{supervoxel_color_stats['label_count']} label values | "
        f"{supervoxel_color_stats['touch_edge_count']} touching pairs | "
        f"{supervoxel_color_stats['color_count']} unique colors"
    )
    print(
        "[display colors] instances: "
        f"{instance_color_stats['label_count']} label values | "
        f"{instance_color_stats['touch_edge_count']} touching pairs | "
        f"{instance_color_stats['color_count']} unique colors"
    )

    viewer.dims.axis_labels = (
        "annotation frame",
        "z",
        "y",
        "x",
    )

    # ----------------------------------------
    # Raw image
    # ----------------------------------------
    viewer.add_image(
        raw,
        name="Raw BioHub",
        scale=scale_4d,
        colormap="gray",
    )

    # ----------------------------------------
    # Stage-6 binary foreground mask
    # ----------------------------------------
    viewer.add_labels(
        stage6_binary_mask,
        name="Stage-6 binary mask",
        scale=scale_4d,
        opacity=0.30,
        visible=False,
    )

    # ----------------------------------------
    # Current corrected pseudo-GT instances
    # ----------------------------------------
    corrected_layer = viewer.add_labels(
        session.corrected,
        name="Corrected instances",
        scale=scale_4d,
        opacity=1.0,
    )

    # Older Napari versions do not accept color= in viewer.add_labels(), but
    # they can still accept the explicit label->RGBA mapping on the created
    # Labels layer. Apply it after construction for compatibility.
    _apply_label_color_dict(
        corrected_layer,
        instance_color_dict,
    )

    # ----------------------------------------
    # Merge-suspect predicted instances
    # ----------------------------------------
    # Static read-only visualization of the ORIGINAL prediction. Save/Undo do
    # not mutate this layer or any existing annotation state.
    if suspect_instances is not None:
        suspect_layer = viewer.add_labels(
            suspect_instances,
            name="Suspect predicted instances",
            scale=scale_4d,
            opacity=1.0,
            visible=False,
        )
        _apply_label_color_dict(
            suspect_layer,
            instance_color_dict,
        )

    # ----------------------------------------
    # Ray-picked seed supervoxel highlights
    # ----------------------------------------
    #
    # One lightweight 3-D mask layer per seed slot gives an unambiguous visual
    # mapping between click order and input box:
    #
    #   Instance 1 -> red
    #   Instance 2 -> blue
    #   Instance 3 -> green
    #   Instance 4 -> magenta
    #
    # These are current-frame-only masks, not another full 20-frame stack.
    seed_highlight_layers = []

    for slot_index, (layer_name, colormap) in enumerate(
        (
            ("Seed 1 highlight", "red"),
            ("Seed 2 highlight", "blue"),
            ("Seed 3 highlight", "green"),
            ("Seed 4 highlight", "magenta"),
        ),
        start=1,
    ):
        seed_layer = viewer.add_image(
            np.zeros(
                session.corrected.shape[1:],
                dtype=np.uint8,
            ),
            name=layer_name,
            scale=DEFAULT_SPACING_ZYX_UM,
            colormap=colormap,
            contrast_limits=(0, 1),
            opacity=0.78,
            blending="additive",
            visible=True,
        )
        seed_highlight_layers.append(seed_layer)

    # ----------------------------------------
    # Atomic supervoxel boundaries
    # ----------------------------------------
    supervoxel_layer = viewer.add_labels(
        supervoxels,
        name="Atomic supervoxel boundaries",
        scale=scale_4d,
        opacity=0.95,
    )

    _apply_label_color_dict(
        supervoxel_layer,
        supervoxel_color_dict,
    )

    try:
        supervoxel_layer.contour = SUPERVOXEL_CONTOUR_WIDTH
    except Exception:
        # Older Napari versions may not expose contour on Labels.
        supervoxel_layer.opacity = 0.25
        print(
            "[napari] Labels.contour is unavailable in this version; "
            "showing translucent supervoxel fills instead."
        )

    # ----------------------------------------
    # Supervoxel ID text
    # ----------------------------------------
    text_spec = {
        "string": "{sv_id}",
        "size": 11,
        "color": "white",
        "anchor": "center",
    }

    # `properties` is supported by old and current Napari versions and is enough
    # for the text-format placeholder.
    text_layer = viewer.add_points(
        text_points,
        ndim=4,
        name="Supervoxel IDs",
        properties=text_properties,
        text=text_spec,
        scale=scale_4d,
        size=1,
        face_color="transparent",
    )

    try:
        text_layer.out_of_slice_display = False
    except Exception:
        pass

    # Red leader lines make the outside numbering unambiguous.
    if leader_lines:
        viewer.add_shapes(
            leader_lines,
            shape_type="line",
            name="SV number leader lines",
            scale=scale_4d,
            edge_color="red",
            edge_width=1.5,
            opacity=0.85,
        )

    # ----------------------------------------
    # Right-side annotation controls
    # ----------------------------------------
    current_frame_label = Label(value="")
    picked_instance_label = Label(
        value="Selected seed supervoxels: none"
    )
    instruction_label = Label(
        value=(
            "Every different SV/instance ID gets its own display color.\n"
            "Single-click visible supervoxels to add split seeds.\n"
            "1st click -> Instance 1; 2nd -> Instance 2.\n"
            "Further clicks use Instance 3/4 if needed.\n"
            "Unentered SVs are assigned automatically."
        )
    )

    box1 = LineEdit(
        label="Instance 1",
    )
    box2 = LineEdit(
        label="Instance 2",
    )
    box3 = LineEdit(
        label="Instance 3",
    )
    box4 = LineEdit(
        label="Instance 4",
    )

    save_button = PushButton(text="Save")
    reset_button = PushButton(text="Reset selections")
    undo_button = PushButton(text="Undo last Save")
    status_label = Label(
        value=(
            "Ready. Click two visible SVs, then Save. Esc resets; Ctrl+Z undoes."
        )
    )

    # Long errors/status messages must wrap instead of expanding the panel's
    # minimum width. This keeps the dock divider freely draggable left/right.
    for label_widget in (
        current_frame_label,
        picked_instance_label,
        instruction_label,
        status_label,
    ):
        _make_magicgui_label_horizontally_shrinkable(
            label_widget
        )

    panel = Container(
        widgets=[
            current_frame_label,
            picked_instance_label,
            instruction_label,
            box1,
            box2,
            box3,
            box4,
            save_button,
            reset_button,
            undo_button,
            status_label,
        ],
        layout="vertical",
        labels=True,
    )

    annotation_dock = viewer.window.add_dock_widget(
        panel,
        area="right",
        name="Merged-cell correction",
    )

    _configure_resizable_annotation_dock(
        annotation_dock,
        panel,
    )

    boxes = (box1, box2, box3, box4)

    def current_local_t() -> int:
        return int(round(viewer.dims.current_step[0]))

    def refresh_instance_graph_colors(
        changed_local_t: int,
    ) -> None:
        """
        Rebuild only the changed frame's raster contact graph, then recolor the
        union graph across all selected frames.
        """
        instance_frame_graphs[changed_local_t] = (
            _build_frame_touch_adjacency(
                session.corrected[
                    changed_local_t
                ]
            )
        )

        color_dict, stats = (
            _color_dict_from_frame_graph_cache(
                instance_frame_graphs
            )
        )

        _apply_label_color_dict(
            corrected_layer,
            color_dict,
        )

        print(
            "[display colors] refreshed instances: "
            f"{stats['label_count']} label values | "
            f"{stats['touch_edge_count']} touching pairs | "
            f"{stats['color_count']} unique colors"
        )

    selected_seed_ids: list[int | None] = [
        None,
        None,
        None,
        None,
    ]

    def _refresh_seed_status_label() -> None:
        parts = []
        for slot_index, sv_id in enumerate(
            selected_seed_ids,
            start=1,
        ):
            if sv_id is not None:
                parts.append(
                    f"{slot_index}: SV {int(sv_id)}"
                )

        picked_instance_label.value = (
            "Selected seed supervoxels: "
            + (
                ", ".join(parts)
                if parts
                else "none"
            )
        )

    def _clear_seed_highlights() -> None:
        for layer in seed_highlight_layers:
            layer.data = np.zeros(
                session.corrected.shape[1:],
                dtype=np.uint8,
            )
            layer.refresh()

    def clear_ray_selection() -> None:
        # Kept under the previous helper name so Save/Undo/time-change call
        # sites continue to reset the click-selection state.
        for index in range(4):
            selected_seed_ids[index] = None

        _clear_seed_highlights()
        _refresh_seed_status_label()

    def show_seed_selection(
        local_t: int,
        slot_index: int,
        sv_id: int,
    ) -> None:
        if not (0 <= slot_index < 4):
            raise AnnotationError(
                f"Invalid seed slot index: {slot_index}"
            )

        sv_mask = (
            supervoxels[local_t] == int(sv_id)
        )

        if not np.any(sv_mask):
            raise AnnotationError(
                f"Supervoxel {sv_id} is absent from the current frame."
            )

        selected_seed_ids[slot_index] = int(sv_id)

        layer = seed_highlight_layers[slot_index]
        layer.data = sv_mask.astype(
            np.uint8,
            copy=False,
        )
        layer.refresh()

        _refresh_seed_status_label()

    def next_empty_seed_slot() -> int | None:
        for slot_index, sv_id in enumerate(
            selected_seed_ids
        ):
            if sv_id is None:
                return slot_index
        return None

    def clear_boxes() -> None:
        for box in boxes:
            box.value = ""


    def reset_selections(
        *,
        message: str | None = None,
    ) -> None:
        clear_boxes()
        clear_ray_selection()

        if message is not None:
            status_label.value = message

    def update_frame_status() -> None:
        local_t = current_local_t()
        dataset_t = timepoints[local_t]
        current_frame_label.value = (
            f"Frame {local_t + 1}/{len(timepoints)}  "
            f"(dataset t={dataset_t})\n"
            f"Corrected merged instances here: "
            f"{session.corrections_in_frame(local_t)}"
        )

        try:
            undo_button.enabled = session.can_undo()
        except Exception:
            pass

    def show_wrapped_error(exc: Exception) -> None:
        error_text = str(exc)
        words = error_text.split()
        wrapped_lines: list[str] = []
        current_line = ""

        for word in words:
            candidate = (
                word
                if not current_line
                else f"{current_line} {word}"
            )

            if len(candidate) > 72 and current_line:
                wrapped_lines.append(current_line)
                current_line = word
            else:
                current_line = candidate

        if current_line:
            wrapped_lines.append(current_line)

        status_label.value = (
            "ERROR: " + "\n".join(wrapped_lines)
        )

        print()
        print("[annotation error]")
        print(exc)

    @viewer.mouse_drag_callbacks.append
    def ray_pick_visible_supervoxel(_viewer, event):
        """
        Plain single-click:
            ray-pick frontmost atomic supervoxel
            -> next empty seed box
            -> corresponding colored seed highlight

        Click-drag:
            leave normal Napari camera navigation untouched.
        """
        button = getattr(event, "button", None)
        button_text = str(button).lower()

        is_left = (
            button is None
            or button == 1
            or "left" in button_text
            or button_text == "1"
        )

        if not is_left:
            return

        dragged = False
        yield

        while getattr(event, "type", None) == "mouse_move":
            dragged = True
            yield

        if dragged:
            return

        local_t = current_local_t()

        sv_id = _ray_pick_frontmost_label(
            supervoxel_layer,
            event,
        )

        if sv_id <= 0:
            status_label.value = (
                "Ray click hit background; no seed was added."
            )
            return

        if sv_id in {
            int(value)
            for value in selected_seed_ids
            if value is not None
        }:
            status_label.value = (
                f"SV {sv_id} is already selected as a seed."
            )
            return

        slot_index = next_empty_seed_slot()

        if slot_index is None:
            status_label.value = (
                "All four seed boxes are already filled. "
                "Press Reset selections or Save."
            )
            return

        boxes[slot_index].value = str(int(sv_id))
        show_seed_selection(
            local_t,
            slot_index,
            int(sv_id),
        )

        status_label.value = (
            f"Added SV {int(sv_id)} to Instance {slot_index + 1}."
        )

    def save_current_split() -> None:
        try:
            groups = [
                parse_supervoxel_group(str(box.value))
                for box in boxes
            ]
            result = session.apply_split(
                current_local_t(),
                groups,
            )
        except Exception as exc:
            show_wrapped_error(exc)
            return

        # The backing array was modified in place. Reassigning + refresh is
        # intentional: it forces Napari to rebuild the label display immediately
        # so the corrected case visibly changes color.
        corrected_layer.data = session.corrected
        corrected_layer.refresh()

        refresh_instance_graph_colors(
            current_local_t()
        )

        reset_selections()
        update_frame_status()

        status_label.value = (
            f"Saved t={result.timepoint}: original instance "
            f"{result.original_instance_id} -> "
            f"{len(result.output_instance_ids)} instances "
            f"{result.output_instance_ids}"
        )

        print()
        print("=" * 72)
        print("CORRECTION SAVED")
        print("=" * 72)
        print(f"Dataset timepoint : {result.timepoint}")
        print(f"Original instance : {result.original_instance_id}")
        for index, (seed_group, sv_group, output_id) in enumerate(
            zip(
                result.seed_groups,
                result.groups,
                result.output_instance_ids,
            ),
            start=1,
        ):
            print(
                f"True instance {index}: seeds={list(seed_group)} | "
                f"assigned_SVs={list(sv_group)} -> label {output_id}"
            )
        print(f"Output directory  : {session.output_dir}")
        print()

    def undo_last_operation() -> None:
        try:
            result = session.undo_last_split()
        except Exception as exc:
            show_wrapped_error(exc)
            return

        # Force Napari to repaint the restored labels.
        corrected_layer.data = session.corrected
        corrected_layer.refresh()

        undo_local_t = timepoints.index(
            int(result.timepoint)
        )

        refresh_instance_graph_colors(
            undo_local_t
        )

        reset_selections()

        # Undo is global/LIFO across the loaded sequence. If the user has moved
        # elsewhere since Save, jump back to the affected frame so the restored
        # cell is immediately visible.
        viewer.dims.set_current_step(
            0,
            undo_local_t,
        )

        update_frame_status()

        status_label.value = (
            f"Undid last Save at t={result.timepoint}: restored instance "
            f"{result.original_instance_id}; removed split labels "
            f"{result.removed_instance_ids}"
        )

        print()
        print("=" * 72)
        print("CORRECTION UNDONE")
        print("=" * 72)
        print(f"Dataset timepoint : {result.timepoint}")
        print(f"Restored instance : {result.original_instance_id}")
        print(f"Removed labels    : {result.removed_instance_ids}")
        print(f"Output directory  : {session.output_dir}")
        print()

    save_button.changed.connect(
        lambda *_: save_current_split()
    )
    reset_button.changed.connect(
        lambda *_: reset_selections(
            message="Selections reset."
        )
    )
    undo_button.changed.connect(
        lambda *_: undo_last_operation()
    )

    @viewer.bind_key("Control-S")
    def _save_with_keyboard(_viewer):
        save_current_split()

    @viewer.bind_key("Control-Z")
    def _undo_with_keyboard(_viewer):
        undo_last_operation()

    @viewer.bind_key("Escape")
    def _reset_with_escape(_viewer):
        reset_selections(
            message="Selections reset."
        )

    # Clear stale typed IDs when the user changes TIME. Moving through z does
    # not clear them.
    last_local_t = {"value": current_local_t()}

    def on_dims_change(_event=None) -> None:
        now = current_local_t()
        if now != last_local_t["value"]:
            reset_selections()
            last_local_t["value"] = now
            status_label.value = (
                "Frame changed. Seed selections and input boxes were cleared; "
                "saved corrections remain applied."
            )
        update_frame_status()

    viewer.dims.events.current_step.connect(on_dims_change)

    # Start near the center z-slice of the first requested frame.
    try:
        viewer.dims.set_current_step(0, 0)
        viewer.dims.set_current_step(1, raw.shape[1] // 2)
    except Exception:
        pass

    update_frame_status()

    viewer.dims.ndisplay = 3

    print()
    print("=" * 72)
    print("ANNOTATION INSTRUCTIONS")
    print("=" * 72)
    print(f"Sample           : {sample_id}")
    print(f"Dataset frames   : {timepoints}")
    print()
    print("1. Navigate through the selected timepoints and z slices.")
    print("2. Find a merged spatial instance.")
    print(
        "3. Every different supervoxel ID and every different segmented "
        "instance ID gets its own display color; colors are not reused."
    )
    print(
        "4. In 3-D, SINGLE-CLICK the first visible seed supervoxel. "
        "Its ID automatically goes into Instance 1."
    )
    print(
        "5. SINGLE-CLICK the second visible seed supervoxel. "
        "Its ID automatically goes into Instance 2."
    )
    print(
        "6. Optional third/fourth clicks fill Instance 3/4."
    )
    print(
        "7. Selected seeds are highlighted with distinct colors: "
        "red, blue, green, magenta."
    )
    print(
        "8. Click-drag still rotates/pans normally; background clicks add nothing."
    )
    print(
        "9. Press Reset selections or Esc to clear all boxes/highlights "
        "without saving."
    )
    print(
        "10. Press Save (or Ctrl+S). The split is applied and selections reset."
    )
    print(
        "11. Press Undo last Save (or Ctrl+Z) to reverse the newest split."
    )
    print()
    print(
        "The tool automatically expands the seeds over the current merged "
        "instance using the weighted supervoxel contact graph."
    )
    print(
        "After Save, the split gets new instance IDs, so its colors change "
        "immediately and remain changed while you move through time."
    )
    print(f"Outputs          : {session.output_dir}")
    print("=" * 72)
    print()

    return viewer


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    args = resolve_paths(parse_args())

    spatial_root = args.spatial_root.resolve()
    supervoxel_root = args.supervoxel_root.resolve()
    stage6_root = resolve_stage6_root(
        args.sample_id,
        args.stage6_root,
    )
    available = completed_spatial_frames(spatial_root)

    if not available:
        raise FileNotFoundError(
            "No complete Investigation-25 frames were found below:\n"
            f"  {spatial_root}\n\n"
            "Expected, for example:\n"
            "  t000/partition/after_split_only.npy"
        )

    timepoints = parse_timepoint_selection(
        args.timepoints,
        available,
    )

    print("=" * 72)
    print("BIOHUB SUPERVOXEL INSTANCE ANNOTATOR V11")
    print("=" * 72)
    print(f"Repository       : {REPO_ROOT}")
    print(f"Sample           : {args.sample_id}")
    print(f"Inv25 instances : {spatial_root}")
    print(f"Inv24 SV root   : {supervoxel_root}")
    print(f"Stage-6 root     : {stage6_root}")
    print(f"Available frames : {available}")
    print(f"Selected frames  : {timepoints}")
    print(f"Raw Zarr         : {args.zarr_path}")
    print(
        f"Binary mask      : "
        f"{args.binary_mask or '(derived from Investigation-25 instances)'}"
    )
    print(f"Spacing ZYX um   : {DEFAULT_SPACING_ZYX_UM}")
    print(f"Output           : {args.output_dir}")
    print(f"Suspect root     : {args.suspect_root}")
    print(f"Suspect threshold: {float(args.suspect_threshold):.3f}")
    print(f"Build suspects   : {bool(args.build_suspects)}")
    print()

    raw = load_raw_frames(
        args.zarr_path.resolve(),
        timepoints,
    )

    stage6_binary_mask = (
        load_stage6_binary_mask_frames(
            stage6_root,
            timepoints,
        )
    )

    print(
        f"[stage6] loaded binary mask stack: "
        f"shape={stage6_binary_mask.shape}, "
        f"dtype={stage6_binary_mask.dtype}"
    )

    # Recommended production path:
    #   instances   -> Investigation 25 final split-only partition
    #   supervoxels -> Investigation 24 atomic watershed reused by Inv25
    if args.supervoxels is None and args.instances is None:
        supervoxels, instances = load_investigation25_frames(
            spatial_root,
            supervoxel_root,
            timepoints,
        )
        supervoxels = supervoxels.astype(np.int32, copy=False)
        instances = instances.astype(np.int32, copy=False)

    # Optional legacy/manual override.
    elif args.supervoxels is not None and args.instances is not None:
        supervoxels = load_selected_label_frames(
            args.supervoxels,
            timepoints,
            role="supervoxels",
        ).astype(np.int32, copy=False)

        instances = load_selected_label_frames(
            args.instances,
            timepoints,
            role="instances",
        ).astype(np.int32, copy=False)

    else:
        raise AnnotationError(
            "--supervoxels and --instances must either both be omitted "
            "(recommended Investigation-25 mode) or both be supplied."
        )

    if args.binary_mask is None:
        foreground = instances > 0
    else:
        foreground = load_selected_label_frames(
            args.binary_mask,
            timepoints,
            role="binary_mask",
        ) > 0

    validate_stacks(
        raw,
        supervoxels,
        instances,
        foreground,
    )

    if stage6_binary_mask.shape != supervoxels.shape:
        raise AnnotationError(
            "Stage-6 binary-mask stack shape does not match spatial data: "
            f"mask={stage6_binary_mask.shape}, "
            f"supervoxels={supervoxels.shape}."
        )

    suspect_instances = None

    if args.build_suspects:
        run_suspect_exporter(
            args=args,
            timepoints=timepoints,
            spatial_root=spatial_root,
            supervoxel_root=supervoxel_root,
        )

    suspect_root = args.suspect_root.resolve()
    if suspect_root.is_dir():
        try:
            suspect_instances, suspect_stats = load_suspect_instance_frames(
                suspect_root=suspect_root,
                timepoints=timepoints,
                instances=instances,
                threshold=float(args.suspect_threshold),
            )
            print(
                "[suspects] display stack ready: "
                f"{suspect_stats['displayed_instances']} threshold-passing "
                f"instances from {suspect_stats['score_rows']} score rows"
            )
        except FileNotFoundError as exc:
            if args.build_suspects:
                raise
            print(
                "[suspects] score directory is incomplete for selected frames; "
                "suspect layer disabled."
            )
            print(exc)
            suspect_instances = None
    else:
        print(
            "[suspects] no score directory found; existing annotator behavior "
            "is unchanged. Use --build-suspects to create it."
        )

    session = AnnotationSession(
        sample_id=args.sample_id,
        timepoints=timepoints,
        supervoxels=supervoxels,
        base_instances=instances,
        output_dir=args.output_dir.resolve(),
        resume=not args.no_resume,
    )

    make_viewer(
        sample_id=args.sample_id,
        timepoints=timepoints,
        raw=raw,
        stage6_binary_mask=stage6_binary_mask,
        supervoxels=supervoxels,
        foreground=foreground,
        session=session,
        suspect_instances=suspect_instances,
    )

    napari.run()


if __name__ == "__main__":
    main()
