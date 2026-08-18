from __future__ import annotations

"""Visual validation of STIR-Net GT geometry target generation.

Run from the repository root, for example:

    python investigations/stirnet/00_GT_validation.py

The script loads the same prepared first-overfit scene used by the current
STIR-Net spatial-first experiment, builds targets with the production
``build_geometry_targets`` function, prints numerical sanity checks, and opens
all target fields in Napari.

Useful options:

    python investigations/stirnet/00_GT_validation.py --full
    python investigations/stirnet/00_GT_validation.py --crop-shape 48 256 256
    python investigations/stirnet/00_GT_validation.py --time-index 2
    python investigations/stirnet/00_GT_validation.py --data-dir <path>
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------------

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from learned.stirnet.data.targets import estimate_model_dref_um
from learned.stirnet.model.geometry.targets import (
    GeometryTargets,
    build_geometry_targets,
)


# ---------------------------------------------------------------------------
# Defaults: intentionally match the current first-overfit investigation scene
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "first_overfit"
    / "BlastoSPIM1_F22_030_034"
)
DEFAULT_TIME_INDEX = 2
DEFAULT_CROP_SHAPE = (64, 320, 320)
DEFAULT_CONTEXT_MARGIN_DREF = 3.0
DEFAULT_VECTOR_STRIDE = (2, 12, 12)
DEFAULT_MAX_VECTORS = 5000
DEFAULT_FLOW_VECTOR_LENGTH_UM = 4.0


@dataclass(frozen=True)
class Scene:
    current_full: np.ndarray
    gt_full: np.ndarray
    raw_full: np.ndarray | None
    spacing_zyx_um: np.ndarray
    dref_um: float
    core_slices: tuple[slice, slice, slice]
    build_slices: tuple[slice, slice, slice]


# ---------------------------------------------------------------------------
# Loading / crop selection
# ---------------------------------------------------------------------------


def _bounded_center_crop(
    shape: tuple[int, int, int],
    center_zyx: np.ndarray,
    crop_shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    shape_arr = np.asarray(shape, dtype=np.int64)
    requested = np.minimum(np.asarray(crop_shape, dtype=np.int64), shape_arr)
    center = np.asarray(center_zyx, dtype=np.int64)
    lower = center - requested // 2
    lower = np.maximum(lower, 0)
    lower = np.minimum(lower, shape_arr - requested)
    upper = lower + requested
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))


def _hard_interfaces(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return hard cell-background surface and cell-cell separator voxels."""
    surface = np.zeros_like(labels, dtype=bool)
    separator = np.zeros_like(labels, dtype=bool)
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        a = labels[tuple(lo)]
        b = labels[tuple(hi)]
        different = a != b
        surface_face = different & ((a == 0) ^ (b == 0))
        separator_face = different & (a > 0) & (b > 0)
        surface[tuple(lo)] |= surface_face
        surface[tuple(hi)] |= surface_face
        separator[tuple(lo)] |= separator_face
        separator[tuple(hi)] |= separator_face
    return surface, separator


def _instance_boxes(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return instance IDs plus inclusive-low / exclusive-high GT bounding boxes."""
    object_slices = ndi.find_objects(labels)
    ids: list[int] = []
    lows: list[list[int]] = []
    highs: list[list[int]] = []
    for instance_id, bbox in enumerate(object_slices, 1):
        if bbox is None:
            continue
        ids.append(instance_id)
        lows.append([int(s.start) for s in bbox])
        highs.append([int(s.stop) for s in bbox])
    if not ids:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0, 3), dtype=np.int64),
        )
    return (
        np.asarray(ids, dtype=np.int64),
        np.asarray(lows, dtype=np.int64),
        np.asarray(highs, dtype=np.int64),
    )


def _adjacent_instance_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    """Collect GT cell-cell adjacency pairs using 6-neighbour voxel faces."""
    pairs: set[tuple[int, int]] = set()
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        a = labels[tuple(lo)]
        b = labels[tuple(hi)]
        valid = (a > 0) & (b > 0) & (a != b)
        if not valid.any():
            continue
        left = a[valid].astype(np.int64, copy=False)
        right = b[valid].astype(np.int64, copy=False)
        pair_array = np.stack([np.minimum(left, right), np.maximum(left, right)], axis=1)
        for x, y in np.unique(pair_array, axis=0):
            pairs.add((int(x), int(y)))
    return pairs


def _auto_complete_cell_crop(
    labels: np.ndarray,
    crop_shape: tuple[int, int, int],
) -> tuple[tuple[slice, slice, slice], dict]:
    """Choose a diagnostic crop dominated by complete GT cells.

    Candidate windows are centered on GT cell bounding-box centers. The score
    strongly rewards fully-contained instances and GT cell-cell contacts while
    penalizing instances that intersect the crop but are cut by its boundary.
    """
    shape = np.asarray(labels.shape, dtype=np.int64)
    requested = np.minimum(np.asarray(crop_shape, dtype=np.int64), shape)

    ids, lows, highs = _instance_boxes(labels)
    if len(ids) == 0:
        center = shape // 2
        crop = _bounded_center_crop(labels.shape, center, tuple(requested))
        return crop, {
            "complete_ids": [],
            "partial_ids": [],
            "contact_edges": 0,
            "score": 0.0,
        }

    centers = np.floor_divide(lows + highs - 1, 2)
    adjacency = _adjacent_instance_pairs(labels)

    best_crop: tuple[slice, slice, slice] | None = None
    best_info: dict | None = None
    best_score = -np.inf

    # Also evaluate centers between adjacent cells, which is useful for separator
    # validation and often captures a compact multi-cell cluster.
    candidate_centers = [centers]
    if adjacency:
        id_to_index = {int(instance_id): i for i, instance_id in enumerate(ids)}
        pair_centers: list[np.ndarray] = []
        for a, b in adjacency:
            if a in id_to_index and b in id_to_index:
                pair_centers.append(
                    np.rint(
                        0.5
                        * (
                            centers[id_to_index[a]].astype(np.float32)
                            + centers[id_to_index[b]].astype(np.float32)
                        )
                    ).astype(np.int64)
                )
        if pair_centers:
            candidate_centers.append(np.stack(pair_centers))

    candidates = np.concatenate(candidate_centers, axis=0)
    if len(candidates) > 1:
        candidates = np.unique(candidates, axis=0)

    for center in candidates:
        crop = _bounded_center_crop(labels.shape, center, tuple(requested))
        crop_low = np.asarray([s.start for s in crop], dtype=np.int64)
        crop_high = np.asarray([s.stop for s in crop], dtype=np.int64)

        intersects = np.all(highs > crop_low[None], axis=1) & np.all(
            lows < crop_high[None], axis=1
        )
        complete = intersects & np.all(lows >= crop_low[None], axis=1) & np.all(
            highs <= crop_high[None], axis=1
        )
        partial = intersects & ~complete

        complete_ids = ids[complete]
        complete_set = set(int(x) for x in complete_ids.tolist())
        contact_edges = sum(
            1 for a, b in adjacency if a in complete_set and b in complete_set
        )

        # Main objective: many intact cells. Partial cells are expensive because
        # they make 3-D target inspection visually misleading. Contacts are
        # rewarded because separator validation needs touching cells.
        score = (
            10.0 * float(complete.sum())
            + 4.0 * float(contact_edges)
            - 7.0 * float(partial.sum())
        )

        # Slight preference for interior crops instead of windows pinned against
        # the scene boundary, all else equal.
        touches_scene_boundary = np.count_nonzero(
            (crop_low == 0) | (crop_high == shape)
        )
        score -= 0.5 * float(touches_scene_boundary)

        if score > best_score:
            best_score = score
            best_crop = crop
            best_info = {
                "complete_ids": [int(x) for x in complete_ids.tolist()],
                "partial_ids": [int(x) for x in ids[partial].tolist()],
                "contact_edges": int(contact_edges),
                "score": float(score),
            }

    assert best_crop is not None and best_info is not None
    return best_crop, best_info


def _expand_build_region(
    gt_full: np.ndarray,
    core: tuple[slice, slice, slice],
    spacing_um: np.ndarray,
    dref_um: float,
    context_margin_dref: float,
) -> tuple[slice, slice, slice]:
    """Build targets with context and complete every cell visible in the core.

    This avoids creating artificial target geometry at the displayed crop edge.
    All GT objects that occur in the visible core are included completely in the
    larger build ROI, then the generated targets are cropped back to the core.
    """
    shape = np.asarray(gt_full.shape, dtype=np.int64)
    lower = np.asarray([s.start for s in core], dtype=np.int64)
    upper = np.asarray([s.stop for s in core], dtype=np.int64)

    selected_ids = np.unique(gt_full[core])
    selected_ids = selected_ids[selected_ids > 0]

    object_slices = ndi.find_objects(gt_full)
    for instance_id in selected_ids.tolist():
        index = int(instance_id) - 1
        if index < 0 or index >= len(object_slices):
            continue
        bbox = object_slices[index]
        if bbox is None:
            continue
        lower = np.minimum(lower, np.asarray([s.start for s in bbox], dtype=np.int64))
        upper = np.maximum(upper, np.asarray([s.stop for s in bbox], dtype=np.int64))

    margin_um = max(float(context_margin_dref) * float(dref_um), float(spacing_um.max()))
    margin_vox = np.ceil(margin_um / spacing_um).astype(np.int64)
    lower = np.maximum(lower - margin_vox, 0)
    upper = np.minimum(upper + margin_vox, shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))


def _relative_slices(
    inner: tuple[slice, slice, slice],
    outer: tuple[slice, slice, slice],
) -> tuple[slice, slice, slice]:
    return tuple(
        slice(int(i.start - o.start), int(i.stop - o.start))
        for i, o in zip(inner, outer)
    )


def load_scene(args: argparse.Namespace) -> Scene:
    data_dir = args.data_dir.resolve()
    instance_path = data_dir / "instance_movie.npy"
    gt_path = data_dir / "gt_movie.npy"
    metadata_path = data_dir / "metadata.json"

    missing = [p for p in (instance_path, gt_path, metadata_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared STIR-Net data is incomplete. Missing:\n  "
            + "\n  ".join(str(p) for p in missing)
        )

    instance_movie = np.load(instance_path, mmap_mode="r")
    gt_movie = np.load(gt_path, mmap_mode="r")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    spacing = np.asarray(metadata["spacing_zyx_um"], dtype=np.float32)

    t = int(args.time_index)
    if not (0 <= t < len(instance_movie)) or not (0 <= t < len(gt_movie)):
        raise IndexError(
            f"time-index {t} is outside available range "
            f"[0, {min(len(instance_movie), len(gt_movie)) - 1}]"
        )

    current_full = np.asarray(instance_movie[t])
    gt_full = np.asarray(gt_movie[t])
    if current_full.shape != gt_full.shape:
        raise RuntimeError(
            f"Current/GT shape mismatch: {current_full.shape} vs {gt_full.shape}"
        )

    dref_um = estimate_model_dref_um(
        current_full,
        tuple(float(v) for v in spacing),
    )

    crop_info: dict | None = None
    if args.full:
        core = tuple(slice(0, int(n)) for n in gt_full.shape)
        build = core
    else:
        core, crop_info = _auto_complete_cell_crop(
            gt_full,
            tuple(args.crop_shape),
        )
        build = _expand_build_region(
            gt_full,
            core,
            spacing,
            dref_um,
            args.context_margin_dref,
        )
        print(
            "Auto crop selection : "
            f"{len(crop_info['complete_ids'])} complete cells, "
            f"{len(crop_info['partial_ids'])} partial cells, "
            f"{crop_info['contact_edges']} complete-cell contacts"
        )
        if crop_info["partial_ids"]:
            print(
                "  Partial GT IDs   : "
                + ", ".join(str(x) for x in crop_info["partial_ids"][:16])
                + (" ..." if len(crop_info["partial_ids"]) > 16 else "")
            )

    raw_full: np.ndarray | None = None
    raw_path = data_dir / "stirnet_source" / "raw_norm_target.npy"
    if raw_path.exists():
        candidate = np.load(raw_path, mmap_mode="r")
        if candidate.shape == gt_full.shape:
            raw_full = candidate
        else:
            print(
                f"[WARN] Raw context skipped: {raw_path.name} shape "
                f"{candidate.shape} != GT shape {gt_full.shape}"
            )
    else:
        print(f"[WARN] Raw context not found: {raw_path}")

    if t != DEFAULT_TIME_INDEX and raw_full is not None:
        print(
            "[WARN] stirnet_source/raw_norm_target.npy is the prepared target-frame "
            "raw image. You selected a non-default time index; verify that this raw "
            "volume corresponds to the selected frame before using it as context."
        )

    return Scene(
        current_full=current_full,
        gt_full=gt_full,
        raw_full=raw_full,
        spacing_zyx_um=spacing,
        dref_um=float(dref_um),
        core_slices=core,
        build_slices=build,
    )


# ---------------------------------------------------------------------------
# Production target generation
# ---------------------------------------------------------------------------


def build_targets(scene: Scene) -> GeometryTargets:
    gt_build = np.asarray(scene.gt_full[scene.build_slices]).astype(np.int64, copy=True)
    current_build = np.asarray(
        scene.current_full[scene.build_slices]
    ).astype(np.int64, copy=True)
    labels = torch.from_numpy(gt_build)
    spacing = torch.from_numpy(scene.spacing_zyx_um.copy())
    dref = torch.tensor(scene.dref_um, dtype=torch.float32)

    return build_geometry_targets(
        labels,
        spacing,
        dref,
        current_labels=torch.from_numpy(current_build),
        device=torch.device("cpu"),
    )


def _target_arrays(
    targets: GeometryTargets,
    relative_core: tuple[slice, slice, slice],
) -> dict[str, np.ndarray]:
    s = relative_core
    return {
        "foreground": targets.foreground[0, 0][s].numpy(),
        "surface": targets.surface[0, 0][s].numpy(),
        "separator": targets.separator[0, 0][s].numpy(),
        "sdf": targets.sdf[0, 0][s].numpy(),
        "sdf_valid": targets.sdf_valid[0, 0][s].numpy(),
        "flow": targets.flow[0][(slice(None), *s)].numpy(),
        "centroid_offset": targets.centroid_offset[0][(slice(None), *s)].numpy(),
        "seed": targets.seed[0, 0][s].numpy(),
    }


# ---------------------------------------------------------------------------
# Numerical validation
# ---------------------------------------------------------------------------


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _print_scalar_stats(name: str, array: np.ndarray) -> None:
    arr = np.asarray(array)
    finite = np.isfinite(arr)
    if arr.size == 0:
        print(f"  {name:18s} empty")
        return
    if not finite.any():
        print(f"  {name:18s} all non-finite")
        return
    values = arr[finite]
    print(
        f"  {name:18s} shape={str(arr.shape):18s} "
        f"min={float(values.min()):9.5f} "
        f"mean={float(values.mean()):9.5f} "
        f"max={float(values.max()):9.5f}"
    )


def validate_targets(
    scene: Scene,
    targets: GeometryTargets,
    arrays: dict[str, np.ndarray],
) -> None:
    core = scene.core_slices
    gt = np.asarray(scene.gt_full[core])
    foreground = gt > 0

    print("\n" + "=" * 88)
    print("STIR-Net GT geometry target validation")
    print("=" * 88)
    print(f"GT full shape        : {tuple(scene.gt_full.shape)}")
    print(f"Visible core         : {core}")
    print(f"Target build ROI     : {scene.build_slices}")
    print(f"Visible core shape   : {tuple(gt.shape)}")
    print(f"Spacing z,y,x (um)   : {tuple(float(x) for x in scene.spacing_zyx_um)}")
    print(f"Model dref (um)      : {scene.dref_um:.5f}")
    visible_ids = np.unique(gt[gt > 0])
    ids, lows, highs = _instance_boxes(scene.gt_full)
    core_low = np.asarray([s.start for s in core], dtype=np.int64)
    core_high = np.asarray([s.stop for s in core], dtype=np.int64)
    intersects = np.all(highs > core_low[None], axis=1) & np.all(
        lows < core_high[None], axis=1
    )
    complete = intersects & np.all(lows >= core_low[None], axis=1) & np.all(
        highs <= core_high[None], axis=1
    )
    partial = intersects & ~complete

    print(f"GT instances in core : {visible_ids.size}")
    print(f"Complete GT cells     : {int(complete.sum())}")
    print(f"Partially cut GT cells: {int(partial.sum())}")

    print("\nTarget scalar statistics")
    for name in ("foreground", "surface", "separator", "sdf", "sdf_valid", "seed"):
        _print_scalar_stats(name, arrays[name])
    _print_scalar_stats("flow magnitude", np.linalg.vector_norm(arrays["flow"], axis=0))
    offset_um = arrays["centroid_offset"] * scene.dref_um
    _print_scalar_stats("offset magnitude um", np.linalg.vector_norm(offset_um, axis=0))

    checks: list[tuple[str, bool, str]] = []

    fg_exact = np.array_equal(arrays["foreground"] >= 0.5, foreground)
    checks.append(("foreground == (GT > 0)", fg_exact, "exact binary agreement"))

    for name in ("foreground", "surface", "separator", "sdf", "flow", "centroid_offset", "seed"):
        checks.append((f"{name} finite", bool(np.isfinite(arrays[name]).all()), "no NaN/Inf"))

    for name in ("surface", "separator", "seed"):
        arr = arrays[name]
        in_range = bool((arr >= -1e-6).all() and (arr <= 1.0 + 1e-6).all())
        checks.append((f"{name} in [0,1]", in_range, f"range=({arr.min():.5f},{arr.max():.5f})"))

    sdf = arrays["sdf"]
    if foreground.any():
        checks.append(
            (
                "SDF positive in foreground",
                bool((sdf[foreground] > 0).all()),
                f"foreground min={float(sdf[foreground].min()):.6f}",
            )
        )
        checks.append(
            (
                "SDF-valid covers foreground",
                bool(arrays["sdf_valid"][foreground].all()),
                "all GT voxels supervised",
            )
        )
    if (~foreground).any():
        checks.append(
            (
                "SDF negative in background",
                bool((sdf[~foreground] < 0).all()),
                f"background max={float(sdf[~foreground].max()):.6f}",
            )
        )

    hard_surface, hard_separator = _hard_interfaces(gt)
    if hard_surface.any():
        minimum = float(arrays["surface"][hard_surface].min())
        checks.append(("surface nonzero on hard interface", minimum > 0.0, f"min={minimum:.6f}"))
    if hard_separator.any():
        minimum = float(arrays["separator"][hard_separator].min())
        checks.append(("separator nonzero on hard interface", minimum > 0.0, f"min={minimum:.6f}"))

    # Per-instance seed maxima should reach 1 for complete cells in the build ROI.
    gt_build = np.asarray(scene.gt_full[scene.build_slices])
    seed_build = targets.seed[0, 0].numpy()
    visible_ids = np.unique(gt)
    visible_ids = visible_ids[visible_ids > 0]
    seed_failures: list[int] = []
    for instance_id in visible_ids.tolist():
        values = seed_build[gt_build == instance_id]
        if values.size == 0 or not np.isclose(float(values.max()), 1.0, atol=1e-5):
            seed_failures.append(int(instance_id))
    checks.append(
        (
            "each visible cell has seed max 1",
            len(seed_failures) == 0,
            "all cells" if not seed_failures else f"failed IDs={seed_failures[:12]}",
        )
    )

    # Centroid-offset contract: x_um + offset*dref must reconstruct the exact
    # GT cell centroid. Use deterministic sparse samples to keep this cheap.
    build_origin = np.asarray([s.start for s in scene.build_slices], dtype=np.float32)
    rel_core = _relative_slices(scene.core_slices, scene.build_slices)
    offset_build = targets.centroid_offset[0].numpy()
    max_centroid_error_um = 0.0
    sampled = 0
    for instance_id in visible_ids.tolist():
        full_coords = np.argwhere(scene.gt_full == instance_id)
        if full_coords.size == 0:
            continue
        expected_centroid_um = full_coords.astype(np.float32).mean(axis=0) * scene.spacing_zyx_um

        local_core_mask = gt == instance_id
        local_points = np.argwhere(local_core_mask)
        if local_points.size == 0:
            continue
        if len(local_points) > 256:
            step = max(1, len(local_points) // 256)
            local_points = local_points[::step][:256]

        core_origin = np.asarray([s.start for s in scene.core_slices], dtype=np.float32)
        global_points = local_points.astype(np.float32) + core_origin
        build_points = global_points - build_origin
        build_points_i = build_points.astype(np.int64)
        offset_values = offset_build[
            :,
            build_points_i[:, 0],
            build_points_i[:, 1],
            build_points_i[:, 2],
        ].T
        reconstructed = (
            global_points * scene.spacing_zyx_um[None]
            + offset_values * scene.dref_um
        )
        error = np.linalg.vector_norm(reconstructed - expected_centroid_um[None], axis=1)
        max_centroid_error_um = max(max_centroid_error_um, float(error.max()))
        sampled += len(error)

    checks.append(
        (
            "centroid-offset reconstruction",
            max_centroid_error_um <= 1e-3,
            f"max error={max_centroid_error_um:.6g} um over {sampled} voxels",
        )
    )

    print("\nContract checks")
    for name, ok, detail in checks:
        print(f"  [{_status(ok):4s}] {name:40s} {detail}")

    failed = [name for name, ok, _ in checks if not ok]
    print("\nResult")
    if failed:
        print(f"  FAIL: {len(failed)} contract check(s) failed.")
        for name in failed:
            print(f"    - {name}")
        print("  Napari will still open so the failed target(s) can be inspected visually.")
    else:
        print("  PASS: all automatic target-generation contract checks passed.")
    print("=" * 88 + "\n")


# ---------------------------------------------------------------------------
# Napari helpers
# ---------------------------------------------------------------------------


def _sample_vectors(
    field: np.ndarray,
    mask: np.ndarray,
    spacing_um: np.ndarray,
    stride: tuple[int, int, int],
    max_vectors: int,
    *,
    physical_scale_um: float | None = None,
    normalized_by_dref: float | None = None,
) -> np.ndarray:
    """Convert dense z/y/x vector field to Napari ``(N,2,3)`` vectors."""
    z, y, x = np.indices(mask.shape)
    select = (
        mask
        & (z % max(1, stride[0]) == 0)
        & (y % max(1, stride[1]) == 0)
        & (x % max(1, stride[2]) == 0)
    )
    points = np.argwhere(select)
    if points.size == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)

    if len(points) > max_vectors:
        step = int(np.ceil(len(points) / max_vectors))
        points = points[::step][:max_vectors]

    values = field[:, points[:, 0], points[:, 1], points[:, 2]].T.astype(np.float32)
    if normalized_by_dref is not None:
        vector_um = values * float(normalized_by_dref)
    elif physical_scale_um is not None:
        vector_um = values * float(physical_scale_um)
    else:
        vector_um = values

    # Napari vector coordinates are in voxels; convert physical displacement to
    # voxel displacement. Layer ``scale`` then restores physical z/y/x spacing.
    vector_vox = vector_um / spacing_um[None]
    return np.stack([points.astype(np.float32), vector_vox.astype(np.float32)], axis=1)


def _points_for_seed_maxima(
    gt: np.ndarray,
    seed: np.ndarray,
) -> np.ndarray:
    points: list[np.ndarray] = []
    for instance_id in np.unique(gt):
        if instance_id <= 0:
            continue
        mask = gt == instance_id
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        values = seed[mask]
        best = coords[int(np.argmax(values))]
        points.append(best.astype(np.float32))
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(points)


def _points_for_gt_centroids(gt: np.ndarray) -> np.ndarray:
    points: list[np.ndarray] = []
    shape = np.asarray(gt.shape)
    for instance_id in np.unique(gt):
        if instance_id <= 0:
            continue
        coords = np.argwhere(gt == instance_id)
        if coords.size == 0:
            continue
        centroid = coords.astype(np.float32).mean(axis=0)
        if np.all(centroid >= 0) and np.all(centroid <= shape - 1):
            points.append(centroid)
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(points)


def open_napari(
    scene: Scene,
    arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is required for this investigation. Install it in the project "
            "environment, for example: pip install 'napari[all]'"
        ) from exc

    core = scene.core_slices
    gt = np.asarray(scene.gt_full[core]).astype(np.int64, copy=False)
    current = np.asarray(scene.current_full[core]).astype(np.int64, copy=False)
    raw = None if scene.raw_full is None else np.asarray(scene.raw_full[core])
    scale = tuple(float(v) for v in scene.spacing_zyx_um)

    flow_magnitude = np.linalg.vector_norm(arrays["flow"], axis=0)
    offset_um = arrays["centroid_offset"] * scene.dref_um
    offset_magnitude_um = np.linalg.vector_norm(offset_um, axis=0)

    flow_vectors = _sample_vectors(
        arrays["flow"],
        gt > 0,
        scene.spacing_zyx_um,
        tuple(args.vector_stride),
        args.max_vectors,
        physical_scale_um=args.flow_vector_length_um,
    )
    offset_vectors = _sample_vectors(
        arrays["centroid_offset"],
        gt > 0,
        scene.spacing_zyx_um,
        tuple(args.vector_stride),
        args.max_vectors,
        normalized_by_dref=scene.dref_um,
    )

    seed_points = _points_for_seed_maxima(gt, arrays["seed"])
    centroid_points = _points_for_gt_centroids(gt)

    viewer = napari.Viewer(title="STIR-Net — 00 GT target validation", ndisplay=3)

    if raw is not None:
        viewer.add_image(
            raw,
            name="00 Raw normalized",
            colormap="gray",
            scale=scale,
        )

    viewer.add_labels(
        current,
        name="01 Current segmentation (context)",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        gt,
        name="02 GT instance labels",
        scale=scale,
    )
    viewer.add_image(
        arrays["foreground"],
        name="03 Target — foreground",
        colormap="gray",
        contrast_limits=(0.0, 1.0),
        scale=scale,
        visible=False,
    )
    viewer.add_image(
        arrays["surface"],
        name="04 Target — surface",
        colormap="inferno",
        contrast_limits=(0.0, 1.0),
        scale=scale,
        blending="additive",
        visible=False,
    )
    viewer.add_image(
        arrays["separator"],
        name="05 Target — separator",
        colormap="magenta",
        contrast_limits=(0.0, 1.0),
        scale=scale,
        blending="additive",
        visible=True,
    )

    sdf_abs = float(max(abs(float(arrays["sdf"].min())), abs(float(arrays["sdf"].max())), 1e-6))
    viewer.add_image(
        arrays["sdf"],
        name="06 Target — signed distance / dref",
        colormap="turbo",
        contrast_limits=(-sdf_abs, sdf_abs),
        scale=scale,
        visible=False,
    )
    viewer.add_image(
        arrays["sdf_valid"].astype(np.float32),
        name="07 Target — SDF valid",
        colormap="gray",
        contrast_limits=(0.0, 1.0),
        scale=scale,
        visible=False,
    )
    viewer.add_image(
        arrays["seed"],
        name="08 Target — seed",
        colormap="inferno",
        contrast_limits=(0.0, 1.0),
        scale=scale,
        visible=False,
    )
    viewer.add_image(
        flow_magnitude,
        name="09 Target — flow magnitude",
        colormap="viridis",
        contrast_limits=(0.0, max(1.0, float(flow_magnitude.max()))),
        scale=scale,
        visible=False,
    )
    viewer.add_image(
        offset_magnitude_um,
        name="10 Target — centroid offset magnitude (um)",
        colormap="viridis",
        scale=scale,
        visible=False,
    )

    if len(flow_vectors):
        viewer.add_vectors(
            flow_vectors,
            name="11 Target — SDF flow vectors",
            scale=scale,
            visible=False,
        )
    if len(offset_vectors):
        viewer.add_vectors(
            offset_vectors,
            name="12 Target — centroid offset vectors",
            scale=scale,
            visible=False,
        )
    if len(seed_points):
        viewer.add_points(
            seed_points,
            name="13 Seed maxima (one chosen per visible cell)",
            scale=scale,
            size=2.0,
            visible=False,
        )
    if len(centroid_points):
        viewer.add_points(
            centroid_points,
            name="14 GT voxel centroids",
            scale=scale,
            size=2.0,
            visible=False,
        )

    # Start near the center of the visible diagnostic crop.
    viewer.dims.current_step = tuple(int(n // 2) for n in gt.shape)

    print("Napari layers loaded.")
    print("Suggested first visual checks:")
    print("  1. GT labels + separator: separator must sit between touching GT cells.")
    print("  2. GT labels + surface: surface must follow cell-background interfaces.")
    print("  3. SDF: positive inside cells, negative in nearby background.")
    print("  4. Seed: smooth interior maxima, reaching 1 independently per cell.")
    print("  5. Flow vectors: per-cell SDF-gradient directions, with no cross-cell leakage.")
    print("  6. Offset vectors: vectors should point toward each cell's voxel centroid.")
    print("Switch Napari to 3-D when useful; the z/y/x layer scale is physical spacing.")

    napari.run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and visualize STIR-Net production GT geometry targets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Prepared STIR-Net scene directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=DEFAULT_TIME_INDEX,
        help=f"Frame to validate (default: {DEFAULT_TIME_INDEX})",
    )
    parser.add_argument(
        "--crop-shape",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=DEFAULT_CROP_SHAPE,
        help=(
            "Maximum visible diagnostic crop. The script automatically searches "
            "for a window with many complete GT cells and cell-cell contacts while "
            "penalizing cells cut by the crop boundary."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Build and display targets for the entire frame instead of a diagnostic crop.",
    )
    parser.add_argument(
        "--context-margin-dref",
        type=float,
        default=DEFAULT_CONTEXT_MARGIN_DREF,
        help="Extra target-build context around the visible crop in multiples of dref.",
    )
    parser.add_argument(
        "--vector-stride",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=DEFAULT_VECTOR_STRIDE,
        help="Napari sampling stride for dense vector layers.",
    )
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=DEFAULT_MAX_VECTORS,
        help="Maximum vectors per Napari vector layer.",
    )
    parser.add_argument(
        "--flow-vector-length-um",
        type=float,
        default=DEFAULT_FLOW_VECTOR_LENGTH_UM,
        help="Display length for unit SDF-flow arrows in micrometres.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = load_scene(args)

    print("Building production geometry targets on CPU...")
    print(f"  data dir       : {args.data_dir}")
    print(f"  time index     : {args.time_index}")
    print(f"  core slices    : {scene.core_slices}")
    print(f"  build slices   : {scene.build_slices}")

    targets = build_targets(scene)
    relative_core = _relative_slices(scene.core_slices, scene.build_slices)
    arrays = _target_arrays(targets, relative_core)

    validate_targets(scene, targets, arrays)
    open_napari(scene, arrays, args)


if __name__ == "__main__":
    main()
