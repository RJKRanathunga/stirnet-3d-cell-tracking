#!/usr/bin/env python
r"""
11_drosophila_absolute_spacing_match.py

Select the ABSOLUTE effective spacing for Drosophila_1 / Drosophila_2 by
matching their cell-size regime to the BioHub Stage-6 processed cache used by
the notebook pipeline.

This version DOES NOT require BioHub ground-truth labels.

BioHub reference source
-----------------------
The repository's notebook pipeline writes Stage-6 instance segmentations to:

    data/sample/processed/stage_6_processed_dataset/
        <sample_id>/
            segmentation/
                t000.npy
                t001.npy
                ...
            cells/
                t000.csv
                ...

Those segmentation volumes are treated here as PSEUDO-GT only for estimating
the target-domain physical cell-size regime. They are not treated as true
competition ground truth.

The script keeps the native Drosophila voxel arrays unchanged and solves for:

    effective_drosophila_spacing_zyx
        = alpha * drosophila_relative_scale_zyx

where the default relative scale is the joint result from investigation 10:

    (3.89410, 1.0, 1.0)

and the BioHub physical spacing defaults to:

    (1.625, 0.40625, 0.40625) um

Independent alpha estimates
---------------------------
alpha_cell_size
    Matches the median equivalent-sphere cell diameter. PRIMARY estimate.

alpha_pca_size
    Matches the median geometric mean of PCA principal-axis lengths.

alpha_neighbour
    Matches median nearest-neighbour centroid distance. SECONDARY estimate,
    because packing density can legitimately differ between organisms.

alpha_balanced
    Geometric mean of alpha_cell_size and alpha_neighbour.

alpha_biohub_xy
    Makes Drosophila X/Y pitch equal to BioHub X/Y pitch. This is an
    acquisition-geometry sanity check.

No source TIFF/NPY/Zarr file is modified or resampled.

Typical use from repository root
--------------------------------

    python .\investigations\stirnet\data\11_drosophila_absolute_spacing_match.py `
        --drosophila D:\Projects\Kaggle\cell-tracking\data\external\NIS3D\NIS3D\Drosophila_1 `
        --drosophila D:\Projects\Kaggle\cell-tracking\data\external\NIS3D\NIS3D\Drosophila_2

By default, ALL available Stage-6 BioHub samples and segmentation frames are
discovered automatically.

Restrict to specific BioHub samples:

    --biohub-sample 44b6_0113de3b --biohub-sample <another_id>

Restrict frames, for example:

    --biohub-frames 0-4,10,15-19

Coordinate order throughout is Z,Y,X.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class CellStats:
    name: str
    source: str
    voxel_counts: np.ndarray
    centroids_zyx: np.ndarray
    covariances_zyx: np.ndarray
    rejected_boundary: int
    rejected_small: int
    rejected_degenerate: int

    @property
    def n_cells(self) -> int:
        return int(self.voxel_counts.size)


# -----------------------------------------------------------------------------
# Repository / path discovery
# -----------------------------------------------------------------------------

def repo_root_from_script() -> Path:
    try:
        # repo/investigations/stirnet/data/11_*.py -> repo
        return Path(__file__).resolve().parents[3]
    except Exception:
        return Path.cwd().resolve()


def resolve_drosophila_dir(spec: str, repo_root: Path) -> Path:
    p = Path(spec).expanduser()

    if p.exists():
        p = p.resolve()
        if p.is_dir() and (p / "GroundTruth.tif").exists():
            return p
        raise FileNotFoundError(
            f"{p} exists but does not contain GroundTruth.tif."
        )

    candidates = [
        repo_root / spec,
        repo_root / "data" / spec,
        repo_root / "data" / "external" / "NIS3D" / spec,
        repo_root / "data" / "external" / "NIS3D" / "NIS3D" / spec,
        repo_root / "data" / "external" / "nis3d" / spec,
        repo_root / "data" / "external" / "nis3d" / "NIS3D" / spec,
    ]

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "GroundTruth.tif").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not locate Drosophila dataset {spec!r}. "
        "Pass its full directory with --drosophila."
    )


def default_stage6_root(repo_root: Path) -> Path:
    return (
        repo_root
        / "data"
        / "sample"
        / "processed"
        / "stage_6_processed_dataset"
    )


def resolve_stage6_root(
    repo_root: Path,
    explicit: Path | None,
) -> Path:
    root = explicit.expanduser().resolve() if explicit else default_stage6_root(repo_root)
    if not root.is_dir():
        raise FileNotFoundError(
            "BioHub Stage-6 processed cache was not found:\n"
            f"  {root}\n\n"
            "Generate it with notebooks/06_generate_processed_dataset.ipynb "
            "or pass --biohub-cache-root."
        )
    return root


def discover_biohub_samples(
    stage6_root: Path,
    requested: Sequence[str] | None,
) -> list[Path]:
    if requested:
        samples = []
        for sample_id in requested:
            p = stage6_root / sample_id
            if not (p / "segmentation").is_dir():
                raise FileNotFoundError(
                    f"Stage-6 segmentation cache not found for {sample_id}:\n"
                    f"  {p / 'segmentation'}"
                )
            samples.append(p.resolve())
        return samples

    samples = sorted(
        p.resolve()
        for p in stage6_root.iterdir()
        if p.is_dir() and (p / "segmentation").is_dir()
    )

    if not samples:
        raise RuntimeError(
            f"No Stage-6 samples with segmentation/ were found under:\n"
            f"  {stage6_root}"
        )

    return samples


# -----------------------------------------------------------------------------
# Frame selection
# -----------------------------------------------------------------------------

_FRAME_RE = re.compile(r"^t(\d+)\.npy$", re.IGNORECASE)


def frame_number(path: Path) -> int:
    m = _FRAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected Stage-6 segmentation filename: {path.name}")
    return int(m.group(1))


def parse_frame_spec(spec: str | None) -> set[int] | None:
    """
    Parse:
        0-4,10,15-19
    """
    if spec is None or not spec.strip():
        return None

    result: set[int] = set()

    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue

        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            if start < 0 or end < start:
                raise ValueError(f"Invalid frame range: {token}")
            result.update(range(start, end + 1))
        else:
            value = int(token)
            if value < 0:
                raise ValueError(f"Invalid frame: {value}")
            result.add(value)

    if not result:
        raise ValueError("--biohub-frames selected no frames.")

    return result


def list_segmentation_frames(
    sample_dir: Path,
    selected_frames: set[int] | None,
    max_frames: int | None,
) -> list[Path]:
    seg_dir = sample_dir / "segmentation"

    files = []
    for p in seg_dir.glob("t*.npy"):
        if _FRAME_RE.match(p.name):
            files.append(p)

    files.sort(key=frame_number)

    if selected_frames is not None:
        files = [p for p in files if frame_number(p) in selected_frames]

    if max_frames is not None:
        files = files[:max_frames]

    if not files:
        raise RuntimeError(
            f"No matching segmentation frames found in:\n  {seg_dir}"
        )

    return files


# -----------------------------------------------------------------------------
# Array loading
# -----------------------------------------------------------------------------

def open_tiff(path: Path) -> np.ndarray:
    try:
        return tifffile.memmap(path)
    except Exception:
        return tifffile.imread(path)


def load_npy(path: Path) -> np.ndarray:
    # mmap keeps Stage-6 frames cheap to open.
    return np.load(path, mmap_mode="r", allow_pickle=False)


# -----------------------------------------------------------------------------
# Instance measurement
# -----------------------------------------------------------------------------

def measure_instances(
    labels: np.ndarray,
    *,
    name: str,
    source: str,
    min_voxels: int,
    max_cells: int | None,
    reject_boundary: bool,
    seed: int,
    quiet: bool = False,
) -> CellStats:
    if labels.ndim != 3:
        raise ValueError(
            f"{source}: expected a 3-D instance-label volume, got {labels.shape}."
        )

    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(
            f"{source}: expected integer instance IDs, got dtype={labels.dtype}."
        )

    objects = ndimage.find_objects(labels)
    ids = np.array(
        [i + 1 for i, slc in enumerate(objects) if slc is not None],
        dtype=np.int64,
    )

    if max_cells is not None and ids.size > max_cells:
        rng = np.random.default_rng(seed)
        ids = np.sort(rng.choice(ids, size=max_cells, replace=False))

    counts: list[int] = []
    centroids: list[np.ndarray] = []
    covariances: list[np.ndarray] = []

    rejected_boundary = 0
    rejected_small = 0
    rejected_degenerate = 0

    shape = tuple(int(v) for v in labels.shape)

    for label_id in ids:
        slc = objects[int(label_id) - 1]
        if slc is None:
            continue

        if reject_boundary and any(
            int(s.start) == 0 or int(s.stop) == shape[d]
            for d, s in enumerate(slc)
        ):
            rejected_boundary += 1
            continue

        roi = labels[slc]
        zz, yy, xx = np.nonzero(roi == label_id)
        n = int(zz.size)

        if n < min_voxels:
            rejected_small += 1
            continue

        coords = np.column_stack(
            (
                zz.astype(np.float64) + float(slc[0].start),
                yy.astype(np.float64) + float(slc[1].start),
                xx.astype(np.float64) + float(slc[2].start),
            )
        )

        centroid = coords.mean(axis=0)
        centered = coords - centroid[None, :]
        cov = centered.T @ centered / float(n)
        cov = 0.5 * (cov + cov.T)

        eig = np.linalg.eigvalsh(cov)
        if not np.all(np.isfinite(eig)) or eig[0] <= 1e-9:
            rejected_degenerate += 1
            continue

        counts.append(n)
        centroids.append(centroid)
        covariances.append(cov)

    if not counts:
        raise RuntimeError(
            f"{name}: no usable instances remained after filtering."
        )

    result = CellStats(
        name=name,
        source=source,
        voxel_counts=np.asarray(counts, dtype=np.float64),
        centroids_zyx=np.asarray(centroids, dtype=np.float64),
        covariances_zyx=np.asarray(covariances, dtype=np.float64),
        rejected_boundary=rejected_boundary,
        rejected_small=rejected_small,
        rejected_degenerate=rejected_degenerate,
    )

    if not quiet:
        print(f"\n{name}")
        print("-" * len(name))
        print(f"source                    : {source}")
        print(f"shape                     : {shape}")
        print(f"available labels          : {len(objects):,}")
        print(f"measured cells            : {result.n_cells:,}")
        print(f"rejected: boundary        : {rejected_boundary:,}")
        print(f"rejected: small           : {rejected_small:,}")
        print(f"rejected: degenerate      : {rejected_degenerate:,}")

    return result


def combine_cell_stats(name: str, groups: Sequence[CellStats]) -> CellStats:
    if not groups:
        raise ValueError("Cannot combine an empty stats list.")

    return CellStats(
        name=name,
        source="multiple volumes",
        voxel_counts=np.concatenate([g.voxel_counts for g in groups]),
        centroids_zyx=np.concatenate([g.centroids_zyx for g in groups], axis=0),
        covariances_zyx=np.concatenate(
            [g.covariances_zyx for g in groups],
            axis=0,
        ),
        rejected_boundary=sum(g.rejected_boundary for g in groups),
        rejected_small=sum(g.rejected_small for g in groups),
        rejected_degenerate=sum(g.rejected_degenerate for g in groups),
    )


# -----------------------------------------------------------------------------
# Physical metrics
# -----------------------------------------------------------------------------

def physical_metrics(
    stats: CellStats,
    spacing_zyx: Sequence[float],
) -> dict[str, np.ndarray]:
    spacing = np.asarray(spacing_zyx, dtype=np.float64)

    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Invalid ZYX spacing: {spacing_zyx}")

    voxel_volume = float(np.prod(spacing))
    physical_volumes = stats.voxel_counts * voxel_volume

    # Equal-volume sphere.
    equivalent_diameter = 2.0 * np.cbrt(
        3.0 * physical_volumes / (4.0 * math.pi)
    )

    # C_phys = S C_vox S where S = diag(spacing).
    cov_phys = (
        stats.covariances_zyx
        * spacing[None, :, None]
        * spacing[None, None, :]
    )

    eig = np.maximum(np.linalg.eigvalsh(cov_phys), 1e-12)

    # For a uniform solid ellipsoid:
    # covariance eigenvalue ~= semi_axis^2 / 5.
    pca_lengths = 2.0 * np.sqrt(5.0 * eig[..., ::-1])  # long, middle, short
    pca_geometric_size = np.cbrt(np.prod(pca_lengths, axis=1))

    centroids_phys = stats.centroids_zyx * spacing[None, :]

    if stats.n_cells >= 2:
        tree = cKDTree(centroids_phys)
        distances, _ = tree.query(centroids_phys, k=2)
        nearest = distances[:, 1]
    else:
        nearest = np.empty(0, dtype=np.float64)

    return {
        "equivalent_diameter": equivalent_diameter,
        "equivalent_radius": 0.5 * equivalent_diameter,
        "pca_geometric_size": pca_geometric_size,
        "pca_lengths": pca_lengths,
        "nearest_neighbour": nearest,
    }


def pooled_metrics_per_volume(
    groups: Sequence[CellStats],
    spacing_zyx: Sequence[float],
) -> dict[str, np.ndarray]:
    """
    Pool cell-size measurements across frames/volumes.

    Nearest-neighbour distance is deliberately computed within each individual
    frame first, then pooled. We must not allow centroids from different frames
    or samples to become artificial neighbours.
    """
    per = [physical_metrics(g, spacing_zyx) for g in groups]

    def cat(key: str) -> np.ndarray:
        arrays = [m[key] for m in per if m[key].size]
        return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64)

    return {
        "equivalent_diameter": cat("equivalent_diameter"),
        "equivalent_radius": cat("equivalent_radius"),
        "pca_geometric_size": cat("pca_geometric_size"),
        "nearest_neighbour": cat("nearest_neighbour"),
        "pca_lengths": np.concatenate(
            [m["pca_lengths"] for m in per],
            axis=0,
        ),
    }


def median(a: np.ndarray) -> float:
    return float(np.median(a)) if a.size else float("nan")


def percentile(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


def print_metric_summary(title: str, metrics: dict[str, np.ndarray]) -> None:
    pca = metrics["pca_lengths"]

    long_mid = pca[:, 0] / pca[:, 1]
    mid_short = pca[:, 1] / pca[:, 2]
    long_short = pca[:, 0] / pca[:, 2]

    print(f"\n{title}")
    print("-" * len(title))
    print(
        "equivalent diameter       : "
        f"median={median(metrics['equivalent_diameter']):.6f}  "
        f"p25={percentile(metrics['equivalent_diameter'],25):.6f}  "
        f"p75={percentile(metrics['equivalent_diameter'],75):.6f}"
    )
    print(
        "equivalent radius         : "
        f"median={median(metrics['equivalent_radius']):.6f}"
    )
    print(
        "PCA geometric size        : "
        f"median={median(metrics['pca_geometric_size']):.6f}"
    )
    print(
        "nearest-neighbour distance: "
        f"median={median(metrics['nearest_neighbour']):.6f}"
    )
    print(f"PCA long/mid              : median={np.median(long_mid):.6f}")
    print(f"PCA mid/short             : median={np.median(mid_short):.6f}")
    print(f"PCA long/short            : median={np.median(long_short):.6f}")


# -----------------------------------------------------------------------------
# Alpha estimation
# -----------------------------------------------------------------------------

def safe_ratio(target: float, source: float) -> float:
    if (
        not np.isfinite(target)
        or not np.isfinite(source)
        or source <= 0
    ):
        return float("nan")
    return float(target / source)


def geometric_mean(values: Sequence[float]) -> float:
    valid = np.asarray(
        [v for v in values if np.isfinite(v) and v > 0],
        dtype=np.float64,
    )
    if valid.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(valid))))


def percent_difference(value: float, target: float) -> float:
    return 100.0 * (value - target) / target


# -----------------------------------------------------------------------------
# BioHub Stage-6 loading
# -----------------------------------------------------------------------------

def load_biohub_stage6_stats(
    sample_dirs: Sequence[Path],
    *,
    selected_frames: set[int] | None,
    max_frames_per_sample: int | None,
    min_voxels: int,
    max_cells_per_frame: int | None,
    reject_boundary: bool,
    seed: int,
) -> tuple[list[CellStats], dict[str, list[int]]]:
    results: list[CellStats] = []
    used_frames: dict[str, list[int]] = {}

    print(f"\n{'=' * 88}")
    print("BIOHUB STAGE-6 PSEUDO-GT")
    print(f"{'=' * 88}")
    print(
        "Using cached instance segmentations as a morphology reference only; "
        "these are NOT true BioHub ground truth."
    )

    frame_counter = 0

    for sample_i, sample_dir in enumerate(sample_dirs):
        files = list_segmentation_frames(
            sample_dir,
            selected_frames=selected_frames,
            max_frames=max_frames_per_sample,
        )

        used_frames[sample_dir.name] = [frame_number(p) for p in files]

        print(
            f"\n{sample_dir.name}: {len(files)} frame(s) from "
            f"{sample_dir / 'segmentation'}"
        )

        sample_cell_count = 0

        for local_i, path in enumerate(files):
            labels = load_npy(path)
            frame = frame_number(path)

            stats = measure_instances(
                labels,
                name=f"BioHub::{sample_dir.name}::t{frame:03d}",
                source=str(path),
                min_voxels=min_voxels,
                max_cells=max_cells_per_frame,
                reject_boundary=reject_boundary,
                seed=seed + 10_000 + frame_counter,
                quiet=True,
            )

            results.append(stats)
            sample_cell_count += stats.n_cells
            frame_counter += 1

            print(
                f"  t{frame:03d}: {stats.n_cells:4d} usable pseudo-instances",
                end="\r",
                flush=True,
            )

        print(" " * 80, end="\r")
        print(
            f"  measured {sample_cell_count:,} pseudo-instances "
            f"across {len(files)} frame(s)"
        )

    if not results:
        raise RuntimeError("No BioHub Stage-6 pseudo-GT frames were loaded.")

    print(
        f"\nBioHub total: {sum(r.n_cells for r in results):,} "
        f"pseudo-instances across {len(results)} frame(s)."
    )

    return results, used_frames


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Estimate Drosophila absolute effective spacing from the BioHub "
            "Stage-6 processed segmentation cache."
        )
    )

    p.add_argument(
        "--drosophila",
        action="append",
        required=True,
        help=(
            "Drosophila dataset directory/name. Repeat for Drosophila_1 and "
            "Drosophila_2."
        ),
    )

    p.add_argument(
        "--biohub-cache-root",
        type=Path,
        default=None,
        help=(
            "Override the Stage-6 root. Default: "
            "data/sample/processed/stage_6_processed_dataset/"
        ),
    )

    p.add_argument(
        "--biohub-sample",
        action="append",
        default=None,
        help=(
            "Use only this Stage-6 sample ID. Repeat for multiple samples. "
            "Default: auto-discover all cached samples."
        ),
    )

    p.add_argument(
        "--biohub-frames",
        type=str,
        default=None,
        help=(
            "Optional frame selection, e.g. '0-4,10,15-19'. "
            "Default: all cached frames."
        ),
    )

    p.add_argument(
        "--max-biohub-frames-per-sample",
        type=int,
        default=0,
        help=(
            "Maximum Stage-6 frames used per sample; 0 means all. "
            "Default: 0."
        ),
    )

    p.add_argument(
        "--dros-relative-scale",
        nargs=3,
        type=float,
        metavar=("Z", "Y", "X"),
        default=(3.89410, 1.0, 1.0),
        help=(
            "Relative Drosophila scale from investigation 10. "
            "Default: 3.89410 1 1."
        ),
    )

    p.add_argument(
        "--biohub-spacing",
        nargs=3,
        type=float,
        metavar=("Z", "Y", "X"),
        default=(1.625, 0.40625, 0.40625),
        help=(
            "BioHub physical spacing Z Y X. "
            "Default: 1.625 0.40625 0.40625 um."
        ),
    )

    p.add_argument(
        "--min-voxels",
        type=int,
        default=100,
        help="Ignore instances below this voxel count. Default: 100.",
    )

    p.add_argument(
        "--max-dros-cells",
        type=int,
        default=2500,
        help=(
            "Maximum measured cells per Drosophila volume; 0 means all. "
            "Default: 2500."
        ),
    )

    p.add_argument(
        "--max-biohub-cells-per-frame",
        type=int,
        default=0,
        help=(
            "Maximum pseudo-instances per BioHub frame; 0 means all. "
            "Default: 0."
        ),
    )

    p.add_argument(
        "--keep-boundary-cells",
        action="store_true",
        help=(
            "Keep instances touching the outer volume boundary. "
            "Default: reject them."
        ),
    )

    p.add_argument(
        "--recommend",
        choices=(
            "cell-size",
            "pca-size",
            "neighbour",
            "balanced",
            "biohub-xy",
        ),
        default="cell-size",
        help=(
            "Which alpha is selected as final recommendation. "
            "Default: cell-size."
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Random seed for optional cell subsampling. Default: 11.",
    )

    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Output JSON. Default: investigations/stirnet/data/_outputs/"
            "11_absolute_spacing_match/recommendation.json"
        ),
    )

    return p


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.min_voxels < 4:
        raise ValueError("--min-voxels must be >= 4.")

    for name in (
        "max_dros_cells",
        "max_biohub_cells_per_frame",
        "max_biohub_frames_per_sample",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_','-')} must be >= 0.")

    relative = np.asarray(args.dros_relative_scale, dtype=np.float64)
    bio_spacing = np.asarray(args.biohub_spacing, dtype=np.float64)

    if (
        relative.shape != (3,)
        or np.any(~np.isfinite(relative))
        or np.any(relative <= 0)
    ):
        raise ValueError("--dros-relative-scale must contain 3 positive values.")

    if (
        bio_spacing.shape != (3,)
        or np.any(~np.isfinite(bio_spacing))
        or np.any(bio_spacing <= 0)
    ):
        raise ValueError("--biohub-spacing must contain 3 positive values.")

    repo_root = repo_root_from_script()
    print(f"Repository root: {repo_root}")

    reject_boundary = not args.keep_boundary_cells

    max_dros_cells = (
        None if args.max_dros_cells == 0 else args.max_dros_cells
    )
    max_bio_cells = (
        None
        if args.max_biohub_cells_per_frame == 0
        else args.max_biohub_cells_per_frame
    )
    max_bio_frames = (
        None
        if args.max_biohub_frames_per_sample == 0
        else args.max_biohub_frames_per_sample
    )

    # ------------------------------------------------------------------
    # Drosophila GT
    # ------------------------------------------------------------------
    print(f"\n{'=' * 88}")
    print("DROSOPHILA GROUND TRUTH")
    print(f"{'=' * 88}")

    dros_stats: list[CellStats] = []

    for i, spec in enumerate(args.drosophila):
        dataset_dir = resolve_drosophila_dir(spec, repo_root)
        gt_path = dataset_dir / "GroundTruth.tif"

        stats = measure_instances(
            open_tiff(gt_path),
            name=dataset_dir.name,
            source=str(gt_path),
            min_voxels=args.min_voxels,
            max_cells=max_dros_cells,
            reject_boundary=reject_boundary,
            seed=args.seed + i,
            quiet=False,
        )

        dros_stats.append(stats)

    # ------------------------------------------------------------------
    # BioHub Stage-6 pseudo-GT
    # ------------------------------------------------------------------
    stage6_root = resolve_stage6_root(
        repo_root,
        args.biohub_cache_root,
    )

    sample_dirs = discover_biohub_samples(
        stage6_root,
        args.biohub_sample,
    )

    print(f"\nStage-6 cache root:")
    print(f"  {stage6_root}")
    print("BioHub samples:")
    for p in sample_dirs:
        print(f"  - {p.name}")

    selected_frames = parse_frame_spec(args.biohub_frames)

    bio_stats, used_frames = load_biohub_stage6_stats(
        sample_dirs,
        selected_frames=selected_frames,
        max_frames_per_sample=max_bio_frames,
        min_voxels=args.min_voxels,
        max_cells_per_frame=max_bio_cells,
        reject_boundary=reject_boundary,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Physical comparison
    # ------------------------------------------------------------------
    dros_relative_metrics = pooled_metrics_per_volume(
        dros_stats,
        relative,
    )
    bio_metrics = pooled_metrics_per_volume(
        bio_stats,
        bio_spacing,
    )

    print(f"\n{'=' * 88}")
    print("ABSOLUTE SPACING MATCH")
    print(f"{'=' * 88}")

    print(
        "Dros relative scale [Z,Y,X] : "
        f"({relative[0]:.6f}, {relative[1]:.6f}, {relative[2]:.6f})"
    )
    print(
        "BioHub spacing [Z,Y,X]      : "
        f"({bio_spacing[0]:.6f}, "
        f"{bio_spacing[1]:.6f}, "
        f"{bio_spacing[2]:.6f})"
    )

    print_metric_summary(
        "Drosophila with relative scale only (alpha=1)",
        dros_relative_metrics,
    )
    print_metric_summary(
        "BioHub Stage-6 pseudo-GT at BioHub spacing",
        bio_metrics,
    )

    alpha_cell = safe_ratio(
        median(bio_metrics["equivalent_diameter"]),
        median(dros_relative_metrics["equivalent_diameter"]),
    )

    alpha_pca = safe_ratio(
        median(bio_metrics["pca_geometric_size"]),
        median(dros_relative_metrics["pca_geometric_size"]),
    )

    alpha_neighbour = safe_ratio(
        median(bio_metrics["nearest_neighbour"]),
        median(dros_relative_metrics["nearest_neighbour"]),
    )

    alpha_balanced = geometric_mean(
        (alpha_cell, alpha_neighbour)
    )

    # Match Dros XY physical pitch to BioHub XY physical pitch.
    alpha_biohub_xy = math.sqrt(
        (bio_spacing[1] / relative[1])
        * (bio_spacing[2] / relative[2])
    )

    estimates = {
        "cell-size": alpha_cell,
        "pca-size": alpha_pca,
        "neighbour": alpha_neighbour,
        "balanced": alpha_balanced,
        "biohub-xy": alpha_biohub_xy,
    }

    print("\nIndependent alpha estimates")
    print("---------------------------")
    print(f"cell-size   : {alpha_cell:.8f}   [PRIMARY]")
    print(f"pca-size    : {alpha_pca:.8f}")
    print(f"neighbour   : {alpha_neighbour:.8f}   [SECONDARY]")
    print(f"balanced    : {alpha_balanced:.8f}")
    print(f"biohub-xy   : {alpha_biohub_xy:.8f}   [SANITY CHECK]")

    chosen_alpha = estimates[args.recommend]

    if not np.isfinite(chosen_alpha) or chosen_alpha <= 0:
        raise RuntimeError(
            f"Recommendation mode {args.recommend!r} produced invalid "
            f"alpha={chosen_alpha}."
        )

    chosen_spacing = chosen_alpha * relative
    biohub_xy_candidate = alpha_biohub_xy * relative

    print(f"\n{'=' * 88}")
    print("RECOMMENDATION")
    print(f"{'=' * 88}")
    print(f"mode         : {args.recommend}")
    print(f"chosen alpha : {chosen_alpha:.8f}")
    print(
        "effective Dros spacing [Z,Y,X]: "
        f"({chosen_spacing[0]:.8f}, "
        f"{chosen_spacing[1]:.8f}, "
        f"{chosen_spacing[2]:.8f})"
    )

    print("\nDifference from BioHub acquisition spacing:")
    for i, axis in enumerate(("Z", "Y", "X")):
        print(
            f"  {axis}: {chosen_spacing[i]:.8f} vs "
            f"{bio_spacing[i]:.8f} "
            f"({percent_difference(chosen_spacing[i], bio_spacing[i]):+.2f}%)"
        )

    print("\nBioHub-XY-pitch candidate:")
    print(f"  alpha = {alpha_biohub_xy:.8f}")
    print(
        "  spacing [Z,Y,X] = "
        f"({biohub_xy_candidate[0]:.8f}, "
        f"{biohub_xy_candidate[1]:.8f}, "
        f"{biohub_xy_candidate[2]:.8f})"
    )
    print(
        "  resulting Z difference vs BioHub Z = "
        f"{percent_difference(biohub_xy_candidate[0], bio_spacing[0]):+.2f}%"
    )

    # What Dros morphology looks like at the chosen physical spacing.
    chosen_dros_metrics = pooled_metrics_per_volume(
        dros_stats,
        chosen_spacing,
    )

    print_metric_summary(
        "Drosophila under recommended effective spacing",
        chosen_dros_metrics,
    )

    # ------------------------------------------------------------------
    # Agreement diagnostic
    # ------------------------------------------------------------------
    finite_estimates = [
        v
        for v in (
            alpha_cell,
            alpha_pca,
            alpha_neighbour,
            alpha_biohub_xy,
        )
        if np.isfinite(v) and v > 0
    ]

    spread = (
        max(finite_estimates) / min(finite_estimates)
        if len(finite_estimates) >= 2
        else float("nan")
    )

    print("\nAgreement diagnostic")
    print("--------------------")
    print(f"max/min alpha ratio : {spread:.4f}")

    if np.isfinite(spread):
        if spread <= 1.10:
            agreement = "very strong"
            print("interpretation        : VERY STRONG agreement (<=10%).")
        elif spread <= 1.20:
            agreement = "good"
            print("interpretation        : good agreement (<=20%).")
        elif spread <= 1.35:
            agreement = "moderate"
            print(
                "interpretation        : moderate agreement; inspect the "
                "cell-size estimate more strongly than packing density."
            )
        else:
            agreement = "weak"
            print(
                "interpretation        : weak agreement. Do not force cell "
                "size and inter-cell packing to match using one scale."
            )
    else:
        agreement = "unavailable"

    # ------------------------------------------------------------------
    # Per-sample BioHub summary
    # ------------------------------------------------------------------
    print("\nBioHub pseudo-GT per-sample cell-size summary")
    print("---------------------------------------------")

    per_sample_summary = {}

    for sample_dir in sample_dirs:
        prefix = f"BioHub::{sample_dir.name}::"
        sample_groups = [
            g for g in bio_stats if g.name.startswith(prefix)
        ]

        sample_metrics = pooled_metrics_per_volume(
            sample_groups,
            bio_spacing,
        )

        eq = median(sample_metrics["equivalent_diameter"])
        pca = median(sample_metrics["pca_geometric_size"])
        nn = median(sample_metrics["nearest_neighbour"])

        per_sample_summary[sample_dir.name] = {
            "frames": used_frames[sample_dir.name],
            "pseudo_instances": int(sum(g.n_cells for g in sample_groups)),
            "median_equivalent_diameter": eq,
            "median_pca_geometric_size": pca,
            "median_nearest_neighbour": nn,
        }

        print(
            f"{sample_dir.name}: "
            f"cells={sum(g.n_cells for g in sample_groups):5d}  "
            f"eq_diam={eq:.5f}  "
            f"pca_size={pca:.5f}  "
            f"NN={nn:.5f}"
        )

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    output_json = args.output_json

    if output_json is None:
        output_json = (
            repo_root
            / "investigations"
            / "stirnet"
            / "data"
            / "_outputs"
            / "11_absolute_spacing_match"
            / "recommendation.json"
        )
    else:
        output_json = output_json.expanduser().resolve()

    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "coordinate_order": "ZYX",
        "units": "um",
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "biohub_reference_is_pseudo_gt": True,
        "biohub_reference_description": (
            "Stage-6 cached instance segmentation from "
            "data/sample/processed/stage_6_processed_dataset/<sample>/"
            "segmentation/tXXX.npy"
        ),
        "stage6_cache_root": str(stage6_root),
        "biohub_samples": per_sample_summary,
        "drosophila_sources": [
            {
                "name": s.name,
                "source": s.source,
                "measured_cells": s.n_cells,
            }
            for s in dros_stats
        ],
        "drosophila_relative_scale_zyx": [
            float(v) for v in relative
        ],
        "biohub_spacing_zyx": [
            float(v) for v in bio_spacing
        ],
        "alpha_estimates": {
            key: float(value)
            for key, value in estimates.items()
        },
        "agreement": {
            "max_min_alpha_ratio": float(spread),
            "interpretation": agreement,
        },
        "recommendation": {
            "mode": args.recommend,
            "alpha": float(chosen_alpha),
            "effective_drosophila_spacing_zyx": [
                float(v) for v in chosen_spacing
            ],
            "resample_arrays": False,
            "intended_use": (
                "Keep native Drosophila arrays unchanged and propagate this "
                "effective spacing consistently through STIR-Net's spacing-aware "
                "data, target-generation, and model pipeline."
            ),
        },
        "biohub_xy_pitch_candidate": {
            "alpha": float(alpha_biohub_xy),
            "effective_drosophila_spacing_zyx": [
                float(v) for v in biohub_xy_candidate
            ],
        },
        "median_metrics": {
            "drosophila_alpha_1": {
                "equivalent_diameter": median(
                    dros_relative_metrics["equivalent_diameter"]
                ),
                "pca_geometric_size": median(
                    dros_relative_metrics["pca_geometric_size"]
                ),
                "nearest_neighbour": median(
                    dros_relative_metrics["nearest_neighbour"]
                ),
            },
            "biohub_stage6_pseudo_gt": {
                "equivalent_diameter": median(
                    bio_metrics["equivalent_diameter"]
                ),
                "pca_geometric_size": median(
                    bio_metrics["pca_geometric_size"]
                ),
                "nearest_neighbour": median(
                    bio_metrics["nearest_neighbour"]
                ),
            },
            "drosophila_recommended": {
                "equivalent_diameter": median(
                    chosen_dros_metrics["equivalent_diameter"]
                ),
                "pca_geometric_size": median(
                    chosen_dros_metrics["pca_geometric_size"]
                ),
                "nearest_neighbour": median(
                    chosen_dros_metrics["nearest_neighbour"]
                ),
            },
        },
        "caveats": [
            (
                "BioHub Stage-6 segmentations are pseudo-GT generated by the "
                "existing preprocessing/segmentation pipeline, not manual labels."
            ),
            (
                "Cell-size matching is therefore used as a coarse domain-scale "
                "calibration, not an accuracy measurement."
            ),
            (
                "Nearest-neighbour distance is secondary because biological "
                "packing density can differ between Drosophila and BioHub."
            ),
            (
                "Boundary-touching instances are rejected by default to avoid "
                "truncated size/shape measurements."
            ),
            (
                "No Drosophila or BioHub source array is modified or resampled."
            ),
        ],
    }

    output_json.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved diagnostic:")
    print(f"  {output_json}")

    print("\nNext decision")
    print("-------------")
    print(
        "If alpha_cell_size, alpha_pca_size, and alpha_biohub_xy are close, "
        "the evidence strongly supports a spacing-only Drosophila adaptation. "
        "The Stage-6 pseudo-GT is adequate for this coarse absolute-scale "
        "calibration even though it is not true BioHub ground truth."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
