#!/usr/bin/env python
r"""
10_drosophila_cell_shape_scaling.py

Diagnostic tool for choosing a global XYZ deformation for Drosophila GT cells.

What this script does
---------------------
1. Loads GroundTruth.tif from one or more datasets.
2. Measures each sufficiently large, non-boundary GT instance using a 3-D
   covariance/PCA model.
3. Reports:
      - PCA axis-ratio distributions
      - axis-aligned extents
      - alignment of each cell's shortest PCA axis with global Z/Y/X
4. Fits a single global diagonal XYZ scale that makes the cells as close as
   possible to a requested target morphology (round by default).
5. Opens an optional Napari 3-D viewer with interactive Z/Y/X scale controls.
   The viewer changes only layer display scale; it does NOT resample or modify
   the source TIFFs.
6. Lets you save the visually chosen scale to JSON.

Typical use
-----------
From the repository root:

    python investigations/stirnet/data/10_drosophila_cell_shape_scaling.py \
        --dataset Drosophila_1 \
        --dataset Drosophila_2

Analyze without opening Napari:

    python investigations/stirnet/data/10_drosophila_cell_shape_scaling.py \
        --dataset Drosophila_1 \
        --dataset Drosophila_2 \
        --no-viewer

Open Drosophila_2 instead of the first dataset after fitting the combined scale:

    python investigations/stirnet/data/10_drosophila_cell_shape_scaling.py \
        --dataset Drosophila_1 \
        --dataset Drosophila_2 \
        --view-index 1

If automatic dataset discovery does not find your folder, pass the full path:

    python investigations/stirnet/data/10_drosophila_cell_shape_scaling.py \
        --dataset "D:\path\to\Drosophila_1"

Important
---------
The Napari XYZ scale is a DISPLAY-ONLY diagnostic. Once a scale is selected for
training, raw data and labels should be resampled from the original source data:
linear interpolation for raw intensity, nearest-neighbour for instance labels,
then STIR-Net dense geometry targets should be regenerated from the transformed
GT/masks. Do not geometrically stretch already-generated vector/SDF targets.

Coordinate order throughout this script is (Z, Y, X).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile
from scipy import ndimage
from scipy.optimize import minimize


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class CellMeasurements:
    dataset_name: str
    dataset_dir: Path
    gt_shape: tuple[int, int, int]
    label_ids: np.ndarray           # [N]
    voxel_counts: np.ndarray        # [N]
    covariances: np.ndarray         # [N, 3, 3], coordinate order Z,Y,X
    extents: np.ndarray             # [N, 3], axis-aligned bbox extents Z,Y,X
    pca_lengths: np.ndarray         # [N, 3], sorted long -> short
    shortest_axis_vectors: np.ndarray  # [N, 3], components in Z,Y,X

    @property
    def n_cells(self) -> int:
        return int(self.covariances.shape[0])


# -----------------------------------------------------------------------------
# Paths / TIFF loading
# -----------------------------------------------------------------------------

def repo_root_from_script() -> Path:
    # .../repo/investigations/stirnet/data/10_*.py -> repo
    try:
        return Path(__file__).resolve().parents[3]
    except Exception:
        return Path.cwd().resolve()


def _looks_like_dataset_dir(path: Path) -> bool:
    return path.is_dir() and (path / "GroundTruth.tif").exists()


def resolve_dataset_dir(spec: str, repo_root: Path) -> Path:
    p = Path(spec).expanduser()
    if p.exists():
        p = p.resolve()
        if not _looks_like_dataset_dir(p):
            raise FileNotFoundError(
                f"{p} exists, but GroundTruth.tif was not found inside it."
            )
        return p

    candidates = [
        repo_root / spec,
        repo_root / "data" / spec,
        repo_root / "data" / "NIS3D" / spec,
        repo_root / "data" / "nis3d" / spec,
        repo_root / "data" / "learned" / "stirnet" / spec,
        repo_root / "data" / "learned" / "stirnet" / "NIS3D" / spec,
        repo_root / "data" / "external" / "NIS3D" / spec,
        repo_root / "data" / "external" / "nis3d" / spec,
    ]
    for candidate in candidates:
        if _looks_like_dataset_dir(candidate):
            return candidate.resolve()

    # Final fallback: search only under repo/data, not the entire repository.
    data_root = repo_root / "data"
    matches: list[Path] = []
    if data_root.exists():
        for gt_path in data_root.rglob("GroundTruth.tif"):
            if gt_path.parent.name.lower() == spec.lower():
                matches.append(gt_path.parent.resolve())

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        options = "\n".join(f"  - {m}" for m in matches)
        raise RuntimeError(
            f"Dataset name {spec!r} matched multiple folders:\n{options}\n"
            "Pass the full dataset path with --dataset."
        )

    raise FileNotFoundError(
        f"Could not locate dataset {spec!r}. "
        "Pass the full dataset directory with --dataset."
    )


def open_tiff(path: Path) -> np.ndarray:
    """
    Prefer a disk-backed TIFF memmap. Fall back to tifffile.imread when the TIFF
    layout cannot be memory-mapped.
    """
    try:
        return tifffile.memmap(path)
    except Exception:
        return tifffile.imread(path)


def find_raw_tiff(dataset_dir: Path) -> Path | None:
    preferred = [
        dataset_dir / "data.tif",
        dataset_dir / "Data.tif",
        dataset_dir / "raw.tif",
        dataset_dir / "Raw.tif",
    ]
    for p in preferred:
        if p.exists():
            return p
    return None


# -----------------------------------------------------------------------------
# Cell measurement
# -----------------------------------------------------------------------------

def _pca_from_covariance(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        lengths: 3 relative PCA lengths, sorted long -> short.
                 For a uniform solid ellipsoid, 2*sqrt(5*lambda) estimates
                 full principal-axis diameters. The constant does not affect
                 ratios.
        shortest_vector: unit eigenvector corresponding to shortest axis,
                         components in global Z,Y,X coordinates.
    """
    vals, vecs = np.linalg.eigh(cov)  # ascending eigenvalues
    vals = np.maximum(vals, 1e-12)

    shortest_vector = vecs[:, 0]
    lengths = 2.0 * np.sqrt(5.0 * vals[::-1])  # long -> short
    return lengths, shortest_vector


def measure_cells(
    dataset_dir: Path,
    *,
    min_voxels: int,
    max_cells: int | None,
    reject_boundary: bool,
    seed: int,
) -> CellMeasurements:
    gt_path = dataset_dir / "GroundTruth.tif"
    gt = open_tiff(gt_path)

    if gt.ndim != 3:
        raise ValueError(
            f"{gt_path} must be a 3-D label volume; got shape {gt.shape}."
        )
    if not np.issubdtype(gt.dtype, np.integer):
        raise TypeError(
            f"{gt_path} must contain integer instance IDs; got dtype {gt.dtype}."
        )

    print(f"\n{'=' * 80}")
    print(f"{dataset_dir.name}")
    print(f"{'=' * 80}")
    print(f"GT file     : {gt_path}")
    print(f"GT shape    : {tuple(int(v) for v in gt.shape)}")
    print(f"GT dtype    : {gt.dtype}")
    print("Scanning label bounding boxes...")

    objects = ndimage.find_objects(gt)
    label_ids = np.array(
        [i + 1 for i, slc in enumerate(objects) if slc is not None],
        dtype=np.int64,
    )

    if max_cells is not None and label_ids.size > max_cells:
        rng = np.random.default_rng(seed)
        label_ids = np.sort(
            rng.choice(label_ids, size=max_cells, replace=False)
        )

    kept_ids: list[int] = []
    voxel_counts: list[int] = []
    covariances: list[np.ndarray] = []
    extents: list[np.ndarray] = []
    pca_lengths: list[np.ndarray] = []
    shortest_vectors: list[np.ndarray] = []

    rejected_small = 0
    rejected_boundary = 0
    rejected_degenerate = 0

    shape = tuple(int(v) for v in gt.shape)

    for idx, label_id in enumerate(label_ids, start=1):
        slc = objects[int(label_id) - 1]
        if slc is None:
            continue

        if reject_boundary:
            touches = any(
                int(s.start) == 0 or int(s.stop) == shape[d]
                for d, s in enumerate(slc)
            )
            if touches:
                rejected_boundary += 1
                continue

        region = gt[slc]
        zz, yy, xx = np.nonzero(region == label_id)
        n = int(zz.size)
        if n < min_voxels:
            rejected_small += 1
            continue

        # Convert local ROI coordinates to full-volume Z,Y,X coordinates.
        coords = np.column_stack(
            (
                zz.astype(np.float64) + float(slc[0].start),
                yy.astype(np.float64) + float(slc[1].start),
                xx.astype(np.float64) + float(slc[2].start),
            )
        )

        centered = coords - coords.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / float(n)
        cov = 0.5 * (cov + cov.T)

        vals = np.linalg.eigvalsh(cov)
        if not np.all(np.isfinite(vals)) or vals[0] <= 1e-9:
            rejected_degenerate += 1
            continue

        lengths, shortest_vec = _pca_from_covariance(cov)
        bbox_extent = np.array(
            [int(s.stop) - int(s.start) for s in slc],
            dtype=np.float64,
        )

        kept_ids.append(int(label_id))
        voxel_counts.append(n)
        covariances.append(cov)
        extents.append(bbox_extent)
        pca_lengths.append(lengths)
        shortest_vectors.append(shortest_vec)

        if idx % 250 == 0 or idx == label_ids.size:
            print(
                f"  examined {idx:5d}/{label_ids.size:5d} labels; "
                f"kept {len(kept_ids):5d}",
                end="\r",
                flush=True,
            )

    print(" " * 90, end="\r")

    if not covariances:
        raise RuntimeError(
            f"No usable cells remained in {dataset_dir.name}. "
            "Lower --min-voxels or inspect the GT."
        )

    result = CellMeasurements(
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir,
        gt_shape=shape,
        label_ids=np.asarray(kept_ids, dtype=np.int64),
        voxel_counts=np.asarray(voxel_counts, dtype=np.int64),
        covariances=np.stack(covariances, axis=0),
        extents=np.stack(extents, axis=0),
        pca_lengths=np.stack(pca_lengths, axis=0),
        shortest_axis_vectors=np.stack(shortest_vectors, axis=0),
    )

    print(f"Available GT labels       : {len(objects):,}")
    print(f"Measured cells            : {result.n_cells:,}")
    print(f"Rejected: small           : {rejected_small:,}")
    print(f"Rejected: volume boundary : {rejected_boundary:,}")
    print(f"Rejected: degenerate      : {rejected_degenerate:,}")

    return result


# -----------------------------------------------------------------------------
# Morphology / scaling math
# -----------------------------------------------------------------------------

def transform_covariances(covs: np.ndarray, scales_zyx: Sequence[float]) -> np.ndarray:
    s = np.asarray(scales_zyx, dtype=np.float64)
    # S C S for diagonal S.
    return covs * s[None, :, None] * s[None, None, :]


def pca_lengths_from_covariances(covs: np.ndarray) -> np.ndarray:
    vals = np.linalg.eigvalsh(covs)  # ascending
    vals = np.maximum(vals, 1e-12)
    lengths = 2.0 * np.sqrt(5.0 * vals[..., ::-1])  # long -> short
    return lengths


def axis_ratios(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    long = lengths[:, 0]
    mid = lengths[:, 1]
    short = lengths[:, 2]
    return long / mid, mid / short, long / short


def morphology_objective(
    covs: np.ndarray,
    scales_zyx: Sequence[float],
    target_long_mid: float,
    target_mid_short: float,
) -> float:
    transformed = transform_covariances(covs, scales_zyx)
    lengths = pca_lengths_from_covariances(transformed)
    long_mid, mid_short, _ = axis_ratios(lengths)

    e1 = np.log(long_mid / target_long_mid)
    e2 = np.log(mid_short / target_mid_short)
    # Robust to unusual cells / imperfect labels.
    return float(np.median(e1 * e1 + e2 * e2))


def fit_global_scale(
    groups: Sequence[CellMeasurements],
    *,
    target_long_mid: float,
    target_mid_short: float,
    max_relative_scale: float,
) -> np.ndarray:
    """
    Optimize a global diagonal scale.

    Shape is invariant to multiplication of all three scales by the same
    constant, so fitting is done in log-space with det(S)=1:
        log(sz) + log(sy) + log(sx) = 0.

    Afterwards the three scales are divided by their median, giving a convenient
    relative deformation in which a typical unaffected axis remains near 1.
    """
    if max_relative_scale <= 1.0:
        raise ValueError("--max-relative-scale must be > 1.")

    bound = math.log(max_relative_scale)

    def unpack(theta: np.ndarray) -> np.ndarray:
        # theta controls log Z and log Y; log X closes the determinant.
        lz = float(theta[0])
        ly = float(theta[1])
        lx = -(lz + ly)
        return np.exp(np.array([lz, ly, lx], dtype=np.float64))

    def objective(theta: np.ndarray) -> float:
        scales = unpack(theta)
        if np.any(scales < 1.0 / max_relative_scale) or np.any(
            scales > max_relative_scale
        ):
            return 1e3 + float(np.sum(np.square(np.log(scales))))
        # Dataset-balanced: Drosophila_1 and Drosophila_2 contribute equally
        # even if one contains more measured cells.
        values = [
            morphology_objective(
                g.covariances,
                scales,
                target_long_mid,
                target_mid_short,
            )
            for g in groups
        ]
        return float(np.mean(values))

    starts = [
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([math.log(1.5), 0.0], dtype=np.float64),
        np.array([0.0, math.log(1.5)], dtype=np.float64),
        np.array([-math.log(1.5), 0.0], dtype=np.float64),
    ]

    best = None
    for x0 in starts:
        result = minimize(
            objective,
            x0=x0,
            method="Powell",
            bounds=[(-bound, bound), (-bound, bound)],
            options={"xtol": 1e-5, "ftol": 1e-7, "maxiter": 300},
        )
        if best is None or result.fun < best.fun:
            best = result

    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("XYZ scale optimization failed.")

    scales = unpack(np.asarray(best.x, dtype=np.float64))

    # Overall magnification is irrelevant to shape. This representation is
    # easier to read: if one axis is compressed, the other two tend to remain
    # around 1 and the compressed axis gets a >1 stretch.
    scales = scales / np.median(scales)

    return scales


def shortest_axis_angles_deg(vectors_zyx: np.ndarray) -> np.ndarray:
    """
    Angle between shortest PCA axis and each GLOBAL axis.
    Sign is irrelevant, so abs(dot) is used.
    Returns [N,3] for Z,Y,X.
    """
    v = vectors_zyx / np.linalg.norm(vectors_zyx, axis=1, keepdims=True)
    cosines = np.clip(np.abs(v), 0.0, 1.0)
    return np.degrees(np.arccos(cosines))


def pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def fmt_triplet(v: Sequence[float], digits: int = 3) -> str:
    return "(" + ", ".join(f"{float(x):.{digits}f}" for x in v) + ")"


def print_dataset_summary(
    m: CellMeasurements,
    recommended_scale: Sequence[float] | None = None,
) -> None:
    long_mid, mid_short, long_short = axis_ratios(m.pca_lengths)
    angles = shortest_axis_angles_deg(m.shortest_axis_vectors)

    print(f"\n--- {m.dataset_name}: native morphology ---")
    print(
        "Axis-aligned bbox extent median [Z,Y,X] : "
        f"{fmt_triplet(np.median(m.extents, axis=0), 2)} vox"
    )
    print(
        "PCA axis length median [long,mid,short] : "
        f"{fmt_triplet(np.median(m.pca_lengths, axis=0), 2)}"
    )
    print(
        "PCA long/mid ratio                     : "
        f"median={np.median(long_mid):.3f}  "
        f"p25={pct(long_mid,25):.3f}  p75={pct(long_mid,75):.3f}"
    )
    print(
        "PCA mid/short ratio                    : "
        f"median={np.median(mid_short):.3f}  "
        f"p25={pct(mid_short,25):.3f}  p75={pct(mid_short,75):.3f}"
    )
    print(
        "PCA long/short ratio                   : "
        f"median={np.median(long_short):.3f}  "
        f"p25={pct(long_short,25):.3f}  p75={pct(long_short,75):.3f}"
    )

    print("\nShortest PCA axis alignment with global axes:")
    for axis_i, axis_name in enumerate(("Z", "Y", "X")):
        a = angles[:, axis_i]
        print(
            f"  {axis_name}: median angle={np.median(a):6.2f} deg  "
            f"p25={pct(a,25):6.2f}  p75={pct(a,75):6.2f}"
        )

    best_axis = ("Z", "Y", "X")[int(np.argmin(np.median(angles, axis=0)))]
    best_angle = float(np.min(np.median(angles, axis=0)))
    print(
        f"Most common global flat-axis alignment : {best_axis} "
        f"(median shortest-axis angle {best_angle:.2f} deg)"
    )

    if recommended_scale is not None:
        transformed = transform_covariances(m.covariances, recommended_scale)
        lengths = pca_lengths_from_covariances(transformed)
        lm, ms, ls = axis_ratios(lengths)
        print("\nAfter recommended global XYZ display scale:")
        print(f"  scale [Z,Y,X]       : {fmt_triplet(recommended_scale, 4)}")
        print(f"  median long/mid     : {np.median(lm):.3f}")
        print(f"  median mid/short    : {np.median(ms):.3f}")
        print(f"  median long/short   : {np.median(ls):.3f}")


# -----------------------------------------------------------------------------
# Napari interactive viewer
# -----------------------------------------------------------------------------

def sampled_percentiles(arr: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    # Sample ~1M voxels at most, preserving a representative spread through ZYX.
    shape = np.asarray(arr.shape, dtype=np.int64)
    target = 1_000_000
    total = int(np.prod(shape))
    if total <= target:
        sample = np.asarray(arr).reshape(-1)
    else:
        step = max(1, int(round((total / target) ** (1.0 / arr.ndim))))
        slc = tuple(slice(None, None, step) for _ in range(arr.ndim))
        sample = np.asarray(arr[slc]).reshape(-1)

    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0, 1.0

    a, b = np.percentile(sample, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b) or a >= b:
        a = float(np.min(sample))
        b = float(np.max(sample))
        if a >= b:
            b = a + 1.0
    return float(a), float(b)


def launch_viewer(
    m: CellMeasurements,
    *,
    recommended_scale: Sequence[float],
    target_long_mid: float,
    target_mid_short: float,
    output_dir: Path,
    raw_percentiles: tuple[float, float],
) -> None:
    try:
        import napari
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QDoubleSpinBox,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSlider,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:
        raise RuntimeError(
            "Napari/Qt could not be imported. Run with --no-viewer for the "
            "mathematical analysis only, or install the project's Napari "
            "dependencies."
        ) from exc

    gt = open_tiff(m.dataset_dir / "GroundTruth.tif")
    raw_path = find_raw_tiff(m.dataset_dir)
    raw = open_tiff(raw_path) if raw_path is not None else None

    viewer = napari.Viewer(ndisplay=3, title=f"STIR-Net cell shape scale — {m.dataset_name}")

    layers = []
    if raw is not None:
        lo, hi = sampled_percentiles(raw, *raw_percentiles)
        raw_layer = viewer.add_image(
            raw,
            name="raw",
            colormap="gray",
            contrast_limits=(lo, hi),
            rendering="mip",
        )
        layers.append(raw_layer)
    else:
        print(
            f"[viewer] No data.tif/raw.tif found in {m.dataset_dir}; "
            "showing GT only."
        )

    gt_layer = viewer.add_labels(
        gt,
        name="GroundTruth",
        opacity=0.55,
    )
    layers.append(gt_layer)

    # ------------------------------------------------------------------
    # Dock widget
    # ------------------------------------------------------------------
    root = QWidget()
    layout = QVBoxLayout(root)

    title = QLabel(
        "<b>XYZ display scaling</b><br>"
        "This changes Napari display geometry only. Source arrays are untouched."
    )
    title.setWordWrap(True)
    layout.addWidget(title)

    recommended_label = QLabel(
        "Mathematical recommendation [Z,Y,X]: "
        f"<b>{fmt_triplet(recommended_scale, 4)}</b>"
    )
    recommended_label.setWordWrap(True)
    layout.addWidget(recommended_label)

    target_label = QLabel(
        f"Target PCA ratios: long/mid={target_long_mid:.3f}, "
        f"mid/short={target_mid_short:.3f}"
    )
    layout.addWidget(target_label)

    metrics_label = QLabel()
    metrics_label.setWordWrap(True)
    layout.addWidget(metrics_label)

    group = QGroupBox("Interactive scales")
    grid = QGridLayout(group)

    axis_names = ("Z", "Y", "X")
    sliders: list[QSlider] = []
    spins: list[QDoubleSpinBox] = []

    SCALE_MIN = 0.25
    SCALE_MAX = 5.00
    SCALE_STEP = 0.01
    slider_max = int(round((SCALE_MAX - SCALE_MIN) / SCALE_STEP))

    current = np.asarray(recommended_scale, dtype=np.float64).copy()

    def slider_to_value(pos: int) -> float:
        return SCALE_MIN + pos * SCALE_STEP

    def value_to_slider(value: float) -> int:
        clipped = float(np.clip(value, SCALE_MIN, SCALE_MAX))
        return int(round((clipped - SCALE_MIN) / SCALE_STEP))

    updating = {"active": False}

    def update_layers_and_metrics() -> None:
        scale_tuple = tuple(float(v) for v in current)
        for layer in layers:
            layer.scale = scale_tuple

        covs_t = transform_covariances(m.covariances, current)
        lengths = pca_lengths_from_covariances(covs_t)
        lm, ms, ls = axis_ratios(lengths)
        obj = morphology_objective(
            m.covariances,
            current,
            target_long_mid,
            target_mid_short,
        )
        metrics_label.setText(
            "Current [Z,Y,X]: "
            f"<b>{fmt_triplet(current, 4)}</b><br>"
            f"median long/mid={np.median(lm):.3f}, "
            f"mid/short={np.median(ms):.3f}, "
            f"long/short={np.median(ls):.3f}<br>"
            f"robust target error={obj:.6f}"
        )

    def set_axis_value(axis_i: int, value: float, source: str) -> None:
        if updating["active"]:
            return
        updating["active"] = True
        try:
            value = float(np.clip(value, SCALE_MIN, SCALE_MAX))
            current[axis_i] = value
            if source != "slider":
                sliders[axis_i].setValue(value_to_slider(value))
            if source != "spin":
                spins[axis_i].setValue(value)
        finally:
            updating["active"] = False
        update_layers_and_metrics()

    for axis_i, axis_name in enumerate(axis_names):
        grid.addWidget(QLabel(axis_name), axis_i, 0)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(slider_max)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        slider.setValue(value_to_slider(float(current[axis_i])))
        grid.addWidget(slider, axis_i, 1)

        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(SCALE_MIN, SCALE_MAX)
        spin.setSingleStep(0.01)
        spin.setValue(float(current[axis_i]))
        grid.addWidget(spin, axis_i, 2)

        slider.valueChanged.connect(
            lambda pos, i=axis_i: set_axis_value(i, slider_to_value(pos), "slider")
        )
        spin.valueChanged.connect(
            lambda value, i=axis_i: set_axis_value(i, float(value), "spin")
        )

        sliders.append(slider)
        spins.append(spin)

    layout.addWidget(group)

    button_row = QHBoxLayout()
    recommended_button = QPushButton("Use recommended")
    identity_button = QPushButton("Identity 1,1,1")
    save_button = QPushButton("Save current scale")
    button_row.addWidget(recommended_button)
    button_row.addWidget(identity_button)
    button_row.addWidget(save_button)
    layout.addLayout(button_row)

    save_status = QLabel("")
    save_status.setWordWrap(True)
    layout.addWidget(save_status)

    def set_all(values: Sequence[float]) -> None:
        updating["active"] = True
        try:
            current[:] = np.asarray(values, dtype=np.float64)
            for i in range(3):
                sliders[i].setValue(value_to_slider(float(current[i])))
                spins[i].setValue(float(current[i]))
        finally:
            updating["active"] = False
        update_layers_and_metrics()

    def save_current() -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{m.dataset_name}_xyz_scale.json"

        covs_t = transform_covariances(m.covariances, current)
        lengths = pca_lengths_from_covariances(covs_t)
        lm, ms, ls = axis_ratios(lengths)

        payload = {
            "dataset": m.dataset_name,
            "dataset_dir": str(m.dataset_dir),
            "coordinate_order": "ZYX",
            "selected_display_scale_zyx": [float(v) for v in current],
            "mathematical_recommendation_zyx": [
                float(v) for v in recommended_scale
            ],
            "target_pca_long_mid": float(target_long_mid),
            "target_pca_mid_short": float(target_mid_short),
            "measured_cells": int(m.n_cells),
            "selected_scale_metrics": {
                "median_long_mid": float(np.median(lm)),
                "median_mid_short": float(np.median(ms)),
                "median_long_short": float(np.median(ls)),
            },
            "display_only": True,
            "note": (
                "This JSON records the chosen geometric deformation. Napari "
                "layer.scale does not resample the TIFFs. Training data must be "
                "resampled separately from original raw/GT data."
            ),
            "saved_at_local": datetime.now().isoformat(timespec="seconds"),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        save_status.setText(f"Saved: <b>{path}</b>")
        print(f"\nSaved selected XYZ scale to:\n  {path}")

    recommended_button.clicked.connect(lambda: set_all(recommended_scale))
    identity_button.clicked.connect(lambda: set_all((1.0, 1.0, 1.0)))
    save_button.clicked.connect(save_current)

    update_layers_and_metrics()

    viewer.window.add_dock_widget(
        root,
        name="STIR-Net XYZ scaling",
        area="right",
    )

    # Put camera into 3-D explicitly.
    viewer.dims.ndisplay = 3

    napari.run()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Measure Drosophila GT cell anisotropy, fit a global XYZ scale, "
            "and optionally tune that scale interactively in Napari."
        )
    )
    p.add_argument(
        "--dataset",
        action="append",
        required=True,
        help=(
            "Dataset name (e.g. Drosophila_1) or full dataset directory. "
            "Repeat for multiple datasets; the fitted scale is dataset-balanced."
        ),
    )
    p.add_argument(
        "--min-voxels",
        type=int,
        default=100,
        help="Ignore GT instances smaller than this many voxels. Default: 100.",
    )
    p.add_argument(
        "--max-cells",
        type=int,
        default=2500,
        help=(
            "Maximum number of labels sampled per dataset for PCA analysis. "
            "Use 0 for all. Default: 2500."
        ),
    )
    p.add_argument(
        "--keep-boundary-cells",
        action="store_true",
        help=(
            "Include GT cells touching the outer volume boundary. By default "
            "they are rejected because cropping distorts their measured shape."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed used only when --max-cells subsamples labels.",
    )
    p.add_argument(
        "--target-long-mid",
        type=float,
        default=1.0,
        help=(
            "Desired median-like PCA long/mid morphology target. "
            "Default 1.0 (round)."
        ),
    )
    p.add_argument(
        "--target-mid-short",
        type=float,
        default=1.0,
        help=(
            "Desired median-like PCA mid/short morphology target. "
            "Default 1.0 (round)."
        ),
    )
    p.add_argument(
        "--max-relative-scale",
        type=float,
        default=4.0,
        help=(
            "Optimization guardrail for any relative axis deformation. "
            "Default: 4.0."
        ),
    )
    p.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run the mathematical analysis only; do not launch Napari.",
    )
    p.add_argument(
        "--view-index",
        type=int,
        default=0,
        help=(
            "Which --dataset entry to open in Napari after fitting. "
            "Default: 0."
        ),
    )
    p.add_argument(
        "--raw-percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(0.5, 99.8),
        help=(
            "Sampled percentiles used as Napari raw contrast limits. "
            "Default: 0.5 99.8."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory used by the viewer's 'Save current scale' button. "
            "Default: investigations/stirnet/data/_outputs/"
            "10_cell_shape_scaling"
        ),
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    if args.min_voxels < 4:
        raise ValueError("--min-voxels must be >= 4.")
    if args.max_cells < 0:
        raise ValueError("--max-cells must be >= 0.")
    if args.target_long_mid < 1.0 or args.target_mid_short < 1.0:
        raise ValueError(
            "PCA ratios are ordered long>=mid>=short, so both target ratios "
            "must be >= 1."
        )
    if not (0.0 <= args.raw_percentiles[0] < args.raw_percentiles[1] <= 100.0):
        raise ValueError("--raw-percentiles must satisfy 0 <= LOW < HIGH <= 100.")

    repo_root = repo_root_from_script()
    print(f"Repository root: {repo_root}")

    dataset_dirs = [
        resolve_dataset_dir(spec, repo_root)
        for spec in args.dataset
    ]

    max_cells = None if args.max_cells == 0 else int(args.max_cells)

    measurements = [
        measure_cells(
            d,
            min_voxels=int(args.min_voxels),
            max_cells=max_cells,
            reject_boundary=not bool(args.keep_boundary_cells),
            seed=int(args.seed) + i,
        )
        for i, d in enumerate(dataset_dirs)
    ]

    for m in measurements:
        print_dataset_summary(m)

    print(f"\n{'=' * 80}")
    print("GLOBAL XYZ MORPHOLOGY FIT")
    print(f"{'=' * 80}")
    print(
        f"Target PCA ratios: long/mid={args.target_long_mid:.3f}, "
        f"mid/short={args.target_mid_short:.3f}"
    )
    print(
        "Fitting one dataset-balanced global diagonal deformation across: "
        + ", ".join(m.dataset_name for m in measurements)
    )

    recommended = fit_global_scale(
        measurements,
        target_long_mid=float(args.target_long_mid),
        target_mid_short=float(args.target_mid_short),
        max_relative_scale=float(args.max_relative_scale),
    )

    print("\nRecommended relative DISPLAY scale [Z,Y,X]:")
    print(f"  Z = {recommended[0]:.5f}")
    print(f"  Y = {recommended[1]:.5f}")
    print(f"  X = {recommended[2]:.5f}")
    print(
        "\nNOTE: multiplying all three values by the same constant does not "
        "change cell shape. The result above is normalized so the median "
        "axis scale is 1."
    )

    for m in measurements:
        print_dataset_summary(m, recommended_scale=recommended)

    all_native = np.concatenate([m.pca_lengths for m in measurements], axis=0)
    native_ls = axis_ratios(all_native)[2]

    all_scaled_lengths = np.concatenate(
        [
            pca_lengths_from_covariances(
                transform_covariances(m.covariances, recommended)
            )
            for m in measurements
        ],
        axis=0,
    )
    scaled_ls = axis_ratios(all_scaled_lengths)[2]

    print(f"\nCombined median long/short before : {np.median(native_ls):.4f}")
    print(f"Combined median long/short after  : {np.median(scaled_ls):.4f}")

    stretched_axis = ("Z", "Y", "X")[int(np.argmax(recommended))]
    print(f"Largest recommended stretch axis  : {stretched_axis}")

    if args.no_viewer:
        return 0

    if not (0 <= args.view_index < len(measurements)):
        raise IndexError(
            f"--view-index {args.view_index} is invalid for "
            f"{len(measurements)} dataset(s)."
        )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            repo_root
            / "investigations"
            / "stirnet"
            / "data"
            / "_outputs"
            / "10_cell_shape_scaling"
        )
    else:
        output_dir = output_dir.expanduser().resolve()

    selected = measurements[int(args.view_index)]

    print(f"\nOpening Napari for: {selected.dataset_name}")
    print(
        "Napari scale is DISPLAY-ONLY. Use the controls to visually tune "
        "Z/Y/X, then click 'Save current scale'."
    )

    launch_viewer(
        selected,
        recommended_scale=recommended,
        target_long_mid=float(args.target_long_mid),
        target_mid_short=float(args.target_mid_short),
        output_dir=output_dir,
        raw_percentiles=(
            float(args.raw_percentiles[0]),
            float(args.raw_percentiles[1]),
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
