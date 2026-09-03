from __future__ import annotations

r"""
Investigation 41 — tiny-supervoxel calibration on the persisted BioHub curation cache.

Purpose
-------
Measure the pathological small-supervoxel tail before adding production
agglomeration. This investigation is READ-ONLY with respect to the preprocessed
volume: it does not modify supervoxels, final instances, Trackastra outputs, or
annotations.

Default input
-------------
    E:\data\biohub\preprocessed\train\44b6_0113de3b\

The current compact dataset-curation cache persists exactly the arrays needed
for this audit:

    movies/supervoxels.npy
    movies/final_instances.npy

For every positive supervoxel, this script measures:

    - voxel count and physical volume
    - number of 6-connected supervoxel neighbours
    - shared interface with each neighbour
    - best neighbour by PHYSICAL shared surface area
    - total supervoxel-to-supervoxel interface
    - best-interface dominance ratio
    - background contact
    - volume-boundary contact
    - overlap with final STIR-Net instances
    - whether it became an exact single-supervoxel final instance

The physical interface calculation respects BioHub anisotropy. For labels in
(Z, Y, X) order with spacing (dz, dy, dx):

    Z-neighbour face area = dy * dx
    Y-neighbour face area = dz * dx
    X-neighbour face area = dz * dy

The script also evaluates a sweep of candidate tiny-SV thresholds. For each
threshold it reports how many supervoxels would be:

    - merged (has >= 1 positive supervoxel neighbour)
    - deleted (has zero positive supervoxel neighbours)

under the proposed production policy. No merge/delete is actually performed.

Typical usage from repository root
----------------------------------
    python .\investigations\stirnet\41_tiny_supervoxel_calibration.py

Custom input:
    python .\investigations\stirnet\41_tiny_supervoxel_calibration.py ^
        --preprocessed-root E:\data\biohub\preprocessed\train\44b6_0113de3b

Quick subset:
    python .\investigations\stirnet\41_tiny_supervoxel_calibration.py ^
        --start-frame 0 --stop-frame 10
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_NAME = "41_tiny_supervoxel_calibration"
DEFAULT_PREPROCESSED_ROOT = Path(
    r"E:\data\biohub\preprocessed\train\44b6_0113de3b"
)
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)
DEFAULT_THRESHOLDS = (
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
)
DEFAULT_TOP_N = 100


# -----------------------------------------------------------------------------
# Repository / paths
# -----------------------------------------------------------------------------


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents, Path.cwd().resolve()):
        if (
            (candidate / "learned" / "stirnet").is_dir()
            and (candidate / "investigations" / "stirnet").is_dir()
            and (candidate / "dataset_curation").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    raise RuntimeError(
        "Could not resolve repository root. Run this script from inside the "
        "cell-tracking repository."
    )


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def default_output_root(preprocessed_root: Path) -> Path:
    volume_id = preprocessed_root.name
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / SCRIPT_NAME
        / volume_id
    ).resolve()


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def parse_float_triplet(value: str, *, name: str) -> tuple[float, float, float]:
    parts = tuple(float(token.strip()) for token in str(value).split(","))
    if len(parts) != 3:
        raise ValueError(f"{name} must have exactly three comma-separated values")
    if any((not math.isfinite(v)) or v <= 0.0 for v in parts):
        raise ValueError(f"{name} values must be finite and positive")
    return parts


def parse_int_list(value: str, *, name: str) -> tuple[int, ...]:
    rows = tuple(sorted({int(token.strip()) for token in str(value).split(",") if token.strip()}))
    if not rows:
        raise ValueError(f"{name} must contain at least one integer")
    if any(v <= 0 for v in rows):
        raise ValueError(f"{name} values must be positive")
    return rows


def format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds >= 3600.0:
        return f"{seconds / 3600.0:.2f} h"
    if seconds >= 60.0:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds:.1f} s"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def quantile_summary(values: Iterable[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    arr = arr[np.isfinite(arr)] if arr.size else arr
    if arr.size == 0:
        return {
            "count": 0,
            "min": None,
            "p0.1": None,
            "p0.5": None,
            "p1": None,
            "p2": None,
            "p5": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    probs = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    q = np.quantile(arr.astype(np.float64, copy=False), probs)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p0.1": float(q[0]),
        "p0.5": float(q[1]),
        "p1": float(q[2]),
        "p2": float(q[3]),
        "p5": float(q[4]),
        "p10": float(q[5]),
        "p25": float(q[6]),
        "median": float(q[7]),
        "p75": float(q[8]),
        "p90": float(q[9]),
        "p95": float(q[10]),
        "p99": float(q[11]),
        "max": float(np.max(arr)),
    }


def print_quantiles(title: str, summary: dict[str, Any], *, integerish: bool) -> None:
    print("\n" + title, flush=True)
    print("-" * 104, flush=True)
    if int(summary.get("count", 0)) == 0:
        print("count         : 0", flush=True)
        return

    keys = ("count", "min", "p0.1", "p0.5", "p1", "p2", "p5", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max")
    for key in keys:
        value = summary.get(key)
        if key == "count":
            print(f"{key:<13} : {int(value):,}", flush=True)
        elif value is None:
            print(f"{key:<13} : n/a", flush=True)
        elif integerish:
            print(f"{key:<13} : {float(value):,.2f}", flush=True)
        else:
            print(f"{key:<13} : {float(value):,.6g}", flush=True)


# -----------------------------------------------------------------------------
# Adjacency / interface measurement
# -----------------------------------------------------------------------------


def _accumulate_axis_interfaces(
    a: np.ndarray,
    b: np.ndarray,
    *,
    face_area_um2: float,
    base: int,
    background_faces: np.ndarray,
    background_area_um2: np.ndarray,
    edge_faces: dict[int, int],
    edge_area_um2: dict[int, float],
) -> None:
    """Accumulate differing 6-neighbour label interfaces for one axis."""
    diff = a != b
    if not np.any(diff):
        return

    left = np.asarray(a[diff], dtype=np.int64)
    right = np.asarray(b[diff], dtype=np.int64)

    # Positive label against background (label 0).
    left_positive = (left > 0) & (right == 0)
    if np.any(left_positive):
        ids = left[left_positive]
        counts = np.bincount(ids, minlength=background_faces.size)
        background_faces[: counts.size] += counts.astype(np.int64, copy=False)
        background_area_um2[: counts.size] += counts.astype(np.float64, copy=False) * float(face_area_um2)

    right_positive = (right > 0) & (left == 0)
    if np.any(right_positive):
        ids = right[right_positive]
        counts = np.bincount(ids, minlength=background_faces.size)
        background_faces[: counts.size] += counts.astype(np.int64, copy=False)
        background_area_um2[: counts.size] += counts.astype(np.float64, copy=False) * float(face_area_um2)

    # Positive label against a different positive label.
    positive_pair = (left > 0) & (right > 0)
    if not np.any(positive_pair):
        return

    p = left[positive_pair]
    q = right[positive_pair]
    lo = np.minimum(p, q)
    hi = np.maximum(p, q)
    keys = lo.astype(np.int64, copy=False) * int(base) + hi.astype(np.int64, copy=False)
    unique_keys, counts = np.unique(keys, return_counts=True)

    for key, count in zip(unique_keys.tolist(), counts.tolist()):
        key_i = int(key)
        count_i = int(count)
        edge_faces[key_i] = edge_faces.get(key_i, 0) + count_i
        edge_area_um2[key_i] = edge_area_um2.get(key_i, 0.0) + count_i * float(face_area_um2)


def measure_interfaces(
    labels: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError(f"Expected 3-D supervoxels, got {labels.shape}")

    max_label = int(labels.max()) if labels.size else 0
    n = max_label + 1
    base = max(n, 1)

    neighbor_count = np.zeros(n, dtype=np.int32)
    total_interface_faces = np.zeros(n, dtype=np.int64)
    total_interface_um2 = np.zeros(n, dtype=np.float64)
    best_neighbor_id = np.zeros(n, dtype=np.int32)
    best_interface_faces = np.zeros(n, dtype=np.int64)
    best_interface_um2 = np.zeros(n, dtype=np.float64)
    background_faces = np.zeros(n, dtype=np.int64)
    background_area_um2 = np.zeros(n, dtype=np.float64)
    touches_volume_boundary = np.zeros(n, dtype=bool)

    dz, dy, dx = (float(v) for v in spacing_zyx_um)
    area_z = dy * dx
    area_y = dz * dx
    area_x = dz * dy

    edge_faces: dict[int, int] = {}
    edge_area_um2: dict[int, float] = {}

    _accumulate_axis_interfaces(
        labels[:-1, :, :],
        labels[1:, :, :],
        face_area_um2=area_z,
        base=base,
        background_faces=background_faces,
        background_area_um2=background_area_um2,
        edge_faces=edge_faces,
        edge_area_um2=edge_area_um2,
    )
    _accumulate_axis_interfaces(
        labels[:, :-1, :],
        labels[:, 1:, :],
        face_area_um2=area_y,
        base=base,
        background_faces=background_faces,
        background_area_um2=background_area_um2,
        edge_faces=edge_faces,
        edge_area_um2=edge_area_um2,
    )
    _accumulate_axis_interfaces(
        labels[:, :, :-1],
        labels[:, :, 1:],
        face_area_um2=area_x,
        base=base,
        background_faces=background_faces,
        background_area_um2=background_area_um2,
        edge_faces=edge_faces,
        edge_area_um2=edge_area_um2,
    )

    # Mark labels touching any of the six outer volume faces.
    if max_label > 0:
        for face in (
            labels[0, :, :],
            labels[-1, :, :],
            labels[:, 0, :],
            labels[:, -1, :],
            labels[:, :, 0],
            labels[:, :, -1],
        ):
            ids = np.unique(face)
            ids = ids[ids > 0]
            touches_volume_boundary[ids.astype(np.int64, copy=False)] = True

    # Convert undirected edge dictionary into per-label neighbour diagnostics.
    for key, physical_area in edge_area_um2.items():
        i = int(key // base)
        j = int(key % base)
        faces = int(edge_faces[key])

        neighbor_count[i] += 1
        neighbor_count[j] += 1
        total_interface_faces[i] += faces
        total_interface_faces[j] += faces
        total_interface_um2[i] += physical_area
        total_interface_um2[j] += physical_area

        if physical_area > best_interface_um2[i]:
            best_interface_um2[i] = physical_area
            best_interface_faces[i] = faces
            best_neighbor_id[i] = j
        if physical_area > best_interface_um2[j]:
            best_interface_um2[j] = physical_area
            best_interface_faces[j] = faces
            best_neighbor_id[j] = i

    dominance = np.zeros(n, dtype=np.float64)
    valid = total_interface_um2 > 0.0
    dominance[valid] = best_interface_um2[valid] / total_interface_um2[valid]

    return {
        "neighbor_count": neighbor_count,
        "total_interface_faces": total_interface_faces,
        "total_interface_um2": total_interface_um2,
        "best_neighbor_id": best_neighbor_id,
        "best_interface_faces": best_interface_faces,
        "best_interface_um2": best_interface_um2,
        "best_interface_fraction": dominance,
        "background_interface_faces": background_faces,
        "background_interface_um2": background_area_um2,
        "touches_background": background_faces > 0,
        "touches_volume_boundary": touches_volume_boundary,
    }


# -----------------------------------------------------------------------------
# Supervoxel <-> final-instance mapping
# -----------------------------------------------------------------------------


def measure_final_mapping(
    supervoxels: np.ndarray,
    final_instances: np.ndarray,
) -> dict[str, np.ndarray]:
    sv = np.asarray(supervoxels)
    final = np.asarray(final_instances)
    if sv.shape != final.shape:
        raise ValueError(f"Supervoxel/final shape mismatch: {sv.shape} vs {final.shape}")

    max_sv = int(sv.max()) if sv.size else 0
    max_final = int(final.max()) if final.size else 0
    n_sv = max_sv + 1
    n_final = max_final + 1

    sv_counts = np.bincount(sv.reshape(-1), minlength=n_sv).astype(np.int64, copy=False)
    final_counts = np.bincount(final.reshape(-1), minlength=n_final).astype(np.int64, copy=False)

    final_overlap_count = np.zeros(n_sv, dtype=np.int32)
    dominant_final_label = np.zeros(n_sv, dtype=np.int32)
    dominant_final_voxels = np.zeros(n_sv, dtype=np.int64)
    final_background_voxels = np.zeros(n_sv, dtype=np.int64)
    final_supervoxel_count = np.zeros(n_final, dtype=np.int32)

    if max_sv <= 0:
        return {
            "sv_counts": sv_counts,
            "final_counts": final_counts,
            "final_overlap_count": final_overlap_count,
            "dominant_final_label": dominant_final_label,
            "dominant_final_voxels": dominant_final_voxels,
            "final_background_voxels": final_background_voxels,
            "final_supervoxel_count": final_supervoxel_count,
            "exact_single_sv_final_instance": np.zeros(n_sv, dtype=bool),
        }

    sv_flat = sv.reshape(-1).astype(np.int64, copy=False)
    final_flat = final.reshape(-1).astype(np.int64, copy=False)
    positive_sv = sv_flat > 0

    base = max(n_final, 1)
    keys = sv_flat[positive_sv] * int(base) + final_flat[positive_sv]
    unique_keys, counts = np.unique(keys, return_counts=True)

    for key, count in zip(unique_keys.tolist(), counts.tolist()):
        sv_id = int(key // base)
        final_id = int(key % base)
        count_i = int(count)

        if final_id <= 0:
            final_background_voxels[sv_id] += count_i
            continue

        final_overlap_count[sv_id] += 1
        final_supervoxel_count[final_id] += 1
        if count_i > dominant_final_voxels[sv_id]:
            dominant_final_voxels[sv_id] = count_i
            dominant_final_label[sv_id] = final_id

    exact_single = np.zeros(n_sv, dtype=bool)
    for sv_id in np.flatnonzero(sv_counts > 0):
        if sv_id == 0:
            continue
        final_id = int(dominant_final_label[sv_id])
        if final_id <= 0:
            continue
        sv_voxels = int(sv_counts[sv_id])
        exact_single[sv_id] = bool(
            final_overlap_count[sv_id] == 1
            and final_background_voxels[sv_id] == 0
            and dominant_final_voxels[sv_id] == sv_voxels
            and final_supervoxel_count[final_id] == 1
            and int(final_counts[final_id]) == sv_voxels
        )

    return {
        "sv_counts": sv_counts,
        "final_counts": final_counts,
        "final_overlap_count": final_overlap_count,
        "dominant_final_label": dominant_final_label,
        "dominant_final_voxels": dominant_final_voxels,
        "final_background_voxels": final_background_voxels,
        "final_supervoxel_count": final_supervoxel_count,
        "exact_single_sv_final_instance": exact_single,
    }


# -----------------------------------------------------------------------------
# Per-frame analysis
# -----------------------------------------------------------------------------


def analyze_frame(
    frame_index: int,
    supervoxels: np.ndarray,
    final_instances: np.ndarray,
    *,
    spacing_zyx_um: tuple[float, float, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()

    sv = np.asarray(supervoxels)
    final = np.asarray(final_instances)
    if sv.ndim != 3 or final.ndim != 3:
        raise ValueError(f"Expected 3-D frames, got {sv.shape} and {final.shape}")
    if sv.shape != final.shape:
        raise ValueError(f"Frame shape mismatch: {sv.shape} vs {final.shape}")

    interfaces = measure_interfaces(sv, spacing_zyx_um)
    mapping = measure_final_mapping(sv, final)

    dz, dy, dx = (float(v) for v in spacing_zyx_um)
    voxel_volume_um3 = dz * dy * dx

    sv_counts = mapping["sv_counts"]
    final_counts = mapping["final_counts"]
    sv_ids = np.flatnonzero(sv_counts > 0)
    sv_ids = sv_ids[sv_ids > 0]
    final_ids = np.flatnonzero(final_counts > 0)
    final_ids = final_ids[final_ids > 0]

    sv_rows: list[dict[str, Any]] = []
    for sv_id_raw in sv_ids.tolist():
        sv_id = int(sv_id_raw)
        voxels = int(sv_counts[sv_id])
        dominant_final = int(mapping["dominant_final_label"][sv_id])
        dominant_voxels = int(mapping["dominant_final_voxels"][sv_id])
        dominant_fraction = dominant_voxels / voxels if voxels > 0 else 0.0

        if dominant_final > 0:
            final_voxels = int(final_counts[dominant_final])
            final_sv_count = int(mapping["final_supervoxel_count"][dominant_final])
        else:
            final_voxels = 0
            final_sv_count = 0

        sv_rows.append(
            {
                "frame": int(frame_index),
                "sv_id": sv_id,
                "voxels": voxels,
                "volume_um3": float(voxels * voxel_volume_um3),
                "neighbor_count": int(interfaces["neighbor_count"][sv_id]),
                "best_neighbor_id": int(interfaces["best_neighbor_id"][sv_id]),
                "best_interface_faces": int(interfaces["best_interface_faces"][sv_id]),
                "best_interface_um2": float(interfaces["best_interface_um2"][sv_id]),
                "total_sv_interface_faces": int(interfaces["total_interface_faces"][sv_id]),
                "total_sv_interface_um2": float(interfaces["total_interface_um2"][sv_id]),
                "best_interface_fraction": float(interfaces["best_interface_fraction"][sv_id]),
                "background_interface_faces": int(interfaces["background_interface_faces"][sv_id]),
                "background_interface_um2": float(interfaces["background_interface_um2"][sv_id]),
                "touches_background": bool(interfaces["touches_background"][sv_id]),
                "touches_volume_boundary": bool(interfaces["touches_volume_boundary"][sv_id]),
                "final_label_count": int(mapping["final_overlap_count"][sv_id]),
                "dominant_final_label": dominant_final,
                "dominant_final_voxels": dominant_voxels,
                "dominant_final_fraction": float(dominant_fraction),
                "final_background_voxels": int(mapping["final_background_voxels"][sv_id]),
                "dominant_final_instance_voxels": final_voxels,
                "dominant_final_instance_volume_um3": float(final_voxels * voxel_volume_um3),
                "dominant_final_instance_supervoxel_count": final_sv_count,
                "exact_single_sv_final_instance": bool(mapping["exact_single_sv_final_instance"][sv_id]),
            }
        )

    final_rows: list[dict[str, Any]] = []
    exact_single_by_final: dict[int, int] = {}
    for sv_id_raw in sv_ids.tolist():
        sv_id = int(sv_id_raw)
        if bool(mapping["exact_single_sv_final_instance"][sv_id]):
            final_id = int(mapping["dominant_final_label"][sv_id])
            exact_single_by_final[final_id] = sv_id

    for final_id_raw in final_ids.tolist():
        final_id = int(final_id_raw)
        voxels = int(final_counts[final_id])
        final_rows.append(
            {
                "frame": int(frame_index),
                "final_id": final_id,
                "voxels": voxels,
                "volume_um3": float(voxels * voxel_volume_um3),
                "supervoxel_count": int(mapping["final_supervoxel_count"][final_id]),
                "exact_single_supervoxel": final_id in exact_single_by_final,
                "single_supervoxel_id": int(exact_single_by_final.get(final_id, 0)),
            }
        )

    sv_sizes = np.asarray([row["voxels"] for row in sv_rows], dtype=np.int64)
    final_sizes = np.asarray([row["voxels"] for row in final_rows], dtype=np.int64)
    single_sizes = np.asarray(
        [row["voxels"] for row in final_rows if row["exact_single_supervoxel"]],
        dtype=np.int64,
    )

    frame_summary = {
        "frame": int(frame_index),
        "supervoxel_count": int(len(sv_rows)),
        "final_instance_count": int(len(final_rows)),
        "exact_single_sv_final_instance_count": int(single_sizes.size),
        "supervoxel_voxels_min": int(sv_sizes.min()) if sv_sizes.size else 0,
        "supervoxel_voxels_p1": float(np.quantile(sv_sizes, 0.01)) if sv_sizes.size else 0.0,
        "supervoxel_voxels_p5": float(np.quantile(sv_sizes, 0.05)) if sv_sizes.size else 0.0,
        "supervoxel_voxels_median": float(np.median(sv_sizes)) if sv_sizes.size else 0.0,
        "final_voxels_min": int(final_sizes.min()) if final_sizes.size else 0,
        "final_voxels_p1": float(np.quantile(final_sizes, 0.01)) if final_sizes.size else 0.0,
        "final_voxels_p5": float(np.quantile(final_sizes, 0.05)) if final_sizes.size else 0.0,
        "final_voxels_median": float(np.median(final_sizes)) if final_sizes.size else 0.0,
        "analysis_seconds": float(time.perf_counter() - started),
    }

    return sv_rows, final_rows, frame_summary


# -----------------------------------------------------------------------------
# Threshold sweep and reports
# -----------------------------------------------------------------------------


def build_threshold_table(
    supervoxel_table: pd.DataFrame,
    thresholds: tuple[int, ...],
) -> pd.DataFrame:
    total_sv = int(len(supervoxel_table))
    total_voxels = int(supervoxel_table["voxels"].sum()) if total_sv else 0
    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        tiny = supervoxel_table[supervoxel_table["voxels"] <= int(threshold)]
        connected = tiny[tiny["neighbor_count"] > 0]
        isolated = tiny[tiny["neighbor_count"] == 0]

        best = connected["best_interface_fraction"].to_numpy(dtype=np.float64, copy=False)
        rows.append(
            {
                "threshold_voxels_le": int(threshold),
                "tiny_supervoxels": int(len(tiny)),
                "tiny_fraction_of_supervoxels": float(len(tiny) / total_sv) if total_sv else 0.0,
                "tiny_voxels": int(tiny["voxels"].sum()),
                "tiny_fraction_of_foreground_voxels": float(tiny["voxels"].sum() / total_voxels) if total_voxels else 0.0,
                "would_merge_connected": int(len(connected)),
                "would_delete_zero_sv_neighbours": int(len(isolated)),
                "one_sv_neighbour": int((tiny["neighbor_count"] == 1).sum()),
                "two_or_more_sv_neighbours": int((tiny["neighbor_count"] >= 2).sum()),
                "touches_background": int(tiny["touches_background"].sum()),
                "touches_volume_boundary": int(tiny["touches_volume_boundary"].sum()),
                "exact_single_sv_final_instances": int(tiny["exact_single_sv_final_instance"].sum()),
                "best_interface_fraction_ge_0_50": int((best >= 0.50).sum()),
                "best_interface_fraction_ge_0_75": int((best >= 0.75).sum()),
                "best_interface_fraction_ge_0_90": int((best >= 0.90).sum()),
                "best_interface_fraction_median_connected": float(np.median(best)) if best.size else float("nan"),
                "best_interface_fraction_p10_connected": float(np.quantile(best, 0.10)) if best.size else float("nan"),
                "best_interface_fraction_p90_connected": float(np.quantile(best, 0.90)) if best.size else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def build_size_histogram(supervoxel_table: pd.DataFrame) -> pd.DataFrame:
    if supervoxel_table.empty:
        return pd.DataFrame(columns=["voxels", "supervoxel_count", "exact_single_sv_final_instance_count"])

    grouped = (
        supervoxel_table.groupby("voxels", as_index=False)
        .agg(
            supervoxel_count=("sv_id", "size"),
            exact_single_sv_final_instance_count=("exact_single_sv_final_instance", "sum"),
        )
        .sort_values("voxels")
        .reset_index(drop=True)
    )
    grouped["exact_single_sv_final_instance_count"] = grouped["exact_single_sv_final_instance_count"].astype(np.int64)
    return grouped


def print_threshold_table(table: pd.DataFrame) -> None:
    print("\nCANDIDATE TINY-SUPERVOXEL THRESHOLDS", flush=True)
    print("-" * 104, flush=True)
    print(
        f"{'<=vox':>7} {'tiny':>8} {'SV%':>7} {'voxel%':>8} {'merge':>8} "
        f"{'delete':>8} {'1-neigh':>8} {'2+-neigh':>9} {'single-final':>13} {'dom>=.90':>9}",
        flush=True,
    )
    for row in table.itertuples(index=False):
        print(
            f"{int(row.threshold_voxels_le):7d} "
            f"{int(row.tiny_supervoxels):8,d} "
            f"{100.0 * float(row.tiny_fraction_of_supervoxels):6.2f}% "
            f"{100.0 * float(row.tiny_fraction_of_foreground_voxels):7.4f}% "
            f"{int(row.would_merge_connected):8,d} "
            f"{int(row.would_delete_zero_sv_neighbours):8,d} "
            f"{int(row.one_sv_neighbour):8,d} "
            f"{int(row.two_or_more_sv_neighbours):9,d} "
            f"{int(row.exact_single_sv_final_instances):13,d} "
            f"{int(row.best_interface_fraction_ge_0_90):9,d}",
            flush=True,
        )


def print_smallest(title: str, table: pd.DataFrame, columns: list[str], n: int) -> None:
    print("\n" + title, flush=True)
    print("-" * 104, flush=True)
    if table.empty:
        print("none", flush=True)
        return
    shown = table.sort_values(["voxels", "frame"]).head(int(n))
    with pd.option_context("display.max_rows", int(n), "display.max_columns", None, "display.width", 220):
        print(shown[columns].to_string(index=False), flush=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Investigation 41: calibrate pathological tiny BioHub supervoxels from the persisted curation cache."
    )
    parser.add_argument(
        "--preprocessed-root",
        default=str(DEFAULT_PREPROCESSED_ROOT),
        help="Volume preprocessed root containing movies/supervoxels.npy and movies/final_instances.npy",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Default: runs/stirnet/evaluation/41_tiny_supervoxel_calibration/<volume_id>",
    )
    parser.add_argument(
        "--spacing-zyx",
        default=",".join(str(v) for v in DEFAULT_SPACING_ZYX_UM),
        help="Physical voxel spacing in micrometres, Z,Y,X",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(v) for v in DEFAULT_THRESHOLDS),
        help="Comma-separated candidate voxel thresholds; statistics use size <= threshold",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--stop-frame",
        type=int,
        default=None,
        help="Exclusive stop frame. Default: analyze all remaining frames.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of smallest objects to save/print in focused tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocessed_root = resolve(args.preprocessed_root)
    spacing_zyx_um = parse_float_triplet(args.spacing_zyx, name="--spacing-zyx")
    thresholds = parse_int_list(args.thresholds, name="--thresholds")

    if args.top <= 0:
        raise ValueError("--top must be positive")

    supervoxel_path = preprocessed_root / "movies" / "supervoxels.npy"
    final_path = preprocessed_root / "movies" / "final_instances.npy"
    if not supervoxel_path.is_file():
        raise FileNotFoundError(supervoxel_path)
    if not final_path.is_file():
        raise FileNotFoundError(final_path)

    output_root = resolve(args.output) if args.output is not None else default_output_root(preprocessed_root)
    output_root.mkdir(parents=True, exist_ok=True)

    supervoxel_movie = np.load(supervoxel_path, mmap_mode="r", allow_pickle=False)
    final_movie = np.load(final_path, mmap_mode="r", allow_pickle=False)

    if supervoxel_movie.ndim != 4 or final_movie.ndim != 4:
        raise ValueError(
            f"Expected T,Z,Y,X movies; got {supervoxel_movie.shape} and {final_movie.shape}"
        )
    if supervoxel_movie.shape != final_movie.shape:
        raise ValueError(
            f"Supervoxel/final movie shape mismatch: {supervoxel_movie.shape} vs {final_movie.shape}"
        )

    frame_count = int(supervoxel_movie.shape[0])
    start = int(args.start_frame)
    stop = frame_count if args.stop_frame is None else int(args.stop_frame)
    if start < 0 or stop < 0 or start >= stop or stop > frame_count:
        raise ValueError(f"Invalid frame range [{start}, {stop}) for frame_count={frame_count}")

    dz, dy, dx = spacing_zyx_um
    voxel_volume_um3 = dz * dy * dx
    area_z = dy * dx
    area_y = dz * dx
    area_x = dz * dy

    print("=" * 112, flush=True)
    print("INVESTIGATION 41 — TINY SUPERVOXEL CALIBRATION", flush=True)
    print("=" * 112, flush=True)
    print(f"preprocessed : {preprocessed_root}", flush=True)
    print(f"supervoxels  : {supervoxel_path}", flush=True)
    print(f"final        : {final_path}", flush=True)
    print(f"movie shape  : {tuple(int(v) for v in supervoxel_movie.shape)}", flush=True)
    print(f"dtype        : SV={supervoxel_movie.dtype} final={final_movie.dtype}", flush=True)
    print(f"frames       : [{start}, {stop}) ({stop - start} frames)", flush=True)
    print(f"spacing ZYX  : {spacing_zyx_um} um", flush=True)
    print(f"voxel volume : {voxel_volume_um3:.9f} um^3", flush=True)
    print(
        f"face areas   : Z-neigh={area_z:.9f}, Y-neigh={area_y:.9f}, X-neigh={area_x:.9f} um^2",
        flush=True,
    )
    print(f"thresholds   : {thresholds}", flush=True)
    print(f"output       : {output_root}", flush=True)
    print("mode         : READ-ONLY calibration; no labels are modified", flush=True)
    print("=" * 112, flush=True)

    run_started = time.perf_counter()
    all_sv_rows: list[dict[str, Any]] = []
    all_final_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []

    for t in range(start, stop):
        frame_started = time.perf_counter()
        sv_rows, final_rows, frame_summary = analyze_frame(
            t,
            np.asarray(supervoxel_movie[t]),
            np.asarray(final_movie[t]),
            spacing_zyx_um=spacing_zyx_um,
        )
        all_sv_rows.extend(sv_rows)
        all_final_rows.extend(final_rows)
        frame_rows.append(frame_summary)

        elapsed = time.perf_counter() - frame_started
        tiny32 = sum(1 for row in sv_rows if int(row["voxels"]) <= 32)
        tiny64 = sum(1 for row in sv_rows if int(row["voxels"]) <= 64)
        isolated64 = sum(
            1
            for row in sv_rows
            if int(row["voxels"]) <= 64 and int(row["neighbor_count"]) == 0
        )
        single64 = sum(
            1
            for row in sv_rows
            if int(row["voxels"]) <= 64 and bool(row["exact_single_sv_final_instance"])
        )
        print(
            f"[t={t:03d}] SV={len(sv_rows):4d} final={len(final_rows):4d} | "
            f"<=32={tiny32:3d} <=64={tiny64:3d} zero-neigh<=64={isolated64:3d} "
            f"single-final<=64={single64:3d} | {elapsed:.2f}s",
            flush=True,
        )

    supervoxel_table = pd.DataFrame(all_sv_rows)
    final_table = pd.DataFrame(all_final_rows)
    frame_table = pd.DataFrame(frame_rows)
    threshold_table = build_threshold_table(supervoxel_table, thresholds)
    size_histogram = build_size_histogram(supervoxel_table)

    # Focused subsets.
    smallest_supervoxels = supervoxel_table.sort_values(["voxels", "frame", "sv_id"]).head(int(args.top)).copy()
    smallest_single_final = (
        supervoxel_table[supervoxel_table["exact_single_sv_final_instance"]]
        .sort_values(["voxels", "frame", "sv_id"])
        .head(int(args.top))
        .copy()
    )
    zero_neighbour = (
        supervoxel_table[supervoxel_table["neighbor_count"] == 0]
        .sort_values(["voxels", "frame", "sv_id"])
        .copy()
    )

    # Persist tables before printing the final report.
    atomic_csv(output_root / "supervoxels.csv", supervoxel_table)
    atomic_csv(output_root / "final_instances.csv", final_table)
    atomic_csv(output_root / "frame_summary.csv", frame_table)
    atomic_csv(output_root / "threshold_summary.csv", threshold_table)
    atomic_csv(output_root / "supervoxel_size_histogram.csv", size_histogram)
    atomic_csv(output_root / "smallest_supervoxels.csv", smallest_supervoxels)
    atomic_csv(output_root / "smallest_single_sv_final_instances.csv", smallest_single_final)
    atomic_csv(output_root / "zero_neighbour_supervoxels.csv", zero_neighbour)

    sv_quantiles = quantile_summary(supervoxel_table["voxels"].to_numpy(dtype=np.float64, copy=False))
    sv_volume_quantiles = quantile_summary(supervoxel_table["volume_um3"].to_numpy(dtype=np.float64, copy=False))
    final_quantiles = quantile_summary(final_table["voxels"].to_numpy(dtype=np.float64, copy=False))

    single_final_table = final_table[final_table["exact_single_supervoxel"]]
    multi_final_table = final_table[final_table["supervoxel_count"] >= 2]
    single_final_quantiles = quantile_summary(single_final_table["voxels"].to_numpy(dtype=np.float64, copy=False))
    multi_final_quantiles = quantile_summary(multi_final_table["voxels"].to_numpy(dtype=np.float64, copy=False))

    connected = supervoxel_table[supervoxel_table["neighbor_count"] > 0]
    dominance_quantiles = quantile_summary(connected["best_interface_fraction"].to_numpy(dtype=np.float64, copy=False))

    summary = {
        "script": SCRIPT_NAME,
        "preprocessed_root": str(preprocessed_root),
        "supervoxels_path": str(supervoxel_path),
        "final_instances_path": str(final_path),
        "output_root": str(output_root),
        "movie_shape_tzyx": [int(v) for v in supervoxel_movie.shape],
        "frame_range": {"start": start, "stop_exclusive": stop},
        "spacing_zyx_um": [float(v) for v in spacing_zyx_um],
        "voxel_volume_um3": float(voxel_volume_um3),
        "face_area_um2": {
            "z_neighbour": float(area_z),
            "y_neighbour": float(area_y),
            "x_neighbour": float(area_x),
        },
        "candidate_thresholds_voxels_le": [int(v) for v in thresholds],
        "counts": {
            "supervoxels": int(len(supervoxel_table)),
            "final_instances": int(len(final_table)),
            "zero_neighbour_supervoxels": int((supervoxel_table["neighbor_count"] == 0).sum()),
            "exact_single_sv_final_instances": int(supervoxel_table["exact_single_sv_final_instance"].sum()),
        },
        "supervoxel_voxels": sv_quantiles,
        "supervoxel_volume_um3": sv_volume_quantiles,
        "final_instance_voxels_all": final_quantiles,
        "final_instance_voxels_exact_single_supervoxel": single_final_quantiles,
        "final_instance_voxels_multi_supervoxel": multi_final_quantiles,
        "best_interface_fraction_connected_supervoxels": dominance_quantiles,
        "threshold_summary": threshold_table.to_dict(orient="records"),
        "elapsed_seconds": float(time.perf_counter() - run_started),
        "read_only": True,
    }
    atomic_json(output_root / "summary.json", summary)

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 41 — GLOBAL RESULTS", flush=True)
    print("=" * 112, flush=True)

    print_quantiles("SUPERVOXEL SIZE — VOXELS", sv_quantiles, integerish=True)
    print_quantiles("SUPERVOXEL SIZE — PHYSICAL VOLUME (um^3)", sv_volume_quantiles, integerish=False)
    print_quantiles("FINAL INSTANCE SIZE — ALL — VOXELS", final_quantiles, integerish=True)
    print_quantiles(
        "FINAL INSTANCE SIZE — EXACT SINGLE-SUPERVOXEL INSTANCES — VOXELS",
        single_final_quantiles,
        integerish=True,
    )
    print_quantiles(
        "FINAL INSTANCE SIZE — MULTI-SUPERVOXEL INSTANCES — VOXELS",
        multi_final_quantiles,
        integerish=True,
    )
    print_quantiles(
        "BEST PHYSICAL INTERFACE FRACTION — CONNECTED SUPERVOXELS",
        dominance_quantiles,
        integerish=False,
    )

    print_threshold_table(threshold_table)

    print_smallest(
        "SMALLEST SUPERVOXELS",
        smallest_supervoxels,
        [
            "frame",
            "sv_id",
            "voxels",
            "volume_um3",
            "neighbor_count",
            "best_neighbor_id",
            "best_interface_um2",
            "best_interface_fraction",
            "touches_background",
            "touches_volume_boundary",
            "dominant_final_label",
            "dominant_final_instance_voxels",
            "dominant_final_instance_supervoxel_count",
            "exact_single_sv_final_instance",
        ],
        min(int(args.top), 30),
    )

    print_smallest(
        "SMALLEST SUPERVOXELS THAT BECAME EXACT SINGLE-SV FINAL INSTANCES",
        smallest_single_final,
        [
            "frame",
            "sv_id",
            "voxels",
            "volume_um3",
            "neighbor_count",
            "best_neighbor_id",
            "best_interface_um2",
            "best_interface_fraction",
            "touches_background",
            "touches_volume_boundary",
            "dominant_final_label",
        ],
        min(int(args.top), 30),
    )

    print("\nOUTPUT FILES", flush=True)
    print("-" * 104, flush=True)
    for name in (
        "summary.json",
        "threshold_summary.csv",
        "supervoxels.csv",
        "final_instances.csv",
        "frame_summary.csv",
        "supervoxel_size_histogram.csv",
        "smallest_supervoxels.csv",
        "smallest_single_sv_final_instances.csv",
        "zero_neighbour_supervoxels.csv",
    ):
        print(f"  {output_root / name}", flush=True)

    print("\nINTERPRETATION CONTRACT", flush=True)
    print("-" * 104, flush=True)
    print(
        "This investigation does NOT choose or apply a production threshold. "
        "Use the measured size tail, the single-SV final-instance tail, and the "
        "threshold sweep to select a conservative cutoff before implementing "
        "tiny-supervoxel agglomeration.",
        flush=True,
    )
    print(
        "Proposed later production policy: tiny + positive neighbour(s) -> merge "
        "to neighbour with maximum physical shared interface; tiny + zero positive "
        "neighbours -> background/delete.",
        flush=True,
    )
    print(f"\ncompleted in {format_seconds(time.perf_counter() - run_started)}", flush=True)


if __name__ == "__main__":
    main()
