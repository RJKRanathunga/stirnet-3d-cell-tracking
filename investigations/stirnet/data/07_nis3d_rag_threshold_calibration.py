from __future__ import annotations

"""
STIR-Net Investigation 06 — RAG merge-threshold calibration.

Purpose
-------
Investigation 05 showed that the learned spatial RAG is strongly discriminative,
but the production merge threshold 0.50 is too permissive.  This investigation
calibrates ONLY the graph-partition threshold.  It does not rerun STIR-Net,
watershed, geometry, or target generation.

It consumes the saved per-crop ``spatial_eval.npz`` artifacts produced by:

    investigations/stirnet/data/05_nis3d_spatial_checkpoint_eval.py

Those files already contain, for each deterministic crop:

    GT labels
    supervision-valid mask
    watershed supervoxels
    RAG edge_index
    learned edge probabilities
    GT merge/cut edge targets
    RAG edge validity
    node purity / GT support / dominant GT

Because the network outputs are frozen, the entire fine calibration is cheap and
normally runs on CPU in seconds.

Default calibration
-------------------
Fine sweep:
    0.800 .. 0.995 in steps of 0.005

Reference thresholds additionally evaluated:
    0.50, 0.90

Selection objective:
    1. maximize mean instance F1 @ IoU 0.50
       (harmonic mean of GT recall and predicted-instance precision)
    2. maximize mean GT recall @ IoU 0.75
    3. maximize mean best-GT IoU
    4. minimize predicted-instance-count bias versus GT

This prevents the calibration from choosing an excessively high threshold merely
because it reduces false merges: too high a threshold leaves cells split into
many supervoxels and hurts predicted-instance precision.

Outputs
-------
runs/stirnet/evaluation/06_nis3d_rag_threshold_calibration/<timestamp>/
    calibration_summary.json
    threshold_aggregate.csv
    per_crop_threshold_metrics.csv
    residual_edges.csv
    recommended_threshold.txt
    rag_threshold_calibration.png

``residual_edges.csv`` is intentionally produced for Investigation 07.  It lists
valid false-merge and false-cut edges at the selected threshold, sorted by how
confidently wrong the model is.

Typical usage
-------------
From the repository root:

    python investigations/stirnet/data/07_nis3d_rag_threshold_calibration.py

Use a particular Investigation-05 run:

    python investigations/stirnet/data/07_nis3d_rag_threshold_calibration.py \
        --source-run runs/stirnet/evaluation/05_nis3d_spatial_checkpoint_eval/checkpoint_step_000600_YYYYMMDD_HHMMSS

Wider/finer sweep:

    python investigations/stirnet/data/07_nis3d_rag_threshold_calibration.py \
        --threshold-min 0.85 \
        --threshold-max 0.999 \
        --threshold-step 0.001
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_NAME = "06_nis3d_rag_threshold_calibration"
SOURCE_STAGE = "05_nis3d_spatial_checkpoint_eval"


# ======================================================================================
# Repository / filesystem
# ======================================================================================


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "learned").is_dir() and (candidate / "src").is_dir():
            return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "learned").is_dir() and (cwd / "src").is_dir():
        return cwd
    raise RuntimeError(
        "Could not resolve repository root. Run from the cell-tracking repository "
        "or place this script under investigations/stirnet/data/."
    )


ROOT = _repo_root()


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _latest_stage05_run() -> Path:
    base = ROOT / "runs" / "stirnet" / "evaluation" / SOURCE_STAGE
    if not base.exists():
        raise FileNotFoundError(
            f"Investigation-05 output directory does not exist:\n  {base}\n\n"
            "Run 05_nis3d_spatial_checkpoint_eval.py first."
        )

    candidates = [
        path
        for path in base.iterdir()
        if path.is_dir()
        and path.name.startswith("checkpoint_step_")
        and (path / "summary.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No completed Investigation-05 run with summary.json was found under:\n  {base}"
        )

    # Prefer the run with the largest checkpoint step, then latest mtime.
    def key(path: Path) -> tuple[int, float]:
        step = -1
        try:
            payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            step = int(payload.get("primary_step", -1))
        except Exception:
            pass
        return step, path.stat().st_mtime

    return max(candidates, key=key)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    preferred = [
        "threshold",
        "sample",
        "crop_index",
        "candidate_type",
        "edge_kind",
        "edge_row",
        "probability",
    ]
    keys = [key for key in preferred if any(key in row for row in rows)]
    seen = set(keys)
    for key in sorted({key for row in rows for key in row}):
        if key not in seen:
            keys.append(key)
            seen.add(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            cooked = {}
            for key in keys:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    cooked[key] = json.dumps(_jsonable(value))
                else:
                    cooked[key] = value
            writer.writerow(cooked)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


# ======================================================================================
# Data model
# ======================================================================================


@dataclass
class CropArtifact:
    path: Path
    sample: str
    crop_index: int
    candidate_type: str
    gt: np.ndarray
    valid: np.ndarray
    supervoxels: np.ndarray
    edge_index: np.ndarray
    edge_probability: np.ndarray
    edge_target: np.ndarray
    edge_valid: np.ndarray
    node_purity: np.ndarray
    node_gt_support: np.ndarray
    node_dominant_gt: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.supervoxels.max(initial=0))

    @property
    def gt_count(self) -> int:
        labels = np.unique(self.gt[self.valid])
        return int(np.count_nonzero(labels > 0))


REQUIRED_KEYS = (
    "gt_labels",
    "supervision_valid",
    "primary_supervoxels",
    "edge_index",
    "edge_probability",
    "edge_target",
    "edge_valid",
    "node_purity",
    "node_gt_support",
    "node_dominant_gt",
)


def _parse_crop_identity(npz_path: Path) -> tuple[str, int, str]:
    # .../<sample>/crop_XX_<type>/spatial_eval.npz
    sample = npz_path.parent.parent.name
    name = npz_path.parent.name
    if not name.startswith("crop_"):
        raise ValueError(f"Unexpected crop directory name: {name}")
    pieces = name.split("_", 2)
    if len(pieces) < 3:
        raise ValueError(f"Could not parse crop identity from: {name}")
    return sample, int(pieces[1]), pieces[2]


def _load_crop(npz_path: Path) -> CropArtifact:
    with np.load(npz_path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise KeyError(
                f"{npz_path} is missing Investigation-05 arrays: {missing}"
            )

        sample, crop_index, candidate_type = _parse_crop_identity(npz_path)
        crop = CropArtifact(
            path=npz_path,
            sample=sample,
            crop_index=crop_index,
            candidate_type=candidate_type,
            gt=np.asarray(data["gt_labels"], dtype=np.int64),
            valid=np.asarray(data["supervision_valid"], dtype=bool),
            supervoxels=np.asarray(data["primary_supervoxels"], dtype=np.int64),
            edge_index=np.asarray(data["edge_index"], dtype=np.int64),
            edge_probability=np.asarray(data["edge_probability"], dtype=np.float64),
            edge_target=np.asarray(data["edge_target"], dtype=bool),
            edge_valid=np.asarray(data["edge_valid"], dtype=bool),
            node_purity=np.asarray(data["node_purity"], dtype=np.float64),
            node_gt_support=np.asarray(data["node_gt_support"], dtype=np.float64),
            node_dominant_gt=np.asarray(data["node_dominant_gt"], dtype=np.int64),
        )

    if crop.edge_index.ndim != 2 or crop.edge_index.shape[0] != 2:
        raise ValueError(
            f"{npz_path}: edge_index must have shape [2,E], got {crop.edge_index.shape}"
        )
    edge_count = crop.edge_index.shape[1]
    for name, array in (
        ("edge_probability", crop.edge_probability),
        ("edge_target", crop.edge_target),
        ("edge_valid", crop.edge_valid),
    ):
        if array.shape != (edge_count,):
            raise ValueError(
                f"{npz_path}: {name} shape {array.shape} does not match E={edge_count}"
            )

    node_count = crop.node_count
    if node_count <= 0:
        raise ValueError(f"{npz_path}: crop contains no watershed supervoxels")
    if crop.edge_index.size:
        if int(crop.edge_index.min()) < 0 or int(crop.edge_index.max()) >= node_count:
            raise ValueError(
                f"{npz_path}: edge_index expects node ids [0,{node_count - 1}], "
                f"got [{crop.edge_index.min()},{crop.edge_index.max()}]"
            )

    for name, array in (
        ("node_purity", crop.node_purity),
        ("node_gt_support", crop.node_gt_support),
        ("node_dominant_gt", crop.node_dominant_gt),
    ):
        if len(array) < node_count:
            raise ValueError(
                f"{npz_path}: {name} has {len(array)} nodes but supervoxels use {node_count}"
            )

    return crop


def _discover_crops(source_run: Path) -> list[CropArtifact]:
    paths = sorted(source_run.glob("*/crop_*/spatial_eval.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No per-crop spatial_eval.npz files were found under:\n  {source_run}"
        )
    crops = [_load_crop(path) for path in paths]
    crops.sort(key=lambda crop: (crop.sample, crop.crop_index))
    return crops


# ======================================================================================
# Graph partitioning
# ======================================================================================


class _UnionFind:
    def __init__(self, count: int):
        self.parent = np.arange(count, dtype=np.int64)
        self.rank = np.zeros(count, dtype=np.int8)

    def find(self, node: int) -> int:
        parent = self.parent
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            nxt = int(parent[node])
            parent[node] = root
            node = nxt
        return root

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _partition_at_threshold(crop: CropArtifact, threshold: float) -> np.ndarray:
    """
    Investigation 05 evaluates one crop per forward, therefore RAG node ids are
    zero-based within the crop and correspond to contiguous supervoxel labels
    1..N.  Reproduce production connected-component agglomeration on CPU.
    """
    uf = _UnionFind(crop.node_count)
    merge_edges = crop.edge_probability >= float(threshold)

    edge_rows = np.flatnonzero(merge_edges)
    for edge_row in edge_rows:
        a = int(crop.edge_index[0, edge_row])
        b = int(crop.edge_index[1, edge_row])
        uf.union(a, b)

    root_to_label: dict[int, int] = {}
    node_to_partition = np.zeros(crop.node_count + 1, dtype=np.int64)

    next_label = 1
    for node in range(crop.node_count):
        root = uf.find(node)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        # node i represents supervoxel label i+1.
        node_to_partition[node + 1] = root_to_label[root]

    max_sv = int(crop.supervoxels.max(initial=0))
    if max_sv >= len(node_to_partition):
        raise RuntimeError(
            f"Supervoxel label {max_sv} exceeds node mapping size {len(node_to_partition) - 1}"
        )
    return node_to_partition[crop.supervoxels]


# ======================================================================================
# Instance metrics
# ======================================================================================


def _contingency(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = pred[valid].astype(np.int64, copy=False)
    g = gt[valid].astype(np.int64, copy=False)

    pred_ids = np.unique(p[p > 0])
    gt_ids = np.unique(g[g > 0])
    if pred_ids.size == 0 or gt_ids.size == 0:
        return (
            np.zeros((pred_ids.size, gt_ids.size), dtype=np.int64),
            pred_ids,
            gt_ids,
        )

    pred_index = {int(label): i for i, label in enumerate(pred_ids)}
    gt_index = {int(label): i for i, label in enumerate(gt_ids)}

    mask = (p > 0) & (g > 0)
    pp = p[mask]
    gg = g[mask]

    matrix = np.zeros((pred_ids.size, gt_ids.size), dtype=np.int64)
    # Compact crop sizes make this pair accumulation sufficiently cheap.
    for pred_id, gt_id in zip(pp.tolist(), gg.tolist()):
        matrix[pred_index[pred_id], gt_index[gt_id]] += 1

    return matrix, pred_ids, gt_ids


def _foreground_dice(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    p = (pred > 0)[valid]
    g = (gt > 0)[valid]
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else 2.0 * tp / denominator


def _harmonic(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    return 0.0 if a + b == 0 else 2.0 * a * b / (a + b)


def _partition_metrics(
    pred: np.ndarray,
    crop: CropArtifact,
    *,
    meaningful_pred_fraction: float = 0.10,
    meaningful_gt_fraction: float = 0.10,
) -> dict[str, float | int]:
    inter, pred_ids, gt_ids = _contingency(pred, crop.gt, crop.valid)
    n_pred = int(pred_ids.size)
    n_gt = int(gt_ids.size)

    result: dict[str, float | int] = {
        "pred_instance_count": n_pred,
        "gt_instance_count": n_gt,
        "instance_count_ratio": (n_pred / n_gt) if n_gt else float("nan"),
        "instance_count_abs_relative_error": (
            abs(n_pred - n_gt) / n_gt if n_gt else float("nan")
        ),
        "foreground_dice": _foreground_dice(pred, crop.gt, crop.valid),
    }

    if n_pred == 0 or n_gt == 0:
        gt_recall50 = 0.0 if n_gt else float("nan")
        pred_precision50 = 0.0 if n_pred else float("nan")
        result.update(
            mean_best_gt_iou=0.0 if n_gt else float("nan"),
            mean_best_pred_iou=0.0 if n_pred else float("nan"),
            mean_best_gt_coverage=0.0 if n_gt else float("nan"),
            gt_recall_iou50=gt_recall50,
            gt_recall_iou75=0.0 if n_gt else float("nan"),
            pred_precision_iou50=pred_precision50,
            instance_f1_iou50=_harmonic(gt_recall50, pred_precision50),
            gt_missing_fraction=1.0 if n_gt else float("nan"),
            merged_pred_count=0,
            merged_pred_fraction=0.0 if n_pred else float("nan"),
            split_gt_count=0,
            split_gt_fraction=0.0 if n_gt else float("nan"),
        )
        return result

    pred_counts = inter.sum(axis=1, keepdims=True)
    gt_counts = inter.sum(axis=0, keepdims=True)

    # Contingency excludes pred-only / GT-only voxels, so recover exact valid
    # label voxel counts for proper IoU denominators.
    p_valid = pred[crop.valid]
    g_valid = crop.gt[crop.valid]
    pred_total = np.asarray(
        [np.count_nonzero(p_valid == label) for label in pred_ids],
        dtype=np.float64,
    )[:, None]
    gt_total = np.asarray(
        [np.count_nonzero(g_valid == label) for label in gt_ids],
        dtype=np.float64,
    )[None, :]

    inter_f = inter.astype(np.float64)
    union = pred_total + gt_total - inter_f
    iou = np.divide(
        inter_f,
        np.maximum(union, 1.0),
        out=np.zeros_like(inter_f),
        where=union > 0,
    )
    coverage = inter_f / np.maximum(gt_total, 1.0)
    pred_fraction = inter_f / np.maximum(pred_total, 1.0)

    best_gt_iou = iou.max(axis=0)
    best_pred_iou = iou.max(axis=1)
    best_gt_coverage = coverage.max(axis=0)

    meaningful = (
        (inter_f > 0)
        & (pred_fraction >= meaningful_pred_fraction)
        & (coverage >= meaningful_gt_fraction)
    )
    merged = meaningful.sum(axis=1) >= 2
    split = meaningful.sum(axis=0) >= 2

    gt_recall50 = float(np.mean(best_gt_iou >= 0.50))
    pred_precision50 = float(np.mean(best_pred_iou >= 0.50))

    result.update(
        mean_best_gt_iou=float(best_gt_iou.mean()),
        mean_best_pred_iou=float(best_pred_iou.mean()),
        mean_best_gt_coverage=float(best_gt_coverage.mean()),
        gt_recall_iou50=gt_recall50,
        gt_recall_iou75=float(np.mean(best_gt_iou >= 0.75)),
        pred_precision_iou50=pred_precision50,
        instance_f1_iou50=_harmonic(gt_recall50, pred_precision50),
        gt_missing_fraction=float(np.mean(best_gt_coverage < 0.25)),
        merged_pred_count=int(merged.sum()),
        merged_pred_fraction=float(merged.mean()),
        split_gt_count=int(split.sum()),
        split_gt_fraction=float(split.mean()),
    )
    return result


# ======================================================================================
# Edge metrics
# ======================================================================================


def _edge_metrics(crop: CropArtifact, threshold: float) -> dict[str, float | int]:
    valid = crop.edge_valid
    probability = crop.edge_probability[valid]
    truth = crop.edge_target[valid]
    pred = probability >= float(threshold)

    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))

    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")

    return {
        "valid_edge_count": int(valid.sum()),
        "true_merge_count": tp,
        "false_merge_count": fp,
        "false_cut_count": fn,
        "true_cut_count": tn,
        "edge_merge_precision": precision,
        "edge_merge_recall": recall,
        "edge_merge_f1": _harmonic(precision, recall),
        "edge_cut_specificity": specificity,
    }


# ======================================================================================
# Calibration sweep
# ======================================================================================


def _threshold_values(
    minimum: float,
    maximum: float,
    step: float,
    reference: Iterable[float],
) -> tuple[float, ...]:
    if not 0.0 < minimum < 1.0:
        raise ValueError("--threshold-min must be in (0,1)")
    if not 0.0 < maximum < 1.0:
        raise ValueError("--threshold-max must be in (0,1)")
    if maximum < minimum:
        raise ValueError("--threshold-max must be >= --threshold-min")
    if step <= 0:
        raise ValueError("--threshold-step must be > 0")

    count = int(math.floor((maximum - minimum) / step + 1e-9))
    values = [minimum + i * step for i in range(count + 1)]
    if not values or values[-1] < maximum - step * 0.25:
        values.append(maximum)

    values.extend(reference)
    return tuple(
        sorted(
            {
                round(float(value), 6)
                for value in values
                if 0.0 < float(value) < 1.0
            }
        )
    )


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        float(row[key])
        for row in rows
        if key in row and isinstance(row[key], (int, float, np.number))
        and math.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else float("nan")


def _sum(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(int(row.get(key, 0)) for row in rows))


def _aggregate_threshold(
    threshold: float,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    # Partition metrics are macro-averaged across deterministic crops so one
    # dense crop does not dominate the selected operating point.
    result = {
        "threshold": float(threshold),
        "crop_count": len(rows),
    }

    mean_keys = (
        "foreground_dice",
        "mean_best_gt_iou",
        "mean_best_pred_iou",
        "mean_best_gt_coverage",
        "gt_recall_iou50",
        "gt_recall_iou75",
        "pred_precision_iou50",
        "instance_f1_iou50",
        "gt_missing_fraction",
        "merged_pred_fraction",
        "split_gt_fraction",
        "instance_count_ratio",
        "instance_count_abs_relative_error",
        "edge_merge_precision",
        "edge_merge_recall",
        "edge_merge_f1",
        "edge_cut_specificity",
    )
    for key in mean_keys:
        result[f"mean_{key}"] = _mean(rows, key)

    sum_keys = (
        "pred_instance_count",
        "gt_instance_count",
        "valid_edge_count",
        "true_merge_count",
        "false_merge_count",
        "false_cut_count",
        "true_cut_count",
    )
    for key in sum_keys:
        result[f"total_{key}"] = _sum(rows, key)

    gt_total = result["total_gt_instance_count"]
    pred_total = result["total_pred_instance_count"]
    result["global_instance_count_ratio"] = (
        pred_total / gt_total if gt_total else float("nan")
    )
    result["global_instance_count_abs_relative_error"] = (
        abs(pred_total - gt_total) / gt_total if gt_total else float("nan")
    )

    tp = result["total_true_merge_count"]
    fp = result["total_false_merge_count"]
    fn = result["total_false_cut_count"]
    tn = result["total_true_cut_count"]

    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    s = tn / (tn + fp) if tn + fp else float("nan")
    result["global_edge_merge_precision"] = p
    result["global_edge_merge_recall"] = r
    result["global_edge_merge_f1"] = _harmonic(p, r)
    result["global_edge_cut_specificity"] = s

    return result


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """
    Lexicographic operating-point selection.

    F1@IoU50 balances under-merging and over-merging.  Higher-IoU recall is the
    first tie-breaker.  Count bias is deliberately only a late tie-breaker.
    """
    def value(name: str, fallback: float = -1e9) -> float:
        item = row.get(name)
        if isinstance(item, (int, float, np.number)) and math.isfinite(float(item)):
            return float(item)
        return fallback

    return (
        value("mean_instance_f1_iou50"),
        value("mean_gt_recall_iou75"),
        value("mean_mean_best_gt_iou"),
        -value("global_instance_count_abs_relative_error", fallback=1e9),
        value("global_edge_merge_f1"),
    )


def _calibrate(
    crops: list[CropArtifact],
    thresholds: tuple[float, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_crop_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        threshold_rows = []
        for crop in crops:
            partition = _partition_at_threshold(crop, threshold)
            row = {
                "threshold": float(threshold),
                "sample": crop.sample,
                "crop_index": crop.crop_index,
                "candidate_type": crop.candidate_type,
                **_partition_metrics(partition, crop),
                **_edge_metrics(crop, threshold),
            }
            threshold_rows.append(row)
            per_crop_rows.append(row)

        aggregate_rows.append(_aggregate_threshold(threshold, threshold_rows))

    return per_crop_rows, aggregate_rows


# ======================================================================================
# Residual errors for the next investigation
# ======================================================================================


def _residual_edges(
    crops: list[CropArtifact],
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for crop in crops:
        pred = crop.edge_probability >= float(threshold)
        valid = crop.edge_valid
        truth = crop.edge_target

        false_merge = valid & pred & ~truth
        false_cut = valid & ~pred & truth

        for edge_kind, mask in (
            ("false_merge", false_merge),
            ("false_cut", false_cut),
        ):
            for edge_row in np.flatnonzero(mask):
                a = int(crop.edge_index[0, edge_row])
                b = int(crop.edge_index[1, edge_row])
                probability = float(crop.edge_probability[edge_row])

                # "wrong_confidence" is larger when the error is farther from
                # the decision boundary in the wrong direction.
                wrong_confidence = (
                    probability - threshold
                    if edge_kind == "false_merge"
                    else threshold - probability
                )

                rows.append(
                    {
                        "sample": crop.sample,
                        "crop_index": crop.crop_index,
                        "candidate_type": crop.candidate_type,
                        "edge_kind": edge_kind,
                        "edge_row": int(edge_row),
                        "probability": probability,
                        "threshold": float(threshold),
                        "wrong_confidence": float(wrong_confidence),
                        "node_a": a,
                        "node_b": b,
                        "node_a_supervoxel_id": a + 1,
                        "node_b_supervoxel_id": b + 1,
                        "node_a_purity": float(crop.node_purity[a]),
                        "node_b_purity": float(crop.node_purity[b]),
                        "node_a_gt_support": float(crop.node_gt_support[a]),
                        "node_b_gt_support": float(crop.node_gt_support[b]),
                        "node_a_dominant_gt": int(crop.node_dominant_gt[a]),
                        "node_b_dominant_gt": int(crop.node_dominant_gt[b]),
                    }
                )

    rows.sort(
        key=lambda row: (
            0 if row["edge_kind"] == "false_merge" else 1,
            -float(row["wrong_confidence"]),
            row["sample"],
            int(row["crop_index"]),
        )
    )
    return rows


# ======================================================================================
# Plot
# ======================================================================================


def _save_plot(path: Path, aggregate_rows: list[dict[str, Any]], selected: float) -> None:
    import matplotlib.pyplot as plt

    threshold = np.asarray([row["threshold"] for row in aggregate_rows], dtype=float)

    figure = plt.figure(figsize=(12, 8))
    grid = figure.add_gridspec(2, 2)

    ax = figure.add_subplot(grid[0, 0])
    ax.plot(threshold, [row["mean_gt_recall_iou50"] for row in aggregate_rows], label="GT recall @ IoU .50")
    ax.plot(threshold, [row["mean_pred_precision_iou50"] for row in aggregate_rows], label="Pred precision @ IoU .50")
    ax.plot(threshold, [row["mean_instance_f1_iou50"] for row in aggregate_rows], label="Instance F1 @ IoU .50")
    ax.axvline(selected, linestyle="--", label=f"selected {selected:.3f}")
    ax.set_xlabel("RAG merge threshold")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = figure.add_subplot(grid[0, 1])
    ax.plot(threshold, [row["mean_gt_recall_iou75"] for row in aggregate_rows], label="GT recall @ IoU .75")
    ax.plot(threshold, [row["mean_mean_best_gt_iou"] for row in aggregate_rows], label="Mean best GT IoU")
    ax.axvline(selected, linestyle="--")
    ax.set_xlabel("RAG merge threshold")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = figure.add_subplot(grid[1, 0])
    ax.plot(threshold, [row["global_instance_count_ratio"] for row in aggregate_rows], label="Pred / GT instance count")
    ax.axhline(1.0, linestyle=":")
    ax.axvline(selected, linestyle="--")
    ax.set_xlabel("RAG merge threshold")
    ax.set_ylabel("count ratio")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = figure.add_subplot(grid[1, 1])
    ax.plot(threshold, [row["global_edge_merge_precision"] for row in aggregate_rows], label="Edge merge precision")
    ax.plot(threshold, [row["global_edge_merge_recall"] for row in aggregate_rows], label="Edge merge recall")
    ax.plot(threshold, [row["global_edge_merge_f1"] for row in aggregate_rows], label="Edge merge F1")
    ax.axvline(selected, linestyle="--")
    ax.set_xlabel("RAG merge threshold")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend()

    figure.suptitle("STIR-Net NIS3D RAG threshold calibration")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


# ======================================================================================
# Main
# ======================================================================================


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    source_run = (
        _latest_stage05_run()
        if args.source_run is None
        else _resolve_repo_path(args.source_run)
    )
    if not source_run.exists():
        raise FileNotFoundError(f"--source-run does not exist:\n  {source_run}")

    summary_path = source_run / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Investigation-05 summary.json is missing:\n  {summary_path}"
        )
    stage05_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    primary_step = int(stage05_summary.get("primary_step", -1))
    if primary_step < 0:
        raise ValueError("Could not resolve primary checkpoint step from stage-05 summary.json")

    crops = _discover_crops(source_run)
    thresholds = _threshold_values(
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
        reference=(args.production_threshold, args.coarse_best_threshold),
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = (
        _resolve_repo_path(args.output_dir)
        if args.output_dir
        else ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / SCRIPT_NAME
        / f"checkpoint_step_{primary_step:06d}_{stamp}"
    )
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 112, flush=True)
    print("STIR-Net Investigation 06 — NIS3D RAG threshold calibration", flush=True)
    print("=" * 112, flush=True)
    print(f"Source run            : {source_run}", flush=True)
    print(f"Checkpoint step       : {primary_step}", flush=True)
    print(f"Crops                 : {len(crops)}", flush=True)
    print(f"Samples               : {sorted({crop.sample for crop in crops})}", flush=True)
    print(f"Fine range            : {args.threshold_min:.3f} .. {args.threshold_max:.3f}", flush=True)
    print(f"Fine step             : {args.threshold_step:.4f}", flush=True)
    print(f"Threshold count       : {len(thresholds)}", flush=True)
    print(f"Production reference  : {args.production_threshold:.3f}", flush=True)
    print(f"Coarse best reference : {args.coarse_best_threshold:.3f}", flush=True)
    print(f"Output                : {output}", flush=True)
    print("=" * 112, flush=True)

    per_crop_rows, aggregate_rows = _calibrate(crops, thresholds)
    selected_row = max(aggregate_rows, key=_selection_key)
    selected_threshold = float(selected_row["threshold"])

    # Reference rows for easy comparison.
    def nearest(value: float) -> dict[str, Any]:
        return min(aggregate_rows, key=lambda row: abs(float(row["threshold"]) - value))

    production_row = nearest(args.production_threshold)
    coarse_row = nearest(args.coarse_best_threshold)

    residual = _residual_edges(crops, selected_threshold)

    _write_csv(output / "per_crop_threshold_metrics.csv", per_crop_rows)
    _write_csv(output / "threshold_aggregate.csv", aggregate_rows)
    _write_csv(output / "residual_edges.csv", residual)

    if not args.no_plot:
        _save_plot(
            output / "rag_threshold_calibration.png",
            aggregate_rows,
            selected_threshold,
        )

    recommendation_text = (
        f"{selected_threshold:.6f}\n\n"
        f"Recommended STIR-Net partition setting:\n"
        f"    spatial_merge_threshold = {selected_threshold:.6f}\n\n"
        f"Selection objective:\n"
        f"    maximize mean instance F1 @ IoU 0.50\n"
        f"    then GT recall @ IoU 0.75\n"
        f"    then mean best GT IoU\n"
        f"    then instance-count calibration\n"
    )
    (output / "recommended_threshold.txt").write_text(
        recommendation_text,
        encoding="utf-8",
    )

    # Count residual error types.
    false_merges = sum(row["edge_kind"] == "false_merge" for row in residual)
    false_cuts = sum(row["edge_kind"] == "false_cut" for row in residual)

    summary = {
        "status": "success",
        "source_run": str(source_run),
        "checkpoint_step": primary_step,
        "samples": sorted({crop.sample for crop in crops}),
        "crop_count": len(crops),
        "threshold_range": {
            "minimum": float(args.threshold_min),
            "maximum": float(args.threshold_max),
            "step": float(args.threshold_step),
            "evaluated_count": len(thresholds),
        },
        "selection_objective": [
            "maximize mean_instance_f1_iou50",
            "maximize mean_gt_recall_iou75",
            "maximize mean_mean_best_gt_iou",
            "minimize global_instance_count_abs_relative_error",
            "maximize global_edge_merge_f1",
        ],
        "production_reference": production_row,
        "coarse_best_reference": coarse_row,
        "recommended": selected_row,
        "recommended_threshold": selected_threshold,
        "residual_valid_edge_errors": {
            "false_merge_count": int(false_merges),
            "false_cut_count": int(false_cuts),
            "total": int(false_merges + false_cuts),
        },
        "output_root": str(output),
    }
    _write_json(output / "calibration_summary.json", summary)

    print("\n" + "=" * 112, flush=True)
    print("CALIBRATION COMPLETE", flush=True)
    print("=" * 112, flush=True)

    def print_row(label: str, row: dict[str, Any]) -> None:
        print(
            f"{label:<22s} "
            f"thr={row['threshold']:.3f} | "
            f"F1@.50={row['mean_instance_f1_iou50']:.4f} | "
            f"R@.50={row['mean_gt_recall_iou50']:.4f} | "
            f"P@.50={row['mean_pred_precision_iou50']:.4f} | "
            f"R@.75={row['mean_gt_recall_iou75']:.4f} | "
            f"IoU={row['mean_mean_best_gt_iou']:.4f} | "
            f"count={row['global_instance_count_ratio']:.4f}x | "
            f"edgeF1={row['global_edge_merge_f1']:.4f}",
            flush=True,
        )

    print_row("production", production_row)
    print_row("coarse 05 best", coarse_row)
    print_row("RECOMMENDED", selected_row)
    print(
        f"\nResidual valid edges at selected threshold: "
        f"false merges={false_merges}, false cuts={false_cuts}",
        flush=True,
    )
    print(f"Residual edge table   : {output / 'residual_edges.csv'}", flush=True)
    print(f"Output                : {output}", flush=True)
    print("=" * 112, flush=True)

    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-calibrate STIR-Net's learned RAG merge threshold from "
            "Investigation-05 saved graph artifacts without rerunning the network."
        )
    )
    parser.add_argument(
        "--source-run",
        default=None,
        help=(
            "Investigation-05 output directory. Default: automatically select "
            "the completed run with the highest checkpoint step."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.80,
        help="Fine-sweep minimum (default: 0.80).",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.995,
        help="Fine-sweep maximum (default: 0.995).",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.005,
        help="Fine-sweep increment (default: 0.005).",
    )
    parser.add_argument(
        "--production-threshold",
        type=float,
        default=0.50,
        help="Existing production threshold included as a reference (default: 0.50).",
    )
    parser.add_argument(
        "--coarse-best-threshold",
        type=float,
        default=0.90,
        help="Best boundary from Investigation 05 included as reference (default: 0.90).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip calibration PNG.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = evaluate(args)
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
