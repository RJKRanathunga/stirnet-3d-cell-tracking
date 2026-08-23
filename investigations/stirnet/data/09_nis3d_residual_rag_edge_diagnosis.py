from __future__ import annotations

"""
STIR-Net Investigation 07 — residual RAG edge diagnosis.

This stage follows:
    05_nis3d_spatial_checkpoint_eval.py
    07_nis3d_rag_threshold_calibration.py

It does NOT rerun the model and does NOT train anything.

Goal
----
At the currently selected development threshold (default: the recommendation
from Investigation 06, currently expected around 0.845), explain the remaining
valid RAG errors:

    false merge:
        GT says two adjacent supervoxels belong to different cells,
        but the RAG probability is above threshold.

    false cut:
        GT says two adjacent supervoxels belong to the same cell,
        but the RAG probability is below threshold.

The key question is whether residual false merges are caused by:

    A. weak / missing separator evidence at a true cell-cell interface,
    B. good separator evidence that the RAG fails to use,
    C. impure / low-support watershed supervoxels,
    D. other geometry cues (surface/SDF/flow/centroid-offset), or
    E. genuinely ambiguous interfaces.

The script uses the exact raw RAG edge feature semantics in production:

    [separator_mean,
     separator_max,
     surface_mean,
     surface_max,
     foreground_mean,
     abs_sdf_mean,
     flow_disagreement,
     centroid_offset_disagreement]

and independently recomputes those features on the saved supervoxel faces to
verify the artifact contract.

Inputs
------
Automatically discovers the latest successful Investigation-06 calibration and
then follows its ``source_run`` back to the corresponding Investigation-05 run.

The Investigation-05 run must still contain:

    <sample>/crop_XX_<type>/
        spatial_eval.npz
        edges.csv

Outputs
-------
runs/stirnet/evaluation/07_nis3d_residual_rag_edge_diagnosis/
    checkpoint_step_XXXXXX_<timestamp>/
        diagnostic_summary.json
        feature_reference.csv
        edge_group_summary.csv
        residual_edge_diagnostics.csv
        false_merges_ranked.csv
        false_cuts_ranked.csv
        category_summary.csv
        interface_feature_validation.csv
        residual_feature_distributions.png
        false_merge_separator_summary.png
        visuals/
            false_merge_rank_00_....png
            false_cut_rank_00_....png

Typical usage
-------------
From repository root:

    python investigations/stirnet/data/09_nis3d_residual_rag_edge_diagnosis.py

Inspect the highest-confidence residual false merge interactively:

    python investigations/stirnet/data/09_nis3d_residual_rag_edge_diagnosis.py \
        --napari \
        --edge-kind false_merge \
        --rank 0

Override the threshold while keeping the same frozen graph outputs:

    python investigations/stirnet/data/09_nis3d_residual_rag_edge_diagnosis.py \
        --threshold 0.90

This is a diagnostic investigation, not an automatic architecture decision.
The category labels are clues and are deliberately data-relative.
"""

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_NAME = "07_nis3d_residual_rag_edge_diagnosis"
STAGE05 = "05_nis3d_spatial_checkpoint_eval"
STAGE06 = "06_nis3d_rag_threshold_calibration"

EDGE_FEATURES = (
    "separator_mean",
    "separator_max",
    "surface_mean",
    "surface_max",
    "foreground_mean",
    "abs_sdf_mean",
    "flow_disagreement",
    "centroid_offset_disagreement",
)

SEPARATOR_FEATURES = ("separator_mean", "separator_max")
GEOMETRY_CUT_FEATURES = (
    "separator_mean",
    "separator_max",
    "surface_mean",
    "surface_max",
    "abs_sdf_mean",
    "flow_disagreement",
    "centroid_offset_disagreement",
)


# ======================================================================================
# Filesystem / serialization
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return

    preferred = [
        "edge_kind",
        "rank",
        "sample",
        "crop_index",
        "candidate_type",
        "edge_row",
        "probability",
        "threshold",
        "wrong_confidence",
        "primary_category",
        "feature",
        "group",
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


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _latest_stage06_run() -> Path:
    base = ROOT / "runs" / "stirnet" / "evaluation" / STAGE06
    if not base.exists():
        raise FileNotFoundError(
            f"Investigation-06 directory not found:\n  {base}\n\n"
            "Run 07_nis3d_rag_threshold_calibration.py first."
        )

    runs = [
        path
        for path in base.iterdir()
        if path.is_dir() and (path / "calibration_summary.json").exists()
    ]
    if not runs:
        raise FileNotFoundError(
            f"No completed Investigation-06 run found under:\n  {base}"
        )

    def key(path: Path) -> tuple[int, float]:
        step = -1
        try:
            payload = json.loads(
                (path / "calibration_summary.json").read_text(encoding="utf-8")
            )
            step = int(payload.get("checkpoint_step", -1))
        except Exception:
            pass
        return step, path.stat().st_mtime

    return max(runs, key=key)


# ======================================================================================
# Stage-05 crop discovery / edge table
# ======================================================================================


@dataclass(frozen=True)
class CropRef:
    sample: str
    crop_index: int
    candidate_type: str
    crop_dir: Path
    npz_path: Path
    edges_csv: Path


def _parse_crop_dir(path: Path) -> tuple[int, str]:
    name = path.name
    if not name.startswith("crop_"):
        raise ValueError(f"Unexpected crop directory: {name}")
    pieces = name.split("_", 2)
    if len(pieces) != 3:
        raise ValueError(f"Cannot parse crop directory: {name}")
    return int(pieces[1]), pieces[2]


def _discover_crops(source_run: Path) -> list[CropRef]:
    refs: list[CropRef] = []
    for npz_path in sorted(source_run.glob("*/crop_*/spatial_eval.npz")):
        crop_dir = npz_path.parent
        crop_index, candidate_type = _parse_crop_dir(crop_dir)
        edges_csv = crop_dir / "edges.csv"
        if not edges_csv.exists():
            raise FileNotFoundError(
                f"Required Investigation-05 edge table is missing:\n  {edges_csv}"
            )
        refs.append(
            CropRef(
                sample=crop_dir.parent.name,
                crop_index=crop_index,
                candidate_type=candidate_type,
                crop_dir=crop_dir,
                npz_path=npz_path,
                edges_csv=edges_csv,
            )
        )
    if not refs:
        raise FileNotFoundError(
            f"No Investigation-05 spatial_eval.npz artifacts found under:\n  {source_run}"
        )
    refs.sort(key=lambda row: (row.sample, row.crop_index))
    return refs


def _first_present(row: dict[str, str], names: Iterable[str]) -> str | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def _load_edge_rows(crop: CropRef, threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with crop.edges_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            edge_row = int(raw["edge_row"])
            probability = float(raw["probability"])
            valid = _parse_bool(raw["valid"])
            target_merge = _parse_bool(raw["target_merge"])
            predicted_merge = probability >= threshold

            if not valid:
                group = "invalid"
            elif predicted_merge and target_merge:
                group = "true_merge"
            elif predicted_merge and not target_merge:
                group = "false_merge"
            elif (not predicted_merge) and target_merge:
                group = "false_cut"
            else:
                group = "true_cut"

            node_a = _int_or_none(
                _first_present(raw, ("node_a", "node_a_global", "node_a_global_id"))
            )
            node_b = _int_or_none(
                _first_present(raw, ("node_b", "node_b_global", "node_b_global_id"))
            )
            sv_a = _int_or_none(
                _first_present(
                    raw,
                    ("sv_a", "node_a_supervoxel_id", "supervoxel_a"),
                )
            )
            sv_b = _int_or_none(
                _first_present(
                    raw,
                    ("sv_b", "node_b_supervoxel_id", "supervoxel_b"),
                )
            )
            if sv_a is None and node_a is not None:
                sv_a = node_a + 1
            if sv_b is None and node_b is not None:
                sv_b = node_b + 1

            row: dict[str, Any] = {
                "sample": crop.sample,
                "crop_index": crop.crop_index,
                "candidate_type": crop.candidate_type,
                "edge_row": edge_row,
                "probability": probability,
                "threshold": threshold,
                "valid": valid,
                "target_merge": target_merge,
                "predicted_merge": predicted_merge,
                "edge_kind": group,
                "node_a": node_a,
                "node_b": node_b,
                "sv_a": sv_a,
                "sv_b": sv_b,
            }

            for feature in EDGE_FEATURES:
                row[feature] = _float_or_nan(raw.get(feature))

            # Accept either verbose or compact Investigation-05 column names.
            aliases = {
                "node_a_purity": ("node_a_purity", "purity_a"),
                "node_b_purity": ("node_b_purity", "purity_b"),
                "node_a_gt_support": ("node_a_gt_support", "support_a"),
                "node_b_gt_support": ("node_b_gt_support", "support_b"),
                "node_a_dominant_gt": ("node_a_dominant_gt", "dominant_gt_a"),
                "node_b_dominant_gt": ("node_b_dominant_gt", "dominant_gt_b"),
                "node_a_z_um": ("node_a_z_um", "z_a_um"),
                "node_a_y_um": ("node_a_y_um", "y_a_um"),
                "node_a_x_um": ("node_a_x_um", "x_a_um"),
                "node_b_z_um": ("node_b_z_um", "z_b_um"),
                "node_b_y_um": ("node_b_y_um", "y_b_um"),
                "node_b_x_um": ("node_b_x_um", "x_b_um"),
            }
            for output_name, candidates in aliases.items():
                value = _first_present(raw, candidates)
                if "dominant_gt" in output_name:
                    row[output_name] = _int_or_none(value)
                else:
                    row[output_name] = _float_or_nan(value)

            if all(
                math.isfinite(float(row[name]))
                for name in (
                    "node_a_z_um",
                    "node_a_y_um",
                    "node_a_x_um",
                    "node_b_z_um",
                    "node_b_y_um",
                    "node_b_x_um",
                )
            ):
                a = np.asarray(
                    [row["node_a_z_um"], row["node_a_y_um"], row["node_a_x_um"]],
                    dtype=np.float64,
                )
                b = np.asarray(
                    [row["node_b_z_um"], row["node_b_y_um"], row["node_b_x_um"]],
                    dtype=np.float64,
                )
                row["node_centroid_distance_um"] = float(np.linalg.norm(a - b))
            else:
                row["node_centroid_distance_um"] = float("nan")

            if group == "false_merge":
                row["wrong_confidence"] = probability - threshold
            elif group == "false_cut":
                row["wrong_confidence"] = threshold - probability
            else:
                row["wrong_confidence"] = 0.0

            rows.append(row)
    return rows


# ======================================================================================
# Feature references / relative evidence scores
# ======================================================================================


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(
        [value for value in values if math.isfinite(value)],
        dtype=np.float64,
    )
    if array.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "q10": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
        }
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _feature_references(valid_rows: list[dict[str, Any]]) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    references: dict[str, dict[str, Any]] = {}
    table: list[dict[str, Any]] = []

    merge_rows = [row for row in valid_rows if row["target_merge"]]
    cut_rows = [row for row in valid_rows if not row["target_merge"]]

    for feature in EDGE_FEATURES:
        merge_stats = _quantiles([float(row[feature]) for row in merge_rows])
        cut_stats = _quantiles([float(row[feature]) for row in cut_rows])

        merge_median = merge_stats["median"]
        cut_median = cut_stats["median"]
        direction = (
            "cut_high"
            if math.isfinite(merge_median)
            and math.isfinite(cut_median)
            and cut_median >= merge_median
            else "cut_low"
        )

        pooled_iqr = 0.5 * (
            (merge_stats["q75"] - merge_stats["q25"])
            + (cut_stats["q75"] - cut_stats["q25"])
        )
        median_gap = cut_median - merge_median
        normalized_gap = (
            abs(median_gap) / max(abs(pooled_iqr), 1e-8)
            if math.isfinite(median_gap)
            else float("nan")
        )

        reference = {
            "feature": feature,
            "direction": direction,
            "merge": merge_stats,
            "cut": cut_stats,
            "median_gap_cut_minus_merge": median_gap,
            "normalized_median_gap_iqr": normalized_gap,
        }
        references[feature] = reference

        flat = {
            "feature": feature,
            "direction": direction,
            "merge_count": len(merge_rows),
            "cut_count": len(cut_rows),
            "median_gap_cut_minus_merge": median_gap,
            "normalized_median_gap_iqr": normalized_gap,
        }
        for prefix, stats in (("merge", merge_stats), ("cut", cut_stats)):
            for key, value in stats.items():
                flat[f"{prefix}_{key}"] = value
        table.append(flat)

    table.sort(
        key=lambda row: (
            -float(row["normalized_median_gap_iqr"])
            if math.isfinite(float(row["normalized_median_gap_iqr"]))
            else float("inf")
        )
    )
    return references, table


def _cut_score(value: float, reference: dict[str, Any]) -> float:
    """
    0 ~= typical true-merge value.
    1 ~= typical true-cut value.
    Values outside [0,1] are intentionally retained.
    """
    merge_median = float(reference["merge"]["median"])
    cut_median = float(reference["cut"]["median"])
    if not (
        math.isfinite(value)
        and math.isfinite(merge_median)
        and math.isfinite(cut_median)
    ):
        return float("nan")
    denominator = cut_median - merge_median
    if abs(denominator) < 1e-8:
        return 0.5
    return (value - merge_median) / denominator


def _mean_finite(values: Iterable[float]) -> float:
    filtered = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(filtered)) if filtered else float("nan")


def _attach_relative_scores(
    row: dict[str, Any],
    references: dict[str, dict[str, Any]],
) -> None:
    for feature in EDGE_FEATURES:
        row[f"{feature}_cut_score"] = _cut_score(
            float(row[feature]),
            references[feature],
        )

    row["separator_cut_score"] = _mean_finite(
        row[f"{feature}_cut_score"] for feature in SEPARATOR_FEATURES
    )
    row["geometry_cut_score"] = _mean_finite(
        row[f"{feature}_cut_score"] for feature in GEOMETRY_CUT_FEATURES
    )


def _diagnostic_category(row: dict[str, Any]) -> tuple[str, list[str]]:
    """
    Heuristic clue classification. It is intentionally conservative and
    data-relative; it is not a training label.
    """
    kind = row["edge_kind"]
    separator_score = float(row["separator_cut_score"])
    geometry_score = float(row["geometry_cut_score"])

    purity_values = [
        float(row.get("node_a_purity", float("nan"))),
        float(row.get("node_b_purity", float("nan"))),
    ]
    support_values = [
        float(row.get("node_a_gt_support", float("nan"))),
        float(row.get("node_b_gt_support", float("nan"))),
    ]
    finite_purity = [value for value in purity_values if math.isfinite(value)]
    finite_support = [value for value in support_values if math.isfinite(value)]
    min_purity = min(finite_purity) if finite_purity else float("nan")
    min_support = min(finite_support) if finite_support else float("nan")

    flags: list[str] = []
    if math.isfinite(min_purity) and min_purity < 0.90:
        flags.append("node_purity_risk")
    if math.isfinite(min_support) and min_support < 0.70:
        flags.append("node_support_risk")

    if kind == "false_merge":
        if flags:
            category = "proposal_or_node_quality_risk"
        elif math.isfinite(separator_score) and separator_score <= 0.35:
            category = "weak_separator_on_true_boundary"
        elif math.isfinite(separator_score) and separator_score >= 0.75:
            category = "rag_merge_despite_strong_separator"
        elif math.isfinite(geometry_score) and geometry_score >= 0.75:
            category = "rag_merge_despite_other_cut_geometry"
        else:
            category = "ambiguous_true_boundary"

        if math.isfinite(separator_score) and separator_score <= 0.35:
            flags.append("weak_separator")
        if math.isfinite(separator_score) and separator_score >= 0.75:
            flags.append("strong_separator")
        if math.isfinite(geometry_score) and geometry_score >= 0.75:
            flags.append("strong_overall_cut_geometry")

    elif kind == "false_cut":
        if math.isfinite(separator_score) and separator_score >= 0.75:
            category = "separator_false_positive_inside_gt"
        elif math.isfinite(geometry_score) and geometry_score >= 0.75:
            category = "other_geometry_false_cut_cue"
        elif (
            math.isfinite(separator_score)
            and math.isfinite(geometry_score)
            and separator_score <= 0.35
            and geometry_score <= 0.50
        ):
            category = "rag_cut_despite_merge_like_geometry"
        else:
            category = "ambiguous_same_gt_fragment"

        if math.isfinite(separator_score) and separator_score >= 0.75:
            flags.append("strong_false_separator")
        if math.isfinite(geometry_score) and geometry_score >= 0.75:
            flags.append("strong_false_cut_geometry")
        if flags and category == "ambiguous_same_gt_fragment":
            category = "proposal_or_node_quality_risk"

    else:
        category = kind

    row["minimum_node_purity"] = min_purity
    row["minimum_node_gt_support"] = min_support
    return category, flags


# ======================================================================================
# Saved NPZ and exact interface-face analysis
# ======================================================================================


NPZ_REQUIRED = (
    "normalized_raw",
    "gt_labels",
    "supervision_valid",
    "spacing_zyx_um",
    "primary_supervoxels",
    "primary_pred_foreground",
    "primary_pred_surface",
    "primary_pred_separator",
    "primary_pred_seed",
    "primary_pred_sdf",
    "primary_pred_flow",
    "primary_pred_centroid_offset",
    "edge_index",
    "edge_probability",
    "edge_target",
    "edge_valid",
)


def _load_npz(crop: CropRef) -> dict[str, np.ndarray]:
    with np.load(crop.npz_path, allow_pickle=False) as data:
        missing = [key for key in NPZ_REQUIRED if key not in data]
        if missing:
            raise KeyError(
                f"{crop.npz_path} is missing Investigation-05 arrays: {missing}"
            )
        return {key: np.asarray(data[key]) for key in data.files}


def _vector_cosine_disagreement(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=0)
    norm_a = np.sqrt(np.sum(a * a, axis=0))
    norm_b = np.sqrt(np.sum(b * b, axis=0))
    denominator = np.maximum(norm_a, 1e-6) * np.maximum(norm_b, 1e-6)
    cosine = np.clip(dot / denominator, -1.0, 1.0)
    return 1.0 - cosine


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")


def _interface_observations(
    arrays: dict[str, np.ndarray],
    needed_edges: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """
    Recompute production RAG face features only for requested residual edges.

    The current production builder averages scalar fields on the two voxels
    touching each 6-connected face and aggregates by undirected supervoxel pair.
    """
    labels = arrays["primary_supervoxels"].astype(np.int64, copy=False)
    separator = arrays["primary_pred_separator"].astype(np.float64, copy=False)
    surface = arrays["primary_pred_surface"].astype(np.float64, copy=False)
    foreground = arrays["primary_pred_foreground"].astype(np.float64, copy=False)
    sdf = arrays["primary_pred_sdf"].astype(np.float64, copy=False)
    flow = arrays["primary_pred_flow"].astype(np.float64, copy=False)
    centroid = arrays["primary_pred_centroid_offset"].astype(np.float64, copy=False)
    spacing = arrays["spacing_zyx_um"].astype(np.float64, copy=False)

    node_count = int(labels.max(initial=0))
    needed_packs = {
        min(int(row["node_a"]), int(row["node_b"])) * node_count
        + max(int(row["node_a"]), int(row["node_b"]))
        for row in needed_edges
        if row.get("node_a") is not None and row.get("node_b") is not None
    }
    if not needed_packs:
        return {}, {"requested_edge_count": 0, "matched_edge_count": 0}

    observations: dict[int, list[tuple[float, ...]]] = {
        pack: [] for pack in needed_packs
    }

    # tuple:
    # axis, face_area, sep, surf, fg, abs_sdf, flow_disagree, offset_disagree
    for axis in range(3):
        a_slice = [slice(None)] * 3
        b_slice = [slice(None)] * 3
        a_slice[axis] = slice(0, -1)
        b_slice[axis] = slice(1, None)
        a_slice = tuple(a_slice)
        b_slice = tuple(b_slice)

        la_raw = labels[a_slice]
        lb_raw = labels[b_slice]
        valid = (la_raw > 0) & (lb_raw > 0) & (la_raw != lb_raw)
        if not np.any(valid):
            continue

        la = la_raw[valid].astype(np.int64) - 1
        lb = lb_raw[valid].astype(np.int64) - 1
        lo = np.minimum(la, lb)
        hi = np.maximum(la, lb)
        packed = lo * node_count + hi

        wanted = np.isin(packed, np.fromiter(needed_packs, dtype=np.int64))
        if not np.any(wanted):
            continue

        # Convert the valid-face linear selection to selections in scalar pairs.
        selected_pack = packed[wanted]

        def pair_scalar(field: np.ndarray) -> np.ndarray:
            av = field[a_slice][valid][wanted]
            bv = field[b_slice][valid][wanted]
            return 0.5 * (av + bv)

        sep = pair_scalar(separator)
        surf = pair_scalar(surface)
        fg = pair_scalar(foreground)
        sdf_abs = pair_scalar(np.abs(sdf))

        flow_a = flow[(slice(None),) + a_slice][:, valid][:, wanted]
        flow_b = flow[(slice(None),) + b_slice][:, valid][:, wanted]
        flow_dis = _vector_cosine_disagreement(flow_a, flow_b)

        cent_a = centroid[(slice(None),) + a_slice][:, valid][:, wanted]
        cent_b = centroid[(slice(None),) + b_slice][:, valid][:, wanted]
        offset_dis = np.linalg.norm(cent_a - cent_b, axis=0)

        other_axes = [idx for idx in range(3) if idx != axis]
        face_area = float(spacing[other_axes[0]] * spacing[other_axes[1]])

        for idx, pack in enumerate(selected_pack.tolist()):
            observations[int(pack)].append(
                (
                    float(axis),
                    face_area,
                    float(sep[idx]),
                    float(surf[idx]),
                    float(fg[idx]),
                    float(sdf_abs[idx]),
                    float(flow_dis[idx]),
                    float(offset_dis[idx]),
                )
            )

    result: dict[int, dict[str, Any]] = {}
    for pack, rows in observations.items():
        if not rows:
            continue
        matrix = np.asarray(rows, dtype=np.float64)

        sep = matrix[:, 2]
        surf = matrix[:, 3]
        fg = matrix[:, 4]
        sdf_abs = matrix[:, 5]
        flow_dis = matrix[:, 6]
        offset_dis = matrix[:, 7]

        result[pack] = {
            "interface_face_count": int(len(matrix)),
            "interface_area_um2": float(matrix[:, 1].sum()),
            "interface_axis_z_fraction": float(np.mean(matrix[:, 0] == 0)),
            "interface_axis_y_fraction": float(np.mean(matrix[:, 0] == 1)),
            "interface_axis_x_fraction": float(np.mean(matrix[:, 0] == 2)),
            "interface_separator_mean": float(sep.mean()),
            "interface_separator_max": float(sep.max()),
            "interface_separator_p90": _percentile(sep, 0.90),
            "interface_separator_fraction_ge_0p5": float(np.mean(sep >= 0.50)),
            "interface_separator_fraction_ge_0p7": float(np.mean(sep >= 0.70)),
            "interface_separator_fraction_ge_0p9": float(np.mean(sep >= 0.90)),
            "interface_surface_mean": float(surf.mean()),
            "interface_surface_max": float(surf.max()),
            "interface_surface_p90": _percentile(surf, 0.90),
            "interface_surface_fraction_ge_0p5": float(np.mean(surf >= 0.50)),
            "interface_foreground_mean": float(fg.mean()),
            "interface_abs_sdf_mean": float(sdf_abs.mean()),
            "interface_flow_disagreement_mean": float(flow_dis.mean()),
            "interface_flow_disagreement_p90": _percentile(flow_dis, 0.90),
            "interface_centroid_offset_disagreement_mean": float(offset_dis.mean()),
            "interface_centroid_offset_disagreement_p90": _percentile(offset_dis, 0.90),
        }

    audit = {
        "requested_edge_count": len(needed_packs),
        "matched_edge_count": len(result),
        "unmatched_edge_count": len(needed_packs) - len(result),
    }
    return result, audit


def _attach_node_volume_stats(
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
) -> None:
    labels = arrays["primary_supervoxels"].astype(np.int64, copy=False)
    spacing = arrays["spacing_zyx_um"].astype(np.float64, copy=False)
    counts = np.bincount(labels.ravel())
    voxel_volume = float(np.prod(spacing))

    for row in rows:
        sv_a = row.get("sv_a")
        sv_b = row.get("sv_b")
        if sv_a is None or sv_b is None:
            continue
        a_count = int(counts[sv_a]) if sv_a < len(counts) else 0
        b_count = int(counts[sv_b]) if sv_b < len(counts) else 0
        row["node_a_voxels"] = a_count
        row["node_b_voxels"] = b_count
        row["node_a_volume_um3"] = a_count * voxel_volume
        row["node_b_volume_um3"] = b_count * voxel_volume
        row["node_volume_ratio_large_to_small"] = (
            max(a_count, b_count) / max(min(a_count, b_count), 1)
        )


def _feature_validation_row(
    crop: CropRef,
    residual_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    differences: dict[str, list[float]] = defaultdict(list)
    for row in residual_rows:
        mapping = {
            "separator_mean": "interface_separator_mean",
            "separator_max": "interface_separator_max",
            "surface_mean": "interface_surface_mean",
            "surface_max": "interface_surface_max",
            "foreground_mean": "interface_foreground_mean",
            "abs_sdf_mean": "interface_abs_sdf_mean",
            "flow_disagreement": "interface_flow_disagreement_mean",
            "centroid_offset_disagreement": "interface_centroid_offset_disagreement_mean",
        }
        for saved_name, recomputed_name in mapping.items():
            a = float(row.get(saved_name, float("nan")))
            b = float(row.get(recomputed_name, float("nan")))
            if math.isfinite(a) and math.isfinite(b):
                differences[saved_name].append(abs(a - b))

    output: dict[str, Any] = {
        "sample": crop.sample,
        "crop_index": crop.crop_index,
        "candidate_type": crop.candidate_type,
        "residual_edge_count": len(residual_rows),
    }
    for feature in EDGE_FEATURES:
        values = differences.get(feature, [])
        output[f"{feature}_max_abs_error"] = max(values) if values else float("nan")
        output[f"{feature}_mean_abs_error"] = (
            float(np.mean(values)) if values else float("nan")
        )
    finite_maxima = [
        float(value)
        for key, value in output.items()
        if key.endswith("_max_abs_error")
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    ]
    output["overall_max_abs_error"] = max(finite_maxima) if finite_maxima else float("nan")
    return output


# ======================================================================================
# Group summaries
# ======================================================================================


def _group_summary(
    valid_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups = ("true_merge", "false_merge", "true_cut", "false_cut")
    result: list[dict[str, Any]] = []

    for group in groups:
        rows = [row for row in valid_rows if row["edge_kind"] == group]
        if not rows:
            continue
        base = {
            "group": group,
            "count": len(rows),
            "mean_probability": _mean_finite(float(row["probability"]) for row in rows),
        }
        for feature in EDGE_FEATURES:
            stats = _quantiles([float(row[feature]) for row in rows])
            for key, value in stats.items():
                base[f"{feature}_{key}"] = value
        result.append(base)
    return result


def _category_summary(residual_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (row["edge_kind"], row["primary_category"]) for row in residual_rows
    )
    totals = Counter(row["edge_kind"] for row in residual_rows)
    output = []
    for (kind, category), count in sorted(counts.items()):
        output.append(
            {
                "edge_kind": kind,
                "primary_category": category,
                "count": count,
                "fraction_within_kind": count / max(totals[kind], 1),
            }
        )
    return output


# ======================================================================================
# Visuals
# ======================================================================================


def _interface_mask_for_pair(
    labels: np.ndarray,
    node_a: int,
    node_b: int,
) -> np.ndarray:
    sv_a = node_a + 1
    sv_b = node_b + 1
    mask = np.zeros(labels.shape, dtype=np.uint8)

    for axis in range(3):
        a_slice = [slice(None)] * 3
        b_slice = [slice(None)] * 3
        a_slice[axis] = slice(0, -1)
        b_slice[axis] = slice(1, None)
        a_slice = tuple(a_slice)
        b_slice = tuple(b_slice)

        la = labels[a_slice]
        lb = labels[b_slice]
        hit = ((la == sv_a) & (lb == sv_b)) | ((la == sv_b) & (lb == sv_a))
        if not np.any(hit):
            continue

        view_a = mask[a_slice]
        view_b = mask[b_slice]
        view_a[hit] = 1
        view_b[hit] = 1

    return mask


def _pair_labels(labels: np.ndarray, node_a: int, node_b: int) -> np.ndarray:
    result = np.zeros(labels.shape, dtype=np.uint8)
    result[labels == node_a + 1] = 1
    result[labels == node_b + 1] = 2
    return result


def _bbox(mask: np.ndarray, margin: int = 12) -> tuple[slice, slice, slice]:
    points = np.argwhere(mask)
    if points.size == 0:
        return tuple(slice(0, size) for size in mask.shape)  # type: ignore[return-value]
    lower = np.maximum(points.min(axis=0) - margin, 0)
    upper = np.minimum(points.max(axis=0) + margin + 1, np.asarray(mask.shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))  # type: ignore[return-value]


def _save_edge_visual(
    path: Path,
    arrays: dict[str, np.ndarray],
    row: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    labels = arrays["primary_supervoxels"].astype(np.int64, copy=False)
    node_a = int(row["node_a"])
    node_b = int(row["node_b"])
    interface = _interface_mask_for_pair(labels, node_a, node_b)
    pair = _pair_labels(labels, node_a, node_b)

    region = _bbox((pair > 0) | (interface > 0), margin=12)
    interface_local = interface[region]

    per_z = interface_local.sum(axis=(1, 2))
    z_local = int(np.argmax(per_z)) if per_z.size else 0
    z_global = int(region[0].start) + z_local

    gt = arrays["gt_labels"]
    raw = arrays["normalized_raw"]
    separator = arrays["primary_pred_separator"]
    surface = arrays["primary_pred_surface"]
    foreground = arrays["primary_pred_foreground"]
    seed = arrays["primary_pred_seed"]
    sdf_abs = np.abs(arrays["primary_pred_sdf"])

    y_slice, x_slice = region[1], region[2]

    panels = [
        ("normalized raw", raw[z_global, y_slice, x_slice], "gray", None, None),
        ("GT labels", gt[z_global, y_slice, x_slice], "nipy_spectral", None, None),
        ("selected SV pair", pair[z_global, y_slice, x_slice], "nipy_spectral", 0, 2),
        ("exact interface", interface[z_global, y_slice, x_slice], "gray", 0, 1),
        ("separator", separator[z_global, y_slice, x_slice], "magma", 0, 1),
        ("surface", surface[z_global, y_slice, x_slice], "magma", 0, 1),
        ("foreground", foreground[z_global, y_slice, x_slice], "magma", 0, 1),
        ("|SDF|", sdf_abs[z_global, y_slice, x_slice], "viridis", None, None),
        ("seed", seed[z_global, y_slice, x_slice], "magma", 0, 1),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    for axis, (name, image, cmap, vmin, vmax) in zip(axes.flat, panels):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(name)
        axis.axis("off")

    title = (
        f"{row['edge_kind']} | rank={row.get('rank', '?')} | "
        f"p={float(row['probability']):.3f} thr={float(row['threshold']):.3f}\n"
        f"{row['sample']} crop={row['crop_index']} edge={row['edge_row']} | "
        f"{row['primary_category']} | sepScore={float(row['separator_cut_score']):.2f}"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_distribution_plot(
    path: Path,
    rows: list[dict[str, Any]],
    threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    valid = [row for row in rows if row["valid"]]
    groups = {
        "true merge": [row for row in valid if row["edge_kind"] == "true_merge"],
        "false merge": [row for row in valid if row["edge_kind"] == "false_merge"],
        "true cut": [row for row in valid if row["edge_kind"] == "true_cut"],
        "false cut": [row for row in valid if row["edge_kind"] == "false_cut"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    feature_panels = (
        "separator_mean",
        "separator_max",
        "flow_disagreement",
        "centroid_offset_disagreement",
    )

    for axis, feature in zip(axes.flat, feature_panels):
        data = []
        labels = []
        for name, group_rows in groups.items():
            values = [
                float(row[feature])
                for row in group_rows
                if math.isfinite(float(row[feature]))
            ]
            if values:
                data.append(values)
                labels.append(name)
        if data:
            axis.boxplot(data, tick_labels=labels, showfliers=False)
        axis.set_title(feature)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"Residual RAG feature groups at merge threshold {threshold:.3f}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_false_merge_separator_plot(
    path: Path,
    residual_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    false_merges = [
        row for row in residual_rows if row["edge_kind"] == "false_merge"
    ]
    if not false_merges:
        return

    x = np.asarray([row["probability"] for row in false_merges], dtype=float)
    y = np.asarray(
        [row.get("interface_separator_mean", row["separator_mean"]) for row in false_merges],
        dtype=float,
    )
    area = np.asarray(
        [row.get("interface_area_um2", 1.0) for row in false_merges],
        dtype=float,
    )
    size = 15.0 + 50.0 * area / max(float(np.nanmedian(area)), 1e-6)
    size = np.clip(size, 15, 180)

    fig, axis = plt.subplots(figsize=(10, 7))
    axis.scatter(x, y, s=size, alpha=0.65)
    axis.axhline(0.5, linestyle=":")
    axis.set_xlabel("RAG merge probability")
    axis.set_ylabel("mean separator probability on shared interface")
    axis.set_title("Residual false merges: RAG confidence vs separator evidence")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _open_napari(
    crop: CropRef,
    row: dict[str, Any],
) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is not installed. Install the visualization requirements "
            "or rerun without --napari."
        ) from exc

    arrays = _load_npz(crop)
    labels = arrays["primary_supervoxels"].astype(np.int64, copy=False)
    node_a = int(row["node_a"])
    node_b = int(row["node_b"])
    pair = _pair_labels(labels, node_a, node_b)
    interface = _interface_mask_for_pair(labels, node_a, node_b)
    scale = tuple(float(value) for value in arrays["spacing_zyx_um"])

    viewer = napari.Viewer(
        title=(
            f"STIR-Net residual {row['edge_kind']} "
            f"rank {row['rank']} p={float(row['probability']):.3f}"
        )
    )
    viewer.add_image(
        arrays["normalized_raw"].astype(np.float32),
        name="normalized raw",
        scale=scale,
    )
    viewer.add_labels(
        arrays["gt_labels"].astype(np.int32),
        name="GT labels",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        labels.astype(np.int32),
        name="all watershed supervoxels",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        pair,
        name="selected supervoxel pair",
        scale=scale,
    )
    viewer.add_labels(
        interface,
        name="shared interface",
        scale=scale,
    )
    viewer.add_image(
        arrays["primary_pred_separator"].astype(np.float32),
        name="separator probability",
        scale=scale,
        opacity=0.65,
        visible=True,
    )
    viewer.add_image(
        arrays["primary_pred_surface"].astype(np.float32),
        name="surface probability",
        scale=scale,
        opacity=0.55,
        visible=False,
    )
    viewer.add_image(
        arrays["primary_pred_foreground"].astype(np.float32),
        name="foreground probability",
        scale=scale,
        opacity=0.45,
        visible=False,
    )
    viewer.add_image(
        arrays["primary_pred_seed"].astype(np.float32),
        name="seed probability",
        scale=scale,
        visible=False,
    )
    viewer.add_image(
        np.abs(arrays["primary_pred_sdf"]).astype(np.float32),
        name="abs SDF",
        scale=scale,
        visible=False,
    )
    napari.run()


# ======================================================================================
# Main
# ======================================================================================


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    calibration_run = (
        _latest_stage06_run()
        if args.calibration_run is None
        else _resolve_repo_path(args.calibration_run)
    )
    calibration_summary_path = calibration_run / "calibration_summary.json"
    if not calibration_summary_path.exists():
        raise FileNotFoundError(
            f"Investigation-06 calibration_summary.json not found:\n"
            f"  {calibration_summary_path}"
        )

    calibration = json.loads(
        calibration_summary_path.read_text(encoding="utf-8")
    )
    source_run = (
        _resolve_repo_path(args.source_run)
        if args.source_run
        else Path(calibration["source_run"]).resolve()
    )
    if not source_run.exists():
        raise FileNotFoundError(
            f"Investigation-05 source run no longer exists:\n  {source_run}"
        )

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(calibration["recommended_threshold"])
    )
    if not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must be in (0,1)")

    checkpoint_step = int(calibration.get("checkpoint_step", -1))
    crops = _discover_crops(source_run)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = (
        _resolve_repo_path(args.output_dir)
        if args.output_dir
        else ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / SCRIPT_NAME
        / f"checkpoint_step_{checkpoint_step:06d}_{stamp}"
    )
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 118, flush=True)
    print("STIR-Net Investigation 07 — residual RAG edge diagnosis", flush=True)
    print("=" * 118, flush=True)
    print(f"Calibration run       : {calibration_run}", flush=True)
    print(f"Source stage-05 run   : {source_run}", flush=True)
    print(f"Checkpoint step       : {checkpoint_step}", flush=True)
    print(f"Working threshold     : {threshold:.6f}", flush=True)
    print(f"Crops                 : {len(crops)}", flush=True)
    print(f"Samples               : {sorted({crop.sample for crop in crops})}", flush=True)
    print(f"Output                : {output}", flush=True)
    print("=" * 118, flush=True)

    # ------------------------------------------------------------------
    # A. Read all frozen RAG edges and construct truth-relative references.
    # ------------------------------------------------------------------
    all_rows: list[dict[str, Any]] = []
    crop_by_key: dict[tuple[str, int], CropRef] = {}
    for crop in crops:
        crop_by_key[(crop.sample, crop.crop_index)] = crop
        all_rows.extend(_load_edge_rows(crop, threshold))

    valid_rows = [row for row in all_rows if row["valid"]]
    references, feature_reference_table = _feature_references(valid_rows)

    for row in valid_rows:
        _attach_relative_scores(row, references)

    residual_rows = [
        row
        for row in valid_rows
        if row["edge_kind"] in {"false_merge", "false_cut"}
    ]

    for row in residual_rows:
        category, flags = _diagnostic_category(row)
        row["primary_category"] = category
        row["diagnostic_flags"] = flags

    # ------------------------------------------------------------------
    # B. Exact saved-interface analysis, crop by crop.
    # ------------------------------------------------------------------
    validation_rows: list[dict[str, Any]] = []
    interface_audits: list[dict[str, Any]] = []

    residual_by_crop: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in residual_rows:
        residual_by_crop[(row["sample"], row["crop_index"])].append(row)

    for key, rows in sorted(residual_by_crop.items()):
        crop = crop_by_key[key]
        arrays = _load_npz(crop)
        interface_stats, audit = _interface_observations(arrays, rows)

        node_count = int(arrays["primary_supervoxels"].max(initial=0))
        for row in rows:
            node_a = int(row["node_a"])
            node_b = int(row["node_b"])
            pack = min(node_a, node_b) * node_count + max(node_a, node_b)
            stats = interface_stats.get(pack)
            if stats is None:
                row["interface_found"] = False
            else:
                row["interface_found"] = True
                row.update(stats)

        _attach_node_volume_stats(arrays, rows)
        validation_rows.append(_feature_validation_row(crop, rows))
        interface_audits.append(
            {
                "sample": crop.sample,
                "crop_index": crop.crop_index,
                "candidate_type": crop.candidate_type,
                **audit,
            }
        )

    # ------------------------------------------------------------------
    # C. Rank residuals.
    # ------------------------------------------------------------------
    false_merges = sorted(
        [row for row in residual_rows if row["edge_kind"] == "false_merge"],
        key=lambda row: (
            -float(row["wrong_confidence"]),
            -float(row.get("interface_area_um2", 0.0)),
        ),
    )
    false_cuts = sorted(
        [row for row in residual_rows if row["edge_kind"] == "false_cut"],
        key=lambda row: (
            -float(row["wrong_confidence"]),
            -float(row.get("interface_area_um2", 0.0)),
        ),
    )

    for rank, row in enumerate(false_merges):
        row["rank"] = rank
    for rank, row in enumerate(false_cuts):
        row["rank"] = rank

    group_summary = _group_summary(valid_rows)
    category_table = _category_summary(residual_rows)

    # ------------------------------------------------------------------
    # D. Separator-specific conclusion.
    # ------------------------------------------------------------------
    fm_count = len(false_merges)
    weak_separator_count = sum(
        "weak_separator" in row["diagnostic_flags"] for row in false_merges
    )
    strong_separator_count = sum(
        "strong_separator" in row["diagnostic_flags"] for row in false_merges
    )
    proposal_risk_count = sum(
        row["primary_category"] == "proposal_or_node_quality_risk"
        for row in false_merges
    )

    weak_fraction = weak_separator_count / fm_count if fm_count else float("nan")
    strong_fraction = strong_separator_count / fm_count if fm_count else float("nan")
    proposal_fraction = proposal_risk_count / fm_count if fm_count else float("nan")

    if fm_count == 0:
        separator_message = "No residual valid false-merge edges at this threshold."
    elif math.isfinite(weak_fraction) and weak_fraction >= 0.50:
        separator_message = (
            "Most residual false merges have separator evidence closer to true-merge "
            "interfaces than true-cut interfaces. This is direct evidence that missed/"
            "weak separator prediction is a major remaining spatial bottleneck."
        )
    elif math.isfinite(strong_fraction) and strong_fraction >= 0.50:
        separator_message = (
            "Most residual false merges already contain strong cut-like separator "
            "evidence. The separator head is therefore not the dominant explanation; "
            "RAG use of geometry/context deserves priority."
        )
    else:
        separator_message = (
            "Residual false merges are mixed: neither weak separator nor ignored strong "
            "separator dominates. Inspect the ranked interface cases before changing "
            "separator loss or RAG architecture."
        )

    # ------------------------------------------------------------------
    # E. Persist tables and plots.
    # ------------------------------------------------------------------
    _write_csv(output / "feature_reference.csv", feature_reference_table)
    _write_csv(output / "edge_group_summary.csv", group_summary)
    _write_csv(output / "residual_edge_diagnostics.csv", residual_rows)
    _write_csv(output / "false_merges_ranked.csv", false_merges)
    _write_csv(output / "false_cuts_ranked.csv", false_cuts)
    _write_csv(output / "category_summary.csv", category_table)
    _write_csv(output / "interface_feature_validation.csv", validation_rows)

    if not args.no_plots:
        _save_distribution_plot(
            output / "residual_feature_distributions.png",
            all_rows,
            threshold,
        )
        _save_false_merge_separator_plot(
            output / "false_merge_separator_summary.png",
            residual_rows,
        )

    # Selected static visual cases.
    if args.png_per_kind > 0 and not args.no_plots:
        selected_visual_rows = (
            false_merges[: args.png_per_kind]
            + false_cuts[: args.png_per_kind]
        )
        visual_by_crop: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in selected_visual_rows:
            visual_by_crop[(row["sample"], row["crop_index"])].append(row)

        visual_dir = output / "visuals"
        for key, rows in visual_by_crop.items():
            crop = crop_by_key[key]
            arrays = _load_npz(crop)
            for row in rows:
                filename = (
                    f"{row['edge_kind']}_rank_{int(row['rank']):02d}_"
                    f"{row['sample']}_crop_{int(row['crop_index']):02d}_"
                    f"edge_{int(row['edge_row']):05d}.png"
                )
                _save_edge_visual(visual_dir / filename, arrays, row)

    overall_validation_max = max(
        (
            float(row["overall_max_abs_error"])
            for row in validation_rows
            if math.isfinite(float(row["overall_max_abs_error"]))
        ),
        default=float("nan"),
    )

    summary = {
        "status": "success",
        "calibration_run": str(calibration_run),
        "source_stage05_run": str(source_run),
        "checkpoint_step": checkpoint_step,
        "threshold": threshold,
        "sample_count": len({crop.sample for crop in crops}),
        "crop_count": len(crops),
        "valid_edge_count": len(valid_rows),
        "true_merge_count": sum(
            row["edge_kind"] == "true_merge" for row in valid_rows
        ),
        "true_cut_count": sum(
            row["edge_kind"] == "true_cut" for row in valid_rows
        ),
        "false_merge_count": len(false_merges),
        "false_cut_count": len(false_cuts),
        "separator_diagnosis": {
            "false_merge_count": fm_count,
            "weak_separator_false_merge_count": weak_separator_count,
            "weak_separator_false_merge_fraction": weak_fraction,
            "strong_separator_ignored_false_merge_count": strong_separator_count,
            "strong_separator_ignored_false_merge_fraction": strong_fraction,
            "proposal_or_node_quality_risk_false_merge_count": proposal_risk_count,
            "proposal_or_node_quality_risk_false_merge_fraction": proposal_fraction,
            "interpretation": separator_message,
        },
        "category_summary": category_table,
        "interface_feature_validation": {
            "overall_max_abs_error": overall_validation_max,
            "per_crop": validation_rows,
            "audit": interface_audits,
            "note": (
                "Recomputed interface means/maxima should closely match the saved "
                "production RAG raw edge features. Small differences can arise from "
                "float16 NPZ storage."
            ),
        },
        "top_false_merges": [
            {
                "rank": row["rank"],
                "sample": row["sample"],
                "crop_index": row["crop_index"],
                "edge_row": row["edge_row"],
                "probability": row["probability"],
                "primary_category": row["primary_category"],
                "separator_cut_score": row["separator_cut_score"],
                "geometry_cut_score": row["geometry_cut_score"],
                "interface_separator_mean": row.get("interface_separator_mean"),
                "interface_separator_fraction_ge_0p5": row.get(
                    "interface_separator_fraction_ge_0p5"
                ),
            }
            for row in false_merges[:10]
        ],
        "top_false_cuts": [
            {
                "rank": row["rank"],
                "sample": row["sample"],
                "crop_index": row["crop_index"],
                "edge_row": row["edge_row"],
                "probability": row["probability"],
                "primary_category": row["primary_category"],
                "separator_cut_score": row["separator_cut_score"],
                "geometry_cut_score": row["geometry_cut_score"],
            }
            for row in false_cuts[:10]
        ],
        "output_root": str(output),
    }
    _write_json(output / "diagnostic_summary.json", summary)

    print("\n" + "=" * 118, flush=True)
    print("RESIDUAL EDGE DIAGNOSIS COMPLETE", flush=True)
    print("=" * 118, flush=True)
    print(f"Valid edges                 : {len(valid_rows)}", flush=True)
    print(f"False merges                : {len(false_merges)}", flush=True)
    print(f"False cuts                  : {len(false_cuts)}", flush=True)
    print(
        f"Weak-separator false merges : {weak_separator_count}/{fm_count} "
        f"({weak_fraction:.1%})" if fm_count else
        "Weak-separator false merges : n/a",
        flush=True,
    )
    print(
        f"Strong-separator ignored    : {strong_separator_count}/{fm_count} "
        f"({strong_fraction:.1%})" if fm_count else
        "Strong-separator ignored    : n/a",
        flush=True,
    )
    print(
        f"Proposal/node-quality risk  : {proposal_risk_count}/{fm_count} "
        f"({proposal_fraction:.1%})" if fm_count else
        "Proposal/node-quality risk  : n/a",
        flush=True,
    )
    print(f"Separator interpretation    : {separator_message}", flush=True)
    print(
        f"Feature reproduction max err: {overall_validation_max:.6g}",
        flush=True,
    )
    print(f"Output                      : {output}", flush=True)
    print("=" * 118, flush=True)

    if args.napari:
        collection = false_merges if args.edge_kind == "false_merge" else false_cuts
        if not collection:
            raise RuntimeError(f"No residual {args.edge_kind} edges to visualize.")
        if not 0 <= args.rank < len(collection):
            raise IndexError(
                f"--rank {args.rank} outside available {args.edge_kind} ranks "
                f"0..{len(collection) - 1}"
            )
        selected = collection[args.rank]
        crop = crop_by_key[(selected["sample"], selected["crop_index"])]
        print(
            f"[napari] opening {args.edge_kind} rank={args.rank} "
            f"p={float(selected['probability']):.4f} "
            f"category={selected['primary_category']}",
            flush=True,
        )
        _open_napari(crop, selected)

    return summary


# ======================================================================================
# CLI
# ======================================================================================


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose residual false-merge / false-cut RAG edges from frozen "
            "Investigation-05 artifacts at the Investigation-06 calibrated threshold."
        )
    )
    parser.add_argument(
        "--calibration-run",
        default=None,
        help=(
            "Investigation-06 run directory. Default: latest completed run at the "
            "highest checkpoint step."
        ),
    )
    parser.add_argument(
        "--source-run",
        default=None,
        help=(
            "Optional Investigation-05 run override. Default: follow source_run "
            "recorded in calibration_summary.json."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional merge-threshold override. Default: recommended_threshold "
            "from Investigation 06."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--png-per-kind",
        type=int,
        default=6,
        help=(
            "Static detailed PNGs for the highest-confidence false merges and "
            "false cuts (default: 6 each)."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all PNG plots/visual summaries.",
    )
    parser.add_argument(
        "--napari",
        action="store_true",
        help="Open one ranked residual edge in Napari after analysis.",
    )
    parser.add_argument(
        "--edge-kind",
        choices=("false_merge", "false_cut"),
        default="false_merge",
        help="Residual edge type to open with --napari (default: false_merge).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank within --edge-kind to open with --napari (default: 0).",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.png_per_kind < 0:
        raise ValueError("--png-per-kind cannot be negative")
    if args.rank < 0:
        raise ValueError("--rank cannot be negative")
    summary = evaluate(args)
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
