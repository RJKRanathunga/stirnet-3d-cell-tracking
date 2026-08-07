"""Fuse duplicate source detections into one component-level review candidate."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .models import SOURCE_NAMES, parse_track_ids


def fuse_candidates(source_candidates: pd.DataFrame) -> pd.DataFrame:
    if source_candidates.empty:
        columns = [
            "candidate_id", "sample_id", "frame", "cell_id", "tier", "priority_score",
            *[f"has_{name}" for name in SOURCE_NAMES],
        ]
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for (sample_id, frame, cell_id), group in source_candidates.groupby(
        ["sample_id", "frame", "cell_id"], sort=True
    ):
        group = group.copy()
        finite_errors = pd.to_numeric(group["volume_sum_log_error"], errors="coerce")
        if finite_errors.notna().any():
            best_idx = finite_errors.idxmin()
        else:
            best_idx = pd.to_numeric(group["source_score"], errors="coerce").idxmax()
        best = group.loc[best_idx]

        sources = set(group["source"].astype(str))
        track_ids: set[int] = set()
        for value in group.get("involved_track_ids", pd.Series(dtype=object)):
            track_ids.update(parse_track_ids(value))
        for column in ("track_a", "track_b"):
            if column in group:
                for value in pd.to_numeric(group[column], errors="coerce").dropna():
                    track_ids.add(int(value))

        if "source2_two_one_two" in sources:
            tier = "A"
            base = 4.0
        elif "source1_multi_to_one" in sources:
            tier = "B"
            base = 3.0
        elif "source3_disappearing_track" in sources:
            tier = "C"
            base = 2.0
        else:
            tier = "D"
            base = 1.0

        support_bonus = 0.35 * max(0, len(sources) - 1)
        source_score = float(pd.to_numeric(group["source_score"], errors="coerce").max())
        volume_error = _finite(best.get("volume_sum_log_error"))
        volume_bonus = math.exp(-volume_error) if math.isfinite(volume_error) else 0.0
        priority = base + support_bonus + 0.25 * source_score + volume_bonus

        record = {
            "candidate_id": str(best["candidate_id"]),
            "sample_id": str(sample_id),
            "frame": int(frame),
            "cell_id": int(cell_id),
            "tier": tier,
            "priority_score": float(priority),
            "source_count": int(len(sources)),
            "sources": ";".join(sorted(sources)),
            "involved_track_ids": ";".join(str(v) for v in sorted(track_ids)),
            "track_a": _nullable_int(best.get("track_a")),
            "track_b": _nullable_int(best.get("track_b")),
            "candidate_volume": _finite(best.get("candidate_volume")),
            "volume_a_reference": _finite(best.get("volume_a_reference")),
            "volume_b_reference": _finite(best.get("volume_b_reference")),
            "volume_sum_ratio": _finite(best.get("volume_sum_ratio")),
            "volume_sum_log_error": volume_error,
            "prediction_a_distance_um": _finite(best.get("prediction_a_distance_um")),
            "prediction_b_distance_um": _finite(best.get("prediction_b_distance_um")),
            "prediction_a_inside": bool(best.get("prediction_a_inside", False)),
            "prediction_b_inside": bool(best.get("prediction_b_inside", False)),
            "edt_peak_count": int(max(pd.to_numeric(group.get("edt_peak_count", 0), errors="coerce").fillna(0).max(), 0)),
            "notes": " | ".join(sorted(set(group["notes"].fillna("").astype(str)) - {""})),
        }
        for name in SOURCE_NAMES:
            record[f"has_{name}"] = name in sources
        rows.append(record)

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["tier", "priority_score", "sample_id", "frame", "cell_id"],
        ascending=[True, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _finite(value) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _nullable_int(value):
    try:
        number = float(value)
        if math.isfinite(number):
            return int(number)
    except (TypeError, ValueError):
        pass
    return pd.NA


__all__ = ["fuse_candidates"]
