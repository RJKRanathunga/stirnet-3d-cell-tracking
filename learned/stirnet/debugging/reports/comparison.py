from __future__ import annotations

from typing import Any
import math

import numpy as np

from ..core.trace import DebugTrace


def _finite_mean(values) -> float:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else float("nan")


def summarize_trace(trace: DebugTrace) -> dict[str, Any]:
    """Collapse a trace into stable run-level metrics for iteration comparison."""

    queries = trace.tables.get("queries", [])
    masks = trace.tables.get("masks", [])

    result: dict[str, Any] = {
        "query_count": trace.metadata.get("query_count", len(queries)),
        "matched_query_count": trace.metadata.get(
            "matched_query_count",
            sum(bool(r.get("matched")) for r in queries),
        ),
        "surviving_query_count": trace.metadata.get(
            "surviving_query_count",
            sum(bool(r.get("survives_final_exist")) for r in queries),
        ),
        "mean_matched_center_error_um": _finite_mean(
            r.get("center_error_um") for r in queries if r.get("matched")
        ),
    }

    for query_type in ("primary", "split", "temporal", "discovery"):
        typed = [r for r in queries if r.get("query_type") == query_type]
        result[f"{query_type}_queries"] = len(typed)
        result[f"{query_type}_survivors"] = sum(
            bool(r.get("survives_final_exist")) for r in typed
        )

    if masks:
        for name in ("learned", "prior", "combined"):
            result[f"mean_{name}_soft_dice"] = _finite_mean(
                r.get(f"{name}_soft_dice") for r in masks
            )
            result[f"mean_{name}_hard_dice"] = _finite_mean(
                r.get(f"{name}_hard_dice") for r in masks
            )
            result[f"mean_{name}_volume_ratio"] = _finite_mean(
                r.get(f"{name}_volume_ratio") for r in masks
            )

    return result


def compare_traces(before: DebugTrace, after: DebugTrace) -> list[dict[str, Any]]:
    """Return row-oriented before/after/delta metrics for two debug traces."""

    left = summarize_trace(before)
    right = summarize_trace(after)
    rows = []
    for metric in sorted(set(left) | set(right)):
        before_value = left.get(metric, float("nan"))
        after_value = right.get(metric, float("nan"))
        try:
            delta = float(after_value) - float(before_value)
        except (TypeError, ValueError):
            delta = float("nan")
        rows.append(
            {
                "metric": metric,
                "before": before_value,
                "after": after_value,
                "delta": delta,
            }
        )
    return rows
