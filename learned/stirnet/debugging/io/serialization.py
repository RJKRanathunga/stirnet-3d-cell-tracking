from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.trace import DebugTrace


def _json_safe(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                result.append(key)
    return result


def save_debug_trace(trace: DebugTrace, out_dir) -> Path:
    out = Path(out_dir)
    tables_dir = out / "tables"
    arrays_dir = out / "arrays"
    tables_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)

    table_manifest = {}
    for name, rows in trace.tables.items():
        path = tables_dir / f"{name}.csv"
        fields = _fieldnames(rows)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if fields:
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: _json_safe(v) for k, v in row.items()})
        table_manifest[name] = str(path.relative_to(out))

    array_manifest = {}
    for name, array in trace.arrays.items():
        rel = Path(*name.split("/")).with_suffix(".npy")
        path = arrays_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(array))
        array_manifest[name] = {
            "path": str(path.relative_to(out)),
            "shape": list(np.asarray(array).shape),
            "dtype": str(np.asarray(array).dtype),
        }

    manifest = {
        "metadata": _json_safe(trace.metadata),
        "tables": table_manifest,
        "arrays": array_manifest,
    }
    with (out / "trace.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=True)

    return out


def _coerce_csv(value: str):
    if value == "":
        return ""
    low = value.lower()
    if low == "nan":
        return float("nan")
    if low in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if low in {"-inf", "-infinity"}:
        return float("-inf")
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_debug_trace(out_dir, *, mmap: bool = False) -> DebugTrace:
    out = Path(out_dir)
    with (out / "trace.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    trace = DebugTrace(metadata=manifest.get("metadata", {}))
    for name, rel in manifest.get("tables", {}).items():
        path = out / rel
        rows = []
        if path.exists() and path.stat().st_size:
            with path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    rows.append({k: _coerce_csv(v) for k, v in row.items()})
        trace.tables[name] = rows

    mode = "r" if mmap else None
    for name, info in manifest.get("arrays", {}).items():
        trace.arrays[name] = np.load(out / info["path"], mmap_mode=mode)

    return trace
