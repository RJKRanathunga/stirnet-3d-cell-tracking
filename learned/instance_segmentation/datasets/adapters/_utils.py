"""Shared I/O and discovery helpers for external 3-D dataset adapters."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
import tifffile

_SUPPORTED_SUFFIXES = {".tif", ".tiff", ".npy"}
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


def tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in re.split(r"[^a-z0-9]+", text.lower()) if token)


def path_tokens(path: Path) -> tuple[str, ...]:
    result: list[str] = []
    for part in path.parts:
        result.extend(tokens(part))
    return tuple(result)


def infer_split(path: Path) -> str:
    for part in reversed(path.parts):
        for token in tokens(part):
            if token in _SPLIT_ALIASES:
                return _SPLIT_ALIASES[token]
        normalized = part.lower().replace("-", "_")
        for alias, canonical in _SPLIT_ALIASES.items():
            if normalized == alias or normalized.startswith(alias + "_"):
                return canonical
    return "unspecified"


def canonical_split(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.lower().replace("-", "_")
    return _SPLIT_ALIASES.get(normalized, normalized)


def load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix in {".tif", ".tiff"}:
        return tifffile.imread(path)
    raise ValueError(f"unsupported 3-D array file: {path}")


def coerce_3d(array: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(array)
    while result.ndim > 3 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != 3:
        raise ValueError(f"{name} must be 3-D after removing singleton leading axes, got {result.shape}")
    return result


def validate_axis_order(order: str) -> str:
    normalized = order.lower().strip()
    if len(normalized) != 3 or set(normalized) != {"x", "y", "z"}:
        raise ValueError("source_axis_order must be a permutation of 'xyz'")
    return normalized


def to_zyx(array: np.ndarray, source_axis_order: str) -> np.ndarray:
    order = validate_axis_order(source_axis_order)
    permutation = tuple(order.index(axis) for axis in "zyx")
    return np.transpose(np.asarray(array), permutation)


def normalize_instance_labels(labels: np.ndarray) -> np.ndarray:
    source = np.asarray(labels)
    if not np.issubdtype(source.dtype, np.integer):
        rounded = np.rint(source)
        if not np.allclose(source, rounded):
            raise TypeError("instance labels contain non-integer values")
        source = rounded
    source = source.astype(np.int64, copy=False)
    if np.any(source < 0):
        raise ValueError("instance labels cannot be negative")
    unique = np.unique(source)
    positive = unique[unique > 0]
    result = np.zeros(source.shape, dtype=np.int32)
    for new_id, old_id in enumerate(positive, start=1):
        result[source == old_id] = new_id
    return result


def find_casefold_file(directory: Path, names: Iterable[str]) -> Path | None:
    wanted = {name.casefold() for name in names}
    try:
        children = list(directory.iterdir())
    except OSError:
        return None
    for child in children:
        if child.is_file() and child.name.casefold() in wanted:
            return child
    return None


def sanitized_sample_id(parts: Iterable[str]) -> str:
    joined = "_".join(part for part in parts if part)
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", joined).strip("_")
    return clean or "sample"


def supported_array_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
    )


def compact_stem(path: Path, drop_tokens: Iterable[str] = ()) -> str:
    drop = {token.lower() for token in drop_tokens}
    stem_tokens = [token for token in tokens(path.stem) if token not in drop]
    return "".join(stem_tokens)
