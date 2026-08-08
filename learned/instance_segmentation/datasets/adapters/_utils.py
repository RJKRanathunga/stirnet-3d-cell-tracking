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
    """Infer train/val/test from path components and archive-style names.

    This deliberately tokenizes names such as ``BlastoSPIM1_train`` and
    ``BlastoSPIM2_test_lowSNR`` rather than requiring the split token to be
    the first characters of a path component.
    """

    # Prefer the path component nearest the file. This avoids an unrelated
    # outer directory such as a temporary/test root overriding an extracted
    # archive name like ``BlastoSPIM1_train``.
    for part in reversed(path.parts):
        part_tokens = tokens(part)
        for token in part_tokens:
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
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return tifffile.imread(path)


def coerce_3d(array: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(array)
    result = np.squeeze(result)
    if result.ndim != 3:
        raise ValueError(f"{name} must be 3-D after squeezing, got {result.shape}")
    return result


def validate_axis_order(source_axis_order: str) -> str:
    order = source_axis_order.lower()
    if sorted(order) != ["x", "y", "z"] or len(order) != 3:
        raise ValueError("source_axis_order must be a permutation of 'zyx'")
    return order


def to_zyx(array: np.ndarray, source_axis_order: str) -> np.ndarray:
    order = validate_axis_order(source_axis_order)
    permutation = tuple(order.index(axis) for axis in "zyx")
    return np.transpose(array, permutation)


def normalize_instance_labels(array: np.ndarray) -> np.ndarray:
    labels_raw = np.asarray(array)
    if np.issubdtype(labels_raw.dtype, np.floating):
        if not np.all(np.isfinite(labels_raw)):
            raise ValueError("instance labels contain non-finite values")
        rounded = np.rint(labels_raw)
        if not np.allclose(labels_raw, rounded, atol=1e-5):
            raise ValueError("floating instance labels are not integer-valued")
        labels = rounded.astype(np.int32)
    else:
        labels = labels_raw.astype(np.int32, copy=False)
    if np.any(labels < 0):
        raise ValueError("instance labels contain negative values")
    return labels


def supported_volume_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
        )
    )


def find_casefold_file(directory: Path, names: Iterable[str]) -> Path | None:
    wanted = {name.lower() for name in names}
    for path in directory.iterdir():
        if path.is_file() and path.name.lower() in wanted:
            return path
    return None


def parse_three_floats(text: str) -> tuple[float, float, float] | None:
    values = re.findall(r"(?<![A-Za-z0-9])([0-9]+(?:\.[0-9]+)?)", text)
    if len(values) < 3:
        return None
    return tuple(float(value) for value in values[:3])  # type: ignore[return-value]


def sanitized_sample_id(parts: Iterable[str]) -> str:
    joined = "_".join(part for part in parts if part)
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", joined).strip("_")
    return clean or "sample"
