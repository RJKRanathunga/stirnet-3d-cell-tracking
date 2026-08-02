"""CSV and JSON helpers preserving existing pandas serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Any

import pandas as pd
from pandas.errors import EmptyDataError


def _validate_columns(df: pd.DataFrame, required_columns: Iterable[str] | None, path: Path) -> None:
    if required_columns is None:
        return
    missing = [name for name in required_columns if name not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")


def load_csv(path: str | Path, *, required_columns: Iterable[str] | None = None) -> pd.DataFrame:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Expected CSV does not exist: {target}")
    frame = pd.read_csv(target)
    _validate_columns(frame, required_columns, target)
    return frame


def load_optional_csv(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if not target.exists() or target.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(target)
    except EmptyDataError:
        return pd.DataFrame()


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)


def load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Expected JSON does not exist: {target}")
    with target.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {target}")
    return value


def save_json(value: Mapping[str, Any], path: str | Path, *, indent: int = 4) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(dict(value), stream, indent=indent)
