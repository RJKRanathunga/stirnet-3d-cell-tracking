from __future__ import annotations

import os
from pathlib import Path

from ._repo import repo_root


BIOHUB_DATA_ROOT = Path(r"E:\data\biohub")
DEFAULT_DATASET = "biohub"
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)

# Legacy first-stage workspace support.
CURATION_ROOT_ENV = "CELL_TRACKING_CURATION_ROOT"

# Advanced override only. Normal curation commands use BIOHUB_DATA_ROOT.
BIOHUB_DATA_ROOT_ENV = "CELL_TRACKING_BIOHUB_DATA_ROOT"


def biohub_data_root(value: str | Path | None = None) -> Path:
    if value is not None:
        return Path(value).expanduser()

    env = os.environ.get(BIOHUB_DATA_ROOT_ENV)
    if env:
        return Path(env).expanduser()

    return BIOHUB_DATA_ROOT


def curation_root(value: str | Path | None = None) -> Path:
    """Legacy workspace resolver retained for backwards compatibility."""
    if value is not None:
        path = Path(value).expanduser()
        return (
            path.resolve()
            if path.is_absolute()
            else (repo_root() / path).resolve()
        )
    env = os.environ.get(CURATION_ROOT_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return (repo_root() / "data" / "curation").resolve()
