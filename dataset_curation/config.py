
from __future__ import annotations
import os
from pathlib import Path
from ._repo import repo_root

DEFAULT_DATASET = "biohub"
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)
CURATION_ROOT_ENV = "CELL_TRACKING_CURATION_ROOT"

def curation_root(value: str | Path | None = None) -> Path:
    if value is not None:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (repo_root() / path).resolve()
    env = os.environ.get(CURATION_ROOT_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return (repo_root() / "data" / "curation").resolve()
