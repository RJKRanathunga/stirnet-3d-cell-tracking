from __future__ import annotations

from pathlib import Path


# DATASET_CURATION_FULL_BIOHUB_ROOT_V2
BIOHUB_DATA_ROOT = Path(r"E:\Data\cell-tracking\Datasets\full_biohub_80\biohub-cell-tracking-during-development")
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)


def biohub_data_root(
    override: str | Path | None = None,
) -> Path:
    """Return the fixed BioHub data root unless an explicit override is given."""
    if override is None:
        return BIOHUB_DATA_ROOT
    return Path(override).expanduser()
