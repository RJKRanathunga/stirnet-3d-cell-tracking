"""Adapter registry used by debugging and future training entry points."""

from __future__ import annotations

from pathlib import Path

from .blastospim import BlastoSPIMAdapter
from .c_elegans import CElegansNucleiAdapter
from .nis3d import NIS3DAdapter

_ALIASES = {
    "c_elegans": "c_elegans",
    "celegans": "c_elegans",
    "c-elegans": "c_elegans",
    "nis3d": "nis3d",
    "blastospim": "blastospim",
    "blasto_spim": "blastospim",
}


def canonical_dataset_name(name: str) -> str:
    key = name.lower().strip()
    if key not in _ALIASES:
        raise ValueError(f"unknown dataset {name!r}; choose from {dataset_choices()}")
    return _ALIASES[key]


def dataset_choices() -> tuple[str, ...]:
    return ("c_elegans", "nis3d", "blastospim")


def make_adapter(
    dataset: str,
    *,
    root: str | Path | None = None,
    source_axis_order: str = "zyx",
    spacing_override_zyx_um: tuple[float, float, float] | None = None,
):
    canonical = canonical_dataset_name(dataset)
    kwargs = {"root": root, "source_axis_order": source_axis_order}
    if canonical == "nis3d":
        return NIS3DAdapter(
            **kwargs,
            spacing_override_zyx_um=spacing_override_zyx_um,
        )
    if canonical == "c_elegans":
        return CElegansNucleiAdapter(**kwargs)
    return BlastoSPIMAdapter(**kwargs)


__all__ = ["canonical_dataset_name", "dataset_choices", "make_adapter"]
