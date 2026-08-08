"""Adapter registry used by debugging tools and future training entry points."""

from __future__ import annotations

from pathlib import Path

from .blastospim import BlastoSPIMAdapter
from .c_elegans import CElegansNucleiAdapter
from .nis3d import NIS3DAdapter

_DATASET_ALIASES = {
    "c_elegans": "c_elegans",
    "c-elegans": "c_elegans",
    "celegans": "c_elegans",
    "c_elegans_nuclei": "c_elegans",
    "nis3d": "nis3d",
    "blastospim": "blastospim",
    "blasto_spim": "blastospim",
}


def dataset_choices() -> tuple[str, ...]:
    return ("c_elegans", "nis3d", "blastospim")


def canonical_dataset_name(name: str) -> str:
    key = name.lower().strip()
    try:
        return _DATASET_ALIASES[key]
    except KeyError as error:
        raise ValueError(
            f"unknown dataset {name!r}; choose one of {', '.join(dataset_choices())}"
        ) from error


def make_adapter(
    dataset: str,
    root: str | Path | None = None,
    *,
    source_axis_order: str = "zyx",
    spacing_override_zyx_um: tuple[float, float, float] | None = None,
):
    """Construct one external-dataset adapter from a canonical/alias name."""

    canonical = canonical_dataset_name(dataset)
    if canonical == "c_elegans":
        if spacing_override_zyx_um is not None:
            raise ValueError(
                "C. elegans spacing is fixed by the published dataset; "
                "--spacing-zyx-um is only needed for NIS3D/BlastoSPIM overrides"
            )
        return CElegansNucleiAdapter(root, source_axis_order=source_axis_order)
    if canonical == "nis3d":
        return NIS3DAdapter(
            root,
            source_axis_order=source_axis_order,
            spacing_override_zyx_um=spacing_override_zyx_um,
        )
    return BlastoSPIMAdapter(
        root,
        source_axis_order=source_axis_order,
        spacing_override_zyx_um=spacing_override_zyx_um,
    )


__all__ = ["canonical_dataset_name", "dataset_choices", "make_adapter"]
