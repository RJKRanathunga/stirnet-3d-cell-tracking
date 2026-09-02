from __future__ import annotations

"""Source-only BioHub volume visualization with no inference dependency."""

from dataset_curation.annotation.source_data import (
    estimate_contrast_limits,
    open_source_movie,
)
from dataset_curation.catalog import VolumeRecord
from dataset_curation.config import DEFAULT_SPACING_ZYX_UM


def view_source_volume(
    record: VolumeRecord,
) -> None:
    import napari

    raw = open_source_movie(
        record.paths.zarr
    )
    low, high = estimate_contrast_limits(
        raw
    )
    scale_tzyx = (
        1.0,
        *DEFAULT_SPACING_ZYX_UM,
    )

    viewer = napari.Viewer(
        ndisplay=3
    )
    viewer.dims.axis_labels = (
        "time",
        "z",
        "y",
        "x",
    )
    viewer.add_image(
        raw,
        name="Raw BioHub",
        scale=scale_tzyx,
        rendering="mip",
        colormap="gray",
        contrast_limits=[
            float(low),
            float(high),
        ],
    )

    try:
        viewer.dims.set_current_step(
            0,
            0,
        )
        viewer.dims.set_current_step(
            1,
            int(raw.shape[1]) // 2,
        )
    except Exception:
        pass

    print("=" * 88)
    print("BIOHUB SOURCE VIEW")
    print("=" * 88)
    print(f"split  : {record.split}")
    print(f"volume : {record.volume_id}")
    print(f"source : {record.paths.zarr}")
    print(f"shape  : {tuple(int(v) for v in raw.shape)}")
    print("=" * 88)

    napari.run()
