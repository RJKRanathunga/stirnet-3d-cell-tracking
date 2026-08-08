"""CLI for verifying external 3-D dataset discovery, axes, spacing, and labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..adapters.registry import dataset_choices, make_adapter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=dataset_choices(), default="c_elegans")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--split", default=None, help="train, val, test, or omitted")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--source-axis-order",
        default="zyx",
        help="native file axis order, e.g. zyx or xyz",
    )
    parser.add_argument(
        "--spacing-zyx-um",
        type=float,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=None,
        help="explicit spacing override for NIS3D/BlastoSPIM when metadata is insufficient",
    )
    parser.add_argument("--list", action="store_true", help="list discovered pairs and exit")
    parser.add_argument("--napari", action="store_true", help="open raw + labels in Napari")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    spacing_override = (
        tuple(float(v) for v in args.spacing_zyx_um)
        if args.spacing_zyx_um is not None
        else None
    )
    adapter = make_adapter(
        args.dataset,
        args.root,
        source_axis_order=args.source_axis_order,
        spacing_override_zyx_um=spacing_override,
    )
    records = adapter.records(split=args.split)
    if args.list:
        for index, record in enumerate(records):
            print(
                f"[{index:03d}] split={record.split:<11} id={record.sample_id} "
                f"image={record.image_path} labels={record.labels_path}"
            )
        print(f"records: {len(records)}")
        return
    if not records:
        raise SystemExit("No matching dataset records found.")
    if not -len(records) <= args.index < len(records):
        raise SystemExit(f"--index {args.index} is outside 0..{len(records)-1}")

    volume = adapter.load(records[args.index])
    ids = volume.instance_ids
    valid = volume.effective_valid_mask
    print(f"dataset: {volume.dataset_name}")
    print(f"sample:  {volume.sample_id}")
    print(f"split:   {volume.split}")
    print(f"shape:   {volume.shape_zyx} [Z,Y,X]")
    print(f"spacing: {volume.spacing_zyx_um} um [Z,Y,X]")
    print(
        f"image:   dtype={volume.image.dtype}, min={np.min(volume.image)}, "
        f"max={np.max(volume.image)}"
    )
    print(
        "labels:  dtype={}, instances={}, max_id={}".format(
            volume.instance_labels.dtype,
            len(ids),
            int(ids.max()) if ids.size else 0,
        )
    )
    print(
        f"valid:   {int(valid.sum())}/{valid.size} voxels "
        f"({100.0 * float(valid.mean()):.3f}%)"
    )
    print(f"normalization bounds: {volume.intensity_bounds}")
    if volume.metadata.get("confidence_counts") is not None:
        print(f"confidence counts: {volume.metadata['confidence_counts']}")

    if args.napari:
        try:
            import napari
        except ImportError as error:
            raise SystemExit("Napari is not installed in this environment.") from error
        viewer = napari.Viewer()
        viewer.add_image(volume.image, name="raw", scale=volume.spacing_zyx_um)
        viewer.add_labels(
            volume.instance_labels,
            name="GT instances",
            scale=volume.spacing_zyx_um,
        )
        if not np.all(valid):
            viewer.add_labels(
                (~valid).astype(np.uint8),
                name="invalid/undefined annotation",
                scale=volume.spacing_zyx_um,
                opacity=0.35,
            )
        napari.run()


if __name__ == "__main__":
    main()
