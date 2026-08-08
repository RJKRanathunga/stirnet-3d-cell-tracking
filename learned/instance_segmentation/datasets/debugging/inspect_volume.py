"""CLI for verifying external 3-D dataset discovery, axes, spacing, and labels."""

from __future__ import annotations

import argparse
import numpy as np

from ..adapters import dataset_choices, make_adapter


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=dataset_choices(), required=True)
    p.add_argument("--root")
    p.add_argument("--split")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--source-axis-order", default="zyx")
    p.add_argument("--spacing-zyx-um", type=float, nargs=3)
    p.add_argument("--list", action="store_true")
    p.add_argument("--napari", action="store_true")
    return p


def main() -> None:
    args = _parser().parse_args()
    adapter = make_adapter(
        args.dataset,
        root=args.root,
        source_axis_order=args.source_axis_order,
        spacing_override_zyx_um=tuple(args.spacing_zyx_um) if args.spacing_zyx_um else None,
    )
    records = adapter.records(split=args.split) if hasattr(adapter, "records") else adapter.discover_records()
    print(f"records: {len(records)}")
    for index, record in enumerate(records):
        print(index, record.sample_id, record.split, record.image_path.name, record.labels_path.name)
    if args.list:
        return
    if not records:
        raise SystemExit("no matching records")
    volume = adapter.load(records[args.index])
    labels = volume.instance_labels
    print("dataset:", volume.dataset_name)
    print("sample:", volume.sample_id)
    print("shape ZYX:", volume.shape_zyx)
    print("spacing ZYX [um]:", volume.spacing_zyx_um)
    print("image:", volume.image.dtype, float(np.min(volume.image)), float(np.max(volume.image)))
    print("instances:", len(volume.instance_ids))
    print("valid fraction:", float(np.mean(volume.effective_valid_mask)))
    print("intensity bounds:", volume.intensity_bounds)
    if args.napari:
        import napari
        viewer = napari.Viewer()
        viewer.add_image(volume.image, name="image", scale=volume.spacing_zyx_um)
        viewer.add_labels(labels, name="labels", scale=volume.spacing_zyx_um)
        viewer.add_labels(volume.effective_valid_mask.astype(np.uint8), name="valid", visible=False, scale=volume.spacing_zyx_um)
        napari.run()


if __name__ == "__main__":
    main()
