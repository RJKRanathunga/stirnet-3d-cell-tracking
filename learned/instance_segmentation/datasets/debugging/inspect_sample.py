"""Build and inspect one complete object-centric canonical CNN sample."""

from __future__ import annotations

import argparse
import numpy as np

from ..adapters import dataset_choices, make_adapter
from ..config import DEFAULT_SAMPLE_BUILD_CONFIG
from ..core.adjacency import build_instance_adjacency
from ..core.sample_builder import SampleBuilder
from ..core.sample_selection import select_valid_sample
from ..core.sampling import pair_groups


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=dataset_choices(), required=True)
    p.add_argument("--root")
    p.add_argument("--split")
    p.add_argument("--volume-index", type=int, default=0)
    p.add_argument("--pair-index", type=int, default=0, help="Nth buildable pair")
    p.add_argument("--raw-pair-index", type=int)
    p.add_argument("--source-axis-order", default="zyx")
    p.add_argument("--spacing-zyx-um", type=float, nargs=3)
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
    volume = adapter.load(records[args.volume_index])
    print("Building native neighboring-pair list...")
    edges = build_instance_adjacency(
        volume.instance_labels,
        volume.spacing_zyx_um,
        max_distance_um=DEFAULT_SAMPLE_BUILD_CONFIG.adjacency_max_distance_um,
    )
    groups = pair_groups(edges)
    print("raw neighboring pairs:", len(groups))
    builder = SampleBuilder()
    if args.raw_pair_index is not None:
        group = groups[args.raw_pair_index]
        sample = builder.build(volume, group)
        raw_index = args.raw_pair_index
        rejections = ()
    else:
        selection = select_valid_sample(volume, groups, builder, valid_index=args.pair_index)
        sample = selection.sample
        group = selection.group
        raw_index = selection.raw_index
        rejections = selection.rejections_before
    print("selected group:", group.instance_ids, group.kind)
    print("raw pair index:", raw_index)
    print("rejections before:", len(rejections))
    print("inputs:", sample.inputs.shape, sample.inputs.dtype)
    print("normalization scale:", sample.transform.normalization_scale)
    print("source bbox [um] ZYX:", sample.transform.component_bbox.extent_um_zyx)
    print("canonical bbox units ZYX:", sample.metadata["canonical_group_bbox_extent_zyx"])
    print("GT centers canonical ZYX:", sample.targets.centers_zyx)
    centers = np.asarray(sample.targets.centers_zyx, dtype=float)
    print("GT centers inverse-mapped native ZYX:", sample.transform.canonical_to_native(centers))
    print("marker count:", len(sample.marker_positions_zyx))

    if args.napari:
        import napari
        viewer = napari.Viewer()
        viewer.add_image(sample.inputs[0], name="fluorescence")
        viewer.add_labels(sample.input_component_mask.astype(np.int32), name="input component")
        viewer.add_labels(sample.targets.instance_labels, name="GT instances")
        viewer.add_image(sample.edt_normalized, name="canonical EDT", visible=False)
        viewer.add_image(sample.marker_heatmap, name="marker heatmap", visible=False)
        viewer.add_image(sample.targets.center[0], name="GT center heatmap", visible=False)
        viewer.add_image(sample.targets.boundary[0], name="GT internal boundary", visible=False)
        if len(centers):
            viewer.add_points(centers, name="GT centers", size=3)
        if sample.marker_positions_zyx:
            viewer.add_points(np.asarray(sample.marker_positions_zyx), name="effective markers", size=3)
        napari.run()


if __name__ == "__main__":
    main()
