"""Build and inspect one complete object-centric cubic CNN sample."""

from __future__ import annotations

import argparse
import numpy as np

from ..adapters import dataset_choices, make_adapter
from ..config import DEFAULT_SAMPLE_BUILD_CONFIG
from ..core.adjacency import build_instance_adjacency
from ..core.models import InstanceGroup
from ..core.sample_builder import SampleBuilder
from ..core.sample_selection import select_valid_sample
from ..core.sampling import pair_groups


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=dataset_choices(), required=True)
    p.add_argument("--root")
    p.add_argument("--split")
    p.add_argument("--volume-index", type=int, default=0)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--pair-index", type=int, default=0, help="Nth buildable neighboring pair")
    selection.add_argument("--raw-pair-index", type=int, help="Raw neighboring-pair index")
    selection.add_argument(
        "--instance-ids", type=int, nargs="+",
        help="Build directly from known instance IDs, bypassing adjacency search",
    )
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
    builder = SampleBuilder()

    if args.instance_ids is not None:
        instance_ids = tuple(int(v) for v in args.instance_ids)
        group_kind = "single" if len(instance_ids) == 1 else "pair_merge" if len(instance_ids) == 2 else "manual_group"
        group = InstanceGroup(instance_ids=instance_ids, kind=group_kind)
        print("Using direct instance IDs; skipping adjacency construction.")
        sample = builder.build(volume, group)
        raw_index = None
        rejections = ()
    else:
        print("Building native neighboring-pair list...")
        edges = build_instance_adjacency(
            volume.instance_labels,
            volume.spacing_zyx_um,
            max_distance_um=DEFAULT_SAMPLE_BUILD_CONFIG.adjacency_max_distance_um,
        )
        groups = pair_groups(edges)
        print("raw neighboring pairs:", len(groups))
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
    print("scale [canonical vox/um]:", sample.transform.scale_vox_per_um)
    print("source bbox [um] ZYX:", sample.transform.component_bbox.extent_um_zyx)
    print("canonical bbox [vox] ZYX:", sample.metadata["canonical_group_bbox_extent_vox_zyx"])
    print("GT centers canonical ZYX:", sample.targets.centers_zyx)
    centers = np.asarray(sample.targets.centers_zyx, dtype=float)
    print("GT centers inverse-mapped native ZYX:", sample.transform.canonical_to_native(centers))
    print("marker count:", len(sample.marker_positions_zyx))

    if args.napari:
        import napari

        scale = DEFAULT_SAMPLE_BUILD_CONFIG.canonical_spacing_zyx
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(sample.inputs[0], name="fluorescence", scale=scale, rendering="mip")
        viewer.add_labels(
            sample.input_component_mask.astype(np.int32), name="input component",
            scale=scale, opacity=0.35,
        )
        viewer.add_labels(sample.targets.instance_labels, name="GT instances", scale=scale, opacity=0.65)
        viewer.add_image(sample.edt_normalized, name="canonical EDT", scale=scale, visible=False, rendering="mip")
        viewer.add_image(sample.marker_heatmap, name="marker heatmap", scale=scale, visible=False, rendering="mip")
        viewer.add_image(sample.targets.center[0], name="GT center heatmap", scale=scale, visible=False, rendering="mip")
        viewer.add_image(sample.targets.boundary[0], name="GT internal boundary", scale=scale, visible=False, rendering="mip")
        if len(centers):
            viewer.add_points(centers, name="GT centers", size=3, scale=scale)
        if sample.marker_positions_zyx:
            viewer.add_points(np.asarray(sample.marker_positions_zyx), name="effective markers", size=3, scale=scale)
        viewer.reset_view()
        napari.run()


if __name__ == "__main__":
    main()
