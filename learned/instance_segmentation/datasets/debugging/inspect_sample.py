"""Build and inspect one valid neighboring-pair Vector-CNN training sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..adapters.registry import dataset_choices, make_adapter
from ..config import DEFAULT_SAMPLE_BUILD_CONFIG
from ..core.adjacency import build_instance_adjacency
from ..core.sample_builder import SampleBuildError, SampleBuilder
from ..core.sample_selection import select_valid_sample
from ..core.sampling import pair_groups


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=dataset_choices(), default="c_elegans")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--volume-index", type=int, default=0)
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
    parser.add_argument(
        "--pair-index",
        type=int,
        default=0,
        help="index among VALID/buildable neighboring pairs (invalid raw pairs are skipped)",
    )
    parser.add_argument(
        "--raw-pair-index",
        type=int,
        default=None,
        help="debug one raw adjacency pair without validity-aware skipping",
    )
    parser.add_argument(
        "--max-raw-candidates",
        type=int,
        default=None,
        help="optional safety limit while searching for the requested valid pair",
    )
    parser.add_argument(
        "--show-rejections",
        type=int,
        default=5,
        help="print up to this many skipped raw-pair diagnostics",
    )
    parser.add_argument("--napari", action="store_true")
    parser.add_argument("--vector-step", type=int, default=4)
    return parser


def _napari_vectors(sample, spacing, step: int) -> np.ndarray:
    labels = sample.targets.instance_labels
    vectors = sample.targets.vectors_normalized
    coords = np.argwhere(labels > 0)
    if step > 1:
        coords = coords[::step]
    if coords.size == 0:
        return np.empty((0, 2, 3), dtype=np.float32)
    normalized = vectors[:, coords[:, 0], coords[:, 1], coords[:, 2]].T
    max_um = float(sample.metadata["config"]["vector_max_distance_um"])
    delta_vox = normalized * max_um / np.asarray(spacing, dtype=np.float32)
    result = np.zeros((len(coords), 2, 3), dtype=np.float32)
    result[:, 0, :] = coords
    result[:, 1, :] = delta_vox
    return result


def main() -> None:
    args = _build_parser().parse_args()
    if args.pair_index < 0:
        raise SystemExit("--pair-index cannot be negative")
    if args.raw_pair_index is not None and args.raw_pair_index < 0:
        raise SystemExit("--raw-pair-index cannot be negative")

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
    volume = adapter.load_index(args.volume_index, split=args.split)
    config = DEFAULT_SAMPLE_BUILD_CONFIG
    edges = build_instance_adjacency(
        volume.instance_labels,
        volume.spacing_zyx_um,
        max_distance_um=config.adjacency_max_distance_um,
    )
    groups = pair_groups(edges)
    if not groups:
        raise SystemExit("No neighboring pair candidates were found in this volume.")

    builder = SampleBuilder(config)
    if args.raw_pair_index is not None:
        if args.raw_pair_index >= len(groups):
            raise SystemExit(
                f"--raw-pair-index {args.raw_pair_index} is outside 0..{len(groups)-1}"
            )
        raw_index = args.raw_pair_index
        group = groups[raw_index]
        try:
            sample = builder.build(volume, group)
        except SampleBuildError as error:
            raise SystemExit(
                f"Raw pair {raw_index} {group.instance_ids} is not buildable:\n{error}"
            ) from error
        rejections = ()
        valid_index_text = "direct raw-pair mode"
    else:
        try:
            selection = select_valid_sample(
                volume,
                groups,
                builder,
                valid_index=args.pair_index,
                max_raw_candidates=args.max_raw_candidates,
            )
        except IndexError as error:
            raise SystemExit(str(error)) from error
        raw_index = selection.raw_index
        group = selection.group
        sample = selection.sample
        rejections = selection.rejections_before
        valid_index_text = str(selection.valid_index)

    if rejections:
        show = max(0, int(args.show_rejections))
        print(
            f"skipped {len(rejections)} invalid raw pair(s) before selected valid pair; "
            f"showing {min(show, len(rejections))}:"
        )
        for rejected in rejections[:show]:
            print(
                f"  raw[{rejected.raw_index}] {rejected.group.instance_ids}: "
                f"{rejected.reason}"
            )
        if len(rejections) > show:
            print(f"  ... {len(rejections) - show} more skipped")

    print(
        f"dataset={volume.dataset_name} sample={volume.sample_id} "
        f"valid_pair_index={valid_index_text} raw_pair_index={raw_index} "
        f"group={group.instance_ids}"
    )
    print(f"raw pair candidates={len(groups)}")
    print(f"inputs={sample.inputs.shape} dtype={sample.inputs.dtype}")
    print(f"markers={sample.marker_positions_zyx}")
    print(f"GT centers={sample.targets.centers_zyx}")
    print(f"GT count={int(sample.targets.instance_labels.max())}")
    print(f"input component voxels={int(sample.input_component_mask.sum())}")
    print(f"GT foreground voxels={int(sample.targets.foreground.sum())}")
    print(f"valid crop voxels={int(sample.valid_mask.sum())}/{sample.valid_mask.size}")

    if args.napari:
        try:
            import napari
        except ImportError as error:
            raise SystemExit("Napari is not installed in this environment.") from error
        spacing = config.target_spacing_zyx_um
        viewer = napari.Viewer()
        viewer.add_image(sample.inputs[0], name="normalized fluorescence", scale=spacing)
        viewer.add_labels(
            sample.input_component_mask.astype(np.uint8),
            name="Stage2-like input mask",
            scale=spacing,
            opacity=0.35,
        )
        viewer.add_labels(
            sample.targets.instance_labels,
            name="GT selected instances",
            scale=spacing,
            opacity=0.55,
        )
        viewer.add_image(sample.edt_normalized, name="physical EDT", scale=spacing, visible=False)
        viewer.add_image(
            sample.marker_heatmap,
            name="effective marker heatmap",
            scale=spacing,
            visible=False,
        )
        viewer.add_image(
            sample.targets.center[0],
            name="GT center heatmap",
            scale=spacing,
            visible=False,
        )
        viewer.add_image(
            sample.targets.boundary[0],
            name="GT internal boundary",
            scale=spacing,
            visible=False,
        )
        if not np.all(sample.valid_mask > 0.5):
            viewer.add_labels(
                (sample.valid_mask[0] <= 0.5).astype(np.uint8),
                name="invalid/undefined target voxels",
                scale=spacing,
                visible=False,
                opacity=0.35,
            )
        marker_points = np.asarray(sample.marker_positions_zyx, dtype=np.float32)
        if marker_points.size:
            viewer.add_points(marker_points, name="effective markers", scale=spacing, size=2.0)
        center_points = np.asarray(sample.targets.centers_zyx, dtype=np.float32)
        if center_points.size:
            viewer.add_points(center_points, name="GT centers", scale=spacing, size=2.5)
        vector_data = _napari_vectors(sample, spacing, max(1, args.vector_step))
        if vector_data.size:
            viewer.add_vectors(
                vector_data,
                name="GT center vectors",
                scale=spacing,
                edge_width=0.4,
            )
        napari.run()


if __name__ == "__main__":
    main()
