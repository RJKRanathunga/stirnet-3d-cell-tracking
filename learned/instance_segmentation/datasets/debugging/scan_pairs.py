"""Scan neighboring pair candidates and summarize TrainingSample rejection reasons."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ..adapters.registry import dataset_choices, make_adapter
from ..config import DEFAULT_SAMPLE_BUILD_CONFIG
from ..core.adjacency import build_instance_adjacency
from ..core.sample_builder import SampleBuildError, SampleBuilder
from ..core.sampling import pair_groups


def _category(reason: str) -> str:
    lower = reason.lower()
    if "voxels after resampling" in lower or "vanished during resampling" in lower:
        return "too_small_after_resampling"
    if "canonical crop border" in lower:
        return "crop_border"
    if "stage-2-like component" in lower:
        return "merge_connectivity"
    if "marker" in lower:
        return "marker_generation"
    if "target" in lower or "center" in lower or "boundary" in lower:
        return "target_generation"
    return "other"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=dataset_choices(), default="c_elegans")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--volume-index", type=int, default=0)
    parser.add_argument("--source-axis-order", default="zyx")
    parser.add_argument("--spacing-zyx-um", type=float, nargs=3, metavar=("Z", "Y", "X"))
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--show-failures", type=int, default=10)
    return parser


def main() -> None:
    args = _parser().parse_args()
    spacing_override = tuple(args.spacing_zyx_um) if args.spacing_zyx_um else None
    adapter = make_adapter(
        args.dataset,
        args.root,
        source_axis_order=args.source_axis_order,
        spacing_override_zyx_um=spacing_override,
    )
    volume = adapter.load_index(args.volume_index, split=args.split)
    config = DEFAULT_SAMPLE_BUILD_CONFIG
    groups = pair_groups(
        build_instance_adjacency(
            volume.instance_labels,
            volume.spacing_zyx_um,
            max_distance_um=config.adjacency_max_distance_um,
        )
    )
    builder = SampleBuilder(config)
    limit = min(len(groups), max(0, args.max_pairs))
    if limit == 0:
        raise SystemExit("No pair candidates to scan.")

    counts: Counter[str] = Counter()
    failures: list[tuple[int, tuple[int, ...], str]] = []
    valid = 0
    for index, group in enumerate(groups[:limit]):
        try:
            builder.build(volume, group)
        except SampleBuildError as error:
            reason = str(error)
            counts[_category(reason)] += 1
            failures.append((index, group.instance_ids, reason))
        else:
            valid += 1

    print(f"dataset={volume.dataset_name} sample={volume.sample_id}")
    print(f"raw pair candidates total={len(groups)} scanned={limit}")
    print(f"valid={valid} rejected={limit-valid} valid_fraction={valid/limit:.3f}")
    for category, count in counts.most_common():
        print(f"rejected[{category}]={count}")
    for index, ids, reason in failures[: max(0, args.show_failures)]:
        print(f"failure raw[{index}] {ids}: {reason}")


if __name__ == "__main__":
    main()
