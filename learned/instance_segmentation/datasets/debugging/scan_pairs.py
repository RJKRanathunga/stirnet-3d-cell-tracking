"""Scan pair candidates and summarize object-centric build rejection reasons."""

from __future__ import annotations

import argparse
from collections import Counter

from ..adapters import dataset_choices, make_adapter
from ..config import DEFAULT_SAMPLE_BUILD_CONFIG
from ..core.adjacency import build_instance_adjacency
from ..core.sample_builder import SampleBuildError, SampleBuilder
from ..core.sampling import pair_groups


def _classify(reason: str) -> str:
    lower = reason.lower()
    if "aspect ratio" in lower:
        return "pathological_aspect"
    if "only" in lower and "voxels" in lower:
        return "too_small_after_normalization"
    if "below minimum" in lower:
        return "too_thin_after_normalization"
    if "border" in lower:
        return "canonical_border"
    if "component" in lower and "found" in lower:
        return "merge_connectivity"
    if "marker" in lower:
        return "marker_generation"
    return "other"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=dataset_choices(), required=True)
    p.add_argument("--root")
    p.add_argument("--split")
    p.add_argument("--volume-index", type=int, default=0)
    p.add_argument("--max-pairs", type=int, default=100)
    p.add_argument("--source-axis-order", default="zyx")
    p.add_argument("--spacing-zyx-um", type=float, nargs=3)
    args = p.parse_args()
    adapter = make_adapter(
        args.dataset,
        root=args.root,
        source_axis_order=args.source_axis_order,
        spacing_override_zyx_um=tuple(args.spacing_zyx_um) if args.spacing_zyx_um else None,
    )
    records = adapter.records(split=args.split) if hasattr(adapter, "records") else adapter.discover_records()
    volume = adapter.load(records[args.volume_index])
    edges = build_instance_adjacency(
        volume.instance_labels,
        volume.spacing_zyx_um,
        max_distance_um=DEFAULT_SAMPLE_BUILD_CONFIG.adjacency_max_distance_um,
    )
    groups = pair_groups(edges)
    builder = SampleBuilder()
    reasons = Counter()
    scales: list[float] = []
    checked = min(args.max_pairs, len(groups))
    for group in groups[:checked]:
        try:
            sample = builder.build(volume, group)
        except SampleBuildError as error:
            reasons[_classify(str(error))] += 1
        else:
            reasons["valid"] += 1
            scales.append(float(sample.transform.normalization_scale))
    print("checked:", checked)
    for key, count in reasons.most_common():
        print(f"{key:34s} {count:6d} ({100*count/max(checked,1):6.2f}%)")
    if scales:
        scales.sort()
        print("normalization scale min/median/max:", scales[0], scales[len(scales)//2], scales[-1])


if __name__ == "__main__":
    main()
