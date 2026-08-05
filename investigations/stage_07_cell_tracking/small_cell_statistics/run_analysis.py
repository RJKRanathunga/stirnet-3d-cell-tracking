"""CLI for the small-cell size and temporal-variation investigation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import SmallCellStatisticsConfig
from .pipeline import run_investigation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare manually curated small-cell scenes with all other same-frame cells "
            "and stable successfully tracked control cells."
        )
    )
    parser.add_argument("--scenes-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--control-min-observations", type=int, default=5)
    parser.add_argument("--matched-controls-per-target", type=int, default=10)
    parser.add_argument("--target-link-max-distance-um", type=float, default=15.0)
    parser.add_argument("--no-mask-volume-validation", action="store_true")
    parser.add_argument("--allow-boundary-controls", action="store_true")
    parser.add_argument("--allow-event-controls", action="store_true")
    parser.add_argument("--allow-gapped-controls", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SmallCellStatisticsConfig(
        scenes_root=args.scenes_root,
        target_link_max_distance_um=args.target_link_max_distance_um,
        control_minimum_observations=args.control_min_observations,
        control_require_contiguous=not args.allow_gapped_controls,
        control_exclude_boundary=not args.allow_boundary_controls,
        control_exclude_event_tracks=not args.allow_event_controls,
        matched_controls_per_target=args.matched_controls_per_target,
        validate_mask_volumes=not args.no_mask_volume_validation,
        create_plots=not args.no_plots,
        strict=args.strict,
    )
    result = run_investigation(config, output_directory=args.output_dir, progress=print)
    print(f"Output: {result.output_directory}")
    print(f"Valid scenes: {result.metadata['valid_scene_count']}")
    print(f"Selected observations: {result.metadata['selected_observation_count']}")
    print(f"Manual small-cell tracks: {result.metadata['manual_small_track_count']}")
    print(f"Stable control tracks: {result.metadata['stable_control_track_count']}")
    print("\nSize distribution summary:")
    columns = [name for name in ("cohort", "count", "q05", "q10", "q25", "q50", "q75", "q90", "q95") if name in result.size_distribution_summary]
    print(result.size_distribution_summary[columns].to_string(index=False))
    print("\nVolume variation comparison:")
    show = result.volume_variation_comparison.loc[
        result.volume_variation_comparison["metric"].isin(
            ["volume_robust_cv", "adjacent_relative_change_median", "adjacent_relative_change_p90"]
        )
    ]
    print(show[[
        "metric", "control_cohort", "target_median", "control_median",
        "median_ratio", "target_median_control_percentile", "mann_whitney_p_value",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
