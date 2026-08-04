"""Command-line entry point for division-characteristics extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import InvestigationConfig
from .pipeline import run_investigation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract raw, preprocessed, morphology, local-background, and temporal "
            "features from manually selected division scenes."
        )
    )
    parser.add_argument(
        "--scenes-root",
        type=Path,
        default=None,
        help=(
            "Division scene directory. Default: data/tracking_scenes/divisions."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: timestamped data/investigations path.",
    )
    parser.add_argument("--core-erosion-um", type=float, default=0.8)
    parser.add_argument("--fixed-radius-um", type=float, default=2.5)
    parser.add_argument("--background-shell-inner-um", type=float, default=0.8)
    parser.add_argument("--background-shell-outer-um", type=float, default=3.0)
    parser.add_argument(
        "--baseline-exclude-last-parent-frames",
        type=int,
        default=1,
        help=(
            "Prefer a parent baseline that excludes this many final pre-division frames. "
            "The analysis automatically falls back when too few frames remain."
        ),
    )
    parser.add_argument("--minimum-baseline-frames", type=int, default=2)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately when any scene is invalid or cannot be processed.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write tables only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = InvestigationConfig(
        scenes_root=args.scenes_root,
        core_erosion_um=args.core_erosion_um,
        fixed_radius_um=args.fixed_radius_um,
        background_shell_inner_um=args.background_shell_inner_um,
        background_shell_outer_um=args.background_shell_outer_um,
        baseline_exclude_last_parent_frames=(
            args.baseline_exclude_last_parent_frames
        ),
        minimum_baseline_frames=args.minimum_baseline_frames,
        create_plots=not args.no_plots,
        strict=args.strict,
    )
    result = run_investigation(config, output_directory=args.output_dir)
    print(f"Division investigation complete: {result.output_directory}")
    print(
        f"Processed cases: {result.metadata['processed_case_count']} | "
        f"Observations: {len(result.observations):,} | "
        f"Event summaries: {len(result.event_summaries):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
