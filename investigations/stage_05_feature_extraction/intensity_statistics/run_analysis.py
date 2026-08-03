"""Command-line entry point for the Stage 05 intensity investigation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    DEFAULT_SAMPLE_ID,
    InvestigationConfig,
    parse_float_list,
    parse_frames,
    parse_int_list,
)
from .pipeline import run_investigation
from .repository_io import discover_available_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare intensity statistics from raw, weakly denoised, and "
            "current preprocessing images using fixed production masks."
        )
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument(
        "--frames",
        default=None,
        help="Frame expression such as 0:20, 0:20:2, or 0,1,5. Default: all aligned frames.",
    )
    parser.add_argument(
        "--weak-sigmas",
        default="0.2,0.4",
        help="Comma-separated physical Gaussian sigmas in micrometers.",
    )
    parser.add_argument(
        "--tracks-csv",
        default=None,
        help="Optional explicit Stage 7 tracks.csv path.",
    )
    parser.add_argument(
        "--track-ids",
        default=None,
        help=(
            "Optional comma-separated manually validated track IDs. "
            "When omitted, structurally complete boundary-free tracks are used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults below data/investigations/.",
    )
    parser.add_argument("--association-radius-um", type=float, default=12.0)
    parser.add_argument("--negatives-per-positive", type=int, default=5)
    parser.add_argument(
        "--allow-preprocessing-mismatch",
        action="store_true",
        help="Continue if canonical preprocessing replay differs from saved Stage 6 data.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write tables only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from src.io import PipelinePaths

    args = build_parser().parse_args(argv)
    paths = PipelinePaths.discover()
    available = discover_available_frames(paths, args.sample_id)
    frames = parse_frames(args.frames, available)

    config = InvestigationConfig(
        sample_id=args.sample_id,
        frame_ids=frames,
        weak_sigmas_um=parse_float_list(args.weak_sigmas),
        association_radius_um=args.association_radius_um,
        negatives_per_positive=args.negatives_per_positive,
        allow_preprocessing_mismatch=args.allow_preprocessing_mismatch,
        create_plots=not args.no_plots,
        manual_track_ids=parse_int_list(args.track_ids),
    )
    result = run_investigation(
        config,
        tracks_csv=args.tracks_csv,
        output_directory=args.output_dir,
    )
    print(f"Investigation complete: {result.output_directory}")
    print(
        f"Cells/method rows: {len(result.cell_statistics):,} | "
        f"Selected tracks: {result.metadata['selected_track_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
