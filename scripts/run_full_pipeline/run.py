"""Run the current cell-tracking pipeline through Stage 11 for full samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import FullPipelineConfig, run_full_pipeline  # noqa: E402


# Edit these two defaults if the local full-dataset/output locations change.
# Both can also be overridden from the command line.
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "full"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "full_processed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover every sample under FULL_DATASET_ROOT/train and run the "
            "current processing pipeline through Stage 11. Stage 9 and Stage 12 "
            "are visualization stages and are not executed."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Full Biohub dataset root (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Batch output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help=(
            "Process only this sample ID. Repeat --sample to select multiple "
            "samples. If omitted, every sample under train/ is processed."
        ),
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=6,
        help=(
            "First stage boundary to run. Values 1-6 all begin with the existing "
            "Stage 6 orchestration of Stages 1-6. Default: 6."
        ),
    )
    parser.add_argument(
        "--end-stage",
        type=int,
        default=11,
        help="Last stage boundary to run. Default: 11.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip stage outputs that already have a valid _SUCCESS.json marker. "
            "A failed/partial stage is deleted and rerun."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first sample failure instead of continuing to later samples.",
    )

    # Current Stage 7 notebook experiment defaults.
    parser.add_argument(
        "--graph-mode",
        choices=("disabled", "shadow", "apply"),
        default="apply",
    )
    parser.add_argument(
        "--graph-algorithm",
        choices=("pairwise", "windowed_4d"),
        default="windowed_4d",
    )
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--maximum-gap-frames", type=int, default=2)
    parser.add_argument("--solver-time-limit-seconds", type=float, default=30.0)
    parser.add_argument("--iterative-fallback-iterations", type=int, default=5)
    parser.add_argument(
        "--save-graph-debug-npz",
        action="store_true",
        help="Save detailed Stage 7 4D graph debug NPZ artifacts.",
    )
    parser.add_argument(
        "--reconciliation-policy",
        default="submission",
        help="Stage 11 reconciliation policy (default: submission).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = FullPipelineConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        sample_ids=tuple(args.sample),
        start_stage=args.start_stage,
        end_stage=args.end_stage,
        resume=args.resume,
        continue_on_error=not args.stop_on_error,
        graph_mode=args.graph_mode,
        graph_algorithm=args.graph_algorithm,
        graph_window_size=args.window_size,
        graph_maximum_gap_frames=args.maximum_gap_frames,
        graph_solver_time_limit_seconds=args.solver_time_limit_seconds,
        graph_iterative_fallback_iterations=args.iterative_fallback_iterations,
        graph_save_debug_npz=args.save_graph_debug_npz,
        reconciliation_policy=args.reconciliation_policy,
    )
    summary = run_full_pipeline(config)
    return 0 if summary.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
