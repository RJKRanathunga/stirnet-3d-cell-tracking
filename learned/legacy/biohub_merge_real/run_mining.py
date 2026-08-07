"""Mine real merge candidates from data/full_processed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from learned.instance_segmentation.datasets.merge_real.config import MergeRealConfig, MergeRealPaths
    from learned.instance_segmentation.datasets.merge_real.mining import mine_all_samples
else:
    from .config import MergeRealConfig, MergeRealPaths
    from .mining import mine_all_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine real merged-cell candidates from full_processed Stage 6 + Stage 11 outputs.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--full-processed-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sample", action="append", default=[], help="Restrict to one sample; repeat for several samples.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = MergeRealPaths.discover(
        args.project_root,
        full_processed_root=args.full_processed_root,
        output_root=args.output_root,
    )
    candidates = mine_all_samples(paths, MergeRealConfig(), sample_ids=tuple(args.sample))
    print(f"Saved {len(candidates)} fused candidates to {paths.mining_dir / 'candidates.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
