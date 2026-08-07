"""Summarize mining and three-pass annotation progress."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from learned.instance_segmentation.datasets.merge_real.config import MergeRealPaths
    from learned.instance_segmentation.datasets.merge_real.review.common import read_csv_or_empty
else:
    from .config import MergeRealPaths
    from .review.common import read_csv_or_empty


def summarize(paths: MergeRealPaths) -> dict[str, object]:
    candidates_path = paths.mining_dir / "candidates.csv"
    candidates = pd.read_csv(candidates_path) if candidates_path.exists() else pd.DataFrame()
    filters = read_csv_or_empty(paths.filter_reviews_csv)
    partitions = read_csv_or_empty(paths.reviews_dir / "partition_reviews.csv")
    center_files = tuple(paths.centers_dir.glob("*.csv")) if paths.centers_dir.exists() else ()
    cases = tuple(p for p in paths.cases_dir.iterdir() if p.is_dir()) if paths.cases_dir.exists() else ()
    return {
        "mined_candidates": len(candidates),
        "tiers": candidates["tier"].value_counts().to_dict() if not candidates.empty else {},
        "pass1_reviewed": len(filters),
        "pass1_labels": filters["classification"].value_counts().to_dict() if not filters.empty else {},
        "pass2_center_annotations": len(center_files),
        "pass3_reviewed": len(partitions),
        "pass3_status": partitions["status"].value_counts().to_dict() if not partitions.empty else {},
        "materialized_cases": len(cases),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--full-processed-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    paths = MergeRealPaths.discover(args.project_root, full_processed_root=args.full_processed_root, output_root=args.output_root)
    for key, value in summarize(paths).items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
