"""Launch pass-2 Napari 3D center annotation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from learned.instance_segmentation.datasets.merge_real.config import MergeRealConfig, MergeRealPaths
    from learned.instance_segmentation.datasets.merge_real.review import launch_center_annotation
else:
    from .config import MergeRealConfig, MergeRealPaths
    from .review import launch_center_annotation


def main() -> int:
    parser = argparse.ArgumentParser(description="Pass 2: place one approximate 3D center per cell in confirmed merge cases.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--full-processed-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    paths = MergeRealPaths.discover(args.project_root, full_processed_root=args.full_processed_root, output_root=args.output_root)
    launch_center_annotation(paths, MergeRealConfig())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
