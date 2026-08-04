"""Benchmark the production peak-based Stage 3 candidate gate on NumPy masks."""

from __future__ import annotations

import argparse
from collections import Counter
from importlib import import_module
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd


pipeline_module = import_module("src.03_segmentation.pipeline")


class _TimedCall:
    def __init__(self, function):
        self.function = function
        self.elapsed = 0.0

    def __call__(self, *args, **kwargs):
        started = perf_counter()
        try:
            return self.function(*args, **kwargs)
        finally:
            self.elapsed += perf_counter() - started


def _selected_frames(mask: np.ndarray, frames: tuple[int, ...] | None):
    if mask.ndim == 3:
        if frames not in (None, (), (0,)):
            raise ValueError("a 3-D mask supports only frame 0")
        return ((0, mask),)
    if mask.ndim != 4:
        raise ValueError("the input must be a ZYX or TZYX NumPy mask")
    indices = frames if frames is not None else tuple(range(mask.shape[0]))
    if any(index < 0 or index >= mask.shape[0] for index in indices):
        raise IndexError("a requested frame is outside the TZYX mask")
    return tuple((index, mask[index]) for index in indices)


def benchmark(
    mask_path: Path,
    frames: tuple[int, ...] | None = None,
    *,
    force_geometry: bool = False,
) -> pd.DataFrame:
    """Run canonical production segmentation and return one row per frame."""

    mask = np.asarray(np.load(mask_path), dtype=bool)
    rows = []
    for frame_index, frame_mask in _selected_frames(mask, frames):
        timed_candidate = _TimedCall(pipeline_module.safely_detect_geometric_candidate)
        timed_geometry = _TimedCall(pipeline_module.safely_complete_geometric_markers)
        started = perf_counter()
        with patch.object(
            pipeline_module,
            "safely_detect_geometric_candidate",
            timed_candidate,
        ), patch.object(
            pipeline_module,
            "safely_complete_geometric_markers",
            timed_geometry,
        ):
            result = pipeline_module.segment_instances_detailed(
                frame_mask,
                force_geometric_analysis=force_geometry,
            )
        total_elapsed = perf_counter() - started
        diagnostics = result.component_diagnostics
        route_counts = Counter(
            route for item in diagnostics for route in item.candidate_routes
        )
        candidates = sum(item.merge_candidate for item in diagnostics)
        geometry_components = sum(item.geometry_executed for item in diagnostics)
        rows.append(
            {
                "sample": mask_path.stem,
                "frame": frame_index,
                "connected_components": len(diagnostics),
                "candidate_components": candidates,
                "candidate_percentage": (
                    100.0 * candidates / len(diagnostics) if diagnostics else 0.0
                ),
                "cross_transform_routes": route_counts["cross_transform"],
                "shape_only_routes": route_counts["shape_only"],
                "suppressed_edt_routes": route_counts["suppressed_edt"],
                "shape_peak_count": sum(item.shape_peak_count for item in diagnostics),
                "proposal_count": sum(
                    item.center_proposal_count for item in diagnostics
                ),
                "geometry_components": geometry_components,
                "supplemental_marker_count": sum(
                    item.supplemental_marker_count for item in diagnostics
                ),
                "candidate_detection_seconds": timed_candidate.elapsed,
                "geometric_completion_seconds": timed_geometry.elapsed,
                "stage3_seconds": total_elapsed,
                "force_geometry": force_geometry,
            }
        )
    return pd.DataFrame(rows)


def _parse_frames(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mask", type=Path, help="3-D ZYX or 4-D TZYX .npy mask")
    parser.add_argument(
        "--frames",
        default=None,
        help="comma-separated T indices (default: every frame; 3-D uses frame 0)",
    )
    parser.add_argument("--force-geometric-analysis", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    arguments = parser.parse_args()
    table = benchmark(
        arguments.mask,
        _parse_frames(arguments.frames),
        force_geometry=arguments.force_geometric_analysis,
    )
    print(table.to_string(index=False))
    if arguments.csv is not None:
        arguments.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(arguments.csv, index=False)


if __name__ == "__main__":
    main()
