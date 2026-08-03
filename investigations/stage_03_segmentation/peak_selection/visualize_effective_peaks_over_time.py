"""Visualize canonical Stage 3 effective peaks across all processed frames.

Run from the repository environment:

    python visualize_effective_peaks_over_time.py --sample-id 44b6_0113de3b

The script:
- reads the complete saved Stage 2 binary mask for each frame;
- runs the canonical full-frame Stage 3 entry point with debug artifacts;
- extracts raw peaks, effective peaks, and final selected markers from that
  exact production execution;
- displays all effective peaks in one TZYX Napari points layer;
- optionally overlays raw data, saved masks, saved labels, and production
  centroids;
- caches the computed peak table so later launches are fast.

It does not modify production artifacts or defaults. H1/H2/H3 is executed
only because it is part of the canonical Stage 3 call; its selected markers
are shown in a separate optional layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Iterable

import dask.array as da
import napari
import numpy as np
import pandas as pd
from scipy import ndimage

from src.io import PipelinePaths, load_npy, load_npy_time_series


peaks_module = import_module("src.03_segmentation.peaks")
pipeline_module = import_module("src.03_segmentation.pipeline")
config_module = import_module("src.03_segmentation.config")

DEFAULT_SEGMENTATION_CONFIG = config_module.DEFAULT_SEGMENTATION_CONFIG
CONNECTIVITY_3D_6 = ndimage.generate_binary_structure(3, 1)
CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class FrameCounts:
    frame: int
    component_count: int
    raw_peak_count: int
    effective_peak_count: int
    failed_component_count: int


def _config_fingerprint(config) -> str:
    """Create a stable-enough fingerprint for diagnostic cache invalidation."""

    return hashlib.sha256(repr(config).encode("utf-8")).hexdigest()[:16]


def _frame_files(directory: Path, frame_count: int) -> list[Path]:
    files = sorted(directory.glob("t*.npy"))
    if not files:
        raise FileNotFoundError(f"No t*.npy files found in {directory}")
    if frame_count > len(files):
        raise ValueError(
            f"Requested {frame_count} frames, but only {len(files)} exist in {directory}"
        )
    return files[:frame_count]


def compute_effective_peaks_for_frame(
    binary_mask: np.ndarray,
    *,
    frame: int,
    config=DEFAULT_SEGMENTATION_CONFIG,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    FrameCounts,
]:
    """Run canonical Stage 3 and extract its raw/effective/selected peak records.

    This calls ``segment_instances_detailed`` on the complete saved Stage 2
    frame. Effective peaks are read from the exact component debug artifacts
    produced by the production implementation.
    """

    mask = np.asarray(binary_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"Expected one ZYX mask, received shape {mask.shape}")

    result = pipeline_module.segment_instances_detailed(
        mask,
        config,
        include_hypothesis_diagnostics=False,
        retain_debug_artifacts=True,
    )

    raw_rows: list[dict[str, object]] = []
    effective_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    padding = int(config.component_padding_voxels)

    for diagnostic in result.component_diagnostics:
        if diagnostic.error:
            failure_rows.append(
                {
                    "frame": int(frame),
                    "component_id": int(diagnostic.component_id),
                    "component_voxels": int(diagnostic.source_voxels),
                    "error_type": "Stage3Fallback",
                    "error": str(diagnostic.error),
                }
            )

    diagnostics_by_component = {
        int(item.component_id): item
        for item in result.component_diagnostics
    }

    for artifact in result.component_debug_artifacts:
        component_id = int(artifact.component_id)
        diagnostic = diagnostics_by_component[component_id]
        starts = np.asarray(
            [int(bounds[0]) for bounds in artifact.bbox_zyx],
            dtype=int,
        )
        effective_ids = {
            int(peak.peak_id) for peak in artifact.effective_peaks
        }
        selected_ids = {
            int(peak.peak_id)
            for peak in artifact.decision.chosen.selected_peaks
        }

        def row_for_peak(peak, *, category: str) -> dict[str, object]:
            local_position = (
                np.asarray(peak.position_zyx, dtype=int) - padding
            )
            global_position = starts + local_position
            return {
                "frame": int(frame),
                "z": int(global_position[0]),
                "y": int(global_position[1]),
                "x": int(global_position[2]),
                "component_id": component_id,
                "peak_id": int(peak.peak_id),
                "category": category,
                "component_voxels": int(diagnostic.source_voxels),
                "raw_peak_count": int(diagnostic.raw_peak_count),
                "effective_peak_count": int(
                    diagnostic.effective_lobe_count
                ),
                "selected_cell_count": int(
                    diagnostic.selected_cell_count
                ),
                "decision_status": str(diagnostic.decision_status),
                "raw_depth_um": float(peak.raw_depth_um),
                "smoothed_depth_um": float(peak.smoothed_depth_um),
                "scale_support": float(peak.scale_support),
                "h_support": float(peak.h_support),
                "setting_support": float(peak.setting_support),
                "detection_count": int(peak.detection_count),
                "persistence_score": float(peak.persistence_score),
                "is_effective": int(peak.peak_id) in effective_ids,
                "is_selected": int(peak.peak_id) in selected_ids,
            }

        raw_rows.extend(
            row_for_peak(peak, category="raw")
            for peak in artifact.peaks
        )
        effective_rows.extend(
            row_for_peak(peak, category="effective")
            for peak in artifact.effective_peaks
        )
        selected_rows.extend(
            row_for_peak(peak, category="selected")
            for peak in artifact.decision.chosen.selected_peaks
        )

    counts = FrameCounts(
        frame=int(frame),
        component_count=len(result.component_diagnostics),
        raw_peak_count=sum(
            int(item.raw_peak_count) for item in result.component_diagnostics
        ),
        effective_peak_count=sum(
            int(item.effective_lobe_count)
            for item in result.component_diagnostics
        ),
        failed_component_count=len(failure_rows),
    )
    return raw_rows, effective_rows, selected_rows, failure_rows, counts


def compute_or_load_peak_cache(
    *,
    paths: PipelinePaths,
    sample_id: str,
    frame_count: int,
    config=DEFAULT_SEGMENTATION_CONFIG,
    recompute: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a valid cache or compute canonical peak tables frame by frame."""

    masking_dir = paths.processed_series(sample_id, "masking")
    mask_files = _frame_files(masking_dir, frame_count)

    cache_dir = (
        paths.project_root
        / "data"
        / "diagnostics"
        / "effective_peaks"
        / sample_id
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = _config_fingerprint(config)
    raw_path = cache_dir / f"raw_peaks_{frame_count}f_{fingerprint}.csv"
    peak_path = cache_dir / f"effective_peaks_{frame_count}f_{fingerprint}.csv"
    selected_path = cache_dir / f"selected_peaks_{frame_count}f_{fingerprint}.csv"
    failure_path = cache_dir / f"failures_{frame_count}f_{fingerprint}.csv"
    count_path = cache_dir / f"frame_counts_{frame_count}f_{fingerprint}.csv"
    metadata_path = cache_dir / f"metadata_{frame_count}f_{fingerprint}.json"

    expected_metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "frame_count": int(frame_count),
        "config_fingerprint": fingerprint,
        "config_repr": repr(config),
        "mask_files": [path.name for path in mask_files],
    }

    if (
        not recompute
        and raw_path.is_file()
        and peak_path.is_file()
        and selected_path.is_file()
        and failure_path.is_file()
        and count_path.is_file()
        and metadata_path.is_file()
    ):
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata == expected_metadata:
            return (
                pd.read_csv(raw_path),
                pd.read_csv(peak_path),
                pd.read_csv(selected_path),
                pd.read_csv(failure_path),
                pd.read_csv(count_path),
            )

    all_raw: list[dict[str, object]] = []
    all_peaks: list[dict[str, object]] = []
    all_selected: list[dict[str, object]] = []
    all_failures: list[dict[str, object]] = []
    all_counts: list[dict[str, object]] = []

    for frame, mask_path in enumerate(mask_files):
        print(f"[effective peaks] frame {frame + 1}/{frame_count}: {mask_path.name}")
        binary_mask = load_npy(mask_path, expected_ndim=3)
        raw, peaks, selected, failures, counts = compute_effective_peaks_for_frame(
            binary_mask,
            frame=frame,
            config=config,
        )
        all_raw.extend(raw)
        all_peaks.extend(peaks)
        all_selected.extend(selected)
        all_failures.extend(failures)
        all_counts.append(counts.__dict__)

    raw_table = pd.DataFrame(all_raw)
    peak_table = pd.DataFrame(all_peaks)
    selected_table = pd.DataFrame(all_selected)
    failure_table = pd.DataFrame(
        all_failures,
        columns=(
            "frame",
            "component_id",
            "component_voxels",
            "error_type",
            "error",
        ),
    )
    count_table = pd.DataFrame(all_counts)

    raw_table.to_csv(raw_path, index=False)
    peak_table.to_csv(peak_path, index=False)
    selected_table.to_csv(selected_path, index=False)
    failure_table.to_csv(failure_path, index=False)
    count_table.to_csv(count_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(expected_metadata, file, indent=2)
        file.write("\n")

    print(f"[effective peaks] cache written to {cache_dir}")
    return raw_table, peak_table, selected_table, failure_table, count_table


def load_production_centroids(
    *,
    paths: PipelinePaths,
    sample_id: str,
    frame_count: int,
) -> pd.DataFrame:
    """Load saved Stage 4 centroids for comparison."""

    cell_dir = paths.processed_series(sample_id, "cells")
    files = sorted(cell_dir.glob("t*.csv"))[:frame_count]
    rows: list[pd.DataFrame] = []

    for frame, path in enumerate(files):
        table = pd.read_csv(path)
        required = {"cell_id", "centroid_z", "centroid_y", "centroid_x"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        selected = table[
            ["cell_id", "centroid_z", "centroid_y", "centroid_x"]
        ].copy()
        selected.insert(0, "frame", frame)
        rows.append(selected)

    if not rows:
        return pd.DataFrame(
            columns=(
                "frame",
                "cell_id",
                "centroid_z",
                "centroid_y",
                "centroid_x",
            )
        )
    return pd.concat(rows, ignore_index=True)


def _properties(table: pd.DataFrame, excluded: Iterable[str]) -> dict[str, np.ndarray]:
    excluded_set = set(excluded)
    return {
        column: table[column].to_numpy()
        for column in table.columns
        if column not in excluded_set
    }


def launch_viewer(
    *,
    paths: PipelinePaths,
    sample_id: str,
    frame_count: int,
    raw_peak_table: pd.DataFrame,
    peak_table: pd.DataFrame,
    selected_peak_table: pd.DataFrame,
    failure_table: pd.DataFrame,
    count_table: pd.DataFrame,
    show_binary: bool,
    show_labels: bool,
    show_centroids: bool,
    point_size: float,
) -> None:
    """Launch the TZYX Napari visualization."""

    voxel_size = tuple(
        float(value)
        for value in DEFAULT_SEGMENTATION_CONFIG.voxel_size_zyx_um
    )
    scale_tzyx = (1.0, *voxel_size)

    raw = da.from_zarr(str(paths.sample_zarr_array(sample_id)))[:frame_count]
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(
        raw,
        name="Raw",
        scale=scale_tzyx,
        rendering="mip",
    )

    if show_binary:
        binary, _ = load_npy_time_series(
            paths.processed_series(sample_id, "masking")
        )
        viewer.add_labels(
            binary[:frame_count].astype(np.uint8),
            name="Binary mask",
            scale=scale_tzyx,
            opacity=0.20,
            rendering="translucent",
            visible=False,
        )

    if show_labels:
        labels, _ = load_npy_time_series(
            paths.processed_series(sample_id, "segmentation")
        )
        viewer.add_labels(
            labels[:frame_count],
            name="Production labels",
            scale=scale_tzyx,
            opacity=0.45,
            rendering="translucent",
            visible=False,
        )

    def table_points(table: pd.DataFrame) -> np.ndarray:
        if table.empty:
            return np.empty((0, 4), dtype=float)
        return table[["frame", "z", "y", "x"]].to_numpy(dtype=float)

    # Keep out_of_slice_display=False. In a 4-D TZYX points layer, time is
    # the sliced dimension; enabling out-of-slice display causes neighboring
    # frames to appear on the current frame.
    viewer.add_points(
        table_points(raw_peak_table),
        name="Raw peaks",
        scale=scale_tzyx,
        size=max(1.0, float(point_size) * 0.50),
        face_color="orange",
        border_color="black",
        border_width=0.10,
        out_of_slice_display=False,
        properties=_properties(
            raw_peak_table,
            excluded=("frame", "z", "y", "x"),
        ),
        visible=False,
    )

    viewer.add_points(
        table_points(peak_table),
        name="Effective peaks",
        scale=scale_tzyx,
        size=float(point_size),
        face_color="lime",
        border_color="black",
        border_width=0.12,
        out_of_slice_display=False,
        properties=_properties(
            peak_table,
            excluded=("frame", "z", "y", "x"),
        ),
    )

    viewer.add_points(
        table_points(selected_peak_table),
        name="Current Stage 3 markers",
        scale=scale_tzyx,
        size=max(1.0, float(point_size) * 1.35),
        symbol="ring",
        face_color="transparent",
        border_color="magenta",
        border_width=0.18,
        out_of_slice_display=False,
        properties=_properties(
            selected_peak_table,
            excluded=("frame", "z", "y", "x"),
        ),
        visible=False,
    )

    production_centroids = pd.DataFrame()
    if show_centroids:
        production_centroids = load_production_centroids(
            paths=paths,
            sample_id=sample_id,
            frame_count=frame_count,
        )
        centroid_points = production_centroids[
            ["frame", "centroid_z", "centroid_y", "centroid_x"]
        ].to_numpy(dtype=float)
        viewer.add_points(
            centroid_points,
            name="Saved Stage 4 centroids",
            scale=scale_tzyx,
            size=max(1.0, float(point_size) * 0.65),
            symbol="ring",
            face_color="transparent",
            border_color="cyan",
            border_width=0.20,
            out_of_slice_display=False,
            properties={
                "cell_id": production_centroids["cell_id"].to_numpy(),
            },
            visible=False,
        )

    try:
        from qtpy.QtWidgets import QLabel

        status = QLabel()
        status.setWordWrap(True)

        def update_status(_event=None) -> None:
            time_index = int(viewer.dims.current_step[0])
            row = count_table.loc[count_table["frame"] == time_index]
            if row.empty:
                status.setText(f"Frame {time_index}: no cached summary")
                return

            values = row.iloc[0]
            production_count = (
                int(
                    np.count_nonzero(
                        production_centroids["frame"].to_numpy() == time_index
                    )
                )
                if not production_centroids.empty
                else 0
            )
            status.setText(
                f"Frame {time_index} | "
                f"components={int(values['component_count'])} | "
                f"raw peaks={int(values['raw_peak_count'])} | "
                f"effective peaks={int(values['effective_peak_count'])} | "
                f"production cells={production_count} | "
                f"failed components={int(values['failed_component_count'])}"
            )

        viewer.dims.events.current_step.connect(update_status)
        viewer.window.add_dock_widget(status, name="Peak summary", area="right")
        update_status()
    except ImportError:
        pass

    viewer.dims.ndisplay = 3
    try:
        viewer.camera.angles = (45, 30, 135)
    except (AttributeError, TypeError, ValueError):
        pass
    viewer.reset_view()

    if not failure_table.empty:
        print("\nComponents whose effective-peak analysis failed:")
        print(failure_table.to_string(index=False))

    napari.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize canonical effective peaks over all processed frames."
    )
    parser.add_argument(
        "--sample-id",
        default="44b6_0113de3b",
        help="Processed sample ID.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=20,
        help="Number of frames to compute and display.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore the diagnostic CSV cache and recompute all peaks.",
    )
    parser.add_argument(
        "--no-binary",
        action="store_true",
        help="Do not add the saved binary-mask layer.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not add the saved production-label layer.",
    )
    parser.add_argument(
        "--no-centroids",
        action="store_true",
        help="Do not add saved production centroids for comparison.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Napari effective-peak point size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    paths = PipelinePaths.discover()
    raw_peak_table, peak_table, selected_peak_table, failure_table, count_table = compute_or_load_peak_cache(
        paths=paths,
        sample_id=args.sample_id,
        frame_count=args.frames,
        config=DEFAULT_SEGMENTATION_CONFIG,
        recompute=args.recompute,
    )

    print(
        f"[effective peaks] {len(peak_table)} peaks across "
        f"{args.frames} frame(s); {len(failure_table)} component failure(s)"
    )

    launch_viewer(
        paths=paths,
        sample_id=args.sample_id,
        frame_count=args.frames,
        raw_peak_table=raw_peak_table,
        peak_table=peak_table,
        selected_peak_table=selected_peak_table,
        failure_table=failure_table,
        count_table=count_table,
        show_binary=not args.no_binary,
        show_labels=not args.no_labels,
        show_centroids=not args.no_centroids,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
