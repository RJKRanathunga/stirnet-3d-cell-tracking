"""Headless canonical Stage 3 orchestration and diagnostic reports."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd

from .models import CanonicalMismatchError, Stage3ComponentRun


peaks_module = import_module("src.03_segmentation.peaks")
watershed_module = import_module("src.03_segmentation.watershed")
pipeline_module = import_module("src.03_segmentation.pipeline")


def run_stage3_component(resolution, component_id: int, config) -> Stage3ComponentRun:
    """Execute the all-effective Stage 3 steps and verify production parity."""

    component_id = int(component_id)
    component_mask = resolution.complete_component_mask(component_id)
    if not component_mask.any():
        raise ValueError("the selected target component is empty")
    padding = config.component_padding_voxels
    padded_mask = np.pad(
        component_mask, padding, mode="constant", constant_values=False
    )

    peak_detail = peaks_module.detect_persistent_distance_peaks_detailed(
        padded_mask, config
    )
    peak_analysis = peak_detail.analysis
    pair_evidence = peaks_module.build_peak_pair_evidence(
        peak_analysis.peaks,
        padded_mask,
        peak_analysis.merge_tree_distance,
        config,
    )
    collapse_result = peaks_module.collapse_same_lobe_peaks(
        peak_analysis.peaks, pair_evidence, config
    )
    effective_peaks = collapse_result.effective_peaks
    if not effective_peaks:
        raise RuntimeError("peak collapse produced no effective peaks")
    if len(effective_peaks) == 1:
        padded_labels = padded_mask.astype(np.int32)
    else:
        padded_labels = watershed_module.build_marker_watershed(
            padded_mask,
            peak_analysis.watershed_distance,
            effective_peaks,
        )

    inner = tuple(
        slice(padding, -padding) if padding else slice(None) for _ in range(3)
    )
    final_labels = np.asarray(padded_labels[inner], dtype=np.int32)
    marker_positions = tuple(
        tuple(int(value - padding) for value in peak.position_zyx)
        for peak in effective_peaks
    )
    canonical = pipeline_module.analyze_component_crop(
        component_mask, config, retain_debug_artifacts=True
    )

    checks = {
        "final labels": np.array_equal(final_labels, canonical.final_labels),
        "raw peaks": peak_analysis.peaks == canonical.raw_peaks,
        "effective peaks": effective_peaks == canonical.effective_peaks,
        "pair evidence": pair_evidence == canonical.pair_evidence,
        "final marker positions": (
            marker_positions == canonical.marker_positions_zyx
        ),
        "final instance count": (
            len(effective_peaks)
            == len(tuple(value for value in np.unique(final_labels) if value > 0))
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise CanonicalMismatchError(
            "manual Stage 3 orchestration differs from production: "
            + ", ".join(failed)
        )
    return Stage3ComponentRun(
        component_id,
        resolution.component_bboxes[component_id],
        component_mask,
        padded_mask,
        peak_detail,
        pair_evidence,
        collapse_result,
        final_labels,
        marker_positions,
        canonical,
        config,
    )


def raw_peaks_dataframe(run: Stage3ComponentRun) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "peak_id": peak.peak_id,
                "z": peak.position_zyx[0],
                "y": peak.position_zyx[1],
                "x": peak.position_zyx[2],
                "raw_depth_um": peak.raw_depth_um,
                "smoothed_depth_um": peak.smoothed_depth_um,
                "scale_support": peak.scale_support,
                "h_support": peak.h_support,
                "setting_support": peak.setting_support,
                "detection_count": peak.detection_count,
                "persistence_score": peak.persistence_score,
            }
            for peak in run.peak_analysis.peaks
        ]
    )


def peak_detections_dataframe(run: Stage3ComponentRun) -> pd.DataFrame:
    return pd.DataFrame([record.__dict__ for record in run.peak_detail.detections])


def pair_evidence_dataframe(run: Stage3ComponentRun) -> pd.DataFrame:
    lobes = run.collapse_result.lobe_id_by_peak
    return pd.DataFrame(
        [
            {
                "peak_a": pair.peak_id_a,
                "peak_b": pair.peak_id_b,
                "separation_um": pair.separation_um,
                "saddle_um": pair.saddle_um,
                "branch_persistence": pair.branch_persistence,
                "branch_balance": pair.branch_balance,
                "separation_support": pair.separation_support,
                "peak_support": pair.peak_support,
                "distinct_lobe_probability": pair.distinct_lobe_probability,
                "same_lobe_probability": pair.same_lobe_probability,
                "collapsed_together": (
                    lobes.get(pair.peak_id_a) == lobes.get(pair.peak_id_b)
                ),
            }
            for pair in run.pair_evidence
        ]
    )


__all__ = [
    "pair_evidence_dataframe",
    "peak_detections_dataframe",
    "raw_peaks_dataframe",
    "run_stage3_component",
]
