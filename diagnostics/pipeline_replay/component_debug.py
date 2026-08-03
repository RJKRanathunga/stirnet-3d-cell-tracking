"""Presentation model for arrays retained by canonical segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ComponentDebugResult:
    """Neutral replay data for one successfully analyzed Stage 3 component."""

    component_id: int
    bbox_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    component_mask: np.ndarray
    padded_component_mask: np.ndarray
    raw_distance: np.ndarray
    merge_tree_distance: np.ndarray
    watershed_distance: np.ndarray
    raw_peak_positions_zyx: np.ndarray
    effective_peak_positions_zyx: np.ndarray
    marker_positions_zyx: np.ndarray
    raw_peak_properties: pd.DataFrame
    effective_peak_properties: pd.DataFrame
    final_labels: np.ndarray
    pair_evidence: pd.DataFrame
    raw_peak_count: int
    effective_peak_count: int
    marker_count: int
    instance_count: int
    processing_status: str
    error: str | None = None


def component_debug_result(artifact, diagnostic, padding: int) -> ComponentDebugResult:
    """Convert retained canonical arrays without recalculating Stage 3."""

    def positions(peaks) -> np.ndarray:
        if not peaks:
            return np.empty((0, 3), dtype=float)
        return np.asarray(
            [peak.position_zyx for peak in peaks], dtype=float
        ) - float(padding)

    effective_ids = {peak.peak_id for peak in artifact.effective_peaks}

    def peak_table(peaks) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "component_id": artifact.component_id,
                    "peak_id": peak.peak_id,
                    "effective": peak.peak_id in effective_ids,
                    "final_marker": peak.peak_id in effective_ids,
                    "raw_depth_um": peak.raw_depth_um,
                    "smoothed_depth_um": peak.smoothed_depth_um,
                    "persistence": peak.persistence_score,
                    "scale_support": peak.scale_support,
                    "h_support": peak.h_support,
                    "collapse_status": (
                        "retained"
                        if peak.peak_id in effective_ids
                        else "collapsed"
                    ),
                }
                for peak in peaks
            ]
        )

    pair_evidence = pd.DataFrame(
        [
            {
                "peak_a": pair.peak_id_a,
                "peak_b": pair.peak_id_b,
                "same_lobe_probability": pair.same_lobe_probability,
                "distinct_lobe_probability": pair.distinct_lobe_probability,
                "branch_persistence": pair.branch_persistence,
                "branch_balance": pair.branch_balance,
                "separation_support": pair.separation_support,
                "peak_support": pair.peak_support,
                "separation_um": pair.separation_um,
                "saddle_um": pair.saddle_um,
            }
            for pair in artifact.pair_evidence
        ]
    )
    return ComponentDebugResult(
        component_id=artifact.component_id,
        bbox_zyx=artifact.bbox_zyx,
        component_mask=artifact.component_mask,
        padded_component_mask=artifact.padded_component_mask,
        raw_distance=artifact.raw_distance,
        merge_tree_distance=artifact.merge_tree_distance,
        watershed_distance=artifact.watershed_distance,
        raw_peak_positions_zyx=positions(artifact.raw_peaks),
        effective_peak_positions_zyx=positions(artifact.effective_peaks),
        marker_positions_zyx=np.asarray(
            artifact.marker_positions_zyx, dtype=float
        ).reshape((-1, 3)),
        raw_peak_properties=peak_table(artifact.raw_peaks),
        effective_peak_properties=peak_table(artifact.effective_peaks),
        final_labels=artifact.final_labels,
        pair_evidence=pair_evidence,
        raw_peak_count=int(diagnostic.raw_peak_count),
        effective_peak_count=int(diagnostic.effective_peak_count),
        marker_count=int(diagnostic.marker_count),
        instance_count=int(diagnostic.instance_count),
        processing_status=str(diagnostic.processing_status),
        error=diagnostic.error,
    )
