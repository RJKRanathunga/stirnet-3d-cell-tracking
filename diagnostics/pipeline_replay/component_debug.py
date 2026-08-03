"""Presentation model for arrays retained by canonical segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ComponentDebugResult:
    component_id: int
    bbox_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    component_mask: np.ndarray
    padded_component_mask: np.ndarray
    raw_distance: np.ndarray
    watershed_distance: np.ndarray
    merge_tree_distance: np.ndarray
    raw_peak_positions_zyx: np.ndarray
    effective_peak_positions_zyx: np.ndarray
    selected_marker_positions_zyx: np.ndarray
    raw_peak_properties: pd.DataFrame
    effective_peak_properties: pd.DataFrame
    hypothesis_h1_labels: np.ndarray | None
    hypothesis_h2_labels: np.ndarray | None
    hypothesis_h3_labels: np.ndarray | None
    selected_labels: np.ndarray
    pair_evidence: pd.DataFrame
    hypothesis_evidence: pd.DataFrame
    selected_cell_count: int
    decision_status: str
    h2_conditional_probability: float
    h2_odds_vs_h1: float
    h3_conditional_probability: float
    h3_odds_vs_h2: float
    error: str | None = None


def component_debug_result(artifact, padding: int) -> ComponentDebugResult:
    """Convert the canonical retained artifact without recalculating evidence."""

    def positions(peaks) -> np.ndarray:
        if not peaks:
            return np.empty((0, 3), dtype=float)
        return np.asarray(peaks, dtype=float) - float(padding)

    hypotheses = {hypothesis.k: hypothesis for hypothesis in artifact.hypotheses}
    effective_ids = {peak.peak_id for peak in artifact.effective_peaks}
    selected_ids = {peak.peak_id for peak in artifact.decision.chosen.selected_peaks}
    def peak_table(peaks) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "component_id": artifact.component_id,
                "peak_id": peak.peak_id,
                "selected": peak.peak_id in selected_ids,
                "effective": peak.peak_id in effective_ids,
                "raw_depth_um": peak.raw_depth_um,
                "smoothed_depth_um": peak.smoothed_depth_um,
                "persistence": peak.persistence_score,
                "scale_support": peak.scale_support,
                "h_support": peak.h_support,
                "collapse_status": "retained" if peak.peak_id in effective_ids else "collapsed",
            }
            for peak in peaks
        ])
    pair_rows = [
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
        }
        for pair in artifact.pair_evidence
    ]
    hypothesis_rows = []
    for k, hypothesis in hypotheses.items():
        row = {
            "k": k,
            "posterior_probability": hypothesis.posterior_probability,
            "prior_probability": hypothesis.prior_probability,
            "log_likelihood": hypothesis.log_likelihood,
            "hard_valid": hypothesis.hard_valid,
            "hard_reasons": "; ".join(hypothesis.hard_reasons),
            "selected_peak_ids": tuple(p.peak_id for p in hypothesis.selected_peaks),
            **hypothesis.evidence.as_dict(),
        }
        hypothesis_rows.append(row)
    inner = tuple(slice(padding, -padding) if padding else slice(None) for _ in range(3))
    return ComponentDebugResult(
        artifact.component_id,
        artifact.bbox_zyx,
        artifact.component_mask,
        artifact.padded_component_mask,
        artifact.raw_distance,
        artifact.watershed_distance,
        artifact.merge_tree_distance,
        positions([peak.position_zyx for peak in artifact.peaks]),
        positions([peak.position_zyx for peak in artifact.effective_peaks]),
        np.asarray(artifact.selected_positions_zyx, dtype=float).reshape((-1, 3)),
        peak_table(artifact.peaks),
        peak_table(artifact.effective_peaks),
        hypotheses[1].labels[inner] if 1 in hypotheses else None,
        hypotheses[2].labels[inner] if 2 in hypotheses else None,
        hypotheses[3].labels[inner] if 3 in hypotheses else None,
        artifact.selected_labels,
        pd.DataFrame(pair_rows),
        pd.DataFrame(hypothesis_rows),
        int(artifact.decision.chosen.k),
        artifact.decision.decision_status,
        artifact.decision.h2_conditional_probability,
        artifact.decision.h2_odds_vs_h1,
        artifact.decision.h3_conditional_probability,
        artifact.decision.h3_odds_vs_h2,
    )
