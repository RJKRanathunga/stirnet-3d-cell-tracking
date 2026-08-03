"""Headless canonical Stage 3 orchestration and diagnostic reports."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd

from .models import CanonicalMismatchError, Stage3ComponentRun


peaks_module = import_module("src.03_segmentation.peaks")
geometry_module = import_module("src.03_segmentation.geometry")
hypotheses_module = import_module("src.03_segmentation.hypotheses")
pipeline_module = import_module("src.03_segmentation.pipeline")

EVIDENCE_NAMES = (
    "lobe_support",
    "coverage_support",
    "marker_quality",
    "neck_support",
    "child_shape",
    "shape_improvement",
    "child_volume",
    "fragment_safety",
)


def run_stage3_component(resolution, component_id: int, config) -> Stage3ComponentRun:
    """Execute the canonical Stage 3 steps and verify the production wrapper."""

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
    merged_description = geometry_module.describe_cell_mask(
        padded_mask, config.voxel_size_zyx_um
    )
    evaluation = hypotheses_module.evaluate_spatial_split_hypotheses(
        padded_mask,
        peak_analysis.raw_distance,
        peak_analysis.watershed_distance,
        collapse_result.effective_peaks,
        pair_evidence,
        merged_description,
        config,
    )
    decision = hypotheses_module.choose_hierarchical_hypothesis(
        evaluation.best_by_k, config
    )
    canonical = pipeline_module.analyze_component_crop(
        component_mask, config, retain_debug_artifacts=True
    )

    inner = tuple(
        slice(padding, -padding) if padding else slice(None) for _ in range(3)
    )
    manual_labels = np.asarray(decision.chosen.labels[inner], dtype=np.int32)
    manual_positions = tuple(
        tuple(int(value - padding) for value in peak.position_zyx)
        for peak in decision.chosen.selected_peaks
    )
    checks = {
        "labels": np.array_equal(manual_labels, canonical.labels),
        "decision status": (
            decision.decision_status == canonical.decision.decision_status
        ),
        "selected peak positions": (
            manual_positions == canonical.selected_positions_zyx
        ),
        "selected peak IDs": (
            tuple(peak.peak_id for peak in decision.chosen.selected_peaks)
            == tuple(
                peak.peak_id for peak in canonical.decision.chosen.selected_peaks
            )
        ),
        "selected cell count": (
            decision.chosen.k == len(canonical.selected_positions_zyx)
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
        merged_description,
        evaluation,
        decision,
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


def hypothesis_posterior(run: Stage3ComponentRun, hypothesis) -> float:
    peak_ids = tuple(peak.peak_id for peak in hypothesis.selected_peaks)
    for best in run.evaluation.best_by_k:
        if best.k == hypothesis.k and tuple(
            peak.peak_id for peak in best.selected_peaks
        ) == peak_ids:
            return float(best.posterior_probability)
    return float("nan")


def all_hypotheses_dataframe(run: Stage3ComponentRun) -> pd.DataFrame:
    weights = run.config.evidence_weights()
    epsilon = run.config.probability_epsilon
    rows = []
    for index, hypothesis in enumerate(run.evaluation.all_hypotheses):
        evidence = hypothesis.evidence.as_dict()
        row = {
            "index": index,
            "k": hypothesis.k,
            "selected_peak_ids": tuple(
                peak.peak_id for peak in hypothesis.selected_peaks
            ),
            "combination_prescore": hypothesis.combination_prescore,
            "hard_valid": hypothesis.hard_valid,
            "hard_reasons": ", ".join(hypothesis.hard_reasons),
            "prior_probability": hypothesis.prior_probability,
            "log_likelihood": hypothesis.log_likelihood,
            "posterior_probability": hypothesis_posterior(run, hypothesis),
            "minimum_child_fraction": hypothesis.minimum_child_fraction,
        }
        row.update(evidence)
        row.update(
            {
                f"{name}_weighted_log_contribution": (
                    weights[name]
                    * np.log(np.clip(evidence[name], epsilon, 1.0))
                )
                for name in EVIDENCE_NAMES
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def final_decision_dataframe(run: Stage3ComponentRun) -> pd.DataFrame:
    posterior = {
        hypothesis.k: hypothesis.posterior_probability
        for hypothesis in run.evaluation.best_by_k
    }
    decision, config = run.decision, run.config
    return pd.DataFrame(
        [
            {
                "decision_status": decision.decision_status,
                "selected_cell_count": decision.chosen.k,
                "fallback_or_error": False,
                "posterior_h1": posterior.get(1, 0.0),
                "posterior_h2": posterior.get(2, 0.0),
                "posterior_h3": posterior.get(3, 0.0),
                "h2_conditional_probability": decision.h2_conditional_probability,
                "h2_odds_vs_h1": decision.h2_odds_vs_h1,
                "h2_required_conditional": config.h2_min_conditional_probability,
                "h2_required_odds": config.h2_min_odds_vs_h1,
                "h3_conditional_probability": decision.h3_conditional_probability,
                "h3_odds_vs_h2": decision.h3_odds_vs_h2,
                "h3_required_conditional": config.h3_min_conditional_probability,
                "h3_required_odds": config.h3_min_odds_vs_h2,
            }
        ]
    )


def hypothesis_summary_text(run: Stage3ComponentRun, index: int) -> str:
    hypothesis = run.evaluation.all_hypotheses[int(index)]
    evidence = ", ".join(
        f"{name}={value:.3f}"
        for name, value in hypothesis.evidence.as_dict().items()
    )
    return (
        f"H{hypothesis.k} #{index}; "
        f"peaks={tuple(peak.peak_id for peak in hypothesis.selected_peaks)}; "
        f"prescore={hypothesis.combination_prescore:.3f}; "
        f"hard_valid={hypothesis.hard_valid}; "
        f"reasons={hypothesis.hard_reasons or ()}; "
        f"prior={hypothesis.prior_probability:.4f}; "
        f"log_likelihood={hypothesis.log_likelihood:.3f}; "
        f"posterior={hypothesis_posterior(run, hypothesis)}; "
        f"min_child_fraction={hypothesis.minimum_child_fraction:.4f}; {evidence}"
    )


__all__ = [
    "all_hypotheses_dataframe",
    "final_decision_dataframe",
    "hypothesis_summary_text",
    "pair_evidence_dataframe",
    "peak_detections_dataframe",
    "raw_peaks_dataframe",
    "run_stage3_component",
]
