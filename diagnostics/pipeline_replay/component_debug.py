"""Presentation model for arrays retained by canonical segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

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
    shape_peak_positions_zyx: np.ndarray
    center_proposal_positions_zyx: np.ndarray
    unrepresented_proposal_positions_zyx: np.ndarray
    candidate_proposal_positions_zyx: np.ndarray
    supplemental_marker_positions_zyx: np.ndarray
    final_marker_positions_zyx: np.ndarray
    marker_positions_zyx: np.ndarray
    raw_peak_properties: pd.DataFrame
    effective_peak_properties: pd.DataFrame
    final_labels: np.ndarray
    pair_evidence: pd.DataFrame
    shape_peaks: pd.DataFrame
    center_proposals: pd.DataFrame
    candidate_summary: pd.DataFrame
    surface_caps: pd.DataFrame
    body_candidates: pd.DataFrame
    cross_sections: pd.DataFrame
    marker_completion: pd.DataFrame
    boundary_positions_zyx: np.ndarray
    boundary_normals_zyx: np.ndarray
    ellipsoid_support_zyx: np.ndarray
    unique_support_zyx: np.ndarray
    binary_log_response: np.ndarray | None
    binary_log_maxima_zyx: np.ndarray
    binary_log_sigma_um: float | None
    raw_peak_count: int
    effective_peak_count: int
    shape_peak_count: int
    center_proposal_count: int
    unrepresented_proposal_count: int
    candidate_proposal_count: int
    merge_candidate: bool
    candidate_routes: tuple[str, ...]
    candidate_processing_status: str
    candidate_error: str | None
    geometry_forced: bool
    geometry_executed: bool
    supplemental_marker_count: int
    final_marker_count: int
    marker_count: int
    instance_count: int
    processing_status: str
    geometry_processing_status: str
    geometry_error: str | None
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
    candidate_detection_module = import_module(
        "src.03_segmentation.candidate_detection"
    )
    candidate_result = artifact.candidate_result
    shape_peaks = candidate_detection_module.shape_peaks_dataframe(
        candidate_result, artifact.component_id
    )
    center_proposals = candidate_detection_module.center_proposals_dataframe(
        candidate_result, artifact.component_id
    )
    candidate_summary = candidate_detection_module.candidate_summary_dataframe(
        candidate_result,
        artifact.component_id,
        len(artifact.raw_peaks),
        len(artifact.effective_peaks),
    )
    for table in (shape_peaks, center_proposals):
        for axis in "zyx":
            if axis in table:
                table[axis] = table[axis] - float(padding)
    marker_completion_module = import_module("src.03_segmentation.marker_completion")
    completion = artifact.geometric_completion
    surface_caps = marker_completion_module.surface_caps_dataframe(
        completion, artifact.component_id
    )
    body_candidates = marker_completion_module.body_candidates_dataframe(
        completion, artifact.component_id
    )
    for table in (surface_caps, body_candidates):
        for axis in "zyx":
            if axis in table:
                table[axis] = table[axis] - float(padding)
    for prefix in ("axis_start_", "axis_end_"):
        for axis in "zyx":
            column = f"{prefix}{axis}"
            if column in body_candidates:
                body_candidates[column] = body_candidates[column] - float(padding)
    cross_sections = marker_completion_module.cross_sections_dataframe(
        completion, artifact.component_id
    )
    marker_completion = marker_completion_module.marker_completion_dataframe(
        completion, artifact.component_id
    )
    for axis in "zyx":
        column = f"marker_{axis}"
        if column in marker_completion:
            marker_completion[column] = marker_completion[column] - float(padding)
    debug = completion.debug_artifacts
    empty_points = np.empty((0, 3), dtype=float)
    boundary_positions = (
        debug.boundary_positions_zyx - float(padding)
        if debug is not None else empty_points
    )
    boundary_normals = (
        debug.boundary_normals_zyx if debug is not None else empty_points
    )
    ellipsoid_support = (
        debug.ellipsoid_support_zyx - float(padding)
        if debug is not None else empty_points
    )
    unique_support = (
        debug.unique_support_zyx - float(padding)
        if debug is not None else empty_points
    )
    supplemental_positions = np.asarray(
        [
            np.asarray(marker.position_zyx, dtype=float) - float(padding)
            for marker in completion.supplemental_markers
        ],
        dtype=float,
    ).reshape((-1, 3))
    final_marker_positions = np.asarray(
        [
            np.asarray(marker.position_zyx, dtype=float) - float(padding)
            for marker in artifact.final_markers
        ],
        dtype=float,
    ).reshape((-1, 3))
    shape_peak_positions = np.asarray(
        [
            np.asarray(peak.position_zyx, dtype=float) - float(padding)
            for peak in candidate_result.shape_peaks
        ],
        dtype=float,
    ).reshape((-1, 3))
    proposal_positions = np.asarray(
        [
            np.asarray(proposal.position_zyx, dtype=float) - float(padding)
            for proposal in candidate_result.proposals
        ],
        dtype=float,
    ).reshape((-1, 3))
    unrepresented_positions = np.asarray(
        [
            np.asarray(proposal.position_zyx, dtype=float) - float(padding)
            for proposal in candidate_result.proposals
            if not proposal.represented
        ],
        dtype=float,
    ).reshape((-1, 3))
    candidate_positions = np.asarray(
        [
            np.asarray(proposal.position_zyx, dtype=float) - float(padding)
            for proposal in candidate_result.proposals
            if proposal.candidate
        ],
        dtype=float,
    ).reshape((-1, 3))
    candidate_debug = candidate_result.debug_artifacts
    binary_log_response = None
    binary_log_maxima = empty_points
    binary_log_sigma = None
    if candidate_debug is not None and candidate_debug.response_volumes:
        selected_scale = max(
            range(len(candidate_debug.sigma_levels_um)),
            key=lambda index: (
                float(np.max(candidate_debug.response_volumes[index])),
                -index,
            ),
        )
        binary_log_response = candidate_debug.response_volumes[selected_scale]
        binary_log_maxima = (
            candidate_debug.raw_maxima_zyx[selected_scale].astype(float)
            - float(padding)
        )
        binary_log_sigma = candidate_debug.sigma_levels_um[selected_scale]
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
        shape_peak_positions_zyx=shape_peak_positions,
        center_proposal_positions_zyx=proposal_positions,
        unrepresented_proposal_positions_zyx=unrepresented_positions,
        candidate_proposal_positions_zyx=candidate_positions,
        supplemental_marker_positions_zyx=supplemental_positions,
        final_marker_positions_zyx=final_marker_positions,
        marker_positions_zyx=np.asarray(
            artifact.marker_positions_zyx, dtype=float
        ).reshape((-1, 3)),
        raw_peak_properties=peak_table(artifact.raw_peaks),
        effective_peak_properties=peak_table(artifact.effective_peaks),
        final_labels=artifact.final_labels,
        pair_evidence=pair_evidence,
        shape_peaks=shape_peaks,
        center_proposals=center_proposals,
        candidate_summary=candidate_summary,
        surface_caps=surface_caps,
        body_candidates=body_candidates,
        cross_sections=cross_sections,
        marker_completion=marker_completion,
        boundary_positions_zyx=boundary_positions,
        boundary_normals_zyx=boundary_normals,
        ellipsoid_support_zyx=ellipsoid_support,
        unique_support_zyx=unique_support,
        binary_log_response=binary_log_response,
        binary_log_maxima_zyx=binary_log_maxima,
        binary_log_sigma_um=binary_log_sigma,
        raw_peak_count=int(diagnostic.raw_peak_count),
        effective_peak_count=int(diagnostic.effective_peak_count),
        shape_peak_count=int(diagnostic.shape_peak_count),
        center_proposal_count=int(diagnostic.center_proposal_count),
        unrepresented_proposal_count=int(diagnostic.unrepresented_proposal_count),
        candidate_proposal_count=int(diagnostic.candidate_proposal_count),
        merge_candidate=bool(diagnostic.merge_candidate),
        candidate_routes=tuple(diagnostic.candidate_routes),
        candidate_processing_status=str(diagnostic.candidate_processing_status),
        candidate_error=diagnostic.candidate_error,
        geometry_forced=bool(diagnostic.geometry_forced),
        geometry_executed=bool(diagnostic.geometry_executed),
        supplemental_marker_count=int(diagnostic.supplemental_marker_count),
        final_marker_count=int(diagnostic.final_marker_count),
        marker_count=int(diagnostic.marker_count),
        instance_count=int(diagnostic.instance_count),
        processing_status=str(diagnostic.processing_status),
        geometry_processing_status=str(diagnostic.geometry_processing_status),
        geometry_error=diagnostic.geometry_error,
        error=diagnostic.error,
    )
