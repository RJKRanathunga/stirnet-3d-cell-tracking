"""Hard-gated continuation candidate generation and soft feature evidence."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .step01_config import TrackReconciliationConfig
from .step02_observations import physical_position, real_track_observations
from .step03_protections import (
    lineage_edge_conflict,
    protected_transition_tracks,
    transition_overlaps_merge,
)
from .step04_motion import (
    backward_prediction,
    bidirectional_disagreement,
    forward_prediction,
)
from .step05_neighborhood import anchor_evidence


SHAPE_FEATURES = (
    "equivalent_radius", "axis_major", "axis_middle", "axis_minor",
    "elongation", "flatness", "anisotropy", "solidity", "compactness",
    "bbox_depth", "bbox_height", "bbox_width",
)
INTENSITY_FEATURES = (
    "intensity_mean", "intensity_median", "intensity_std", "intensity_sum",
)


def _finite_median(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return math.nan
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else math.nan


def _feature_error(
    source: pd.DataFrame,
    target: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    logarithmic: bool = False,
) -> float:
    errors: list[float] = []
    for column in columns:
        first = _finite_median(source, column)
        second = _finite_median(target, column)
        if not math.isfinite(first) or not math.isfinite(second):
            continue
        if logarithmic and first > 0 and second > 0:
            errors.append(abs(math.log(second / first)))
        else:
            denominator = max(abs(first), abs(second), 1e-8)
            errors.append(abs(second - first) / denominator)
    return float(np.median(errors)) if errors else math.nan


def _target_quality(
    target_real: pd.DataFrame,
    sequence_last_frame: int,
    config: TrackReconciliationConfig,
) -> float:
    count = len(target_real)
    if count == 0:
        return 0.0
    persistence = 1.0 - math.exp(-count / 2.0)
    if int(target_real["frame"].max()) == sequence_last_frame:
        persistence = max(persistence, 0.7)
    smoothness = 0.6
    ordered = target_real.sort_values("frame", kind="mergesort").head(5)
    if len(ordered) >= 3:
        positions = ordered[["z", "y", "x"]].to_numpy(dtype=float) * np.asarray(
            config.voxel_size_zyx_um, dtype=float
        )
        frames = ordered["frame"].to_numpy(dtype=float)
        velocities = np.diff(positions, axis=0) / np.diff(frames)[:, None]
        center = np.median(velocities, axis=0)
        dispersion = float(np.median(np.linalg.norm(velocities - center, axis=1)))
        smoothness = math.exp(-dispersion / config.position_score_scale_um)
    association = math.nan
    if "association_probability" in target_real:
        values = pd.to_numeric(target_real["association_probability"], errors="coerce")
        values = values[np.isfinite(values)]
        if len(values):
            association = float(np.clip(values.head(3).median(), 0.0, 1.0))
    components = [persistence, smoothness]
    if math.isfinite(association):
        components.append(association)
    return float(np.mean(components))


def _stage7_alternative(
    source_real: pd.DataFrame,
    target_start: pd.Series,
    association_candidates: pd.DataFrame | None,
) -> dict[str, object]:
    defaults = {
        "stage7_candidate_available": False,
        "stage7_candidate_distance_um": math.nan,
        "stage7_candidate_pair_cost": math.nan,
        "stage7_candidate_probability": math.nan,
        "stage7_candidate_rank": math.nan,
    }
    if association_candidates is None or association_candidates.empty:
        return defaults
    required = {"track_id", "to_frame", "detection_cell_index"}
    if not required.issubset(association_candidates.columns):
        return defaults
    source_track_id = int(source_real.iloc[-1]["track_id"])
    # A Stage 8 provenance remap makes the pre-Stage-8 identity unsafe.
    if "source_track_id" in source_real:
        provenance = pd.to_numeric(source_real["source_track_id"], errors="coerce")
        provenance = provenance[np.isfinite(provenance)]
        if len(provenance) and not (provenance.astype(int) == source_track_id).all():
            return defaults
    matches = association_candidates.loc[
        (pd.to_numeric(association_candidates["track_id"], errors="coerce") == source_track_id)
        & (pd.to_numeric(association_candidates["to_frame"], errors="coerce") == int(target_start["frame"]))
        & (pd.to_numeric(association_candidates["detection_cell_index"], errors="coerce") == int(target_start["cell"]))
    ].copy()
    if matches.empty:
        return defaults
    sort_columns = [column for column in (
        "candidate_rank_by_pair_cost", "pair_cost", "detection_cell_index"
    ) if column in matches]
    row = matches.sort_values(sort_columns, kind="mergesort").iloc[0]
    def numeric(name: str) -> float:
        try:
            value = float(row.get(name, math.nan))
            return value if math.isfinite(value) else math.nan
        except (TypeError, ValueError):
            return math.nan
    return {
        "stage7_candidate_available": True,
        "stage7_candidate_distance_um": numeric("distance_um"),
        "stage7_candidate_pair_cost": numeric("pair_cost"),
        "stage7_candidate_probability": numeric("association_probability"),
        "stage7_candidate_rank": numeric("candidate_rank_by_pair_cost"),
    }


def generate_candidates(
    observations: pd.DataFrame,
    endpoints: pd.DataFrame,
    *,
    sample_id: str,
    sequence_last_frame: int,
    segmentation_events: pd.DataFrame | None,
    division_events: pd.DataFrame | None,
    lineage_edges: pd.DataFrame | None,
    global_motion: pd.DataFrame | None,
    association_candidates: pd.DataFrame | None,
    spatial_shape_zyx: tuple[int, int, int] | None,
    config: TrackReconciliationConfig,
) -> list[dict[str, object]]:
    """Generate candidates only inside the configured temporal/physical gates."""

    if endpoints.empty:
        return []
    starts_by_frame: dict[int, pd.DataFrame] = {
        int(frame): group.sort_values("track_id", kind="mergesort")
        for frame, group in endpoints.groupby("first_real_frame", dropna=True, sort=True)
    }
    records: list[dict[str, object]] = []
    sources = endpoints.loc[endpoints["source_eligible"].astype(bool)].sort_values(
        ["last_real_frame", "track_id"], kind="mergesort"
    )
    spacing = np.asarray(config.voxel_size_zyx_um, dtype=float)
    for source in sources.itertuples(index=False):
        source_track_id = int(source.track_id)
        source_end_frame = int(source.last_real_frame)
        source_real = real_track_observations(observations, source_track_id)
        source_end = observations.loc[int(source.last_real_observation_index)]
        source_position = source_end[["z", "y", "x"]].to_numpy(dtype=float) * spacing
        for target_frame in range(
            source_end_frame + 1,
            source_end_frame + config.maximum_gap_frames + 1,
        ):
            starts = starts_by_frame.get(target_frame)
            if starts is None or starts.empty:
                continue
            gap = target_frame - source_end_frame
            radius = min(
                config.maximum_candidate_radius_um,
                config.candidate_radius_base_um
                + config.candidate_radius_per_additional_gap_um * (gap - 1),
            )
            start_indices = starts["first_real_observation_index"].astype(int).to_numpy()
            start_rows = observations.loc[start_indices]
            positions = start_rows[["z", "y", "x"]].to_numpy(dtype=float) * spacing
            distances = np.linalg.norm(positions - source_position[None, :], axis=1)
            inside = np.flatnonzero(distances <= radius + 1e-12)
            for position_index in inside:
                target = starts.iloc[int(position_index)]
                target_track_id = int(target["track_id"])
                if target_track_id == source_track_id:
                    continue
                target_start = observations.loc[int(target["first_real_observation_index"])]
                target_real = real_track_observations(observations, target_track_id)
                reason = "admissible"
                if not bool(target["target_eligible"]):
                    reason = str(target["target_exclusion_reason"])
                elif int(target["first_real_frame"]) <= source_end_frame:
                    reason = "excluded_temporal_overlap"
                elif int(source.last_frame) >= int(target["first_frame"]):
                    reason = "excluded_temporal_overlap"
                elif transition_overlaps_merge(
                    source_track_id, target_track_id, source_end_frame,
                    target_frame, segmentation_events,
                ):
                    reason = "excluded_merge_event"
                elif lineage_edge_conflict(source_track_id, target_track_id, lineage_edges):
                    reason = "excluded_confirmed_division"
                protected_ids = protected_transition_tracks(
                    source_end_frame, target_frame, division_events, segmentation_events
                )
                forward = forward_prediction(
                    source_real, target_start, global_motion, config
                )
                backward = backward_prediction(target_real, source_end, config)
                anchors = anchor_evidence(
                    observations, source_end, target_start,
                    source_track_id=source_track_id,
                    target_track_id=target_track_id,
                    protected_track_ids=protected_ids,
                    spatial_shape_zyx=spatial_shape_zyx,
                    config=config,
                )
                # With no complete Stage 7 global-motion path, persistent
                # local anchors provide the preferred estimate of scene/local
                # displacement. This precedes the robust source-only fallback.
                if (
                    not bool(forward["forward_used_global_motion"])
                    and int(anchors["anchor_count"])
                    >= config.minimum_anchor_count_for_strong_support
                    and math.isfinite(float(anchors["anchor_prediction_error_um"]))
                ):
                    forward["forward_predicted_z_um"] = anchors["anchor_predicted_z_um"]
                    forward["forward_predicted_y_um"] = anchors["anchor_predicted_y_um"]
                    forward["forward_predicted_x_um"] = anchors["anchor_predicted_x_um"]
                    forward["forward_error_um"] = anchors["anchor_prediction_error_um"]
                source_reference = source_real.tail(config.feature_history)
                target_reference = target_real.head(config.feature_history)
                source_volume = _finite_median(source_reference, "volume_voxels")
                target_volume = _finite_median(target_reference, "volume_voxels")
                if not math.isfinite(source_volume):
                    source_volume = _finite_median(source_reference, "volume")
                if not math.isfinite(target_volume):
                    target_volume = _finite_median(target_reference, "volume")
                volume_error = (
                    abs(math.log(target_volume / source_volume))
                    if source_volume > 0 and target_volume > 0 else math.nan
                )
                stage7 = _stage7_alternative(
                    source_real, target_start, association_candidates
                )
                records.append({
                    "candidate_id": (
                        f"{sample_id}:continuation:{source_track_id}:{target_track_id}:"
                        f"{source_end_frame}:{target_frame}"
                    ),
                    "sample_id": str(sample_id),
                    "source_track_id": source_track_id,
                    "target_track_id": target_track_id,
                    "source_end_frame": source_end_frame,
                    "target_start_frame": target_frame,
                    "gap_frames": gap,
                    "hard_search_radius_um": float(radius),
                    "direct_endpoint_distance_um": float(distances[position_index]),
                    "admissible": reason == "admissible",
                    "admissibility_reason": reason,
                    **forward,
                    **backward,
                    "bidirectional_disagreement_um": bidirectional_disagreement(
                        forward, backward, source_end, target_start, config
                    ),
                    **anchors,
                    "source_reference_volume": source_volume,
                    "target_reference_volume": target_volume,
                    "volume_log_error": volume_error,
                    "shape_error": _feature_error(
                        source_reference, target_reference, SHAPE_FEATURES
                    ),
                    "intensity_error": _feature_error(
                        source_reference, target_reference, INTENSITY_FEATURES,
                        logarithmic=True,
                    ),
                    "target_real_observation_count": int(len(target_real)),
                    "candidate_quality_score": _target_quality(
                        target_real, sequence_last_frame, config
                    ),
                    **stage7,
                })
    return sorted(
        records,
        key=lambda row: (
            int(row["source_end_frame"]), int(row["source_track_id"]),
            int(row["target_start_frame"]), int(row["target_track_id"]),
        ),
    )
