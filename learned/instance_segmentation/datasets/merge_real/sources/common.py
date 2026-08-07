"""Helpers shared by candidate sources."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import MergeRealConfig
from ..evidence.component import component_is_boundary
from ..evidence.spatial import PointComponentMatch
from ..evidence.volume import volume_sum_evidence
from ..models import CandidateEvidence
from ..observations import ObservationIndex
from ..repository_io import SampleArtifacts


def candidate_volume(sample: SampleArtifacts, frame: int, cell_id: int) -> float:
    cells = sample.cells(frame)
    match = cells.loc[pd.to_numeric(cells["cell_id"], errors="coerce") == int(cell_id)]
    if match.empty:
        return math.nan
    row = match.iloc[0]
    for column in ("volume_voxels", "volume"):
        if column in row.index:
            try:
                value = float(row[column])
                if math.isfinite(value) and value > 0:
                    return value
            except (TypeError, ValueError):
                pass
    labels = sample.labels(frame, mmap=True)
    return float(np.count_nonzero(labels == int(cell_id)))


def make_pair_evidence(
    *,
    source: str,
    sample: SampleArtifacts,
    index: ObservationIndex,
    config: MergeRealConfig,
    frame: int,
    cell_id: int,
    track_a: int,
    track_b: int,
    match_a: PointComponentMatch | None = None,
    match_b: PointComponentMatch | None = None,
    base_score: float = 0.0,
    notes: str = "",
) -> CandidateEvidence | None:
    involved = tuple(sorted({int(track_a), int(track_b)}))
    if index.division_contaminated(involved, frame):
        return None
    if component_is_boundary(sample, frame, cell_id, config):
        return None

    current = candidate_volume(sample, frame, cell_id)
    volume_a = index.reference_volume(track_a, frame)
    volume_b = index.reference_volume(track_b, frame)
    volume = volume_sum_evidence(current, volume_a, volume_b, config)
    if not volume.broad_match:
        return None

    ma = match_a or PointComponentMatch(cell_id, math.nan, False)
    mb = match_b or PointComponentMatch(cell_id, math.nan, False)
    spatial_score = 0.0
    for match in (ma, mb):
        if match.inside:
            spatial_score += 0.5
        elif math.isfinite(match.distance_um):
            spatial_score += 0.5 * max(0.0, 1.0 - match.distance_um / config.prediction_component_radius_um)
    score = float(base_score + volume.score + spatial_score)
    return CandidateEvidence(
        sample_id=sample.sample_id,
        frame=int(frame),
        cell_id=int(cell_id),
        source=source,
        track_a=int(track_a),
        track_b=int(track_b),
        involved_track_ids=involved,
        candidate_volume=current,
        volume_a_reference=volume_a,
        volume_b_reference=volume_b,
        volume_sum_ratio=volume.ratio,
        volume_sum_log_error=volume.log_error,
        prediction_a_distance_um=ma.distance_um,
        prediction_b_distance_um=mb.distance_um,
        prediction_a_inside=ma.inside,
        prediction_b_inside=mb.inside,
        source_score=score,
        notes=notes,
    )


__all__ = ["candidate_volume", "make_pair_evidence"]
