"""Source 4: tracking-independent large/abnormal Stage 6 components."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import MergeRealConfig
from ..evidence.component import component_is_boundary, count_edt_peaks
from ..evidence.spatial import PointComponentMatch
from ..models import CandidateEvidence
from ..observations import ObservationIndex
from ..repository_io import SampleArtifacts
from .common import candidate_volume, make_pair_evidence


def mine_source4(
    sample: SampleArtifacts,
    index: ObservationIndex,
    config: MergeRealConfig,
):
    records = []
    for frame in range(1, sample.frame_count):
        cells = sample.cells(frame)
        if cells.empty or "cell_id" not in cells:
            continue
        volume_column = "volume_voxels" if "volume_voxels" in cells else None
        if volume_column is None:
            continue
        volumes = pd.to_numeric(cells[volume_column], errors="coerce").to_numpy(dtype=float)
        finite = volumes[np.isfinite(volumes) & (volumes > 0)]
        if finite.size == 0:
            continue
        frame_median = float(np.median(finite))
        for row in cells.itertuples(index=False):
            cell_id = int(row.cell_id)
            current = float(getattr(row, volume_column))
            if not math.isfinite(current) or current <= 0:
                continue
            if component_is_boundary(sample, frame, cell_id, config):
                continue
            owners = index.track_ids_for_cell(frame, cell_id)
            owner = owners[0] if owners else None
            track_ratio = math.nan
            if owner is not None:
                reference = index.reference_volume(owner, frame)
                if math.isfinite(reference) and reference > 0:
                    track_ratio = current / reference
            frame_ratio = current / max(frame_median, 1e-8)
            if (
                frame_ratio < config.anomaly_frame_volume_ratio
                and (not math.isfinite(track_ratio) or track_ratio < config.anomaly_track_volume_ratio)
            ):
                continue
            peak_count = count_edt_peaks(sample, frame, cell_id, config)
            if peak_count < config.anomaly_min_edt_peaks:
                continue

            # Prefer a pair interpretation when an owner plus a nearby previous track
            # explain the large component volume. Otherwise retain a source-4-only case.
            best = None
            if owner is not None:
                owner_pos = index.predict_zyx(owner, frame)
                previous = index.tracks_in_frame(frame - 1)
                if owner_pos is not None and not previous.empty:
                    spacing = np.asarray(config.voxel_size_zyx_um, dtype=float)
                    owner_um = owner_pos * spacing
                    candidates = []
                    for prev in previous.itertuples(index=False):
                        other = int(prev.track_id)
                        if other == owner:
                            continue
                        pred = index.predict_zyx(other, frame)
                        if pred is None:
                            continue
                        distance = float(np.linalg.norm((pred * spacing) - owner_um))
                        if distance <= 2.0 * config.prediction_component_radius_um:
                            candidates.append((distance, other))
                    for _, other in sorted(candidates)[:6]:
                        evidence = make_pair_evidence(
                            source="source4_component_anomaly",
                            sample=sample,
                            index=index,
                            config=config,
                            frame=frame,
                            cell_id=cell_id,
                            track_a=owner,
                            track_b=other,
                            match_a=PointComponentMatch(cell_id, 0.0, True),
                            match_b=PointComponentMatch(cell_id, math.nan, False),
                            base_score=1.0 + min(frame_ratio - 1.0, 1.0),
                            notes=f"large component with {peak_count} EDT peak regions",
                        )
                        if evidence is not None:
                            evidence = CandidateEvidence(
                                **{**evidence.__dict__, "edt_peak_count": int(peak_count)}
                            )
                            if best is None or evidence.volume_sum_log_error < best.volume_sum_log_error:
                                best = evidence
            if best is not None:
                records.append(best)
            else:
                # Keep a pure anomaly candidate for manual inspection even when no
                # trustworthy two-track interpretation exists.
                involved = tuple(int(v) for v in owners)
                if index.division_contaminated(involved, frame):
                    continue
                records.append(
                    CandidateEvidence(
                        sample_id=sample.sample_id,
                        frame=frame,
                        cell_id=cell_id,
                        source="source4_component_anomaly",
                        track_a=owner,
                        involved_track_ids=involved,
                        candidate_volume=candidate_volume(sample, frame, cell_id),
                        edt_peak_count=int(peak_count),
                        source_score=float(1.0 + min(frame_ratio - 1.0, 1.0)),
                        notes=f"volume anomaly (frame_ratio={frame_ratio:.3f}, track_ratio={track_ratio:.3f}) with {peak_count} EDT peaks",
                    )
                )
    return records


__all__ = ["mine_source4"]
