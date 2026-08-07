"""Source 3: an interior ending projects into a continuing track's component."""

from __future__ import annotations

from ..config import MergeRealConfig
from ..evidence.spatial import associate_point_to_component
from ..observations import ObservationIndex
from ..repository_io import SampleArtifacts
from .common import make_pair_evidence


def mine_source3(
    sample: SampleArtifacts,
    index: ObservationIndex,
    config: MergeRealConfig,
):
    records = []
    last_sequence_frame = sample.frame_count - 1
    for track_a in index.track_ids:
        end = index.last_frame(track_a)
        if end is None or end >= last_sequence_frame - 0:
            continue
        frame = end + 1
        if frame >= sample.frame_count:
            continue
        if index.endpoint_is_boundary(track_a, end):
            continue
        predicted_a = index.predict_zyx(track_a, frame)
        match_a = associate_point_to_component(sample, frame, predicted_a, config)
        if match_a.cell_id is None:
            continue
        for track_b in index.track_ids_for_cell(frame, match_a.cell_id):
            if track_b == track_a:
                continue
            # This source specifically means A disappears beside an already existing B.
            if not index.present(track_b, end):
                continue
            if (
                config.source3_require_continuing_next_frame
                and frame + 1 < sample.frame_count
                and not index.present(track_b, frame + 1)
            ):
                continue
            predicted_b = index.predict_zyx(track_b, frame)
            match_b = associate_point_to_component(sample, frame, predicted_b, config)
            if match_b.cell_id != match_a.cell_id:
                continue
            evidence = make_pair_evidence(
                source="source3_disappearing_track",
                sample=sample,
                index=index,
                config=config,
                frame=frame,
                cell_id=match_a.cell_id,
                track_a=track_a,
                track_b=track_b,
                match_a=match_a,
                match_b=match_b,
                base_score=2.0 + (0.5 if index.stage11_unresolved_support(track_a, end) else 0.0),
                notes=(
                    "track A ends at t-1 and projects into continuing track B's component at t"
                    + ("; Stage 11 reports the ending as unresolved" if index.stage11_unresolved_support(track_a, end) else "")
                ),
            )
            if evidence is not None:
                records.append(evidence)
    return records


__all__ = ["mine_source3"]
