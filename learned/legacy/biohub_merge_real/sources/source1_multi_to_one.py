"""Source 1: multiple previous track predictions converge on one component."""

from __future__ import annotations

from itertools import combinations

from ..config import MergeRealConfig
from ..evidence.spatial import associate_point_to_component
from ..observations import ObservationIndex
from ..repository_io import SampleArtifacts
from .common import make_pair_evidence


def mine_source1(
    sample: SampleArtifacts,
    index: ObservationIndex,
    config: MergeRealConfig,
):
    records = []
    for frame in range(1, sample.frame_count):
        groups: dict[int, list[tuple[int, object]]] = {}
        for row in index.tracks_in_frame(frame - 1).itertuples(index=False):
            track_id = int(row.track_id)
            predicted = index.predict_zyx(track_id, frame)
            match = associate_point_to_component(sample, frame, predicted, config)
            if match.cell_id is None:
                continue
            groups.setdefault(int(match.cell_id), []).append((track_id, match))

        for cell_id, members in groups.items():
            unique = {}
            for track_id, match in members:
                unique.setdefault(track_id, match)
            if len(unique) < 2:
                continue
            best = None
            for track_a, track_b in combinations(sorted(unique), 2):
                evidence = make_pair_evidence(
                    source="source1_multi_to_one",
                    sample=sample,
                    index=index,
                    config=config,
                    frame=frame,
                    cell_id=cell_id,
                    track_a=track_a,
                    track_b=track_b,
                    match_a=unique[track_a],
                    match_b=unique[track_b],
                    base_score=3.0,
                    notes="two previous track predictions map to the same current Stage 6 component",
                )
                if evidence is None:
                    continue
                if best is None or evidence.volume_sum_log_error < best.volume_sum_log_error:
                    best = evidence
            if best is not None:
                records.append(best)
    return records


__all__ = ["mine_source1"]
