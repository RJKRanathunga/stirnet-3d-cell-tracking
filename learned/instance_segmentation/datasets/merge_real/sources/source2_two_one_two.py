"""Source 2: high-confidence 2 -> 1 -> 2 temporal topology."""

from __future__ import annotations

from ..config import MergeRealConfig
from ..evidence.spatial import associate_point_to_component
from ..observations import ObservationIndex
from ..repository_io import SampleArtifacts
from .common import make_pair_evidence


def mine_source2(
    sample: SampleArtifacts,
    index: ObservationIndex,
    config: MergeRealConfig,
):
    records = []
    for track_a in index.track_ids:
        frames_a = set(index.track(track_a)["frame"].astype(int).tolist())
        for frame in range(1, sample.frame_count - 1):
            if frame - 1 not in frames_a or frame in frames_a or frame + 1 not in frames_a:
                continue
            predicted_a = index.predict_zyx(track_a, frame)
            match_a = associate_point_to_component(sample, frame, predicted_a, config)
            if match_a.cell_id is None:
                continue

            before_ids = set(index.tracks_in_frame(frame - 1)["track_id"].astype(int))
            after_ids = set(index.tracks_in_frame(frame + 1)["track_id"].astype(int))
            stable_neighbors = sorted((before_ids & after_ids) - {track_a})
            for track_b in stable_neighbors:
                predicted_b = index.predict_zyx(track_b, frame)
                match_b = associate_point_to_component(sample, frame, predicted_b, config)
                if match_b.cell_id != match_a.cell_id:
                    continue
                remap_support = index.stage11_remap_support(track_a, frame)
                evidence = make_pair_evidence(
                    source="source2_two_one_two",
                    sample=sample,
                    index=index,
                    config=config,
                    frame=frame,
                    cell_id=match_a.cell_id,
                    track_a=track_a,
                    track_b=track_b,
                    match_a=match_a,
                    match_b=match_b,
                    base_score=4.0 + (0.5 if remap_support else 0.0),
                    notes=(
                        "tracks A and B are separate at t-1/t+1 and both project to one Stage 6 component at t"
                        + ("; Stage 11 remap supports the broken identity" if remap_support else "")
                    ),
                )
                if evidence is not None:
                    records.append(evidence)
    return records


__all__ = ["mine_source2"]
