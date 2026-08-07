"""Small data models and manifest schemas used by merge_real."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SOURCE_NAMES = (
    "source1_multi_to_one",
    "source2_two_one_two",
    "source3_disappearing_track",
    "source4_component_anomaly",
)

FILTER_LABELS = (
    "confirmed_2_cell_merge",
    "confirmed_3plus_cell_merge",
    "not_merge",
    "other_segmentation_error",
    "division_or_birth",
    "ambiguous",
    "skip",
)

PARTITION_LABELS = (
    "accepted",
    "accepted_corrected",
    "rejected",
    "ambiguous",
)


@dataclass(frozen=True)
class CandidateEvidence:
    sample_id: str
    frame: int
    cell_id: int
    source: str
    track_a: int | None = None
    track_b: int | None = None
    involved_track_ids: tuple[int, ...] = ()
    candidate_volume: float = float("nan")
    volume_a_reference: float = float("nan")
    volume_b_reference: float = float("nan")
    volume_sum_ratio: float = float("nan")
    volume_sum_log_error: float = float("nan")
    prediction_a_distance_um: float = float("nan")
    prediction_b_distance_um: float = float("nan")
    prediction_a_inside: bool = False
    prediction_b_inside: bool = False
    edt_peak_count: int = 0
    source_score: float = 0.0
    notes: str = ""

    @property
    def candidate_id(self) -> str:
        return candidate_id(self.sample_id, self.frame, self.cell_id)

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["candidate_id"] = self.candidate_id
        record["involved_track_ids"] = ";".join(str(v) for v in self.involved_track_ids)
        return record


def candidate_id(sample_id: str, frame: int, cell_id: int) -> str:
    return f"{sample_id}_t{int(frame):03d}_c{int(cell_id):05d}"


def parse_track_ids(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ()
    result: list[int] = []
    for token in text.replace(",", ";").split(";"):
        token = token.strip()
        if token:
            try:
                result.append(int(float(token)))
            except ValueError:
                continue
    return tuple(sorted(set(result)))


__all__ = [
    "CandidateEvidence",
    "FILTER_LABELS",
    "PARTITION_LABELS",
    "SOURCE_NAMES",
    "candidate_id",
    "parse_track_ids",
]
