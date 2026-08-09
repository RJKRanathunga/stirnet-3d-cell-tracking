"""Validity-aware selection of buildable object-centric groups."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .models import AnnotatedVolume, InstanceGroup, TrainingSample
from .sample_builder import SampleBuildError, SampleBuilder


@dataclass(frozen=True)
class GroupRejection:
    raw_index: int
    group: InstanceGroup
    reason: str


@dataclass(frozen=True)
class ValidSampleSelection:
    valid_index: int
    raw_index: int
    group: InstanceGroup
    sample: TrainingSample
    rejections_before: tuple[GroupRejection, ...]


def select_valid_sample(
    volume: AnnotatedVolume,
    groups: Iterable[InstanceGroup],
    builder: SampleBuilder,
    *,
    valid_index: int = 0,
    max_raw_candidates: int | None = None,
) -> ValidSampleSelection:
    if valid_index < 0:
        raise ValueError("valid_index cannot be negative")
    if max_raw_candidates is not None and max_raw_candidates <= 0:
        raise ValueError("max_raw_candidates must be positive when provided")
    rejections: list[GroupRejection] = []
    valid_seen = 0
    raw_seen = 0
    for raw_index, group in enumerate(groups):
        if max_raw_candidates is not None and raw_index >= max_raw_candidates:
            break
        raw_seen = raw_index + 1
        try:
            sample = builder.build(volume, group)
        except SampleBuildError as error:
            rejections.append(GroupRejection(raw_index, group, str(error)))
            continue
        if valid_seen == valid_index:
            return ValidSampleSelection(
                valid_index, raw_index, group, sample, tuple(rejections)
            )
        valid_seen += 1
    limit_note = (
        f" within first {max_raw_candidates} raw candidates"
        if max_raw_candidates is not None else ""
    )
    raise IndexError(
        f"valid sample index {valid_index} was not found{limit_note}; "
        f"examined {raw_seen} raw candidates and found {valid_seen} valid samples"
    )


__all__ = ["GroupRejection", "ValidSampleSelection", "select_valid_sample"]
