"""Minimal common diagnostic models shared by pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Provenance:
    source_type: str
    source_stage: str
    source_frame: int | None = None
    source_instance_id: int | None = None
    source_cell_id: int | None = None
    source_track_ids: tuple[int, ...] = ()
    source_candidate_id: str | int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionRecord:
    decision_type: str
    outcome: str
    frame: int | None = None
    subject_id: str | int | None = None
    reason: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None


@dataclass
class StageTrace:
    stage_name: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    intermediates: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    decisions: list[DecisionRecord] = field(default_factory=list)
    provenance: dict[str, Provenance] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class DiagnosticTrace:
    scene_id: str | None = None
    frame_range: tuple[int, int] | None = None
    stages: dict[str, StageTrace] = field(default_factory=dict)
    final_result: Any | None = None
