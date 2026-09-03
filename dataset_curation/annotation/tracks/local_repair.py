from __future__ import annotations

# DATASET_CURATION_LOCAL_TRACK_REPAIR_V2
#
# Conservative annotation-time track repair after a spatial split.
#
# Reused design ideas:
#   Stage 7  -> physical motion prediction + Hungarian one-to-one assignment
#   Stage 8  -> local gap closing around broken fragments
#   Stage 11 -> hard gates + conservative ambiguity margins
#
# This module intentionally does not import numbered pipeline stages. Those
# modules depend on stage-specific DataFrame schemas and global pipeline state,
# while annotation repair has a much smaller graph/node contract.

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from dataset_curation.annotation.tracks.graph import (
    Edge,
    Node,
    _canonical_edge,
)
from dataset_curation.annotation.tracks.session import TrackAnnotationSession


@dataclass(frozen=True)
class LocalTrackRepairConfig:
    maximum_gap_frames: int = 2
    motion_history_nodes: int = 4

    candidate_radius_base_um: float = 12.0
    candidate_radius_per_additional_gap_um: float = 4.0
    maximum_candidate_radius_um: float = 16.0

    hard_maximum_volume_ratio: float = 3.0
    conservative_maximum_volume_ratio: float = 2.25

    distance_weight: float = 0.78
    volume_weight: float = 0.17
    temporal_gap_weight: float = 0.05

    unmatched_cost: float = 0.90
    invalid_assignment_cost: float = 1.0e6

    maximum_accepted_cost: float = 0.72
    maximum_accepted_distance_fraction: float = 0.72
    minimum_assignment_margin: float = 0.12

    def validate(self) -> None:
        if self.maximum_gap_frames < 1:
            raise ValueError("maximum_gap_frames must be >= 1")
        if self.motion_history_nodes < 1:
            raise ValueError("motion_history_nodes must be >= 1")
        if self.candidate_radius_base_um <= 0:
            raise ValueError("candidate_radius_base_um must be > 0")
        if (
            self.maximum_candidate_radius_um
            < self.candidate_radius_base_um
        ):
            raise ValueError(
                "maximum_candidate_radius_um cannot be below the base radius"
            )
        if self.hard_maximum_volume_ratio < 1:
            raise ValueError("hard_maximum_volume_ratio must be >= 1")
        if (
            self.conservative_maximum_volume_ratio
            > self.hard_maximum_volume_ratio
        ):
            raise ValueError(
                "conservative_maximum_volume_ratio cannot exceed the hard gate"
            )
        if self.unmatched_cost <= 0:
            raise ValueError("unmatched_cost must be > 0")
        if self.invalid_assignment_cost <= self.unmatched_cost:
            raise ValueError(
                "invalid_assignment_cost must exceed unmatched_cost"
            )


@dataclass(frozen=True)
class RepairProposal:
    source: Node
    target: Node
    side: str
    frame_gap: int
    distance_um: float
    candidate_radius_um: float
    volume_ratio: float
    cost: float
    source_margin: float
    target_margin: float

    @property
    def edge(self) -> Edge:
        return _canonical_edge(self.source, self.target)

    def audit_record(self) -> dict[str, object]:
        return {
            "side": str(self.side),
            "source": [int(self.source[0]), int(self.source[1])],
            "target": [int(self.target[0]), int(self.target[1])],
            "frame_gap": int(self.frame_gap),
            "distance_um": float(self.distance_um),
            "candidate_radius_um": float(self.candidate_radius_um),
            "volume_ratio": float(self.volume_ratio),
            "cost": float(self.cost),
            "source_margin": (
                None
                if not math.isfinite(self.source_margin)
                else float(self.source_margin)
            ),
            "target_margin": (
                None
                if not math.isfinite(self.target_margin)
                else float(self.target_margin)
            ),
        }


@dataclass(frozen=True)
class SideRepairSummary:
    side: str
    source_count: int
    target_count: int
    candidate_count: int
    accepted: tuple[RepairProposal, ...]
    ambiguous_assignments: int
    unmatched_sources: int

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    def audit_record(self) -> dict[str, object]:
        return {
            "side": str(self.side),
            "source_count": int(self.source_count),
            "target_count": int(self.target_count),
            "candidate_count": int(self.candidate_count),
            "accepted_count": int(self.accepted_count),
            "ambiguous_assignments": int(self.ambiguous_assignments),
            "unmatched_sources": int(self.unmatched_sources),
            "accepted": [
                item.audit_record()
                for item in self.accepted
            ],
        }


@dataclass(frozen=True)
class LocalTrackRepairResult:
    frame: int
    incoming: SideRepairSummary
    outgoing: SideRepairSummary
    applied_edges: tuple[Edge, ...]

    @property
    def applied_count(self) -> int:
        return len(self.applied_edges)

    @property
    def ambiguous_count(self) -> int:
        return (
            int(self.incoming.ambiguous_assignments)
            + int(self.outgoing.ambiguous_assignments)
        )

    def summary_text(self) -> str:
        return (
            f"{self.applied_count} continuation edge(s) applied "
            f"(incoming={self.incoming.accepted_count}, "
            f"outgoing={self.outgoing.accepted_count}); "
            f"{self.ambiguous_count} ambiguous assignment(s) left manual"
        )


def _active_edges_without_global_analysis(
    session: TrackAnnotationSession,
) -> set[Edge]:
    # Do not call session.active_edges here. That property also rebuilds
    # components and endpoint diagnostics, which the viewer deliberately moved
    # to its serialized background refresh worker.
    candidates = (
        set(session.base_edges)
        | set(session.forced_edges)
    ) - set(session.broken_edges)

    valid = session.valid_nodes
    return {
        edge
        for edge in candidates
        if edge[0] in valid and edge[1] in valid
    }


def _adjacency(
    edges: Iterable[Edge],
) -> tuple[dict[Node, list[Node]], dict[Node, list[Node]]]:
    predecessors: dict[Node, list[Node]] = {}
    successors: dict[Node, list[Node]] = {}

    for left, right in edges:
        predecessors.setdefault(right, []).append(left)
        successors.setdefault(left, []).append(right)

    for mapping in (predecessors, successors):
        for node in mapping:
            mapping[node].sort()

    return predecessors, successors


def _trace_one_to_one(
    node: Node,
    *,
    direction: str,
    predecessors: dict[Node, list[Node]],
    successors: dict[Node, list[Node]],
    birth_edges: set[Edge],
    limit: int,
) -> list[Node]:
    result = [node]
    current = node

    while len(result) < int(limit):
        if direction == "backward":
            candidates = predecessors.get(current, [])
            if len(candidates) != 1:
                break
            neighbour = candidates[0]
            edge = _canonical_edge(neighbour, current)
        elif direction == "forward":
            candidates = successors.get(current, [])
            if len(candidates) != 1:
                break
            neighbour = candidates[0]
            edge = _canonical_edge(current, neighbour)
        else:
            raise ValueError(direction)

        # Never learn ordinary continuation motion through a lineage branch.
        if edge in birth_edges:
            break

        result.append(neighbour)
        current = neighbour

    return sorted(
        result,
        key=lambda value: (int(value[0]), int(value[1])),
    )


def _physical_center(
    node: Node,
    *,
    centers: dict[Node, np.ndarray],
    spacing_zyx_um: np.ndarray,
) -> np.ndarray | None:
    value = centers.get(node)
    if value is None:
        return None

    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return None

    return point * spacing_zyx_um


def _robust_velocity_um_per_frame(
    nodes: list[Node],
    *,
    centers: dict[Node, np.ndarray],
    spacing_zyx_um: np.ndarray,
) -> np.ndarray:
    observations: list[tuple[int, np.ndarray]] = []

    for node in nodes:
        point = _physical_center(
            node,
            centers=centers,
            spacing_zyx_um=spacing_zyx_um,
        )
        if point is not None:
            observations.append((int(node[0]), point))

    observations.sort(key=lambda value: value[0])

    velocities: list[np.ndarray] = []
    for (frame_a, point_a), (frame_b, point_b) in zip(
        observations[:-1],
        observations[1:],
    ):
        dt = int(frame_b) - int(frame_a)
        if dt <= 0:
            continue
        velocities.append(
            (point_b - point_a) / float(dt)
        )

    if not velocities:
        return np.zeros((3,), dtype=np.float64)

    return np.median(
        np.asarray(velocities, dtype=np.float64),
        axis=0,
    )


class _VolumeCache:
    def __init__(
        self,
        labels_for_frame: Callable[[int], np.ndarray],
    ) -> None:
        self._labels_for_frame = labels_for_frame
        self._counts: dict[int, np.ndarray] = {}

    def _frame_counts(self, frame: int) -> np.ndarray:
        frame = int(frame)
        cached = self._counts.get(frame)
        if cached is not None:
            return cached

        labels = np.asarray(
            self._labels_for_frame(frame)
        )
        if labels.ndim != 3:
            raise ValueError(
                f"Expected a 3-D corrected label frame at t={frame}; "
                f"got {labels.shape}."
            )
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(
                f"Corrected labels at t={frame} must be integer-valued."
            )

        counts = np.bincount(
            labels.reshape(-1).astype(np.int64, copy=False)
        )
        self._counts[frame] = counts
        return counts

    def node_volume(self, node: Node) -> float:
        counts = self._frame_counts(int(node[0]))
        instance_id = int(node[1])

        if instance_id <= 0 or instance_id >= counts.size:
            return math.nan

        value = int(counts[instance_id])
        return float(value) if value > 0 else math.nan

    def reference_volume(
        self,
        nodes: Iterable[Node],
    ) -> float:
        finite = [
            value
            for value in (
                self.node_volume(node)
                for node in nodes
            )
            if math.isfinite(value) and value > 0
        ]
        if not finite:
            return math.nan

        return float(
            np.median(
                np.asarray(finite, dtype=np.float64)
            )
        )


def _volume_ratio(a: float, b: float) -> float:
    if (
        not math.isfinite(a)
        or not math.isfinite(b)
        or a <= 0
        or b <= 0
    ):
        # Missing volume evidence is neutral rather than a reason to invent a
        # hard rejection. Distance/motion still must pass all conservative gates.
        return 1.0

    return float(max(a, b) / min(a, b))


def _candidate_radius(
    frame_gap: int,
    config: LocalTrackRepairConfig,
) -> float:
    return float(
        min(
            config.maximum_candidate_radius_um,
            config.candidate_radius_base_um
            + config.candidate_radius_per_additional_gap_um
            * max(int(frame_gap) - 1, 0),
        )
    )


def _candidate_cost(
    *,
    distance_um: float,
    radius_um: float,
    volume_ratio: float,
    frame_gap: int,
    config: LocalTrackRepairConfig,
) -> float:
    distance_term = float(distance_um) / max(float(radius_um), 1.0e-9)

    volume_term = (
        0.0
        if volume_ratio <= 1.0
        else math.log(float(volume_ratio)) / math.log(2.0)
    )

    gap_term = float(max(int(frame_gap) - 1, 0))

    return float(
        config.distance_weight * distance_term
        + config.volume_weight * volume_term
        + config.temporal_gap_weight * gap_term
    )


def _margin(
    matrix: np.ndarray,
    *,
    row: int,
    column: int,
    axis: str,
) -> float:
    if axis == "row":
        alternatives = np.delete(
            np.asarray(matrix[row, :], dtype=np.float64),
            column,
        )
    elif axis == "column":
        alternatives = np.delete(
            np.asarray(matrix[:, column], dtype=np.float64),
            row,
        )
    else:
        raise ValueError(axis)

    alternatives = alternatives[np.isfinite(alternatives)]
    if alternatives.size == 0:
        return math.inf

    return float(
        np.min(alternatives)
        - float(matrix[row, column])
    )


def _solve_side(
    *,
    side: str,
    sources: list[Node],
    targets: list[Node],
    split_frame: int,
    centers: dict[Node, np.ndarray],
    predecessors: dict[Node, list[Node]],
    successors: dict[Node, list[Node]],
    birth_edges: set[Edge],
    volumes: _VolumeCache,
    spacing_zyx_um: np.ndarray,
    config: LocalTrackRepairConfig,
) -> SideRepairSummary:
    if not sources or not targets:
        return SideRepairSummary(
            side=side,
            source_count=len(sources),
            target_count=len(targets),
            candidate_count=0,
            accepted=(),
            ambiguous_assignments=0,
            unmatched_sources=len(sources),
        )

    pair_cost = np.full(
        (len(sources), len(targets)),
        np.inf,
        dtype=np.float64,
    )
    pair_metrics: dict[
        tuple[int, int],
        tuple[int, float, float, float],
    ] = {}

    for source_index, source in enumerate(sources):
        if side == "incoming":
            chain = _trace_one_to_one(
                source,
                direction="backward",
                predecessors=predecessors,
                successors=successors,
                birth_edges=birth_edges,
                limit=config.motion_history_nodes,
            )
            source_center = _physical_center(
                source,
                centers=centers,
                spacing_zyx_um=spacing_zyx_um,
            )
            if source_center is None:
                continue

            velocity = _robust_velocity_um_per_frame(
                chain,
                centers=centers,
                spacing_zyx_um=spacing_zyx_um,
            )
            frame_gap = int(split_frame) - int(source[0])
            predicted = source_center + velocity * float(frame_gap)
            reference_volume = volumes.reference_volume(chain)

        elif side == "outgoing":
            chain = _trace_one_to_one(
                source,
                direction="forward",
                predecessors=predecessors,
                successors=successors,
                birth_edges=birth_edges,
                limit=config.motion_history_nodes,
            )
            source_center = _physical_center(
                source,
                centers=centers,
                spacing_zyx_um=spacing_zyx_um,
            )
            if source_center is None:
                continue

            velocity = _robust_velocity_um_per_frame(
                chain,
                centers=centers,
                spacing_zyx_um=spacing_zyx_um,
            )
            frame_gap = int(source[0]) - int(split_frame)
            predicted = source_center - velocity * float(frame_gap)
            reference_volume = volumes.reference_volume(chain)

        else:
            raise ValueError(side)

        if frame_gap < 1 or frame_gap > config.maximum_gap_frames:
            continue

        radius_um = _candidate_radius(frame_gap, config)

        for target_index, target in enumerate(targets):
            target_center = _physical_center(
                target,
                centers=centers,
                spacing_zyx_um=spacing_zyx_um,
            )
            if target_center is None:
                continue

            distance_um = float(
                np.linalg.norm(predicted - target_center)
            )
            if distance_um > radius_um:
                continue

            target_volume = volumes.node_volume(target)
            volume_ratio = _volume_ratio(
                reference_volume,
                target_volume,
            )
            if volume_ratio > config.hard_maximum_volume_ratio:
                continue

            cost = _candidate_cost(
                distance_um=distance_um,
                radius_um=radius_um,
                volume_ratio=volume_ratio,
                frame_gap=frame_gap,
                config=config,
            )
            pair_cost[source_index, target_index] = cost
            pair_metrics[(source_index, target_index)] = (
                int(frame_gap),
                float(distance_um),
                float(radius_um),
                float(volume_ratio),
            )

    candidate_count = int(np.isfinite(pair_cost).sum())

    # Augment only with per-source unmatched columns. Targets can naturally be
    # left unused, while every source can decline a bad real match.
    augmented = np.full(
        (
            len(sources),
            len(targets) + len(sources),
        ),
        float(config.invalid_assignment_cost),
        dtype=np.float64,
    )
    augmented[:, : len(targets)] = np.where(
        np.isfinite(pair_cost),
        pair_cost,
        float(config.invalid_assignment_cost),
    )
    for source_index in range(len(sources)):
        augmented[
            source_index,
            len(targets) + source_index,
        ] = float(config.unmatched_cost)

    rows, columns = linear_sum_assignment(augmented)

    accepted: list[RepairProposal] = []
    ambiguous = 0
    unmatched = 0

    for row, column in zip(rows.tolist(), columns.tolist()):
        if column >= len(targets):
            unmatched += 1
            continue

        if not np.isfinite(pair_cost[row, column]):
            unmatched += 1
            continue

        (
            frame_gap,
            distance_um,
            radius_um,
            volume_ratio,
        ) = pair_metrics[(row, column)]
        cost = float(pair_cost[row, column])

        source_margin = _margin(
            pair_cost,
            row=row,
            column=column,
            axis="row",
        )
        target_margin = _margin(
            pair_cost,
            row=row,
            column=column,
            axis="column",
        )

        conservative = (
            cost <= config.maximum_accepted_cost
            and distance_um
            <= radius_um * config.maximum_accepted_distance_fraction
            and volume_ratio
            <= config.conservative_maximum_volume_ratio
            and source_margin >= config.minimum_assignment_margin
            and target_margin >= config.minimum_assignment_margin
        )

        if not conservative:
            ambiguous += 1
            continue

        source = sources[row]
        target = targets[column]

        if side == "incoming":
            left, right = source, target
        else:
            left, right = target, source

        accepted.append(
            RepairProposal(
                source=left,
                target=right,
                side=side,
                frame_gap=int(frame_gap),
                distance_um=float(distance_um),
                candidate_radius_um=float(radius_um),
                volume_ratio=float(volume_ratio),
                cost=float(cost),
                source_margin=float(source_margin),
                target_margin=float(target_margin),
            )
        )

    return SideRepairSummary(
        side=side,
        source_count=len(sources),
        target_count=len(targets),
        candidate_count=candidate_count,
        accepted=tuple(accepted),
        ambiguous_assignments=int(ambiguous),
        unmatched_sources=int(unmatched),
    )


def repair_split_tracks(
    *,
    frame: int,
    new_instance_ids: Iterable[int],
    track_session: TrackAnnotationSession,
    track_centers: dict[Node, np.ndarray],
    labels_for_frame: Callable[[int], np.ndarray],
    spacing_zyx_um: tuple[float, float, float],
    config: LocalTrackRepairConfig | None = None,
) -> LocalTrackRepairResult:
    config = (
        LocalTrackRepairConfig()
        if config is None
        else config
    )
    config.validate()

    frame = int(frame)
    spacing = np.asarray(
        spacing_zyx_um,
        dtype=np.float64,
    )
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError(
            "spacing_zyx_um must contain three positive values"
        )

    # The targets are ONLY the fresh detections created by this split. This is
    # the strongest spatial-edit prior and prevents unrelated nearby cells from
    # becoming targets.
    targets = sorted(
        {
            (frame, int(instance_id))
            for instance_id in new_instance_ids
            if int(instance_id) > 0
            and (frame, int(instance_id))
            in track_session.valid_nodes
        }
    )

    active_edges = _active_edges_without_global_analysis(
        track_session
    )
    predecessors, successors = _adjacency(active_edges)

    birth_edges = set(track_session.birth_edges)
    birth_nodes = {
        node
        for edge in birth_edges
        for node in edge
    }

    ignored_starts = set(track_session.ignored_start_nodes)
    ignored_ends = set(track_session.ignored_end_nodes)

    # Endpoints only. Candidate generation then applies the much stronger
    # physical motion/volume gates against the new split detections.
    incoming_sources = sorted(
        node
        for node in track_session.valid_nodes
        if (
            frame - config.maximum_gap_frames
            <= int(node[0])
            < frame
            and node not in successors
            and node not in track_session.boundary_exit_nodes
            and node not in ignored_ends
            and node not in birth_nodes
        )
    )

    outgoing_sources = sorted(
        node
        for node in track_session.valid_nodes
        if (
            frame
            < int(node[0])
            <= frame + config.maximum_gap_frames
            and node not in predecessors
            and node not in track_session.boundary_entry_nodes
            and node not in ignored_starts
            and node not in birth_nodes
        )
    )

    volumes = _VolumeCache(labels_for_frame)

    incoming = _solve_side(
        side="incoming",
        sources=incoming_sources,
        targets=targets,
        split_frame=frame,
        centers=track_centers,
        predecessors=predecessors,
        successors=successors,
        birth_edges=birth_edges,
        volumes=volumes,
        spacing_zyx_um=spacing,
        config=config,
    )

    outgoing = _solve_side(
        side="outgoing",
        sources=outgoing_sources,
        targets=targets,
        split_frame=frame,
        centers=track_centers,
        predecessors=predecessors,
        successors=successors,
        birth_edges=birth_edges,
        volumes=volumes,
        spacing_zyx_um=spacing,
        config=config,
    )

    proposed_edges = tuple(
        proposal.edge
        for proposal in (
            *incoming.accepted,
            *outgoing.accepted,
        )
    )

    details = {
        "algorithm": "local_hungarian_conservative_v2",
        "split_frame": int(frame),
        "new_instance_ids": [
            int(node[1])
            for node in targets
        ],
        "incoming": incoming.audit_record(),
        "outgoing": outgoing.audit_record(),
    }

    applied = track_session.apply_auto_repair(
        proposed_edges,
        focus_frame=frame,
        details=details,
    )

    result = LocalTrackRepairResult(
        frame=frame,
        incoming=incoming,
        outgoing=outgoing,
        applied_edges=tuple(applied),
    )

    print(
        f"[track repair] t={frame}: {result.summary_text()}",
        flush=True,
    )
    return result
