"""Ambiguity seed classification without blind high-confidence anchors."""

from __future__ import annotations

import math

import numpy as np

from .config import FourDGraphConfig
from .types import ObservationStore, TemporalEdge


def ambiguity_seeds(
    *,
    observations: ObservationStore,
    edges: list[TemporalEdge],
    config: FourDGraphConfig,
) -> set[int]:
    seeds: set[int] = set()
    outgoing: dict[int, list[TemporalEdge]] = {}
    incoming: dict[int, list[TemporalEdge]] = {}
    for edge in edges:
        outgoing.setdefault(edge.source, []).append(edge)
        incoming.setdefault(edge.target, []).append(edge)
        if edge.frame_gap > 1:
            seeds.update((edge.source, edge.target))
        if edge.provisional_selected:
            if (
                math.isfinite(edge.provisional_probability)
                and edge.provisional_probability < config.ambiguous_probability_threshold
            ) or (
                math.isfinite(edge.provisional_margin)
                and edge.provisional_margin < config.ambiguous_margin_threshold
            ):
                seeds.update((edge.source, edge.target))
            if min(
                edge.global_motion_residual_um,
                edge.relative_motion_residual_um,
            ) > config.graph_inconsistency_threshold:
                # Confidence alone never locks a severe structural contradiction.
                seeds.update((edge.source, edge.target))

    for node in range(observations.node_count):
        node_out = sorted(outgoing.get(node, []), key=lambda edge: (edge.unary_cost, edge.target))
        node_in = sorted(incoming.get(node, []), key=lambda edge: (edge.unary_cost, edge.source))
        selected_outgoing = next(
            (edge for edge in node_out if edge.provisional_selected), None
        )
        selected_incoming = next(
            (edge for edge in node_in if edge.provisional_selected), None
        )
        if selected_outgoing is not None:
            alternatives = [edge for edge in node_out if not edge.provisional_selected]
            if alternatives:
                alternative = alternatives[0]
                selected_is_uncertain = (
                    not math.isfinite(selected_outgoing.provisional_margin)
                    or selected_outgoing.provisional_margin
                    < 2.0 * config.ambiguous_margin_threshold
                )
                if (
                    alternative.unary_cost
                    <= selected_outgoing.unary_cost + config.ambiguous_cost_delta
                    and selected_is_uncertain
                ):
                    seeds.update((node, selected_outgoing.target, alternative.target))
        if selected_incoming is not None:
            alternatives = [edge for edge in node_in if not edge.provisional_selected]
            if alternatives:
                alternative = alternatives[0]
                selected_is_uncertain = (
                    not math.isfinite(selected_incoming.provisional_margin)
                    or selected_incoming.provisional_margin
                    < 2.0 * config.ambiguous_margin_threshold
                )
                if (
                    alternative.unary_cost
                    <= selected_incoming.unary_cost + config.ambiguous_cost_delta
                    and selected_is_uncertain
                ):
                    seeds.update((node, selected_incoming.source, alternative.source))
        reference = (
            selected_outgoing.unary_cost
            if selected_outgoing is not None else config.generic_death_cost
        )
        for edge in node_out:
            if (
                edge.graph_expanded
                and edge.unary_cost <= reference + config.ambiguous_cost_delta
            ):
                seeds.update((edge.source, edge.target))

    selected_in = {edge.target for edge in edges if edge.provisional_selected}
    selected_out = {edge.source for edge in edges if edge.provisional_selected}
    selected_predecessor = {
        edge.target: edge for edge in edges if edge.provisional_selected
    }
    selected_successor = {
        edge.source: edge for edge in edges if edge.provisional_selected
    }
    for middle in sorted(set(selected_predecessor) & set(selected_successor)):
        first = selected_predecessor[middle]
        second = selected_successor[middle]
        velocity_first = (
            observations.positions_zyx_um[middle]
            - observations.positions_zyx_um[first.source]
        ) / first.frame_gap
        velocity_second = (
            observations.positions_zyx_um[second.target]
            - observations.positions_zyx_um[middle]
        ) / second.frame_gap
        if float(np.linalg.norm(velocity_second - velocity_first)) > (
            config.graph_inconsistency_threshold
        ):
            seeds.update((first.source, middle, second.target))
    first_frame = int(observations.frames.min()) if observations.node_count else 0
    last_frame = int(observations.frames.max()) if observations.node_count else 0
    for node in range(observations.node_count):
        frame = int(observations.frames[node])
        if frame > first_frame and node not in selected_in:
            seeds.add(node)
        if frame < last_frame and node not in selected_out:
            seeds.add(node)
    return seeds
