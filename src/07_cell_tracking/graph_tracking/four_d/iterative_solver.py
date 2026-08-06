"""Deterministic unary-flow fallback with iterative structural reweighting."""

from __future__ import annotations

from dataclasses import replace

from .config import FourDGraphConfig
from .milp_model import solve_component_milp
from .types import (
    AmbiguityComponent,
    ComponentSolution,
    ObservationStore,
    PairFactor,
    TemporalEdge,
)


def solve_component_iteratively(
    *,
    component: AmbiguityComponent,
    edges: list[TemporalEdge],
    observations: ObservationStore,
    pair_factors: list[PairFactor],
    provisional_selected_edges: set[int],
    config: FourDGraphConfig,
) -> ComponentSolution:
    adjustments: dict[int, float] = {}
    previous: tuple[int, ...] | None = None
    final: ComponentSolution | None = None
    cumulative_runtime = 0.0
    for iteration in range(1, config.iterative_solver_iterations + 1):
        solution = solve_component_milp(
            component=component,
            edges=edges,
            observations=observations,
            pair_factors=[],
            provisional_selected_edges=provisional_selected_edges,
            config=config,
            solver_type="iterative_fallback",
            edge_cost_adjustments=adjustments,
            enforce_exact_limits=False,
        )
        cumulative_runtime += solution.runtime_seconds
        if not solution.success:
            return replace(
                solution,
                iterations=iteration,
                runtime_seconds=cumulative_runtime,
            )
        final = solution
        if solution.selected_edge_indices == previous:
            return replace(
                solution,
                iterations=iteration,
                status="stable",
                runtime_seconds=cumulative_runtime,
            )
        previous = solution.selected_edge_indices
        selected = set(solution.selected_edge_indices)
        updated: dict[int, float] = {}
        for factor in pair_factors:
            if factor.first_edge in selected and factor.second_edge in selected:
                updated[factor.first_edge] = updated.get(factor.first_edge, 0.0) + factor.cost / 2.0
                updated[factor.second_edge] = updated.get(factor.second_edge, 0.0) + factor.cost / 2.0
        adjustments = {
            edge: 0.5 * adjustments.get(edge, 0.0) + 0.5 * cost
            for edge, cost in updated.items()
        }
    assert final is not None
    return replace(
        final,
        iterations=config.iterative_solver_iterations,
        status="iteration_limit",
        runtime_seconds=cumulative_runtime,
    )
