"""Sparse binary flow MILP with linearized trajectory/spatial factors."""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix

from .boundary_events import end_event_options, start_event_options
from .config import FourDGraphConfig
from .types import (
    AmbiguityComponent,
    ComponentSolution,
    ObservationStore,
    PairFactor,
    TemporalEdge,
)


def solve_component_milp(
    *,
    component: AmbiguityComponent,
    edges: list[TemporalEdge],
    observations: ObservationStore,
    pair_factors: list[PairFactor],
    provisional_selected_edges: set[int],
    config: FourDGraphConfig,
    solver_type: str = "exact_milp",
    edge_cost_adjustments: dict[int, float] | None = None,
    enforce_exact_limits: bool = True,
) -> ComponentSolution:
    started = perf_counter()
    node_set = set(int(value) for value in component.node_indices)
    edge_indices = [
        int(value) for value in component.edge_indices
        if edges[int(value)].hard_safety_valid
    ]
    edge_set = set(edge_indices)
    if enforce_exact_limits and (
        len(node_set) > config.maximum_exact_component_nodes
        or len(edge_indices) > config.maximum_exact_component_edges
        or len(pair_factors) > config.maximum_exact_pair_factors
    ):
        return ComponentSolution(
            False, (), {}, {}, math.nan, solver_type, "limits_exceeded",
            "component exceeds exact MILP limits", 0, 0, len(pair_factors),
            math.nan, 0, perf_counter() - started,
        )

    fixed_incoming: dict[int, int] = {node: 0 for node in node_set}
    fixed_outgoing: dict[int, int] = {node: 0 for node in node_set}
    for edge_index in provisional_selected_edges:
        edge = edges[edge_index]
        if edge_index in edge_set:
            continue
        if edge.target in node_set and edge.source not in node_set:
            fixed_incoming[edge.target] += 1
        if edge.source in node_set and edge.target not in node_set:
            fixed_outgoing[edge.source] += 1
    if any(value > 1 for value in fixed_incoming.values()) or any(
        value > 1 for value in fixed_outgoing.values()
    ):
        return ComponentSolution(
            False, (), {}, {}, math.nan, solver_type, "invalid_boundary_conditions",
            "multiple fixed predecessors or successors", 0, 0, 0,
            math.nan, 0, perf_counter() - started,
        )

    objectives: list[float] = []
    descriptors: list[tuple] = []
    edge_variable: dict[int, int] = {}
    adjustments = edge_cost_adjustments or {}
    for edge_index in edge_indices:
        edge_variable[edge_index] = len(objectives)
        objectives.append(
            edges[edge_index].total_cost
            + float(adjustments.get(edge_index, 0.0))
            + config.deterministic_tie_break * (edge_index + 1)
        )
        descriptors.append(("edge", edge_index))

    start_variables: dict[int, list[int]] = {}
    end_variables: dict[int, list[int]] = {}
    start_details: dict[int, tuple[str, str, float, float]] = {}
    end_details: dict[int, tuple[str, str, float, float]] = {}
    component_edges = [edges[index] for index in edge_indices]
    edges_by_source: dict[int, list[TemporalEdge]] = {node: [] for node in node_set}
    edges_by_target: dict[int, list[TemporalEdge]] = {node: [] for node in node_set}
    for edge in component_edges:
        edges_by_source[edge.source].append(edge)
        edges_by_target[edge.target].append(edge)
    first_frame = int(observations.frames.min()) if observations.node_count else 0
    last_frame = int(observations.frames.max()) if observations.node_count else 0
    for node in sorted(node_set):
        if fixed_incoming[node] == 0:
            for event_type, face, cost, support in start_event_options(
                node,
                edges=edges_by_source[node],
                observations=observations,
                config=config,
                first_frame=first_frame,
            ):
                variable = len(objectives)
                start_variables.setdefault(node, []).append(variable)
                start_details[variable] = (event_type, face, cost, support)
                objectives.append(cost + config.deterministic_tie_break * (variable + 1))
                descriptors.append(("start", node, event_type, face))
        if fixed_outgoing[node] == 0:
            for event_type, face, cost, support in end_event_options(
                node,
                edges=edges_by_target[node],
                observations=observations,
                config=config,
                last_frame=last_frame,
            ):
                variable = len(objectives)
                end_variables.setdefault(node, []).append(variable)
                end_details[variable] = (event_type, face, cost, support)
                objectives.append(cost + config.deterministic_tie_break * (variable + 1))
                descriptors.append(("end", node, event_type, face))

    factor_variables: list[tuple[int, PairFactor]] = []
    for factor in pair_factors:
        if factor.first_edge not in edge_variable or factor.second_edge not in edge_variable:
            continue
        variable = len(objectives)
        factor_variables.append((variable, factor))
        objectives.append(float(factor.cost))
        descriptors.append(("factor", factor.first_edge, factor.second_edge, factor.factor_type))

    variable_count = len(objectives)
    estimated_constraints = 2 * len(node_set) + 3 * len(factor_variables)
    if variable_count > config.maximum_variables or estimated_constraints > config.maximum_constraints:
        return ComponentSolution(
            False, (), {}, {}, math.nan, solver_type, "model_limits_exceeded",
            "MILP variable or constraint cap exceeded", variable_count,
            estimated_constraints, len(factor_variables), math.nan, 0,
            perf_counter() - started,
        )

    row_indices: list[int] = []
    column_indices: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    incoming: dict[int, list[int]] = {node: [] for node in node_set}
    outgoing: dict[int, list[int]] = {node: [] for node in node_set}
    for edge_index, variable in edge_variable.items():
        edge = edges[edge_index]
        outgoing[edge.source].append(variable)
        incoming[edge.target].append(variable)

    def add_constraint(coefficients: list[tuple[int, float]], low: float, high: float) -> None:
        row = len(lower)
        for column, value in coefficients:
            row_indices.append(row)
            column_indices.append(column)
            data.append(value)
        lower.append(low)
        upper.append(high)

    for node in sorted(node_set):
        add_constraint(
            [(value, 1.0) for value in incoming[node] + start_variables.get(node, [])],
            1.0 - fixed_incoming[node], 1.0 - fixed_incoming[node],
        )
        add_constraint(
            [(value, 1.0) for value in outgoing[node] + end_variables.get(node, [])],
            1.0 - fixed_outgoing[node], 1.0 - fixed_outgoing[node],
        )
    for variable, factor in factor_variables:
        first = edge_variable[factor.first_edge]
        second = edge_variable[factor.second_edge]
        add_constraint([(variable, 1.0), (first, -1.0)], -math.inf, 0.0)
        add_constraint([(variable, 1.0), (second, -1.0)], -math.inf, 0.0)
        add_constraint(
            [(variable, 1.0), (first, -1.0), (second, -1.0)], -1.0, math.inf
        )

    matrix = coo_matrix(
        (data, (row_indices, column_indices)),
        shape=(len(lower), variable_count),
        dtype=float,
    ).tocsr()
    try:
        if solver_type == "iterative_fallback" and not factor_variables:
            # This is a node-edge incidence flow with unit supplies. Its LP
            # relaxation is integral, and HiGHS dual simplex is substantially
            # faster than routing large unary windows through the MILP wrapper.
            result = linprog(
                c=np.asarray(objectives, dtype=float),
                A_eq=matrix,
                b_eq=np.asarray(lower, dtype=float),
                bounds=(0.0, 1.0),
                method="highs-ds",
                options={
                    "time_limit": config.solver_time_limit_seconds,
                    "presolve": True,
                },
            )
        else:
            result = milp(
                c=np.asarray(objectives, dtype=float),
                integrality=np.ones(variable_count, dtype=np.int8),
                bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
                constraints=LinearConstraint(
                    matrix, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
                ),
                options={
                    "time_limit": config.solver_time_limit_seconds,
                    "mip_rel_gap": config.solver_relative_mip_gap,
                    "presolve": True,
                },
            )
    except Exception as error:  # SciPy/HiGHS errors are isolated by component.
        return ComponentSolution(
            False, (), {}, {}, math.nan, solver_type, "solver_exception",
            f"{type(error).__name__}: {error}", variable_count, len(lower),
            len(factor_variables), math.nan, 0, perf_counter() - started,
        )
    success = bool(result.success and result.x is not None)
    if not success:
        return ComponentSolution(
            False, (), {}, {}, float(getattr(result, "fun", math.nan)), solver_type,
            f"solver_status_{result.status}", str(result.message), variable_count,
            len(lower), len(factor_variables),
            float(getattr(result, "mip_gap", math.nan)), 0,
            perf_counter() - started,
        )
    values = np.asarray(result.x)
    if not factor_variables:
        maximum_fractionality = float(np.max(np.abs(values - np.rint(values))))
        if maximum_fractionality > 1e-6:
            return ComponentSolution(
                False, (), {}, {}, float(result.fun), solver_type,
                "fractional_flow_solution",
                f"unary flow relaxation was non-integral ({maximum_fractionality:.3g})",
                variable_count, len(lower), 0, math.nan, 0,
                perf_counter() - started,
            )
    selected = tuple(sorted(
        edge_index for edge_index, variable in edge_variable.items()
        if values[variable] >= 0.5
    ))
    starts: dict[int, tuple[str, str]] = {}
    ends: dict[int, tuple[str, str]] = {}
    for node, variables in start_variables.items():
        chosen = [variable for variable in variables if values[variable] >= 0.5]
        if chosen:
            detail = start_details[chosen[0]]
            starts[node] = (detail[0], detail[1])
    for node, variables in end_variables.items():
        chosen = [variable for variable in variables if values[variable] >= 0.5]
        if chosen:
            detail = end_details[chosen[0]]
            ends[node] = (detail[0], detail[1])
    mip_gap = getattr(result, "mip_gap", 0.0)
    return ComponentSolution(
        True, selected, starts, ends, float(result.fun), solver_type, "optimal",
        str(result.message), variable_count, len(lower), len(factor_variables),
        float(0.0 if mip_gap is None else mip_gap), 1, perf_counter() - started,
    )
