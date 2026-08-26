"""Exact globally constrained reconciliation decoder using SciPy MILP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from ..config import DecoderConfig


@dataclass(frozen=True)
class DecoderProblem:
    """Event utilities for one local reconciliation component.

    All high-purity tracklets are retained.  Every tracklet receives exactly
    one incoming explanation (appearance, continuation, or division-child) and
    one outgoing explanation (termination, continuation, or division-parent).
    """

    num_tracklets: int
    continuation_source: np.ndarray
    continuation_target: np.ndarray
    continuation_utility: np.ndarray
    division_parent: np.ndarray
    division_child_a: np.ndarray
    division_child_b: np.ndarray
    division_utility: np.ndarray
    appearance_utility: np.ndarray
    termination_utility: np.ndarray

    def validate(self) -> None:
        n = int(self.num_tracklets)
        if n <= 0:
            raise ValueError("num_tracklets must be positive")
        arrays = [
            self.continuation_source,
            self.continuation_target,
            self.continuation_utility,
        ]
        if len({len(np.asarray(x)) for x in arrays}) != 1:
            raise ValueError("continuation arrays must have equal length")
        arrays = [
            self.division_parent,
            self.division_child_a,
            self.division_child_b,
            self.division_utility,
        ]
        if len({len(np.asarray(x)) for x in arrays}) != 1:
            raise ValueError("division arrays must have equal length")
        if np.asarray(self.appearance_utility).shape != (n,) or np.asarray(self.termination_utility).shape != (n,):
            raise ValueError("appearance and termination utilities must be [N]")
        for values in (
            self.continuation_source,
            self.continuation_target,
            self.division_parent,
            self.division_child_a,
            self.division_child_b,
        ):
            arr = np.asarray(values, dtype=int)
            if arr.size and (arr.min() < 0 or arr.max() >= n):
                raise IndexError("decoder event references an invalid tracklet")
        if np.any(np.asarray(self.division_child_a) == np.asarray(self.division_child_b)):
            raise ValueError("division daughters must be distinct")


@dataclass(frozen=True)
class DecoderResult:
    continuation_selected: np.ndarray
    division_selected: np.ndarray
    appearance_selected: np.ndarray
    termination_selected: np.ndarray
    objective_utility: float
    status: int
    message: str


class MILPDecoder:
    """Globally choose legal continuation/division/appearance/termination events."""

    def __init__(self, config: DecoderConfig | None = None) -> None:
        self.config = config or DecoderConfig()

    def solve(self, problem: DecoderProblem) -> DecoderResult:
        problem.validate()
        n = int(problem.num_tracklets)
        ce = len(problem.continuation_utility)
        dv = len(problem.division_utility)
        # Variable blocks: continuation | division | appearance | termination.
        c0 = 0
        d0 = ce
        a0 = ce + dv
        t0 = ce + dv + n
        total = ce + dv + 2 * n

        utility = np.concatenate(
            [
                np.asarray(problem.continuation_utility, dtype=float),
                np.asarray(problem.division_utility, dtype=float),
                np.asarray(problem.appearance_utility, dtype=float),
                np.asarray(problem.termination_utility, dtype=float),
            ]
        )
        if not np.all(np.isfinite(utility)):
            raise ValueError("all decoder utilities must be finite")

        # 2N equality constraints: exactly one incoming and one outgoing event.
        matrix = lil_matrix((2 * n, total), dtype=float)
        lower = np.ones(2 * n, dtype=float)
        upper = np.ones(2 * n, dtype=float)
        for i in range(n):
            matrix[i, a0 + i] = 1.0
            matrix[n + i, t0 + i] = 1.0
        for e, (s, t) in enumerate(
            zip(np.asarray(problem.continuation_source, dtype=int), np.asarray(problem.continuation_target, dtype=int))
        ):
            matrix[t, c0 + e] = 1.0
            matrix[n + s, c0 + e] = 1.0
        for h, (p, a, b) in enumerate(
            zip(
                np.asarray(problem.division_parent, dtype=int),
                np.asarray(problem.division_child_a, dtype=int),
                np.asarray(problem.division_child_b, dtype=int),
            )
        ):
            matrix[a, d0 + h] = 1.0
            matrix[b, d0 + h] = 1.0
            matrix[n + p, d0 + h] = 1.0

        options: dict[str, float] = {}
        if self.config.milp_time_limit_seconds is not None:
            options["time_limit"] = float(self.config.milp_time_limit_seconds)
        if self.config.mip_relative_gap is not None:
            options["mip_rel_gap"] = float(self.config.mip_relative_gap)
        result = milp(
            c=-utility,  # SciPy minimizes; we maximize event utility.
            integrality=np.ones(total, dtype=int),
            bounds=Bounds(np.zeros(total), np.ones(total)),
            constraints=LinearConstraint(matrix.tocsr(), lower, upper),
            options=options or None,
        )
        if result.x is None:
            raise RuntimeError(f"reconciliation MILP failed: {result.message}")
        selected = result.x > 0.5
        return DecoderResult(
            continuation_selected=selected[c0:d0],
            division_selected=selected[d0:a0],
            appearance_selected=selected[a0:t0],
            termination_selected=selected[t0:],
            objective_utility=float(utility @ selected.astype(float)),
            status=int(result.status),
            message=str(result.message),
        )
