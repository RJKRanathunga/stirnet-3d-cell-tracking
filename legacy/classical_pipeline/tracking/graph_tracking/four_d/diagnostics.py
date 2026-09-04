"""Schema-stable diagnostics helpers for the 4D tracker."""

from __future__ import annotations

import pandas as pd

from . import schemas


def table(rows: list[dict], columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def empty_window_summary() -> pd.DataFrame:
    return table([], schemas.WINDOW_SUMMARY_COLUMNS)


def empty_component_summary() -> pd.DataFrame:
    return table([], schemas.COMPONENT_SUMMARY_COLUMNS)


def empty_temporal_edges() -> pd.DataFrame:
    return table([], schemas.TEMPORAL_EDGE_COLUMNS)


def empty_assignment_changes() -> pd.DataFrame:
    return table([], schemas.ASSIGNMENT_CHANGE_COLUMNS)


def empty_boundary_events() -> pd.DataFrame:
    return table([], schemas.BOUNDARY_EVENT_COLUMNS)


def empty_solver_diagnostics() -> pd.DataFrame:
    return table([], schemas.SOLVER_DIAGNOSTIC_COLUMNS)


def empty_track_id_map() -> pd.DataFrame:
    return table([], schemas.TRACK_ID_MAP_COLUMNS)
