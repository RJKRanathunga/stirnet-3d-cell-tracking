"""Diagnostic table utilities."""

from __future__ import annotations

import pandas as pd

from . import schemas


def table(rows: list[dict], columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def empty_candidate_evidence() -> pd.DataFrame:
    return table([], schemas.GRAPH_CANDIDATE_EVIDENCE_COLUMNS)


def empty_anchor_votes() -> pd.DataFrame:
    return table([], schemas.GRAPH_ANCHOR_VOTE_COLUMNS)


def empty_boundary_hypotheses() -> pd.DataFrame:
    return table([], schemas.GRAPH_BOUNDARY_HYPOTHESIS_COLUMNS)


def empty_refinement_events() -> pd.DataFrame:
    return table([], schemas.GRAPH_REFINEMENT_EVENT_COLUMNS)


def empty_transition_summary() -> pd.DataFrame:
    return table([], schemas.GRAPH_TRANSITION_SUMMARY_COLUMNS)
