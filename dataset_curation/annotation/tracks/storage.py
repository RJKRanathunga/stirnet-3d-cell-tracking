from __future__ import annotations

"""Track-annotation output paths and atomic serialization."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dataset_curation.annotation.tracks.graph import (
    AnnotationError,
    Edge,
    Node,
    _canonical_edge,
)


@dataclass(frozen=True)
class OutputPaths:
    root: Path

    @property
    def state_json(self) -> Path:
        return self.root / "track_annotations.json"

    @property
    def corrected_edges_csv(self) -> Path:
        return self.root / "corrected_edges.csv"

    @property
    def overrides_csv(self) -> Path:
        return self.root / "edge_overrides.csv"

    @property
    def birth_events_csv(self) -> Path:
        return self.root / "birth_events.csv"

    @property
    def current_tracks_csv(self) -> Path:
        """Canonical annotation-owned corrected track table."""
        return self.root / "current_tracks.csv"


def _atomic_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    try:
        tmp.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(
            tmp,
            path,
        )
    finally:
        tmp.unlink(
            missing_ok=True
        )


def _atomic_csv(
    path: Path,
    frame: pd.DataFrame,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    try:
        frame.to_csv(
            tmp,
            index=False,
        )
        os.replace(
            tmp,
            path,
        )
    finally:
        tmp.unlink(
            missing_ok=True
        )


def _node_json(
    node: Node,
) -> list[int]:
    return [
        int(node[0]),
        int(node[1]),
    ]


def _edge_json(
    edge: Edge,
) -> list[list[int]]:
    return [
        _node_json(edge[0]),
        _node_json(edge[1]),
    ]


def _parse_node(
    value: Any,
) -> Node:
    if (
        not isinstance(
            value,
            (list, tuple),
        )
        or len(value) != 2
    ):
        raise AnnotationError(
            f"Invalid serialized node: {value!r}"
        )
    return (
        int(value[0]),
        int(value[1]),
    )


def _parse_edge(
    value: Any,
) -> Edge:
    if (
        not isinstance(
            value,
            (list, tuple),
        )
        or len(value) != 2
    ):
        raise AnnotationError(
            f"Invalid serialized edge: {value!r}"
        )
    return _canonical_edge(
        _parse_node(value[0]),
        _parse_node(value[1]),
    )
