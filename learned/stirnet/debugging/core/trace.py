from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DebugTrace:
    """Serializable result of one STIR-Net inspection pass."""

    metadata: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def table(self, name: str) -> list[dict[str, Any]]:
        return self.tables.setdefault(name, [])

    def add_array(self, name: str, value: np.ndarray) -> None:
        self.arrays[name] = np.asarray(value)

    def get_array(self, name: str, default=None):
        return self.arrays.get(name, default)


@dataclass
class HookCapture:
    module_stats: list[dict[str, Any]] = field(default_factory=list)
    encoder_pyramid: Any = None
    graph_node_embeddings: Any = None
    pooled_tracklets: Any = None
    temporal_initial: Any = None
    temporal_before_cr1: Any = None
    temporal_after_cr1: Any = None
    temporal_before_cr2: Any = None
    temporal_after_cr2: Any = None
    query_initial: Any = None
    decoder_layers: list[Any] = field(default_factory=list)
