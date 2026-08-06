"""Make the standalone graph package importable from its Stage 7 location."""

from __future__ import annotations

import sys
from pathlib import Path


GRAPH_PACKAGE_PARENT = (
    Path(__file__).resolve().parents[2] / "src" / "07_cell_tracking"
)
sys.path.insert(0, str(GRAPH_PACKAGE_PARENT))
