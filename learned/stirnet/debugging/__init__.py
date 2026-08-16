"""Reusable trace I/O retained from the historical V1 debugging package.

The query/native-mask inspector is intentionally not exported by V2.
"""

from .core import DebugConfig, DebugTrace
from .io import load_debug_trace, save_debug_trace
from .reports import compare_traces, summarize_trace

__all__ = [
    "DebugConfig",
    "DebugTrace",
    "load_debug_trace",
    "save_debug_trace",
    "compare_traces",
    "summarize_trace",
]
