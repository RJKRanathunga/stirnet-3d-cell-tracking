"""Structured STIR-Net diagnostic and observability tools."""

from .core import DebugConfig, DebugTrace, StirNetInspector
from .io import load_debug_trace, save_debug_trace
from .reports import compare_traces, summarize_trace

__all__ = [
    "DebugConfig",
    "DebugTrace",
    "StirNetInspector",
    "load_debug_trace",
    "save_debug_trace",
    "compare_traces",
    "summarize_trace",
]
