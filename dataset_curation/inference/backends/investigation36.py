"""Deprecated compatibility import.

The production backend no longer calls Investigation 36 or any other file under
investigations/. Keep this module only so existing dataset_curation imports do
not break.
"""

from .stirnet_trackastra import (
    StirNetTrackastraBackend,
)


Investigation36Backend = StirNetTrackastraBackend

__all__ = [
    "Investigation36Backend",
    "StirNetTrackastraBackend",
]
