"""Canonical foreground masking API."""

from .config import DEFAULT_MASKING_CONFIG, MaskingConfig
from .pipeline import create_binary_mask

__all__ = ["DEFAULT_MASKING_CONFIG", "MaskingConfig", "create_binary_mask"]
