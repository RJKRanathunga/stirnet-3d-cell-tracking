"""Adapters for existing repository tracking/reconciliation tables."""

from .manifest import (
    TRACKLET_STRUCTURED_PRIMITIVES, TRACKLET_STRUCTURED_DIM,
    TRACKLET_RELIABILITY_FEATURES,
)
from .stage11 import (
    STAGE11_PAIR_FEATURES,
    STAGE11_PAIR_FEATURE_GROUPS,
    tensorize_stage11_pair_features,
    add_global_only_predictions,
    prediction_columns_from_stage11,
)

__all__ = [
    "TRACKLET_STRUCTURED_PRIMITIVES",
    "TRACKLET_STRUCTURED_DIM",
    "TRACKLET_RELIABILITY_FEATURES",
    "STAGE11_PAIR_FEATURES",
    "STAGE11_PAIR_FEATURE_GROUPS",
    "tensorize_stage11_pair_features",
    "add_global_only_predictions",
    "prediction_columns_from_stage11",
]
