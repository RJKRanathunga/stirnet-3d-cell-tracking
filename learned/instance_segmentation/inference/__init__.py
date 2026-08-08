"""Inference helpers for component normalization, voting, and inverse mapping."""

from .component_crop import InferenceROI, build_inference_roi
from .voting import vector_vote_accumulator

__all__ = ["InferenceROI", "build_inference_roi", "vector_vote_accumulator"]
