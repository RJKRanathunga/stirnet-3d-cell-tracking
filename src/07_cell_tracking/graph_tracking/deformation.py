"""Robust local translation, similarity, and affine deformation models."""

from __future__ import annotations

import numpy as np

from .config import GraphTrackingConfig
from .types import LocalTransform


def _normalized_weights(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    total = float(values.sum())
    if total <= 0 or not np.isfinite(total):
        return np.full(len(values), 1.0 / max(len(values), 1), dtype=float)
    return values / total


def fit_translation(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> LocalTransform:
    normalized = _normalized_weights(weights)
    displacement = target - source
    translation = np.sum(displacement * normalized[:, None], axis=0)
    residual = float(np.sqrt(np.sum(normalized * np.sum((source + translation - target) ** 2, axis=1))))
    return LocalTransform("translation", np.eye(3), translation, 1.0, residual)


def fit_similarity(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> LocalTransform | None:
    if len(source) < 3:
        return None
    w = _normalized_weights(weights)
    source_center = np.sum(source * w[:, None], axis=0)
    target_center = np.sum(target * w[:, None], axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = (target_centered * w[:, None]).T @ source_centered
    try:
        u, singular, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    denominator = float(np.sum(w * np.sum(source_centered**2, axis=1)))
    if denominator <= 1.0e-12:
        return None
    scale = float(np.sum(singular) / denominator)
    linear = scale * rotation
    translation = target_center - source_center @ linear.T
    predicted = source @ linear.T + translation
    residual = float(np.sqrt(np.sum(w * np.sum((predicted - target) ** 2, axis=1))))
    condition = float(singular[0] / max(singular[-1], 1.0e-12))
    return LocalTransform("similarity", linear, translation, condition, residual)


def fit_affine(source: np.ndarray, target: np.ndarray, weights: np.ndarray, maximum_condition: float) -> LocalTransform | None:
    if len(source) < 4:
        return None
    design = np.column_stack([source, np.ones(len(source))])
    w = np.sqrt(np.maximum(np.asarray(weights, dtype=float), 1.0e-12))
    weighted_design = design * w[:, None]
    weighted_target = target * w[:, None]
    try:
        condition = float(np.linalg.cond(weighted_design))
        if not np.isfinite(condition) or condition > maximum_condition:
            return None
        coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    linear = coefficients[:3, :].T
    translation = coefficients[3, :]
    predicted = source @ linear.T + translation
    normalized = _normalized_weights(weights)
    residual = float(np.sqrt(np.sum(normalized * np.sum((predicted - target) ** 2, axis=1))))
    return LocalTransform("affine", linear, translation, condition, residual)


def fit_best_local_transform(
    source_anchor_positions: np.ndarray,
    target_anchor_positions: np.ndarray,
    weights: np.ndarray,
    config: GraphTrackingConfig,
) -> LocalTransform:
    source = np.asarray(source_anchor_positions, dtype=float)
    target = np.asarray(target_anchor_positions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    candidates = [fit_translation(source, target, weights)]
    if config.enable_similarity_transform and len(source) >= config.minimum_similarity_anchors:
        model = fit_similarity(source, target, weights)
        if model is not None:
            candidates.append(model)
    if config.enable_affine_transform and len(source) >= config.minimum_affine_anchors:
        model = fit_affine(source, target, weights, config.affine_maximum_condition_number)
        if model is not None:
            candidates.append(model)
    # Avoid selecting a complex model for negligible gain.
    candidates.sort(key=lambda model: (model.residual_um + {"translation": 0.0, "similarity": 0.05, "affine": 0.10}[model.transform_type], model.transform_type))
    return candidates[0]
