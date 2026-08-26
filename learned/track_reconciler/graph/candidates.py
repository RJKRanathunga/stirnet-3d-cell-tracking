"""Deterministic high-recall candidate gating.

This is deliberately not learned.  Like supervoxel construction in STIR-Net,
the gate reduces the search space so the learned model only compares plausible
biological explanations.
"""

from __future__ import annotations

import numpy as np

from ..config import CandidateConfig


def gate_candidate_pairs(
    source_ids: np.ndarray,
    source_end_frame: np.ndarray,
    source_end_xyz_um: np.ndarray,
    target_ids: np.ndarray,
    target_start_frame: np.ndarray,
    target_start_xyz_um: np.ndarray,
    *,
    expected_global: dict[tuple[int, int], np.ndarray] | None = None,
    expected_global_relative: dict[tuple[int, int], np.ndarray] | None = None,
    expected_local: dict[tuple[int, int], np.ndarray] | None = None,
    extra_pairs: set[tuple[int, int]] | None = None,
    config: CandidateConfig | None = None,
) -> list[tuple[int, int]]:
    """Union proximity around endpoint/prediction anchors and external candidates.

    Prediction dictionaries are keyed by `(source_id, target_frame)` so the
    expected position can depend on the actual temporal gap.
    """

    cfg = config or CandidateConfig()
    source_ids = np.asarray(source_ids)
    target_ids = np.asarray(target_ids)
    pairs: set[tuple[int, int]] = set(extra_pairs or ())
    for si, source_id in enumerate(source_ids.tolist()):
        ranked: list[tuple[float, int]] = []
        sf = int(source_end_frame[si])
        base = np.asarray(source_end_xyz_um[si], dtype=float)
        for ti, target_id in enumerate(target_ids.tolist()):
            tf = int(target_start_frame[ti])
            gap = tf - sf
            if gap < 1 or gap > cfg.maximum_gap_frames:
                continue
            radius = min(
                cfg.radius_max_um,
                cfg.radius_base_um + cfg.radius_per_extra_gap_um * (gap - 1),
            )
            target = np.asarray(target_start_xyz_um[ti], dtype=float)
            anchors = [base]
            key = (int(source_id), tf)
            for mapping in (expected_global, expected_global_relative, expected_local):
                if mapping is not None and key in mapping:
                    value = np.asarray(mapping[key], dtype=float)
                    if value.shape == (3,) and np.all(np.isfinite(value)):
                        anchors.append(value)
            distance = min(float(np.linalg.norm(target - anchor)) for anchor in anchors)
            if distance <= radius:
                ranked.append((distance, int(target_id)))
        ranked.sort()
        for _, target_id in ranked[: cfg.maximum_targets_per_source]:
            pairs.add((int(source_id), target_id))
    return sorted(pairs)
