"""Mapping from current `src/11_track_reconciliation` candidates to model evidence.

The default manifest intentionally uses primitive measurements, availability,
counts/ranks and reliability—not the current hand-weighted continuation score.
This lets the network learn evidence weighting rather than merely imitate Stage 11.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import Tensor


STAGE11_PAIR_FEATURES: tuple[str, ...] = (
    # temporal / raw geometry
    "gap_frames",
    "direct_endpoint_distance_um",
    "hard_search_radius_um",
    # forward expert
    "forward_error_um",
    "forward_history_count",
    "forward_used_global_motion",
    "forward_used_relative_velocity",
    "forward_uncertainty_um",
    # backward expert
    "backward_error_um",
    "backward_history_count",
    "backward_prediction_available",
    "bidirectional_disagreement_um",
    # local-neighbour expert
    "anchor_count",
    "anchor_prediction_error_um",
    "neighborhood_distance_error_um",
    "local_survival_ratio",
    # morphology / appearance
    "volume_log_error",
    "shape_error",
    "intensity_error",
    "intensity_mean_error",
    "intensity_median_error",
    "intensity_std_error",
    "intensity_iqr_error",
    "intensity_cv_error",
    "target_real_observation_count",
    # legacy Stage-7 / Trackastra-like association evidence
    "stage7_candidate_available",
    "stage7_candidate_distance_um",
    "stage7_candidate_pair_cost",
    "stage7_candidate_probability",
    "stage7_candidate_rank",
    # ambiguity / competition
    "candidate_quality_score",
    "source_candidate_count",
    "target_predecessor_count",
    "source_rank",
    "target_rank",
    "source_score_margin",
    "target_score_margin",
    "mutual_best",
    # size reliability / provenance
    "effective_pair_volume",
    "small_cell_history_exception",
)

# Group IDs are useful for optional feature-group dropout in a trainer.
STAGE11_PAIR_FEATURE_GROUPS: tuple[int, ...] = (
    *([0] * 3),   # geometry/time
    *([1] * 5),   # forward
    *([2] * 4),   # backward
    *([3] * 4),   # local
    *([4] * 9),   # morphology/appearance
    *([5] * 5),   # previous association model
    *([6] * 8),   # ambiguity
    *([7] * 2),   # reliability/provenance
)
assert len(STAGE11_PAIR_FEATURES) == 40
assert len(STAGE11_PAIR_FEATURE_GROUPS) == 40

_BOOLEAN = {
    "forward_used_global_motion",
    "forward_used_relative_velocity",
    "backward_prediction_available",
    "stage7_candidate_available",
    "mutual_best",
    "small_cell_history_exception",
}
_COUNT = {
    "forward_history_count",
    "backward_history_count",
    "anchor_count",
    "target_real_observation_count",
    "source_candidate_count",
    "target_predecessor_count",
    "source_rank",
    "target_rank",
    "stage7_candidate_rank",
}
_POSITIVE = {
    "direct_endpoint_distance_um",
    "hard_search_radius_um",
    "forward_error_um",
    "forward_uncertainty_um",
    "backward_error_um",
    "bidirectional_disagreement_um",
    "anchor_prediction_error_um",
    "neighborhood_distance_error_um",
    "stage7_candidate_distance_um",
    "stage7_candidate_pair_cost",
    "effective_pair_volume",
}


def _numeric(series: pd.Series, name: str) -> np.ndarray:
    if name in _BOOLEAN:
        return series.fillna(False).astype(bool).to_numpy(dtype=np.float32)
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    finite = np.isfinite(values)
    if name in _COUNT or name in _POSITIVE:
        values[finite] = np.log1p(np.maximum(values[finite], 0.0))
    values[~finite] = 0.0
    return values.astype(np.float32)


def tensorize_stage11_pair_features(
    candidates: pd.DataFrame,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Return `[E,80]` model features and `[E,40]` availability mask.

    The first 40 channels are primitive values and the next 40 are their
    availability flags.  This matters because a missing non-negative error
    cannot safely be encoded as numeric zero (zero also means a perfect match).
    Continuous normalization statistics must be fitted on the training split.
    """

    columns = []
    available = []
    for name in STAGE11_PAIR_FEATURES:
        if name in candidates:
            raw = candidates[name]
            avail = ~pd.isna(raw)
            values = _numeric(raw, name)
        else:
            avail = pd.Series(False, index=candidates.index)
            values = np.zeros(len(candidates), dtype=np.float32)
        columns.append(values)
        available.append(avail.to_numpy(dtype=np.float32))
    matrix = np.stack(columns, axis=-1) if len(candidates) else np.zeros((0, 40), np.float32)
    validity = np.stack(available, axis=-1) if len(candidates) else np.zeros((0, 40), np.float32)
    if mean is not None or std is not None:
        if mean is None or std is None:
            raise ValueError("mean and std must be supplied together")
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        if mean.shape != (40,) or std.shape != (40,):
            raise ValueError("Stage-11 normalization stats must have shape [40]")
        matrix = (matrix - mean) / np.maximum(std, 1e-6)
        matrix *= validity
    combined = np.concatenate((matrix, validity), axis=-1).astype(np.float32, copy=False)
    return torch.as_tensor(combined, device=device), torch.as_tensor(validity, device=device)


def _xyz(frame: pd.DataFrame, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    names = [f"{prefix}_{axis}_um" for axis in ("z", "y", "x")]
    valid = np.ones(len(frame), dtype=bool)
    values = np.zeros((len(frame), 3), dtype=np.float32)
    if not all(name in frame for name in names):
        return values, np.zeros(len(frame), dtype=bool)
    for axis, name in enumerate(names):
        column = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        valid &= np.isfinite(column)
        values[:, axis] = np.nan_to_num(column, nan=0.0).astype(np.float32)
    values[~valid] = 0.0
    return values, valid



def add_global_only_predictions(
    candidates: pd.DataFrame,
    observations: pd.DataFrame,
    global_motion: pd.DataFrame,
    *,
    voxel_size_zyx_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
) -> pd.DataFrame:
    """Persist the newly required global-shift-only expected position.

    This is intentionally separate from current Stage-11 `forward_predicted_*`,
    which includes relative motion (and its residual correction).  Missing
    global-shift steps yield NaN rather than silently falling back to another
    motion expert.
    """

    required = {"source_track_id", "source_end_frame", "target_start_frame"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"candidates missing {sorted(required - set(candidates.columns))}")
    obs_required = {"track_id", "frame", "z", "y", "x"}
    if not obs_required.issubset(observations.columns):
        raise ValueError(f"observations missing {sorted(obs_required - set(observations.columns))}")
    gm_required = {"from_frame", "to_frame"}
    if not gm_required.issubset(global_motion.columns):
        raise ValueError("global_motion must contain from_frame/to_frame")
    alternatives = (
        ("final_shift_z_um", "final_shift_y_um", "final_shift_x_um"),
        ("global_shift_z_um", "global_shift_y_um", "global_shift_x_um"),
        ("shift_z_um", "shift_y_um", "shift_x_um"),
    )
    shift_cols = next((cols for cols in alternatives if set(cols).issubset(global_motion.columns)), None)
    if shift_cols is None:
        raise ValueError("global_motion does not contain a recognized physical shift triplet")
    lookup: dict[tuple[int, int], np.ndarray] = {}
    for row in global_motion.itertuples(index=False):
        shift = np.asarray([getattr(row, c) for c in shift_cols], dtype=float)
        if np.all(np.isfinite(shift)):
            lookup[(int(row.from_frame), int(row.to_frame))] = shift
    spacing = np.asarray(voxel_size_zyx_um, dtype=float)
    endpoints = observations.sort_values(["track_id", "frame"], kind="mergesort").groupby("track_id", sort=False)
    by_track = {int(tid): group for tid, group in endpoints}
    result = candidates.copy()
    predictions = np.full((len(result), 3), np.nan, dtype=float)
    for i, row in enumerate(result.itertuples(index=False)):
        tid = int(row.source_track_id)
        sf = int(row.source_end_frame)
        tf = int(row.target_start_frame)
        group = by_track.get(tid)
        if group is None:
            continue
        endpoint = group.loc[pd.to_numeric(group["frame"], errors="coerce") == sf]
        if endpoint.empty:
            continue
        p = endpoint.iloc[-1][["z", "y", "x"]].to_numpy(dtype=float) * spacing
        shifts: list[np.ndarray] = []
        complete = True
        for frame in range(sf + 1, tf + 1):
            shift = lookup.get((frame - 1, frame))
            if shift is None:
                complete = False
                break
            shifts.append(shift)
        if complete:
            predictions[i] = p + (np.sum(shifts, axis=0) if shifts else 0.0)
    for axis, name in enumerate(("z", "y", "x")):
        result[f"global_predicted_{name}_um"] = predictions[:, axis]
    return result


def prediction_columns_from_stage11(candidates: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract candidate-gap-specific motion experts.

    Current Stage 11 already persists `forward_predicted_*` (global + relative
    motion + residual), `anchor_predicted_*` (local motion), and backward
    prediction.  The newly requested global-only prediction should be persisted
    under `global_predicted_{z,y,x}_um`; if absent it is marked unavailable
    rather than fabricated from another expert.
    """

    global_xyz, vg = _xyz(candidates, "global_predicted")
    global_relative, vgr = _xyz(candidates, "forward_predicted")
    local_xyz, vl = _xyz(candidates, "anchor_predicted")
    backward_xyz, vb = _xyz(candidates, "backward_predicted")
    return {
        "expected_global_xyz_um": global_xyz,
        "expected_global_relative_xyz_um": global_relative,
        "expected_local_xyz_um": local_xyz,
        "expected_backward_source_xyz_um": backward_xyz,
        "prediction_valid": np.stack((vg, vgr, vl, vb), axis=-1),
    }
