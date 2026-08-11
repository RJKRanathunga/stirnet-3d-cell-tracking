from __future__ import annotations

import numpy as np

from ..core.stats import cosine_similarity_rows


def _np(state, key):
    if state is None:
        return None
    value = state.get(key)
    if value is None:
        return None
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def build_temporal_table(hook_capture):
    if hook_capture is None or hook_capture.temporal_initial is None:
        return []

    initial = hook_capture.temporal_initial
    cr1 = hook_capture.temporal_after_cr1
    cr2 = hook_capture.temporal_after_cr2
    t0 = _np(initial, "tokens")
    refs = _np(initial, "ref_um")
    sal = _np(initial, "salience")
    rel = _np(initial, "reliability")
    status = _np(initial, "status")
    t1 = _np(cr1, "tokens")
    t2 = _np(cr2, "tokens")
    n = len(t0)

    norm0 = np.linalg.norm(t0, axis=-1)
    norm1 = np.linalg.norm(t1, axis=-1) if t1 is not None else np.full(n, np.nan)
    norm2 = np.linalg.norm(t2, axis=-1) if t2 is not None else np.full(n, np.nan)
    cos01 = cosine_similarity_rows(t0, t1) if t1 is not None else np.full(n, np.nan)
    cos12 = cosine_similarity_rows(t1, t2) if t1 is not None and t2 is not None else np.full(n, np.nan)

    rows = []
    for i in range(n):
        row = {
            "temporal_index": i,
            "ref_z_um": float(refs[i, 0]),
            "ref_y_um": float(refs[i, 1]),
            "ref_x_um": float(refs[i, 2]),
            "salience": float(sal[i, 0]),
            "reliability": float(rel[i, 0]),
            "initial_token_norm": float(norm0[i]),
            "after_cr1_token_norm": float(norm1[i]),
            "after_cr2_token_norm": float(norm2[i]),
            "cosine_initial_to_cr1": float(cos01[i]),
            "cosine_cr1_to_cr2": float(cos12[i]),
        }
        if status is not None:
            for j, value in enumerate(status[i]):
                row[f"status_{j}"] = float(value)
        rows.append(row)
    return rows
