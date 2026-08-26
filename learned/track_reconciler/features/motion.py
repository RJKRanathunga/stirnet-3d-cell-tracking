"""Explicit physical relation features for candidate A -> B transitions."""

from __future__ import annotations

import torch
from torch import Tensor


# 7 vectors * 3 + 7 scalar terms + 4 validity bits + 2 gap terms = 34.
MOTION_RELATION_DIM = 34


def _gather_tracklet(values: Tensor, indices: Tensor) -> Tensor:
    """Gather `[B,N,D]` with `[B,E]` indices -> `[B,E,D]`."""

    b = torch.arange(values.shape[0], device=values.device)[:, None]
    return values[b, indices]


def build_motion_relation_features(
    *,
    source_end_xyz_um: Tensor,
    target_start_xyz_um: Tensor,
    expected_global_xyz_um: Tensor,
    expected_global_relative_xyz_um: Tensor,
    expected_local_xyz_um: Tensor,
    expected_backward_source_xyz_um: Tensor,
    prediction_valid: Tensor,
    gap_frames: Tensor,
    distance_scale_um: float,
) -> Tensor:
    """Build normalized, explicit prediction and residual features.

    Crucially, the expected global-only and global+relative positions are not
    collapsed into just residual magnitudes.  Their source-relative prediction
    vectors are retained so the edge reasoner can compare the motion experts.
    Invalid experts are zeroed and accompanied by explicit validity bits.
    """

    scale = max(float(distance_scale_um), 1e-6)
    valid = prediction_valid.to(source_end_xyz_um.dtype)

    direct = (target_start_xyz_um - source_end_xyz_um) / scale
    pred_g = (expected_global_xyz_um - source_end_xyz_um) / scale
    pred_gr = (expected_global_relative_xyz_um - source_end_xyz_um) / scale
    pred_l = (expected_local_xyz_um - source_end_xyz_um) / scale
    residual_g = (target_start_xyz_um - expected_global_xyz_um) / scale
    residual_gr = (target_start_xyz_um - expected_global_relative_xyz_um) / scale
    residual_l = (target_start_xyz_um - expected_local_xyz_um) / scale
    residual_back = (source_end_xyz_um - expected_backward_source_xyz_um) / scale

    def gated(vector: Tensor, which: int) -> Tensor:
        return vector * valid[..., which : which + 1]

    pred_g = gated(pred_g, 0)
    pred_gr = gated(pred_gr, 1)
    pred_l = gated(pred_l, 2)
    residual_g = gated(residual_g, 0)
    residual_gr = gated(residual_gr, 1)
    residual_l = gated(residual_l, 2)
    residual_back = gated(residual_back, 3)

    norm = lambda x: torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    disagreement = pred_gr - direct - residual_back  # forward displacement - backward-implied displacement
    disagreement = disagreement * (valid[..., 1:2] * valid[..., 3:4])

    # Keep the most informative vectors.  Direct + 3 predictions + 3 forward
    # residuals = 7 vectors (21 dimensions).
    vectors = [direct, pred_g, pred_gr, pred_l, residual_g, residual_gr, residual_l]
    scalars = [
        norm(direct),
        norm(residual_g),
        norm(residual_gr),
        norm(residual_l),
        norm(residual_back),
        norm(disagreement),
        torch.sum(
            torch.nn.functional.normalize(pred_gr, dim=-1, eps=1e-6)
            * torch.nn.functional.normalize(direct, dim=-1, eps=1e-6),
            dim=-1,
            keepdim=True,
        ) * valid[..., 1:2],
    ]
    gap = gap_frames.to(source_end_xyz_um.dtype).unsqueeze(-1)
    gap_terms = [gap / 4.0, torch.log1p(gap) / torch.log(torch.tensor(5.0, device=gap.device, dtype=gap.dtype))]
    result = torch.cat(vectors + scalars + [valid] + gap_terms, dim=-1)
    if result.shape[-1] != MOTION_RELATION_DIM:
        raise RuntimeError(f"motion relation dimension drifted to {result.shape[-1]}")
    return result


def gather_edge_endpoints(
    start_xyz_um: Tensor,
    end_xyz_um: Tensor,
    edge_index: Tensor,
) -> tuple[Tensor, Tensor]:
    source = edge_index[..., 0]
    target = edge_index[..., 1]
    source_end = _gather_tracklet(end_xyz_um, source)
    target_start = _gather_tracklet(start_xyz_um, target)
    return source_end, target_start
