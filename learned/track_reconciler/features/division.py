"""Explicit symmetric biological features for parent -> daughter-pair hypotheses."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


DIVISION_PAIR_DIM = 16


def build_division_pair_features(
    *,
    parent_xyz_um: Tensor,
    child_a_xyz_um: Tensor,
    child_b_xyz_um: Tensor,
    expected_parent_future_xyz_um: Tensor,
    parent_volume: Tensor,
    child_a_volume: Tensor,
    child_b_volume: Tensor,
    parent_fingerprint: Tensor,
    child_a_fingerprint: Tensor,
    child_b_fingerprint: Tensor,
    child_a_gap_frames: Tensor,
    child_b_gap_frames: Tensor,
    distance_scale_um: float = 10.0,
) -> Tensor:
    """Return 16 daughter-order-invariant division features.

    This head receives *pair* evidence that independent A->B and A->C edge
    scores cannot represent: volume conservation, branch geometry, midpoint
    agreement and the relation among the two daughter appearances.
    """

    eps = 1e-6
    scale = max(float(distance_scale_um), eps)
    pv = parent_volume.clamp_min(eps)
    av = child_a_volume.clamp_min(eps)
    bv = child_b_volume.clamp_min(eps)
    combined_ratio = (av + bv) / pv
    combined_log = torch.log(combined_ratio)
    daughter_balance = torch.abs(torch.log(av / bv))

    pa = child_a_xyz_um - parent_xyz_um
    pb = child_b_xyz_um - parent_xyz_um
    child_delta = child_a_xyz_um - child_b_xyz_um
    midpoint = 0.5 * (child_a_xyz_um + child_b_xyz_um)
    da = torch.linalg.vector_norm(pa, dim=-1)
    db = torch.linalg.vector_norm(pb, dim=-1)
    separation = torch.linalg.vector_norm(child_delta, dim=-1)
    midpoint_expected = torch.linalg.vector_norm(
        midpoint - expected_parent_future_xyz_um, dim=-1
    )
    midpoint_parent = torch.linalg.vector_norm(midpoint - parent_xyz_um, dim=-1)
    branch_cos = torch.sum(
        F.normalize(pa, dim=-1, eps=eps) * F.normalize(pb, dim=-1, eps=eps),
        dim=-1,
    )

    pf = F.normalize(parent_fingerprint, dim=-1, eps=eps)
    af = F.normalize(child_a_fingerprint, dim=-1, eps=eps)
    bf = F.normalize(child_b_fingerprint, dim=-1, eps=eps)
    pca = torch.sum(pf * af, dim=-1)
    pcb = torch.sum(pf * bf, dim=-1)
    ccb = torch.sum(af * bf, dim=-1)
    pda = torch.linalg.vector_norm(pf - af, dim=-1)
    pdb = torch.linalg.vector_norm(pf - bf, dim=-1)
    cdb = torch.linalg.vector_norm(af - bf, dim=-1)

    simultaneous = (child_a_gap_frames == child_b_gap_frames).to(parent_xyz_um.dtype)
    mean_gap = 0.5 * (
        child_a_gap_frames.to(parent_xyz_um.dtype)
        + child_b_gap_frames.to(parent_xyz_um.dtype)
    ) / 4.0

    features = torch.stack(
        (
            combined_log,
            torch.abs(combined_log),
            daughter_balance,
            separation / scale,
            0.5 * (da + db) / scale,
            torch.abs(da - db) / scale,
            midpoint_expected / scale,
            branch_cos,
            midpoint_parent / scale,
            0.5 * (pca + pcb),
            torch.abs(pca - pcb),
            ccb,
            0.5 * (pda + pdb),
            cdb,
            simultaneous,
            mean_gap,
        ),
        dim=-1,
    )
    if features.shape[-1] != DIVISION_PAIR_DIM:
        raise RuntimeError("division feature dimension drifted")
    return features
