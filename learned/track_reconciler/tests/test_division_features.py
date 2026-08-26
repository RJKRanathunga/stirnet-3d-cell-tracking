import torch

from learned.track_reconciler.features.division import build_division_pair_features


def test_division_features_are_symmetric_in_daughters():
    torch.manual_seed(7)
    p = torch.randn(2, 3)
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    expected = torch.randn(2, 3)
    pv = torch.rand(2) + 2
    av = torch.rand(2) + 1
    bv = torch.rand(2) + 1
    pf = torch.randn(2, 96)
    af = torch.randn(2, 96)
    bf = torch.randn(2, 96)
    ga = torch.tensor([1, 2])
    gb = torch.tensor([1, 2])
    first = build_division_pair_features(
        parent_xyz_um=p, child_a_xyz_um=a, child_b_xyz_um=b,
        expected_parent_future_xyz_um=expected,
        parent_volume=pv, child_a_volume=av, child_b_volume=bv,
        parent_fingerprint=pf, child_a_fingerprint=af, child_b_fingerprint=bf,
        child_a_gap_frames=ga, child_b_gap_frames=gb,
    )
    swapped = build_division_pair_features(
        parent_xyz_um=p, child_a_xyz_um=b, child_b_xyz_um=a,
        expected_parent_future_xyz_um=expected,
        parent_volume=pv, child_a_volume=bv, child_b_volume=av,
        parent_fingerprint=pf, child_a_fingerprint=bf, child_b_fingerprint=af,
        child_a_gap_frames=gb, child_b_gap_frames=ga,
    )
    assert first.shape[-1] == 16
    assert torch.allclose(first, swapped, atol=1e-6)
