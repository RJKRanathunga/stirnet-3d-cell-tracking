import torch

from learned.track_reconciler.features.geometry import pairwise_segment_distance, segment_segment_distance


def test_crossing_segments_have_zero_distance():
    p0 = torch.tensor([0.0, 0.0, 0.0])
    p1 = torch.tensor([1.0, 1.0, 0.0])
    q0 = torch.tensor([0.0, 1.0, 0.0])
    q1 = torch.tensor([1.0, 0.0, 0.0])
    distance = segment_segment_distance(p0, p1, q0, q1)
    assert float(distance) < 1e-5


def test_pairwise_segment_distance_is_symmetric():
    start = torch.randn(2, 7, 3)
    end = start + torch.randn(2, 7, 3)
    d = pairwise_segment_distance(start, end)
    assert torch.allclose(d, d.transpose(1, 2), atol=1e-6)
    assert torch.allclose(torch.diagonal(d, dim1=1, dim2=2), torch.zeros(2, 7), atol=1e-6)
