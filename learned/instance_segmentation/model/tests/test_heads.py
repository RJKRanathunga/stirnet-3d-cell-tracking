import torch

from learned.instance_segmentation.model.heads import (
    ScalarPredictionHead,
    VectorCNNHeads,
    VectorPredictionHead,
)


def test_scalar_head_outputs_one_channel():
    head = ScalarPredictionHead(8, groups=4).eval()
    x = torch.randn(2, 8, 4, 16, 16)
    with torch.no_grad():
        y = head(x)
    assert y.shape == (2, 1, 4, 16, 16)
    assert torch.isfinite(y).all()


def test_vector_head_outputs_three_tanh_bounded_channels():
    head = VectorPredictionHead(8, groups=4).eval()
    x = torch.randn(2, 8, 4, 16, 16)
    with torch.no_grad():
        y = head(x)
    assert y.shape == (2, 3, 4, 16, 16)
    assert torch.isfinite(y).all()
    assert float(y.min()) >= -1.0
    assert float(y.max()) <= 1.0


def test_combined_heads_contract():
    heads = VectorCNNHeads(8, groups=4).eval()
    x = torch.randn(1, 8, 4, 16, 16)
    with torch.no_grad():
        foreground, vectors, boundary, center = heads(x)

    assert foreground.shape == (1, 1, 4, 16, 16)
    assert vectors.shape == (1, 3, 4, 16, 16)
    assert boundary.shape == (1, 1, 4, 16, 16)
    assert center.shape == (1, 1, 4, 16, 16)
