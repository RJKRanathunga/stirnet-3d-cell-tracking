import torch

from learned.track_reconciler.features.parental import parental_softmax


def test_parental_softmax_has_explicit_no_parent_mass():
    logits = torch.tensor([[0.0, 0.0, 4.0]])
    target = torch.tensor([[2, 2, 2]])
    gap = torch.tensor([[1, 1, 2]])
    mask = torch.ones_like(logits, dtype=torch.bool)
    p, q = parental_softmax(logits, target, gap, mask)
    assert torch.allclose(p[0, :2], torch.tensor([1/3, 1/3]), atol=1e-6)
    assert torch.allclose(q[0, :2], torch.tensor([1/3, 1/3]), atol=1e-6)
    # gap=2 is normalized separately: sigmoid(4)
    assert torch.allclose(p[0, 2], torch.sigmoid(torch.tensor(4.0)), atol=1e-6)
