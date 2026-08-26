import torch

from learned.track_reconciler import TrackletReconciliationNetwork
from learned.track_reconciler.smoke_test import make_batch


def test_forward_shapes_and_gradients():
    model = TrackletReconciliationNetwork()
    batch = make_batch("cpu")
    output = model(batch)
    assert output.continuation_logits.shape == (1, 6)
    assert output.parental_probabilities.shape == (1, 6)
    assert output.division_logits is not None and output.division_logits.shape == (1, 2)
    assert output.appearance_logits.shape == (1, 5)
    loss = output.continuation_logits.sum() + output.division_logits.sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
