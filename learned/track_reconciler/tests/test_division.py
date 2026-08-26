import torch

from learned.track_reconciler import TrackletReconciliationNetwork
from learned.track_reconciler.smoke_test import make_batch


def test_division_head_is_daughter_order_invariant():
    torch.manual_seed(3)
    model = TrackletReconciliationNetwork().eval()
    first = make_batch("cpu")
    second = make_batch("cpu")
    # Make inputs exactly equal, then swap daughter-edge order only.
    second = first
    swapped = second.divisions.edge_pair_index.flip(-1).clone()
    original = second.divisions.edge_pair_index
    with torch.no_grad():
        out_a = model(first).division_logits
        second.divisions.edge_pair_index = swapped
        out_b = model(second).division_logits
        second.divisions.edge_pair_index = original
    assert torch.allclose(out_a, out_b, atol=1e-6)
