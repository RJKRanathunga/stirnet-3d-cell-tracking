import torch

from ...model import VectorCNNLoss, VectorCNNTargets, VectorInstanceCNN


def test_model_and_loss_contract():
    model = VectorInstanceCNN()
    x = torch.rand(1, 4, 16, 64, 64)
    output = model(x)
    assert output.vectors_normalized.shape == (1, 3, 16, 64, 64)
    scalar = (1, 1, 16, 64, 64)
    targets = VectorCNNTargets(
        foreground=torch.zeros(scalar),
        vectors_normalized=torch.zeros_like(output.vectors_normalized),
        boundary=torch.zeros(scalar),
        center=torch.zeros(scalar),
        valid_mask=torch.ones(scalar),
    )
    loss = VectorCNNLoss()(output, targets)
    assert torch.isfinite(loss.total)
