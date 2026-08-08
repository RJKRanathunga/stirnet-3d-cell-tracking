import torch

from ...model import VectorCNNConfig, VectorCNNLoss, VectorCNNTargets, VectorInstanceCNN


def test_model_and_loss_contract_on_small_test_cube():
    config = VectorCNNConfig(
        channels=(4, 8, 12, 16, 20),
        group_norm_groups=4,
        input_shape_zyx=(16, 16, 16),
    )
    model = VectorInstanceCNN(config)
    x = torch.rand(1, 4, 16, 16, 16)
    output = model(x)
    assert output.vectors_normalized.shape == (1, 3, 16, 16, 16)
    scalar = (1, 1, 16, 16, 16)
    targets = VectorCNNTargets(
        foreground=torch.zeros(scalar),
        vectors_normalized=torch.zeros_like(output.vectors_normalized),
        boundary=torch.zeros(scalar),
        center=torch.zeros(scalar),
        valid_mask=torch.ones(scalar),
    )
    loss = VectorCNNLoss()(output, targets)
    assert torch.isfinite(loss.total)
