import pytest
import torch

from learned.instance_segmentation.model.losses import (
    VectorCNNLoss,
    VectorCNNTargets,
    focal_bce_with_logits,
    soft_dice_loss_from_logits,
)
from learned.instance_segmentation.model.vector_cnn import VectorCNNOutput


def _make_output_and_targets(shape=(2, 1, 4, 8, 8)):
    batch, _, z, y, x = shape
    foreground = (torch.rand(batch, 1, z, y, x) > 0.55).float()
    boundary = (torch.rand(batch, 1, z, y, x) > 0.85).float()
    center = (torch.rand(batch, 1, z, y, x) > 0.95).float()
    target_vectors = torch.zeros(batch, 3, z, y, x)
    target_vectors[:, 0] = 0.15
    target_vectors[:, 1] = -0.10
    target_vectors[:, 2] = 0.05
    target_vectors *= foreground
    output = VectorCNNOutput(
        foreground_logits=torch.randn(batch, 1, z, y, x, requires_grad=True),
        vectors_normalized=torch.tanh(torch.randn(batch, 3, z, y, x, requires_grad=True)),
        boundary_logits=torch.randn(batch, 1, z, y, x, requires_grad=True),
        center_logits=torch.randn(batch, 1, z, y, x, requires_grad=True),
    )
    targets = VectorCNNTargets(
        foreground=foreground,
        vectors_normalized=target_vectors,
        boundary=boundary,
        center=center,
        valid_mask=torch.ones(batch, 1, z, y, x),
    )
    return output, targets


def test_multitask_loss_is_finite_and_backpropagates():
    output, targets = _make_output_and_targets()
    losses = VectorCNNLoss()(output, targets)
    assert set(losses.as_dict()) == {"total", "foreground", "vector", "direction", "boundary", "center"}
    for value in losses.as_dict().values():
        assert value.ndim == 0
        assert torch.isfinite(value)
    losses.total.backward()
    assert output.foreground_logits.grad is not None
    assert output.boundary_logits.grad is not None
    assert output.center_logits.grad is not None


def test_perfect_binary_logits_have_lower_dice_loss_than_wrong_logits():
    target = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])
    good_logits = torch.where(target > 0.5, torch.tensor(8.0), torch.tensor(-8.0))
    bad_logits = -good_logits
    assert soft_dice_loss_from_logits(good_logits, target) < soft_dice_loss_from_logits(bad_logits, target)


def test_valid_mask_removes_invalid_voxels_from_focal_loss():
    logits_a = torch.zeros(1, 1, 1, 1, 2)
    logits_b = logits_a.clone()
    logits_b[..., 1] = 100.0
    target = torch.zeros_like(logits_a)
    valid = torch.tensor([[[[[1.0, 0.0]]]]])
    assert torch.allclose(
        focal_bce_with_logits(logits_a, target, valid_mask=valid),
        focal_bce_with_logits(logits_b, target, valid_mask=valid),
    )


def test_loss_rejects_vector_shape_mismatch():
    output, targets = _make_output_and_targets(shape=(1, 1, 4, 8, 8))
    bad_targets = VectorCNNTargets(
        foreground=targets.foreground,
        vectors_normalized=torch.zeros(1, 3, 4, 8, 7),
        boundary=targets.boundary,
        center=targets.center,
        valid_mask=targets.valid_mask,
    )
    with pytest.raises(ValueError, match="vector output/target shapes must match"):
        VectorCNNLoss()(output, bad_targets)
