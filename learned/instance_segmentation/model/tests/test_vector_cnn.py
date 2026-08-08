import pytest
import torch

from learned.instance_segmentation.model.vector_cnn import (
    VectorCNNConfig,
    VectorCNNOutput,
    VectorInstanceCNN,
)


def _small_config():
    return VectorCNNConfig(
        input_channels=4,
        channels=(8, 16, 24, 32, 40),
        group_norm_groups=8,
        dropout=0.0,
    )


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        VectorCNNConfig(input_channels=0)
    with pytest.raises(ValueError):
        VectorCNNConfig(channels=(8, 16, 24, 32))
    with pytest.raises(ValueError):
        VectorCNNConfig(group_norm_groups=0)
    with pytest.raises(ValueError):
        VectorCNNConfig(dropout=1.0)


def test_forward_preserves_spatial_shape_and_head_contract():
    model = VectorInstanceCNN(_small_config()).eval()
    x = torch.rand(2, 4, 8, 32, 32)

    with torch.no_grad():
        output = model(x)

    assert isinstance(output, VectorCNNOutput)
    assert output.foreground_logits.shape == (2, 1, 8, 32, 32)
    assert output.vectors_normalized.shape == (2, 3, 8, 32, 32)
    assert output.boundary_logits.shape == (2, 1, 8, 32, 32)
    assert output.center_logits.shape == (2, 1, 8, 32, 32)

    assert torch.isfinite(output.foreground_logits).all()
    assert torch.isfinite(output.vectors_normalized).all()
    assert torch.isfinite(output.boundary_logits).all()
    assert torch.isfinite(output.center_logits).all()

    assert float(output.vectors_normalized.min()) >= -1.0
    assert float(output.vectors_normalized.max()) <= 1.0


def test_probability_properties_are_bounded():
    output = VectorCNNOutput(
        foreground_logits=torch.tensor([[[[[-10.0, 10.0]]]]]),
        vectors_normalized=torch.zeros(1, 3, 1, 1, 2),
        boundary_logits=torch.tensor([[[[[-10.0, 10.0]]]]]),
        center_logits=torch.tensor([[[[[-10.0, 10.0]]]]]),
    )

    for probability in (
        output.foreground_probability,
        output.boundary_probability,
        output.center_probability,
    ):
        assert torch.all(probability >= 0.0)
        assert torch.all(probability <= 1.0)


def test_canonical_vector_conversion_uses_axis_fractions():
    vectors = torch.zeros(1, 3, 4, 5, 6)
    vectors[:, 0] = 0.5
    vectors[:, 1] = -0.25
    vectors[:, 2] = 1.0

    output = VectorCNNOutput(
        foreground_logits=torch.zeros(1, 1, 4, 5, 6),
        vectors_normalized=vectors,
        boundary_logits=torch.zeros(1, 1, 4, 5, 6),
        center_logits=torch.zeros(1, 1, 4, 5, 6),
    )

    displacement = output.vectors_to_canonical_displacement()

    assert torch.allclose(displacement[:, 0], torch.full((1, 4, 5, 6), 1.5))
    assert torch.allclose(displacement[:, 1], torch.full((1, 4, 5, 6), -1.0))
    assert torch.allclose(displacement[:, 2], torch.full((1, 4, 5, 6), 5.0))


def test_input_validation():
    model = VectorInstanceCNN(_small_config()).eval()

    with pytest.raises(ValueError, match=r"\[B,C,Z,Y,X\]"):
        model(torch.rand(4, 8, 32, 32))

    with pytest.raises(ValueError, match="expected 4 input channels"):
        model(torch.rand(1, 3, 8, 32, 32))

    with pytest.raises(ValueError, match="Z>=4"):
        model(torch.rand(1, 4, 3, 32, 32))

    with pytest.raises(ValueError, match="Y>=16"):
        model(torch.rand(1, 4, 8, 15, 32))
