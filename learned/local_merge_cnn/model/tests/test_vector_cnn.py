import pytest
import torch

from learned.instance_segmentation.model.vector_cnn import (
    VectorCNNConfig,
    VectorCNNOutput,
    VectorInstanceCNN,
)


def _small_config(shape=(16, 16, 16)):
    return VectorCNNConfig(
        input_channels=4,
        channels=(4, 8, 12, 16, 20),
        group_norm_groups=4,
        dropout=0.0,
        input_shape_zyx=shape,
    )


def test_default_model_contract_is_64_cube():
    config = VectorCNNConfig()
    assert config.input_shape_zyx == (64, 64, 64)
    assert config.channels == (16, 32, 64, 128, 192)


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        VectorCNNConfig(input_channels=0)
    with pytest.raises(ValueError):
        VectorCNNConfig(channels=(8, 16, 24, 32))
    with pytest.raises(ValueError):
        VectorCNNConfig(group_norm_groups=0)
    with pytest.raises(ValueError):
        VectorCNNConfig(dropout=1.0)
    with pytest.raises(ValueError):
        VectorCNNConfig(input_shape_zyx=(15, 64, 64))
    with pytest.raises(ValueError):
        VectorCNNConfig(input_shape_zyx=(30, 64, 64))


def test_forward_preserves_spatial_shape_and_head_contract():
    model = VectorInstanceCNN(_small_config()).eval()
    x = torch.rand(1, 4, 16, 16, 16)
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, VectorCNNOutput)
    assert output.foreground_logits.shape == (1, 1, 16, 16, 16)
    assert output.vectors_normalized.shape == (1, 3, 16, 16, 16)
    assert output.boundary_logits.shape == (1, 1, 16, 16, 16)
    assert output.center_logits.shape == (1, 1, 16, 16, 16)
    assert torch.isfinite(output.foreground_logits).all()
    assert torch.isfinite(output.vectors_normalized).all()
    assert float(output.vectors_normalized.min()) >= -1.0
    assert float(output.vectors_normalized.max()) <= 1.0


def test_encoder_downsamples_isotropically_every_level():
    model = VectorInstanceCNN(_small_config(shape=(32, 32, 32))).eval()
    x = torch.rand(1, 4, 32, 32, 32)
    with torch.no_grad():
        features = model.encoder(x)
    assert features.level0.shape[-3:] == (32, 32, 32)
    assert features.level1.shape[-3:] == (16, 16, 16)
    assert features.level2.shape[-3:] == (8, 8, 8)
    assert features.level3.shape[-3:] == (4, 4, 4)
    assert features.bottleneck.shape[-3:] == (2, 2, 2)


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
        model(torch.rand(4, 16, 16, 16))
    with pytest.raises(ValueError, match="expected 4 input channels"):
        model(torch.rand(1, 3, 16, 16, 16))
    with pytest.raises(ValueError, match="expected spatial shape"):
        model(torch.rand(1, 4, 32, 32, 32))
