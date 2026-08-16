from __future__ import annotations

import torch

from learned.stirnet import StirNet
from learned.stirnet.model.spatial.blocks import AxisFactorizedConv

from .conftest import small_model_config, synthetic_batch


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_depthwise_axis_block_has_same_shape_finite_gradients_and_lower_cost():
    torch.manual_seed(18)
    dense = AxisFactorizedConv(
        32, 16, variant="dense", bottleneck_ratio=0.5
    )
    depthwise = AxisFactorizedConv(
        32, 16, variant="depthwise", bottleneck_ratio=0.5
    )
    x = torch.randn((2, 32, 5, 9, 9), requires_grad=True)
    acquisition = torch.randn((2, 16), requires_grad=True)
    output = depthwise(x, acquisition)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert acquisition.grad is not None and torch.isfinite(acquisition.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in depthwise.parameters()
    )
    assert _parameter_count(depthwise) < _parameter_count(dense)
    assert (
        depthwise.estimated_conv_macs_per_voxel()
        < dense.estimated_conv_macs_per_voxel()
    )


def test_dense_axis_checkpoint_contract_is_unchanged():
    source = AxisFactorizedConv(12, 8, variant="dense")
    state = source.state_dict()
    assert set(state) == {
        "conv_z.weight",
        "conv_y.weight",
        "conv_x.weight",
        "gate.weight",
        "gate.bias",
        "fuse.weight",
    }
    restored = AxisFactorizedConv(12, 8, variant="dense")
    restored.load_state_dict(state, strict=True)
    for key, value in state.items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_depthwise_variant_runs_the_complete_spatial_geometry_path():
    config = small_model_config()
    config.spatial.axis_conv_variant = "depthwise"
    config.spatial.axis_conv_bottleneck_ratio = 0.5
    config.validate()
    model = StirNet(config).eval()
    batch = synthetic_batch(temporal=False)
    with torch.no_grad():
        output = model(
            batch["spatial_inputs"],
            batch["spacing_um"],
            batch["dref_um"],
            execution_stage="geometry",
        )
    assert output.geometry.sdf.shape[-3:] == batch["spatial_inputs"].shape[-3:]
    assert torch.isfinite(output.geometry.sdf).all()
    assert torch.isfinite(output.geometry.flow).all()


def test_depthwise_variant_is_opt_in():
    config = small_model_config()
    assert config.spatial.axis_conv_variant == "dense"

