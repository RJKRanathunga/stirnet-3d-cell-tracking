from __future__ import annotations

import torch

from learned.stirnet import StirNet
from learned.stirnet.model.utils.physical import (
    canonical_resample_spec,
    resample_continuous_volume,
    resample_labels_volume,
)

from .conftest import small_model_config, synthetic_batch


def test_canonical_grid_preserves_physical_extent_and_anisotropy():
    shape = (6, 12, 12)
    spacing = torch.tensor([[2.0, 0.208, 0.208]])
    target, effective = canonical_resample_spec(
        shape, spacing, (2.0, 0.4, 0.4)
    )
    assert target[0] == shape[0]
    assert target[1] < shape[1] and target[2] < shape[2]
    native_extent = (torch.tensor(shape) - 1) * spacing[0]
    target_extent = (torch.tensor(target) - 1) * effective[0]
    torch.testing.assert_close(target_extent, native_extent)
    assert effective[0, 0] > effective[0, 1]


def test_continuous_round_trip_preserves_positions_and_vector_units():
    shape = (7, 13, 15)
    z = torch.linspace(-2, 2, shape[0])[:, None, None]
    y = torch.linspace(-1, 1, shape[1])[None, :, None]
    x = torch.linspace(-3, 3, shape[2])[None, None, :]
    scalar = (z + 2 * y - 0.5 * x)[None, None]
    reduced = resample_continuous_volume(scalar, (7, 8, 9))
    restored = resample_continuous_volume(reduced, shape)
    torch.testing.assert_close(restored, scalar, atol=1e-6, rtol=1e-6)

    flow = torch.zeros((1, 3, *shape))
    flow[:, 0] = 0.25
    flow[:, 1] = -0.5
    flow[:, 2] = 0.75
    offset = 1.7 * flow
    restored_flow = resample_continuous_volume(
        resample_continuous_volume(flow, (7, 8, 9)), shape
    )
    restored_offset = resample_continuous_volume(
        resample_continuous_volume(offset, (7, 8, 9)), shape
    )
    torch.testing.assert_close(restored_flow, flow)
    torch.testing.assert_close(restored_offset, offset)
    torch.testing.assert_close(
        torch.linalg.vector_norm(restored_offset, dim=1),
        torch.linalg.vector_norm(offset, dim=1),
    )


def test_label_resampling_is_nearest_neighbor_and_integer_only():
    labels = torch.zeros((5, 9, 11), dtype=torch.long)
    labels[1:4, 2:5, 2:5] = 3
    labels[1:4, 5:8, 6:10] = 9
    reduced = resample_labels_volume(labels, (5, 6, 7))
    restored = resample_labels_volume(reduced, labels.shape)
    assert reduced.dtype == torch.long and restored.dtype == torch.long
    assert set(torch.unique(reduced).tolist()).issubset({0, 3, 9})
    assert set(torch.unique(restored).tolist()).issubset({0, 3, 9})


def test_opt_in_model_restores_native_explicit_geometry_shape():
    config = small_model_config()
    config.spatial.canonical_spacing_um = (1.6, 0.6, 0.6)
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
    native_shape = batch["spatial_inputs"].shape[-3:]
    assert output.geometry.sdf.shape[-3:] == native_shape
    assert output.geometry.flow.shape[-3:] == native_shape
    assert output.geometry.centroid_offset.shape[-3:] == native_shape
    assert output.decoded_spatial.d0.shape[-3:] != native_shape
    assert output.geometry.features.shape[-3:] == output.decoded_spatial.d0.shape[-3:]
    assert output.geometry.feature_spacing_um is not None


def test_canonical_spacing_is_off_by_default_and_uses_no_label_input():
    config = small_model_config()
    assert config.spatial.canonical_spacing_um is None
    # The resampling contract is determined only by native shape/spacing and
    # the requested physical grid; ground-truth labels are not an input.
    target_a, spacing_a = canonical_resample_spec(
        (6, 12, 12), torch.tensor([[1.6, 0.4, 0.4]]), (1.6, 0.6, 0.6)
    )
    target_b, spacing_b = canonical_resample_spec(
        (6, 12, 12), torch.tensor([[1.6, 0.4, 0.4]]), (1.6, 0.6, 0.6)
    )
    assert target_a == target_b
    torch.testing.assert_close(spacing_a, spacing_b)

