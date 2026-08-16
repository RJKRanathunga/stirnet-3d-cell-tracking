from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage as ndi

from learned.stirnet import StirNet, StirNetConfig, build_geometry_targets

from .conftest import small_model_config, synthetic_batch


def test_public_imports():
    assert StirNet is not None
    assert StirNetConfig is not None


def test_geometry_targets_use_anisotropic_physical_spacing():
    labels = torch.zeros((1, 5, 9, 9), dtype=torch.long)
    labels[:, 1:4, 2:7, 2:7] = 1
    target = build_geometry_targets(
        labels,
        torch.tensor([[2.0, 0.5, 0.5]]),
        torch.tensor([4.0]),
    )
    assert target.sdf.shape == (1, 1, 5, 9, 9)
    assert target.flow.shape == (1, 3, 5, 9, 9)
    assert torch.isfinite(target.sdf).all()
    assert not torch.allclose(target.flow[:, 0], target.flow[:, 1])


def test_spatial_and_temporal_forward_partition_invariants():
    model = StirNet(small_model_config()).eval()
    batch = synthetic_batch(temporal=True)
    with torch.no_grad():
        output = model(
            batch["spatial_inputs"],
            batch["spacing_um"],
            batch["dref_um"],
            graph_x=batch["graph_x"],
            graph_edge_index=batch["graph_edge_index"],
            graph_edge_attr=batch["graph_edge_attr"],
            tracklet_id=batch["tracklet_id"],
            temporal_ref_um=batch["temporal_ref_um"],
            temporal_status=batch["temporal_status"],
            temporal_batch=batch["temporal_batch"],
            node_instance_grid=batch["node_instance_grid"],
            node_history_valid=batch["node_history_valid"],
            run_refinement=False,
            apply_existence_filter=False,
        )
    assert output.final_labels[0].shape == batch["instance_labels"][0].shape
    assert output.final_labels[0].dtype == torch.long
    assert output.temporal.tokens.shape == (2, model.cfg.temporal.d_model)
    assert output.reasoning.recovery_logits.shape == (2,)
    assert output.reasoning.recovery_scores.shape == (2,)
    labels = output.final_labels[0].cpu().numpy()
    assert np.all(labels >= 0)
    for instance_id in range(1, int(labels.max()) + 1):
        _, count = ndi.label(labels == instance_id, structure=ndi.generate_binary_structure(3, 1))
        assert count == 1
    assert output.centers_um[0].shape[0] == int(labels.max())
    spacing = batch["spacing_um"][0]
    extent = (torch.tensor(labels.shape) - 1) * spacing
    for row, center in enumerate(output.centers_um[0], 1):
        voxel = torch.round((center + 0.5 * extent) / spacing).long()
        assert int(output.final_labels[0][tuple(voxel.tolist())]) == row


def test_spatial_forward_needs_no_temporal_evidence():
    model = StirNet(small_model_config()).eval()
    batch = synthetic_batch(temporal=False)
    with torch.no_grad():
        output = model(
            batch["spatial_inputs"],
            batch["spacing_um"],
            batch["dref_um"],
            run_refinement=False,
            apply_existence_filter=False,
        )
    assert output.temporal.is_empty
    assert torch.equal(
        output.reasoning.final_edge_logits, output.rag.spatial_edge_logits
    )
