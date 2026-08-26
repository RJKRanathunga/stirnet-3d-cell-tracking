from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage as ndi

from learned.stirnet.model.config import InferenceConfig
from learned.stirnet.model.postprocess.source_core_split import (
    SourceCoreSplitOnlyFilter,
)


def _case():
    shape = (8, 20, 20)
    final = np.zeros(shape, np.int64)
    final[1:4, 1:5, 1:5] = 1
    final[1:4, 1:5, 7:11] = 2
    final[1:4, 1:5, 13:17] = 3
    final[2:7, 7:18, 3:17] = 4

    source = np.zeros(shape, np.float32)
    source[2:4, 2:4, 2:4] = 1
    source[2:4, 2:4, 8:10] = 1
    source[2:4, 2:4, 14:16] = 1
    source[3:6, 9:13, 5:8] = 1
    source[3:6, 9:13, 12:15] = 1

    separator = np.zeros(shape, np.float32)
    separator[:, :, 10:12] = 0.95

    # Atomic supervoxels. The large final component has four existing atomic
    # pieces separated along X. The two source masks anchor the left and right
    # sides; the graph watershed must split only along these existing SVs.
    sv = np.zeros(shape, np.int64)
    sv[1:4, 1:5, 1:5] = 1
    sv[1:4, 1:5, 7:11] = 2
    sv[1:4, 1:5, 13:17] = 3
    large = final == 4
    for x0, x1, sid in ((3, 7, 4), (7, 10, 5), (10, 13, 6), (13, 17, 7)):
        mask = large.copy()
        x = np.zeros(shape, dtype=bool)
        x[:, :, x0:x1] = True
        sv[mask & x] = sid

    return final, source, separator, sv


def _filter(
    final,
    source,
    separator,
    *,
    supervoxels=None,
    source_instances=None,
    anchor_mode="prefer_source_instances",
    method="supervoxel_graph",
    threshold=0.70,
):
    cfg = InferenceConfig()
    cfg.source_core_split_method = method
    cfg.source_core_split_anchor_mode = anchor_mode
    cfg.source_core_split_min_core_voxels = 4
    cfg.source_core_split_min_reference_components = 1
    cfg.source_core_split_min_core_containment = 0.80
    cfg.source_core_split_min_core_separation_dref = 0.50
    cfg.source_core_split_min_child_fraction = 0.10
    cfg.source_core_split_confidence_threshold = threshold
    cfg.source_core_split_volume_ratio_center = 1.20
    cfg.source_core_split_supervoxel_min_anchor_voxels = 1
    return SourceCoreSplitOnlyFilter(cfg)(
        [torch.as_tensor(final, dtype=torch.long)],
        torch.as_tensor(source[None], dtype=torch.float32),
        torch.as_tensor(separator[None], dtype=torch.float32),
        torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        torch.tensor([4.0], dtype=torch.float32),
        supervoxel_labels=(
            None
            if supervoxels is None
            else [torch.as_tensor(supervoxels, dtype=torch.long)]
        ),
        source_instance_labels=(
            None
            if source_instances is None
            else torch.as_tensor(source_instances[None], dtype=torch.long)
        ),
    )


def test_default_method_is_supervoxel_graph_and_filter_remains_toggleable():
    cfg = InferenceConfig()
    assert cfg.source_core_split_method == "supervoxel_graph"
    assert cfg.source_core_split_anchor_mode == "prefer_source_instances"
    # The architecture can still be globally/per-call switched off as before.
    assert cfg.source_core_split_enabled is False


def test_supervoxel_graph_splits_multi_mask_final_component():
    final, source, separator, supervoxels = _case()
    state = _filter(
        final,
        source,
        separator,
        supervoxels=supervoxels,
    )
    assert state.applied_count >= 1

    output = state.labels[0].numpy()
    ids = np.unique(output[final == 4])
    assert len(ids[ids > 0]) >= 2

    row = next(
        r for r in state.records
        if r.get("final_component_id") == 4
        and r.get("status") == "applied"
    )
    assert row["method"] == "supervoxel_graph"
    assert row["supervoxel_count"] >= 4
    assert row["cut_supervoxel_edge_count"] >= 1


def test_supervoxel_graph_never_cuts_through_an_atomic_supervoxel():
    final, source, separator, supervoxels = _case()
    state = _filter(
        final,
        source,
        separator,
        supervoxels=supervoxels,
    )
    output = state.labels[0].numpy()

    for sv_id in np.unique(supervoxels[supervoxels > 0]):
        old_ids = np.unique(final[supervoxels == sv_id])
        if len(old_ids[old_ids > 0]) != 1:
            continue
        new_ids = np.unique(output[supervoxels == sv_id])
        new_ids = new_ids[new_ids > 0]
        assert len(new_ids) <= 1


def test_legacy_voxel_watershed_is_still_available():
    final, source, separator, _ = _case()
    state = _filter(
        final,
        source,
        separator,
        method="voxel_watershed",
        supervoxels=None,
    )
    assert state.applied_count >= 1
    row = next(
        r for r in state.records
        if r.get("final_component_id") == 4
        and r.get("status") == "applied"
    )
    assert row["method"] == "voxel_watershed"


def test_merged_source_mask_never_merges_graph_separated_components():
    shape = (6, 16, 16)
    final = np.zeros(shape, np.int64)
    final[:, 2:8, 2:7] = 1
    final[:, 2:8, 9:14] = 2

    source = np.zeros(shape, np.float32)
    source[2:4, 4:6, 4:12] = 1

    separator = np.ones(shape, np.float32)
    supervoxels = final.copy()

    state = _filter(
        final,
        source,
        separator,
        supervoxels=supervoxels,
    )
    output = state.labels[0].numpy()

    left = set(np.unique(output[final == 1]).tolist()) - {0}
    right = set(np.unique(output[final == 2]).tolist()) - {0}
    assert left.isdisjoint(right)
    assert state.applied_count == 0


def test_separator_only_boosts_split_confidence():
    final, source, strong, supervoxels = _case()
    weak = np.zeros_like(strong)

    weak_state = _filter(
        final,
        source,
        weak,
        supervoxels=supervoxels,
        threshold=0.99,
    )
    strong_state = _filter(
        final,
        source,
        strong,
        supervoxels=supervoxels,
        threshold=0.99,
    )

    weak_row = next(
        r for r in weak_state.records
        if r.get("final_component_id") == 4
        and "split_confidence" in r
    )
    strong_row = next(
        r for r in strong_state.records
        if r.get("final_component_id") == 4
        and "split_confidence" in r
    )
    assert strong_row["split_confidence"] >= weak_row["split_confidence"]



# STIRNET_SOURCE_INSTANCE_ANCHOR_SPLIT_ONLY_V1
def test_touching_binary_cells_split_when_two_discrete_source_ids_exist():
    final, _source, separator, supervoxels = _case()

    source_instances = np.zeros(final.shape, dtype=np.int64)
    # Face-touching labels: binary foreground is ONE 6-connected component.
    source_instances[3:6, 9:13, 5:10] = 101
    source_instances[3:6, 9:13, 10:15] = 202
    source_binary = (source_instances > 0).astype(np.float32)

    _components, count = ndi.label(
        source_binary > 0,
        structure=ndi.generate_binary_structure(3, 1),
    )
    assert count == 1

    historical = _filter(
        final,
        source_binary,
        separator,
        supervoxels=supervoxels,
        anchor_mode="binary_components",
    )
    assert historical.applied_count == 0

    upgraded = _filter(
        final,
        source_binary,
        separator,
        supervoxels=supervoxels,
        source_instances=source_instances,
        anchor_mode="prefer_source_instances",
    )
    assert upgraded.applied_count >= 1

    output = upgraded.labels[0].numpy()
    ids = np.unique(output[final == 4])
    assert len(ids[ids > 0]) >= 2

    row = next(
        record
        for record in upgraded.records
        if record.get("final_component_id") == 4
        and record.get("status") == "applied"
    )
    assert row["source_anchor_mode"] == "source_instances"
    assert set(row["source_core_ids"]) == {101, 202}


def test_prefer_source_instances_falls_back_when_labels_are_absent():
    final, source, separator, supervoxels = _case()

    historical = _filter(
        final,
        source,
        separator,
        supervoxels=supervoxels,
        anchor_mode="binary_components",
    )
    fallback = _filter(
        final,
        source,
        separator,
        supervoxels=supervoxels,
        source_instances=None,
        anchor_mode="prefer_source_instances",
    )

    assert historical.applied_count == fallback.applied_count
    assert np.array_equal(
        historical.labels[0].numpy(),
        fallback.labels[0].numpy(),
    )


def test_discrete_source_ids_can_never_merge_final_components():
    shape = (6, 16, 16)
    final = np.zeros(shape, np.int64)
    final[:, 2:8, 2:8] = 1
    final[:, 2:8, 8:14] = 2

    source_instances = np.zeros(shape, np.int64)
    source_instances[2:4, 4:6, 4:8] = 11
    source_instances[2:4, 4:6, 8:12] = 22
    source_binary = (source_instances > 0).astype(np.float32)

    separator = np.ones(shape, np.float32)
    supervoxels = final.copy()

    state = _filter(
        final,
        source_binary,
        separator,
        supervoxels=supervoxels,
        source_instances=source_instances,
        anchor_mode="prefer_source_instances",
    )

    output = state.labels[0].numpy()
    left = set(np.unique(output[final == 1]).tolist()) - {0}
    right = set(np.unique(output[final == 2]).tolist()) - {0}

    assert left.isdisjoint(right)
    assert state.applied_count == 0
