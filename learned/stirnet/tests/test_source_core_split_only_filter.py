from __future__ import annotations

import numpy as np
import torch

from learned.stirnet.model.config import InferenceConfig
from learned.stirnet.model.postprocess.source_core_split import SourceCoreSplitOnlyFilter


def _filter(final, source, separator, *, threshold=0.70):
    cfg = InferenceConfig()
    cfg.source_core_split_min_core_voxels = 4
    cfg.source_core_split_min_reference_components = 1
    cfg.source_core_split_min_core_containment = 0.80
    cfg.source_core_split_min_core_separation_dref = 0.50
    cfg.source_core_split_min_child_fraction = 0.10
    cfg.source_core_split_confidence_threshold = threshold
    cfg.source_core_split_volume_ratio_center = 1.20
    return SourceCoreSplitOnlyFilter(cfg)(
        [torch.as_tensor(final, dtype=torch.long)],
        torch.as_tensor(source[None], dtype=torch.float32),
        torch.as_tensor(separator[None], dtype=torch.float32),
        torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        torch.tensor([4.0], dtype=torch.float32),
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
    return final, source, separator


def test_two_bright_cores_can_split_one_final_component():
    final, source, separator = _case()
    state = _filter(final, source, separator)
    assert state.applied_count >= 1
    output = state.labels[0].numpy()
    ids = np.unique(output[final == 4])
    assert len(ids[ids > 0]) >= 2


def test_merged_source_core_never_merges_graph_separated_components():
    shape = (6, 16, 16)
    final = np.zeros(shape, np.int64)
    final[:, 2:8, 2:7] = 1
    final[:, 2:8, 9:14] = 2
    source = np.zeros(shape, np.float32)
    source[2:4, 4:6, 4:12] = 1  # one connected source bridge
    separator = np.ones(shape, np.float32)
    state = _filter(final, source, separator)
    output = state.labels[0].numpy()
    left = set(np.unique(output[final == 1]).tolist()) - {0}
    right = set(np.unique(output[final == 2]).tolist()) - {0}
    assert left.isdisjoint(right)
    assert state.applied_count == 0


def test_separator_only_boosts_split_confidence():
    final, source, strong = _case()
    weak = np.zeros_like(strong)
    weak_state = _filter(final, source, weak, threshold=0.99)
    strong_state = _filter(final, source, strong, threshold=0.99)
    weak_row = next(r for r in weak_state.records if r.get("final_component_id") == 4)
    strong_row = next(r for r in strong_state.records if r.get("final_component_id") == 4)
    assert strong_row["split_confidence"] >= weak_row["split_confidence"]


def test_default_is_off():
    assert InferenceConfig().source_core_split_enabled is False
