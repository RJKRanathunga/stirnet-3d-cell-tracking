from __future__ import annotations

import numpy as np
import torch

from learned.stirnet.model.geometry.targets import (
    build_geometry_targets,
    build_source_conditioned_separator_interface,
)
from learned.stirnet.training.crops import build_crop_candidate_cache


def _scene_non_touching_merge():
    gt = np.zeros((7, 14, 18), dtype=np.int64)
    gt[2:5, 3:11, 2:6] = 1
    gt[2:5, 3:11, 12:16] = 2

    current = np.zeros_like(gt)
    current[1:6, 2:12, 1:17] = 7
    return gt, current


def _targets(gt, current=None):
    return build_geometry_targets(
        torch.from_numpy(gt),
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor(4.0),
        current_labels=None if current is None else torch.from_numpy(current),
        separator_target_sigma_um=1.0,
        separator_source_min_overlap_voxels=4,
        separator_source_min_gt_fraction=0.01,
    )


def test_non_touching_gt_cells_gain_bounded_separator_sheet_inside_merged_source():
    gt, current = _scene_non_touching_merge()

    baseline = _targets(gt)
    corrected = _targets(gt, current)

    baseline_separator = baseline.separator[0, 0].numpy()
    corrected_separator = corrected.separator[0, 0].numpy()

    assert float(baseline_separator.max()) == 0.0

    bridge = corrected_separator[2:5, 3:11, 6:12]
    per_row = bridge.max(axis=2)
    assert float(corrected_separator.max()) > 0.5
    assert float((per_row > 0.10).mean()) > 0.90

    assert np.all(corrected_separator[current == 0] == 0.0)


def test_disconnected_islands_with_same_source_id_do_not_create_false_sheet():
    gt = np.zeros((7, 14, 18), dtype=np.int64)
    gt[2:5, 3:11, 2:6] = 1
    gt[2:5, 3:11, 12:16] = 2

    current = np.zeros_like(gt)
    current[1:6, 2:12, 1:7] = 9
    current[1:6, 2:12, 11:17] = 9

    targets = _targets(gt, current)
    assert float(targets.separator.max()) == 0.0


def test_single_gt_cell_inside_source_does_not_create_internal_separator():
    gt = np.zeros((7, 14, 18), dtype=np.int64)
    gt[2:5, 3:11, 4:10] = 1

    current = np.zeros_like(gt)
    current[1:6, 2:12, 2:13] = 5

    targets = _targets(gt, current)
    assert float(targets.separator.max()) == 0.0


def test_direct_gt_contact_separator_is_preserved_when_current_already_separates():
    gt = np.zeros((7, 14, 18), dtype=np.int64)
    gt[2:5, 3:11, 2:8] = 1
    gt[2:5, 3:11, 8:14] = 2

    current = np.zeros_like(gt)
    current[2:5, 3:11, 2:8] = 11
    current[2:5, 3:11, 8:14] = 12

    baseline = _targets(gt)
    corrected = _targets(gt, current)

    assert float(baseline.separator.max()) > 0.0
    assert float(corrected.separator.max()) > 0.0
    assert np.all(corrected.separator.numpy() >= baseline.separator.numpy())


def test_hard_source_conditioned_interface_exists_for_non_touching_merge():
    gt, current = _scene_non_touching_merge()
    interface = build_source_conditioned_separator_interface(
        gt,
        current,
        (1.0, 1.0, 1.0),
        min_overlap_voxels=4,
        min_gt_fraction=0.01,
    )
    assert interface.dtype == np.bool_
    assert interface.any()
    assert interface[2:5, 3:11, 6:12].any()


def test_crop_sampler_can_target_source_conditioned_separator_sheet():
    gt, current = _scene_non_touching_merge()
    cache = build_crop_candidate_cache(
        torch.from_numpy(gt)[None],
        torch.from_numpy(current)[None],
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([4.0]),
    )
    candidates = cache.candidates[0]["separator"]
    assert candidates.ndim == 2
    assert candidates.shape[1] == 3
    assert candidates.shape[0] > 0
    assert bool(((candidates[:, 2] >= 6) & (candidates[:, 2] <= 11)).any())
