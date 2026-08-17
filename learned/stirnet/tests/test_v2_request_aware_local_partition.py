from __future__ import annotations

import torch

from learned.stirnet.model.partition.local_update import (
    _LocalPartitionTask,
    _cluster_request_tasks,
    reconcile_request_local_labels,
)


def _box(z0, z1, y0, y1, x0, x1):
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def test_halo_overlap_does_not_coalesce_independent_request_tasks():
    left = _LocalPartitionTask(
        0, 0, "edge", 0,
        _box(1, 3, 1, 3, 1, 3),
        _box(0, 6, 0, 6, 0, 6),
        (1, 2),
    )
    right = _LocalPartitionTask(
        0, 1, "edge", 1,
        _box(4, 6, 4, 6, 4, 6),
        _box(2, 8, 2, 8, 2, 8),
        (3, 4),
    )
    assert len(_cluster_request_tasks([left, right])) == 2


def test_shared_ownership_clusters_requests():
    left = _LocalPartitionTask(
        0, 0, "split", 0,
        _box(1, 3, 1, 3, 1, 3),
        _box(0, 4, 0, 4, 0, 4),
        (7, 8),
    )
    right = _LocalPartitionTask(
        0, 1, "edge", 1,
        _box(5, 7, 5, 7, 5, 7),
        _box(4, 8, 4, 8, 4, 8),
        (8, 9),
    )
    assert len(_cluster_request_tasks([left, right])) == 1


def test_owned_labels_can_merge_without_fallback():
    base = torch.zeros((5, 7, 9), dtype=torch.long)
    base[1:4, 2:5, 1:4] = 1
    base[1:4, 2:5, 5:8] = 2
    box = _box(0, 5, 1, 6, 0, 9)
    old = base[box]
    writable = (old == 1) | (old == 2)
    writable[1:4, 1:4, 1:8] = True
    local = torch.zeros_like(old)
    local[1:4, 1:4, 1:8] = 1

    updated, reason = reconcile_request_local_labels(
        base, local, box, writable, (1, 2)
    )
    assert reason == ""
    assert updated is not None
    positive = torch.unique(updated[updated > 0])
    assert positive.numel() == 1
    assert int(positive.item()) in {1, 2}


def test_owned_merge_cannot_absorb_protected_external_object():
    base = torch.zeros((5, 7, 11), dtype=torch.long)
    base[1:4, 2:5, 1:3] = 1
    base[1:4, 2:5, 4:6] = 2
    base[1:4, 2:5, 8:10] = 3
    box = _box(0, 5, 1, 6, 0, 11)
    old = base[box]
    writable = (old == 1) | (old == 2)
    writable[1:4, 1:4, 1:8] |= (old[1:4, 1:4, 1:8] == 0)
    local = torch.zeros_like(old)
    local[1:4, 1:4, 1:10] = 1

    updated, reason = reconcile_request_local_labels(
        base, local, box, writable, (1, 2)
    )
    assert updated is None
    assert reason == "owned component touches protected external label"


def test_owned_label_can_split_when_whole_old_object_is_writable():
    base = torch.zeros((5, 7, 9), dtype=torch.long)
    base[1:4, 2:5, 1:8] = 5
    box = _box(0, 5, 1, 6, 0, 9)
    old = base[box]
    writable = old == 5
    local = torch.zeros_like(old)
    local[1:4, 1:4, 1:3] = 1
    local[1:4, 1:4, 5:8] = 2

    updated, reason = reconcile_request_local_labels(
        base, local, box, writable, (5,)
    )
    assert reason == ""
    assert updated is not None
    positive = torch.unique(updated[updated > 0])
    assert positive.numel() == 2
    assert 5 in positive.tolist()
