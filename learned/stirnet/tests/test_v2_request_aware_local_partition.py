from __future__ import annotations

import torch

from learned.stirnet.model.partition.local_update import (
    _LocalPartitionTask,
    _cluster_request_tasks,
    _owned_supervoxel_ids,
    reconcile_request_local_labels,
)
from learned.stirnet.model.types import (
    InstanceState,
    RAGState,
    RefinementRequest,
)


def _box(z0, z1, y0, y1, x0, x1):
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def _task(index, kind, core, dependency, owned):
    return _LocalPartitionTask(
        0, index, kind, index, core, dependency, owned
    )


def test_halo_overlap_does_not_coalesce_independent_request_tasks():
    left = _task(
        0, "edge",
        _box(1, 3, 1, 3, 1, 3),
        _box(0, 6, 0, 6, 0, 6),
        (1, 2),
    )
    right = _task(
        1, "edge",
        _box(4, 6, 4, 6, 4, 6),
        _box(2, 8, 2, 8, 2, 8),
        (3, 4),
    )
    assert len(_cluster_request_tasks([left, right])) == 2


def test_core_overlap_does_not_coalesce_independent_requests():
    left = _task(
        0, "edge",
        _box(1, 5, 1, 5, 1, 5),
        _box(0, 6, 0, 6, 0, 6),
        (1, 2),
    )
    right = _task(
        1, "edge",
        _box(3, 7, 3, 7, 3, 7),
        _box(2, 8, 2, 8, 2, 8),
        (3, 4),
    )
    assert len(_cluster_request_tasks([left, right])) == 2


def test_recovery_requests_are_singletons_even_when_cores_overlap():
    left = _task(
        0, "recovery",
        _box(1, 5, 1, 5, 1, 5),
        _box(0, 6, 0, 6, 0, 6),
        (),
    )
    right = _task(
        1, "recovery",
        _box(3, 7, 3, 7, 3, 7),
        _box(2, 8, 2, 8, 2, 8),
        (),
    )
    assert len(_cluster_request_tasks([left, right])) == 2


def test_shared_ownership_still_clusters_requests():
    left = _task(
        0, "split",
        _box(1, 3, 1, 3, 1, 3),
        _box(0, 4, 0, 4, 0, 4),
        (7, 8),
    )
    right = _task(
        1, "edge",
        _box(5, 7, 5, 7, 5, 7),
        _box(4, 8, 4, 8, 4, 8),
        (8, 9),
    )
    assert len(_cluster_request_tasks([left, right])) == 1


def test_edge_ownership_is_exact_endpoint_supervoxels_not_whole_instances():
    node_count = 4
    rag = RAGState(
        node_features=torch.zeros((node_count, 1)),
        node_embeddings=torch.zeros((node_count, 1)),
        node_batch=torch.zeros((node_count,), dtype=torch.long),
        node_supervoxel_id=torch.tensor([10, 11, 12, 13]),
        node_centroid_um=torch.zeros((node_count, 3)),
        node_volume_voxels=torch.ones((node_count,)),
        edge_index=torch.tensor([[0], [2]], dtype=torch.long),
        edge_features=torch.zeros((1, 1)),
        edge_embeddings=torch.zeros((1, 1)),
        spatial_edge_logits=torch.zeros((1,)),
        edge_batch=torch.zeros((1,), dtype=torch.long),
        supervoxel_labels=[torch.zeros((2, 2, 2), dtype=torch.long)],
        node_offsets=torch.tensor([0, node_count], dtype=torch.long),
        statistics=None,
    )
    instances = InstanceState(
        tokens=torch.zeros((2, 1)),
        ref_um=torch.zeros((2, 3)),
        batch_index=torch.zeros((2,), dtype=torch.long),
        local_ids=torch.tensor([1, 2]),
        quality_logits=torch.zeros((2,)),
        labels=[torch.zeros((2, 2, 2), dtype=torch.long)],
        token_offsets=torch.tensor([0, 2], dtype=torch.long),
        node_to_instance=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )
    request = RefinementRequest(
        batch_index=0,
        center_um=torch.zeros((3,)),
        query_token=torch.zeros((1,)),
        kind="edge",
        source_index=0,
        score=1.0,
    )
    assert _owned_supervoxel_ids(request, rag, instances) == (10, 12)


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


def test_protected_labels_are_hard_barriers_for_same_watershed_id():
    base = torch.zeros((1, 3, 7), dtype=torch.long)
    base[:, :, 2] = 1
    base[:, :, 4] = 3
    box = _box(0, 1, 0, 3, 0, 7)
    old = base[box]
    writable = old == 0
    local = torch.ones_like(old)

    updated, reason = reconcile_request_local_labels(
        base, local, box, writable, ()
    )
    assert reason == ""
    assert updated is not None
    assert torch.equal(updated[:, :, 2], base[:, :, 2])
    assert torch.equal(updated[:, :, 4], base[:, :, 4])

    left_id = int(updated[0, 1, 1].item())
    middle_id = int(updated[0, 1, 3].item())
    right_id = int(updated[0, 1, 5].item())
    assert left_id > 3 and middle_id > 3 and right_id > 3
    assert len({left_id, middle_id, right_id}) == 3


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
