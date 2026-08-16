from __future__ import annotations

import torch

from learned.stirnet.model.instances.tokenizer import centers_from_labels
from learned.stirnet.model.utils.tensor_ops import reduce_labeled_voxels


def _reference(labels: torch.Tensor, spacing: torch.Tensor, field: torch.Tensor):
    shape = torch.tensor(labels.shape, dtype=torch.float32)
    axes = [
        (torch.arange(size) - 0.5 * (size - 1)) * step
        for size, step in zip(labels.shape, spacing)
    ]
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    rows = []
    for label_id in range(1, int(labels.max()) + 1):
        mask = labels == label_id
        if not mask.any():
            continue
        xyz = coordinates[mask]
        values = field[mask]
        points = torch.nonzero(mask, as_tuple=False)
        rows.append(
            {
                "count": mask.sum().float(),
                "centroid": xyz.mean(0),
                "variance": xyz.var(0, unbiased=False),
                "minimum": points.min(0).values,
                "maximum": points.max(0).values,
                "mean": values.mean(),
                "max": values.max(),
                "argmax": points[values.argmax()],
            }
        )
    return rows


def test_vectorized_reductions_match_reference_with_noncontiguous_labels():
    torch.manual_seed(13)
    labels = torch.zeros((4, 6, 7), dtype=torch.long)
    labels[0:2, 1:4, 1:3] = 1
    labels[1:4, 3:6, 4:7] = 3
    spacing = torch.tensor([1.7, 0.5, 0.3])
    field = torch.randn(labels.shape)
    stats = reduce_labeled_voxels(
        labels, spacing, fields={"value": field}, argmax_field=field
    )
    reference = _reference(labels, spacing, field)

    assert stats.counts.tolist() == [12.0, 0.0, 27.0]
    for row, label_id in zip(reference, (0, 2)):
        torch.testing.assert_close(stats.counts[label_id], row["count"])
        torch.testing.assert_close(stats.centroid_um[label_id], row["centroid"])
        torch.testing.assert_close(stats.variance_um2[label_id], row["variance"])
        torch.testing.assert_close(stats.min_voxel[label_id], row["minimum"])
        torch.testing.assert_close(stats.max_voxel[label_id], row["maximum"])
        torch.testing.assert_close(stats.field_means["value"][label_id], row["mean"])
        torch.testing.assert_close(stats.field_maxima["value"][label_id], row["max"])
        flat = stats.argmax_flat_index[label_id]
        point = torch.tensor(
            [
                flat // (labels.shape[1] * labels.shape[2]),
                (flat % (labels.shape[1] * labels.shape[2])) // labels.shape[2],
                flat % labels.shape[2],
            ]
        )
        torch.testing.assert_close(point, row["argmax"])
    assert torch.equal(stats.centroid_um[1], torch.zeros(3))


def test_vectorized_centers_are_in_mask_and_use_sdf_argmax():
    labels = torch.zeros((3, 5, 6), dtype=torch.long)
    labels[:, 1:3, 1:3] = 1
    labels[:, 3:5, 3:6] = 2
    sdf = torch.zeros((1, 1, *labels.shape))
    sdf[0, 0, 2, 2, 2] = 7
    sdf[0, 0, 1, 4, 5] = 9
    spacing = torch.tensor([[2.0, 0.5, 0.25]])
    centers = centers_from_labels([labels], spacing, sdf)[0]
    extent = (torch.tensor(labels.shape) - 1) * spacing[0]
    voxels = torch.round((centers + 0.5 * extent) / spacing[0]).long()
    assert voxels.tolist() == [[2, 2, 2], [1, 4, 5]]
    assert [int(labels[tuple(point.tolist())]) for point in voxels] == [1, 2]


def test_vectorized_reduction_handles_empty_and_two_hundred_labels():
    empty = torch.zeros((2, 3, 4), dtype=torch.long)
    stats = reduce_labeled_voxels(empty, torch.ones(3))
    assert stats.counts.shape == (0,)

    labels = torch.arange(1, 201, dtype=torch.long).reshape(5, 5, 8)
    values = torch.linspace(-1, 1, labels.numel()).reshape_as(labels).float()
    stats = reduce_labeled_voxels(
        labels,
        torch.tensor([2.0, 0.4, 0.4]),
        fields={"value": values},
        argmax_field=values,
    )
    assert stats.counts.shape == (200,)
    assert torch.equal(stats.counts, torch.ones(200))
    assert torch.isfinite(stats.centroid_um).all()
    assert torch.isfinite(stats.field_means["value"]).all()
