from __future__ import annotations

import torch

from learned.stirnet.model.geometry.targets import GeometryTargets
from learned.stirnet.training.crops import CropBatch
from learned.stirnet.training.spatial_augmentation import (
    XY_FLIP_X,
    XY_FLIP_XY,
    XY_FLIP_Y,
    XY_IDENTITY,
    apply_xy_flip_codes,
    sample_xy_flip_codes,
)


def _fixture():
    scalar = torch.arange(4 * 1 * 2 * 3 * 4).reshape(4, 1, 2, 3, 4).float()
    labels = scalar[:, 0].long()

    flow = torch.zeros((4, 3, 2, 3, 4))
    flow[:, 0] = 10.0
    flow[:, 1] = 20.0
    flow[:, 2] = 30.0
    offset = -flow / 10.0

    targets = GeometryTargets(
        foreground=scalar.clone(),
        surface=scalar.clone() + 1,
        separator=scalar.clone() + 2,
        sdf=scalar.clone() + 3,
        sdf_valid=scalar > 5,
        flow=flow,
        centroid_offset=offset,
        seed=scalar.clone() + 4,
    )
    crop = CropBatch(
        batch={
            "spatial_inputs": scalar.repeat(1, 5, 1, 1, 1),
            "instance_labels": labels.clone(),
            "supervision_valid_mask": torch.ones_like(labels, dtype=torch.bool),
            "spacing_um": torch.ones((4, 3)),
            "dref_um": torch.ones(4),
        },
        gt_labels=labels.clone(),
        geometry_targets=None,
        specs=[],
    )
    return crop, targets


def _expected(row, code):
    dims = []
    if code & XY_FLIP_Y:
        dims.append(-2)
    if code & XY_FLIP_X:
        dims.append(-1)
    return torch.flip(row, dims=dims) if dims else row


def test_explicit_xy_reflections_transform_scalars_and_vector_signs():
    crop, targets = _fixture()
    codes = torch.tensor(
        [XY_IDENTITY, XY_FLIP_X, XY_FLIP_Y, XY_FLIP_XY],
        dtype=torch.uint8,
    )
    changed, transformed, counts = apply_xy_flip_codes(crop, targets, codes)

    assert counts == {
        "identity": 1,
        "flip_x": 1,
        "flip_y": 1,
        "flip_xy": 1,
    }

    for row, code in enumerate(codes.tolist()):
        torch.testing.assert_close(
            changed.gt_labels[row],
            _expected(crop.gt_labels[row], code),
        )
        torch.testing.assert_close(
            transformed.foreground[row],
            _expected(targets.foreground[row], code),
        )

        expected_flow = _expected(targets.flow[row], code).clone()
        expected_offset = _expected(targets.centroid_offset[row], code).clone()
        if code & XY_FLIP_X:
            expected_flow[2] *= -1
            expected_offset[2] *= -1
        if code & XY_FLIP_Y:
            expected_flow[1] *= -1
            expected_offset[1] *= -1

        torch.testing.assert_close(transformed.flow[row], expected_flow)
        torch.testing.assert_close(
            transformed.centroid_offset[row],
            expected_offset,
        )

    torch.testing.assert_close(
        changed.batch["spacing_um"],
        crop.batch["spacing_um"],
    )
    torch.testing.assert_close(
        changed.batch["dref_um"],
        crop.batch["dref_um"],
    )


def test_sampling_is_resume_deterministic():
    a = sample_xy_flip_codes(64, probability=0.5, seed=123456)
    b = sample_xy_flip_codes(64, probability=0.5, seed=123456)
    c = sample_xy_flip_codes(64, probability=0.5, seed=123457)
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, c)


def test_probability_extremes():
    a = sample_xy_flip_codes(8, probability=0.0, seed=1)
    b = sample_xy_flip_codes(8, probability=1.0, seed=1)
    assert bool((a == XY_IDENTITY).all())
    assert bool((b == XY_FLIP_XY).all())
