from __future__ import annotations

import numpy as np
import torch

from learned.stirnet.model import StirNetConfig
import pytest

from learned.stirnet.model.geometry.edt_backend import (
    cupy_edt_available,
    geometry_edt_backend,
)
from learned.stirnet.model.geometry.targets import build_geometry_targets
from learned.stirnet.training.prepared_geometry import (
    build_prepared_geometry_targets,
    build_static_geometry_targets,
    compose_source_conditioned_geometry_targets,
)


def _scene():
    gt = np.zeros((7, 14, 18), dtype=np.int64)
    gt[2:5, 3:11, 2:6] = 1
    gt[2:5, 3:11, 12:16] = 2
    current = np.zeros_like(gt)
    current[1:6, 2:12, 1:17] = 7
    return (
        torch.from_numpy(gt),
        torch.from_numpy(current),
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor(4.0),
    )


def _cfg():
    cfg = StirNetConfig().geometry
    cfg.separator_target_sigma_um = 1.0
    cfg.separator_source_min_overlap_voxels = 4
    cfg.separator_source_min_gt_fraction = 0.01
    return cfg


def test_static_plus_dynamic_matches_existing_builder():
    gt, current, spacing, dref = _scene()
    cfg = _cfg()
    legacy = build_geometry_targets(
        gt,
        spacing,
        dref,
        current_labels=current,
        separator_target_sigma_um=cfg.separator_target_sigma_um,
        separator_source_min_overlap_voxels=cfg.separator_source_min_overlap_voxels,
        separator_source_min_gt_fraction=cfg.separator_source_min_gt_fraction,
    )
    prepared = build_prepared_geometry_targets(
        gt,
        spacing,
        dref,
        current_labels=current,
        geometry_config=cfg,
        backend="scipy",
    )
    for name in legacy.__dict__:
        a, b = getattr(legacy, name), getattr(prepared, name)
        assert torch.equal(a, b)


def test_corruption_reuses_static_fields_and_changes_only_separator():
    gt, current, spacing, dref = _scene()
    cfg = _cfg()
    static = build_static_geometry_targets(
        gt, spacing, dref, geometry_config=cfg, backend="scipy"
    )
    composed = compose_source_conditioned_geometry_targets(
        static,
        gt,
        current,
        spacing,
        geometry_config=cfg,
        backend="scipy",
    )
    assert float(static.separator.max()) == 0.0
    assert float(composed.separator.max()) > 0.5
    for name in (
        "foreground", "surface", "sdf", "sdf_valid",
        "flow", "centroid_offset", "seed",
    ):
        assert torch.equal(getattr(static, name), getattr(composed, name))


def test_auto_backend_can_be_forced_to_reference_on_small_arrays():
    gt, current, spacing, dref = _scene()
    with geometry_edt_backend("auto", gpu_min_voxels=10**9):
        auto = build_geometry_targets(gt, spacing, dref, current_labels=current)
    reference = build_geometry_targets(gt, spacing, dref, current_labels=current)
    for name in reference.__dict__:
        assert torch.equal(getattr(auto, name), getattr(reference, name))


@pytest.mark.skipif(not cupy_edt_available(), reason="CuPy/CUDA unavailable")
def test_cupy_geometry_matches_scipy_on_non_tied_scene():
    gt, current, spacing, dref = _scene()
    cfg = _cfg()
    reference = build_prepared_geometry_targets(
        gt, spacing, dref, current_labels=current, geometry_config=cfg, backend="scipy"
    )
    gpu = build_prepared_geometry_targets(
        gt, spacing, dref, current_labels=current, geometry_config=cfg,
        backend="cupy", gpu_min_voxels=1
    )
    for name in reference.__dict__:
        a, b = getattr(reference, name), getattr(gpu, name)
        if a.dtype == torch.bool:
            assert torch.equal(a, b)
        else:
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
