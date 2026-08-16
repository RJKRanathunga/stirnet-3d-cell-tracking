from __future__ import annotations

import torch
import pytest

from learned.stirnet import StirNet
from learned.stirnet.inference import PostprocessConfig, StirNetRefiner
from learned.stirnet.training.checkpoint import load_checkpoint, save_checkpoint

from .conftest import small_model_config, synthetic_batch


def test_refiner_returns_partition_and_centers():
    model = StirNet(small_model_config()).eval()
    refiner = StirNetRefiner(
        model, device="cpu", postprocess_config=PostprocessConfig(min_object_voxels=1)
    )
    results, output = refiner.refine_batch(synthetic_batch(temporal=True))
    assert len(results) == 1
    assert results[0]["labels"].ndim == 3
    assert results[0]["centers_um"].shape[-1] == 3
    assert results[0]["instance_count"] == results[0]["centers_um"].shape[0]
    assert output.final_labels[0].ndim == 3


def test_v2_checkpoint_contract_and_v1_failure(tmp_path):
    cfg = small_model_config()
    source = StirNet(cfg)
    path = tmp_path / "v2.pt"
    save_checkpoint(path, model=source, step=7, epoch=2, model_config=cfg)
    target = StirNet(small_model_config())
    checkpoint = load_checkpoint(path, target)
    assert checkpoint["architecture"] == "spatial_first_v2"
    assert checkpoint["global_step"] == 7

    old_path = tmp_path / "v1.pt"
    torch.save({"model": source.state_dict(), "step": 3}, old_path)
    with pytest.raises(ValueError, match="intentionally incompatible"):
        load_checkpoint(old_path, target)
