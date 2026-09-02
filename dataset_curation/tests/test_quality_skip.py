from __future__ import annotations

# DATASET_CURATION_CANONICAL_SKIP_V1

from pathlib import Path

import numpy as np
import pytest

from dataset_curation.inference.quality import (
    SourceQualityRejected,
    evaluate_source_mask,
    validate_source_mask,
)
from dataset_curation.paths import BioHubVolumePaths


def test_small_connected_components_pass_quality_gate():
    mask = np.zeros((32, 64, 64), dtype=np.uint8)
    mask[2:8, 2:18, 2:18] = 1
    mask[20:27, 40:55, 40:55] = 1
    metrics = validate_source_mask(0, mask)
    assert metrics.connected_components == 2
    assert metrics.largest_component_voxels < 100_000


def test_near_volume_connected_component_is_rejected():
    mask = np.zeros((32, 64, 64), dtype=np.uint8)
    mask[:30, :, :] = 1
    with pytest.raises(SourceQualityRejected) as captured:
        validate_source_mask(7, mask)
    error = captured.value
    assert error.frame == 7
    assert error.metrics.largest_component_voxels >= 100_000
    assert error.metrics.largest_component_fraction_of_foreground >= 0.50
    assert error.metrics.largest_component_fraction_of_volume >= 0.05


def test_metrics_use_6_connected_components():
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[1, 1, 1] = 1
    metrics = evaluate_source_mask(mask)
    assert metrics.connected_components == 2


def test_skip_marker_is_terminal_path_state(tmp_path: Path):
    paths = BioHubVolumePaths(tmp_path, "train", "sample")
    paths.preprocessed_root.mkdir(parents=True)
    paths.skip_marker.write_text(
        '{"status":"skipped","reason_code":"x"}',
        encoding="utf-8",
    )
    assert paths.inference_skipped()
    assert not paths.inference_complete(frame_count=1)


def test_quality_hook_precedes_source_segmentation_in_production_input():
    root = Path(__file__).resolve().parents[2]
    text = (
        root / "learned" / "stirnet" / "inference" / "spatial_input.py"
    ).read_text(encoding="utf-8")
    validator = text.index("source_mask_validator(")
    segmentation = text.index("source_labels = segment_instances(")
    assert validator < segmentation
