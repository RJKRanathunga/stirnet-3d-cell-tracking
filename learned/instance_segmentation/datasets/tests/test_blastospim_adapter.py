from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from learned.instance_segmentation.datasets.adapters.blastospim import BlastoSPIMAdapter
from learned.instance_segmentation.datasets.config import BLASTOSPIM_SPACING_ZYX_UM


def test_blastospim_pairs_images_with_corrected_gt_and_infers_split(tmp_path: Path) -> None:
    archive = tmp_path / "BlastoSPIM1_train"
    image_dir = archive / "Images"
    corrected_dir = archive / "Corrected_Segmentation"
    expected_dir = archive / "Expected_Segmentation"
    image_dir.mkdir(parents=True)
    corrected_dir.mkdir(parents=True)
    expected_dir.mkdir(parents=True)

    image = np.arange(5 * 6 * 7, dtype=np.uint16).reshape(5, 6, 7)
    corrected = np.zeros_like(image, dtype=np.uint16)
    corrected[1:4, 2:5, 2:5] = 11
    expected = np.zeros_like(image, dtype=np.uint16)
    expected[1:4, 2:5, 2:5] = 99

    tifffile.imwrite(image_dir / "embryo_001.tif", image, photometric="minisblack")
    tifffile.imwrite(corrected_dir / "embryo_001.tif", corrected, photometric="minisblack")
    tifffile.imwrite(expected_dir / "embryo_001.tif", expected, photometric="minisblack")

    adapter = BlastoSPIMAdapter(tmp_path)
    records = adapter.discover_records()
    assert len(records) == 1
    assert records[0].split == "train"
    assert "Corrected_Segmentation" in str(records[0].labels_path)

    volume = adapter.load(records[0])
    assert volume.spacing_zyx_um == BLASTOSPIM_SPACING_ZYX_UM
    assert int(volume.instance_labels.max()) == 11
    assert volume.metadata["version"] == "1.0"


def test_blastospim_root_can_point_directly_to_archive(tmp_path: Path) -> None:
    archive = tmp_path / "BlastoSPIM2_test_lowSNR"
    image_dir = archive / "raw"
    gt_dir = archive / "ground_truth"
    image_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    image = np.zeros((3, 4, 5), dtype=np.uint8)
    labels = np.zeros_like(image, dtype=np.uint16)
    labels[:, 1:3, 1:3] = 1
    tifffile.imwrite(image_dir / "sample42.tif", image, photometric="minisblack")
    tifffile.imwrite(gt_dir / "sample42.tif", labels, photometric="minisblack")

    adapter = BlastoSPIMAdapter(archive)
    records = adapter.discover_records()
    assert len(records) == 1
    assert records[0].split == "test"
    volume = adapter.load(records[0])
    assert volume.metadata["version"] == "2.0"
