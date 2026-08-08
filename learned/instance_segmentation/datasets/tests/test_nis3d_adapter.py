from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from learned.instance_segmentation.datasets.adapters.nis3d import NIS3DAdapter


def test_nis3d_discovers_sidecars_spacing_and_confidence_mask(tmp_path: Path) -> None:
    sample_dir = tmp_path / "Zebrafish_1"
    sample_dir.mkdir()

    image = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
    labels = np.zeros_like(image, dtype=np.uint16)
    labels[1:3, 1:3, 1:3] = 7
    confidence = np.zeros_like(image, dtype=np.uint8)
    confidence[0, 0, 0] = 1
    confidence[1:3, 1:3, 1:3] = 4

    tifffile.imwrite(sample_dir / "Data.tif", image, photometric="minisblack")
    tifffile.imwrite(sample_dir / "GroundTruth.tif", labels, photometric="minisblack")
    tifffile.imwrite(sample_dir / "ConfidenceScore.tif", confidence, photometric="minisblack")
    (sample_dir / "Info.txt").write_text("Voxel size: 0.43 um x 0.43 um x 2.5 um\n")

    adapter = NIS3DAdapter(tmp_path)
    records = adapter.discover_records()
    assert len(records) == 1
    volume = adapter.load(records[0])

    assert volume.dataset_name == "nis3d"
    assert volume.spacing_zyx_um == (2.5, 0.43, 0.43)
    assert volume.image.shape == image.shape
    assert volume.instance_labels.dtype == np.int32
    assert int(volume.instance_labels.max()) == 7
    assert not bool(volume.valid_mask[0, 0, 0])
    assert bool(volume.valid_mask[1, 1, 1])
    assert volume.metadata["confidence_counts"][1] == 1


def test_nis3d_axis_conversion_and_spacing_override(tmp_path: Path) -> None:
    sample_dir = tmp_path / "Mouse_unknown"
    sample_dir.mkdir()

    # Stored XYZ; adapter must produce ZYX.
    image_xyz = np.zeros((6, 5, 4), dtype=np.uint8)
    labels_xyz = np.zeros_like(image_xyz, dtype=np.uint16)
    labels_xyz[2:4, 1:3, 1:3] = 2
    tifffile.imwrite(sample_dir / "Data.tif", image_xyz, photometric="minisblack")
    tifffile.imwrite(sample_dir / "gt.tif", labels_xyz, photometric="minisblack")

    adapter = NIS3DAdapter(
        tmp_path,
        source_axis_order="xyz",
        spacing_override_zyx_um=(1.7, 0.4, 0.4),
    )
    volume = adapter.load_index(0)

    assert volume.image.shape == (4, 5, 6)
    assert volume.instance_labels.shape == (4, 5, 6)
    assert volume.spacing_zyx_um == (1.7, 0.4, 0.4)
