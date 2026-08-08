from pathlib import Path

import numpy as np
import tifffile

from learned.instance_segmentation.datasets.adapters.c_elegans import CElegansNucleiAdapter


def test_adapter_discovers_and_loads_common_layout(tmp_path: Path) -> None:
    image_dir = tmp_path / "train" / "images"
    mask_dir = tmp_path / "train" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.arange(6 * 8 * 10, dtype=np.uint16).reshape(6, 8, 10)
    labels = np.zeros_like(image, dtype=np.uint16)
    labels[1:3, 2:5, 2:5] = 1
    labels[3:5, 3:6, 6:9] = 2
    tifffile.imwrite(image_dir / "001.tif", image)
    tifffile.imwrite(mask_dir / "001.tif", labels)

    adapter = CElegansNucleiAdapter(tmp_path)
    records = adapter.discover_records()
    assert len(records) == 1
    assert records[0].split == "train"

    volume = adapter.load(records[0])
    assert volume.image.shape == image.shape
    assert volume.instance_labels.shape == labels.shape
    assert tuple(volume.instance_ids.tolist()) == (1, 2)
    assert volume.spacing_zyx_um == (0.122, 0.116, 0.116)


def test_adapter_explicit_axis_order_conversion(tmp_path: Path) -> None:
    image_dir = tmp_path / "val" / "images"
    mask_dir = tmp_path / "val" / "labels"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    # Stored as [X,Y,Z]. Explicit xyz should map it to [Z,Y,X].
    image_xyz = np.zeros((10, 8, 6), dtype=np.uint16)
    labels_xyz = np.zeros_like(image_xyz, dtype=np.uint16)
    labels_xyz[2:5, 2:5, 1:4] = 1
    tifffile.imwrite(image_dir / "002.tif", image_xyz)
    tifffile.imwrite(mask_dir / "002.tif", labels_xyz)

    adapter = CElegansNucleiAdapter(tmp_path, source_axis_order="xyz")
    volume = adapter.load(adapter.discover_records()[0])
    assert volume.shape_zyx == (6, 8, 10)
