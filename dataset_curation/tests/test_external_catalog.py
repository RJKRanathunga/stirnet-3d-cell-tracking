import json
from pathlib import Path

from dataset_curation.catalog import BioHubCatalog


def _make_volume(root: Path, split: str, volume_id: str, frames: int, gt: bool):
    sample = root / "source" / split / volume_id
    array = sample / f"{volume_id}.zarr" / "0"
    array.mkdir(parents=True)
    (array / "zarr.json").write_text(
        json.dumps({"shape": [frames, 64, 256, 256]}),
        encoding="utf-8",
    )
    (sample / "README.txt").write_text(
        "THIS MUST NOT BE USED FOR DISCOVERY",
        encoding="utf-8",
    )
    if gt:
        gt_root = sample / "ground_truth"
        gt_root.mkdir()
        (gt_root / "ground_truth_nodes.csv").write_text(
            "id\n1\n",
            encoding="utf-8",
        )
        (gt_root / "ground_truth_edges.csv").write_text(
            "source,target\n1,2\n",
            encoding="utf-8",
        )


def test_catalog_discovers_train_and_test_without_readme_dependency(tmp_path: Path):
    _make_volume(tmp_path, "train", "44b6_train", 100, True)
    _make_volume(tmp_path, "test", "44b6_test", 100, False)

    catalog = BioHubCatalog(tmp_path)

    train = catalog.discover("train")
    test = catalog.discover("test")

    assert [item.volume_id for item in train] == ["44b6_train"]
    assert [item.volume_id for item in test] == ["44b6_test"]
    assert train[0].frame_count == 100
    assert test[0].frame_count == 100
    assert train[0].has_ground_truth is True
    assert test[0].has_ground_truth is False
