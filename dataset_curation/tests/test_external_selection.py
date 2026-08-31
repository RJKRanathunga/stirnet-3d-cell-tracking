import json
from pathlib import Path

from dataset_curation.annotation.selection import (
    annotation_started,
    select_annotation_volume,
    touch_annotation_session,
)
from dataset_curation.catalog import BioHubCatalog


def _make_ready_volume(root: Path, volume_id: str):
    sample = root / "source" / "train" / volume_id
    array = sample / f"{volume_id}.zarr" / "0"
    array.mkdir(parents=True)
    (array / "zarr.json").write_text(
        json.dumps({"shape": [2, 2, 2, 2]}),
        encoding="utf-8",
    )

    run = root / "preprocessed" / "train" / volume_id / "current"
    movies = run / "movies"
    trackastra = run / "trackastra"
    movies.mkdir(parents=True)
    trackastra.mkdir(parents=True)

    import numpy as np

    shape = (2, 2, 2, 2)
    for name, dtype in (
        ("raw.npy", np.uint16),
        ("preprocessed.npy", np.float16),
        ("binary_mask.npy", np.uint8),
        ("source_instances.npy", np.int32),
        ("supervoxels.npy", np.int32),
        ("final_instances.npy", np.int32),
    ):
        np.save(movies / name, np.zeros(shape, dtype=dtype))

    (run / "cells_all.csv").write_text("frame,cell_id\n", encoding="utf-8")
    (run / "_SPATIAL_SUCCESS.json").write_text("{}", encoding="utf-8")

    for name in (
        "track_graph.pkl",
        "tracked_masks.npy",
        "napari_tracks.npy",
        "napari_graph.json",
        "tracks.csv",
        "summary.json",
    ):
        (trackastra / name).write_bytes(b"x")


def test_next_and_resume_use_session_marker(tmp_path: Path):
    _make_ready_volume(tmp_path, "a")
    _make_ready_volume(tmp_path, "b")

    catalog = BioHubCatalog(tmp_path)
    a = catalog.get("a", split="train")

    selected = select_annotation_volume(
        catalog,
        split="train",
        kind="instances",
        run_id="current",
        annotation_set="main",
        next_volume=True,
    )
    assert selected.volume_id == "a"

    touch_annotation_session(
        a,
        kind="instances",
        annotation_set="main",
        run_id="current",
    )
    assert annotation_started(
        a,
        kind="instances",
        annotation_set="main",
    )

    selected = select_annotation_volume(
        catalog,
        split="train",
        kind="instances",
        run_id="current",
        annotation_set="main",
        next_volume=True,
    )
    assert selected.volume_id == "b"

    selected = select_annotation_volume(
        catalog,
        split="train",
        kind="instances",
        run_id="current",
        annotation_set="main",
        resume=True,
    )
    assert selected.volume_id == "a"
