from __future__ import annotations

# DATASET_CURATION_CANONICAL_SKIP_V1

import json
from pathlib import Path

import numpy as np
import pytest

from dataset_curation.annotation.selection import (
    annotation_started,
    select_annotation_volume,
    touch_annotation_session,
)
from dataset_curation.catalog import BioHubCatalog
from dataset_curation.errors import ArtifactError


def _make_ready_volume(root: Path, volume_id: str):
    sample = root / "source" / "train" / volume_id
    array = sample / f"{volume_id}.zarr" / "0"
    array.mkdir(parents=True)
    (array / "zarr.json").write_text(
        json.dumps({"shape": [2, 2, 2, 2]}),
        encoding="utf-8",
    )

    run = root / "preprocessed" / "train" / volume_id
    movies = run / "movies"
    trackastra = run / "trackastra"
    movies.mkdir(parents=True)
    trackastra.mkdir(parents=True)

    shape = (2, 2, 2, 2)
    np.save(movies / "supervoxels.npy", np.zeros(shape, dtype=np.uint16))
    np.save(movies / "final_instances.npy", np.zeros(shape, dtype=np.uint16))
    (run / "cells_all.csv").write_text("frame,cell_id\n", encoding="utf-8")
    (run / "_SPATIAL_SUCCESS.json").write_text("{}", encoding="utf-8")
    (run / "curation_manifest.json").write_text(
        json.dumps({"inference_id": f"inference-{volume_id}"}),
        encoding="utf-8",
    )

    for name in (
        "track_graph.pkl",
        "napari_tracks.npy",
        "napari_graph.json",
        "tracks.csv",
        "summary.json",
    ):
        (trackastra / name).write_bytes(b"x")


def _make_skipped_volume(root: Path, volume_id: str):
    sample = root / "source" / "train" / volume_id
    array = sample / f"{volume_id}.zarr" / "0"
    array.mkdir(parents=True)
    (array / "zarr.json").write_text(
        json.dumps({"shape": [2, 2, 2, 2]}),
        encoding="utf-8",
    )
    run = root / "preprocessed" / "train" / volume_id
    run.mkdir(parents=True)
    (run / "_SKIPPED.json").write_text(
        json.dumps(
            {
                "status": "skipped",
                "reason_code": "pathological_connected_foreground",
                "trigger_frame": 0,
            }
        ),
        encoding="utf-8",
    )


def test_next_and_resume_use_session_marker(tmp_path: Path):
    _make_ready_volume(tmp_path, "a")
    _make_ready_volume(tmp_path, "b")

    catalog = BioHubCatalog(tmp_path)
    a = catalog.get("a", split="train")

    selected = select_annotation_volume(
        catalog,
        split="train",
        annotation_set="main",
        next_volume=True,
    )
    assert selected.volume_id == "a"

    touch_annotation_session(a, annotation_set="main")
    assert annotation_started(a, annotation_set="main")

    selected = select_annotation_volume(
        catalog,
        split="train",
        annotation_set="main",
        next_volume=True,
    )
    assert selected.volume_id == "b"

    selected = select_annotation_volume(
        catalog,
        split="train",
        annotation_set="main",
        resume=True,
    )
    assert selected.volume_id == "a"


def test_skipped_volume_is_not_annotation_ready(tmp_path: Path):
    _make_skipped_volume(tmp_path, "bad")
    catalog = BioHubCatalog(tmp_path)

    with pytest.raises(ArtifactError, match="skipped"):
        select_annotation_volume(
            catalog,
            split="train",
            annotation_set="main",
            volume_id="bad",
        )
