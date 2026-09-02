from __future__ import annotations

from pathlib import Path

import numpy as np

from dataset_curation.paths import BioHubVolumePaths


def test_compact_spatial_contract_accepts_only_uint16_labels(
    tmp_path: Path,
):
    paths = BioHubVolumePaths(
        tmp_path,
        "train",
        "sample",
    )
    run_id = "current"
    paths.movies(run_id).mkdir(
        parents=True,
        exist_ok=True,
    )

    shape = (2, 3, 4, 5)
    np.save(
        paths.supervoxels(run_id),
        np.zeros(shape, dtype=np.uint16),
        allow_pickle=False,
    )
    np.save(
        paths.final_instances(run_id),
        np.zeros(shape, dtype=np.uint16),
        allow_pickle=False,
    )
    paths.cells_csv(run_id).write_text(
        "frame,cell_id\n",
        encoding="utf-8",
    )
    paths.spatial_success(run_id).write_text(
        "{}",
        encoding="utf-8",
    )

    assert paths.spatial_complete(
        run_id,
        frame_count=2,
    )

    np.save(
        paths.final_instances(run_id),
        np.zeros(shape, dtype=np.int32),
        allow_pickle=False,
    )
    assert not paths.spatial_complete(
        run_id,
        frame_count=2,
    )


def test_compact_paths_do_not_expose_ephemeral_full_volume_movies(
    tmp_path: Path,
):
    paths = BioHubVolumePaths(
        tmp_path,
        "train",
        "sample",
    )
    for name in (
        "raw",
        "preprocessed",
        "binary_mask",
        "source_instances",
        "tracked_masks",
    ):
        assert not hasattr(
            paths,
            name,
        )


def test_tracking_complete_does_not_require_tracked_masks(
    tmp_path: Path,
):
    paths = BioHubVolumePaths(
        tmp_path,
        "train",
        "sample",
    )
    run_id = "current"
    paths.trackastra_root(
        run_id
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in (
        paths.track_graph(run_id),
        paths.napari_tracks(run_id),
        paths.napari_graph(run_id),
        paths.tracks_csv(run_id),
        paths.trackastra_summary(run_id),
    ):
        path.write_bytes(b"x")

    assert paths.tracking_complete(
        run_id
    )
    assert not (
        paths.trackastra_root(
            run_id
        )
        / "tracked_masks.npy"
    ).exists()


def test_unified_annotation_preserves_compact_source_contract():
    root = Path(
        __file__
    ).resolve().parents[2]

    sink = (
        root
        / "dataset_curation"
        / "inference"
        / "spatial_sink.py"
    ).read_text(
        encoding="utf-8"
    )
    trackastra = (
        root
        / "dataset_curation"
        / "inference"
        / "trackastra.py"
    ).read_text(
        encoding="utf-8"
    )
    runner = (
        root
        / "dataset_curation"
        / "annotation"
        / "curation_runner.py"
    ).read_text(
        encoding="utf-8"
    )
    source_data = (
        root
        / "dataset_curation"
        / "annotation"
        / "source_data.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "open_memmap" in sink
    assert "dtype=PERSISTED_LABEL_DTYPE" in sink
    assert "self._raw" not in sink
    assert "self._preprocessed" not in sink
    assert "self._binary" not in sink
    assert "self._source" not in sink

    assert "tracked_masks_persisted" in trackastra
    assert "_dask_from_source_zarr" in trackastra
    assert "paths.zarr" in trackastra

    assert "open_source_movie" in runner
    assert "mmap_mode=\"r\"" in runner
    assert "load_raw_and_binary_frames" not in runner
    assert "BinaryMaskFrameCache" in source_data
    assert "load_raw_and_binary_frames" not in source_data
