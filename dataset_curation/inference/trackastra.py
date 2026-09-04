"""Dataset-curation persistence adapter for the active Trackastra runner."""

from __future__ import annotations

import gc
import os
from pathlib import Path
import pickle

import dask.array as da
import numpy as np
import pandas as pd
import torch

from dataset_curation.io.atomic import atomic_json
from src.tracking import (
    TrackastraConfig,
    assign_nearest_instance_ids,
    run_trackastra as run_trackastra_core,
)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _dask_from_source_zarr(sample_zarr: Path):
    from src.io import open_sample

    source = open_sample(sample_zarr)
    shape = tuple(int(v) for v in source.shape)
    if len(shape) != 4:
        raise ValueError(f"Raw source must be (T,Z,Y,X), got {shape}")
    source_chunks = getattr(source, "chunks", None)
    if source_chunks is None or len(source_chunks) != 4:
        source_chunks = (1, *shape[1:])
    raw = da.from_array(
        source,
        chunks=source_chunks,
        asarray=False,
        fancy=False,
    )
    return source, raw


def _dask_from_label_memmap(path: Path):
    labels = np.load(path, mmap_mode="r", allow_pickle=False)
    if labels.ndim != 4:
        raise ValueError(f"Final instances must be (T,Z,Y,X), got {labels.shape}")
    if labels.dtype != np.dtype(np.uint16):
        raise TypeError(f"Expected compact uint16 final instances, got {labels.dtype}")
    chunks = (1, *tuple(int(v) for v in labels.shape[1:]))
    lazy = da.from_array(labels, chunks=chunks, asarray=False, fancy=False)
    return labels, lazy


def run_trackastra(
    paths,
    *,
    model_name: str = "ctc",
    mode: str = "greedy",
    device: str = "cuda",
    rebuild: bool = False,
) -> None:
    """Run primary tracking and persist curation-owned artifacts."""
    output = paths.trackastra_root
    output.mkdir(parents=True, exist_ok=True)

    stale_tracked_masks = output / "tracked_masks.npy"
    if stale_tracked_masks.is_file():
        stale_tracked_masks.unlink()
        print(
            f"[cache] removed obsolete tracked mask movie: {stale_tracked_masks}",
            flush=True,
        )

    if paths.tracking_complete() and not rebuild:
        print("[trackastra] reusing cached graph/tracks", flush=True)
        return

    source_handle, raw_movie = _dask_from_source_zarr(paths.zarr)
    final_handle, final_movie = _dask_from_label_memmap(paths.final_instances)
    if tuple(raw_movie.shape) != tuple(final_movie.shape):
        raise ValueError(
            "Raw source/final-instance movie shape mismatch: "
            f"{raw_movie.shape} vs {final_movie.shape}"
        )

    print("", flush=True)
    print("=" * 96, flush=True)
    print("DATASET CURATION — TRACKASTRA", flush=True)
    print("=" * 96, flush=True)
    print(f"model     : {model_name}", flush=True)
    print(f"mode      : {mode}", flush=True)
    print(f"device    : {device}", flush=True)
    print(f"raw       : {paths.zarr} (lazy Dask/Zarr)", flush=True)
    print(
        f"instances : {paths.final_instances} (uint16 memmap/Dask)",
        flush=True,
    )
    print("=" * 96, flush=True)

    result = run_trackastra_core(
        raw_movie,
        final_movie,
        config=TrackastraConfig(
            model_name=str(model_name),
            mode=str(mode),
            device=str(device),
        ),
    )

    with paths.track_graph.open("wb") as handle:
        pickle.dump(result.graph, handle)
    np.save(paths.napari_tracks, result.napari_tracks, allow_pickle=False)
    atomic_json(
        paths.napari_graph,
        {str(child): parent for child, parent in result.napari_graph.items()},
    )

    tracks_df = pd.DataFrame(
        result.napari_tracks,
        columns=["track_id", "frame", "z", "y", "x"],
    )
    tracks_df["track_id"] = tracks_df["track_id"].astype(np.int64)
    tracks_df["frame"] = tracks_df["frame"].astype(np.int64)
    cells = pd.read_csv(paths.cells_csv)
    tracks_with_instances = assign_nearest_instance_ids(
        tracks_df,
        cells,
        output_column="cell_id",
    )
    _atomic_csv(paths.tracks_csv, tracks_with_instances)

    summary = dict(result.summary)
    summary.update(
        {
            "raw_input": "source_zarr_dask",
            "instance_input": "uint16_memmap_dask",
            "tracked_masks_persisted": False,
        }
    )
    atomic_json(paths.trackastra_summary, summary)

    del result, raw_movie, final_movie, source_handle, final_handle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "[trackastra] "
        f"nodes={summary['graph_nodes']} "
        f"edges={summary['graph_edges']} "
        f"tracklets={summary['napari_tracklets']} "
        f"time={float(summary['seconds']):.1f}s",
        flush=True,
    )


__all__ = ["run_trackastra"]
