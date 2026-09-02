from __future__ import annotations

# DATASET_CURATION_CANONICAL_SKIP_V1

import gc
import os
from pathlib import Path
import pickle
import time

import dask.array as da
import numpy as np
import pandas as pd
import torch

from dataset_curation.io.atomic import atomic_json


def _atomic_csv(
    path: Path,
    frame: pd.DataFrame,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    try:
        frame.to_csv(
            tmp,
            index=False,
        )
        os.replace(
            tmp,
            path,
        )
    finally:
        tmp.unlink(
            missing_ok=True
        )


def _dask_from_source_zarr(
    sample_zarr: Path,
):
    from src.io import open_sample

    source = open_sample(
        sample_zarr
    )
    shape = tuple(
        int(v)
        for v in source.shape
    )
    if len(shape) != 4:
        raise ValueError(
            f"Raw source must be (T,Z,Y,X), got {shape}"
        )

    source_chunks = getattr(
        source,
        "chunks",
        None,
    )
    if (
        source_chunks is None
        or len(source_chunks) != 4
    ):
        source_chunks = (
            1,
            *shape[1:],
        )

    raw = da.from_array(
        source,
        chunks=source_chunks,
        asarray=False,
        fancy=False,
    )
    return source, raw


def _dask_from_label_memmap(
    path: Path,
):
    labels = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if labels.ndim != 4:
        raise ValueError(
            f"Final instances must be (T,Z,Y,X), got {labels.shape}"
        )
    if labels.dtype != np.dtype(np.uint16):
        raise TypeError(
            f"Expected compact uint16 final instances, got {labels.dtype}"
        )

    chunks = (
        1,
        *tuple(
            int(v)
            for v in labels.shape[1:]
        ),
    )
    lazy = da.from_array(
        labels,
        chunks=chunks,
        asarray=False,
        fancy=False,
    )
    return labels, lazy


def run_trackastra(
    paths,
    *,
    model_name: str = "ctc",
    mode: str = "greedy",
    device: str = "cuda",
    rebuild: bool = False,
) -> None:
    """
    Production curation adapter around Trackastra.

    Trackastra 0.5.5 supports Dask-backed large-array inference, so the raw
    movie is read lazily from the canonical source Zarr instead of being copied
    into preprocessed/<volume>/movies/raw.npy.
    """
    output = paths.trackastra_root
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # The compact contract never persists a full relabeled Trackastra mask
    # movie. Remove one even when the small graph/tracks cache can be reused.
    stale_tracked_masks = (
        output
        / "tracked_masks.npy"
    )
    if stale_tracked_masks.is_file():
        stale_tracked_masks.unlink()
        print(
            f"[cache] removed obsolete tracked mask movie: "
            f"{stale_tracked_masks}",
            flush=True,
        )

    if (
        paths.tracking_complete()
        and not rebuild
    ):
        print(
            "[trackastra] reusing cached graph/tracks",
            flush=True,
        )
        return

    try:
        from trackastra.model import Trackastra
        from trackastra.tracking.utils import (
            graph_to_napari_tracks,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Trackastra is required for curation tracking. "
            "Activate the repository environment containing "
            "the configured Trackastra package."
        ) from exc

    source_handle, raw_movie = (
        _dask_from_source_zarr(
            paths.zarr
        )
    )
    final_handle, final_movie = (
        _dask_from_label_memmap(
            paths.final_instances
        )
    )

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
        f"instances : {paths.final_instances} "
        "(uint16 memmap/Dask)",
        flush=True,
    )
    print("=" * 96, flush=True)

    started = time.perf_counter()
    model = Trackastra.from_pretrained(
        model_name,
        device=device,
    )
    track_graph, tracked_masks = model.track(
        raw_movie,
        final_movie,
        mode=mode,
    )
    seconds = (
        time.perf_counter()
        - started
    )

    with paths.track_graph.open("wb") as handle:
        pickle.dump(
            track_graph,
            handle,
        )

    (
        napari_tracks,
        napari_graph,
        _properties,
    ) = graph_to_napari_tracks(
        track_graph
    )
    napari_tracks = np.asarray(
        napari_tracks,
        dtype=np.float64,
    )

    if (
        napari_tracks.ndim != 2
        or napari_tracks.shape[1] != 5
    ):
        raise RuntimeError(
            "Expected Trackastra 3-D Napari "
            "tracks [track_id,time,z,y,x], "
            f"got {napari_tracks.shape}"
        )

    np.save(
        paths.napari_tracks,
        napari_tracks,
        allow_pickle=False,
    )

    serializable_graph = {
        str(int(child)): (
            [
                int(value)
                for value in parent
            ]
            if isinstance(
                parent,
                (list, tuple, set),
            )
            else int(parent)
        )
        for child, parent
        in napari_graph.items()
    }
    atomic_json(
        paths.napari_graph,
        serializable_graph,
    )

    tracks_df = pd.DataFrame(
        napari_tracks,
        columns=[
            "track_id",
            "frame",
            "z",
            "y",
            "x",
        ],
    )
    tracks_df["track_id"] = (
        tracks_df["track_id"]
        .astype(np.int64)
    )
    tracks_df["frame"] = (
        tracks_df["frame"]
        .astype(np.int64)
    )

    cells = pd.read_csv(
        paths.cells_csv
    )
    from src.api import (
        prepare_visualization_data,
    )

    visualization = (
        prepare_visualization_data(
            tracks_df,
            cells,
            assign_cell_ids=True,
        )
    )
    _atomic_csv(
        paths.tracks_csv,
        visualization.tracks,
    )

    summary = {
        "model": str(model_name),
        "mode": str(mode),
        "device": str(device),
        "seconds": float(seconds),
        "raw_input": "source_zarr_dask",
        "instance_input": "uint16_memmap_dask",
        "tracked_masks_persisted": False,
        "graph_nodes": int(
            track_graph.number_of_nodes()
        ),
        "graph_edges": int(
            track_graph.number_of_edges()
        ),
        "napari_track_rows": int(
            len(napari_tracks)
        ),
        "napari_tracklets": int(
            np.unique(
                napari_tracks[:, 0]
            ).size
            if napari_tracks.size
            else 0
        ),
        "napari_parent_relations": int(
            len(napari_graph)
        ),
    }
    atomic_json(
        paths.trackastra_summary,
        summary,
    )

    del (
        model,
        tracked_masks,
        raw_movie,
        final_movie,
        source_handle,
        final_handle,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"[trackastra] "
        f"nodes={summary['graph_nodes']} "
        f"edges={summary['graph_edges']} "
        f"tracklets={summary['napari_tracklets']} "
        f"time={seconds:.1f}s",
        flush=True,
    )
