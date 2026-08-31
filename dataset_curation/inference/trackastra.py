from __future__ import annotations

import gc
import os
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import torch

from dataset_curation.io.atomic import (
    atomic_json,
)


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


def run_trackastra(
    paths,
    *,
    run_id: str = "current",
    model_name: str = "ctc",
    mode: str = "greedy",
    device: str = "cuda",
    rebuild: bool = False,
) -> None:
    """Production curation adapter around the external Trackastra package."""
    if (
        paths.tracking_complete(run_id)
        and not rebuild
    ):
        print(
            "[trackastra] reusing cached "
            "graph/masks/tracks",
            flush=True,
        )
        return

    output = paths.trackastra_root(
        run_id
    )
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        from trackastra.model import (
            Trackastra,
        )
        from trackastra.tracking.utils import (
            graph_to_napari_tracks,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Trackastra is required for curation tracking. "
            "Activate the repository environment containing "
            "the configured Trackastra package."
        ) from exc

    raw_movie = np.load(
        paths.raw(run_id),
        mmap_mode="r",
        allow_pickle=False,
    )
    final_movie = np.load(
        paths.final_instances(run_id),
        mmap_mode="r",
        allow_pickle=False,
    )

    print("", flush=True)
    print("=" * 96, flush=True)
    print(
        "DATASET CURATION — TRACKASTRA",
        flush=True,
    )
    print("=" * 96, flush=True)
    print(
        f"model     : {model_name}",
        flush=True,
    )
    print(
        f"mode      : {mode}",
        flush=True,
    )
    print(
        f"device    : {device}",
        flush=True,
    )
    print(
        f"raw       : {paths.raw(run_id)}",
        flush=True,
    )
    print(
        f"instances : "
        f"{paths.final_instances(run_id)}",
        flush=True,
    )
    print("=" * 96, flush=True)

    started = time.perf_counter()
    model = Trackastra.from_pretrained(
        model_name,
        device=device,
    )
    track_graph, tracked_masks = (
        model.track(
            raw_movie,
            final_movie,
            mode=mode,
        )
    )
    seconds = (
        time.perf_counter()
        - started
    )

    with paths.track_graph(
        run_id
    ).open("wb") as handle:
        pickle.dump(
            track_graph,
            handle,
        )

    np.save(
        paths.tracked_masks(run_id),
        np.asarray(tracked_masks),
        allow_pickle=False,
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
        paths.napari_tracks(run_id),
        napari_tracks,
        allow_pickle=False,
    )

    serializable_graph = {
        str(int(child)): (
            [int(value) for value in parent]
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
        paths.napari_graph(run_id),
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
        paths.cells_csv(run_id)
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
        paths.tracks_csv(run_id),
        visualization.tracks,
    )

    summary = {
        "model": str(
            model_name
        ),
        "mode": str(
            mode
        ),
        "device": str(
            device
        ),
        "seconds": float(
            seconds
        ),
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
        paths.trackastra_summary(
            run_id
        ),
        summary,
    )

    del model, tracked_masks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"[trackastra] "
        f"nodes={summary['graph_nodes']} "
        f"edges={summary['graph_edges']} "
        f"tracklets="
        f"{summary['napari_tracklets']} "
        f"time={seconds:.1f}s",
        flush=True,
    )
