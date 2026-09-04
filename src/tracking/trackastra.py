"""Framework-neutral Trackastra execution for the current pipeline.

Persistence belongs to callers such as dataset_curation or Kaggle code.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrackastraConfig:
    model_name: str = "ctc"
    mode: str = "greedy"
    device: str = "cuda"


@dataclass
class TrackastraResult:
    graph: Any
    napari_tracks: np.ndarray
    napari_graph: dict[int, int | list[int]]
    seconds: float
    summary: dict[str, object]


def _normalize_parent_graph(graph: dict) -> dict[int, int | list[int]]:
    result: dict[int, int | list[int]] = {}
    for child, parent in graph.items():
        child_id = int(child)
        if isinstance(parent, (list, tuple, set)):
            result[child_id] = [int(value) for value in parent]
        else:
            result[child_id] = int(parent)
    return result


def run_trackastra(
    raw_movie,
    instance_movie,
    *,
    config: TrackastraConfig = TrackastraConfig(),
) -> TrackastraResult:
    """Run Trackastra on aligned ``(T,Z,Y,X)`` raw and label movies."""
    if tuple(raw_movie.shape) != tuple(instance_movie.shape):
        raise ValueError(
            "Raw/instance movie shape mismatch: "
            f"{raw_movie.shape} vs {instance_movie.shape}"
        )
    if len(tuple(raw_movie.shape)) != 4:
        raise ValueError(
            f"Trackastra expects (T,Z,Y,X), got {tuple(raw_movie.shape)}"
        )
    try:
        from trackastra.model import Trackastra
        from trackastra.tracking.utils import graph_to_napari_tracks
    except ImportError as exc:
        raise RuntimeError(
            "Trackastra is required for primary tracking. "
            "Activate the repository environment containing Trackastra."
        ) from exc

    started = time.perf_counter()
    model = Trackastra.from_pretrained(config.model_name, device=config.device)
    graph, tracked_masks = model.track(raw_movie, instance_movie, mode=config.mode)
    seconds = float(time.perf_counter() - started)

    napari_tracks, napari_graph, _properties = graph_to_napari_tracks(graph)
    napari_tracks = np.asarray(napari_tracks, dtype=np.float64)
    if napari_tracks.ndim != 2 or napari_tracks.shape[1] != 5:
        raise RuntimeError(
            "Expected Trackastra 3-D Napari tracks "
            "[track_id,time,z,y,x], "
            f"got {napari_tracks.shape}"
        )
    parent_graph = _normalize_parent_graph(napari_graph)
    summary: dict[str, object] = {
        "model": str(config.model_name),
        "mode": str(config.mode),
        "device": str(config.device),
        "seconds": seconds,
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "napari_track_rows": int(len(napari_tracks)),
        "napari_tracklets": int(
            np.unique(napari_tracks[:, 0]).size if napari_tracks.size else 0
        ),
        "napari_parent_relations": int(len(parent_graph)),
    }
    del tracked_masks
    return TrackastraResult(
        graph=graph,
        napari_tracks=napari_tracks,
        napari_graph=parent_graph,
        seconds=seconds,
        summary=summary,
    )


__all__ = ["TrackastraConfig", "TrackastraResult", "run_trackastra"]
