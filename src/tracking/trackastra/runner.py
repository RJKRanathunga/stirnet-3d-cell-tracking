"""Framework-neutral two-pass Trackastra execution for the current pipeline.

Persistence belongs to callers such as dataset_curation or Kaggle code.
"""

from __future__ import annotations

import gc
import time

import numpy as np

from .config import TrackastraConfig, TrackastraResult
from .engine import load_trackastra_model, run_trackastra_pass
from .global_motion import estimate_global_motion
from .stabilization import build_stabilized_movies, restore_graph_coordinates


TRACKING_SCHEMA_VERSION = 2
BOOTSTRAP_STRATEGY = "bootstrap_global_motion_v1"


def _normalize_parent_graph(graph: dict) -> dict[int, int | list[int]]:
    result: dict[int, int | list[int]] = {}
    for child, parent in graph.items():
        child_id = int(child)
        if isinstance(parent, (list, tuple, set)):
            result[child_id] = [int(value) for value in parent]
        else:
            result[child_id] = int(parent)
    return result


def _graph_to_public_tracks(graph):
    try:
        from trackastra.tracking.utils import graph_to_napari_tracks
    except ImportError as exc:
        raise RuntimeError("Trackastra is required for primary tracking.") from exc

    napari_tracks, napari_graph, _properties = graph_to_napari_tracks(graph)
    napari_tracks = np.asarray(napari_tracks, dtype=np.float64)
    if napari_tracks.ndim != 2 or napari_tracks.shape[1] != 5:
        raise RuntimeError(
            "Expected Trackastra 3-D Napari tracks [track_id,time,z,y,x], "
            f"got {napari_tracks.shape}"
        )
    return napari_tracks, _normalize_parent_graph(napari_graph)


def _cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_trackastra(
    raw_movie,
    instance_movie,
    *,
    config: TrackastraConfig = TrackastraConfig(),
) -> TrackastraResult:
    """Run production Trackastra primary tracking.

    Global-motion mode:
      1. pass 1 in source coordinates;
      2. robust motion estimate from pass-1 unambiguous continuations;
      3. lazy zero-padded stabilization of raw + instance movies;
      4. pass 2 with the same loaded Trackastra model;
      5. transform final graph coordinates back to source coordinates.
    """
    raw_shape = tuple(int(v) for v in raw_movie.shape)
    label_shape = tuple(int(v) for v in instance_movie.shape)
    if raw_shape != label_shape:
        raise ValueError(
            f"Raw/instance movie shape mismatch: {raw_shape} vs {label_shape}"
        )
    if len(raw_shape) != 4:
        raise ValueError(f"Trackastra expects (T,Z,Y,X), got {raw_shape}")

    total_started = time.perf_counter()
    model = load_trackastra_model(config)

    try:
        pass1 = run_trackastra_pass(
            model,
            raw_movie,
            instance_movie,
            config=config,
            pass_name="pass1_source",
        )

        motion = None
        if bool(config.global_motion.enabled) and int(raw_shape[0]) >= 2:
            motion = estimate_global_motion(
                pass1.graph,
                frame_count=int(raw_shape[0]),
                spatial_shape_zyx=(
                    int(raw_shape[1]),
                    int(raw_shape[2]),
                    int(raw_shape[3]),
                ),
                config=config.global_motion,
            )
            stabilized_raw, stabilized_labels = build_stabilized_movies(
                raw_movie,
                instance_movie,
                motion,
            )
            pass2 = run_trackastra_pass(
                model,
                stabilized_raw,
                stabilized_labels,
                config=config,
                pass_name="pass2_stabilized",
            )
            final_graph = restore_graph_coordinates(pass2.graph, motion)
            pass_summaries = (dict(pass1.summary), dict(pass2.summary))
            strategy = BOOTSTRAP_STRATEGY
            del stabilized_raw, stabilized_labels, pass2
        else:
            final_graph = pass1.graph
            pass_summaries = (dict(pass1.summary),)
            strategy = "single_pass"

        napari_tracks, parent_graph = _graph_to_public_tracks(final_graph)
        seconds = float(time.perf_counter() - total_started)

        summary: dict[str, object] = {
            "schema_version": TRACKING_SCHEMA_VERSION,
            "strategy": strategy,
            "model": str(config.model_name),
            "mode": str(config.mode),
            "device": str(config.device),
            "batch_size": (
                int(config.batch_size)
                if config.batch_size is not None
                else None
            ),
            "passes": int(len(pass_summaries)),
            "seconds": seconds,
            "graph_nodes": int(final_graph.number_of_nodes()),
            "graph_edges": int(final_graph.number_of_edges()),
            "napari_track_rows": int(len(napari_tracks)),
            "napari_tracklets": int(
                np.unique(napari_tracks[:, 0]).size if napari_tracks.size else 0
            ),
            "napari_parent_relations": int(len(parent_graph)),
            "tracked_masks_materialized": False,
            "coordinates_restored_to_source": True,
            "global_motion_enabled": bool(config.global_motion.enabled),
            "global_motion": motion.summary() if motion is not None else None,
            "pass_summaries": [dict(row) for row in pass_summaries],
        }

        return TrackastraResult(
            graph=final_graph,
            napari_tracks=napari_tracks,
            napari_graph=parent_graph,
            seconds=seconds,
            summary=summary,
            global_motion=motion,
            pass_summaries=pass_summaries,
        )
    finally:
        del model
        _cleanup_cuda()


__all__ = [
    "BOOTSTRAP_STRATEGY",
    "TRACKING_SCHEMA_VERSION",
    "run_trackastra",
]
