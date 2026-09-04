"""One-pass Trackastra neural association and graph solving."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import time
from typing import Any

from .config import TrackastraConfig
from .greedy import solve_greedy


@dataclass
class TrackastraPassResult:
    graph: Any
    seconds: float
    summary: dict[str, object]


def load_trackastra_model(config: TrackastraConfig):
    try:
        from trackastra.model import Trackastra
    except ImportError as exc:
        raise RuntimeError(
            "Trackastra is required for primary tracking. "
            "Activate the repository environment containing Trackastra."
        ) from exc

    kwargs: dict[str, object] = {"device": str(config.device)}
    if config.batch_size is not None:
        kwargs["batch_size"] = int(config.batch_size)

    return Trackastra.from_pretrained(str(config.model_name), **kwargs)


def run_trackastra_pass(
    model,
    raw_movie,
    instance_movie,
    *,
    config: TrackastraConfig,
    pass_name: str,
) -> TrackastraPassResult:
    """Run Trackastra prediction without creating a full tracked-mask movie."""
    raw_shape = tuple(int(v) for v in raw_movie.shape)
    label_shape = tuple(int(v) for v in instance_movie.shape)
    if raw_shape != label_shape:
        raise ValueError(
            f"Raw/instance movie shape mismatch: {raw_shape} vs {label_shape}"
        )
    if len(raw_shape) != 4:
        raise ValueError(f"Trackastra expects (T,Z,Y,X), got {raw_shape}")

    started = time.perf_counter()
    predict_started = time.perf_counter()
    predict_kwargs: dict[str, object] = {}
    if config.batch_size is not None:
        predict_kwargs["batch_size"] = int(config.batch_size)

    predictions = model._predict(raw_movie, instance_movie, **predict_kwargs)
    prediction_seconds = float(time.perf_counter() - predict_started)
    prediction_nodes = int(len(predictions["nodes"]))
    prediction_weights = int(len(predictions["weights"]))

    if str(config.mode) in {"greedy", "greedy_nodiv"}:
        from trackastra.tracking.tracking import build_graph

        candidate_started = time.perf_counter()
        candidate_graph = build_graph(
            nodes=predictions["nodes"],
            weights=predictions["weights"],
            use_distance=False,
            max_distance=256,
            max_neighbors=10,
            delta_t=1,
        )
        candidate_seconds = float(time.perf_counter() - candidate_started)
        graph, solver_summary = solve_greedy(
            candidate_graph,
            allow_divisions=(str(config.mode) == "greedy"),
            threshold=0.5,
            edge_attr="weight",
        )
        candidate_edges = int(candidate_graph.number_of_edges())
        del candidate_graph
    else:
        candidate_seconds = None
        candidate_edges = None
        solver_started = time.perf_counter()
        graph = model._track_from_predictions(predictions, mode=str(config.mode))
        solver_summary = {
            "solver": "trackastra_ilp",
            "seconds": float(time.perf_counter() - solver_started),
        }

    del predictions
    gc.collect()

    seconds = float(time.perf_counter() - started)
    summary: dict[str, object] = {
        "pass": str(pass_name),
        "seconds": seconds,
        "prediction_seconds": prediction_seconds,
        "prediction_nodes": prediction_nodes,
        "prediction_weights": prediction_weights,
        "candidate_graph_seconds": candidate_seconds,
        "candidate_edges": candidate_edges,
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        **solver_summary,
    }
    return TrackastraPassResult(graph=graph, seconds=seconds, summary=summary)


__all__ = [
    "TrackastraPassResult",
    "load_trackastra_model",
    "run_trackastra_pass",
]
