"""Compare provisional and windowed-4D Stage 7 results beyond endpoint counts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _edge_set(tracks: pd.DataFrame) -> set[tuple[int, object, int, object]]:
    result: set[tuple[int, object, int, object]] = set()
    for _, group in tracks.groupby("track_id", sort=True):
        ordered = group.sort_values(["frame", "cell"], kind="mergesort")
        values = list(zip(ordered["frame"], ordered["cell"]))
        result.update((int(a[0]), a[1], int(b[0]), b[1]) for a, b in zip(values[:-1], values[1:]))
    return result


def _median_acceleration(tracks: pd.DataFrame, spacing: np.ndarray) -> float:
    values: list[float] = []
    for _, group in tracks.groupby("track_id", sort=True):
        ordered = group.sort_values("frame", kind="mergesort")
        if len(ordered) < 3:
            continue
        positions = ordered[["z", "y", "x"]].to_numpy(dtype=float) * spacing
        frames = ordered["frame"].to_numpy(dtype=float)
        velocities = np.diff(positions, axis=0) / np.diff(frames)[:, None]
        values.extend(np.linalg.norm(np.diff(velocities, axis=0), axis=1).tolist())
    return float(np.median(values)) if values else math.nan


def compare_results(
    *,
    provisional_tracks: pd.DataFrame,
    optimized_tracks: pd.DataFrame,
    temporal_edges: pd.DataFrame,
    component_summary: pd.DataFrame,
    boundary_events: pd.DataFrame,
    runtime_by_phase: dict[str, float] | None = None,
    voxel_size_zyx_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
) -> dict[str, object]:
    provisional = _edge_set(provisional_tracks)
    optimized = _edge_set(optimized_tracks)
    candidates = {
        (
            int(row.source_frame), int(row.source_detection_index),
            int(row.target_frame), int(row.target_detection_index),
        )
        for row in temporal_edges.itertuples(index=False)
    } if not temporal_edges.empty else set()
    selected = temporal_edges.loc[
        temporal_edges["optimized_selected"].astype(bool)
    ] if not temporal_edges.empty else temporal_edges
    changed_from = provisional - optimized
    changed_to = optimized - provisional
    provisional_match_to_death = sum(
        not any(edge[:2] == old[:2] for edge in optimized) for old in changed_from
    )
    provisional_death_to_match = sum(
        not any(edge[:2] == new[:2] for edge in provisional) for new in changed_to
    )
    fallback_count = int(component_summary["fallback_used"].astype(bool).sum()) if not component_summary.empty else 0
    spacing = np.asarray(voxel_size_zyx_um, dtype=float)
    return {
        "candidate_recall": float(len(provisional & candidates) / max(len(provisional), 1)),
        "changed_edge_count": int(len(provisional ^ optimized)),
        "provisional_match_to_optimized_match": int(len(changed_from) - provisional_match_to_death),
        "provisional_match_to_death": int(provisional_match_to_death),
        "provisional_death_to_optimized_match": int(provisional_death_to_match),
        "provisional_birth_to_matched": int(provisional_death_to_match),
        "selected_gap_edges": int((selected["frame_gap"] > 1).sum()) if not selected.empty else 0,
        "graph_expanded_selected_edges": int(selected["graph_expanded"].astype(bool).sum()) if not selected.empty else 0,
        "boundary_entries": int((boundary_events["event_type"] == "boundary_entry").sum()) if not boundary_events.empty else 0,
        "boundary_exits": int((boundary_events["event_type"] == "boundary_exit").sum()) if not boundary_events.empty else 0,
        "provisional_track_fragmentation": int(provisional_tracks["track_id"].nunique()),
        "optimized_track_fragmentation": int(optimized_tracks["track_id"].nunique()),
        "provisional_median_trajectory_acceleration_um": _median_acceleration(provisional_tracks, spacing),
        "optimized_median_trajectory_acceleration_um": _median_acceleration(optimized_tracks, spacing),
        "mean_selected_neighbor_relation_residual": float(
            selected["graph_cost"].mean()
        ) if not selected.empty else math.nan,
        "solver_fallback_rate": float(fallback_count / max(len(component_summary), 1)),
        "runtime_seconds_by_phase": runtime_by_phase or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage7_directory", type=Path)
    parser.add_argument("--provisional-tracks", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.stage7_directory
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    report = compare_results(
        provisional_tracks=pd.read_csv(arguments.provisional_tracks),
        optimized_tracks=pd.read_csv(root / "tracks.csv"),
        temporal_edges=pd.read_csv(root / "graph4d_temporal_edges.csv"),
        component_summary=pd.read_csv(root / "graph4d_component_summary.csv"),
        boundary_events=pd.read_csv(root / "graph4d_boundary_events.csv"),
        runtime_by_phase=metadata.get("graph_tracking", {}).get("runtime_seconds_by_phase", {}),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
