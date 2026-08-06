from __future__ import annotations

import numpy as np
import pandas as pd

from graph_tracking.assignment import solve_assignment
from graph_tracking.config import GraphTrackingConfig


def state(track_id: int, position, *, boundary: bool = False, volume: float = 100.0):
    position = np.asarray(position, dtype=float)
    return {
        "track_id": track_id,
        "last_frame": 0,
        "last_position_physical": position,
        "last_detection": {
            "centroid_z": position[0],
            "centroid_y": position[1],
            "centroid_x": position[2],
            "volume_voxels": volume,
            "touches_boundary": boundary,
            "boundary_faces": "x_max" if boundary else "",
        },
        "relative_motion_confidence_next_frame": 1.0,
    }


def detections(rows):
    return pd.DataFrame([
        {
            "cell_id": index + 1,
            "centroid_z": position[0],
            "centroid_y": position[1],
            "centroid_x": position[2],
            "volume_voxels": volume,
            "touches_boundary": boundary,
            "boundary_faces": face,
        }
        for index, (position, boundary, face, volume) in enumerate(rows)
    ])


def base_assignment(pair_cost, safety_invalid, miss_cost, birth_cost, distance, config=None):
    config = config or GraphTrackingConfig()
    result = solve_assignment(
        np.asarray(pair_cost, dtype=float),
        np.asarray(miss_cost, dtype=float),
        np.asarray(birth_cost, dtype=float),
        safety_invalid=np.asarray(safety_invalid, dtype=bool),
        config=config,
    )
    result["safety_invalid"] = np.asarray(safety_invalid, dtype=bool)
    result["distance_matrix"] = np.asarray(distance, dtype=float)
    return result
