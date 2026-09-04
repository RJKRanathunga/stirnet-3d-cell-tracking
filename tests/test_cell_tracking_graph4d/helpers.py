from __future__ import annotations

from importlib import import_module

import pandas as pd
from src.api import GraphTrackingConfig, run_cell_tracking


FourDGraphConfig = import_module(
    "legacy.classical_pipeline.tracking.graph_tracking.four_d"
).FourDGraphConfig


def detection(cell_id: int, z: float, y: float, x: float, volume: float = 100.0):
    return {
        "cell_id": cell_id,
        "centroid_z": z,
        "centroid_y": y,
        "centroid_x": x,
        "volume_voxels": volume,
        "z_min": max(0.0, z - 2),
        "y_min": max(0.0, y - 3),
        "x_min": max(0.0, x - 3),
        "z_max": min(64.0, z + 3),
        "y_max": min(256.0, y + 4),
        "x_max": min(256.0, x + 4),
        "extent": 0.8,
        "equivalent_radius": 3.0,
        "elongation": 1.2,
        "flatness": 1.1,
        "anisotropy": 1.3,
        "solidity": 0.9,
        "compactness": 0.8,
        "intensity_mean": 0.5,
        "intensity_std": 0.1,
        "intensity_cv": 0.2,
        "bbox_depth": 5.0,
        "bbox_height": 7.0,
        "bbox_width": 7.0,
    }


def moving_frames(count: int = 4) -> list[pd.DataFrame]:
    return [
        pd.DataFrame([
            detection(1, 20, 90 + frame, 90),
            detection(2, 20, 110 + frame, 90),
            detection(3, 20, 100 + frame, 110),
        ])
        for frame in range(count)
    ]


def ambiguous_gap_frames() -> list[pd.DataFrame]:
    return [
        pd.DataFrame([
            detection(1, 20, 100, 100),
            detection(2, 20, 80, 100),
            detection(3, 20, 120, 100),
        ]),
        pd.DataFrame([
            detection(2, 20, 81, 100),
            detection(3, 20, 121, 100),
        ]),
        pd.DataFrame([
            detection(1, 20, 102, 100),
            detection(2, 20, 82, 100),
            detection(3, 20, 122, 100),
        ]),
    ]
