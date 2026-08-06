"""Build compact, stable observation nodes from all detection frames."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..geometry import FACE_NAMES, distances_to_volume_faces
from .config import FourDGraphConfig
from .types import ObservationStore


_POSITION_COLUMNS = ("centroid_z", "centroid_y", "centroid_x")
_FEATURE_COLUMNS = (
    "volume_voxels", "intensity_mean", "intensity_median", "intensity_std",
    "intensity_iqr", "intensity_cv", "intensity_sum", "equivalent_radius",
    "axis_major", "axis_middle", "axis_minor", "elongation", "flatness",
    "anisotropy", "solidity", "compactness", "bbox_depth", "bbox_height",
    "bbox_width", "segmentation_reliability", "detection_reliability",
)


def _faces_from_row(
    row: pd.Series,
    centroid: np.ndarray,
    spatial_shape_zyx: np.ndarray,
) -> tuple[str, ...]:
    raw = row.get("boundary_faces", "")
    if pd.notna(raw) and str(raw).strip():
        return tuple(sorted(value for value in str(raw).split("|") if value))
    result: list[str] = []
    bbox = {
        "z_min": row.get("z_min", centroid[0]),
        "z_max": row.get("z_max", centroid[0]),
        "y_min": row.get("y_min", centroid[1]),
        "y_max": row.get("y_max", centroid[1]),
        "x_min": row.get("x_min", centroid[2]),
        "x_max": row.get("x_max", centroid[2]),
    }
    for axis, name in enumerate(("z", "y", "x")):
        if float(bbox[f"{name}_min"]) <= 0.0:
            result.append(f"{name}_min")
        if float(bbox[f"{name}_max"]) >= float(spatial_shape_zyx[axis] - 1):
            result.append(f"{name}_max")
    return tuple(result)


def build_observations(
    *,
    time_frames: list[pd.DataFrame],
    provisional_tracks: pd.DataFrame,
    spatial_shape_zyx: np.ndarray,
    voxel_size_zyx_um: np.ndarray,
    config: FourDGraphConfig,
) -> ObservationStore:
    """Create deterministic ``(frame, detection_index)`` nodes."""

    shape = np.asarray(spatial_shape_zyx, dtype=float)
    spacing = np.asarray(voxel_size_zyx_um, dtype=float)
    if shape.shape != (3,) or spacing.shape != (3,):
        raise ValueError("spatial shape and voxel size must have shape (3,)")

    track_lookup: dict[tuple[int, object], pd.Series] = {}
    if not provisional_tracks.empty:
        for _, row in provisional_tracks.iterrows():
            track_lookup[(int(row["frame"]), row["cell"])] = row

    records: list[dict[str, object]] = []
    frames: list[int] = []
    detection_indices: list[int] = []
    cell_ids: list[int] = []
    positions: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    volumes: list[float] = []
    boundary_flags: list[bool] = []
    boundary_faces: list[tuple[str, ...]] = []
    face_distances: list[np.ndarray] = []
    provisional_ids: list[int] = []
    provisional_confidences: list[float] = []
    reliabilities: list[float] = []
    node_by_key: dict[tuple[int, int], int] = {}
    nodes_by_frame: dict[int, np.ndarray] = {}

    for frame_index, detections in enumerate(time_frames):
        missing = [column for column in _POSITION_COLUMNS if column not in detections]
        if missing:
            raise ValueError(f"Frame {frame_index} is missing centroid columns: {missing}")
        frame_nodes: list[int] = []
        for detection_index, (_, row) in enumerate(detections.iterrows()):
            node_index = len(records)
            centroid = row[list(_POSITION_COLUMNS)].to_numpy(dtype=float)
            position = centroid * spacing
            volume = float(row.get("volume_voxels", 1.0))
            if not math.isfinite(volume) or volume <= 0:
                volume = 1.0
            faces = _faces_from_row(row, centroid, shape)
            boundary = bool(row.get("touches_boundary", bool(faces)))
            distances = distances_to_volume_faces(position, shape, spacing)
            cell_label = detections.index[detection_index]
            track_row = track_lookup.get((frame_index, cell_label))
            provisional_id = int(track_row["track_id"]) if track_row is not None else -1
            confidence = (
                float(track_row.get("association_probability", math.nan))
                if track_row is not None else math.nan
            )
            if not math.isfinite(confidence):
                confidence = 1.0 if frame_index == 0 else 0.0
            reliability = float(row.get("segmentation_reliability", 1.0))
            if not math.isfinite(reliability):
                reliability = 1.0
            if volume <= config.small_cell_volume_threshold:
                reliability *= max(0.25, volume / config.small_cell_volume_threshold)

            record: dict[str, object] = {
                "node_index": node_index,
                "frame": frame_index,
                "detection_index": detection_index,
                "cell_index": cell_label,
                "cell_id": int(row.get("cell_id", detection_index + 1)),
                "position_z_um": float(position[0]),
                "position_y_um": float(position[1]),
                "position_x_um": float(position[2]),
                "centroid_z": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_x": float(centroid[2]),
                "volume_voxels": volume,
                "touches_boundary": boundary,
                "boundary_faces": "|".join(faces),
                "provisional_track_id": provisional_id,
                "provisional_association_confidence": confidence,
                "small_cell_reliability": reliability,
            }
            for face_index, face in enumerate(FACE_NAMES):
                record[f"distance_{face}_um"] = float(distances[face_index])
            for column in _FEATURE_COLUMNS:
                if column not in record:
                    record[column] = row.get(column, math.nan)
            records.append(record)
            frames.append(frame_index)
            detection_indices.append(detection_index)
            cell_ids.append(int(record["cell_id"]))
            positions.append(position)
            centroids.append(centroid)
            volumes.append(volume)
            boundary_flags.append(boundary)
            boundary_faces.append(faces)
            face_distances.append(distances)
            provisional_ids.append(provisional_id)
            provisional_confidences.append(confidence)
            reliabilities.append(reliability)
            node_by_key[(frame_index, detection_index)] = node_index
            frame_nodes.append(node_index)
        nodes_by_frame[frame_index] = np.asarray(frame_nodes, dtype=np.int32)

    return ObservationStore(
        table=pd.DataFrame(records),
        frames=np.asarray(frames, dtype=np.int32),
        detection_indices=np.asarray(detection_indices, dtype=np.int32),
        cell_ids=np.asarray(cell_ids, dtype=np.int64),
        positions_zyx_um=np.asarray(positions, dtype=float).reshape(-1, 3),
        centroids_zyx_voxel=np.asarray(centroids, dtype=float).reshape(-1, 3),
        volumes=np.asarray(volumes, dtype=float),
        boundary_flags=np.asarray(boundary_flags, dtype=bool),
        boundary_faces=tuple(boundary_faces),
        distances_to_faces_um=np.asarray(face_distances, dtype=float).reshape(-1, 6),
        provisional_track_ids=np.asarray(provisional_ids, dtype=np.int64),
        provisional_confidences=np.asarray(provisional_confidences, dtype=float),
        small_cell_reliability=np.asarray(reliabilities, dtype=float),
        node_by_frame_detection=node_by_key,
        nodes_by_frame=nodes_by_frame,
    )

