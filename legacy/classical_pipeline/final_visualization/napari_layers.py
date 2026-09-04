"""Napari layer construction for final reconciled tracks."""

from __future__ import annotations

import numpy as np

from .step04_napari_data import to_napari_points, to_napari_tracks
from .napari_widget import add_final_failure_navigator


def _add_track_group(
    viewer,
    frame,
    *,
    track_name: str,
    point_name: str,
    color: str,
    scale,
    tail_length: int,
    visible: bool,
):
    if frame.empty:
        return None, None
    track_layer = viewer.add_tracks(
        to_napari_tracks(frame), name=track_name, scale=scale,
        tail_length=tail_length,
    )
    point_layer = viewer.add_points(
        to_napari_points(frame),
        name=point_name,
        scale=scale,
        size=5,
        face_color=color,
        properties={
            "track_id": frame["track_id"].to_numpy(),
            "cell_id": frame["cell_id"].to_numpy(),
        },
    )
    track_layer.visible = visible
    point_layer.visible = visible
    return track_layer, point_layer


def create_final_visualization_viewer(
    visualization,
    *,
    raw_volume,
    preprocessed_volume=None,
    binary_mask_volume=None,
    instance_labels_volume=None,
    config=None,
    add_navigator: bool = True,
):
    """Create a complete Stage 12 viewer while keeping Stage 9 untouched."""

    try:
        import napari
    except ImportError as exc:  # pragma: no cover - depends on GUI environment
        raise RuntimeError("Napari is required to create the Stage 12 viewer") from exc

    tail_length = int(getattr(config, "tail_length", 20))
    show_boundary = bool(getattr(config, "show_boundary_tracks", False))
    show_modified = bool(getattr(config, "show_modified_tracks", False))
    show_valid_events = bool(getattr(config, "show_valid_event_points", False))
    scale = (1.0, *visualization.voxel_size_zyx)
    viewer = napari.Viewer(ndisplay=3)
    raw_limits = [float(np.percentile(raw_volume, 1)), float(np.percentile(raw_volume, 99.8))]
    viewer.add_image(
        raw_volume, name="Raw Volume", scale=scale, rendering="mip",
        colormap="gray", contrast_limits=raw_limits,
    )
    if preprocessed_volume is not None:
        viewer.add_image(
            preprocessed_volume, name="Preprocessed Volume", scale=scale,
            rendering="mip", colormap="gray", contrast_limits=(0.0, 1.0),
            visible=False,
        )
    if binary_mask_volume is not None:
        viewer.add_labels(binary_mask_volume, name="Binary Mask", scale=scale, visible=False)
    if instance_labels_volume is not None:
        viewer.add_labels(instance_labels_volume, name="Instance Labels", scale=scale, visible=False)

    all_track_kwargs = {
        "name": "Final Tracks - all", "scale": scale, "tail_length": tail_length,
    }
    if visualization.lineage_graph:
        all_track_kwargs["graph"] = visualization.lineage_graph
    all_tracks = viewer.add_tracks(visualization.tracks_array, **all_track_kwargs)
    all_tracks.visible = False
    all_points = viewer.add_points(
        visualization.points_array, name="Final Centroids - all", scale=scale,
        size=4, face_color="red",
        properties={
            "track_id": visualization.track_ids,
            "cell_id": visualization.cell_ids,
        },
        text={
            "string": "C{cell_id}", "size": 8, "color": "white",
            "anchor": "center",
        },
    )
    all_points.visible = False

    track_id_labels = viewer.add_points(
        visualization.points_array,
        name="Track IDs",
        scale=scale,
        size=1,
        face_color="transparent",
        border_color="transparent",
        properties={
            "track_id": visualization.track_ids,
            "cell_id": visualization.cell_ids,
        },
        text={
            "string": "T{track_id}",
            "size": 8,
            "color": "white",
            "anchor": "center",
        },
        visible=False,
    )

    groups = visualization.groups
    _add_track_group(viewer, groups.suspicious_termination_tracks,
        track_name="Failure - early terminations", point_name="Failure endpoints - terminations",
        color="red", scale=scale, tail_length=tail_length, visible=True)
    _add_track_group(viewer, groups.suspicious_birth_tracks,
        track_name="Failure - late births", point_name="Failure endpoints - births",
        color="lime", scale=scale, tail_length=tail_length, visible=True)
    _add_track_group(viewer, groups.temporal_gap_tracks,
        track_name="Failure - temporal gaps", point_name="Gap track centroids",
        color="magenta", scale=scale, tail_length=tail_length, visible=True)
    _add_track_group(viewer, groups.unresolved_tracks,
        track_name="Stage 11 - unresolved endings", point_name="Unresolved track centroids",
        color="yellow", scale=scale, tail_length=tail_length, visible=True)
    _add_track_group(viewer, groups.forced_repair_tracks,
        track_name="Stage 11 - forced repairs", point_name="Forced repair centroids",
        color="violet", scale=scale, tail_length=tail_length, visible=False)
    _add_track_group(viewer, groups.stage11_modified_tracks,
        track_name="Stage 11 - modified tracks", point_name="Modified track centroids",
        color="blue", scale=scale, tail_length=tail_length, visible=show_modified)
    _add_track_group(viewer, groups.short_lived_tracks,
        track_name="Audit - short tracks", point_name="Short track centroids",
        color="white", scale=scale, tail_length=tail_length, visible=False)
    _add_track_group(viewer, groups.division_tracks,
        track_name="Valid - division tracks", point_name="Division track centroids",
        color="white", scale=scale, tail_length=tail_length, visible=False)
    _add_track_group(viewer, groups.merge_tracks,
        track_name="Valid - merge-related tracks", point_name="Merge track centroids",
        color="orange", scale=scale, tail_length=tail_length, visible=False)
    _add_track_group(viewer, groups.boundary_entry_tracks,
        track_name="Valid - boundary entries", point_name="Boundary entry centroids",
        color="cyan", scale=scale, tail_length=tail_length, visible=show_boundary)
    _add_track_group(viewer, groups.boundary_exit_tracks,
        track_name="Valid - boundary exits", point_name="Boundary exit centroids",
        color="orange", scale=scale, tail_length=tail_length, visible=show_boundary)

    event_frame = visualization.diagnostic_events
    if not event_frame.empty:
        visible_events = event_frame[event_frame["is_failure"].astype(bool) | (event_frame["severity"] == "warning")]
        valid_events = event_frame[~event_frame.index.isin(visible_events.index)]
        if not visible_events.empty:
            viewer.add_points(
                visible_events[["frame", "z", "y", "x"]].to_numpy(float),
                name="Final diagnostic events", scale=scale, size=8,
                face_color="yellow",
                properties={column: visible_events[column].to_numpy() for column in (
                    "event_id", "track_id", "event_type", "classification", "severity", "reason"
                )},
                text={
                    "string": "{classification}", "size": 8, "color": "white",
                    "anchor": "upper_left",
                },
            )
        if show_valid_events and not valid_events.empty:
            viewer.add_points(
                valid_events[["frame", "z", "y", "x"]].to_numpy(float),
                name="Valid endpoint events", scale=scale, size=6,
                face_color="cyan", visible=False,
                properties={column: valid_events[column].to_numpy() for column in (
                    "event_id", "track_id", "event_type", "classification", "reason"
                )},
            )

    try:
        viewer.camera.angles = (45, 30, 135)
    except Exception:
        pass
    navigator = add_final_failure_navigator(viewer, visualization) if add_navigator else None
    return viewer, navigator
