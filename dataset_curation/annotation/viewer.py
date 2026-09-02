from __future__ import annotations

# DATASET_CURATION_EMPTY_TRACKS_SAFE_V1

"""Unified Napari viewer for spatial and track annotation."""

from dataclasses import dataclass
from typing import Any

import napari
import numpy as np
import pandas as pd

from dataset_curation.annotation.instances.centers import (
    frame_instance_centers,
    supervoxel_interior_points,
)
from dataset_curation.annotation.instances.session import (
    AnnotationSession,
    parse_supervoxel_group,
)
from dataset_curation.annotation.instances.split import AnnotationError
from dataset_curation.annotation.layers import (
    apply_label_color_dict,
    edges_to_tracks_array,
    filter_track_rows_for_completed,
    label_color_dict,
    nodes_to_points_array,
    ray_pick_frontmost_label,
    track_frame_arrays,
)
from dataset_curation.annotation.source_data import (
    BinaryMaskFrameCache,
    estimate_contrast_limits,
)
from dataset_curation.annotation.tracks.diagnostics import (
    CATEGORY_COLORS,
    EndpointTrackGroups,
    diagnostic_node_categories,
)
from dataset_curation.annotation.tracks.graph import Node
from dataset_curation.annotation.tracks.session import TrackAnnotationSession

try:
    from magicgui.widgets import (
        Container,
        Label,
        LineEdit,
        PushButton,
    )
except ImportError as exc:
    raise ImportError(
        "magicgui is required for dataset curation. It normally ships with "
        "Napari."
    ) from exc

try:
    from qtpy.QtWidgets import QSizePolicy
except ImportError:
    QSizePolicy = None


MODE_SPATIAL = "spatial"
MODE_TRACKING = "tracking"
SUPERVOXEL_CONTOUR_WIDTH = 1

_CATEGORY_RGBA = {
    "default": (1.0, 1.0, 1.0, 1.0),
    "broken": (1.0, 0.0, 0.0, 1.0),
    "new": (0.0, 1.0, 0.0, 1.0),
    "boundary_entry": (0.0, 1.0, 1.0, 1.0),
    "boundary_exit": (1.0, 0.55, 0.0, 1.0),
}


def _make_label_shrinkable(widget) -> None:
    native = getattr(
        widget,
        "native",
        None,
    )
    if native is None:
        return
    try:
        native.setWordWrap(True)
    except Exception:
        pass
    try:
        native.setMinimumWidth(0)
    except Exception:
        pass
    if QSizePolicy is not None:
        try:
            native.setSizePolicy(
                QSizePolicy.Ignored,
                QSizePolicy.Preferred,
            )
        except Exception:
            pass


def _configure_dock(dock, panel) -> None:
    native = getattr(
        panel,
        "native",
        None,
    )
    if native is not None:
        try:
            native.setMinimumWidth(300)
        except Exception:
            pass
    try:
        dock.setMinimumWidth(320)
        dock.setMaximumWidth(16_777_215)
    except Exception:
        pass


def _sync_tracks_layer(
    viewer,
    layer,
    data: np.ndarray,
    *,
    name: str,
    scale_tzyx,
    tail_length: int,
    visible: bool,
):
    # Napari Tracks cannot safely construct/update from shape (0, 5) in
    # some versions. Empty track sets are valid curation states, so represent
    # them by absence of the Tracks layer and recreate the layer when rows
    # reappear.
    array = np.asarray(
        data,
        dtype=np.float64,
    )
    if (
        array.ndim != 2
        or array.shape[1] != 5
    ):
        raise ValueError(
            f"{name}: expected tracks array with shape (N,5), "
            f"got {array.shape}"
        )

    if array.shape[0] == 0:
        if layer is not None:
            try:
                viewer.layers.remove(
                    layer
                )
            except (
                ValueError,
                KeyError,
            ):
                pass
        return None

    if layer is None:
        layer = viewer.add_tracks(
            array,
            name=name,
            scale=scale_tzyx,
            tail_length=tail_length,
        )
    else:
        layer.data = array
        layer.refresh()

    layer.visible = bool(
        visible
    )
    return layer


def _track_group_layers(
    viewer,
    frame: pd.DataFrame,
    *,
    track_name: str,
    point_name: str,
    point_color: str,
    scale_tzyx,
    visible: bool,
    tail_length: int,
):
    tracks_array, points_array, properties = (
        track_frame_arrays(frame)
    )
    track_layer = _sync_tracks_layer(
        viewer,
        None,
        tracks_array,
        name=track_name,
        scale_tzyx=scale_tzyx,
        tail_length=tail_length,
        visible=visible,
    )
    point_layer = viewer.add_points(
        points_array,
        name=point_name,
        scale=scale_tzyx,
        size=4,
        face_color=point_color,
        properties=properties,
    )
    point_layer.visible = bool(visible)
    return track_layer, point_layer


@dataclass
class _DiagnosticLayerGroup:
    frame: pd.DataFrame
    track_name: str
    track_layer: Any | None
    track_visible: bool
    point_layer: Any


def make_viewer(
    *,
    sample_id: str,
    raw,
    binary_cache: BinaryMaskFrameCache,
    supervoxels,
    spatial_session: AnnotationSession,
    track_session: TrackAnnotationSession,
    track_centers: dict[Node, np.ndarray],
    original_tracks: pd.DataFrame,
    diagnostics: EndpointTrackGroups,
    spacing_zyx: tuple[float, float, float],
) -> napari.Viewer:
    frame_count = int(raw.shape[0])
    spatial_shape = tuple(
        int(v)
        for v in raw.shape[-3:]
    )
    scale_tzyx = (
        1.0,
        *tuple(float(v) for v in spacing_zyx),
    )

    viewer = napari.Viewer(
        ndisplay=3
    )
    viewer.dims.axis_labels = (
        "time",
        "z",
        "y",
        "x",
    )

    low, high = estimate_contrast_limits(raw)
    raw_layer = viewer.add_image(
        raw,
        name="Raw BioHub",
        scale=scale_tzyx,
        rendering="mip",
        colormap="gray",
        contrast_limits=[
            float(low),
            float(high),
        ],
    )

    binary_layer = viewer.add_labels(
        np.zeros(
            spatial_shape,
            dtype=np.uint8,
        ),
        name="Stage-6 binary mask",
        scale=spacing_zyx,
        opacity=0.30,
        visible=False,
    )

    initial_frame = spatial_session.frame(0)
    corrected_layer = viewer.add_labels(
        initial_frame,
        name="Corrected instances",
        scale=spacing_zyx,
        opacity=1.0,
    )
    apply_label_color_dict(
        corrected_layer,
        label_color_dict(initial_frame),
    )

    initial_supervoxels = np.asarray(
        supervoxels[0]
    )
    supervoxel_layer = viewer.add_labels(
        initial_supervoxels,
        name="Atomic supervoxel boundaries",
        scale=spacing_zyx,
        opacity=0.95,
    )
    apply_label_color_dict(
        supervoxel_layer,
        label_color_dict(initial_supervoxels),
    )
    try:
        supervoxel_layer.contour = (
            SUPERVOXEL_CONTOUR_WIDTH
        )
    except Exception:
        supervoxel_layer.opacity = 0.25

    sv_points, sv_properties = (
        supervoxel_interior_points(
            initial_supervoxels,
            hidden_ids=(),
            spacing_zyx=spacing_zyx,
        )
    )
    supervoxel_id_layer = viewer.add_points(
        sv_points,
        name="Supervoxel IDs",
        scale=spacing_zyx,
        size=1,
        face_color="transparent",
        properties=sv_properties,
        text={
            "string": "{sv_id}",
            "size": 11,
            "color": "white",
            "anchor": "center",
        },
    )
    try:
        supervoxel_id_layer.out_of_slice_display = False
    except Exception:
        pass

    cell_centers_layer = viewer.add_points(
        np.zeros(
            (0, 3),
            dtype=np.float32,
        ),
        name="Cell instance centers",
        scale=spacing_zyx,
        size=4.5,
        face_color="white",
        properties={
            "instance_id": np.zeros(
                (0,),
                dtype=np.int64,
            ),
            "category": np.zeros(
                (0,),
                dtype=object,
            ),
        },
        text={
            "string": "{instance_id}",
            "size": 7,
            "color": "white",
            "anchor": "center",
        },
    )

    seed_highlight_layers = []
    for layer_name, colormap in (
        ("Seed 1 highlight", "red"),
        ("Seed 2 highlight", "blue"),
        ("Seed 3 highlight", "green"),
        ("Seed 4 highlight", "magenta"),
    ):
        seed_highlight_layers.append(
            viewer.add_image(
                np.zeros(
                    spatial_shape,
                    dtype=np.uint8,
                ),
                name=layer_name,
                scale=spacing_zyx,
                colormap=colormap,
                contrast_limits=(0, 1),
                opacity=0.78,
                blending="additive",
            )
        )

    track_selection_a_layer = viewer.add_image(
        np.zeros(
            spatial_shape,
            dtype=np.uint8,
        ),
        name="Track Cell A selection",
        scale=spacing_zyx,
        colormap="yellow",
        contrast_limits=(0, 1),
        opacity=0.88,
        blending="additive",
    )
    track_selection_b_layer = viewer.add_image(
        np.zeros(
            spatial_shape,
            dtype=np.uint8,
        ),
        name="Track Cell B selection",
        scale=spacing_zyx,
        colormap="magenta",
        contrast_limits=(0, 1),
        opacity=0.88,
        blending="additive",
    )

    # ------------------------------------------------------------------
    # Notebook-09 Trackastra visualization
    # ------------------------------------------------------------------
    all_tracks_array, all_points_array, all_properties = (
        track_frame_arrays(original_tracks)
    )
    all_tracks_layer = _sync_tracks_layer(
        viewer,
        None,
        all_tracks_array,
        name="Tracks - all",
        scale_tzyx=scale_tzyx,
        tail_length=frame_count,
        visible=False,
    )

    all_centers_layer = viewer.add_points(
        all_points_array,
        name="Centroids - all",
        scale=scale_tzyx,
        size=4,
        face_color="red",
        properties=all_properties,
        text={
            "string": "{cell_id}",
            "size": 8,
            "color": "white",
            "anchor": "center",
        },
    )
    all_centers_layer.visible = False

    diagnostic_layers: dict[
        str,
        _DiagnosticLayerGroup,
    ] = {}

    def add_diagnostic(
        key: str,
        frame: pd.DataFrame,
        *,
        track_name: str,
        point_name: str,
        color: str,
        visible: bool,
    ) -> None:
        track_layer, point_layer = (
            _track_group_layers(
                viewer,
                frame,
                track_name=track_name,
                point_name=point_name,
                point_color=color,
                scale_tzyx=scale_tzyx,
                visible=visible,
                tail_length=frame_count,
            )
        )
        diagnostic_layers[key] = (
            _DiagnosticLayerGroup(
                frame=frame,
                track_name=track_name,
                track_layer=track_layer,
                track_visible=bool(
                    visible
                ),
                point_layer=point_layer,
            )
        )

    add_diagnostic(
        "broken",
        diagnostics.ended_failure_tracks,
        track_name="Broken Tracks",
        point_name="Broken Track Centers",
        color=CATEGORY_COLORS["broken"],
        visible=True,
    )
    add_diagnostic(
        "new",
        diagnostics.new_failure_tracks,
        track_name="New Tracks",
        point_name="New Track Centers",
        color=CATEGORY_COLORS["new"],
        visible=True,
    )
    add_diagnostic(
        "boundary_entry",
        diagnostics.boundary_entry_tracks,
        track_name="Boundary Entry Tracks",
        point_name="Boundary Entry Centers",
        color=CATEGORY_COLORS["boundary_entry"],
        visible=False,
    )
    add_diagnostic(
        "boundary_exit",
        diagnostics.boundary_exit_tracks,
        track_name="Boundary Exit Tracks",
        point_name="Boundary Exit Centers",
        color=CATEGORY_COLORS["boundary_exit"],
        visible=False,
    )

    active_tracks_layer = _sync_tracks_layer(
        viewer,
        None,
        edges_to_tracks_array(
            track_session.visible_edges,
            track_centers,
        ),
        name="Corrected Tracks - active",
        scale_tzyx=scale_tzyx,
        tail_length=frame_count,
        visible=True,
    )

    hidden_tracks_layer = _sync_tracks_layer(
        viewer,
        None,
        edges_to_tracks_array(
            track_session.hidden_edges,
            track_centers,
        ),
        name="Hidden tracks",
        scale_tzyx=scale_tzyx,
        tail_length=frame_count,
        visible=False,
    )

    hidden_track_centers_layer = viewer.add_points(
        nodes_to_points_array(
            track_session.hidden_nodes,
            track_centers,
        ),
        name="Hidden track centers",
        scale=scale_tzyx,
        size=4,
        face_color="gray",
    )
    hidden_track_centers_layer.visible = False

    diagnostic_categories = (
        diagnostic_node_categories(
            diagnostics
        )
    )

    # ------------------------------------------------------------------
    # Unified annotation controls
    # ------------------------------------------------------------------
    mode = {
        "value": MODE_SPATIAL,
    }
    selected_seed_ids: list[
        int | None
    ] = [
        None,
        None,
        None,
        None,
    ]
    last_selected_sv = {
        "value": None,
    }
    last_frame = {
        "value": -1,
    }
    last_binary_frame = {
        "value": -1,
    }

    mode_label = Label(
        value="Mode: Spatial"
    )
    frame_label = Label(value="")
    selection_label = Label(value="")
    graph_label = Label(value="")
    status_label = Label(
        value=(
            "Spatial mode: click visible supervoxels. "
            "Use Save Split or Hallucination."
        )
    )

    spatial_mode_button = PushButton(
        text="Spatial mode"
    )
    tracking_mode_button = PushButton(
        text="Tracking mode"
    )

    box1 = LineEdit(label="Instance 1")
    box2 = LineEdit(label="Instance 2")
    box3 = LineEdit(label="Instance 3")
    box4 = LineEdit(label="Instance 4")
    boxes = (
        box1,
        box2,
        box3,
        box4,
    )

    save_split_button = PushButton(
        text="Save Split"
    )
    hallucination_button = PushButton(
        text="Hallucination"
    )
    reset_spatial_button = PushButton(
        text="Reset Spatial"
    )
    undo_spatial_button = PushButton(
        text="Undo Spatial"
    )

    continue_track_button = PushButton(
        text="Continue Track"
    )
    break_track_button = PushButton(
        text="Break Track"
    )
    complete_track_button = PushButton(
        text="Complete Track"
    )
    reset_track_button = PushButton(
        text="Reset Track"
    )
    undo_track_button = PushButton(
        text="Undo Track"
    )

    for widget in (
        mode_label,
        frame_label,
        selection_label,
        graph_label,
        status_label,
    ):
        _make_label_shrinkable(widget)

    panel = Container(
        widgets=[
            mode_label,
            spatial_mode_button,
            tracking_mode_button,
            frame_label,
            selection_label,
            graph_label,
            box1,
            box2,
            box3,
            box4,
            save_split_button,
            hallucination_button,
            reset_spatial_button,
            undo_spatial_button,
            continue_track_button,
            break_track_button,
            complete_track_button,
            reset_track_button,
            undo_track_button,
            status_label,
        ],
        layout="vertical",
        labels=True,
    )
    dock = viewer.window.add_dock_widget(
        panel,
        area="right",
        name="Unified cell + track annotation",
    )
    _configure_dock(
        dock,
        panel,
    )

    def current_frame() -> int:
        return int(
            round(
                viewer.dims.current_step[0]
            )
        )

    def show_error(exc: Exception) -> None:
        status_label.value = (
            "ERROR: " + str(exc)
        )
        print()
        print("[annotation error]")
        print(exc)

    def clear_seed_highlights() -> None:
        zero = np.zeros(
            spatial_shape,
            dtype=np.uint8,
        )
        for layer in seed_highlight_layers:
            layer.data = zero
            layer.refresh()

    def clear_spatial_selection(
        *,
        persist_message: str | None = None,
    ) -> None:
        for index in range(4):
            selected_seed_ids[index] = None
        last_selected_sv["value"] = None
        for box in boxes:
            box.value = ""
        clear_seed_highlights()
        if persist_message is not None:
            status_label.value = persist_message

    def next_empty_seed_slot() -> int | None:
        for index, value in enumerate(
            selected_seed_ids
        ):
            if value is None:
                return index
        return None

    def show_seed(
        frame: int,
        slot: int,
        sv_id: int,
    ) -> None:
        sv_frame = np.asarray(
            supervoxels[frame]
        )
        mask = (
            sv_frame
            == int(sv_id)
        )
        layer = seed_highlight_layers[
            int(slot)
        ]
        layer.data = mask.astype(
            np.uint8,
            copy=False,
        )
        layer.refresh()
        selected_seed_ids[slot] = int(sv_id)
        last_selected_sv["value"] = int(sv_id)

    def clear_track_selection(
        *,
        persist: bool = True,
    ) -> None:
        if persist:
            track_session.reset_selections()
        else:
            track_session.selections.clear()
        zero = np.zeros(
            spatial_shape,
            dtype=np.uint8,
        )
        track_selection_a_layer.data = zero
        track_selection_b_layer.data = zero
        track_selection_a_layer.refresh()
        track_selection_b_layer.refresh()

    def refresh_track_selection_layers() -> None:
        frame = current_frame()
        corrected = spatial_session.frame(
            frame
        )
        zero = np.zeros(
            spatial_shape,
            dtype=np.uint8,
        )
        for slot, layer in enumerate(
            (
                track_selection_a_layer,
                track_selection_b_layer,
            )
        ):
            if (
                slot
                < len(track_session.selections)
            ):
                node = track_session.selections[
                    slot
                ]
                if node[0] == frame:
                    layer.data = (
                        corrected
                        == int(node[1])
                    ).astype(
                        np.uint8,
                        copy=False,
                    )
                else:
                    layer.data = zero
            else:
                layer.data = zero
            layer.refresh()

    def refresh_binary_layer() -> None:
        if not bool(
            binary_layer.visible
        ):
            return
        frame = current_frame()
        if (
            frame
            == last_binary_frame["value"]
        ):
            return
        binary_layer.data = binary_cache.frame(
            frame
        )
        binary_layer.refresh()
        last_binary_frame["value"] = frame

    def _replace_frame_centers(
        frame: int,
        centers: dict[int, np.ndarray],
    ) -> None:
        stale = [
            node
            for node in track_centers
            if node[0] == int(frame)
        ]
        for node in stale:
            del track_centers[node]
        for instance_id, center in centers.items():
            track_centers[
                (
                    int(frame),
                    int(instance_id),
                )
            ] = np.asarray(
                center,
                dtype=np.float64,
            )

    def refresh_diagnostic_layers() -> None:
        for group in diagnostic_layers.values():
            filtered = (
                filter_track_rows_for_completed(
                    group.frame,
                    track_session.completed_nodes,
                )
            )
            tracks_array, points_array, properties = (
                track_frame_arrays(filtered)
            )

            if group.track_layer is not None:
                try:
                    group.track_visible = bool(
                        group.track_layer.visible
                    )
                except Exception:
                    pass

            group.track_layer = (
                _sync_tracks_layer(
                    viewer,
                    group.track_layer,
                    tracks_array,
                    name=group.track_name,
                    scale_tzyx=scale_tzyx,
                    tail_length=frame_count,
                    visible=group.track_visible,
                )
            )

            group.point_layer.data = points_array
            try:
                group.point_layer.properties = properties
            except Exception:
                pass
            group.point_layer.refresh()

    def refresh_track_graph_layers() -> None:
        nonlocal active_tracks_layer, hidden_tracks_layer

        active_visible = True
        if active_tracks_layer is not None:
            try:
                active_visible = bool(
                    active_tracks_layer.visible
                )
            except Exception:
                pass

        hidden_visible = False
        if hidden_tracks_layer is not None:
            try:
                hidden_visible = bool(
                    hidden_tracks_layer.visible
                )
            except Exception:
                pass

        active_tracks_layer = (
            _sync_tracks_layer(
                viewer,
                active_tracks_layer,
                edges_to_tracks_array(
                    track_session.visible_edges,
                    track_centers,
                ),
                name="Corrected Tracks - active",
                scale_tzyx=scale_tzyx,
                tail_length=frame_count,
                visible=active_visible,
            )
        )

        hidden_tracks_layer = (
            _sync_tracks_layer(
                viewer,
                hidden_tracks_layer,
                edges_to_tracks_array(
                    track_session.hidden_edges,
                    track_centers,
                ),
                name="Hidden tracks",
                scale_tzyx=scale_tzyx,
                tail_length=frame_count,
                visible=hidden_visible,
            )
        )

        hidden_track_centers_layer.data = (
            nodes_to_points_array(
                track_session.hidden_nodes,
                track_centers,
            )
        )
        hidden_track_centers_layer.refresh()

        refresh_diagnostic_layers()

    def refresh_current_frame_layers(
        *,
        refresh_tracks: bool = True,
    ) -> None:
        frame = current_frame()
        corrected = spatial_session.frame(
            frame
        )

        hallucinated = (
            spatial_session.hallucinated_supervoxels(
                frame
            )
        )
        sv_frame = np.asarray(
            supervoxels[frame]
        )
        if hallucinated:
            sv_display = sv_frame.copy()
            sv_display[
                np.isin(
                    sv_display,
                    np.asarray(
                        sorted(hallucinated),
                        dtype=sv_display.dtype,
                    ),
                )
            ] = 0
        else:
            sv_display = sv_frame

        corrected_layer.data = corrected
        corrected_layer.refresh()
        apply_label_color_dict(
            corrected_layer,
            label_color_dict(corrected),
        )

        supervoxel_layer.data = sv_display
        supervoxel_layer.refresh()
        apply_label_color_dict(
            supervoxel_layer,
            label_color_dict(sv_display),
        )

        sv_points, sv_properties = (
            supervoxel_interior_points(
                sv_display,
                hidden_ids=hallucinated,
                spacing_zyx=spacing_zyx,
            )
        )
        supervoxel_id_layer.data = sv_points
        try:
            supervoxel_id_layer.properties = (
                sv_properties
            )
        except Exception:
            pass
        supervoxel_id_layer.refresh()

        centers = frame_instance_centers(
            corrected
        )
        _replace_frame_centers(
            frame,
            centers,
        )
        track_session.set_frame_nodes(
            frame,
            centers.keys(),
        )

        ordered_ids = sorted(
            centers
        )
        center_points = np.asarray(
            [
                centers[instance_id]
                for instance_id in ordered_ids
            ],
            dtype=np.float32,
        )
        if center_points.size == 0:
            center_points = np.zeros(
                (0, 3),
                dtype=np.float32,
            )

        categories = [
            diagnostic_categories.get(
                (
                    frame,
                    int(instance_id),
                ),
                "default",
            )
            for instance_id in ordered_ids
        ]
        center_colors = np.asarray(
            [
                _CATEGORY_RGBA[
                    category
                ]
                for category in categories
            ],
            dtype=np.float32,
        )
        if center_colors.size == 0:
            center_colors = np.zeros(
                (0, 4),
                dtype=np.float32,
            )

        cell_centers_layer.data = (
            center_points
        )
        try:
            cell_centers_layer.properties = {
                "instance_id": np.asarray(
                    ordered_ids,
                    dtype=np.int64,
                ),
                "category": np.asarray(
                    categories,
                    dtype=object,
                ),
            }
        except Exception:
            pass
        try:
            cell_centers_layer.face_color = (
                center_colors
            )
        except Exception:
            # Older Napari fallback. Diagnostic point layers still preserve
            # the requested red/lime/cyan/orange category colors.
            cell_centers_layer.face_color = (
                "white"
            )
        cell_centers_layer.refresh()

        refresh_track_selection_layers()
        refresh_binary_layer()

        if refresh_tracks:
            refresh_track_graph_layers()

    def refresh_status() -> None:
        frame = current_frame()
        selection_parts = []
        for index, sv_id in enumerate(
            selected_seed_ids,
            start=1,
        ):
            if sv_id is not None:
                selection_parts.append(
                    f"SV{index}={sv_id}"
                )
        for index, node in enumerate(
            track_session.selections,
            start=1,
        ):
            selection_parts.append(
                f"{chr(64 + index)}=(t={node[0]},id={node[1]})"
            )

        frame_label.value = (
            f"Frame t={frame}/{frame_count - 1} | "
            f"splits={spatial_session.split_corrections_in_frame(frame)} | "
            f"hallucinations={spatial_session.hallucinations_in_frame(frame)}"
        )
        selection_label.value = (
            "Selected: "
            + (
                " | ".join(selection_parts)
                if selection_parts
                else "none"
            )
        )
        graph_label.value = (
            f"Tracks: visible edges={len(track_session.visible_edges)} | "
            f"hidden nodes={len(track_session.hidden_nodes)} | "
            f"manual continues={len(track_session.forced_edges)} | "
            f"manual breaks={len(track_session.broken_edges)}"
        )

        try:
            undo_spatial_button.enabled = (
                spatial_session.can_undo()
            )
            continue_track_button.enabled = (
                len(track_session.selections)
                == 2
            )
            break_track_button.enabled = (
                len(track_session.selections)
                == 2
            )
            complete_track_button.enabled = (
                bool(
                    track_session.selections
                )
            )
            undo_track_button.enabled = (
                bool(
                    track_session.history
                )
            )

            spatial_enabled = (
                mode["value"]
                == MODE_SPATIAL
            )
            for widget in (
                box1,
                box2,
                box3,
                box4,
                save_split_button,
                hallucination_button,
                reset_spatial_button,
                undo_spatial_button,
            ):
                widget.enabled = spatial_enabled

            tracking_enabled = (
                mode["value"]
                == MODE_TRACKING
            )
            for widget in (
                continue_track_button,
                break_track_button,
                complete_track_button,
                reset_track_button,
                undo_track_button,
            ):
                widget.enabled = tracking_enabled
        except Exception:
            pass

    def set_mode(value: str) -> None:
        if value not in {
            MODE_SPATIAL,
            MODE_TRACKING,
        }:
            raise ValueError(value)
        mode["value"] = value
        if value == MODE_SPATIAL:
            clear_track_selection()
            mode_label.value = (
                "Mode: Spatial"
            )
            status_label.value = (
                "Spatial mode: click supervoxels. "
                "Save Split or mark the last selected SV as Hallucination."
            )
        else:
            clear_spatial_selection()
            mode_label.value = (
                "Mode: Tracking"
            )
            status_label.value = (
                "Tracking mode: click cell A, move in time, click cell B, "
                "then Continue Track or Break Track."
            )
        refresh_status()

    @viewer.mouse_drag_callbacks.append
    def unified_picker(_viewer, event):
        button = getattr(
            event,
            "button",
            None,
        )
        button_text = str(
            button
        ).lower()
        is_left = (
            button is None
            or button == 1
            or button_text == "1"
            or "left" in button_text
        )
        if not is_left:
            return

        dragged = False
        yield
        while getattr(
            event,
            "type",
            None,
        ) == "mouse_move":
            dragged = True
            yield
        if dragged:
            return

        frame = current_frame()

        if (
            mode["value"]
            == MODE_SPATIAL
        ):
            sv_id = (
                ray_pick_frontmost_label(
                    supervoxel_layer,
                    event,
                )
            )
            if sv_id <= 0:
                status_label.value = (
                    "Click hit background; no supervoxel selected."
                )
                return
            if sv_id in {
                int(v)
                for v in selected_seed_ids
                if v is not None
            }:
                last_selected_sv[
                    "value"
                ] = int(sv_id)
                status_label.value = (
                    f"SV {sv_id} is already selected; it is now the "
                    "Hallucination target."
                )
                return
            slot = next_empty_seed_slot()
            if slot is None:
                status_label.value = (
                    "All four split seed slots are filled. "
                    "Save Split, Hallucination, or Reset Spatial."
                )
                return
            selected_seed_ids[
                slot
            ] = int(sv_id)
            boxes[slot].value = str(
                int(sv_id)
            )
            show_seed(
                frame,
                slot,
                int(sv_id),
            )
            status_label.value = (
                f"Selected SV {sv_id} as split seed {slot + 1}."
            )
            refresh_status()
            return

        instance_id = (
            ray_pick_frontmost_label(
                corrected_layer,
                event,
            )
        )
        if instance_id <= 0:
            status_label.value = (
                "Click hit background; no corrected cell selected."
            )
            return

        node = (
            int(frame),
            int(instance_id),
        )
        try:
            track_session.add_selection(
                node
            )
        except Exception as exc:
            show_error(exc)
            return
        refresh_track_selection_layers()
        refresh_status()
        status_label.value = (
            f"Selected track cell t={frame}, id={instance_id}."
        )

    def save_split() -> None:
        try:
            groups = [
                parse_supervoxel_group(
                    str(box.value)
                )
                for box in boxes
            ]
            result = (
                spatial_session.apply_split(
                    current_frame(),
                    groups,
                )
            )
        except Exception as exc:
            show_error(exc)
            return

        clear_spatial_selection()
        refresh_current_frame_layers()
        refresh_status()
        status_label.value = (
            f"SPLIT saved at t={result.timepoint}: "
            f"{result.original_instance_id} -> "
            f"{result.output_instance_ids}. "
            "No track association was invented for the new detections."
        )

    def mark_hallucination() -> None:
        sv_id = last_selected_sv[
            "value"
        ]
        if sv_id is None:
            show_error(
                AnnotationError(
                    "Select one visible supervoxel first."
                )
            )
            return

        try:
            record = (
                spatial_session.apply_hallucination(
                    current_frame(),
                    int(sv_id),
                )
            )
        except Exception as exc:
            show_error(exc)
            return

        clear_spatial_selection()
        refresh_current_frame_layers()
        refresh_status()
        status_label.value = (
            f"HALLUCINATION saved: t={record['timepoint']} "
            f"SV={record['supervoxel_id']} removed from corrected "
            "instances and supervoxel visualization."
        )

    def undo_spatial() -> None:
        try:
            result = spatial_session.undo()
        except Exception as exc:
            show_error(exc)
            return

        viewer.dims.set_current_step(
            0,
            int(result.timepoint),
        )
        clear_spatial_selection()
        refresh_current_frame_layers()
        refresh_status()
        status_label.value = (
            f"Undid spatial {result.operation_type} at "
            f"t={result.timepoint}: {result.message}."
        )

    def continue_track() -> None:
        try:
            edge = (
                track_session.connect_selected()
            )
        except Exception as exc:
            show_error(exc)
            return
        refresh_track_selection_layers()
        refresh_track_graph_layers()
        refresh_status()
        status_label.value = (
            f"CONTINUE: t={edge[0][0]} id={edge[0][1]} -> "
            f"t={edge[1][0]} id={edge[1][1]} "
            f"(gap={edge[1][0] - edge[0][0]})."
        )

    def break_track() -> None:
        try:
            edge = (
                track_session.break_selected()
            )
        except Exception as exc:
            show_error(exc)
            return
        refresh_track_selection_layers()
        refresh_track_graph_layers()
        refresh_status()
        status_label.value = (
            f"BREAK: t={edge[0][0]} id={edge[0][1]} -> "
            f"t={edge[1][0]} id={edge[1][1]}."
        )

    def complete_track() -> None:
        try:
            nodes = (
                track_session.complete_selected_components()
            )
        except Exception as exc:
            show_error(exc)
            return
        refresh_track_selection_layers()
        refresh_track_graph_layers()
        refresh_status()
        status_label.value = (
            f"Completed {len(nodes)} detections. "
            "The track is removed from active/diagnostic layers and is "
            "available in Hidden tracks."
        )

    def undo_track() -> None:
        try:
            op = track_session.undo()
        except Exception as exc:
            show_error(exc)
            return
        focus = int(
            op.get(
                "focus_frame",
                current_frame(),
            )
        )
        focus = int(
            np.clip(
                focus,
                0,
                frame_count - 1,
            )
        )
        viewer.dims.set_current_step(
            0,
            focus,
        )
        refresh_track_selection_layers()
        refresh_track_graph_layers()
        refresh_status()
        status_label.value = (
            f"Undid last track {op.get('type', 'operation')}."
        )

    spatial_mode_button.changed.connect(
        lambda *_: set_mode(
            MODE_SPATIAL
        )
    )
    tracking_mode_button.changed.connect(
        lambda *_: set_mode(
            MODE_TRACKING
        )
    )
    save_split_button.changed.connect(
        lambda *_: save_split()
    )
    hallucination_button.changed.connect(
        lambda *_: mark_hallucination()
    )
    reset_spatial_button.changed.connect(
        lambda *_: (
            clear_spatial_selection(
                persist_message=(
                    "Spatial selections reset."
                )
            ),
            refresh_status(),
        )
    )
    undo_spatial_button.changed.connect(
        lambda *_: undo_spatial()
    )
    continue_track_button.changed.connect(
        lambda *_: continue_track()
    )
    break_track_button.changed.connect(
        lambda *_: break_track()
    )
    complete_track_button.changed.connect(
        lambda *_: complete_track()
    )
    reset_track_button.changed.connect(
        lambda *_: (
            clear_track_selection(),
            refresh_status(),
        )
    )
    undo_track_button.changed.connect(
        lambda *_: undo_track()
    )

    @viewer.bind_key("Escape")
    def reset_current_mode(_viewer):
        if (
            mode["value"]
            == MODE_SPATIAL
        ):
            clear_spatial_selection(
                persist_message=(
                    "Spatial selections reset."
                )
            )
        else:
            clear_track_selection()
            status_label.value = (
                "Track selections reset."
            )
        refresh_status()

    @viewer.bind_key("Control-S")
    def save_spatial_shortcut(_viewer):
        if (
            mode["value"]
            == MODE_SPATIAL
        ):
            save_split()

    @viewer.bind_key("Control-Z")
    def undo_mode_shortcut(_viewer):
        if (
            mode["value"]
            == MODE_SPATIAL
        ):
            undo_spatial()
        else:
            undo_track()

    def on_dims_change(_event=None) -> None:
        now = current_frame()
        if now == last_frame["value"]:
            return

        clear_spatial_selection()
        clear_track_selection()
        last_frame["value"] = now
        last_binary_frame["value"] = -1
        refresh_current_frame_layers()
        refresh_status()

        if (
            mode["value"]
            == MODE_SPATIAL
        ):
            status_label.value = (
                "Frame changed. Spatial selections were cleared."
            )
        else:
            status_label.value = (
                "Frame changed. Track selections were cleared."
            )

    viewer.dims.events.current_step.connect(
        on_dims_change
    )

    def on_binary_visibility_change(_event=None) -> None:
        if bool(binary_layer.visible):
            last_binary_frame["value"] = -1
            refresh_binary_layer()

    binary_layer.events.visible.connect(
        on_binary_visibility_change
    )

    try:
        viewer.dims.set_current_step(
            0,
            0,
        )
        viewer.dims.set_current_step(
            1,
            spatial_shape[0] // 2,
        )
    except Exception:
        pass

    last_frame["value"] = current_frame()
    refresh_current_frame_layers()
    refresh_status()
    set_mode(MODE_SPATIAL)

    print()
    print("=" * 96)
    print("UNIFIED DATASET CURATION")
    print("=" * 96)
    print(f"sample              : {sample_id}")
    print(f"frames              : {frame_count}")
    print("SV number leaders   : removed")
    print("SV IDs              : interior EDT centers")
    print("cell centers        : dynamic corrected-instance centers")
    print("track diagnostics   : notebook-09 broken/new/boundary groups")
    print("spatial controls    : Save Split | Hallucination | Undo Spatial")
    print("track controls      : Continue Track | Break Track | Complete Track")
    print("completed tracks    : Hidden tracks layer")
    print("=" * 96)

    return viewer
