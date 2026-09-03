from __future__ import annotations

# DATASET_CURATION_LOCAL_TRACK_REPAIR_V2

# DATASET_CURATION_RAW_RAY_BIRTH_AUTOHIDE_V1

# DATASET_CURATION_EMPTY_TRACKS_SAFE_V1

# DATASET_CURATION_EXACT_BOUNDARY_TOUCH_V1

# DATASET_CURATION_POINTS_TEXT_SYNC_V1

"""Unified Napari viewer for spatial and track annotation."""

from dataclasses import dataclass
from typing import Any

import napari
import numpy as np
import pandas as pd

from dataset_curation.annotation.background_refresh import (
    TrackRefreshCoordinator,
    TrackRefreshResult,
    capture_track_refresh_snapshot,
)

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
    label_color_dict,
    nodes_to_points_array,
    ray_pick_label_from_raw,
    track_frame_arrays,
)
from dataset_curation.annotation.tracks.current import (
    build_current_track_table,
    diagnostic_events_for_node,
    filter_diagnostic_rows_for_frame,
    persist_current_track_table,
    prepare_current_endpoint_groups,
)
from dataset_curation.annotation.source_data import (
    BinaryMaskFrameCache,
    estimate_contrast_limits,
)
from dataset_curation.annotation.tracks.diagnostics import (
    CATEGORY_COLORS,
    EndpointTrackGroups,
    boundary_touching_instance_ids,
    diagnostic_node_categories,
)
from dataset_curation.annotation.tracks.graph import Node
from dataset_curation.annotation.tracks.local_repair import (
    repair_split_tracks,
)
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
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QSizePolicy
except ImportError:
    QTimer = None
    QSizePolicy = None


MODE_SPATIAL = "spatial"
MODE_TRACKING = "tracking"

# Full tracking data remains on disk; these only bound rendering.
TRACK_HISTORY_FRAMES = 5
DIAGNOSTIC_HORIZON_FRAMES = 5

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


def _sync_points_data_and_features(
    layer,
    data: np.ndarray,
    *,
    properties: dict[str, np.ndarray] | None = None,
    face_color=None,
    text_spec: dict[str, Any] | None = None,
) -> None:
    """
    Replace a dynamic Napari Points payload without exposing mismatched
    point/feature lengths to feature-backed text rendering.

    Napari/Vispy may render text synchronously from the `data` event. Therefore
    assigning data first and properties second is unsafe whenever N changes
    between frames. Clearing text before either mutation prevents the renderer
    from indexing stale FormatStringEncoding values.

    `features` is used for the replacement because it is the canonical table
    backing Points properties/text in current Napari.
    """
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(
            f"Points data must be 2-D, got shape {array.shape}."
        )

    row_count = int(array.shape[0])
    normalized: dict[str, np.ndarray] = {}

    if properties is not None:
        for key, values in properties.items():
            values_array = np.asarray(values)
            if values_array.ndim == 0:
                values_array = np.repeat(
                    values_array.reshape(1),
                    row_count,
                )
            if int(values_array.shape[0]) != row_count:
                raise ValueError(
                    f"Points property {key!r} has "
                    f"{values_array.shape[0]} rows for {row_count} points."
                )
            normalized[str(key)] = values_array

    # A feature-backed TextManager can receive a Vispy callback immediately
    # from layer.data. Remove it first; restore it only after the new feature
    # table has the same row count as the new coordinates.
    if text_spec is not None:
        layer.text = None

    layer.data = array

    if properties is not None:
        layer.features = pd.DataFrame(
            normalized
        )

    if face_color is not None:
        layer.face_color = face_color

    if text_spec is not None:
        layer.text = dict(text_spec)

    layer.refresh()


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
        layer.tail_length = int(tail_length)
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

    print("[viewer] creating Napari window...", flush=True)
    viewer = napari.Viewer(
        ndisplay=3
    )
    print("[viewer] Napari window created", flush=True)
    viewer.dims.axis_labels = (
        "time",
        "z",
        "y",
        "x",
    )

    low, high = estimate_contrast_limits(raw)
    print("[viewer] adding lazy raw Zarr layer...", flush=True)
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

    print("[viewer] raw layer ready", flush=True)

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
        opacity=0.35,
    )
    apply_label_color_dict(
        supervoxel_layer,
        label_color_dict(initial_supervoxels),
    )
    # Napari does not render Labels.contour in 3-D. Deliberately use
    # translucent label fills instead of assigning contour and triggering
    # repeated "Contours are not displayed during 3D rendering" warnings.

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

    for layer in seed_highlight_layers:
        layer.visible = False

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
    track_selection_c_layer = viewer.add_image(
        np.zeros(
            spatial_shape,
            dtype=np.uint8,
        ),
        name="Track Cell C selection",
        scale=spacing_zyx,
        colormap="cyan",
        contrast_limits=(0, 1),
        opacity=0.88,
        blending="additive",
    )
    track_selection_a_layer.visible = False
    track_selection_b_layer.visible = False
    track_selection_c_layer.visible = False

    print("[viewer] spatial annotation layers ready", flush=True)

    # ------------------------------------------------------------------
    # Notebook-09 Trackastra visualization
    # ------------------------------------------------------------------
    print("[viewer] adding corrected track diagnostic layers...", flush=True)

    current_tracks = build_current_track_table(
        valid_nodes=track_session.valid_nodes,
        active_edges=track_session.active_edges,
        centers=track_centers,
    )
    persist_current_track_table(
        track_session.output.current_tracks_csv,
        current_tracks,
    )
    current_diagnostics = prepare_current_endpoint_groups(
        current_tracks,
        unresolved_start_nodes=track_session.unresolved_start_nodes,
        unresolved_end_nodes=track_session.unresolved_end_nodes,
        boundary_entry_nodes=track_session.boundary_entry_nodes,
        boundary_exit_nodes=track_session.boundary_exit_nodes,
        hidden_nodes=track_session.hidden_nodes,
        ignored_start_nodes=track_session.ignored_start_nodes,
        ignored_end_nodes=track_session.ignored_end_nodes,
    )

    all_tracks_array, all_points_array, all_properties = (
        track_frame_arrays(current_tracks)
    )
    all_tracks_layer = _sync_tracks_layer(
        viewer,
        None,
        all_tracks_array,
        name="Tracks - all",
        scale_tzyx=scale_tzyx,
        tail_length=TRACK_HISTORY_FRAMES,
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
                tail_length=TRACK_HISTORY_FRAMES,
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
        filter_diagnostic_rows_for_frame(
            current_diagnostics.ended_failure_tracks,
            category="broken",
            current_frame=0,
            horizon_frames=DIAGNOSTIC_HORIZON_FRAMES,
        ),
        track_name="Broken Tracks",
        point_name="Broken Track Centers",
        color=CATEGORY_COLORS["broken"],
        visible=True,
    )
    add_diagnostic(
        "new",
        filter_diagnostic_rows_for_frame(
            current_diagnostics.new_failure_tracks,
            category="new",
            current_frame=0,
            horizon_frames=DIAGNOSTIC_HORIZON_FRAMES,
        ),
        track_name="New Tracks",
        point_name="New Track Centers",
        color=CATEGORY_COLORS["new"],
        visible=True,
    )
    add_diagnostic(
        "boundary_entry",
        current_diagnostics.boundary_entry_tracks,
        track_name="Boundary Entry Tracks",
        point_name="Boundary Entry Centers",
        color=CATEGORY_COLORS["boundary_entry"],
        visible=False,
    )
    add_diagnostic(
        "boundary_exit",
        current_diagnostics.boundary_exit_tracks,
        track_name="Boundary Exit Tracks",
        point_name="Boundary Exit Centers",
        color=CATEGORY_COLORS["boundary_exit"],
        visible=False,
    )

    print("[viewer] Trackastra diagnostic layers ready", flush=True)
    print("[viewer] analyzing corrected tracking graph...", flush=True)

    active_tracks_layer = _sync_tracks_layer(
        viewer,
        None,
        edges_to_tracks_array(
            track_session.visible_edges,
            track_centers,
        ),
        name="Corrected Tracks - active",
        scale_tzyx=scale_tzyx,
        tail_length=TRACK_HISTORY_FRAMES,
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
        tail_length=TRACK_HISTORY_FRAMES,
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

    print("[viewer] corrected tracking graph layers ready", flush=True)

    diagnostic_categories = (
        diagnostic_node_categories(
            current_diagnostics
        )
    )

    # The initial session load remains fully synchronous so all existing
    # exports are known-good. From this point onward the GUI writes only the
    # small canonical JSON synchronously; derived tables/graph materialization
    # are handled by one serialized background worker.
    track_status_cache = {
        "visible_edges": len(
            track_session.visible_edges
        ),
        "hidden_nodes": len(
            track_session.hidden_nodes
        ),
        "unresolved_starts": len(
            track_session.unresolved_start_nodes
        ),
        "unresolved_ends": len(
            track_session.unresolved_end_nodes
        ),
    }
    track_session.set_deferred_derived_persistence(
        True
    )
    background_coordinator = (
        TrackRefreshCoordinator()
    )
    background_generation = {
        "value": 0,
    }

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

    # Lightweight metadata caches only. No extra 3-D/4-D movie copies.
    corrected_color_cache: dict[int, dict] = {}
    sv_metadata_cache: dict[
        tuple[int, tuple[int, ...]],
        tuple[np.ndarray, dict[str, np.ndarray], dict],
    ] = {}

    centers_by_frame: dict[int, dict[int, np.ndarray]] = {}
    for (frame_id, instance_id), center in track_centers.items():
        centers_by_frame.setdefault(
            int(frame_id),
            {},
        )[int(instance_id)] = np.asarray(
            center,
            dtype=np.float64,
        )

    mode_label = Label(
        value="Mode: Spatial"
    )
    frame_label = Label(value="")
    selection_label = Label(value="")
    graph_label = Label(value="")
    background_label = Label(
        value="Background: idle"
    )
    status_label = Label(
        value=(
            "Spatial mode: click visible supervoxels. "
            "Use Save Split, Save Merge, Hallucination, or Ignore."
        )
    )

    spatial_mode_button = PushButton(
        text="Spatial mode"
    )
    tracking_mode_button = PushButton(
        text="Tracking mode"
    )
    ignore_button = PushButton(
        text="Ignore"
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
    save_merge_button = PushButton(
        text="Save Merge"
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
    birth_button = PushButton(
        text="Birth"
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
        background_label,
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
            background_label,
            ignore_button,
            box1,
            box2,
            box3,
            box4,
            save_split_button,
            save_merge_button,
            hallucination_button,
            reset_spatial_button,
            undo_spatial_button,
            continue_track_button,
            break_track_button,
            birth_button,
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
        # Hiding is far cheaper than rewriting four full 3-D zero arrays.
        for layer in seed_highlight_layers:
            layer.visible = False

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
        layer.visible = True
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
        # Selection overlays are transient. Hide them instead of copying
        # three full 3-D zero arrays on every frame change/reset.
        track_selection_a_layer.visible = False
        track_selection_b_layer.visible = False
        track_selection_c_layer.visible = False

    def refresh_track_selection_layers() -> None:
        frame = current_frame()
        corrected = spatial_session.frame(
            frame
        )
        for slot, layer in enumerate(
            (
                track_selection_a_layer,
                track_selection_b_layer,
                track_selection_c_layer,
            )
        ):
            if slot >= len(track_session.selections):
                layer.visible = False
                continue

            node = track_session.selections[slot]
            if node[0] != frame:
                layer.visible = False
                continue

            layer.data = (
                corrected
                == int(node[1])
            ).astype(
                np.uint8,
                copy=False,
            )
            layer.visible = True
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
        frame = int(frame)
        for instance_id in centers_by_frame.get(frame, {}):
            track_centers.pop(
                (frame, int(instance_id)),
                None,
            )

        normalized = {
            int(instance_id): np.asarray(
                center,
                dtype=np.float64,
            )
            for instance_id, center in centers.items()
        }
        centers_by_frame[frame] = normalized

        for instance_id, center in normalized.items():
            track_centers[
                (frame, instance_id)
            ] = center

    def invalidate_frame_metadata(frame: int) -> None:
        frame = int(frame)
        corrected_color_cache.pop(frame, None)
        # Supervoxels are immutable. Their cache key already includes the
        # hallucinated-ID set, so Split/Merge must not throw away expensive
        # supervoxel interior-point metadata.

    def _diagnostic_frame_for_key(
        key: str,
        groups: EndpointTrackGroups,
    ) -> pd.DataFrame:
        source_frame = {
            "broken": groups.ended_failure_tracks,
            "new": groups.new_failure_tracks,
            "boundary_entry": groups.boundary_entry_tracks,
            "boundary_exit": groups.boundary_exit_tracks,
        }[key]

        return filter_diagnostic_rows_for_frame(
            source_frame,
            category=key,
            current_frame=current_frame(),
            horizon_frames=DIAGNOSTIC_HORIZON_FRAMES,
        )

    def _refresh_current_center_colors() -> None:
        frame = current_frame()
        ids = np.asarray(
            cell_centers_layer.properties.get(
                "instance_id",
                np.zeros((0,), dtype=np.int64),
            ),
            dtype=np.int64,
        )
        categories = [
            diagnostic_categories.get(
                (frame, int(instance_id)),
                "default",
            )
            for instance_id in ids.tolist()
        ]
        colors = np.asarray(
            [
                _CATEGORY_RGBA[category]
                for category in categories
            ],
            dtype=np.float32,
        )
        if colors.size == 0:
            colors = np.zeros((0, 4), dtype=np.float32)

        try:
            cell_centers_layer.properties = {
                "instance_id": ids,
                "category": np.asarray(
                    categories,
                    dtype=object,
                ),
            }
            cell_centers_layer.face_color = colors
            cell_centers_layer.refresh()
        except Exception:
            pass

    def refresh_diagnostic_layers(
        groups: EndpointTrackGroups,
    ) -> None:
        nonlocal diagnostic_categories

        localized_frames: dict[str, pd.DataFrame] = {}

        for key, group in diagnostic_layers.items():
            group.frame = _diagnostic_frame_for_key(
                key,
                groups,
            )
            localized_frames[key] = group.frame
            tracks_array, points_array, properties = (
                track_frame_arrays(group.frame)
            )

            if group.track_layer is not None:
                try:
                    group.track_visible = bool(
                        group.track_layer.visible
                    )
                except Exception:
                    pass

            group.track_layer = _sync_tracks_layer(
                viewer,
                group.track_layer,
                tracks_array,
                name=group.track_name,
                scale_tzyx=scale_tzyx,
                tail_length=TRACK_HISTORY_FRAMES,
                visible=group.track_visible,
            )
            _sync_points_data_and_features(
                group.point_layer,
                points_array,
                properties=properties,
            )

        diagnostic_categories = {}

        def assign_current_category(
            frame: pd.DataFrame,
            category: str,
        ) -> None:
            now = current_frame()
            for row in frame.itertuples(index=False):
                if int(row.frame) != now:
                    continue
                diagnostic_categories[
                    (
                        int(row.frame),
                        int(row.cell_id),
                    )
                ] = category

        # Same priority as the existing notebook-style categories.
        assign_current_category(
            localized_frames.get(
                "boundary_entry",
                pd.DataFrame(),
            ),
            "boundary_entry",
        )
        assign_current_category(
            localized_frames.get(
                "boundary_exit",
                pd.DataFrame(),
            ),
            "boundary_exit",
        )
        assign_current_category(
            localized_frames.get(
                "new",
                pd.DataFrame(),
            ),
            "new",
        )
        assign_current_category(
            localized_frames.get(
                "broken",
                pd.DataFrame(),
            ),
            "broken",
        )

        _refresh_current_center_colors()

    def refresh_track_graph_layers() -> None:
        nonlocal active_tracks_layer, hidden_tracks_layer
        nonlocal all_tracks_layer, current_tracks, current_diagnostics
        nonlocal diagnostic_categories

        current_tracks = build_current_track_table(
            valid_nodes=track_session.valid_nodes,
            active_edges=track_session.active_edges,
            centers=track_centers,
        )
        persist_current_track_table(
            track_session.output.current_tracks_csv,
            current_tracks,
        )
        current_diagnostics = prepare_current_endpoint_groups(
            current_tracks,
            unresolved_start_nodes=track_session.unresolved_start_nodes,
            unresolved_end_nodes=track_session.unresolved_end_nodes,
            boundary_entry_nodes=track_session.boundary_entry_nodes,
            boundary_exit_nodes=track_session.boundary_exit_nodes,
            hidden_nodes=track_session.hidden_nodes,
            ignored_start_nodes=track_session.ignored_start_nodes,
            ignored_end_nodes=track_session.ignored_end_nodes,
        )

        all_visible = False
        if all_tracks_layer is not None:
            try:
                all_visible = bool(all_tracks_layer.visible)
            except Exception:
                pass

        all_tracks_array, all_points_array, all_properties = (
            track_frame_arrays(current_tracks)
        )
        all_tracks_layer = _sync_tracks_layer(
            viewer,
            all_tracks_layer,
            all_tracks_array,
            name="Tracks - all",
            scale_tzyx=scale_tzyx,
            tail_length=TRACK_HISTORY_FRAMES,
            visible=all_visible,
        )
        _sync_points_data_and_features(
            all_centers_layer,
            all_points_array,
            properties=all_properties,
            text_spec={
                "string": "{cell_id}",
                "size": 8,
                "color": "white",
                "anchor": "center",
            },
        )

        active_visible = True
        if active_tracks_layer is not None:
            try:
                active_visible = bool(active_tracks_layer.visible)
            except Exception:
                pass

        hidden_visible = False
        if hidden_tracks_layer is not None:
            try:
                hidden_visible = bool(hidden_tracks_layer.visible)
            except Exception:
                pass

        active_tracks_layer = _sync_tracks_layer(
            viewer,
            active_tracks_layer,
            edges_to_tracks_array(
                track_session.visible_edges,
                track_centers,
            ),
            name="Corrected Tracks - active",
            scale_tzyx=scale_tzyx,
            tail_length=TRACK_HISTORY_FRAMES,
            visible=active_visible,
        )

        hidden_tracks_layer = _sync_tracks_layer(
            viewer,
            hidden_tracks_layer,
            edges_to_tracks_array(
                track_session.hidden_edges,
                track_centers,
            ),
            name="Hidden tracks",
            scale_tzyx=scale_tzyx,
            tail_length=TRACK_HISTORY_FRAMES,
            visible=hidden_visible,
        )

        hidden_track_centers_layer.data = nodes_to_points_array(
            track_session.hidden_nodes,
            track_centers,
        )
        hidden_track_centers_layer.refresh()

        track_status_cache.update(
            {
                "visible_edges": len(
                    track_session.visible_edges
                ),
                "hidden_nodes": len(
                    track_session.hidden_nodes
                ),
                "unresolved_starts": len(
                    track_session.unresolved_start_nodes
                ),
                "unresolved_ends": len(
                    track_session.unresolved_end_nodes
                ),
            }
        )

        refresh_diagnostic_layers(
            current_diagnostics
        )

    def request_background_track_refresh(
        reason: str,
    ) -> None:
        background_generation["value"] += 1
        generation = int(
            background_generation["value"]
        )

        snapshot = capture_track_refresh_snapshot(
            generation=generation,
            reason=str(reason),
            track_session=track_session,
            track_centers=track_centers,
        )
        background_coordinator.request(
            snapshot
        )
        background_label.value = (
            f"Background: updating… generation {generation} ({reason})"
        )


    def _apply_background_track_result(
        result: TrackRefreshResult,
    ) -> None:
        nonlocal all_tracks_layer
        nonlocal active_tracks_layer
        nonlocal hidden_tracks_layer
        nonlocal current_tracks
        nonlocal current_diagnostics

        current_tracks = result.current_tracks
        current_diagnostics = result.diagnostics

        track_status_cache.update(
            {
                "visible_edges": int(
                    result.visible_edge_count
                ),
                "hidden_nodes": int(
                    result.hidden_node_count
                ),
                "unresolved_starts": int(
                    result.unresolved_start_count
                ),
                "unresolved_ends": int(
                    result.unresolved_end_count
                ),
            }
        )

        all_visible = False
        if all_tracks_layer is not None:
            try:
                all_visible = bool(
                    all_tracks_layer.visible
                )
            except Exception:
                pass

        all_tracks_layer = _sync_tracks_layer(
            viewer,
            all_tracks_layer,
            result.all_tracks_array,
            name="Tracks - all",
            scale_tzyx=scale_tzyx,
            tail_length=TRACK_HISTORY_FRAMES,
            visible=all_visible,
        )

        _sync_points_data_and_features(
            all_centers_layer,
            result.all_points_array,
            properties=result.all_properties,
            text_spec={
                "string": "{cell_id}",
                "size": 8,
                "color": "white",
                "anchor": "center",
            },
        )

        active_visible = True
        if active_tracks_layer is not None:
            try:
                active_visible = bool(
                    active_tracks_layer.visible
                )
            except Exception:
                pass

        active_tracks_layer = _sync_tracks_layer(
            viewer,
            active_tracks_layer,
            result.active_tracks_array,
            name="Corrected Tracks - active",
            scale_tzyx=scale_tzyx,
            tail_length=TRACK_HISTORY_FRAMES,
            visible=active_visible,
        )

        hidden_visible = False
        if hidden_tracks_layer is not None:
            try:
                hidden_visible = bool(
                    hidden_tracks_layer.visible
                )
            except Exception:
                pass

        hidden_tracks_layer = _sync_tracks_layer(
            viewer,
            hidden_tracks_layer,
            result.hidden_tracks_array,
            name="Hidden tracks",
            scale_tzyx=scale_tzyx,
            tail_length=TRACK_HISTORY_FRAMES,
            visible=hidden_visible,
        )

        hidden_track_centers_layer.data = (
            result.hidden_points_array
        )
        hidden_track_centers_layer.refresh()

        refresh_diagnostic_layers(
            current_diagnostics
        )
        refresh_status()


    def poll_background_track_refresh() -> None:
        try:
            result = background_coordinator.poll()
        except Exception as exc:
            background_label.value = (
                "Background: ERROR — " + str(exc)
            )
            print()
            print("[background annotation refresh error]")
            print(exc)
            return

        if result is None:
            return

        latest = int(
            background_coordinator.latest_generation
        )

        if int(result.generation) != latest:
            background_label.value = (
                f"Background: updating… generation {latest}"
            )
            return

        _apply_background_track_result(
            result
        )

        if background_coordinator.busy:
            background_label.value = (
                f"Background: updating… generation {latest}"
            )
        else:
            background_label.value = (
                f"Background: idle — applied generation {latest}"
            )

    def refresh_current_frame_layers(
        *,
        refresh_tracks: bool = True,
        spatial_authority_changed: bool = False,
    ) -> None:
        frame = current_frame()
        corrected = spatial_session.frame(frame)

        if spatial_authority_changed:
            invalidate_frame_metadata(
                frame
            )

        hallucinated = spatial_session.hallucinated_supervoxels(
            frame
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

        corrected_mapping = corrected_color_cache.get(frame)
        if corrected_mapping is None:
            corrected_mapping = label_color_dict(
                corrected
            )
            corrected_color_cache[frame] = corrected_mapping
        apply_label_color_dict(
            corrected_layer,
            corrected_mapping,
        )

        supervoxel_layer.data = sv_display
        supervoxel_layer.refresh()

        sv_key = (
            int(frame),
            tuple(sorted(int(v) for v in hallucinated)),
        )
        sv_metadata = sv_metadata_cache.get(sv_key)
        if sv_metadata is None:
            sv_points, sv_properties = supervoxel_interior_points(
                sv_display,
                hidden_ids=hallucinated,
                spacing_zyx=spacing_zyx,
            )
            sv_mapping = label_color_dict(
                sv_display
            )
            sv_metadata = (
                sv_points,
                sv_properties,
                sv_mapping,
            )
            sv_metadata_cache[sv_key] = sv_metadata
        else:
            (
                sv_points,
                sv_properties,
                sv_mapping,
            ) = sv_metadata

        apply_label_color_dict(
            supervoxel_layer,
            sv_mapping,
        )
        _sync_points_data_and_features(
            supervoxel_id_layer,
            sv_points,
            properties=sv_properties,
            text_spec={
                "string": "{sv_id}",
                "size": 11,
                "color": "white",
                "anchor": "center",
            },
        )

        if spatial_authority_changed:
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
            track_session.set_frame_exact_boundary_touch_nodes(
                frame,
                boundary_touching_instance_ids(corrected),
            )
        else:
            centers = dict(
                centers_by_frame.get(
                    int(frame),
                    {},
                )
            )
            if not centers:
                # Defensive fallback for a detection table missing this frame.
                centers = frame_instance_centers(
                    corrected
                )
                _replace_frame_centers(
                    frame,
                    centers,
                )

        ordered_ids = sorted(centers)
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
                (frame, int(instance_id)),
                "default",
            )
            for instance_id in ordered_ids
        ]
        center_colors = np.asarray(
            [
                _CATEGORY_RGBA[category]
                for category in categories
            ],
            dtype=np.float32,
        )
        if center_colors.size == 0:
            center_colors = np.zeros(
                (0, 4),
                dtype=np.float32,
            )

        _sync_points_data_and_features(
            cell_centers_layer,
            center_points,
            properties={
                "instance_id": np.asarray(
                    ordered_ids,
                    dtype=np.int64,
                ),
                "category": np.asarray(
                    categories,
                    dtype=object,
                ),
            },
            face_color=center_colors,
            text_spec={
                "string": "{instance_id}",
                "size": 7,
                "color": "white",
                "anchor": "center",
            },
        )

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
            f"merges={spatial_session.merge_corrections_in_frame(frame)} | "
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
            f"Tracks: visible edges={track_status_cache['visible_edges']} | "
            f"hidden nodes={track_status_cache['hidden_nodes']} | "
            f"manual continues={len(track_session.forced_edges - track_session.birth_edges)} | "
            f"manual breaks={len(track_session.broken_edges)} | "
            f"births={len(track_session.birth_events)} | "
            f"ignored review={len(track_session.ignored_events)} | "
            f"unresolved starts={track_status_cache['unresolved_starts']} | "
            f"unresolved ends={track_status_cache['unresolved_ends']}"
        )

        try:
            undo_spatial_button.enabled = (
                spatial_session.can_undo()
            )
            ignore_button.enabled = (
                (
                    mode["value"] == MODE_SPATIAL
                    and any(
                        value is not None
                        for value in selected_seed_ids
                    )
                )
                or (
                    mode["value"] == MODE_TRACKING
                    and len(track_session.selections) == 1
                )
            )
            continue_track_button.enabled = (
                len(track_session.selections)
                == 2
            )
            break_track_button.enabled = (
                len(track_session.selections)
                == 2
            )
            birth_button.enabled = (
                len(track_session.selections)
                == 3
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
                save_merge_button,
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
                birth_button,
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
                "Spatial mode: click supervoxels. Save Split separates one "
                "instance; Save Merge joins the current instances containing "
                "the selected supervoxels; Hallucination removes all selected SVs; Ignore defers an edge case for later analysis."
            )
        else:
            clear_spatial_selection()
            mode_label.value = (
                "Mode: Tracking"
            )
            status_label.value = (
                "Tracking mode: select two cells for Continue/Break, or "
                "select one parent in an earlier frame and two daughters in "
                "the next frame for Birth. Select one cell and press Ignore to defer a broken/new edge case for later analysis. Track selections survive time changes."
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
                ray_pick_label_from_raw(
                    raw_layer,
                    np.asarray(
                        supervoxel_layer.data
                    ),
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
                    "All four spatial selection slots are filled. "
                    "Save Split, Save Merge, Hallucination, or Reset Spatial."
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
            ray_pick_label_from_raw(
                raw_layer,
                spatial_session.frame(
                    frame
                ),
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

    def _selected_node_for_ignore() -> tuple[Node, str]:
        frame = current_frame()

        if mode["value"] == MODE_SPATIAL:
            selected_sv_ids = [
                int(value)
                for value in selected_seed_ids
                if value is not None
            ]
            if not selected_sv_ids:
                raise AnnotationError(
                    "Select a spatial supervoxel first, then press Ignore."
                )

            instance_ids = {
                int(
                    spatial_session.current_instance_for_supervoxel(
                        frame,
                        sv_id,
                    )
                )
                for sv_id in selected_sv_ids
            }
            instance_ids.discard(0)

            if not instance_ids:
                raise AnnotationError(
                    "The selected spatial supervoxels are currently background."
                )
            if len(instance_ids) != 1:
                raise AnnotationError(
                    "Ignore is ambiguous because the selected supervoxels belong "
                    "to multiple corrected instances. Reset and select one instance."
                )

            return (
                (
                    int(frame),
                    int(next(iter(instance_ids))),
                ),
                MODE_SPATIAL,
            )

        if len(track_session.selections) != 1:
            raise AnnotationError(
                "Tracking Ignore requires exactly one selected cell."
            )

        return (
            track_session.selections[0],
            MODE_TRACKING,
        )


    def ignore_selected() -> None:
        try:
            node, source_mode = _selected_node_for_ignore()
            related_events = diagnostic_events_for_node(
                current_diagnostics,
                node,
                current_frame=current_frame(),
                horizon_frames=DIAGNOSTIC_HORIZON_FRAMES,
            )
            added = track_session.ignore_events(
                selected_node=node,
                source_mode=source_mode,
                events=related_events,
            )
        except Exception as exc:
            show_error(exc)
            return

        clear_spatial_selection()
        clear_track_selection()
        request_background_track_refresh("ignore")
        refresh_status()

        if added:
            event_types = ", ".join(
                str(record.get("event_type"))
                for record in added
            )
            status_label.value = (
                f"IGNORE: t={node[0]} id={node[1]} marked for later analysis "
                f"({event_types}). Related broken/new events are hidden. "
                "Review queue: tracks/ignored_events.csv."
            )
        else:
            status_label.value = (
                f"IGNORE: t={node[0]} id={node[1]} was already marked "
                "for later analysis."
            )

    def save_split() -> None:
        frame = current_frame()

        try:
            groups = [
                parse_supervoxel_group(
                    str(box.value)
                )
                for box in boxes
            ]
            result = (
                spatial_session.apply_split(
                    frame,
                    groups,
                )
            )
        except Exception as exc:
            show_error(exc)
            return

        clear_spatial_selection()
        refresh_current_frame_layers(
            refresh_tracks=False,
            spatial_authority_changed=True,
        )

        try:
            repair_result = repair_split_tracks(
                frame=frame,
                new_instance_ids=result.output_instance_ids,
                track_session=track_session,
                track_centers=track_centers,
                labels_for_frame=spatial_session.frame,
                spacing_zyx_um=spacing_zyx,
            )
            repair_text = repair_result.summary_text()
        except Exception as exc:
            # The spatial edit is already canonical. Automatic tracking is a
            # convenience layer and must never roll back a valid split.
            repair_text = (
                "automatic track repair failed; "
                f"manual tracking remains available ({exc})"
            )
            print()
            print("[local track repair error]")
            print(exc)

        request_background_track_refresh("split")
        refresh_status()
        status_label.value = (
            f"SPLIT saved at t={result.timepoint}: "
            f"{result.original_instance_id} -> "
            f"{result.output_instance_ids}. "
            f"Track repair: {repair_text}."
        )

    def save_merge() -> None:
        try:
            selected_supervoxels: list[int] = []
            for box in boxes:
                selected_supervoxels.extend(
                    parse_supervoxel_group(
                        str(box.value)
                    )
                )

            result = spatial_session.apply_merge(
                current_frame(),
                selected_supervoxels,
            )
        except Exception as exc:
            show_error(exc)
            return

        clear_spatial_selection()
        refresh_current_frame_layers(
            refresh_tracks=False,
            spatial_authority_changed=True,
        )
        request_background_track_refresh("merge")
        refresh_status()
        status_label.value = (
            f"MERGE saved at t={result.timepoint}: "
            f"instances {result.source_instance_ids} -> "
            f"{result.output_instance_id}, selected by SVs "
            f"{result.selected_supervoxel_ids}. "
            "A fresh corrected instance ID was created; track associations "
            "were not inherited automatically."
        )

    # DATASET_CURATION_MULTI_HALLUCINATION_V1
    def mark_hallucination() -> None:
        try:
            selected_supervoxels: list[
                int
            ] = []
            for box in boxes:
                selected_supervoxels.extend(
                    parse_supervoxel_group(
                        str(
                            box.value
                        )
                    )
                )

            if not selected_supervoxels:
                raise AnnotationError(
                    "Select one or more visible supervoxels first."
                )

            records = (
                spatial_session.apply_hallucinations(
                    current_frame(),
                    selected_supervoxels,
                )
            )
        except Exception as exc:
            show_error(exc)
            return

        clear_spatial_selection()
        refresh_current_frame_layers(
            refresh_tracks=False,
            spatial_authority_changed=True,
        )
        request_background_track_refresh(
            "hallucination"
        )
        refresh_status()

        saved_ids = tuple(
            int(
                record[
                    "supervoxel_id"
                ]
            )
            for record in records
        )
        status_label.value = (
            f"HALLUCINATION saved: t={records[0]['timepoint']} "
            f"SVs={saved_ids} removed from corrected instances "
            "and supervoxel visualization."
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
        refresh_current_frame_layers(
            refresh_tracks=False,
            spatial_authority_changed=True,
        )
        request_background_track_refresh("undo-spatial")
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
        request_background_track_refresh("continue")
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
        request_background_track_refresh("break")
        refresh_status()
        status_label.value = (
            f"BREAK: t={edge[0][0]} id={edge[0][1]} -> "
            f"t={edge[1][0]} id={edge[1][1]}."
        )

    def mark_birth() -> None:
        try:
            event = (
                track_session.birth_selected()
            )
        except Exception as exc:
            show_error(exc)
            return

        refresh_track_selection_layers()
        request_background_track_refresh("birth")
        refresh_status()

        parent = event["parent"]
        daughters = event["daughters"]
        status_label.value = (
            f"BIRTH: parent t={parent[0]} id={parent[1]} -> "
            f"daughters t={daughters[0][0]} "
            f"id={daughters[0][1]}, id={daughters[1][1]}. "
            "The two parent-to-daughter edges are now part of the corrected "
            "graph."
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
        request_background_track_refresh("undo-track")
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
    ignore_button.changed.connect(
        lambda *_: ignore_selected()
    )
    save_split_button.changed.connect(
        lambda *_: save_split()
    )
    save_merge_button.changed.connect(
        lambda *_: save_merge()
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
    birth_button.changed.connect(
        lambda *_: mark_birth()
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

    if QTimer is None:
        raise RuntimeError(
            "Qt QTimer is required for asynchronous annotation refresh."
        )

    background_timer = QTimer()
    background_timer.setInterval(40)
    background_timer.timeout.connect(
        poll_background_track_refresh
    )
    background_timer.start()

    # Keep Python references alive for the lifetime of the Napari viewer.
    viewer._dataset_curation_background_timer = (
        background_timer
    )
    viewer._dataset_curation_background_coordinator = (
        background_coordinator
    )


    def _shutdown_background_refresh(*_args) -> None:
        try:
            background_timer.stop()
        except Exception:
            pass
        try:
            background_coordinator.shutdown()
        except Exception:
            pass


    try:
        viewer.window._qt_window.destroyed.connect(
            _shutdown_background_refresh
        )
    except Exception:
        pass

    def reset_all_selections(_source=None) -> None:
        clear_spatial_selection()
        clear_track_selection()
        status_label.value = (
            "Spatial and tracking selections reset."
        )
        refresh_status()


    def reset_with_escape(_viewer=None) -> None:
        reset_all_selections(_viewer)


    try:
        viewer.bind_key(
            "Escape",
            reset_with_escape,
            overwrite=True,
        )
    except TypeError:
        viewer.bind_key(
            "Escape",
            reset_with_escape,
        )


    def _bind_escape_to_layer(layer) -> None:
        try:
            layer.bind_key(
                "Escape",
                reset_all_selections,
                overwrite=True,
            )
        except TypeError:
            try:
                layer.bind_key(
                    "Escape",
                    reset_all_selections,
                )
            except Exception:
                pass
        except Exception:
            pass


    for _layer in list(viewer.layers):
        _bind_escape_to_layer(_layer)


    def _bind_escape_to_inserted_layer(event) -> None:
        layer = getattr(event, "value", None)
        if layer is not None:
            _bind_escape_to_layer(layer)


    try:
        viewer.layers.events.inserted.connect(
            _bind_escape_to_inserted_layer
        )
    except Exception:
        pass
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

        if (
            mode["value"]
            == MODE_SPATIAL
        ):
            clear_spatial_selection()

        # Tracking selections deliberately survive a time change. Continue,
        # Break and Birth all require selecting detections across frames.
        last_frame["value"] = now
        last_binary_frame["value"] = -1
        # Time navigation changes only the current 3-D overlays. The global
        # corrected graph/diagnostic layers are unchanged until an annotation
        # operation actually edits the graph or spatial detections.
        refresh_current_frame_layers(
            refresh_tracks=False
        )
        # Lightweight only: do not rebuild the graph/current_tracks.csv while
        # scrubbing. Just update the small local diagnostic windows.
        refresh_diagnostic_layers(
            current_diagnostics
        )
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
                "Frame changed. Track selections were preserved."
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

    print("[viewer] initializing current-frame overlays...", flush=True)
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
    print("track diagnostics   : broken look-ahead/new recent window = 5 frames")
    print(f"track tail          : {TRACK_HISTORY_FRAMES} time units max history")
    print("spatial controls    : Save Split | Save Merge | Hallucination | Undo Spatial")
    print("track controls      : Continue Track | Break Track | Birth | Undo Track")
    print("common control      : Ignore -> tracks/ignored_events.csv (pending review)")
    print("background refresh  : serialized + latest-state coalescing")
    print("canonical save      : synchronous JSON; derived CSV/layers asynchronous")
    print("track completion    : automatic; complete components move to Hidden tracks")
    print("ray picking         : always derived from Raw BioHub")
    print("3-D contours        : disabled; translucent SV fills avoid Napari warning")
    print("=" * 96)

    return viewer
