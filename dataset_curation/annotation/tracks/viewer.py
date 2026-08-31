from __future__ import annotations

"""Napari picking and UI for Trackastra association correction."""

from dataset_curation.annotation.tracks.graph import (
    AnnotationError,
    Edge,
    Node,
    build_trackastra_detection_edges,
)

from dataset_curation.annotation.tracks.storage import (
    OutputPaths,
    SourcePaths,
)

from dataset_curation.annotation.tracks.session import (
    TrackAnnotationSession,
)

import colorsys

import json

from typing import Any, Iterable

import napari

import numpy as np

import pandas as pd

try:
    from magicgui.widgets import Container, Label, PushButton
except ImportError as exc:
    raise ImportError(
        "magicgui is required for the track annotation panel. It normally "
        "comes with Napari. Install it with: pip install magicgui"
    ) from exc

try:
    from qtpy.QtWidgets import QSizePolicy
except ImportError:
    QSizePolicy = None

def _validate_source_arrays(
    raw: np.ndarray,
    binary_mask: np.ndarray,
    instances: np.ndarray,
) -> None:
    if raw.ndim != 4:
        raise AnnotationError(f"raw.npy must be (T,Z,Y,X), got {raw.shape}")
    if binary_mask.shape != raw.shape:
        raise AnnotationError(
            f"binary mask shape {binary_mask.shape} != raw shape {raw.shape}"
        )
    if instances.shape != raw.shape:
        raise AnnotationError(
            f"final instances shape {instances.shape} != raw shape {raw.shape}"
        )

def _normalize_cells(cells: pd.DataFrame) -> pd.DataFrame:
    required = {
        "frame",
        "cell_id",
        "centroid_z",
        "centroid_y",
        "centroid_x",
    }
    missing = sorted(required - set(cells.columns))
    if missing:
        raise AnnotationError(f"cells_all.csv is missing columns: {missing}")

    result = cells.copy()
    result["frame"] = result["frame"].astype(np.int64)
    result["cell_id"] = result["cell_id"].astype(np.int64)

    duplicate = result.duplicated(["frame", "cell_id"], keep=False)
    if duplicate.any():
        rows = result.loc[duplicate, ["frame", "cell_id"]].head(10)
        raise AnnotationError(
            "cells_all.csv contains duplicate (frame, cell_id) detections:\n"
            + rows.to_string(index=False)
        )
    return result

def _normalize_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    required = {"track_id", "frame", "cell_id", "z", "y", "x"}
    missing = sorted(required - set(tracks.columns))
    if missing:
        raise AnnotationError(f"trackastra/tracks.csv is missing columns: {missing}")

    result = tracks.copy()
    result["track_id"] = result["track_id"].astype(np.int64)
    result["frame"] = result["frame"].astype(np.int64)
    result["cell_id"] = result["cell_id"].astype(np.int64)
    result = result.loc[result["cell_id"] > 0].copy()
    return result

def _ray_intersections_data(layer, event) -> tuple[np.ndarray, np.ndarray] | None:
    view_direction = getattr(event, "view_direction", None)
    dims_displayed = getattr(event, "dims_displayed", None)
    if view_direction is None or dims_displayed is None:
        return None

    try:
        start, end = layer.get_ray_intersections(
            position=event.position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=True,
        )
    except TypeError:
        try:
            start, end = layer.get_ray_intersections(
                event.position,
                view_direction,
                dims_displayed,
                world=True,
            )
        except Exception:
            return None
    except Exception:
        return None

    if start is None or end is None:
        return None
    start = np.asarray(start, dtype=np.float64).reshape(-1)
    end = np.asarray(end, dtype=np.float64).reshape(-1)
    if start.shape != end.shape or not np.all(np.isfinite(start)) or not np.all(
        np.isfinite(end)
    ):
        return None
    return start, end

def _first_foreground_point_along_ray(
    mask: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    samples_per_voxel: float = 4.0,
) -> np.ndarray | None:
    data = np.asarray(mask)
    if start.size != data.ndim or end.size != data.ndim:
        return None

    delta = end - start
    max_axis_distance = float(np.max(np.abs(delta)))
    sample_count = max(
        2,
        int(np.ceil(max_axis_distance * float(samples_per_voxel))) + 1,
    )
    shape = np.asarray(data.shape, dtype=np.int64)

    for alpha in np.linspace(0.0, 1.0, sample_count, dtype=np.float64):
        point = start + alpha * delta
        index = np.rint(point).astype(np.int64)
        if np.any(index < 0) or np.any(index >= shape):
            continue
        if int(data[tuple(index.tolist())]) > 0:
            return point
    return None

def _distance_points_to_segment(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    direction = end - start
    denom = float(np.dot(direction, direction))
    if denom <= 1e-12:
        return np.linalg.norm(points - start[None, :], axis=1)
    alpha = ((points - start[None, :]) @ direction) / denom
    alpha = np.clip(alpha, 0.0, 1.0)
    closest = start[None, :] + alpha[:, None] * direction[None, :]
    return np.linalg.norm(points - closest, axis=1)

class DetectionPicker:
    def __init__(
        self,
        *,
        cells: pd.DataFrame,
        instances: np.ndarray,
        binary_mask: np.ndarray,
        spacing_zyx: tuple[float, float, float],
        max_ray_distance_um: float,
        session: TrackAnnotationSession,
    ) -> None:
        self.cells = cells
        self.instances = instances
        self.binary_mask = binary_mask
        self.spacing = np.asarray(spacing_zyx, dtype=np.float64)
        self.max_ray_distance_um = float(max_ray_distance_um)
        self.session = session

        self.rows_by_frame: dict[int, pd.DataFrame] = {
            int(frame): group.reset_index(drop=True)
            for frame, group in cells.groupby("frame", sort=False)
        }

    def _nearest_to_point(self, frame: int, point_zyx: np.ndarray) -> Node:
        rows = self.rows_by_frame.get(frame)
        if rows is None or rows.empty:
            raise AnnotationError(f"No detections are available at t={frame}.")

        centers = rows[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(float)
        distances = np.linalg.norm(
            (centers - np.asarray(point_zyx)[None, :]) * self.spacing[None, :],
            axis=1,
        )
        order = np.argsort(distances)
        for index in order:
            node = (frame, int(rows.iloc[int(index)]["cell_id"]))
            if node not in self.session.completed_nodes:
                return node
        raise AnnotationError(f"Every detection at t={frame} is already completed.")

    def _nearest_to_ray(
        self,
        frame: int,
        start_zyx: np.ndarray,
        end_zyx: np.ndarray,
    ) -> tuple[Node, float]:
        rows = self.rows_by_frame.get(frame)
        if rows is None or rows.empty:
            raise AnnotationError(f"No detections are available at t={frame}.")

        active_indices = [
            index
            for index, row in rows.iterrows()
            if (frame, int(row.cell_id)) not in self.session.completed_nodes
        ]
        if not active_indices:
            raise AnnotationError(f"Every detection at t={frame} is already completed.")

        active = rows.loc[active_indices].reset_index(drop=True)
        centers = active[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(float)
        centers_um = centers * self.spacing[None, :]
        start_um = np.asarray(start_zyx, dtype=np.float64) * self.spacing
        end_um = np.asarray(end_zyx, dtype=np.float64) * self.spacing
        distances = _distance_points_to_segment(centers_um, start_um, end_um)
        best = int(np.argmin(distances))
        distance = float(distances[best])

        if self.max_ray_distance_um > 0 and distance > self.max_ray_distance_um:
            raise AnnotationError(
                f"Closest cell center is {distance:.2f} um from the click ray, "
                f"above the {self.max_ray_distance_um:.2f} um guard. Click closer "
                "to the target cell or increase --max-ray-distance-um."
            )

        node = (frame, int(active.iloc[best]["cell_id"]))
        return node, distance

    def pick(
        self,
        *,
        frame: int,
        event,
        binary_layer,
        reference_layer,
    ) -> tuple[Node, str]:
        ray = _ray_intersections_data(reference_layer, event)

        # Binary-mask mode: frontmost foreground is authoritative.
        if bool(binary_layer.visible):
            binary_ray = _ray_intersections_data(binary_layer, event) or ray
            if binary_ray is None:
                raise AnnotationError(
                    "Could not resolve a 3-D camera ray for Binary Mask picking."
                )
            hit = _first_foreground_point_along_ray(
                self.binary_mask,
                binary_ray[0],
                binary_ray[1],
            )
            if hit is None:
                raise AnnotationError("The click ray did not hit Binary Mask foreground.")

            index = np.rint(hit).astype(np.int64)
            index[0] = int(frame)
            shape = np.asarray(self.instances.shape, dtype=np.int64)
            index = np.clip(index, 0, shape - 1)
            instance_id = int(self.instances[tuple(index.tolist())])

            if instance_id > 0:
                node = (frame, instance_id)
                self.session._validate_node(node)
                if node in self.session.completed_nodes:
                    raise AnnotationError(
                        f"The frontmost mask hit is t={frame} instance={instance_id}, "
                        "which belongs to a completed track."
                    )
                return node, "binary-mask front hit"

            node = self._nearest_to_point(frame, hit[-3:])
            return node, "binary-mask hit -> nearest centroid"

        # Normal mode: closest center to the click ray.
        if ray is None:
            # 2-D / compatibility fallback: convert the click position to layer
            # data coordinates and use that as a degenerate ray point.
            try:
                point = np.asarray(reference_layer.world_to_data(event.position), dtype=float)
            except Exception as exc:
                raise AnnotationError("Could not resolve mouse position in data space.") from exc
            if point.size < 3:
                raise AnnotationError("Mouse position does not contain Z/Y/X coordinates.")
            node = self._nearest_to_point(frame, point[-3:])
            return node, "nearest centroid to click point"

        node, distance = self._nearest_to_ray(frame, ray[0][-3:], ray[1][-3:])
        return node, f"nearest centroid to ray ({distance:.2f} um)"

def _display_color(label_id: int) -> tuple[float, float, float, float]:
    hue = (0.03 + int(label_id) * 0.6180339887498949) % 1.0
    saturation = 0.82 if int(label_id) % 2 == 0 else 0.70
    value = 0.98 if int(label_id) % 3 else 0.86
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return float(r), float(g), float(b), 0.62

def _apply_current_frame_colors(layer, labels: np.ndarray) -> None:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    mapping: dict[int, tuple[float, float, float, float]] = {
        0: (0.0, 0.0, 0.0, 0.0)
    }
    for label_id in ids.tolist():
        mapping[int(label_id)] = _display_color(int(label_id))
    try:
        layer.color = mapping
    except Exception:
        pass
    layer.refresh()

def _edge_tracks_array(
    edges: Iterable[Edge],
    centers: dict[Node, np.ndarray],
    hidden_nodes: set[Node],
) -> np.ndarray:
    rows: list[list[float]] = []
    track_id = 1
    for left, right in sorted(edges):
        if left in hidden_nodes or right in hidden_nodes:
            continue
        if left not in centers or right not in centers:
            continue
        rows.append([float(track_id), float(left[0]), *centers[left].tolist()])
        rows.append([float(track_id), float(right[0]), *centers[right].tolist()])
        track_id += 1
    if not rows:
        return np.empty((0, 5), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)

def _active_points_array(
    centers: dict[Node, np.ndarray],
    hidden_nodes: set[Node],
) -> np.ndarray:
    rows = [
        [float(node[0]), *center.tolist()]
        for node, center in sorted(centers.items())
        if node not in hidden_nodes
    ]
    return np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 4), float)

def _make_label_shrinkable(widget) -> None:
    native = getattr(widget, "native", None)
    if native is None:
        return
    try:
        native.setWordWrap(True)
        native.setMinimumWidth(0)
    except Exception:
        pass
    if QSizePolicy is not None:
        try:
            native.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        except Exception:
            pass

def open_viewer(
    *,
    source: SourcePaths,
    output: OutputPaths,
    sample_id: str,
    spacing_zyx: tuple[float, float, float],
    max_ray_distance_um: float,
    resume: bool,
) -> None:
    raw = np.load(source.raw, mmap_mode="r", allow_pickle=False)
    binary_mask = np.load(source.binary_mask, mmap_mode="r", allow_pickle=False)
    instances = np.load(source.final_instances, mmap_mode="r", allow_pickle=False)
    _validate_source_arrays(raw, binary_mask, instances)

    cells = _normalize_cells(pd.read_csv(source.cells_csv))
    tracks = _normalize_tracks(pd.read_csv(source.tracks_csv))
    lineage_payload = json.loads(source.napari_graph.read_text(encoding="utf-8"))
    if not isinstance(lineage_payload, dict):
        raise AnnotationError(
            f"Expected dict in {source.napari_graph}, got "
            f"{type(lineage_payload).__name__}."
        )

    valid_nodes: set[Node] = {
        (int(row.frame), int(row.cell_id))
        for row in cells.itertuples(index=False)
    }
    centers: dict[Node, np.ndarray] = {
        (int(row.frame), int(row.cell_id)): np.asarray(
            [row.centroid_z, row.centroid_y, row.centroid_x],
            dtype=np.float64,
        )
        for row in cells.itertuples(index=False)
    }
    base_edges = build_trackastra_detection_edges(tracks, lineage_payload)

    session = TrackAnnotationSession(
        sample_id=sample_id,
        source_root=source.root,
        output=output,
        valid_nodes=valid_nodes,
        base_edges=base_edges,
        resume=resume,
    )
    picker = DetectionPicker(
        cells=cells,
        instances=instances,
        binary_mask=binary_mask,
        spacing_zyx=spacing_zyx,
        max_ray_distance_um=max_ray_distance_um,
        session=session,
    )

    scale_tzyx = (1.0, *spacing_zyx)
    spatial_shape = tuple(int(v) for v in raw.shape[-3:])

    viewer = napari.Viewer(ndisplay=3)
    low, high = np.percentile(np.asarray(raw), [1.0, 99.8])
    raw_layer = viewer.add_image(
        raw,
        name="Raw Volume",
        scale=scale_tzyx,
        rendering="mip",
        colormap="gray",
        contrast_limits=[float(low), float(high)],
    )
    binary_layer = viewer.add_labels(
        binary_mask,
        name="Binary Mask",
        scale=scale_tzyx,
        visible=False,
        opacity=0.35,
    )

    # Current-frame-only candidate layer. It is rebuilt when t changes so
    # completed detections can disappear without copying the full 4-D label movie.
    active_cells_layer = viewer.add_labels(
        np.zeros(spatial_shape, dtype=np.int32),
        name="Active Spatial Instances",
        scale=spacing_zyx,
        opacity=0.62,
    )

    if source.tracked_masks.is_file():
        tracked_masks = np.load(source.tracked_masks, mmap_mode="r", allow_pickle=False)
        viewer.add_labels(
            tracked_masks,
            name="Trackastra Tracked Masks (original)",
            scale=scale_tzyx,
            visible=False,
            opacity=0.45,
        )

    if source.napari_tracks.is_file():
        original_tracks = np.load(source.napari_tracks, mmap_mode="r", allow_pickle=False)
        original_layer = viewer.add_tracks(
            np.asarray(original_tracks),
            name="Trackastra Tracks (original)",
            scale=scale_tzyx,
            tail_length=int(raw.shape[0]),
        )
        original_layer.visible = False

    corrected_tracks_layer = viewer.add_tracks(
        _edge_tracks_array(session.active_edges, centers, session.completed_nodes),
        name="Corrected Edges - active",
        scale=scale_tzyx,
        tail_length=int(raw.shape[0]),
    )
    corrected_tracks_layer.visible = True

    active_points_layer = viewer.add_points(
        _active_points_array(centers, session.completed_nodes),
        name="Centroids - active",
        scale=scale_tzyx,
        size=3.5,
        face_color="red",
        opacity=0.8,
    )

    selection_a_layer = viewer.add_labels(
        np.zeros(spatial_shape, dtype=np.uint8),
        name="Selected Cell A",
        scale=spacing_zyx,
        opacity=0.88,
        color={0: (0, 0, 0, 0), 1: (1.0, 0.95, 0.05, 1.0)},
    )
    selection_b_layer = viewer.add_labels(
        np.zeros(spatial_shape, dtype=np.uint8),
        name="Selected Cell B",
        scale=spacing_zyx,
        opacity=0.88,
        color={0: (0, 0, 0, 0), 1: (1.0, 0.05, 0.75, 1.0)},
    )

    frame_label = Label(value="")
    selection_label = Label(value="Selected: none")
    graph_label = Label(value="")
    status_label = Label(
        value=(
            "Click cell A, move in time, click cell B, then Connect or Break. "
            "Escape resets the pair."
        )
    )
    connect_button = PushButton(text="Connect")
    break_button = PushButton(text="Break")
    complete_button = PushButton(text="Complete Track")
    undo_button = PushButton(text="Undo")
    reset_button = PushButton(text="Reset")

    for widget in (frame_label, selection_label, graph_label, status_label):
        _make_label_shrinkable(widget)

    panel = Container(
        widgets=[
            frame_label,
            selection_label,
            graph_label,
            connect_button,
            break_button,
            complete_button,
            undo_button,
            reset_button,
            status_label,
        ],
        layout="vertical",
    )
    dock = viewer.window.add_dock_widget(panel, name="Track annotation", area="right")
    try:
        dock.setMinimumWidth(300)
    except Exception:
        pass

    last_frame = [-1]

    def current_frame() -> int:
        return int(round(viewer.dims.current_step[0]))

    def refresh_selection_layers() -> None:
        frame = current_frame()
        zero = np.zeros(spatial_shape, dtype=np.uint8)
        layers = (selection_a_layer, selection_b_layer)
        for slot, layer in enumerate(layers):
            if slot < len(session.selections):
                node = session.selections[slot]
                if node[0] == frame:
                    layer.data = (np.asarray(instances[frame]) == node[1]).astype(
                        np.uint8, copy=False
                    )
                else:
                    layer.data = zero
            else:
                layer.data = zero
            layer.refresh()

    def refresh_current_candidate_cells() -> None:
        frame = current_frame()
        frame_labels = np.asarray(instances[frame])
        completed_ids = {
            node[1] for node in session.completed_nodes if node[0] == frame
        }
        if completed_ids:
            display = np.asarray(frame_labels).copy()
            display[np.isin(display, np.fromiter(completed_ids, dtype=np.int64))] = 0
        else:
            display = np.asarray(frame_labels)
        active_cells_layer.data = display
        _apply_current_frame_colors(active_cells_layer, display)

    def refresh_graph_layers() -> None:
        corrected_tracks_layer.data = _edge_tracks_array(
            session.active_edges,
            centers,
            session.completed_nodes,
        )
        corrected_tracks_layer.refresh()
        active_points_layer.data = _active_points_array(
            centers,
            session.completed_nodes,
        )
        active_points_layer.refresh()

    def refresh_status_labels() -> None:
        frame = current_frame()
        active_here = sum(
            1 for node in valid_nodes if node[0] == frame and node not in session.completed_nodes
        )
        completed_here = sum(1 for node in session.completed_nodes if node[0] == frame)
        frame_label.value = (
            f"Frame t={frame} / {raw.shape[0] - 1} | active cells={active_here} | "
            f"completed here={completed_here}"
        )
        if session.selections:
            selection_label.value = "Selected: " + " | ".join(
                f"{chr(65 + i)}=(t={node[0]}, id={node[1]})"
                for i, node in enumerate(session.selections)
            )
        else:
            selection_label.value = "Selected: none"
        graph_label.value = (
            f"Graph: active edges={len(session.active_edges)} | "
            f"manual connects={len(session.forced_edges)} | "
            f"manual breaks={len(session.broken_edges)} | "
            f"completed nodes={len(session.completed_nodes)} | "
            f"undo depth={len(session.history)}"
        )
        try:
            undo_button.enabled = bool(session.history)
            connect_button.enabled = len(session.selections) == 2
            break_button.enabled = len(session.selections) == 2
            complete_button.enabled = bool(session.selections)
        except Exception:
            pass

    def refresh_all(*, force_candidate: bool = True) -> None:
        if force_candidate:
            refresh_current_candidate_cells()
        refresh_selection_layers()
        refresh_graph_layers()
        refresh_status_labels()

    def show_error(exc: Exception) -> None:
        status_label.value = "ERROR: " + str(exc)
        print("\n[track annotation error]")
        print(exc)

    @viewer.mouse_drag_callbacks.append
    def ray_pick_cell(_viewer, event):
        button = getattr(event, "button", None)
        button_text = str(button).lower()
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
        while getattr(event, "type", None) == "mouse_move":
            dragged = True
            yield
        if dragged:
            return

        try:
            node, mode = picker.pick(
                frame=current_frame(),
                event=event,
                binary_layer=binary_layer,
                reference_layer=raw_layer,
            )
            session.add_selection(node)
        except Exception as exc:
            show_error(exc)
            return

        refresh_selection_layers()
        refresh_status_labels()
        slot = chr(64 + len(session.selections))
        status_label.value = (
            f"Selected {slot}: t={node[0]} instance={node[1]} via {mode}."
        )

    def connect_selected() -> None:
        try:
            edge = session.connect_selected()
        except Exception as exc:
            show_error(exc)
            return
        refresh_all(force_candidate=False)
        status_label.value = (
            f"CONNECTED t={edge[0][0]} id={edge[0][1]} -> "
            f"t={edge[1][0]} id={edge[1][1]} "
            f"(gap={edge[1][0] - edge[0][0]})."
        )

    def break_selected() -> None:
        try:
            edge = session.break_selected()
        except Exception as exc:
            show_error(exc)
            return
        refresh_all(force_candidate=False)
        status_label.value = (
            f"BROKE t={edge[0][0]} id={edge[0][1]} -> "
            f"t={edge[1][0]} id={edge[1][1]}."
        )

    def complete_selected() -> None:
        try:
            nodes = session.complete_selected_components()
        except Exception as exc:
            show_error(exc)
            return
        refresh_all(force_candidate=True)
        status_label.value = (
            f"Completed and hid corrected component: {len(nodes)} newly hidden "
            "detections."
        )

    def undo_last() -> None:
        try:
            op = session.undo()
        except Exception as exc:
            show_error(exc)
            return
        focus_frame = int(op.get("focus_frame", current_frame()))
        focus_frame = int(np.clip(focus_frame, 0, raw.shape[0] - 1))
        viewer.dims.set_current_step(0, focus_frame)
        last_frame[0] = -1
        refresh_all(force_candidate=True)
        status_label.value = f"Undid last {op.get('type', 'operation')}."

    def reset_selected() -> None:
        session.reset_selections()
        refresh_selection_layers()
        refresh_status_labels()
        status_label.value = "Selections reset."

    connect_button.clicked.connect(connect_selected)
    break_button.clicked.connect(break_selected)
    complete_button.clicked.connect(complete_selected)
    undo_button.clicked.connect(undo_last)
    reset_button.clicked.connect(reset_selected)

    @viewer.bind_key("Escape", overwrite=True)
    def _reset_shortcut(_viewer):
        reset_selected()

    def on_dims_change(_event=None) -> None:
        frame = current_frame()
        if frame == last_frame[0]:
            return
        last_frame[0] = frame
        refresh_current_candidate_cells()
        refresh_selection_layers()
        refresh_status_labels()

    viewer.dims.events.current_step.connect(on_dims_change)

    # Initial state and resume selection highlights.
    refresh_all(force_candidate=True)

    print("=" * 96)
    print("BIOHUB TRACK ANNOTATOR")
    print("=" * 96)
    print(f"sample             : {sample_id}")
    print(f"source             : {source.root}")
    print(f"output             : {output.root}")
    print(f"detections         : {len(valid_nodes)}")
    print(f"Trackastra edges   : {len(base_edges)}")
    print(f"manual connects    : {len(session.forced_edges)}")
    print(f"manual breaks      : {len(session.broken_edges)}")
    print(f"completed nodes    : {len(session.completed_nodes)}")
    print(f"max ray distance   : {max_ray_distance_um} um")
    print("Binary Mask visible: frontmost-mask-hit picking")
    print("Binary Mask hidden : closest-centroid-to-ray picking")
    print("=" * 96)

    napari.run()
