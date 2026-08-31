from __future__ import annotations

"""Napari viewer and ray-picking UI for merged-instance correction."""

from dataset_curation.annotation.instances.split import (
    AnnotationError,
    DEFAULT_SPACING_ZYX_UM,
)

from dataset_curation.annotation.instances.session import (
    AnnotationSession,
    _dominant_parent_instance,
    parse_supervoxel_group,
)

from dataset_curation.annotation.instances.io import (
    AnnotationError,
)

import math

import napari

import numpy as np

from scipy import ndimage

try:
    from magicgui.widgets import Container, Label, LineEdit, PushButton
except ImportError as exc:
    raise ImportError(
        "magicgui is required for the annotation panel. It normally comes with "
        "Napari. Install it with: pip install magicgui"
    ) from exc

try:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QSizePolicy
except ImportError:
    Qt = None
    QSizePolicy = None

LABEL_SURFACE_OFFSET_PX = 0.5

LABEL_MIN_SEPARATION_PX = 10.0

LABEL_REPOSITION_RADIUS_PX = 24

SUPERVOXEL_CONTOUR_WIDTH = 1

def _centroid_2d(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise AnnotationError("Cannot compute centroid of an empty mask.")
    return coords.mean(axis=0).astype(np.float64)

def _fallback_nearest_background(
    foreground_2d: np.ndarray,
    source_yx: np.ndarray,
) -> np.ndarray:
    source = np.rint(source_yx).astype(int)
    source[0] = np.clip(source[0], 0, foreground_2d.shape[0] - 1)
    source[1] = np.clip(source[1], 0, foreground_2d.shape[1] - 1)

    if not foreground_2d[tuple(source)]:
        return source.astype(np.float64)

    # For every foreground pixel, scipy returns the coordinate of its nearest
    # zero/background pixel.
    _, nearest = ndimage.distance_transform_edt(
        foreground_2d,
        return_indices=True,
    )
    y = int(nearest[0, source[0], source[1]])
    x = int(nearest[1, source[0], source[1]])
    return np.array([y, x], dtype=np.float64)

def _ray_to_background(
    foreground_2d: np.ndarray,
    start_yx: np.ndarray,
    direction_yx: np.ndarray,
) -> np.ndarray:
    norm = float(np.linalg.norm(direction_yx))
    if norm < 1e-6:
        return _fallback_nearest_background(foreground_2d, start_yx)

    unit = direction_yx / norm
    h, w = foreground_2d.shape
    max_steps = int(math.ceil(math.hypot(h, w))) + 2

    last_inside = np.asarray(start_yx, dtype=np.float64)

    for step in range(max_steps):
        p = start_yx + unit * float(step)
        y, x = np.rint(p).astype(int)

        if y < 0 or y >= h or x < 0 or x >= w:
            break

        last_inside = p
        if not foreground_2d[y, x]:
            candidate = p + unit * LABEL_SURFACE_OFFSET_PX
            candidate[0] = np.clip(candidate[0], 0, h - 1)
            candidate[1] = np.clip(candidate[1], 0, w - 1)

            cy, cx = np.rint(candidate).astype(int)
            if not foreground_2d[cy, cx]:
                return candidate

            return p

    return _fallback_nearest_background(foreground_2d, last_inside)

def _reposition_to_avoid_overlap(
    candidate_yx: np.ndarray,
    foreground_2d: np.ndarray,
    used_yx: list[np.ndarray],
) -> np.ndarray:
    if not used_yx:
        return candidate_yx

    def valid(point: np.ndarray) -> bool:
        y, x = np.rint(point).astype(int)
        if y < 0 or y >= foreground_2d.shape[0]:
            return False
        if x < 0 or x >= foreground_2d.shape[1]:
            return False
        if foreground_2d[y, x]:
            return False
        return all(
            np.linalg.norm(point - previous) >= LABEL_MIN_SEPARATION_PX
            for previous in used_yx
        )

    if valid(candidate_yx):
        return candidate_yx

    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)

    for radius in range(2, LABEL_REPOSITION_RADIUS_PX + 1, 2):
        for angle in angles:
            offset = radius * np.array(
                [math.sin(angle), math.cos(angle)],
                dtype=np.float64,
            )
            point = candidate_yx + offset
            if valid(point):
                return point

    return candidate_yx

def compute_supervoxel_label_anchor(
    sv_frame: np.ndarray,
    instance_frame: np.ndarray,
    foreground_frame: np.ndarray,
    sv_id: int,
    used_by_z: dict[int, list[np.ndarray]],
) -> tuple[float, float, float, float, float]:
    """Return z, outside-label-y/x, and same-slice SV-target-y/x."""
    mask_3d = sv_frame == sv_id
    z_counts = mask_3d.reshape(mask_3d.shape[0], -1).sum(axis=1)

    if z_counts.max() <= 0:
        raise AnnotationError(f"Supervoxel {sv_id} is empty.")

    # Put both the text and leader line on the z slice where this supervoxel
    # has its largest visible cross-section.
    z = int(np.argmax(z_counts))
    sv_2d = mask_3d[z]
    sv_center = _centroid_2d(sv_2d)

    parent_id = _dominant_parent_instance(
        sv_frame,
        instance_frame,
        sv_id,
    )

    if parent_id > 0:
        parent_2d = instance_frame[z] == parent_id
        if np.any(parent_2d):
            parent_center = _centroid_2d(parent_2d)
        else:
            parent_center = sv_center
    else:
        parent_center = sv_center

    direction = sv_center - parent_center

    # A central SV can have an almost-zero radial direction. Give it a stable
    # direction based on its ID, then cast toward background.
    if np.linalg.norm(direction) < 1e-4:
        golden_angle = 2.399963229728653
        angle = float(sv_id) * golden_angle
        direction = np.array(
            [math.sin(angle), math.cos(angle)],
            dtype=np.float64,
        )

    candidate = _ray_to_background(
        foreground_frame[z],
        sv_center,
        direction,
    )

    used = used_by_z.setdefault(z, [])
    candidate = _reposition_to_avoid_overlap(
        candidate,
        foreground_frame[z],
        used,
    )
    used.append(candidate.copy())

    return (
        float(z),
        float(candidate[0]),
        float(candidate[1]),
        float(sv_center[0]),
        float(sv_center[1]),
    )

def build_supervoxel_text_points(
    supervoxels: np.ndarray,
    instances: np.ndarray,
    foreground: np.ndarray,
    timepoints: tuple[int, ...],
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[np.ndarray],
]:
    points: list[tuple[float, float, float, float]] = []
    leader_lines: list[np.ndarray] = []
    sv_ids: list[int] = []
    dataset_times: list[int] = []

    for local_t, dataset_t in enumerate(timepoints):
        frame = supervoxels[local_t]
        ids = np.unique(frame)
        ids = ids[ids > 0]

        print(
            f"[labels] t={dataset_t}: computing outside-surface positions for "
            f"{len(ids)} supervoxels"
        )

        used_by_z: dict[int, list[np.ndarray]] = {}

        for sv_id in ids.tolist():
            (
                z,
                label_y,
                label_x,
                target_y,
                target_x,
            ) = compute_supervoxel_label_anchor(
                frame,
                instances[local_t],
                foreground[local_t],
                int(sv_id),
                used_by_z,
            )

            points.append(
                (
                    float(local_t),
                    z,
                    label_y,
                    label_x,
                )
            )

            # Each leader line stays inside one t/z plane:
            #
            # outside number --------> actual supervoxel center
            #
            # This makes it visually unambiguous which number belongs to which
            # atomic region.
            leader_lines.append(
                np.asarray(
                    [
                        [
                            float(local_t),
                            z,
                            label_y,
                            label_x,
                        ],
                        [
                            float(local_t),
                            z,
                            target_y,
                            target_x,
                        ],
                    ],
                    dtype=np.float32,
                )
            )

            sv_ids.append(int(sv_id))
            dataset_times.append(int(dataset_t))

    point_array = np.asarray(points, dtype=np.float32)
    if point_array.size == 0:
        point_array = np.zeros((0, 4), dtype=np.float32)

    properties = {
        "sv_id": np.asarray(sv_ids, dtype=np.int64),
        "dataset_t": np.asarray(dataset_times, dtype=np.int64),
    }

    return point_array, properties, leader_lines

_GRAPH_COLOR_BASE_RGBA = (
    (0.90, 0.12, 0.12, 1.0),  # red
    (0.12, 0.36, 0.95, 1.0),  # blue
    (0.10, 0.74, 0.20, 1.0),  # green
    (0.92, 0.12, 0.72, 1.0),  # magenta
    (0.00, 0.74, 0.80, 1.0),  # cyan
    (0.96, 0.70, 0.05, 1.0),  # amber
    (0.54, 0.22, 0.86, 1.0),  # purple
    (0.98, 0.42, 0.05, 1.0),  # orange
    (0.42, 0.84, 0.06, 1.0),  # lime
    (0.96, 0.34, 0.55, 1.0),  # pink
    (0.18, 0.72, 0.54, 1.0),  # teal-green
    (0.43, 0.47, 0.96, 1.0),  # periwinkle
)

def _hsv_to_rgba(
    hue: float,
    saturation: float,
    value: float,
) -> tuple[float, float, float, float]:
    """Dependency-free HSV -> RGBA conversion."""
    hue = float(hue) % 1.0
    saturation = float(np.clip(saturation, 0.0, 1.0))
    value = float(np.clip(value, 0.0, 1.0))

    h6 = hue * 6.0
    sector = int(np.floor(h6)) % 6
    fraction = h6 - np.floor(h6)

    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))

    if sector == 0:
        r, g, b = value, t, p
    elif sector == 1:
        r, g, b = q, value, p
    elif sector == 2:
        r, g, b = p, value, t
    elif sector == 3:
        r, g, b = p, q, value
    elif sector == 4:
        r, g, b = t, p, value
    else:
        r, g, b = value, p, q

    return float(r), float(g), float(b), 1.0

def _display_color_for_unique_index(
    unique_index: int,
) -> tuple[float, float, float, float]:
    """
    Return a deterministic, non-repeating display color for one label rank.

    Consecutive indices are deliberately far apart in hue using golden-ratio
    stepping. That is particularly useful here because neighbouring watershed
    IDs are often numerically close.

    For the few hundred labels in these BioHub frames this produces a large
    practical palette without exact color reuse.
    """
    unique_index = int(unique_index)

    if unique_index < 0:
        raise ValueError(
            f"unique_index must be non-negative, got {unique_index}"
        )

    # Keep the first few colors maximally obvious.
    if unique_index < len(_GRAPH_COLOR_BASE_RGBA):
        return _GRAPH_COLOR_BASE_RGBA[unique_index]

    extra = unique_index - len(_GRAPH_COLOR_BASE_RGBA)

    golden_ratio_conjugate = 0.6180339887498949
    hue = (
        0.03
        + (extra + 1) * golden_ratio_conjugate
    ) % 1.0

    # Cycle saturation/value independently from hue. This increases separation
    # between colors whose hue eventually comes close after many labels.
    saturation_cycle = (
        0.82,
        0.68,
        0.92,
        0.74,
    )
    value_cycle = (
        0.98,
        0.86,
        0.94,
    )

    saturation = saturation_cycle[
        extra % len(saturation_cycle)
    ]
    value = value_cycle[
        (extra // len(saturation_cycle))
        % len(value_cycle)
    ]

    return _hsv_to_rgba(
        hue,
        saturation,
        value,
    )

def _build_frame_touch_adjacency(
    labels_zyx: np.ndarray,
) -> tuple[
    dict[int, set[int]],
    set[int],
]:
    """
    Build the 6-neighbour face-contact graph for ONE 3-D label volume.

    Two positive labels are adjacent iff at least one z/y/x voxel face separates
    them. Background 0 is excluded.
    """
    frame = np.asarray(labels_zyx)

    if frame.ndim != 3:
        raise ValueError(
            "Frame adjacency expects (Z,Y,X), got "
            f"{frame.shape}."
        )

    positive_ids = np.unique(frame)
    positive_ids = positive_ids[positive_ids > 0]

    all_labels = {
        int(value)
        for value in positive_ids.tolist()
    }
    adjacency: dict[int, set[int]] = {
        label_id: set()
        for label_id in all_labels
    }

    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)

        a = frame[tuple(left)]
        b = frame[tuple(right)]

        touching = (
            (a > 0)
            & (b > 0)
            & (a != b)
        )

        if not np.any(touching):
            continue

        aa = a[touching].astype(
            np.int64,
            copy=False,
        )
        bb = b[touching].astype(
            np.int64,
            copy=False,
        )

        pairs = np.stack(
            [
                np.minimum(aa, bb),
                np.maximum(aa, bb),
            ],
            axis=1,
        )

        # Thousands of voxel faces can represent the same graph edge.
        pairs = np.unique(
            pairs,
            axis=0,
        )

        for u, v in pairs.tolist():
            u = int(u)
            v = int(v)

            adjacency.setdefault(u, set()).add(v)
            adjacency.setdefault(v, set()).add(u)

            all_labels.add(u)
            all_labels.add(v)

    return adjacency, all_labels

def _build_frame_graph_cache(
    labels_tzyx: np.ndarray,
) -> list[
    tuple[
        dict[int, set[int]],
        set[int],
    ]
]:
    """
    Cache one contact graph per selected timepoint.

    This intentionally does NOT connect labels between t and t+1.
    """
    data = np.asarray(labels_tzyx)

    if data.ndim == 3:
        return [
            _build_frame_touch_adjacency(data)
        ]

    if data.ndim != 4:
        raise ValueError(
            "Graph-coloring expects (Z,Y,X) or (T,Z,Y,X), got "
            f"{data.shape}."
        )

    return [
        _build_frame_touch_adjacency(
            data[t]
        )
        for t in range(data.shape[0])
    ]

def _merge_frame_graph_cache(
    frame_graphs: list[
        tuple[
            dict[int, set[int]],
            set[int],
        ]
    ],
) -> tuple[
    dict[int, set[int]],
    set[int],
]:
    """
    Merge per-frame graphs by LABEL VALUE.

    Napari Labels colors are keyed by label value, not by (time,label), so if
    values 12 and 19 touch in any selected frame they must receive different
    global display colors. This union graph guarantees that.
    """
    merged_adjacency: dict[int, set[int]] = {}
    all_labels: set[int] = set()

    for adjacency, labels in frame_graphs:
        all_labels.update(
            int(value)
            for value in labels
        )

        for node, neighbours in adjacency.items():
            target = merged_adjacency.setdefault(
                int(node),
                set(),
            )
            target.update(
                int(value)
                for value in neighbours
            )

    for label_id in all_labels:
        merged_adjacency.setdefault(
            int(label_id),
            set(),
        )

    return merged_adjacency, all_labels

def _color_dict_from_frame_graph_cache(
    frame_graphs: list[
        tuple[
            dict[int, set[int]],
            set[int],
        ]
    ],
) -> tuple[
    dict[int, tuple[float, float, float, float]],
    dict[str, int],
]:
    """
    Build a UNIQUE label-value -> RGBA mapping.

    The frame contact graphs are still merged for diagnostics, but unlike the
    previous graph-coloring implementation we never reuse a display color
    between two different positive label IDs.
    """
    adjacency, all_labels = (
        _merge_frame_graph_cache(
            frame_graphs
        )
    )

    ordered_labels = sorted(
        int(label_id)
        for label_id in all_labels
    )

    color_dict: dict[
        int,
        tuple[float, float, float, float],
    ] = {
        0: (0.0, 0.0, 0.0, 0.0),
    }

    for unique_index, label_id in enumerate(
        ordered_labels
    ):
        color_dict[int(label_id)] = (
            _display_color_for_unique_index(
                int(unique_index)
            )
        )

    edge_count = (
        sum(
            len(neighbours)
            for neighbours in adjacency.values()
        )
        // 2
    )

    # Defensive invariant: every touching pair must have different RGBA values.
    # This is now implied by unique-per-label coloring, but checking it here
    # protects future modifications to the color generator.
    for node, neighbours in adjacency.items():
        node_color = color_dict[int(node)]

        for neighbour in neighbours:
            if node_color == color_dict[int(neighbour)]:
                raise RuntimeError(
                    "Unique display-color invariant failed for touching "
                    f"labels {node} and {neighbour}."
                )

    stats = {
        "label_count": int(len(ordered_labels)),
        "touch_edge_count": int(edge_count),
        "color_count": int(len(ordered_labels)),
    }

    return color_dict, stats

def _apply_label_color_dict(
    layer,
    color_dict: dict[
        int,
        tuple[float, float, float, float],
    ],
) -> None:
    """
    Update a Napari Labels layer's explicit label -> RGBA mapping.

    Napari's API differs across versions:
        newer: layer.color = mapping
        older: layer.color_mode / internal direct-color machinery may be needed

    Try public APIs first and only then fall back to the layer's direct-colormap
    interface if exposed.
    """
    first_error = None

    try:
        layer.color = color_dict
        layer.refresh()
        return
    except Exception as exc:
        first_error = exc

    # Some versions expose a setter through the property but require direct
    # color mode before assigning the mapping.
    try:
        if hasattr(layer, "color_mode"):
            try:
                layer.color_mode = "direct"
            except Exception:
                pass

        layer.color = color_dict
        layer.refresh()
        return
    except Exception:
        pass

    # Older Labels implementations may expose `_direct_colormap` or
    # `direct_colormap`. We avoid assuming one exact class/API shape.
    for attr_name in (
        "direct_colormap",
        "_direct_colormap",
    ):
        if not hasattr(layer, attr_name):
            continue

        try:
            colormap = getattr(layer, attr_name)

            if hasattr(colormap, "color_dict"):
                colormap.color_dict = color_dict
                layer.refresh()
                return

            if hasattr(colormap, "colors"):
                colormap.colors = color_dict
                layer.refresh()
                return
        except Exception:
            continue

    raise RuntimeError(
        "This Napari version did not accept the adjacency-aware Labels color "
        "mapping after layer creation either. "
        f"Original error: {first_error}"
    )

def _coerce_positive_label_value(value) -> int:
    """
    Convert Napari Labels.get_value() output to one positive integer label.

    Most Napari versions return a scalar label for Labels. We deliberately
    reject ambiguous multi-value outputs and fall back to explicit ray
    traversal instead of guessing.
    """
    if value is None:
        return 0

    array = np.asarray(value)

    if array.ndim == 0:
        try:
            result = int(array.item())
        except (TypeError, ValueError, OverflowError):
            return 0
        return result if result > 0 else 0

    if array.size == 1:
        try:
            result = int(array.reshape(-1)[0])
        except (TypeError, ValueError, OverflowError):
            return 0
        return result if result > 0 else 0

    return 0

def _first_nonzero_label_along_data_ray(
    labels: np.ndarray,
    start_point: np.ndarray,
    end_point: np.ndarray,
    *,
    samples_per_voxel: float = 4.0,
) -> int:
    """
    Return the first positive label encountered from start_point -> end_point.

    start_point is the camera-near intersection supplied by Napari, so traversal
    order directly implements "frontmost visible cell".

    The points are in full n-D layer data coordinates. Sampling at 4 samples per
    voxel along the largest-changing dimension is intentionally conservative for
    label picking while still requiring only ~10^3 samples for these volumes.
    """
    data = np.asarray(labels)
    start = np.asarray(start_point, dtype=np.float64).reshape(-1)
    end = np.asarray(end_point, dtype=np.float64).reshape(-1)

    if start.shape != end.shape:
        return 0
    if start.size != data.ndim:
        return 0
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        return 0

    delta = end - start
    max_axis_distance = float(np.max(np.abs(delta)))

    sample_count = max(
        2,
        int(np.ceil(max_axis_distance * float(samples_per_voxel))) + 1,
    )

    shape = np.asarray(data.shape, dtype=np.int64)

    for alpha in np.linspace(
        0.0,
        1.0,
        sample_count,
        endpoint=True,
        dtype=np.float64,
    ):
        point = start + alpha * delta
        index = np.rint(point).astype(np.int64)

        if np.any(index < 0) or np.any(index >= shape):
            continue

        value = int(data[tuple(index.tolist())])
        if value > 0:
            return value

    return 0

def _ray_pick_frontmost_label(
    layer,
    event,
) -> int:
    """
    Pick the frontmost positive Labels value under a Napari mouse event.

    Preferred path:
        Labels.get_value(... view_direction ...)

    Fallback:
        get_ray_intersections() + explicit front-to-back label traversal.
    """
    view_direction = getattr(event, "view_direction", None)
    dims_displayed = getattr(event, "dims_displayed", None)

    # In true 3-D, ask Napari for its native ray-aware top value first.
    if view_direction is not None and dims_displayed is not None:
        try:
            value = layer.get_value(
                event.position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=True,
            )
            label_id = _coerce_positive_label_value(value)
            if label_id > 0:
                return label_id
        except Exception:
            # Fall through to explicit ray traversal.
            pass

        try:
            start_point, end_point = layer.get_ray_intersections(
                position=event.position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=True,
            )
        except TypeError:
            # Compatibility with versions accepting positional parameters.
            try:
                start_point, end_point = layer.get_ray_intersections(
                    event.position,
                    view_direction,
                    dims_displayed,
                    world=True,
                )
            except Exception:
                start_point, end_point = None, None
        except Exception:
            start_point, end_point = None, None

        if start_point is not None and end_point is not None:
            return _first_nonzero_label_along_data_ray(
                np.asarray(layer.data),
                np.asarray(start_point),
                np.asarray(end_point),
            )

    # 2-D compatibility path. It is not the main workflow, but clicking still
    # selects the label directly under the cursor if the user switches ndisplay.
    try:
        value = layer.get_value(
            event.position,
            world=True,
        )
        return _coerce_positive_label_value(value)
    except Exception:
        return 0

def _make_magicgui_label_horizontally_shrinkable(widget) -> None:
    """Prevent long QLabel text from forcing the entire dock width."""
    native = getattr(widget, "native", None)
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

def _configure_resizable_annotation_dock(dock_widget, panel) -> None:
    """
    Keep the annotation panel horizontally resizable.

    The critical part is allowing child widgets, especially long status/error
    labels, to shrink. Otherwise Qt's sizeHint can make the dock appear locked
    at a huge width after one long error message.
    """
    panel_native = getattr(panel, "native", None)

    if panel_native is not None:
        try:
            panel_native.setMinimumWidth(260)
        except Exception:
            pass

        if QSizePolicy is not None:
            try:
                panel_native.setSizePolicy(
                    QSizePolicy.Preferred,
                    QSizePolicy.Expanding,
                )
            except Exception:
                pass

    # Napari returns a Qt dock widget in current versions. Keep only a modest
    # minimum width and no artificial maximum width. The divider between the
    # canvas and dock can then be dragged left/right normally.
    if dock_widget is not None:
        try:
            dock_widget.setMinimumWidth(280)
        except Exception:
            pass

        try:
            dock_widget.setMaximumWidth(16_777_215)
        except Exception:
            pass

        try:
            if Qt is not None and hasattr(dock_widget, "setFeatures"):
                features = dock_widget.features()
                dock_widget.setFeatures(features)
        except Exception:
            pass

def make_viewer(
    *,
    sample_id: str,
    timepoints: tuple[int, ...],
    raw: np.ndarray,
    stage6_binary_mask: np.ndarray,
    supervoxels: np.ndarray,
    foreground: np.ndarray,
    session: AnnotationSession,
    suspect_instances: np.ndarray | None = None,
) -> napari.Viewer:
    print()
    print("=" * 72)
    print("Building supervoxel text positions")
    print("=" * 72)

    (
        text_points,
        text_properties,
        leader_lines,
    ) = build_supervoxel_text_points(
        supervoxels,
        session.base_instances,
        foreground,
        timepoints,
    )

    viewer = napari.Viewer(ndisplay=2)
    scale_4d = (1.0, *DEFAULT_SPACING_ZYX_UM)

    print()
    print("=" * 72)
    print("Computing unique per-label display colors")
    print("=" * 72)

    supervoxel_frame_graphs = (
        _build_frame_graph_cache(
            supervoxels
        )
    )
    instance_frame_graphs = (
        _build_frame_graph_cache(
            session.corrected
        )
    )

    (
        supervoxel_color_dict,
        supervoxel_color_stats,
    ) = _color_dict_from_frame_graph_cache(
        supervoxel_frame_graphs
    )

    (
        instance_color_dict,
        instance_color_stats,
    ) = _color_dict_from_frame_graph_cache(
        instance_frame_graphs
    )

    print(
        "[display colors] supervoxels: "
        f"{supervoxel_color_stats['label_count']} label values | "
        f"{supervoxel_color_stats['touch_edge_count']} touching pairs | "
        f"{supervoxel_color_stats['color_count']} unique colors"
    )
    print(
        "[display colors] instances: "
        f"{instance_color_stats['label_count']} label values | "
        f"{instance_color_stats['touch_edge_count']} touching pairs | "
        f"{instance_color_stats['color_count']} unique colors"
    )

    viewer.dims.axis_labels = (
        "annotation frame",
        "z",
        "y",
        "x",
    )

    # ----------------------------------------
    # Raw image
    # ----------------------------------------
    viewer.add_image(
        raw,
        name="Raw BioHub",
        scale=scale_4d,
        colormap="gray",
    )

    # ----------------------------------------
    # Stage-6 binary foreground mask
    # ----------------------------------------
    viewer.add_labels(
        stage6_binary_mask,
        name="Stage-6 binary mask",
        scale=scale_4d,
        opacity=0.30,
        visible=False,
    )

    # ----------------------------------------
    # Current corrected pseudo-GT instances
    # ----------------------------------------
    corrected_layer = viewer.add_labels(
        session.corrected,
        name="Corrected instances",
        scale=scale_4d,
        opacity=1.0,
    )

    # Older Napari versions do not accept color= in viewer.add_labels(), but
    # they can still accept the explicit label->RGBA mapping on the created
    # Labels layer. Apply it after construction for compatibility.
    _apply_label_color_dict(
        corrected_layer,
        instance_color_dict,
    )

    # ----------------------------------------
    # Merge-suspect predicted instances
    # ----------------------------------------
    # Static read-only visualization of the ORIGINAL prediction. Save/Undo do
    # not mutate this layer or any existing annotation state.
    if suspect_instances is not None:
        suspect_layer = viewer.add_labels(
            suspect_instances,
            name="Suspect predicted instances",
            scale=scale_4d,
            opacity=1.0,
            visible=False,
        )
        _apply_label_color_dict(
            suspect_layer,
            instance_color_dict,
        )

    # ----------------------------------------
    # Ray-picked seed supervoxel highlights
    # ----------------------------------------
    #
    # One lightweight 3-D mask layer per seed slot gives an unambiguous visual
    # mapping between click order and input box:
    #
    #   Instance 1 -> red
    #   Instance 2 -> blue
    #   Instance 3 -> green
    #   Instance 4 -> magenta
    #
    # These are current-frame-only masks, not another full 20-frame stack.
    seed_highlight_layers = []

    for slot_index, (layer_name, colormap) in enumerate(
        (
            ("Seed 1 highlight", "red"),
            ("Seed 2 highlight", "blue"),
            ("Seed 3 highlight", "green"),
            ("Seed 4 highlight", "magenta"),
        ),
        start=1,
    ):
        seed_layer = viewer.add_image(
            np.zeros(
                session.corrected.shape[1:],
                dtype=np.uint8,
            ),
            name=layer_name,
            scale=DEFAULT_SPACING_ZYX_UM,
            colormap=colormap,
            contrast_limits=(0, 1),
            opacity=0.78,
            blending="additive",
            visible=True,
        )
        seed_highlight_layers.append(seed_layer)

    # ----------------------------------------
    # Atomic supervoxel boundaries
    # ----------------------------------------
    supervoxel_layer = viewer.add_labels(
        supervoxels,
        name="Atomic supervoxel boundaries",
        scale=scale_4d,
        opacity=0.95,
    )

    _apply_label_color_dict(
        supervoxel_layer,
        supervoxel_color_dict,
    )

    try:
        supervoxel_layer.contour = SUPERVOXEL_CONTOUR_WIDTH
    except Exception:
        # Older Napari versions may not expose contour on Labels.
        supervoxel_layer.opacity = 0.25
        print(
            "[napari] Labels.contour is unavailable in this version; "
            "showing translucent supervoxel fills instead."
        )

    # ----------------------------------------
    # Supervoxel ID text
    # ----------------------------------------
    text_spec = {
        "string": "{sv_id}",
        "size": 11,
        "color": "white",
        "anchor": "center",
    }

    # `properties` is supported by old and current Napari versions and is enough
    # for the text-format placeholder.
    text_layer = viewer.add_points(
        text_points,
        ndim=4,
        name="Supervoxel IDs",
        properties=text_properties,
        text=text_spec,
        scale=scale_4d,
        size=1,
        face_color="transparent",
    )

    try:
        text_layer.out_of_slice_display = False
    except Exception:
        pass

    # Red leader lines make the outside numbering unambiguous.
    if leader_lines:
        viewer.add_shapes(
            leader_lines,
            shape_type="line",
            name="SV number leader lines",
            scale=scale_4d,
            edge_color="red",
            edge_width=1.5,
            opacity=0.85,
        )

    # ----------------------------------------
    # Right-side annotation controls
    # ----------------------------------------
    current_frame_label = Label(value="")
    picked_instance_label = Label(
        value="Selected seed supervoxels: none"
    )
    instruction_label = Label(
        value=(
            "Every different SV/instance ID gets its own display color.\n"
            "Single-click visible supervoxels to add split seeds.\n"
            "1st click -> Instance 1; 2nd -> Instance 2.\n"
            "Further clicks use Instance 3/4 if needed.\n"
            "Unentered SVs are assigned automatically."
        )
    )

    box1 = LineEdit(
        label="Instance 1",
    )
    box2 = LineEdit(
        label="Instance 2",
    )
    box3 = LineEdit(
        label="Instance 3",
    )
    box4 = LineEdit(
        label="Instance 4",
    )

    save_button = PushButton(text="Save")
    reset_button = PushButton(text="Reset selections")
    undo_button = PushButton(text="Undo last Save")
    status_label = Label(
        value=(
            "Ready. Click two visible SVs, then Save. Esc resets; Ctrl+Z undoes."
        )
    )

    # Long errors/status messages must wrap instead of expanding the panel's
    # minimum width. This keeps the dock divider freely draggable left/right.
    for label_widget in (
        current_frame_label,
        picked_instance_label,
        instruction_label,
        status_label,
    ):
        _make_magicgui_label_horizontally_shrinkable(
            label_widget
        )

    panel = Container(
        widgets=[
            current_frame_label,
            picked_instance_label,
            instruction_label,
            box1,
            box2,
            box3,
            box4,
            save_button,
            reset_button,
            undo_button,
            status_label,
        ],
        layout="vertical",
        labels=True,
    )

    annotation_dock = viewer.window.add_dock_widget(
        panel,
        area="right",
        name="Merged-cell correction",
    )

    _configure_resizable_annotation_dock(
        annotation_dock,
        panel,
    )

    boxes = (box1, box2, box3, box4)

    def current_local_t() -> int:
        return int(round(viewer.dims.current_step[0]))

    def refresh_instance_graph_colors(
        changed_local_t: int,
    ) -> None:
        """
        Rebuild only the changed frame's raster contact graph, then recolor the
        union graph across all selected frames.
        """
        instance_frame_graphs[changed_local_t] = (
            _build_frame_touch_adjacency(
                session.corrected[
                    changed_local_t
                ]
            )
        )

        color_dict, stats = (
            _color_dict_from_frame_graph_cache(
                instance_frame_graphs
            )
        )

        _apply_label_color_dict(
            corrected_layer,
            color_dict,
        )

        print(
            "[display colors] refreshed instances: "
            f"{stats['label_count']} label values | "
            f"{stats['touch_edge_count']} touching pairs | "
            f"{stats['color_count']} unique colors"
        )

    selected_seed_ids: list[int | None] = [
        None,
        None,
        None,
        None,
    ]

    def _refresh_seed_status_label() -> None:
        parts = []
        for slot_index, sv_id in enumerate(
            selected_seed_ids,
            start=1,
        ):
            if sv_id is not None:
                parts.append(
                    f"{slot_index}: SV {int(sv_id)}"
                )

        picked_instance_label.value = (
            "Selected seed supervoxels: "
            + (
                ", ".join(parts)
                if parts
                else "none"
            )
        )

    def _clear_seed_highlights() -> None:
        for layer in seed_highlight_layers:
            layer.data = np.zeros(
                session.corrected.shape[1:],
                dtype=np.uint8,
            )
            layer.refresh()

    def clear_ray_selection() -> None:
        # Kept under the previous helper name so Save/Undo/time-change call
        # sites continue to reset the click-selection state.
        for index in range(4):
            selected_seed_ids[index] = None

        _clear_seed_highlights()
        _refresh_seed_status_label()

    def show_seed_selection(
        local_t: int,
        slot_index: int,
        sv_id: int,
    ) -> None:
        if not (0 <= slot_index < 4):
            raise AnnotationError(
                f"Invalid seed slot index: {slot_index}"
            )

        sv_mask = (
            supervoxels[local_t] == int(sv_id)
        )

        if not np.any(sv_mask):
            raise AnnotationError(
                f"Supervoxel {sv_id} is absent from the current frame."
            )

        selected_seed_ids[slot_index] = int(sv_id)

        layer = seed_highlight_layers[slot_index]
        layer.data = sv_mask.astype(
            np.uint8,
            copy=False,
        )
        layer.refresh()

        _refresh_seed_status_label()

    def next_empty_seed_slot() -> int | None:
        for slot_index, sv_id in enumerate(
            selected_seed_ids
        ):
            if sv_id is None:
                return slot_index
        return None

    def clear_boxes() -> None:
        for box in boxes:
            box.value = ""


    def reset_selections(
        *,
        message: str | None = None,
    ) -> None:
        clear_boxes()
        clear_ray_selection()

        if message is not None:
            status_label.value = message

    def update_frame_status() -> None:
        local_t = current_local_t()
        dataset_t = timepoints[local_t]
        current_frame_label.value = (
            f"Frame {local_t + 1}/{len(timepoints)}  "
            f"(dataset t={dataset_t})\n"
            f"Corrected merged instances here: "
            f"{session.corrections_in_frame(local_t)}"
        )

        try:
            undo_button.enabled = session.can_undo()
        except Exception:
            pass

    def show_wrapped_error(exc: Exception) -> None:
        error_text = str(exc)
        words = error_text.split()
        wrapped_lines: list[str] = []
        current_line = ""

        for word in words:
            candidate = (
                word
                if not current_line
                else f"{current_line} {word}"
            )

            if len(candidate) > 72 and current_line:
                wrapped_lines.append(current_line)
                current_line = word
            else:
                current_line = candidate

        if current_line:
            wrapped_lines.append(current_line)

        status_label.value = (
            "ERROR: " + "\n".join(wrapped_lines)
        )

        print()
        print("[annotation error]")
        print(exc)

    @viewer.mouse_drag_callbacks.append
    def ray_pick_visible_supervoxel(_viewer, event):
        """
        Plain single-click:
            ray-pick frontmost atomic supervoxel
            -> next empty seed box
            -> corresponding colored seed highlight

        Click-drag:
            leave normal Napari camera navigation untouched.
        """
        button = getattr(event, "button", None)
        button_text = str(button).lower()

        is_left = (
            button is None
            or button == 1
            or "left" in button_text
            or button_text == "1"
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

        local_t = current_local_t()

        sv_id = _ray_pick_frontmost_label(
            supervoxel_layer,
            event,
        )

        if sv_id <= 0:
            status_label.value = (
                "Ray click hit background; no seed was added."
            )
            return

        if sv_id in {
            int(value)
            for value in selected_seed_ids
            if value is not None
        }:
            status_label.value = (
                f"SV {sv_id} is already selected as a seed."
            )
            return

        slot_index = next_empty_seed_slot()

        if slot_index is None:
            status_label.value = (
                "All four seed boxes are already filled. "
                "Press Reset selections or Save."
            )
            return

        boxes[slot_index].value = str(int(sv_id))
        show_seed_selection(
            local_t,
            slot_index,
            int(sv_id),
        )

        status_label.value = (
            f"Added SV {int(sv_id)} to Instance {slot_index + 1}."
        )

    def save_current_split() -> None:
        try:
            groups = [
                parse_supervoxel_group(str(box.value))
                for box in boxes
            ]
            result = session.apply_split(
                current_local_t(),
                groups,
            )
        except Exception as exc:
            show_wrapped_error(exc)
            return

        # The backing array was modified in place. Reassigning + refresh is
        # intentional: it forces Napari to rebuild the label display immediately
        # so the corrected case visibly changes color.
        corrected_layer.data = session.corrected
        corrected_layer.refresh()

        refresh_instance_graph_colors(
            current_local_t()
        )

        reset_selections()
        update_frame_status()

        status_label.value = (
            f"Saved t={result.timepoint}: original instance "
            f"{result.original_instance_id} -> "
            f"{len(result.output_instance_ids)} instances "
            f"{result.output_instance_ids}"
        )

        print()
        print("=" * 72)
        print("CORRECTION SAVED")
        print("=" * 72)
        print(f"Dataset timepoint : {result.timepoint}")
        print(f"Original instance : {result.original_instance_id}")
        for index, (seed_group, sv_group, output_id) in enumerate(
            zip(
                result.seed_groups,
                result.groups,
                result.output_instance_ids,
            ),
            start=1,
        ):
            print(
                f"True instance {index}: seeds={list(seed_group)} | "
                f"assigned_SVs={list(sv_group)} -> label {output_id}"
            )
        print(f"Output directory  : {session.output_dir}")
        print()

    def undo_last_operation() -> None:
        try:
            result = session.undo_last_split()
        except Exception as exc:
            show_wrapped_error(exc)
            return

        # Force Napari to repaint the restored labels.
        corrected_layer.data = session.corrected
        corrected_layer.refresh()

        undo_local_t = timepoints.index(
            int(result.timepoint)
        )

        refresh_instance_graph_colors(
            undo_local_t
        )

        reset_selections()

        # Undo is global/LIFO across the loaded sequence. If the user has moved
        # elsewhere since Save, jump back to the affected frame so the restored
        # cell is immediately visible.
        viewer.dims.set_current_step(
            0,
            undo_local_t,
        )

        update_frame_status()

        status_label.value = (
            f"Undid last Save at t={result.timepoint}: restored instance "
            f"{result.original_instance_id}; removed split labels "
            f"{result.removed_instance_ids}"
        )

        print()
        print("=" * 72)
        print("CORRECTION UNDONE")
        print("=" * 72)
        print(f"Dataset timepoint : {result.timepoint}")
        print(f"Restored instance : {result.original_instance_id}")
        print(f"Removed labels    : {result.removed_instance_ids}")
        print(f"Output directory  : {session.output_dir}")
        print()

    save_button.changed.connect(
        lambda *_: save_current_split()
    )
    reset_button.changed.connect(
        lambda *_: reset_selections(
            message="Selections reset."
        )
    )
    undo_button.changed.connect(
        lambda *_: undo_last_operation()
    )

    @viewer.bind_key("Control-S")
    def _save_with_keyboard(_viewer):
        save_current_split()

    @viewer.bind_key("Control-Z")
    def _undo_with_keyboard(_viewer):
        undo_last_operation()

    @viewer.bind_key("Escape")
    def _reset_with_escape(_viewer):
        reset_selections(
            message="Selections reset."
        )

    # Clear stale typed IDs when the user changes TIME. Moving through z does
    # not clear them.
    last_local_t = {"value": current_local_t()}

    def on_dims_change(_event=None) -> None:
        now = current_local_t()
        if now != last_local_t["value"]:
            reset_selections()
            last_local_t["value"] = now
            status_label.value = (
                "Frame changed. Seed selections and input boxes were cleared; "
                "saved corrections remain applied."
            )
        update_frame_status()

    viewer.dims.events.current_step.connect(on_dims_change)

    # Start near the center z-slice of the first requested frame.
    try:
        viewer.dims.set_current_step(0, 0)
        viewer.dims.set_current_step(1, raw.shape[1] // 2)
    except Exception:
        pass

    update_frame_status()

    viewer.dims.ndisplay = 3

    print()
    print("=" * 72)
    print("ANNOTATION INSTRUCTIONS")
    print("=" * 72)
    print(f"Sample           : {sample_id}")
    print(f"Dataset frames   : {timepoints}")
    print()
    print("1. Navigate through the selected timepoints and z slices.")
    print("2. Find a merged spatial instance.")
    print(
        "3. Every different supervoxel ID and every different segmented "
        "instance ID gets its own display color; colors are not reused."
    )
    print(
        "4. In 3-D, SINGLE-CLICK the first visible seed supervoxel. "
        "Its ID automatically goes into Instance 1."
    )
    print(
        "5. SINGLE-CLICK the second visible seed supervoxel. "
        "Its ID automatically goes into Instance 2."
    )
    print(
        "6. Optional third/fourth clicks fill Instance 3/4."
    )
    print(
        "7. Selected seeds are highlighted with distinct colors: "
        "red, blue, green, magenta."
    )
    print(
        "8. Click-drag still rotates/pans normally; background clicks add nothing."
    )
    print(
        "9. Press Reset selections or Esc to clear all boxes/highlights "
        "without saving."
    )
    print(
        "10. Press Save (or Ctrl+S). The split is applied and selections reset."
    )
    print(
        "11. Press Undo last Save (or Ctrl+Z) to reverse the newest split."
    )
    print()
    print(
        "The tool automatically expands the seeds over the current merged "
        "instance using the weighted supervoxel contact graph."
    )
    print(
        "After Save, the split gets new instance IDs, so its colors change "
        "immediately and remain changed while you move through time."
    )
    print(f"Outputs          : {session.output_dir}")
    print("=" * 72)
    print()

    return viewer
