from __future__ import annotations

"""Napari-independent layer-data helpers for unified curation."""

import colorsys
from typing import Iterable

import numpy as np
import pandas as pd

from dataset_curation.annotation.tracks.graph import Edge, Node


_BASE_RGBA = (
    (0.90, 0.12, 0.12, 1.0),
    (0.12, 0.36, 0.95, 1.0),
    (0.10, 0.74, 0.20, 1.0),
    (0.92, 0.12, 0.72, 1.0),
    (0.00, 0.74, 0.80, 1.0),
    (0.96, 0.70, 0.05, 1.0),
    (0.54, 0.22, 0.86, 1.0),
    (0.98, 0.42, 0.05, 1.0),
    (0.42, 0.84, 0.06, 1.0),
    (0.96, 0.34, 0.55, 1.0),
    (0.18, 0.72, 0.54, 1.0),
    (0.43, 0.47, 0.96, 1.0),
)


def _color_for_rank(rank: int) -> tuple[float, float, float, float]:
    rank = int(rank)
    if rank < len(_BASE_RGBA):
        return _BASE_RGBA[rank]

    extra = rank - len(_BASE_RGBA)
    hue = (
        0.03
        + (extra + 1) * 0.6180339887498949
    ) % 1.0
    saturation = (
        0.82,
        0.68,
        0.92,
        0.74,
    )[extra % 4]
    value = (
        0.98,
        0.86,
        0.94,
    )[(extra // 4) % 3]
    r, g, b = colorsys.hsv_to_rgb(
        hue,
        saturation,
        value,
    )
    return (
        float(r),
        float(g),
        float(b),
        1.0,
    )


def label_color_dict(
    labels: np.ndarray,
) -> dict[int, tuple[float, float, float, float]]:
    ids = np.unique(
        np.asarray(labels)
    )
    ids = ids[ids > 0]
    result: dict[
        int,
        tuple[float, float, float, float],
    ] = {
        0: (
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    }
    for rank, raw_id in enumerate(
        sorted(int(v) for v in ids.tolist())
    ):
        result[int(raw_id)] = _color_for_rank(rank)
    return result


def apply_label_color_dict(
    layer,
    mapping,
) -> None:
    first_error = None
    try:
        layer.color = mapping
        layer.refresh()
        return
    except Exception as exc:
        first_error = exc

    try:
        if hasattr(layer, "color_mode"):
            try:
                layer.color_mode = "direct"
            except Exception:
                pass
        layer.color = mapping
        layer.refresh()
        return
    except Exception:
        pass

    for attr_name in (
        "direct_colormap",
        "_direct_colormap",
    ):
        if not hasattr(layer, attr_name):
            continue
        try:
            colormap = getattr(
                layer,
                attr_name,
            )
            if hasattr(
                colormap,
                "color_dict",
            ):
                colormap.color_dict = mapping
                layer.refresh()
                return
            if hasattr(
                colormap,
                "colors",
            ):
                colormap.colors = mapping
                layer.refresh()
                return
        except Exception:
            continue

    raise RuntimeError(
        "Napari did not accept the explicit label color mapping. "
        f"Original error: {first_error}"
    )


def _coerce_positive_label_value(value) -> int:
    if value is None:
        return 0
    array = np.asarray(value)
    if array.size != 1:
        return 0
    try:
        result = int(array.reshape(-1)[0])
    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return 0
    return result if result > 0 else 0


def _first_nonzero_label_along_ray(
    labels: np.ndarray,
    start_point: np.ndarray,
    end_point: np.ndarray,
    *,
    samples_per_voxel: float = 4.0,
) -> int:
    data = np.asarray(labels)
    start = np.asarray(
        start_point,
        dtype=np.float64,
    ).reshape(-1)
    end = np.asarray(
        end_point,
        dtype=np.float64,
    ).reshape(-1)

    if start.shape != end.shape:
        return 0
    if start.size != data.ndim:
        return 0
    if (
        not np.all(np.isfinite(start))
        or not np.all(np.isfinite(end))
    ):
        return 0

    delta = end - start
    max_axis_distance = float(
        np.max(np.abs(delta))
    )
    sample_count = max(
        2,
        int(
            np.ceil(
                max_axis_distance
                * float(samples_per_voxel)
            )
        )
        + 1,
    )
    shape = np.asarray(
        data.shape,
        dtype=np.int64,
    )

    for alpha in np.linspace(
        0.0,
        1.0,
        sample_count,
        endpoint=True,
        dtype=np.float64,
    ):
        point = start + alpha * delta
        index = np.rint(
            point
        ).astype(np.int64)

        if (
            np.any(index < 0)
            or np.any(index >= shape)
        ):
            continue

        value = int(
            data[
                tuple(
                    index.tolist()
                )
            ]
        )
        if value > 0:
            return value
    return 0


def ray_pick_frontmost_label(
    layer,
    event,
) -> int:
    """
    Pick the camera-nearest positive label from a current-frame 3-D layer.
    """
    view_direction = getattr(
        event,
        "view_direction",
        None,
    )
    dims_displayed = getattr(
        event,
        "dims_displayed",
        None,
    )

    if (
        view_direction is not None
        and dims_displayed is not None
    ):
        try:
            value = layer.get_value(
                event.position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=True,
            )
            label_id = _coerce_positive_label_value(
                value
            )
            if label_id > 0:
                return label_id
        except Exception:
            pass

        try:
            start_point, end_point = (
                layer.get_ray_intersections(
                    position=event.position,
                    view_direction=view_direction,
                    dims_displayed=dims_displayed,
                    world=True,
                )
            )
        except TypeError:
            try:
                start_point, end_point = (
                    layer.get_ray_intersections(
                        event.position,
                        view_direction,
                        dims_displayed,
                        world=True,
                    )
                )
            except Exception:
                start_point, end_point = (
                    None,
                    None,
                )
        except Exception:
            start_point, end_point = (
                None,
                None,
            )

        if (
            start_point is not None
            and end_point is not None
        ):
            start = np.asarray(
                start_point
            ).reshape(-1)
            end = np.asarray(
                end_point
            ).reshape(-1)
            if start.size != np.asarray(
                layer.data
            ).ndim:
                start = start[
                    -np.asarray(layer.data).ndim:
                ]
                end = end[
                    -np.asarray(layer.data).ndim:
                ]
            return _first_nonzero_label_along_ray(
                np.asarray(layer.data),
                start,
                end,
            )

    try:
        value = layer.get_value(
            event.position,
            world=True,
        )
        return _coerce_positive_label_value(
            value
        )
    except Exception:
        return 0


def edges_to_tracks_array(
    edges: Iterable[Edge],
    centers: dict[Node, np.ndarray],
) -> np.ndarray:
    rows: list[list[float]] = []
    segment_id = 1

    for left, right in sorted(edges):
        if (
            left not in centers
            or right not in centers
        ):
            continue
        left_center = np.asarray(
            centers[left],
            dtype=np.float64,
        )
        right_center = np.asarray(
            centers[right],
            dtype=np.float64,
        )
        if (
            left_center.shape != (3,)
            or right_center.shape != (3,)
        ):
            continue

        rows.append(
            [
                float(segment_id),
                float(left[0]),
                *left_center.tolist(),
            ]
        )
        rows.append(
            [
                float(segment_id),
                float(right[0]),
                *right_center.tolist(),
            ]
        )
        segment_id += 1

    if not rows:
        return np.zeros(
            (0, 5),
            dtype=np.float64,
        )
    return np.asarray(
        rows,
        dtype=np.float64,
    )


def nodes_to_points_array(
    nodes: Iterable[Node],
    centers: dict[Node, np.ndarray],
) -> np.ndarray:
    rows: list[list[float]] = []
    for node in sorted(nodes):
        center = centers.get(node)
        if center is None:
            continue
        center = np.asarray(
            center,
            dtype=np.float64,
        )
        if center.shape != (3,):
            continue
        rows.append(
            [
                float(node[0]),
                *center.tolist(),
            ]
        )

    if not rows:
        return np.zeros(
            (0, 4),
            dtype=np.float64,
        )
    return np.asarray(
        rows,
        dtype=np.float64,
    )


def filter_track_rows_for_completed(
    frame: pd.DataFrame,
    completed_nodes: set[Node],
) -> pd.DataFrame:
    if frame.empty or not completed_nodes:
        return frame.copy()

    keep = [
        (
            int(row.frame),
            int(row.cell_id),
        )
        not in completed_nodes
        for row in frame.itertuples(
            index=False
        )
    ]
    return frame.loc[
        np.asarray(
            keep,
            dtype=bool,
        )
    ].copy()


def track_frame_arrays(
    frame: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    if frame.empty:
        return (
            np.zeros(
                (0, 5),
                dtype=np.float64,
            ),
            np.zeros(
                (0, 4),
                dtype=np.float64,
            ),
            {
                "track_id": np.zeros(
                    (0,),
                    dtype=np.int64,
                ),
                "cell_id": np.zeros(
                    (0,),
                    dtype=np.int64,
                ),
            },
        )

    tracks = frame[
        [
            "track_id",
            "frame",
            "z",
            "y",
            "x",
        ]
    ].to_numpy(
        dtype=np.float64
    )
    points = frame[
        [
            "frame",
            "z",
            "y",
            "x",
        ]
    ].to_numpy(
        dtype=np.float64
    )
    properties = {
        "track_id": frame[
            "track_id"
        ].to_numpy(
            dtype=np.int64
        ),
        "cell_id": frame[
            "cell_id"
        ].to_numpy(
            dtype=np.int64
        ),
    }
    return (
        tracks,
        points,
        properties,
    )
