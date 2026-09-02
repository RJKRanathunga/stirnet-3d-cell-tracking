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


def _first_nonzero_label_along_ray(
    labels: np.ndarray,
    start_point: np.ndarray,
    end_point: np.ndarray,
    *,
    samples_per_voxel: float = 4.0,
) -> int:
    """
    Traverse a spatial label volume from camera-near to camera-far.

    The caller is responsible for deriving the ray from the RAW image layer.
    This deliberately does not ask a Labels layer for a rendered value.
    """
    data = np.asarray(labels)
    start = np.asarray(
        start_point,
        dtype=np.float64,
    ).reshape(-1)
    end = np.asarray(
        end_point,
        dtype=np.float64,
    ).reshape(-1)

    if data.ndim != 3:
        raise ValueError(
            f"Ray picking expects a 3-D ZYX label frame, got {data.shape}."
        )
    if start.shape != (3,) or end.shape != (3,):
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


def ray_pick_label_from_raw(
    raw_layer,
    labels_zyx: np.ndarray,
    event,
) -> int:
    """
    Select the first positive spatial label hit by the RAW-volume camera ray.

    The ray always comes from ``Raw BioHub``. In spatial mode the same ray is
    tested against the current supervoxel frame; in tracking mode it is tested
    against the current corrected-instance frame.

    This keeps click geometry independent of which overlay is active, visible,
    translucent, or rendered differently by a Napari version.
    """
    labels = np.asarray(labels_zyx)
    if labels.ndim != 3:
        raise ValueError(
            f"Expected current-frame labels (Z,Y,X), got {labels.shape}."
        )

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
            start_point, end_point = (
                raw_layer.get_ray_intersections(
                    position=event.position,
                    view_direction=view_direction,
                    dims_displayed=dims_displayed,
                    world=True,
                )
            )
        except TypeError:
            try:
                start_point, end_point = (
                    raw_layer.get_ray_intersections(
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
                start_point,
                dtype=np.float64,
            ).reshape(-1)
            end = np.asarray(
                end_point,
                dtype=np.float64,
            ).reshape(-1)

            if (
                start.size >= 3
                and end.size >= 3
            ):
                return _first_nonzero_label_along_ray(
                    labels,
                    start[-3:],
                    end[-3:],
                )

    # 2-D compatibility path: convert the canvas/world click through the raw
    # layer and sample only the spatial coordinates.
    try:
        data_position = np.asarray(
            raw_layer.world_to_data(
                event.position
            ),
            dtype=np.float64,
        ).reshape(-1)
    except Exception:
        return 0

    if data_position.size < 3:
        return 0

    index = np.rint(
        data_position[-3:]
    ).astype(np.int64)
    shape = np.asarray(
        labels.shape,
        dtype=np.int64,
    )
    if (
        np.any(index < 0)
        or np.any(index >= shape)
    ):
        return 0

    value = int(
        labels[
            tuple(index.tolist())
        ]
    )
    return value if value > 0 else 0


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


def filter_track_rows_for_hidden(
    frame: pd.DataFrame,
    hidden_nodes: set[Node],
) -> pd.DataFrame:
    if frame.empty or not hidden_nodes:
        return frame.copy()

    keep = [
        (
            int(row.frame),
            int(row.cell_id),
        )
        not in hidden_nodes
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


def filter_diagnostic_track_rows(
    frame: pd.DataFrame,
    *,
    category: str,
    hidden_nodes: set[Node],
    unresolved_start_nodes: set[Node],
    unresolved_end_nodes: set[Node],
) -> pd.DataFrame:
    """
    Remove diagnostic tracks whose ORIGINAL endpoint is no longer unresolved.

    After Continue/Birth resolves one gap, that old red/lime diagnostic
    disappears even if the resulting component still contains another break.
    """
    if frame.empty:
        return frame.copy()

    if category not in {
        "new",
        "broken",
    }:
        return filter_track_rows_for_hidden(
            frame,
            hidden_nodes,
        )

    unresolved = (
        unresolved_start_nodes
        if category == "new"
        else unresolved_end_nodes
    )

    pieces: list[pd.DataFrame] = []
    for _track_id, original_group in frame.groupby(
        "track_id",
        sort=False,
    ):
        ordered = original_group.sort_values(
            "frame"
        )
        endpoint = (
            ordered.iloc[0]
            if category == "new"
            else ordered.iloc[-1]
        )
        node = (
            int(endpoint["frame"]),
            int(endpoint["cell_id"]),
        )
        if node not in unresolved:
            continue

        visible_piece = filter_track_rows_for_hidden(
            original_group,
            hidden_nodes,
        )
        if not visible_piece.empty:
            pieces.append(
                visible_piece
            )

    if not pieces:
        return frame.iloc[0:0].copy()

    return pd.concat(
        pieces,
        ignore_index=True,
    )


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
