"""Lazy zero-padded stabilization for Trackastra pass 2."""

from __future__ import annotations

import numpy as np

from .config import GlobalMotionEstimate


def pad_frame_to_canvas(
    frame_zyx: np.ndarray,
    *,
    placement_zyx: tuple[int, int, int] | np.ndarray,
    canvas_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    frame = np.asarray(frame_zyx)
    if frame.ndim != 3:
        raise ValueError(f"Expected one (Z,Y,X) frame, got {frame.shape}")

    placement = np.asarray(placement_zyx, dtype=np.int64)
    canvas_shape = np.asarray(canvas_shape_zyx, dtype=np.int64)
    frame_shape = np.asarray(frame.shape, dtype=np.int64)

    if placement.shape != (3,):
        raise ValueError(
            f"placement_zyx must have shape (3,), got {placement.shape}"
        )
    if np.any(placement < 0):
        raise ValueError(
            f"placement_zyx must be nonnegative, got {placement.tolist()}"
        )
    stop = placement + frame_shape
    if np.any(stop > canvas_shape):
        raise ValueError(
            "Frame placement exceeds canvas: "
            f"placement={placement.tolist()} frame={frame_shape.tolist()} "
            f"canvas={canvas_shape.tolist()}"
        )

    output = np.zeros(
        tuple(int(v) for v in canvas_shape.tolist()),
        dtype=frame.dtype,
    )
    z0, y0, x0 = (int(v) for v in placement.tolist())
    z1, y1, x1 = (int(v) for v in stop.tolist())
    output[z0:z1, y0:y1, x0:x1] = frame
    return output


def _as_frame_chunked_dask(movie):
    import dask.array as da

    shape = tuple(int(v) for v in movie.shape)
    if len(shape) != 4:
        raise ValueError(f"Expected movie shape (T,Z,Y,X), got {shape}")

    chunks = (1, shape[1], shape[2], shape[3])
    if isinstance(movie, da.Array):
        return movie.rechunk(chunks)
    return da.from_array(movie, chunks=chunks, asarray=False)


def build_stabilized_movies(
    raw_movie,
    instance_movie,
    estimate: GlobalMotionEstimate,
):
    """Return lazy stabilized Dask movies with one full spatial frame per chunk.

    Full-frame chunks are intentional: Trackastra normalizes Dask input with
    map_blocks, so smaller spatial chunks would normalize subregions
    independently and change the model input distribution.
    """
    import dask.array as da

    raw_shape = tuple(int(v) for v in raw_movie.shape)
    label_shape = tuple(int(v) for v in instance_movie.shape)
    if raw_shape != label_shape:
        raise ValueError(
            f"Raw/instance movie shape mismatch: {raw_shape} vs {label_shape}"
        )
    if len(raw_shape) != 4:
        raise ValueError(f"Expected (T,Z,Y,X), got {raw_shape}")

    frame_count = int(raw_shape[0])
    if estimate.placement_zyx.shape != (frame_count, 3):
        raise ValueError(
            "Motion placement/frame mismatch: "
            f"{estimate.placement_zyx.shape} vs frame_count={frame_count}"
        )

    raw = _as_frame_chunked_dask(raw_movie)
    labels = _as_frame_chunked_dask(instance_movie)

    original_shape = np.asarray(raw_shape[1:], dtype=np.int64)
    canvas_shape = np.asarray(estimate.canvas_shape_zyx, dtype=np.int64)
    raw_frames = []
    label_frames = []

    for t in range(frame_count):
        before = np.asarray(estimate.placement_zyx[t], dtype=np.int64)
        after = canvas_shape - before - original_shape
        if np.any(before < 0) or np.any(after < 0):
            raise RuntimeError(
                f"Invalid stabilized padding at t={t}: "
                f"before={before.tolist()} after={after.tolist()}"
            )
        pad_width = tuple(
            (int(lo), int(hi))
            for lo, hi in zip(before.tolist(), after.tolist())
        )
        raw_frames.append(
            da.pad(raw[t], pad_width, mode="constant", constant_values=0)
        )
        label_frames.append(
            da.pad(labels[t], pad_width, mode="constant", constant_values=0)
        )

    full_chunks = (
        1,
        int(canvas_shape[0]),
        int(canvas_shape[1]),
        int(canvas_shape[2]),
    )
    stabilized_raw = da.stack(raw_frames, axis=0).rechunk(full_chunks)
    stabilized_labels = da.stack(label_frames, axis=0).rechunk(full_chunks)
    return stabilized_raw, stabilized_labels


def restore_graph_coordinates(
    graph,
    estimate: GlobalMotionEstimate,
):
    """Restore pass-2 graph coordinates from padded space to source space."""
    restored = graph.copy()
    frame_count = int(estimate.placement_zyx.shape[0])

    for node_id, attributes in restored.nodes(data=True):
        frame = int(attributes["time"])
        if not (0 <= frame < frame_count):
            raise RuntimeError(
                f"Trackastra graph node {node_id!r} has invalid frame {frame}"
            )
        coords = np.asarray(attributes["coords"], dtype=np.float64)
        if coords.shape != (3,):
            raise RuntimeError(
                f"Trackastra node {node_id!r} has invalid coords {coords.shape}"
            )
        attributes["coords"] = (
            coords
            - estimate.placement_zyx[frame].astype(np.float64)
        )

    return restored


__all__ = [
    "build_stabilized_movies",
    "pad_frame_to_canvas",
    "restore_graph_coordinates",
]
