from __future__ import annotations

import numpy as np

from ..core.trace import DebugTrace


_QUERY_COLORS = {
    0: "deepskyblue",   # primary
    1: "orange",        # split
    2: "yellow",        # temporal
    3: "magenta",       # discovery
}


def _spacing(trace: DebugTrace):
    return tuple(float(v) for v in trace.metadata["spacing_um"])


def _query_row(trace: DebugTrace, query_index: int):
    for row in trace.tables.get("queries", []):
        if int(row.get("query", -1)) == int(query_index):
            return row
    return None


def _mask_row(trace: DebugTrace, query_index: int):
    for row in trace.tables.get("masks", []):
        if int(row.get("query", -1)) == int(query_index):
            return row
    return None


def _add_query_points(viewer, trace: DebugTrace):
    spacing = _spacing(trace)
    final = trace.arrays.get("queries/final_points")
    initial = trace.arrays.get("queries/initial_points")

    if initial is not None:
        for type_id, name in ((0, "primary"), (1, "split"), (2, "temporal"), (3, "discovery")):
            subset = np.asarray(initial)[np.asarray(initial)[:, 4].astype(int) == type_id]
            if len(subset):
                viewer.add_points(
                    subset[:, :3],
                    name=f"Initial {name} references",
                    scale=spacing,
                    size=3,
                    face_color=_QUERY_COLORS[type_id],
                    opacity=0.55,
                    visible=False,
                )

    if final is not None:
        final = np.asarray(final)
        for type_id, name in ((0, "primary"), (1, "split"), (2, "temporal"), (3, "discovery")):
            subset = final[final[:, 4].astype(int) == type_id]
            if len(subset):
                viewer.add_points(
                    subset[:, :3],
                    name=f"Final {name} centers",
                    scale=spacing,
                    size=5,
                    face_color=_QUERY_COLORS[type_id],
                    opacity=0.9,
                    visible=(name == "temporal"),
                )


def add_query_mask_layers(viewer, trace: DebugTrace, query_index: int):
    row = _mask_row(trace, query_index)
    if row is None:
        raise KeyError(f"No native-mask decomposition stored for query {query_index}")

    spacing = np.asarray(_spacing(trace), dtype=np.float32)
    origin = np.asarray(
        [row["crop_z0"], row["crop_y0"], row["crop_x0"]],
        dtype=np.float32,
    )
    translate = tuple((origin * spacing).tolist())
    prefix = f"masks/q{int(query_index):03d}"

    for key, colormap, visible in (
        ("learned_prob", "green", True),
        ("prior_prob", "magenta", False),
        ("combined_prob", "cyan", False),
    ):
        array = trace.arrays.get(f"{prefix}/{key}")
        if array is not None:
            viewer.add_image(
                array,
                name=f"Q{query_index} {key}",
                scale=tuple(spacing),
                translate=translate,
                colormap=colormap,
                opacity=0.65,
                contrast_limits=(0.0, 1.0),
                visible=visible,
            )

    gt = trace.arrays.get(f"{prefix}/gt_mask")
    if gt is not None:
        viewer.add_labels(
            gt.astype(np.uint8),
            name=f"Q{query_index} matched GT crop",
            scale=tuple(spacing),
            translate=translate,
            opacity=0.45,
            visible=False,
        )


def open_debug_viewer(trace: DebugTrace, *, query_index: int | None = None, ndisplay: int = 3):
    try:
        import napari
    except ImportError as exc:
        raise ImportError(
            "Napari is optional. Install napari in the active environment to use the 3D debug viewer."
        ) from exc

    spacing = _spacing(trace)
    viewer = napari.Viewer(ndisplay=ndisplay, title="STIR-Net Debug Trace")

    raw = trace.arrays.get("scene/raw")
    if raw is not None:
        viewer.add_image(raw, name="Raw", scale=spacing, colormap="gray", rendering="mip", opacity=0.65)

    current = trace.arrays.get("scene/current_labels")
    if current is not None:
        viewer.add_labels(current, name="Current instances", scale=spacing, opacity=0.45, visible=False)

    gt = trace.arrays.get("scene/gt_labels")
    if gt is not None:
        viewer.add_labels(gt, name="Ground truth", scale=spacing, opacity=0.55, visible=False)

    for key, title, cmap in (
        ("dense/foreground", "CNN foreground probability", "green"),
        ("dense/center_heatmap", "CNN center heatmap", "yellow"),
        ("dense/boundary", "CNN boundary probability", "magenta"),
    ):
        arr = trace.arrays.get(key)
        if arr is not None:
            viewer.add_image(
                arr,
                name=title,
                scale=spacing,
                colormap=cmap,
                opacity=0.55,
                contrast_limits=(0.0, 1.0),
                visible=False,
            )

    _add_query_points(viewer, trace)

    if query_index is not None:
        add_query_mask_layers(viewer, trace, int(query_index))

    return viewer
