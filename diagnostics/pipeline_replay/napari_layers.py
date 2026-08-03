"""Napari layer ownership and rendering for replay results."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.segmentation import find_boundaries

from .comparison import diagnose_instance_centers


OWNED_PREFIXES = ("Production | ", "Trial | ", "Debug | ", "Difference | ")


def remove_pipeline_replay_layers(viewer: Any) -> None:
    """Remove only layers owned by this workbench."""

    for layer in list(viewer.layers):
        if any(str(layer.name).startswith(prefix) for prefix in OWNED_PREFIXES):
            viewer.layers.remove(layer)


def _replace(viewer: Any, kind: str, data, *, name: str, **kwargs):
    for layer in list(viewer.layers):
        if layer.name == name:
            viewer.layers.remove(layer)
    return getattr(viewer, f"add_{kind}")(data, name=name, **kwargs)


def _transform(source):
    scale = tuple(float(value) for value in source.voxel_size_zyx_um)
    translate = tuple(
        float(origin) * spacing
        for origin, spacing in zip(source.crop_origin_zyx, scale)
    )
    return scale, translate


def add_production_layers(viewer: Any, source, baseline, selected_cell_id: int):
    scale, translate = _transform(source)
    crop = source.crop_slices
    raw = baseline.raw[crop]
    processed = baseline.preprocessed[crop]
    mask = baseline.binary_mask[crop]
    labels = baseline.instance_labels[crop]
    selected = (labels == int(selected_cell_id)).astype(np.uint8)
    points = baseline.cells[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
    if len(points):
        points -= np.asarray(source.crop_origin_zyx, dtype=float)
    layers = [
        _replace(viewer, "image", raw, name="Production | Raw", scale=scale, translate=translate),
        _replace(viewer, "image", processed, name="Production | Preprocessed", scale=scale, translate=translate, visible=False),
        _replace(viewer, "labels", mask.astype(np.uint8), name="Production | Binary mask", scale=scale, translate=translate, visible=False),
        _replace(viewer, "labels", labels, name="Production | Instance labels", scale=scale, translate=translate),
        _replace(viewer, "points", points, name="Production | Centroids", scale=scale, translate=translate, size=2.5, face_color="cyan"),
        _replace(viewer, "labels", selected, name="Production | Selected cell", scale=scale, translate=translate),
    ]
    return layers


def add_trial_layers(viewer: Any, source, runner, result):
    scale, translate = _transform(source)
    trace = runner.state.preprocessing_trace
    normalized = denoised = background = None
    if trace is not None and runner._display_slices is not None:
        normalized = trace.intermediates["normalized"][runner._display_slices]
        denoised = trace.intermediates["denoised"][runner._display_slices]
        background = trace.intermediates["background"][runner._display_slices]
    arrays = (
        ("image", normalized, "Trial | Normalized", False),
        ("image", denoised, "Trial | Denoised", False),
        ("image", background, "Trial | Estimated background", False),
        ("image", result.display_preprocessed, "Trial | Preprocessed", True),
        ("labels", result.display_binary_mask, "Trial | Binary mask", False),
        ("labels", result.display_connected_components, "Trial | Connected components", False),
        ("labels", result.display_instance_labels, "Trial | Instance labels", True),
    )
    layers = []
    for kind, data, name, visible in arrays:
        if data is not None:
            layers.append(_replace(viewer, kind, data, name=name, scale=scale, translate=translate, visible=visible))
    if result.display_markers is not None:
        marker_points = np.argwhere(result.display_markers > 0)
        layers.append(_replace(
            viewer, "points", marker_points, name="Trial | Segmentation markers",
            scale=scale, translate=translate, size=2.2, face_color="magenta",
            properties={"instance_id": result.display_markers[result.display_markers > 0]},
        ))
    if result.cells is not None:
        centroids = result.cells[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
        centroids -= np.asarray(source.crop_origin_zyx, dtype=float)
        layers.append(_replace(
            viewer, "points", centroids, name="Trial | Geometric centroids",
            scale=scale, translate=translate, size=2.2, face_color="yellow",
            properties={"instance_id": result.cells["cell_id"].to_numpy()},
        ))
    if result.display_instance_labels is not None:
        centers = diagnose_instance_centers(runner._labels_work)
        proxies = np.asarray([item.in_body_center_zyx for item in centers], dtype=float).reshape((-1, 3))
        proxies += np.asarray(result.work_origin_zyx, dtype=float)
        proxies -= np.asarray(source.crop_origin_zyx, dtype=float)
        layers.append(_replace(
            viewer, "points", proxies, name="Trial | In-body center proxies",
            scale=scale, translate=translate, size=2.2, face_color="lime",
            properties={
                "instance_id": np.asarray([item.instance_id for item in centers]),
                "centroid_inside": np.asarray([item.centroid_inside_mask for item in centers]),
                "connected_parts": np.asarray([item.connected_part_count for item in centers]),
                "proxy_distance_voxels": np.asarray([item.in_body_distance_voxels for item in centers]),
            },
        ))
    return layers


def add_debug_layers(viewer: Any, source, result, component_index: int = 0):
    if not result.component_results:
        return []
    component = result.component_results[component_index]
    scale, scene_translate = _transform(source)
    padding = tuple(
        (array_size - mask_size) // 2
        for array_size, mask_size in zip(
            component.raw_distance.shape, component.component_mask.shape
        )
    )
    component_origin = tuple(bounds[0] - pad + work for bounds, pad, work in zip(component.bbox_zyx, padding, result.work_origin_zyx))
    debug_translate = tuple(float(value) * spacing for value, spacing in zip(component_origin, scale))
    layers = []
    for data, name in (
        (component.raw_distance, "Debug | Raw EDT"),
        (component.merge_tree_distance, "Debug | Merge-tree EDT"),
        (component.watershed_distance, "Debug | Watershed EDT"),
    ):
        layers.append(_replace(viewer, "image", data, name=name, scale=scale, translate=debug_translate, visible=False))
    bbox_start_global = np.asarray([bounds[0] for bounds in component.bbox_zyx]) + np.asarray(result.work_origin_zyx)
    def local_points(values):
        return np.asarray(values, dtype=float) + bbox_start_global - np.asarray(source.crop_origin_zyx)
    raw_properties = {
        column: component.raw_peak_properties[column].to_numpy()
        for column in component.raw_peak_properties.columns
    }
    layers.append(_replace(viewer, "points", local_points(component.raw_peak_positions_zyx), name="Debug | Raw peaks", scale=scale, translate=scene_translate, size=2, face_color="orange", properties=raw_properties))
    effective_properties = {
        column: component.effective_peak_properties[column].to_numpy()
        for column in component.effective_peak_properties.columns
    }
    layers.append(_replace(viewer, "points", local_points(component.effective_peak_positions_zyx), name="Debug | Effective peaks (final markers)", scale=scale, translate=scene_translate, size=2.5, face_color="red", properties=effective_properties))
    peak_by_id = {
        int(peak_id): point
        for peak_id, point in zip(
            component.raw_peak_properties.get("peak_id", []),
            local_points(component.raw_peak_positions_zyx),
        )
    }
    lines, properties = [], {}
    if not component.pair_evidence.empty:
        valid_rows = []
        for row in component.pair_evidence.to_dict("records"):
            if row["peak_a"] in peak_by_id and row["peak_b"] in peak_by_id:
                lines.append(np.asarray([peak_by_id[row["peak_a"]], peak_by_id[row["peak_b"]]]))
                valid_rows.append(row)
        properties = {column: np.asarray([row[column] for row in valid_rows]) for column in component.pair_evidence.columns}
    if lines:
        layers.append(_replace(viewer, "shapes", lines, shape_type="line", name="Debug | Peak pair evidence", scale=scale, translate=scene_translate, edge_width=1, properties=properties))
    final_origin = tuple(
        (bounds[0] + work) * spacing
        for bounds, work, spacing in zip(
            component.bbox_zyx, result.work_origin_zyx, scale
        )
    )
    layers.append(
        _replace(
            viewer,
            "labels",
            component.final_labels,
            name="Debug | Final labels",
            scale=scale,
            translate=final_origin,
            visible=False,
        )
    )
    layers.append(
        _replace(
            viewer,
            "labels",
            find_boundaries(component.final_labels, mode="inner").astype(np.uint8),
            name="Debug | Final boundaries",
            scale=scale,
            translate=final_origin,
            visible=False,
        )
    )
    return layers


def add_difference_layers(viewer: Any, source, baseline, result, selected_cell_id: int, matched_trial_id: int | None):
    if result.display_binary_mask is None or result.display_instance_labels is None:
        return []
    scale, translate = _transform(source)
    crop = source.crop_slices
    production_mask = baseline.binary_mask[crop].astype(bool)
    trial_mask = result.display_binary_mask.astype(bool)
    production_labels = baseline.instance_labels[crop]
    trial_labels = result.display_instance_labels
    values = (
        (trial_mask & ~production_mask, "Difference | Mask added", "green"),
        (production_mask & ~trial_mask, "Difference | Mask removed", "red"),
        (find_boundaries(production_labels), "Difference | Production boundaries", "cyan"),
        (find_boundaries(trial_labels), "Difference | Trial boundaries", "yellow"),
        (production_labels == selected_cell_id, "Difference | Production selected cell", "cyan"),
        (trial_labels == matched_trial_id if matched_trial_id is not None else np.zeros_like(trial_labels, dtype=bool), "Difference | Matched trial cell", "magenta"),
    )
    return [
        _replace(viewer, "image", data.astype(np.uint8), name=name, scale=scale, translate=translate, colormap=color, blending="additive", visible=False)
        for data, name, color in values
    ]


def add_tracking_context_layers(viewer: Any, source, context, frame: int):
    """Add read-only overlays from saved Stage 7 and Stage 8 outputs."""

    scale, translate = _transform(source)
    origin = np.asarray(source.crop_origin_zyx, dtype=float)
    layers = []
    stage7_tracks = context.stage7.get("tracks")
    if stage7_tracks is not None and not stage7_tracks.empty and {"z", "y", "x"}.issubset(stage7_tracks.columns):
        for track_id, group in stage7_tracks.sort_values("frame").groupby("track_id"):
            points = group[["z", "y", "x"]].to_numpy(dtype=float) - origin
            if len(points) >= 2:
                layers.append(_replace(
                    viewer, "shapes", [points], shape_type="path",
                    name=f"Production | Stage 7 track path {int(track_id)}",
                    scale=scale, translate=translate, edge_color="cyan", edge_width=1.5,
                    properties={"track_id": np.asarray([int(track_id)])},
                ))
    stage8_tracks = context.stage8.get("tracks")
    if stage8_tracks is not None and not stage8_tracks.empty and {"z", "y", "x"}.issubset(stage8_tracks.columns):
        current = stage8_tracks[stage8_tracks["frame"] == frame] if "frame" in stage8_tracks else stage8_tracks
        virtual = (
            current["is_virtual_merge"].fillna(False).astype(bool)
            if "is_virtual_merge" in current else np.zeros(len(current), dtype=bool)
        )
        for select, name, color in (
            (~virtual, "Production | Stage 8 centers", "blue"),
            (virtual, "Production | Stage 8 virtual merge centers", "magenta"),
        ):
            table = current.loc[select]
            if table.empty:
                continue
            points = table[["z", "y", "x"]].to_numpy(dtype=float) - origin
            property_names = (
                "track_id", "cell_id", "is_virtual_merge", "source_merged_cell_id",
                "merge_event_id", "merge_role",
            )
            properties = {
                column: table[column].to_numpy()
                for column in property_names if column in table.columns
            }
            layers.append(_replace(
                viewer, "points", points, name=name, scale=scale, translate=translate,
                size=2.5, face_color=color, properties=properties,
            ))
    trajectories = context.stage8.get("merge_center_trajectories")
    if trajectories is not None and not trajectories.empty and {"z", "y", "x"}.issubset(trajectories.columns):
        group_column = "merge_event_id" if "merge_event_id" in trajectories else None
        groups = trajectories.groupby(group_column) if group_column else [(0, trajectories)]
        paths, event_ids = [], []
        for event_id, group in groups:
            points = group.sort_values("frame")[["z", "y", "x"]].to_numpy(dtype=float) - origin
            if len(points) >= 2:
                paths.append(points); event_ids.append(event_id)
        if paths:
            layers.append(_replace(
                viewer, "shapes", paths, shape_type="path",
                name="Production | Stage 8 merge-center trajectories",
                scale=scale, translate=translate, edge_color="magenta", edge_width=1.5,
                properties={"merge_event_id": np.asarray(event_ids)},
            ))
    return layers
