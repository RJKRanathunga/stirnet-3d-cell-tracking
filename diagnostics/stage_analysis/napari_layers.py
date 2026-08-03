"""Owned TZYX Napari layers for the reusable Stage 3 diagnostic widget."""

from __future__ import annotations

from importlib import import_module

import numpy as np
from skimage.segmentation import find_boundaries

from .models import Stage3ComponentRun, Stage3FrameSelection
from .source import build_display_scope


OWNER = "stage3-analysis"
peaks_module = import_module("src.03_segmentation.peaks")

ANALYSIS_LAYER_NAMES = {
    "EDT | Raw",
    "EDT | Merge tree",
    "EDT | Watershed",
    "Peaks | Raw",
    "Peaks | Effective",
    "Peaks | Selected",
    "Pairs | Evidence",
    "Peak scan | Smoothed EDT",
    "Peak scan | H-maxima",
    "Peak scan | Peaks",
    "Hyp | Labels",
    "Hyp | Boundary",
    "Hyp | Markers",
    "Final | Labels",
    "Final | Boundary",
    "Compare | Prod boundary",
    "Compare | Trial boundary",
}


def crop_origin(crop) -> tuple[int, int, int]:
    return tuple(int(axis.start) for axis in crop)


def crop_shape(crop) -> tuple[int, int, int]:
    return tuple(int(axis.stop - axis.start) for axis in crop)


def embed_component(data, component_bbox, diagnostic_crop, fill=0) -> np.ndarray:
    output = np.full(
        crop_shape(diagnostic_crop), fill, dtype=np.asarray(data).dtype
    )
    destination = tuple(
        slice(box.start - view.start, box.stop - view.start)
        for box, view in zip(component_bbox, diagnostic_crop)
    )
    output[destination] = data
    return output


def padded_peak_to_crop(
    position, component_bbox, diagnostic_crop, padding
) -> tuple[int, int, int]:
    return tuple(
        int(
            position[axis]
            - padding
            + component_bbox[axis].start
            - diagnostic_crop[axis].start
        )
        for axis in range(3)
    )


def current_tzyx(data, time_count: int, time_index: int, fill=0) -> np.ndarray:
    """Place one loaded ZYX frame into its scene-aligned TZYX slot."""

    array = np.asarray(data)
    output = np.full(
        (int(time_count), *array.shape), fill, dtype=array.dtype
    )
    output[int(time_index)] = array
    return output


def _time_points(points_zyx, time_index: int) -> np.ndarray:
    points = np.asarray(points_zyx, dtype=float).reshape((-1, 3))
    if not len(points):
        return np.empty((0, 4), dtype=float)
    return np.column_stack(
        (np.full(len(points), float(time_index)), points)
    )


def _time_lines(lines_zyx, time_index: int) -> list[np.ndarray]:
    return [
        np.column_stack(
            (np.full(len(line), float(time_index)), np.asarray(line, dtype=float))
        )
        for line in lines_zyx
    ]


def capture_camera(viewer) -> dict[str, object]:
    """Capture camera fields that remain stable across a time-frame refresh."""

    state = {}
    for name in ("center", "zoom", "angles", "perspective"):
        if hasattr(viewer.camera, name):
            value = getattr(viewer.camera, name)
            state[name] = tuple(value) if name in {"center", "angles"} else value
    return state


def restore_camera(viewer, state: dict[str, object]) -> None:
    for name, value in state.items():
        try:
            setattr(viewer.camera, name, value)
        except (AttributeError, TypeError, ValueError):
            pass


def remove_owned_layers(viewer, owned_registry=()) -> None:
    """Remove only layers owned by this package or registered by object/name."""

    registered_objects = {
        id(value) for value in owned_registry if not isinstance(value, str)
    }
    registered_names = {
        value for value in owned_registry if isinstance(value, str)
    }
    for layer in list(viewer.layers):
        metadata = getattr(layer, "metadata", {}) or {}
        if (
            metadata.get("owner") == OWNER
            or id(layer) in registered_objects
            or getattr(layer, "name", None) in registered_names
        ):
            viewer.layers.remove(layer)


class Stage3LayerManager:
    """Create short-named, ownership-tagged diagnostic layers."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.registry = []

    def enforce_3d(self) -> None:
        self.viewer.dims.ndisplay = 3

    def clear(self) -> None:
        remove_owned_layers(self.viewer, self.registry)
        self.registry.clear()
        self.enforce_3d()

    def clear_analysis(self) -> None:
        for layer in list(self.viewer.layers):
            if layer.name in ANALYSIS_LAYER_NAMES and (
                layer.metadata.get("owner") == OWNER or layer in self.registry
            ):
                self.viewer.layers.remove(layer)
                if layer in self.registry:
                    self.registry.remove(layer)
        self.enforce_3d()

    def _replace(self, name: str) -> None:
        for layer in list(self.viewer.layers):
            if layer.name == name and layer.metadata.get("owner") == OWNER:
                self.viewer.layers.remove(layer)
                if layer in self.registry:
                    self.registry.remove(layer)

    def add(self, method: str, data, name: str, **kwargs):
        self._replace(name)
        if method == "add_image":
            kwargs.setdefault("rendering", "mip")
        elif method == "add_labels":
            kwargs.setdefault("rendering", "translucent")
        elif method == "add_points":
            kwargs.setdefault("out_of_slice_display", True)
        elif method == "add_shapes":
            kwargs.setdefault("ndim", 4)
        layer = getattr(self.viewer, method)(data, name=name, **kwargs)
        layer.metadata["owner"] = OWNER
        self.registry.append(layer)
        self.enforce_3d()
        return layer


def _common_layer_kwargs(frame: Stage3FrameSelection, voxel_size):
    return {
        "scale": (1.0, *tuple(float(value) for value in voxel_size)),
        "translate": (
            0.0,
            *tuple(
                float(origin * spacing)
                for origin, spacing in zip(
                    crop_origin(frame.display_crop), voxel_size
                )
            ),
        ),
    }


def peak_properties(peaks, run: Stage3ComponentRun) -> dict[str, list]:
    effective = {
        peak.peak_id for peak in run.collapse_result.effective_peaks
    }
    selected = {peak.peak_id for peak in run.decision.chosen.selected_peaks}
    return {
        "peak_id": [peak.peak_id for peak in peaks],
        "raw_depth": [peak.raw_depth_um for peak in peaks],
        "smoothed_depth": [peak.smoothed_depth_um for peak in peaks],
        "scale_support": [peak.scale_support for peak in peaks],
        "h_support": [peak.h_support for peak in peaks],
        "setting_support": [peak.setting_support for peak in peaks],
        "detection_count": [peak.detection_count for peak in peaks],
        "persistence_score": [peak.persistence_score for peak in peaks],
        "retained": [peak.peak_id in effective for peak in peaks],
        "collapsed": [peak.peak_id not in effective for peak in peaks],
        "selected": [peak.peak_id in selected for peak in peaks],
    }


def render_input_layers(
    manager: Stage3LayerManager,
    frame: Stage3FrameSelection,
    component_id: int | None,
    selected_only: bool,
    time_count: int,
    voxel_size,
) -> None:
    """Render cached current-frame input/production data as scene TZYX layers."""

    production_frame = frame.production_frame
    resolution = frame.resolution
    crop = frame.display_crop
    scope = build_display_scope(
        production_frame.instance_labels,
        production_frame.binary_mask,
        resolution,
        selected_only=selected_only,
        crop=crop,
    )
    common = _common_layer_kwargs(frame, voxel_size)
    if component_id is None:
        target = np.zeros(crop_shape(crop), dtype=bool)
    else:
        target = resolution.component_labels[crop] == int(component_id)
    selected_production = np.where(
        np.isin(production_frame.instance_labels[crop], resolution.selected_ids),
        production_frame.instance_labels[crop],
        0,
    )

    def volume(data):
        return current_tzyx(data, time_count, frame.scene_time_index)

    manager.add(
        "add_image",
        volume(production_frame.raw[crop]),
        "Input | Raw",
        blending="additive",
        **common,
    )
    manager.add(
        "add_image",
        volume(production_frame.preprocessed[crop]),
        "Input | Preprocessed",
        blending="additive",
        visible=False,
        **common,
    )
    manager.add(
        "add_labels",
        volume(scope.binary_mask.astype(np.uint8)),
        "Mask | Saved",
        opacity=0.25,
        **common,
    )
    manager.add(
        "add_labels",
        volume(target.astype(np.uint8)),
        "Mask | Target",
        opacity=0.35,
        **common,
    )
    manager.add(
        "add_labels",
        volume(find_boundaries(target, mode="inner").astype(np.uint8)),
        "Boundary | Target",
        **common,
    )
    manager.add(
        "add_labels",
        volume(scope.production_labels.astype(np.int32)),
        "Prod | Labels",
        opacity=0.45,
        **common,
    )
    manager.add(
        "add_labels",
        volume(
            find_boundaries(scope.production_labels, mode="inner").astype(
                np.uint8
            )
        ),
        "Prod | Boundary",
        **common,
    )
    manager.add(
        "add_labels",
        volume(selected_production.astype(np.int32)),
        "Prod | Selected",
        opacity=0.60,
        **common,
    )


def render_analysis_layers(
    manager: Stage3LayerManager,
    run: Stage3ComponentRun,
    frame: Stage3FrameSelection,
    time_count: int,
    voxel_size,
) -> None:
    crop, bbox = frame.display_crop, run.component_bbox
    time_index = frame.scene_time_index
    padding = run.config.component_padding_voxels
    inner = tuple(
        slice(padding, -padding) if padding else slice(None) for _ in range(3)
    )
    common = _common_layer_kwargs(frame, voxel_size)

    for field, name in (
        (run.peak_analysis.raw_distance, "EDT | Raw"),
        (run.peak_analysis.merge_tree_distance, "EDT | Merge tree"),
        (run.peak_analysis.watershed_distance, "EDT | Watershed"),
    ):
        manager.add(
            "add_image",
            current_tzyx(
                embed_component(field[inner], bbox, crop),
                time_count,
                time_index,
            ),
            name,
            visible=False,
            **common,
        )

    def add_peak_layer(name, peak_values, size):
        positions = [
            padded_peak_to_crop(
                peak.position_zyx, bbox, crop, padding
            )
            for peak in peak_values
        ]
        manager.add(
            "add_points",
            _time_points(positions, time_index),
            name,
            size=size,
            properties=peak_properties(peak_values, run),
            text={"string": "{peak_id}", "color": "white"},
            **common,
        )

    add_peak_layer("Peaks | Raw", run.peak_analysis.peaks, 5)
    add_peak_layer("Peaks | Effective", run.collapse_result.effective_peaks, 7)
    add_peak_layer("Peaks | Selected", run.decision.chosen.selected_peaks, 9)

    by_id = {peak.peak_id: peak for peak in run.peak_analysis.peaks}
    property_names = (
        "peak_a",
        "peak_b",
        "separation_um",
        "saddle_um",
        "branch_persistence",
        "branch_balance",
        "separation_support",
        "peak_support",
        "distinct_lobe_probability",
        "same_lobe_probability",
    )
    lines, properties = [], {name: [] for name in property_names}
    for pair in run.pair_evidence:
        lines.append(
            np.asarray(
                [
                    padded_peak_to_crop(
                        by_id[pair.peak_id_a].position_zyx,
                        bbox,
                        crop,
                        padding,
                    ),
                    padded_peak_to_crop(
                        by_id[pair.peak_id_b].position_zyx,
                        bbox,
                        crop,
                        padding,
                    ),
                ],
                dtype=float,
            )
        )
        values = (
            pair.peak_id_a,
            pair.peak_id_b,
            pair.separation_um,
            pair.saddle_um,
            pair.branch_persistence,
            pair.branch_balance,
            pair.separation_support,
            pair.peak_support,
            pair.distinct_lobe_probability,
            pair.same_lobe_probability,
        )
        for key, value in zip(properties, values):
            properties[key].append(value)
    manager.add(
        "add_shapes",
        _time_lines(lines, time_index),
        "Pairs | Evidence",
        shape_type="line",
        properties=properties,
        edge_width=2,
        **common,
    )

    final_labels = embed_component(run.canonical.labels, bbox, crop)
    trial_boundary = find_boundaries(final_labels, mode="inner")
    production_boundary = find_boundaries(
        frame.production_frame.instance_labels[crop], mode="inner"
    )
    comparison = (
        production_boundary.astype(np.uint8)
        + 2 * trial_boundary.astype(np.uint8)
    )
    manager.add(
        "add_labels",
        current_tzyx(final_labels, time_count, time_index),
        "Final | Labels",
        opacity=0.50,
        **common,
    )
    manager.add(
        "add_labels",
        current_tzyx(
            trial_boundary.astype(np.uint8), time_count, time_index
        ),
        "Final | Boundary",
        **common,
    )
    manager.add(
        "add_labels",
        current_tzyx(comparison, time_count, time_index),
        "Compare | Prod boundary",
        opacity=0.85,
        **common,
    )
    manager.add(
        "add_labels",
        current_tzyx(
            trial_boundary.astype(np.uint8), time_count, time_index
        ),
        "Compare | Trial boundary",
        visible=False,
        **common,
    )


def render_peak_setting(
    manager: Stage3LayerManager,
    run: Stage3ComponentRun,
    frame: Stage3FrameSelection,
    time_count: int,
    sigma_um: float,
    h_um: float,
    voxel_size,
) -> None:
    crop, bbox = frame.display_crop, run.component_bbox
    time_index = frame.scene_time_index
    padding = run.config.component_padding_voxels
    inner = tuple(
        slice(padding, -padding) if padding else slice(None) for _ in range(3)
    )
    setting = peaks_module.inspect_peak_detection_setting(
        run.padded_mask, run.config, float(sigma_um), float(h_um)
    )
    common = _common_layer_kwargs(frame, voxel_size)
    manager.add(
        "add_image",
        current_tzyx(
            embed_component(setting.smoothed_distance[inner], bbox, crop),
            time_count,
            time_index,
        ),
        "Peak scan | Smoothed EDT",
        visible=False,
        **common,
    )
    manager.add(
        "add_labels",
        current_tzyx(
            embed_component(
                setting.h_maxima_mask[inner].astype(np.uint8), bbox, crop
            ),
            time_count,
            time_index,
        ),
        "Peak scan | H-maxima",
        **common,
    )
    points = [
        padded_peak_to_crop(position, bbox, crop, padding)
        for position in setting.representative_positions_zyx
    ]
    manager.add(
        "add_points",
        _time_points(points, time_index),
        "Peak scan | Peaks",
        size=6,
        **common,
    )


def render_hypothesis(
    manager: Stage3LayerManager,
    run: Stage3ComponentRun,
    frame: Stage3FrameSelection,
    time_count: int,
    hypothesis_index: int,
    voxel_size,
) -> None:
    crop, bbox = frame.display_crop, run.component_bbox
    time_index = frame.scene_time_index
    padding = run.config.component_padding_voxels
    inner = tuple(
        slice(padding, -padding) if padding else slice(None) for _ in range(3)
    )
    hypothesis = run.evaluation.all_hypotheses[int(hypothesis_index)]
    labels = embed_component(
        hypothesis.labels[inner].astype(np.int32), bbox, crop
    )
    common = _common_layer_kwargs(frame, voxel_size)
    manager.add(
        "add_labels",
        current_tzyx(labels, time_count, time_index),
        "Hyp | Labels",
        opacity=0.50,
        **common,
    )
    manager.add(
        "add_labels",
        current_tzyx(
            find_boundaries(labels, mode="inner").astype(np.uint8),
            time_count,
            time_index,
        ),
        "Hyp | Boundary",
        **common,
    )
    points = [
        padded_peak_to_crop(peak.position_zyx, bbox, crop, padding)
        for peak in hypothesis.selected_peaks
    ]
    manager.add(
        "add_points",
        _time_points(points, time_index),
        "Hyp | Markers",
        size=9,
        properties={
            "peak_id": [peak.peak_id for peak in hypothesis.selected_peaks]
        },
        text={"string": "{peak_id}", "color": "white"},
        **common,
    )


__all__ = [
    "ANALYSIS_LAYER_NAMES",
    "OWNER",
    "Stage3LayerManager",
    "capture_camera",
    "current_tzyx",
    "remove_owned_layers",
    "render_analysis_layers",
    "render_hypothesis",
    "render_input_layers",
    "render_peak_setting",
    "restore_camera",
]
