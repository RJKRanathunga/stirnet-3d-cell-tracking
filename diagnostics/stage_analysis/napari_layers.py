"""Owned TZYX Napari layers for the reusable Stage 3 diagnostic widget."""

from __future__ import annotations

from importlib import import_module

import numpy as np
from skimage.segmentation import find_boundaries

from .models import Stage3ComponentRun, Stage3FrameSelection
from .source import build_display_scope


OWNER = "stage3-analysis"
peaks_module = import_module("src.source_instances.segmentation.peaks")

ANALYSIS_LAYER_NAMES = {
    "EDT | Raw",
    "EDT | Merge tree",
    "EDT | Watershed",
    "Peaks | Raw",
    "Peaks | Effective",
    "Pairs | Evidence",
    "Geometry | Shape center peaks",
    "Geometry | Center proposals",
    "Geometry | Unrepresented proposals",
    "Geometry | Candidate proposals",
    "Geometry | Binary LoG response",
    "Geometry | Binary LoG maxima",
    "Geometry | Boundary samples",
    "Geometry | Surface caps",
    "Geometry | Cap normals",
    "Geometry | Candidate body axes",
    "Geometry | Rejected body axes",
    "Geometry | Selected body axes",
    "Geometry | Effective EDT markers",
    "Geometry | Supplemental markers",
    "Geometry | Final markers",
    "Geometry | Cross-section planes",
    "Geometry | Ellipsoid support",
    "Geometry | Unique support",
    "Peak scan | Smoothed EDT",
    "Peak scan | H-maxima",
    "Peak scan | Peaks",
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
    }


def _geometry_to_crop(position, run, frame):
    return padded_peak_to_crop(
        position,
        run.component_bbox,
        frame.display_crop,
        run.config.component_padding_voxels,
    )


def render_geometry_layers(
    manager: Stage3LayerManager,
    run: Stage3ComponentRun,
    frame: Stage3FrameSelection,
    time_count: int,
    voxel_size,
) -> None:
    """Render retained production geometry without recalculating it."""

    completion = run.geometric_completion
    debug = completion.debug_artifacts
    time_index = frame.scene_time_index
    common = _common_layer_kwargs(frame, voxel_size)
    spacing = np.asarray(voxel_size, dtype=float)

    shape_peaks = run.candidate_result.shape_peaks
    if shape_peaks:
        manager.add(
            "add_points",
            _time_points(
                [_geometry_to_crop(peak.position_zyx, run, frame) for peak in shape_peaks],
                time_index,
            ),
            "Geometry | Shape center peaks",
            size=5,
            face_color="orange",
            properties={
                "shape_peak_id": [peak.peak_id for peak in shape_peaks],
                "best_scale_um": [peak.best_scale_um for peak in shape_peaks],
                "response": [peak.response for peak in shape_peaks],
                "relative_response": [peak.relative_response for peak in shape_peaks],
                "scale_support": [peak.scale_support for peak in shape_peaks],
                "detection_count": [peak.detection_count for peak in shape_peaks],
                "interior_depth_um": [peak.interior_depth_um for peak in shape_peaks],
                "local_depth_ratio": [peak.local_depth_ratio for peak in shape_peaks],
            },
            visible=False,
            **common,
        )

    def proposal_properties(proposals):
        return {
            "proposal_id": [proposal.proposal_id for proposal in proposals],
            "source_types": [
                ";".join(
                    source
                    for source, present in (
                        ("suppressed_edt", bool(proposal.raw_peak_ids)),
                        ("binary_log", bool(proposal.shape_peak_ids)),
                    )
                    if present
                )
                for proposal in proposals
            ],
            "route": [proposal.route or "" for proposal in proposals],
            "represented": [proposal.represented for proposal in proposals],
            "physical_separation_um": [
                proposal.nearest_effective_distance_um for proposal in proposals
            ],
            "normalized_separation": [
                proposal.normalized_effective_separation for proposal in proposals
            ],
            "raw_persistence": [proposal.raw_persistence for proposal in proposals],
            "shape_response": [proposal.shape_relative_response for proposal in proposals],
            "shape_scale_support": [proposal.shape_scale_support for proposal in proposals],
            "shape_local_depth_ratio": [proposal.shape_local_depth_ratio for proposal in proposals],
            "reasons": [";".join(proposal.reasons) for proposal in proposals],
        }

    proposals = list(run.candidate_result.proposals)
    unrepresented = [proposal for proposal in proposals if not proposal.represented]
    candidates = [proposal for proposal in proposals if proposal.candidate]
    for name, values, color, visible in (
        ("Geometry | Center proposals", proposals, "gray", False),
        ("Geometry | Unrepresented proposals", unrepresented, "yellow", False),
        ("Geometry | Candidate proposals", candidates, "magenta", True),
    ):
        if values:
            manager.add(
                "add_points",
                _time_points(
                    [_geometry_to_crop(value.position_zyx, run, frame) for value in values],
                    time_index,
                ),
                name,
                size=6,
                face_color=color,
                properties=proposal_properties(values),
                **common,
                visible=visible,
            )

    candidate_debug = run.candidate_result.debug_artifacts
    if candidate_debug is not None and candidate_debug.response_volumes:
        selected_scale = max(
            range(len(candidate_debug.sigma_levels_um)),
            key=lambda index: (
                float(np.max(candidate_debug.response_volumes[index])),
                -index,
            ),
        )
        padding = run.config.component_padding_voxels
        inner = tuple(
            slice(padding, -padding) if padding else slice(None) for _ in range(3)
        )
        manager.add(
            "add_image",
            current_tzyx(
                embed_component(
                    candidate_debug.response_volumes[selected_scale][inner],
                    run.component_bbox,
                    frame.display_crop,
                ),
                time_count,
                time_index,
            ),
            "Geometry | Binary LoG response",
            visible=False,
            **common,
        )
        maxima = candidate_debug.raw_maxima_zyx[selected_scale]
        if len(maxima):
            manager.add(
                "add_points",
                _time_points(
                    [_geometry_to_crop(value, run, frame) for value in maxima],
                    time_index,
                ),
                "Geometry | Binary LoG maxima",
                size=4,
                face_color="orange",
                visible=False,
                properties={
                    "sigma_um": [
                        candidate_debug.sigma_levels_um[selected_scale]
                    ] * len(maxima)
                },
                **common,
            )

    if debug is not None and len(debug.boundary_positions_zyx):
        boundary = [
            _geometry_to_crop(position, run, frame)
            for position in debug.boundary_positions_zyx
        ]
        manager.add(
            "add_points",
            _time_points(boundary, time_index),
            "Geometry | Boundary samples",
            size=1.2,
            face_color="gray",
            visible=False,
            **common,
        )

    cap_positions = [
        _geometry_to_crop(cap.center_zyx, run, frame)
        for cap in completion.surface_caps
    ]
    if cap_positions:
        cap_properties = {
            "cap_id": [cap.cap_id for cap in completion.surface_caps],
            "area_proxy_um2": [cap.area_proxy_um2 for cap in completion.surface_caps],
            "prominence_um": [cap.prominence_um for cap in completion.surface_caps],
            "normal_coherence": [cap.normal_coherence for cap in completion.surface_caps],
            "curvature_score": [cap.curvature_score for cap in completion.surface_caps],
            "scale_support": [cap.scale_support for cap in completion.surface_caps],
        }
        manager.add(
            "add_points",
            _time_points(cap_positions, time_index),
            "Geometry | Surface caps",
            size=5,
            face_color="gold",
            properties=cap_properties,
            text={"string": "{cap_id}", "color": "white"},
            **common,
        )
        vectors = []
        for position, cap in zip(cap_positions, completion.surface_caps):
            origin = np.asarray((float(time_index), *position), dtype=float)
            direction = np.asarray(
                (0.0, *(2.0 * np.asarray(cap.mean_normal) / spacing)),
                dtype=float,
            )
            vectors.append(np.stack((origin, direction)))
        manager.add(
            "add_vectors",
            np.asarray(vectors),
            "Geometry | Cap normals",
            edge_color="gold",
            properties=cap_properties,
            **common,
        )

    selected_ids = {body.body_id for body in completion.selected_bodies}
    candidates = [body for body in completion.body_candidates if body.valid]
    rejected = [body for body in completion.body_candidates if not body.valid]

    def body_lines(bodies):
        return _time_lines(
            [
                np.asarray(
                    [
                        _geometry_to_crop(body.axis_endpoints_zyx[0], run, frame),
                        _geometry_to_crop(body.axis_endpoints_zyx[1], run, frame),
                    ]
                )
                for body in bodies
            ],
            time_index,
        )

    def body_properties(bodies):
        return {
            "body_id": [body.body_id for body in bodies],
            "body_score": [body.score for body in bodies],
            "valid": [body.valid for body in bodies],
            "selected": [body.body_id in selected_ids for body in bodies],
            "rejection_reasons": [";".join(body.rejection_reasons) for body in bodies],
            "represented_effective_peak_ids": [
                ";".join(str(value) for value in body.represented_by_effective_peak_ids)
                for body in bodies
            ],
        }

    for name, bodies, color, visible in (
        ("Geometry | Candidate body axes", candidates, "cyan", False),
        ("Geometry | Rejected body axes", rejected, "red", False),
        ("Geometry | Selected body axes", list(completion.selected_bodies), "lime", True),
    ):
        if bodies:
            manager.add(
                "add_shapes",
                body_lines(bodies),
                name,
                shape_type="line",
                edge_color=color,
                edge_width=2,
                properties=body_properties(bodies),
                visible=visible,
                **common,
            )

    marker_groups = (
        (
            "Geometry | Effective EDT markers",
            [marker for marker in run.final_markers if marker.source == "effective_edt"],
            "red",
        ),
        (
            "Geometry | Supplemental markers",
            list(completion.supplemental_markers),
            "magenta",
        ),
        ("Geometry | Final markers", list(run.final_markers), "white"),
    )
    for name, markers, color in marker_groups:
        if not markers:
            continue
        manager.add(
            "add_points",
            _time_points(
                [
                    _geometry_to_crop(marker.position_zyx, run, frame)
                    for marker in markers
                ],
                time_index,
            ),
            name,
            size=7,
            face_color=color,
            properties={
                "source": [marker.source for marker in markers],
                "source_reference_id": [marker.source_reference_id for marker in markers],
                "confidence": [marker.confidence for marker in markers],
            },
            **common,
        )

    # Display cross-sections only for the first selected body to avoid an
    # unreadable all-candidate plane cloud. The body's properties identify it.
    if completion.selected_bodies:
        body = completion.selected_bodies[0]
        axis = body.rotation_matrix[:, 0]
        radial_a = body.rotation_matrix[:, 1]
        radial_b = body.rotation_matrix[:, 2]
        center_um = np.asarray(body.center_um)
        polygons = []
        for section in body.cross_sections:
            section_center_um = center_um + section.t_um * axis
            extent_a = max(section.radius_major_um, 0.2)
            extent_b = max(section.radius_minor_um, 0.2)
            corners_um = [
                section_center_um + sign_a * extent_a * radial_a + sign_b * extent_b * radial_b
                for sign_a, sign_b in ((-1, -1), (-1, 1), (1, 1), (1, -1))
            ]
            corners_zyx = np.asarray(corners_um) / spacing
            polygons.append(
                np.asarray(
                    [
                        (float(time_index), *_geometry_to_crop(point, run, frame))
                        for point in corners_zyx
                    ]
                )
            )
        if polygons:
            manager.add(
                "add_shapes",
                polygons,
                "Geometry | Cross-section planes",
                shape_type="polygon",
                edge_color="yellow",
                face_color="transparent",
                properties={"body_id": [body.body_id] * len(polygons)},
                visible=False,
                **common,
            )

    for name, values, color in (
        (
            "Geometry | Ellipsoid support",
            None if debug is None else debug.ellipsoid_support_zyx,
            "blue",
        ),
        (
            "Geometry | Unique support",
            None if debug is None else debug.unique_support_zyx,
            "green",
        ),
    ):
        if values is not None and len(values):
            manager.add(
                "add_points",
                _time_points(
                    [_geometry_to_crop(point, run, frame) for point in values],
                    time_index,
                ),
                name,
                size=1,
                face_color=color,
                visible=False,
                **common,
            )
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

    final_labels = embed_component(run.final_labels, bbox, crop)
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
    render_geometry_layers(
        manager, run, frame, time_count, voxel_size
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


__all__ = [
    "ANALYSIS_LAYER_NAMES",
    "OWNER",
    "Stage3LayerManager",
    "capture_camera",
    "current_tzyx",
    "remove_owned_layers",
    "render_analysis_layers",
    "render_geometry_layers",
    "render_input_layers",
    "render_peak_setting",
    "restore_camera",
]
