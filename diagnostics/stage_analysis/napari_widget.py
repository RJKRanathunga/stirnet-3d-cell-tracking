"""Reusable Qt/Napari widget for one-frame-at-a-time Stage 3 diagnostics."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from pathlib import Path
import traceback

import numpy as np
from IPython.display import display
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.io import PipelinePaths

from .napari_layers import (
    Stage3LayerManager,
    capture_camera,
    render_analysis_layers,
    render_input_layers,
    render_peak_setting,
    restore_camera,
)
from .runner import (
    body_candidates_dataframe,
    candidate_summary_dataframe,
    center_proposals_dataframe,
    cross_sections_dataframe,
    marker_completion_dataframe,
    pair_evidence_dataframe,
    peak_detections_dataframe,
    raw_peaks_dataframe,
    run_stage3_component,
    shape_peaks_dataframe,
    surface_caps_dataframe,
)
from .source import Stage3AnalysisSource, discover_categories, discover_scenes


DEFAULT_SEGMENTATION_CONFIG = import_module(
    "src.03_segmentation.config"
).DEFAULT_SEGMENTATION_CONFIG

PARAMETER_SPECS = (
    ("watershed_sigma_um", float),
    ("merge_tree_sigma_um", float),
    ("peak_cluster_radius_um", float),
    ("sigma_levels_um", tuple),
    ("h_levels_um", tuple),
    ("same_lobe_collapse_probability", float),
    ("pair_prior_distinct", float),
    ("pair_likelihood_temperature", float),
)


class Stage3AnalysisWidget(QWidget):
    """Notebook-independent Stage 3 controls driven by Napari's T slider."""

    def __init__(
        self,
        viewer,
        pipeline_paths: PipelinePaths,
        *,
        scenes_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.viewer = viewer
        self.paths = pipeline_paths
        self.scenes_root = Path(scenes_root or pipeline_paths.tracking_scenes)
        self.layers = Stage3LayerManager(viewer)
        self.source: Stage3AnalysisSource | None = None
        self.frame_selection = None
        self.run_result = None
        self.parameter_widgets = {}
        self._current_time_index: int | None = None
        self._handling_dimension_event = False
        self._build_ui()
        self.restore_defaults()
        self.viewer.dims.events.current_step.connect(
            self._on_dimension_step_changed
        )
        self.refresh_categories()
        self._enforce_3d()

    def _enforce_3d(self) -> None:
        self.viewer.dims.ndisplay = 3

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()
        self.category_combo = QComboBox()
        self.scene_combo = QComboBox()
        self.original_frame_label = QLabel("No scene loaded")
        self.selected_ids_label = QLabel("No scene loaded")
        self.selected_ids_label.setWordWrap(True)
        self.selected_only = QCheckBox("Show selected cells only")
        self.selected_only.setChecked(True)
        self.component_combo = QComboBox()
        self.sigma_combo = QComboBox()
        self.h_combo = QComboBox()
        form.addRow("Scene category", self.category_combo)
        form.addRow("Scene", self.scene_combo)
        form.addRow("Original frame", self.original_frame_label)
        form.addRow("Selected production IDs", self.selected_ids_label)
        form.addRow("", self.selected_only)
        form.addRow("Target binary component", self.component_combo)
        form.addRow("Peak sigma", self.sigma_combo)
        form.addRow("Peak H-level", self.h_combo)
        root.addLayout(form)

        parameter_box = QGroupBox(
            "Temporary immutable SegmentationConfig changes"
        )
        parameter_form = QFormLayout(parameter_box)
        for name, kind in PARAMETER_SPECS:
            if kind is tuple:
                widget = QLineEdit()
            elif kind is int:
                widget = QSpinBox()
                widget.setRange(1, 1_000_000)
            else:
                widget = QDoubleSpinBox()
                widget.setDecimals(6)
                widget.setRange(0.0, 1_000_000.0)
                widget.setSingleStep(0.01)
            self.parameter_widgets[name] = widget
            parameter_form.addRow(name, widget)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(parameter_box)
        scroll.setMinimumHeight(330)
        root.addWidget(scroll)

        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run analysis")
        self.restore_button = QPushButton("Restore defaults")
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.restore_button)
        root.addLayout(buttons)
        self.result_summary = QLabel("Not run")
        self.error_summary = QLabel("")
        for label in (
            self.result_summary,
            self.error_summary,
        ):
            label.setWordWrap(True)
        self.error_summary.setStyleSheet(
            "color: #ff6666; font-weight: bold;"
        )
        root.addWidget(self.result_summary)
        root.addWidget(self.error_summary)

        self.category_combo.currentTextChanged.connect(self.refresh_scenes)
        self.scene_combo.currentTextChanged.connect(self.load_scene)
        self.component_combo.currentIndexChanged.connect(
            self.component_changed
        )
        self.selected_only.toggled.connect(self.refresh_input_layers)
        self.sigma_combo.currentIndexChanged.connect(
            self.peak_setting_changed
        )
        self.h_combo.currentIndexChanged.connect(self.peak_setting_changed)
        self.run_button.clicked.connect(self.run_analysis)
        self.restore_button.clicked.connect(self.restore_defaults)

    def set_error(self, message: str) -> None:
        self.error_summary.setText(str(message))

    def _with_preserved_view(self, callback) -> None:
        camera = capture_camera(self.viewer)
        previous_guard = self._handling_dimension_event
        self._handling_dimension_event = True
        try:
            callback()
            if self._current_time_index is not None and self.viewer.dims.ndim >= 4:
                self.viewer.dims.set_current_step(
                    0, self._current_time_index
                )
            self._enforce_3d()
            restore_camera(self.viewer, camera)
        finally:
            self._handling_dimension_event = previous_guard

    def refresh_categories(self) -> None:
        self.category_combo.blockSignals(True)
        self.category_combo.clear()
        self.category_combo.addItems(discover_categories(self.scenes_root))
        self.category_combo.blockSignals(False)
        self.refresh_scenes(self.category_combo.currentText())
        self._enforce_3d()

    def refresh_scenes(self, category: str) -> None:
        self.scene_combo.blockSignals(True)
        self.scene_combo.clear()
        self.scene_combo.addItems(
            discover_scenes(self.scenes_root, category) if category else []
        )
        self.scene_combo.blockSignals(False)
        self.load_scene(self.scene_combo.currentText())
        self._enforce_3d()

    def load_scene(self, scene_name: str) -> None:
        self._handling_dimension_event = True
        try:
            self.layers.clear()
            self.source = None
            self.frame_selection = None
            self.run_result = None
            self._current_time_index = None
            self.component_combo.clear()
            self.sigma_combo.clear()
            self.h_combo.clear()
            category = self.category_combo.currentText().strip()
            if not category or not scene_name:
                self.run_button.setEnabled(False)
                self.original_frame_label.setText("No scene loaded")
                self.set_error("Select a valid scene category and scene.")
                return
            self.source = Stage3AnalysisSource.from_scene(
                self.scenes_root / category / scene_name,
                self.paths,
            )
        except Exception as error:
            self.run_button.setEnabled(False)
            self.set_error(
                "Invalid scene metadata, missing sample Zarr, or source error: "
                f"{type(error).__name__}: {error}"
            )
            return
        finally:
            self._handling_dimension_event = False
            self._enforce_3d()
        self._load_time_index(0, preserve_camera=False)

    def _on_dimension_step_changed(self, _event=None) -> None:
        self._enforce_3d()
        if self._handling_dimension_event or self.source is None:
            return
        steps = tuple(int(round(value)) for value in self.viewer.dims.current_step)
        if len(steps) < 4:
            return
        time_index = steps[0]
        if time_index == self._current_time_index:
            return
        if 0 <= time_index < self.source.time_count:
            self._load_time_index(time_index, preserve_camera=True)

    def _load_time_index(
        self, scene_time_index: int, *, preserve_camera: bool
    ) -> None:
        if self.source is None:
            return
        camera = capture_camera(self.viewer) if preserve_camera else {}
        self._handling_dimension_event = True
        try:
            self.layers.clear()
            self.frame_selection = self.source.load_time_index(
                scene_time_index
            )
            self._current_time_index = int(scene_time_index)
            self.run_result = None
            self.result_summary.setText("Not run for current frame")
            self.sigma_combo.clear()
            self.h_combo.clear()
            self.original_frame_label.setText(
                str(self.frame_selection.original_frame_number)
            )
            selected_ids = self.frame_selection.selected_ids
            self.selected_ids_label.setText(
                ", ".join(map(str, selected_ids))
                if selected_ids
                else "No target cells saved for this frame"
            )
            self._populate_components_and_errors()
            self._render_inputs()
            if self.viewer.dims.ndim >= 4:
                self.viewer.dims.set_current_step(0, scene_time_index)
            self._enforce_3d()
            if preserve_camera:
                restore_camera(self.viewer, camera)
        except Exception as error:
            self.frame_selection = None
            self.run_result = None
            self.run_button.setEnabled(False)
            self.set_error(
                "Missing Stage 6 files, invalid frame, or resolution error: "
                f"{type(error).__name__}: {error}"
            )
        finally:
            self._handling_dimension_event = False
            self._enforce_3d()

    def _populate_components_and_errors(self) -> None:
        frame = self.frame_selection
        resolution = frame.resolution
        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        for component_id in resolution.target_component_ids:
            selected_ids = resolution.component_selected_ids[component_id]
            self.component_combo.addItem(
                f"Component {component_id} — IDs {selected_ids}", component_id
            )
        self.component_combo.blockSignals(False)

        problems = []
        if not frame.selected_ids:
            problems.append(
                "No target cells were saved for this frame. Stage 3 execution is disabled."
            )
        if resolution.missing_production_ids:
            problems.append(
                "selected IDs absent from production labels: "
                f"{resolution.missing_production_ids}"
            )
        if resolution.selected_ids_without_stage2_foreground:
            problems.append(
                "selected production labels overlap no Stage 2 foreground: "
                f"{resolution.selected_ids_without_stage2_foreground}"
            )
        if frame.selected_ids and not resolution.target_component_ids:
            problems.append("No non-empty target Stage 2 component was resolved.")
        if len(resolution.target_component_ids) > 1:
            problems.append(
                f"Selected IDs map to {len(resolution.target_component_ids)} "
                "components; choose one above."
            )
        fatal = bool(
            not frame.selected_ids
            or resolution.missing_production_ids
            or resolution.selected_ids_without_stage2_foreground
            or not resolution.target_component_ids
        )
        self.run_button.setEnabled(not fatal)
        self.set_error("; ".join(problems))

    def _selected_component_id(self) -> int | None:
        if self.component_combo.currentIndex() < 0:
            return None
        return int(self.component_combo.currentData())

    def _render_inputs(self) -> None:
        if self.frame_selection is None or self.source is None:
            return
        render_input_layers(
            self.layers,
            self.frame_selection,
            self._selected_component_id(),
            self.selected_only.isChecked(),
            self.source.time_count,
            self.source.voxel_size_zyx_um,
        )

    def component_changed(self, *_args) -> None:
        self.run_result = None
        self.layers.clear_analysis()
        self.result_summary.setText("Not run for selected component")
        self.refresh_input_layers()
        self._enforce_3d()

    def refresh_input_layers(self, *_args) -> None:
        if self.frame_selection is None:
            return
        self._with_preserved_view(self._render_inputs)
        self._enforce_3d()

    def restore_defaults(self) -> None:
        for name, kind in PARAMETER_SPECS:
            value = getattr(DEFAULT_SEGMENTATION_CONFIG, name)
            widget = self.parameter_widgets[name]
            if kind is tuple:
                widget.setText(", ".join(map(str, value)))
            else:
                widget.setValue(value)
        self.result_summary.setText(
            "Defaults restored; press Run analysis to apply."
        )
        self._enforce_3d()

    def configured_state(self):
        changes = {}
        for name, kind in PARAMETER_SPECS:
            widget = self.parameter_widgets[name]
            if kind is tuple:
                values = tuple(
                    float(value.strip())
                    for value in widget.text().split(",")
                    if value.strip()
                )
                if not values:
                    raise ValueError(
                        f"{name} must contain at least one comma-separated value"
                    )
                changes[name] = values
            elif kind is int:
                changes[name] = int(widget.value())
            else:
                changes[name] = float(widget.value())
        return replace(DEFAULT_SEGMENTATION_CONFIG, **changes)

    def run_analysis(self) -> None:
        if (
            self.frame_selection is None
            or self.source is None
            or self._selected_component_id() is None
        ):
            self.set_error("Load a frame with a valid target component first.")
            return
        self.set_error("")
        self.result_summary.setText("Running Stage 3…")
        try:
            config = self.configured_state()
            self.run_result = run_stage3_component(
                self.frame_selection.resolution,
                self._selected_component_id(),
                config,
            )
            self._with_preserved_view(
                lambda: render_analysis_layers(
                    self.layers,
                    self.run_result,
                    self.frame_selection,
                    self.source.time_count,
                    config.voxel_size_zyx_um,
                )
            )
            self._populate_analysis_pickers(config)
            self.peak_setting_changed()
            effective_count = len(
                self.run_result.collapse_result.effective_peaks
            )
            instance_count = int(np.max(self.run_result.final_labels))
            candidate = self.run_result.candidate_result
            geometry = self.run_result.geometric_completion
            self.result_summary.setText(
                f"processed; effective peaks={effective_count}; "
                f"shape peaks={len(candidate.shape_peaks)}; "
                f"candidate proposals={len(candidate.candidate_proposal_ids)}; "
                f"geometry={geometry.processing_status}; "
                f"final instances={instance_count}; canonical wrapper verified"
            )
            self.display_analysis_tables()
        except Exception as error:
            self.run_result = None
            self.layers.clear_analysis()
            message = f"{type(error).__name__}: {error}"
            self.result_summary.setText(
                "Stage 3 analysis failed; see the error details."
            )
            self.set_error(message + "\n" + traceback.format_exc())
        self._enforce_3d()

    def _populate_analysis_pickers(self, config) -> None:
        for combo, values in (
            (self.sigma_combo, config.sigma_levels_um),
            (self.h_combo, config.h_levels_um),
        ):
            combo.blockSignals(True)
            combo.clear()
            for value in values:
                combo.addItem(str(value), float(value))
            combo.blockSignals(False)

    def peak_setting_changed(self, *_args) -> None:
        if (
            self.run_result is None
            or self.frame_selection is None
            or self.source is None
            or self.sigma_combo.currentIndex() < 0
            or self.h_combo.currentIndex() < 0
        ):
            return
        self._with_preserved_view(
            lambda: render_peak_setting(
                self.layers,
                self.run_result,
                self.frame_selection,
                self.source.time_count,
                self.sigma_combo.currentData(),
                self.h_combo.currentData(),
                self.run_result.config.voxel_size_zyx_um,
            )
        )
        self._enforce_3d()

    def display_analysis_tables(self) -> None:
        print("Raw peaks")
        display(raw_peaks_dataframe(self.run_result))
        print("Per-setting peak detections and final cluster assignments")
        display(peak_detections_dataframe(self.run_result))
        print("Pair evidence")
        display(pair_evidence_dataframe(self.run_result))
        print("Binary-LoG shape peaks")
        display(shape_peaks_dataframe(self.run_result))
        print("Center proposals")
        display(center_proposals_dataframe(self.run_result))
        print("Candidate summary")
        display(candidate_summary_dataframe(self.run_result))
        print("Surface caps")
        display(surface_caps_dataframe(self.run_result))
        print("Geometric body candidates")
        display(body_candidates_dataframe(self.run_result))
        print("Cross-sections")
        display(cross_sections_dataframe(self.run_result))
        print("Marker completion")
        display(marker_completion_dataframe(self.run_result))


def add_stage3_analysis_widget(
    viewer,
    *,
    paths: PipelinePaths | None = None,
    scenes_root: str | Path | None = None,
) -> Stage3AnalysisWidget:
    """Install the reusable Stage 3 analysis widget and return it."""

    pipeline_paths = paths or PipelinePaths.discover()
    widget = Stage3AnalysisWidget(
        viewer,
        pipeline_paths,
        scenes_root=scenes_root,
    )
    viewer.window.add_dock_widget(
        widget, name="Stage 3 analysis", area="right"
    )
    viewer.dims.ndisplay = 3
    return widget


__all__ = ["Stage3AnalysisWidget", "add_stage3_analysis_widget"]
