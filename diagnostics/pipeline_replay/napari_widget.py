"""Reusable Qt dock widget for interactive pipeline replay."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path
from typing import Any

from .comparison import compare_feature_rows, match_instance_by_iou
from .models import ReplayMode
from .napari_layers import (
    add_debug_layers,
    add_difference_layers,
    add_production_layers,
    add_tracking_context_layers,
    add_trial_layers,
    remove_pipeline_replay_layers,
)
from .runner import PipelineReplayRunner
from .source import PipelineReplaySource
from src.io import PipelinePaths

try:
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import (
        QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
        QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSpinBox,
        QTabWidget, QVBoxLayout, QWidget,
    )
    QT_AVAILABLE = True
except ImportError:
    QWidget = object
    QT_AVAILABLE = False


MAIN_SEGMENTATION_FIELDS = (
    "watershed_sigma_um", "merge_tree_sigma_um", "peak_cluster_radius_um",
    "same_lobe_collapse_probability", "pair_prior_distinct",
    "pair_likelihood_temperature",
)
ADVANCED_SEGMENTATION_FIELDS = ("sigma_levels_um", "h_levels_um")


def discover_categories(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def discover_scenes(root: Path, category: str) -> list[str]:
    directory = root / category
    if not directory.is_dir():
        return []
    return sorted(path.name for path in directory.iterdir() if (path / "scene.json").is_file())


class PipelineReplayWorkbenchWidget(QWidget):
    def __init__(self, *, viewer: Any, scenes_root: str | Path, pipeline_paths: PipelinePaths | None = None) -> None:
        if not QT_AVAILABLE:
            raise ImportError("qtpy is required to create the pipeline replay workbench")
        super().__init__()
        self.viewer = viewer
        self.scenes_root = Path(scenes_root)
        self.pipeline_paths = pipeline_paths or PipelinePaths.discover()
        self.source: PipelineReplaySource | None = None
        self.runner: PipelineReplayRunner | None = None
        self._future: Future | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline-replay")
        self.parameter_controls: dict[tuple[int, str], Any] = {}
        self._build_ui()
        self._connect()
        self.refresh_categories()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        selection = QGroupBox("Saved tracking scene")
        form = QFormLayout(selection)
        self.category_combo, self.scene_combo = QComboBox(), QComboBox()
        self.frame_combo, self.cell_combo = QComboBox(), QComboBox()
        form.addRow("Category", self.category_combo)
        form.addRow("Scene", self.scene_combo)
        form.addRow("Original frame", self.frame_combo)
        form.addRow("Production cell", self.cell_combo)
        self.reload_button = QPushButton("Reload production baseline")
        form.addRow(self.reload_button)
        root.addWidget(selection)

        replay = QGroupBox("Replay")
        replay_form = QFormLayout(replay)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([ReplayMode.EXACT.value, ReplayMode.LOCAL.value])
        replay_form.addRow("Mode", self.mode_combo)
        self.halo_edits = []
        halo_row = QHBoxLayout()
        for _ in range(3):
            edit = QSpinBox(); edit.setRange(0, 10000); edit.setSpecialValueText("Auto")
            self.halo_edits.append(edit); halo_row.addWidget(edit)
        replay_form.addRow("Halo Z/Y/X (voxels)", halo_row)
        self.stage_combo = QComboBox(); self.stage_combo.addItems([f"Stage {value}" for value in range(1, 6)])
        replay_form.addRow("Selected stage", self.stage_combo)
        root.addWidget(replay)

        tabs = QTabWidget()
        tabs.addTab(self._config_tab(1, "Preprocessing", ("low_percentile", "high_percentile", "denoise_sigma_um", "background_sigma_um")), "Stage 1")
        tabs.addTab(self._config_tab(2, "Masking", ("threshold_multiplier", "threshold_offset")), "Stage 2")
        stage3 = QWidget(); stage3_layout = QVBoxLayout(stage3)
        stage3_layout.addWidget(self._config_group(3, "Segmentation", MAIN_SEGMENTATION_FIELDS))
        stage3_layout.addWidget(self._config_group(3, "Advanced", ADVANCED_SEGMENTATION_FIELDS))
        stage3_layout.addStretch(1)
        tabs.addTab(stage3, "Stage 3")
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(tabs)
        root.addWidget(scroll, 1)

        buttons = QHBoxLayout()
        self.run_stage_button = QPushButton("Run stage")
        self.run_downstream_button = QPushButton("Run stage + downstream")
        buttons.addWidget(self.run_stage_button); buttons.addWidget(self.run_downstream_button)
        root.addLayout(buttons)
        buttons2 = QHBoxLayout()
        self.reset_button = QPushButton("Reset selected stage")
        self.defaults_button = QPushButton("Restore all production defaults")
        buttons2.addWidget(self.reset_button); buttons2.addWidget(self.defaults_button)
        root.addLayout(buttons2)

        summary_group = QGroupBox("Result summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_label = QLabel("Load a scene to begin."); self.summary_label.setWordWrap(True)
        self.context_label = QLabel(""); self.context_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label); summary_layout.addWidget(self.context_label)
        root.addWidget(summary_group)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True); root.addWidget(self.status_label)

    def _config_tab(self, stage: int, title: str, names: tuple[str, ...]):
        widget = QWidget(); layout = QVBoxLayout(widget)
        layout.addWidget(self._config_group(stage, title, names)); layout.addStretch(1)
        return widget

    def _config_group(self, stage: int, title: str, names: tuple[str, ...]):
        group = QGroupBox(title); form = QFormLayout(group)
        for name in names:
            if name in ADVANCED_SEGMENTATION_FIELDS:
                control = QLineEdit()
            else:
                control = QDoubleSpinBox(); control.setDecimals(6); control.setRange(-1e6, 1e6); control.setSingleStep(0.01)
            self.parameter_controls[(stage, name)] = control
            form.addRow(name, control)
        return group

    def _connect(self) -> None:
        self.category_combo.currentTextChanged.connect(self.refresh_scenes)
        self.scene_combo.currentTextChanged.connect(self.load_scene_metadata)
        self.frame_combo.currentTextChanged.connect(self._refresh_cells)
        self.reload_button.clicked.connect(self.reload_baseline)
        self.run_stage_button.clicked.connect(lambda: self._start_run(False))
        self.run_downstream_button.clicked.connect(lambda: self._start_run(True))
        self.reset_button.clicked.connect(self._reset_stage)
        self.defaults_button.clicked.connect(self._restore_defaults)

    def refresh_categories(self) -> None:
        self.category_combo.clear(); self.category_combo.addItems(discover_categories(self.scenes_root))
        self.refresh_scenes(self.category_combo.currentText())

    def refresh_scenes(self, category: str) -> None:
        self.scene_combo.clear(); self.scene_combo.addItems(discover_scenes(self.scenes_root, category))
        self.load_scene_metadata(self.scene_combo.currentText())

    def load_scene_metadata(self, scene_name: str) -> None:
        if not scene_name or not self.category_combo.currentText():
            return
        try:
            path = self.scenes_root / self.category_combo.currentText() / scene_name
            self.source = PipelineReplaySource.from_scene(path, self.pipeline_paths)
            self.runner = PipelineReplayRunner(self.source)
            self.frame_combo.clear(); self.frame_combo.addItems([str(value) for value in self.source.frames])
            self._refresh_cells(); self._load_controls(); self.reload_baseline()
        except Exception as error:
            self._show_error(error)

    def _refresh_cells(self, *_args) -> None:
        self.cell_combo.clear()
        if self.source is None or not self.frame_combo.currentText():
            return
        frame = int(self.frame_combo.currentText())
        self.cell_combo.addItems([str(value) for value in self.source.selected_cells.get(frame, ())])

    def reload_baseline(self) -> None:
        if self.runner is None or not self.frame_combo.currentText() or not self.cell_combo.currentText():
            return
        try:
            frame, cell = int(self.frame_combo.currentText()), int(self.cell_combo.currentText())
            self.runner.reload_baseline(frame)
            if not (self.runner.baseline.instance_labels == cell).any():
                raise ValueError(f"selected cell {cell} is absent from saved Stage 6 segmentation")
            remove_pipeline_replay_layers(self.viewer)
            add_production_layers(self.viewer, self.source, self.runner.baseline, cell)
            context = self.source.load_tracking_context(frame, cell)
            add_tracking_context_layers(self.viewer, self.source, context, frame)
            self.context_label.setText(self._format_tracking_context(context, frame, cell))
            self.status_label.setText(f"Loaded frame {frame}, production cell {cell} from raw Zarr + Stage 6.")
        except Exception as error:
            self._show_error(error)

    def _format_tracking_context(self, context, frame: int, cell: int) -> str:
        """Render compact read-only Stage 7/8 evidence without rerunning tracking."""

        lines = [
            f"Stage 7 source: {self.pipeline_paths.stage7_tracking / 'tracks.csv'}",
            f"Stage 8 source: {self.pipeline_paths.stage8_stitching / 'tracks.csv'}",
        ]
        tracks = context.stage7.get("tracks")
        track_ids = []
        if tracks is not None and not tracks.empty and "track_id" in tracks:
            track_ids = sorted(set(tracks["track_id"].astype(int)))
            points = []
            for row in tracks.sort_values("frame").itertuples():
                coordinate = tuple(
                    round(float(getattr(row, axis)), 2)
                    for axis in ("z", "y", "x")
                    if hasattr(row, axis)
                )
                points.append(f"t{int(row.frame)} {coordinate}")
            lines.append(f"Selected Stage 7 track(s): {track_ids}; nearby points: {'; '.join(points)}")
        events = context.stage7.get("association_events")
        if events is not None and not events.empty:
            row = events.iloc[-1]
            values = []
            for column in (
                "decision_type", "association_probability", "probability_margin",
                "distance_um", "position_cost", "volume_cost", "shape_cost",
                "intensity_cost", "motion_cost",
            ):
                if column in row.index:
                    values.append(f"{column}={row[column]}")
            lines.append("Selected association: " + ", ".join(values))
        candidates = context.stage7.get("association_candidates")
        if candidates is not None and not candidates.empty:
            rank_column = "candidate_rank_by_pair_cost" if "candidate_rank_by_pair_cost" in candidates else None
            top = candidates.sort_values(rank_column).head(3) if rank_column else candidates.head(3)
            lines.append(
                "Top competing associations: "
                + "; ".join(
                    f"track {getattr(row, 'track_id', '?')} → detection {getattr(row, 'detection_cell_index', '?')} "
                    f"p={getattr(row, 'association_probability', '?')} cost={getattr(row, 'pair_cost', '?')}"
                    for row in top.itertuples()
                )
            )
        stage8_tracks = context.stage8.get("tracks")
        if stage8_tracks is not None and not stage8_tracks.empty:
            virtual = (
                int(stage8_tracks["is_virtual_merge"].fillna(False).astype(bool).sum())
                if "is_virtual_merge" in stage8_tracks else 0
            )
            merge_events = (
                sorted(stage8_tracks["merge_event_id"].dropna().unique().tolist())
                if "merge_event_id" in stage8_tracks else []
            )
            lines.append(
                f"Stage 8 centers: {len(stage8_tracks)} rows ({virtual} virtual merge); "
                f"merge event IDs: {merge_events}"
            )
        evidence = [
            f"{name}={len(table)}"
            for name, table in context.stage8.items()
            if name != "tracks" and not table.empty
        ]
        lines.append("Stage 8 provenance/evidence: " + (", ".join(evidence) or "none for this selection"))
        return "\n".join(lines)

    def _config_for_stage(self, stage: int):
        return {1: self.runner.state.preprocessing_config, 2: self.runner.state.masking_config, 3: self.runner.state.segmentation_config}[stage]

    def _load_controls(self) -> None:
        if self.runner is None: return
        for (stage, name), control in self.parameter_controls.items():
            value = getattr(self._config_for_stage(stage), name)
            if isinstance(control, QLineEdit): control.setText(", ".join(str(item) for item in value))
            else: control.setValue(value)

    def _apply_controls(self) -> None:
        method_names = {1: "preprocessing", 2: "masking", 3: "segmentation"}
        for stage, method_name in method_names.items():
            changes = {}
            for (control_stage, name), control in self.parameter_controls.items():
                if control_stage != stage:
                    continue
                if isinstance(control, QLineEdit):
                    changes[name] = tuple(
                        float(part.strip())
                        for part in control.text().split(",")
                        if part.strip()
                    )
                else:
                    changes[name] = control.value()
            getattr(self.runner, f"update_{method_name}_config")(**changes)

    def _start_run(self, downstream: bool) -> None:
        if self.runner is None or self._future is not None:
            return
        try: self._apply_controls()
        except Exception as error: self._show_error(error); return
        stage = self.stage_combo.currentIndex() + 1
        mode = ReplayMode(self.mode_combo.currentText())
        halo_values = tuple(edit.value() for edit in self.halo_edits)
        halo = None if not any(halo_values) else halo_values
        self.status_label.setText(f"Running {mode.value}, Stage {stage}{' through 5' if downstream else ''}…")
        self._set_buttons(False)
        self._future = self._executor.submit(self.runner.run, stage, downstream=downstream, mode=mode, halo_zyx=halo)
        QTimer.singleShot(100, self._poll_future)

    def _poll_future(self) -> None:
        if self._future is None: return
        if not self._future.done(): QTimer.singleShot(100, self._poll_future); return
        future, self._future = self._future, None; self._set_buttons(True)
        try:
            result = future.result(); self._render_result(result)
        except Exception as error: self._show_error(error)

    def _render_result(self, result) -> None:
        cell = int(self.cell_combo.currentText())
        add_trial_layers(self.viewer, self.source, self.runner, result)
        add_debug_layers(self.viewer, self.source, result)
        work_crop = self.runner.state.work_crop_global
        production_work = self.runner.baseline.instance_labels[work_crop]
        match = match_instance_by_iou(production_work, self.runner._labels_work, cell, self.source.voxel_size_zyx_um)
        add_difference_layers(self.viewer, self.source, self.runner.baseline, result, cell, match.trial_instance_id)
        component = result.component_results[0] if result.component_results else None
        component_summary = (
            f"Stage 3 processing: {component.processing_status}\n"
            f"Raw/effective peaks: {component.raw_peak_count}/{component.effective_peak_count}\n"
            f"Shape peaks/proposals: {component.shape_peak_count}/{component.center_proposal_count}\n"
            f"Unrepresented/candidate proposals: {component.unrepresented_proposal_count}/{component.candidate_proposal_count}\n"
            f"Merge candidate/routes: {component.merge_candidate}/{component.candidate_routes or '-'}\n"
            f"Candidate processing: {component.candidate_processing_status}; error: {component.candidate_error or '-'}\n"
            f"Geometry forced/executed: {component.geometry_forced}/{component.geometry_executed}\n"
            f"Final markers/instances: {component.marker_count}/{component.instance_count}\n"
            f"Pair evidence records: {len(component.pair_evidence)}\n"
            f"Stage 3 error: {component.error or '-'}\n"
            if component is not None
            else "Stage 3 processing: no retained component debug result\n"
        )
        feature_table = compare_feature_rows(self.runner.baseline.cells, result.features, cell, match.trial_instance_id) if result.features is not None else None
        self.summary_label.setText(
            f"Frame: {result.frame}\nProduction cell: {cell}\nMode: {result.mode.value}\n"
            f"Relevant components: {self.runner.state.relevant_component_ids}\n"
            f"{component_summary}"
            f"Best trial instance: {match.trial_instance_id}\nIoU: {match.iou:.4f} ({match.status})\n"
            f"Centroid displacement: {match.centroid_displacement_voxels if match.centroid_displacement_voxels is not None else '-'} voxels; "
            f"{match.centroid_displacement_um if match.centroid_displacement_um is not None else '-'} µm\n"
            f"Production/trial volume: {match.production_volume}/{match.trial_volume}\n"
            f"Disconnected parts: {match.trial_disconnected_parts}\n"
            f"Compared Stage 5 features: {0 if feature_table is None else len(feature_table)}"
        )
        self.status_label.setText(result.warning or "Replay completed with canonical full-frame processing.")

    def _reset_stage(self) -> None:
        stage = self.stage_combo.currentIndex() + 1
        if self.runner is not None:
            self.runner.reset_stage(stage)
            self._load_controls()
            self.status_label.setText(
                f"Stage {stage} reset to production defaults."
                if stage <= 3 else f"Stage {stage} cached result cleared."
            )

    def _restore_defaults(self) -> None:
        if self.runner is not None: self.runner.restore_production_defaults(); self._load_controls(); self.status_label.setText("All production defaults restored.")

    def _set_buttons(self, enabled: bool) -> None:
        self.run_stage_button.setEnabled(enabled); self.run_downstream_button.setEnabled(enabled)

    def _show_error(self, error: Exception) -> None:
        message = str(error); self.status_label.setText(f"Error: {message}")
        QMessageBox.critical(self, "Pipeline replay", message)

    def closeEvent(self, event) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


def add_pipeline_replay_workbench(
    viewer: Any,
    *,
    scenes_root: str | Path,
    pipeline_paths: PipelinePaths | None = None,
    dock_area: str = "right",
) -> PipelineReplayWorkbenchWidget:
    widget = PipelineReplayWorkbenchWidget(viewer=viewer, scenes_root=scenes_root, pipeline_paths=pipeline_paths)
    viewer.window.add_dock_widget(widget, area=dock_area, name="Pipeline Replay Workbench")
    return widget


__all__ = ["PipelineReplayWorkbenchWidget", "add_pipeline_replay_workbench", "QT_AVAILABLE"]
