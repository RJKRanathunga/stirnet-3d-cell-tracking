"""Napari dock widget for filtering and jumping to final diagnostic events."""

from __future__ import annotations

import pandas as pd


def add_final_failure_navigator(viewer, visualization, *, dock_name: str = "Final Track Audit"):
    """Add a compact event browser without importing Qt during non-GUI use."""

    try:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QComboBox, QFormLayout, QHBoxLayout, QLabel, QPushButton,
            QPlainTextEdit, QVBoxLayout, QWidget,
        )
    except ImportError as exc:  # pragma: no cover - depends on Napari environment
        raise RuntimeError("QtPy is required for the Stage 12 Napari navigator") from exc

    events = visualization.diagnostic_events.copy()
    summaries = visualization.track_summary.set_index("track_id", drop=False)

    class FinalFailureNavigatorWidget(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.filtered = pd.DataFrame()
            self.index = 0
            self.filter_combo = QComboBox()
            self.filter_combo.addItems([
                "Failures and warnings", "Failures only", "All events",
                "Track starts", "Track ends", "Temporal gaps", "Stage 11 repairs",
            ])
            self.event_combo = QComboBox()
            self.previous_button = QPushButton("Previous")
            self.next_button = QPushButton("Next")
            self.jump_button = QPushButton("Jump to event")
            self.details = QPlainTextEdit()
            self.details.setReadOnly(True)
            self.details.setMinimumHeight(180)
            self.status = QLabel("")
            self.status.setWordWrap(True)

            form = QFormLayout()
            form.addRow("Show", self.filter_combo)
            form.addRow("Event", self.event_combo)
            buttons = QHBoxLayout()
            buttons.addWidget(self.previous_button)
            buttons.addWidget(self.next_button)
            layout = QVBoxLayout(self)
            layout.addLayout(form)
            layout.addLayout(buttons)
            layout.addWidget(self.jump_button)
            layout.addWidget(self.status)
            layout.addWidget(self.details)

            self.selected_layer = viewer.add_points(
                data=[], name="Selected Final Diagnostic", ndim=4,
                scale=(1.0, *visualization.voxel_size_zyx), size=9,
                face_color="yellow",
            )
            self.filter_combo.currentTextChanged.connect(self._apply_filter)
            self.event_combo.currentIndexChanged.connect(self._select_combo)
            self.previous_button.clicked.connect(lambda: self._move(-1))
            self.next_button.clicked.connect(lambda: self._move(1))
            self.jump_button.clicked.connect(self._jump)
            self._apply_filter()

        def _apply_filter(self, *_args) -> None:
            mode = self.filter_combo.currentText()
            if mode == "Failures and warnings":
                mask = events["is_failure"].astype(bool) | (events["severity"] == "warning")
            elif mode == "Failures only":
                mask = events["is_failure"].astype(bool)
            elif mode == "Track starts":
                mask = events["event_type"] == "track_start"
            elif mode == "Track ends":
                mask = events["event_type"] == "track_end"
            elif mode == "Temporal gaps":
                mask = events["event_type"] == "temporal_gap"
            elif mode == "Stage 11 repairs":
                mask = events["event_type"] == "stage11_repair"
            else:
                mask = pd.Series(True, index=events.index)
            self.filtered = events.loc[mask].sort_values(
                ["frame", "track_id", "event_type"], kind="stable"
            ).reset_index(drop=True)
            self.event_combo.blockSignals(True)
            self.event_combo.clear()
            for row in self.filtered.itertuples(index=False):
                self.event_combo.addItem(
                    f"f{int(row.frame):03d} · T{int(row.track_id)} · {row.classification}"
                )
            self.event_combo.blockSignals(False)
            self.index = 0
            self._render()

        def _select_combo(self, index: int) -> None:
            if index >= 0:
                self.index = index
                self._render()

        def _move(self, delta: int) -> None:
            if self.filtered.empty:
                return
            self.index = (self.index + delta) % len(self.filtered)
            self.event_combo.setCurrentIndex(self.index)
            self._render()

        def _jump(self) -> None:
            if self.filtered.empty:
                return
            row = self.filtered.iloc[self.index]
            frame = int(row["frame"])
            viewer.dims.set_current_step(0, frame)
            point = [[frame, float(row["z"]), float(row["y"]), float(row["x"])]]
            self.selected_layer.data = point
            self.selected_layer.visible = True

        def _render(self) -> None:
            if self.filtered.empty:
                self.status.setText("No events match this filter.")
                self.details.setPlainText("")
                self.selected_layer.data = []
                return
            self.index = min(self.index, len(self.filtered) - 1)
            row = self.filtered.iloc[self.index]
            track_id = int(row["track_id"])
            summary = summaries.loc[track_id]
            self.status.setText(f"Event {self.index + 1} of {len(self.filtered)}")
            lines = [
                f"Track: {track_id}",
                f"Event: {row['event_type']}",
                f"Frame: {int(row['frame'])}",
                f"Classification: {row['classification']}",
                f"Severity: {row['severity']}",
                f"Reason: {row['reason']}",
                "",
                f"Frames: {int(summary.first_frame)}–{int(summary.last_frame)}",
                f"Observations: {int(summary.observation_count)}",
                f"Start: {summary.start_classification}",
                f"End: {summary.end_classification}",
                f"Failure score: {float(summary.failure_score):.3f}",
                f"Failure reasons: {summary.failure_reasons or 'none'}",
                f"Original segments: {summary.original_segment_ids}",
                f"Stage 11 repairs: {int(summary.repair_count)}",
                f"Forced repairs: {int(summary.forced_repair_count)}",
                f"Unresolved ending: {bool(summary.unresolved_ending)}",
            ]
            self.details.setPlainText("\n".join(lines))
            self._jump()

    widget = FinalFailureNavigatorWidget()
    viewer.window.add_dock_widget(widget, name=dock_name, area="right")
    return widget
