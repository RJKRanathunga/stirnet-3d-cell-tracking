"""Pass 1: rapidly classify mined candidates as merge/non-merge/ambiguous."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from ..config import MergeRealConfig, MergeRealPaths
from ..models import FILTER_LABELS
from .common import candidate_summary, review_candidates, temporal_case_data, upsert_review


def launch_case_filter(paths: MergeRealPaths, config: MergeRealConfig) -> None:
    import napari
    from qtpy.QtWidgets import (
        QComboBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    candidates = review_candidates(paths, confirmed_only=False)
    if candidates.empty:
        raise RuntimeError("No candidates found. Run run_mining.py first.")

    viewer = napari.Viewer()
    state = {"index": 0}

    panel = QWidget()
    layout = QVBoxLayout(panel)
    info = QLabel()
    info.setWordWrap(True)
    layout.addWidget(info)

    expected = QSpinBox()
    expected.setRange(1, 12)
    expected.setValue(2)
    layout.addWidget(QLabel("Expected cell count"))
    layout.addWidget(expected)

    confidence = QComboBox()
    confidence.addItems(["high", "medium", "low"])
    layout.addWidget(QLabel("Confidence"))
    layout.addWidget(confidence)

    notes = QLineEdit()
    notes.setPlaceholderText("Optional notes")
    layout.addWidget(notes)

    button_rows = []
    labels = [
        ("2-cell merge", "confirmed_2_cell_merge", 2),
        ("3+ merge", "confirmed_3plus_cell_merge", -3),
        ("Not merge", "not_merge", None),
        ("Other error", "other_segmentation_error", None),
        ("Division/birth", "division_or_birth", None),
        ("Ambiguous", "ambiguous", None),
        ("Skip", "skip", None),
    ]
    for text, classification, count in labels:
        button = QPushButton(text)
        button.clicked.connect(
            lambda _=False, c=classification, n=count: save_and_next(c, n)
        )
        layout.addWidget(button)
        button_rows.append(button)

    nav = QHBoxLayout()
    previous = QPushButton("Previous")
    next_button = QPushButton("Next")
    nav.addWidget(previous)
    nav.addWidget(next_button)
    layout.addLayout(nav)
    previous.clicked.connect(lambda: move(-1))
    next_button.clicked.connect(lambda: move(1))

    viewer.window.add_dock_widget(panel, area="right", name="Merge case filter")

    def current():
        return candidates.iloc[state["index"]]

    def load_current():
        candidate = current()
        data = temporal_case_data(paths, config, candidate)
        viewer.layers.clear()
        scale = (1.0, *config.voxel_size_zyx_um)
        if data["raw"] is not None:
            viewer.add_image(data["raw"], name="Raw", scale=scale, visible=False)
        viewer.add_image(data["preprocessed"], name="Preprocessed", scale=scale)
        viewer.add_labels(data["labels"], name="Production Labels", scale=scale, opacity=0.35)
        viewer.add_labels(data["candidate_mask"], name="Candidate Component", scale=scale, opacity=0.55)
        for track_id, points in data["track_points"].items():
            viewer.add_points(
                points,
                name=f"Track {track_id}",
                scale=scale,
                size=4,
            )
        if data["predicted_points"]:
            predicted = np.stack(list(data["predicted_points"].values()), axis=0)
            properties = {"track_id": np.asarray(list(data["predicted_points"].keys()), dtype=int)}
            viewer.add_points(
                predicted,
                name="Predicted Positions",
                scale=scale,
                size=6,
                properties=properties,
                symbol="cross",
            )
        viewer.dims.set_current_step(0, int(data["event_index"]))
        info.setText(
            f"{state['index'] + 1}/{len(candidates)}\n" + candidate_summary(candidate)
        )
        expected.setValue(2)
        notes.clear()

    def move(delta: int):
        state["index"] = max(0, min(len(candidates) - 1, state["index"] + int(delta)))
        load_current()

    def save_and_next(classification: str, default_count: int | None):
        candidate = current()
        if default_count == 2:
            expected.setValue(2)
        elif default_count == -3:
            expected.setValue(max(3, expected.value()))
        record = {
            "candidate_id": str(candidate.candidate_id),
            "sample_id": str(candidate.sample_id),
            "frame": int(candidate.frame),
            "cell_id": int(candidate.cell_id),
            "classification": classification,
            "expected_cell_count": int(expected.value()),
            "confidence": confidence.currentText(),
            "notes": notes.text().strip(),
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        upsert_review(paths.filter_reviews_csv, record)
        if state["index"] < len(candidates) - 1:
            move(1)

    load_current()
    napari.run()


__all__ = ["launch_case_filter"]
