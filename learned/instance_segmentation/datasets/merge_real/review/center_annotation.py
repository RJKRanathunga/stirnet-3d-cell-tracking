"""Pass 2: place one approximate 3D center marker per real cell."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import MergeRealConfig, MergeRealPaths
from ..extraction import crop_origin, load_review_volume
from ..repository_io import FullProcessedRepository
from .common import candidate_summary, review_candidates


def launch_center_annotation(paths: MergeRealPaths, config: MergeRealConfig) -> None:
    import napari
    from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

    candidates = review_candidates(paths, confirmed_only=True)
    if candidates.empty:
        raise RuntimeError("No confirmed merge cases. Complete pass 1 first.")
    repository = FullProcessedRepository(paths)
    viewer = napari.Viewer()
    state = {"index": 0, "slices": None}

    panel = QWidget()
    layout = QVBoxLayout(panel)
    info = QLabel()
    info.setWordWrap(True)
    layout.addWidget(info)
    save = QPushButton("Save centers")
    layout.addWidget(save)
    nav = QHBoxLayout()
    previous = QPushButton("Previous")
    next_button = QPushButton("Next")
    nav.addWidget(previous)
    nav.addWidget(next_button)
    layout.addLayout(nav)
    viewer.window.add_dock_widget(panel, area="right", name="Cell center annotation")

    def current():
        return candidates.iloc[state["index"]]

    def centers_path(candidate) -> object:
        return paths.centers_dir / f"{candidate.candidate_id}.csv"

    def load_current():
        candidate = current()
        sample = repository.sample(str(candidate.sample_id))
        volumes = load_review_volume(sample, int(candidate.frame), int(candidate.cell_id), config)
        slices = volumes["slices"]
        state["slices"] = slices
        viewer.layers.clear()
        scale = config.voxel_size_zyx_um
        if "raw" in volumes:
            viewer.add_image(volumes["raw"], name="Raw", scale=scale, visible=False)
        viewer.add_image(volumes["preprocessed"], name="Preprocessed", scale=scale)
        viewer.add_labels(volumes["production_labels"], name="Context Labels", scale=scale, opacity=0.25)
        viewer.add_labels(volumes["candidate_mask"], name="Candidate Mask", scale=scale, opacity=0.45)
        saved = centers_path(candidate)
        if saved.exists():
            frame = pd.read_csv(saved)
            origin = np.asarray(crop_origin(slices), dtype=float)
            points = frame[["z", "y", "x"]].to_numpy(dtype=float) - origin[None, :]
        else:
            points = np.empty((0, 3), dtype=float)
        viewer.add_points(points, name="Cell Centers", scale=scale, size=5)
        info.setText(
            f"{state['index'] + 1}/{len(candidates)}\n"
            + candidate_summary(candidate)
            + f"\nExpected centers: {int(candidate.expected_cell_count)}"
        )

    def save_current():
        candidate = current()
        layer = viewer.layers["Cell Centers"]
        points = np.asarray(layer.data, dtype=float)
        expected = int(candidate.expected_cell_count)
        if len(points) != expected:
            info.setText(info.text() + f"\nERROR: expected {expected} centers, found {len(points)}")
            return
        origin = np.asarray(crop_origin(state["slices"]), dtype=float)
        global_points = points + origin[None, :]
        frame = pd.DataFrame(
            {
                "candidate_id": str(candidate.candidate_id),
                "center_index": np.arange(1, len(points) + 1, dtype=int),
                "z": global_points[:, 0],
                "y": global_points[:, 1],
                "x": global_points[:, 2],
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        path = centers_path(candidate)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        info.setText(info.text() + "\nSaved centers.")

    def move(delta: int):
        state["index"] = max(0, min(len(candidates) - 1, state["index"] + int(delta)))
        load_current()

    save.clicked.connect(save_current)
    previous.clicked.connect(lambda: move(-1))
    next_button.clicked.connect(lambda: move(1))
    load_current()
    napari.run()


__all__ = ["launch_center_annotation"]
