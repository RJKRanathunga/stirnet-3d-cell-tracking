"""Pass 3: verify/correct the automatically generated 3D instance partition."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from ..config import MergeRealConfig, MergeRealPaths
from ..extraction import crop_origin, load_review_volume, materialize_case
from ..repository_io import FullProcessedRepository
from .common import candidate_summary, review_candidates, upsert_review
from .partition_generation import generate_partition


def launch_partition_review(paths: MergeRealPaths, config: MergeRealConfig) -> None:
    import napari
    from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

    all_candidates = review_candidates(paths, confirmed_only=True)
    if all_candidates.empty:
        raise RuntimeError("No confirmed candidates. Complete pass 1 first.")
    with_centers = []
    for _, candidate in all_candidates.iterrows():
        if (paths.centers_dir / f"{candidate.candidate_id}.csv").exists():
            with_centers.append(candidate)
    if not with_centers:
        raise RuntimeError("No center annotations found. Complete pass 2 first.")
    candidates = pd.DataFrame(with_centers).reset_index(drop=True)
    repository = FullProcessedRepository(paths)
    viewer = napari.Viewer()
    state = {"index": 0, "volumes": None, "local_centers": None}

    panel = QWidget()
    layout = QVBoxLayout(panel)
    info = QLabel()
    info.setWordWrap(True)
    layout.addWidget(info)
    regenerate = QPushButton("Regenerate watershed")
    accept = QPushButton("Accept")
    accept_corrected = QPushButton("Accept corrected")
    ambiguous = QPushButton("Ambiguous")
    reject = QPushButton("Reject")
    for button in (regenerate, accept, accept_corrected, ambiguous, reject):
        layout.addWidget(button)
    nav = QHBoxLayout()
    previous = QPushButton("Previous")
    next_button = QPushButton("Next")
    nav.addWidget(previous)
    nav.addWidget(next_button)
    layout.addLayout(nav)
    viewer.window.add_dock_widget(panel, area="right", name="3D partition review")

    def current():
        return candidates.iloc[state["index"]]

    def partition_dir(candidate):
        return paths.partitions_dir / str(candidate.candidate_id)

    def load_current():
        candidate = current()
        sample = repository.sample(str(candidate.sample_id))
        volumes = load_review_volume(sample, int(candidate.frame), int(candidate.cell_id), config)
        state["volumes"] = volumes
        origin = np.asarray(crop_origin(volumes["slices"]), dtype=float)
        centers = pd.read_csv(paths.centers_dir / f"{candidate.candidate_id}.csv")
        local_centers = centers[["z", "y", "x"]].to_numpy(dtype=float) - origin[None, :]
        state["local_centers"] = local_centers

        saved_dir = partition_dir(candidate)
        saved_partition = saved_dir / "instance_labels.npy"
        saved_uncertain = saved_dir / "uncertain_mask.npy"
        if saved_partition.exists():
            partition = np.load(saved_partition, allow_pickle=False)
        else:
            partition = generate_partition(
                volumes["candidate_mask"], local_centers, config.voxel_size_zyx_um
            )
        uncertain = (
            np.load(saved_uncertain, allow_pickle=False)
            if saved_uncertain.exists()
            else np.zeros_like(partition, dtype=np.uint8)
        )

        viewer.layers.clear()
        scale = config.voxel_size_zyx_um
        if "raw" in volumes:
            viewer.add_image(volumes["raw"], name="Raw", scale=scale, visible=False)
        viewer.add_image(volumes["preprocessed"], name="Preprocessed", scale=scale)
        viewer.add_labels(volumes["candidate_mask"], name="Candidate Mask", scale=scale, opacity=0.2)
        viewer.add_points(local_centers, name="Centers", scale=scale, size=5)
        viewer.add_labels(partition, name="Proposed Instances", scale=scale, opacity=0.55)
        viewer.add_labels(uncertain, name="Uncertain Boundary", scale=scale, opacity=0.45)
        info.setText(
            f"{state['index'] + 1}/{len(candidates)}\n" + candidate_summary(candidate)
            + "\nEdit 'Proposed Instances' only if needed; paint optional uncertainty in 'Uncertain Boundary'."
        )

    def regenerate_current():
        volumes = state["volumes"]
        partition = generate_partition(
            volumes["candidate_mask"], state["local_centers"], config.voxel_size_zyx_um
        )
        viewer.layers["Proposed Instances"].data = partition

    def save_status(status: str, materialize: bool):
        candidate = current()
        partition = np.asarray(viewer.layers["Proposed Instances"].data, dtype=np.int32).copy()
        uncertain = np.asarray(viewer.layers["Uncertain Boundary"].data, dtype=np.uint8).copy()
        candidate_mask = np.asarray(state["volumes"]["candidate_mask"], dtype=bool)
        partition[~candidate_mask] = 0
        uncertain[~candidate_mask] = 0
        expected = int(candidate.expected_cell_count)
        actual = int(len(np.unique(partition[partition > 0])))
        if materialize and actual != expected:
            info.setText(info.text() + f"\nERROR: expected {expected} nonzero instances, found {actual}.")
            return
        output = partition_dir(candidate)
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "instance_labels.npy", partition, allow_pickle=False)
        np.save(output / "uncertain_mask.npy", uncertain, allow_pickle=False)
        with (output / "review.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "candidate_id": str(candidate.candidate_id),
                    "status": status,
                    "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        upsert_review(
            paths.reviews_dir / "partition_reviews.csv",
            {
                "candidate_id": str(candidate.candidate_id),
                "sample_id": str(candidate.sample_id),
                "frame": int(candidate.frame),
                "cell_id": int(candidate.cell_id),
                "status": status,
                "instance_count": int(len(np.unique(partition[partition > 0]))),
                "has_uncertain_voxels": bool(np.any(uncertain)),
                "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        if materialize:
            materialize_case(paths, config, candidate, partition, uncertain, partition_status=status)
        if state["index"] < len(candidates) - 1:
            move(1)

    def move(delta: int):
        state["index"] = max(0, min(len(candidates) - 1, state["index"] + int(delta)))
        load_current()

    regenerate.clicked.connect(regenerate_current)
    accept.clicked.connect(lambda: save_status("accepted", True))
    accept_corrected.clicked.connect(lambda: save_status("accepted_corrected", True))
    ambiguous.clicked.connect(lambda: save_status("ambiguous", False))
    reject.clicked.connect(lambda: save_status("rejected", False))
    previous.clicked.connect(lambda: move(-1))
    next_button.clicked.connect(lambda: move(1))
    load_current()
    napari.run()


__all__ = ["launch_partition_review"]
