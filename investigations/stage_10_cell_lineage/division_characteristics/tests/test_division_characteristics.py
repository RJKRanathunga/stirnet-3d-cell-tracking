"""Focused synthetic tests for transition inference and feature extraction."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from investigations.stage_10_cell_lineage.division_characteristics.config import (
    InvestigationConfig,
)
from investigations.stage_10_cell_lineage.division_characteristics.feature_extraction import (
    extract_cell_observation,
)
from investigations.stage_10_cell_lineage.division_characteristics.repository_io import (
    FrameArtifacts,
)
from investigations.stage_10_cell_lineage.division_characteristics.scene_cases import (
    DivisionSceneError,
    parse_division_scene,
)


class DivisionSceneTests(unittest.TestCase):
    def _write_scene(
        self,
        root: Path,
        selected_cells: dict[int, list[int]],
    ) -> Path:
        path = root / "001"
        path.mkdir(parents=True)
        frames = np.asarray(sorted(selected_cells), dtype=np.int32)
        shape = (len(frames), 4, 8, 8)
        np.save(path / "frames.npy", frames, allow_pickle=False)
        np.save(path / "binary_mask.npy", np.zeros(shape, dtype=bool), allow_pickle=False)
        np.save(path / "instance_labels.npy", np.zeros(shape, dtype=np.int32), allow_pickle=False)
        np.save(path / "raw.npy", np.zeros(shape, dtype=np.uint16), allow_pickle=False)
        metadata = {
            "schema_version": 1,
            "sample_id": "sample",
            "selected_cells": {str(k): v for k, v in selected_cells.items()},
            "files": {
                "frames": "frames.npy",
                "binary_mask": "binary_mask.npy",
                "instance_labels": "instance_labels.npy",
                "masked_images": {"raw": "raw.npy"},
            },
        }
        with (path / "scene.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file)
        return path

    def test_variable_window_and_transition_gap_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = self._write_scene(
                root,
                {3: [10], 4: [11], 7: [20, 21], 9: [22, 23]},
            )
            case = parse_division_scene(scene, root=root)
            self.assertEqual(case.event_frame, 7)
            self.assertEqual(case.previous_parent_frame, 4)
            self.assertEqual(case.transition_gap_frames, 3)
            self.assertTrue(case.warnings)

    def test_one_cell_after_event_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = self._write_scene(root, {1: [10], 2: [20, 21], 3: [22]})
            with self.assertRaises(DivisionSceneError):
                parse_division_scene(scene, root=root)


class FeatureExtractionTests(unittest.TestCase):
    def test_raw_and_mask_robust_features_are_extracted(self) -> None:
        labels = np.zeros((7, 15, 15), dtype=np.int32)
        labels[2:5, 5:10, 5:10] = 4
        raw = np.full(labels.shape, 10, dtype=np.float32)
        raw[labels == 4] = 50
        preprocessed = raw / 50.0
        binary = labels > 0
        cells = pd.DataFrame({"cell_id": [4], "volume_voxels": [75]})
        artifacts = FrameArtifacts(
            sample_id="sample",
            frame=3,
            raw=raw,
            preprocessed=preprocessed,
            binary_mask=binary,
            instance_labels=labels,
            cells=cells,
        )
        result = extract_cell_observation(
            artifacts,
            4,
            config=InvestigationConfig(
                core_erosion_um=0.2,
                fixed_radius_um=2.0,
                background_shell_inner_um=0.5,
                background_shell_outer_um=2.0,
            ),
        )
        self.assertEqual(result["volume_voxels"], 75)
        self.assertAlmostEqual(result["raw_mask_mean"], 50.0)
        self.assertAlmostEqual(result["raw_background_median"], 10.0)
        self.assertAlmostEqual(result["raw_background_corrected_mean"], 40.0)
        self.assertTrue(np.isfinite(result["sphericity"]))


if __name__ == "__main__":
    unittest.main()
