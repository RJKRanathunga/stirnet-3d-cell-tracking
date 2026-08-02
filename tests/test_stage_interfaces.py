"""Focused contracts for stage APIs, diagnostics, and shared I/O."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.api import create_binary_mask, preprocess_volume
from src.diagnostics import DiagnosticTrace, StageTrace
from src.diagnostics.invariants import validate_tracks
from src.io import (
    load_csv,
    load_npy,
    load_optional_csv,
    load_tracking_scene,
    save_csv,
    save_json,
    save_npy,
)


class StageInterfaceTests(unittest.TestCase):
    def test_optional_diagnostics_do_not_change_normal_output(self) -> None:
        volume = np.linspace(0, 1, 5 * 9 * 9, dtype=np.float32).reshape(5, 9, 9)
        normal = preprocess_volume(volume)
        diagnosed, trace = preprocess_volume(volume, return_diagnostics=True)
        np.testing.assert_array_equal(diagnosed, normal)
        self.assertIsInstance(trace, StageTrace)
        self.assertIn("background", trace.intermediates)

        mask = create_binary_mask(normal)
        diagnosed_mask, mask_trace = create_binary_mask(
            normal, return_diagnostics=True
        )
        np.testing.assert_array_equal(diagnosed_mask, mask)
        self.assertEqual(mask_trace.stage_name, "02_masking")

    def test_diagnostic_trace_combines_stage_traces(self) -> None:
        trace = DiagnosticTrace()
        trace.stages["01_preprocessing"] = StageTrace("01_preprocessing")
        self.assertEqual(list(trace.stages), ["01_preprocessing"])

    def test_array_and_table_io_preserve_dtype_order_and_empty_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            array = np.arange(12, dtype=np.uint16).reshape(3, 2, 2)
            save_npy(array, root / "array.npy")
            loaded = load_npy(root / "array.npy", expected_dtype=np.uint16)
            np.testing.assert_array_equal(loaded, array)

            table = pd.DataFrame({"second": [2, 1], "first": ["b", "a"]})
            save_csv(table, root / "table.csv")
            restored = pd.read_csv(root / "table.csv")
            self.assertEqual(restored.columns.tolist(), ["second", "first"])
            self.assertEqual(restored["second"].tolist(), [2, 1])

            save_csv(pd.DataFrame(), root / "empty.csv")
            self.assertTrue(load_optional_csv(root / "empty.csv").empty)

            with self.assertRaises(ValueError):
                load_csv(root / "table.csv", required_columns=("missing",))
            with self.assertRaises(FileNotFoundError):
                load_npy(root / "missing.npy")

    def test_track_invariants_distinguish_virtual_rows(self) -> None:
        tracks = pd.DataFrame({
            "track_id": [1, 2], "frame": [0, 0], "cell": [0, 0],
            "cell_id": [4, 4], "z": [1., 1.], "y": [2., 2.], "x": [3., 3.],
            "is_virtual_merge": [False, True], "merge_event_id": [pd.NA, 0],
            "source_track_id": [1, 9], "source_merged_cell_id": [pd.NA, 4],
        })
        self.assertEqual(validate_tracks(tracks), [])

    def test_tracking_scene_loader_validates_aligned_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_npy(np.array([2, 3], dtype=np.int64), root / "frames.npy")
            save_npy(np.zeros((2, 3, 4, 5), dtype=bool), root / "binary_mask.npy")
            save_npy(np.zeros((2, 3, 4, 5), dtype=np.int32), root / "labels.npy")
            save_npy(np.ones((2, 3, 4, 5), dtype=np.float32), root / "raw.npy")
            save_json({
                "schema_version": 1,
                "files": {
                    "frames": "frames.npy",
                    "binary_mask": "binary_mask.npy",
                    "instance_labels": "labels.npy",
                    "masked_images": {"raw": "raw.npy"},
                },
            }, root / "scene.json")
            scene = load_tracking_scene(root)
            self.assertEqual(scene.instance_labels.shape, (2, 3, 4, 5))
            self.assertEqual(list(scene.images), ["raw"])


if __name__ == "__main__":
    unittest.main()
