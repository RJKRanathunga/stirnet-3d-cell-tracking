"""Stage 9 preparation contract migrated from the visualization notebook."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

from src.api import prepare_visualization_data

overlay_module = import_module("src.09_visualization.step04_scene_overlay")
prepare_stage8_scene_overlay = overlay_module.prepare_stage8_scene_overlay


class VisualizationPreparationTests(unittest.TestCase):
    def test_nearest_cell_mapping_and_napari_array_order(self) -> None:
        tracks = pd.DataFrame({
            "track_id": [8, 7],
            "frame": [1, 0],
            "z": [2.0, 1.0],
            "y": [4.1, 3.0],
            "x": [6.0, 5.0],
        })
        cells = pd.DataFrame({
            "frame": [0, 1],
            "cell_id": [11, 12],
            "centroid_z": [1.0, 2.0],
            "centroid_y": [3.0, 4.0],
            "centroid_x": [5.0, 6.0],
        })

        result, trace = prepare_visualization_data(
            tracks, cells, return_diagnostics=True
        )
        self.assertEqual(result.tracks["cell_id"].tolist(), [12, 11])
        self.assertEqual(result.cell_ids.tolist(), [12, 11])
        np.testing.assert_array_equal(
            result.tracks_array,
            tracks[["track_id", "frame", "z", "y", "x"]].to_numpy(float),
        )
        np.testing.assert_array_equal(
            result.points_array,
            tracks[["frame", "z", "y", "x"]].to_numpy(float),
        )
        self.assertEqual(trace.stage_name, "09_visualization")

    def test_endpoint_groups_filter_boundary_entries_and_exits(self) -> None:
        rows = []
        paths = {
            0: [(0, 20, 30, 30), (1, 20, 31, 30), (2, 20, 32, 30)],
            1: [(1, 20, 80, 80), (2, 20, 81, 80)],
            2: [(1, 0, 100, 100), (2, 1, 101, 100)],
            3: [(0, 20, 140, 140), (1, 20, 141, 140)],
            4: [(0, 0, 180, 180), (1, 1, 181, 180)],
        }
        for track_id, observations in paths.items():
            for frame, z, y, x in observations:
                rows.append({
                    "track_id": track_id,
                    "frame": frame,
                    "z": float(z),
                    "y": float(y),
                    "x": float(x),
                })
        tracks = pd.DataFrame(rows)
        cells = tracks.reset_index(drop=True).copy()
        cells["cell_id"] = np.arange(1, len(cells) + 1)
        cells["centroid_z"] = cells["z"]
        cells["centroid_y"] = cells["y"]
        cells["centroid_x"] = cells["x"]
        cells["z_min"] = np.maximum(cells["z"] - 1, 0)
        cells["y_min"] = np.maximum(cells["y"] - 2, 0)
        cells["x_min"] = np.maximum(cells["x"] - 2, 0)
        cells["z_max"] = cells["z"] + 2
        cells["y_max"] = cells["y"] + 3
        cells["x_max"] = cells["x"] + 3

        result = prepare_visualization_data(
            tracks,
            cells,
            spatial_shape_zyx=(64, 256, 256),
        )
        groups = result.endpoint_groups
        self.assertEqual(set(groups.new_failure_tracks.track_id), {1})
        self.assertEqual(set(groups.ended_failure_tracks.track_id), {3})
        self.assertEqual(set(groups.boundary_entry_tracks.track_id), {2})
        self.assertEqual(set(groups.boundary_exit_tracks.track_id), {4})
        self.assertEqual(result.voxel_size_zyx, (1.625, 0.40625, 0.40625))

    def test_scene_overlay_resolves_provenance_labels_and_both_transforms(self) -> None:
        class Scene:
            frames = np.asarray([10, 11])
            sample_id = "sample-a"
            crop_origin_zyx = (2, 20, 30)
            voxel_size_zyx = (1.625, 0.40625, 0.40625)
            metadata = {
                "cell_id_column": "cell_id",
                "selected_cells": {"10": [71]},
            }

        tracks = pd.DataFrame([
            {
                "track_id": 5, "frame": 10, "cell_id": 71,
                "z": 3., "y": 22., "x": 33., "is_virtual_merge": False,
                "merge_event_id": np.nan, "merge_role": np.nan,
                "source_track_id": 5, "source_merged_cell_id": np.nan,
            },
            {
                "track_id": 5, "frame": 11, "cell_id": 72,
                "z": 4., "y": 23., "x": 34., "is_virtual_merge": False,
                "merge_event_id": np.nan, "merge_role": np.nan,
                "source_track_id": 5, "source_merged_cell_id": np.nan,
            },
            {
                "track_id": 6, "frame": 10, "cell_id": 80,
                "z": 3.5, "y": 22.5, "x": 33.5, "is_virtual_merge": True,
                "merge_event_id": 2, "merge_role": "parent_a",
                "source_track_id": 9, "source_merged_cell_id": 71,
            },
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracks.to_csv(root / "tracks.csv", index=False)
            (root / "metadata.json").write_text(
                json.dumps({"sample_id": "sample-a"}), encoding="utf-8"
            )
            local = prepare_stage8_scene_overlay(
                Scene(),
                tracks_path=root / "tracks.csv",
                metadata_path=root / "metadata.json",
            )
            original = prepare_stage8_scene_overlay(
                Scene(),
                tracks_path=root / "tracks.csv",
                metadata_path=root / "metadata.json",
                use_original_coordinates=True,
            )

        self.assertEqual(local.summary["track_ids"], [5, 6])
        self.assertEqual(local.scale, (1.0, 1.625, 0.40625, 0.40625))
        self.assertEqual(local.translate, (10.0, 0.0, 0.0, 0.0))
        self.assertEqual(
            original.translate,
            (10.0, 3.25, 8.125, 12.1875),
        )
        np.testing.assert_array_equal(local.track_data, original.track_data)
        self.assertEqual(local.ordinary_properties["display_label"].tolist(), ["C71", "C72"])
        self.assertEqual(
            local.virtual_properties["virtual_display_label"].tolist(),
            ["C71 | parent_a"],
        )
        required = {
            "track_id", "cell_id", "frame", "is_virtual_merge",
            "merge_event_id", "merge_role", "source_track_id",
            "source_merged_cell_id",
        }
        self.assertTrue(required.issubset(local.virtual_properties))
        local_physical = (
            local.ordinary_point_data[0] * np.asarray(local.scale)
            + np.asarray(original.translate)
        )
        expected_physical = np.asarray([10., 3 * 1.625, 22 * 0.40625, 33 * 0.40625])
        np.testing.assert_allclose(local_physical, expected_physical)

    def test_visualization_notebooks_restore_scaled_layers_and_frame_display(self) -> None:
        root = Path(__file__).resolve().parents[1]
        stage9 = json.loads(
            (root / "notebooks" / "09_visualization.ipynb").read_text(encoding="utf-8")
        )
        stage9_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in stage9["cells"]
            if cell.get("cell_type") == "code"
        )
        tree = ast.parse(stage9_source)
        spatial_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"add_image", "add_labels", "add_tracks", "add_points"}
        ]
        self.assertTrue(spatial_calls)
        for call in spatial_calls:
            self.assertIn("scale", {keyword.arg for keyword in call.keywords})
        for layer_name in (
            "Raw Volume", "Preprocessed Volume", "Binary Mask", "Instance Labels",
            "Tracks - all", "Centroids - all", "Ended Tracks", "Ended Centroids",
            "New Tracks", "New Centroids", "Boundary Entry Tracks",
            "Boundary Entry Centroids", "Boundary Exit Tracks",
            "Boundary Exit Centroids",
        ):
            self.assertIn(layer_name, stage9_source)
        self.assertIn('"string": "{cell_id}"', stage9_source)

        scene_notebook = json.loads(
            (root / "notebooks" / "diagnostics" / "general_scene_visualizer.ipynb")
            .read_text(encoding="utf-8")
        )
        scene_source = "\n".join(
            "".join(cell.get("source", [])) for cell in scene_notebook["cells"]
        )
        self.assertIn("prepare_stage8_scene_overlay", scene_source)
        self.assertIn("Current scene time index:", scene_source)
        self.assertIn("Current original frame:", scene_source)
        self.assertIn("viewer.dims.events.current_step.connect", scene_source)
        self.assertIn('"string": "{display_label}"', scene_source)
        self.assertIn('"string": "{virtual_display_label}"', scene_source)


if __name__ == "__main__":
    unittest.main()
