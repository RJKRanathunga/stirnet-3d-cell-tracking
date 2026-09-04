"""Focused non-GUI contracts for the Stage 3 analysis notebook APIs."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from diagnostics.pipeline_replay.models import ProductionFrame
from diagnostics.stage_analysis.napari_layers import (
    OWNER,
    current_tzyx,
    remove_owned_layers,
)
from diagnostics.stage_analysis.runner import run_stage3_component
from diagnostics.stage_analysis.source import (
    Stage3AnalysisSource,
    build_display_scope,
    resolve_target_components,
)


config_module = import_module("src.source_instances.segmentation.config")
peaks_module = import_module("src.source_instances.segmentation.peaks")
DEFAULT_CONFIG = config_module.DEFAULT_SEGMENTATION_CONFIG


class TargetComponentResolutionTests(unittest.TestCase):
    def test_complete_component_is_recovered_beyond_scene_sized_selection(self) -> None:
        binary = np.zeros((3, 12, 12), dtype=bool)
        binary[1, 1:11, 6] = True
        production = np.zeros(binary.shape, dtype=np.int32)
        production[1, 5:8, 6] = 41

        result = resolve_target_components(
            production, binary, (41,), display_padding=0
        )

        self.assertEqual(result.target_component_ids, (1,))
        self.assertEqual(int(result.complete_component_mask(1).sum()), 10)
        self.assertEqual(result.component_bboxes[1][1], slice(1, 11))
        self.assertEqual(result.diagnostic_crop[1], slice(1, 11))

    def test_multiple_ids_in_one_component_produce_one_target(self) -> None:
        binary = np.zeros((3, 9, 9), dtype=bool)
        binary[1, 2:7, 4] = True
        production = np.zeros(binary.shape, dtype=np.int32)
        production[1, 2:4, 4] = 5
        production[1, 5:7, 4] = 8

        result = resolve_target_components(production, binary, (5, 8))

        self.assertEqual(result.target_component_ids, (1,))
        self.assertEqual(result.component_selected_ids[1], (5, 8))

    def test_ids_in_separate_components_produce_multiple_choices(self) -> None:
        binary = np.zeros((3, 9, 9), dtype=bool)
        binary[1, 2, 2] = True
        binary[1, 6, 6] = True
        production = np.zeros(binary.shape, dtype=np.int32)
        production[1, 2, 2] = 5
        production[1, 6, 6] = 8

        result = resolve_target_components(production, binary, (5, 8))

        self.assertEqual(result.target_component_ids, (1, 2))
        self.assertEqual(result.component_selected_ids, {1: (5,), 2: (8,)})

    def test_missing_and_foreground_free_ids_are_reported(self) -> None:
        binary = np.zeros((2, 4, 4), dtype=bool)
        production = np.zeros(binary.shape, dtype=np.int32)
        production[0, 1, 1] = 4

        result = resolve_target_components(production, binary, (4, 99))

        self.assertEqual(result.missing_production_ids, (99,))
        self.assertEqual(result.selected_ids_without_stage2_foreground, (4,))

    def test_selected_only_is_a_pure_display_scope_change(self) -> None:
        binary = np.zeros((2, 7, 7), dtype=bool)
        binary[0, 1:4, 2] = True
        binary[0, 5, 5] = True
        production = np.zeros(binary.shape, dtype=np.int32)
        production[0, 1:4, 2] = 7
        production[0, 5, 5] = 12
        result = resolve_target_components(production, binary, (7,), display_padding=4)

        selected = build_display_scope(
            production, binary, result, selected_only=True
        )
        all_cells = build_display_scope(
            production, binary, result, selected_only=False
        )

        self.assertEqual(set(np.unique(selected.production_labels)), {0, 7})
        self.assertEqual(set(np.unique(all_cells.production_labels)), {0, 7, 12})
        self.assertEqual(int(selected.binary_mask.sum()), 3)
        self.assertEqual(int(all_cells.binary_mask.sum()), 4)

    def test_layer_cleanup_is_ownership_safe(self) -> None:
        owned = SimpleNamespace(name="Input | Raw", metadata={"owner": OWNER})
        registered = SimpleNamespace(name="Temporary", metadata={})
        unrelated_same_name = SimpleNamespace(name="Input | Raw", metadata={})
        unrelated = SimpleNamespace(name="User", metadata={})
        viewer = SimpleNamespace(
            layers=[owned, registered, unrelated_same_name, unrelated]
        )

        remove_owned_layers(viewer, (registered,))

        self.assertEqual(viewer.layers, [unrelated_same_name, unrelated])

    def test_current_frame_is_placed_at_the_mapped_tzyx_index(self) -> None:
        frame = np.arange(12, dtype=np.uint16).reshape(1, 3, 4)

        scene = current_tzyx(frame, time_count=3, time_index=1)

        self.assertEqual(scene.shape, (3, 1, 3, 4))
        np.testing.assert_array_equal(scene[1], frame)
        self.assertFalse(scene[0].any())
        self.assertFalse(scene[2].any())


class Stage3AnalysisSourceTests(unittest.TestCase):
    def test_time_index_maps_to_original_frame_and_loads_only_that_frame(self) -> None:
        shape = (2, 5, 5)
        binary = np.zeros(shape, dtype=bool)
        binary[0, 2, 2] = True
        labels = binary.astype(np.int32) * 9
        production_frame = ProductionFrame(
            13,
            np.ones(shape, dtype=np.uint16),
            np.ones(shape, dtype=np.float32),
            binary,
            labels,
            pd.DataFrame(
                {
                    "cell_id": [9],
                    "centroid_z": [0.0],
                    "centroid_y": [2.0],
                    "centroid_x": [2.0],
                }
            ),
        )

        class FakeReplaySource:
            frames = (4, 13, 22)
            selected_cells = {13: (9,)}
            voxel_size_zyx_um = (1.625, 0.40625, 0.40625)
            crop_slices = (slice(0, 2), slice(0, 5), slice(0, 5))

            def __init__(self):
                self.loaded = []

            def load_frame(self, frame):
                self.loaded.append(frame)
                return production_frame

        replay = FakeReplaySource()
        source = Stage3AnalysisSource(replay)

        selected = source.load_time_index(1)

        self.assertEqual(source.scene_time_to_original_frame, (4, 13, 22))
        self.assertEqual(selected.original_frame_number, 13)
        self.assertEqual(selected.selected_ids, (9,))
        self.assertEqual(replay.loaded, [13])
        np.testing.assert_array_equal(
            selected.production_frame.binary_mask, binary
        )


class PeakDiagnosticApiTests(unittest.TestCase):
    @staticmethod
    def _two_lobe_mask() -> np.ndarray:
        shape = (7, 31, 31)
        grid = np.indices(shape).transpose(1, 2, 3, 0)
        voxel = np.asarray(DEFAULT_CONFIG.voxel_size_zyx_um)
        physical = grid * voxel
        first = np.asarray((3, 12, 15)) * voxel
        second = np.asarray((3, 19, 15)) * voxel
        return (
            (np.linalg.norm(physical - first, axis=-1) <= 2.4)
            | (np.linalg.norm(physical - second, axis=-1) <= 2.4)
        )

    def test_detailed_peak_api_is_array_equal_to_production_api(self) -> None:
        mask = self._two_lobe_mask()
        config = replace(
            DEFAULT_CONFIG,
            sigma_levels_um=DEFAULT_CONFIG.sigma_levels_um[:2],
            h_levels_um=DEFAULT_CONFIG.h_levels_um[:2],
        )

        normal = peaks_module.detect_persistent_distance_peaks(mask, config)
        detailed = peaks_module.detect_persistent_distance_peaks_detailed(
            mask, config
        )

        np.testing.assert_array_equal(normal.raw_distance, detailed.analysis.raw_distance)
        np.testing.assert_array_equal(
            normal.merge_tree_distance, detailed.analysis.merge_tree_distance
        )
        np.testing.assert_array_equal(
            normal.watershed_distance, detailed.analysis.watershed_distance
        )
        self.assertEqual(normal.peaks, detailed.analysis.peaks)

    def test_on_demand_setting_reproduces_detailed_detections(self) -> None:
        mask = self._two_lobe_mask()
        config = replace(
            DEFAULT_CONFIG,
            sigma_levels_um=DEFAULT_CONFIG.sigma_levels_um[:2],
            h_levels_um=DEFAULT_CONFIG.h_levels_um[:2],
        )
        detailed = peaks_module.detect_persistent_distance_peaks_detailed(mask, config)
        sigma, h_value = config.sigma_levels_um[1], config.h_levels_um[0]

        setting = peaks_module.inspect_peak_detection_setting(
            mask, config, sigma, h_value
        )
        records = tuple(
            record
            for record in detailed.detections
            if record.sigma_um == sigma and record.h_um == h_value
        )

        self.assertEqual(
            {record.position_zyx for record in records},
            set(setting.representative_positions_zyx),
        )
        for record in records:
            self.assertEqual(
                record.smoothed_depth_um,
                float(setting.smoothed_distance[record.position_zyx]),
            )

    def test_manual_sequence_matches_canonical_component_wrapper(self) -> None:
        component_mask = self._two_lobe_mask()
        config = replace(
            DEFAULT_CONFIG,
            sigma_levels_um=DEFAULT_CONFIG.sigma_levels_um[:2],
            h_levels_um=DEFAULT_CONFIG.h_levels_um[:2],
        )
        production = component_mask.astype(np.int32)
        resolution = resolve_target_components(
            production, component_mask, (1,), display_padding=0
        )

        run = run_stage3_component(resolution, 1, config)

        np.testing.assert_array_equal(run.final_labels, run.canonical.final_labels)
        self.assertEqual(
            run.collapse_result.effective_peaks, run.canonical.effective_peaks
        )
        self.assertEqual(
            run.marker_positions_zyx, run.canonical.marker_positions_zyx
        )
        self.assertEqual(
            int(run.final_labels.max()), len(run.collapse_result.effective_peaks)
        )

    def test_arbitrary_effective_peak_count_uses_every_peak_as_final_marker(self) -> None:
        shape = (11, 81, 61)
        voxel = np.asarray(DEFAULT_CONFIG.voxel_size_zyx_um)
        physical = np.indices(shape).transpose(1, 2, 3, 0) * voxel
        center = np.asarray((5, 40, 30)) * voxel
        component_mask = np.zeros(shape, dtype=bool)
        for offset in (-7.2, -2.4, 2.4, 7.2):
            lobe_center = center + np.asarray((0.0, offset, 0.0))
            component_mask |= (
                np.linalg.norm(physical - lobe_center, axis=-1) <= 2.8
            )
        resolution = resolve_target_components(
            component_mask.astype(np.int32),
            component_mask,
            (1,),
            display_padding=0,
        )

        run = run_stage3_component(resolution, 1, DEFAULT_CONFIG)

        self.assertGreater(len(run.collapse_result.effective_peaks), 3)
        self.assertEqual(
            int(run.final_labels.max()), len(run.collapse_result.effective_peaks)
        )
        self.assertEqual(
            run.marker_positions_zyx,
            tuple(
                tuple(
                    coordinate - DEFAULT_CONFIG.component_padding_voxels
                    for coordinate in peak.position_zyx
                )
                for peak in run.collapse_result.effective_peaks
            ),
        )


class NotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.path = (
            cls.root
            / "notebooks"
            / "diagnostics"
            / "stage_analysis"
            / "03_instance_segmentation_analysis.ipynb"
        )
        cls.notebook = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell.get("source", ()))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        cls.code = "\n".join(cls.code_cells)
        package = cls.root / "diagnostics" / "stage_analysis"
        cls.widget_code = (package / "napari_widget.py").read_text(
            encoding="utf-8"
        )
        cls.source_code = (package / "source.py").read_text(encoding="utf-8")
        cls.layers_code = (package / "napari_layers.py").read_text(
            encoding="utf-8"
        )
        cls.runner_code = (package / "runner.py").read_text(encoding="utf-8")
        cls.models_code = (package / "models.py").read_text(encoding="utf-8")
        cls.visualizer_code = (
            cls.root
            / "investigations"
            / "stage_03_segmentation"
            / "peak_selection"
            / "visualize_effective_peaks_over_time.py"
        ).read_text(encoding="utf-8")

    def test_notebook_json_and_each_python_cell_compile(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        for index, source in enumerate(self.code_cells):
            compile(source, f"notebook-cell-{index}", "exec")

    def test_notebook_is_a_thin_explicit_3d_launcher(self) -> None:
        self.assertLessEqual(len(self.code_cells), 4)
        self.assertNotIn("\ndef ", "\n" + self.code)
        self.assertNotIn("\nclass ", "\n" + self.code)
        self.assertIn("napari.Viewer(ndisplay=3)", self.code)
        self.assertIn("add_stage3_analysis_widget", self.code)
        self.assertIn("napari.run()", self.code)

    def test_saved_ids_and_complete_stage2_frame_are_automatic(self) -> None:
        self.assertIn(
            "self.replay_source.selected_cells.get(frame_number, ())",
            self.source_code,
        )
        self.assertIn(
            "production_frame = self.replay_source.load_frame(frame_number)",
            self.source_code,
        )
        self.assertIn("production_frame.binary_mask", self.source_code)
        self.assertNotIn("form.addRow(\"Cell ID\"", self.widget_code)
        self.assertNotIn("form.addRow(\"Cell IDs\"", self.widget_code)

    def test_native_time_slider_replaces_custom_frame_controls(self) -> None:
        self.assertNotIn("frame_combo", self.widget_code)
        self.assertNotIn("wheelEvent", self.widget_code)
        self.assertNotIn("wheel_event", self.widget_code)
        self.assertIn("self.original_frame_label", self.widget_code)
        self.assertIn("events.current_step.connect", self.widget_code)
        self.assertIn("self.source.load_time_index", self.widget_code)

    def test_time_and_display_callbacks_do_not_run_stage3(self) -> None:
        time_body = self.widget_code.split(
            "    def _on_dimension_step_changed", 1
        )[1].split("    def _load_time_index", 1)[0]
        load_body = self.widget_code.split("    def _load_time_index", 1)[1].split(
            "    def _populate_components_and_errors", 1
        )[0]
        display_body = self.widget_code.split(
            "    def refresh_input_layers", 1
        )[1].split("    def restore_defaults", 1)[0]
        self.assertNotIn("run_stage3_component", time_body)
        self.assertNotIn("run_stage3_component", load_body)
        self.assertNotIn("run_stage3_component", display_body)

    def test_tzyx_layers_have_anisotropic_scale_and_3d_enforcement(self) -> None:
        self.assertIn(
            '"scale": (1.0, *tuple(float(value) for value in voxel_size))',
            self.layers_code,
        )
        self.assertIn("current_tzyx", self.layers_code)
        self.assertIn("self.viewer.dims.ndisplay = 3", self.widget_code)
        self.assertIn("viewer.dims.ndisplay = 3", self.widget_code)

    def test_all_required_short_layer_names_are_present(self) -> None:
        required = {
            "Input | Raw", "Input | Preprocessed", "Mask | Saved", "Mask | Target",
            "Boundary | Target", "Prod | Labels", "Prod | Boundary", "Prod | Selected",
            "EDT | Raw", "EDT | Merge tree", "EDT | Watershed", "Peaks | Raw",
            "Peaks | Effective", "Pairs | Evidence",
            "Peak scan | Smoothed EDT", "Peak scan | H-maxima", "Peak scan | Peaks",
            "Final | Labels", "Final | Boundary", "Compare | Prod boundary",
            "Compare | Trial boundary",
        }
        self.assertEqual(
            {name for name in required if name not in self.layers_code}, set()
        )

    def test_no_hypothesis_controls_models_tables_or_layers_remain(self) -> None:
        combined = "\n".join(
            (self.widget_code, self.layers_code, self.runner_code, self.models_code)
        )
        forbidden = (
            "Hypothesis",
            "hypothesis",
            "H1",
            "H2",
            "H3",
            "posterior",
            "conditional_probability",
            "odds_vs",
            "selected_peaks",
            "decision_status",
            "combination_prescore",
            "Peaks | Selected",
            "Hyp | Labels",
            "Hyp | Boundary",
            "Hyp | Markers",
        )
        for value in forbidden:
            self.assertNotIn(value, combined)

    def test_all_frame_visualizer_uses_effective_peaks_as_final_markers(self) -> None:
        self.assertIn("CACHE_SCHEMA_VERSION = 3", self.visualizer_code)
        self.assertIn('name="Effective peaks"', self.visualizer_code)
        self.assertIn("out_of_slice_display=False", self.visualizer_code)
        self.assertIn("Saved Stage 4 centroids", self.visualizer_code)
        self.assertNotIn("selected_peak", self.visualizer_code)
        self.assertNotIn("Current Stage 3 markers", self.visualizer_code)


if __name__ == "__main__":
    unittest.main()
