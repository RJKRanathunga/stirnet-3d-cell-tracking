"""Focused non-GUI contracts for the pipeline replay workbench."""

from __future__ import annotations

import tempfile
import inspect
import json
import unittest
from dataclasses import fields
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import zarr

from diagnostics.pipeline_replay import (
    PipelineReplayRunner,
    PipelineReplaySource,
    ReplayMode,
    diagnose_instance_centers,
    match_instance_by_iou,
)
from diagnostics.pipeline_replay.models import ProductionFrame
from diagnostics.pipeline_replay.component_debug import ComponentDebugResult
from diagnostics.pipeline_replay.napari_layers import (
    add_debug_layers,
    remove_pipeline_replay_layers,
)
from src.api import create_binary_mask, preprocess_volume
from src.io import PipelinePaths, save_csv, save_json, save_npy

preprocessing_config = import_module("src.source_instances.preprocessing.config")
masking_config = import_module("src.source_instances.foreground.config")
replay_runner_module = import_module("diagnostics.pipeline_replay.runner")
segmentation_config = import_module("src.source_instances.segmentation.config")


class _SyntheticSource:
    crop_origin_zyx = (1, 2, 2)
    crop_stop_zyx = (4, 7, 7)
    crop_shape_zyx = (3, 5, 5)
    voxel_size_zyx_um = (1.625, 0.40625, 0.40625)

    def __init__(self, raw: np.ndarray):
        zeros = np.zeros(raw.shape, dtype=np.int32)
        self.frame = ProductionFrame(
            0, raw, np.zeros_like(raw, dtype=float), zeros.astype(bool), zeros,
            pd.DataFrame(columns=("cell_id", "centroid_z", "centroid_y", "centroid_x")),
        )

    def load_frame(self, frame: int):
        return self.frame

    def global_to_local(self, coordinate):
        return tuple(float(value) - origin for value, origin in zip(coordinate, self.crop_origin_zyx))

    def local_to_global(self, coordinate):
        return tuple(float(value) + origin for value, origin in zip(coordinate, self.crop_origin_zyx))


class PipelineReplayTests(unittest.TestCase):
    def test_scene_loads_raw_zarr_and_never_masked_scene_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir(); (root / "notebooks").mkdir()
            (root / "pyproject.toml").write_text("", encoding="utf-8")
            paths = PipelinePaths(root)
            sample_id, frame = "sample-a", 1
            raw = np.arange(3 * 4 * 5 * 6, dtype=np.uint16).reshape(3, 4, 5, 6)
            array_path = paths.sample_zarr_array(sample_id)
            array_path.parent.mkdir(parents=True)
            array = zarr.open_array(str(array_path), mode="w", shape=raw.shape, dtype=raw.dtype)
            array[:] = raw
            stage6 = paths.processed_dataset(sample_id)
            volume = raw[frame]
            save_npy(volume.astype(np.float32), stage6 / "preprocessing" / "t001.npy")
            save_npy(volume > 0, stage6 / "masking" / "t001.npy")
            labels = (volume > 0).astype(np.int32)
            save_npy(labels, stage6 / "segmentation" / "t001.npy")
            save_csv(pd.DataFrame({
                "cell_id": [1], "centroid_z": [1.0], "centroid_y": [2.0], "centroid_x": [3.0],
            }), stage6 / "cells" / "t001.csv")
            scene = root / "scene"
            scene.mkdir()
            save_npy(np.full((1, 2, 3, 3), 999, dtype=np.uint16), scene / "masked_raw.npy")
            save_json({
                "sample_id": sample_id,
                "frame_numbers": [frame],
                "selected_cells": {"1": [1]},
                "crop_origin_zyx": [1, 1, 2],
                "crop_stop_zyx": [3, 4, 5],
                "crop_shape_zyx": [2, 3, 3],
                "voxel_size_zyx": [1.625, 0.40625, 0.40625],
                "files": {"masked_images": {"raw": "masked_raw.npy"}},
            }, scene / "scene.json")
            source = PipelineReplaySource.from_scene(scene, paths)
            loaded = source.load_frame(frame)
            np.testing.assert_array_equal(loaded.raw, volume)
            self.assertFalse(np.any(loaded.raw == 999))
            np.testing.assert_array_equal(loaded.raw[source.crop_slices], volume[1:3, 1:4, 2:5])

    def test_scene_coordinate_round_trip(self) -> None:
        source = _SyntheticSource(np.zeros((5, 9, 9), dtype=float))
        source.crop_origin_zyx = (2, 20, 30)
        global_point = (3.5, 24.0, 38.25)
        self.assertEqual(source.local_to_global(source.global_to_local(global_point)), global_point)

    def test_default_preprocessing_and_masking_configs_preserve_results(self) -> None:
        rng = np.random.default_rng(12)
        volume = rng.random((3, 9, 9), dtype=np.float32)
        np.testing.assert_array_equal(
            preprocess_volume(volume),
            preprocess_volume(volume, config=preprocessing_config.DEFAULT_PREPROCESSING_CONFIG),
        )
        processed = preprocess_volume(volume)
        np.testing.assert_array_equal(
            create_binary_mask(processed),
            create_binary_mask(processed, config=masking_config.DEFAULT_MASKING_CONFIG),
        )

    def test_exact_mode_preprocessing_matches_canonical_full_frame(self) -> None:
        raw = np.random.default_rng(3).random((5, 9, 9), dtype=np.float32)
        runner = PipelineReplayRunner(_SyntheticSource(raw))
        runner.reload_baseline(0)
        result = runner.run(1, downstream=False, mode=ReplayMode.EXACT)
        canonical = preprocess_volume(raw)
        np.testing.assert_array_equal(
            result.display_preprocessed,
            canonical[1:4, 2:7, 2:7],
        )

    def test_local_mode_is_explicitly_approximate(self) -> None:
        raw = np.random.default_rng(4).random((5, 9, 9), dtype=np.float32)
        runner = PipelineReplayRunner(_SyntheticSource(raw)); runner.reload_baseline(0)
        result = runner.run(1, downstream=False, mode=ReplayMode.LOCAL, halo_zyx=(1, 1, 1))
        self.assertEqual(result.mode.value, "Local approximation")
        self.assertIn("Local approximation", result.warning or "")

    def test_intersecting_component_uses_complete_work_extent(self) -> None:
        runner = PipelineReplayRunner(_SyntheticSource(np.zeros((5, 9, 9), dtype=float)))
        mask = np.zeros((5, 9, 9), dtype=bool)
        mask[2, 0:8, 4] = True  # intersects ROI y=2:7 but extends well outside it
        runner._mask_work = mask
        runner._display_slices = (slice(1, 4), slice(2, 7), slice(2, 7))
        captured = {}
        def fake_segment(value, *_args, **kwargs):
            captured["mask"] = value.copy()
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                final_labels=value.astype(np.int32),
                markers=np.zeros_like(value, dtype=np.int32),
                component_diagnostics=(),
                component_debug_artifacts=(),
            )
        with patch.object(replay_runner_module.segmentation_module, "segment_instances_detailed", side_effect=fake_segment):
            runner._run_stage_3()
        self.assertEqual(int(captured["mask"].sum()), 8)
        self.assertTrue(captured["mask"][2, 0, 4])
        self.assertTrue(captured["mask"][2, 7, 4])
        self.assertEqual(captured["kwargs"], {"retain_debug_artifacts": True})

    def test_stage3_replay_uses_all_effective_peaks_without_a_count_cap(self) -> None:
        shape = (11, 81, 61)
        voxel = np.asarray(
            segmentation_config.DEFAULT_SEGMENTATION_CONFIG.voxel_size_zyx_um
        )
        physical = np.indices(shape).transpose(1, 2, 3, 0) * voxel
        center = np.asarray((5, 40, 30)) * voxel
        mask = np.zeros(shape, dtype=bool)
        for offset in (-7.2, -2.4, 2.4, 7.2):
            lobe_center = center + np.asarray((0.0, offset, 0.0))
            mask |= np.linalg.norm(physical - lobe_center, axis=-1) <= 2.8

        source = _SyntheticSource(np.zeros(shape, dtype=np.float32))
        source.crop_origin_zyx = (0, 0, 0)
        source.crop_stop_zyx = shape
        source.crop_shape_zyx = shape
        runner = PipelineReplayRunner(source)
        runner.reload_baseline(0)
        runner._raw_work = source.frame.raw
        runner._mask_work = mask
        runner._display_slices = tuple(slice(0, value) for value in shape)
        runner.state.work_crop_global = runner._display_slices

        runner._run_stage_3()
        result = runner.result()
        component = result.component_results[0]

        self.assertGreater(component.effective_peak_count, 3)
        self.assertEqual(
            component.marker_count, component.effective_peak_count
        )
        self.assertEqual(
            component.instance_count, component.effective_peak_count
        )
        self.assertEqual(
            len(component.effective_peak_positions_zyx),
            component.effective_peak_count,
        )
        np.testing.assert_array_equal(
            component.marker_positions_zyx,
            component.effective_peak_positions_zyx,
        )
        self.assertEqual(
            int(result.display_instance_labels.max()),
            component.effective_peak_count,
        )
        np.testing.assert_array_equal(result.display_instance_labels > 0, mask)
        self.assertEqual(component.processing_status, "processed")
        self.assertIsNone(component.error)

        class Viewer:
            def __init__(self):
                self.layers = []

            def _add(self, data, name, **kwargs):
                layer = SimpleNamespace(data=data, name=name, **kwargs)
                self.layers.append(layer)
                return layer

            add_image = _add
            add_labels = _add
            add_points = _add
            add_shapes = _add

        viewer = Viewer()
        add_debug_layers(viewer, source, result)
        names = {layer.name for layer in viewer.layers}
        self.assertIn("Debug | Effective peaks (final markers)", names)
        self.assertIn("Debug | Final labels", names)
        self.assertIn("Debug | Final boundaries", names)
        self.assertIn("Debug | Peak pair evidence", names)
        self.assertTrue(
            {"Debug | Raw EDT", "Debug | Merge-tree EDT", "Debug | Watershed EDT"}
            .issubset(names)
        )
        self.assertFalse(
            any(
                token in name
                for name in names
                for token in ("H1", "H2", "H3", "Selected", "Hypothesis")
            )
        )
        final_marker_layer = next(
            layer
            for layer in viewer.layers
            if layer.name == "Debug | Effective peaks (final markers)"
        )
        self.assertEqual(len(final_marker_layer.data), component.marker_count)

    def test_stage3_replay_model_and_runner_have_no_removed_contracts(self) -> None:
        model_fields = {field.name for field in fields(ComponentDebugResult)}
        runner_source = inspect.getsource(replay_runner_module)
        debug_source = inspect.getsource(
            import_module("diagnostics.pipeline_replay.component_debug")
        )
        forbidden_fields = {
            "hypothesis_evidence",
            "selected_marker_positions_zyx",
            "selected_cell_count",
            "decision_status",
        }
        self.assertTrue(forbidden_fields.isdisjoint(model_fields))
        for value in (
            "include_hypothesis_diagnostics",
            "hypothesis_diagnostics",
            "selected_peaks",
            "posterior",
            "conditional_probability",
            "odds_vs",
        ):
            self.assertNotIn(value, runner_source + debug_source)

    def test_iou_matching_ignores_numeric_label_equality_and_reports_topology(self) -> None:
        production = np.zeros((1, 5, 5), dtype=np.int32)
        trial = np.zeros_like(production)
        production[:, 1:4, 1:4] = 42
        trial[:, 1:4, 1:4] = 7
        match = match_instance_by_iou(production, trial, 42, (1, 1, 1))
        self.assertEqual(match.trial_instance_id, 7)
        self.assertEqual(match.iou, 1.0)
        self.assertEqual(match.status, "clear_overlap")

    def test_outside_centroid_and_disconnected_instance_are_identified(self) -> None:
        labels = np.zeros((1, 9, 9), dtype=np.int32)
        labels[0, 1:8, 1] = 1; labels[0, 1:8, 7] = 1
        labels[0, 1, 1:8] = 1; labels[0, 7, 1:8] = 1  # hollow ring
        labels[0, 1, 3] = 2; labels[0, 7, 3] = 2
        diagnostics = {item.instance_id: item for item in diagnose_instance_centers(labels)}
        self.assertFalse(diagnostics[1].centroid_inside_mask)
        self.assertEqual(diagnostics[2].connected_part_count, 2)

    def test_parameter_invalidation_and_reset_are_stage_scoped(self) -> None:
        runner = PipelineReplayRunner(_SyntheticSource(np.zeros((5, 9, 9), dtype=float)))
        sentinel = object()
        runner.state.dirty_from_stage = 6
        runner.state.preprocessing_trace = sentinel
        runner.state.masking_trace = sentinel
        runner.state.trial_labels = np.ones((1, 1, 1))
        runner.update_masking_config(threshold_offset=0.1)
        self.assertEqual(runner.state.dirty_from_stage, 2)
        self.assertIs(runner.state.preprocessing_trace, sentinel)
        self.assertIsNone(runner.state.masking_trace)
        runner.reset_stage(2)
        self.assertEqual(runner.state.masking_config, masking_config.DEFAULT_MASKING_CONFIG)

    def test_layer_cleanup_only_removes_owned_prefixes(self) -> None:
        unrelated = SimpleNamespace(name="User layer")
        layers = [SimpleNamespace(name="Trial | Binary mask"), unrelated, SimpleNamespace(name="Debug | Effective peaks (final markers)")]
        viewer = SimpleNamespace(layers=layers)
        remove_pipeline_replay_layers(viewer)
        self.assertEqual(viewer.layers, [unrelated])

    def test_gui_module_import_is_optional_without_constructing_qt(self) -> None:
        module = import_module("diagnostics.pipeline_replay.napari_widget")
        self.assertTrue(hasattr(module, "add_pipeline_replay_workbench"))

    def test_replay_workbench_notebook_json_and_code_cells_are_valid(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "diagnostics"
            / "pipeline_replay_workbench.ipynb"
        )
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [
            "".join(cell.get("source", ()))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        for index, source in enumerate(code_cells):
            compile(source, f"pipeline-replay-cell-{index}", "exec")


if __name__ == "__main__":
    unittest.main()
