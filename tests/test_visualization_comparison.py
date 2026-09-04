"""Focused contracts for the Stage 9 before/after comparison."""

from __future__ import annotations

import ast
from dataclasses import replace
from importlib import import_module
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.io import PipelinePaths


comparison_module = import_module("legacy.classical_pipeline.visualization.step05_comparison")
endpoint_module = import_module("legacy.classical_pipeline.visualization.step02_endpoints")
napari_module = import_module("legacy.classical_pipeline.visualization.napari_layers")

Stage9Snapshot = comparison_module.Stage9Snapshot
compare_stage9_snapshots = comparison_module.compare_stage9_snapshots
load_stage9_snapshot = comparison_module.load_stage9_snapshot
prepare_stage9_snapshot = comparison_module.prepare_stage9_snapshot
save_stage9_snapshot = comparison_module.save_stage9_snapshot
EndpointTrackGroups = endpoint_module.EndpointTrackGroups
add_track_group = napari_module.add_track_group


TRACK_COLUMNS = ["track_id", "frame", "z", "y", "x", "cell_id"]
ENDPOINT_COLUMNS = [
    "track_id",
    "frame",
    "z",
    "y",
    "x",
    "cell_id",
    "is_boundary_endpoint",
]


def _event_tables(events, endpoint: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    track_rows = []
    endpoint_rows = []
    for track_id, frame, cell_id in events:
        other_frame = frame - 1 if endpoint == "end" else frame + 1
        observations = (
            [(other_frame, cell_id + 10_000), (frame, cell_id)]
            if endpoint == "end"
            else [(frame, cell_id), (other_frame, cell_id + 10_000)]
        )
        for observation_frame, observation_cell_id in observations:
            track_rows.append(
                {
                    "track_id": track_id,
                    "frame": observation_frame,
                    "z": float(track_id),
                    "y": float(observation_cell_id),
                    "x": float(observation_frame),
                    "cell_id": observation_cell_id,
                }
            )
        endpoint_rows.append(
            {
                "track_id": track_id,
                "frame": frame,
                "z": float(track_id),
                "y": float(cell_id),
                "x": float(frame),
                "cell_id": cell_id,
                "is_boundary_endpoint": False,
            }
        )
    return (
        pd.DataFrame(track_rows, columns=TRACK_COLUMNS),
        pd.DataFrame(endpoint_rows, columns=ENDPOINT_COLUMNS),
    )


def _metadata(**updates) -> dict:
    metadata = {
        "sample_id": "sample-a",
        "boundary_margin_um": 4.0,
        "voxel_size_zyx": [1.625, 0.40625, 0.40625],
        "spatial_shape_zyx": [64, 256, 256],
        "source_stage": "stage_08_track_stitching",
        "stage8_metadata": {},
        "ended_track_count": 0,
        "new_track_count": 0,
        "snapshot_created_utc": "2026-08-07T00:00:00+00:00",
    }
    metadata.update(updates)
    return metadata


def _snapshot(*, ended=(), new=(), metadata=None) -> Stage9Snapshot:
    ended_tracks, ended_endpoints = _event_tables(ended, "end")
    new_tracks, new_endpoints = _event_tables(new, "start")
    return Stage9Snapshot(
        ended_tracks=ended_tracks,
        ended_endpoints=ended_endpoints,
        new_tracks=new_tracks,
        new_endpoints=new_endpoints,
        metadata=dict(metadata or _metadata()),
    )


def _notebook_code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


class Stage9ComparisonTests(unittest.TestCase):
    def test_same_event_with_renumbered_track_id_is_unchanged(self) -> None:
        previous = _snapshot(ended=((10, 5, 101),))
        new = _snapshot(ended=((99, 5, 101),))

        result = compare_stage9_snapshots(previous, new)

        self.assertTrue(result.removed_ended_tracks.empty)
        self.assertTrue(result.added_ended_tracks.empty)
        self.assertEqual(result.summary["removed_ended_tracks"], 0)
        self.assertEqual(result.summary["added_ended_tracks"], 0)

    def test_previous_only_and_new_only_ended_events(self) -> None:
        previous = _snapshot(ended=((10, 5, 101), (11, 7, 303)))
        new = _snapshot(ended=((20, 6, 202), (98, 7, 303)))

        result = compare_stage9_snapshots(previous, new)

        self.assertEqual(set(result.removed_ended_tracks.track_id), {10})
        self.assertEqual(set(result.added_ended_tracks.track_id), {20})
        self.assertEqual(
            set(result.removed_ended_endpoints.cell_id),
            {101},
        )
        self.assertEqual(set(result.added_ended_endpoints.cell_id), {202})

    def test_previous_only_and_new_only_start_events(self) -> None:
        previous = _snapshot(new=((10, 5, 101), (11, 7, 303)))
        new = _snapshot(new=((20, 6, 202), (98, 7, 303)))

        result = compare_stage9_snapshots(previous, new)

        self.assertEqual(set(result.removed_new_tracks.track_id), {10})
        self.assertEqual(set(result.added_new_tracks.track_id), {20})
        self.assertEqual(set(result.removed_new_endpoints.cell_id), {101})
        self.assertEqual(set(result.added_new_endpoints.cell_id), {202})

    def test_short_interior_fragment_is_added_to_both_event_groups(self) -> None:
        previous = _snapshot()
        new_tracks = pd.DataFrame(
            [
                {"track_id": 7, "frame": 2, "z": 5.0, "y": 20.0, "x": 30.0, "cell_id": 20},
                {"track_id": 7, "frame": 3, "z": 5.0, "y": 21.0, "x": 30.0, "cell_id": 21},
            ],
            columns=TRACK_COLUMNS,
        )
        new = Stage9Snapshot(
            ended_tracks=new_tracks.copy(),
            ended_endpoints=pd.DataFrame(
                [{**new_tracks.iloc[-1].to_dict(), "is_boundary_endpoint": False}],
                columns=ENDPOINT_COLUMNS,
            ),
            new_tracks=new_tracks.copy(),
            new_endpoints=pd.DataFrame(
                [{**new_tracks.iloc[0].to_dict(), "is_boundary_endpoint": False}],
                columns=ENDPOINT_COLUMNS,
            ),
            metadata=_metadata(ended_track_count=1, new_track_count=1),
        )

        result = compare_stage9_snapshots(previous, new)

        self.assertEqual(set(result.added_ended_tracks.track_id), {7})
        self.assertEqual(set(result.added_new_tracks.track_id), {7})
        self.assertEqual(len(result.added_ended_tracks), 2)
        self.assertEqual(len(result.added_new_tracks), 2)

    def test_empty_groups_compare_without_special_cases(self) -> None:
        result = compare_stage9_snapshots(_snapshot(), _snapshot())

        self.assertTrue(result.removed_ended_tracks.empty)
        self.assertTrue(result.added_ended_tracks.empty)
        self.assertTrue(result.removed_new_tracks.empty)
        self.assertTrue(result.added_new_tracks.empty)
        self.assertEqual(set(result.summary.values()), {0})

    def test_snapshot_save_load_round_trip(self) -> None:
        ended_tracks, ended_endpoints = _event_tables(((1, 5, 101),), "end")
        new_tracks, new_endpoints = _event_tables(((2, 6, 202),), "start")
        empty_tracks = pd.DataFrame(columns=TRACK_COLUMNS)
        groups = EndpointTrackGroups(
            track_summary=pd.DataFrame(),
            new_track_endpoints=new_endpoints,
            ended_track_endpoints=ended_endpoints,
            new_failure_tracks=new_tracks,
            ended_failure_tracks=ended_tracks,
            boundary_entry_tracks=empty_tracks,
            boundary_exit_tracks=empty_tracks,
        )
        snapshot = prepare_stage9_snapshot(
            groups,
            sample_id="sample-a",
            boundary_margin_um=4.0,
            voxel_size_zyx=(1.625, 0.40625, 0.40625),
            spatial_shape_zyx=(64, 256, 256),
            stage8_metadata={"algorithm": "old"},
        )

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "snapshot"
            saved = save_stage9_snapshot(snapshot, target)
            loaded = load_stage9_snapshot(target)

        self.assertEqual(saved, target)
        pd.testing.assert_frame_equal(
            loaded.ended_tracks, snapshot.ended_tracks, check_dtype=False
        )
        pd.testing.assert_frame_equal(
            loaded.ended_endpoints, snapshot.ended_endpoints, check_dtype=False
        )
        pd.testing.assert_frame_equal(
            loaded.new_tracks, snapshot.new_tracks, check_dtype=False
        )
        pd.testing.assert_frame_equal(
            loaded.new_endpoints, snapshot.new_endpoints, check_dtype=False
        )
        self.assertEqual(loaded.metadata, snapshot.metadata)
        self.assertEqual(loaded.metadata["stage8_metadata"], {"algorithm": "old"})

    def test_overwrite_protection_and_replacement_remove_stale_files(self) -> None:
        snapshot = _snapshot(ended=((1, 5, 101),))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "snapshot"
            save_stage9_snapshot(snapshot, target)
            stale = target / "stale.csv"
            stale.write_text("stale", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "non-empty"):
                save_stage9_snapshot(snapshot, target)

            save_stage9_snapshot(snapshot, target, overwrite=True)
            self.assertFalse(stale.exists())
            self.assertEqual(
                {path.name for path in target.iterdir()},
                {
                    "ended_tracks.csv",
                    "ended_endpoints.csv",
                    "new_tracks.csv",
                    "new_endpoints.csv",
                    "metadata.json",
                },
            )

    def test_compatibility_validation(self) -> None:
        previous = _snapshot()
        cases = {
            "sample ID": {"sample_id": "sample-b"},
            "voxel sizes": {"voxel_size_zyx": [2.0, 1.0, 1.0]},
            "spatial shapes": {"spatial_shape_zyx": [32, 128, 128]},
            "boundary margin": {"boundary_margin_um": 5.0},
        }
        for message, update in cases.items():
            with self.subTest(message=message):
                new = _snapshot(metadata=_metadata(**update))
                with self.assertRaisesRegex(ValueError, message):
                    compare_stage9_snapshots(previous, new)

    def test_duplicate_event_key_is_rejected(self) -> None:
        previous = _snapshot(ended=((1, 5, 101),))
        duplicate = pd.concat(
            [previous.ended_endpoints, previous.ended_endpoints],
            ignore_index=True,
        )
        previous = replace(previous, ended_endpoints=duplicate)

        with self.assertRaisesRegex(ValueError, r"\(frame, cell_id\).*unique"):
            compare_stage9_snapshots(previous, _snapshot())

    def test_negative_endpoint_cell_id_is_rejected(self) -> None:
        previous = _snapshot(ended=((1, 5, -1),))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            compare_stage9_snapshots(previous, _snapshot())

    def test_shared_napari_helper_preserves_stage9_styling(self) -> None:
        class Layer:
            visible = True

        class Viewer:
            def __init__(self):
                self.calls = []

            def add_tracks(self, data, **kwargs):
                self.calls.append(("tracks", data, kwargs))
                return Layer()

            def add_points(self, data, **kwargs):
                self.calls.append(("points", data, kwargs))
                return Layer()

        frame, _ = _event_tables(((3, 5, 101),), "end")
        viewer = Viewer()
        scale = (1.0, 1.625, 0.40625, 0.40625)
        add_track_group(
            viewer,
            frame,
            track_name="Ended Tracks",
            point_name="Ended Centroids",
            color="red",
            scale=scale,
            visible=False,
        )

        self.assertEqual([call[0] for call in viewer.calls], ["tracks", "points"])
        self.assertEqual(viewer.calls[0][2]["tail_length"], 20)
        self.assertEqual(viewer.calls[0][2]["scale"], scale)
        self.assertEqual(viewer.calls[1][2]["size"], 4)
        self.assertEqual(viewer.calls[1][2]["face_color"], "red")
        self.assertEqual(viewer.calls[1][2]["scale"], scale)
        self.assertEqual(
            set(viewer.calls[1][2]["properties"]),
            {"track_id", "cell_id"},
        )

    def test_comparison_notebook_layer_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = _notebook_code(
            root / "notebooks" / "comparisons" / "09_visualization_comparison.ipynb"
        )
        for layer_name in (
            "Raw Volume",
            "Preprocessed Volume",
            "Binary Mask",
            "Instance Labels",
            "Previous | Ended Tracks",
            "Previous | Ended Centroids",
            "Previous | New Tracks",
            "Previous | New Centroids",
            "New | Ended Tracks",
            "New | Ended Centroids",
            "New | New Tracks",
            "New | New Centroids",
            "Removed | Ended Tracks",
            "Removed | Ended Centroids",
            "Removed | New Tracks",
            "Removed | New Centroids",
            "Added | Ended Tracks",
            "Added | Ended Centroids",
            "Added | New Tracks",
            "Added | New Centroids",
        ):
            self.assertIn(layer_name, source)
        self.assertIn(
            '(previous_snapshot.ended_tracks, "Previous | Ended Tracks", '
            '"Previous | Ended Centroids", "red", False)',
            source,
        )
        self.assertIn(
            '(new_snapshot.new_tracks, "New | New Tracks", '
            '"New | New Centroids", "lime", False)',
            source,
        )
        self.assertIn(
            '(comparison.removed_ended_tracks, "Removed | Ended Tracks", '
            '"Removed | Ended Centroids", "red", True)',
            source,
        )
        self.assertIn(
            '(comparison.added_new_tracks, "Added | New Tracks", '
            '"Added | New Centroids", "lime", True)',
            source,
        )

        tree = ast.parse(source)
        spatial_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"add_image", "add_labels"}
                )
                or (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "add_track_group"
                )
            )
        ]
        self.assertTrue(spatial_calls)
        for call in spatial_calls:
            self.assertIn("scale", {keyword.arg for keyword in call.keywords})
        self.assertIn("spatial_shape_zyx=raw.shape[-3:]", source)

    def test_existing_stage9_notebook_keeps_original_layers_and_scale(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = _notebook_code(root / "notebooks" / "09_visualization.ipynb")
        for layer_name in (
            "Raw Volume",
            "Preprocessed Volume",
            "Binary Mask",
            "Instance Labels",
            "Tracks - all",
            "Centroids - all",
            "Ended Tracks",
            "Ended Centroids",
            "New Tracks",
            "New Centroids",
            "Boundary Entry Tracks",
            "Boundary Entry Centroids",
            "Boundary Exit Tracks",
            "Boundary Exit Centroids",
        ):
            self.assertIn(layer_name, source)

        tree = ast.parse(source)
        local_helpers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "add_track_group"
        ]
        self.assertFalse(local_helpers)
        self.assertIn("legacy.classical_pipeline.visualization.napari_layers", source)
        spatial_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr
                    in {"add_image", "add_labels", "add_tracks", "add_points"}
                )
                or (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "add_track_group"
                )
            )
        ]
        self.assertTrue(spatial_calls)
        for call in spatial_calls:
            self.assertIn("scale", {keyword.arg for keyword in call.keywords})

    def test_stage9_comparison_path_is_outside_processed_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            self.assertEqual(
                paths.stage9_comparisons,
                Path(directory) / "data" / "comparisons" / "stage_09",
            )


if __name__ == "__main__":
    unittest.main()
