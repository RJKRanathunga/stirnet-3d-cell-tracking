"""Synthetic production regressions for Stage 10 cell-lineage detection."""

from __future__ import annotations

import tempfile
import unittest
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

from src.api import run_cell_lineage
from src.io import PipelinePaths, load_stage10_outputs, save_lineage_result


lineage_module = import_module("legacy.classical_pipeline.lineage.step06_pipeline")
config_module = import_module("legacy.classical_pipeline.lineage.step01_config")
CellLineageConfig = config_module.CellLineageConfig


class SyntheticLineageData:
    """Build aligned cell tables, instance labels, raw frames, and track rows."""

    shape = (20, 48, 48)

    def __init__(self, root: Path, frame_count: int) -> None:
        self.root = root
        self.frame_count = frame_count
        self.labels = [np.zeros(self.shape, dtype=np.int32) for _ in range(frame_count)]
        self.raw = np.full((frame_count, *self.shape), 10.0, dtype=np.float32)
        self.detections: list[list[dict[str, object]]] = [[] for _ in range(frame_count)]
        self.track_rows: list[dict[str, object]] = []

    def add(
        self,
        track_id: int,
        frame: int,
        center: tuple[float, float, float],
        volume: float,
        *,
        mask_voxels: int | None = None,
        boundary: bool = False,
        virtual: bool = False,
        merge_event_id: int | None = None,
        raw_intensity: float = 20.0,
    ) -> None:
        cell = len(self.detections[frame])
        cell_id = cell + 1
        count = int(round(volume)) if mask_voxels is None else int(mask_voxels)
        grid = np.indices(self.shape).reshape(3, -1).T
        available = self.labels[frame].reshape(-1) == 0
        delta = grid.astype(float) - np.asarray(center, dtype=float)
        distance = (
            (delta[:, 0] * 1.625) ** 2
            + (delta[:, 1] * 0.40625) ** 2
            + (delta[:, 2] * 0.40625) ** 2
        )
        ordered = np.flatnonzero(available)[np.argsort(distance[available], kind="mergesort")]
        chosen = ordered[:count]
        if len(chosen) != count:
            raise ValueError("Synthetic frame has insufficient free voxels")
        self.labels[frame].reshape(-1)[chosen] = cell_id
        self.raw[frame].reshape(-1)[chosen] = raw_intensity
        coordinates = grid[chosen]
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0) + 1
        radius = float((3 * max(count, 1) / (4 * np.pi)) ** (1 / 3))
        detection = {
            "cell_id": cell_id,
            "volume_voxels": float(count),
            "intensity_sum": float(count * raw_intensity),
            "intensity_mean": float(raw_intensity),
            "intensity_std": 0.0,
            "equivalent_radius": radius,
            "axis_major": radius * 2.2,
            "axis_middle": radius * 2.0,
            "axis_minor": radius * 1.8,
            "elongation": 1.2,
            "flatness": 1.1,
            "anisotropy": 1.3,
            "solidity": 0.9,
            "compactness": 0.8,
            "bbox_depth": int(maximum[0] - minimum[0]),
            "bbox_height": int(maximum[1] - minimum[1]),
            "bbox_width": int(maximum[2] - minimum[2]),
            "z_min": int(minimum[0]), "y_min": int(minimum[1]), "x_min": int(minimum[2]),
            "z_max": int(maximum[0]), "y_max": int(maximum[1]), "x_max": int(maximum[2]),
            "touches_boundary": bool(boundary),
            "boundary_faces": "z_min" if boundary else "",
            "distance_to_boundary_um": 0.0 if boundary else 8.0,
        }
        self.detections[frame].append(detection)
        self.track_rows.append({
            "track_id": track_id,
            "frame": frame,
            "cell": cell,
            "cell_id": cell_id,
            "z": float(center[0]), "y": float(center[1]), "x": float(center[2]),
            "volume": float(volume),
            "is_virtual_merge": bool(virtual),
            "merge_event_id": merge_event_id if merge_event_id is not None else pd.NA,
            "merge_role": "A" if virtual else pd.NA,
        })

    def finish(self) -> tuple[pd.DataFrame, list[pd.DataFrame], np.ndarray, list[Path]]:
        segmentation_files = []
        for frame, labels in enumerate(self.labels):
            path = self.root / f"t{frame:03d}.npy"
            np.save(path, labels)
            segmentation_files.append(path)
        time_frames = [pd.DataFrame(rows) for rows in self.detections]
        return pd.DataFrame(self.track_rows), time_frames, self.raw, segmentation_files


def add_clean_division(
    data: SyntheticLineageData,
    *,
    parent: int = 10,
    child_a: int = 20,
    child_b: int = 30,
    parent_frames: tuple[int, ...] = (0, 1, 2),
    child_frames: tuple[int, ...] = (3, 4, 5),
    parent_volumes: tuple[float, ...] | None = None,
    child_volume: float = 50.0,
) -> None:
    parent_volumes = parent_volumes or tuple(100.0 for _ in parent_frames)
    for frame, volume in zip(parent_frames, parent_volumes):
        data.add(parent, frame, (10, 24, 24), volume, raw_intensity=24 if frame == parent_frames[-1] else 20)
    for offset, frame in enumerate(child_frames):
        data.add(child_a, frame, (10, 24, 21 - offset), child_volume,
                 raw_intensity=20 + 3 * offset)
        data.add(child_b, frame, (10, 24, 27 + offset), child_volume,
                 raw_intensity=20 + 3 * offset)


class CellLineageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_data(self, data: SyntheticLineageData, **kwargs):
        tracks, frames, raw, segmentations = data.finish()
        result = run_cell_lineage(
            tracks, frames, raw, segmentations, sample_id="synthetic", **kwargs
        )
        return result, tracks, frames, raw, segmentations

    def test_clean_confirmed_division(self) -> None:
        data = SyntheticLineageData(self.root, 6)
        add_clean_division(data)
        result, tracks, *_ = self.run_data(data)
        self.assertEqual(result.division_events["decision"].tolist(), ["confirmed"])
        self.assertEqual(len(result.lineage_edges), 2)
        self.assertEqual(set(result.protected_tracks["track_id"]), {10, 20, 30})
        lineage = result.track_lineage.set_index("track_id")
        self.assertEqual(lineage.loc[20, "generation"], lineage.loc[10, "generation"] + 1)
        self.assertEqual(set(lineage.index), set(tracks["track_id"]))
        self.assertFalse(result.metadata["track_ids_rewritten"])

    def test_continuation_plus_unrelated_birth_is_rejected(self) -> None:
        data = SyntheticLineageData(self.root, 6)
        for frame in (0, 1, 2):
            data.add(10, frame, (10, 24, 24), 200)
        for frame in (3, 4, 5):
            data.add(20, frame, (10, 24, 24), 180)
            data.add(30, frame, (10, 24, 29), 20)
        result, *_ = self.run_data(data)
        candidate = result.division_candidates.iloc[0]
        self.assertGreater(candidate["best_continuation_score"], candidate["division_score"])
        self.assertEqual(candidate["rejection_reason"], "continuation_hypothesis_stronger")
        self.assertTrue(result.lineage_edges.empty)

    def test_tiny_fragment_is_rejected(self) -> None:
        data = SyntheticLineageData(self.root, 6)
        for frame in (0, 1, 2):
            data.add(10, frame, (10, 24, 24), 100)
        for frame in (3, 4, 5):
            data.add(20, frame, (10, 24, 21), 97)
            data.add(30, frame, (10, 24, 27), 3, mask_voxels=3)
        result, *_ = self.run_data(data)
        self.assertEqual(result.division_candidates.iloc[0]["rejection_reason"], "tiny_child_fragment")
        self.assertTrue(result.division_events.empty)
        self.assertTrue(result.lineage_edges.empty)

    def test_combined_volume_mismatch_is_rejected_early(self) -> None:
        data = SyntheticLineageData(self.root, 6)
        for frame in (0, 1, 2):
            data.add(10, frame, (10, 24, 24), 100)
        for frame in (3, 4, 5):
            data.add(20, frame, (10, 24, 21), 80)
            data.add(30, frame, (10, 24, 27), 80)
        result, *_ = self.run_data(data)
        candidate = result.division_candidates.iloc[0]
        self.assertEqual(candidate["rejection_reason"], "combined_volume_mismatch")
        self.assertFalse(candidate["birth_masks_available"])

    def test_boundary_parent_is_excluded(self) -> None:
        data = SyntheticLineageData(self.root, 6)
        for frame in (0, 1):
            data.add(10, frame, (10, 24, 24), 100)
        data.add(10, 2, (10, 24, 24), 100, boundary=True)
        for frame in (3, 4, 5):
            data.add(20, frame, (10, 24, 21), 50)
            data.add(30, frame, (10, 24, 27), 50)
        result, *_ = self.run_data(data)
        self.assertTrue(result.division_candidates.empty)
        self.assertEqual(result.metadata["boundary_rejection_count"], 1)

    def test_child_disappears_immediately(self) -> None:
        data = SyntheticLineageData(self.root, 7)
        for frame in (0, 1, 2):
            data.add(10, frame, (10, 24, 24), 100)
        for frame in (3, 4, 5, 6):
            data.add(20, frame, (10, 24, 21 - (frame - 3)), 50)
        data.add(30, 3, (10, 24, 27), 50)
        result, *_ = self.run_data(data)
        candidate = result.division_candidates.iloc[0]
        self.assertFalse(candidate["both_children_persist"])
        self.assertEqual(candidate["rejection_reason"], "insufficient_persistence")
        self.assertNotEqual(candidate["decision"], "confirmed")

    def test_truncated_final_window_can_be_probable_only(self) -> None:
        data = SyntheticLineageData(self.root, 4)
        add_clean_division(data, child_frames=(3,))
        result, *_ = self.run_data(data)
        candidate = result.division_candidates.iloc[0]
        self.assertTrue(candidate["future_window_truncated"])
        self.assertEqual(candidate["decision"], "probable")
        self.assertTrue(result.lineage_edges.empty)
        self.assertTrue(result.protected_tracks.empty)

    def test_conflict_resolution_is_deterministic(self) -> None:
        data = SyntheticLineageData(self.root, 6)
        for frame in (0, 1, 2):
            data.add(10, frame, (10, 24, 24), 100)
        for frame in (3, 4, 5):
            data.add(20, frame, (10, 24, 21 - (frame - 3)), 50)
            data.add(30, frame, (10, 24, 27 + (frame - 3)), 50)
            data.add(40, frame, (10, 27, 24), 50)
        tracks, frames, raw, segmentations = data.finish()
        permissive = CellLineageConfig(
            confirmed_minimum_score=0.0, probable_minimum_score=0.0,
            confirmed_minimum_margin=-1.0, probable_minimum_margin=-1.0,
        )
        first = run_cell_lineage(tracks, frames, raw, segmentations, config=permissive)
        second = run_cell_lineage(
            tracks.sample(frac=1, random_state=7), frames, raw, segmentations,
            config=permissive,
        )
        pd.testing.assert_frame_equal(first.division_candidates, second.division_candidates)
        self.assertEqual((first.division_candidates["decision"] == "confirmed").sum(), 1)
        self.assertGreaterEqual(
            (first.division_candidates["decision"] == "rejected_conflict").sum(), 1
        )

    def test_earlier_merge_history_remains_eligible(self) -> None:
        data = SyntheticLineageData(self.root, 6)
        add_clean_division(data)
        tracks, frames, raw, segmentations = data.finish()
        mask = (tracks["track_id"] == 10) & (tracks["frame"] == 1)
        tracks.loc[mask, "merge_event_id"] = 7
        result = run_cell_lineage(tracks, frames, raw, segmentations)
        self.assertEqual(result.division_events.iloc[0]["decision"], "confirmed")

    def test_virtual_endpoint_is_excluded(self) -> None:
        data = SyntheticLineageData(self.root, 6)
        for frame in (0, 1):
            data.add(10, frame, (10, 24, 24), 100)
        data.add(10, 2, (10, 24, 24), 100, virtual=True, merge_event_id=1)
        for frame in (3, 4, 5):
            data.add(20, frame, (10, 24, 21), 50)
            data.add(30, frame, (10, 24, 27), 50)
        result, *_ = self.run_data(data)
        self.assertTrue(result.division_events.empty)
        self.assertEqual(result.metadata["virtual_observation_rejection_count"], 1)

    def test_missing_optional_intensity_does_not_stop_geometry(self) -> None:
        data = SyntheticLineageData(self.root, 6)
        add_clean_division(data)
        tracks, frames, raw, segmentations = data.finish()

        class MissingHistoryRaw:
            def __getitem__(self, frame):
                if int(frame) < 2:
                    raise OSError("historical raw frame unavailable")
                return raw[int(frame)]

        result = run_cell_lineage(tracks, frames, MissingHistoryRaw(), segmentations)
        self.assertEqual(len(result.division_candidates), 1)
        self.assertTrue(np.isnan(
            result.division_candidates.iloc[0]["parent_final_integrated_intensity_ratio"]
        ))
        self.assertGreaterEqual(result.metadata["missing_optional_intensity_count"], 1)

    def test_empty_output_schemas(self) -> None:
        data = SyntheticLineageData(self.root, 2)
        data.add(1, 0, (10, 24, 24), 100)
        data.add(1, 1, (10, 24, 24), 100)
        result, *_ = self.run_data(data)
        self.assertEqual(result.division_candidates.columns.tolist(), list(config_module.DIVISION_CANDIDATE_COLUMNS))
        self.assertEqual(result.division_events.columns.tolist(), list(config_module.DIVISION_EVENT_COLUMNS))
        self.assertEqual(result.lineage_edges.columns.tolist(), list(config_module.LINEAGE_EDGE_COLUMNS))
        self.assertEqual(result.track_lineage.columns.tolist(), list(config_module.TRACK_LINEAGE_COLUMNS))
        self.assertEqual(result.protected_tracks.columns.tolist(), list(config_module.PROTECTED_TRACK_COLUMNS))

    def test_diagnostic_equivalence(self) -> None:
        data = SyntheticLineageData(self.root, 6)
        add_clean_division(data)
        tracks, frames, raw, segmentations = data.finish()
        normal = run_cell_lineage(tracks, frames, raw, segmentations)
        diagnosed, trace = run_cell_lineage(
            tracks, frames, raw, segmentations, return_diagnostics=True
        )
        for name in (
            "division_candidates", "division_events", "lineage_edges",
            "track_lineage", "protected_tracks",
        ):
            pd.testing.assert_frame_equal(getattr(normal, name), getattr(diagnosed, name))
        self.assertEqual(normal.metadata, diagnosed.metadata)
        self.assertEqual(trace.stage_name, "10_cell_lineage")
        self.assertEqual(len(trace.decisions), len(normal.division_candidates))

    def test_save_load_round_trip(self) -> None:
        data = SyntheticLineageData(self.root / "inputs", 6)
        data.root.mkdir()
        add_clean_division(data)
        result, *_ = self.run_data(data)
        paths = PipelinePaths(self.root)
        save_lineage_result(result, paths.stage10_lineage)
        loaded = load_stage10_outputs(paths=paths)
        for name in (
            "division_candidates", "division_events", "lineage_edges",
            "track_lineage", "protected_tracks",
        ):
            pd.testing.assert_frame_equal(
                getattr(result, name), getattr(loaded, name), check_dtype=False
            )
        self.assertEqual(result.metadata, loaded.metadata)

    def test_lineage_roots_and_generations(self) -> None:
        data = SyntheticLineageData(self.root, 9)
        add_clean_division(data, child_frames=(3, 4, 5), parent_volumes=(100, 100, 100))
        # Child 20 grows before its own later division.
        for row in data.track_rows:
            if row["track_id"] == 20 and row["frame"] in (4, 5):
                row["volume"] = 100.0
        # The masks remain 50 voxels at frames 4/5; only birth masks are structural.
        for offset, frame in enumerate((6, 7, 8)):
            data.add(40, frame, (10, 24, 18 - offset), 50)
            data.add(50, frame, (10, 24, 24 + offset), 50)
        result, *_ = self.run_data(data)
        confirmed = result.division_events.loc[result.division_events["decision"] == "confirmed"]
        self.assertEqual(len(confirmed), 2)
        lineage = result.track_lineage.set_index("track_id")
        self.assertEqual(lineage.loc[20, "root_track_id"], 10)
        self.assertEqual(lineage.loc[40, "root_track_id"], 10)
        self.assertEqual(lineage.loc[40, "generation"], 2)
        self.assertEqual(lineage.loc[50, "generation"], 2)

    def test_duplicate_track_frame_observations_are_rejected(self) -> None:
        data = SyntheticLineageData(self.root, 6)
        add_clean_division(data)
        tracks, frames, raw, segmentations = data.finish()
        tracks = pd.concat([tracks, tracks.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate.*track_id, frame"):
            run_cell_lineage(tracks, frames, raw, segmentations)


if __name__ == "__main__":
    unittest.main()
