"""Focused synthetic production tests for Stage 11 track reconciliation."""

from __future__ import annotations

import tempfile
import unittest
from importlib import import_module
from pathlib import Path

import pandas as pd

from src.api import run_track_reconciliation
from src.diagnostics import StageTrace
from src.io import (
    PipelinePaths,
    load_stage11_outputs,
    save_track_reconciliation_result,
)


config_module = import_module("src.11_track_reconciliation.step01_config")
lineage_schemas = import_module("src.10_cell_lineage.step01_config")
TrackReconciliationConfig = config_module.TrackReconciliationConfig


class SyntheticTracks:
    def __init__(self, frame_count: int = 8) -> None:
        self.frame_count = frame_count
        self._detections: list[list[dict[str, object]]] = [
            [] for _ in range(frame_count)
        ]
        self._tracks: list[dict[str, object]] = []
        self._next_cell_id = 1

    def add_track(
        self,
        track_id: int,
        positions: dict[int, tuple[float, float, float]],
        *,
        volume: float | dict[int, float] = 100.0,
        shape: float | dict[int, float] = 2.0,
        boundary_frames: set[int] | None = None,
        virtual_frames: set[int] | None = None,
    ) -> None:
        boundary_frames = boundary_frames or set()
        virtual_frames = virtual_frames or set()
        for frame, (z, y, x) in sorted(positions.items()):
            cell = len(self._detections[frame])
            cell_id = self._next_cell_id
            self._next_cell_id += 1
            row_volume = volume[frame] if isinstance(volume, dict) else volume
            row_shape = shape[frame] if isinstance(shape, dict) else shape
            boundary = frame in boundary_frames
            self._detections[frame].append({
                "cell_id": cell_id,
                "volume_voxels": float(row_volume),
                "centroid_z": float(z),
                "centroid_y": float(y),
                "centroid_x": float(x),
                "equivalent_radius": float(row_shape),
                "axis_major": float(row_shape) * 1.2,
                "axis_middle": float(row_shape),
                "axis_minor": float(row_shape) * 0.8,
                "intensity_mean": 10.0,
                "intensity_median": 9.5,
                "intensity_std": 1.0,
                "intensity_sum": 1000.0,
                "touches_boundary": boundary,
                "boundary_faces": "z_min" if boundary else "",
                "distance_to_boundary_um": 0.0 if boundary else 50.0,
            })
            self._tracks.append({
                "track_id": track_id,
                "frame": frame,
                "cell": cell,
                "cell_id": cell_id,
                "z": float(z),
                "y": float(y),
                "x": float(x),
                "volume": float(row_volume),
                "is_virtual_merge": frame in virtual_frames,
            })

    @property
    def tracks(self) -> pd.DataFrame:
        columns = (
            "track_id", "frame", "cell", "cell_id", "z", "y", "x",
            "volume", "is_virtual_merge",
        )
        return pd.DataFrame(self._tracks, columns=columns)

    @property
    def frames(self) -> list[pd.DataFrame]:
        columns = (
            "cell_id", "volume_voxels", "centroid_z", "centroid_y",
            "centroid_x", "equivalent_radius", "axis_major", "axis_middle",
            "axis_minor", "intensity_mean", "intensity_median",
            "intensity_std", "intensity_sum", "touches_boundary",
            "boundary_faces", "distance_to_boundary_um",
        )
        return [pd.DataFrame(rows, columns=columns) for rows in self._detections]


def simple_break(*, frame_count: int = 6, target_frame: int = 2) -> SyntheticTracks:
    data = SyntheticTracks(frame_count)
    data.add_track(1, {0: (10, 10, 10), 1: (10, 11, 10)})
    data.add_track(2, {
        target_frame: (10, 10 + target_frame, 10),
        target_frame + 1: (10, 11 + target_frame, 10),
    })
    return data


def division_events(decision: str = "confirmed") -> pd.DataFrame:
    row = {
        "division_event_id": 0,
        "sample_id": "sample",
        "parent_track_id": 10,
        "parent_end_frame": 1,
        "child_birth_frame": 2,
        "child_track_a": 20,
        "child_track_b": 30,
        "decision": decision,
        "confidence": 0.9,
        "division_score": 0.9,
        "best_continuation_score": 0.2,
        "division_margin": 0.7,
        "combined_volume_ratio": 1.0,
        "weighted_centroid_error_um": 1.0,
        "birth_separation_um": 3.0,
        "future_window_truncated": False,
    }
    return pd.DataFrame([row], columns=lineage_schemas.DIVISION_EVENT_COLUMNS)


def lineage_edges() -> pd.DataFrame:
    return pd.DataFrame([
        {"division_event_id": 0, "event_frame": 2, "parent_track_id": 10,
         "child_track_id": 20, "relation": "division", "confidence": 0.9},
        {"division_event_id": 0, "event_frame": 2, "parent_track_id": 10,
         "child_track_id": 30, "relation": "division", "confidence": 0.9},
    ], columns=lineage_schemas.LINEAGE_EDGE_COLUMNS)


def protected_tracks() -> pd.DataFrame:
    return pd.DataFrame([
        {"division_event_id": 0, "track_id": 10, "role": "parent",
         "protected_reason": "confirmed_division_lineage"},
        {"division_event_id": 0, "track_id": 20, "role": "child_a",
         "protected_reason": "confirmed_division_lineage"},
        {"division_event_id": 0, "track_id": 30, "role": "child_b",
         "protected_reason": "confirmed_division_lineage"},
    ], columns=lineage_schemas.PROTECTED_TRACK_COLUMNS)


class TrackReconciliationTests(unittest.TestCase):
    def run_stage(self, data: SyntheticTracks, policy: str = "submission", **kwargs):
        return run_track_reconciliation(
            data.tracks, data.frames,
            config=TrackReconciliationConfig(policy=policy),
            sample_id="sample", **kwargs,
        )

    def test_01_clear_one_frame_break_is_conservative(self) -> None:
        result = self.run_stage(simple_break(), "conservative")
        self.assertEqual(result.track_id_remap["original_segment_track_id"].tolist(), [2])
        self.assertEqual(set(result.tracks["track_id"]), {1})

    def test_02_multiframe_gap_inside_limit(self) -> None:
        result = self.run_stage(simple_break(target_frame=4, frame_count=8))
        self.assertEqual(result.track_id_remap.iloc[0]["gap_frames"], 3)

    def test_03_large_feature_change_does_not_override_anchors(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 11, 10)}, volume=100, shape=2)
        data.add_track(2, {2: (10, 12, 10), 3: (10, 13, 10)}, volume=600, shape=6)
        data.add_track(9, {0: (10, 20, 10), 1: (10, 21, 10), 2: (10, 22, 10), 3: (10, 23, 10), 4: (10, 24, 10), 5: (10, 25, 10)})
        result = self.run_stage(data, "conservative")
        self.assertFalse(result.track_id_remap.empty)

    def test_04_forward_backward_prediction_selects_target(self) -> None:
        data = SyntheticTracks(7)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 12, 10)})
        data.add_track(2, {2: (10, 14, 10), 3: (10, 16, 10), 4: (10, 18, 10)})
        data.add_track(3, {2: (10, 12, 10), 3: (10, 10, 10), 4: (10, 8, 10)})
        result = self.run_stage(data)
        decision = result.track_id_remap.loc[result.track_id_remap["predecessor_track_id"] == 1]
        self.assertEqual(int(decision.iloc[0]["original_segment_track_id"]), 2)

    def test_05_anchor_motion_selects_local_target(self) -> None:
        data = SyntheticTracks(7)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 10, 10)})
        data.add_track(2, {2: (10, 30, 10), 3: (10, 31, 10), 4: (10, 32, 10)})
        data.add_track(3, {2: (10, 10, 10), 3: (10, 10, 10), 4: (10, 10, 10)})
        data.add_track(9, {0: (10, 15, 10), 1: (10, 15, 10), 2: (10, 35, 10), 3: (10, 36, 10), 4: (10, 37, 10), 5: (10, 38, 10), 6: (10, 39, 10)})
        data.add_track(8, {0: (10, 5, 10), 1: (10, 5, 10), 2: (10, 25, 10), 3: (10, 26, 10), 4: (10, 27, 10), 5: (10, 28, 10), 6: (10, 29, 10)})
        result = self.run_stage(data)
        selected = result.track_id_remap.loc[result.track_id_remap["predecessor_track_id"] == 1]
        self.assertEqual(int(selected.iloc[0]["original_segment_track_id"]), 2)

    def test_06_conservative_leaves_symmetric_ambiguity(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 10, 10)})
        data.add_track(2, {2: (10, 11, 10), 3: (10, 11, 10)})
        data.add_track(3, {2: (10, 9, 10), 3: (10, 9, 10)})
        result = self.run_stage(data, "conservative")
        self.assertTrue(result.track_id_remap.empty)
        self.assertIn("unresolved_conservative_threshold", set(result.unresolved_endings["reason"]))

    def test_07_submission_forces_best_ambiguous_target(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 10, 10)})
        data.add_track(2, {2: (10, 11, 10), 3: (10, 11, 10)})
        data.add_track(3, {2: (10, 9, 10), 3: (10, 9, 10)})
        result = self.run_stage(data)
        self.assertEqual(result.track_id_remap.iloc[0]["decision"], "forced_best_candidate")
        self.assertTrue(bool(result.track_id_remap.iloc[0]["forced"]))

    def test_08_diagnostic_proposes_without_rewrite(self) -> None:
        result = self.run_stage(simple_break(), "diagnostic")
        self.assertTrue(result.track_id_remap.empty)
        self.assertIn("diagnostic_proposal", set(result.continuation_decisions["decision"]))
        pd.testing.assert_frame_equal(
            result.tracks.sort_values(["track_id", "frame"]).reset_index(drop=True),
            simple_break().tracks.sort_values(["track_id", "frame"]).reset_index(drop=True),
        )

    def test_09_no_new_target_remains_unresolved(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 11, 10)})
        result = self.run_stage(data)
        self.assertTrue(result.track_id_remap.empty)
        self.assertIn("unresolved_no_candidate", set(result.unresolved_endings["reason"]))

    def test_10_outside_physical_radius_never_forced(self) -> None:
        data = simple_break()
        data._tracks = [row for row in data._tracks if row["track_id"] != 2]
        data._detections[2] = []
        data._detections[3] = []
        data.add_track(2, {2: (10, 100, 10), 3: (10, 101, 10)})
        result = self.run_stage(data)
        self.assertTrue(result.track_id_remap.empty)

    def test_11_outside_temporal_window_never_forced(self) -> None:
        result = self.run_stage(simple_break(target_frame=6, frame_count=9))
        self.assertTrue(result.track_id_remap.empty)

    def test_12_boundary_source_is_excluded(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 11, 10)}, boundary_frames={1})
        data.add_track(2, {2: (10, 12, 10), 3: (10, 13, 10)})
        result = self.run_stage(data)
        endpoint = result.endpoint_classifications.set_index("track_id").loc[1]
        self.assertEqual(endpoint["source_exclusion_reason"], "excluded_boundary_exit")

    def test_13_boundary_target_is_excluded(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 11, 10)})
        data.add_track(2, {2: (10, 12, 10), 3: (10, 13, 10)}, boundary_frames={2})
        result = self.run_stage(data)
        self.assertTrue(result.track_id_remap.empty)

    def test_14_virtual_merge_endpoint_is_excluded(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 11, 10)}, virtual_frames={1})
        data.add_track(2, {2: (10, 12, 10), 3: (10, 13, 10)})
        result = self.run_stage(data)
        self.assertEqual(
            result.endpoint_classifications.set_index("track_id").loc[1, "source_exclusion_reason"],
            "excluded_virtual_endpoint",
        )

    def test_15_stage8_merge_transition_is_excluded(self) -> None:
        event = pd.DataFrame([{
            "event_id": 0, "track_a": 1, "track_b": 9, "merged_track": 2,
            "merged_interval_start": 2, "merged_interval_end": 2,
            "split_frame": 3, "split_track_a": 1, "split_track_b": 9,
        }])
        result = self.run_stage(simple_break(), segmentation_events=event)
        self.assertTrue(result.track_id_remap.empty)

    def _division_data(self) -> SyntheticTracks:
        data = SyntheticTracks(7)
        data.add_track(10, {0: (10, 10, 10), 1: (10, 10, 10)})
        data.add_track(20, {2: (10, 9, 10), 3: (10, 9, 10)})
        data.add_track(30, {2: (10, 11, 10), 3: (10, 11, 10), 4: (10, 11, 10), 5: (10, 11, 10), 6: (10, 11, 10)})
        return data

    def test_16_confirmed_parent_endpoint_is_protected(self) -> None:
        result = self.run_stage(
            self._division_data(), division_events=division_events(),
            protected_tracks=protected_tracks(),
        )
        reason = result.endpoint_classifications.set_index("track_id").loc[10, "source_exclusion_reason"]
        self.assertEqual(reason, "excluded_confirmed_division")

    def test_17_confirmed_child_births_are_protected(self) -> None:
        data = self._division_data()
        data.add_track(99, {0: (10, 9, 10), 1: (10, 9, 10)})
        result = self.run_stage(
            data, division_events=division_events(), protected_tracks=protected_tracks()
        )
        self.assertFalse(set(result.track_id_remap["original_segment_track_id"]) & {20, 30})

    def test_18_division_child_can_be_repaired_later(self) -> None:
        data = self._division_data()
        data.add_track(40, {4: (10, 9, 10), 5: (10, 9, 10), 6: (10, 9, 10)})
        result = self.run_stage(
            data, division_events=division_events(), lineage_edges=lineage_edges(),
            protected_tracks=protected_tracks(),
        )
        row = result.track_id_remap.loc[result.track_id_remap["original_segment_track_id"] == 40]
        self.assertEqual(int(row.iloc[0]["canonical_track_id"]), 20)

    def test_19_probable_division_is_not_force_collapsed(self) -> None:
        result = self.run_stage(
            self._division_data(), division_events=division_events("probable")
        )
        reasons = result.endpoint_classifications.set_index("track_id")
        self.assertEqual(reasons.loc[10, "source_exclusion_reason"], "excluded_probable_division")

    def test_20_two_sources_two_targets_use_global_one_to_one(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 10, 10)})
        data.add_track(2, {0: (10, 14, 10), 1: (10, 14, 10)})
        data.add_track(3, {2: (10, 11, 10), 3: (10, 11, 10)})
        data.add_track(4, {2: (10, 15, 10), 3: (10, 15, 10)})
        result = self.run_stage(data)
        selected = result.track_id_remap.loc[result.track_id_remap["predecessor_track_id"].isin([1, 2])]
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected["original_segment_track_id"].nunique(), 2)

    def test_21_two_sources_one_target_leaves_one_unresolved(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 10, 10)})
        data.add_track(2, {0: (10, 12, 10), 1: (10, 12, 10)})
        data.add_track(3, {2: (10, 11, 10), 3: (10, 11, 10)})
        result = self.run_stage(data)
        self.assertEqual(len(result.track_id_remap), 1)
        self.assertIn("unresolved_no_unused_candidate", set(result.unresolved_endings["reason"]))

    def test_22_target_is_never_assigned_twice(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 10, 10)})
        data.add_track(2, {0: (10, 12, 10), 1: (10, 12, 10)})
        data.add_track(3, {2: (10, 11, 10), 3: (10, 11, 10)})
        result = self.run_stage(data)
        self.assertFalse(result.track_id_remap["original_segment_track_id"].duplicated().any())

    def test_23_remap_chain_resolves_to_oldest_id(self) -> None:
        data = SyntheticTracks(7)
        data.add_track(81, {0: (10, 10, 10), 1: (10, 11, 10)})
        data.add_track(143, {2: (10, 12, 10), 3: (10, 13, 10)})
        data.add_track(206, {4: (10, 14, 10), 5: (10, 15, 10)})
        result = self.run_stage(data)
        self.assertEqual(set(result.tracks["track_id"]), {81})
        self.assertEqual(set(result.track_id_remap["canonical_track_id"]), {81})

    def test_24_no_duplicate_track_frames_after_remap(self) -> None:
        result = self.run_stage(simple_break())
        self.assertFalse(result.tracks.duplicated(["track_id", "frame"]).any())

    def test_25_lineage_edges_survive_canonicalization(self) -> None:
        data = self._division_data()
        data.add_track(40, {4: (10, 9, 10), 5: (10, 9, 10), 6: (10, 9, 10)})
        result = self.run_stage(
            data, division_events=division_events(), lineage_edges=lineage_edges(),
            protected_tracks=protected_tracks(),
        )
        self.assertEqual(len(result.lineage_edges), 2)
        self.assertIn(20, set(result.lineage_edges["child_track_id"]))

    def test_26_track_lineage_contains_only_final_ids(self) -> None:
        result = self.run_stage(simple_break())
        self.assertEqual(set(result.track_lineage["track_id"]), set(result.tracks["track_id"]))

    def test_27_empty_inputs_have_stable_schemas(self) -> None:
        empty = pd.DataFrame(columns=("track_id", "frame", "cell", "cell_id", "z", "y", "x", "volume"))
        result = run_track_reconciliation(empty, [], sample_id="sample")
        self.assertEqual(result.continuation_candidates.columns.tolist(), list(config_module.CONTINUATION_CANDIDATE_COLUMNS))
        self.assertEqual(result.track_id_remap.columns.tolist(), list(config_module.TRACK_ID_REMAP_COLUMNS))

    def test_28_missing_stage7_diagnostics_do_not_fail(self) -> None:
        result = self.run_stage(simple_break(), global_motion=None, association_candidates=None)
        self.assertFalse(result.track_id_remap.empty)

    def test_29_shuffled_input_is_deterministic(self) -> None:
        data = simple_break()
        first = self.run_stage(data)
        shuffled_tracks = data.tracks.sample(frac=1.0, random_state=4).reset_index(drop=True)
        second = run_track_reconciliation(
            shuffled_tracks, data.frames,
            config=TrackReconciliationConfig(policy="submission"), sample_id="sample",
        )
        pd.testing.assert_frame_equal(first.continuation_candidates, second.continuation_candidates)
        pd.testing.assert_frame_equal(first.track_id_remap, second.track_id_remap)

    def test_30_stage_is_idempotent(self) -> None:
        data = simple_break()
        first = self.run_stage(data)
        second = run_track_reconciliation(
            first.tracks, data.frames,
            config=TrackReconciliationConfig(policy="submission"), sample_id="sample",
        )
        self.assertTrue(second.track_id_remap.empty)

    def test_31_save_load_empty_and_populated(self) -> None:
        for result in (
            self.run_stage(simple_break()),
            run_track_reconciliation(
                pd.DataFrame(columns=("track_id", "frame", "cell", "cell_id", "z", "y", "x", "volume")),
                [], sample_id="sample",
            ),
        ):
            with tempfile.TemporaryDirectory() as directory:
                paths = PipelinePaths(Path(directory))
                save_track_reconciliation_result(result, paths.stage11_reconciliation)
                loaded = load_stage11_outputs(paths=paths)
                self.assertEqual(loaded.tracks.columns.tolist(), result.tracks.columns.tolist())
                self.assertEqual(loaded.track_id_remap.columns.tolist(), result.track_id_remap.columns.tolist())

    def test_32_diagnostics_preserve_normal_result_tables(self) -> None:
        data = simple_break()
        normal = self.run_stage(data)
        diagnosed, trace = run_track_reconciliation(
            data.tracks, data.frames,
            config=TrackReconciliationConfig(policy="submission"),
            sample_id="sample", return_diagnostics=True,
        )
        self.assertIsInstance(trace, StageTrace)
        for name in normal.__dataclass_fields__:
            first = getattr(normal, name)
            second = getattr(diagnosed, name)
            if isinstance(first, pd.DataFrame):
                pd.testing.assert_frame_equal(first, second)
            else:
                self.assertEqual(first, second)

    def test_config_validation_and_json_stability(self) -> None:
        self.assertEqual(TrackReconciliationConfig().policy, "submission")
        self.assertIsInstance(TrackReconciliationConfig().as_dict()["voxel_size_zyx_um"], list)
        with self.assertRaises(ValueError):
            TrackReconciliationConfig(policy="unsafe")
        with self.assertRaises(ValueError):
            TrackReconciliationConfig(maximum_candidate_radius_um=1.0)

    def test_invalid_duplicate_input_is_rejected(self) -> None:
        data = simple_break()
        duplicated = pd.concat([data.tracks, data.tracks.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            run_track_reconciliation(duplicated, data.frames)


if __name__ == "__main__":
    unittest.main()
