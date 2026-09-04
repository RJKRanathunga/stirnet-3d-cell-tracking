"""Synthetic tests for Stage 12 final visualization and endpoint auditing."""

from __future__ import annotations

import unittest
from importlib import import_module

import pandas as pd


stage12 = import_module("legacy.classical_pipeline.final_visualization")
FinalVisualizationConfig = stage12.FinalVisualizationConfig
prepare_final_visualization_data = stage12.prepare_final_visualization_data


def cells_for(tracks: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "frame": tracks["frame"].astype(int),
        "cell_id": tracks["cell_id"].astype(int),
        "centroid_z": tracks["z"].astype(float),
        "centroid_y": tracks["y"].astype(float),
        "centroid_x": tracks["x"].astype(float),
    })


def endpoint_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "observation_count": 2, "real_observation_count": 2,
        "first_observation_index": 0, "last_observation_index": 1,
        "first_real_observation_index": 0, "last_real_observation_index": 1,
        "first_is_boundary": False, "last_is_boundary": False,
        "first_is_virtual": False, "last_is_virtual": False,
        "source_eligible": False, "target_eligible": False,
        "source_exclusion_reason": "excluded_last_frame",
        "target_exclusion_reason": "excluded_sequence_start",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


class FinalVisualizationTests(unittest.TestCase):
    def test_unexplained_interior_birth_and_end_are_failures(self) -> None:
        tracks = pd.DataFrame([
            {"track_id": 1, "frame": 1, "cell_id": 1, "z": 5, "y": 5, "x": 5},
            {"track_id": 1, "frame": 2, "cell_id": 2, "z": 5, "y": 6, "x": 5},
        ])
        endpoints = endpoint_rows([{
            "track_id": 1, "first_frame": 1, "last_frame": 2,
            "first_real_frame": 1, "last_real_frame": 2,
            "source_eligible": True, "target_eligible": True,
            "source_exclusion_reason": "eligible",
            "target_exclusion_reason": "eligible",
        }])
        result = prepare_final_visualization_data(
            tracks, cells_for(tracks), endpoint_classifications=endpoints,
            sequence_last_frame=4,
        )
        summary = result.track_summary.iloc[0]
        self.assertTrue(bool(summary.suspicious_start))
        self.assertTrue(bool(summary.suspicious_end))
        self.assertEqual(len(result.failure_events), 2)

    def test_boundary_endpoints_are_valid(self) -> None:
        tracks = pd.DataFrame([
            {"track_id": 2, "frame": 1, "cell_id": 3, "z": 0, "y": 5, "x": 5},
            {"track_id": 2, "frame": 2, "cell_id": 4, "z": 0, "y": 6, "x": 5},
        ])
        endpoints = endpoint_rows([{
            "track_id": 2, "first_frame": 1, "last_frame": 2,
            "first_real_frame": 1, "last_real_frame": 2,
            "first_is_boundary": True, "last_is_boundary": True,
            "source_exclusion_reason": "excluded_boundary_exit",
            "target_exclusion_reason": "excluded_boundary_entry_target",
        }])
        result = prepare_final_visualization_data(
            tracks, cells_for(tracks), endpoint_classifications=endpoints,
            sequence_last_frame=4,
        )
        summary = result.track_summary.iloc[0]
        self.assertEqual(summary.start_classification, "boundary_entry")
        self.assertEqual(summary.end_classification, "boundary_exit")
        self.assertFalse(bool(summary.suspicious_start))
        self.assertFalse(bool(summary.suspicious_end))

    def test_confirmed_division_is_not_reported_as_track_loss(self) -> None:
        tracks = pd.DataFrame([
            {"track_id": 10, "frame": 0, "cell_id": 1, "z": 5, "y": 5, "x": 5},
            {"track_id": 10, "frame": 1, "cell_id": 2, "z": 5, "y": 6, "x": 5},
            {"track_id": 20, "frame": 2, "cell_id": 3, "z": 5, "y": 6, "x": 4},
            {"track_id": 20, "frame": 4, "cell_id": 4, "z": 5, "y": 7, "x": 4},
            {"track_id": 30, "frame": 2, "cell_id": 5, "z": 5, "y": 6, "x": 6},
            {"track_id": 30, "frame": 4, "cell_id": 6, "z": 5, "y": 7, "x": 6},
        ])
        endpoints = endpoint_rows([
            {"track_id": 10, "first_frame": 0, "last_frame": 1,
             "first_real_frame": 0, "last_real_frame": 1,
             "source_exclusion_reason": "excluded_confirmed_division"},
            {"track_id": 20, "first_frame": 2, "last_frame": 4,
             "first_real_frame": 2, "last_real_frame": 4,
             "target_exclusion_reason": "excluded_confirmed_division"},
            {"track_id": 30, "first_frame": 2, "last_frame": 4,
             "first_real_frame": 2, "last_real_frame": 4,
             "target_exclusion_reason": "excluded_confirmed_division"},
        ])
        divisions = pd.DataFrame([{
            "parent_track_id": 10, "parent_end_frame": 1,
            "child_birth_frame": 2, "child_track_a": 20,
            "child_track_b": 30, "decision": "confirmed",
        }])
        result = prepare_final_visualization_data(
            tracks, cells_for(tracks), endpoint_classifications=endpoints,
            division_events=divisions, sequence_last_frame=4,
        )
        summary = result.track_summary.set_index("track_id")
        self.assertEqual(summary.loc[10, "end_classification"], "confirmed_division_parent")
        self.assertEqual(summary.loc[20, "start_classification"], "confirmed_division_child")
        self.assertFalse(bool(summary.loc[10, "suspicious_end"]))

    def test_remapped_track_retains_repair_provenance_and_gap(self) -> None:
        tracks = pd.DataFrame([
            {"track_id": 1, "frame": 0, "cell_id": 1, "z": 5, "y": 5, "x": 5},
            {"track_id": 1, "frame": 1, "cell_id": 2, "z": 5, "y": 6, "x": 5},
            {"track_id": 1, "frame": 3, "cell_id": 3, "z": 5, "y": 8, "x": 5},
            {"track_id": 1, "frame": 4, "cell_id": 4, "z": 5, "y": 9, "x": 5},
        ])
        endpoints = endpoint_rows([
            {"track_id": 1, "first_frame": 0, "last_frame": 1,
             "first_real_frame": 0, "last_real_frame": 1,
             "source_exclusion_reason": "eligible"},
            {"track_id": 2, "first_frame": 3, "last_frame": 4,
             "first_real_frame": 3, "last_real_frame": 4,
             "target_exclusion_reason": "eligible"},
        ])
        remap = pd.DataFrame([{
            "decision_id": "d1", "original_segment_track_id": 2,
            "predecessor_track_id": 1, "canonical_track_id": 1,
            "from_frame": 3, "to_frame": 4, "gap_frames": 1,
            "decision": "forced_best_candidate", "policy_phase": "submission",
            "forced": True, "continuation_score": 0.5, "assignment_cost": 0.5,
        }])
        result = prepare_final_visualization_data(
            tracks, cells_for(tracks), endpoint_classifications=endpoints,
            track_id_remap=remap, sequence_last_frame=4,
        )
        summary = result.track_summary.iloc[0]
        self.assertTrue(bool(summary.stage11_modified))
        self.assertEqual(int(summary.forced_repair_count), 1)
        self.assertEqual(int(summary.missing_frame_count), 1)
        self.assertIn("2", summary.original_segment_ids)
        self.assertTrue((result.diagnostic_events["event_type"] == "temporal_gap").any())
        self.assertTrue((result.diagnostic_events["event_type"] == "stage11_repair").any())


if __name__ == "__main__":
    unittest.main()
