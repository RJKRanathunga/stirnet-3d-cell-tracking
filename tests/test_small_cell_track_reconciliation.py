"""Focused small-cell reliability and protection tests for Stage 11."""

from __future__ import annotations

from importlib import import_module
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.api import run_track_reconciliation
from src.io import PipelinePaths, load_stage11_outputs, save_track_reconciliation_result
from tests.test_track_reconciliation import (
    SyntheticTracks,
    division_events,
    protected_tracks,
)


config_module = import_module("src.11_track_reconciliation.step01_config")
reliability = import_module("src.11_track_reconciliation.small_cell_reliability")
TrackReconciliationConfig = config_module.TrackReconciliationConfig


class SmallCellTrackReconciliationTests(unittest.TestCase):
    def run_stage(
        self,
        data: SyntheticTracks,
        *,
        policy: str = "conservative",
        config: TrackReconciliationConfig | None = None,
        **kwargs,
    ):
        return run_track_reconciliation(
            data.tracks,
            data.frames,
            config=config or TrackReconciliationConfig(policy=policy),
            sample_id="small-cell-test",
            **kwargs,
        )

    @staticmethod
    def supported_break(
        *, source_volume: float = 40.0, target_volume: float = 120.0,
        source_shape: float = 2.0, target_shape: float = 8.0,
    ) -> SyntheticTracks:
        data = SyntheticTracks(6)
        data.add_track(
            1, {0: (10, 10, 10), 1: (10, 11, 10)},
            volume=source_volume, shape=source_shape,
        )
        data.add_track(
            2, {2: (10, 12, 10), 3: (10, 13, 10)},
            volume=target_volume, shape=target_shape,
        )
        return data

    def test_small_cell_large_volume_change_can_reconcile(self) -> None:
        result = self.run_stage(self.supported_break())
        decision = result.track_id_remap.iloc[0]
        candidate = result.continuation_candidates.iloc[0]
        self.assertEqual(decision["decision"], "accepted_small_cell_supported")
        self.assertFalse(bool(decision["forced"]))
        self.assertEqual(float(candidate["effective_pair_volume"]), 40.0)
        self.assertEqual(candidate["small_cell_regime"], "extremely_small")
        self.assertGreaterEqual(int(candidate["small_cell_support_count"]), 2)

    def test_extremely_small_sixteen_fold_change_needs_support(self) -> None:
        result = self.run_stage(
            self.supported_break(source_volume=6.0, target_volume=97.0)
        )
        self.assertEqual(
            result.track_id_remap.iloc[0]["decision"],
            "accepted_small_cell_supported",
        )
        self.assertAlmostEqual(
            float(result.continuation_candidates.iloc[0]["volume_log_score_scale_used"]),
            math.log(8.0),
        )

    def test_small_cell_large_shape_change_does_not_dominate(self) -> None:
        result = self.run_stage(
            self.supported_break(source_shape=1.0, target_shape=20.0)
        )
        candidate = result.continuation_candidates.iloc[0]
        self.assertTrue(bool(candidate["small_cell_special_acceptance"]))
        self.assertAlmostEqual(float(candidate["shape_weight_multiplier"]), 0.10)

    def test_small_cell_requires_strong_forward_support(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 10, 10)}, volume=6)
        data.add_track(2, {2: (10, 30, 10), 3: (10, 30, 10)}, volume=97)
        config = TrackReconciliationConfig(
            policy="conservative", conservative_minimum_score=0.99
        )
        result = self.run_stage(data, config=config)
        candidate = result.continuation_candidates.iloc[0]
        self.assertFalse(bool(candidate["small_cell_strong_forward"]))
        self.assertFalse(bool(candidate["small_cell_special_acceptance"]))
        self.assertTrue(result.track_id_remap.empty)

    def test_small_cell_requires_independent_contextual_support(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 11, 10)}, volume=6)
        data.add_track(2, {2: (10, 12, 10)}, volume=97)
        config = TrackReconciliationConfig(
            policy="conservative", conservative_minimum_score=0.99
        )
        result = self.run_stage(data, config=config)
        candidate = result.continuation_candidates.iloc[0]
        self.assertTrue(bool(candidate["small_cell_strong_forward"]))
        self.assertEqual(int(candidate["small_cell_support_count"]), 1)
        self.assertFalse(bool(candidate["small_cell_special_acceptance"]))
        self.assertTrue(result.track_id_remap.empty)

    def test_small_cell_ambiguous_candidates_remain_unresolved(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 10, 10)}, volume=40)
        data.add_track(2, {2: (10, 9, 10), 3: (10, 9, 10)}, volume=120)
        data.add_track(3, {2: (10, 11, 10), 3: (10, 11, 10)}, volume=120)
        config = TrackReconciliationConfig(
            policy="conservative", conservative_minimum_score=0.99
        )
        result = self.run_stage(data, config=config)
        source = result.continuation_candidates.loc[
            result.continuation_candidates["source_track_id"] == 1
        ]
        self.assertFalse(source["small_cell_special_acceptance"].astype(bool).any())
        self.assertTrue(result.track_id_remap.empty)

    def test_small_cell_special_acceptance_is_not_forced(self) -> None:
        result = self.run_stage(self.supported_break(), policy="submission")
        decision = result.continuation_decisions.loc[
            result.continuation_decisions["decision"]
            == "accepted_small_cell_supported"
        ].iloc[0]
        self.assertEqual(decision["policy_phase"], "small_cell_supported")
        self.assertFalse(bool(decision["forced"]))
        self.assertEqual(result.metadata["forced_assignment_count"], 0)

    def test_small_cell_stage_trace_explains_support_and_reduced_weights(self) -> None:
        data = self.supported_break()
        _, trace = run_track_reconciliation(
            data.tracks,
            data.frames,
            config=TrackReconciliationConfig(policy="conservative"),
            sample_id="small-cell-test",
            return_diagnostics=True,
        )
        decision = next(
            item for item in trace.decisions
            if item.decision_type == "reconciliation_decision"
            and item.outcome == "accepted_small_cell_supported"
        )
        self.assertTrue(decision.metrics["small_cell_strong_forward"])
        self.assertTrue(decision.metrics["small_cell_strong_backward"])
        self.assertLess(decision.metrics["volume_weight_multiplier"], 1.0)
        self.assertLess(decision.metrics["shape_weight_multiplier"], 1.0)

    def test_small_cell_singleton_source_uses_only_supported_path(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {1: (10, 11, 10)}, volume=40)
        data.add_track(2, {2: (10, 12, 10), 3: (10, 13, 10)}, volume=120)
        result = self.run_stage(data, policy="submission")
        candidate = result.continuation_candidates.loc[
            (result.continuation_candidates["source_track_id"] == 1)
            & (result.continuation_candidates["target_track_id"] == 2)
        ].iloc[0]
        self.assertTrue(bool(candidate["small_cell_history_exception"]))
        self.assertTrue(bool(candidate["small_cell_special_acceptance"]))
        decision = result.track_id_remap.iloc[0]
        self.assertEqual(decision["decision"], "accepted_small_cell_supported")
        self.assertFalse(bool(decision["forced"]))

    def test_normal_singleton_source_cannot_use_history_exception(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {1: (10, 11, 10)}, volume=700)
        data.add_track(2, {2: (10, 12, 10), 3: (10, 13, 10)}, volume=800)
        result = self.run_stage(data, policy="submission")
        candidate = result.continuation_candidates.iloc[0]
        self.assertTrue(bool(candidate["small_cell_history_exception"]))
        self.assertFalse(bool(candidate["admissible"]))
        self.assertEqual(candidate["admissibility_reason"], "excluded_insufficient_history")
        self.assertTrue(result.track_id_remap.empty)

    def test_small_cell_singleton_source_uses_only_supported_path(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {1: (10, 11, 10)}, volume=40)
        data.add_track(2, {2: (10, 12, 10), 3: (10, 13, 10)}, volume=120)
        result = self.run_stage(data, policy="submission")
        candidate = result.continuation_candidates.loc[
            (result.continuation_candidates["source_track_id"] == 1)
            & (result.continuation_candidates["target_track_id"] == 2)
        ].iloc[0]
        self.assertTrue(bool(candidate["small_cell_history_exception"]))
        self.assertTrue(bool(candidate["small_cell_special_acceptance"]))
        decision = result.track_id_remap.iloc[0]
        self.assertEqual(decision["decision"], "accepted_small_cell_supported")
        self.assertFalse(bool(decision["forced"]))

    def test_normal_singleton_source_cannot_use_history_exception(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {1: (10, 11, 10)}, volume=700)
        data.add_track(2, {2: (10, 12, 10), 3: (10, 13, 10)}, volume=800)
        result = self.run_stage(data, policy="submission")
        candidate = result.continuation_candidates.iloc[0]
        self.assertTrue(bool(candidate["small_cell_history_exception"]))
        self.assertFalse(bool(candidate["admissible"]))
        self.assertEqual(candidate["admissibility_reason"], "excluded_insufficient_history")
        self.assertTrue(result.track_id_remap.empty)

    def _division_data(self) -> SyntheticTracks:
        data = SyntheticTracks(7)
        data.add_track(10, {0: (10, 10, 10), 1: (10, 10, 10)}, volume=40)
        data.add_track(20, {2: (10, 9, 10), 3: (10, 9, 10)}, volume=120)
        data.add_track(
            30,
            {2: (10, 11, 10), 3: (10, 11, 10), 4: (10, 11, 10)},
            volume=120,
        )
        return data

    def test_small_cell_does_not_override_confirmed_division(self) -> None:
        result = self.run_stage(
            self._division_data(),
            division_events=division_events("confirmed"),
            protected_tracks=protected_tracks(),
        )
        endpoint = result.endpoint_classifications.set_index("track_id")
        self.assertEqual(endpoint.loc[10, "source_exclusion_reason"], "excluded_confirmed_division")
        self.assertFalse(set(result.track_id_remap["original_segment_track_id"]) & {20, 30})

    def test_small_cell_does_not_override_probable_division(self) -> None:
        result = self.run_stage(
            self._division_data(), division_events=division_events("probable")
        )
        endpoint = result.endpoint_classifications.set_index("track_id")
        self.assertEqual(endpoint.loc[10, "source_exclusion_reason"], "excluded_probable_division")
        self.assertTrue(result.track_id_remap.empty)

    def test_small_cell_does_not_override_merge_protection(self) -> None:
        merge = pd.DataFrame([{
            "event_id": 0, "track_a": 1, "track_b": 9, "merged_track": 2,
            "merged_interval_start": 2, "merged_interval_end": 2,
            "split_frame": 3, "split_track_a": 1, "split_track_b": 9,
        }])
        result = self.run_stage(self.supported_break(), segmentation_events=merge)
        self.assertTrue(result.track_id_remap.empty)

    def test_small_cell_does_not_join_confirmed_boundary_exit(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(
            1, {1: (10, 11, 10)},
            volume=40, boundary_frames={1},
        )
        data.add_track(2, {2: (10, 12, 10), 3: (10, 13, 10)}, volume=120)
        result = self.run_stage(data)
        self.assertTrue(result.track_id_remap.empty)
        self.assertEqual(
            result.endpoint_classifications.set_index("track_id").loc[
                1, "source_exclusion_reason"
            ],
            "excluded_boundary_exit",
        )

    def test_small_cell_does_not_attach_to_running_target(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 11, 10)}, volume=40)
        data.add_track(
            2, {1: (10, 11, 10), 2: (10, 12, 10), 3: (10, 13, 10)}, volume=120
        )
        result = self.run_stage(data, policy="submission")
        self.assertTrue(result.track_id_remap.empty)
        self.assertFalse(
            ((result.continuation_candidates["source_track_id"] == 1)
             & (result.continuation_candidates["target_track_id"] == 2)).any()
        )

    def test_normal_cell_scoring_is_unchanged(self) -> None:
        data = self.supported_break(source_volume=700, target_volume=900)
        enabled = self.run_stage(data, policy="submission")
        disabled = self.run_stage(
            data,
            config=TrackReconciliationConfig(
                policy="submission", small_cell_mode_enabled=False
            ),
        )
        first = enabled.continuation_candidates.iloc[0]
        second = disabled.continuation_candidates.iloc[0]
        self.assertFalse(bool(first["small_cell_mode_applied"]))
        self.assertEqual(float(first["continuation_score"]), float(second["continuation_score"]))
        self.assertEqual(float(first["size_aware_score_delta"]), 0.0)
        self.assertEqual(
            enabled.track_id_remap["decision"].tolist(),
            disabled.track_id_remap["decision"].tolist(),
        )

    def test_transition_volume_interpolates_reliability(self) -> None:
        config = TrackReconciliationConfig()
        at_small = reliability.evidence_weight_multipliers(200.0, config)
        at_transition = reliability.evidence_weight_multipliers(350.0, config)
        at_normal = reliability.evidence_weight_multipliers(600.0, config)
        self.assertLess(at_small["volume"], at_transition["volume"])
        self.assertLess(at_transition["volume"], at_normal["volume"])
        self.assertGreater(at_small["forward_position"], at_transition["forward_position"])
        self.assertGreater(at_transition["forward_position"], at_normal["forward_position"])
        scale = reliability.volume_log_score_scale(350.0, config)
        self.assertGreater(scale, config.volume_log_score_scale)
        self.assertLess(scale, config.small_cell_volume_log_scale_at_250)

    def test_missing_optional_small_cell_evidence_is_omitted(self) -> None:
        data = self.supported_break()
        optional = {
            "intensity_median", "intensity_std", "intensity_sum",
            "equivalent_radius", "axis_major", "axis_middle", "axis_minor",
        }
        frames = [frame.drop(columns=list(optional & set(frame.columns))) for frame in data.frames]
        result = run_track_reconciliation(
            data.tracks,
            frames,
            config=TrackReconciliationConfig(policy="conservative"),
            sample_id="small-cell-test",
        )
        candidate = result.continuation_candidates.iloc[0]
        self.assertTrue(math.isfinite(float(candidate["continuation_score"])))
        self.assertTrue(math.isnan(float(candidate["shape_error"])))
        self.assertTrue(math.isfinite(float(candidate["intensity_error"])))
        sum_only = reliability.size_aware_intensity_error(
            {"intensity_sum_error": 2.0},
            2.0,
            40.0,
            TrackReconciliationConfig(),
        )
        self.assertTrue(math.isnan(sum_only))

    def test_small_cell_global_assignment_resolves_competition(self) -> None:
        data = SyntheticTracks(6)
        data.add_track(1, {0: (10, 10, 10), 1: (10, 10, 10)}, volume=40)
        data.add_track(2, {0: (10, 20, 10), 1: (10, 20, 10)}, volume=50)
        data.add_track(3, {2: (10, 11, 10), 3: (10, 11, 10)}, volume=120)
        data.add_track(4, {2: (10, 21, 10), 3: (10, 21, 10)}, volume=130)
        result = self.run_stage(data)
        accepted = result.track_id_remap.loc[
            result.track_id_remap["decision"] == "accepted_small_cell_supported"
        ]
        self.assertEqual(len(accepted), 2)
        self.assertEqual(accepted["original_segment_track_id"].nunique(), 2)
        self.assertEqual(accepted["predecessor_track_id"].nunique(), 2)

    def test_small_cell_chained_fragments_are_canonicalized(self) -> None:
        data = SyntheticTracks(9)
        data.add_track(129, {0: (10, 10, 10), 1: (10, 11, 10)}, volume=40)
        data.add_track(243, {2: (10, 12, 10), 3: (10, 13, 10)}, volume=80)
        data.add_track(317, {4: (10, 14, 10), 5: (10, 15, 10)}, volume=120)
        data.add_track(332, {6: (10, 16, 10), 7: (10, 17, 10)}, volume=60)
        result = self.run_stage(data)
        self.assertEqual(set(result.tracks["track_id"]), {129})
        self.assertEqual(set(result.track_id_remap["canonical_track_id"]), {129})
        self.assertEqual(result.tracks["frame"].tolist(), list(range(8)))

    def test_small_cell_reconciliation_is_deterministic(self) -> None:
        data = self.supported_break()
        first = self.run_stage(data)
        shuffled = data.tracks.sample(frac=1.0, random_state=19).reset_index(drop=True)
        second = run_track_reconciliation(
            shuffled,
            data.frames,
            config=TrackReconciliationConfig(policy="conservative"),
            sample_id="small-cell-test",
        )
        pd.testing.assert_frame_equal(
            first.continuation_candidates, second.continuation_candidates
        )
        pd.testing.assert_frame_equal(first.track_id_remap, second.track_id_remap)

    def test_small_cell_reconciliation_is_idempotent(self) -> None:
        data = self.supported_break()
        first = self.run_stage(data)
        second = run_track_reconciliation(
            first.tracks,
            data.frames,
            config=TrackReconciliationConfig(policy="conservative"),
            sample_id="small-cell-test",
        )
        self.assertTrue(second.track_id_remap.empty)
        pd.testing.assert_frame_equal(first.tracks, second.tracks)

    def test_small_cell_candidate_schema_serializes_empty_results(self) -> None:
        empty = pd.DataFrame(columns=(
            "track_id", "frame", "cell", "cell_id", "z", "y", "x", "volume",
        ))
        result = run_track_reconciliation(empty, [], sample_id="small-cell-test")
        self.assertEqual(
            result.continuation_candidates.columns.tolist(),
            list(config_module.CONTINUATION_CANDIDATE_COLUMNS),
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            save_track_reconciliation_result(result, paths.stage11_reconciliation)
            loaded = load_stage11_outputs(paths=paths)
            self.assertEqual(
                loaded.continuation_candidates.columns.tolist(),
                list(config_module.CONTINUATION_CANDIDATE_COLUMNS),
            )


if __name__ == "__main__":
    unittest.main()
