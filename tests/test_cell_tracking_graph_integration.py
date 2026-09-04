"""Production Stage 7 integration tests for optional graph refinement."""

from __future__ import annotations

import tempfile
import unittest
from importlib import import_module
from pathlib import Path

import pandas as pd

from src.api import run_cell_tracking
from src.io import PipelinePaths, load_stage7_outputs, save_tracking_result


GraphTrackingConfig = import_module(
    "legacy.classical_pipeline.tracking.graph_tracking"
).GraphTrackingConfig


def detection(
    cell_id: int,
    z: float,
    y: float,
    x: float,
    *,
    volume: float = 100.0,
) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "centroid_z": z,
        "centroid_y": y,
        "centroid_x": x,
        "volume_voxels": volume,
        "z_min": max(0.0, z - 2.0),
        "y_min": max(0.0, y - 3.0),
        "x_min": max(0.0, x - 3.0),
        "z_max": min(64.0, z + 3.0),
        "y_max": min(256.0, y + 4.0),
        "x_max": min(256.0, x + 4.0),
        "extent": 0.8,
        "equivalent_radius": 3.0,
        "elongation": 1.2,
        "flatness": 1.1,
        "anisotropy": 1.3,
        "solidity": 0.9,
        "compactness": 0.8,
        "intensity_mean": 0.5,
        "intensity_std": 0.1,
        "intensity_cv": 0.2,
        "bbox_depth": 5.0,
        "bbox_height": 7.0,
        "bbox_width": 7.0,
    }


def graph_config(mode: str, **overrides) -> object:
    values = {
        "mode": mode,
        "maximum_radius_um": 25.0,
        "anchor_minimum_association_probability": 0.25,
        "anchor_minimum_probability_margin": 0.01,
        "anchor_maximum_position_error_um": 5.0,
        "graph_pair_weight": 8.0,
        "maximum_pair_cost_reduction": 8.0,
        "maximum_pair_cost_penalty": 8.0,
        "boundary_minimum_consensus": 0.20,
        "boundary_minimum_directional_agreement": 0.50,
        "outside_vote_minimum_distance_um": 0.10,
        "outside_vote_strong_distance_um": 1.0,
    }
    values.update(overrides)
    return GraphTrackingConfig(**values)


def continuation_frames(
    *,
    outlier: bool = False,
    central_volume: float = 130.0,
    wrong_x: float = 112.0,
) -> list[pd.DataFrame]:
    first = pd.DataFrame(
        [
            detection(1, 20, 100, 100),
            detection(2, 20, 80, 100),
            detection(3, 20, 120, 100),
            detection(4, 20, 100, 80),
            detection(5, 20, 100, 120),
        ]
    )
    last_anchor_y = 116 if outlier else 100
    second = pd.DataFrame(
        [
            detection(2, 20, 80, 105),
            detection(3, 20, 120, 105),
            detection(4, 20, 100, 85),
            detection(5, 20, last_anchor_y, 125),
            detection(1, 20, 100, 105, volume=central_volume),
            detection(6, 20, 100, wrong_x),
        ]
    )
    return [first, second]


def exit_frames() -> list[pd.DataFrame]:
    return [
        pd.DataFrame(
            [
                detection(1, 20, 100, 254),
                detection(2, 20, 60, 230),
                detection(3, 20, 140, 230),
            ]
        ),
        pd.DataFrame(
            [
                detection(2, 20, 60, 245),
                detection(3, 20, 140, 245),
            ]
        ),
        pd.DataFrame(
            [
                detection(2, 20, 60, 245),
                detection(3, 20, 140, 245),
                detection(1, 20, 100, 254),
            ]
        ),
    ]


class CellTrackingGraphIntegrationTests(unittest.TestCase):
    def test_disabled_default_is_exactly_explicit_disabled(self) -> None:
        frames = continuation_frames()
        default = run_cell_tracking(frames, sample_id="graph-disabled")
        disabled = run_cell_tracking(
            frames,
            sample_id="graph-disabled",
            graph_config=GraphTrackingConfig(mode="disabled"),
        )
        for name in (
            "tracks",
            "boundary_events",
            "missing_predictions",
            "global_motion",
            "association_events",
            "association_candidates",
        ):
            pd.testing.assert_frame_equal(
                getattr(default, name),
                getattr(disabled, name),
            )
        self.assertTrue(disabled.graph_transition_summary.empty)
        self.assertTrue(disabled.graph_candidate_evidence.empty)

    def test_shadow_has_diagnostics_but_preserves_production_decisions(self) -> None:
        frames = continuation_frames()
        disabled = run_cell_tracking(frames, sample_id="graph-shadow")
        shadow = run_cell_tracking(
            frames,
            sample_id="graph-shadow",
            graph_config=graph_config("shadow"),
        )
        pd.testing.assert_frame_equal(shadow.tracks, disabled.tracks)
        pd.testing.assert_frame_equal(
            shadow.association_events[
                ["decision_type", "track_id", "detection_position_index"]
            ],
            disabled.association_events[
                ["decision_type", "track_id", "detection_position_index"]
            ],
        )
        self.assertFalse(shadow.graph_transition_summary.empty)
        self.assertFalse(shadow.graph_candidate_evidence.empty)
        graph_columns = {
            "graph_available",
            "graph_anchor_count",
            "graph_inlier_count",
            "graph_vote_score",
            "graph_confidence",
            "graph_cost_delta",
            "base_selected",
            "graph_selected",
            "assignment_changed_by_graph",
            "graph_boundary_event_type",
            "graph_boundary_confidence",
        }
        self.assertTrue(graph_columns.issubset(shadow.association_candidates))
        self.assertTrue(graph_columns.issubset(shadow.association_events))

    def test_apply_corrects_ambiguous_continuation(self) -> None:
        frames = continuation_frames()
        disabled = run_cell_tracking(frames, sample_id="graph-apply")
        applied = run_cell_tracking(
            frames,
            sample_id="graph-apply",
            graph_config=graph_config("apply"),
        )
        disabled_target = int(
            disabled.tracks.loc[
                (disabled.tracks["track_id"] == 0)
                & (disabled.tracks["frame"] == 1),
                "cell",
            ].iloc[0]
        )
        applied_target = int(
            applied.tracks.loc[
                (applied.tracks["track_id"] == 0)
                & (applied.tracks["frame"] == 1),
                "cell",
            ].iloc[0]
        )
        self.assertEqual(disabled_target, 5)
        self.assertEqual(applied_target, 4)
        self.assertTrue(applied.graph_refinement_events["changed"].any())

    def test_outlier_anchor_is_rejected_without_changing_correct_match(self) -> None:
        frames = continuation_frames(
            outlier=True,
            central_volume=100.0,
            wrong_x=106.0,
        )
        disabled = run_cell_tracking(frames, sample_id="graph-outlier")
        applied = run_cell_tracking(
            frames,
            sample_id="graph-outlier",
            graph_config=graph_config(
                "apply",
                anchor_minimum_association_probability=0.01,
                anchor_minimum_probability_margin=-1.0,
                anchor_maximum_position_error_um=10.0,
                ambiguous_probability_threshold=0.0,
                ambiguous_margin_threshold=-1.0,
                vote_inlier_radius_um=2.0,
            ),
        )
        central_disabled = disabled.tracks.loc[
            disabled.tracks["track_id"] == 0,
            ["track_id", "frame", "cell"],
        ].reset_index(drop=True)
        central_applied = applied.tracks.loc[
            applied.tracks["track_id"] == 0,
            ["track_id", "frame", "cell"],
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(central_applied, central_disabled)
        evidence = applied.graph_candidate_evidence
        self.assertTrue((evidence["inlier_count"] < evidence["anchor_count"]).any())

    def test_graph_exit_stays_pending_and_is_cancelled_by_reacquisition(self) -> None:
        result = run_cell_tracking(
            exit_frames(),
            sample_id="graph-exit",
            graph_config=graph_config("apply"),
        )
        hypotheses = result.graph_boundary_hypotheses
        supported_exit = hypotheses.loc[
            (hypotheses["event_type"] == "exit")
            & hypotheses["supported"].astype(bool)
        ]
        self.assertFalse(supported_exit.empty)
        events = set(result.boundary_events["event_type"])
        self.assertIn("graph_exit_predicted", events)
        self.assertIn("graph_exit_pending", events)
        self.assertIn("graph_exit_rejected_reacquired", events)
        track_zero_frames = result.tracks.loc[
            result.tracks["track_id"] == 0,
            "frame",
        ].tolist()
        self.assertEqual(track_zero_frames, [0, 2])

    def test_graph_exit_confirms_only_after_existing_pending_window(self) -> None:
        frames = exit_frames()[:2]
        frames.extend(
            [
                pd.DataFrame(
                    [
                        detection(2, 20, 60, 245),
                        detection(3, 20, 140, 245),
                    ]
                ),
                pd.DataFrame(
                    [
                        detection(2, 20, 60, 245),
                        detection(3, 20, 140, 245),
                    ]
                ),
            ]
        )
        result = run_cell_tracking(
            frames,
            sample_id="graph-exit-confirmed",
            graph_config=graph_config("apply"),
        )
        track_events = result.boundary_events.loc[
            result.boundary_events["track_id"] == 0,
            ["frame", "event_type"],
        ]
        confirmed = track_events.loc[
            track_events["event_type"] == "graph_exit_confirmed"
        ]
        self.assertEqual(confirmed["frame"].tolist(), [3])

    def test_backward_votes_support_boundary_entry(self) -> None:
        frames = [
            pd.DataFrame(
                [
                    detection(1, 20, 90, 20),
                    detection(2, 20, 110, 20),
                ]
            ),
            pd.DataFrame(
                [
                    detection(1, 20, 90, 30),
                    detection(2, 20, 110, 30),
                    detection(3, 20, 100, 2),
                ]
            ),
        ]
        result = run_cell_tracking(
            frames,
            sample_id="graph-entry",
            graph_config=graph_config("apply"),
        )
        self.assertIn(
            "graph_entry_supported",
            set(result.boundary_events["event_type"]),
        )
        entry = result.graph_boundary_hypotheses.loc[
            result.graph_boundary_hypotheses["event_type"] == "entry"
        ]
        self.assertTrue(entry["supported"].astype(bool).any())

    def test_inside_predecessor_vote_can_oppose_birth(self) -> None:
        frames = [
            pd.DataFrame(
                [
                    detection(1, 20, 90, 20),
                    detection(2, 20, 110, 20),
                ]
            ),
            pd.DataFrame(
                [
                    detection(1, 20, 90, 30),
                    detection(2, 20, 110, 30),
                    detection(3, 20, 100, 10),
                ]
            ),
        ]
        applied = run_cell_tracking(
            frames,
            sample_id="inside-predecessor",
            graph_config=graph_config(
                "apply",
                graph_pair_weight=20.0,
                maximum_pair_cost_reduction=20.0,
                inside_backward_vote_birth_penalty=4.0,
            ),
        )
        hypotheses = applied.graph_boundary_hypotheses
        self.assertIn(
            "inside_predecessor_vote",
            set(hypotheses["decision"]),
        )
        birth_event = applied.graph_refinement_events.loc[
            applied.graph_refinement_events["entity_type"] == "detection"
        ].iloc[0]
        self.assertGreater(birth_event["graph_cost"], birth_event["base_cost"])

    def test_low_anchor_fallback_and_safety_invalid_pairs(self) -> None:
        frames = continuation_frames()
        disabled = run_cell_tracking(frames, sample_id="graph-fallback")
        fallback = run_cell_tracking(
            frames,
            sample_id="graph-fallback",
            graph_config=graph_config("apply", minimum_anchor_votes=10),
        )
        pd.testing.assert_frame_equal(fallback.tracks, disabled.tracks)
        self.assertTrue(fallback.graph_candidate_evidence.empty)

        far_frames = [
            pd.DataFrame(
                [
                    detection(1, 20, 30, 30),
                    detection(2, 20, 80, 80),
                    detection(3, 20, 120, 120),
                ]
            ),
            pd.DataFrame(
                [
                    detection(2, 20, 80, 80),
                    detection(3, 20, 120, 120),
                    detection(4, 50, 220, 220),
                ]
            ),
        ]
        far = run_cell_tracking(
            far_frames,
            sample_id="graph-safety",
            graph_config=graph_config("apply", minimum_anchor_votes=1),
        )
        track_zero = far.association_events.loc[
            far.association_events["track_id"] == 0
        ]
        self.assertEqual(track_zero["decision_type"].tolist(), ["miss"])
        far_birth = far.association_events.loc[
            (far.association_events["decision_type"] == "birth")
            & (far.association_events["detection_position_index"] == 2)
        ]
        self.assertFalse(far_birth.empty)

    def test_determinism_trace_and_stage7_artifacts(self) -> None:
        frames = continuation_frames()
        first, trace = run_cell_tracking(
            frames,
            sample_id="graph-deterministic",
            graph_config=graph_config("shadow"),
            return_diagnostics=True,
        )
        second = run_cell_tracking(
            frames,
            sample_id="graph-deterministic",
            graph_config=graph_config("shadow"),
        )
        for name in (
            "tracks",
            "association_events",
            "graph_transition_summary",
            "graph_candidate_evidence",
            "graph_anchor_votes",
            "graph_boundary_hypotheses",
            "graph_refinement_events",
        ):
            pd.testing.assert_frame_equal(
                getattr(first, name),
                getattr(second, name),
            )
            self.assertIn(name, trace.intermediates | {"tracks": first.tracks})

        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            save_tracking_result(first, paths.stage7_tracking)
            loaded = load_stage7_outputs(paths=paths)
            pd.testing.assert_frame_equal(
                loaded.graph_candidate_evidence,
                first.graph_candidate_evidence,
                check_dtype=False,
            )
            graph_files = tuple(paths.stage7_tracking.glob("graph_*.csv"))
            self.assertEqual(len(graph_files), 5)
            for path in graph_files:
                path.unlink()
            legacy = load_stage7_outputs(paths=paths)
            self.assertTrue(legacy.graph_transition_summary.empty)
            self.assertTrue(legacy.graph_candidate_evidence.empty)


if __name__ == "__main__":
    unittest.main()
