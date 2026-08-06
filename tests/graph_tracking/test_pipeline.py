from __future__ import annotations

import unittest

import numpy as np

from graph_tracking import GraphTrackingConfig, refine_transition_with_graph
from helpers import base_assignment, detections, state


class PipelineTests(unittest.TestCase):
    def continuation_case(self):
        states = [
            state(0, [5, 5, 5]),
            state(1, [5, 2, 5]),
            state(2, [5, 8, 5]),
            state(3, [2, 5, 5]),
        ]
        frame = detections([
            ([5, 2, 7], False, "", 100),
            ([5, 8, 7], False, "", 100),
            ([2, 5, 7], False, "", 100),
            ([5, 5, 7], False, "", 100),  # correct
            ([5, 5, 10], False, "", 100),  # wrong base choice
        ])
        invalid = np.ones((4, 5), dtype=bool)
        invalid[0, 3:5] = False
        invalid[1, 0] = False
        invalid[2, 1] = False
        invalid[3, 2] = False
        pair = np.full((4, 5), 1e6)
        pair[0, 3] = 0.75
        pair[0, 4] = 0.50
        pair[1, 0] = pair[2, 1] = pair[3, 2] = 0.05
        distance = np.full((4, 5), 99.0)
        distance[0, 3] = 2.0
        distance[0, 4] = 5.0
        distance[1, 0] = distance[2, 1] = distance[3, 2] = 0.2
        base = base_assignment(pair, invalid, [5, 5, 5, 5], [5, 5, 5, 0.9, 0.9], distance)
        return states, frame, base

    def test_shadow_mode_preserves_base_assignment(self) -> None:
        states, frame, base = self.continuation_case()
        config = GraphTrackingConfig(
            mode="shadow",
            maximum_radius_um=20,
            anchor_minimum_association_probability=0.50,
            anchor_minimum_probability_margin=0.05,
            graph_pair_weight=2.0,
            maximum_pair_cost_reduction=4.0,
            maximum_pair_cost_penalty=4.0,
        )
        result = refine_transition_with_graph(
            eligible_states=states,
            detections=frame,
            current_frame=1,
            base_assignment=base,
            volume_shape_zyx=np.asarray([20, 20, 20]),
            voxel_size_zyx_um=np.ones(3),
            config=config,
        )
        np.testing.assert_array_equal(result.assignment["cols"], base["cols"])
        self.assertFalse(result.candidate_evidence.empty)

    def test_apply_mode_corrects_deformed_candidate(self) -> None:
        states, frame, base = self.continuation_case()
        config = GraphTrackingConfig(
            mode="apply",
            maximum_radius_um=20,
            anchor_minimum_association_probability=0.50,
            anchor_minimum_probability_margin=0.05,
            graph_pair_weight=2.0,
            maximum_pair_cost_reduction=4.0,
            maximum_pair_cost_penalty=4.0,
        )
        result = refine_transition_with_graph(
            eligible_states=states,
            detections=frame,
            current_frame=1,
            base_assignment=base,
            volume_shape_zyx=np.asarray([20, 20, 20]),
            voxel_size_zyx_um=np.ones(3),
            config=config,
        )
        selected = {int(row): int(col) for row, col in zip(result.assignment["rows"], result.assignment["cols"])}
        self.assertEqual(selected[0], 3)
        self.assertTrue(bool(result.refinement_events.loc[result.refinement_events["entity_index"] == 0, "changed"].iloc[0]))

    def test_disabled_mode_is_identity(self) -> None:
        states, frame, base = self.continuation_case()
        result = refine_transition_with_graph(
            eligible_states=states,
            detections=frame,
            current_frame=1,
            base_assignment=base,
            volume_shape_zyx=np.asarray([20, 20, 20]),
            voxel_size_zyx_um=np.ones(3),
            config=GraphTrackingConfig(mode="disabled"),
        )
        self.assertIs(result.assignment, base)
        self.assertTrue(result.candidate_evidence.empty)


if __name__ == "__main__":
    unittest.main()
