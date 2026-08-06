from __future__ import annotations

import unittest

import numpy as np

from graph_tracking import GraphTrackingConfig, refine_transition_with_graph
from helpers import base_assignment, detections, state


class BoundaryTests(unittest.TestCase):
    def config(self, mode="shadow"):
        return GraphTrackingConfig(
            mode=mode,
            maximum_radius_um=20,
            anchor_minimum_association_probability=0.50,
            anchor_minimum_probability_margin=0.02,
            boundary_minimum_consensus=0.20,
            boundary_minimum_directional_agreement=0.50,
            outside_vote_minimum_distance_um=0.1,
            outside_vote_strong_distance_um=1.0,
        )

    def test_forward_votes_support_exit(self) -> None:
        states = [
            state(0, [5, 5, 9.5], boundary=True),
            state(1, [5, 3, 8]),
            state(2, [5, 7, 8]),
        ]
        frame = detections([
            ([5, 3, 10], False, "", 100),
            ([5, 7, 10], False, "", 100),
        ])
        invalid = np.ones((3, 2), dtype=bool)
        invalid[1, 0] = False
        invalid[2, 1] = False
        pair = np.full((3, 2), 1e6)
        pair[1, 0] = pair[2, 1] = 0.05
        distance = np.full((3, 2), 99.0)
        distance[1, 0] = distance[2, 1] = 0.2
        base = base_assignment(pair, invalid, [0.5, 5, 5], [5, 5], distance, self.config())
        result = refine_transition_with_graph(
            eligible_states=states,
            detections=frame,
            current_frame=1,
            base_assignment=base,
            volume_shape_zyx=np.asarray([11, 11, 11]),
            voxel_size_zyx_um=np.ones(3),
            config=self.config(),
        )
        exits = result.boundary_hypotheses[result.boundary_hypotheses["event_type"] == "exit"]
        self.assertFalse(exits.empty)
        self.assertEqual(exits.iloc[0]["boundary_face"], "x_max")
        self.assertTrue(bool(exits.iloc[0]["supported"]))

    def test_backward_votes_support_entry(self) -> None:
        states = [
            state(1, [5, 3, 0]),
            state(2, [5, 7, 0]),
        ]
        frame = detections([
            ([5, 3, 2], False, "", 100),
            ([5, 7, 2], False, "", 100),
            ([5, 5, 0.5], True, "x_min", 100),
        ])
        invalid = np.ones((2, 3), dtype=bool)
        invalid[0, 0] = False
        invalid[1, 1] = False
        pair = np.full((2, 3), 1e6)
        pair[0, 0] = pair[1, 1] = 0.05
        distance = np.full((2, 3), 99.0)
        distance[0, 0] = distance[1, 1] = 0.2
        base = base_assignment(pair, invalid, [5, 5], [5, 5, 0.5], distance, self.config())
        result = refine_transition_with_graph(
            eligible_states=states,
            detections=frame,
            current_frame=1,
            base_assignment=base,
            volume_shape_zyx=np.asarray([11, 11, 11]),
            voxel_size_zyx_um=np.ones(3),
            config=self.config(),
        )
        entries = result.boundary_hypotheses[result.boundary_hypotheses["event_type"] == "entry"]
        self.assertFalse(entries.empty)
        self.assertEqual(entries.iloc[0]["boundary_face"], "x_min")
        self.assertTrue(bool(entries.iloc[0]["supported"]))


if __name__ == "__main__":
    unittest.main()
