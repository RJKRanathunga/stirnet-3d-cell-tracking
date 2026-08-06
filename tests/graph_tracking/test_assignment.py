from __future__ import annotations

import unittest

import numpy as np

from graph_tracking.assignment import augmented_assignment, bipartite_components


class AssignmentTests(unittest.TestCase):
    def test_match_miss_birth_solution(self) -> None:
        result = augmented_assignment(
            np.asarray([[0.1, 10.0], [10.0, 10.0]]),
            np.asarray([2.0, 0.2]),
            np.asarray([2.0, 0.3]),
        )
        self.assertEqual(list(zip(result["rows"], result["cols"])), [(0, 0)])
        self.assertEqual(result["missed_state_indices"].tolist(), [1])
        self.assertEqual(result["birth_detection_indices"].tolist(), [1])

    def test_components(self) -> None:
        valid = np.asarray([[True, False, False], [True, False, False], [False, False, True]])
        components = bipartite_components(valid, {0, 1, 2}, {0, 1, 2})
        normalized = {(tuple(tracks), tuple(detections)) for tracks, detections in components}
        self.assertIn(((0, 1), (0,)), normalized)
        self.assertIn(((2,), (2,)), normalized)
        self.assertIn(((), (1,)), normalized)


if __name__ == "__main__":
    unittest.main()
