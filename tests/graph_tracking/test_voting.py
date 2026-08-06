from __future__ import annotations

import unittest

import numpy as np

from graph_tracking import GraphTrackingConfig
from graph_tracking.spatial_graph import build_spatial_graph
from graph_tracking.types import TemporalAnchor
from graph_tracking.voting import build_forward_vote_bundle, evaluate_candidate, robust_vote_consensus


def graph(frame: int, positions: list[list[float]], ids: list[int]):
    return build_spatial_graph(
        frame=frame,
        node_ids=np.asarray(ids),
        positions_zyx_um=np.asarray(positions, dtype=float),
        volumes=np.full(len(ids), 100.0),
        touches_boundary=np.zeros(len(ids), dtype=bool),
        boundary_faces=tuple("" for _ in ids),
        config=GraphTrackingConfig(maximum_radius_um=20, maximum_neighbors=10),
    )


class VotingTests(unittest.TestCase):
    def test_forward_votes_preserve_relative_vectors(self) -> None:
        source = graph(0, [[5, 5, 5], [5, 2, 5], [5, 8, 5], [2, 5, 5]], [0, 1, 2, 3])
        target = graph(1, [[5, 2, 7], [5, 8, 7], [2, 5, 7], [5, 5, 7], [5, 5, 10]], [0, 1, 2, 3, 4])
        anchors = {
            1: TemporalAnchor(1, 1, 0, 0.9, 0.3, 0.5, 1.0),
            2: TemporalAnchor(2, 2, 1, 0.9, 0.3, 0.5, 1.0),
            3: TemporalAnchor(3, 3, 2, 0.9, 0.3, 0.5, 1.0),
        }
        config = GraphTrackingConfig(maximum_radius_um=20, vote_kernel_sigma_um=2.0)
        bundle = build_forward_vote_bundle(source_graph=source, target_graph=target, source_node_index=0, anchors_by_source_index=anchors, config=config)
        self.assertIsNotNone(bundle)
        np.testing.assert_allclose(bundle.consensus.position_zyx_um, [5, 5, 7], atol=1e-6)
        correct = evaluate_candidate(bundle=bundle, source_graph=source, target_graph=target, candidate_node_index=3, config=config)
        wrong = evaluate_candidate(bundle=bundle, source_graph=source, target_graph=target, candidate_node_index=4, config=config)
        self.assertGreater(correct.vote_score, wrong.vote_score)
        self.assertGreater(correct.vector_score, wrong.vector_score)

    def test_outlier_vote_is_rejected(self) -> None:
        config = GraphTrackingConfig(vote_inlier_radius_um=2.0)
        votes = np.asarray([[0, 0, 0], [0.2, 0, 0], [-0.1, 0, 0], [20, 0, 0]], dtype=float)
        consensus = robust_vote_consensus(votes, np.ones(4), config)
        self.assertEqual(int(consensus.inlier_mask.sum()), 3)
        self.assertLess(np.linalg.norm(consensus.position_zyx_um), 0.2)


if __name__ == "__main__":
    unittest.main()
