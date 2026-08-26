import numpy as np

from learned.track_reconciler.graph.decoder import DecoderProblem, MILPDecoder


def test_milp_prefers_legal_division_over_competing_continuations():
    # Tracklet 0 can either continue to one child or divide into 1+2.
    problem = DecoderProblem(
        num_tracklets=3,
        continuation_source=np.array([0, 0]),
        continuation_target=np.array([1, 2]),
        continuation_utility=np.array([2.0, 1.8]),
        division_parent=np.array([0]),
        division_child_a=np.array([1]),
        division_child_b=np.array([2]),
        division_utility=np.array([5.0]),
        appearance_utility=np.array([0.0, -1.0, -1.0]),
        termination_utility=np.array([-1.0, 0.0, 0.0]),
    )
    result = MILPDecoder().solve(problem)
    assert result.division_selected.tolist() == [True]
    assert result.continuation_selected.tolist() == [False, False]
    assert result.appearance_selected[0]
    assert result.termination_selected[1] and result.termination_selected[2]
