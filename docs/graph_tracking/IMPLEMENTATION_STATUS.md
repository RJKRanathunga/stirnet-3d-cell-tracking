# Implementation status

## Implemented

- Sparse shared-node spatial graphs using `scipy.spatial.cKDTree`.
- Radius-limited and K-limited deterministic neighbourhoods.
- Reliable temporal anchors selected from current Stage 7 probabilities,
  margins, position errors, boundary state, and motion confidence.
- Vector-preserving forward Hough voting.
- Robust weighted geometric-median consensus and outlier rejection.
- Translation, similarity, and affine local deformation prediction.
- Candidate vector, radial, relative-volume, consensus, and deformation scores.
- Bounded graph cost changes for existing Stage 7 pair costs.
- Forward outside-volume exit hypotheses and miss-cost support.
- Backward outside-volume entry hypotheses and birth-cost support.
- Backward inside-volume predecessor evidence that penalizes false births.
- Locked reliable matches during graph refinement.
- Internal augmented Hungarian assignment with optional callback to the
  repository's existing solver.
- Disabled, shadow, and apply modes.
- Stable diagnostic DataFrame schemas.
- Strict Stage 7 integration-input validation.
- Integration guide and synthetic unit tests.
- Production Stage 7 integration after final base-assignment selection and
  before state mutation.
- Disabled, shadow, and apply behavior in the public Stage 7 entry point.
- Graph-supported entry/exit lifecycle events, StageTrace diagnostics, and
  backward-compatible Stage 7 artifact I/O.
- Production integration regressions for refinement, fallbacks, boundaries,
  deterministic output, safety gates, and legacy artifact loading.

## Intentionally deferred

- Persistent multi-frame neighbour-history state.
- Five-frame sliding-window graph optimization.
- Learned GNN or graph-matching model.
- Automatic threshold calibration from labelled competition data.

These deferred items should follow only after adjacent-frame shadow-mode
validation on the curated failure scenes.
