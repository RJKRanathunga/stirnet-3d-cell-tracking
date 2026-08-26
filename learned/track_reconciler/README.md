# Learned Track Reconciler

`learned/track_reconciler` is a conservative learned repair layer for the project's primary tracker. It is **not** a replacement frame-to-frame tracker. The package assumes that Trackastra (or another primary tracker) already produced mostly correct trajectories and that suspicious internal edges have been split into **high-purity tracklets**.

The central design principle is the same one that made supervoxels valuable in STIR-Net: make the learned decision unit reliable and the candidate search small before asking a neural network to reason.

## Architecture

```text
cell crop (raw/mask/EDT/SDF/context) -> small 3-D fingerprint CNN
                                      |
structured per-cell history ----------+-> separate temporal Transformers
                                           -> incoming HEAD / outgoing TAIL states

high-recall deterministic candidate gate
             |
             v
candidate A -> B
  - A tail, B head
  - explicit global-only expected position
  - explicit global+relative expected position
  - explicit local-anchor expected position
  - backward expected source position
  - residual vectors and uncertainty
  - morphology/intensity/volume evidence
  - Trackastra/Stage-7 evidence
  - source + target ambiguity
             |
             v
edge token -> 4-layer geometry-biased EDGE Transformer
             |
             +-> continuation logits + parental softmax
             +-> sparse symmetric division hypotheses
             +-> appearance / termination / division priors
             |
             v
SciPy MILP decoder -> legal lineage graph
```

### Why edge-centric?

HOCT (Bragantini, Theodoro & Royer, arXiv:2607.11754, 2026) shows that candidate tracking graphs are strongly non-homophilic: edges sharing a cell are mostly competing alternatives rather than labels that should be averaged by a conventional node GNN. This package therefore reasons over **candidate transition edges**. Each edge-attention layer receives the exact minimum distance between its finite 3-D line segment and every other candidate segment, with alternating attractive and repulsive attention heads.

### Why separate temporal streams?

Motion/geometry and appearance have different reliability failure modes. Tiny cells may have weak morphology while a long trajectory has excellent kinematics; a short/new tracklet may have the opposite situation. The network therefore encodes fingerprints and structured history separately and fuses them with tracklet reliability.

### Head and tail states

A tracklet is directional. The same pooled token is not used for both roles:

- `head`: contextualized early state, used when the tracklet is a successor.
- `tail`: contextualized recent state, used when the tracklet is a predecessor.
- `pooled`: global tracklet summary used for event priors.

### Division is an explicit event

A division is not inferred from two independent continuation scores. Only the top plausible child edges are paired; a symmetric head scores `parent -> {child1, child2}` using the refined edge embeddings, parent division prior, and explicit biological pair features. Swapping daughter order cannot alter the logit.

### Global legality is deterministic

`MILPDecoder` does not ask the network to learn constraints that are already known. Every high-purity tracklet receives exactly one incoming explanation (appearance, continuation, division-child) and one outgoing explanation (termination, continuation, division-parent). This directly excludes many-to-one fusion and incompatible competing events.

## Existing Stage-11 integration

`integration/stage11.py` defines 40 primitive features already available in the current reconciliation tables. It intentionally excludes the handcrafted final `continuation_score` from the default model input.

Current Stage 11 already stores the combined forward prediction (`forward_predicted_*`), local-anchor prediction, and backward prediction. The adapter can compute and persist the newly requested **global-only expected position** as:

```text
global_predicted_z_um
global_predicted_y_um
global_predicted_x_um
```

Use `add_global_only_predictions(...)` to generate these columns from Stage-11 observations and global motion. If they are still absent, the adapter marks that expert invalid; it never silently substitutes another prediction. Pair input contains 40 values plus 40 per-feature availability flags (80 channels total), so missing evidence cannot masquerade as a perfect zero-error match.

## Fingerprint crop channels

Default `in_channels=5`; recommended physical-FOV crop channels are:

1. normalized raw intensity
2. target-cell binary mask
3. target-cell EDT
4. boundary / signed-distance representation
5. surrounding-instance occupancy

STIR foreground/separator maps can be appended by increasing `FingerprintConfig.in_channels`. Crop extraction should use a fixed **physical field of view** because BioHub voxels are anisotropic.

## Training notes

- Precompute fingerprints when I/O/memory dominates; `TrackletBatch` accepts either crops or precomputed embeddings.
- Fit scalar feature normalization on the training split only.
- Over-sample true Trackastra breaks, near ties, swaps, gap closures and divisions; do not drown them in trivial links.
- Use hard nearby negatives, not random cells.
- Add the fingerprint metric loss using reliable same-track positives and nearby different-cell negatives.
- Use modality/feature-group dropout so the network cannot simply copy Trackastra or the old heuristic.
- Calibrate final event logits on held-out data (`TemperatureScaler`) before thresholding automatic corrections.

## Smoke test

From repository root after copying the folder under `learned/`:

```bash
python -m learned.track_reconciler.smoke_test
pytest learned/track_reconciler/tests -q
```

No new runtime dependencies are required beyond the repository's existing PyTorch, NumPy, pandas and SciPy stack.


### Windows pytest import bootstrap

The test suite includes `tests/conftest.py` so the repository root is inserted into `sys.path` when tests are launched through the Windows `pytest.exe` console entry point. This is necessary because the current repository `pyproject.toml` installs `src` and `diagnostics`, but not `learned`. It does not affect runtime model imports.
