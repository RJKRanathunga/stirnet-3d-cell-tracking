# Architecture and information-flow contract

## 1. Atomic reasoning unit: high-purity tracklet

A primary-track association is allowed to become an atomic tracklet only after an internal confidence audit. Low-confidence, low-margin or otherwise suspicious primary-track edges should be cut first. This ensures the reconciler can repair a bad association instead of hiding it inside an indivisible tracklet.

This package does not dictate the audit threshold because it must be calibrated on the project's actual Trackastra score distribution. The invariant is architectural: **uncertain internal associations are boundaries, not immutable history**.

## 2. Per-observation representation

The image branch produces a compact cell fingerprint. The structured branch should include normalized primitives such as:

- physical displacement / velocity / acceleration summaries
- global shift and relative motion
- volume and log-volume trend
- axis lengths, elongation, flatness, anisotropy, solidity, compactness
- intensity mean/median/std/IQR/CV
- boundary distance / boundary flags
- association quality and observation reliability

Absolute physical location is kept out of identity-like structured features where possible; absolute z,y,x is retained separately for candidate geometry, boundary modeling and positional encoding.

## 3. Temporal encoding

Appearance and structured sequences receive independent 2-layer temporal Transformers. Missing history/future is represented with masks. For every tracklet the model exposes:

- HEAD: incoming/early representation
- TAIL: outgoing/recent representation
- POOLED: whole-tracklet representation

Reliability-conditioned gates fuse the two modalities only after each has had the chance to model its own temporal evolution.

## 4. Candidate construction

Candidate construction is deterministic and high recall. For source A at candidate gap dt, take the union of targets near:

- A endpoint
- expected position from global shift only
- expected position from global shift + A relative motion
- expected position from persistent local-neighbour motion
- externally supplied primary-tracker candidates

This turns a global search problem into a small local multiple-choice problem.

## 5. Candidate edge representation

For A -> B, the model receives both role-specific tracklet tokens and primitive relation features. The motion feature builder preserves:

- direct A->B displacement vector
- global-only prediction vector
- global+relative prediction vector
- local prediction vector
- residual vector to each prediction
- backward source-prediction residual
- bidirectional agreement
- temporal gap and validity

Expected positions are therefore first-class evidence, not just pre-scored distances.

## 6. Edge-centric geometric reasoning

The candidate transition itself is the Transformer token. Four attention layers compare edge tokens globally within a small reconciliation component. Queries and keys receive 3-D rotary position encoding at the edge midpoint. Attention receives an additional per-head bias proportional to the exact finite-segment distance between candidate transition lines.

Alternating heads are fixed as attractive or repulsive; only their non-negative magnitude is learned. This follows the stability argument and ablation evidence in HOCT rather than allowing each head to arbitrarily flip the geometry prior.

## 7. Source/target competition

Continuation logits are normalized by target tracklet and temporal gap with a unit no-parent state. Thus weak candidate sets remain weak rather than becoming artificially confident through ordinary softmax normalization.

Source-side conflicts are handled jointly with target-side conflicts by the edge Transformer and exactly by the final ILP.

## 8. Division reasoning

A parent tracklet has an independent division prior from its temporal state. Only top candidate child edges are paired. The pair head is permutation invariant and combines:

- parent tail representation
- refined A->B and A->C edge representations
- edge sum, absolute difference and product
- parent division prior
- explicit pair features such as combined-volume conservation, daughter balance, daughter separation, branch angle, midpoint prediction residual and fingerprint relations

This keeps the daughter-pair decision easy: candidate search and edge reasoning have already done most of the work.

## 9. Event decoder

The neural network emits utilities/logits for:

- CONTINUE(A,B)
- DIVIDE(A,B,C)
- APPEAR(B)
- TERMINATE(A)

The MILP enforces exactly one incoming and one outgoing explanation per accepted tracklet. A division event consumes one outgoing parent decision and two incoming child decisions. This yields legal lineage structure without asking the neural network to learn bookkeeping constraints.

## 10. Uncertainty / human-in-the-loop

Raw logits should be calibrated on a held-out correction set. At inference, uncertainty can be estimated from:

- calibrated event probability / entropy
- parental no-match probability
- source and target alternative margins
- best feasible ILP solution vs best constrained alternative (future extension)

Only high-confidence repairs should be applied automatically. Low-margin regions should be surfaced to the annotation UI.

## Research decisions used

- Bragantini, Theodoro & Royer, *Higher-Order Cell Tracking Transformer*, 2026: edge-centric attention, finite segment distance bias, attractive/repulsive heads, per-gap parental softmax, high-purity two-pass tracklets + ILP.
- Cetintas et al., *SUSHI*, CVPR 2023: conservative tracklet hierarchy for long-term association.
- OrganoidTracker 2.0, Nature Methods 2025: learned 3-D link evidence, explicit temporal division evidence, calibrated uncertainty and global graph optimization.
- Ultrack, Nature Methods 2025: explicit global optimization under biological constraints.
- Cell-tracking GNN / metric-learning work (ECCV 2022): learned crop embeddings and hard-negative identity supervision.
