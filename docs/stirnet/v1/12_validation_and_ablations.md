# 12 — Validation, Ablations, and V2 Triggers

## 1. Validation philosophy

STIR-Net is a correction model.

The most important question is not:

> Does voxel accuracy improve?

It is:

> Does the model correct real instance errors without damaging cells that were already correct?

## 2. Primary segmentation metrics

Report:

```text
instance precision
instance recall
instance F1
matched mask Dice
matched mask IoU
center precision
center recall
center localization error in µm
count MAE
```

## 3. Failure-specific metrics

### Under-segmentation recovery

Measure recall separately for:

```text
2 -> 1 merges
3 -> 1 merges
4+ clusters
```

A merge is recovered if the correct number of usable cells is reconstructed with acceptable center/mask matching.

### Over-segmentation recovery

Fraction of oversplit GT cells reconstructed as one biological instance.

### Missing-cell recovery

Recall of GT cells absent from the input segmentation.

### False-positive removal

Fraction of input artifacts correctly removed.

### Boundary correction

Matched-mask Dice/IoU improvement on cells whose identity/count was already correct but boundary was wrong.

## 4. No-harm metrics

These are mandatory.

```text
clean-instance preservation rate
false correction rate
false split rate
false merge rate
correct-cell Dice before vs after
```

A refinement model that improves rare failures while damaging many correct cells is unacceptable.

## 5. Downstream metric

After refinement:

```text
corrected instances
 |
Trackastra pass 2
```

measure final tracking/competition metrics.

This is an evaluation metric, not a training loss.

## 6. Runtime metrics

Record:

```text
GPU memory
seconds/frame
seconds/sequence
queries/tile
full masks rendered/tile
Trackastra pass-1 overhead
Trackastra pass-2 overhead
```

V1 should be compared against the current pipeline end to end.

## 7. Required ablation ladder

### A — ordinary native CNN

```text
ordinary CNN
no temporal input
query decoder
```

Purpose:

baseline.

### B — physical-aware CNN

```text
spacing-conditioned axis-factorized CNN
no temporal input
query decoder
```

Purpose:

test native-spacing strategy.

### C — simple temporal priors

```text
physical-aware CNN
temporal information rasterized/simple conditioned
no graph cross-attention
```

Purpose:

test whether temporal clues help at all.

### D — graph encoder + simple fusion

```text
physical-aware CNN
GATv2 temporal encoder
simple feature conditioning
```

Purpose:

test graph representation without complex co-reasoning.

### E — one co-reasoning block

Purpose:

measure value of actual cross-modal reasoning.

### F — full V1

```text
two co-reasoning blocks
salience/reliability
full query system
```

## 8. Component ablations

Test independently:

```text
with/without split companions
with/without temporal repair queries
with/without discovery queries
with/without marker channel
with/without physical EDT
with/without salience
with/without reliability
with/without acquisition conditioning
with/without initial mask priors
```

## 9. Temporal clue ablations

Measure separately on:

```text
stable complete tracks
interior starts
interior ends
gaps
division neighbourhoods
boundary exits
```

We should verify that anomaly salience actually improves hard areas rather than merely adding complexity.

## 10. Multi-dataset generalization tests

Evaluate:

1. train/test within one dataset;
2. train on several datasets, test on held-out sample;
3. leave-one-dataset-out generalization;
4. synthetic spacing perturbation;
5. strongly anisotropic versus isotropic acquisition.

The purpose is to test whether native-resolution physical-coordinate handling avoids the information-loss/generalization tradeoff of global resampling.

## 11. Acceptance criteria before V2

Do not begin major V2 architecture work until V1 establishes:

- stable training;
- correct query matching;
- low clean-cell damage;
- clear improvement over spatial-only baseline on real failures;
- measurable value from temporal clues;
- acceptable runtime.

## 12. V2 triggers

### Trigger: spacing-conditioned CNN fails across acquisition geometries

Consider:

```text
physical-coordinate deformable sampling
```

where learned offsets are expressed in µm/cell-scale units and native feature maps are sampled with `grid_sample`.

### Trigger: long-range spatial interactions remain unresolved

Consider an additional sparse/global object-level attention layer, not a full voxel transformer.

### Trigger: discovery queries fail to find fully missed cells

Consider:

- center-proposal generation;
- denser learned discovery queries;
- dedicated coarse objectness proposal head.

### Trigger: complex 3+ merges fail

Consider:

- more split companions per suspicious instance;
- adaptive query spawning;
- set-query decoder extensions.

### Trigger: temporal priors remain weak

Consider feeding richer Trackastra internal embeddings/association probabilities if accessible and stable.

### Trigger: boundary quality is limited despite correct counts/centers

Consider:

- higher-capacity native decoder;
- physical deformable convolution;
- explicit signed-distance/boundary-vector auxiliary targets.

## 13. Changes that are not justified by novelty alone

Do not add:

```text
Transformer everywhere
more attention heads
deeper GNN
full 4D CNN
more hand-crafted channels
tracking loss
```

unless an ablation or failure analysis identifies the missing dependency.

## 14. Architecture-change process

Any V1 architecture change should follow:

```text
observed failure
 |
hypothesis
 |
small targeted experiment
 |
ablation result
 |
documentation update
 |
implementation update
```

The documentation is the source of truth for the intended model behavior.

## Historical-evidence validation

The required ladder is:

```text
A current baseline (history disabled; new dynamics zeroed)
B enriched hypothesis dynamics only
C B + detection history encoder/fusion
D C + temporal-to-spatial history support bias
E full production configuration, when distinct from D
```

Report 2->1, 3->1, and larger-cluster recovery; clean-cell preservation; false
split rate; instance precision/recall/F1; matched Dice; count and center error;
pass-2 tracking when available; runtime; and CUDA peak allocated/reserved memory.
Stratify by converging versus non-converging tracks, history absent/past-only/
future-only/both, and strong versus weak component overlap.

Unit validation covers physical-spacing invariance, invalid-history neutrality,
chunk equivalence, translational coordinates, nearest support selection,
support-based component assignment, 22-D reverse edges, convergence sign,
flips/dropout/false clues, empty/no-history forward, curriculum ownership,
finite backward, and checkpoint migration.

`history_overfit.py` trains the A-D ladder on one ambiguous two-cell merge and
reports predicted count, matched-positive proxy, existence, center error,
coarse Dice, and native Dice rather than treating total loss as success.
`history_memory_profile.py` resets CUDA peaks per trial and reports node,
hypothesis, and query counts; compact tensor bytes; spatial feature shapes; and
peak memory. Initial marginal allocated-memory target is 0.5-0.7 GB. If it is
exceeded, optimize checkpointing, chunks, dtype, and chunk-local sampling before
reducing support channels or the 12-cube resolution.

## 17. Hierarchical temporal-memory validation

`test_v1_temporal_memory.py` covers complete directed candidate topology,
accepted-relation marking, tracklet invariance, no cross-batch edges, explicit
safety errors, node retention, permutation equivariance, empty/missing memory,
split-slot attention diversity, query-to-history gradient reachability,
checkpoint migration, and curriculum ownership.

`debugging/acceptance/hierarchical_memory.py` trains one synthetic converging
two-cell/current-merge scene whose accepted graph explains only one branch, then
evaluates the same weights under full, zero/shuffled node, tracklet-only,
node-only, and accepted-graph-only modes. It reports split attention and
per-layer sibling center separation. `backward_gate.py` accepts the same memory
and graph ablation switches for the uncropped real scene and reports path-level
gradient sums plus per-phase CUDA peaks.
