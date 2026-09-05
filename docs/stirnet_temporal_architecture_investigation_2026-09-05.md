# STIR-Net temporal architecture investigation — 2026-09-05

## Executive summary

The largest verified bottleneck was not candidate information or selector
optimization. It was a split-only partition-contract bug.

Investigation 42/46 evaluated and selected temporal edge logits under a
split-only policy, but the final graph partitioner first solved the complete
frame RAG. Only afterward did `enforce_split_only` intersect the answer with
the frozen spatial components. With the default final union-find backend, a
candidate CUT inside one current component could be bypassed by a KEEP path
through another current component. Forbidden cross-component temporal logits
therefore changed the apparent partition inside the component being scored.

This is large enough to alter the scientific conclusion:

- Raw candidate: 12/14 CUT, 260/321 KEEP, 7/12 exact merges.
- Writable logits with the old whole-frame solve: the same edge counts, but
  only 3/12 exact merges.
- Writable logits with the constraint enforced inside the solve: 7/12 exact
  merges again and zero pre-enforcement cross-component violations.
- A perfect editable-edge oracle rose from 6/12 to 12/12 exact recovery.
- The existing train-calibrated V3 edge selector rose from 3/12 to **6/12
  exact**, with identical 8/14 CUT, 319/321 KEEP, and only 1/2350 clean false
  splits. This is a real improvement obtained without retraining or validation
  threshold fitting.

The remaining bottleneck is a combination of candidate selection and candidate
generation:

- A candidate-vs-spatial edge oracle reaches 10/12 after the solver fix. Thus
  four of the six current final failures are selector false negatives.
- The other two failures are candidate-generation errors: the candidate misses
  the required CUT in frames 32 and 33.
- The candidate directly receives temporal information only after it has been
  compressed through temporal node encoding and local cross-attention. Strong
  component evidence such as relative volume and predecessor growth is not
  directly exposed to the edge-delta head.
- The Investigation-45 pre-solver association graph is not encoded. The direct
  BioHub adapter sets `hypothesis_edge_index` and `hypothesis_edge_attr` to
  `None`; only a rejected-predecessor count is injected into a reused scalar
  node channel. Removing that count caused no prediction change in this audit.

The hard cutter and stabilized coordinates are helpful, but secondary to the
partition fix. Removing hard-cutter topology reduced candidate CUT recall from
12/14 to 9/14. Raw coordinates reduced it to 11/14. Explicit physical channels
mostly improved preservation rather than merge recall. These are development
results on frames 30–39, which are not an untouched holdout.

## Current architecture and actual data flow

The evaluated path is:

1. A frozen cached spatial RAG and its current spatial partition are loaded.
   Ignored/deferred targets are removed from supervision. Temporal actions are
   allowed only on trusted RAG edges whose endpoints are already in the same
   current spatial component.
2. `InstanceTokenizer` pools D0/D1/D2 CNN features, geometry-field means and
   maxima, and shape statistics over each current spatial instance. Its
   reference point is the pooled SDF argmax.
3. The direct BioHub temporal adapter selects Trackastra detections in a
   radius-2 window. It creates 32-D detection features, 15-D accepted-link and
   same-frame-neighbour edge features, tracklet assignments, 10-D tracklet
   status, and tracklet reference positions.
4. Investigation 46 synchronously cuts suspicious one-to-one Trackastra links.
   The threshold 4.072968 was calibrated on frames 2–25 only. Temporal
   positions and velocities are expressed in target-relative,
   global-motion-stabilized coordinates.
5. Investigation 46 reuses otherwise neutral detection channels for local
   volume anomaly, positive parent growth, incoming/outgoing hard-cut flags,
   stabilized prediction error, nearby broken tracks, cutter score, plausible
   and rejected predecessor counts, and accepted-link confidence.
6. `TemporalGraphEncoder` embeds detections, message-passes over accepted links
   and local neighbours, pools detections into tracklets, and predicts temporal
   salience/reliability. The production encoder supports a structured
   tracklet-hypothesis graph, but this adapter supplies none.
7. `InstanceTemporalReasoner` cross-attends both instance tokens and RAG-node
   embeddings to nearby temporal tracklets. Its edge-delta head receives the
   frozen RAG edge embedding, left/right attended RAG-node contexts, their
   absolute difference, and the spatial edge logit. It does not receive the
   instance token, split logit, component volume evidence, or rejected-link
   hypotheses directly.
8. The temporal candidate is `spatial_logit + edge_temporal_delta`. The
   original four-scalar gate uses spatial uncertainty, left/right temporal
   support, and whether the endpoints share a provisional instance.
9. Investigation 46 V3 instead uses a trained discrete edge selector. V4 uses a
   whole-current-component spatial-vs-candidate selector.
10. Before this investigation, final partitioning ran on the entire RAG and
    split-only was imposed only after partitioning. Investigation 47 adds an
    optional parent-component constraint to `GraphPartitioner`; cross-parent
    edges are excluded before union-find or multicut is solved.

The tiled temporal inference path also omitted `stage="final"` on its last
partitioner call, so it silently used the spatial backend with a final-stage
threshold. This differed from the full-volume `StirNet.forward` path. The call
now explicitly selects the final backend and has a regression assertion.

## Baseline and corrected measurements

All rows use curated development frames 30–39. CUT/KEEP and exact values include
counts. “Pre violations” counts final components that crossed current spatial
components before the legacy post-hoc intersection. Reported split-only
violations after enforcement are zero for every row.

| Method | CUT | KEEP | Exact bad components | Clean false splits | Pre violations |
|---|---:|---:|---:|---:|---:|
| Frozen spatial | 0/14 | 321/321 | 0/12 | 0/2350 | 394 |
| Raw candidate, legacy solve (not writable) | 12/14 | 260/321 | 7/12 | 42/2350 | 200 |
| Legal logits, legacy whole-frame solve | 12/14 | 260/321 | 3/12 | 31/2350 | 401 |
| **Legal candidate, constrained solve** | **12/14** | **260/321** | **7/12** | **42/2350** | **0** |
| Current V4 component selector, constrained | 2/14 | 319/321 | 1/12 | 1/2350 | 0 |
| V3 edge selector, legacy solve | 8/14 | 319/321 | 3/12 | 1/2350 | 394 |
| **V3 edge selector, constrained solve** | **8/14** | **319/321** | **6/12** | **1/2350** | **0** |
| Candidate-choice edge oracle, legacy solve | 12/14 | 321/321 | 5/12 | 0/2350 | 395 |
| Candidate-choice edge oracle, constrained | 12/14 | 321/321 | 10/12 | 0/2350 | 0 |
| Perfect editable-edge oracle, legacy solve | 14/14 | 321/321 | 6/12 | 0/2350 | 395 |
| Perfect editable-edge oracle, constrained | 14/14 | 321/321 | 12/12 | 0/2350 | 0 |

The raw and legal candidates have identical supervised edge metrics because
those metrics already mask to editable edges. Their different exact recovery
therefore could only come from the partition contract. Across the ten frames,
the raw candidate changed 2,535 forbidden logits and changed the binary
prediction on 1,075 of them.

## Failure analysis

There are 12 curated merge components in frames 30–39.

### Contract-dependent apparent successes

Four components were exact under the raw whole-frame candidate but not under
legal logits with the legacy solver: frame/component 30/53, 31/41, 32/51, and
38/113. None has an internal non-editable edge. Their result changed because
cross-component edges created external union paths. All four become legally
exact with constrained partitioning.

### Candidate false-cut failures

Three components have every required CUT but also cut an edge belonging to one
of the target cells: 30/108, 31/37, and 34/104. The candidate-choice oracle and
the V3 selector can suppress those extra cuts. With constrained partitioning,
all three are recoverable.

### Candidate missing-cut failures

The candidate misses the only required CUT for 32/60 and 33/109. These remain
wrong even under the constrained candidate-choice oracle, but become exact
under the perfect editable-edge oracle. They are true candidate-generation
failures, not RAG/action-space failures. Both components have candidate split
probability near opposite extremes (0.978 and 0.045), demonstrating that the
split head is not a reliable coordinator for candidate edge writes.

### Selector false negatives

The constrained V3 selector exactly recovers 30/53, 30/108, 31/37, 32/51,
34/104, and 37/75. It misses four components that the candidate-choice oracle
can recover: 31/41, 31/79, 38/113, and 39/94. This is the remaining selection
headroom. Two additional components require a better candidate.

The complete per-component table, including target IDs, physical evidence,
required cuts, candidate false cuts, selector probabilities, and all oracle
flags, is in `bad_validation_components.csv` in the run directory.

## Exact-recovery metric audit

The target interpretation itself is sound for this split-only question:

- Edge supervision requires both endpoint nodes to have positive, non-ignored
  target IDs.
- A scored current component must be fully trusted; ignored/deferred nodes
  therefore cannot turn a component into a supervised positive or negative.
- Target IDs spanning more than one current component are excluded from exact
  recovery because a split-only action cannot repair that existing over-split.
- Exact recovery requires every target ID in the bad component to map to one
  unique final component, and that final component may contain no trusted node
  from another target.
- Clean false split counts whether a trusted clean current component becomes
  more than one final component.

The audit found no ignored-ID or target-ID contamination. The metric defect was
comparability: raw, legal, final, and oracle logits were passed through a solver
whose admissible graph did not match the split-only action contract. Reporting
only the post-intersection violation count hid this mismatch.

## Experiments performed

### 1. Candidate contract audit

Hypothesis: raw exact recovery uses logits that the split-only writer cannot
legally apply.

Change: Investigation 47 computes raw and `where(editable, candidate, spatial)`
logits separately and records forbidden changes.

Result: edge metrics were identical, while exact recovery fell from 7/12 to
3/12 under the legacy partitioner. The hypothesis was verified, but the deeper
cause was the ordering of partitioning and constraint enforcement.

### 2. In-solver parent-component constraint

Hypothesis: post-hoc intersection does not prevent external graph paths from
changing a split-only solution.

Change: `GraphPartitioner.forward` now accepts optional
`node_parent_component`. Edges crossing parent components are excluded before
either union-find or multicut. Existing callers are unchanged unless they opt
in.

Result: the legal candidate recovered 7/12 instead of 3/12. The perfect
editable-edge oracle recovered 12/12 instead of 6/12. Pre-enforcement
violations fell to zero. Targeted tests cover union-find, multicut, validation,
and the external-bridge counterexample.

### 3. Existing V3 selector under the corrected solver

Hypothesis: the trained edge selector is better than previously reported, but
its partition was corrupted by external paths.

Change: none to weights or threshold. The V3 step-175 checkpoint and its
train-only threshold 0.594719 were reused exactly.

Result: exact recovery doubled from 3/12 to 6/12. CUT stayed 8/14, KEEP stayed
319/321, and clean false splits stayed 1/2350. This is the best precise final
result in the audit.

### 4. Frozen temporal-input ablations

Hypothesis: hard-cutter topology, stabilization, explicit physical evidence,
and rejected-predecessor count contribute differently to the candidate.

No weights or thresholds were fitted. The same checkpoint was evaluated after
one input intervention at a time. These are the constrained-solver results:

| Input | Candidate CUT / KEEP | Candidate exact / clean split | V3 final CUT / KEEP | V3 final exact / clean split |
|---|---:|---:|---:|---:|
| Full cutter + stabilized + physical | 12/14 / 260/321 | 7/12 / 42/2350 | 8/14 / 319/321 | 6/12 / 1/2350 |
| No hard-cutter topology | 9/14 / 282/321 | 5/12 / 28/2350 | 7/14 / 320/321 | 5/12 / 1/2350 |
| Raw coordinates + physical | 11/14 / 257/321 | 6/12 / 43/2350 | 7/14 / 319/321 | 5/12 / 1/2350 |
| Stabilized, no explicit physical | 12/14 / 257/321 | 7/12 / 44/2350 | 8/14 / 319/321 | 6/12 / 1/2350 |
| Stabilized physical, no rejected count | 12/14 / 260/321 | 7/12 / 42/2350 | 8/14 / 319/321 | 6/12 / 1/2350 |

Removing hard-cutter topology reduced candidate CUT recall from 12/14 to 9/14
and final exact recovery from 6/12 to 5/12. Reverting to raw coordinates reduced
candidate CUT recall to 11/14 and final exact to 5/12. Removing all explicit
physical channels kept CUT recall but caused three more candidate KEEP errors
and two more clean splits. Removing only rejected-predecessor count caused no
change at all.

This supports hard cutting and stabilization. It does not support the claim
that the current scalar rejected-predecessor count materially affects the
frozen candidate.

### 5. Optimization audit of existing artifacts

No additional expensive training sweep was justified after the contract bug
was found. Existing histories were inspected:

- V2 candidate tuning had finite, substantial gradients (roughly 3.5–8.9 in
  the ten-step run) but reduced candidate CUT recall, so this is not a
  vanishing-gradient explanation.
- V3 trained on 40 HELP and 81 SUPPRESS edges. Its train-only calibration had
  AUC 0.9238, HELP recall 0.775, and harmful-write rate 0.0864.
- V4 trained on only 11 HELP and 57 SUPPRESS components. Its train AUC was
  0.9761, yet validation used only one of seven raw-candidate helpful
  components. More importantly, V4 labels were generated from the invalid raw
  whole-frame candidate partition. Those labels should not be trusted until
  regenerated with constrained partitioning.

## Architecture changes

### Production utility

`learned/stirnet/model/partition/partitioner.py`

- Added optional `node_parent_component` to `GraphPartitioner.forward`.
- Validates shape and integer dtype.
- Filters RAG edges crossing parent components before union-find or multicut.
- Default behavior is unchanged when the argument is omitted.

This is a general, hard partition constraint rather than a BioHub threshold or
heuristic. It should be used whenever a correction stage is contractually
split-only relative to an existing partition.

### Tests

`learned/stirnet/tests/test_multicut_partitioner.py`

- Added an explicit external union-path regression test.
- Added multicut coverage.
- Added argument shape/dtype validation coverage.

`learned/stirnet/inference/tiled_dense.py` and
`learned/stirnet/tests/test_v2_tiled_inference.py`

- Fixed the tiled temporal path to request `stage="final"` explicitly.
- Added a regression assertion that the last tiled partition uses the final
  backend.

### Experiment

`investigations/stirnet/47_biohub_temporal_candidate_contract_audit.py`

- Reconstructs the current curated data, motion, hard cutter, candidate, V3
  selector, and V4 selector from existing artifacts.
- Reports spatial/raw/legal/current/oracle metrics under legacy and constrained
  solvers.
- Emits component diagnostics and temporal-input ablations.

No parameter count, learned weight, threshold, annotation, cache, or checkpoint
was changed.

## Best result obtained

The best precise final method is the existing V3 discrete edge selector plus
the new in-solver parent-component constraint:

- CUT: **8/14 = 57.14%**
- KEEP: **319/321 = 99.38%**
- Exact bad-component recovery: **6/12 = 50.0%**
- Clean false-split rate: **1/2350 = 0.0426%**
- Split-only violations: **0**, including before post-processing

Compared with the exact same logits and threshold under the legacy solver,
exact recovery improved from 3/12 to 6/12 without changing edge accuracy or
clean preservation.

The high-recall candidate remains 12/14 CUT and 7/12 exact, but its 260/321
KEEP and 42/2350 clean false splits are not yet acceptable as a final method.

## What did not work or was not supported

- Merely replacing forbidden candidate logits with spatial logits is not
  sufficient. The whole-frame solver can still route through other spatial
  components.
- Post-hoc split-only intersection reports zero final violations while hiding
  hundreds of invalid pre-intersection joins. The old violation metric was
  therefore necessary but not diagnostic.
- The V4 whole-component selector remains at 1/12 exact after the solver fix.
  It is both too coarse for candidate false-cut cases and trained from invalid
  raw-partition labels.
- Candidate fine-tuning did not fail because gradients were absent; it moved a
  useful initializer in the wrong direction under a small, imbalanced dataset.
- The current rejected-predecessor scalar count had no measurable frozen-model
  effect. It is not equivalent to encoding the rejected association graph.
- A perfect selector cannot repair the two candidate missing-CUT cases.

## Recommended next architecture

Promote now:

1. Use the parent-component-constrained partitioner for every explicitly
   split-only temporal correction path. Do not rely on post-hoc intersection.
2. Keep the V3 edge selector as the current precise final method; its labels are
   edge-local and were not invalidated by the partition bug.
3. Keep hard-cutter topology and stabilized positions/velocities.

Keep experimental:

1. Retrain/evaluate selectors only after recomputing all partitions and labels
   with the constrained solver. V4's current component labels are invalid.
2. Replace the binary whole-candidate component choice with a hierarchical
   decision: a component-level merge-worthiness head followed by coordinated
   edge/partition scoring. The constrained candidate-choice oracle (10/12)
   establishes substantial headroom for this design.
3. Expose physical component evidence directly to the candidate edge head, for
   example by broadcasting a learned component evidence token to its RAG edges.
   Avoid relying on reused temporal-node channels to survive several pooling
   stages.
4. Build and encode the actual pre-solver association-hypothesis graph rather
   than only a rejected-count scalar. The production temporal encoder already
   supports `hypothesis_edge_index`/`hypothesis_edge_attr`; the BioHub direct
   adapter currently discards this input.
5. Give the two candidate-miss cases targeted attention. Both need one CUT, but
   their split probabilities are 0.978 and 0.045, so training the existing split
   head alone is unlikely to coordinate them reliably.

Most valuable additional data:

- More curated merge components across volumes, with divisions and boundary
  artifacts explicitly represented.
- A fresh untouched volume/frame range for final threshold and architecture
  selection.
- Reviewed predecessor alternatives, not merely final accepted Trackastra
  links, so structured hypothesis supervision is possible.

## Reproducibility

Main audit and ablations:

```powershell
.\.venv\Scripts\python.exe .\investigations\stirnet\47_biohub_temporal_candidate_contract_audit.py
```

Canonical-only rerun:

```powershell
.\.venv\Scripts\python.exe .\investigations\stirnet\47_biohub_temporal_candidate_contract_audit.py --skip-ablations
```

Verification:

```powershell
.\.venv\Scripts\python.exe -m pytest .\learned\stirnet\tests\test_multicut_partitioner.py -q
.\.venv\Scripts\python.exe -m pytest .\learned\stirnet\tests\test_v2_model.py -q
.\.venv\Scripts\python.exe -m pytest .\learned\stirnet\tests\test_v2_tiled_inference.py -q
```

The first non-escalated pytest attempt could not write napari's user cache and
failed during plugin setup. Rerunning with permission to write that generated
cache produced 9/9, 4/4, and 5/5 passing tests respectively (18 total).

Checkpoints:

- Temporal initializer:
  `runs/stirnet/investigations/42_biohub_curated_temporal_finetune/44b6_0113de3b/best.pt`
- V3 edge selector:
  `runs/stirnet/investigations/46_biohub_hard_cutter_temporal_training_v3/44b6_0113de3b/best_selector_v3.pt`
- V4 component selector:
  `runs/stirnet/investigations/46_biohub_hard_cutter_temporal_training_v4/44b6_0113de3b/best_component_selector_v4.pt`

Outputs:

- `runs/stirnet/investigations/47_biohub_temporal_candidate_contract_audit/44b6_0113de3b/contract_audit_summary.json`
- `runs/stirnet/investigations/47_biohub_temporal_candidate_contract_audit/44b6_0113de3b/canonical_metrics.csv`
- `runs/stirnet/investigations/47_biohub_temporal_candidate_contract_audit/44b6_0113de3b/bad_validation_components.csv`
- `runs/stirnet/investigations/47_biohub_temporal_candidate_contract_audit/44b6_0113de3b/candidate_input_ablations.csv`

The canonical evaluation takes about 88–111 seconds on the available RTX 4050
Laptop GPU after cached setup. The final full five-mode audit took 465.5 seconds
(7 minutes 46 seconds), including shared setup.
No git-mutating operation was performed, and no commit was created.
