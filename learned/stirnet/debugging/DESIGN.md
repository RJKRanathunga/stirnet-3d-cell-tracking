# STIR-Net debugger design

## 1. Purpose

The debugger is an observability layer for STIR-Net V1. It must answer *where*
a prediction becomes wrong without changing model numerics or creating a second
implementation of the network.

The standard diagnostic path is:

```text
input scene
  -> spatial encoder E0..E3
  -> temporal graph / tracklet state
  -> co-reasoning CR1
  -> decoder to E2
  -> co-reasoning CR2
  -> query builder
  -> query decoder L1 -> L2 -> L3
  -> learned native mask + query prior
  -> final inference/postprocessing
```

## 2. Observation strategy

### Spatial branch

Capture bounded statistics for E0..E3 and decoder stages. In deep mode, save
only the explicit dense foreground, center-heatmap and boundary probabilities
as full scene arrays.

### Temporal branch

Trace temporal hypothesis tokens at construction, after CR1 and after CR2.
Report references, salience, reliability, token norms and cosine changes.

### Query branch

Use final Hungarian matching as an identity anchor. For every query, trace:

- query type and source instance;
- initial reference;
- existence probability after each decoder layer;
- center after each decoder layer;
- center error to matched GT after each decoder layer;
- coarse soft Dice after each decoder layer;
- final survival at the inference existence threshold.

### Native mask branch

For a small selected query set, decompose the native mask into:

```text
learned logits = native_mask_embedding dot mask_features
prior logits   = seeded-instance prior OR temporal Gaussian OR zero
combined       = learned + prior
```

Compute full-scene metrics in chunks and save only compact query-centric crops.
This directly tests whether a hand-designed prior dominates learned geometry.

### Gradients

Summarize parameter norm, gradient norm, nonzero count, finite state and
`gradient_norm / parameter_norm` by major model subsystem. A dedicated backward
probe never steps the optimizer.

## 3. Memory policy

The full logical biological scene is retained. Memory is bounded by:

- reductions instead of copying full internal feature maps;
- CPU copies only for small structured query/temporal states;
- spatial chunking for native-mask metrics;
- full probability arrays only for interpretable dense heads;
- cropped mask arrays only for selected queries.

Debugging must not introduce cell-by-cell training semantics.

## 4. Tool boundaries

- `core/`: orchestration, configuration, hooks, trace schema.
- `probes/`: one scientifically meaningful diagnostic per module.
- `visualization/`: optional Napari frontend.
- `io/`: machine-readable persistence.
- `reports/`: run comparison.
- `acceptance/`: existing first-overfit forward/backward gates.
- `cli/`: reproducible command-line entry point.

## 5. V0.1 acceptance criteria

The first release must be able to:

1. run an unchanged checkpoint on an ordinary STIR-Net batch;
2. reproduce the current Hungarian matching;
3. show query existence/center progression across all three decoder layers;
4. show temporal token changes through CR1/CR2;
5. decompose selected native masks into learned/prior/combined contributions;
6. expose CNN dense foreground/center/boundary predictions in Napari;
7. summarize gradients by major model module;
8. save/load a trace and compare two traces;
9. remain compatible with existing `first_overfit_acceptance` imports.

## 6. Deferred features

### Attention replay

Current V1 query self-attention is configured with `need_weights=False`, query
cross-attention returns only the projected message, and co-reasoning uses
bounded helper methods rather than an ordinary `forward()` that returns
weights. V0.1 therefore does not monkey-patch these methods.

If query/mask/temporal traces indicate attention is the failing stage, V0.2
should add a **selected-query replay probe** that recomputes attention from the
captured inputs and existing module parameters without changing forward
behavior.

### Loss-specific gradient attribution

V0.1 provides total-loss module gradients. If loss competition remains unclear,
V0.2 can backpropagate individual named losses in isolated diagnostic passes.

## 7. Iteration workflow

```text
checkpoint A
 -> standard light trace
 -> deep trace on representative failures
 -> hypothesis
 -> one source change
 -> checkpoint B
 -> same trace contract
 -> compare_traces(A, B)
 -> Napari inspection
 -> keep/revert change based on evidence
```
