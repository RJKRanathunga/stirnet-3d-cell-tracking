# Current production pipeline

`src.pipeline` owns composition for the current project architecture.
Model implementations remain in their own packages.

```text
BioHub raw volume
      |
      v
src/source_instances/
      |
      v
learned/stirnet/
      |
      v
src/tracking/trackastra/
      |
      |-- Trackastra pass 1 in source coordinates
      |-- robust global-motion estimate from pass-1 continuations
      |-- lazy padded stabilization of raw + instance movies
      |-- Trackastra pass 2 in stabilized coordinates
      `-- restore final graph coordinates to source coordinates
      |
      v
learned/track_reconciler/
      |
      v
final exporter
```

The learned track stitcher is still being finalized. The pipeline has an
explicit Stage-4 boundary but never falls back silently to the historical
heuristic stitcher.

## Primary-tracking contract

`src.tracking` owns Trackastra execution. Bootstrap global-motion compensation
is internal to the primary-tracking stage. Downstream stitching, visualization
and export receive the final graph in the original source coordinate system.

Global motion is estimated only from first-pass Trackastra predictions.
Production tracking never imports annotations, investigations, or ground truth.

## Target-frame-anchored temporal coordinates

Any STIR-Net temporal reasoning that consumes Trackastra evidence must use the
float cumulative global-motion estimate produced by primary tracking. Temporal
detections are expressed relative to the current target frame as

```text
p_temporal(t | t0) = p_source_relative(t) - (G(t) - G(t0))
```

This keeps the target frame and current spatial RAG unchanged while removing
common stage/embryo translation from historical and future detections. Temporal
velocities are recomputed from accepted Trackastra continuations after this
coordinate transform.

The temporal path must not materialize another padded movie. The Trackastra
graph, visualization, stitching and export continue to use original BioHub
source coordinates. `src.pipeline.stages.temporal_evidence` owns the bridge
from `TrackastraResult.global_motion` to STIR-Net's temporal graph builder.

The shared STIR-Net temporal cache contract is versioned so raw-motion temporal
caches cannot be silently reused after this change.

## Dependency rules

1. Active production code must not import `legacy.classical_pipeline`.
2. Legacy code may import stable active utilities for reproducibility.
3. `dataset_curation/`, Kaggle code, and other consumers should use active
   stage APIs rather than own model execution logic.
4. `notebooks/`, `investigations/`, and `experiments/` are proof-of-path
   artifacts and are intentionally not rewritten by this migration.
5. Historical stage numbers must not be introduced into new active APIs.
