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
src/tracking/trackastra.py
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

## Dependency rules

1. Active production code must not import `legacy.classical_pipeline`.
2. Legacy code may import stable active utilities for reproducibility.
3. `dataset_curation/`, Kaggle code, and other consumers should use active
   stage APIs rather than own model execution logic.
4. `notebooks/`, `investigations/`, and `experiments/` are proof-of-path
   artifacts and are intentionally not rewritten by this migration.
5. Historical stage numbers must not be introduced into new active APIs.
