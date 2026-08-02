# Pipeline refactor compatibility notes

## Canonical behavior

- Stages 1–5 use the implementations that were already under `src/`. Notebook
  copies were removed even where their exploratory sequencing differed.
- Stage 2 retains the canonical Otsu threshold and binary-mask contract. The
  notebook's connected-component inspection is analysis, not part of mask
  production.
- Stage 3 retains the source probabilistic component analysis and deterministic
  instance-label/marker alignment. Older notebook-local segmentation paths are
  no longer executable.
- Stages 4 and 5 retain the source dataframe construction, feature names, column
  order, and use of the preprocessed intensity volume.
- Stage 6 writes the existing `tNNN.npy` and `tNNN.csv` artifacts to the same
  stage directories through shared I/O helpers.

## Tracking and stitching preservation

- Stage 7 configuration, candidate construction, cost calculations, assignment,
  state mutation, ID allocation, record append order, and final table ordering
  were migrated from the notebook without algorithmic changes. Notebook globals
  became local state inside `run_cell_tracking`.
- Stage 8 cell-feature enrichment, onset scoring, conflict resolution, hidden
  center reconstruction, merged-interval tracing, split assignment, remapping,
  correction order, ending classification, and serialization order were migrated
  in their original cell order into `run_track_stitching`.
- Full-sample comparisons matched all ten Stage 7 CSV tables and all ten Stage 8
  CSV tables. The Stage 8 baseline contains no accepted merge onsets, so focused
  synthetic positive and negative tests additionally cover correction and
  rejection behavior.

## Identifiers and serialization

- `cell` remains the zero-based positional row index within a frame's detection
  table. `cell_id` remains the one-based instance label.
- Coordinates remain voxel-order `ZYX`; physical calculations retain the
  existing voxel size `(1.625, 0.40625, 0.40625)` micrometers.
- Shared savers use the existing pandas/NumPy serialization. A dataframe with no
  columns still produces the same headerless blank CSV behavior.
- Optional diagnostics do not add columns to normal stage outputs. Provenance for
  decisions remains in traces unless Stage 8's existing corrected-track schema
  already contains merge provenance fields.

## Scope exclusions

- Stage 10 lineage behavior was not changed.
- No storage redesign, configuration persistence, algorithm optimization, or
  heuristic change is included in this refactor.
