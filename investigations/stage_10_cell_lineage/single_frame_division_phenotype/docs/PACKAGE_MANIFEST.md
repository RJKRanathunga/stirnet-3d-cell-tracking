# Package manifest

This archive adds a new investigation package without changing existing files.

## Added package

`investigations/stage_10_cell_lineage/single_frame_division_phenotype/`

- configuration and scene validation;
- repository-aware raw/Stage 6/Stage 8/optional Stage 10 loading;
- full same-frame population extraction;
- static morphology, intensity-distribution, radial, peak, and texture features;
- separate parent and daughter exports;
- global, clean, local, and clean-local population contrasts;
- unadjusted size features and separate size-adjusted residuals;
- feature-effect summaries, plots, and shared-scale isolated-cell galleries;
- command-line runner and package documentation.

## Added test

`tests/test_single_frame_division_phenotype.py`

## Validation performed

- Python bytecode compilation succeeded.
- Five synthetic unit tests passed.
- A synthetic repository-shaped end-to-end run succeeded with tables, plots, and galleries enabled.

## Maintenance fix

- Treat missing or zero-byte optional `segmentation_events.csv` and `protected_tracks.csv` files as empty tables.
- Include captured frame errors in the final failure message when no frame succeeds.
- Add regression tests for both empty optional CSV cases.
