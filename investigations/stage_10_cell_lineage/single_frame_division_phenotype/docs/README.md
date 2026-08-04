# Single-frame division phenotype investigation

This is a new, read-only Stage 10 investigation. It does not modify or import the existing `division_characteristics` investigation.

## Question

Which static properties make a cell look like:

- a parent immediately before division; or
- a newly born daughter;

when only one 3-D frame is available?

The saved scenes under `data/tracking_scenes/divisions/` are used only to label parent and daughter cells. No previous-frame, next-frame, persistence, velocity, divergence, track-duration, or parent-history feature is used as a predictor.

## Population design

For every manually labelled target observation, the investigation extracts identical features for every segmented cell in the same full frame. It produces comparisons against:

- all other cells;
- clean normal cells, excluding boundary, virtual-merge, known-lineage, and Stage 8 event observations;
- all nearby cells inside the configured physical radius;
- clean nearby cells.

Cell size is retained as a valid phenotype feature. A separate size-adjusted residual is also calculated so that the analysis can distinguish "the daughter is small" from "the daughter is unusual even for its size."

## Primary labels

- `parent_final`: last saved one-cell frame before the first saved two-cell frame;
- `daughter_birth`: both selected cells in the first saved two-cell frame.

Secondary labels are retained separately:

- `parent_earlier`;
- `daughter_later`.

## Static features

The package extracts:

- physical volume, bounding box, PCA axes, extent, surface area, sphericity, solidity, compactness, EDT thickness, and geometric peak count;
- raw and production-preprocessed intensity distributions;
- percentiles, MAD, CV, skewness, kurtosis, entropy, Gini concentration, brightest-fraction means, and integrated intensity;
- core, middle, and outer radial intensity regions;
- local-background and whole-frame corrected intensity;
- intensity-centroid and brightest-voxel offsets;
- bright connected regions and internal peak structure;
- gradient and Laplacian texture energy.

All target and normal cells pass through exactly the same extractor.

## Run

From the repository root:

```powershell
python -m investigations.stage_10_cell_lineage.single_frame_division_phenotype.run_analysis
```

Explicit paths:

```powershell
python -m investigations.stage_10_cell_lineage.single_frame_division_phenotype.run_analysis `
  --scenes-root data/tracking_scenes/divisions `
  --output-dir data/investigations/stage_10_cell_lineage/single_frame_division_phenotype/manual_run
```

Faster table-only run:

```powershell
python -m investigations.stage_10_cell_lineage.single_frame_division_phenotype.run_analysis `
  --no-plots --no-galleries
```

Strict validation:

```powershell
python -m investigations.stage_10_cell_lineage.single_frame_division_phenotype.run_analysis --strict
```

## Main outputs

A timestamped run is written to:

```text
data/investigations/stage_10_cell_lineage/single_frame_division_phenotype/<timestamp>/
```

Important tables:

- `scene_validation.csv`
- `target_labels.csv`
- `frame_manifest.csv`
- `static_cell_features.csv`
- `normal_cell_features.csv`
- `target_features.csv`
- `parent_features.csv`
- `daughter_features.csv`
- `primary_parent_features.csv`
- `birth_daughter_features.csv`
- `population_manifest.csv`
- `population_contrasts.csv`
- `feature_effect_summary.csv`
- `gallery_manifest.csv`

`population_contrasts.csv` contains each target's percentile, robust z-score, Cliff's delta, and optional size-adjusted residual for every feature and population.

`feature_effect_summary.csv` summarizes each feature at scene level. With only five verified scenes, use direction consistency and case-level percentile separation for feature discovery; do not treat these outputs as final production accuracy estimates.

## Galleries

Galleries isolate each target/control instance inside a fixed physical crop and use one shared raw-intensity scale for all cells from the same frame. They show each target beside nearby and deterministic random same-frame normal cells. Crops are not independently normalized because that would hide relative brightness differences.

## Production safety

The investigation only reads raw, Stage 6, Stage 8, and optional Stage 10 outputs. It writes only under `data/investigations/` unless an explicit output directory is supplied.
