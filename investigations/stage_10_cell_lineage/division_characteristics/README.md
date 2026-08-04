# Stage 10 division-characteristics investigation

This read-only investigation analyzes manually extracted scenes under:

```text
data/tracking_scenes/divisions/
```

It does **not** require a fixed `[-4, +4]` time window. Every valid scene may contain a different number of parent and child frames.

## Division-frame inference

For each scene, the tool reads `scene.json -> selected_cells` and expects this pattern across the saved selections:

```text
one selected cell, one selected cell, ... -> two selected cells, two selected cells, ...
```

The **division event frame** is the first saved frame containing two selected cells. That is treated as the frame in which the two children are born.

A gap between the last saved parent frame and the first saved two-child frame is accepted in non-strict mode and recorded in the outputs. A reversal back to one selected cell after the event is rejected because it is not an unambiguous 1-to-2 scene.

## Why the full repository data are reloaded

The tracking-scene extractor saves raw and preprocessed arrays after zeroing voxels outside the selected cells. Those files are useful for visualization, but they cannot support reliable local-background or fixed-radius measurements.

This investigation therefore uses each scene as an annotation and reloads:

- the full raw Zarr frame;
- saved Stage 6 preprocessed data;
- saved binary masks;
- saved instance labels;
- saved cell tables;
- Stage 8 tracks, falling back to Stage 7 tracks when necessary.

No production artifact is modified.

## Extracting scenes

Create the category directory if it does not already exist:

```text
data/tracking_scenes/divisions/
```

For each manually identified division:

1. Save one selected parent cell in every available pre-division frame.
2. In the first frame where two children are visible, save both child cell IDs.
3. Continue saving both children for as many later frames as available.
4. Prefer consecutive frames, but incomplete temporal windows are supported.
5. Do not include unrelated nearby cells in the manual selection.

The child labels `daughter_a` and `daughter_b` are analysis labels only. At the event frame they are ordered deterministically by position. In later frames, the assignment that minimizes physical centroid displacement is used, so changing cell IDs do not swap the daughter trajectories.

## Run

From the repository root with the project environment activated:

```powershell
python -m investigations.stage_10_cell_lineage.division_characteristics.run_analysis
```

Use an explicit scene or output directory:

```powershell
python -m investigations.stage_10_cell_lineage.division_characteristics.run_analysis `
  --scenes-root data/tracking_scenes/divisions `
  --output-dir data/investigations/stage_10_cell_lineage/division_characteristics/manual_run
```

Run validation strictly:

```powershell
python -m investigations.stage_10_cell_lineage.division_characteristics.run_analysis --strict
```

Write tables without figures:

```powershell
python -m investigations.stage_10_cell_lineage.division_characteristics.run_analysis --no-plots
```

## Extracted features

### Geometry

- voxel and physical volume;
- physical centroid;
- physical bounding-box dimensions;
- equivalent radius;
- physical PCA axes;
- elongation, flatness and anisotropy;
- exposed-face physical surface area;
- sphericity;
- convex-hull volume and solidity.

### Intensity representations

Every intensity measurement is calculated from both:

- full raw fluorescence;
- saved production-preprocessed fluorescence.

For the instance mask, eroded core, fixed-radius sphere, clean fixed-radius sphere and local background shell, the tool calculates:

- voxel count;
- mean and median;
- standard deviation and MAD;
- minimum and maximum;
- integrated intensity;
- P10, P25, P75, P90 and P95;
- IQR, CV and range;
- brightest 10% and brightest 25% means.

The clean fixed-radius sphere excludes voxels belonging to other segmented cells while preserving target-cell and background voxels.

### Background and frame correction

The tool calculates:

- local-background-corrected mean intensity;
- local-background-corrected integrated intensity;
- raw cell intensity divided by the full-frame foreground median;
- equivalent ratios for selected preprocessed measurements.

These measurements help distinguish real daughter brightening from changing segmentation boundaries or whole-frame illumination variation.

## Variable parent baseline

The preferred baseline excludes the final pre-division parent frame because that frame may already contain division-associated changes.

Default behavior:

1. use all available parent frames except the final one;
2. require at least two baseline frames;
3. if too few remain, use all available parent frames;
4. if only one parent frame exists, use it and record `limited_parent_history`.

The exact baseline frames and fallback mode are written to `parent_baselines.csv`.

## Outputs

A timestamped directory is created under:

```text
data/investigations/stage_10_cell_lineage/division_characteristics/<timestamp>/
```

Important tables:

- `scene_validation.csv`: valid and invalid scene parsing results;
- `case_manifest.csv`: event frame, temporal coverage, transition gap and processing status;
- `frame_reference.csv`: full-frame foreground/background intensity references;
- `observation_features.csv`: one wide row per selected parent or child observation;
- `normalized_trajectories.csv`: long-form feature trajectories relative to the parent baseline;
- `parent_baselines.csv`: baseline value, MAD, frames and fallback source;
- `combined_children.csv`: child sums, daughter separation and asymmetry;
- `event_summaries.csv`: final-parent, first-child and combined-child event descriptors;
- `feature_consistency.csv`: cross-case direction and frequency summaries for event ratios.

Figures are grouped into per-case trajectories, combined-child trajectories and event-aligned normalized trajectories.

## Interpretation

Five manually selected scenes are enough to identify promising patterns, but not enough to establish final production thresholds. Prefer signals that:

- move in the same direction in most cases;
- exceed ordinary parent baseline variation;
- agree between raw, frame-corrected and mask-robust measurements;
- persist across more than one frame;
- are not explained solely by changing mask size.
