# Small-cell size and temporal-variation investigation

This read-only investigation analyzes manually curated scenes under:

```text
data/tracking_scenes/small_cells/
```

It answers four separate questions:

1. What voxel-volume range do the manually selected “small cells” occupy?
2. Where does each selected cell fall within the complete same-frame cell population?
3. How much does each selected manual cell trajectory vary over time?
4. Is that variation larger than the variation of stable, successfully tracked control cells—including size-matched controls?

The investigation reloads full Stage 6 cell tables and segmentation labels. It does **not** restrict the comparison population to the saved scene crop. The saved `scene.json` selections are used only as manual labels.

## Installation

Extract the ZIP at the repository root so that this directory becomes:

```text
investigations/stage_07_cell_tracking/small_cell_statistics/
```

No production pipeline files are modified.

## Run

From the repository root:

```powershell
python -m investigations.stage_07_cell_tracking.small_cell_statistics.run_analysis
```

Strict run without figures:

```powershell
python -m investigations.stage_07_cell_tracking.small_cell_statistics.run_analysis `
  --strict `
  --no-plots
```

Custom scene and output directories:

```powershell
python -m investigations.stage_07_cell_tracking.small_cell_statistics.run_analysis `
  --scenes-root data/tracking_scenes/small_cells `
  --output-dir data/investigations/stage_07_cell_tracking/small_cell_statistics/manual_run
```

Useful controls:

```text
--control-min-observations 5
--matched-controls-per-target 10
--target-link-max-distance-um 15
--no-mask-volume-validation
--allow-boundary-controls
--allow-event-controls
--allow-gapped-controls
--no-plots
--strict
```

## Manual trajectory handling

A scene may contain one or several selected cells in each frame. The investigation reconstructs **manual trajectories** across selected frames using deterministic Hungarian assignment in physical Z/Y/X coordinates. This is intentionally independent of production track IDs. Production IDs are retained in the output so fragmentation and ID changes remain visible.

## Stable control definition

A production track is a stable control by default only when it:

- has at least five observations;
- contains no duplicate frame;
- is temporally contiguous;
- never touches the volume boundary;
- contains no virtual-merge observation;
- is not referenced by known Stage 8 merge or Stage 10 division/protection tables;
- is not one of the manually selected small-cell production tracks.

The package additionally selects nearest size-matched stable controls for each manual target track using log median-volume distance and track-length difference.

## Main outputs

All tables are written below `tables/`.

### Size and same-frame comparisons

- `selected_small_cell_observations.csv` — every manually selected observation with all Stage 6 features, recomputed mask volume, manual trajectory ID, and production track ID.
- `all_cell_observations.csv` — full Stage 6 population for every loaded sample.
- `all_other_cell_observations.csv` — full population excluding selected targets.
- `frame_population_feature_comparisons.csv` — target value, same-frame population quantiles, percentile rank, robust z-score, and median ratio for every available numeric feature.
- `selected_feature_population_summary.csv` — feature-level summary of median percentile, tail frequency, robust z-score, and population-median ratio.
- `frame_size_summary.csv` — compact per-scene/per-frame volume comparison.
- `size_distribution_summary.csv` — volume quantiles for targets, all other cells, stable-control observations, and track medians.
- `small_size_threshold_candidates.csv` — candidate thresholds from target quantiles and population percentiles, with selected-cell recall and control prevalence.

### Temporal variation

- `selected_small_track_summary.csv` — per-manual-track volume range, CV, robust CV, IQR/range ratios, slopes, and adjacent-frame change statistics.
- `stable_control_track_summary.csv` — the same metrics for successful controls.
- `track_volume_transitions.csv` — one row per temporal transition, including symmetric relative change, log change, frame gap, and volume ratio.
- `feature_within_track_variation.csv` — within-track variation for volume, morphology, intensity, and bounding-box features.
- `feature_variation_comparison.csv` — aggregate selected-versus-control variation effects for every tracked feature.
- `stable_control_variation_by_size_bin.csv` — empirical normal variation curve across stable-control median-volume bins.
- `volume_variation_comparison.csv` — selected-versus-control distribution comparison, Mann–Whitney p-value, effect direction, and selected median percentile within controls.
- `size_matched_control_pairs.csv` — stable controls matched to each selected manual trajectory by median size and duration.

### Validation and provenance

- `scene_validation.csv`
- `track_observations_with_features.csv`
- `stable_control_observations.csv`
- `control_track_manifest.csv`
- `run_metadata.json`

## Figures

When plots are enabled:

1. full-population versus selected-cell volume distributions;
2. selected-cell same-frame volume percentiles;
3. manual target volume trajectories;
4. volume instability versus median cell size;
5. target/control variation boxplots;
6. candidate small-cell threshold trade-off.

## Interpreting the result

Do not choose a production “small cell” threshold from one number alone. Look for agreement among:

- the selected observation and selected track-median ranges;
- same-frame percentile ranks;
- the threshold recall/prevalence table;
- variation-versus-size plots;
- stable controls and size-matched stable controls.

A useful implementation decision should separate two effects:

1. **expected size-dependent segmentation variation** shared by correctly tracked small controls;
2. **failure-specific instability** present mainly in the manually curated failure tracks.
