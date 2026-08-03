# Deferred Experiment Proposal: Raw and Frame-Gain-Normalized Intensity Features

| Field | Value |
|---|---|
| Status | **Deferred / not approved for production implementation** |
| Evidence level | Preliminary, single-sample investigation |
| Primary stage | Stage 05 — feature extraction |
| Downstream stages | Stage 07 — cell tracking; Stage 08 — track stitching and merge analysis |
| Related investigation | `investigations/stage_05_feature_extraction/intensity_statistics/` |
| Related investigation commit | `40ae959a2a28f007866f8c4e8b57d0877ea815f7` |
| Repository state reviewed | `main` through `5a789c1eb40050094b22299ddefa8a267856d07d` |
| Proposed document location | `docs/experiments/stage_05_frame_gain_normalized_intensity_features.md` |
| Decision | Preserve the idea for later validation; do not change production behavior now |

## 1. Executive summary

The current pipeline uses a strongly preprocessed image for two different purposes:

1. producing the binary mask and instance segmentation; and
2. calculating per-cell intensity features.

That preprocessing is appropriate for segmentation because it suppresses noise,
corrects broad background variation, and improves foreground separation.
However, it changes the original fluorescence distribution through percentile
normalization, Gaussian denoising, smooth-background subtraction, clipping, and
rescaling.

A Stage 05 investigation compared intensity measurements from:

- raw volumes;
- weak Gaussian denoising at `0.2 µm`;
- weak Gaussian denoising at `0.4 µm`;
- the current normalized intermediate;
- the current denoised intermediate;
- the fully preprocessed production image.

All methods used the same saved production masks and instance labels. Therefore,
the observed differences came from the intensity representation rather than
different cell boundaries.

The preliminary result suggested that:

- weak Gaussian denoising preserved raw features but did not materially improve
  temporal stability;
- the fully preprocessed image reduced useful differences between cells;
- raw-derived features were more discriminative for correct versus nearby
  incorrect tracking candidates;
- a large part of raw temporal variation appeared to be a frame-wide intensity
  trend rather than random voxel noise;
- post-hoc frame-level gain correction was promising.

These observations are not sufficient to justify a production change. They were
obtained from one sample, twenty frames, automatically selected tracks, and
associations produced by the current tracker rather than external ground truth.

This document preserves a possible future implementation:

```text
raw volume
├── existing strong preprocessing
│   └── binary mask → instance labels → geometry and shape
│
└── raw intensity measurement
    ├── raw per-cell statistics
    └── frame-gain normalization
        └── tracking-oriented intensity statistics
```

The current production behavior should remain unchanged until stronger evidence
is available.

---

## 2. Current repository behavior

### 2.1 Stage 06 orchestration

The canonical dataset-processing flow currently performs:

```python
raw = load_timepoint(sample_path, frame)
preprocessed = preprocess_volume(raw)
binary_mask = create_binary_mask(preprocessed)
labels = segment_instances(binary_mask)
cells = detect_cells(labels)
cells = extract_cell_features(cells, labels, preprocessed)
```

Therefore, current `intensity_*` columns are calculated from the fully
preprocessed image, not the original raw fluorescence volume.

The relevant file is:

```text
src/dataset_processing.py
```

### 2.2 Stage 05 feature extraction

The current Stage 05 implementation calculates per-cell intensity values using:

```python
mask = labels == cell_id
intensities = volume[mask]
```

It then produces:

```text
intensity_mean
intensity_median
intensity_std
intensity_min
intensity_max
intensity_q25
intensity_q75
intensity_sum
intensity_range
intensity_iqr
intensity_cv
```

The same function also calculates geometry and morphology from the fixed
instance labels.

The relevant files are:

```text
src/05_feature_extraction/features.py
src/05_feature_extraction/pipeline.py
```

### 2.3 Stage 06 serialization

The complete per-cell dataframe is saved to:

```text
data/sample/processed/stage_6_processed_dataset/<sample_id>/cells/tNNN.csv
```

The existing loader requires only the stable detection columns and tolerates
additional feature columns. Therefore, a future implementation could append
new intensity columns without introducing a second feature table or changing
the Stage 06 directory layout.

The relevant file is:

```text
src/io/stage_io.py
```

### 2.4 Stage 07 intensity usage

Stage 07 currently defines:

```python
INTENSITY_FEATURES = [
    "intensity_mean",
    "intensity_std",
    "intensity_cv",
]
```

For each feature, it calculates a symmetric relative absolute difference:

```text
|previous - current|
────────────────────
max(|previous|, |current|) + epsilon
```

The feature-group residual is the average across available intensity features.
A Student-t negative-log cost is then applied. The intensity term currently has
moderate influence:

```text
interior intensity weight = 0.30
boundary intensity weight = 0.25
```

The relevant files are:

```text
src/07_cell_tracking/step01_config.py
src/07_cell_tracking/step02_association.py
src/07_cell_tracking/step03_pipeline.py
```

### 2.5 Stage 08 intensity usage

Stage 08 enriches track observations with Stage 06 intensity features and uses
`intensity_sum` when evaluating whether two parent tracks could explain a newly
observed merged cell.

The relevant file is:

```text
src/08_track_stitching/step01_pipeline.py
```

---

## 3. Why the current preprocessed intensity may be suboptimal

The current preprocessing path is optimized for cell separation, not
quantitative fluorescence preservation.

Conceptually, it performs:

```text
raw
→ percentile normalization
→ Gaussian denoising
→ broad background estimation
→ background subtraction
→ negative clipping
→ frame-wise rescaling
```

These operations are useful for segmentation but can alter intensity features.

### 3.1 Percentile normalization

Each frame is rescaled according to its own intensity distribution. Consequently:

- equal raw intensities can map to different normalized values in different
  frames;
- different raw intensities can map to similar normalized values;
- absolute fluorescence units are lost;
- high-intensity values can be compressed by percentile clipping.

### 3.2 Spatial Gaussian denoising

Gaussian filtering mixes values across the cell boundary:

- part of a cell's signal can move outside its fixed instance mask;
- neighbouring or background signal can move inside the mask;
- maxima and internal heterogeneity are reduced;
- close cells can become more similar.

### 3.3 Broad background subtraction

A smooth background model may contain part of a large or diffuse cell's signal.
Subtracting it can remove legitimate low-frequency cellular intensity.

### 3.4 Negative clipping

Values below the estimated background are clipped to zero. This changes:

- the lower tail of the distribution;
- mean and variance;
- low percentiles;
- coefficient of variation;
- weak peripheral cell signal.

### 3.5 Final frame-wise rescaling

Dividing by a frame-dependent maximum makes every cell depend on the brightest
region in that frame. A single unusually bright object can influence all
intensity features.

### 3.6 Practical interpretation

The fully preprocessed image remains useful as a contrast representation.
It is not necessarily a faithful quantitative measurement of the original
fluorescence signal.

---

## 4. Preliminary evidence from the completed investigation

The investigation used:

```text
sample: 44b6_0113de3b
frames: 0–19
fixed masks: saved production binary masks and instance labels
selected complete tracks: 40
```

The following results are preliminary and must not be treated as ground-truth
benchmark results.

### 4.1 Weak denoising preserved raw statistics

Median relative changes compared with raw were approximately:

| Method | Mean | Median | Standard deviation | IQR | CV |
|---|---:|---:|---:|---:|---:|
| Weak Gaussian `0.2 µm` | `0.40%` | `0.47%` | `1.24%` | `1.14%` | `0.83%` |
| Weak Gaussian `0.4 µm` | `1.86%` | `2.07%` | `3.84%` | `2.93%` | `1.94%` |

Spearman rank agreement with raw was approximately `0.997` or higher for the
tested features.

However, weak Gaussian filtering reduced measurements inside the fixed masks.
For example, the retained maximum was approximately:

```text
0.2 µm: 98.12%
0.4 µm: 94.93%
```

This is consistent with Gaussian signal spreading across fixed instance
boundaries.

### 4.2 Weak denoising did not materially improve temporal stability

Median temporal coefficients of variation across the selected tracks were
approximately:

| Method | Per-cell mean | Per-cell median |
|---|---:|---:|
| Raw | `4.28%` | `4.36%` |
| Weak Gaussian `0.2 µm` | `4.28%` | `4.35%` |
| Weak Gaussian `0.4 µm` | `4.26%` | `4.42%` |
| Current normalized intermediate | `3.17%` | `3.29%` |
| Current denoised intermediate | `3.08%` | `2.95%` |
| Fully preprocessed production image | `4.16%` | `4.23%` |

The tested weak Gaussian filters were effectively indistinguishable from raw
for cell-level mean and median stability.

### 4.3 A frame-wide intensity trend was present

Across the twenty-frame sequence:

```text
raw foreground mean increased by approximately 8.0%
raw foreground median increased by approximately 7.5%
```

Most selected tracks had a similar positive intensity slope. After linear
detrending, the residual variation of raw mean was close to the current
denoised representation.

This suggests that much of the apparent raw temporal instability may be a
shared frame-level trend rather than independent voxel noise.

### 4.4 Raw-derived features were more discriminative for tracking candidates

The investigation compared the current Stage 07 intensity feature combination
on correct consecutive-frame pairs and nearby incorrect candidates.

Approximate separation results were:

| Method | Correct median cost | Wrong median cost | Separation AUC |
|---|---:|---:|---:|
| Raw | `0.0471` | `0.2402` | `0.9089` |
| Weak Gaussian `0.2 µm` | `0.0479` | `0.2414` | `0.9080` |
| Weak Gaussian `0.4 µm` | `0.0462` | `0.2419` | `0.9077` |
| Current denoised intermediate | `0.0337` | `0.1757` | `0.8863` |
| Current normalized intermediate | `0.0397` | `0.1829` | `0.8827` |
| Fully preprocessed production image | `0.0356` | `0.1793` | `0.8475` |

Higher AUC is better.

The current denoised image made the same cell more stable, but also made nearby
incorrect cells more similar. The fully preprocessed image had the weakest
candidate discrimination in this experiment.

### 4.5 Robust feature combinations appeared preferable

Approximate single-feature AUC values for raw measurements were:

| Feature | AUC |
|---|---:|
| Mean | `0.9175` |
| Median | `0.9120` |
| Standard deviation | `0.8823` |
| CV | `0.8622` |
| IQR | `0.8725` |

Approximate raw feature-combination results were:

| Combination | AUC |
|---|---:|
| Current `mean + std + CV` | `0.9089` |
| Mean only | `0.9175` |
| Mean + median | `0.9190` |
| Mean + median + IQR | `0.9208` |

This suggests that `std` and especially `CV` may weaken the tracking appearance
signal in this sample.

### 4.6 Post-hoc frame correction was promising

A preliminary post-hoc calculation divided raw cell statistics by a frame-level
foreground reference.

It approximately reduced the median temporal CV of raw cell mean from:

```text
4.28% → 3.20%
```

Approximate association results were:

```text
frame-corrected raw, mean + std + CV:       AUC ≈ 0.9095
frame-corrected raw, mean + median + IQR:   AUC ≈ 0.9229
```

These calculations were not part of a production implementation and did not
test all downstream behavior.

---

## 5. Why implementation is deferred

The evidence is suggestive but incomplete.

### 5.1 Only one sample was analyzed

The result may depend on:

- sample brightness;
- density;
- illumination field;
- acquisition settings;
- segmentation quality;
- degree of photobleaching or frame drift;
- cell morphology.

The other samples may behave differently.

### 5.2 Tracks were automatically selected

The forty complete tracks were selected using structural criteria, not
independent manual confirmation. Some may contain identity errors.

### 5.3 Correct associations came from the current tracker

The association labels were derived from current Stage 07 outputs rather than
external ground truth. This introduces circularity.

### 5.4 Global intensity change may be biological

A frame-level trend is not automatically an imaging artifact. If the complete
cell population genuinely becomes brighter or dimmer, gain normalization would
remove a real biological trend.

### 5.5 Local background was not evaluated

The experiment did not test:

- per-cell local background shells;
- local background subtraction;
- reliability in dense regions;
- background contamination from adjacent cells.

### 5.6 Stage 08 effects were not evaluated

Changing `intensity_sum` semantics could affect merge-onset evidence. The
completed investigation did not evaluate:

- parent-sum versus merged-sum conservation;
- merge candidate rejection distributions;
- accepted merge repairs;
- false merge evidence.

### 5.7 No full pipeline regression was performed

There is no current evidence that the proposed representation improves:

- unexpected track loss;
- track fragmentation;
- complete track count;
- division accuracy;
- merge recovery;
- final competition metrics.

For these reasons, this document records a hypothesis and implementation design,
not an approved change.

---

## 6. Future hypothesis

The future experiment would test the following hypothesis:

> Intensity features calculated from raw voxels inside fixed production
> instance masks, after removing a robust frame-wide multiplicative gain, retain
> more cell-specific appearance information than the current fully preprocessed
> representation while achieving comparable temporal stability.

A secondary hypothesis is:

> Robust features such as mean, median, and IQR provide more useful Stage 07
> appearance evidence than the current mean, standard deviation, and CV
> combination.

---

## 7. Measurement model

A simplified raw measurement model is:

```text
observed cell signal at frame t
    = true cell signal
      × frame-wide acquisition gain
      + local background
      + random noise
      + segmentation-boundary effects
```

The proposed first experiment would address only the multiplicative
frame-wide term.

It would not initially attempt to correct local background.

---

## 8. Proposed future architecture

```text
raw volume
├── existing Stage 01 preprocessing
│   └── Stage 02 mask
│       └── Stage 03 instances
│           └── Stage 04 detections
│               └── geometry and morphology
│
└── raw Stage 05 intensity branch
    ├── raw per-cell statistics inside Stage 03 instances
    ├── robust frame reference
    └── gain-normalized per-cell statistics
        ├── optional Stage 07 appearance features
        └── optional Stage 08 intensity-conservation feature
```

The existing segmentation path must remain unchanged.

---

## 9. Proposed feature schema

The future implementation should not immediately change the meaning of current
`intensity_*` columns.

### 9.1 Preserve legacy processed features

Retain:

```text
intensity_mean
intensity_median
intensity_std
intensity_min
intensity_max
intensity_q25
intensity_q75
intensity_sum
intensity_range
intensity_iqr
intensity_cv
```

These would continue to represent the fully preprocessed image during the
validation phase.

### 9.2 Add explicit raw features

Append:

```text
intensity_raw_mean
intensity_raw_median
intensity_raw_std
intensity_raw_min
intensity_raw_max
intensity_raw_q25
intensity_raw_q75
intensity_raw_iqr
intensity_raw_sum
intensity_raw_range
intensity_raw_cv
intensity_raw_p05
intensity_raw_p95
intensity_raw_mad
intensity_raw_p95_p05_range
```

All values must come from:

```python
raw_volume[labels == cell_id]
```

### 9.3 Add explicit frame-gain-normalized features

Append:

```text
intensity_gain_normalized_mean
intensity_gain_normalized_median
intensity_gain_normalized_std
intensity_gain_normalized_q25
intensity_gain_normalized_q75
intensity_gain_normalized_iqr
intensity_gain_normalized_sum
intensity_gain_normalized_p05
intensity_gain_normalized_p95
intensity_gain_normalized_mad
```

Use the term `gain_normalized`, not the ambiguous term `corrected`, because a
later experiment may introduce local background correction.

### 9.4 Add reference diagnostics

Append:

```text
intensity_frame_reference
intensity_frame_reference_method
intensity_frame_reference_cell_count
intensity_frame_reference_valid
```

Possible method values:

```text
interior_cell_medians
all_cell_medians
foreground_median
invalid
```

---

## 10. Proposed frame-reference estimator

### 10.1 Primary reference

For frame `t`, calculate each cell's raw median:

```text
m(i,t) = median raw intensity of cell i in frame t
```

Then calculate:

```text
R(t) = median of m(i,t) across eligible cells
```

The median of per-cell medians gives each cell approximately equal influence.
It avoids allowing very large cells to dominate through voxel count.

### 10.2 Eligible cells

The primary reference should use cells satisfying:

- finite raw median;
- positive raw median;
- no contact with the physical image boundary.

Using the current upper-exclusive bounding boxes, an interior cell satisfies:

```python
is_interior = (
    (z_min > 0)
    & (y_min > 0)
    & (x_min > 0)
    & (z_max < labels.shape[0])
    & (y_max < labels.shape[1])
    & (x_max < labels.shape[2])
)
```

Boundary exclusion reduces the influence of partially observed cells.

### 10.3 Deterministic fallback hierarchy

Use:

```text
1. Median of interior, finite, positive per-cell medians
2. Median of all finite, positive per-cell medians
3. Median of raw foreground voxels where labels > 0
4. Invalid reference
```

The first method should require a minimum number of eligible cells, tentatively
ten.

No invalid reference should be silently replaced with `1.0`.

---

## 11. Proposed gain-normalization formulas

For cell `i`, frame `t`, raw statistic `S(i,t)`, and frame reference `R(t)`:

```text
S_gain_normalized(i,t) = S_raw(i,t) / R(t)
```

Examples:

```python
intensity_gain_normalized_mean = (
    intensity_raw_mean / intensity_frame_reference
)

intensity_gain_normalized_median = (
    intensity_raw_median / intensity_frame_reference
)

intensity_gain_normalized_iqr = (
    intensity_raw_iqr / intensity_frame_reference
)

intensity_gain_normalized_sum = (
    intensity_raw_sum / intensity_frame_reference
)
```

A sequence-level multiplier is unnecessary for tracking because the existing
Stage 07 relative-difference cost is invariant to a constant common scale.

### 11.1 Interpretation

A gain-normalized value represents cell intensity relative to the robust
intensity level of its frame.

It is not an absolute fluorescence measurement.

Therefore, both raw and gain-normalized values should be retained.

---

## 12. Proposed Stage 05 source design

### 12.1 New files

```text
src/05_feature_extraction/config.py
src/05_feature_extraction/intensity.py
```

### 12.2 Configuration

Possible future configuration:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class IntensityFeatureConfig:
    reference_method: str = "median_of_cell_medians"
    exclude_image_boundary: bool = True
    minimum_interior_reference_cells: int = 10
    include_legacy_processed_features: bool = True


DEFAULT_INTENSITY_FEATURE_CONFIG = IntensityFeatureConfig()
```

### 12.3 Pure intensity functions

Possible functions:

```python
def summarize_intensity(
    values: np.ndarray,
) -> dict[str, float]:
    ...


def extract_raw_intensity_table(
    cell_ids: Sequence[int],
    labels: np.ndarray,
    raw_volume: np.ndarray,
) -> pd.DataFrame:
    ...


def estimate_frame_reference(
    raw_statistics: pd.DataFrame,
    cells_df: pd.DataFrame,
    labels_shape: tuple[int, int, int],
    config: IntensityFeatureConfig,
) -> FrameIntensityReference:
    ...


def add_gain_normalized_statistics(
    statistics: pd.DataFrame,
    reference: FrameIntensityReference,
) -> pd.DataFrame:
    ...
```

Reference result:

```python
@dataclass(frozen=True)
class FrameIntensityReference:
    value: float
    method: str
    cell_count: int
    valid: bool
```

### 12.4 Preserve geometry implementation

Do not rewrite the existing:

- PCA shape features;
- convex hull;
- compactness;
- centroid;
- bounding-box calculations.

A future change should isolate intensity semantics from geometry behavior.

---

## 13. Proposed public API migration

Current compatible call:

```python
extract_cell_features(
    cells_df,
    labels,
    preprocessed_volume,
)
```

Possible future signature:

```python
def extract_cell_features(
    cells_df: pd.DataFrame,
    labels: np.ndarray,
    volume: np.ndarray,
    *,
    raw_volume: np.ndarray | None = None,
    config: IntensityFeatureConfig = DEFAULT_INTENSITY_FEATURE_CONFIG,
    return_diagnostics: bool = False,
):
    ...
```

Interpretation:

- `volume` remains the legacy preprocessed image;
- `raw_volume` is optional;
- without `raw_volume`, behavior remains backward compatible;
- with `raw_volume`, new explicit raw and gain-normalized columns are appended.

Required validation:

```python
labels.shape == volume.shape
labels.shape == raw_volume.shape
```

when raw data is supplied.

Existing columns should retain their current names, order, and values during the
validation phase.

---

## 14. Proposed Stage 06 integration

Current call:

```python
cells = extract_cell_features(
    cells,
    labels,
    preprocessed,
)
```

Possible future call:

```python
cells = extract_cell_features(
    cells,
    labels,
    preprocessed,
    raw_volume=raw,
)
```

No additional arrays need to be saved.

Continue using:

```text
cells/tNNN.csv
```

Do not introduce:

- a second feature CSV;
- a separate raw intensity table;
- a new Stage 06 directory;
- duplicated cell identifiers.

---

## 15. Proposed Stage 05 diagnostics

Change ambiguous diagnostic inputs from:

```python
inputs={
    "cells": cells_df,
    "labels": labels,
    "volume": volume,
}
```

to:

```python
inputs={
    "cells": cells_df,
    "labels": labels,
    "preprocessed_volume": volume,
    "raw_volume": raw_volume,
}
```

Potential metrics:

```text
raw_intensity_available
frame_reference_value
frame_reference_method
frame_reference_cell_count
frame_reference_valid
legacy_intensity_features_preserved
```

---

## 16. Proposed pipeline replay behavior

The replay workbench currently supplies only the preprocessed work volume to
Stage 05.

A future exact-mode replay should call:

```python
self._features_work = extract_cell_features(
    self._cells_work,
    self._labels_work,
    self._processed_work,
    raw_volume=self._raw_work,
)
```

### 16.1 Exact mode

Exact full-frame replay should estimate the frame reference from the complete
frame and match production Stage 05.

### 16.2 Local mode

Local replay must not silently estimate a frame-wide reference from a crop.

Preferred behavior:

1. load the saved full-frame reference from the baseline cell table;
2. apply that reference to the local trial features;
3. label the result as using a production full-frame reference.

Fallback behavior, if the saved reference is unavailable:

- mark gain-normalized features unavailable; or
- explicitly label a crop-derived reference as approximate.

---

## 17. Proposed Stage 07 migration

### 17.1 Preserve a legacy mode

Possible configuration:

```python
TRACKING_INTENSITY_MODE = "legacy_processed"
```

Supported modes:

```text
legacy_processed
gain_normalized_robust
```

Feature mappings:

```python
LEGACY_INTENSITY_FEATURES = [
    "intensity_mean",
    "intensity_std",
    "intensity_cv",
]

GAIN_NORMALIZED_INTENSITY_FEATURES = [
    "intensity_gain_normalized_mean",
    "intensity_gain_normalized_median",
    "intensity_gain_normalized_iqr",
]
```

### 17.2 Resolve the mode once

At the start of a tracking run:

1. inspect all frames for required columns;
2. verify that values are sufficiently finite;
3. resolve the requested mode;
4. use a visible fallback for old datasets;
5. record the resolution in metadata.

Example:

```text
requested mode: gain_normalized_robust
resolved mode: legacy_processed
fallback reason: required gain-normalized columns absent
```

Fallback must not be silent.

### 17.3 Pass feature names explicitly

Avoid changing a module-level list at runtime.

Possible future signature:

```python
assign_track_states(
    ...,
    intensity_feature_names=active_intensity_features,
)
```

Thread the list through:

```text
make_feature_template
update_feature_template
feature_group_cost
assign_track_states
```

### 17.4 Preserve weights initially

Do not initially change:

```text
INTENSITY_RELATIVE_SCALE
INTERIOR_W_INTENSITY
BOUNDARY_W_INTENSITY
```

First isolate the effect of the representation and feature set.

### 17.5 Add diagnostics

Potential Stage 07 metadata:

```text
requested_intensity_mode
resolved_intensity_mode
intensity_feature_names
intensity_fallback_used
intensity_fallback_reason
intensity_relative_scale
interior_intensity_weight
boundary_intensity_weight
```

Potential association diagnostics:

```text
intensity_cost
intensity_feature_mode
```

---

## 18. Proposed Stage 08 migration

Add optional columns:

```text
intensity_raw_sum
intensity_gain_normalized_sum
intensity_gain_normalized_mean
intensity_gain_normalized_median
intensity_gain_normalized_iqr
```

Resolve the merge intensity feature:

```python
if "intensity_gain_normalized_sum" is available:
    merge_intensity_sum_column = "intensity_gain_normalized_sum"
else:
    merge_intensity_sum_column = "intensity_sum"
```

Replace hard-coded references to `intensity_sum` with the resolved column.

### 18.1 Why normalized sum remains additive within a frame

For frame reference `R(t)`:

```text
sum(A) / R(t) + sum(B) / R(t)
    = [sum(A) + sum(B)] / R(t)
```

Therefore, gain normalization preserves parent-sum additivity within the same
frame.

Across adjacent frames, it may improve comparability by removing an estimated
global gain difference.

### 18.2 Keep thresholds unchanged initially

Do not immediately retune:

```text
MERGE_MAX_INTENSITY_SUM_REL_ERROR
```

First measure the new intensity-error distribution.

Record:

```text
merge_intensity_feature
merge_intensity_fallback_used
```

---

## 19. Required future tests

### 19.1 Stage 05 raw statistics

Using synthetic labels and raw values, verify every raw statistic directly from:

```python
raw[labels == cell_id]
```

### 19.2 Geometry independence

With identical labels but different intensity volumes, verify that all geometry
and morphology columns are identical.

### 19.3 Legacy compatibility

Without `raw_volume`, verify that current Stage 05 columns and values remain
unchanged.

### 19.4 Gain invariance

Create:

```python
raw_b = raw_a * 1.7
```

with identical labels.

Expected:

```text
raw features scale by 1.7
gain-normalized features remain equal
```

### 19.5 Reference fallback

Test:

- adequate interior cells;
- boundary exclusion;
- insufficient interior cells;
- all-cell fallback;
- foreground fallback;
- invalid empty frame;
- invalid zero reference.

### 19.6 Gain-normalized sum additivity

Verify:

```text
gain-normalized sum(A)
+ gain-normalized sum(B)
= gain-normalized sum(A ∪ B)
```

within numerical tolerance.

### 19.7 Output schema

Verify:

- all old columns remain;
- old column order remains;
- new columns are appended;
- identifiers and geometry are unchanged.

### 19.8 Stage 06 integration

Verify that `process_dataset()` supplies both:

```text
preprocessed volume
raw_volume=raw
```

while writing the same artifact paths.

### 19.9 Replay

Verify:

- exact replay equals direct production Stage 05;
- local replay uses a full-frame reference;
- crop-derived reference is never silent.

### 19.10 Stage 07

Verify:

- new data resolves the new mode;
- old data falls back to legacy;
- fallback appears in metadata;
- synthetic global multiplication does not change gain-normalized costs;
- position, volume, shape, and motion terms remain unchanged;
- changing the mode changes only intensity-related terms.

### 19.11 Stage 08

Verify:

- normalized sum is preferred when available;
- legacy sum is used when unavailable;
- selected feature is recorded;
- current merge acceptance and rejection tests continue to pass.

---

## 20. Required future validation experiment

A production implementation should not be enabled immediately.

### 20.1 Dataset scope

Run the experiment on all available samples and all available frames.

### 20.2 Track verification

Construct a manually verified evaluation set including:

- complete isolated tracks;
- difficult shape-changing tracks;
- low-signal cells;
- bright cells;
- crowded cells;
- boundary cells;
- merge and split intervals;
- unexpected loss scenes.

### 20.3 Compared Stage 07 modes

At minimum:

```text
A. current legacy processed mean + std + CV
B. raw mean + median + IQR
C. gain-normalized raw mean + median + IQR
D. no intensity term
```

The `no intensity` baseline is essential. It determines whether the intensity
features improve tracking beyond position, motion, volume, shape, and bounding
box evidence.

### 20.4 Metrics

Measure:

```text
correct association rate
incorrect association rate
miss rate
birth rate
track fragmentation
identity switches
complete track count
mean and median track length
unexpected track losses
association probability
probability margin
intensity cost
assignment objective
```

Where ground truth is available, also calculate:

```text
edge Jaccard
division accuracy
competition score
```

### 20.5 Assignment-difference table

For every changed assignment:

```text
frame
track_id
legacy_cell_id
new_cell_id
legacy_pair_cost
new_pair_cost
legacy_intensity_cost
new_intensity_cost
legacy_probability
new_probability
legacy_margin
new_margin
distance_um
```

Every changed assignment should be reviewable in the general scene visualizer.

### 20.6 Stage 08 metrics

Compare:

```text
merge onset candidate count
intensity-error distribution
candidates rejected by intensity
accepted merge onsets
merge repairs
false merge evidence
```

---

## 21. Suggested acceptance criteria

A future implementation should not become the default unless:

- all existing tests pass;
- legacy mode reproduces the previous Stage 07 output;
- all normal frames produce valid references;
- gain-normalized features pass synthetic gain invariance;
- candidate discrimination improves consistently across samples;
- manually verified association accuracy does not decrease;
- unexpected track loss does not increase;
- probability margins do not materially degrade;
- Stage 08 does not show unexplained merge behavior changes;
- final evaluation metrics improve or remain neutral;
- benefits are not limited to one sample.

A result that merely reduces temporal variance is insufficient. It must retain
or improve discrimination between different cells.

---

## 22. Risks

### 22.1 Removing real biological change

Population-wide biological brightening or dimming would be interpreted as
frame gain and removed from normalized features.

Mitigation:

- retain raw features;
- use normalized features only for tracking;
- record frame references;
- analyze population-level raw trends separately.

### 22.2 Reference contamination

Changes in population composition, segmentation errors, or many boundary cells
could bias the frame reference.

Mitigation:

- robust median estimator;
- boundary exclusion;
- minimum cell count;
- fallback diagnostics;
- per-frame reliability flag.

### 22.3 New segmentation behavior changes the reference

Because the reference is estimated from detected instances, a future
segmentation change could alter the normalized features.

Mitigation:

- record segmentation version or processing metadata;
- compare references after segmentation changes;
- keep reference diagnostics in every cell table.

### 22.4 Stage 08 semantic changes

Changing the source of `intensity_sum` may affect merge evidence.

Mitigation:

- explicit feature resolution;
- legacy fallback;
- isolated Stage 08 validation;
- no initial threshold retuning.

### 22.5 Schema complexity

Keeping legacy, raw, and normalized feature families increases CSV width.

Mitigation:

- explicit prefixes;
- one canonical per-cell table;
- document feature purpose;
- remove legacy features only after a formal migration.

---

## 23. Non-goals of the first future implementation

The initial experiment should not:

- alter Stage 01 preprocessing;
- alter Stage 02 masking;
- alter Stage 03 segmentation;
- alter Stage 04 detection;
- add weak Gaussian filtering;
- add local background subtraction;
- change Stage 07 intensity weights;
- retune Stage 08 thresholds;
- remove existing intensity columns;
- redesign Stage 06 storage;
- claim biological calibration of raw intensity.

---

## 24. Possible later local-background experiment

If frame-gain normalization is validated, a separate experiment may compare:

```text
raw
gain-normalized raw
local-background-subtracted raw
gain-normalized + local-background-subtracted raw
```

A possible local background estimator would use a shell around each instance,
excluding all labeled cells.

Required concerns:

- anisotropic physical shell construction;
- dense regions with insufficient background voxels;
- image-boundary truncation;
- neighbouring-cell contamination;
- fallback to larger local regions;
- reliability metadata.

Possible future columns:

```text
intensity_local_background_median
intensity_local_background_mad
intensity_background_corrected_mean
intensity_background_corrected_median
intensity_background_corrected_iqr
intensity_background_estimate_method
intensity_background_voxel_count
intensity_background_reliable
```

This must remain separate from frame-gain normalization because additive local
background and multiplicative global gain are different effects.

---

## 25. Suggested future implementation sequence

### Change set 1 — feature generation only

Potential files:

```text
src/05_feature_extraction/config.py
src/05_feature_extraction/intensity.py
src/05_feature_extraction/features.py
src/05_feature_extraction/pipeline.py
src/dataset_processing.py
diagnostics/pipeline_replay/runner.py
diagnostics/pipeline_replay/comparison.py
tests/test_feature_intensity_sources.py
tests/test_pipeline_replay.py
```

Tracking remains in legacy mode.

### Change set 2 — optional Stage 07 and Stage 08 modes

Potential files:

```text
src/07_cell_tracking/step01_config.py
src/07_cell_tracking/step02_association.py
src/07_cell_tracking/step03_pipeline.py
src/08_track_stitching/step01_pipeline.py
existing Stage 07 tests
existing Stage 08 tests
```

Default remains:

```text
legacy_processed
```

### Change set 3 — validation tooling

Add:

- legacy versus candidate comparison runner;
- assignment-difference table;
- summary metrics;
- visualizer integration.

### Change set 4 — isolated default switch

Only after validation:

```python
TRACKING_INTENSITY_MODE = "gain_normalized_robust"
```

The default switch should be a small, independent commit.

---

## 26. Rollback strategy

The design should preserve:

- legacy processed intensity columns;
- legacy Stage 07 mode;
- legacy Stage 08 intensity fallback;
- unchanged Stage 06 paths.

Rollback would therefore require only selecting:

```text
legacy_processed
```

No data migration should be necessary.

---

## 27. Conditions that should trigger reconsideration

Revisit this proposal when one or more of the following occur:

1. Stage 07 intensity features are implicated in manually verified tracking
   failures.
2. Multi-sample analysis confirms shared frame-wide intensity drift.
3. A no-intensity baseline performs worse than gain-normalized intensity.
4. Gain-normalized robust features improve manually verified associations.
5. Stage 08 merge intensity conservation improves across samples.
6. Ground-truth or competition metrics show a measurable benefit.
7. A later imaging-data specification clarifies that population-wide intensity
   changes are acquisition artifacts rather than biological signal.

Do not implement the proposal merely because it is architecturally cleaner.

---

## 28. Open questions

Before implementation, answer:

1. Is the frame-wide trend an acquisition artifact or a biological population
   trend?
2. Is the trend multiplicative, additive, or mixed?
3. Should the reference use all cells, stable cells, or background?
4. How stable is the reference under segmentation-count changes?
5. Does gain normalization improve identity preservation on manually verified
   difficult scenes?
6. Does a no-intensity tracker perform equally well?
7. Does normalized sum improve Stage 08 merge conservation?
8. Are per-sample and cross-sample intensity semantics required?
9. Should raw intensity values remain `uint16`-scale floats or be calibrated to
   physical fluorescence units if metadata becomes available?
10. When, if ever, should legacy processed intensity columns be removed?

---

## 29. Codex handoff checklist for a future implementation

When this experiment is approved, instruct Codex to:

- inspect the latest repository before editing;
- preserve current Stage 01–04 algorithms;
- preserve current artifact paths and identifiers;
- preserve all legacy Stage 05 columns and values initially;
- add explicit raw and gain-normalized feature families;
- use production instance labels as the only per-cell mask;
- calculate one robust reference per complete frame;
- implement deterministic fallbacks and reliability metadata;
- keep the Stage 05 positional API backward compatible;
- pass `raw_volume=raw` only from canonical Stage 06;
- update exact and local replay semantics explicitly;
- add optional Stage 07 feature-mode resolution;
- keep Stage 07 weights unchanged initially;
- add visible old-dataset fallback;
- add optional Stage 08 normalized-sum resolution;
- keep Stage 08 thresholds unchanged initially;
- add focused unit and regression tests;
- produce a baseline-versus-candidate comparison report;
- avoid Git operations unless explicitly requested;
- do not make the new mode the default without validation evidence.

---

## 30. Current decision

**No production implementation is currently recommended.**

The current segmentation preprocessing remains appropriate for segmentation.
The existing intensity representation remains usable as a tracking feature, even
though preliminary evidence suggests it may not be optimal.

The proposed raw and frame-gain-normalized feature branch should remain a
documented future experiment until multi-sample, manually verified, downstream
evidence justifies the added complexity and semantic change.
