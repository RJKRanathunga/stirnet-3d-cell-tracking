# Effective Peak Validation

This investigation validates whether Stage 3 effective peaks correspond to
individual biological cell bodies across complete processed frames.

It was created after merged-instance failures suggested that Stage 3 was
detecting valid cell centers but discarding some of them later in the pipeline.

## Files

- `visualize_effective_peaks_over_time.py`  
  Runs canonical Stage 3 diagnostics across multiple frames and opens a 3D
  Napari viewer.

- `findings.md`  
  Records the observations, validated conclusion, current production behavior,
  and remaining caveats.

## Investigation question

The main question was:

> Do effective peaks already represent the correct instance centers, or do they
> frequently over-segment individual cells?

The investigation compared:

- raw multi-scale EDT peak candidates;
- effective peaks retained after same-lobe collapse;
- final Stage 3 markers;
- saved Stage 4 centroids from production instance labels.

## Validated result

Across the inspected 20-frame subset:

- effective peaks usually corresponded to visible biological cell bodies;
- most suspected merged-cell failures already contained separate effective
  peaks for the individual cells;
- components with more than three valid peaks occurred in briefly touching
  cell clusters;
- only two convincing duplicate-effective-peak cases were found:
  - one ordinary cell;
  - one partially cut boundary cell.

This showed that the former fixed H1/H2/H3 selection stage was discarding valid
effective peaks and causing merged-instance failures.

## Current Stage 3 behavior

Stage 3 now uses every effective peak as a final watershed marker:

```text
complete 6-connected binary component
→ raw EDT peak detection
→ pair evidence
→ same-lobe collapse
→ every effective peak becomes a final marker
→ marker-controlled watershed using all effective peaks
→ final instance labels
```

There is no global cell-count limit and no truncation of effective peaks.

- One effective peak preserves the complete component as one instance.
- Multiple effective peaks produce one deterministic watershed child per peak.
- Marker IDs and child labels are assigned in effective-peak order.

## Running the visualizer

Run from the repository root:

```powershell
python investigations/stage_03_segmentation/peak_selection/visualize_effective_peaks_over_time.py `
    --sample-id 44b6_0113de3b `
    --frames 20
```

Force recomputation after changing Stage 3 parameters or implementation:

```powershell
python investigations/stage_03_segmentation/peak_selection/visualize_effective_peaks_over_time.py `
    --sample-id 44b6_0113de3b `
    --frames 20 `
    --recompute
```

Use another sample by changing `--sample-id`.

## Napari layers

The visualizer includes these main layers.

### Raw

The original TZYX image volume.

### Raw peaks

Multi-scale EDT peak candidates before same-lobe collapse.

These are intentionally overcomplete and are useful for diagnosing the peak
detector.

### Effective peaks

Peaks retained after same-lobe collapse.

These are now the final Stage 3 watershed markers.

### Saved Stage 4 centroids

Geometric centroids calculated from the saved production instance labels.

These are a comparison layer only. They are not EDT peaks or Stage 3 markers.

### Optional mask and label layers

The saved Stage 2 binary mask and production instance labels may be enabled for
spatial comparison.

## Time navigation

The viewer uses native TZYX time navigation.

Use the Napari time slider or the normal Ctrl+mouse-wheel interaction to move
between frames.

Point layers use:

```python
out_of_slice_display = False
```

so points from neighboring timepoints are not displayed on the current frame.

## Cache

Computed peak diagnostics are cached under:

```text
data/diagnostics/effective_peaks/<sample_id>/
```

The cache is versioned. Use `--recompute` whenever the Stage 3 implementation,
configuration, or diagnostic schema changes.

## How to interpret disagreements

### Multiple effective peaks inside one production instance

This may indicate:

- a previously missed split;
- a genuine duplicate effective peak;
- incomplete geometry near the image boundary;
- an outdated saved production result generated before the current Stage 3
  algorithm.

Inspect the Stage 2 mask, effective peaks, final labels, and saved centroid layer
together before classifying the case.

### One effective peak covering several visible cells

This indicates that raw peak detection or same-lobe collapsing failed to
preserve separate centers.

### Several effective peaks in a large connected component

This is expected when multiple cells touch through narrow or brief contacts.
The current production algorithm uses all of them without imposing a maximum
cell count.

## Boundary caveat

Partially observed cells at the volume boundary may produce distorted distance
geometry and duplicate effective peaks.

Boundary cases should be recorded separately and should not be used alone to
tune global same-lobe-collapse parameters.

Future validation should include:

- additional samples;
- elongated single cells;
- strong cell overlaps;
- noisy protrusions;
- unequal cell sizes;
- large connected clusters;
- dedicated boundary cases.

## Outcome

This investigation directly motivated the removal of the fixed H1/H2/H3
hypothesis-selection architecture.

Effective peaks are now used as arbitrary-count final instance markers, and the
production, dedicated diagnostic, and pipeline replay paths all follow the same
Stage 3 behavior.
