# Effective Peak Visualization Findings

## Context

This diagnostic used:

```text
visualize_effective_peaks_over_time.py
```

to inspect the canonical Stage 3 peak outputs across all 20 processed frames.

The visualizer compares four concepts:

- **Raw peaks**: multi-scale EDT peak candidates before same-lobe collapsing.
- **Effective peaks**: peaks retained after same-lobe collapsing.
- **Current Stage 3 markers**: peaks selected by the existing H1/H2/H3 hypothesis decision.
- **Saved Stage 4 centroids**: geometric centroids calculated from the saved production instance labels.

The script uses the complete saved Stage 2 binary mask and the canonical Stage 3 implementation with diagnostic artifacts enabled. It does not modify production data or parameters.

## Main observations

Across the 20 inspected frames, effective peaks were usually consistent with the visible biological cell bodies.

Only two convincing cases were found where one apparent cell contained more than one effective peak:

1. One ordinary cell with two effective peaks.
2. One cell at the volume boundary, where the cell body was partially cut by the image boundary and its geometry was incomplete.

The boundary case is not strong evidence of a general effective-peak failure because partially observed cells can produce ambiguous distance geometry.

In contrast, many previously suspected merged-cell failures already contained distinct effective peaks for the individual visible cell bodies.

## Primary finding

The dominant Stage 3 failure appears to occur **after effective-peak generation**.

The observed sequence is:

```text
Correct Stage 2 binary component
→ raw EDT peaks
→ effective peaks that usually correspond to individual cells
→ H1/H2/H3 hypothesis selection
→ valid effective peaks are discarded
→ too few watershed markers are retained
→ multiple cells remain merged in the final labels
```

The current hypothesis model can represent only one, two, or three cells per connected component. Large connected clusters can contain more than three valid effective lobes, especially when several cells touch through short or thin necks.

For such components, every available H1/H2/H3 explanation is incomplete. The algorithm still selects the best available incomplete hypothesis, which can force several valid cell centers into one merged output.

## Interpretation

The effective-peak stage appears substantially more reliable than initially expected.

For this inspected subset:

- effective peaks rarely over-segmented ordinary cells;
- most merged-cell failures already had separate effective peaks;
- the fixed-count hypothesis stage frequently removed useful markers;
- the final merged outputs therefore do not necessarily indicate failure in peak detection or same-lobe collapsing.

This suggests that effective peaks should be treated as provisional cell centers rather than merely as candidates for a global H1/H2/H3 cell-count decision.

## Recommended algorithm direction

A more appropriate Stage 3 flow is:

```text
Raw peaks
→ same-lobe collapse
→ use all effective peaks as watershed markers
→ validate watershed children
→ locally merge only children that are likely to belong to the same cell
→ final instances
```

This avoids imposing a global maximum of three cells on a connected component.

The local validation and merge stage should consider:

- child voxel count;
- equivalent physical radius;
- connectedness;
- marker containment;
- child volume fraction;
- shared interface area;
- EDT depth at the shared interface;
- merge-tree branch persistence;
- same-lobe probability;
- shape plausibility before and after merging;
- whether the child touches the image boundary.

## Boundary handling

Boundary cells should be treated separately from fully observed cells.

A partially cut boundary cell may produce:

- distorted EDT maxima;
- multiple effective peaks;
- incomplete shape evidence;
- misleading child-volume fractions.

Possible handling strategies include:

- marking boundary components as uncertain;
- applying more conservative splitting near the boundary;
- excluding severe boundary cases from parameter calibration;
- evaluating boundary cells with dedicated rules.

## Important caveat

This inspection is strong evidence for the current sample and 20-frame subset, but it is not yet a full benchmark.

The conclusion should be validated on:

- additional samples;
- elongated single cells;
- strongly merged two-cell cases;
- three-cell clusters;
- large multi-cell clusters;
- noisy protrusions;
- unequal cell sizes;
- boundary cells.

## Next experiment

Add an experimental Stage 3 mode that uses every effective peak as a watershed marker.

Keep the existing hypothesis path temporarily for comparison:

```text
hypothesis mode
    current H1/H2/H3 production behavior

all-effective mode
    all effective peaks become initial watershed markers
```

For each frame, compare:

- effective peak count;
- final child count;
- recovered merged cells;
- newly introduced false splits;
- boundary-related failures;
- child-size distribution;
- production cell count;
- tracking continuity in later stages.

## Provisional conclusion

The current evidence indicates that the main Stage 3 limitation is not failure to discover individual cell centers. The algorithm often discovers them correctly and then discards them during the fixed H1/H2/H3 hypothesis decision.

The next implementation should therefore test an arbitrary-count, all-effective-peak watershed followed by conservative local child merging.
