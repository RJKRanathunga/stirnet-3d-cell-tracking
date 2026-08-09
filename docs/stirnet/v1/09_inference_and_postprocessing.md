# 09 — Inference and Postprocessing

## 1. End-to-end inference

```text
preprocessed native sequence
 |
current instance segmentation
 |
Trackastra pass 1
 |
temporal clue extraction
 |
patch/tile generation
 |
STIR-Net
 |
query filtering
 |
native-resolution mask rendering
 |
voxel conflict resolution
 |
instance assembly
 |
Trackastra pass 2
```

## 2. Physical tiling

Inference tiles are defined in physical/cell-scale space.

Context size:

$$8d_\text{ref}$$

Valid ownership core:

$$6d_\text{ref}.$$

Tiles overlap so every output cell center belongs to exactly one valid core.

## 3. Tile ownership rule

A query output is owned by a tile if:

```text
predicted center lies inside the tile's valid core
```

The mask itself may extend into the context region.

This prevents duplicate ownership across overlapping tiles.

## 4. Candidate filtering before full mask rendering

For all queries compute:

- existence probability;
- predicted center;
- coarse mask;
- query type.

Initial candidate criterion:

```python
exist_prob > 0.30
```

for deciding whether to render a full native mask.

This threshold is intentionally lower than the final threshold to avoid prematurely dropping valid cells.

## 5. Final existence filtering

After native mask rendering and basic mask validation:

```python
final_exist_threshold = 0.50
```

as a starting value.

Tune on validation data.

## 6. Native mask rendering

For each surviving query:

$$L_i(v)=m_i^TF_{mask}(v)+L_i^{prior}(v).$$

Probability:

$$P_i(v)=\sigma(L_i(v)).$$

Do not render masks for discarded low-existence queries.

## 7. One connected component per query

Threshold each candidate mask.

Then:

```text
connected components
 |
select component containing predicted center
```

If center voxel is not inside any positive component, select the component with minimum physical distance to the predicted center.

Reject tiny disconnected components.

## 8. Voxel conflict resolution

Multiple cell masks may overlap.

Define:

$$score_i(v) = P_i(\text{exist}) \cdot P_i(v).$$

Assign each voxel to the surviving query with maximal score, subject to a minimum mask score threshold.

Optional auxiliary foreground probability may suppress assignments in strong background regions.

## 9. Background rule

If no query score exceeds threshold:

```text
voxel = background
```

Do not force every current foreground voxel to remain foreground.

This is necessary for false-positive removal and boundary correction.

## 10. Output labels

Renumber accepted cells sequentially within the target frame:

```text
1,2,3,...
```

Original input IDs are not preserved.

Trackastra pass 2 establishes temporal identity later.

## 11. Merge correction behavior

Typical merge:

```text
input:
1 current instance
+
primary query
+
split companion
+
possibly multiple temporal queries

decoder:
two cell hypotheses survive

output:
2 masks
```

No explicit `split` operation is required after the model.

## 12. Over-segmentation behavior

Typical over-split case:

```text
input:
2 current instances
+
4 seeded primary/split queries

decoder:
only one biological hypothesis survives
others -> no-object

output:
1 mask
```

## 13. Missing-cell behavior

Possible sources:

```text
temporal repair query
or
discovery query
```

produces a valid cell where the current segmentation contains no instance.

## 14. Correct-cell behavior

```text
primary query -> same/near-same mask
split companion -> no-object
temporal duplicate -> no-object or suppressed by query competition
```

## 15. Tile stitching

Because ownership is center-based, stitching is primarily concatenation of owned instances.

For rare cross-tile conflicts:

- compare physical centers;
- compare mask overlap in overlapping context;
- suppress near-duplicate outputs.

A duplicate rule may use high 3D IoU plus center proximity.

## 16. Trackastra pass 2

After complete corrected target-frame instance labels are assembled for the sequence:

```text
corrected instance sequence
 |
Trackastra
 |
final tracks
```

Do not preserve pass-1 track IDs or assume query identity corresponds to final Trackastra identity.

## 17. Diagnostic outputs

Inference should optionally save:

```text
existence score
query type
predicted center
input-instance associations
temporal clue salience/reliability
mask confidence
number of queries before/after filtering
```

These are essential for failure analysis.

## 18. Conservative deployment mode

A future safety mode may keep the original segmentation when refinement confidence is low.

This is not part of the base V1 architecture, but inference APIs should make it possible to compare original and refined instances before committing changes.
