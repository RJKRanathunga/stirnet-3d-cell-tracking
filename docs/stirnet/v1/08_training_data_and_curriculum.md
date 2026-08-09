# 08 — Training Data, Corruption, and Curriculum

## 1. Training objective

The network must learn:

```text
imperfect instances + raw image + imperfect temporal clues
                ->
correct instances
```

Therefore training data must reproduce the kinds of errors produced by the actual pipeline.

## 2. Data sources

Use three sources.

### 2.1 Clean annotated sequences

Input segmentation is GT or near-GT.

Purpose:

- learn no-op behavior;
- learn correct boundaries;
- prevent aggressive over-correction.

### 2.2 Synthetic corruptions of annotated sequences

Create controlled, realistic segmentation failures.

Purpose:

- large-scale supervision;
- balance rare failure categories;
- generate merges and missing-cell cases cheaply.

### 2.3 Real pipeline failures

Run the actual current segmentation pipeline and collect failures with GT or manually validated corrections.

Purpose:

- match deployment error distribution;
- reveal failure mechanisms synthetic corruption misses.

## 3. Initial clean/corrupt mixture

Starting sampling policy:

```text
40% clean
60% corrupted
```

This is tunable.

The key requirement is that clean examples remain common.

## 4. Synthetic corruption operators

### 4.1 Adjacent-cell merge

Choose physically adjacent GT cells and replace their separate input labels with one current instance.

Preserve raw image unchanged.

Possible variants:

- 2-to-1;
- 3-to-1;
- cluster merges.

### 4.2 Over-segmentation

Split one GT cell using:

- perturbed watershed;
- random plane;
- marker overpopulation;
- shape-aware partition.

Input fragments should remain plausible.

### 4.3 Missing cell

Delete one GT instance from the input labels.

Emphasize:

- small cells;
- weak cells;
- crowded cells;
- temporally persistent cells.

### 4.4 False positive

Insert a plausible non-cell object or fragment.

Avoid obviously synthetic shapes that create shortcuts.

### 4.5 Boundary erosion

Shrink instance mask.

### 4.6 Boundary dilation

Expand into local background or neighbouring regions.

### 4.7 Wrong partition

Move a separation surface between neighbours.

### 4.8 Marker corruption

Delete, duplicate, or shift the marker heatmap.

### 4.9 Small-cell loss

Preferentially remove small true instances to reflect observed segmentation failure modes.

## 5. Corruptions happen on native grids

Do not resample before corruption.

All operations must preserve:

```text
native raw image
native spacing metadata
native GT masks
```

## 6. Temporal corruption pipeline

The most realistic training process is:

```text
GT sequence
 |
synthetic/current-pipeline segmentation corruption
 |
Trackastra pass 1
 |
provisional temporal graph
 |
STIR-Net training example
```

Trackastra should therefore be run **after segmentation corruption**, not on the clean GT and then paired with corrupted masks.

## 7. Offline cache

Do not run Trackastra inside every SGD step.

Create an offline cache containing:

```text
sample id
target time
native raw volume references
corrupted instance labels
Trackastra graph tensors
tracklet mapping
temporal references
status features
GT instances
corruption metadata
```

The training loader performs local augmentation/cropping on top of the cached structural data.

## 8. Temporal clue corruption

Trackastra clues are intentionally imperfect during training.

Starting perturbations:

```text
10% temporal-hypothesis dropout
10% graph-edge dropout
position jitter ~ N(0, 0.1*dref)
occasional larger jitter up to ~0.5*dref
~5% plausible false temporal hypothesis insertion
association-confidence corruption/dropout
```

All percentages are starting values.

## 9. Why corrupt temporal clues?

Without temporal corruption the model can learn:

```text
Trackastra clue == truth
```

which is undesirable.

Desired rule:

```text
Trackastra clue == evidence
```

The image and current segmentation must still be able to override temporal priors.

## 10. Spatial augmentation

Apply consistently to:

- raw image;
- current labels;
- GT labels;
- centers;
- graph coordinates;
- motion vectors;
- temporal references.

Recommended:

```text
axis flips
permitted 90-degree rotations
small physical rotations
small elastic deformation
intensity scale / shift
gamma
noise
blur
```

Do not rotate coordinates without rotating all dependent geometry.

## 11. Multi-dataset training

Every dataset should provide:

```text
native spacing
native image
native labels
physical units
```

Cross-dataset normalization occurs through:

- explicit physical coordinates;
- \(d_\text{ref}\)-normalized geometry;
- acquisition conditioning;
- intensity augmentation.

No dataset is forced to imitate BioHub's voxel dimensions.

## 12. Sampling hard examples

Oversample patches containing:

```text
true merged cases
3+ cell clusters
small cells
dense neighbours
weak intensity
interior track end
interior track start
short temporal gap
division neighbourhood
boundary cells
hard but correct touching cells
```

Easy isolated cells should not dominate optimization.

## 13. Curriculum

### Phase A — spatial correction pretraining

Enable:

- physical-aware spatial encoder/decoder;
- primary queries;
- split companion queries;
- discovery queries;
- query decoder;
- matcher/losses.

Disable:

- graph encoder;
- temporal queries;
- co-reasoning.

Goal:

> Geometry alone must learn useful instance correction.

### Phase B — temporal integration

Load Phase-A weights.

Enable:

- detection graph encoder;
- temporal hypothesis construction;
- temporal repair queries;
- one coarse co-reasoning block.

Goal:

> Prove temporal clues improve ambiguous cases.

### Phase C — full V1

Enable:

- second co-reasoning block;
- anomaly salience;
- reliability;
- temporal clue corruption;
- end-to-end training of STIR-Net.

Trackastra remains frozen and external.

## 14. Optimizer defaults

Starting values:

```python
optimizer = AdamW
lr = 2e-4
weight_decay = 1e-4
max_grad_norm = 1.0
```

Schedule:

```text
linear warmup
+
cosine decay
```

Use mixed precision where supported.

## 15. Checkpoints

Save:

```text
model state
optimizer state
scheduler state
scaler state
epoch / global step
config
dataset manifests
normalization statistics
metric history
git commit if available
```

The model code is expected under `learned/stirnet/`, but checkpoints should be stored in a dedicated learned-model/output location rather than committed to source control.
