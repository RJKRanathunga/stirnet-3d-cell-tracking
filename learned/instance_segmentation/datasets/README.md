# Learned Instance Segmentation — External Datasets

This package converts densely annotated 3-D microscopy datasets into the four
input channels and dense supervision targets consumed by `VectorInstanceCNN`.
The dataset-specific boundary is intentionally narrow: each adapter produces an
`AnnotatedVolume`; every later operation is shared across C. elegans, NIS3D,
and BlastoSPIM.

```text
native dataset files
        ↓
adapters/<dataset>.py
        ↓
AnnotatedVolume [Z,Y,X] + physical spacing
        ↓
physical adjacency / neighboring-pair selection
        ↓
centered physical crop + resampling to Biohub grid
        ↓
synthetic Stage-2 merged component
        ↓
physical EDT + production effective-marker detector
        ↓
4-channel CNN input + foreground/vector/boundary/center targets
```

## Implemented adapters

### C. elegans nuclei

Adapter: `adapters/c_elegans.py`

Default source spacing in internal ZYX order:

```text
(0.122, 0.116, 0.116) µm
```

Root resolution:

1. explicit `--root`,
2. `C_ELEGANS_NUCLEI_DIR`,
3. the current project Windows path when present,
4. `data/external/c_elegans_nuclei`.

### NIS3D

Adapter: `adapters/nis3d.py`

The adapter recursively discovers sample folders containing `Data.tif` and
`GroundTruth.tif` (with common mirror aliases such as `gt.tif`). If present,
`ConfidenceScore.tif` is loaded and NIS3D undefined voxels are excluded through
the canonical `valid_mask`. Physical spacing is parsed from `Info.txt` when
possible. An explicit override can always be supplied with
`--spacing-zyx-um Z Y X`.

Root resolution:

1. explicit `--root`,
2. `NIS3D_DIR`,
3. `data/external/nis3d` / `data/external/NIS3D`.

### BlastoSPIM

Adapter: `adapters/blastospim.py`

The adapter recursively searches TIFF/NPY volumes and pairs raw-image files
with instance-label files. It supports an umbrella directory containing several
extracted BlastoSPIM archives or a root pointing directly at one archive. When
both corrected/expert segmentation and expected/automatic segmentation are
present, corrected/ground-truth annotations are preferred.

Default source spacing in internal ZYX order:

```text
(2.0, 0.208, 0.208) µm
```

Root resolution:

1. explicit `--root`,
2. `BLASTOSPIM_DIR`,
3. `data/external/blastospim` / `data/external/BlastoSPIM`.

Because BlastoSPIM releases have multiple archive/layout variants, run
`inspect_volume --list` immediately after extraction. If your downloaded tree
uses an unrecognized naming pattern, the adapter will fail explicitly rather
than silently pairing unrelated files.

## Canonical model grid

All training samples are generated at the Biohub model grid:

```text
shape   = (16, 64, 64) [Z,Y,X]
spacing = (1.625, 0.40625, 0.40625) µm [Z,Y,X]
```

Whole external volumes are not blindly resampled. Pair selection happens in the
native annotation grid; only the physical crop required for a sample is
resampled.

## Valid-pair indexing

Some source nuclei are too small to survive conversion to the Biohub target
grid, especially datasets with much finer Z sampling. `SampleBuilder` correctly
rejects those examples. Debugging `--pair-index` is therefore an index among
**buildable/valid pairs**, not raw adjacency edges.

For example, if raw pairs 0 and 1 disappear after resampling and raw pair 2 is
valid:

```text
--pair-index 0  → raw pair 2
```

Use `--raw-pair-index N` when you specifically want to reproduce one rejected
raw candidate. Rejection messages include native voxel count, native bounding
box, physical bounding-box size, and physical volume.

## Inspect dataset discovery first

All debugging commands use the same generic interface.

### C. elegans

```powershell
python -m learned.instance_segmentation.datasets.debugging.inspect_volume `
  --dataset c_elegans `
  --root "D:\Projects\Kaggle\cell-tracking\data\external\c_elegans_nuclei" `
  --split train --index 0
```

### NIS3D

```powershell
python -m learned.instance_segmentation.datasets.debugging.inspect_volume `
  --dataset nis3d `
  --root "D:\Projects\Kaggle\cell-tracking\data\external\NIS3D" `
  --list
```

Then inspect a discovered volume:

```powershell
python -m learned.instance_segmentation.datasets.debugging.inspect_volume `
  --dataset nis3d `
  --root "D:\Projects\Kaggle\cell-tracking\data\external\NIS3D" `
  --index 0 --napari
```

If a volume's `Info.txt` does not expose parseable spacing, supply the known
spacing explicitly:

```text
--spacing-zyx-um Z Y X
```

### BlastoSPIM

```powershell
python -m learned.instance_segmentation.datasets.debugging.inspect_volume `
  --dataset blastospim `
  --root "<PATH_TO_EXTRACTED_BLASTOSPIM>" `
  --list
```

Inspect one record:

```powershell
python -m learned.instance_segmentation.datasets.debugging.inspect_volume `
  --dataset blastospim `
  --root "<PATH_TO_EXTRACTED_BLASTOSPIM>" `
  --split train --index 0 --napari
```

If the downloaded TIFF stack is stored in a different axis order, specify it
explicitly, for example `--source-axis-order xyz`. The code does not guess axis
permutations.

## Build one complete CNN sample

The same command works for every adapter:

```powershell
python -m learned.instance_segmentation.datasets.debugging.inspect_sample `
  --dataset c_elegans `
  --root "D:\Projects\Kaggle\cell-tracking\data\external\c_elegans_nuclei" `
  --split train --volume-index 0 --pair-index 0 --napari
```

Replace `--dataset` and `--root` for NIS3D or BlastoSPIM.

The command:

1. finds naturally neighboring GT instances,
2. skips pairs that cannot safely survive canonical resampling,
3. keeps the real fluorescence crop unchanged,
4. corrupts only the Stage-2-like foreground mask to create one component,
5. computes physical EDT from that imperfect input mask,
6. runs the repository's actual Stage-3 effective-peak detector with geometric
   completion disabled,
7. creates the marker Gaussian channel,
8. creates foreground, physical center-vector, boundary, and center targets.

## Scan pair usability

Before training on a new source, quantify how many neighboring pairs remain
usable at Biohub resolution:

```powershell
python -m learned.instance_segmentation.datasets.debugging.scan_pairs `
  --dataset c_elegans `
  --root "D:\Projects\Kaggle\cell-tracking\data\external\c_elegans_nuclei" `
  --split train --volume-index 0 --max-pairs 100
```

The report separates failures such as:

- too small after resampling,
- crop-border truncation,
- merge-connectivity failure,
- marker-generation failure,
- target-generation failure.

This is useful for deciding how strongly each external source should contribute
to training.

## CNN input contract

`TrainingSample.inputs` is `float32 [4,Z,Y,X]`:

1. robust-normalized fluorescence,
2. Stage-2-like foreground/component mask,
3. normalized physical EDT,
4. effective-marker Gaussian heatmap.

## Target contract

- `foreground`: `[1,Z,Y,X]`
- `vectors_normalized`: `[3,Z,Y,X]`, physical `(Z,Y,X)` offsets divided by
  `vector_max_distance_um`
- `boundary`: `[1,Z,Y,X]`
- `center`: `[1,Z,Y,X]`
- `instance_labels`: `[Z,Y,X]`, local IDs `0..K`
- `valid_mask`: `[1,Z,Y,X]`

The default `vector_max_distance_um=16.0` must remain synchronized with
`model.VectorCNNConfig.vector_max_distance_um`.

## Anti-leakage rule

EDT and the effective-marker channel are always calculated from the corrupted
Stage-2-like **input mask**. They are never generated from GT centers or GT
instance count. Dense instance labels are used only to construct supervision.

## Tests

From the repository root:

```powershell
python -m pytest learned/instance_segmentation/datasets/tests -q
```

The tests include synthetic adapter layouts for C. elegans, NIS3D and
BlastoSPIM, axis conversion, NIS3D confidence masking, valid-pair skipping,
resampling, mask corruption, target generation, and the complete sample-builder
contract.

## Dependencies

Required:

- `numpy`
- `scipy`
- `tifffile`

Optional:

- `napari` for visual debugging
- `pytest` for tests

The effective-marker channel additionally requires the repository's existing
`src/03_segmentation` implementation at runtime.
