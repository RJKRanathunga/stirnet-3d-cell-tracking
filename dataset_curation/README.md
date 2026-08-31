# dataset_curation

`dataset_curation` is the human-in-the-loop inference and annotation workflow
for the BioHub cell-tracking dataset.

## External data root

The production root is hard-coded as:

```text
E:\data\biohub
```

Normal commands therefore do not require a data-path argument. An advanced
`--data-root` override exists only for recovery/testing if the drive letter
changes.

The expected layout is:

```text
E:\data\biohub\
├── source\
│   ├── train\
│   │   └── <volume-id>\
│   │       ├── <volume-id>.zarr\
│   │       ├── ground_truth\
│   │       │   ├── ground_truth_nodes.csv
│   │       │   └── ground_truth_edges.csv
│   │       └── README.txt
│   └── test\
│       └── <volume-id>\
│           └── <volume-id>.zarr\
│
├── preprocessed\
│   ├── train\
│   │   └── <volume-id>\
│   │       └── current\
│   │           ├── movies\
│   │           │   ├── raw.npy
│   │           │   ├── preprocessed.npy
│   │           │   ├── binary_mask.npy
│   │           │   ├── source_instances.npy
│   │           │   ├── supervoxels.npy
│   │           │   └── final_instances.npy
│   │           ├── trackastra\
│   │           ├── cells_all.csv
│   │           └── curation_manifest.json
│   └── test\
│       └── ...
│
└── annotations\
    ├── train\
    │   └── <volume-id>\
    │       └── main\
    │           ├── manifest.json
    │           ├── instances\
    │           └── tracks\
    └── test\
        └── ...
```

`source/` is treated as read-only. Generated inference/preprocessing data never
go inside source sample directories.

`README.txt` is ignored by discovery and is never used as a source of dataset
metadata.

The train `ground_truth/` CSVs are sparse graph supervision. Their presence is
reported by `status`, but they are **not** treated as complete full-volume GT and
are never used to decide whether inference is complete. Test volumes do not
need a `ground_truth/` directory.

## Inspect downloaded volumes

```powershell
python -m dataset_curation status
```

Train and test together:

```powershell
python -m dataset_curation status --split all
```

Example columns:

```text
SPLIT  VOLUME          FRAMES  SPARSE_GT  INFERENCE  INST_ANN  TRACK_ANN
train  44b6_0c582fdc   100     yes        missing    -         -
```

## Batch inference

Run inference on the next 5 volumes that do **not** already have a complete
preprocessed/inference cache:

```powershell
python -m dataset_curation infer --split train --count 5
```

Run one exact volume:

```powershell
python -m dataset_curation infer `
    --split train `
    --id 44b6_0c582fdc
```

Run several exact IDs:

```powershell
python -m dataset_curation infer `
    --split train `
    --id 44b6_0c582fdc `
    --id 44b6_12dfb391
```

Run every currently missing train volume:

```powershell
python -m dataset_curation infer --split train --all
```

Test volumes use the same command:

```powershell
python -m dataset_curation infer --split test --all
```

The selector considers the **number of pending volumes**, not the first N
source directories. For example, `--count 5` selects 5 volumes that actually
need work even if earlier IDs are already complete.

A complete volume is skipped automatically. A partial cache is not considered
complete and is sent back through the existing Investigation-36 cache logic.
Batch processing continues to later volumes after a failure; add `--fail-fast`
to stop immediately.

Force a fresh inference for an exact volume:

```powershell
python -m dataset_curation infer `
    --split train `
    --id 44b6_0c582fdc `
    --force
```

Extra Investigation-36 arguments can be forwarded after `--`:

```powershell
python -m dataset_curation infer `
    --split train `
    --count 3 `
    -- --checkpoint <checkpoint-path>
```

The Zarr time dimension is read from Zarr metadata, so a 100-frame full volume
is automatically passed to inference as 100 frames.

## Annotation: one volume at a time

### Instance correction

Open the next inference-ready volume that has never had an instance-annotation
session opened:

```powershell
python -m dataset_curation annotate-instances --next
```

Because `--next` is the default, this is equivalent:

```powershell
python -m dataset_curation annotate-instances
```

Resume the most recently opened instance-annotation session:

```powershell
python -m dataset_curation annotate-instances --resume
```

Open a specific volume:

```powershell
python -m dataset_curation annotate-instances `
    --id 44b6_0c582fdc
```

Limit the loaded frames when desired:

```powershell
python -m dataset_curation annotate-instances `
    --id 44b6_0c582fdc `
    --timepoints 0-19
```

The current supervoxel split algorithm, ray picking, Save, Undo, Reset/Escape,
unique colors, and resumable correction files are reused unchanged.

A `_session.json` marker is written when the viewer is opened. This is
intentional: if you inspect an entire volume and find **zero corrections**, it
still counts as already reviewed for `--next`. If the session was interrupted,
`--resume` reopens it.

### Track correction

Next fresh track-annotation volume:

```powershell
python -m dataset_curation annotate-tracks --next
```

Resume the most recently opened track annotation:

```powershell
python -m dataset_curation annotate-tracks --resume
```

Exact volume:

```powershell
python -m dataset_curation annotate-tracks `
    --id 44b6_0c582fdc
```

The existing Trackastra Connect/Break/Complete Track, arbitrary-gap edges,
division one-to-many edges, binary-mask picking, undo history, and persistent
graph overrides are reused.

## Why supervoxels are now part of inference output

The current STIR-Net spatial inference already computes atomic watershed
supervoxels. Investigation 36 previously discarded that tensor after each
frame. Dataset curation needs it for merged-cell correction, so the cache now
persists:

```text
movies/supervoxels.npy
```

This means newly inferred full BioHub volumes are annotation-ready without
registering old Investigation-24/25 output directories.
