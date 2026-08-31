# dataset_curation

Production human-in-the-loop curation package for the BioHub cell-tracking
dataset.

## Data layout

The production data root is `E:\data\biohub`.

```text
E:\data\biohub\
├── source\
│   ├── train\<volume-id>\<volume-id>.zarr
│   └── test\<volume-id>\<volume-id>.zarr
├── preprocessed\
│   ├── train\<volume-id>\<run-id>\...
│   └── test\<volume-id>\<run-id>\...
└── annotations\
    ├── train\<volume-id>\<annotation-set>\...
    └── test\<volume-id>\<annotation-set>\...
```

`source/` is input-only. Train `ground_truth/` CSV files are sparse graph
supervision; they are not treated as complete full-volume ground truth.

## Architecture

Spatial inference is owned by `learned/stirnet/inference/`.
`dataset_curation` consumes that production API directly.

There is no `_compat/` layer, generic legacy workspace, Investigation36 backend
alias, historical point annotator, or compatibility evaluation wrapper. Git
history is the recovery mechanism.

Instance correction lives in:

```text
annotation/instances/
├── io.py
├── session.py
├── viewer.py
└── curation_runner.py
```

Track correction lives in:

```text
annotation/tracks/
├── graph.py
├── storage.py
├── session.py
├── viewer.py
└── curation_runner.py
```

## Commands

```powershell
python -m dataset_curation status --split all
python -m dataset_curation infer --split train --count 5
python -m dataset_curation infer --split train --id 44b6_0c582fdc

python -m dataset_curation annotate-instances --next
python -m dataset_curation annotate-instances --resume
python -m dataset_curation annotate-instances --id 44b6_0c582fdc

python -m dataset_curation annotate-tracks --next
python -m dataset_curation annotate-tracks --resume
python -m dataset_curation annotate-tracks --id 44b6_0c582fdc
```
