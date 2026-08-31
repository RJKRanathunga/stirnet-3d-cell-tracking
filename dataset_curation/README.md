
# dataset_curation

`dataset_curation` is the persistent human-in-the-loop curation layer for the
cell-tracking project.

## Current tasks migrated

Current repository sources are preserved under `_compat/` during the first
behavior-preserving migration:

- `evaluation/segmentation/scripts/02_supervoxel_instance_annotator.py`
  - merged-instance correction using atomic supervoxel seed groups,
  - weighted contact-graph expansion,
  - ray picking, Napari UI, Save/Undo/Reset, resumable corrected labels.
- `evaluation/segmentation/scripts/03_biohub_merge_suspect_export.py`
  - leak-free causal merge-suspect inference and threshold-independent scores.
- `evaluation/segmentation/scripts/annotate_points.py`
  - historical raw/binary-boundary point annotation workflow.
- `evaluation/track_annotation/01_track_annotator.py`
  - Trackastra Connect/Break/Complete Track correction using
    `(frame, spatial_instance_id)` nodes and resumable graph overrides.

The old paths become thin wrappers, so existing commands remain valid.

## Persistent layout

```text
<curation-root>/<dataset>/<sample-id>/
    sample.json
    prepared/
    inference/
        <run-id>/
            manifest.json
            movies/
            trackastra/
            ...
    annotations/
        <annotation-set>/
            manifest.json
            instances/
            tracks/
            points/
```

Inference artifacts are base predictions. Human corrections are stored in a
separate annotation set that is bound to one inference run, preventing old
annotations from silently being applied to a newer prediction with different
instance IDs.

By default the root is `data/curation/`. Set
`CELL_TRACKING_CURATION_ROOT` or pass `--root` to keep large artifacts on
another drive.

## Register a source movie

```powershell
python -m dataset_curation setup `
    --sample-id 44b6_0113de3b `
    --source-zarr <path-to-sample.zarr>
```

## Run current STIR-Net + Trackastra inference

The backend calls the existing
`investigations/stirnet/36_biohub_spatial_trackastra_visualization.py`
implementation. Model code is not duplicated.

```powershell
python -m dataset_curation infer `
    --sample-id 44b6_0113de3b `
    --run-id current `
    -- --frame-count 20
```

The output is written directly below the named curation inference run.

## Track annotation

```powershell
python -m dataset_curation annotate-tracks `
    --sample-id 44b6_0113de3b `
    --run-id current `
    --annotation-set main
```

This uses the current `evaluation/track_annotation/01_track_annotator.py`
behavior, but writes the resumable state below the curation annotation set.

## Current instance-annotation contract

The current instance annotator consumes:
- Investigation-25 final split-only instances,
- Investigation-24 atomic watershed supervoxels,
- Stage-6 preprocessing/masking/segmentation,
- raw Zarr.

Register those exact existing artifacts once:

```powershell
python -m dataset_curation register-spatial `
    --sample-id 44b6_0113de3b `
    --run-id current `
    --instances-root <inv25-sample-variant-root> `
    --supervoxels-root <inv24-h100-root> `
    --stage6-root <stage6-sample-root>
```

Generate/reuse merge-suspect scores:

```powershell
python -m dataset_curation suspects `
    --sample-id 44b6_0113de3b `
    --run-id current `
    -- --timepoints all
```

Open/resume instance correction:

```powershell
python -m dataset_curation annotate-instances `
    --sample-id 44b6_0113de3b `
    --run-id current `
    --annotation-set main `
    -- --timepoints all --suspect-threshold 0.70
```

## Why `_compat/` exists

This first migration changes architecture without rewriting the scientific or
annotation behavior. The public package boundaries already separate workspace,
inference, instance sessions/splitting, track graph/session, and point viewer
responsibilities. Once regression fixtures are established, the large
compatibility implementations can be decomposed physically behind the same
public interfaces without changing outputs or user interaction.
