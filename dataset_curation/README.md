# Dataset Curation

Production BioHub inference and unified human annotation.

<!-- DATASET_CURATION_CANONICAL_SKIP_V1 -->

## Commands

```powershell
python -m dataset_curation status --split all
python -m dataset_curation infer --split train --count 5
python -m dataset_curation infer --split train --id <volume-id>
python -m dataset_curation annotate --next
python -m dataset_curation annotate --resume
python -m dataset_curation annotate --id <volume-id>
python -m dataset_curation view-source --id <volume-id>
```

There is exactly one canonical production inference cache per source volume.
There is no inference `run_id` and no `current/` subdirectory.

## Canonical inference storage

```text
preprocessed/<split>/<volume>/
├── movies/
│   ├── supervoxels.npy
│   └── final_instances.npy
├── cells/
│   └── tXXX.csv
├── cells_all.csv
├── spatial_summary.json
├── _SPATIAL_SUCCESS.json
├── curation_manifest.json
└── trackastra/
    ├── track_graph.pkl
    ├── napari_tracks.npy
    ├── napari_graph.json
    ├── tracks.csv
    └── summary.json
```

`curation_manifest.json` carries an immutable `inference_id`. Annotation sets
bind to that ID. If canonical inference is regenerated, a new inference ID is
created so stale annotations cannot silently bind to changed instance IDs.

## Pathological source-volume skipping

Immediately after production binary-mask construction, before source-instance
segmentation, dataset curation performs one cheap 6-connected-component pass.
A frame is outside the supported production regime only when all three default
conditions hold:

```text
largest component voxels                 >= 100000
largest component / foreground voxels    >= 0.50
largest component / complete frame       >= 0.05
```

When that condition is reached, expensive EDT/peak-pair segmentation is never
entered for that frame. The whole volume is rejected for production curation,
partial inference artifacts are removed, and the terminal record is:

```text
preprocessed/<split>/<volume>/_SKIPPED.json
```

Normal `infer --count ...` and `infer --all` selection excludes recorded skips.
To deliberately re-evaluate one after the pipeline has changed:

```powershell
python -m dataset_curation infer --split train --id <volume-id> --retry-skipped
```

`--force` alone does not override a skip record.

`status` reports `complete`, `partial`, `missing`, or `skipped`. Skipped volumes
are never selected by `annotate --next`, but `view-source` remains available.

## Unified viewer

Spatial mode:

- atomic supervoxel boundaries
- supervoxel IDs placed at an interior EDT center
- corrected cell-instance centers
- split seeds for up to four output instances
- `Save Split`
- `Hallucination`
- spatial undo

Tracking mode:

- notebook-09-equivalent Trackastra diagnostics
- `Broken Tracks` / red centers
- `New Tracks` / lime centers
- boundary-entry cyan centers
- boundary-exit orange centers
- corrected active track edges
- `Continue Track`
- `Break Track`
- `Complete Track`
- completed components move to the hidden `Hidden tracks` layer
- track undo

A spatial edit is authoritative. If a split or hallucination removes a previous
Trackastra detection ID, graph edges incident to that detection become inactive.
New split detections are selectable for manual `Continue Track`, but the tool
does not invent track associations automatically.

## Hallucinations

`Hallucination` applies to the last selected visible supervoxel. The supervoxel
is removed from the corrected instance raster and from the displayed atomic-SV
layer, while the immutable inference `supervoxels.npy` remains unchanged.

Active hallucinations are exported to:

```text
annotations/<split>/<volume>/<annotation-set>/instances/hallucinations.csv
```

Spatial operations are LIFO-undoable.

## Compact persistent cache

The only persistent full-volume spatial movies are:

```text
movies/supervoxels.npy      uint16
movies/final_instances.npy  uint16
```

Raw intensity remains in the canonical source Zarr. Preprocessed intensity,
binary masks, source-instance labels, the five-channel STIR-Net tensor, and
dense network outputs are not persisted as extra full-volume movies.

The unified viewer reads raw data lazily and reconstructs the Stage-6 binary
mask only for requested frames through a small RAM cache.

## Source-only viewer

`view-source` does not require inference and does not import PyTorch:

```powershell
python -m dataset_curation view-source --id 44b6_0113de3b
```

If the same ID exists in more than one split, pass `--split train` or
`--split test`.

## Annotation storage

One annotation set remains bound to one exact canonical inference ID.

```text
annotations/<split>/<volume>/<annotation-set>/
├── _session.json
├── manifest.json
├── instances/
│   ├── spatial_operations.json
│   ├── hallucinations.csv
│   └── manual_instances_tXXX.npy
└── tracks/
    ├── track_annotations.json
    ├── corrected_edges.csv
    ├── edge_overrides.csv
    └── completed_nodes.csv
```

The repository intentionally has no compatibility layer for the removed
run-directory hierarchy or the removed separate annotation viewers. Git history
is the recovery mechanism.
