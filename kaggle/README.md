# Kaggle spatial STIR-Net submission package

This folder is the frozen deployment path for the first real Biohub leaderboard measurement of the current spatial branch.

Generated against repository commit:

```text
ac981bf4fa79f9a5f8db2aa24f11adcd87754482
refactor(rag): remove experimental separator fusion
```

## Scientific scope of this submission

The scored path is intentionally:

```text
Biohub test OME-Zarr
  -> current canonical preprocessing / binary mask / source watershed
  -> current tiled STIR-Net spatial model
  -> current production signed multicut partition
  -> source-instance anchored split-only postfilter (Investigation 25)
  -> recompute cell detections/features from corrected labels
  -> Stage 7 apply-mode windowed 4D tracking
  -> Stage 8 stitching / merge repair
  -> Stage 10 lineage
  -> Stage 11 submission-policy reconciliation
  -> submission.csv
```

Learned temporal STIR-Net reasoning is **disabled**. This makes the resulting leaderboard score a clean spatial-only baseline.

## Files

- `run_submission.py` — hidden-test-safe end-to-end inference and CSV generation.
- `validate_submission.py` — strict schema, graph, dataset-coverage, temporal-direction and Zarr-bound validator.
- `build_inference_bundle.py` — creates the private offline Kaggle input bundle from the committed repository plus the spatial checkpoint.
- `biohub_stirnet_submission.ipynb` — thin Kaggle notebook used for the committed competition run.

## 1. Place this folder at repository root

After extracting the ZIP you should have:

```text
cell-tracking/
├── kaggle/
│   ├── README.md
│   ├── run_submission.py
│   ├── validate_submission.py
│   ├── build_inference_bundle.py
│   └── biohub_stirnet_submission.ipynb
├── learned/
├── src/
├── investigations/
└── ...
```

For this first baseline, confirm the repository has not changed:

```powershell
git rev-parse HEAD
```

Expected:

```text
ac981bf4fa79f9a5f8db2aa24f11adcd87754482
```

If HEAD changes later, review/regenerate these Kaggle files against that revision rather than silently submitting mixed code.

## 2. Build the private inference bundle

From repository root:

```powershell
python .\kaggle\build_inference_bundle.py
```

The builder first tries the current Investigation-17 recovery checkpoint:

```text
runs/stirnet/investigations/17_morphology_rag_multicrop_training/
recovery/drosophila_12_morphology_rag_multicrop_v1/best_checkpoint.pt
```

If your final spatial checkpoint is elsewhere, specify it explicitly:

```powershell
python .\kaggle\build_inference_bundle.py `
    --checkpoint .\path\to\your\checkpoint.pt
```

The builder:

1. verifies the Git revision;
2. refuses dirty inference-critical source files by default;
3. uses `git archive HEAD` for code, so untracked/local files cannot enter accidentally;
4. copies only `learned/`, `src/`, the two required inference helper scripts, and package metadata;
5. converts the training checkpoint to an inference-only checkpoint;
6. excludes `.git`, `data/`, `runs/`, `.env`, `kaggle.json`, and backup folders;
7. SHA256-hashes every file;
8. writes both an unpacked directory and a ZIP.

Outputs:

```text
kaggle/dist/stirnet_kaggle_bundle/
kaggle/dist/stirnet_kaggle_bundle.zip
```

### Optional offline wheels

Do **not** add packages unless the Kaggle preflight actually shows that one is missing/incompatible. If an offline wheel set becomes necessary:

```powershell
python .\kaggle\build_inference_bundle.py `
    --wheel-dir .\kaggle\wheels
```

The notebook installs bundled wheels with `--no-index --no-deps` before importing the pipeline.

## 3. Create a private Kaggle Dataset

Upload `stirnet_kaggle_bundle.zip` as a **private** Kaggle Dataset and attach it to the competition notebook.

The notebook can consume either:

- an attached dataset containing the unpacked `stirnet_kaggle_bundle/` directory, or
- an attached dataset containing `stirnet_kaggle_bundle.zip`.

If the ZIP is attached, the notebook verifies the file and extracts it under `/kaggle/working/` before execution.

Do not make the checkpoint bundle public unless you explicitly intend to release the model.

## 4. Create the Kaggle competition notebook

Import `biohub_stirnet_submission.ipynb` into Kaggle, then:

1. attach the **Biohub - Cell Tracking During Development** competition data;
2. attach the private STIR-Net inference bundle dataset;
3. select a GPU accelerator;
4. set **Internet = Off**;
5. run all cells once interactively on the visible test placeholders;
6. inspect the final validator result;
7. use **Save Version / Save & Run All** for the scored run;
8. submit the generated `/kaggle/working/submission.csv`.

The notebook deliberately refuses to run without CUDA so an accidental CPU commit does not consume a submission attempt/runtime window.

## 5. What the runtime does with hidden test data

`run_submission.py` discovers every `.zarr` below the mounted `test/` directory at runtime. It does not contain any fixed sample IDs, fixed frame count, or dependency on the local `44b6_0113de3b` cache.

For each dataset it processes one frame at a time, persists only the corrected segmentation and cell table needed by the existing tracking stack, releases dense STIR-Net GPU state, and moves to the next frame. After the sample is spatially processed it runs Stages 7, 8, 10 and 11, converts the reconciled tracks/lineage into Kaggle nodes and edges, then releases sample-level state.

A failure in any hidden-test sample aborts the notebook. It does **not** silently submit a partial dataset set.

## Submission validation

The final validator requires the exact Kaggle columns:

```text
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
```

It also checks:

- `id` is exactly `0..N-1`;
- every hidden test dataset is represented and no extra dataset appears;
- node IDs are unique within a dataset;
- node/edge sentinel fields use `-1` correctly;
- all edge endpoints exist;
- all edges move forward in time;
- indegree is at most 1 and outdegree is at most 2;
- duplicate/self edges are absent;
- node coordinates are integer voxel coordinates inside the actual test Zarr bounds.

You can also validate a local CSV manually:

```powershell
python .\kaggle\validate_submission.py .\submission.csv `
    --test-root .\path\to\test
```

## Important baseline rule

Do not enable temporal STIR-Net or change spatial thresholds in this notebook before obtaining the first score. The purpose of this deployment is to measure the current spatial system plus the already-established tracking pipeline. Temporal reasoning can then be added as a controlled second submission.
