# Real merged-cell dataset (`merge_real`)

This package mines **real 3D under-segmented cell components** from the multi-sample pipeline output in `data/full_processed`, then turns them into trustworthy learned instance-segmentation examples through a three-pass Napari review workflow.

It is intentionally isolated from `src/03_segmentation`. The classical pipeline supplies evidence and images; this package creates a reviewed dataset for the learned Stage 3 replacement/augmentation.

## Input contract

The implementation reads the per-sample layout produced by the full-dataset runner:

```text
data/full_processed/
├── pipeline_manifest.csv
└── train/
    └── <sample_id>/
        ├── source.json
        ├── stage_6_processed_dataset/
        │   ├── preprocessing/tXXX.npy
        │   ├── masking/tXXX.npy
        │   ├── segmentation/tXXX.npy
        │   └── cells/tXXX.csv
        └── stage_11_track_reconciliation/
            ├── tracks.csv
            ├── endpoint_classifications.csv
            ├── continuation_candidates.csv
            ├── continuation_decisions.csv
            ├── track_id_remap.csv
            ├── unresolved_endings.csv
            ├── division_events.csv
            └── segmentation_events.csv
```

`source.json` is used to locate the original Zarr volume for optional raw-image display/extraction.

## Important Stage 3 configuration note

At repository commit `6702b31`, the full-dataset runner calls `process_dataset(...)` without a custom segmentation configuration, while `SegmentationConfig.enable_geometric_completion` defaults to `True`.

Therefore, verify how `data/full_processed` was generated if the intended mining baseline is **effective peaks only**. The miner does not alter or rerun Stage 3; it mines exactly what is present in `data/full_processed`.

## Candidate sources

The names are local to this package and intentionally numbered 1–4.

1. **`source1_multi_to_one`** — two previous track predictions map to one current Stage 6 component.
2. **`source2_two_one_two`** — a track is present at `t-1` and `t+1` but missing at `t`, while another identity occupies the shared component. This is the strongest temporal source.
3. **`source3_disappearing_track`** — an interior track ending projects into a component occupied by another continuing track. This is expected to be the highest-volume source.
4. **`source4_component_anomaly`** — tracking-independent large/abnormal components with multiple 3D EDT peak regions.

All sources use broad combined-volume evidence:

```text
V_candidate ≈ V_track_A + V_track_B
```

The actual stored feature is the continuous ratio and its log error. Thresholds are deliberately permissive because candidates are **not labels**.

Confirmed/probable division transitions and boundary components are excluded from positive pair mining to reduce parent/daughter contamination.

## Candidate fusion and ranking

All detections are fused by:

```text
(sample_id, frame, cell_id)
```

A component can therefore receive support from several sources without appearing several times in the review queue.

Initial tiers:

```text
A  source 2 present
B  source 1 present
C  source 3 present
D  source 4 only
```

Within tiers, volume consistency, spatial agreement, and independent-source support determine review priority. This is an annotation priority, **not a learned merge probability**.

## Three-pass annotation

### Pass 1 — filter cases

Classify each temporal 3D scene as:

- `confirmed_2_cell_merge`
- `confirmed_3plus_cell_merge`
- `not_merge`
- `other_segmentation_error`
- `division_or_birth`
- `ambiguous`
- `skip`

Only confirmed merges proceed.

### Pass 2 — place centers

Place one approximate Napari 3D point inside each real cell. These are **seed markers**, not final mathematically exact CNN centers. Exact training centers can later be derived from the accepted instance masks.

### Pass 3 — review the 3D partition

The package generates a physical-EDT marker-controlled 3D watershed from the human centers. The reviewer can:

- accept it directly,
- edit the `Proposed Instances` Labels layer and accept the correction,
- mark uncertain voxels in `Uncertain Boundary`,
- reject/mark ambiguous.

A complete separation plane is **not manually drawn by default**. Manual voxel painting is only a fallback for the local region where the automatic partition is wrong.

## Output layout

```text
data/learned/instance_segmentation/merge_real/
├── mining/
│   ├── source_candidates.csv
│   ├── candidates.csv
│   ├── summary.json
│   └── run_metadata.json
├── reviews/
│   ├── filter_reviews.csv
│   ├── centers/<candidate_id>.csv
│   ├── partitions/<candidate_id>/
│   │   ├── instance_labels.npy
│   │   ├── uncertain_mask.npy
│   │   └── review.json
│   └── partition_reviews.csv
└── cases/
    └── <candidate_id>/
        ├── raw.npy                    # when raw Zarr loading is available
        ├── preprocessed.npy
        ├── binary_mask.npy
        ├── production_labels.npy
        ├── candidate_mask.npy
        ├── instance_labels.npy        # human-verified split, labels 1..N
        ├── uncertain_mask.npy         # optional
        └── metadata.json
```

CNN vector fields, center heatmaps, foreground targets, and boundary targets should be generated later from `instance_labels.npy`; they are not manually annotated here.

## Usage

Run commands from the repository root.

### 1. Mine every completed sample

```powershell
python learned/instance_segmentation/dataset/merge_real/run_mining.py
```

One or several samples only:

```powershell
python learned/instance_segmentation/dataset/merge_real/run_mining.py `
    --sample 44b6_0b24845f `
    --sample 44b6_0113de3b
```

Override paths if needed:

```powershell
python learned/instance_segmentation/dataset/merge_real/run_mining.py `
    --full-processed-root D:/somewhere/full_processed `
    --output-root D:/somewhere/merge_real
```

### 2. Pass 1

```powershell
python learned/instance_segmentation/dataset/merge_real/run_filter.py
```

### 3. Pass 2

```powershell
python learned/instance_segmentation/dataset/merge_real/run_center_annotation.py
```

### 4. Pass 3

```powershell
python learned/instance_segmentation/dataset/merge_real/run_partition_review.py
```

### 5. Run tests

From the repository root:

```powershell
pytest learned/instance_segmentation/dataset/merge_real/tests -q
```

## Implementation sequence

The code is organized so the research loop can be evaluated early:

```text
full_processed Stage 6 + Stage 11
        ↓
source 1–4 candidate mining
        ↓
component-level fusion/ranking
        ↓
Pass 1: merge classification
        ↓
Pass 2: approximate 3D centers
        ↓
automatic physical-EDT watershed
        ↓
Pass 3: accept/correct/uncertainty
        ↓
materialized real-case dataset
```

The first useful metric is not CNN performance yet. It is **manual precision by candidate source/tier**. After the first 50–100 reviewed candidates, use the review outcomes to recalibrate the mining thresholds before spending substantial time annotating centers and partitions.
