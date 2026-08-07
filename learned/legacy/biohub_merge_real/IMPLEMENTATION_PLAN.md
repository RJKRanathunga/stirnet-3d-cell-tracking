# `merge_real` implementation plan

## Goal

Create a high-recall, human-verified dataset of real 3D merged-cell components for the learned instance-segmentation model, using only the already processed multi-sample outputs under `data/full_processed` plus the original raw Zarr path recorded per sample.

## Inputs

For every completed sample:

- Stage 6 preprocessing, binary masks, instance labels, and cell feature tables.
- Stage 11 final reconciled tracks.
- Stage 11 endpoint classifications, unresolved endings, continuation evidence, remaps, division events, and segmentation events.
- `source.json` for sample shape and optional raw-Zarr loading.

The miner does not mutate any pipeline output.

## Phase 1 — normalize sample state

Build one `ObservationIndex` per sample that:

- filters virtual merge observations when that provenance is present,
- indexes final canonical tracks by track/frame/cell,
- provides robust history-based volume references,
- provides physical-space one-step motion prediction,
- maps original Stage 11 segment IDs to canonical IDs through `track_id_remap`,
- exposes Stage 11 unresolved/remap support,
- identifies Stage 11 boundary endpoints,
- rejects confirmed/probable division contamination around the candidate frame.

## Phase 2 — independent candidate sources

### Source 1: multi-to-one

Predict every real track present at `t-1` into frame `t`. Group predictions by the Stage 6 component they reach. For groups with at least two identities, retain the pair whose historical volumes best explain the component volume.

### Source 2: two-one-two

Find an identity present at `t-1` and `t+1` but absent at `t`. Find another identity present on both sides whose prediction reaches the same Stage 6 component at `t`. Rank these as the strongest temporal candidates. Stage 11 remap evidence boosts priority.

### Source 3: disappearing track

Find final canonical interior endings. Predict the ending into `t+1`; if it lands in the component of another track that existed before the event and continues afterward, evaluate combined-volume consistency. Explicit Stage 11 unresolved-ending evidence boosts priority. Stage 11 boundary endpoints are excluded.

### Source 4: component anomaly

Independently inspect unusually large Stage 6 components. Require a frame-level or track-history volume anomaly and multiple separated physical-EDT peak regions. When possible, associate the component with a plausible pair of nearby previous identities; otherwise retain it as a pure anomaly candidate.

## Phase 3 — evidence and fusion

For track-pair candidates, calculate:

- `V_candidate / (V_A + V_B)`,
- absolute log volume-sum error,
- physical prediction-to-component distances,
- direct predicted-inside-component support,
- Stage 11 unresolved/remap support,
- optional 3D EDT peak count.

Use broad gates during mining. These are annotation-priority features, not pseudo-labels.

Fuse all source records by `(sample_id, frame, cell_id)` and retain source provenance. Initial priority tiers are A/B/C/D for Sources 2/1/3/4 respectively, with bonuses for independent-source agreement and volume/spatial consistency.

## Phase 4 — three manual passes

### Pass 1: candidate filtering

Review a fixed temporal crop around each candidate in Napari using raw/preprocessed data, production labels, involved tracks, and predicted positions. Store only the categorical result, expected cell count, confidence, and notes.

Only confirmed merges move forward.

### Pass 2: center annotation

Place one approximate 3D point per real cell. These points are segmentation seeds, not exact final CNN centers. Save global Z/Y/X coordinates so partitions can be regenerated deterministically.

### Pass 3: partition review

Generate an anisotropic physical-EDT marker-controlled watershed inside the merged component. Review the complete 3D result in Napari.

- Accept directly when correct.
- Edit the instance-label layer locally when needed.
- Optionally paint uncertain interface voxels.
- Reject/mark ambiguous when a reliable partition cannot be obtained.

A complete manually drawn 3D separation plane is not required.

## Phase 5 — permanent accepted cases

Materialize only accepted cases under `data/learned/instance_segmentation/merge_real/cases/` with:

- raw image when available,
- preprocessed volume,
- binary mask,
- context production labels,
- original candidate mask,
- final verified local instance labels,
- optional uncertainty mask,
- crop/sample/source metadata.

The final instance mask is the annotation ground truth. CNN foreground, vector, boundary, center-heatmap, and loss-validity targets should be derived later and should not be manually authored here.

## First evaluation checkpoint

Before substantial center/partition annotation, manually classify approximately 50–100 top candidates and measure:

- precision by source,
- precision by tier,
- source overlap,
- confirmed-case volume-ratio distribution,
- division/boundary contamination rate,
- dominant false-positive reasons.

Use those results to recalibrate the high-recall miner before expanding the real-case dataset.
