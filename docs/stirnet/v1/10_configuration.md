# 10 — V1 Configuration

## 1. Frozen architecture defaults

```python
# representation
D_MODEL = 128
MASK_DIM = 32
DROPOUT = 0.10

# spatial backbone
SPATIAL_CHANNELS = (16, 32, 64, 128)
SPATIAL_BLOCKS_PER_LEVEL = 2
ANISOTROPY_THRESHOLD = 1.5

# temporal graph
TEMPORAL_RADIUS = 2
GRAPH_LAYERS = 2
GRAPH_HEADS = 4
GRAPH_FFN_DIM = 256
K_SPATIAL_NEIGHBORS = 6

# co-reasoning
COREASONING_BLOCKS = 2
COREASONING_HEADS = 4

# queries
SPLIT_COMPANIONS_PER_INSTANCE = 1
MAX_SPLIT_COMPANIONS_PER_INSTANCE = 8
SPLIT_VOLUME_RATIO_PER_HYPOTHESIS = 1.0
N_DISCOVERY_QUERIES = 8
MAX_QUERIES = None

# query decoder
QUERY_LAYERS = 3
QUERY_HEADS = 4
QUERY_FFN_DIM = 512

# bounded-memory execution defaults
COREASONING_TEMPORAL_QUERY_CHUNK = 8
COREASONING_SPATIAL_QUERY_CHUNK = 8192
COREASONING_SPATIAL_KEY_CHUNK = 65536

# training activation-memory controls
ACTIVATION_CHECKPOINTING = True
CHECKPOINT_SPATIAL = True
CHECKPOINT_COREASONING = True
CHECKPOINT_LOSSES = True
```

Activation checkpointing recomputes selected forward regions during backward
instead of retaining their intermediate activations. The three component flags
allow the spatial encoder/decoder, co-reasoning blocks, and streamed loss chunks
to be controlled independently under the master flag. Checkpointing is applied
only while the corresponding module is in training mode, gradients are enabled,
and at least one tensor input requires a gradient. Evaluation and `torch.no_grad()`
therefore use the direct forward path and preserve inference behavior.

## 2. Physical/cell-scale defaults

```python
PATCH_CONTEXT_DIAMETERS = 8.0
PATCH_VALID_DIAMETERS = 6.0

TEMPORAL_BASE_RADIUS_DREF = 1.5
TEMPORAL_MAX_RADIUS_DREF = 2.5
SPATIAL_NEIGHBOR_RADIUS_DREF = 2.5
TEMPORAL_GAUSSIAN_SIGMA_DREF = 0.75
NATIVE_SUPPORT_RADIUS_DREF = 1.5
NATIVE_SOURCE_DILATION_DREF = 0.5
TEMPORAL_MATCH_RADIUS_DREF = 1.0
DISCOVERY_MATCH_RADIUS_DREF = 1.5
```

These values are expressed relative to $d_\text{ref}$, not voxel counts.

The split-companion baseline is a minimum. Observable within-volume source
volume ratios can raise it up to the configured maximum; GT annotations are not
an input to query construction. Temporal match distance uses the initial clue
reference, while discovery match distance uses the final decoded center.

## 3. Query decoder mask defaults

```python
MASK_ATTENTION_THRESHOLD = 0.20
INITIAL_PRIOR_LOGIT_INSIDE = +1.5
INITIAL_PRIOR_LOGIT_OUTSIDE = -1.5
NATIVE_BACKGROUND_LOGIT = -20.0

PRIMARY_CENTER_STEP_DREF = 0.50
SPLIT_CENTER_STEP_DREF = 0.75
TEMPORAL_CENTER_STEP_DREF = 0.25
DISCOVERY_CENTER_STEP_DREF = 1.00
```

Physical dilation widths should be specified in multiples of $d_\text{ref}$ and converted to each feature grid using current effective spacing.

## 4. Loss defaults

```python
LOSS_EXIST = 2.0

LOSS_DICE_HI = 5.0
LOSS_FOCAL_HI = 2.0

LOSS_DICE_COARSE = 1.0
LOSS_FOCAL_COARSE = 0.5

LOSS_CENTER = 2.0
LOSS_COUNT = 0.25
LOSS_OVERLAP = 0.00

LOSS_FOREGROUND = 0.50
LOSS_CENTER_HEATMAP = 1.00
LOSS_BOUNDARY = 0.50

AUX_LAYER_WEIGHT = 0.50

MASK_SUPERVISION_RADIUS_DREF = 1.5
MASK_FOCAL_ALPHA_POS = 0.75
MASK_FOCAL_GAMMA = 2.0
```

These are tuning defaults, not permanent architecture constraints.

## 5. Existence focal defaults

```python
EXIST_FOCAL_GAMMA = 2.0
EXIST_FOCAL_ALPHA_POS = 0.75
EXIST_FOCAL_ALPHA_NEG = 0.25
```

## 6. Boundary target defaults

```python
BOUNDARY_WIDTH_UM = 1.0
BOUNDARY_POS_WEIGHT = 4.0
```

## 7. Temporal corruption defaults

```python
TEMP_HYPOTHESIS_DROPOUT = 0.10
TEMP_EDGE_DROPOUT = 0.10

TEMP_POSITION_JITTER_DREF = 0.10
TEMP_LARGE_JITTER_DREF = 0.50
TEMP_FALSE_CLUE_PROB = 0.05
```

## 8. Optional curriculum defaults

```python
CURRICULUM_ENABLED = False
SPATIAL_DENSE_STEPS = 0
TEMPORAL_DENSE_STEPS = 0
QUERY_BOOTSTRAP_STEPS = 0
NATIVE_BOOTSTRAP_STEPS = 0
JOINT_SPATIAL_LR_SCALE = 0.10
JOINT_DENSE_LR_SCALE = 0.50
```

Durations remain zero by default because production lengths must be chosen by
an explicit experiment. Disabled mode preserves all-at-once training.

## 9. Training data mixture

```python
CLEAN_SAMPLE_PROB = 0.40
CORRUPTED_SAMPLE_PROB = 0.60
```

## 10. Optimization defaults

```python
LR = 2e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
```

Use AdamW, warmup, cosine decay, and mixed precision.

## 11. Inference defaults

```python
RENDER_EXIST_THRESHOLD = 0.30
FINAL_EXIST_THRESHOLD = 0.50
```

Mask threshold and conflict score threshold must be selected from validation data and should not be hardcoded before calibration.

## 12. Config dataclass structure

Recommended:

```python
@dataclass
class StirNetConfig:
    spatial: SpatialConfig
    temporal: TemporalConfig
    coreasoning: CoReasoningConfig
    queries: QueryConfig
    decoder: DecoderConfig
    losses: LossConfig
    training: TrainingConfig
    curriculum: CurriculumConfig
    inference: InferenceConfig
```

Configuration should be serializable to YAML/JSON and stored with every checkpoint.
