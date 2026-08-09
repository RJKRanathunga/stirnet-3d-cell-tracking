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
N_DISCOVERY_QUERIES = 8
MAX_QUERIES = 128

# query decoder
QUERY_LAYERS = 3
QUERY_HEADS = 4
QUERY_FFN_DIM = 512
```

## 2. Physical/cell-scale defaults

```python
PATCH_CONTEXT_DIAMETERS = 8.0
PATCH_VALID_DIAMETERS = 6.0

TEMPORAL_BASE_RADIUS_DREF = 1.5
TEMPORAL_MAX_RADIUS_DREF = 2.5
SPATIAL_NEIGHBOR_RADIUS_DREF = 2.5
TEMPORAL_GAUSSIAN_SIGMA_DREF = 0.75
```

These values are expressed relative to $d_\text{ref}$, not voxel counts.

## 3. Query decoder mask defaults

```python
MASK_ATTENTION_THRESHOLD = 0.20
INITIAL_PRIOR_LOGIT_INSIDE = +1.5
INITIAL_PRIOR_LOGIT_OUTSIDE = -1.5
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
LOSS_OVERLAP = 0.10

LOSS_FOREGROUND = 0.50
LOSS_CENTER_HEATMAP = 1.00
LOSS_BOUNDARY = 0.50

AUX_LAYER_WEIGHT = 0.50
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

## 8. Training data mixture

```python
CLEAN_SAMPLE_PROB = 0.40
CORRUPTED_SAMPLE_PROB = 0.60
```

## 9. Optimization defaults

```python
LR = 2e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
```

Use AdamW, warmup, cosine decay, and mixed precision.

## 10. Inference defaults

```python
RENDER_EXIST_THRESHOLD = 0.30
FINAL_EXIST_THRESHOLD = 0.50
```

Mask threshold and conflict score threshold must be selected from validation data and should not be hardcoded before calibration.

## 11. Config dataclass structure

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
    inference: InferenceConfig
```

Configuration should be serializable to YAML/JSON and stored with every checkpoint.
