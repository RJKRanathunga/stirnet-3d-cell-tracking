# STIR-Net GPU Runtime Benchmarks

**Benchmark date:** 2026-08-15

This document records the GPU memory and training-throughput experiments used to select the current STIR-Net runtime profiles.

The goal of these experiments was not to change the STIR-Net architecture or checkpoint weights. The experiments only changed runtime execution settings such as activation checkpointing, attention chunk sizes, streamed-loss chunk sizes, and the number of local-mask requests decoded per training step.

---

## 1. Reference workload

All comparable GPU benchmarks used the same prepared BlastoSPIM first-overfit sample and the same reduced STIR-Net debugging architecture.

### Sample

| Property | Value |
|---|---:|
| ROI shape | `59 × 662 × 703` |
| Current cells | 36 |
| Ground-truth cells | 33 |
| Temporal tracklets | 52 |
| Split companions | 44 |
| Discovery queries | 8 |
| Required queries | 140 |
| Graph nodes | 178 |

The query decomposition is:

```text
36 primary
+ 44 split companions
+ 52 temporal queries
+ 8 discovery queries
= 140 required queries
```

### Model and training setup

| Property | Value |
|---|---:|
| Parameters | 653,582 |
| Trainable parameters | 653,582 |
| Curriculum stage | `joint` |
| AMP | FP16 |
| Decoder max spatial tokens | 2048 |
| Local-mask proposal matches | 33 |
| Architecture | `_reduced_config()` |
| Model/checkpoint compatibility | unchanged across runtime profiles |

The benchmark performs a real optimizer step:

```text
forward
→ loss computation
→ GradScaler backward
→ gradient unscale
→ gradient clipping
→ optimizer step
```

---

## 2. Runtime-profile principle

Runtime profiles change execution strategy only.

They may change:

- activation checkpointing
- spatial checkpointing
- co-reasoning checkpointing
- history checkpointing
- streamed-loss checkpointing
- temporal/spatial attention chunk sizes
- history node chunk size
- streamed loss chunk sizes
- local-mask training request cap

They must not change:

- model dimensions
- channel counts
- model geometry
- learned parameters
- output definitions
- loss weights
- checkpoint tensor shapes

This allows checkpoints to remain compatible between local and cloud execution profiles.

---

## 3. L4 experiments

### Hardware

```text
GPU: NVIDIA L4
CUDA-visible VRAM: 22.03 GiB
```

### 3.1 Spatial checkpoint ON, co-reasoning checkpoint ON

Configuration:

```text
checkpoint_spatial        = True
checkpoint_coreasoning    = True
checkpoint_history        = False
checkpoint_losses         = False

temporal_query_chunk_size = 64
spatial_query_chunk_size  = 32768
spatial_key_chunk_size    = 131072

history_node_chunk_size   = 512
native_chunk_voxels       = 1048576
dense_chunk_voxels        = 1048576

local_mask_train_cap      = 8
```

Result:

| Metric | Value |
|---|---:|
| Median step time | **14.468 s** |
| Mean step time | 14.453 s |
| Max peak allocated | 15.862 GiB |
| Max peak reserved | 21.059 GiB |
| Local masks/step | 8 |
| Mean local masks/s | 0.554 |

This was the fastest validated L4 configuration.

---

### 3.2 Same checkpointing with smaller chunks

Configuration:

```text
checkpoint_spatial        = True
checkpoint_coreasoning    = True

temporal_query_chunk_size = 16
spatial_query_chunk_size  = 16384
spatial_key_chunk_size    = 65536
```

Result:

| Metric | Value |
|---|---:|
| Median step time | 14.801 s |
| Mean step time | 14.775 s |
| Max peak allocated | 15.791 GiB |
| Max peak reserved | 20.566 GiB |
| Mean local masks/s | 0.541 |

Smaller chunks reduced peak memory only slightly and made training slower.

**Conclusion:** the larger `64 / 32768 / 131072` chunk configuration is preferable when co-reasoning checkpointing is enabled.

---

### 3.3 Co-reasoning checkpoint OFF

Two configurations were tested:

```text
large chunks:
64 / 32768 / 131072

smaller chunks:
16 / 16384 / 65536
```

Both OOMed.

The smaller-chunk run reached approximately:

```text
allocated: 21.70 GiB
free:      13 MiB
```

and failed while trying to allocate only another 16 MiB.

**Conclusion:** co-reasoning checkpointing is mandatory for this workload on an L4-class memory budget.

---

### 3.4 Spatial checkpoint OFF

Configuration:

```text
checkpoint_spatial       = False
checkpoint_coreasoning   = True
```

The run OOMed during the high-resolution spatial decoder.

At failure:

```text
PyTorch allocated: ~20.92 GiB
process usage:      ~21.88 GiB
free:               ~145 MiB
requested:          ~210 MiB
```

**Conclusion:** spatial checkpointing is also mandatory for this workload on the L4.

---

## 4. L40S experiments

### Hardware

```text
GPU: NVIDIA L40S
CUDA-visible VRAM: 44.39 GiB
```

---

### 4.1 Spatial checkpoint OFF, co-reasoning checkpoint OFF

Configuration:

```text
checkpoint_spatial       = False
checkpoint_coreasoning   = False
checkpoint_history       = False
checkpoint_losses        = False
```

with the large cloud chunks:

```text
temporal_query_chunk_size = 64
spatial_query_chunk_size  = 32768
spatial_key_chunk_size    = 131072
```

The run OOMed in co-reasoning.

At failure:

```text
PyTorch allocated: 43.77 GiB
process usage:      44.35 GiB
free:               33 MiB
requested:          26 MiB
```

**Conclusion:** even an L40S cannot retain both the full spatial and co-reasoning activation graphs for this workload.

---

### 4.2 Spatial checkpoint OFF, co-reasoning checkpoint ON

Configuration:

```text
checkpoint_spatial        = False
checkpoint_coreasoning    = True
checkpoint_history        = False
checkpoint_losses         = False

temporal_query_chunk_size = 64
spatial_query_chunk_size  = 32768
spatial_key_chunk_size    = 131072

history_node_chunk_size   = 512
native_chunk_voxels       = 1048576
dense_chunk_voxels        = 1048576

local_mask_train_cap      = 8
```

Measured steps:

| Step | Time | Peak allocated | Peak reserved | Local masks/s |
|---|---:|---:|---:|---:|
| 1 | 5.070 s | 32.97 GiB | 34.35 GiB | 1.578 |
| 2 | 5.051 s | 32.71 GiB | 34.35 GiB | 1.584 |
| 3 | 5.083 s | 33.24 GiB | 34.66 GiB | 1.574 |

Summary:

| Metric | Value |
|---|---:|
| Median step time | **5.070 s** |
| Mean step time | 5.068 s |
| Max peak allocated | 33.24 GiB |
| Max peak reserved | 34.66 GiB |
| Reserved headroom | 9.73 GiB |
| Local masks/step | 8 |
| Mean local masks/s | **1.579** |

This is the current preferred full-training configuration.

---

## 5. RTX 4050 local experiments

### Hardware

```text
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
CUDA-visible dedicated VRAM: 6.00 GiB
Windows WDDM environment
```

---

### 5.1 Local-mask cap = 2

Configuration:

```text
checkpoint_spatial       = True
checkpoint_coreasoning   = True
checkpoint_history       = True
checkpoint_losses        = True

temporal_query_chunk_size = 8
spatial_query_chunk_size  = 8192
spatial_key_chunk_size    = 65536

history_node_chunk_size   = 128
native_chunk_voxels       = 262144
dense_chunk_voxels        = 524288

local_mask_train_cap      = 2
```

The run OOMed during backward checkpoint recomputation in the spatial decoder.

---

### 5.2 Local-mask cap = 1

The same configuration was retried with:

```text
local_mask_train_cap = 1
```

The run completed, but only under severe GPU-memory overcommit/paging pressure.

Measured results:

| Metric | Value |
|---|---:|
| Median step time | **292.965 s** |
| Mean step time | 292.965 s |
| Max peak allocated | 10.91 GiB |
| Max peak reserved | 11.54 GiB |
| Local masks/step | 1 |
| Mean local masks/s | **0.0034** |

The reported CUDA-visible physical VRAM remained 6.00 GiB, so the much larger allocator values indicate severe Windows/WDDM memory overcommit rather than a genuinely resident 10–11 GiB GPU working set.

The workload therefore completes only with extremely poor performance.

---

## 6. Cross-hardware comparison

### Best measured practical configurations

| GPU | Spatial CKPT | CR CKPT | Local masks/step | Median step | Local masks/s |
|---|---:|---:|---:|---:|---:|
| RTX 4050 6 GB | ON | ON | 1 | 292.965 s | 0.0034 |
| NVIDIA L4 | ON | ON | 8 | 14.468 s | 0.554 |
| NVIDIA L40S | **OFF** | ON | 8 | **5.070 s** | **1.579** |

### L40S versus local RTX 4050

```text
optimizer-step speedup:
292.965 / 5.070 ≈ 57.8×

local-mask supervision throughput:
1.579 / 0.003415 ≈ 462×
```

The L40S therefore provides approximately:

- **57.8× faster optimizer steps**
- **462× higher local-mask supervision throughput**

for this specific full joint-training reference workload.

---

## 7. Final runtime profiles

### `local_6gb`

Intended use:

- smoke tests
- unit tests
- architecture/debugging checks
- reduced/cropped ROIs
- short forward/backward validation

It is **not recommended for full-volume production training**.

```python
RuntimeProfile.LOCAL_6GB: _RuntimeSettings(
    activation_checkpointing=True,
    checkpoint_spatial=True,
    checkpoint_coreasoning=True,
    checkpoint_history=True,
    checkpoint_losses=True,
    temporal_query_chunk_size=8,
    spatial_query_chunk_size=8_192,
    spatial_key_chunk_size=65_536,
    history_node_chunk_size=128,
    native_chunk_voxels=262_144,
    dense_chunk_voxels=524_288,
    local_mask_train_cap=1,
)
```

The full `59 × 662 × 703` joint workload exceeded the practical dedicated-VRAM budget even with aggressive checkpointing.

---

### `cloud_48gb`

Intended use:

- full-volume training
- first-overfit experiments
- architecture comparisons
- long optimization runs
- evaluation on the full reference workload

```python
RuntimeProfile.CLOUD_48GB: _RuntimeSettings(
    activation_checkpointing=True,
    checkpoint_spatial=False,
    checkpoint_coreasoning=True,
    checkpoint_history=False,
    checkpoint_losses=False,
    temporal_query_chunk_size=64,
    spatial_query_chunk_size=32_768,
    spatial_key_chunk_size=131_072,
    history_node_chunk_size=512,
    native_chunk_voxels=1_048_576,
    dense_chunk_voxels=1_048_576,
    local_mask_train_cap=8,
)
```

Rationale:

- the 48 GB budget is large enough to retain spatial activations
- retaining spatial activations removes expensive spatial recomputation
- co-reasoning still requires checkpointing because retaining both graphs exceeded the L40S memory budget
- history and streamed losses do not need checkpoint recomputation on this profile

---

## 8. Key conclusions

1. **Co-reasoning checkpointing is mandatory** on the tested large workload, including on the L40S when spatial activations are also retained.
2. **Spatial checkpointing is mandatory on L4-class memory budgets.**
3. The L40S can safely disable spatial checkpointing while retaining co-reasoning checkpointing.
4. Disabling both spatial and co-reasoning checkpointing exceeded the tested 44.39 GiB L40S memory budget.
5. Smaller attention chunks did not materially reduce the validated L4 peak and made the step slower.
6. Full-volume joint training on the RTX 4050 is technically possible only under severe memory overcommit with `local_mask_train_cap=1`, and is not practically useful.
7. The RTX 4050 should be used for reduced local development workloads rather than full-volume training.
8. The L40S is the preferred production-training target for the current STIR-Net workload.

---

## 9. Benchmark scripts

The benchmark workflow is implemented in:

```text
modal/02_benchmark_stirnet.py
modal/03_benchmark_stirnet_local.py
```

The cloud benchmark is used for Modal GPU experiments.

The local benchmark runs the same real BlastoSPIM joint-training workload with the `local_6gb` profile and reports:

- optimizer-step time
- CUDA peak allocated memory
- CUDA peak reserved memory
- local masks per step
- local masks per second
- comparison against the measured L40S reference

---

## 10. Reproducibility note

These numbers are workload-specific and should not be treated as universal GPU performance measurements.

When comparing future runs, keep the following fixed:

```text
ROI shape             59 × 662 × 703
current cells         36
GT cells              33
temporal tracklets    52
split companions      44
discovery queries     8
required queries      140
parameters            653,582
curriculum stage      joint
AMP                   FP16
```

If the model architecture, ROI, query count, local-mask cap, loss configuration, PyTorch version, or GPU runtime changes, record a new benchmark rather than directly comparing the new numbers with this table.
