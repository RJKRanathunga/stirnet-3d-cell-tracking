# STIR-Net V1 debugging toolkit

This package is a structured observability layer for the current STIR-Net V1
implementation. It is intentionally separate from the model code: the debugger
uses PyTorch hooks, existing structured outputs, the existing Hungarian matcher,
and replay of the current native-mask formula. It does not change the model's
numerical forward path.

## Why this exists

The first full-scene overfit proved that STIR-Net can optimize, but the final
Napari result exposed several different possible failure locations:

- query survival / existence ranking;
- center drift through the query decoder;
- temporal-reference quality;
- co-reasoning corruption or collapse;
- CNN boundary / foreground representation;
- learned native-mask geometry;
- hand-designed mask priors dominating learned logits;
- unbalanced gradient flow.

The debugger is designed to identify **where the information first becomes
wrong**, instead of inspecting millions of scalar weights.

## Folder layout

```text
debugging/
├── __init__.py
├── README.md
├── first_overfit_acceptance.py          # compatibility wrapper
├── first_overfit_backward_gate.py       # compatibility wrapper
│
├── core/
│   ├── __init__.py
│   ├── config.py                        # light/deep debug presets
│   ├── hooks.py                         # source-neutral PyTorch instrumentation
│   ├── inspector.py                     # main orchestration API
│   ├── stats.py                         # bounded tensor summaries
│   └── trace.py                         # structured DebugTrace
│
├── probes/
│   ├── __init__.py
│   ├── matching.py                      # final Hungarian identity anchor
│   ├── queries.py                       # initial -> decoder1 -> decoder2 -> decoder3
│   ├── masks.py                         # learned/prior/combined native masks
│   ├── temporal.py                      # builder -> CR1 -> CR2 temporal states
│   ├── spatial.py                       # CNN dense outputs / scene arrays
│   └── gradients.py                     # per-major-module gradient summaries
│
├── visualization/
│   ├── __init__.py
│   └── napari_viewer.py                 # 3D frontend
│
├── io/
│   ├── __init__.py
│   └── serialization.py                 # JSON + CSV + NPY trace persistence
│
├── cli/
│   ├── __init__.py
│   └── inspect_checkpoint.py
│
├── acceptance/
│   ├── __init__.py
│   ├── first_overfit.py                 # existing all-cell sample builder/gate
│   └── backward_gate.py                 # existing one-step backward gate
│
└── tests/
    ├── __init__.py
    └── test_debugging_core.py
```

The two old root acceptance module names are retained because Notebook 05/06
already import them.

## Main API

```python
from learned.stirnet.debugging import DebugConfig, StirNetInspector

inspector = StirNetInspector(
    model,
    DebugConfig.deep(
        selected_query_indices=(72, 84),
        max_selected_queries=6,
    ),
)

trace = inspector.inspect(batch)
```

The ordinary STIR-Net batch may be on CPU. The inspector moves model inputs to
its configured device while leaving the large target label maps on CPU.

## What one trace contains

### Query table

One row per query, including:

```text
query index
query type
source instance id
temporal index / salience / reliability
initial reference position
layer-1 existence / center / coarse Dice
layer-2 existence / center / coarse Dice
layer-3 existence / center / coarse Dice
final Hungarian GT match
center error in micrometres
whether the query survives the inference existence threshold
```

The final Hungarian match is used as a stable identity anchor, then the same
query/GT pair is traced backwards through the three decoder layers.

### Native-mask decomposition

For selected queries the debugger computes three full-scene metrics:

```text
learned-only mask      = native_mask_embedding · mask_features
prior-only mask        = seeded-instance prior or temporal Gaussian prior
combined mask          = learned logits + prior logits
```

Metrics are streamed in bounded voxel chunks, so a full `[Q,Z,Y,X]` tensor is
never materialized. Only a compact crop around the selected query / matched GT
is retained for Napari.

Each mask row includes soft Dice, hard Dice, predicted volume, GT volume,
volume ratio, mean probability inside GT, and mean probability outside GT.

### Temporal trace

For every temporal hypothesis:

```text
physical reference
salience
reliability
initial temporal-token norm
token norm after CR1
token norm after CR2
cosine(initial, CR1)
cosine(CR1, CR2)
status vector
```

This tells us whether temporal representations change sharply in a
co-reasoning block even though their physical reference is unchanged.

### CNN / spatial trace

Large encoder/decoder feature tensors are summarized instead of copied:

```text
shape
dtype
finite fraction
zero fraction
mean/std/min/max
RMS / absolute maximum
optional per-channel mean/std
```

Deep mode also stores the explicit dense outputs that are biologically
interpretable:

```text
foreground probability
center heatmap probability
boundary probability
```

### Gradients

After a normal `backward()`:

```python
from learned.stirnet.debugging.probes import summarize_gradients
rows = summarize_gradients(model)
```

The rows aggregate parameter norm, gradient norm, nonzero gradient count,
finite status, maximum absolute gradient, and gradient/parameter ratio for:

```text
spatial encoder
graph encoder
tracklet pooler
temporal state builder
CR1
spatial decoder
CR2
query builder
query decoder
native mask head
dense heads
```

A dedicated total-loss backward probe is also provided, but it does not run by
default because the validated full-scene backward is expensive.

## Light vs deep

### `DebugConfig.light()`

Use for nearly every checkpoint/run:

- one evaluation forward;
- module statistics;
- final Hungarian matching;
- query trajectory;
- temporal trajectory;
- no full native-mask rendering;
- no full scene copies.

### `DebugConfig.deep()`

Use after a failure is visible:

- everything from light mode;
- raw/current/GT arrays;
- CNN dense probability fields;
- automatic selection of representative bad queries;
- streamed learned/prior/combined native-mask diagnostics;
- query-centric 3D crops for Napari.

## Query selection in deep mode

Explicit query indices are always taken first. Remaining slots are filled with
representative cases:

1. worst matched center errors;
2. worst matched coarse Dice;
3. a high-existence surviving temporal query;
4. a high-existence split query.

This keeps deep inspection bounded and makes repeated runs comparable.

## Napari

```python
from learned.stirnet.debugging.visualization import open_debug_viewer

viewer = open_debug_viewer(trace, query_index=72)
```

The viewer can show:

```text
raw
current instances
GT instances
CNN foreground probability
CNN center heatmap
CNN boundary probability
initial query references by query type
final query centers by query type
selected-query learned mask probability
selected-query prior probability
selected-query combined probability
matched GT crop
```

Napari is optional and is deliberately not added to
`learned/stirnet/requirements.txt` by this package.

## Save / reload

```python
from learned.stirnet.debugging.io import save_debug_trace, load_debug_trace

save_debug_trace(trace, "runs/stirnet/debug/step_025")
trace = load_debug_trace("runs/stirnet/debug/step_025")
```

The saved structure is:

```text
step_025/
├── trace.json
├── tables/
│   ├── modules.csv
│   ├── matching.csv
│   ├── queries.csv
│   ├── temporal.csv
│   └── masks.csv
└── arrays/
    ├── scene/
    ├── dense/
    ├── queries/
    └── masks/qXXX/
```

Generated debug outputs/checkpoints are run artifacts and should not be
committed to Git.

## Current limitation: attention maps

The current V1 query decoder calls self-attention with `need_weights=False` and
the local co-reasoning attention is implemented through chunked helper methods.
The V1 debugger therefore does **not** monkey-patch those modules just to obtain
attention matrices.

The first debugging version uses query trajectories, support-conditioned mask
outputs, temporal token changes and module statistics. If those diagnostics
identify attention routing as the unresolved failure, a second phase can add a
selected-query attention replay/probe. That keeps instrumentation from changing
model behavior during the current debugging cycle.

## Comparing iterations

The trace schema is designed for repeated architecture experiments:

```python
from learned.stirnet.debugging import (
    load_debug_trace,
    compare_traces,
)

before = load_debug_trace("runs/stirnet/debug/before")
after = load_debug_trace("runs/stirnet/debug/after")
rows = compare_traces(before, after)
```

This compares query survival by type, matched center error, and selected-query
learned/prior/combined mask metrics so an architectural change can be judged
against the same diagnostic contract.

## Attention scope

V0.1 deliberately does not monkey-patch attention methods.  The current V1
query self-attention uses `need_weights=False`, query cross-attention returns
only its message, and co-reasoning invokes chunked attention helper methods
directly.  The first debugger release therefore identifies *which query/stage*
is failing using references, layer outputs, masks and token changes.  A later
selected-query attention replay can be added without changing model numerics.
