# 13 — Hierarchical Temporal Memory

## Data flow

```text
                    FIVE-FRAME WINDOW
                           |
                           v
                detection observations
                           |
            scalar features + local 3-D grid
                           |
                           v
                 candidate Detection GNN
                           |
                           v
                  NODE MEMORY [N,D]
                           |
              +------------+------------+
              |                         |
              v                         |
        TrackletPooler                   |
              |                         |
              v                         |
     TRACKLET MEMORY [M,D]              |
              |                         |
              +------------+------------+
                           |
            +--------------+--------------+
            |                             |
            v                             v
       CR1 / CR2                 component/query memory
   tracklet-level dense             cross-attention
       reasoning                          |
            |                             |
            +--------------+--------------+
                           v
                   query decoder
                           |
                           v
                    instance masks
```

Only the target frame enters the dense spatial CNN. Historical morphology
remains sparse: each detection has one `4 x 12 x 12 x 12` local physical grid.
No five-frame dense CNN or global isotropic resampling is introduced.

## Candidate graph versus accepted graph

The candidate detection graph contains all directed non-self node pairs within
one logical sample. It grants permission to reason; it does not assert a
biological match. Accepted Trackastra temporal/division records are used
separately to build tracklets and are then exposed as edge measurements. The
15-D edge schema is:

```text
0       signed dt
1:4     relative z,y,x / dref
4       distance / dref
5       log volume ratio
6       intensity difference
7       motion-prediction residual / dref
8       Trackastra score or zero
9       score-valid flag
10:14   temporal-forward, temporal-reverse, division, same-frame flags
14      accepted-Trackastra relation flag
```

The old 14 fields are unchanged. Generic candidates have columns 8, 9, and 14
zero unless they coincide with accepted evidence. Candidate construction is
exact and CPU-chunked. `max_candidate_edges` may raise a clear safety error but
never truncates or ranks evidence.

## Memory levels

`TemporalNodeMemory.tokens [N,D]` is the output of history/scalar fusion and all
detection-GNN layers. Its explicit physical metadata is observed zyx,
tracklet-projected target zyx, signed time, tracklet ID, batch index, history
validity, and optional node ID. Every input detection produces one row.

`TemporalState.tokens [M,D]` is the existing learned tracklet pool followed by
status projection and CR1/CR2 updates. Pooling is no longer destructive because
the node memory stays attached to the state. CR1/CR2 remain tracklet-level to
avoid dense voxel-to-node cost.

## Shared memory attention

Component and query fusion instantiate the same generic multi-head attention.
Dot-product token similarity is augmented with a learned 10-D raw relation:
observed displacement (3), projected displacement (3), observed/projected
distance (2), normalized signed time (1), and history-validity (1). Physical
math and softmax logits are FP32 under AMP. Spatial quantities use each sample's
`dref`; coordinates remain native physical z,y,x.

The primitive iterates logical samples, so cross-batch attention is impossible.
Node and tracklet messages use separate projections and conservative sigmoid
gates initialized with negative bias. The residual path is nonzero and remains
fully differentiable at initialization.

## Component and iterative query fusion

`InstanceQueryBuilder` fuses both memories into the pooled current-component
embedding before primary and dynamic split companions are created. All sibling
slots see the same memory; learned slot embeddings create their distinct query
identities.

Every one of the three decoder layers executes:

```text
query self-attention
hierarchical node + tracklet attention
spatial cross-attention
FFN and prediction heads
```

Layer `l+1` uses the physical query reference updated by layer `l`. Temporal
queries are retained and follow the unchanged center bounds and matching rules.

## Cache, migration, and curriculum

Reusable preprocessing lives at `sample/temporal_v3/temporal_graph.pt` and
contains graph scalars, complete candidates, accepted-association metadata,
node ordering/physical metadata, tracklet/hypothesis tensors, history grids,
and projected support. Learned node tokens are never cached. Older nonempty
contracts must be rebuilt explicitly.

Checkpoint migration copies legacy detection projection columns 0-13 exactly,
zeros new column 14, preserves the existing 8-to-22 hypothesis migration, and
initializes new memory modules from the receiving model. Optimizer state with a
different parameter structure is rejected rather than silently misloaded.

Candidate-GNN/history/node construction remains in the temporal curriculum
group. Component and decoder memory modules are nested under query builder and
decoder and therefore enter at `query_bootstrap`. The five stages and persistent
optimizer objects are unchanged.

## Debugging and memory strategy

`return_debug=True` exposes detached node metadata/tokens, component and
per-layer query entropy/max/top-k weights, query-slot mappings, and ablation
labels. `return_full_temporal_attention=True` additionally returns full
batch-block-diagonal attention matrices for small diagnostics.

Same-weight forward ablations are `zero_node`, `shuffle_node`, `tracklet_only`,
and `node_only`; the detection graph separately supports `accepted_only`.
Attention tensors scale with query-by-memory size, while the candidate GNN is
the principal new activation cost. No evidence is silently removed for memory
optimization.
