# Windowed 4D architecture

## Graph representation

An observation is `(frame, detection_index)`. Intrinsic positions, volume,
appearance/shape values, six physical face distances, provisional identity and
reliability are stored once in a columnar `ObservationStore`.

Each frame has a physical-coordinate `cKDTree` graph with at most eight
neighbours per node inside 25 micrometres. Undirected edges are unique and store
relative ZYX vector, distance, unit direction, log-volume ratio, face visibility
and boundary observability. CSR arrays provide neighbourhood lookup.

Temporal edges are forward directed and support adjacent continuation plus
configurable gaps. Their union contains all provisional continuations, row and
column top-K candidates, locally competitive candidates, and graph expansions.
Expansion propagates stable matched neighbours, preserves full relative vectors,
uses robust median consensus and queries the target KD-tree. An explicitly
safety-invalid Stage 7 pair cannot be revived.

## Persistent relations and factors

For persistent provisional neighbour identities `a,b`, histories retain recent
`p_b - p_a` vectors, distances, relative log volumes, visibility and boundary
coverage. Confidence combines persistence, robust dispersion and observability;
small/unreliable detections have reduced appearance and anchoring influence.

The objective combines Stage 7 unary cost, robust acceleration, sparse spatial
relative-vector/deformation compatibility, persistent-history compatibility,
gap penalties and event alternatives. Trajectory and spatial terms exist only
for structurally related candidate pairs. Each quadratic pair is linearized with
binary `z = x1*x2` and the three standard MILP inequalities.

## Flow and events

Every observation has exactly one predecessor alternative and one successor
alternative:

```text
incoming temporal + birth + face entries + sequence start = 1
outgoing temporal + death + face exits + sequence end = 1
```

Because all temporal edges advance in frame, the solution is acyclic and
one-to-one. Gap edges are ordinary flow edges and do not generate synthetic
track rows. Face-specific costs use physical proximity and directional support;
a supported gap can beat death plus later birth/entry.

## Components, windows, and fallback

Seeds include low probability/margin, births/deaths, competitive alternatives,
gaps, graph expansion, severe motion residuals and trajectory contradictions.
Regions expand through bounded temporal and spatial hops and overlapping regions
merge. Reliable graph-consistent continuations outside these components stay
fixed; confidence alone does not protect a contradictory continuation.

Small components use sparse `scipy.optimize.milp`/HiGHS. Long components use
seven-frame overlapping windows with central commits. Oversized components use
deterministic unary-flow LP relaxations (integral by the flow structure) followed
by structural reweighting. Every exact or fallback solve receives the configured
finite HiGHS time limit. A timeout, non-integral fallback result, solver error,
model-limit failure, or window consistency failure restores only that
component's provisional edges. The assembled graph is always validated before
extraction.

Track starts are ordered by `(frame, detection_index)` and assigned deterministic
new IDs. A many-row overlap table maps provisional IDs to optimized IDs.
