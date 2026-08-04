# Stage 11 — track reconciliation

Stage 11 is the final identity-repair stage after Stage 10. It supports one
topology only: one unexplained interior ended segment continuing as one newly
started segment. It does not detect divisions, reinterpret Stage 8 merges, or
attach an ending to a track that was already running.

## Resolution policies

`diagnostic` produces endpoints, candidates, rankings, and global proposals but
does not rewrite IDs. `conservative` applies only globally selected edges that
pass the configured score, margin, and strong-support rules. `submission`, the
production default, runs the conservative phase first and then globally assigns
remaining eligible sources to remaining admissible targets.

The submission fallback is deliberately bounded. It reuses the already-built
candidate graph, so it never searches beyond `maximum_gap_frames`, enlarges the
gap-dependent physical radius, or invents a target. An ending with no admissible
unused newly started track remains unresolved.

The fallback will not:

- search beyond the configured time window;
- attach to an already-running track;
- attach to a boundary entry;
- override an accepted or probable division;
- override Stage 8 merge handling;
- fabricate a successor when no new track exists.

## Evidence and hard protections

Candidate positions and distances use anisotropic physical ZYX coordinates.
Soft evidence combines forward and backward motion, stable-neighbour anchors,
local distance preservation, gap length, volume, shape, intensity, target
persistence, optional Stage 7 alternatives, and mutual uniqueness. Missing
optional evidence is omitted from the weighted denominator and is never treated
as negative evidence.

Boundary exits and entries, virtual merge endpoints, active merge transitions,
confirmed division parent/child endpoints, probable division topology, temporal
overlap, lineage conflicts, and candidates outside the hard physical gate are
not admissible. Protections are endpoint-specific: a confirmed division child
may still be repaired after an independent later break.

## Identity and lineage output

For an accepted `ended E → new N` continuation, every row of original segment
`N` is rewritten to canonical identity `E`. Chains such as `81 → 143 → 206`
resolve to `81`. Stage 11 creates no detections and does not interpolate missing
frames; a final track can therefore contain frame-number gaps.

Stage 8 segmentation events and Stage 10 division, edge, lineage, and protection
tables are copied into the Stage 11 output with canonical IDs. Confirmed lineage
edges are retained, probable divisions remain probable, and `track_lineage` is
rebuilt with one row per final canonical ID.

The stage writes final tracks, canonical event tables, endpoint classifications,
candidate evidence, decisions, remap provenance, unresolved endings, validation
checks, and metadata to `stage_11_track_reconciliation`. Optional `StageTrace`
diagnostics expose the same normal result tables plus intermediate evidence and
one provenance record per candidate and source decision.

## Current limitations

Only broken 1-to-1 continuations are repaired. Stage 11 does not solve new merge,
split, division, many-to-one, or one-to-many topologies. Its thresholds and
weights are provisional configurable engineering defaults, not calibrated
biological probabilities. Optional Stage 7 evidence is used only when identifiers
and target detections can be matched safely.
