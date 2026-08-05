# Stage 11 — track reconciliation

Stage 11 is the final identity-repair stage after Stage 10. It supports one
topology only: one unexplained interior ended segment continuing as one newly
started segment. It does not detect divisions, reinterpret Stage 8 merges, or
attach an ending to a track that was already running.

## Resolution policies

`diagnostic` produces endpoints, candidates, rankings, and global proposals but
does not rewrite IDs. `conservative` applies globally selected ordinary edges
that pass the configured score, margin, and strong-support rules, followed by
globally selected evidence-supported small-cell edges. `submission`, the
production default, runs those two phases first and then globally assigns
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

## Small-cell reliability

Segmentation-derived volume, morphology, intensity dispersion, and intensity sum
are unusually unstable for the small objects identified by the current Stage 7
investigation. Stage 11 therefore calculates
`effective_pair_volume = min(source_reference_volume, target_reference_volume)`:
if either endpoint is very small, the pair uses the less reliable regime. The
diagnostic regimes are `extremely_small` through 100 voxels, `small` through 200,
`transition` through 350, and `normal` above 350. Missing endpoint volume leaves
the regime `unknown` and retains normal scoring.

For size-aware volume scoring, the log-error scale is interpolated in log-volume
space through the following calibration knots: `log(8)` at 75 voxels,
`log(5)` at 150, `log(3)` at 250, and `log(2)` at 400. It returns smoothly to the
configured ordinary Stage 11 scale at 600 voxels. At and above 600 voxels the
ordinary score is used exactly.

Evidence weights are candidate-local. At or below 100 voxels, spatial, anchor,
neighbourhood, and uniqueness evidence is strengthened while volume and shape
weights use 0.10 multipliers and intensity uses 0.50. At 200 voxels the volume,
shape, and intensity multipliers are 0.35, 0.30, and 0.70. All multipliers
interpolate in log-volume space back to 1.0 at 600 voxels. The global
configuration object is never mutated. Intensity mean and median retain most of
the within-intensity weight for small cells; standard deviation, IQR, and CV are
strongly reduced, and intensity sum is omitted. Available components are
normalized without penalizing missing columns. Normal cells retain the previous
combined mean/median/std/sum behavior.

The `accepted_small_cell_supported` decision is distinct from forced submission
repair. It is available only through 200 voxels and requires an admissible
structurally valid candidate, strong forward prediction, at least two supports
among forward, backward, stable anchors, and neighbourhood preservation,
mutual-best ownership, and configured source/target score margins. These
candidates remain inside component-wise global one-to-one assignment. They are
selected after ordinary conservative assignments and before bounded submission
fallback; their decision rows have `forced = false`.

A one-observation source normally remains excluded for insufficient motion
history. Small-cell mode lets such a source enter candidate scoring only when the
pair is at most 200 voxels. The edge still has to pass the complete special
support and uniqueness predicate; it is explicitly barred from ordinary
conservative and forced fallback assignment. This narrow exception allows
auditable chains containing tiny one-frame fragments without weakening the
normal-cell history requirement.

Candidate artifacts record the effective volume, regime, applied volume scale,
all weight multipliers, four support flags, support count, uniqueness predicate,
special-acceptance predicate, ordinary score, size-aware score, and score delta.
Decision rows repeat the key regime and support facts. `StageTrace` records why
the mode applied, which supports passed, which feature weights were reduced, and
whether the special path was eligible.

Boundary exits and entries, virtual merge endpoints, active merge transitions,
confirmed division parent/child endpoints, probable division topology, temporal
overlap, lineage conflicts, and candidates outside the hard physical gate are
not admissible. Small-cell reliability never overrides these protections and
never attaches to an already-running target. Protections are endpoint-specific:
a confirmed division child may still be repaired after an independent later
break.

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
biological probabilities or biological constants. They are empirical defaults
derived from the current small-cell investigation and should be recalibrated as
curated evidence grows. Optional Stage 7 evidence is used only when identifiers
and target detections can be matched safely. Small-cell mode does not expand the
candidate radius or interpolate missing detections.
