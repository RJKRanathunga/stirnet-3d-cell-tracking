# Implementation status

## Complete production path

- Exact disabled behavior and the existing pairwise shadow/apply backend.
- Algorithm selection with frozen nested `FourDGraphConfig` defaults.
- Complete provisional-sequence evidence retention in memory.
- Columnar 4D observations and physical face distances.
- Per-frame deterministic KD-tree spatial graphs with CSR adjacency.
- Adjacent and `t -> t+2` temporal candidates (`t -> t+3` configurable).
- Union of provisional, row/column top-K, locally competitive, gap, and
  neighbour-vote-expanded candidates; hard Stage 7 safety exclusions remain
  hard.
- Persistent neighbour-identity/vector histories with visibility-aware robust
  confidence.
- Ambiguity components that unlock high-confidence but motion/trajectory-
  inconsistent provisional edges.
- Sparse robust trajectory, spatial-vector, and persistent-relation factors.
- Face-specific entry/exit, generic birth/death, and sequence endpoint events.
- Sparse `scipy.optimize.milp` binary flow model with linearized pair factors.
- Exact component limits, overlapping windows, deterministic iterative
  reweighting fallback, and component-local provisional fallback on failure.
- Deterministic path extraction, optimized ID assignment, and ID remap table.
- Strict uniqueness, flow, safety, acyclicity, gap, boundary, schema, and ID
  validation.
- Shadow/apply Stage 7 integration, rebuilt apply artifacts, StageTrace records,
  optional I/O, evaluation utility, and backwards-compatible legacy loading.

## Intentional limitations

- The model remains one-to-one; it does not infer divisions or segmentation
  merges.
- Persistent histories are recomputed from the provisional identity prior for
  each complete run; iterative fallback reweights them but does not create a
  learned identity model.
- Missing frames are represented by direct gap edges. Synthetic detections are
  never written into `tracks.csv`.
- Thresholds are conservative engineering defaults, not calibrated biological
  claims. Apply mode should follow manual review of shadow-mode real-data diffs.
- On the current 20-frame sample (4,421 observations), a complete default shadow
  run exceeded the bounded 300-second development benchmark. Profiling isolated
  the dominant cost to repeated large unary-flow solves for one oversized,
  highly connected ambiguity component; candidate and pair-factor construction
  were not the dominant phases. Solver calls are individually time-limited and
  fail component-locally back to provisional edges, but the aggregate of several
  window/iteration calls has no whole-run deadline yet.
- Remaining performance work is deterministic subdivision of giant ambiguity
  components, reuse of window flow-model structure across reweighting
  iterations, and an optional whole-optimizer runtime budget. Until that work is
  validated on all five samples, keep the backend in shadow mode.
