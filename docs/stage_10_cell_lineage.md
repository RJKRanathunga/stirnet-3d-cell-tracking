# Stage 10 cell lineage

Stage 10 detects a deliberately narrow division topology:

```text
one interior parent track ends at frame t
two interior child tracks begin at frame t + 1
```

It runs after Stage 8 because merge repair can insert virtual observations and remap outgoing tracks. The corrected Stage 8 identifiers are canonical: Stage 10 creates lineage relationships but never rewrites those identifiers and never returns a modified track table.

## Candidate generation

Track starts and endings are derived directly from corrected Stage 8 tracks. An eligible parent must end before the sequence ends, have enough observations, and end on a real, nonboundary observation. Only real, nonboundary tracks whose first observation is exactly one frame later are considered as children. Nearby starts are indexed by frame, filtered in physical ZYX space, and enumerated as deterministic unordered pairs.

The parent birth-frame position is predicted from the component-wise median physical velocity over recent real observations. With insufficient motion history, the final parent position is used. Recent real parent volumes provide a median reference volume; parent swelling is retained only as diagnostic evidence.

## Hard structural gates

Candidates are rejected when any of these conditions hold:

- a relevant endpoint is boundary or virtual, or the transition overlaps an active Stage 8 merge representation;
- the combined daughter volume differs too much from the recent parent reference;
- the volume-weighted daughter centroid is too far from the predicted parent position;
- daughter separation at birth is implausibly large;
- either birth instance mask is missing;
- either daughter falls below the hard voxel-count or parent-volume-fraction fragment limit.

The approximately three-voxel daughter seen in the investigation is therefore treated as a segmentation fragment, not biological evidence. Unequal daughters are otherwise permitted.

## Soft evidence and scoring

The primary score components are spatial compatibility, combined-volume conservation, and persistence of the weaker daughter. Daughter divergence is supporting evidence and does not have to be monotonic. Small temporary separation decreases are allowed, while strong collapse lowers the score.

Raw fluorescence is loaded lazily only for candidates that pass cheap structural gates. Stage 6 intensity columns remain available as observation context but are not treated as raw fluorescence. Parent raw intensity compares the final parent frame with preceding real observations, excluding the final frame from the baseline.

Daughter intensity is not expected to be elevated at the first split frame. Delayed paired-daughter core or background-corrected intensity is examined over the next one to three frames. Intensity is secondary, never a hard gate, and missing measurements are omitted from the weighted denominator. Morphology is used only in ordinary-continuation alternatives.

Candidates that pass hard fragment limits but miss preferred voxel or volume-fraction limits receive a bounded artifact penalty. All thresholds, score scales, and weights are provisional and configurable through `CellLineageConfig`; the four clean investigation cases are not enough to establish final biological thresholds.

## Continuation alternatives

For each daughter, Stage 10 scores the competing hypothesis that the parent simply continued as that daughter. The continuation score combines predicted-position agreement, individual volume agreement, and available size/axis similarity. The division margin is the division score minus the stronger continuation score. This prevents an ordinary broken track plus an unrelated nearby birth from being accepted solely because it has a raw `1 → 2` topology.

## Decisions and conflicts

`confirmed` requires all hard gates, both daughters meeting the persistence requirement, and the configured confirmed score and continuation margin. `probable` meets the lower score and margin thresholds but is retained for review. A truncated future window can yield a probable event; it cannot be confirmed without the required observations for both daughters. Other candidates are `rejected` with an explicit primary reason.

Preliminary confirmed candidates rank ahead of probable candidates. Within those groups, conflicts are resolved by decreasing division margin, division score, and combined-volume score, followed by stable frame and track-ID ordering. A selected parent can have one division and a selected child can have one biological parent. Competing candidates become `rejected_conflict`.

## Outputs

- `division_candidates.csv` contains every evaluated pair, its measurements, component scores, alternatives, decision, and rejection diagnostics.
- `division_events.csv` contains conflict-selected confirmed and probable events.
- `lineage_edges.csv` contains two parent-to-child edges for each confirmed event only.
- `track_lineage.csv` contains every corrected Stage 8 track, its biological parent when confirmed, root, generation, and status.
- `protected_tracks.csv` contains the parent and two children from each confirmed event for a future Stage 11 recovery system.

Probable events do not create production lineage edges. Only confirmed events protect tracks.

## Known limitations

The first production detector does not support `parent continues + one child starts`, missing-frame gap divisions, three-or-more-child events, global lineage optimization, fragment repair, or track-ID rewriting. It also does not perform Stage 11 fallback stitching. These cases must remain unsupported rather than being silently converted into confirmed divisions.
