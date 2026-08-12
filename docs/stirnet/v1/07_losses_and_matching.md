# 07 — Matching and Losses

## 1. Principle

Temporal and discovery queries form an unordered fallback set, but primary and
split queries are seeded from a specific current instance. Matching therefore
preserves source semantics before globally assigning fallback queries.

Tracking identities are not used.

## 2. Ground-truth set

For one target patch:

$$G=\{G_1,\ldots,G_K\}.$$

A GT cell belongs to the target set if its center lies in the valid central output region.

Its mask may extend into the context margin.

## 3. Structured Hungarian matching

For predicted query i and GT cell j:

$$C_{ij} = 2C_{exist} + 5C_{dice} + 2C_{focal} + 2C_{center}.$$

Build a compact CPU-resident current-source by GT overlap-count matrix while
constructing the target. A positive overlap defines compatibility; no
source-by-GT-by-volume tensor is created.

Stage A assigns primary/split queries only to GT cells overlapping their own
source. A source overlapping zero GTs contributes no positive seeded query. For
a one-GT source only the primary is eligible. For a source overlapping two or
more GTs, primary and split compete for compatible GTs. The global seeded
assignment also resolves oversegmentation: when multiple source fragments
overlap one GT, at most one primary owns it.

Stage B removes Stage-A-owned GTs and matches temporal/discovery queries
globally to the remainder. Seeded queries never enter Stage B. Both stages are
one-to-one, and incompatible seeded matches are filtered rather than forced.

## 4. Existence matching cost

$$C_{exist} = -\log\sigma(e_i).$$

## 5. Dice matching cost

Use coarse decoder mask probability $p_i(v)$ inside the GT-local support:

$$S_j(v)=\mathbb{1}[\|x(v)-c_j\|_2\le1.5d_{ref}]\lor g_j(v).$$

$$C_{dice} = 1- \frac{ 2\sum_{v\in S_j}p_i(v)g_j(v)+\epsilon }{ \sum_{v\in S_j}p_i(v)+\sum_{v\in S_j}g_j(v)+\epsilon }.$$

Use a coarse feature grid for matching efficiency.

The standard target representation may retain one integer native label map
plus target IDs and centers. Downsample that label map once per required coarse
resolution, then construct instance masks on the coarse grid. Do not eagerly
materialize `K x Z x Y x X` native target masks.

## 6. Focal mask matching cost

Use a binary focal mask cost over the same GT-local coarse support. The exact
effective coarse-grid spacing emitted by the decoder defines physical
coordinates after token capping. Matched-mask focal defaults are
`alpha_positive=0.75`, `gamma=2.0`.

This term adds local voxelwise discrimination beyond Dice overlap.

## 7. Center matching cost

Use physical/cell-scale normalized centers:

$$C_{center} = \frac{ \|\hat c_i-c_j\|_1 }{ d_\text{ref} }.$$

## 8. Existence loss

Matched queries:

```text
target = 1
```

Unmatched queries owned by the valid output region:

```text
target = 0
```

Context-only queries may be ignored.

Use binary focal loss:

$$FL(p,y) = -\alpha_y(1-p_t)^\gamma\log p_t.$$

Initial:

```python
gamma = 2.0
alpha_positive = 0.75
alpha_negative = 0.25
```

Call:

$$L_{exist}.$$

## 9. High-resolution mask loss

Matched positive queries only. Dice and focal reductions use the same physical
GT-local support $S_j$; every positive GT voxel is included even when an
elongated cell extends beyond the nominal radius. Native reduction remains
spatially streamed, and focal normalizes by supported voxels rather than the
complete scene volume.

$$L_{mask}^{hi} = 5L_{Dice}^{hi} + 2L_{Focal}^{hi}.$$

Average per instance before averaging over the batch. Do not allow
high-resolution datasets or remote background to dominate simply because they
contain more voxels.

Render predictions and targets in bounded spatial/query chunks (or equivalent
query-local supports). The streamed reduction must preserve the same per-cell
Dice and focal objectives without allocating every matched native mask at once.

In training, native-mask and dense auxiliary loss chunks use non-reentrant
activation checkpointing when configured. Their backward recomputation
rematerializes only the current target chunk, so CPU-backed label maps remain
the canonical target representation and the full set of per-chunk logits is not
retained on the accelerator. Evaluation and no-gradient loss calculation use
the direct streamed path.

## 10. Coarse mask loss

Applied to matched queries at decoder scales within the same GT-local physical
support (radial support OR positive target voxels):

$$L_{mask}^{coarse} = 1.0L_{Dice}^{coarse} + 0.5L_{Focal}^{coarse}.$$

This directly supervises masks used for masked cross-attention.

## 11. Center regression loss

Matched queries:

$$L_{center} = SmoothL1 \left( \frac{ \hat c-c }{ d_\text{ref} } \right).$$

Use physical coordinates.

## 12. Count consistency loss

Expected number of cells:

$$\hat N= \sum_i\sigma(e_i).$$

$$L_{count} = \frac{ SmoothL1(\hat N,N_{GT}) }{ \max(1,N_{GT}) }.$$

This is a weak auxiliary loss.

## 13. Overlap loss

The original global all-matched-query overlap formula is retained as an
experimental helper but disabled in corrected V1 (`LOSS_OVERLAP = 0`). At
initialization its sum over many diffuse masks dominated the objective and made
near-zero masks the easiest solution. Corrected V1 does not introduce a
replacement overlap objective yet, and skips this computation when its weight
is non-positive while returning a structural zero loss entry.

## 14. Dense foreground loss

Target:

$$F_{GT}(v)= \mathbb{1}\left[ \bigcup_jG_j(v) \right].$$

Loss:

$$L_{fg}=BCE+Dice.$$

## 15. Dense center heatmap loss

Generate a physical Gaussian around each GT center.

For desired physical sigma $\sigma_{\mu m}$,

$$\sigma_z^{vox}=\sigma_{\mu m}/s_z$$

and equivalently for Y/X.

Use a focal heatmap loss.

Call:

$$L_{centerHeat}.$$

## 16. Dense boundary target

Generate boundaries from GT instance labels.

Expand to an approximately fixed physical width, e.g.:

$$w_{boundary}\approx1\mu m.$$

Because native spacings vary, physical expansion radius is axis dependent.

Loss:

$$L_{boundary} = WeightedBCE+Dice.$$

Initial positive BCE weight:

```python
boundary_pos_weight = 4.0
```

## 17. Final V1 objective

Initial weighting:

$$\begin{aligned} L_{final}=&\ 2L_{exist}\\ &+5L_{dice}^{hi} +2L_{focal}^{hi}\\ &+1L_{dice}^{coarse} +0.5L_{focal}^{coarse}\\ &+2L_{center}\\ &+0.25L_{count}\\ &+0.00L_{overlap}\\ &+0.50L_{fg}\\ &+1.00L_{centerHeat}\\ &+0.50L_{boundary}. \end{aligned}$$

These are starting values, not fixed scientific constants.

## 18. Decoder deep supervision

Decoder layers 1 and 2 emit:

- existence logits;
- center prediction;
- coarse mask logits.

Apply corresponding auxiliary loss:

$$L_{total} = L_{final} + 0.5L_{aux}^{(1)} + 0.5L_{aux}^{(2)}.$$

Do not render high-resolution masks for intermediate decoder layers.

Run structured Hungarian matching once from the final decoder output. Reuse
the exact final `pred_indices` and `target_indices` for both auxiliary layers;
intermediate layers construct masks/supports at their own resolution but never
change query-to-GT identity.

## 19. No tracking loss

Explicitly absent:

- edge association loss;
- track identity loss;
- lineage loss;
- Trackastra consistency loss.

Temporal components receive gradient only through segmentation-instance losses.

## 20. No explicit preservation loss in V1

Correct input instances are represented heavily in the training distribution.

The normal mask/existence/center losses already reward preserving them.

A separate "stay close to input segmentation" term risks preserving subtle segmentation errors.

Only add a preservation regularizer later if clean-instance damage remains a measured failure.

## 21. Loss logging

Training must log:

```text
total loss
existence
high-res Dice
high-res focal
coarse Dice
coarse focal
center
count
overlap
foreground
center heatmap
boundary
```

Additionally log gradient norms for major groups:

```text
spatial backbone
temporal encoder
co-reasoning
query decoder
```

This is required before tuning loss weights.
