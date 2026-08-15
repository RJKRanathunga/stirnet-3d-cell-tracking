# 07 — Matching and Losses

## 1. Principle

Primary/split, temporal, and discovery queries have different semantics.
Matching preserves source ownership first, then admits only physically
plausible recovery queries. A GT without a plausible candidate is legally
unmatched.

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

In spatial-proposal mode, Stage A retains source-overlap eligibility for
source-linked proposals and initial-anchor proximity eligibility for off-mask
proposals. Its assignment cost is immutable-initial-anchor distance to the GT
center, so evolving centers or coarse masks cannot exchange biological target
ownership between nearby proposal queries. Excess proposals remain unmatched.
Legacy mode retains the source-compatible primary/split Stage A behavior.

Stage B1 considers temporal queries against GTs left by Stage A. An edge is
eligible only when the immutable temporal initial reference is within
`temporal_match_radius_dref` (default `1.0`) of the GT center in Euclidean
cell-scale distance. The final decoded center cannot redefine the temporal
clue's semantic eligibility.

Stage B2 considers discovery queries only against GTs left by B1. Because a
discovery query has no temporal anchor, eligibility uses its final decoded
center and `discovery_match_radius_dref` (default `1.5`). Seeded queries never
enter either recovery stage.

Every stage is one-to-one. Within a recovery stage, an augmented Hungarian
problem gives real queries and GTs legal dummy assignments. The unmatched
penalty dominates all variation among eligible real costs, while forbidden
edges cost more than remaining unmatched. This produces maximum eligible-edge
cardinality first and minimum cost second; an ineligible edge is never returned.
Unmatched queries and unmatched GTs are both valid outcomes.

## 4. Existence matching cost

$$C_{exist} = -\log\sigma(e_i).$$

## 5. Dice matching cost

Use coarse decoder mask probability $p_i(v)$ inside the GT-local support:

$$S_j(v)=\mathbb{1}[\|x(v)-c_j\|_2\le1.5d_{ref}]\lor g_j(v).$$

$$C_{dice} = 1- \frac{ 2\sum_{v\in S_j}p_i(v)g_j(v)+\epsilon }{ \sum_{v\in S_j}p_i(v)+\sum_{v\in S_j}g_j(v)+\epsilon }.$$

Use a coarse feature grid for matching efficiency.

The standard target representation retains one integer native label map plus
target IDs and centers. For every required decoder shape, selected foreground
label voxels scatter directly to `[target_index, coarse_linear_index]`; a coarse
bin is positive when any native voxel from that instance maps to it. Different
instances may overlap at coarse resolution. This occupancy-preserving path is
used by final matching, final coarse losses, and auxiliary-layer losses and
never constructs `K x Z x Y x X` native targets. Nearest multiclass-label
sampling remains debug-only because it can erase small instances.

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

Matched positive queries only. Spatial proposals are decoded after matching on
native crops around their immutable anchors. Their target crop is derived
directly from the CPU integer label map and the complete physical `1.5*dref`
sphere is supervised: every allowed voxel is either matched-GT positive or
explicit negative. GT voxels outside the allowed sphere are not unioned into
support. Dense probability side evidence is detached, while D0 remains attached
so joint local-mask loss can train the spatial decoder/backbone.

Non-spatial queries retain the legacy streamed native objective and its
role-specific support/prior semantics. Proposal-local and legacy results are
combined by matched-query count while preserving the public `dice_hi` and
`focal_hi` metrics.

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

Let $N_{matched}$ be the number of positive existence targets created by the
final structured assignment:

$$L_{count} = \frac{ SmoothL1(\hat N,N_{matched}) }
{ \max(1,N_{matched}) }.$$

This is a weak auxiliary loss. When all GTs are matched it is identical to the
old GT-count target. When recovery candidates are exhausted it does not force
unrelated queries positive merely to reach the raw GT count. Diagnostics report
both raw GT count and matched-positive count. The staged curriculum delays this
loss until joint training because early count gradients can oppose positive
existence gradients.

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

## Historical evidence and matching eligibility

History may change existence, center, mask, and temporal representations through
the existing segmentation losses. It adds no track identity, association,
lineage, or Trackastra-consistency loss. Temporal-recovery eligibility remains
defined by the immutable initial temporal reference before decoder refinement;
projected support must not redefine which GT instance a temporal clue was
physically eligible to recover.
