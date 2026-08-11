# 05 — Spatiotemporal Co-Reasoning

## 1. Purpose

Co-reasoning is the central architectural mechanism of STIR-Net.

Its purpose is **not** to concatenate a CNN vector and a graph vector.

Instead, spatial and temporal representations repeatedly interrogate each other before instance decoding.

Each block performs:

```text
temporal hypotheses read spatial evidence
             |
             v
temporal hypotheses update
             |
             v
hypothesis graph reasoning
             |
             v
spatial features read updated temporal hypotheses
             |
             v
spatial convolutional reasoning
```

## 2. Number and placement

V1 uses:

```python
COREASONING_BLOCKS = 2
COREASONING_HEADS = 4
D_MODEL = 128
HEAD_DIM = 32
```

### CR-1

At deepest encoder level E3.

Purpose:

- coarse object/context interpretation;
- identify likely numbers of cells;
- connect temporal anomalies to broad spatial regions.

### CR-2

At decoder/E2 scale.

Purpose:

- medium-scale localization;
- refine center allocation;
- refine candidate separation regions.

Fine native boundaries are still resolved by the CNN decoder after fusion.

## 3. Spatial tokenization

Given feature tensor:

```python
F: [B, C, Zl, Yl, Xl]
```

project to:

```python
S: [B, Ns, 128]
Ns = Zl * Yl * Xl
```

and generate physical coordinates:

```python
S_pos_um: [B,Ns,3]
```

using the feature level's effective spacing.

Use padding masks when variable shapes are padded.

## 4. Temporal tokens

Input:

```python
T: [B or packed M,128]
T_ref_um: [...,3]
salience: [...,1]
reliability: [...,1]
```

## 5. Relative physical positional bias

For temporal hypothesis i and spatial token j:

$$\Delta p_{ij} = p_j^{spatial} - p_i^{temporal}.$$

Build normalized vector:

$$r_{ij} = [ \Delta z/d_\text{ref}, \Delta y/d_\text{ref}, \Delta x/d_\text{ref}, \|\Delta p\|/d_\text{ref} ].$$

MLP:

```text
4 -> 32 -> 4
```

produces one bias value per attention head.

## 6. Temporal reads spatial

Queries:

$$Q=T W_Q^T$$

Keys/values:

$$K=S W_K^S,\qquad V=S W_V^S.$$

For head h:

$$\ell_{ij}^{(h)} = \frac{ Q_i^{(h)}\cdot K_j^{(h)} }{ \sqrt{32} } + b_h(r_{ij}) + M_{ij}.$$

### Search radius

For temporal hypothesis i:

$$R_i = (1.5+a_i)d_\text{ref}.$$

Thus:

- low-salience track → ~1.5 cell diameters;
- high-salience anomaly → up to ~2.5 cell diameters.

Mask:

$$M_{ij}= \begin{cases} 0,& \|\Delta p_{ij}\|\le R_i\\ -\infty,&\text{otherwise} \end{cases}$$

plus spatial padding masks.

The implementation must determine local candidates before materializing
unbounded pairwise tensors, or use configurable query/key chunking with an
online softmax. Chunking is an internal execution strategy: it must retain all
tokens and preserve one global all-cell sample.

### Output

$$U_i = \operatorname{MHA}(T_i,S,S).$$

## 7. Gated temporal update

Concatenate:

$$[T_i,U_i,a_i,r_i].$$

Gate:

$$g_i= \sigma(W_g[T_i,U_i,a_i,r_i]).$$

Update:

$$T_i' = T_i + g_i\odot W_OU_i.$$

Then LayerNorm and FFN may be applied.

This gate is important because Trackastra-derived hypotheses can be wrong.

## 8. Graph reasoning after image inspection

Run one GATv2 hypothesis block:

$$T'' = \operatorname{HypothesisGATv2}(T',E_H).$$

The graph now reasons over **image-aware temporal hypotheses**.

This allows the model to learn relationships such as:

```text
A sees convincing spatial evidence.
B sees convincing spatial evidence.
A and B are both close to the same current instance.
=> current instance may contain two cells.
```

## 9. Spatial reads temporal

Spatial queries:

$$Q=S W_Q^S$$

Temporal keys/values:

$$K=T'' W_K^T,\qquad V=T'' W_V^T.$$

For spatial token j and temporal hypothesis i:

$$\ell_{ji}^{(h)} = \frac{ Q_j^{(h)}\cdot K_i^{(h)} }{ \sqrt{32} } + b_h(r_{ji}) + \lambda_h a_i + \eta_h\log(r_i+\epsilon) + M_{ji}.$$

### Salience bias

$$\lambda_h=\operatorname{softplus}(\theta_h)$$

so highly diagnostic anomalies can receive a learned positive prior.

### Reliability term

Reliability modifies influence separately from anomaly salience.

The exact parameterization of $\eta_h$ may remain unconstrained or be learned with a conservative initialization.

### Empty-neighbour rule

If a spatial token has no temporal hypothesis inside the allowed radius:

```text
temporal update = zero
```

Never perform softmax over all `-inf`.

## 10. Gated spatial update

Let temporal message be $U_j^S$.

Use:

$$g_j^S= \sigma(W_g^S[S_j,U_j^S]).$$

Then:

$$S_j' = S_j + g_j^S\odot W_O^SU_j^S.$$

## 11. Convolution after fusion

Reshape spatial tokens back to:

```python
[B,128,Zl,Yl,Xl]
```

and run two PhysicalAwareResBlocks.

Reason:

- attention communicates object-level evidence;
- convolution restores local geometric coherence.

The final mask boundary is not inferred directly from attention weights.

## 12. Residual outputs

Each co-reasoning block returns:

```python
updated_spatial_feature
updated_temporal_state
```

Both are passed onward.

CR-2 starts from the temporal state already updated by CR-1.

## 13. Spatial-only mode

If temporal state is empty:

```text
temporal-to-spatial message = 0
spatial-to-temporal branch skipped
hypothesis GAT skipped
spatial residual blocks still run
```

This behavior is required for:

- spatial-only pretraining;
- missing Trackastra data;
- ablations.
