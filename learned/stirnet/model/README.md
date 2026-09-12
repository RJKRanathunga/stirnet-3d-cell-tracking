# STIR-Net spatial-first model package

This package contains the current redesigned STIR-Net model implementation.

## Architectural rule

**Dense geometry first, sparse temporal reasoning second.**

The current frame is first solved as a complete 3-D instance-segmentation problem. Temporal evidence can inspect the spatial representation, alter uncertain object/RAG decisions, and request bounded local geometry refinement, but it never globally overwrites the native spatial trunk.

## Research-derived decisions

- **Omnipose:** predict a structured distance field and its gradient/flow instead of making a single center coordinate carry object geometry.
- **NucMM:** jointly represent foreground, inter-instance contour/separator, and signed distance.
- **NISNet3D:** retain an auxiliary 3-D centroid-vector representation and use learned 3-D geometry to improve markers/separation.
- **PlantSeg:** intentionally oversegment with marker-controlled watershed, build a region-adjacency graph, and obtain cells through graph partitioning rather than independent overlapping masks.
- **Embedding-style instance segmentation:** keep a centroid-offset field as redundant instance-grouping evidence rather than the sole representation.
- **Masked/local query reasoning:** use object/temporal tokens only on relevant local geometry, especially in the native-resolution refiner.

## Forward path

```text
raw image + segmentation priors
        -> separate evidence stems + gated prior fusion
        -> anisotropy-aware residual 3-D U-Net
        -> native dense geometry
           {foreground, outer surface, separator, SDF,
            SDF-flow, centroid offset, seed score}
        -> marker-controlled 3-D watershed
        -> supervoxels
        -> learned spatial RAG
        -> provisional connected instances
        -> one token per connected spatial instance

historical detection graph
        -> temporal graph encoder
        -> temporal tokens READ D1/D2/native geometry

instance tokens + RAG nodes <-> local temporal memory
        -> existence/split/recovery evidence
        -> gated temporal residuals on RAG edges
        -> optional bounded native geometry refinement
        -> re-run watershed/RAG if geometry changed
        -> final connected graph partition
        -> one in-mask max-SDF center per final instance
```

## Important invariants

1. Final instance IDs are unions of adjacent supervoxels, so disconnected islands cannot share one ID.
2. Final masks are mutually exclusive because they come from one partition rather than independent mask painting.
3. Centers are derived from final masks (maximum-SDF voxel), so every final instance has exactly one interior center.
4. D0 is used by the geometry decoder and local refinement before final instance decisions.
5. Temporal evidence is residual and reliability/locality gated.
6. Raw microscopy has an independent stem, allowing recovery of cells missing from the current mask prior.
7. Watershed is deliberately non-differentiable; geometry and RAG logits receive direct supervised losses.

## Main hierarchy

- `spatial/`: evidence fusion and anisotropy-aware residual U-Net.
- `geometry/`: dense geometry decoder, GT target construction, losses.
- `partition/`: markers, 3-D watershed, RAG construction/network, connected graph partitioner.
- `instances/`: connected-instance tokenization and mask-derived center extraction.
- `temporal/`: history encoding, detection graph encoding, one-way spatial observation, instance/RAG temporal reasoning.
- `refinement/`: ambiguity/recovery request selection and native-resolution geometry residual refiner.

## Integration notes

`StirNet.forward` accepts either a `TemporalInput` object or the core legacy temporal tensors (`graph_x`, `graph_edge_index`, `graph_edge_attr`, `tracklet_id`, `temporal_ref_um`, `temporal_status`, `temporal_batch`). The existing data/training/inference packages will still need to be adapted because the output contract is intentionally no longer query-mask based.

The built-in graph solver is a dependency-free greedy affinity agglomeration with the same connectivity principle as PlantSeg's RAG stage. Its interface is isolated in `partition/partitioner.py`, so a future GASP/Multicut/Mutex-Watershed backend can replace it without changing the neural architecture.

## Verification

From the parent directory containing `model/`:

```bash
python -m model.smoke_test
```

The smoke test exercises the spatial, temporal, historical, graph-partition, and local-refinement paths.
