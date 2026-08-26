"""Tensor contracts shared by preprocessing, model, training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


def _shape(tensor: Tensor, ndim: int, name: str) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got {tuple(tensor.shape)}")


@dataclass
class TrackletBatch:
    """Padded high-purity tracklets.

    Dimensions
    ----------
    B: independent reconciliation components / examples
    N: tracklets in a component
    K: observations retained per tracklet
    S: structured observation features
    F: fingerprint embedding size
    R: tracklet reliability features

    `times` are frame numbers or relative frame coordinates.  `start_xyz_um`
    and `end_xyz_um` are physical positions in z,y,x order.
    """

    structured: Tensor                  # [B, N, K, S]
    observation_mask: Tensor            # [B, N, K] bool
    tracklet_mask: Tensor               # [B, N] bool
    times: Tensor                       # [B, N, K]
    start_xyz_um: Tensor                # [B, N, 3]
    end_xyz_um: Tensor                  # [B, N, 3]
    reliability: Tensor                 # [B, N, R]
    crops: Optional[Tensor] = None      # [B, N, K, C, D, H, W]
    fingerprints: Optional[Tensor] = None  # [B, N, K, F]

    def validate(self) -> None:
        _shape(self.structured, 4, "structured")
        b, n, k, _ = self.structured.shape
        expected = {
            "observation_mask": (b, n, k),
            "tracklet_mask": (b, n),
            "times": (b, n, k),
            "start_xyz_um": (b, n, 3),
            "end_xyz_um": (b, n, 3),
        }
        for name, shape in expected.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if self.reliability.shape[:2] != (b, n):
            raise ValueError("reliability must start with [B, N]")
        if self.crops is None and self.fingerprints is None:
            raise ValueError("either crops or precomputed fingerprints must be supplied")
        if self.crops is not None and self.crops.shape[:3] != (b, n, k):
            raise ValueError("crops must start with [B, N, K]")
        if self.fingerprints is not None and self.fingerprints.shape[:3] != (b, n, k):
            raise ValueError("fingerprints must start with [B, N, K]")
        if self.observation_mask.dtype != torch.bool or self.tracklet_mask.dtype != torch.bool:
            raise TypeError("observation_mask and tracklet_mask must be bool tensors")
        if (self.tracklet_mask & ~self.observation_mask.any(dim=-1)).any():
            raise ValueError("every active tracklet must contain at least one valid observation")
        if (self.observation_mask & ~self.tracklet_mask.unsqueeze(-1)).any():
            raise ValueError("padded tracklets cannot contain valid observations")


@dataclass
class CandidateEdgeBatch:
    """High-recall candidate transition edges between tracklets.

    Expected-position fields are candidate-gap-specific predictions generated
    from the source tracklet A.  They intentionally keep the global-only,
    global+relative-motion, and local-neighbour experts separate.
    """

    edge_index: Tensor                    # [B, E, 2], source/target tracklet index
    edge_mask: Tensor                     # [B, E] bool
    gap_frames: Tensor                    # [B, E]
    pair_features: Tensor                 # [B, E, P] standardized primitive evidence

    expected_global_xyz_um: Tensor        # [B, E, 3]
    expected_global_relative_xyz_um: Tensor  # [B, E, 3]
    expected_local_xyz_um: Tensor         # [B, E, 3]
    expected_backward_source_xyz_um: Tensor  # [B, E, 3]
    prediction_valid: Tensor              # [B, E, 4] bool, same order as predictions

    def validate(self, num_tracklets: int | None = None) -> None:
        _shape(self.edge_index, 3, "edge_index")
        b, e, two = self.edge_index.shape
        if two != 2:
            raise ValueError("edge_index last dimension must be 2")
        if self.edge_mask.shape != (b, e):
            raise ValueError("edge_mask must be [B, E]")
        if self.gap_frames.shape != (b, e):
            raise ValueError("gap_frames must be [B, E]")
        if self.pair_features.shape[:2] != (b, e):
            raise ValueError("pair_features must start with [B, E]")
        for name in (
            "expected_global_xyz_um",
            "expected_global_relative_xyz_um",
            "expected_local_xyz_um",
            "expected_backward_source_xyz_um",
        ):
            if getattr(self, name).shape != (b, e, 3):
                raise ValueError(f"{name} must be [B, E, 3]")
        if self.prediction_valid.shape != (b, e, 4):
            raise ValueError("prediction_valid must be [B, E, 4]")
        if self.edge_mask.dtype != torch.bool or self.prediction_valid.dtype != torch.bool:
            raise TypeError("edge_mask and prediction_valid must be bool tensors")
        if num_tracklets is not None and self.edge_mask.any():
            active = self.edge_index[self.edge_mask]
            if active.min() < 0 or active.max() >= num_tracklets:
                raise IndexError("edge_index contains an invalid tracklet index")


@dataclass
class DivisionHypothesisBatch:
    """Sparse explicit parent -> {child1, child2} hypotheses.

    `edge_pair_index` points at the two candidate continuation edges that share
    the same parent.  `features` stores explicit symmetric biological evidence
    such as volume conservation, daughter balance and branch angle.
    """

    edge_pair_index: Tensor              # [B, D, 2]
    hypothesis_mask: Tensor              # [B, D] bool
    features: Tensor                     # [B, D, Q]

    def validate(self, num_edges: int | None = None) -> None:
        _shape(self.edge_pair_index, 3, "edge_pair_index")
        b, d, two = self.edge_pair_index.shape
        if two != 2:
            raise ValueError("edge_pair_index last dimension must be 2")
        if self.hypothesis_mask.shape != (b, d):
            raise ValueError("hypothesis_mask must be [B, D]")
        if self.features.shape[:2] != (b, d):
            raise ValueError("division features must start with [B, D]")
        if num_edges is not None and self.hypothesis_mask.any():
            active = self.edge_pair_index[self.hypothesis_mask]
            if active.min() < 0 or active.max() >= num_edges:
                raise IndexError("division hypothesis references an invalid edge")


@dataclass
class ReconciliationBatch:
    tracklets: TrackletBatch
    edges: CandidateEdgeBatch
    divisions: Optional[DivisionHypothesisBatch] = None

    def validate(self) -> None:
        self.tracklets.validate()
        b, n = self.tracklets.tracklet_mask.shape
        self.edges.validate(num_tracklets=n)
        if self.edges.edge_index.shape[0] != b:
            raise ValueError("tracklet and edge batch sizes disagree")
        if self.divisions is not None:
            self.divisions.validate(num_edges=self.edges.edge_index.shape[1])
            if self.divisions.edge_pair_index.shape[0] != b:
                raise ValueError("tracklet and division batch sizes disagree")


@dataclass
class TrackletEncoding:
    head: Tensor                         # [B, N, H] - incoming / early state
    tail: Tensor                         # [B, N, H] - outgoing / recent state
    pooled: Tensor                       # [B, N, H]
    fingerprint_sequence: Tensor         # [B, N, K, F]


@dataclass
class ReconciliationOutput:
    tracklets: TrackletEncoding
    edge_embeddings: Tensor              # [B, E, H]
    continuation_logits: Tensor           # [B, E]
    parental_probabilities: Tensor        # [B, E]
    no_parent_probability: Tensor         # [B, E], repeated group q for convenience
    division_prior_logits: Tensor         # [B, N]
    division_logits: Optional[Tensor]     # [B, D]
    appearance_logits: Tensor             # [B, N]
    termination_logits: Tensor            # [B, N]
