from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from .targets import build_gt_targets, estimate_model_dref_um, extract_instance_metadata, make_instance_boundary
from .graph_builder import DETECTION_EDGE_DIM
from ..temporal_events import TEMPORAL_NODE_EVENT_FEATURE_DIM


SPATIAL_CHANNEL_NAMES = (
    "raw",
    "current_foreground_prior",
    "current_edt_prior",
    "current_boundary_prior",
    "current_marker_prior",
)

TRUSTED_MODEL_DREF_SOURCES = frozenset(
    {
        "current_segmentation",
        "fixed_acquisition_prior",
        "explicit_non_gt",
    }
)


def resolve_model_dref(
    instance_labels: np.ndarray,
    spacing_um,
    *,
    cached_dref_um: float | torch.Tensor | None = None,
    metadata: dict | None = None,
) -> tuple[float, str, bool]:
    """Resolve model scale from explicit non-GT provenance or current labels.

    The presence of a cached numeric value is intentionally insufficient: old
    caches may have serialized a GT-derived dref without provenance metadata.
    """
    provenance = dict(metadata or {})
    source = provenance.get("model_dref_source")
    candidate = cached_dref_um
    if candidate is None:
        candidate = provenance.get("model_dref_um")
    if source in TRUSTED_MODEL_DREF_SOURCES and candidate is not None:
        value = float(torch.as_tensor(candidate))
        if np.isfinite(value) and value > 0:
            return value, str(source), True
    value = float(estimate_model_dref_um(np.asarray(instance_labels), spacing_um))
    return value, "current_segmentation", False


def renormalize_cached_dref(
    sample: dict,
    *,
    cached_dref_um: float,
    model_dref_um: float,
    model_dref_source: str = "current_segmentation",
) -> dict:
    """Convert dref-normalized cached inputs without using GT-derived scale."""
    result = dict(sample)
    scale = float(cached_dref_um) / max(float(model_dref_um), 1e-6)
    if "spatial_inputs" in result:
        spatial = torch.as_tensor(result["spatial_inputs"]).clone()
        if spatial.ndim == 5:
            spatial[:, 2] *= scale
        elif spatial.ndim == 4:
            spatial[2] *= scale
        else:
            raise ValueError("spatial_inputs must be [C,Z,Y,X] or [B,C,Z,Y,X]")
        result["spatial_inputs"] = spatial
    if "graph_x" in result:
        graph_x = torch.as_tensor(result["graph_x"]).clone()
        for start, stop in ((1, 4), (5, 11), (17, 23), (25, 27)):
            graph_x[:, start:stop] *= scale
        result["graph_x"] = graph_x
    if "graph_edge_attr" in result:
        edge_attr = torch.as_tensor(result["graph_edge_attr"]).clone()
        edge_attr[:, 1:5] *= scale
        edge_attr[:, 7] *= scale
        result["graph_edge_attr"] = edge_attr
    history = result.get("node_instance_grid")
    if history is not None and torch.as_tensor(history).numel():
        history = torch.as_tensor(history)
        original_dtype = history.dtype
        values = history.float()
        extent_scale = float(model_dref_um) / max(float(cached_dref_um), 1e-6)
        theta = torch.zeros(
            (values.shape[0], 3, 4), device=values.device, dtype=values.dtype
        )
        theta[:, 0, 0] = extent_scale
        theta[:, 1, 1] = extent_scale
        theta[:, 2, 2] = extent_scale
        grid = F.affine_grid(theta, values.shape, align_corners=True)
        values = F.grid_sample(
            values,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        values[:, 1].mul_(scale).clamp_(-1.0, 1.0)
        result["node_instance_grid"] = values.to(original_dtype)
    result["dref_um"] = torch.as_tensor(model_dref_um, dtype=torch.float32)
    metadata = dict(result.get("metadata", {}))
    metadata.update(
        {
            "model_dref_um": float(model_dref_um),
            "model_dref_source": model_dref_source,
            "legacy_cached_dref_um": float(cached_dref_um),
        }
    )
    result["metadata"] = metadata
    return result


def robust_normalize(raw: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.8) -> np.ndarray:
    raw=np.asarray(raw,np.float32)
    lo,hi=np.percentile(raw,[low_pct,high_pct])
    if hi<=lo:return np.zeros_like(raw,np.float32)
    return np.clip((raw-lo)/(hi-lo),0,1).astype(np.float32)


def build_spatial_channels(raw_norm: np.ndarray, instance_labels: np.ndarray, spacing_um, dref_um: float, marker_heatmap=None) -> np.ndarray:
    foreground=(instance_labels>0).astype(np.float32)
    edt=np.zeros_like(raw_norm,np.float32)
    for label, bbox in enumerate(ndi.find_objects(instance_labels), 1):
        if bbox is None:
            continue
        local = instance_labels[bbox] == label
        padded = np.pad(local, 1, mode="constant", constant_values=False)
        distance = ndi.distance_transform_edt(
            padded, sampling=spacing_um
        )[tuple(slice(1, -1) for _ in range(3))]
        view = edt[bbox]
        view[local] = distance[local] / max(dref_um, 1e-6)
    boundary=make_instance_boundary(instance_labels).astype(np.float32)
    marker=np.zeros_like(raw_norm,np.float32) if marker_heatmap is None else np.asarray(marker_heatmap,np.float32)
    spatial = np.stack([raw_norm,foreground,edt,boundary,marker],axis=0)
    if spatial.shape[0] != len(SPATIAL_CHANNEL_NAMES):
        raise RuntimeError("STIR-Net spatial channel contract is inconsistent")
    return spatial


def build_cached_sample(
    raw: np.ndarray,
    instance_labels: np.ndarray,
    gt_labels: np.ndarray,
    spacing_um,
    *,
    marker_heatmap: np.ndarray | None = None,
    temporal_graph: dict | None = None,
    dref_um: float | None = None,
    model_dref_source: str | None = None,
    normalize_raw: bool = True,
    metadata: dict | None = None,
) -> dict:
    spacing=tuple(float(v) for v in spacing_um)
    raw_norm=robust_normalize(raw) if normalize_raw else np.asarray(raw,np.float32)
    sample_metadata = dict(metadata or {})
    if model_dref_source is not None:
        sample_metadata["model_dref_source"] = model_dref_source
    dref, resolved_source, _ = resolve_model_dref(
        np.asarray(instance_labels),
        spacing,
        cached_dref_um=dref_um,
        metadata=sample_metadata,
    )
    sample_metadata["model_dref_source"] = resolved_source
    sample_metadata["model_dref_um"] = dref
    spatial=build_spatial_channels(raw_norm,np.asarray(instance_labels),spacing,dref,marker_heatmap)
    inst=extract_instance_metadata(np.asarray(instance_labels),raw_norm,spacing,dref,spatial[4])
    target=build_gt_targets(
        np.asarray(gt_labels), spacing, dref, current_labels=np.asarray(instance_labels)
    )
    sample={
        "spatial_inputs":torch.as_tensor(spatial,dtype=torch.float32),
        "instance_labels":torch.as_tensor(instance_labels,dtype=torch.long),
        "spacing_um":torch.tensor(spacing,dtype=torch.float32),
        "dref_um":torch.tensor(dref,dtype=torch.float32),
        "instance_ids":inst.ids,
        "instance_features":inst.features,
        "instance_centroids_um":inst.centroids_um,
        "target":target,
        "metadata":sample_metadata,
    }
    if temporal_graph: sample.update(temporal_graph)
    else:
        sample.update({
            "graph_x":torch.zeros((0,32)),"graph_edge_index":torch.zeros((2,0),dtype=torch.long),"graph_edge_attr":torch.zeros((0,DETECTION_EDGE_DIM)),
            "node_event_features":torch.zeros((0,TEMPORAL_NODE_EVENT_FEATURE_DIM)),
            "accepted_association_edge_index":torch.zeros((2,0),dtype=torch.long),
            "accepted_association_edge_attr":torch.zeros((0,3)),
            "tracklet_id":torch.zeros((0,),dtype=torch.long),"temporal_ref_um":torch.zeros((0,3)),"temporal_status":torch.zeros((0,10)),
            "node_ids":torch.zeros((0,),dtype=torch.long),"node_observed_ref_um":torch.zeros((0,3)),"node_time_offset":torch.zeros((0,)),
            "hypothesis_edge_index":torch.zeros((2,0),dtype=torch.long),"hypothesis_edge_attr":torch.zeros((0,22)),
            "node_instance_grid":torch.zeros((0,4,12,12,12),dtype=torch.float16),
            "node_history_valid":torch.zeros((0,),dtype=torch.bool),
            "history_support":torch.zeros((0,2,2,12,12,12),dtype=torch.float16),
            "history_support_valid":torch.zeros((0,2),dtype=torch.bool),
            "history_support_dt":torch.zeros((0,2)),
            "history_support_center_um":torch.zeros((0,2,3)),
            "history_support_extent_um":torch.zeros((0,2)),
            "best_current_component_id":torch.zeros((0,),dtype=torch.long),
            "best_component_overlap":torch.zeros((0,)),
            "second_best_component_overlap":torch.zeros((0,)),
        })
    return sample
