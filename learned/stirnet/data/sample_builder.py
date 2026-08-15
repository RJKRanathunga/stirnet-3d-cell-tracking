from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage as ndi

from .targets import build_gt_targets, estimate_dref_um, extract_instance_metadata, make_instance_boundary
from .graph_builder import DETECTION_EDGE_DIM
from ..temporal_events import TEMPORAL_NODE_EVENT_FEATURE_DIM


def robust_normalize(raw: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.8) -> np.ndarray:
    raw=np.asarray(raw,np.float32)
    lo,hi=np.percentile(raw,[low_pct,high_pct])
    if hi<=lo:return np.zeros_like(raw,np.float32)
    return np.clip((raw-lo)/(hi-lo),0,1).astype(np.float32)


def build_spatial_channels(raw_norm: np.ndarray, instance_labels: np.ndarray, spacing_um, dref_um: float, marker_heatmap=None) -> np.ndarray:
    foreground=(instance_labels>0).astype(np.float32)
    edt=np.zeros_like(raw_norm,np.float32)
    for label in np.unique(instance_labels):
        if label<=0:continue
        m=instance_labels==label
        d=ndi.distance_transform_edt(m,sampling=spacing_um)/max(dref_um,1e-6)
        edt[m]=d[m]
    boundary=make_instance_boundary(instance_labels).astype(np.float32)
    marker=np.zeros_like(raw_norm,np.float32) if marker_heatmap is None else np.asarray(marker_heatmap,np.float32)
    return np.stack([raw_norm,foreground,edt,boundary,marker],axis=0)


def build_cached_sample(
    raw: np.ndarray,
    instance_labels: np.ndarray,
    gt_labels: np.ndarray,
    spacing_um,
    *,
    marker_heatmap: np.ndarray | None = None,
    temporal_graph: dict | None = None,
    dref_um: float | None = None,
    normalize_raw: bool = True,
    metadata: dict | None = None,
) -> dict:
    spacing=tuple(float(v) for v in spacing_um)
    raw_norm=robust_normalize(raw) if normalize_raw else np.asarray(raw,np.float32)
    dref=float(dref_um if dref_um is not None else estimate_dref_um(gt_labels,spacing))
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
        "metadata":metadata or {},
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
