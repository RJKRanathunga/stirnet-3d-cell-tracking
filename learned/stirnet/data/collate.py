from __future__ import annotations

import torch

from .graph_builder import DETECTION_EDGE_DIM, HYPOTHESIS_EDGE_DIM
from ..temporal_events import TEMPORAL_NODE_EVENT_FEATURE_DIM


def _hypothesis_edges_compatible(value: torch.Tensor) -> torch.Tensor:
    """Pad legacy 8-D edges while preserving their unchanged semantic prefix."""
    if value.shape[-1] == HYPOTHESIS_EDGE_DIM:
        return value
    if value.shape[-1] == 8:
        return torch.nn.functional.pad(value, (0, HYPOTHESIS_EDGE_DIM - 8))
    raise ValueError(
        f"hypothesis_edge_attr has width {value.shape[-1]}; expected 8 or {HYPOTHESIS_EDGE_DIM}"
    )


def _cat(samples,key,shape_tail,dtype=torch.float32):
    vals=[s.get(key) for s in samples]
    vals=[v for v in vals if v is not None]
    if vals:return torch.cat(vals,dim=0)
    return torch.zeros((0,*shape_tail),dtype=dtype)


def stirnet_collate(samples:list[dict])->dict:
    if not samples: raise ValueError("empty batch")
    shapes={tuple(s["spatial_inputs"].shape) for s in samples}
    if len(shapes)!=1:
        raise ValueError(f"STIR-Net uses spacing/shape buckets; got incompatible shapes: {shapes}")
    channels={int(s["spatial_inputs"].shape[0]) for s in samples}
    if channels != {5}:
        raise ValueError(
            "spatial_inputs must use the five-channel V2 order: raw, current "
            "foreground, current EDT, current boundary, current marker"
        )
    B=len(samples)
    batch={
        "spatial_inputs":torch.stack([s["spatial_inputs"] for s in samples]),
        "instance_labels":torch.stack([s["instance_labels"] for s in samples]),
        "spacing_um":torch.stack([torch.as_tensor(s["spacing_um"]).float() for s in samples]),
        "dref_um":torch.stack([torch.as_tensor(s["dref_um"]).float() for s in samples]),
        "targets":[s["target"] for s in samples],
    }
    if all(isinstance(s.get("geometry_targets_static"), dict) for s in samples):
        field_names = tuple(samples[0]["geometry_targets_static"].keys())
        batch["geometry_targets_static"] = {
            name: torch.stack([torch.as_tensor(s["geometry_targets_static"][name]) for s in samples])
            for name in field_names
        }
    # instances
    inst_feats=[];inst_ids=[];inst_cent=[];inst_batch=[]
    for b,s in enumerate(samples):
        n=len(s["instance_ids"])
        inst_feats.append(s["instance_features"]);inst_ids.append(s["instance_ids"]);inst_cent.append(s["instance_centroids_um"])
        inst_batch.append(torch.full((n,),b,dtype=torch.long))
    batch["instance_features"]=torch.cat(inst_feats) if inst_feats else torch.zeros((0,14))
    batch["instance_ids"]=torch.cat(inst_ids) if inst_ids else torch.zeros((0,),dtype=torch.long)
    batch["instance_centroids_um"]=torch.cat(inst_cent) if inst_cent else torch.zeros((0,3))
    batch["instance_batch"]=torch.cat(inst_batch) if inst_batch else torch.zeros((0,),dtype=torch.long)

    # packed detection graph + hypothesis graph
    gx=[]; gei=[]; gea=[]; aei=[]; aea=[]; tid=[]; node_ids=[]; node_refs=[]; node_dt=[]
    node_events=[]; complete_event_contract=True
    tref=[]; tstatus=[]; hei=[]; hea=[]; tb=[]
    node_grids=[];node_valid=[]
    supports=[];support_valid=[];support_dt=[];support_centers=[];support_extents=[]
    best_ids=[];best_overlap=[];second_overlap=[]
    node_off=0; hyp_off=0
    for b,s in enumerate(samples):
        x=s.get("graph_x",torch.zeros((0,32))); ei=s.get("graph_edge_index",torch.zeros((2,0),dtype=torch.long)); ea=s.get("graph_edge_attr",torch.zeros((0,DETECTION_EDGE_DIM)))
        if len(x) and ea.shape[-1] != DETECTION_EDGE_DIM:
            raise ValueError(
                f"Non-empty cached detection graph has edge width {ea.shape[-1]}; "
                f"temporal cache contract v3 requires {DETECTION_EDGE_DIM}. Rebuild the cache "
                "so legacy accepted-edge topology is not mistaken for the candidate graph."
            )
        if not len(x) and ea.shape[-1] != DETECTION_EDGE_DIM:
            ea=torch.zeros((0,DETECTION_EDGE_DIM),dtype=ea.dtype,device=ea.device)
        accepted_ei=s.get("accepted_association_edge_index",torch.zeros((2,0),dtype=torch.long))
        accepted_ea=s.get("accepted_association_edge_attr",torch.zeros((0,3)))
        tr=s.get("temporal_ref_um",torch.zeros((0,3))); st=s.get("temporal_status",torch.zeros((len(tr),10)))
        ti=s.get("tracklet_id",torch.zeros((len(x),),dtype=torch.long))
        hi=s.get("hypothesis_edge_index",torch.zeros((2,0),dtype=torch.long)); ha=_hypothesis_edges_compatible(s.get("hypothesis_edge_attr",torch.zeros((0,HYPOTHESIS_EDGE_DIM))))
        grid=s.get("node_instance_grid",torch.zeros((len(x),4,12,12,12),dtype=torch.float16))
        valid=s.get("node_history_valid",torch.zeros((len(x),),dtype=torch.bool))
        support=s.get("history_support",torch.zeros((len(tr),2,2,12,12,12),dtype=torch.float16))
        support_ok=s.get("history_support_valid",torch.zeros((len(tr),2),dtype=torch.bool))
        support_delta=s.get("history_support_dt",torch.zeros((len(tr),2)))
        support_center=s.get("history_support_center_um",torch.zeros((len(tr),2,3)))
        support_extent=s.get("history_support_extent_um",torch.zeros((len(tr),2)))
        node_grids.append(grid);node_valid.append(valid)
        supports.append(support);support_valid.append(support_ok);support_dt.append(support_delta)
        support_centers.append(support_center);support_extents.append(support_extent)
        best_ids.append(s.get("best_current_component_id",torch.full((len(tr),),-1,dtype=torch.long)))
        best_overlap.append(s.get("best_component_overlap",torch.zeros((len(tr),))))
        second_overlap.append(s.get("second_best_component_overlap",torch.zeros((len(tr),))))
        gx.append(x); gea.append(ea); gei.append(ei+node_off if ei.numel() else ei)
        aei.append(accepted_ei+node_off if accepted_ei.numel() else accepted_ei)
        aea.append(accepted_ea)
        node_ids.append(s.get("node_ids",torch.arange(len(x),dtype=torch.long)))
        node_refs.append(s.get("node_observed_ref_um",x[:,1:4]*torch.as_tensor(s["dref_um"])))
        node_dt.append(s.get("node_time_offset",x[:,0]*2.0))
        event_features=s.get("node_event_features")
        if event_features is None:
            # Leave reconstruction to StirNet, which knows the configured
            # temporal radius. This keeps old cache migration exact even when
            # a non-default radius was used.
            complete_event_contract=False
        else:
            if event_features.shape != (
                len(x),TEMPORAL_NODE_EVENT_FEATURE_DIM
            ):
                raise ValueError(
                    "node_event_features must align with graph_x and have shape [N,8]"
                )
            node_events.append(event_features)
        # local tracklet ids become global hypothesis ids
        tid.append(ti+hyp_off if ti.numel() else ti)
        tref.append(tr);tstatus.append(st);tb.append(torch.full((len(tr),),b,dtype=torch.long))
        hei.append(hi+hyp_off if hi.numel() else hi);hea.append(ha)
        node_off+=len(x);hyp_off+=len(tr)
    batch["graph_x"]=torch.cat(gx) if gx else torch.zeros((0,32))
    batch["graph_edge_index"]=torch.cat(gei,dim=1) if gei else torch.zeros((2,0),dtype=torch.long)
    batch["graph_edge_attr"]=torch.cat(gea) if gea else torch.zeros((0,DETECTION_EDGE_DIM))
    batch["accepted_association_edge_index"]=torch.cat(aei,dim=1) if aei else torch.zeros((2,0),dtype=torch.long)
    batch["accepted_association_edge_attr"]=torch.cat(aea) if aea else torch.zeros((0,3))
    batch["tracklet_id"]=torch.cat(tid) if tid else torch.zeros((0,),dtype=torch.long)
    batch["node_ids"]=torch.cat(node_ids) if node_ids else torch.zeros((0,),dtype=torch.long)
    batch["node_observed_ref_um"]=torch.cat(node_refs) if node_refs else torch.zeros((0,3))
    batch["node_time_offset"]=torch.cat(node_dt) if node_dt else torch.zeros((0,))
    if complete_event_contract:
        batch["node_event_features"]=(
            torch.cat(node_events)
            if node_events
            else torch.zeros((0,TEMPORAL_NODE_EVENT_FEATURE_DIM))
        )
    batch["temporal_ref_um"]=torch.cat(tref) if tref else torch.zeros((0,3))
    batch["temporal_status"]=torch.cat(tstatus) if tstatus else torch.zeros((0,10))
    batch["temporal_batch"]=torch.cat(tb) if tb else torch.zeros((0,),dtype=torch.long)
    batch["hypothesis_edge_index"]=torch.cat(hei,dim=1) if hei else torch.zeros((2,0),dtype=torch.long)
    batch["hypothesis_edge_attr"]=torch.cat(hea) if hea else torch.zeros((0,HYPOTHESIS_EDGE_DIM))
    batch["node_instance_grid"]=torch.cat(node_grids) if node_grids else torch.zeros((0,4,12,12,12),dtype=torch.float16)
    batch["node_history_valid"]=torch.cat(node_valid) if node_valid else torch.zeros((0,),dtype=torch.bool)
    batch["history_support"]=torch.cat(supports) if supports else torch.zeros((0,2,2,12,12,12),dtype=torch.float16)
    batch["history_support_valid"]=torch.cat(support_valid) if support_valid else torch.zeros((0,2),dtype=torch.bool)
    batch["history_support_dt"]=torch.cat(support_dt) if support_dt else torch.zeros((0,2))
    batch["history_support_center_um"]=torch.cat(support_centers) if support_centers else torch.zeros((0,2,3))
    batch["history_support_extent_um"]=torch.cat(support_extents) if support_extents else torch.zeros((0,2))
    batch["best_current_component_id"]=torch.cat(best_ids) if best_ids else torch.zeros((0,),dtype=torch.long)
    batch["best_component_overlap"]=torch.cat(best_overlap) if best_overlap else torch.zeros((0,))
    batch["second_best_component_overlap"]=torch.cat(second_overlap) if second_overlap else torch.zeros((0,))
    return batch
