from __future__ import annotations

import torch


def _cat(samples,key,shape_tail,dtype=torch.float32):
    vals=[s.get(key) for s in samples]
    vals=[v for v in vals if v is not None]
    if vals:return torch.cat(vals,dim=0)
    return torch.zeros((0,*shape_tail),dtype=dtype)


def stirnet_collate(samples:list[dict])->dict:
    if not samples: raise ValueError("empty batch")
    shapes={tuple(s["spatial_inputs"].shape) for s in samples}
    if len(shapes)!=1:
        raise ValueError(f"STIR-Net V1 uses spacing/shape buckets; got incompatible shapes: {shapes}")
    B=len(samples)
    batch={
        "spatial_inputs":torch.stack([s["spatial_inputs"] for s in samples]),
        "instance_labels":torch.stack([s["instance_labels"] for s in samples]),
        "spacing_um":torch.stack([torch.as_tensor(s["spacing_um"]).float() for s in samples]),
        "dref_um":torch.stack([torch.as_tensor(s["dref_um"]).float() for s in samples]),
        "targets":[s["target"] for s in samples],
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
    gx=[]; gei=[]; gea=[]; tid=[]; tref=[]; tstatus=[]; hei=[]; hea=[]; tb=[]
    node_off=0; hyp_off=0
    for b,s in enumerate(samples):
        x=s.get("graph_x",torch.zeros((0,32))); ei=s.get("graph_edge_index",torch.zeros((2,0),dtype=torch.long)); ea=s.get("graph_edge_attr",torch.zeros((0,14)))
        tr=s.get("temporal_ref_um",torch.zeros((0,3))); st=s.get("temporal_status",torch.zeros((len(tr),10)))
        ti=s.get("tracklet_id",torch.zeros((len(x),),dtype=torch.long))
        hi=s.get("hypothesis_edge_index",torch.zeros((2,0),dtype=torch.long)); ha=s.get("hypothesis_edge_attr",torch.zeros((0,8)))
        gx.append(x); gea.append(ea); gei.append(ei+node_off if ei.numel() else ei)
        # local tracklet ids become global hypothesis ids
        tid.append(ti+hyp_off if ti.numel() else ti)
        tref.append(tr);tstatus.append(st);tb.append(torch.full((len(tr),),b,dtype=torch.long))
        hei.append(hi+hyp_off if hi.numel() else hi);hea.append(ha)
        node_off+=len(x);hyp_off+=len(tr)
    batch["graph_x"]=torch.cat(gx) if gx else torch.zeros((0,32))
    batch["graph_edge_index"]=torch.cat(gei,dim=1) if gei else torch.zeros((2,0),dtype=torch.long)
    batch["graph_edge_attr"]=torch.cat(gea) if gea else torch.zeros((0,14))
    batch["tracklet_id"]=torch.cat(tid) if tid else torch.zeros((0,),dtype=torch.long)
    batch["temporal_ref_um"]=torch.cat(tref) if tref else torch.zeros((0,3))
    batch["temporal_status"]=torch.cat(tstatus) if tstatus else torch.zeros((0,10))
    batch["temporal_batch"]=torch.cat(tb) if tb else torch.zeros((0,),dtype=torch.long)
    batch["hypothesis_edge_index"]=torch.cat(hei,dim=1) if hei else torch.zeros((2,0),dtype=torch.long)
    batch["hypothesis_edge_attr"]=torch.cat(hea) if hea else torch.zeros((0,8))
    return batch
