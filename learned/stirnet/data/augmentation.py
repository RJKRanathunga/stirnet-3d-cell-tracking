from __future__ import annotations

import random
import torch
from torch import Tensor


def random_flip_sample(sample: dict, p: float = 0.5) -> dict:
    """Flip dense zyx tensors and all relative zyx coordinates consistently."""
    out=dict(sample)
    dense_keys=["spatial_inputs","instance_labels"]
    target_dense=["masks","label_map","foreground","center_heatmap","boundary"]
    coord_keys=["instance_centroids_um","temporal_ref_um"]
    for axis in range(3):
        if random.random()>=p: continue
        dense_axis=axis+1  # spatial_inputs C,Z,Y,X
        if "spatial_inputs" in out: out["spatial_inputs"]=torch.flip(out["spatial_inputs"],[dense_axis])
        if "instance_labels" in out: out["instance_labels"]=torch.flip(out["instance_labels"],[axis])
        if "target" in out:
            t=dict(out["target"])
            if "masks" in t: t["masks"]=torch.flip(t["masks"],[axis+1])
            for k in ["label_map","foreground","center_heatmap","boundary"]:
                if k in t: t[k]=torch.flip(t[k],[axis])
            if "centers_um" in t: t["centers_um"]=t["centers_um"].clone(); t["centers_um"][:,axis]*=-1
            if "centers_cellscale" in t: t["centers_cellscale"]=t["centers_cellscale"].clone(); t["centers_cellscale"][:,axis]*=-1
            out["target"]=t
        for k in coord_keys:
            if k in out:
                out[k]=out[k].clone(); out[k][:,axis]*=-1
        if "graph_x" in out and out["graph_x"].numel():
            gx=out["graph_x"].clone()
            gx[:,1+axis]*=-1
            gx[:,17+axis]*=-1
            gx[:,20+axis]*=-1
            out["graph_x"]=gx
        if "graph_edge_attr" in out and out["graph_edge_attr"].numel():
            ea=out["graph_edge_attr"].clone(); ea[:,1+axis]*=-1; out["graph_edge_attr"]=ea
        if "hypothesis_edge_attr" in out and out["hypothesis_edge_attr"].numel():
            ha=out["hypothesis_edge_attr"].clone(); ha[:,axis]*=-1; out["hypothesis_edge_attr"]=ha
    return out


def random_intensity(sample: dict, scale=(0.8,1.2), shift=(-0.1,0.1), gamma=(0.8,1.2), noise_std=(0.0,0.03)) -> dict:
    out=dict(sample); x=out["spatial_inputs"].clone()
    raw=x[0]
    s=raw.new_empty(()).uniform_(*scale); sh=raw.new_empty(()).uniform_(*shift); g=raw.new_empty(()).uniform_(*gamma)
    raw=(raw*s+sh).clamp(0,1).pow(g)
    ns=float(raw.new_empty(()).uniform_(*noise_std).item())
    if ns>0: raw=(raw+torch.randn_like(raw)*ns).clamp(0,1)
    x[0]=raw; out["spatial_inputs"]=x
    return out


def corrupt_temporal_clues(sample: dict, *, hypothesis_dropout=0.10, edge_dropout=0.10,
                            position_jitter_dref=0.10, large_jitter_dref=0.50,
                            false_clue_prob=0.05) -> dict:
    """Perturb cached temporal clues while keeping packed graph indices consistent."""
    out=dict(sample)
    tref=out.get("temporal_ref_um")
    if tref is None or len(tref)==0:
        return out
    dref=float(torch.as_tensor(out["dref_um"]).item())
    M=len(tref)
    keep=torch.rand(M)>=hypothesis_dropout
    # keep at least one if any existed
    if not keep.any(): keep[torch.randint(0,M,(1,))]=True
    old_to_new=torch.full((M,),-1,dtype=torch.long)
    old_to_new[keep]=torch.arange(int(keep.sum()))

    ti=out.get("tracklet_id",torch.zeros(0,dtype=torch.long))
    node_keep=keep[ti] if len(ti) else torch.zeros(0,dtype=torch.bool)
    if len(ti):
        node_map=torch.full((len(ti),),-1,dtype=torch.long); node_map[node_keep]=torch.arange(int(node_keep.sum()))
        out["graph_x"]=out["graph_x"][node_keep]
        out["tracklet_id"]=old_to_new[ti[node_keep]]
        ei=out["graph_edge_index"]
        if ei.numel():
            ek=node_keep[ei[0]] & node_keep[ei[1]] & (torch.rand(ei.shape[1])>=edge_dropout)
            out["graph_edge_index"]=node_map[ei[:,ek]]
            out["graph_edge_attr"]=out["graph_edge_attr"][ek]
    out["temporal_ref_um"]=tref[keep].clone()
    out["temporal_status"]=out["temporal_status"][keep]
    hi=out.get("hypothesis_edge_index",torch.zeros((2,0),dtype=torch.long))
    if hi.numel():
        hk=keep[hi[0]] & keep[hi[1]] & (torch.rand(hi.shape[1])>=edge_dropout)
        out["hypothesis_edge_index"]=old_to_new[hi[:,hk]]
        out["hypothesis_edge_attr"]=out["hypothesis_edge_attr"][hk]
    # position noise
    n=len(out["temporal_ref_um"])
    if n:
        jitter=torch.randn_like(out["temporal_ref_um"])*(position_jitter_dref*dref)
        if torch.rand(())<0.1:
            jitter=jitter+torch.randn_like(jitter)*(large_jitter_dref*dref)
        out["temporal_ref_um"] += jitter
    # False hypothesis with no supporting graph nodes is valid: its status embedding still creates a token.
    if torch.rand(())<false_clue_prob:
        false_ref=(torch.rand((1,3))*2-1)*(2*dref)
        out["temporal_ref_um"]=torch.cat([out["temporal_ref_um"],false_ref],0)
        false_status=torch.zeros((1,out["temporal_status"].shape[1]),dtype=out["temporal_status"].dtype)
        false_status[0,-1]=1
        out["temporal_status"]=torch.cat([out["temporal_status"],false_status],0)
    return out
