from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import torch


@dataclass
class DetectionRecord:
    node_id: int
    time_offset: int
    position_um: tuple[float,float,float]  # relative to target patch center, zyx
    physical_volume_um3: float
    bbox_um: tuple[float,float,float] = (0,0,0)
    pca_axes_um: tuple[float,float,float] = (0,0,0)
    # Canonical model-ready values: log1p(non-negative PCA axis ratio), using
    # acquisition resolution as the denominator floor. The standard
    # extract_instance_metadata() path produces exactly this representation.
    elongation: float = 1.0
    flatness: float = 1.0
    solidity: float = 1.0
    compactness: float = 1.0
    intensity_mean: float = 0.0
    intensity_std: float = 0.0
    backward_velocity_um: tuple[float,float,float] = (0,0,0)
    forward_velocity_um: tuple[float,float,float] = (0,0,0)
    distance_to_volume_boundary_um: float = 1e6
    distance_to_patch_boundary_um: float = 1e6
    boundary_related: bool = False


@dataclass
class AssociationRecord:
    src_node_id: int
    dst_node_id: int
    score: float | None = None
    relation: Literal["temporal","division"] = "temporal"


class _UnionFind:
    def __init__(self,n): self.p=list(range(n))
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a,b):
        a=self.find(a); b=self.find(b)
        if a!=b:self.p[b]=a


def _tracklets(records: list[DetectionRecord], associations: list[AssociationRecord]):
    id_to_idx={r.node_id:i for i,r in enumerate(records)}
    indeg=[0]*len(records); outdeg=[0]*len(records)
    temporal=[]
    for a in associations:
        if a.relation!="temporal" or a.src_node_id not in id_to_idx or a.dst_node_id not in id_to_idx: continue
        s=id_to_idx[a.src_node_id]; d=id_to_idx[a.dst_node_id]
        outdeg[s]+=1; indeg[d]+=1; temporal.append((s,d))
    uf=_UnionFind(len(records))
    for s,d in temporal:
        if outdeg[s]==1 and indeg[d]==1:
            uf.union(s,d)
    roots={}; ids=[]
    for i in range(len(records)):
        root=uf.find(i)
        if root not in roots: roots[root]=len(roots)
        ids.append(roots[root])
    return np.asarray(ids,np.int64),len(roots)


def _reference_for_tracklet(track: list[DetectionRecord]) -> np.ndarray:
    track=sorted(track,key=lambda r:r.time_offset)
    by_t={r.time_offset:r for r in track}
    if 0 in by_t:return np.asarray(by_t[0].position_um,np.float32)
    past=[r for r in track if r.time_offset<0]; future=[r for r in track if r.time_offset>0]
    if past and future:
        a=max(past,key=lambda r:r.time_offset); b=min(future,key=lambda r:r.time_offset)
        f=(0-a.time_offset)/(b.time_offset-a.time_offset)
        return np.asarray(a.position_um)*(1-f)+np.asarray(b.position_um)*f
    if len(past)>=2:
        a,b=sorted(past,key=lambda r:r.time_offset)[-2:]
        v=(np.asarray(b.position_um)-np.asarray(a.position_um))/(b.time_offset-a.time_offset)
        return np.asarray(b.position_um)+v*(0-b.time_offset)
    if past:return np.asarray(past[-1].position_um,np.float32)
    if len(future)>=2:
        a,b=sorted(future,key=lambda r:r.time_offset)[:2]
        v=(np.asarray(b.position_um)-np.asarray(a.position_um))/(b.time_offset-a.time_offset)
        return np.asarray(a.position_um)-v*a.time_offset
    return np.asarray(future[0].position_um,np.float32)


def build_temporal_graph(
    records: Iterable[DetectionRecord],
    associations: Iterable[AssociationRecord],
    *,
    dref_um: float,
    temporal_radius: int = 2,
    k_spatial_neighbors: int = 6,
    spatial_radius_dref: float = 2.5,
    current_labels: np.ndarray | None = None,
    spacing_um: tuple[float,float,float] | None = None,
) -> dict:
    records=list(records); associations=list(associations)
    if not records:
        return {
            "graph_x":torch.zeros((0,32),dtype=torch.float32),
            "graph_edge_index":torch.zeros((2,0),dtype=torch.long),
            "graph_edge_attr":torch.zeros((0,14),dtype=torch.float32),
            "tracklet_id":torch.zeros((0,),dtype=torch.long),
            "temporal_ref_um":torch.zeros((0,3),dtype=torch.float32),
            "temporal_status":torch.zeros((0,10),dtype=torch.float32),
            "hypothesis_edge_index":torch.zeros((2,0),dtype=torch.long),
            "hypothesis_edge_attr":torch.zeros((0,8),dtype=torch.float32),
        }
    id_to_idx={r.node_id:i for i,r in enumerate(records)}
    volumes=np.asarray([max(r.physical_volume_um3,1e-6) for r in records])
    med_vol=float(np.median(volumes))
    # Track lengths from provisional temporal connectivity.
    tracklet_id,M=_tracklets(records,associations)
    track_groups=[[records[i] for i in np.where(tracklet_id==m)[0]] for m in range(M)]
    lengths_before=np.zeros(len(records)); lengths_after=np.zeros(len(records))
    for group in track_groups:
        ts=sorted(r.time_offset for r in group)
        for r in group:
            i=id_to_idx[r.node_id]; lengths_before[i]=sum(t<=r.time_offset for t in ts); lengths_after[i]=sum(t>=r.time_offset for t in ts)
    division_nodes=set()
    for a in associations:
        if a.relation=="division": division_nodes.update([a.src_node_id,a.dst_node_id])
    # Interior start/end from tracklet extent, excluding explicitly boundary-related records.
    gx=[]
    for i,r in enumerate(records):
        group=track_groups[tracklet_id[i]]; ts=[g.time_offset for g in group]
        interior_start=(r.time_offset==min(ts) and min(ts)>-temporal_radius and not r.boundary_related)
        interior_end=(r.time_offset==max(ts) and max(ts)<temporal_radius and not r.boundary_related)
        gx.append([
            r.time_offset/max(temporal_radius,1),
            *(np.asarray(r.position_um)/dref_um),
            np.log(max(r.physical_volume_um3,1e-6)/max(med_vol,1e-6)),
            *(np.asarray(r.bbox_um)/dref_um),
            *(np.asarray(r.pca_axes_um)/dref_um),
            r.elongation,r.flatness,r.solidity,r.compactness,
            r.intensity_mean,r.intensity_std,
            *(np.asarray(r.backward_velocity_um)/dref_um),
            *(np.asarray(r.forward_velocity_um)/dref_um),
            lengths_before[i],lengths_after[i],
            r.distance_to_volume_boundary_um/dref_um,
            r.distance_to_patch_boundary_um/dref_um,
            float(r.time_offset==0),float(interior_start),float(interior_end),
            float(r.node_id in division_nodes),float(r.boundary_related),
        ])
    gx=np.asarray(gx,np.float32)
    assert gx.shape[1]==32

    edges=[]; attrs=[]
    def add_edge(si,di,relation,score=None):
        s=records[si]; d=records[di]
        delta=np.asarray(d.position_um)-np.asarray(s.position_um)
        dist=float(np.linalg.norm(delta))
        motion=np.asarray(s.forward_velocity_um)
        residual=float(np.linalg.norm(delta-motion*max(d.time_offset-s.time_offset,1)))
        rel_types=[0,0,0,0]
        rel_types[{"temporal_fwd":0,"temporal_rev":1,"division":2,"spatial":3}[relation]]=1
        attrs.append([
            d.time_offset-s.time_offset,* (delta/dref_um),dist/dref_um,
            np.log(max(d.physical_volume_um3,1e-6)/max(s.physical_volume_um3,1e-6)),
            d.intensity_mean-s.intensity_mean,residual/dref_um,
            0.0 if score is None else float(score),float(score is not None),*rel_types,
        ])
        edges.append((si,di))
    for a in associations:
        if a.src_node_id not in id_to_idx or a.dst_node_id not in id_to_idx:continue
        s=id_to_idx[a.src_node_id]; d=id_to_idx[a.dst_node_id]
        if a.relation=="division": add_edge(s,d,"division",a.score)
        else:
            add_edge(s,d,"temporal_fwd",a.score); add_edge(d,s,"temporal_rev",a.score)
    # same-frame spatial edges
    for t in sorted(set(r.time_offset for r in records)):
        idx=np.asarray([i for i,r in enumerate(records) if r.time_offset==t],dtype=int)
        if len(idx)<2:continue
        pos=np.asarray([records[i].position_um for i in idx])
        D=np.linalg.norm(pos[:,None]-pos[None],axis=-1)
        for a,i in enumerate(idx):
            order=np.argsort(D[a])
            count=0
            for b in order:
                if b==a:continue
                if D[a,b]>spatial_radius_dref*dref_um:break
                add_edge(i,int(idx[b]),"spatial",None);count+=1
                if count>=k_spatial_neighbors:break
    edge_index=np.asarray(edges,np.int64).T if edges else np.zeros((2,0),np.int64)
    edge_attr=np.asarray(attrs,np.float32).reshape(-1,14)

    refs=np.stack([_reference_for_tracklet(g) for g in track_groups]).astype(np.float32)
    status=np.zeros((M,10),np.float32)
    assoc_scores={m:[] for m in range(M)}
    for a in associations:
        if a.src_node_id in id_to_idx:
            m=tracklet_id[id_to_idx[a.src_node_id]]
            if a.score is not None:assoc_scores[m].append(a.score)
    for m,g in enumerate(track_groups):
        ts=sorted(r.time_offset for r in g)
        gaps=any((b-a)>1 for a,b in zip(ts[:-1],ts[1:]))
        boundary=any(r.boundary_related for r in g)
        division=any(r.node_id in division_nodes for r in g)
        interior_start=min(ts)>-temporal_radius and not boundary
        interior_end=max(ts)<temporal_radius and not boundary
        complete=(min(ts)<=-temporal_radius and max(ts)>=temporal_radius and not gaps)
        uncertain=bool(assoc_scores[m] and np.mean(assoc_scores[m])<0.5)
        status[m]=[complete,interior_start,interior_end,gaps,division,boundary,
                   min(ts)<=-temporal_radius,max(ts)>=temporal_radius,0.0,uncertain]

    # Hypothesis component id at target frame if current labels are provided.
    comp=np.full(M,-1,np.int64)
    if current_labels is not None and spacing_um is not None:
        spacing=np.asarray(spacing_um,float); center=0.5*(np.asarray(current_labels.shape)-1)*spacing
        vox=np.rint((refs+center)/spacing).astype(int)
        for m,v in enumerate(vox):
            if np.all(v>=0) and np.all(v<np.asarray(current_labels.shape)):
                comp[m]=int(current_labels[tuple(v)])
    hedges=[];hattrs=[]
    for i in range(M):
        for j in range(i+1,M):
            delta=refs[j]-refs[i]; dist=float(np.linalg.norm(delta))
            same_comp=comp[i]>0 and comp[i]==comp[j]
            lineage=False
            gap_related=bool(status[i,3] or status[j,3])
            if dist<=spatial_radius_dref*dref_um or same_comp or gap_related:
                attr=[*(delta/dref_um),dist/dref_um,float(same_comp),float(lineage),float(gap_related),1.0]
                hedges.extend([(i,j),(j,i)]);hattrs.extend([attr,[-attr[0],-attr[1],-attr[2],attr[3],attr[4],attr[5],attr[6],attr[7]]])
    hidx=np.asarray(hedges,np.int64).T if hedges else np.zeros((2,0),np.int64)
    hattr=np.asarray(hattrs,np.float32).reshape(-1,8)
    return {
        "graph_x":torch.as_tensor(gx),
        "graph_edge_index":torch.as_tensor(edge_index,dtype=torch.long),
        "graph_edge_attr":torch.as_tensor(edge_attr),
        "tracklet_id":torch.as_tensor(tracklet_id,dtype=torch.long),
        "temporal_ref_um":torch.as_tensor(refs),
        "temporal_status":torch.as_tensor(status),
        "hypothesis_edge_index":torch.as_tensor(hidx,dtype=torch.long),
        "hypothesis_edge_attr":torch.as_tensor(hattr),
    }
