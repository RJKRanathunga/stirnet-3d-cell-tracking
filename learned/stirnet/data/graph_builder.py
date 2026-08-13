from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import torch

from .historical_instances import (
    component_overlap_from_projected_support,
    pairwise_convergence_statistics,
    projected_support_overlap,
    select_nearest_history_support,
)


DETECTION_EDGE_DIM = 15
DETECTION_EDGE_ACCEPTED_COLUMN = 14
HYPOTHESIS_EDGE_DIM = 22


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
    instance_grid: np.ndarray | torch.Tensor | None = None
    history_valid: bool = False


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
    members: dict[int,list[int]]={}
    for i in range(len(records)):
        members.setdefault(uf.find(i),[]).append(i)
    ordered_roots=sorted(
        members,
        key=lambda root:min(records[i].node_id for i in members[root]),
    )
    roots={root:index for index,root in enumerate(ordered_roots)}
    ids=[roots[uf.find(i)] for i in range(len(records))]
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
    node_instance_grid: torch.Tensor | None = None,
    node_history_valid: torch.Tensor | None = None,
    history_extent_dref: float = 2.5,
    candidate_graph_enabled: bool = True,
    max_candidate_edges: int | None = None,
    candidate_edge_chunk_size: int = 65_536,
) -> dict:
    records=list(records); associations=list(associations)
    if not records:
        return {
            "graph_x":torch.zeros((0,32),dtype=torch.float32),
            "graph_edge_index":torch.zeros((2,0),dtype=torch.long),
            "graph_edge_attr":torch.zeros((0,DETECTION_EDGE_DIM),dtype=torch.float32),
            "accepted_association_edge_index":torch.zeros((2,0),dtype=torch.long),
            "accepted_association_edge_attr":torch.zeros((0,3),dtype=torch.float32),
            "tracklet_id":torch.zeros((0,),dtype=torch.long),
            "node_ids":torch.zeros((0,),dtype=torch.long),
            "node_observed_ref_um":torch.zeros((0,3),dtype=torch.float32),
            "node_time_offset":torch.zeros((0,),dtype=torch.float32),
            "temporal_ref_um":torch.zeros((0,3),dtype=torch.float32),
            "temporal_status":torch.zeros((0,10),dtype=torch.float32),
            "hypothesis_edge_index":torch.zeros((2,0),dtype=torch.long),
            "hypothesis_edge_attr":torch.zeros((0,HYPOTHESIS_EDGE_DIM),dtype=torch.float32),
            "node_instance_grid":torch.zeros((0,4,12,12,12),dtype=torch.float16),
            "node_history_valid":torch.zeros((0,),dtype=torch.bool),
            "history_support":torch.zeros((0,2,2,12,12,12),dtype=torch.float16),
            "history_support_valid":torch.zeros((0,2),dtype=torch.bool),
            "history_support_dt":torch.zeros((0,2),dtype=torch.float32),
            "history_support_center_um":torch.zeros((0,2,3),dtype=torch.float32),
            "history_support_extent_um":torch.zeros((0,2),dtype=torch.float32),
            "best_current_component_id":torch.zeros((0,),dtype=torch.long),
            "best_component_overlap":torch.zeros((0,),dtype=torch.float32),
            "second_best_component_overlap":torch.zeros((0,),dtype=torch.float32),
        }
    id_to_idx={r.node_id:i for i,r in enumerate(records)}
    volumes=np.asarray([max(r.physical_volume_um3,1e-6) for r in records])
    med_vol=float(np.median(volumes))
    # Track lengths from provisional temporal connectivity.
    tracklet_id,M=_tracklets(records,associations)
    track_groups=[[records[i] for i in np.where(tracklet_id==m)[0]] for m in range(M)]
    if node_instance_grid is None:
        first_grid = next((r.instance_grid for r in records if r.instance_grid is not None), None)
        grid_size = int(np.asarray(first_grid).shape[-1]) if first_grid is not None else 12
        grids = []
        inferred_valid = []
        for record in records:
            if record.instance_grid is None:
                grids.append(torch.zeros((4, grid_size, grid_size, grid_size), dtype=torch.float16))
                inferred_valid.append(False)
            else:
                grid = torch.as_tensor(record.instance_grid)
                if tuple(grid.shape) != (4, grid_size, grid_size, grid_size):
                    raise ValueError("all node history grids must share [4,G,G,G]")
                grids.append(grid.to(torch.float16))
                inferred_valid.append(bool(record.history_valid))
        node_instance_grid = torch.stack(grids)
        if node_history_valid is None:
            node_history_valid = torch.tensor(inferred_valid, dtype=torch.bool)
    else:
        node_instance_grid = torch.as_tensor(node_instance_grid)
    if node_history_valid is None:
        node_history_valid = torch.ones((len(records),), dtype=torch.bool)
    node_history_valid = torch.as_tensor(node_history_valid, dtype=torch.bool)
    if node_instance_grid.shape[0] != len(records) or node_history_valid.shape != (len(records),):
        raise ValueError("node history tensors must align one-to-one with detection records")
    support_data = select_nearest_history_support(
        records,
        tracklet_id,
        node_instance_grid,
        node_history_valid,
        n_tracklets=M,
        dref_um=dref_um,
        extent_dref=history_extent_dref,
    )
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
    accepted_edges=[]; accepted_attrs=[]

    # Accepted associations remain a separate prior. They determine tracklets
    # above and only annotate candidate-GNN edges below; candidate topology
    # never feeds back into `_tracklets`.
    accepted: dict[tuple[int,int], tuple[str,float|None]] = {}
    for association in associations:
        if association.src_node_id not in id_to_idx or association.dst_node_id not in id_to_idx:
            continue
        source=id_to_idx[association.src_node_id]
        destination=id_to_idx[association.dst_node_id]
        relation="division" if association.relation=="division" else "temporal_fwd"
        accepted[(source,destination)]=(relation,association.score)
        accepted_edges.append((source,destination))
        accepted_attrs.append([
            0.0 if association.score is None else float(association.score),
            float(association.score is not None),
            float(association.relation=="division"),
        ])
        if association.relation=="temporal":
            accepted[(destination,source)]=( "temporal_rev",association.score)

    def add_edge(si,di,relation,score=None,is_accepted=False):
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
            float(is_accepted),
        ])
        edges.append((si,di))
    if candidate_graph_enabled:
        n=len(records)
        edge_count=n*(n-1)
        if max_candidate_edges is not None and edge_count>max_candidate_edges:
            raise RuntimeError(
                f"Complete candidate detection graph requires {edge_count} directed edges, "
                f"exceeding max_candidate_edges={max_candidate_edges}. Increase or disable "
                "the explicit safety limit; STIR-Net will not silently drop observations."
            )
        if candidate_edge_chunk_size<=0:
            raise ValueError("candidate_edge_chunk_size must be positive")
        # CPU chunks bound temporary construction state without changing the
        # exact all-pairs directed semantics.
        for start in range(0,n,candidate_edge_chunk_size):
            stop=min(start+candidate_edge_chunk_size,n)
            for si in range(start,stop):
                for di in range(n):
                    if si==di:
                        continue
                    relation=(
                        "temporal_fwd" if records[di].time_offset>records[si].time_offset
                        else "temporal_rev" if records[di].time_offset<records[si].time_offset
                        else "spatial"
                    )
                    accepted_relation=accepted.get((si,di))
                    score=None
                    is_accepted=accepted_relation is not None
                    if accepted_relation is not None:
                        relation,score=accepted_relation
                    add_edge(si,di,relation,score,is_accepted)
    else:
        for (source,destination),(relation,score) in accepted.items():
            add_edge(source,destination,relation,score,True)
        # Legacy accepted-graph topology retains bounded same-frame neighbours.
        for time in sorted(set(record.time_offset for record in records)):
            indices=np.asarray([i for i,record in enumerate(records) if record.time_offset==time],dtype=int)
            if len(indices)<2:continue
            positions=np.asarray([records[i].position_um for i in indices])
            distances=np.linalg.norm(positions[:,None]-positions[None],axis=-1)
            for row,source in enumerate(indices):
                count=0
                for column in np.argsort(distances[row]):
                    if column==row:continue
                    if distances[row,column]>spatial_radius_dref*dref_um:break
                    destination=int(indices[column])
                    if (int(source),destination) not in accepted:
                        add_edge(int(source),destination,"spatial",None,False)
                    count+=1
                    if count>=k_spatial_neighbors:break
    edge_index=np.asarray(edges,np.int64).T if edges else np.zeros((2,0),np.int64)
    edge_attr=np.asarray(attrs,np.float32).reshape(-1,DETECTION_EDGE_DIM)
    accepted_edge_index=(
        np.asarray(accepted_edges,np.int64).T
        if accepted_edges else np.zeros((2,0),np.int64)
    )
    accepted_edge_attr=np.asarray(accepted_attrs,np.float32).reshape(-1,3)

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

    # Projected historical support is the primary target-component association.
    comp=np.full(M,-1,np.int64)
    best_overlap=np.zeros(M,np.float32)
    second_overlap=np.zeros(M,np.float32)
    overlap_available=np.zeros(M,np.bool_)
    if current_labels is not None and spacing_um is not None:
        comp,best_overlap,second_overlap,overlap_available = component_overlap_from_projected_support(
            support_data["history_support"],
            support_data["history_support_valid"],
            refs,
            support_data["history_support_extent_um"],
            current_labels,
            spacing_um,
        )
        # Backward-compatible reference lookup only when no support was usable.
        spacing=np.asarray(spacing_um,float); center=0.5*(np.asarray(current_labels.shape)-1)*spacing
        vox=np.rint((refs+center)/spacing).astype(int)
        for m,v in enumerate(vox):
            if not overlap_available[m] and np.all(v>=0) and np.all(v<np.asarray(current_labels.shape)):
                comp[m]=int(current_labels[tuple(v)])
    track_reliability=np.ones(M,np.float32)
    for m,scores in assoc_scores.items():
        if scores: track_reliability[m]=float(np.mean(scores))
    lineage_pairs=set()
    for association in associations:
        if association.relation != "division":
            continue
        if association.src_node_id in id_to_idx and association.dst_node_id in id_to_idx:
            a=int(tracklet_id[id_to_idx[association.src_node_id]])
            b=int(tracklet_id[id_to_idx[association.dst_node_id]])
            lineage_pairs.add(tuple(sorted((a,b))))

    def representative_volume(group):
        candidates=[r for r in group if r.time_offset != 0]
        chosen=min(candidates or group,key=lambda r:abs(r.time_offset))
        return max(float(chosen.physical_volume_um3),1e-6)

    component_volumes={}
    if current_labels is not None and spacing_um is not None:
        voxel_volume=float(np.prod(np.asarray(spacing_um,float)))
        ids,counts=np.unique(current_labels[current_labels>0],return_counts=True)
        component_volumes={int(i):float(c)*voxel_volume for i,c in zip(ids,counts)}
    hedges=[];hattrs=[]
    for i in range(M):
        for j in range(i+1,M):
            delta=refs[j]-refs[i]; dist=float(np.linalg.norm(delta))
            same_component=comp[i]>0 and comp[i]==comp[j]
            same_comp_conf=(
                float(min(best_overlap[i],best_overlap[j]))
                if same_component and overlap_available[i] and overlap_available[j]
                else float(same_component)
            )
            lineage=tuple(sorted((i,j))) in lineage_pairs
            gap_related=bool(status[i,3] or status[j,3])
            convergence=pairwise_convergence_statistics(track_groups[i],track_groups[j],dref_um)
            support_overlap,support_pair_valid=projected_support_overlap(
                support_data["history_support"][i],support_data["history_support_valid"][i],refs[i],support_data["history_support_extent_um"][i],
                support_data["history_support"][j],support_data["history_support_valid"][j],refs[j],support_data["history_support_extent_um"][j],
            )
            component_valid=bool(overlap_available[i] and overlap_available[j])
            current_volume=component_volumes.get(int(comp[i]),0.0) if same_component else 0.0
            volume_ratio=(representative_volume(track_groups[i])+representative_volume(track_groups[j]))/current_volume if current_volume>0 else 0.0
            combined_reliability=float(np.sqrt(track_reliability[i]*track_reliability[j]))
            attr=[
                *(delta/dref_um),dist/dref_um,same_comp_conf,float(lineage),float(gap_related),combined_reliability,
                float(convergence["nearest_distance"]),float(convergence["older_distance"]),
                float(convergence["distance_change"]),float(convergence["closing_speed"]),
                *np.asarray(convergence["relative_velocity"],np.float32),support_overlap,
                float(best_overlap[i]),float(best_overlap[j]),float(volume_ratio),
                float(convergence["nearest_valid"]),float(convergence["older_valid"]),
                float(component_valid and support_pair_valid),
            ]
            if dist<=spatial_radius_dref*dref_um or same_component or gap_related or convergence["nearest_valid"]:
                reverse=list(attr)
                reverse[0:3]=[-v for v in attr[0:3]]
                reverse[12:15]=[-v for v in attr[12:15]]
                reverse[16],reverse[17]=attr[17],attr[16]
                hedges.extend([(i,j),(j,i)]);hattrs.extend([attr,reverse])
    hidx=np.asarray(hedges,np.int64).T if hedges else np.zeros((2,0),np.int64)
    hattr=np.asarray(hattrs,np.float32).reshape(-1,HYPOTHESIS_EDGE_DIM)
    return {
        "graph_x":torch.as_tensor(gx),
        "graph_edge_index":torch.as_tensor(edge_index,dtype=torch.long),
        "graph_edge_attr":torch.as_tensor(edge_attr),
        "accepted_association_edge_index":torch.as_tensor(accepted_edge_index,dtype=torch.long),
        "accepted_association_edge_attr":torch.as_tensor(accepted_edge_attr),
        "tracklet_id":torch.as_tensor(tracklet_id,dtype=torch.long),
        "node_ids":torch.tensor([record.node_id for record in records],dtype=torch.long),
        "node_observed_ref_um":torch.tensor(
            [record.position_um for record in records],dtype=torch.float32
        ),
        "node_time_offset":torch.tensor(
            [record.time_offset for record in records],dtype=torch.float32
        ),
        "temporal_ref_um":torch.as_tensor(refs),
        "temporal_status":torch.as_tensor(status),
        "hypothesis_edge_index":torch.as_tensor(hidx,dtype=torch.long),
        "hypothesis_edge_attr":torch.as_tensor(hattr),
        "node_instance_grid":node_instance_grid.to(torch.float16),
        "node_history_valid":node_history_valid,
        **support_data,
        "best_current_component_id":torch.as_tensor(comp,dtype=torch.long),
        "best_component_overlap":torch.as_tensor(best_overlap,dtype=torch.float32),
        "second_best_component_overlap":torch.as_tensor(second_overlap,dtype=torch.float32),
    }
