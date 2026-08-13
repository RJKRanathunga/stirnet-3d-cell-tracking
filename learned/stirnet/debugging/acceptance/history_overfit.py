"""Targeted same-sample diagnostic for compact historical instance evidence.

Run with ``python -m learned.stirnet.debugging.acceptance.history_overfit``.
This is deliberately a diagnostic, not a CI test: it trains four independent
models on one synthetic converging-track merge and reports task-level outputs.
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch
import torch.nn.functional as F

from ...data.collate import stirnet_collate
from ...data.graph_builder import AssociationRecord, DetectionRecord, build_temporal_graph
from ...data.historical_instances import build_node_instance_grids
from ...data.sample_builder import build_cached_sample
from ...model import RefinementCriterion, StirNet, StirNetConfig
from ...training.trainer import model_forward_from_batch, move_batch_to_device


def _tiny_config() -> StirNetConfig:
    cfg=StirNetConfig()
    cfg.spatial.channels=(4,8,16,32);cfg.spatial.blocks_per_level=1;cfg.spatial.mask_dim=8
    cfg.temporal.d_model=32;cfg.temporal.graph_ffn_dim=64
    cfg.coreasoning.d_model=32;cfg.coreasoning.dropout=0
    cfg.coreasoning.temporal_query_chunk_size=2;cfg.coreasoning.spatial_key_chunk_size=256
    cfg.queries.d_model=32;cfg.queries.discovery_queries=2
    cfg.decoder.d_model=32;cfg.decoder.ffn_dim=64;cfg.decoder.mask_dim=8;cfg.decoder.dropout=0;cfg.decoder.max_spatial_tokens=128
    cfg.training.activation_checkpointing=False
    return cfg


def build_converging_merge_batch() -> dict:
    shape=(8,24,24);spacing=(1.5,0.75,0.75);dref=6.0
    zz,yy,xx=np.indices(shape)
    def ball(center,radius=3.2):
        physical=(zz-center[0])**2*spacing[0]**2+(yy-center[1])**2*spacing[1]**2+(xx-center[2])**2*spacing[2]**2
        return physical<=radius**2
    gt=np.zeros(shape,np.int32)
    gt[ball((4,12,8))]=1;gt[ball((4,12,16))]=2
    current=(gt>0)|((zz>=2)&(zz<=6)&(yy>=10)&(yy<=14)&(xx>=8)&(xx<=16))
    current_labels=current.astype(np.int32)
    raw=(0.15+0.75*ndi_gaussian(gt>0,1.0)).astype(np.float32)

    records=[];associations=[];observations=[];node_id=0
    tracks=[]
    for cell, target_x in ((1,8),(2,16)):
        ids=[]
        direction=-1 if cell==1 else 1
        for time,extra in ((-2,5),(-1,2),(1,2)):
            center_vox=np.array((4,12,target_x+direction*extra),np.float32)
            labels=ball(tuple(center_vox)).astype(np.int32)*cell
            hist_raw=(0.1+0.8*ndi_gaussian(labels>0,0.8)).astype(np.float32)
            center_abs=center_vox*np.asarray(spacing)
            patch_center=0.5*(np.asarray(shape)-1)*np.asarray(spacing)
            records.append(DetectionRecord(node_id,time,tuple(center_abs-patch_center),float((labels>0).sum()*np.prod(spacing))))
            observations.append((hist_raw,labels,cell,spacing,dref,center_abs))
            ids.append(node_id);node_id+=1
        tracks.append(ids)
        associations.extend([AssociationRecord(ids[0],ids[1],0.9),AssociationRecord(ids[1],ids[2],0.8)])
    grids,valid=build_node_instance_grids(observations)
    graph=build_temporal_graph(
        records,associations,dref_um=dref,current_labels=current_labels,spacing_um=spacing,
        node_instance_grid=grids,node_history_valid=valid,
    )
    sample=build_cached_sample(raw,current_labels,gt,spacing,dref_um=dref,normalize_raw=False,temporal_graph=graph,
        metadata={"diagnostic":"converging_two_to_one"})
    return stirnet_collate([sample])


def ndi_gaussian(mask: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(mask.astype(np.float32),sigma)


def _task_metrics(model: StirNet, output, target: dict) -> dict[str,float|int]:
    probabilities=output.exist_logits[0].sigmoid().detach()
    valid=~output.query_padding_mask[0]
    selected=torch.nonzero(valid&(probabilities>=0.5),as_tuple=False).flatten()
    top=torch.topk(probabilities.masked_fill(~valid,-1),k=min(2,int(valid.sum()))).indices
    gt_centers=target["centers_cellscale"].to(output.centers_cellscale.device)
    center_error=float(torch.cdist(output.centers_cellscale[0,top],gt_centers).min(dim=1).values.mean()) if len(gt_centers) else 0.0
    gt_map=target["label_map"].to(output.coarse_mask_logits.device)
    gt_masks=torch.stack([gt_map==label for label in target["ids"].tolist()]).float()
    gt_coarse=F.interpolate(gt_masks[:,None],size=output.coarse_mask_logits.shape[-3:],mode="nearest")[:,0]
    pred=output.coarse_mask_logits[0,top].sigmoid()
    dice=(2*(pred[:,None]*gt_coarse[None]).flatten(2).sum(-1)/(pred[:,None].flatten(2).sum(-1)+gt_coarse[None].flatten(2).sum(-1)+1e-6)).amax(dim=1).mean()
    native=model.render_masks(output,[top])[0].sigmoid()
    native_dice=(2*(native[:,None]*gt_masks[None]).flatten(2).sum(-1)/(native[:,None].flatten(2).sum(-1)+gt_masks[None].flatten(2).sum(-1)+1e-6)).amax(dim=1).mean()
    return {"predicted_instance_count":int(selected.numel()),"matched_positive_proxy":int((probabilities[top]>=0.5).sum()),
            "existence_top2_mean":float(probabilities[top].mean()),"center_error_dref":center_error,
            "coarse_dice_top2":float(dice),"native_dice_top2":float(native_dice)}


def run(steps: int=200,device: str|None=None,selected: set[str]|None=None) -> dict:
    device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    source=build_converging_merge_batch();results={}
    ablations={
        "A_baseline":{"history":False,"bias":False,"zero_dynamics":True},
        "B_enriched_dynamics":{"history":False,"bias":False,"zero_dynamics":False},
        "C_history_encoder":{"history":True,"bias":False,"zero_dynamics":False},
        "D_full_history_support":{"history":True,"bias":True,"zero_dynamics":False},
    }
    for name,options in ablations.items():
        if selected is not None and name not in selected:
            continue
        torch.manual_seed(17)
        # Construct the same full module graph for every trial so common
        # downstream weights receive identical seeded initialization.
        cfg=_tiny_config()
        batch=copy.deepcopy(source)
        if options["zero_dynamics"]: batch["hypothesis_edge_attr"][:,8:]=0
        batch=move_batch_to_device(batch,device)
        model=StirNet(cfg).to(device)
        if not options["history"]:
            model.cfg.history.enabled=False
        if not options["bias"]:
            model.cr1.cross.history_bias=None;model.cr2.cross.history_bias=None
        criterion=RefinementCriterion(cfg.losses,cfg.queries,cfg.training).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3)
        first_loss=None
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True);output=model_forward_from_batch(model,batch)
            losses=criterion(output,batch["targets"]);losses["loss"].backward();optimizer.step()
            first_loss=float(losses["loss"].detach()) if first_loss is None else first_loss
        model.eval()
        with torch.no_grad(): output=model_forward_from_batch(model,batch)
        metrics=_task_metrics(model,output,batch["targets"][0]);metrics["initial_loss"]=first_loss;metrics["final_loss"]=float(criterion(output,batch["targets"])["loss"])
        results[name]=metrics
    return results


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--steps",type=int,default=200);parser.add_argument("--device",default=None)
    parser.add_argument("--ablations",nargs="*",default=None,help="Optional subset of A_baseline/B_enriched_dynamics/C_history_encoder/D_full_history_support")
    args=parser.parse_args();print(json.dumps(run(args.steps,args.device,set(args.ablations) if args.ablations else None),indent=2))


if __name__=="__main__": main()
