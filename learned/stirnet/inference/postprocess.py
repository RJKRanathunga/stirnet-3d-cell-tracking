from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage as ndi

from ..model.types import StirNetOutput


def _component_near_center(mask: np.ndarray, center_vox: np.ndarray, min_voxels: int) -> np.ndarray:
    cc,n=ndi.label(mask)
    if n==0:return np.zeros_like(mask,bool)
    center=np.rint(center_vox).astype(int)
    if np.all(center>=0) and np.all(center<np.asarray(mask.shape)):
        lab=cc[tuple(center)]
        if lab>0 and np.count_nonzero(cc==lab)>=min_voxels:return cc==lab
    best=None;best_d=np.inf
    for lab in range(1,n+1):
        pts=np.argwhere(cc==lab)
        if len(pts)<min_voxels:continue
        d=np.min(np.linalg.norm(pts-center[None],axis=1))
        if d<best_d:best_d=d;best=lab
    return cc==best if best is not None else np.zeros_like(mask,bool)


@torch.no_grad()
def postprocess_batch(model,outputs:StirNetOutput,render_threshold=0.30,final_exist_threshold=0.50,mask_threshold=0.50,min_mask_voxels=8,valid_min_rel_um=None,valid_max_rel_um=None):
    probs=torch.sigmoid(outputs.exist_logits).masked_fill(outputs.query_padding_mask,0)
    selected=[torch.nonzero(probs[b]>render_threshold,as_tuple=False).flatten() for b in range(probs.shape[0])]
    rendered=model.render_masks(outputs,selected)
    results=[]
    for b,idx in enumerate(selected):
        shape=outputs.instance_labels.shape[-3:]
        accepted=[]
        spacing=outputs.spacing_um[b].detach().cpu().numpy();dref=float(outputs.dref_um[b].item())
        extent=(np.asarray(shape)-1)*spacing
        for local,qi in enumerate(idx.tolist()):
            ep=float(probs[b,qi].item())
            if ep<final_exist_threshold:continue
            center_rel=(outputs.centers_cellscale[b,qi]*outputs.dref_um[b]).detach().cpu().numpy()
            if valid_min_rel_um is not None and valid_max_rel_um is not None:
                vmin=np.asarray(valid_min_rel_um[b]); vmax=np.asarray(valid_max_rel_um[b])
                if np.any(center_rel < vmin) or np.any(center_rel > vmax):
                    continue
            mp=torch.sigmoid(rendered[b][local]).detach().cpu().numpy()
            mask=mp>mask_threshold
            center_vox=(center_rel+0.5*extent)/spacing
            mask=_component_near_center(mask,center_vox,min_mask_voxels)
            if mask.sum()<min_mask_voxels:continue
            accepted.append((qi,ep,mp,mask))
        labels=np.zeros(shape,np.int32)
        if accepted:
            scores=np.stack([ep*mp for _,ep,mp,_ in accepted],axis=0)
            valid=np.stack([m for *_,m in accepted],axis=0)
            scores=np.where(valid,scores,-np.inf)
            winner=np.argmax(scores,axis=0);best=np.max(scores,axis=0)
            for i in range(len(accepted)):
                labels[(winner==i)&np.isfinite(best)&(best>0)]=i+1
        results.append({"labels":labels,"accepted_queries":[q for q,_,_,_ in accepted],"exist_probs":[p for _,p,_,_ in accepted]})
    return results
