from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from ..model import RefinementCriterion, StirNet, StirNetConfig
from .checkpoint import save_checkpoint


def move_to_device(x,device):
    if torch.is_tensor(x): return x.to(device,non_blocking=True)
    if isinstance(x,dict): return {k:move_to_device(v,device) for k,v in x.items()}
    if isinstance(x,list): return [move_to_device(v,device) for v in x]
    if isinstance(x,tuple): return tuple(move_to_device(v,device) for v in x)
    return x


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move model inputs while retaining potentially huge target maps on CPU."""
    return {
        key: value if key == "targets" else move_to_device(value, device)
        for key, value in batch.items()
    }


def model_forward_from_batch(model: StirNet,b:dict):
    return model(
        b["spatial_inputs"],b["instance_labels"],b["spacing_um"],b["dref_um"],
        b["instance_features"],b["instance_ids"],b["instance_batch"],b["instance_centroids_um"],
        b["graph_x"],b["graph_edge_index"],b["graph_edge_attr"],b["tracklet_id"],
        b["temporal_ref_um"],b["temporal_status"],b["hypothesis_edge_index"],
        b["hypothesis_edge_attr"],b["temporal_batch"],
        b.get("spatial_padding_mask"),
    )


class Trainer:
    def __init__(self, model: StirNet, cfg: StirNetConfig, device=None, amp_dtype: str = "fp16"):
        self.model=model
        self.cfg=cfg
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.criterion=RefinementCriterion(cfg.losses,cfg.queries).to(self.device)
        self.optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.training.lr,weight_decay=cfg.training.weight_decay)
        self.scheduler=None
        self.amp_dtype=amp_dtype
        self.scaler=torch.amp.GradScaler("cuda",enabled=self.device.type=="cuda" and amp_dtype=="fp16")
        self.global_step=0

    def _autocast(self):
        if self.device.type!="cuda": return nullcontext()
        dtype=torch.float16 if self.amp_dtype=="fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda",dtype=dtype)

    def train_step(self,batch:dict)->dict[str,float]:
        self.model.train(); b=move_batch_to_device(batch,self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            out=model_forward_from_batch(self.model,b)
            losses=self.criterion(out,b["targets"])
            loss=losses["loss"]
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg.training.max_grad_norm)
        self.scaler.step(self.optimizer);self.scaler.update()
        if self.scheduler is not None:self.scheduler.step()
        self.global_step+=1
        return {k:float(v.detach().cpu()) for k,v in losses.items()}

    @torch.no_grad()
    def eval_step(self,batch:dict)->dict[str,float]:
        self.model.eval();b=move_batch_to_device(batch,self.device)
        with self._autocast():
            out=model_forward_from_batch(self.model,b);losses=self.criterion(out,b["targets"])
        return {k:float(v.detach().cpu()) for k,v in losses.items()}

    def fit(self,train_loader,val_loader=None,epochs=1,out_dir="runs/stirnet",checkpoint_every=1):
        out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
        for epoch in range(epochs):
            sums={};n=0
            for batch in train_loader:
                m=self.train_step(batch);n+=1
                for k,v in m.items():sums[k]=sums.get(k,0.0)+v
            train_avg={k:v/max(n,1) for k,v in sums.items()}
            print(f"epoch {epoch+1}: train {train_avg}")
            if val_loader is not None:
                vs={};vn=0
                for batch in val_loader:
                    m=self.eval_step(batch);vn+=1
                    for k,v in m.items():vs[k]=vs.get(k,0.0)+v
                print(f"epoch {epoch+1}: val { {k:v/max(vn,1) for k,v in vs.items()} }")
            if (epoch+1)%checkpoint_every==0:
                save_checkpoint(out/f"epoch_{epoch+1:04d}.pt",model=self.model,optimizer=self.optimizer,
                                scheduler=self.scheduler,scaler=self.scaler,step=self.global_step,epoch=epoch+1,config=self.cfg)
