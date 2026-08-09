from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(path, *, model, optimizer=None, scheduler=None, scaler=None, step=0, epoch=0, config=None, extra=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload={
        "model":model.state_dict(),"step":step,"epoch":epoch,
        "config":config.to_dict() if hasattr(config,"to_dict") else config,
        "extra":extra or {},
    }
    if optimizer is not None: payload["optimizer"]=optimizer.state_dict()
    if scheduler is not None: payload["scheduler"]=scheduler.state_dict()
    if scaler is not None: payload["scaler"]=scaler.state_dict()
    torch.save(payload,path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu", strict=True):
    ckpt=torch.load(path,map_location=map_location,weights_only=False)
    model.load_state_dict(ckpt["model"],strict=strict)
    if optimizer is not None and "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
    return ckpt
