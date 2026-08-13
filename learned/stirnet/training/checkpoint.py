from __future__ import annotations

from pathlib import Path
import torch


def migrate_history_checkpoint_state_dict(model, state_dict: dict) -> tuple[dict, list[str]]:
    """Migrate pre-history V1 weights without discarding the hypothesis GNN.

    The legacy eight hypothesis-edge semantics are the unchanged prefix of the
    22-D schema. New columns are initialized to zero; newly introduced history
    modules retain the receiving model's conservative initialization.
    """
    current=model.state_dict()
    migrated=dict(state_dict)
    notes=[]
    edge_suffix="hyp_graph.block.attn.edge.weight"
    for key,target in current.items():
        source=migrated.get(key)
        if source is not None and source.shape!=target.shape and key.endswith(edge_suffix):
            if source.ndim==2 and source.shape[0]==target.shape[0] and source.shape[1]==8 and target.shape[1]==22:
                value=torch.zeros_like(target)
                value[:,:8]=source.to(value.dtype)
                migrated[key]=value
                notes.append(f"expanded {key}: 8 -> 22 edge inputs")
                continue
        if source is None and (
            key.startswith("history_encoder.")
            or key.startswith("history_fusion.")
            or ".cross.history_bias." in key
        ):
            migrated[key]=target.clone()
            notes.append(f"initialized new history parameter {key}")
    return migrated,notes


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


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu", strict=True, migrate_history=True):
    ckpt=torch.load(path,map_location=map_location,weights_only=False)
    state=ckpt["model"]
    if migrate_history:
        state,notes=migrate_history_checkpoint_state_dict(model,state)
        if notes: ckpt["history_migration"]=notes
    model.load_state_dict(state,strict=strict)
    if optimizer is not None and "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
    return ckpt
