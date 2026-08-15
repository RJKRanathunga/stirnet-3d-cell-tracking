from __future__ import annotations

from pathlib import Path
import copy
import torch


def migrate_stirnet_checkpoint_config(config: dict | None) -> tuple[dict | None, list[str]]:
    """Migrate serialized V1 configuration names without changing the model version."""
    if config is None or not isinstance(config, dict):
        return config, []
    migrated = copy.deepcopy(config)
    notes: list[str] = []
    decoder = migrated.get("decoder")
    if (
        isinstance(decoder, dict)
        and "proposal_center_step_dref" in decoder
        and "proposal_center_max_offset_dref" not in decoder
    ):
        decoder["proposal_center_max_offset_dref"] = decoder.pop(
            "proposal_center_step_dref"
        )
        notes.append(
            "renamed decoder.proposal_center_step_dref to "
            "decoder.proposal_center_max_offset_dref"
        )
    return migrated, notes


def migrate_history_checkpoint_state_dict(model, state_dict: dict) -> tuple[dict, list[str]]:
    """Migrate earlier V1 weights into the current additive V1 architecture.

    The legacy eight hypothesis-edge semantics are the unchanged prefix of the
    22-D schema. The legacy 14 detection-edge semantics are likewise the exact
    prefix of the 15-D candidate-edge schema. New columns are initialized to
    zero; newly introduced history/memory and spatial-proposal modules retain
    the receiving model's conservative initialization.
    """
    current=model.state_dict()
    migrated=dict(state_dict)
    notes=[]
    edge_suffix="hyp_graph.block.attn.edge.weight"
    for key,target in current.items():
        source=migrated.get(key)
        if (
            key == "query_builder.type_embedding.weight"
            and source is not None
            and source.ndim == 2
            and source.shape[0] == 4
            and target.shape == (5, source.shape[1])
        ):
            value = target.clone()
            value[:4] = source.to(value.dtype)
            value[4] = 0.5 * (
                source[0].to(value.dtype) + source[1].to(value.dtype)
            )
            migrated[key] = value
            notes.append(f"expanded {key}: 4 -> 5 query types")
            continue
        if (
            source is not None
            and source.shape != target.shape
            and key.startswith("graph_encoder.layers.")
            and key.endswith("attn.edge.weight")
            and source.ndim == 2
            and source.shape[0] == target.shape[0]
            and source.shape[1] == 14
            and target.shape[1] == 15
        ):
            value=torch.zeros_like(target)
            value[:,:14]=source.to(value.dtype)
            migrated[key]=value
            notes.append(f"expanded {key}: 14 -> 15 edge inputs")
            continue
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
            or ".temporal_fusion." in key
            or key.startswith("spatial_proposal_generator.")
            or key.startswith("query_builder.proposal_proj.")
            or key.startswith("query_builder.proposal_score_proj.")
            or key.startswith("query_builder.component_context_gate.")
            or key.startswith("local_mask_decoder.")
        ):
            migrated[key]=target.clone()
            notes.append(f"initialized new STIR-Net V1 parameter {key}")
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


def _optimizer_structure_matches(optimizer, state: dict) -> bool:
    current=optimizer.state_dict()
    old_groups=state.get("param_groups",[])
    new_groups=current.get("param_groups",[])
    return len(old_groups)==len(new_groups) and all(
        len(old.get("params",[]))==len(new.get("params",[]))
        for old,new in zip(old_groups,new_groups)
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu", strict=True, migrate_history=True):
    ckpt=torch.load(path,map_location=map_location,weights_only=False)
    state=ckpt["model"]
    if migrate_history:
        state,notes=migrate_history_checkpoint_state_dict(model,state)
        migrated_config, config_notes = migrate_stirnet_checkpoint_config(
            ckpt.get("config")
        )
        if "config" in ckpt:
            ckpt["config"] = migrated_config
        notes.extend(config_notes)
        if notes:
            ckpt["history_migration"]=notes
            ckpt["temporal_migration"]=notes
            ckpt["model_migration"]=notes
    model.load_state_dict(state,strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        if not _optimizer_structure_matches(optimizer,ckpt["optimizer"]):
            raise ValueError(
                "Checkpoint optimizer state is incompatible with the current STIR-Net "
                "parameter groups (including spatial-proposal and local-mask groups). The model state "
                "was migrated, but optimizer state cannot be mapped safely; load without an "
                "optimizer and start a new one."
            )
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
    return ckpt
