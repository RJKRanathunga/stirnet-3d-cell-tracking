"""CUDA marginal-memory profile for the STIR-Net history ablation ladder."""
from __future__ import annotations

import copy
import json

import torch

from .history_overfit import _tiny_config, build_converging_merge_batch
from ...model import StirNet
from ...training.trainer import model_forward_from_batch, move_batch_to_device


def run() -> dict:
    if not torch.cuda.is_available():
        return {"available":False,"reason":"CUDA is unavailable"}
    device=torch.device("cuda");source=build_converging_merge_batch();report={"available":True,"trials":{}}
    trials={"baseline":(False,False,True),"enriched_graph":(False,False,False),"history_encoder":(True,False,False),"history_support_bias":(True,True,False)}
    for name,(history,bias,zero_dynamics) in trials.items():
        cfg=_tiny_config();cfg.history.enabled=history;cfg.history.attention_bias_enabled=bias
        batch=copy.deepcopy(source)
        if zero_dynamics: batch["hypothesis_edge_attr"][:,8:]=0
        batch=move_batch_to_device(batch,device);model=StirNet(cfg).to(device).train()
        shapes=[];handle=model.encoder.register_forward_hook(lambda _m,_i,o:shapes.extend(tuple(f.shape) for f in o.features))
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
        output=model_forward_from_batch(model,batch);output.exist_logits.square().mean().backward();torch.cuda.synchronize();handle.remove()
        report["trials"][name]={"max_memory_allocated":torch.cuda.max_memory_allocated(device),"max_memory_reserved":torch.cuda.max_memory_reserved(device),
            "N_nodes":len(batch["graph_x"]),"N_hypotheses":len(batch["temporal_ref_um"]),"history_tensor_bytes":batch["node_instance_grid"].numel()*batch["node_instance_grid"].element_size(),
            "support_tensor_bytes":batch["history_support"].numel()*batch["history_support"].element_size(),"query_count":int((~output.query_padding_mask).sum()),"spatial_feature_shapes":shapes}
        del output,model,batch
    return report


if __name__=="__main__": print(json.dumps(run(),indent=2))
