from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def capture_dense_arrays(outputs, *, dtype: str = "float16"):
    np_dtype = np.float16 if dtype == "float16" else np.float32
    arrays = {}
    for key, logits in outputs.dense_outputs.items():
        prob = torch.sigmoid(logits.detach().float())[0, 0]
        arrays[f"dense/{key.removesuffix('_logits')}"] = prob.cpu().numpy().astype(np_dtype, copy=False)
    return arrays


def capture_scene_arrays(batch_cpu: dict, targets: list[dict], *, dtype: str = "float16"):
    np_dtype = np.float16 if dtype == "float16" else np.float32
    spatial = batch_cpu["spatial_inputs"]
    current = batch_cpu["instance_labels"]
    raw = spatial[0, 0].detach().float().cpu().numpy() if torch.is_tensor(spatial) else np.asarray(spatial)[0, 0]
    current_np = current[0].detach().cpu().numpy() if torch.is_tensor(current) else np.asarray(current)[0]
    target_map = targets[0]["label_map"]
    gt = target_map.detach().cpu().numpy() if torch.is_tensor(target_map) else np.asarray(target_map)
    return {
        "scene/raw": raw.astype(np_dtype, copy=False),
        "scene/current_labels": current_np.astype(np.int32, copy=False),
        "scene/gt_labels": gt.astype(np.int32, copy=False),
    }
