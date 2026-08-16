from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage as ndi
from torch import Tensor

from ..model.instances.tokenizer import centers_from_labels
from ..model.types import StirNetOutput


@dataclass
class PostprocessConfig:
    min_object_voxels: int = 4


def _valid_region_mask(
    shape: tuple[int, int, int],
    spacing_um: np.ndarray,
    valid_min_rel_um,
    valid_max_rel_um,
) -> np.ndarray:
    axes = [
        np.arange(size, dtype=np.float32) * spacing_um[axis]
        - 0.5 * (size - 1) * spacing_um[axis]
        for axis, size in enumerate(shape)
    ]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    coordinates = np.stack([zz, yy, xx], axis=-1)
    minimum = np.asarray(valid_min_rel_um, dtype=np.float32)
    maximum = np.asarray(valid_max_rel_um, dtype=np.float32)
    return np.all((coordinates >= minimum) & (coordinates <= maximum), axis=-1)


def postprocess_labels(
    labels: Tensor,
    spacing_um: Tensor,
    *,
    min_object_voxels: int = 4,
    valid_min_rel_um=None,
    valid_max_rel_um=None,
) -> Tensor:
    """Crop, split disconnected remnants, reject tiny objects, and relabel."""
    array = labels.detach().cpu().numpy().astype(np.int64, copy=True)
    if (valid_min_rel_um is None) != (valid_max_rel_um is None):
        raise ValueError("valid_min_rel_um and valid_max_rel_um must be provided together")
    if valid_min_rel_um is not None:
        valid = _valid_region_mask(
            tuple(array.shape),
            spacing_um.detach().cpu().numpy(),
            valid_min_rel_um,
            valid_max_rel_um,
        )
        array[~valid] = 0
    result = np.zeros_like(array, dtype=np.int64)
    next_id = 1
    connectivity = ndi.generate_binary_structure(3, 1)
    for instance_id in np.unique(array):
        if instance_id <= 0:
            continue
        components, count = ndi.label(array == instance_id, structure=connectivity)
        for component_id in range(1, count + 1):
            mask = components == component_id
            if int(mask.sum()) < min_object_voxels:
                continue
            result[mask] = next_id
            next_id += 1
    return torch.from_numpy(result).to(device=labels.device, dtype=torch.long)


@torch.no_grad()
def postprocess_batch(
    outputs: StirNetOutput,
    spacing_um: Tensor,
    *,
    config: PostprocessConfig | None = None,
    valid_min_rel_um=None,
    valid_max_rel_um=None,
) -> list[dict]:
    cfg = config or PostprocessConfig()
    results: list[dict] = []
    for batch_index, source in enumerate(outputs.final_labels):
        minimum = None if valid_min_rel_um is None else valid_min_rel_um[batch_index]
        maximum = None if valid_max_rel_um is None else valid_max_rel_um[batch_index]
        labels = postprocess_labels(
            source,
            spacing_um[batch_index],
            min_object_voxels=cfg.min_object_voxels,
            valid_min_rel_um=minimum,
            valid_max_rel_um=maximum,
        )
        centers = centers_from_labels(
            [labels],
            spacing_um[batch_index : batch_index + 1],
            outputs.geometry.sdf[batch_index : batch_index + 1],
        )[0]
        results.append(
            {
                "labels": labels.detach().cpu(),
                "centers_um": centers.detach().cpu(),
                "instance_count": int(labels.max().item()),
            }
        )
    return results


__all__ = ["PostprocessConfig", "postprocess_batch", "postprocess_labels"]
