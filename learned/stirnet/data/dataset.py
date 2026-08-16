from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .targets import (
    add_source_gt_compatibility,
    build_gt_targets,
    estimate_model_dref_um,
    extract_instance_metadata,
    make_internal_boundary_target,
    make_instance_boundary,
)
from .trackastra_cache import load_cache
from .sample_builder import renormalize_cached_dref


class CachedStirNetDataset(Dataset):
    """Dataset for pre-built STIR-Net patch caches.

    Each `.pt` file may already contain all tensors. If native arrays are present instead,
    `_materialize` derives the standard five spatial channels and instance metadata.
    """
    def __init__(self, files: Sequence[str | Path], transform: Callable[[dict],dict] | None = None):
        self.files=[Path(f) for f in files]
        self.transform=transform

    def __len__(self): return len(self.files)

    def shape_key(self,index):
        s=load_cache(self.files[index],map_location="cpu")
        if "spatial_inputs" in s:
            shape=tuple(s["spatial_inputs"].shape[-3:])
        elif "raw" in s:
            shape=tuple(np.asarray(s["raw"]).shape[-3:])
        else:
            shape=()
        spacing=tuple(round(float(v),5) for v in s.get("spacing_um",(0,0,0)))
        return spacing,shape

    def _materialize(self,s:dict)->dict:
        if "spatial_inputs" in s:
            if "target" in s and "instance_labels" in s:
                s = dict(s)
                metadata = dict(s.get("metadata", {}))
                source = metadata.get("model_dref_source")
                trusted_sources = {
                    "current_segmentation",
                    "fixed_acquisition_prior",
                    "explicit_non_gt",
                }
                if source not in trusted_sources:
                    spacing = tuple(float(value) for value in s["spacing_um"])
                    model_dref = estimate_model_dref_um(
                        torch.as_tensor(s["instance_labels"]).cpu().numpy(), spacing
                    )
                    cached_dref = float(torch.as_tensor(s.get("dref_um", model_dref)))
                    s = renormalize_cached_dref(
                        s,
                        cached_dref_um=cached_dref,
                        model_dref_um=model_dref,
                    )
                    target = dict(s["target"])
                    if "centers_um" in target:
                        target["centers_cellscale"] = (
                            torch.as_tensor(target["centers_um"]).float()
                            / max(model_dref, 1e-6)
                        )
                    s["target"] = target
                target = add_source_gt_compatibility(
                    s["target"], s["instance_labels"]
                )
                if "internal_boundary" not in target and "label_map" in target:
                    target = dict(target)
                    target["internal_boundary"] = torch.as_tensor(
                        make_internal_boundary_target(
                            torch.as_tensor(target["label_map"]).cpu().numpy(),
                            tuple(float(value) for value in s["spacing_um"]),
                        )
                    )
                s["target"] = target
            return s
        raw=np.asarray(s["raw"],np.float32)
        labels=np.asarray(s["instance_labels"],np.int64)
        gt=np.asarray(s["gt_labels"],np.int64)
        spacing=tuple(float(x) for x in s["spacing_um"])
        dref=float(s.get("dref_um",estimate_model_dref_um(labels,spacing)))
        foreground=(labels>0).astype(np.float32)
        from scipy import ndimage as ndi
        edt=np.zeros_like(raw,np.float32)
        for label in np.unique(labels):
            if label<=0:continue
            m=labels==label
            edt[m]=ndi.distance_transform_edt(m,sampling=spacing)[m]/max(dref,1e-6)
        boundary=make_instance_boundary(labels).astype(np.float32)
        marker=np.asarray(s.get("marker_heatmap",np.zeros_like(raw)),np.float32)
        spatial=np.stack([raw,foreground,edt,boundary,marker])
        meta=extract_instance_metadata(labels,raw,spacing,dref,marker)
        target=build_gt_targets(gt,spacing,dref,current_labels=labels)
        s=dict(s)
        s.update({
            "spatial_inputs":torch.as_tensor(spatial),
            "instance_labels":torch.as_tensor(labels,dtype=torch.long),
            "spacing_um":torch.tensor(spacing,dtype=torch.float32),
            "dref_um":torch.tensor(dref,dtype=torch.float32),
            "metadata": {
                **dict(s.get("metadata", {})),
                "model_dref_um": dref,
                "model_dref_source": (
                    dict(s.get("metadata", {})).get("model_dref_source")
                    or ("explicit_non_gt" if "dref_um" in s else "current_segmentation")
                ),
            },
            "instance_ids":meta.ids,
            "instance_features":meta.features,
            "instance_centroids_um":meta.centroids_um,
            "target":target,
        })
        return s

    def __getitem__(self,index):
        s=load_cache(self.files[index],map_location="cpu")
        s=self._materialize(s)
        if self.transform is not None:s=self.transform(s)
        return s
