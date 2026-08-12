from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .targets import (
    add_source_gt_compatibility,
    build_gt_targets,
    estimate_dref_um,
    extract_instance_metadata,
    make_instance_boundary,
)


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
        s=torch.load(self.files[index],map_location="cpu",weights_only=False)
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
                s["target"] = add_source_gt_compatibility(
                    s["target"], s["instance_labels"]
                )
            return s
        raw=np.asarray(s["raw"],np.float32)
        labels=np.asarray(s["instance_labels"],np.int64)
        gt=np.asarray(s["gt_labels"],np.int64)
        spacing=tuple(float(x) for x in s["spacing_um"])
        dref=float(s.get("dref_um",estimate_dref_um(gt,spacing)))
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
            "instance_ids":meta.ids,
            "instance_features":meta.features,
            "instance_centroids_um":meta.centroids_um,
            "target":target,
        })
        return s

    def __getitem__(self,index):
        s=torch.load(self.files[index],map_location="cpu",weights_only=False)
        s=self._materialize(s)
        if self.transform is not None:s=self.transform(s)
        return s
