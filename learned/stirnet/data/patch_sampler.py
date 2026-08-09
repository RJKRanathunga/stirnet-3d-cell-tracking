from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PhysicalPatchSpec:
    context_diameters: float = 8.0
    valid_diameters: float = 6.0

    def voxel_shape(self, spacing_um, dref_um: float) -> tuple[int,int,int]:
        length = self.context_diameters * dref_um
        return tuple(max(4, int(np.ceil(length / float(s)))) for s in spacing_um)

    def valid_voxel_shape(self, spacing_um, dref_um: float) -> tuple[int,int,int]:
        length = self.valid_diameters * dref_um
        return tuple(max(2, int(np.floor(length / float(s)))) for s in spacing_um)


def centered_slices(center_zyx, patch_shape, volume_shape):
    starts=[]; ends=[]; pads=[]
    for c,n,N in zip(center_zyx,patch_shape,volume_shape):
        start=int(round(c))-n//2; end=start+n
        left=max(0,-start); right=max(0,end-N)
        starts.append(max(0,start)); ends.append(min(N,end)); pads.append((left,right))
    return tuple(slice(a,b) for a,b in zip(starts,ends)), tuple(pads)


def extract_padded(volume: np.ndarray, slices, pads, constant=0):
    crop=volume[slices]
    return np.pad(crop,pads,mode="constant",constant_values=constant)
