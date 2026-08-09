from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..data.patch_sampler import PhysicalPatchSpec, centered_slices


@dataclass
class Tile:
    center_zyx: tuple[int,int,int]
    slices: tuple[slice,slice,slice]
    pads: tuple[tuple[int,int],tuple[int,int],tuple[int,int]]
    valid_min_um: np.ndarray
    valid_max_um: np.ndarray


def generate_tiles(volume_shape,spacing_um,dref_um,context_diameters=8.0,valid_diameters=6.0):
    spec=PhysicalPatchSpec(context_diameters,valid_diameters)
    patch=np.asarray(spec.voxel_shape(spacing_um,dref_um),int)
    valid=np.asarray(spec.valid_voxel_shape(spacing_um,dref_um),int)
    shape=np.asarray(volume_shape,int);spacing=np.asarray(spacing_um,float)
    # Centers spaced by valid core size; guarantee first/last coverage.
    axes=[]
    for N,v in zip(shape,valid):
        if N<=v: axes.append([N//2]);continue
        start=v//2; vals=list(range(start,N,max(1,v)))
        if vals[-1] < N-v//2-1: vals.append(N-v//2-1)
        axes.append(vals)
    for z in axes[0]:
        for y in axes[1]:
            for x in axes[2]:
                center=np.array([z,y,x])
                sl,pads=centered_slices(center,patch,shape)
                c_um=center*spacing
                half=0.5*valid*spacing
                yield Tile((int(z),int(y),int(x)),sl,pads,c_um-half,c_um+half)
