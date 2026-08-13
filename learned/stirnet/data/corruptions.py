from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
from scipy import ndimage as ndi


@dataclass
class CorruptionResult:
    labels: np.ndarray
    kind: str
    metadata: dict


@dataclass
class SequenceCorruptionResult:
    labels: np.ndarray
    target_index: int
    kind: str
    metadata: dict


def merge_instances(labels: np.ndarray, ids: list[int]) -> CorruptionResult:
    out=labels.copy()
    ids=[int(i) for i in ids if i>0]
    if len(ids)<2: return CorruptionResult(out,"none",{})
    keep=ids[0]
    for i in ids[1:]: out[out==i]=keep
    return CorruptionResult(out,"merge",{"ids":ids,"kept":keep})


def merge_sequence_at_target(
    label_sequence: np.ndarray,
    target_index: int,
    ids: list[int] | None = None,
    *,
    require_separate_context: bool = True,
    rng: np.random.Generator | None = None,
) -> SequenceCorruptionResult:
    """Merge persistent cells only at the target frame of an annotated sequence.

    The surrounding frames remain provisional segmentation inputs; this helper
    does not construct perfect tracks or inject GT identities into model tensors.
    It is the minimal corruption primitive used before a Trackastra pass-1 cache
    is built.
    """
    sequence=np.asarray(label_sequence)
    if sequence.ndim!=4: raise ValueError("label_sequence must be [T,Z,Y,X]")
    if not 0<=target_index<sequence.shape[0]: raise IndexError("target_index out of range")
    rng=rng or np.random.default_rng()
    target=sequence[target_index]
    if ids is None:
        candidate=merge_adjacent_instances(target,rng)
        ids=list(candidate.metadata.get("ids",[]))
    ids=[int(i) for i in (ids or []) if i>0]
    if len(ids)<2:
        return SequenceCorruptionResult(sequence.copy(),target_index,"none",{})
    if require_separate_context:
        context=[t for t in range(sequence.shape[0]) if t!=target_index]
        if not any(all(np.any(sequence[t]==i) for i in ids) for t in context):
            return SequenceCorruptionResult(sequence.copy(),target_index,"none",{"reason":"missing_separate_context"})
    out=sequence.copy()
    merged=merge_instances(out[target_index],ids)
    out[target_index]=merged.labels
    return SequenceCorruptionResult(
        out,target_index,"sequence_merge",
        {"ids":ids,"kept":ids[0],"past_available":target_index>0,"future_available":target_index+1<sequence.shape[0]},
    )


def remove_instance(labels: np.ndarray, label: int) -> CorruptionResult:
    out=labels.copy(); out[out==label]=0
    return CorruptionResult(out,"missing",{"id":int(label)})


def erode_instance(labels: np.ndarray, label: int, iterations: int = 1) -> CorruptionResult:
    out=labels.copy(); m=labels==label
    e=ndi.binary_erosion(m,iterations=iterations)
    out[m]=0; out[e]=label
    return CorruptionResult(out,"erode",{"id":int(label),"iterations":iterations})


def dilate_instance(labels: np.ndarray, label: int, iterations: int = 1) -> CorruptionResult:
    out=labels.copy(); m=labels==label
    d=ndi.binary_dilation(m,iterations=iterations)
    # only take background to avoid deleting a different labelled cell in simple V1 corruption
    out[d & (out==0)] = label
    return CorruptionResult(out,"dilate",{"id":int(label),"iterations":iterations})


def split_instance_plane(labels: np.ndarray, label: int, rng: np.random.Generator | None = None) -> CorruptionResult:
    rng=rng or np.random.default_rng()
    out=labels.copy(); coords=np.argwhere(labels==label)
    if len(coords)<8: return CorruptionResult(out,"none",{})
    center=coords.mean(0); normal=rng.normal(size=3); normal/=np.linalg.norm(normal)+1e-8
    side=((coords-center)@normal)>0
    if side.all() or (~side).all(): return CorruptionResult(out,"none",{})
    new_id=int(labels.max())+1
    out[tuple(coords[side].T)]=new_id
    return CorruptionResult(out,"split",{"id":int(label),"new_id":new_id,"normal":normal.tolist()})


def random_corruption(labels: np.ndarray, rng: np.random.Generator | None = None) -> CorruptionResult:
    rng=rng or np.random.default_rng()
    ids=np.unique(labels); ids=ids[ids>0]
    if len(ids)==0: return CorruptionResult(labels.copy(),"none",{})
    op=rng.choice(["remove","erode","dilate","split","merge","false_positive"])
    label=int(rng.choice(ids))
    if op=="remove": return remove_instance(labels,label)
    if op=="erode": return erode_instance(labels,label,int(rng.integers(1,3)))
    if op=="dilate": return dilate_instance(labels,label,int(rng.integers(1,3)))
    if op=="merge": return merge_adjacent_instances(labels,rng)
    if op=="false_positive": return insert_false_blob(labels,rng=rng)
    return split_instance_plane(labels,label,rng)


def merge_adjacent_instances(labels: np.ndarray, rng: np.random.Generator | None = None) -> CorruptionResult:
    rng=rng or np.random.default_rng()
    ids=np.unique(labels);ids=ids[ids>0]
    pairs=[]
    for label in ids:
        dil=ndi.binary_dilation(labels==label,iterations=1)
        neigh=np.unique(labels[dil & (labels!=label)]);neigh=neigh[neigh>0]
        for n in neigh:
            if label<n:pairs.append((int(label),int(n)))
    if not pairs:return CorruptionResult(labels.copy(),"none",{})
    return merge_instances(labels,list(pairs[int(rng.integers(len(pairs)))]))


def insert_false_blob(labels: np.ndarray, center=None, radius: int = 2, rng: np.random.Generator | None = None) -> CorruptionResult:
    rng=rng or np.random.default_rng();out=labels.copy();shape=np.asarray(labels.shape)
    if center is None:center=np.asarray([rng.integers(0,n) for n in shape])
    center=np.asarray(center)
    zz,yy,xx=np.ogrid[:shape[0],:shape[1],:shape[2]]
    blob=(zz-center[0])**2+(yy-center[1])**2+(xx-center[2])**2<=radius**2
    new_id=int(labels.max())+1;out[blob & (out==0)]=new_id
    return CorruptionResult(out,"false_positive",{"new_id":new_id,"center":center.tolist(),"radius":radius})
