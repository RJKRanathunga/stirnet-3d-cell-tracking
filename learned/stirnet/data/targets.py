from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull


@dataclass
class InstanceMetadata:
    ids: torch.Tensor
    features: torch.Tensor
    centroids_um: torch.Tensor


def _pca_axes_um(coords_um: np.ndarray) -> np.ndarray:
    if len(coords_um) < 3:
        return np.zeros(3, np.float32)
    cov = np.cov(coords_um.T)
    vals = np.linalg.eigvalsh(cov)
    vals = np.maximum(vals, 0)[::-1]
    # 4*sqrt(lambda) is a stable extent-like feature, not an exact ellipsoid diameter.
    return (4.0 * np.sqrt(vals)).astype(np.float32)


def extract_instance_metadata(
    labels: np.ndarray,
    raw: np.ndarray,
    spacing_um: tuple[float, float, float],
    dref_um: float,
    marker_heatmap: np.ndarray | None = None,
) -> InstanceMetadata:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    spacing = np.asarray(spacing_um, np.float32)
    patch_center_um = 0.5 * (np.asarray(labels.shape, np.float32) - 1) * spacing
    voxel_volume = float(np.prod(spacing))
    feats, centers = [], []
    for label in ids:
        coords = np.argwhere(labels == label)
        coords_um_abs = coords.astype(np.float32) * spacing[None]
        center_abs = coords_um_abs.mean(0)
        centers.append(center_abs - patch_center_um)
        volume_um3 = len(coords) * voxel_volume
        lo = coords.min(0); hi = coords.max(0) + 1
        bbox_um = (hi - lo).astype(np.float32) * spacing
        axes = _pca_axes_um(coords_um_abs)
        elong = float(axes[0] / max(axes[1], 1e-6))
        flat = float(axes[1] / max(axes[2], 1e-6))
        bbox_volume = float(np.prod(np.maximum(bbox_um, 1e-6)))
        compact = float(volume_um3 / max(bbox_volume, 1e-6))
        solidity = 1.0
        if len(coords) >= 8:
            try:
                hull = ConvexHull(coords_um_abs)
                solidity = float(min(1.0, volume_um3 / max(hull.volume, 1e-6)))
            except Exception:
                pass
        vals = raw[labels == label]
        mean_i = float(vals.mean()) if vals.size else 0.0
        std_i = float(vals.std()) if vals.size else 0.0
        marker_count = 0.0
        if marker_heatmap is not None:
            local = marker_heatmap * (labels == label)
            peaks = local == ndi.maximum_filter(local, size=3)
            marker_count = float(np.count_nonzero(peaks & (local > 0.5)))
        feats.append([
            np.log1p(volume_um3),
            *(bbox_um / max(dref_um, 1e-6)),
            *(axes / max(dref_um, 1e-6)),
            elong, flat, solidity, compact, mean_i, std_i, marker_count,
        ])
    return InstanceMetadata(
        ids=torch.as_tensor(ids, dtype=torch.long),
        features=torch.as_tensor(np.asarray(feats, np.float32).reshape(len(ids), 14)),
        centroids_um=torch.as_tensor(np.asarray(centers, np.float32).reshape(len(ids), 3)),
    )


def estimate_dref_um(labels: np.ndarray, spacing_um: tuple[float,float,float], min_voxels: int = 20) -> float:
    voxel_volume = float(np.prod(spacing_um))
    diameters = []
    for label in np.unique(labels):
        if label <= 0: continue
        n = int(np.count_nonzero(labels == label))
        if n < min_voxels: continue
        vol = n * voxel_volume
        diameters.append(2 * ((3 * vol) / (4 * np.pi)) ** (1 / 3))
    if not diameters:
        return float(np.mean(spacing_um) * 8.0)
    arr = np.asarray(diameters)
    lo, hi = np.percentile(arr, [10, 90]) if len(arr) >= 10 else (arr.min(), arr.max())
    good = arr[(arr >= lo) & (arr <= hi)]
    return float(np.median(good if len(good) else arr))


def make_center_heatmap(shape: tuple[int,int,int], centers_um_relative: np.ndarray, spacing_um, sigma_um: float = 2.0) -> np.ndarray:
    spacing = np.asarray(spacing_um, np.float32)
    center_abs_um = 0.5 * (np.asarray(shape, np.float32) - 1) * spacing
    heat = np.zeros(shape, np.float32)
    for c_rel in centers_um_relative:
        c_vox = (np.asarray(c_rel) + center_abs_um) / spacing
        idx = np.round(c_vox).astype(int)
        if np.all(idx >= 0) and np.all(idx < np.asarray(shape)):
            heat[tuple(idx)] = 1.0
    sigma_vox = tuple(float(sigma_um / max(s, 1e-6)) for s in spacing)
    heat = ndi.gaussian_filter(heat, sigma=sigma_vox, mode="constant")
    if heat.max() > 0:
        heat /= heat.max()
    return heat.astype(np.float32)


def make_instance_boundary(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(labels, dtype=bool)
    for axis in range(3):
        sl1=[slice(None)]*3; sl2=[slice(None)]*3
        sl1[axis]=slice(1,None); sl2[axis]=slice(None,-1)
        a=labels[tuple(sl1)]; b=labels[tuple(sl2)]
        diff=(a!=b) & ((a>0)|(b>0))
        view1=boundary[tuple(sl1)]; view2=boundary[tuple(sl2)]
        view1 |= diff; view2 |= diff
    return boundary


def make_boundary_target(labels: np.ndarray, spacing_um, width_um: float = 1.0) -> np.ndarray:
    boundary = make_instance_boundary(labels)
    spacing=np.asarray(spacing_um,float)
    rv=np.ceil(width_um/spacing).astype(int)
    if rv.max() == 0: return boundary.astype(np.float32)
    zz,yy,xx=np.ogrid[-rv[0]:rv[0]+1,-rv[1]:rv[1]+1,-rv[2]:rv[2]+1]
    structure=(zz*spacing[0])**2+(yy*spacing[1])**2+(xx*spacing[2])**2 <= width_um**2
    return ndi.binary_dilation(boundary, structure=structure).astype(np.float32)


def build_gt_targets(gt_labels: np.ndarray, spacing_um, dref_um: float, center_sigma_um: float = 2.0, boundary_width_um: float = 1.0) -> dict:
    ids=np.unique(gt_labels); ids=ids[ids>0]
    masks=np.stack([(gt_labels==i) for i in ids],axis=0) if len(ids) else np.zeros((0,*gt_labels.shape),bool)
    spacing=np.asarray(spacing_um,np.float32)
    center_abs=0.5*(np.asarray(gt_labels.shape,np.float32)-1)*spacing
    centers=[]
    for i in ids:
        c=np.argwhere(gt_labels==i).mean(0)*spacing-center_abs
        centers.append(c)
    centers=np.asarray(centers,np.float32).reshape(len(ids),3)
    return {
        "ids": torch.as_tensor(ids,dtype=torch.long),
        "masks": torch.as_tensor(masks,dtype=torch.bool),
        "centers_um": torch.as_tensor(centers,dtype=torch.float32),
        "centers_cellscale": torch.as_tensor(centers/max(dref_um,1e-6),dtype=torch.float32),
        "foreground": torch.as_tensor((gt_labels>0).astype(np.float32)),
        "center_heatmap": torch.as_tensor(make_center_heatmap(gt_labels.shape,centers,spacing_um,center_sigma_um)),
        "boundary": torch.as_tensor(make_boundary_target(gt_labels,spacing_um,boundary_width_um)),
    }
