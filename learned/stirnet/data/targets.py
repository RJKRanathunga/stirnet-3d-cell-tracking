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


def stable_log_shape_ratio(
    numerator: float,
    denominator: float,
    *,
    denominator_floor: float,
) -> float:
    """Return a stable model feature for a non-negative PCA axis ratio.

    PCA axes estimated from voxel centres can be exactly zero for one-voxel,
    line-like, or plane-like components. The finest physical voxel extent is
    the smallest shape scale the acquisition can resolve, so it is a meaningful
    denominator floor. ``log1p`` then keeps valid elongated objects on a scale
    comparable with the other neural geometry inputs.
    """
    num = max(float(numerator), 0.0)
    den = max(float(denominator), float(denominator_floor), np.finfo(np.float32).tiny)
    return float(np.log1p(num / den))


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
    shape_resolution_um = float(np.min(spacing[spacing > 0])) if np.any(spacing > 0) else 1.0
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
        elong = stable_log_shape_ratio(
            axes[0], axes[1], denominator_floor=shape_resolution_um
        )
        flat = stable_log_shape_ratio(
            axes[1], axes[2], denominator_floor=shape_resolution_um
        )
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


def build_source_gt_compatibility(
    current_labels: np.ndarray,
    gt_labels: np.ndarray,
    gt_ids: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build compact current-source x GT overlap counts on CPU.

    The returned tensors scale with the number of instances, not the native
    volume. A positive count is the source-aware matching compatibility rule.
    """
    current = np.asarray(current_labels)
    ground_truth = np.asarray(gt_labels)
    if current.shape != ground_truth.shape:
        raise ValueError(
            "current_labels and gt_labels must share one native shape; got "
            f"{current.shape} and {ground_truth.shape}"
        )
    source_ids = np.unique(current)
    source_ids = source_ids[source_ids > 0].astype(np.int64, copy=False)
    if gt_ids is None:
        gt_ids = np.unique(ground_truth)
        gt_ids = gt_ids[gt_ids > 0]
    gt_ids = np.asarray(gt_ids, dtype=np.int64)
    overlap = np.zeros((len(source_ids), len(gt_ids)), dtype=np.int64)
    if len(source_ids) and len(gt_ids):
        source_flat = current.reshape(-1)
        gt_flat = ground_truth.reshape(-1)
        positive = (source_flat > 0) & (gt_flat > 0)
        if np.any(positive):
            pairs, counts = np.unique(
                np.stack([source_flat[positive], gt_flat[positive]], axis=1),
                axis=0,
                return_counts=True,
            )
            source_rows = np.searchsorted(source_ids, pairs[:, 0])
            gt_cols = np.searchsorted(gt_ids, pairs[:, 1])
            valid = (
                (source_rows < len(source_ids))
                & (gt_cols < len(gt_ids))
                & (source_ids[source_rows] == pairs[:, 0])
                & (gt_ids[gt_cols] == pairs[:, 1])
            )
            overlap[source_rows[valid], gt_cols[valid]] = counts[valid]
    return (
        torch.as_tensor(source_ids, dtype=torch.long),
        torch.as_tensor(overlap, dtype=torch.long),
    )


def add_source_gt_compatibility(
    target: dict,
    current_labels: np.ndarray | torch.Tensor,
) -> dict:
    """Return a shallow target copy carrying compact source-aware metadata."""
    if "source_ids" in target and "source_gt_overlap" in target:
        return target
    if "label_map" not in target:
        return target
    result = dict(target)
    gt_labels = torch.as_tensor(target["label_map"]).cpu().numpy()
    source_ids, overlap = build_source_gt_compatibility(
        torch.as_tensor(current_labels).cpu().numpy(),
        gt_labels,
        torch.as_tensor(target["ids"]).cpu().numpy(),
    )
    result["source_ids"] = source_ids
    result["source_gt_overlap"] = overlap
    return result


def build_gt_targets(
    gt_labels: np.ndarray,
    spacing_um,
    dref_um: float,
    center_sigma_um: float = 2.0,
    boundary_width_um: float = 1.0,
    *,
    include_dense_masks: bool = False,
    current_labels: np.ndarray | None = None,
) -> dict:
    """Build targets around one integer label map.

    Per-instance native-resolution masks are optional because their
    ``K x Z x Y x X`` allocation is prohibitive for full all-cell scenes. The
    matcher and criterion derive coarse masks and matched native target chunks
    from ``label_map`` and ``ids`` instead.
    """
    ids=np.unique(gt_labels); ids=ids[ids>0]
    spacing=np.asarray(spacing_um,np.float32)
    center_abs=0.5*(np.asarray(gt_labels.shape,np.float32)-1)*spacing
    centers=[]
    for i in ids:
        c=np.argwhere(gt_labels==i).mean(0)*spacing-center_abs
        centers.append(c)
    centers=np.asarray(centers,np.float32).reshape(len(ids),3)
    target = {
        "ids": torch.as_tensor(ids,dtype=torch.long),
        "label_map": torch.as_tensor(np.asarray(gt_labels, dtype=np.int32)),
        "centers_um": torch.as_tensor(centers,dtype=torch.float32),
        "centers_cellscale": torch.as_tensor(centers/max(dref_um,1e-6),dtype=torch.float32),
        "foreground": torch.as_tensor((gt_labels>0).astype(np.float32)),
        "center_heatmap": torch.as_tensor(make_center_heatmap(gt_labels.shape,centers,spacing_um,center_sigma_um)),
        "boundary": torch.as_tensor(make_boundary_target(gt_labels,spacing_um,boundary_width_um)),
    }
    if include_dense_masks:
        masks=np.stack([(gt_labels==i) for i in ids],axis=0) if len(ids) else np.zeros((0,*gt_labels.shape),bool)
        target["masks"] = torch.as_tensor(masks,dtype=torch.bool)
    if current_labels is not None:
        source_ids, overlap = build_source_gt_compatibility(
            current_labels, gt_labels, ids
        )
        target["source_ids"] = source_ids
        target["source_gt_overlap"] = overlap
    return target
