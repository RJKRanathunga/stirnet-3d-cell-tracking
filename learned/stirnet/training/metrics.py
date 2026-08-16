from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def instance_iou_matrix(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError("pred and gt label maps must have identical shapes")
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)
    pids, pred_inverse, pred_counts = np.unique(
        pred_flat[pred_flat > 0], return_inverse=True, return_counts=True
    )
    gids, gt_inverse, gt_counts = np.unique(
        gt_flat[gt_flat > 0], return_inverse=True, return_counts=True
    )
    intersections = np.zeros((len(pids), len(gids)), dtype=np.int64)
    positive = (pred_flat > 0) & (gt_flat > 0)
    if positive.any() and len(pids) and len(gids):
        pred_rows = np.searchsorted(pids, pred_flat[positive])
        gt_columns = np.searchsorted(gids, gt_flat[positive])
        packed = pred_rows * len(gids) + gt_columns
        intersections = np.bincount(
            packed, minlength=len(pids) * len(gids)
        ).reshape(len(pids), len(gids))
    unions = pred_counts[:, None] + gt_counts[None, :] - intersections
    mat = np.divide(
        intersections,
        unions,
        out=np.zeros(intersections.shape, dtype=np.float32),
        where=unions > 0,
    )
    return mat, pids, gids


def instance_metrics(pred: np.ndarray, gt: np.ndarray, iou_threshold: float = 0.5) -> dict:
    iou,pids,gids=instance_iou_matrix(pred,gt)
    if len(pids) and len(gids):
        r,c=linear_sum_assignment(1-iou)
        good=iou[r,c]>=iou_threshold
        tp=int(good.sum()); matched_iou=iou[r[good],c[good]] if tp else np.array([])
    else:
        tp=0; matched_iou=np.array([])
    fp=len(pids)-tp;fn=len(gids)-tp
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    f1=2*precision*recall/max(precision+recall,1e-8)
    return {"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,"f1":f1,
            "mean_matched_iou":float(matched_iou.mean()) if len(matched_iou) else 0.0,
            "count_error":abs(len(pids)-len(gids))}
