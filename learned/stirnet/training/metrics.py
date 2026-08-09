from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def instance_iou_matrix(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    pids=np.unique(pred);pids=pids[pids>0]
    gids=np.unique(gt);gids=gids[gids>0]
    mat=np.zeros((len(pids),len(gids)),np.float32)
    for i,p in enumerate(pids):
        pm=pred==p
        for j,g in enumerate(gids):
            gm=gt==g
            inter=np.count_nonzero(pm & gm); union=np.count_nonzero(pm | gm)
            mat[i,j]=inter/union if union else 0
    return mat,pids,gids


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
