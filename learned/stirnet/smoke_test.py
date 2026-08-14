"""Small CPU smoke test for the complete V1 forward + criterion path."""
from __future__ import annotations

import torch

from .model import RefinementCriterion, StirNet, StirNetConfig


def run():
    cfg=StirNetConfig()
    cfg.queries.discovery_queries=2
    cfg.queries.max_queries=16
    cfg.proposals.max_proposals=12
    cfg.proposals.candidate_pool_size=32
    cfg.decoder.max_spatial_tokens=4096
    model=StirNet(cfg)
    B,Z,Y,X=1,8,32,32
    spatial=torch.randn(B,5,Z,Y,X)
    labels=torch.zeros(B,Z,Y,X,dtype=torch.long)
    labels[:,2:6,10:20,10:20]=1
    spacing=torch.tensor([[1.625,0.40625,0.40625]])
    dref=torch.tensor([8.0])
    instance_features=torch.zeros(1,14)
    instance_ids=torch.tensor([1])
    instance_batch=torch.tensor([0])
    instance_centroids=torch.zeros(1,3)
    gx=torch.zeros(2,32);gx[:,0]=torch.tensor([-0.5,0.5])
    ei=torch.tensor([[0,1],[1,0]],dtype=torch.long);ea=torch.zeros(2,14)
    tracklet=torch.tensor([0,0])
    tref=torch.zeros(1,3);status=torch.zeros(1,10);status[:,3]=1
    hei=torch.zeros(2,0,dtype=torch.long);hea=torch.zeros(0,8);tb=torch.tensor([0])
    outputs=model(spatial,labels,spacing,dref,instance_features,instance_ids,instance_batch,instance_centroids,
                  gx,ei,ea,tracklet,tref,status,hei,hea,tb)
    target_mask=labels[0:1]==1
    targets=[{
        "ids":torch.tensor([1]),
        "masks":target_mask,
        "centers_cellscale":torch.zeros(1,3),
        "foreground":target_mask[0].float(),
        "center_heatmap":torch.zeros(Z,Y,X),
        "boundary":torch.zeros(Z,Y,X),
        "internal_boundary":torch.zeros(Z,Y,X),
        "source_ids":torch.tensor([1]),
        "source_gt_overlap":torch.ones(1,1,dtype=torch.long),
    }]
    criterion=RefinementCriterion(
        cfg.losses,cfg.queries,cfg.training,cfg.proposals
    )
    loss=criterion(outputs,targets)["loss"]
    loss.backward()
    print("STIR-Net smoke test OK",float(loss.detach()))


if __name__=="__main__":
    run()
