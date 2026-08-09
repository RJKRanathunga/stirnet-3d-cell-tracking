from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data import CachedStirNetDataset, stirnet_collate
from ..model import StirNet, StirNetConfig
from .trainer import Trainer


def _read_list(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-list",required=True,help="Text file containing one cached .pt sample path per line")
    ap.add_argument("--val-list")
    ap.add_argument("--out",default="runs/stirnet_v1")
    ap.add_argument("--epochs",type=int,default=50)
    ap.add_argument("--batch-size",type=int,default=1,help="Use >1 only with spacing/shape-bucketed sample lists")
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--device")
    args=ap.parse_args()
    cfg=StirNetConfig();model=StirNet(cfg)
    train=CachedStirNetDataset(_read_list(args.train_list))
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,
                            pin_memory=True,collate_fn=stirnet_collate)
    val_loader=None
    if args.val_list:
        val=CachedStirNetDataset(_read_list(args.val_list))
        val_loader=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,
                              pin_memory=True,collate_fn=stirnet_collate)
    Trainer(model,cfg,args.device).fit(train_loader,val_loader,args.epochs,args.out)


if __name__=="__main__":main()
