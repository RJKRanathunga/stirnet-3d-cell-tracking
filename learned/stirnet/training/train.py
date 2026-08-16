from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from ..data import CachedStirNetDataset, stirnet_collate
from ..model import StirNet, StirNetConfig
from .config import TrainingConfig
from .trainer import Trainer


def _read_list(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _loader(files, *, batch_size: int, workers: int, shuffle: bool):
    dataset = CachedStirNetDataset(files)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=stirnet_collate,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list")
    parser.add_argument("--out", default="runs/stirnet_v2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--stage",
        choices=(
            "geometry_bootstrap",
            "spatial_partition",
            "instance_temporal",
            "refinement_joint",
        ),
    )
    args = parser.parse_args()

    model = StirNet(StirNetConfig())
    training_config = TrainingConfig(lr=args.learning_rate, amp_dtype=args.amp_dtype)
    training_config.curriculum.fixed_stage = args.stage
    train_loader = _loader(
        _read_list(args.train_list),
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
    )
    validation_loader = (
        _loader(
            _read_list(args.val_list),
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=False,
        )
        if args.val_list
        else None
    )
    Trainer(model, training_config, args.device).fit(
        train_loader,
        validation_loader,
        epochs=args.epochs,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
