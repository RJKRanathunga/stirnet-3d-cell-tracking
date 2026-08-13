from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from learned.stirnet import StirNet
from learned.stirnet.model.config import (
    CoReasoningConfig,
    DecoderConfig,
    InferenceConfig,
    HistoryConfig,
    LossConfig,
    QueryConfig,
    SpatialConfig,
    StirNetConfig,
    TemporalConfig,
    TrainingConfig,
)
from learned.stirnet.training.checkpoint import load_checkpoint

from ..acceptance.first_overfit import build_real_batch
from ..core import DebugConfig, StirNetInspector
from ..io import save_debug_trace


def config_from_dict(data: dict | None) -> StirNetConfig:
    if not data:
        return StirNetConfig()
    return StirNetConfig(
        spatial=SpatialConfig(**data.get("spatial", {})),
        temporal=TemporalConfig(**data.get("temporal", {})),
        history=HistoryConfig(**data.get("history", {})),
        coreasoning=CoReasoningConfig(**data.get("coreasoning", {})),
        queries=QueryConfig(**data.get("queries", {})),
        decoder=DecoderConfig(**data.get("decoder", {})),
        losses=LossConfig(**data.get("losses", {})),
        training=TrainingConfig(**data.get("training", {})),
        inference=InferenceConfig(**data.get("inference", {})),
    )


def _parse_queries(text: str):
    if not text.strip():
        return ()
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one STIR-Net checkpoint and batch.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--batch", type=Path, help="Cached torch batch (.pt)")
    source.add_argument("--first-overfit-data-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preset", choices=("light", "deep"), default="light")
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", choices=("fp16", "bf16", "none"), default="fp16")
    parser.add_argument("--queries", default="", help="Comma-separated explicit query indices")
    parser.add_argument("--open-napari", action="store_true")
    args = parser.parse_args()

    raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = config_from_dict(raw_ckpt.get("config"))
    model = StirNet(cfg)
    load_checkpoint(args.checkpoint, model, map_location="cpu", strict=True)

    if args.batch is not None:
        batch = torch.load(args.batch, map_location="cpu", weights_only=False)
    else:
        batch, sample = build_real_batch(args.first_overfit_data_dir)
        print("Sample:", sample)

    debug_cfg = DebugConfig.deep() if args.preset == "deep" else DebugConfig.light()
    debug_cfg = replace(
        debug_cfg,
        device=args.device,
        amp_dtype=args.amp,
        selected_query_indices=_parse_queries(args.queries),
    )

    inspector = StirNetInspector(model, debug_cfg)
    trace = inspector.inspect(batch)
    save_debug_trace(trace, args.out)

    print("Saved debug trace:", args.out)
    print("Queries:", trace.metadata.get("query_count"))
    print("Matched:", trace.metadata.get("matched_query_count"))
    print("Surviving:", trace.metadata.get("surviving_query_count"))
    print("Selected deep queries:", trace.metadata.get("selected_queries", []))

    if args.open_napari:
        from ..visualization import open_debug_viewer
        selected = trace.metadata.get("selected_queries", [])
        open_debug_viewer(trace, query_index=selected[0] if selected else None)
        import napari
        napari.run()


if __name__ == "__main__":
    main()
