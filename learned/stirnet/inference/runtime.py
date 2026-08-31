from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, is_dataclass, replace
import hashlib
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from learned.stirnet import StirNet
from learned.stirnet.model.config import ModelConfig
from learned.stirnet.training.checkpoint import load_checkpoint


@dataclass(frozen=True)
class SpatialInferenceConfig:
    """Production configuration for tiled spatial STIR-Net inference."""

    spacing_zyx_um: tuple[float, float, float] = (
        1.625,
        0.40625,
        0.40625,
    )
    tile_shape_zyx: tuple[int, int, int] = (32, 128, 128)
    tile_overlap_zyx: tuple[int, int, int] = (8, 32, 32)
    tile_halo_zyx: tuple[int, int, int] = (4, 16, 16)
    tile_batch_size: int = 1

    def validate(self) -> None:
        if len(self.spacing_zyx_um) != 3 or any(
            float(v) <= 0 for v in self.spacing_zyx_um
        ):
            raise ValueError("spacing_zyx_um must contain 3 positive values")
        if len(self.tile_shape_zyx) != 3 or any(
            int(v) <= 0 for v in self.tile_shape_zyx
        ):
            raise ValueError("tile_shape_zyx must contain 3 positive integers")
        if len(self.tile_overlap_zyx) != 3 or any(
            int(v) < 0 for v in self.tile_overlap_zyx
        ):
            raise ValueError("tile_overlap_zyx must contain 3 non-negative integers")
        if len(self.tile_halo_zyx) != 3 or any(
            int(v) < 0 for v in self.tile_halo_zyx
        ):
            raise ValueError("tile_halo_zyx must contain 3 non-negative integers")
        if any(
            int(overlap) >= int(tile)
            for overlap, tile in zip(
                self.tile_overlap_zyx,
                self.tile_shape_zyx,
            )
        ):
            raise ValueError("Every tile overlap must be smaller than its tile size")
        if any(
            2 * int(halo) >= int(tile)
            for halo, tile in zip(
                self.tile_halo_zyx,
                self.tile_shape_zyx,
            )
        ):
            raise ValueError("Every tile halo must be smaller than half its tile size")
        if int(self.tile_batch_size) < 1:
            raise ValueError("tile_batch_size must be positive")


@dataclass
class SpatialModelRuntime:
    model: Any
    model_cfg: ModelConfig
    inference_cfg: Any
    device: torch.device
    checkpoint_path: Path
    checkpoint_step: int
    checkpoint_sha256: str


def resolve_device(value: str | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        device = value
    else:
        token = str(value).strip().lower()
        if token == "auto":
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            device = torch.device(token)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False"
        )
    return device


def _sha256(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_torch(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    try:
        payload = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            path,
            map_location=map_location,
        )
    if not isinstance(payload, dict):
        raise TypeError(
            f"Checkpoint must contain a dictionary, got "
            f"{type(payload).__name__}"
        )
    return payload


def _hydrate_dataclass(
    instance: Any,
    payload: dict[str, Any],
    *,
    path: str,
) -> Any:
    """Strictly hydrate a nested dataclass from serialized checkpoint config."""
    if not is_dataclass(instance):
        raise TypeError(f"{path} is not a dataclass instance")

    for key, value in payload.items():
        if not hasattr(instance, key):
            raise KeyError(
                f"Checkpoint config contains unknown field {path}.{key}. "
                "Checkpoint and current repository are not configuration-compatible."
            )
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _hydrate_dataclass(
                current,
                value,
                path=f"{path}.{key}",
            )
        elif isinstance(current, tuple) and isinstance(
            value,
            (tuple, list),
        ):
            setattr(instance, key, tuple(value))
        else:
            setattr(instance, key, value)

    return instance


def build_inference_config(
    model_cfg: ModelConfig,
    config: SpatialInferenceConfig,
):
    config.validate()
    return replace(
        model_cfg.inference,
        mode="tiled",
        tiled_dense_enabled=True,
        tile_shape_zyx=tuple(
            int(v) for v in config.tile_shape_zyx
        ),
        tile_overlap_zyx=tuple(
            int(v) for v in config.tile_overlap_zyx
        ),
        tile_halo_zyx=tuple(
            int(v) for v in config.tile_halo_zyx
        ),
        tile_batch_size=int(config.tile_batch_size),
    )


def load_spatial_runtime(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "auto",
    config: SpatialInferenceConfig | None = None,
) -> SpatialModelRuntime:
    """
    Load a STIR-Net checkpoint for production inference.

    Only model_config and model state are inference-critical. Experiment-local
    training_config metadata is deliberately ignored rather than routed through
    an investigation/evaluation helper.
    """
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    device_obj = resolve_device(device)
    cfg = config or SpatialInferenceConfig()
    cfg.validate()

    payload = _load_torch(
        checkpoint_path,
        map_location="cpu",
    )

    serialized_model_cfg = payload.get("model_config")
    if not isinstance(serialized_model_cfg, dict):
        raise ValueError(
            "Checkpoint does not contain a serialized model_config"
        )

    model_cfg = ModelConfig()
    _hydrate_dataclass(
        model_cfg,
        serialized_model_cfg,
        path="model_config",
    )
    model_cfg.validate()

    model = StirNet(model_cfg)
    load_checkpoint(
        checkpoint_path,
        model,
        map_location="cpu",
        strict=True,
    )
    model.to(device_obj)
    model.eval()

    inference_cfg = build_inference_config(
        model_cfg,
        cfg,
    )

    runtime = SpatialModelRuntime(
        model=model,
        model_cfg=model_cfg,
        inference_cfg=inference_cfg,
        device=device_obj,
        checkpoint_path=checkpoint_path,
        checkpoint_step=int(payload.get("global_step", -1)),
        checkpoint_sha256=_sha256(checkpoint_path),
    )

    print(
        f"[checkpoint] step={runtime.checkpoint_step} "
        f"device={runtime.device} "
        f"sha256={runtime.checkpoint_sha256[:12]}...",
        flush=True,
    )
    return runtime


def amp_context(
    device: torch.device,
):
    if device.type != "cuda":
        return nullcontext(), "fp32"
    if torch.cuda.is_bf16_supported():
        return (
            torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ),
            "bf16",
        )
    return (
        torch.autocast(
            "cuda",
            dtype=torch.float16,
        ),
        "fp16",
    )


def tensor_numpy(
    value: Any,
    dtype=None,
) -> np.ndarray:
    tensor = torch.as_tensor(value).detach()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    array = tensor.cpu().numpy()
    return (
        array.astype(dtype, copy=False)
        if dtype is not None
        else array
    )


def run_tiled_spatial(
    runtime: SpatialModelRuntime,
    spatial: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
    dref_um: float,
):
    """Run only the CUDA/torch spatial call for one prepared frame."""
    from learned.stirnet.inference.tiled_dense import (
        tiled_spatial_inference,
    )

    spatial_tensor = torch.from_numpy(spatial)[None].to(
        device=runtime.device,
        dtype=torch.float32,
        non_blocking=False,
    )
    spacing_tensor = torch.tensor(
        [spacing_zyx_um],
        device=runtime.device,
        dtype=torch.float32,
    )
    dref_tensor = torch.tensor(
        [float(dref_um)],
        device=runtime.device,
        dtype=torch.float32,
    )

    if runtime.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(runtime.device)

    amp, amp_name = amp_context(runtime.device)
    started = time.perf_counter()

    with torch.inference_mode(), amp:
        result = tiled_spatial_inference(
            runtime.model,
            spatial_tensor,
            spacing_tensor,
            dref_tensor,
            config=runtime.inference_cfg,
        )

    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)

    seconds = time.perf_counter() - started
    peak_gib = (
        float(
            torch.cuda.max_memory_allocated(runtime.device)
            / 2**30
        )
        if runtime.device.type == "cuda"
        else 0.0
    )

    return (
        result,
        spatial_tensor,
        amp_name,
        float(seconds),
        float(peak_gib),
    )
