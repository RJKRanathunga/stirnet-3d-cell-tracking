"""Stage 2: STIR-Net spatial instance refinement."""

from learned.stirnet.inference import (
    SpatialInferenceConfig,
    load_spatial_runtime,
    run_parallel_spatial_volume,
)

__all__ = [
    "SpatialInferenceConfig",
    "load_spatial_runtime",
    "run_parallel_spatial_volume",
]
