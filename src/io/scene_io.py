"""Core loading for fixed tracking-scene artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .arrays import load_npy
from .tables import load_json


@dataclass(frozen=True)
class TrackingSceneData:
    path: Path
    metadata: dict[str, Any]
    frames: np.ndarray
    binary_mask: np.ndarray
    instance_labels: np.ndarray
    images: dict[str, np.ndarray]


def load_tracking_scene(path: str | Path) -> TrackingSceneData:
    root = Path(path)
    metadata = load_json(root / "scene.json")
    files = metadata.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("scene.json field 'files' must be an object")
    frames = load_npy(root / str(files.get("frames", "frames.npy")), expected_ndim=1)
    mask = load_npy(root / str(files.get("binary_mask", "binary_mask.npy")), expected_ndim=4)
    labels = load_npy(root / str(files.get("instance_labels", "instance_labels.npy")), expected_ndim=4)
    if mask.shape != labels.shape or mask.shape[0] != len(frames):
        raise ValueError("Tracking-scene arrays are not aligned")
    images: dict[str, np.ndarray] = {}
    image_files = files.get("masked_images", {})
    if not isinstance(image_files, dict):
        raise ValueError("scene.json field 'files.masked_images' must be an object")
    for name, filename in image_files.items():
        image = load_npy(root / str(filename), expected_ndim=4)
        if image.shape != labels.shape:
            raise ValueError(f"Scene image {name!r} has shape {image.shape}, expected {labels.shape}")
        images[str(name)] = image
    return TrackingSceneData(root, metadata, frames, mask, labels, images)
