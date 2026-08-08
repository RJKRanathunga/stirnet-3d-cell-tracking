"""Adapter for Zenodo record 5942575: C. elegans 3-D nuclei."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import tifffile

from ..config import C_ELEGANS_SPACING_ZYX_UM, default_c_elegans_root
from ..core.models import AnnotatedVolume, VolumeRecord
from ..core.normalization import robust_intensity_bounds

_SUPPORTED_SUFFIXES = {".tif", ".tiff", ".npy"}
_MASK_TOKENS = {
    "mask",
    "masks",
    "label",
    "labels",
    "gt",
    "groundtruth",
    "ground_truth",
    "seg",
    "segmentation",
    "instance",
    "instances",
    "annotation",
    "annotations",
}
_IMAGE_TOKENS = {"image", "images", "img", "imgs", "raw", "raws", "data"}
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in re.split(r"[^a-z0-9]+", text.lower()) if token)


def _path_tokens(path: Path) -> tuple[str, ...]:
    result: list[str] = []
    for part in path.parts:
        result.extend(_tokens(part))
    return tuple(result)


def _infer_split(relative_path: Path) -> str:
    for part in relative_path.parts:
        normalized = part.lower().replace("-", "_")
        for alias, canonical in _SPLIT_ALIASES.items():
            if normalized == alias or normalized.startswith(alias + "_"):
                return canonical
    return "unspecified"


def _is_mask_path(relative_path: Path) -> bool:
    tokens = set(_path_tokens(relative_path))
    return bool(tokens & _MASK_TOKENS)


def _normalized_pair_key(path: Path) -> str:
    """Normalize image/mask naming differences while preserving sample identity."""

    stem_tokens = list(_tokens(path.stem))
    filtered = [
        token
        for token in stem_tokens
        if token not in _MASK_TOKENS and token not in _IMAGE_TOKENS
    ]
    if not filtered:
        # Separate image/mask directories often intentionally use identical
        # numeric stems such as 001.tif. Keep the original stem in that case.
        filtered = stem_tokens
    return "_".join(filtered)


def _semantic_path_score(image: Path, mask: Path) -> tuple[int, int, str]:
    """Prefer same stem and nearby sibling directories when pairing files."""

    exact_stem_penalty = 0 if image.stem.lower() == mask.stem.lower() else 1
    image_parts = image.parts[:-1]
    mask_parts = mask.parts[:-1]
    common = 0
    for a, b in zip(image_parts, mask_parts):
        if a.lower() != b.lower():
            break
        common += 1
    directory_penalty = (len(image_parts) - common) + (len(mask_parts) - common)
    return exact_stem_penalty, directory_penalty, str(mask)


def _load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return tifffile.imread(path)


def _coerce_3d(array: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(array)
    result = np.squeeze(result)
    if result.ndim != 3:
        raise ValueError(f"{name} must be 3-D after squeezing, got {result.shape}")
    return result


def _to_zyx(array: np.ndarray, source_axis_order: str) -> np.ndarray:
    order = source_axis_order.lower()
    if sorted(order) != ["x", "y", "z"] or len(order) != 3:
        raise ValueError("source_axis_order must be a permutation of 'zyx'")
    permutation = tuple(order.index(axis) for axis in "zyx")
    return np.transpose(array, permutation)


class CElegansNucleiAdapter:
    """Discover and load the downloaded C. elegans nuclei dataset.

    The adapter intentionally tolerates common extracted layouts, e.g.::

        train/images/001.tif      train/masks/001.tif
        train/raw/001.tif         train/labels/001.tif
        images/train/001.tif      masks/train/001.tif

    It never guesses axis permutations. Loaded TIFF stacks must already have
    matching [Z,Y,X] image/mask shapes; a mismatch raises immediately.
    """

    dataset_name = "c_elegans_nuclei"
    spacing_zyx_um = C_ELEGANS_SPACING_ZYX_UM

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        source_axis_order: str = "zyx",
    ) -> None:
        self.root = Path(root) if root is not None else default_c_elegans_root()
        order = source_axis_order.lower()
        if sorted(order) != ["x", "y", "z"] or len(order) != 3:
            raise ValueError("source_axis_order must be a permutation of 'zyx'")
        self.source_axis_order = order

    def _files(self) -> tuple[Path, ...]:
        if not self.root.exists():
            raise FileNotFoundError(
                f"C. elegans dataset directory does not exist: {self.root}. "
                "Pass root=... or set C_ELEGANS_NUCLEI_DIR."
            )
        return tuple(
            sorted(
                path
                for path in self.root.rglob("*")
                if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
            )
        )

    def discover_records(self) -> tuple[VolumeRecord, ...]:
        files = self._files()
        if not files:
            raise FileNotFoundError(
                f"no TIFF/NPY volume files were found below {self.root}"
            )

        by_split_images: dict[str, list[Path]] = defaultdict(list)
        by_split_masks: dict[str, list[Path]] = defaultdict(list)
        for path in files:
            relative = path.relative_to(self.root)
            split = _infer_split(relative)
            if _is_mask_path(relative):
                by_split_masks[split].append(path)
            else:
                by_split_images[split].append(path)

        records: list[VolumeRecord] = []
        diagnostics: list[str] = []
        all_splits = sorted(set(by_split_images) | set(by_split_masks))
        for split in all_splits:
            images = by_split_images.get(split, [])
            masks = by_split_masks.get(split, [])
            if not images or not masks:
                diagnostics.append(
                    f"split={split}: images={len(images)}, masks={len(masks)}"
                )
                continue

            masks_by_key: dict[str, list[Path]] = defaultdict(list)
            for mask in masks:
                masks_by_key[_normalized_pair_key(mask)].append(mask)

            used_masks: set[Path] = set()
            for image in images:
                key = _normalized_pair_key(image)
                candidates = [m for m in masks_by_key.get(key, []) if m not in used_masks]
                if not candidates:
                    # Exact-stem fallback handles layouts whose naming contains
                    # unfamiliar directory tokens but identical file names.
                    candidates = [
                        m
                        for m in masks
                        if m not in used_masks and m.stem.lower() == image.stem.lower()
                    ]
                if not candidates:
                    diagnostics.append(
                        f"unpaired image in split={split}: {image.relative_to(self.root)}"
                    )
                    continue
                mask = min(candidates, key=lambda candidate: _semantic_path_score(image, candidate))
                used_masks.add(mask)
                sample_key = key or image.stem
                sample_id = f"{split}_{sample_key}" if split != "unspecified" else sample_key
                records.append(
                    VolumeRecord(
                        sample_id=sample_id,
                        split=split,
                        image_path=image,
                        labels_path=mask,
                    )
                )

        if not records:
            details = "; ".join(diagnostics[:10])
            raise RuntimeError(
                "could not pair raw images with instance masks under "
                f"{self.root}. Discovery details: {details}"
            )

        records.sort(key=lambda record: (record.split, record.sample_id))
        return tuple(records)

    def records(self, *, split: str | None = None) -> tuple[VolumeRecord, ...]:
        records = self.discover_records()
        if split is None:
            return records
        canonical = _SPLIT_ALIASES.get(split.lower(), split.lower())
        return tuple(record for record in records if record.split == canonical)

    def load(self, record: VolumeRecord) -> AnnotatedVolume:
        image = _to_zyx(_coerce_3d(_load_array(record.image_path), name="image"), self.source_axis_order)
        labels_raw = _to_zyx(_coerce_3d(_load_array(record.labels_path), name="instance labels"), self.source_axis_order)
        if image.shape != labels_raw.shape:
            raise ValueError(
                "C. elegans image and label stacks must have identical [Z,Y,X] shape; "
                f"got {image.shape} and {labels_raw.shape} for {record.sample_id}. "
                "The adapter intentionally does not guess axis permutations."
            )
        if np.issubdtype(labels_raw.dtype, np.floating):
            if not np.all(np.isfinite(labels_raw)):
                raise ValueError("instance labels contain non-finite values")
            rounded = np.rint(labels_raw)
            if not np.allclose(labels_raw, rounded, atol=1e-5):
                raise ValueError("floating instance labels are not integer-valued")
            labels = rounded.astype(np.int32)
        else:
            labels = labels_raw.astype(np.int32, copy=False)
        if np.any(labels < 0):
            raise ValueError("instance labels contain negative values")

        valid = np.ones(image.shape, dtype=bool)
        bounds = robust_intensity_bounds(image)
        return AnnotatedVolume(
            image=np.asarray(image),
            instance_labels=labels,
            spacing_zyx_um=self.spacing_zyx_um,
            valid_mask=valid,
            intensity_bounds=bounds,
            dataset_name=self.dataset_name,
            sample_id=record.sample_id,
            split=record.split,
            metadata={
                "image_path": str(record.image_path),
                "labels_path": str(record.labels_path),
                "source": "Zenodo 10.5281/zenodo.5942575",
                "source_axis_order": self.source_axis_order,
            },
        )

    def load_index(self, index: int = 0, *, split: str | None = None) -> AnnotatedVolume:
        records = self.records(split=split)
        if not records:
            raise IndexError(f"no records available for split={split!r}")
        return self.load(records[index])
