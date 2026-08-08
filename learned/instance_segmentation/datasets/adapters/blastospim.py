"""Adapter for the BlastoSPIM 1.0/2.0 mouse-embryo nuclei datasets."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import numpy as np

from ..config import BLASTOSPIM_SPACING_ZYX_UM, default_blastospim_root
from ..core.models import AnnotatedVolume, VolumeRecord
from ..core.normalization import robust_intensity_bounds
from ._utils import (
    canonical_split,
    coerce_3d,
    infer_split,
    load_array,
    normalize_instance_labels,
    path_tokens,
    sanitized_sample_id,
    supported_volume_files,
    to_zyx,
    tokens,
    validate_axis_order,
)

_LABEL_TOKENS = {
    "label",
    "labels",
    "mask",
    "masks",
    "seg",
    "segmentation",
    "segmentations",
    "groundtruth",
    "ground",
    "truth",
    "gt",
    "annotation",
    "annotations",
    "corrected",
}
_IMAGE_TOKENS = {"image", "images", "img", "raw", "volume", "volumes", "data"}
_IGNORE_TOKENS = {
    "preview",
    "projection",
    "projections",
    "mip",
    "visualize",
    "visualization",
    "visulize",
    "roi",
    "rois",
}
_SEMANTIC_TOKENS = _LABEL_TOKENS | _IMAGE_TOKENS | {
    "expected",
    "corrected",
    "manual",
    "expert",
}


def _is_ignored(path: Path) -> bool:
    return bool(set(path_tokens(path)) & _IGNORE_TOKENS)


def _is_label(path: Path) -> bool:
    return bool(set(path_tokens(path)) & _LABEL_TOKENS)


def _is_explicit_image(path: Path) -> bool:
    path_set = set(path_tokens(path))
    return bool(path_set & _IMAGE_TOKENS) and not bool(path_set & _LABEL_TOKENS)


def _pair_key(path: Path) -> str:
    stem = [token for token in tokens(path.stem) if token not in _SEMANTIC_TOKENS]
    if not stem:
        stem = list(tokens(path.stem))
    return "_".join(stem)


def _context_tokens(path: Path) -> set[str]:
    return {
        token
        for part in path.parts[:-1]
        for token in tokens(part)
        if token not in _SEMANTIC_TOKENS
        and token not in {"train", "training", "val", "validation", "test", "testing"}
    }


def _label_preference(path: Path) -> int:
    value = str(path).lower()
    path_set = set(path_tokens(path))
    if "corrected" in path_set or "groundtruth" in path_set or "gt" in path_set:
        return 0
    if "manual" in path_set or "expert" in path_set or "annotation" in path_set:
        return 1
    if "expected" in path_set:
        return 5
    if "segmentation" in path_set or "labels" in path_set or "label" in path_set:
        return 2
    return 3 + int("expected" in value)


def _candidate_score(image: Path, label: Path) -> tuple[int, int, int, str]:
    exact_stem = 0 if image.stem.lower() == label.stem.lower() else 1
    shared_context = len(_context_tokens(image) & _context_tokens(label))
    return (_label_preference(label), exact_stem, -shared_context, str(label))


def _sample_context(image: Path, root: Path) -> tuple[str, ...]:
    relative = image.relative_to(root)
    parts: list[str] = []
    for part in relative.parts[:-1]:
        part_tokens = tokens(part)
        if set(part_tokens) & _IMAGE_TOKENS:
            continue
        if part.lower().startswith(("train", "val", "test")):
            continue
        parts.append(part)
    parts.append(_pair_key(image) or image.stem)
    return tuple(parts)


def _infer_version(path: Path) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", str(path).lower())
    if "blastospim2" in compact:
        return "2.0"
    if "blastospim1" in compact:
        return "1.0"
    # The official site states that the series literally named "Blast" is 2.0.
    if any(part.lower() == "blast" for part in path.parts):
        return "2.0"
    return "unspecified"


class BlastoSPIMAdapter:
    """Discover raw/ground-truth TIFF pairs from BlastoSPIM archives.

    The official downloads are split into train/validation/two test archives.
    This adapter tolerates both an umbrella directory containing those
    extracted archives and a root pointing directly at one extracted archive.
    It prefers expert/corrected/ground-truth segmentation directories over
    automatically generated ``Expected_Segmentation`` files when both exist.
    """

    dataset_name = "blastospim"

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        source_axis_order: str = "zyx",
        spacing_override_zyx_um: tuple[float, float, float] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else default_blastospim_root()
        self.source_axis_order = validate_axis_order(source_axis_order)
        spacing = spacing_override_zyx_um or BLASTOSPIM_SPACING_ZYX_UM
        if len(spacing) != 3 or any(float(v) <= 0 for v in spacing):
            raise ValueError("spacing_override_zyx_um must contain three positive values")
        self.spacing_zyx_um = tuple(float(v) for v in spacing)

    def discover_records(self) -> tuple[VolumeRecord, ...]:
        if not self.root.exists():
            raise FileNotFoundError(
                f"BlastoSPIM dataset directory does not exist: {self.root}. "
                "Pass --root or set BLASTOSPIM_DIR."
            )
        files = tuple(path for path in supported_volume_files(self.root) if not _is_ignored(path))
        if not files:
            raise FileNotFoundError(f"no TIFF/NPY volumes found below {self.root}")

        labels = [path for path in files if _is_label(path)]
        explicit_images = [path for path in files if _is_explicit_image(path)]
        if explicit_images:
            images = explicit_images
        else:
            images = [path for path in files if path not in labels]

        if not images or not labels:
            raise RuntimeError(
                "Could not identify BlastoSPIM raw images and ground-truth labels. "
                "Expected image/raw directories plus label/segmentation/ground-truth "
                "directories. Run inspect_volume --list after checking extraction."
            )

        labels_by_split_key: dict[tuple[str, str], list[Path]] = defaultdict(list)
        for label in labels:
            split = infer_split(Path(self.root.name) / label.relative_to(self.root))
            labels_by_split_key[(split, _pair_key(label))].append(label)

        records: list[VolumeRecord] = []
        used_labels: set[Path] = set()
        unpaired: list[Path] = []
        for image in sorted(images):
            relative = image.relative_to(self.root)
            split = infer_split(Path(self.root.name) / relative)
            key = _pair_key(image)
            candidates = [
                label
                for label in labels_by_split_key.get((split, key), [])
                if label not in used_labels
            ]
            if not candidates:
                candidates = [
                    label
                    for label in labels
                    if label not in used_labels
                    and label.stem.lower() == image.stem.lower()
                    and infer_split(Path(self.root.name) / label.relative_to(self.root)) == split
                ]
            if not candidates and split == "unspecified":
                candidates = [
                    label
                    for label in labels
                    if label not in used_labels and _pair_key(label) == key
                ]
            if not candidates:
                unpaired.append(image)
                continue

            label = min(candidates, key=lambda candidate: _candidate_score(image, candidate))
            used_labels.add(label)
            sample_id = sanitized_sample_id(_sample_context(image, self.root))
            version = _infer_version(Path(self.root.name) / relative)
            records.append(
                VolumeRecord(
                    sample_id=sample_id,
                    split=split,
                    image_path=image,
                    labels_path=label,
                    metadata={
                        "version": version,
                        "image_relative_path": str(relative),
                        "labels_relative_path": str(label.relative_to(self.root)),
                    },
                )
            )

        if not records:
            examples = ", ".join(str(path.relative_to(self.root)) for path in unpaired[:5])
            raise RuntimeError(
                "No BlastoSPIM image/label pairs could be formed. "
                f"Example unpaired images: {examples or 'none'}"
            )
        records.sort(key=lambda record: (record.split, record.sample_id))
        return tuple(records)

    def records(self, *, split: str | None = None) -> tuple[VolumeRecord, ...]:
        records = self.discover_records()
        wanted = canonical_split(split)
        if wanted is None:
            return records
        return tuple(record for record in records if record.split == wanted)

    def load(self, record: VolumeRecord) -> AnnotatedVolume:
        image = to_zyx(
            coerce_3d(load_array(record.image_path), name="image"),
            self.source_axis_order,
        )
        labels = normalize_instance_labels(
            to_zyx(
                coerce_3d(load_array(record.labels_path), name="instance labels"),
                self.source_axis_order,
            )
        )
        if image.shape != labels.shape:
            raise ValueError(
                f"BlastoSPIM image/GT shape mismatch for {record.sample_id}: "
                f"{image.shape} versus {labels.shape}. If the TIFFs are stored in "
                "a different axis order, pass --source-axis-order explicitly."
            )
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
                "version": record.metadata.get("version", "unspecified"),
                "source": "BlastoSPIM 1.0/2.0",
                "source_axis_order": self.source_axis_order,
            },
        )

    def load_index(self, index: int = 0, *, split: str | None = None) -> AnnotatedVolume:
        records = self.records(split=split)
        if not records:
            raise IndexError(f"no BlastoSPIM records available for split={split!r}")
        return self.load(records[index])


__all__ = ["BlastoSPIMAdapter"]
