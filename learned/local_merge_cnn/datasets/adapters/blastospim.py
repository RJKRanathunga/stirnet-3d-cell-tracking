"""Adapter for extracted BlastoSPIM 3-D mouse-embryo nuclei datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import BLASTOSPIM_SPACING_ZYX_UM, default_blastospim_root
from ..core.models import AnnotatedVolume, VolumeRecord
from ..core.normalization import robust_intensity_bounds
from ._utils import (
    canonical_split,
    coerce_3d,
    compact_stem,
    infer_split,
    load_array,
    normalize_instance_labels,
    supported_array_files,
    to_zyx,
    validate_axis_order,
)

_LABEL_WORDS = {
    "label", "labels", "mask", "masks", "seg", "segmentation", "gt",
    "groundtruth", "corrected", "correction", "expert", "instance", "instances",
}
_AUTO_WORDS = {"expected", "automatic", "auto", "prediction", "predicted"}
_IMAGE_WORDS = {"image", "images", "img", "raw", "volume", "stack", "data"}


def _label_score(path: Path) -> int:
    lower = path.stem.lower()
    score = 0
    if any(token in lower for token in ("correct", "expert", "groundtruth", "gt")):
        score += 20
    if any(token in lower for token in ("label", "mask", "seg", "instance")):
        score += 10
    if any(token in lower for token in _AUTO_WORDS):
        score -= 10
    return score


def _is_label(path: Path) -> bool:
    return _label_score(path) > 0


def _key(path: Path) -> str:
    return compact_stem(path, _LABEL_WORDS | _AUTO_WORDS | _IMAGE_WORDS)


def _version_from_path(path: Path) -> str | None:
    lower = str(path).lower()
    if "blastospim2" in lower or "2.0" in lower:
        return "2.0"
    if "blastospim1" in lower or "1.0" in lower:
        return "1.0"
    return None


class BlastoSPIMAdapter:
    dataset_name = "blastospim"

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        source_axis_order: str = "zyx",
        spacing_zyx_um: tuple[float, float, float] = BLASTOSPIM_SPACING_ZYX_UM,
    ) -> None:
        self.root = Path(root) if root is not None else default_blastospim_root()
        self.source_axis_order = validate_axis_order(source_axis_order)
        if len(spacing_zyx_um) != 3 or any(float(v) <= 0 for v in spacing_zyx_um):
            raise ValueError("spacing_zyx_um must contain three positive values")
        self.spacing_zyx_um = tuple(float(v) for v in spacing_zyx_um)

    def discover_records(self) -> tuple[VolumeRecord, ...]:
        if not self.root.exists():
            raise FileNotFoundError(
                f"BlastoSPIM directory does not exist: {self.root}. Pass --root or set BLASTOSPIM_DIR."
            )
        files = supported_array_files(self.root)
        label_files = [p for p in files if _is_label(p)]
        image_files = [p for p in files if not _is_label(p)]
        if not label_files or not image_files:
            raise RuntimeError(
                "Could not discover BlastoSPIM raw/annotation arrays. If this release uses "
                "different names, inspect the extracted tree and extend the adapter patterns."
            )

        image_by_key: dict[str, list[Path]] = {}
        for image in image_files:
            image_by_key.setdefault(_key(image), []).append(image)

        label_by_key: dict[str, list[Path]] = {}
        for label in label_files:
            label_by_key.setdefault(_key(label), []).append(label)

        records: list[VolumeRecord] = []
        for key, labels in label_by_key.items():
            images = image_by_key.get(key)
            if not images:
                continue
            # Prefer corrected/expert labels over automatic/expected variants.
            label = sorted(labels, key=lambda p: (-_label_score(p), str(p)))[0]
            image = sorted(images, key=str)[0]
            split = infer_split(label.relative_to(self.root))
            sample_id = f"{split}_{key}" if split != "unspecified" else key
            records.append(
                VolumeRecord(
                    sample_id=sample_id,
                    split=split,
                    image_path=image,
                    labels_path=label,
                    metadata={
                        "version": _version_from_path(label),
                        "source_directory": str(label.parent),
                    },
                )
            )
        if not records:
            raise RuntimeError("No BlastoSPIM raw/annotation pairs could be matched")
        records.sort(key=lambda r: (r.split, r.sample_id))
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
                f"BlastoSPIM image/label shape mismatch for {record.sample_id}: "
                f"{image.shape} versus {labels.shape}"
            )
        return AnnotatedVolume(
            image=np.asarray(image),
            instance_labels=labels,
            spacing_zyx_um=self.spacing_zyx_um,
            dataset_name=self.dataset_name,
            sample_id=record.sample_id,
            split=record.split,
            intensity_bounds=robust_intensity_bounds(image),
            metadata={
                **dict(record.metadata),
                "image_path": str(record.image_path),
                "labels_path": str(record.labels_path),
                "source_axis_order": self.source_axis_order,
            },
        )

    def load_index(self, index: int = 0, *, split: str | None = None) -> AnnotatedVolume:
        records = self.records(split=split)
        if not records:
            raise IndexError(f"no BlastoSPIM records available for split={split!r}")
        return self.load(records[index])


__all__ = ["BlastoSPIMAdapter"]
