"""Adapter for Zenodo record 5942575: C. elegans 3-D nuclei."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import C_ELEGANS_SPACING_ZYX_UM, default_c_elegans_root
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

_LABEL_TOKENS = {
    "label", "labels", "mask", "masks", "seg", "segmentation", "gt",
    "groundtruth", "ground", "truth", "instance", "instances",
}
_IMAGE_TOKENS = {"image", "images", "img", "raw", "volume", "stack"}


def _is_label_file(path: Path) -> bool:
    words = set(path.stem.lower().replace("-", "_").split("_"))
    lower = path.stem.lower()
    return bool(words & _LABEL_TOKENS) or any(
        token in lower for token in ("mask", "label", "segmentation", "groundtruth")
    )


def _key(path: Path) -> str:
    return compact_stem(path, _LABEL_TOKENS | _IMAGE_TOKENS)


class CElegansNucleiAdapter:
    dataset_name = "c_elegans_nuclei"

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        source_axis_order: str = "zyx",
        spacing_zyx_um: tuple[float, float, float] = C_ELEGANS_SPACING_ZYX_UM,
    ) -> None:
        self.root = Path(root) if root is not None else default_c_elegans_root()
        self.source_axis_order = validate_axis_order(source_axis_order)
        if len(spacing_zyx_um) != 3 or any(float(v) <= 0 for v in spacing_zyx_um):
            raise ValueError("spacing_zyx_um must contain three positive values")
        self.spacing_zyx_um = tuple(float(v) for v in spacing_zyx_um)

    def discover_records(self) -> tuple[VolumeRecord, ...]:
        if not self.root.exists():
            raise FileNotFoundError(
                f"C. elegans dataset directory does not exist: {self.root}. "
                "Pass --root or set C_ELEGANS_NUCLEI_DIR."
            )
        files = supported_array_files(self.root)
        labels = [path for path in files if _is_label_file(path)]
        images = [path for path in files if not _is_label_file(path)]
        if not labels or not images:
            raise RuntimeError(
                "Could not identify C. elegans image/label files. Expected TIFF/NPY "
                "volumes with label/mask/seg tokens on annotation files."
            )

        image_by_key: dict[str, list[Path]] = {}
        for path in images:
            image_by_key.setdefault(_key(path), []).append(path)

        records: list[VolumeRecord] = []
        used_images: set[Path] = set()
        for label in labels:
            key = _key(label)
            candidates = image_by_key.get(key, [])
            if not candidates:
                # Fallback: longest common normalized stem.
                candidates = sorted(
                    images,
                    key=lambda p: len(set(_key(p)) & set(key)),
                    reverse=True,
                )[:1]
            if not candidates:
                continue
            image = candidates[0]
            if image in used_images:
                continue
            used_images.add(image)
            split = infer_split(label.relative_to(self.root))
            sample_stem = key or image.stem
            sample_id = f"{split}_{sample_stem}" if split != "unspecified" else sample_stem
            records.append(
                VolumeRecord(
                    sample_id=sample_id,
                    split=split,
                    image_path=image,
                    labels_path=label,
                    metadata={"source_directory": str(label.parent)},
                )
            )
        if not records:
            raise RuntimeError("No C. elegans image/label pairs could be matched")
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
                f"C. elegans image/label shape mismatch for {record.sample_id}: "
                f"{image.shape} versus {labels.shape}"
            )
        bounds = robust_intensity_bounds(image)
        return AnnotatedVolume(
            image=np.asarray(image),
            instance_labels=labels,
            spacing_zyx_um=self.spacing_zyx_um,
            dataset_name=self.dataset_name,
            sample_id=record.sample_id,
            split=record.split,
            intensity_bounds=bounds,
            metadata={
                **dict(record.metadata),
                "image_path": str(record.image_path),
                "labels_path": str(record.labels_path),
                "source_axis_order": self.source_axis_order,
                "source": "C. elegans nuclei, Zenodo 5942575",
            },
        )

    def load_index(self, index: int = 0, *, split: str | None = None) -> AnnotatedVolume:
        records = self.records(split=split)
        if not records:
            raise IndexError(f"no records available for split={split!r}")
        return self.load(records[index])


__all__ = ["CElegansNucleiAdapter"]
