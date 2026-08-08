"""Adapter for the NIS3D dense 3-D nuclei benchmark."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from ..config import default_nis3d_root
from ..core.models import AnnotatedVolume, VolumeRecord
from ..core.normalization import robust_intensity_bounds
from ._utils import (
    canonical_split,
    coerce_3d,
    find_casefold_file,
    infer_split,
    load_array,
    normalize_instance_labels,
    sanitized_sample_id,
    to_zyx,
    validate_axis_order,
)

# Values explicitly stated in the NIS3D paper. The source reports spacing in
# XYZ order; this table stores our internal ZYX order. Mus musculus spacing is
# intentionally not guessed: its bundled Info.txt is parsed instead.
_KNOWN_SPACING_ZYX_UM: dict[str, tuple[float, float, float]] = {
    "zebrafish1": (2.5, 0.43, 0.43),
    "zebrafish2": (1.0, 1.0, 1.0),
    "drosophila1": (1.0, 1.0, 1.0),
    "drosophila2": (1.0, 1.0, 1.0),
}

_DATA_NAMES = ("Data.tif", "data.tif", "Data.tiff", "data.tiff")
_GT_NAMES = (
    "GroundTruth.tif",
    "groundtruth.tif",
    "ground_truth.tif",
    "gt.tif",
    "GroundTruth.tiff",
    "gt.tiff",
)
_CONFIDENCE_NAMES = (
    "ConfidenceScore.tif",
    "confidencescore.tif",
    "scoreOfConfidence.tif",
    "scoreofconfidence.tif",
    "confidence.tif",
    "ConfidenceScore.tiff",
)
_INFO_NAMES = ("Info.txt", "info.txt")


def _compact_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _spacing_from_info_text(text: str) -> tuple[float, float, float] | None:
    """Parse common NIS3D Info.txt spacing forms and return ZYX micrometres."""

    lower = text.lower().replace("μ", "u").replace("µ", "u")

    # Prefer explicitly axis-labelled values when present.
    axis_values: dict[str, float] = {}
    for axis in "xyz":
        patterns = (
            rf"\b{axis}\s*(?:voxel\s*)?(?:size|resolution|spacing)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            rf"\b{axis}\s*[- ]?(?:resolution|spacing)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
        )
        for pattern in patterns:
            match = re.search(pattern, lower)
            if match:
                axis_values[axis] = float(match.group(1))
                break
    if len(axis_values) == 3:
        return (axis_values["z"], axis_values["y"], axis_values["x"])

    # NIS3D documentation and paper state voxel sizes in XYZ order, e.g.
    # "0.43 um x 0.43 um x 2.5 um".
    for line in lower.splitlines():
        if not any(token in line for token in ("voxel", "resolution", "spacing", "pixel")):
            continue
        values = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(?:u?m)?", line)
        if len(values) >= 3:
            x, y, z = (float(value) for value in values[:3])
            if x > 0 and y > 0 and z > 0:
                return (z, y, x)
    return None


def _known_spacing_for_sample(sample_id: str) -> tuple[float, float, float] | None:
    compact = _compact_name(sample_id)
    for key, spacing in _KNOWN_SPACING_ZYX_UM.items():
        if key in compact:
            return spacing
    return None


class NIS3DAdapter:
    """Load NIS3D folders containing Data/GT/confidence/Info sidecars.

    Official sample folders contain ``Data.tif``, ``GroundTruth.tif``,
    ``ConfidenceScore.tif``, ``Visulize.tif`` and ``Info.txt``. A few mirrors
    use ``gt.tif`` / ``scoreOfConfidence.tif``; those aliases are supported.

    Confidence value 1 denotes the NIS3D "undefined mask" region and is
    converted to ``valid_mask=False`` so those voxels can be excluded from
    training losses. Values 0 and 2..4 remain valid.
    """

    dataset_name = "nis3d"

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        source_axis_order: str = "zyx",
        spacing_override_zyx_um: tuple[float, float, float] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else default_nis3d_root()
        self.source_axis_order = validate_axis_order(source_axis_order)
        if spacing_override_zyx_um is not None:
            if len(spacing_override_zyx_um) != 3 or any(float(v) <= 0 for v in spacing_override_zyx_um):
                raise ValueError("spacing_override_zyx_um must contain three positive values")
            spacing_override_zyx_um = tuple(float(v) for v in spacing_override_zyx_um)
        self.spacing_override_zyx_um = spacing_override_zyx_um

    def discover_records(self) -> tuple[VolumeRecord, ...]:
        if not self.root.exists():
            raise FileNotFoundError(
                f"NIS3D dataset directory does not exist: {self.root}. "
                "Pass --root or set NIS3D_DIR."
            )

        candidate_dirs = [self.root]
        candidate_dirs.extend(path for path in self.root.rglob("*") if path.is_dir())
        records: list[VolumeRecord] = []
        seen_images: set[Path] = set()
        for directory in candidate_dirs:
            image = find_casefold_file(directory, _DATA_NAMES)
            labels = find_casefold_file(directory, _GT_NAMES)
            if image is None or labels is None or image in seen_images:
                continue
            seen_images.add(image)
            confidence = find_casefold_file(directory, _CONFIDENCE_NAMES)
            info = find_casefold_file(directory, _INFO_NAMES)
            relative = directory.relative_to(self.root) if directory != self.root else Path(directory.name)
            split = infer_split(Path(self.root.name) / relative)
            sample_id = sanitized_sample_id(relative.parts if directory != self.root else (directory.name,))
            records.append(
                VolumeRecord(
                    sample_id=sample_id,
                    split=split,
                    image_path=image,
                    labels_path=labels,
                    metadata={
                        "confidence_path": str(confidence) if confidence is not None else None,
                        "info_path": str(info) if info is not None else None,
                        "sample_directory": str(directory),
                    },
                )
            )

        if not records:
            raise RuntimeError(
                "No NIS3D sample folders were found. Expected each sample folder "
                "to contain Data.tif plus GroundTruth.tif (or gt.tif)."
            )
        records.sort(key=lambda record: (record.split, record.sample_id))
        return tuple(records)

    def records(self, *, split: str | None = None) -> tuple[VolumeRecord, ...]:
        records = self.discover_records()
        wanted = canonical_split(split)
        if wanted is None:
            return records
        return tuple(record for record in records if record.split == wanted)

    def _spacing_for_record(self, record: VolumeRecord) -> tuple[float, float, float]:
        if self.spacing_override_zyx_um is not None:
            return self.spacing_override_zyx_um

        info_value = record.metadata.get("info_path")
        if info_value:
            info_path = Path(str(info_value))
            if info_path.exists():
                try:
                    parsed = _spacing_from_info_text(info_path.read_text(errors="ignore"))
                except OSError:
                    parsed = None
                if parsed is not None:
                    return parsed

        known = _known_spacing_for_sample(record.sample_id)
        if known is not None:
            return known

        raise ValueError(
            f"Could not determine physical spacing for NIS3D sample {record.sample_id!r}. "
            "Its Info.txt did not contain a parseable XYZ voxel size and this sample "
            "has no documented fallback. Pass --spacing-zyx-um Z Y X explicitly."
        )

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
                f"NIS3D image/GT shape mismatch for {record.sample_id}: "
                f"{image.shape} versus {labels.shape}"
            )

        confidence_path_value = record.metadata.get("confidence_path")
        valid = np.ones(image.shape, dtype=bool)
        confidence_counts: dict[int, int] | None = None
        if confidence_path_value:
            confidence_path = Path(str(confidence_path_value))
            if confidence_path.exists():
                confidence = to_zyx(
                    coerce_3d(load_array(confidence_path), name="confidence score"),
                    self.source_axis_order,
                )
                if confidence.shape != image.shape:
                    raise ValueError(
                        f"NIS3D confidence/image shape mismatch for {record.sample_id}: "
                        f"{confidence.shape} versus {image.shape}"
                    )
                confidence_int = np.rint(confidence).astype(np.int16, copy=False)
                valid = confidence_int != 1
                unique, counts = np.unique(confidence_int, return_counts=True)
                confidence_counts = {
                    int(value): int(count) for value, count in zip(unique, counts)
                }

        spacing = self._spacing_for_record(record)
        bounds = robust_intensity_bounds(image, valid_mask=valid)
        return AnnotatedVolume(
            image=np.asarray(image),
            instance_labels=labels,
            spacing_zyx_um=spacing,
            valid_mask=valid,
            intensity_bounds=bounds,
            dataset_name=self.dataset_name,
            sample_id=record.sample_id,
            split=record.split,
            metadata={
                "image_path": str(record.image_path),
                "labels_path": str(record.labels_path),
                "confidence_path": confidence_path_value,
                "info_path": record.metadata.get("info_path"),
                "confidence_counts": confidence_counts,
                "source": "NIS3D, Zenodo 10.5281/zenodo.11456029",
                "source_axis_order": self.source_axis_order,
            },
        )

    def load_index(self, index: int = 0, *, split: str | None = None) -> AnnotatedVolume:
        records = self.records(split=split)
        if not records:
            raise IndexError(f"no NIS3D records available for split={split!r}")
        return self.load(records[index])


__all__ = ["NIS3DAdapter"]
