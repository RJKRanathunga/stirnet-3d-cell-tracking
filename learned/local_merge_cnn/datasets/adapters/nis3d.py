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

# Trusted values from the bundled NIS3D Info.txt files / documentation.
_KNOWN_SPACING_ZYX_UM: dict[str, tuple[float, float, float]] = {
    "drosophila1": (1.0, 1.0, 1.0),
    "drosophila2": (1.0, 1.0, 1.0),
    "musmusculus1": (1.0, 1.0, 1.0),
    "musmusculus2": (1.0, 1.0, 1.0),
    "zebrafish1": (2.5, 0.43, 0.43),
    "zebrafish2": (1.0, 1.0, 1.0),
}
_PRIMARY_ORDER = {
    "drosophila1": 0,
    "drosophila2": 1,
    "musmusculus1": 2,
    "musmusculus2": 3,
    "zebrafish1": 4,
    "zebrafish2": 5,
}

_DATA_NAMES = ("Data.tif", "data.tif", "Data.tiff", "data.tiff")
_GT_NAMES = (
    "GroundTruth.tif", "groundtruth.tif", "ground_truth.tif", "gt.tif",
    "GroundTruth.tiff", "gt.tiff",
)
_CONFIDENCE_NAMES = (
    "ConfidenceScore.tif", "confidencescore.tif", "scoreOfConfidence.tif",
    "scoreofconfidence.tif", "confidence.tif", "ConfidenceScore.tiff",
)
_INFO_NAMES = ("Info.txt", "info.txt")


def _compact_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _parse_length_um(value: str, unit: str | None) -> float:
    number = float(value)
    normalized = (unit or "um").lower().replace("μ", "u").replace("µ", "u")
    if normalized in {"um", "u m"}:
        return number
    if normalized == "nm":
        return number / 1000.0
    if normalized == "mm":
        return number * 1000.0
    raise ValueError(f"unsupported spacing unit {unit!r}")


def _spacing_from_info_text(text: str) -> tuple[float, float, float] | None:
    """Parse the explicit NIS3D Resolution field; never scrape prose numbers."""
    normalized = text.replace("μ", "u").replace("µ", "u")
    lines = normalized.splitlines()

    # First preference: a line immediately following a Resolution: header.
    candidates: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.fullmatch(r"resolution\s*:\s*", stripped, flags=re.I):
            if i + 1 < len(lines):
                candidates.append(lines[i + 1].strip())
        match = re.match(r"^\s*resolution\s*:\s*(.+?)\s*$", line, flags=re.I)
        if match and match.group(1):
            candidates.append(match.group(1).strip())

    # Secondary explicit form: "voxel size is A x B x C".
    for line in lines:
        match = re.search(
            r"voxel\s+size\s+(?:is|=|:)\s*([^.;\n]+)",
            line,
            flags=re.I,
        )
        if match:
            candidates.append(match.group(1).strip())

    triple_pattern = re.compile(
        r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(nm|um|mm)?\s*[x×]\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*(nm|um|mm)?\s*[x×]\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*(nm|um|mm)?\s*$",
        flags=re.I,
    )
    for candidate in candidates:
        match = triple_pattern.match(candidate)
        if not match:
            continue
        x = _parse_length_um(match.group(1), match.group(2))
        y = _parse_length_um(match.group(3), match.group(4))
        z = _parse_length_um(match.group(5), match.group(6))
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
                f"NIS3D dataset directory does not exist: {self.root}. Pass --root or set NIS3D_DIR."
            )
        candidate_dirs = [self.root]
        candidate_dirs.extend(path for path in self.root.rglob("*") if path.is_dir())
        records: list[VolumeRecord] = []
        seen_images: set[Path] = set()
        for directory in candidate_dirs:
            compact_path = _compact_name(str(directory))
            if "suggestivesplitting" in compact_path:
                continue
            image = find_casefold_file(directory, _DATA_NAMES)
            labels = find_casefold_file(directory, _GT_NAMES)
            if image is None or labels is None or image in seen_images:
                continue
            seen_images.add(image)
            confidence = find_casefold_file(directory, _CONFIDENCE_NAMES)
            info = find_casefold_file(directory, _INFO_NAMES)
            relative = directory.relative_to(self.root) if directory != self.root else Path(directory.name)
            sample_id = directory.name if _known_spacing_for_sample(directory.name) else sanitized_sample_id(relative.parts)
            records.append(
                VolumeRecord(
                    sample_id=sample_id,
                    split=infer_split(Path(self.root.name) / relative),
                    image_path=image,
                    labels_path=labels,
                    metadata={
                        "confidence_path": str(confidence) if confidence else None,
                        "info_path": str(info) if info else None,
                        "sample_directory": str(directory),
                    },
                )
            )
        if not records:
            raise RuntimeError("No NIS3D sample folders found containing Data.tif and GT labels")

        primary = [r for r in records if _known_spacing_for_sample(r.sample_id) is not None]
        if primary:
            records = primary
        records.sort(
            key=lambda r: (
                _PRIMARY_ORDER.get(_compact_name(r.sample_id), 999),
                r.sample_id.lower(),
            )
        )
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
            path = Path(str(info_value))
            if path.exists():
                try:
                    parsed = _spacing_from_info_text(path.read_text(errors="ignore"))
                except OSError:
                    parsed = None
                if parsed is not None:
                    return parsed
        known = _known_spacing_for_sample(record.sample_id)
        if known is not None:
            return known
        raise ValueError(
            f"Could not determine spacing for NIS3D sample {record.sample_id!r}; "
            "pass --spacing-zyx-um Z Y X explicitly."
        )

    def load(self, record: VolumeRecord) -> AnnotatedVolume:
        image = to_zyx(coerce_3d(load_array(record.image_path), name="image"), self.source_axis_order)
        labels = normalize_instance_labels(
            to_zyx(coerce_3d(load_array(record.labels_path), name="instance labels"), self.source_axis_order)
        )
        if image.shape != labels.shape:
            raise ValueError(f"NIS3D image/GT shape mismatch: {image.shape} vs {labels.shape}")
        valid = np.ones(image.shape, dtype=bool)
        confidence_counts = None
        confidence_value = record.metadata.get("confidence_path")
        if confidence_value:
            path = Path(str(confidence_value))
            if path.exists():
                confidence = to_zyx(coerce_3d(load_array(path), name="confidence"), self.source_axis_order)
                if confidence.shape != image.shape:
                    raise ValueError("NIS3D confidence/image shape mismatch")
                confidence_int = np.rint(confidence).astype(np.int16, copy=False)
                valid = confidence_int != 1
                unique, counts = np.unique(confidence_int, return_counts=True)
                confidence_counts = {int(v): int(c) for v, c in zip(unique, counts)}
        spacing = self._spacing_for_record(record)
        return AnnotatedVolume(
            image=np.asarray(image),
            instance_labels=labels,
            spacing_zyx_um=spacing,
            valid_mask=valid,
            intensity_bounds=robust_intensity_bounds(image, valid_mask=valid),
            dataset_name=self.dataset_name,
            sample_id=record.sample_id,
            split=record.split,
            metadata={
                **dict(record.metadata),
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


__all__ = ["NIS3DAdapter", "_spacing_from_info_text"]
