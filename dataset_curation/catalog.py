from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dataset_curation.config import biohub_data_root
from dataset_curation.paths import (
    BioHubVolumePaths,
    VALID_SPLITS,
    validate_split,
)


def _frame_count_from_metadata(zarr: Path) -> int | None:
    """Read Zarr metadata only. README.txt is deliberately never consulted."""
    candidates = (
        zarr / "0" / "zarr.json",
        zarr / "0" / ".zarray",
    )

    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            shape = payload.get("shape")
            if (
                isinstance(shape, (list, tuple))
                and len(shape) >= 1
                and int(shape[0]) > 0
            ):
                return int(shape[0])
        except Exception:
            continue

    # Fallback for the observed BioHub Zarr v3 chunk hierarchy:
    # <id>.zarr/0/c/<time-chunk>/...
    chunk_root = zarr / "0" / "c"
    if chunk_root.is_dir():
        numeric = [
            child
            for child in chunk_root.iterdir()
            if child.is_dir() and child.name.isdigit()
        ]
        if numeric:
            return len(numeric)

    return None


@dataclass(frozen=True)
class VolumeRecord:
    volume_id: str
    split: str
    paths: BioHubVolumePaths
    frame_count: int | None

    @property
    def has_ground_truth(self) -> bool:
        # Sparse GT presence is metadata only. It does not affect inference
        # completeness or annotation selection.
        return self.paths.has_ground_truth_files()


class BioHubCatalog:
    def __init__(
        self,
        data_root: str | Path | None = None,
    ) -> None:
        self.data_root = biohub_data_root(data_root)

    def validate_root(self) -> None:
        source = self.data_root / "source"
        if not source.is_dir():
            raise FileNotFoundError(
                "BioHub source directory does not exist:\n"
                f"  {source}\n"
                r"Expected the external-drive root E:\data\biohub"
            )

    def ensure_output_roots(self) -> None:
        (self.data_root / "preprocessed").mkdir(
            parents=True,
            exist_ok=True,
        )
        (self.data_root / "annotations").mkdir(
            parents=True,
            exist_ok=True,
        )

    def discover(
        self,
        split: str,
    ) -> list[VolumeRecord]:
        self.validate_root()
        split = validate_split(split)
        source_root = self.data_root / "source" / split

        if not source_root.is_dir():
            return []

        records: list[VolumeRecord] = []

        # Only sample directories containing <id>/<id>.zarr/0 are accepted.
        # README.txt and unrelated files are therefore ignored by construction.
        for sample_dir in sorted(
            (path for path in source_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            volume_id = sample_dir.name
            paths = BioHubVolumePaths(
                self.data_root,
                split,
                volume_id,
            )

            if not paths.zarr_array.is_dir():
                continue

            records.append(
                VolumeRecord(
                    volume_id=volume_id,
                    split=split,
                    paths=paths,
                    frame_count=_frame_count_from_metadata(paths.zarr),
                )
            )

        return records

    def discover_all(self) -> list[VolumeRecord]:
        result: list[VolumeRecord] = []
        for split in VALID_SPLITS:
            result.extend(self.discover(split))
        return result

    def get(
        self,
        volume_id: str,
        *,
        split: str,
    ) -> VolumeRecord:
        split = validate_split(split)
        volume_id = str(volume_id).strip()

        for record in self.discover(split):
            if record.volume_id == volume_id:
                return record

        raise KeyError(
            f"Volume {volume_id!r} was not found below "
            f"{self.data_root / 'source' / split}"
        )
