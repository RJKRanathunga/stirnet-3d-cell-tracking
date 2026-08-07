"""Discovery of complete samples under FULL_DATASET_ROOT/train."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FullDatasetSample:
    sample_id: str
    sample_directory: Path
    zarr_path: Path
    zarr_array_path: Path
    ground_truth_nodes_path: Path | None
    ground_truth_edges_path: Path | None
    readme_path: Path | None

    @property
    def has_ground_truth(self) -> bool:
        return (
            self.ground_truth_nodes_path is not None
            and self.ground_truth_edges_path is not None
        )


def _optional_file(path: Path) -> Path | None:
    return path if path.is_file() else None


def _discover_one(sample_directory: Path) -> FullDatasetSample:
    sample_id = sample_directory.name
    zarr_path = sample_directory / f"{sample_id}.zarr"
    zarr_array_path = zarr_path / "0"

    if not zarr_path.is_dir():
        raise FileNotFoundError(
            f"Sample {sample_id} does not contain {zarr_path.name}: {sample_directory}"
        )
    if not zarr_array_path.exists():
        raise FileNotFoundError(
            f"Sample {sample_id} does not contain Zarr array 0: {zarr_array_path}"
        )

    ground_truth = sample_directory / "ground_truth"
    return FullDatasetSample(
        sample_id=sample_id,
        sample_directory=sample_directory,
        zarr_path=zarr_path,
        zarr_array_path=zarr_array_path,
        ground_truth_nodes_path=_optional_file(
            ground_truth / "ground_truth_nodes.csv"
        ),
        ground_truth_edges_path=_optional_file(
            ground_truth / "ground_truth_edges.csv"
        ),
        readme_path=_optional_file(sample_directory / "README.txt"),
    )


def discover_samples(
    dataset_root: str | Path,
    *,
    train_subdirectory: str = "train",
    sample_ids: tuple[str, ...] = (),
) -> tuple[FullDatasetSample, ...]:
    root = Path(dataset_root).expanduser().resolve()
    train_root = root / train_subdirectory
    if not train_root.is_dir():
        raise FileNotFoundError(
            f"Expected full-dataset training directory does not exist: {train_root}"
        )

    if sample_ids:
        requested = tuple(dict.fromkeys(str(value) for value in sample_ids))
        missing = [
            sample_id
            for sample_id in requested
            if not (train_root / sample_id).is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                "Requested sample directories were not found under "
                f"{train_root}: {missing}"
            )
        directories = [train_root / sample_id for sample_id in requested]
    else:
        directories = [
            path
            for path in sorted(train_root.iterdir(), key=lambda value: value.name)
            if path.is_dir() and not path.name.startswith(".")
        ]

    if not directories:
        raise FileNotFoundError(f"No sample directories were found in: {train_root}")

    return tuple(_discover_one(path) for path in directories)


__all__ = ["FullDatasetSample", "discover_samples"]
