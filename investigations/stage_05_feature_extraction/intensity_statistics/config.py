"""Configuration for the Stage 05 intensity-statistics investigation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


DEFAULT_SAMPLE_ID = "44b6_0113de3b"
DEFAULT_VOXEL_SIZE_ZYX_UM = (1.625, 0.40625, 0.40625)


def sigma_method_name(sigma_um: float) -> str:
    """Return a filesystem- and column-friendly name for one weak Gaussian."""
    value = f"{float(sigma_um):g}".replace(".", "p")
    return f"weak_gaussian_{value}um"


@dataclass(frozen=True)
class InvestigationConfig:
    """Parameters that define one reproducible investigation run."""

    sample_id: str = DEFAULT_SAMPLE_ID
    frame_ids: tuple[int, ...] = tuple(range(20))
    weak_sigmas_um: tuple[float, ...] = (0.2, 0.4)
    voxel_size_zyx_um: tuple[float, float, float] = DEFAULT_VOXEL_SIZE_ZYX_UM
    association_radius_um: float = 12.0
    negatives_per_positive: int = 5
    preprocessing_rtol: float = 1e-5
    preprocessing_atol: float = 1e-6
    allow_preprocessing_mismatch: bool = False
    create_plots: bool = True
    manual_track_ids: tuple[int, ...] = ()
    temporal_features: tuple[str, ...] = (
        "mean",
        "median",
        "std",
        "cv",
        "iqr",
        "p95_p05_range",
    )
    association_features: tuple[str, ...] = ("mean", "std", "cv")

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id cannot be empty")
        if not self.frame_ids:
            raise ValueError("frame_ids cannot be empty")
        if any(frame < 0 for frame in self.frame_ids):
            raise ValueError("frame_ids must be non-negative")
        if tuple(sorted(set(self.frame_ids))) != self.frame_ids:
            raise ValueError("frame_ids must be sorted and unique")
        if any(sigma < 0 for sigma in self.weak_sigmas_um):
            raise ValueError("weak_sigmas_um cannot contain negative values")
        if len(self.voxel_size_zyx_um) != 3 or any(
            value <= 0 for value in self.voxel_size_zyx_um
        ):
            raise ValueError("voxel_size_zyx_um must contain three positive values")
        if self.association_radius_um <= 0:
            raise ValueError("association_radius_um must be positive")
        if self.negatives_per_positive < 1:
            raise ValueError("negatives_per_positive must be at least one")

    @property
    def method_names(self) -> tuple[str, ...]:
        weak = tuple(sigma_method_name(value) for value in self.weak_sigmas_um)
        return (
            "raw",
            *weak,
            "normalized",
            "current_denoised",
            "production_preprocessed",
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_int_list(value: str | None) -> tuple[int, ...]:
    """Parse comma-separated integers."""
    if value is None or not value.strip():
        return ()
    return tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))


def parse_float_list(value: str) -> tuple[float, ...]:
    """Parse comma-separated floating-point values."""
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError("at least one floating-point value is required")
    return values


def parse_frames(value: str | None, available: Iterable[int]) -> tuple[int, ...]:
    """Resolve ``0:20``, ``0,2,5``, or an omitted frame expression."""
    available_ids = tuple(sorted(set(int(frame) for frame in available)))
    if not available_ids:
        raise ValueError("no available frames were found")
    if value is None or not value.strip():
        return available_ids

    text = value.strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError("frame range must be start:stop or start:stop:step")
        start = int(parts[0]) if parts[0] else available_ids[0]
        stop = int(parts[1]) if parts[1] else available_ids[-1] + 1
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        requested = tuple(range(start, stop, step))
    else:
        requested = parse_int_list(text)

    missing = sorted(set(requested).difference(available_ids))
    if missing:
        raise ValueError(f"requested frames are unavailable: {missing}")
    return tuple(sorted(set(requested)))
