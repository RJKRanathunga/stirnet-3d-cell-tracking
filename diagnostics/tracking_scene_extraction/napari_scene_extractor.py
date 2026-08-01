from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QComboBox,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QPlainTextEdit,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    QT_AVAILABLE = True
except ImportError:  # Allows non-GUI tests and syntax validation.
    Qt = None
    QWidget = object
    QT_AVAILABLE = False


SCHEMA_VERSION = 1
BBOX_COLUMNS = (
    "z_min",
    "y_min",
    "x_min",
    "z_max",
    "y_max",
    "x_max",
)
PREVIEW_LAYER_NAME = "Scene Capture Bounds"


class SceneCaptureError(ValueError):
    """Raised when a scene selection cannot be validated or saved."""


def _materialize(array_like: Any) -> np.ndarray:
    """Convert a NumPy, Zarr, or Dask slice into a NumPy array."""
    if hasattr(array_like, "compute"):
        array_like = array_like.compute()
    return np.asarray(array_like)


def _json_ready(value: Any) -> Any:
    """Recursively convert Paths and NumPy values to JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _sanitize_scene_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("._-")
    return name


def _sanitize_array_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip().lower())
    name = name.strip("_-")
    return name or "volume"


def _parse_cell_ids(text: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens:
        raise SceneCaptureError("Enter at least one cell ID.")

    values: list[int] = []
    invalid_tokens: list[str] = []

    for token in tokens:
        try:
            value = int(token)
        except ValueError:
            invalid_tokens.append(token)
            continue

        if value < 0:
            invalid_tokens.append(token)
            continue

        if value not in values:
            values.append(value)

    if invalid_tokens:
        raise SceneCaptureError(
            "Invalid cell ID value(s): " + ", ".join(invalid_tokens)
        )

    return tuple(values)


@dataclass(frozen=True)
class CropBounds:
    start_zyx: tuple[int, int, int]
    stop_zyx: tuple[int, int, int]

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(
            stop - start
            for start, stop in zip(self.start_zyx, self.stop_zyx)
        )

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return tuple(
            slice(start, stop)
            for start, stop in zip(self.start_zyx, self.stop_zyx)
        )


class TrackingSceneCaptureModel:
    """State and extraction logic for manually curated tracking scenes."""

    def __init__(
        self,
        *,
        cells: pd.DataFrame,
        instance_labels_volume: Any,
        binary_mask_volume: Any | None,
        image_volumes: Mapping[str, Any] | None,
        sample_id: str,
        save_root: str | Path,
        voxel_size_zyx: Sequence[float],
        cell_id_column: str = "cell_id",
        frame_column: str = "frame",
        source_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.cells = cells.copy()
        self.instance_labels_volume = instance_labels_volume
        self.binary_mask_volume = binary_mask_volume
        self.image_volumes = dict(image_volumes or {})
        self.sample_id = str(sample_id)
        self.save_root = Path(save_root)
        self.voxel_size_zyx = tuple(float(value) for value in voxel_size_zyx)
        self.cell_id_column = cell_id_column
        self.frame_column = frame_column
        self.source_metadata = dict(source_metadata or {})
        self.selections: dict[int, tuple[int, ...]] = {}
        self._bbox_cache: dict[tuple[int, int], tuple[int, int, int, int, int, int]] = {}

        self._validate_inputs()
        self.save_root.mkdir(parents=True, exist_ok=True)

    @property
    def num_frames(self) -> int:
        return int(self.instance_labels_volume.shape[0])

    @property
    def spatial_shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.instance_labels_volume.shape[-3:])

    def _validate_inputs(self) -> None:
        missing_columns = {
            self.cell_id_column,
            self.frame_column,
        } - set(self.cells.columns)
        if missing_columns:
            raise SceneCaptureError(
                "The cells table is missing required column(s): "
                + ", ".join(sorted(missing_columns))
            )

        if len(self.voxel_size_zyx) != 3:
            raise SceneCaptureError("voxel_size_zyx must contain exactly three values.")

        if getattr(self.instance_labels_volume, "ndim", None) != 4:
            raise SceneCaptureError(
                "instance_labels_volume must have shape (T, Z, Y, X)."
            )

        expected_shape = tuple(self.instance_labels_volume.shape)

        if self.binary_mask_volume is not None:
            if tuple(self.binary_mask_volume.shape) != expected_shape:
                raise SceneCaptureError(
                    "binary_mask_volume must have the same shape as "
                    "instance_labels_volume."
                )

        for name, volume in self.image_volumes.items():
            if tuple(volume.shape) != expected_shape:
                raise SceneCaptureError(
                    f"Image volume '{name}' has shape {tuple(volume.shape)}, "
                    f"expected {expected_shape}."
                )

    def list_categories(self) -> list[str]:
        return sorted(
            (
                path.name
                for path in self.save_root.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ),
            key=str.casefold,
        )

    def record_frame(self, frame: int, cell_ids: Sequence[int]) -> None:
        frame = int(frame)
        if frame < 0 or frame >= self.num_frames:
            raise SceneCaptureError(
                f"Frame {frame} is outside the valid range 0-{self.num_frames - 1}."
            )

        ids = tuple(dict.fromkeys(int(value) for value in cell_ids))
        if not ids:
            raise SceneCaptureError("Enter at least one cell ID.")

        frame_cells = self.cells.loc[
            self.cells[self.frame_column].astype(int) == frame,
            self.cell_id_column,
        ]
        available_ids = set(frame_cells.astype(int).tolist())
        invalid_ids = [cell_id for cell_id in ids if cell_id not in available_ids]

        if invalid_ids:
            raise SceneCaptureError(
                f"Cell ID(s) not found in frame {frame}: "
                + ", ".join(str(value) for value in invalid_ids)
            )

        # Verify that every selected ID has a real instance mask. This catches
        # cells-table/segmentation mismatches before the scene is saved.
        labels_frame = _materialize(self.instance_labels_volume[frame])
        absent_masks = [cell_id for cell_id in ids if not np.any(labels_frame == cell_id)]
        if absent_masks:
            raise SceneCaptureError(
                f"Cell ID(s) have no instance-label voxels in frame {frame}: "
                + ", ".join(str(value) for value in absent_masks)
            )

        self.selections[frame] = ids

    def remove_frame(self, frame: int) -> None:
        self.selections.pop(int(frame), None)

    def clear(self) -> None:
        self.selections.clear()

    def _bbox_from_cells_table(
        self,
        frame: int,
        cell_id: int,
    ) -> tuple[int, int, int, int, int, int] | None:
        if not set(BBOX_COLUMNS).issubset(self.cells.columns):
            return None

        rows = self.cells[
            (self.cells[self.frame_column].astype(int) == int(frame))
            & (self.cells[self.cell_id_column].astype(int) == int(cell_id))
        ]
        if rows.empty:
            return None

        values = rows.iloc[0][list(BBOX_COLUMNS)].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            return None

        z_min, y_min, x_min, z_max, y_max, x_max = values
        bounds = (
            int(np.floor(z_min)),
            int(np.floor(y_min)),
            int(np.floor(x_min)),
            int(np.ceil(z_max)),
            int(np.ceil(y_max)),
            int(np.ceil(x_max)),
        )

        if any(stop <= start for start, stop in zip(bounds[:3], bounds[3:])):
            return None

        return bounds

    def _bbox_from_labels(
        self,
        frame: int,
        cell_id: int,
    ) -> tuple[int, int, int, int, int, int]:
        labels_frame = _materialize(self.instance_labels_volume[int(frame)])
        coordinates = np.argwhere(labels_frame == int(cell_id))
        if coordinates.size == 0:
            raise SceneCaptureError(
                f"Cell ID {cell_id} has no instance-label voxels in frame {frame}."
            )

        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0) + 1
        return (
            int(minimum[0]),
            int(minimum[1]),
            int(minimum[2]),
            int(maximum[0]),
            int(maximum[1]),
            int(maximum[2]),
        )

    def get_cell_bbox(
        self,
        frame: int,
        cell_id: int,
    ) -> tuple[int, int, int, int, int, int]:
        key = (int(frame), int(cell_id))
        if key not in self._bbox_cache:
            bounds = self._bbox_from_cells_table(*key)
            if bounds is None:
                bounds = self._bbox_from_labels(*key)
            self._bbox_cache[key] = bounds
        return self._bbox_cache[key]

    def compute_crop_bounds(
        self,
        padding_zyx: Sequence[int],
    ) -> CropBounds:
        if not self.selections:
            raise SceneCaptureError("No frame selections have been recorded.")

        padding = tuple(int(value) for value in padding_zyx)
        if len(padding) != 3 or any(value < 0 for value in padding):
            raise SceneCaptureError(
                "padding_zyx must contain three non-negative integers."
            )

        all_bounds = [
            self.get_cell_bbox(frame, cell_id)
            for frame, cell_ids in self.selections.items()
            for cell_id in cell_ids
        ]

        starts = np.min(np.asarray([bounds[:3] for bounds in all_bounds]), axis=0)
        stops = np.max(np.asarray([bounds[3:] for bounds in all_bounds]), axis=0)

        spatial_shape = np.asarray(self.spatial_shape_zyx, dtype=int)
        starts = np.maximum(starts - np.asarray(padding, dtype=int), 0)
        stops = np.minimum(stops + np.asarray(padding, dtype=int), spatial_shape)

        return CropBounds(
            start_zyx=tuple(int(value) for value in starts),
            stop_zyx=tuple(int(value) for value in stops),
        )

    def _category_path(self, category: str) -> Path:
        category = category.strip()
        if not category:
            raise SceneCaptureError("Select a scene category.")

        categories = self.list_categories()
        if category not in categories:
            raise SceneCaptureError(
                f"Category '{category}' is not a directory inside {self.save_root}. "
                "Create the directory and refresh the category list."
            )

        return self.save_root / category

    def _next_scene_name(self, category_path: Path) -> str:
        """Return the next automatic numeric scene name: 001, 002, 003, ..."""
        existing_numbers = [
            int(path.name)
            for path in category_path.iterdir()
            if path.is_dir() and path.name.isdigit()
        ]

        next_number = max(existing_numbers, default=0) + 1
        return f"{next_number:03d}"

    def save_scene(
        self,
        *,
        category: str,
        padding_zyx: Sequence[int],
    ) -> Path:
        if not self.selections:
            raise SceneCaptureError("No frame selections have been recorded.")

        category_path = self._category_path(category)
        safe_name = self._next_scene_name(category_path)
        final_path = category_path / safe_name

        crop = self.compute_crop_bounds(padding_zyx)
        frame_start = min(self.selections)
        frame_end = max(self.selections)
        frames = np.arange(frame_start, frame_end + 1, dtype=np.int32)
        missing_frames = [
            int(frame) for frame in frames if int(frame) not in self.selections
        ]

        binary_frames: list[np.ndarray] = []
        label_frames: list[np.ndarray] = []
        image_frames: dict[str, list[np.ndarray]] = {
            name: [] for name in self.image_volumes
        }

        crop_slices = crop.slices
        label_dtype = np.dtype(self.instance_labels_volume.dtype)

        for frame_value in frames:
            frame = int(frame_value)
            selected_ids = self.selections.get(frame, ())

            if selected_ids:
                labels_crop = _materialize(
                    self.instance_labels_volume[(frame, *crop_slices)]
                )

                selected_mask = np.isin(labels_crop, selected_ids)

                if self.binary_mask_volume is not None:
                    source_binary_crop = _materialize(
                        self.binary_mask_volume[(frame, *crop_slices)]
                    ).astype(bool, copy=False)
                    selected_mask &= source_binary_crop

                missing_ids = [
                    cell_id
                    for cell_id in selected_ids
                    if not np.any(labels_crop == cell_id)
                ]
                if missing_ids:
                    raise SceneCaptureError(
                        f"Selected cell ID(s) disappeared from frame {frame}: "
                        + ", ".join(str(value) for value in missing_ids)
                    )

                selected_labels = np.where(selected_mask, labels_crop, 0).astype(
                    label_dtype,
                    copy=False,
                )
            else:
                selected_mask = np.zeros(crop.shape_zyx, dtype=bool)
                selected_labels = np.zeros(crop.shape_zyx, dtype=label_dtype)

            binary_frames.append(selected_mask.astype(bool, copy=False))
            label_frames.append(selected_labels)

            for name, volume in self.image_volumes.items():
                if selected_ids:
                    image_crop = _materialize(volume[(frame, *crop_slices)])
                    masked_image = np.where(selected_mask, image_crop, 0).astype(
                        image_crop.dtype,
                        copy=False,
                    )
                else:
                    masked_image = np.zeros(
                        crop.shape_zyx,
                        dtype=np.dtype(volume.dtype),
                    )
                image_frames[name].append(masked_image)

        temporary_path = category_path / f".{safe_name}.tmp-{uuid.uuid4().hex}"
        temporary_path.mkdir(parents=False, exist_ok=False)

        try:
            np.save(
                temporary_path / "frames.npy",
                frames,
                allow_pickle=False,
            )
            np.save(
                temporary_path / "binary_mask.npy",
                np.stack(binary_frames, axis=0),
                allow_pickle=False,
            )
            np.save(
                temporary_path / "instance_labels.npy",
                np.stack(label_frames, axis=0),
                allow_pickle=False,
            )

            image_files: dict[str, str] = {}
            used_names: set[str] = set()
            for name, arrays in image_frames.items():
                safe_array_name = _sanitize_array_name(name)
                candidate = safe_array_name
                suffix = 2
                while candidate in used_names:
                    candidate = f"{safe_array_name}_{suffix}"
                    suffix += 1
                used_names.add(candidate)

                filename = f"{candidate}.npy"
                np.save(
                    temporary_path / filename,
                    np.stack(arrays, axis=0),
                    allow_pickle=False,
                )
                image_files[name] = filename

            metadata = {
                "schema_version": SCHEMA_VERSION,
                "scene_name": safe_name,
                "category": category,
                "sample_id": self.sample_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "cell_id_column": self.cell_id_column,
                "frame_column": self.frame_column,
                "selected_cells": {
                    str(frame): [int(value) for value in cell_ids]
                    for frame, cell_ids in sorted(self.selections.items())
                },
                "frame_start": int(frame_start),
                "frame_end": int(frame_end),
                "frame_numbers": [int(value) for value in frames],
                "missing_selection_frames": missing_frames,
                "voxel_size_zyx": list(self.voxel_size_zyx),
                "crop_origin_zyx": list(crop.start_zyx),
                "crop_stop_zyx": list(crop.stop_zyx),
                "crop_shape_zyx": list(crop.shape_zyx),
                "padding_zyx": [int(value) for value in padding_zyx],
                "spatial_coordinate_system": (
                    "All saved frames use one fixed crop. Add crop_origin_zyx "
                    "to local ZYX coordinates to recover original-volume coordinates."
                ),
                "files": {
                    "frames": "frames.npy",
                    "binary_mask": "binary_mask.npy",
                    "instance_labels": "instance_labels.npy",
                    "masked_images": image_files,
                },
                "source": self.source_metadata,
            }

            with (temporary_path / "scene.json").open("w", encoding="utf-8") as file:
                json.dump(_json_ready(metadata), file, indent=2)
                file.write("\n")

            temporary_path.rename(final_path)
        except Exception:
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise

        return final_path


class TrackingSceneExtractorWidget(QWidget):
    """Napari dock widget for manual frame-by-frame scene extraction."""

    def __init__(
        self,
        *,
        viewer: Any,
        model: TrackingSceneCaptureModel,
        default_padding_zyx: Sequence[int],
    ) -> None:
        if not QT_AVAILABLE:
            raise ImportError(
                "qtpy is required to create the tracking scene extractor widget."
            )

        super().__init__()
        self.viewer = viewer
        self.model = model
        self.default_padding_zyx = tuple(int(value) for value in default_padding_zyx)
        if len(self.default_padding_zyx) != 3:
            raise SceneCaptureError(
                "default_padding_zyx must contain exactly three integers."
            )

        self._build_ui()
        self._connect_events()
        self.refresh_categories()
        self._update_frame_label()
        self._update_summary()

    def _build_ui(self) -> None:
        root = QVBoxLayout()
        self.setLayout(root)

        self.current_frame_label = QLabel("Current frame: -")
        root.addWidget(self.current_frame_label)

        selection_group = QGroupBox("Frame selection")
        selection_layout = QVBoxLayout()
        selection_group.setLayout(selection_layout)

        self.cell_ids_input = QLineEdit()
        self.cell_ids_input.setPlaceholderText("Cell IDs, for example: 42, 47")
        selection_layout.addWidget(self.cell_ids_input)

        selection_buttons = QHBoxLayout()
        self.save_frame_button = QPushButton("Save frame selection")
        self.remove_frame_button = QPushButton("Remove current frame")
        selection_buttons.addWidget(self.save_frame_button)
        selection_buttons.addWidget(self.remove_frame_button)
        selection_layout.addLayout(selection_buttons)

        self.clear_button = QPushButton("Clear scene selection")
        selection_layout.addWidget(self.clear_button)
        root.addWidget(selection_group)

        crop_group = QGroupBox("Scene crop padding (voxels)")
        crop_layout = QGridLayout()
        crop_group.setLayout(crop_layout)

        self.padding_spins: list[QSpinBox] = []
        for column, (axis, default) in enumerate(
            zip(("Z", "Y", "X"), self.default_padding_zyx)
        ):
            crop_layout.addWidget(QLabel(axis), 0, column)
            spin = QSpinBox()
            spin.setRange(0, 4096)
            spin.setValue(int(default))
            crop_layout.addWidget(spin, 1, column)
            self.padding_spins.append(spin)
        root.addWidget(crop_group)

        save_group = QGroupBox("Save scene")
        save_layout = QFormLayout()
        save_group.setLayout(save_layout)

        category_row = QWidget()
        category_layout = QHBoxLayout()
        category_layout.setContentsMargins(0, 0, 0, 0)
        category_row.setLayout(category_layout)
        self.category_combo = QComboBox()
        self.refresh_categories_button = QPushButton("Refresh")
        category_layout.addWidget(self.category_combo, 1)
        category_layout.addWidget(self.refresh_categories_button)
        save_layout.addRow("Category", category_row)

        self.save_scene_button = QPushButton("Save scene")
        save_layout.addRow(self.save_scene_button)
        root.addWidget(save_group)

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(160)
        root.addWidget(self.summary)

        self.root_path_label = QLabel(f"Save root:\n{self.model.save_root}")
        self.root_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.root_path_label.setWordWrap(True)
        root.addWidget(self.root_path_label)

    def _connect_events(self) -> None:
        self.save_frame_button.clicked.connect(self._save_current_frame)
        self.remove_frame_button.clicked.connect(self._remove_current_frame)
        self.clear_button.clicked.connect(self._clear_scene)
        self.refresh_categories_button.clicked.connect(self.refresh_categories)
        self.save_scene_button.clicked.connect(self._save_scene)

        for spin in self.padding_spins:
            spin.valueChanged.connect(self._selection_changed)

        self.viewer.dims.events.current_step.connect(self._on_dims_changed)

    def _current_frame(self) -> int:
        if self.viewer.dims.ndim < 4:
            raise SceneCaptureError(
                "The viewer must have a leading time axis before using scene capture."
            )
        return int(round(self.viewer.dims.current_step[0]))

    def _padding(self) -> tuple[int, int, int]:
        return tuple(spin.value() for spin in self.padding_spins)

    def _notify(self, level: str, message: str) -> None:
        try:
            from napari.utils.notifications import (
                show_error,
                show_info,
                show_warning,
            )

            functions = {
                "error": show_error,
                "warning": show_warning,
                "info": show_info,
            }
            functions[level](message)
        except Exception:
            print(f"[{level.upper()}] {message}")

    def _on_dims_changed(self, event: Any = None) -> None:
        self._update_frame_label()
        self._load_current_selection_into_input()
        self._update_preview()

    def _update_frame_label(self) -> None:
        try:
            frame = self._current_frame()
            self.current_frame_label.setText(f"Current frame: {frame}")
        except SceneCaptureError:
            self.current_frame_label.setText("Current frame: unavailable")

    def _load_current_selection_into_input(self) -> None:
        try:
            frame = self._current_frame()
        except SceneCaptureError:
            return

        selected_ids = self.model.selections.get(frame)
        if selected_ids is not None:
            self.cell_ids_input.setText(", ".join(map(str, selected_ids)))
        else:
            self.cell_ids_input.clear()

    def _save_current_frame(self) -> None:
        try:
            frame = self._current_frame()
            cell_ids = _parse_cell_ids(self.cell_ids_input.text())
            self.model.record_frame(frame, cell_ids)
        except SceneCaptureError as error:
            self._notify("error", str(error))
            return

        self._notify(
            "info",
            f"Saved frame {frame}: " + ", ".join(map(str, cell_ids)),
        )
        self._selection_changed()

    def _remove_current_frame(self) -> None:
        try:
            frame = self._current_frame()
        except SceneCaptureError as error:
            self._notify("error", str(error))
            return

        if frame not in self.model.selections:
            self._notify("warning", f"Frame {frame} has no saved selection.")
            return

        self.model.remove_frame(frame)
        self.cell_ids_input.clear()
        self._notify("info", f"Removed the saved selection for frame {frame}.")
        self._selection_changed()

    def _clear_scene(self) -> None:
        self.model.clear()
        self.cell_ids_input.clear()
        self._remove_preview_layer()
        self._update_summary()
        self._notify("info", "Cleared the current scene selection.")

    def refresh_categories(self) -> None:
        previous = self.category_combo.currentText().strip()
        categories = self.model.list_categories()

        self.category_combo.blockSignals(True)
        self.category_combo.clear()
        self.category_combo.addItems(categories)
        if previous in categories:
            self.category_combo.setCurrentText(previous)
        self.category_combo.blockSignals(False)

        if not categories:
            self._notify(
                "warning",
                "No category directories were found. Create a subdirectory inside "
                f"{self.model.save_root} and click Refresh.",
            )

    def _save_scene(self) -> None:
        try:
            saved_path = self.model.save_scene(
                category=self.category_combo.currentText(),
                padding_zyx=self._padding(),
            )
        except SceneCaptureError as error:
            self._notify("error", str(error))
            return
        except Exception as error:
            self._notify(
                "error",
                f"Scene extraction failed: {type(error).__name__}: {error}",
            )
            return

        self._notify("info", f"Saved scene to: {saved_path}")

    def _selection_changed(self, *args: Any) -> None:
        self._update_summary()
        self._update_preview()

    def _update_summary(self) -> None:
        if not self.model.selections:
            self.summary.setPlainText("No frames have been recorded.")
            return

        lines = ["Captured frames:"]
        for frame, cell_ids in sorted(self.model.selections.items()):
            lines.append(
                f"  t{frame:03d}: " + ", ".join(str(value) for value in cell_ids)
            )

        try:
            crop = self.model.compute_crop_bounds(self._padding())
            lines.extend(
                [
                    "",
                    "Fixed scene crop:",
                    f"  origin ZYX: {crop.start_zyx}",
                    f"  stop ZYX:   {crop.stop_zyx}",
                    f"  shape ZYX:  {crop.shape_zyx}",
                ]
            )

            frame_start = min(self.model.selections)
            frame_end = max(self.model.selections)
            missing = [
                frame
                for frame in range(frame_start, frame_end + 1)
                if frame not in self.model.selections
            ]
            if missing:
                lines.extend(
                    [
                        "",
                        "Frames without a saved selection:",
                        "  " + ", ".join(map(str, missing)),
                    ]
                )
        except SceneCaptureError as error:
            lines.extend(["", f"Crop error: {error}"])

        self.summary.setPlainText("\n".join(lines))

    def _remove_preview_layer(self) -> None:
        try:
            layer = self.viewer.layers[PREVIEW_LAYER_NAME]
        except Exception:
            return

        try:
            self.viewer.layers.remove(layer)
        except Exception:
            pass

    @staticmethod
    def _box_corners(
        frame: int,
        crop: CropBounds,
    ) -> np.ndarray:
        z0, y0, x0 = crop.start_zyx
        z1, y1, x1 = (value - 1 for value in crop.stop_zyx)
        return np.asarray(
            [
                [frame, z0, y0, x0],
                [frame, z0, y0, x1],
                [frame, z0, y1, x0],
                [frame, z0, y1, x1],
                [frame, z1, y0, x0],
                [frame, z1, y0, x1],
                [frame, z1, y1, x0],
                [frame, z1, y1, x1],
            ],
            dtype=float,
        )

    @staticmethod
    def _box_paths(corners: np.ndarray) -> list[np.ndarray]:
        edge_pairs = (
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
        return [corners[[start, stop]] for start, stop in edge_pairs]

    def _update_preview(self) -> None:
        self._remove_preview_layer()
        if not self.model.selections:
            return

        try:
            crop = self.model.compute_crop_bounds(self._padding())
            frame = self._current_frame()
        except SceneCaptureError:
            return

        corners = self._box_corners(frame, crop)
        paths = self._box_paths(corners)

        try:
            self.viewer.add_shapes(
                paths,
                shape_type=["path"] * len(paths),
                name=PREVIEW_LAYER_NAME,
                scale=(1, *self.model.voxel_size_zyx),
                edge_color="yellow",
                edge_width=2,
            )
        except Exception:
            # A point-corner fallback keeps the preview usable on Napari
            # versions that cannot render 3D path shapes in the current mode.
            self.viewer.add_points(
                corners,
                name=PREVIEW_LAYER_NAME,
                scale=(1, *self.model.voxel_size_zyx),
                size=3,
                face_color="yellow",
            )


def add_tracking_scene_extractor(
    *,
    viewer: Any,
    cells: pd.DataFrame,
    instance_labels_volume: Any,
    binary_mask_volume: Any | None,
    image_volumes: Mapping[str, Any] | None,
    sample_id: str,
    save_root: str | Path,
    voxel_size_zyx: Sequence[float],
    default_padding_zyx: Sequence[int] = (2, 12, 12),
    cell_id_column: str = "cell_id",
    frame_column: str = "frame",
    source_metadata: Mapping[str, Any] | None = None,
    dock_area: str = "right",
) -> TrackingSceneExtractorWidget:
    """Create and dock the manual tracking-scene extraction tool."""
    model = TrackingSceneCaptureModel(
        cells=cells,
        instance_labels_volume=instance_labels_volume,
        binary_mask_volume=binary_mask_volume,
        image_volumes=image_volumes,
        sample_id=sample_id,
        save_root=save_root,
        voxel_size_zyx=voxel_size_zyx,
        cell_id_column=cell_id_column,
        frame_column=frame_column,
        source_metadata=source_metadata,
    )

    widget = TrackingSceneExtractorWidget(
        viewer=viewer,
        model=model,
        default_padding_zyx=default_padding_zyx,
    )

    viewer.window.add_dock_widget(
        widget,
        name="Tracking Scene Extractor",
        area=dock_area,
    )
    return widget


__all__ = [
    "CropBounds",
    "SceneCaptureError",
    "TrackingSceneCaptureModel",
    "TrackingSceneExtractorWidget",
    "add_tracking_scene_extractor",
]
