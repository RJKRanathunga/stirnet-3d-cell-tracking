"""Interactive Napari widget for previewing and saving 3D cell volumes.

The widget reads the current Napari time frame, looks up a user-entered cell
ID, draws the exact extraction boundary as a red 3D wireframe, and saves the
volume when the user clicks the button or presses ``E`` while the canvas has
focus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    from .extraction import (
        BOX_SIZE,
        OUTPUT_DIR,
        PAD_VALUE,
        VOXEL_SIZE,
        calculate_box_bounds,
        find_cell,
        save_cell_extraction,
        validate_box_size,
        validate_cell_table,
        validate_image_volume,
    )
except ImportError:  # Allows direct execution during development.
    from extraction import (  # type: ignore
        BOX_SIZE,
        OUTPUT_DIR,
        PAD_VALUE,
        VOXEL_SIZE,
        calculate_box_bounds,
        find_cell,
        save_cell_extraction,
        validate_box_size,
        validate_cell_table,
        validate_image_volume,
    )


BOX_LAYER_NAME = "Cell extraction box"


def make_box_wireframe(
    frame: int,
    requested_start_zyx: Sequence[int],
    requested_stop_zyx: Sequence[int],
) -> list[np.ndarray]:
    """Return 12 four-dimensional line segments for a voxel-aligned cuboid.

    The crop contains voxel indices ``start`` through ``stop - 1``. Therefore,
    the visual boundaries are placed half a voxel outside those voxel centres.
    Each line uses ``(T, Z, Y, X)`` coordinates.
    """

    start = np.asarray(requested_start_zyx, dtype=float) - 0.5
    stop = np.asarray(requested_stop_zyx, dtype=float) - 0.5

    if start.shape != (3,) or stop.shape != (3,):
        raise ValueError("Box start and stop must both contain (Z, Y, X).")

    z0, y0, x0 = start
    z1, y1, x1 = stop

    corners_zyx = np.asarray(
        [
            [z0, y0, x0],
            [z0, y0, x1],
            [z0, y1, x0],
            [z0, y1, x1],
            [z1, y0, x0],
            [z1, y0, x1],
            [z1, y1, x0],
            [z1, y1, x1],
        ],
        dtype=float,
    )

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

    time_value = float(frame)
    edges: list[np.ndarray] = []

    for first, second in edge_pairs:
        edges.append(
            np.asarray(
                [
                    [time_value, *corners_zyx[first]],
                    [time_value, *corners_zyx[second]],
                ],
                dtype=float,
            )
        )

    return edges


class CellVolumeExtractorWidget(QWidget):
    """Dock widget for interactive cell-volume collection."""

    def __init__(
        self,
        *,
        viewer: Any,
        cells: pd.DataFrame,
        image_volume: Any,
        sample_id: str,
        output_dir: Path | str = OUTPUT_DIR,
        voxel_size_zyx: Sequence[float] = VOXEL_SIZE,
        default_box_size: Sequence[int] = BOX_SIZE,
        pad_value: int | float = PAD_VALUE,
        source_cells_dir: Path | str | None = None,
        source_zarr_array: Path | str | None = None,
        time_axis: int = 0,
        extract_key: str = "E",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        validate_cell_table(cells)
        validate_image_volume(image_volume)

        self.viewer = viewer
        self.cells = cells
        self.image_volume = image_volume
        self.sample_id = str(sample_id)
        self.output_dir = Path(output_dir)
        self.voxel_size_zyx = tuple(float(value) for value in voxel_size_zyx)
        self.pad_value = pad_value
        self.source_cells_dir = (
            Path(source_cells_dir) if source_cells_dir is not None else None
        )
        self.source_zarr_array = (
            Path(source_zarr_array) if source_zarr_array is not None else None
        )
        self.time_axis = int(time_axis)
        self.extract_key = str(extract_key)

        if len(self.voxel_size_zyx) != 3:
            raise ValueError("voxel_size_zyx must contain (Z, Y, X).")
        if not 0 <= self.time_axis < len(self.viewer.dims.current_step):
            raise IndexError(
                f"time_axis {self.time_axis} is invalid for viewer dimensions "
                f"{self.viewer.dims.current_step}."
            )

        self._box_layer = None
        self._preview_signature: tuple[int, int, tuple[int, int, int]] | None = None
        self._updating = False
        self._last_frame = self.current_frame()

        self._build_ui(validate_box_size(default_box_size))
        self._connect_events()
        self._bind_extract_key()
        self._update_frame_label()
        self._set_status(
            f"Go to a frame, enter a cell ID, and preview the box. "
            f"Press {self.extract_key} to extract."
        )

    # --------------------------------------------------------
    # UI construction
    # --------------------------------------------------------

    def _build_ui(self, default_box_size: np.ndarray) -> None:
        layout = QVBoxLayout(self)

        self.frame_label = QLabel()
        self.frame_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.frame_label)

        form = QFormLayout()

        self.cell_id_spin = QSpinBox()
        self.cell_id_spin.setRange(0, 2_147_483_647)
        self.cell_id_spin.setKeyboardTracking(False)
        self.cell_id_spin.setValue(1)
        form.addRow("Cell ID", self.cell_id_spin)

        self.z_size_spin = self._make_size_spin(int(default_box_size[0]))
        self.y_size_spin = self._make_size_spin(int(default_box_size[1]))
        self.x_size_spin = self._make_size_spin(int(default_box_size[2]))

        form.addRow("Box depth Z", self.z_size_spin)
        form.addRow("Box height Y", self.y_size_spin)
        form.addRow("Box width X", self.x_size_spin)

        layout.addLayout(form)

        self.auto_update_checkbox = QCheckBox("Update preview when box size changes")
        self.auto_update_checkbox.setChecked(True)
        layout.addWidget(self.auto_update_checkbox)

        self.overwrite_checkbox = QCheckBox("Overwrite an existing extraction")
        self.overwrite_checkbox.setChecked(False)
        layout.addWidget(self.overwrite_checkbox)

        button_row = QHBoxLayout()
        self.preview_button = QPushButton("Preview box")
        self.extract_button = QPushButton(f"Extract ({self.extract_key})")
        self.clear_button = QPushButton("Clear")

        button_row.addWidget(self.preview_button)
        button_row.addWidget(self.extract_button)
        button_row.addWidget(self.clear_button)
        layout.addLayout(button_row)

        output_label = QLabel(f"Output: {self.output_dir}")
        output_label.setWordWrap(True)
        output_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(output_label)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    @staticmethod
    def _make_size_spin(value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(1, 100_001)
        spin.setSingleStep(2)
        spin.setKeyboardTracking(False)
        spin.setValue(value)
        return spin

    def _connect_events(self) -> None:
        self.preview_button.clicked.connect(self.preview_box)
        self.extract_button.clicked.connect(self.extract_current_cell)
        self.clear_button.clicked.connect(self.clear_preview)
        self.cell_id_spin.valueChanged.connect(self._on_cell_id_changed)

        for spin in (self.z_size_spin, self.y_size_spin, self.x_size_spin):
            spin.valueChanged.connect(self._on_box_size_changed)

        try:
            self.viewer.dims.events.current_step.connect(self._on_frame_changed)
        except AttributeError:
            # The widget remains fully usable; the frame label is refreshed when
            # preview or extraction is requested.
            pass

    def _bind_extract_key(self) -> None:
        def extract_from_key(_viewer: Any) -> None:
            self.extract_current_cell()

        self._extract_key_callback = extract_from_key
        self.viewer.bind_key(
            self.extract_key,
            self._extract_key_callback,
            overwrite=True,
        )

    # --------------------------------------------------------
    # Current values
    # --------------------------------------------------------

    def current_frame(self) -> int:
        return int(self.viewer.dims.current_step[self.time_axis])

    def current_box_size(self) -> tuple[int, int, int]:
        size = validate_box_size(
            (
                self.z_size_spin.value(),
                self.y_size_spin.value(),
                self.x_size_spin.value(),
            )
        )
        return tuple(int(value) for value in size)

    def _current_signature(self) -> tuple[int, int, tuple[int, int, int]]:
        return (
            self.current_frame(),
            int(self.cell_id_spin.value()),
            self.current_box_size(),
        )

    def _update_frame_label(self) -> None:
        frame = self.current_frame()
        maximum = int(self.image_volume.shape[0]) - 1
        self.frame_label.setText(f"Current frame: {frame} / {maximum}")

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    def preview_box(self, *_args: Any, report_errors: bool = True) -> bool:
        """Draw the exact requested extraction box in red."""

        try:
            self._update_frame_label()
            frame, cell_id, box_size = self._current_signature()
            row = find_cell(self.cells, frame=frame, cell_id=cell_id)
            centroid_zyx = row[
                ["centroid_z", "centroid_y", "centroid_x"]
            ].to_numpy(dtype=float)

            bounds = calculate_box_bounds(
                centroid_zyx=centroid_zyx,
                box_size=box_size,
                spatial_shape_zyx=self.image_volume.shape[1:],
            )

            edges = make_box_wireframe(
                frame=frame,
                requested_start_zyx=bounds["requested_start_zyx"],
                requested_stop_zyx=bounds["requested_stop_zyx"],
            )

            self._remove_box_layer()
            self._box_layer = self.viewer.add_shapes(
                edges,
                shape_type="line",
                name=BOX_LAYER_NAME,
                edge_color="red",
                edge_width=3,
                opacity=1.0,
                scale=(1.0, *self.voxel_size_zyx),
            )

            try:
                self._box_layer.mode = "pan_zoom"
            except (AttributeError, ValueError):
                pass

            self._preview_signature = (frame, cell_id, box_size)
            padding_before = bounds["padding_before_zyx"]
            padding_after = bounds["padding_after_zyx"]
            padding_used = any(padding_before) or any(padding_after)
            padding_message = " Boundary padding will be used." if padding_used else ""

            self._set_status(
                f"Previewing frame {frame}, cell {cell_id}, box {box_size}."
                f"{padding_message}"
            )
            return True

        except Exception as error:
            self._preview_signature = None
            self._remove_box_layer()
            if report_errors:
                self._set_error(error)
            return False

    def clear_preview(self, *_args: Any, update_status: bool = True) -> None:
        self._remove_box_layer()
        self._preview_signature = None
        if update_status:
            self._set_status("Extraction preview cleared.")

    def _remove_box_layer(self) -> None:
        if self._box_layer is None:
            return

        try:
            self.viewer.layers.remove(self._box_layer)
        except (ValueError, KeyError):
            pass
        finally:
            self._box_layer = None

    # --------------------------------------------------------
    # Extraction
    # --------------------------------------------------------

    def extract_current_cell(self, *_args: Any) -> None:
        """Extract the current frame/cell/box and save it to disk."""

        try:
            signature = self._current_signature()
            if self._preview_signature != signature:
                if not self.preview_box(report_errors=True):
                    return

            frame, cell_id, box_size = signature
            source_cell_file = self._source_cell_file(frame)

            volume_path, metadata_path, crop, _metadata = save_cell_extraction(
                image_volume=self.image_volume,
                cells=self.cells,
                sample_id=self.sample_id,
                frame=frame,
                cell_id=cell_id,
                box_size=box_size,
                output_dir=self.output_dir,
                voxel_size_zyx=self.voxel_size_zyx,
                pad_value=self.pad_value,
                overwrite=self.overwrite_checkbox.isChecked(),
                source_cell_file=source_cell_file,
                source_zarr_array=self.source_zarr_array,
            )

            message = (
                f"Saved frame {frame}, cell {cell_id}: {volume_path.name} "
                f"with shape {crop.shape}. Metadata: {metadata_path.name}"
            )
            self._set_status(message)
            self.viewer.status = message

        except Exception as error:
            self._set_error(error)

    def _source_cell_file(self, frame: int) -> Path | None:
        if self.source_cells_dir is None:
            return None

        files = sorted(self.source_cells_dir.glob("t*.csv"))
        if 0 <= int(frame) < len(files):
            return files[int(frame)]
        return None

    # --------------------------------------------------------
    # Events and status
    # --------------------------------------------------------

    def _on_frame_changed(self, _event: Any = None) -> None:
        if self._updating:
            return

        frame = self.current_frame()
        self._update_frame_label()
        if frame == self._last_frame:
            return

        self._last_frame = frame
        if self._preview_signature is not None:
            self.clear_preview(update_status=False)
            self._set_status("Frame changed. Enter/confirm the cell ID and preview again.")

    def _on_cell_id_changed(self, _value: int) -> None:
        if self._preview_signature is not None:
            self.clear_preview(update_status=False)
            self._set_status("Cell ID changed. Click Preview box.")

    def _on_box_size_changed(self, _value: int) -> None:
        if (
            self._preview_signature is not None
            and self.auto_update_checkbox.isChecked()
        ):
            self.preview_box(report_errors=False)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _set_error(self, error: Exception) -> None:
        message = f"{type(error).__name__}: {error}"
        self.status_label.setText(message)
        self.viewer.status = message


def add_cell_volume_extractor(
    *,
    viewer: Any,
    cells: pd.DataFrame,
    image_volume: Any,
    sample_id: str,
    output_dir: Path | str = OUTPUT_DIR,
    voxel_size_zyx: Sequence[float] = VOXEL_SIZE,
    default_box_size: Sequence[int] = BOX_SIZE,
    pad_value: int | float = PAD_VALUE,
    source_cells_dir: Path | str | None = None,
    source_zarr_array: Path | str | None = None,
    time_axis: int = 0,
    extract_key: str = "E",
    dock_area: str = "right",
) -> CellVolumeExtractorWidget:
    """Create and dock the interactive extractor in an existing viewer."""

    widget = CellVolumeExtractorWidget(
        viewer=viewer,
        cells=cells,
        image_volume=image_volume,
        sample_id=sample_id,
        output_dir=output_dir,
        voxel_size_zyx=voxel_size_zyx,
        default_box_size=default_box_size,
        pad_value=pad_value,
        source_cells_dir=source_cells_dir,
        source_zarr_array=source_zarr_array,
        time_axis=time_axis,
        extract_key=extract_key,
    )

    viewer.window.add_dock_widget(
        widget,
        name="Cell Volume Extraction",
        area=dock_area,
    )

    return widget
