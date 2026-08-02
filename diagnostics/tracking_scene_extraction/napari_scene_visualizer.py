from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.io.scene_io import load_tracking_scene as load_scene_data

try:
    from qtpy.QtWidgets import (
        QCheckBox,
        QComboBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    QT_AVAILABLE = True
except ImportError:  # Allows importing the module in non-GUI environments.
    QWidget = object
    QT_AVAILABLE = False


LAYER_PREFIX = "Tracking Scene | "
SUPPORTED_SCHEMA_VERSIONS = {1}


class TrackingSceneVisualizationError(RuntimeError):
    """Raised when a saved tracking scene cannot be loaded or displayed."""


@dataclass(frozen=True)
class TrackingScene:
    """Validated saved tracking-scene data."""

    path: Path
    metadata: Mapping[str, Any]
    frames: np.ndarray
    binary_mask: np.ndarray
    instance_labels: np.ndarray
    masked_images: Mapping[str, np.ndarray]

    @property
    def name(self) -> str:
        return str(self.metadata.get("scene_name", self.path.name))

    @property
    def category(self) -> str:
        return str(self.metadata.get("category", self.path.parent.name))

    @property
    def sample_id(self) -> str:
        return str(self.metadata.get("sample_id", ""))

    @property
    def voxel_size_zyx(self) -> tuple[float, float, float]:
        values = self.metadata.get("voxel_size_zyx", (1.0, 1.0, 1.0))
        if len(values) != 3:
            raise TrackingSceneVisualizationError(
                "scene.json field 'voxel_size_zyx' must contain three values."
            )
        return tuple(float(value) for value in values)

    @property
    def crop_origin_zyx(self) -> tuple[int, int, int]:
        values = self.metadata.get("crop_origin_zyx", (0, 0, 0))
        if len(values) != 3:
            raise TrackingSceneVisualizationError(
                "scene.json field 'crop_origin_zyx' must contain three values."
            )
        return tuple(int(value) for value in values)


def discover_categories(scenes_root: str | Path) -> list[str]:
    """Return category directories inside ``scenes_root``."""
    root = Path(scenes_root)
    if not root.exists():
        return []

    return sorted(
        (
            path.name
            for path in root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=str.casefold,
    )


def discover_scenes(scenes_root: str | Path, category: str) -> list[str]:
    """Return valid scene directory names in one category."""
    category_path = Path(scenes_root) / category
    if not category_path.is_dir():
        return []

    def sort_key(name: str) -> tuple[int, int | str]:
        if name.isdigit():
            return (0, int(name))
        return (1, name.casefold())

    names = [
        path.name
        for path in category_path.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and (path / "scene.json").is_file()
    ]
    return sorted(names, key=sort_key)


def load_tracking_scene(scene_path: str | Path) -> TrackingScene:
    """Load and validate one scene directory created by the extractor."""
    try:
        loaded = load_scene_data(scene_path)
    except Exception as error:
        raise TrackingSceneVisualizationError(
            f"Could not load tracking scene: {error}"
        ) from error

    path = loaded.path
    metadata = loaded.metadata

    schema_version = int(metadata.get("schema_version", -1))
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise TrackingSceneVisualizationError(
            f"Unsupported scene schema version {schema_version}. "
            f"Supported versions: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    expected_shape = metadata.get("crop_shape_zyx")
    if expected_shape is not None:
        expected_shape = tuple(int(value) for value in expected_shape)
        if tuple(loaded.instance_labels.shape[1:]) != expected_shape:
            raise TrackingSceneVisualizationError(
                "Saved volume shape does not match scene.json field 'crop_shape_zyx'."
            )

    return TrackingScene(
        path=path,
        metadata=metadata,
        frames=loaded.frames.astype(np.int64, copy=False),
        binary_mask=loaded.binary_mask.astype(bool, copy=False),
        instance_labels=loaded.instance_labels,
        masked_images=loaded.images,
    )


def _layer_transform(
    scene: TrackingScene,
    *,
    use_original_coordinates: bool,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Return Napari scale and translate values for (T, Z, Y, X)."""
    scale = (1.0, *scene.voxel_size_zyx)

    frame_origin = float(scene.frames[0]) if len(scene.frames) else 0.0
    if use_original_coordinates:
        spatial_translate = tuple(
            origin * spacing
            for origin, spacing in zip(
                scene.crop_origin_zyx,
                scene.voxel_size_zyx,
            )
        )
    else:
        spatial_translate = (0.0, 0.0, 0.0)

    translate = (frame_origin, *spatial_translate)
    return scale, translate


def remove_tracking_scene_layers(viewer: Any) -> None:
    """Remove only layers previously created by this visualizer."""
    for layer in list(viewer.layers):
        if str(layer.name).startswith(LAYER_PREFIX):
            viewer.layers.remove(layer)


def add_scene_to_viewer(
    viewer: Any,
    scene: TrackingScene | str | Path,
    *,
    use_original_coordinates: bool = False,
    remove_existing: bool = True,
) -> list[Any]:
    """Add one saved tracking scene to an existing Napari viewer."""
    if not isinstance(scene, TrackingScene):
        scene = load_tracking_scene(scene)

    if remove_existing:
        remove_tracking_scene_layers(viewer)

    scale, translate = _layer_transform(
        scene,
        use_original_coordinates=use_original_coordinates,
    )

    created_layers: list[Any] = []

    for index, (name, image) in enumerate(scene.masked_images.items()):
        layer = viewer.add_image(
            image,
            name=f"{LAYER_PREFIX}{name}",
            scale=scale,
            translate=translate,
            blending="additive",
            visible=index == 0,
        )
        created_layers.append(layer)

    binary_layer = viewer.add_labels(
        scene.binary_mask.astype(np.uint8, copy=False),
        name=f"{LAYER_PREFIX}Binary mask",
        scale=scale,
        translate=translate,
        visible=False,
    )
    created_layers.append(binary_layer)

    labels_layer = viewer.add_labels(
        scene.instance_labels,
        name=f"{LAYER_PREFIX}Instance labels",
        scale=scale,
        translate=translate,
        visible=True,
    )
    created_layers.append(labels_layer)

    viewer.dims.ndisplay = 3
    if len(scene.frames):
        viewer.dims.set_current_step(0, 0)

    try:
        viewer.reset_view()
    except Exception:
        pass

    return created_layers


class TrackingSceneVisualizerWidget(QWidget):
    """Reusable Napari dock widget for browsing extracted tracking scenes."""

    def __init__(
        self,
        *,
        viewer: Any,
        scenes_root: str | Path,
        use_original_coordinates: bool = False,
    ) -> None:
        if not QT_AVAILABLE:
            raise ImportError(
                "qtpy is required to create the tracking scene visualizer widget."
            )

        super().__init__()
        self.viewer = viewer
        self.scenes_root = Path(scenes_root)
        self.current_scene: TrackingScene | None = None

        self._build_ui(use_original_coordinates)
        self._connect_events()
        self.refresh_categories()

    def _build_ui(self, use_original_coordinates: bool) -> None:
        root = QVBoxLayout()
        self.setLayout(root)

        selection_group = QGroupBox("Tracking scene")
        selection_form = QFormLayout()
        selection_group.setLayout(selection_form)

        self.category_combo = QComboBox()
        selection_form.addRow("Category", self.category_combo)

        self.scene_combo = QComboBox()
        selection_form.addRow("Scene", self.scene_combo)

        self.original_coordinates_checkbox = QCheckBox(
            "Use original volume coordinates"
        )
        self.original_coordinates_checkbox.setChecked(use_original_coordinates)
        selection_form.addRow(self.original_coordinates_checkbox)

        root.addWidget(selection_group)

        button_row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.load_button = QPushButton("Load scene")
        button_row.addWidget(self.refresh_button)
        button_row.addWidget(self.load_button)
        root.addLayout(button_row)

        self.clear_button = QPushButton("Clear scene layers")
        root.addWidget(self.clear_button)

        info_group = QGroupBox("Scene information")
        info_layout = QVBoxLayout()
        info_group.setLayout(info_layout)
        self.info_label = QLabel("No scene loaded.")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        root.addWidget(info_group)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        root.addStretch(1)

    def _connect_events(self) -> None:
        self.category_combo.currentTextChanged.connect(self.refresh_scenes)
        self.refresh_button.clicked.connect(self.refresh_categories)
        self.load_button.clicked.connect(self.load_selected_scene)
        self.clear_button.clicked.connect(self.clear_scene)
        self.original_coordinates_checkbox.toggled.connect(
            self._reload_current_scene
        )

    def _show_error(self, message: str) -> None:
        self.status_label.setText(f"Error: {message}")
        QMessageBox.critical(self, "Tracking scene visualizer", message)

    def refresh_categories(self) -> None:
        previous = self.category_combo.currentText()
        categories = discover_categories(self.scenes_root)

        self.category_combo.blockSignals(True)
        self.category_combo.clear()
        self.category_combo.addItems(categories)
        if previous in categories:
            self.category_combo.setCurrentText(previous)
        self.category_combo.blockSignals(False)

        self.refresh_scenes(self.category_combo.currentText())

        if categories:
            self.status_label.setText(
                f"Found {len(categories)} categor{'y' if len(categories) == 1 else 'ies'}."
            )
        else:
            self.status_label.setText(
                f"No category directories found in {self.scenes_root}."
            )

    def refresh_scenes(self, category: str | None = None) -> None:
        category = category or self.category_combo.currentText()
        previous = self.scene_combo.currentText()
        scenes = discover_scenes(self.scenes_root, category) if category else []

        self.scene_combo.clear()
        self.scene_combo.addItems(scenes)
        if previous in scenes:
            self.scene_combo.setCurrentText(previous)

        self.load_button.setEnabled(bool(scenes))

    def selected_scene_path(self) -> Path:
        category = self.category_combo.currentText().strip()
        scene_name = self.scene_combo.currentText().strip()
        if not category or not scene_name:
            raise TrackingSceneVisualizationError(
                "Select a category and scene before loading."
            )
        return self.scenes_root / category / scene_name

    def load_selected_scene(self) -> None:
        try:
            scene = load_tracking_scene(self.selected_scene_path())
            add_scene_to_viewer(
                self.viewer,
                scene,
                use_original_coordinates=(
                    self.original_coordinates_checkbox.isChecked()
                ),
                remove_existing=True,
            )
        except Exception as error:
            self._show_error(str(error))
            return

        self.current_scene = scene
        self._update_info(scene)
        self.status_label.setText(
            f"Loaded {scene.category}/{scene.name}."
        )

    def _reload_current_scene(self, _checked: bool) -> None:
        if self.current_scene is None:
            return
        try:
            add_scene_to_viewer(
                self.viewer,
                self.current_scene,
                use_original_coordinates=(
                    self.original_coordinates_checkbox.isChecked()
                ),
                remove_existing=True,
            )
        except Exception as error:
            self._show_error(str(error))

    def clear_scene(self) -> None:
        remove_tracking_scene_layers(self.viewer)
        self.current_scene = None
        self.info_label.setText("No scene loaded.")
        self.status_label.setText("Scene layers cleared.")

    def _update_info(self, scene: TrackingScene) -> None:
        selected_cells = scene.metadata.get("selected_cells", {})
        selected_count = sum(len(values) for values in selected_cells.values())
        first_frame = int(scene.frames[0]) if len(scene.frames) else "-"
        last_frame = int(scene.frames[-1]) if len(scene.frames) else "-"
        shape = tuple(int(value) for value in scene.instance_labels.shape[1:])

        self.info_label.setText(
            f"Category: {scene.category}\n"
            f"Scene: {scene.name}\n"
            f"Sample: {scene.sample_id}\n"
            f"Frames: {first_frame}–{last_frame} ({len(scene.frames)})\n"
            f"Selected cell entries: {selected_count}\n"
            f"Crop shape ZYX: {shape}\n"
            f"Crop origin ZYX: {scene.crop_origin_zyx}\n"
            f"Voxel size ZYX: {scene.voxel_size_zyx}"
        )


def add_tracking_scene_visualizer(
    viewer: Any,
    scenes_root: str | Path,
    *,
    dock_area: str = "right",
    use_original_coordinates: bool = False,
    widget_name: str = "Tracking Scene Visualizer",
) -> TrackingSceneVisualizerWidget:
    """Create and dock the reusable scene visualizer in a Napari viewer."""
    widget = TrackingSceneVisualizerWidget(
        viewer=viewer,
        scenes_root=scenes_root,
        use_original_coordinates=use_original_coordinates,
    )
    viewer.window.add_dock_widget(
        widget,
        area=dock_area,
        name=widget_name,
    )
    return widget


__all__ = [
    "TrackingScene",
    "TrackingSceneVisualizationError",
    "TrackingSceneVisualizerWidget",
    "add_scene_to_viewer",
    "add_tracking_scene_visualizer",
    "discover_categories",
    "discover_scenes",
    "load_tracking_scene",
    "remove_tracking_scene_layers",
]
