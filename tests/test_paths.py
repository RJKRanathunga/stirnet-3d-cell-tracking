"""Focused tests for packaging-independent project-root discovery."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.io.paths as paths_module
from src.io import PipelinePaths, find_project_root


ENV_NAME = "CELL_TRACKING_PROJECT_ROOT"


def make_project_root(parent: Path, name: str = "project") -> Path:
    root = parent / name
    (root / "src").mkdir(parents=True)
    (root / "notebooks").mkdir()
    (root / "pyproject.toml").touch()
    return root.resolve()


class ProjectRootTests(unittest.TestCase):
    def test_no_argument_uses_installed_module_repository(self) -> None:
        expected = Path(paths_module.__file__).resolve().parents[2]
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV_NAME, None)
            self.assertEqual(find_project_root(), expected)

    def test_valid_environment_override_is_respected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_project_root(Path(directory))
            with patch.dict(os.environ, {ENV_NAME: str(root)}):
                self.assertEqual(find_project_root(), root)

    def test_invalid_environment_override_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory).resolve()
            with patch.dict(os.environ, {ENV_NAME: str(invalid)}):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"CELL_TRACKING_PROJECT_ROOT environment override.*src, notebooks, pyproject.toml",
                ):
                    find_project_root()

    def test_empty_environment_override_does_not_fall_back(self) -> None:
        with patch.dict(os.environ, {ENV_NAME: ""}):
            with self.assertRaisesRegex(RuntimeError, r"environment override: <empty>"):
                find_project_root()

    def test_explicit_nested_directory_searches_upward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_project_root(Path(directory))
            nested = root / "notebooks" / "diagnostics" / "case"
            nested.mkdir(parents=True)
            self.assertEqual(find_project_root(nested), root)

    def test_explicit_file_searches_from_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_project_root(Path(directory))
            notebook = root / "notebooks" / "example.ipynb"
            notebook.touch()
            self.assertEqual(find_project_root(notebook), root)

    def test_invalid_explicit_input_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory).resolve()
            with self.assertRaisesRegex(
                RuntimeError,
                r"explicit input.*src, notebooks, pyproject.toml",
            ):
                find_project_root(invalid)

    def test_pipeline_paths_discover_without_argument(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV_NAME, None)
            paths = PipelinePaths.discover()
        self.assertEqual(paths.project_root, find_project_root(paths_module.__file__))

    def test_pipeline_paths_explicit_root_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_project_root(Path(directory))
            self.assertEqual(PipelinePaths.discover(root).project_root, root)
            self.assertEqual(PipelinePaths.discover(str(root)).project_root, root)

    def test_existing_storage_layout_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_project_root(Path(directory))
            paths = PipelinePaths.discover(root)
            sample = "sample-id"

            self.assertEqual(paths.data_root, root / "data" / "sample")
            self.assertEqual(paths.processed_root, root / "data" / "sample" / "processed")
            self.assertEqual(
                paths.training_root,
                root / "data" / "sample" / "biohub_5samples_20timepoints" / "train",
            )
            self.assertEqual(
                paths.sample_zarr(sample),
                paths.training_root / sample / f"{sample}.zarr",
            )
            self.assertEqual(paths.sample_zarr_array(sample), paths.sample_zarr(sample) / "0")
            self.assertEqual(
                paths.processed_dataset(sample),
                paths.processed_root / "stage_6_processed_dataset" / sample,
            )
            self.assertEqual(
                paths.processed_series(sample, "cells"),
                paths.processed_dataset(sample) / "cells",
            )
            self.assertEqual(
                paths.stage7_tracking,
                paths.processed_root / "stage_7_cell_tracking",
            )
            self.assertEqual(
                paths.stage8_stitching,
                paths.processed_root / "stage_8_track_stitching",
            )
            self.assertEqual(paths.tracking_scenes, root / "data" / "tracking_scenes")

    def test_environment_patch_restores_original_value(self) -> None:
        original = os.environ.get(ENV_NAME)
        with tempfile.TemporaryDirectory() as directory:
            root = make_project_root(Path(directory))
            with patch.dict(os.environ, {ENV_NAME: str(root)}):
                self.assertEqual(os.environ[ENV_NAME], str(root))
        self.assertEqual(os.environ.get(ENV_NAME), original)

    def test_returned_root_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_project_root(Path(directory))
            nested = root / "notebooks" / ".." / "notebooks"
            self.assertEqual(find_project_root(nested), root)
            self.assertTrue(find_project_root(nested).is_absolute())


if __name__ == "__main__":
    unittest.main()
