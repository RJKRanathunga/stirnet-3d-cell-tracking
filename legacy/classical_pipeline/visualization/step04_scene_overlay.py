"""Prepare Stage 8 paths and centers for an extracted tracking scene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.io import load_csv, load_json

from .transforms import scene_layer_transform


OVERLAY_PROPERTY_COLUMNS = (
    "track_id",
    "cell_id",
    "frame",
    "is_virtual_merge",
    "merge_event_id",
    "merge_role",
    "source_track_id",
    "source_merged_cell_id",
    "display_label",
    "virtual_display_label",
)


@dataclass(frozen=True)
class Stage8SceneOverlay:
    rows: pd.DataFrame
    track_data: np.ndarray
    ordinary_point_data: np.ndarray
    virtual_point_data: np.ndarray
    ordinary_properties: dict[str, np.ndarray]
    virtual_properties: dict[str, np.ndarray]
    scale: tuple[float, float, float, float]
    translate: tuple[float, float, float, float]
    summary: dict[str, Any]


def _parse_boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def _property_values(series: pd.Series) -> np.ndarray:
    values = series.astype(object).where(pd.notna(series), "")
    return values.to_numpy()


def _point_properties(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        column: _property_values(rows[column])
        for column in OVERLAY_PROPERTY_COLUMNS
        if column in rows.columns
    }


def _format_identifier(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "?" if pd.isna(numeric) else str(int(numeric))


def prepare_stage8_scene_overlay(
    scene,
    *,
    tracks_path: str | Path,
    metadata_path: str | Path,
    use_original_coordinates: bool = False,
) -> Stage8SceneOverlay:
    """Resolve a scene's saved cell references against the latest Stage 8 rows."""

    tracks = load_csv(
        tracks_path,
        required_columns=("track_id", "frame", "z", "y", "x"),
    )
    metadata = load_json(metadata_path)
    stage8_sample = str(metadata.get("sample_id", ""))
    if stage8_sample and scene.sample_id and stage8_sample != scene.sample_id:
        raise ValueError(
            "The selected scene and Stage 8 output belong to different samples "
            f"({scene.sample_id!r} versus {stage8_sample!r})."
        )

    scene_cell_column = str(scene.metadata.get("cell_id_column", "cell_id"))
    if scene_cell_column not in tracks.columns:
        fallbacks = [name for name in ("cell_id", "cell") if name in tracks.columns]
        if not fallbacks:
            raise ValueError(
                "Could not find the scene cell-ID column in Stage 8 tracks.csv."
            )
        scene_cell_column = fallbacks[0]

    selected_cells = scene.metadata.get("selected_cells", {})
    if not isinstance(selected_cells, dict) or not selected_cells:
        raise ValueError("The selected scene does not contain saved cell references.")
    references = [
        (int(frame), int(cell_id))
        for frame, cell_ids in selected_cells.items()
        for cell_id in cell_ids
    ]
    if not references:
        raise ValueError("The selected scene contains an empty selected_cells mapping.")

    numeric_frame = pd.to_numeric(tracks["frame"], errors="coerce")
    numeric_cell = pd.to_numeric(tracks[scene_cell_column], errors="coerce")
    if "source_merged_cell_id" in tracks.columns:
        numeric_merged_cell = pd.to_numeric(
            tracks["source_merged_cell_id"], errors="coerce"
        )
    else:
        numeric_merged_cell = pd.Series(np.nan, index=tracks.index, dtype=float)
    matched = pd.Series(False, index=tracks.index)
    for frame, cell_id in references:
        matched |= numeric_frame.eq(frame) & (
            numeric_cell.eq(cell_id) | numeric_merged_cell.eq(cell_id)
        )
    matched_rows = tracks.loc[matched]
    if matched_rows.empty:
        raise ValueError(
            "None of the scene's saved frame/cell references were found in "
            "the current Stage 8 tracks.csv."
        )

    track_ids = sorted(
        pd.to_numeric(matched_rows["track_id"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    scene_frames = {int(frame) for frame in scene.frames}
    rows = tracks[
        pd.to_numeric(tracks["track_id"], errors="coerce").isin(track_ids)
        & numeric_frame.isin(scene_frames)
    ].copy()
    rows["track_id"] = pd.to_numeric(rows["track_id"], errors="raise").astype(int)
    rows["frame"] = pd.to_numeric(rows["frame"], errors="raise").astype(int)
    rows["is_virtual_merge"] = (
        _parse_boolean_series(rows["is_virtual_merge"])
        if "is_virtual_merge" in rows.columns
        else False
    )
    rows = rows.sort_values(["track_id", "frame"]).reset_index(drop=True)

    frame_to_local = {
        int(frame): int(local_time) for local_time, frame in enumerate(scene.frames)
    }
    rows["scene_time"] = rows["frame"].map(frame_to_local)
    for column in ("z", "y", "x"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.dropna(subset=["scene_time", "z", "y", "x"]).copy()
    if rows.empty:
        raise ValueError("All relevant Stage 8 rows have invalid coordinates.")
    rows["scene_time"] = rows["scene_time"].astype(int)

    source_cell_ids = pd.to_numeric(rows[scene_cell_column], errors="coerce")
    if "source_merged_cell_id" in rows.columns:
        merged_source_ids = pd.to_numeric(
            rows["source_merged_cell_id"], errors="coerce"
        )
        use_merged_source = rows["is_virtual_merge"] & merged_source_ids.notna()
        source_cell_ids = source_cell_ids.where(
            ~use_merged_source,
            merged_source_ids,
        ).fillna(merged_source_ids)
    rows["display_cell_id"] = [_format_identifier(value) for value in source_cell_ids]
    rows["display_label"] = "C" + rows["display_cell_id"]
    roles = (
        rows["merge_role"].astype(object).where(pd.notna(rows["merge_role"]), "")
        if "merge_role" in rows.columns
        else pd.Series("", index=rows.index, dtype=object)
    ).astype(str).str.strip()
    rows["virtual_display_label"] = rows["display_label"]
    has_role = roles.ne("")
    rows.loc[has_role, "virtual_display_label"] = (
        rows.loc[has_role, "display_label"] + " | " + roles.loc[has_role]
    )

    local_coordinates = rows[["z", "y", "x"]].to_numpy(dtype=float) - np.asarray(
        scene.crop_origin_zyx, dtype=float
    )
    rows[["scene_z", "scene_y", "scene_x"]] = local_coordinates
    scale, translate = scene_layer_transform(
        scene, use_original_coordinates=use_original_coordinates
    )
    track_data = rows[
        ["track_id", "scene_time", "scene_z", "scene_y", "scene_x"]
    ].to_numpy(dtype=float)
    ordinary_rows = rows.loc[~rows["is_virtual_merge"]].copy()
    virtual_rows = rows.loc[rows["is_virtual_merge"]].copy()
    point_columns = ["scene_time", "scene_z", "scene_y", "scene_x"]

    return Stage8SceneOverlay(
        rows=rows,
        track_data=track_data,
        ordinary_point_data=ordinary_rows[point_columns].to_numpy(dtype=float),
        virtual_point_data=virtual_rows[point_columns].to_numpy(dtype=float),
        ordinary_properties=_point_properties(ordinary_rows),
        virtual_properties=_point_properties(virtual_rows),
        scale=scale,
        translate=translate,
        summary={
            "selected_reference_count": len(references),
            "matched_reference_count": int(matched.sum()),
            "track_ids": track_ids,
            "track_row_count": len(rows),
            "virtual_center_count": int(rows["is_virtual_merge"].sum()),
            "scene_cell_column": scene_cell_column,
            "scene_frame_count": len(scene.frames),
        },
    )


__all__ = [
    "OVERLAY_PROPERTY_COLUMNS",
    "Stage8SceneOverlay",
    "prepare_stage8_scene_overlay",
    "scene_layer_transform",
]
