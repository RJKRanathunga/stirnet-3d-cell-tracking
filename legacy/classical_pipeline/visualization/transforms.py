"""Shared scene-coordinate transforms for Napari TZYX layers."""

from __future__ import annotations


def scene_layer_transform(
    scene,
    *,
    use_original_coordinates: bool,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Return the scale and translation for scene-local TZYX coordinates."""

    scale = (1.0, *tuple(float(value) for value in scene.voxel_size_zyx))
    frame_origin = float(scene.frames[0]) if len(scene.frames) else 0.0
    if use_original_coordinates:
        spatial_translate = tuple(
            float(origin) * float(spacing)
            for origin, spacing in zip(scene.crop_origin_zyx, scene.voxel_size_zyx)
        )
    else:
        spatial_translate = (0.0, 0.0, 0.0)
    return scale, (frame_origin, *spatial_translate)


__all__ = ["scene_layer_transform"]
