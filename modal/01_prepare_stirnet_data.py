from __future__ import annotations

import gc
import json
import shutil
import sys
from importlib import import_module
from pathlib import Path

import modal


app = modal.App("stirnet-data-prep")
data_volume = modal.Volume.from_name("stirnet-data")

# Save this file as: <repo>/modal/prepare_stirnet_data.py
LOCAL_REPO_ROOT = Path(__file__).resolve().parents[1]

# IMPORTANT: keep container paths as POSIX strings here.
# This file is imported on Windows before Modal sends it to Linux, so using
# pathlib.Path("/workspace/...") at module import time would turn them into
# Windows paths such as "\\workspace\\...".
REMOTE_REPO_ROOT = "/workspace/cell-tracking"
DATA_MOUNT = "/workspace/cell-tracking/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        # Match the repository's recorded local preprocessing environment.
        "numpy==2.4.6",
        "scipy==1.17.1",
        "scikit-image==0.26.0",
    )
    .workdir(REMOTE_REPO_ROOT)
    .add_local_dir(
        LOCAL_REPO_ROOT / "src",
        remote_path="/workspace/cell-tracking/src",
    )
)

SOURCE_DIR = (
    "/workspace/cell-tracking/data/source/"
    "BlastoSPIM1_F22_030_034_source"
)

OUTPUT_DIR = (
    "/workspace/cell-tracking/data/learned/stirnet/first_overfit/"
    "BlastoSPIM1_F22_030_034"
)

SPACING_ZYX_UM = (2.0, 0.208, 0.208)
FRAME_NUMBERS = [30, 31, 32, 33, 34]
TARGET_LOCAL_T = 2


def _robust_normalize(
    raw,
    low_pct: float = 1.0,
    high_pct: float = 99.8,
):
    import numpy as np

    raw = np.asarray(raw, np.float32)
    lo, hi = np.percentile(raw, [low_pct, high_pct])
    if hi <= lo:
        return np.zeros_like(raw, np.float32)
    return np.clip(
        (raw - lo) / (hi - lo),
        0,
        1,
    ).astype(np.float32)


def _estimate_dref_um(
    labels,
    spacing_um,
    min_voxels: int = 20,
) -> float:
    import numpy as np

    voxel_volume = float(np.prod(spacing_um))
    diameters = []

    for label in np.unique(labels):
        if label <= 0:
            continue

        n = int(np.count_nonzero(labels == label))
        if n < min_voxels:
            continue

        volume_um3 = n * voxel_volume
        diameters.append(
            2
            * (
                (3 * volume_um3)
                / (4 * np.pi)
            )
            ** (1 / 3)
        )

    if not diameters:
        return float(np.mean(spacing_um) * 8.0)

    values = np.asarray(diameters)

    if len(values) >= 10:
        low, high = np.percentile(values, [10, 90])
    else:
        low, high = values.min(), values.max()

    good = values[
        (values >= low)
        & (values <= high)
    ]

    return float(
        np.median(
            good if len(good) else values
        )
    )


def _make_instance_boundary(labels):
    import numpy as np

    boundary = np.zeros_like(
        labels,
        dtype=bool,
    )

    for axis in range(3):
        sl1 = [slice(None)] * 3
        sl2 = [slice(None)] * 3

        sl1[axis] = slice(1, None)
        sl2[axis] = slice(None, -1)

        a = labels[tuple(sl1)]
        b = labels[tuple(sl2)]

        diff = (
            (a != b)
            & ((a > 0) | (b > 0))
        )

        boundary[tuple(sl1)] |= diff
        boundary[tuple(sl2)] |= diff

    return boundary


def _per_instance_edt_local(
    labels,
    spacing_um,
    dref_um: float,
):
    import numpy as np
    from scipy import ndimage as ndi

    labels = np.asarray(labels)

    output = np.zeros(
        labels.shape,
        dtype=np.float32,
    )

    objects = ndi.find_objects(labels)

    for label_id, bbox in enumerate(
        objects,
        start=1,
    ):
        if bbox is None:
            continue

        local_labels = labels[bbox]
        local_mask = (
            local_labels == label_id
        )

        if not local_mask.any():
            continue

        # Notebook 02 explicitly pads so components touching
        # acquisition boundaries still see background.
        padded = np.pad(
            local_mask,
            pad_width=1,
            mode="constant",
            constant_values=False,
        )

        local_edt = ndi.distance_transform_edt(
            padded,
            sampling=spacing_um,
        )[1:-1, 1:-1, 1:-1]

        local_edt = (
            local_edt
            / max(float(dref_um), 1e-6)
        ).astype(
            np.float32,
            copy=False,
        )

        output_view = output[bbox]
        output_view[local_mask] = (
            local_edt[local_mask]
        )

    return output


@app.function(
    image=image,
    cpu=8.0,
    memory=16384,
    timeout=3 * 60 * 60,
    volumes={
        str(DATA_MOUNT): data_volume,
    },
)
def prepare_spatial_data():
    import numpy as np
    from scipy import ndimage as ndi

    remote_repo_root = Path(REMOTE_REPO_ROOT)
    source_dir = Path(SOURCE_DIR)
    output_dir = Path(OUTPUT_DIR)

    sys.path.insert(
        0,
        REMOTE_REPO_ROOT,
    )

    preprocess_volume = import_module(
        "src.01_preprocessing.pipeline"
    ).preprocess_volume

    PreprocessingConfig = import_module(
        "src.01_preprocessing.config"
    ).PreprocessingConfig

    create_binary_mask = import_module(
        "src.02_masking.pipeline"
    ).create_binary_mask

    MaskingConfig = import_module(
        "src.02_masking.config"
    ).MaskingConfig

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source directory not found: "
            f"{source_dir}"
        )

    if output_dir.exists():
        raise RuntimeError(
            f"Output already exists: "
            f"{output_dir}\n"
            "Delete it explicitly before "
            "rebuilding."
        )

    # Modal recommends doing transformations on local SSD
    # and copying only final artifacts to the Volume.
    work_root = Path(
        "/tmp/stirnet-prep"
    )

    source_local = (
        work_root / "source"
    )

    output_local = (
        work_root
        / "BlastoSPIM1_F22_030_034"
    )

    shutil.rmtree(
        work_root,
        ignore_errors=True,
    )

    source_local.mkdir(
        parents=True
    )

    output_local.mkdir(
        parents=True
    )

    print(
        "Copying source volumes "
        "from Modal Volume to local SSD...",
        flush=True,
    )

    for frame in FRAME_NUMBERS:
        for kind in (
            "image",
            "masks",
        ):
            name = (
                f"F22_{frame:03d}_"
                f"{kind}_0001.npy"
            )

            src = source_dir / name

            if not src.exists():
                raise FileNotFoundError(src)

            shutil.copy2(
                src,
                source_local / name,
            )

    raw_frames = []
    gt_frames = []

    for frame in FRAME_NUMBERS:
        raw = np.load(
            source_local
            / (
                f"F22_{frame:03d}_"
                "image_0001.npy"
            ),
            allow_pickle=False,
        )

        gt = np.load(
            source_local
            / (
                f"F22_{frame:03d}_"
                "masks_0001.npy"
            ),
            allow_pickle=False,
        )

        if (
            raw.ndim != 3
            or gt.ndim != 3
            or raw.shape != gt.shape
        ):
            raise ValueError(
                f"Bad source pair at "
                f"F22_{frame:03d}: "
                f"raw={raw.shape}, "
                f"gt={gt.shape}"
            )

        if (
            raw_frames
            and raw.shape
            != raw_frames[0].shape
        ):
            raise ValueError(
                "The five raw volumes "
                "do not share one shape."
            )

        raw_frames.append(raw)
        gt_frames.append(gt)

    raw_movie = np.stack(
        raw_frames,
        axis=0,
    )

    gt_movie = np.stack(
        gt_frames,
        axis=0,
    )

    preprocessing_config = (
        PreprocessingConfig(
            low_percentile=1.0,
            high_percentile=99.5,
            denoise_sigma_um=0.8,
            background_sigma_um=4.0,
            voxel_size_zyx_um=(
                SPACING_ZYX_UM
            ),
        )
    )

    masking_config = MaskingConfig()

    print(
        "Running canonical "
        "preprocessing...",
        flush=True,
    )

    processed_frames = []

    for t in range(
        raw_movie.shape[0]
    ):
        processed = preprocess_volume(
            raw_movie[t],
            config=(
                preprocessing_config
            ),
            return_diagnostics=False,
        )

        processed_frames.append(
            processed
        )

        print(
            f"  processed {t + 1}/5",
            flush=True,
        )

    processed_movie = np.stack(
        processed_frames,
        axis=0,
    )

    print(
        "Running canonical "
        "Otsu masking...",
        flush=True,
    )

    binary_frames = []

    for t in range(
        processed_movie.shape[0]
    ):
        binary = create_binary_mask(
            processed_movie[t],
            config=masking_config,
            return_diagnostics=False,
        )

        binary_frames.append(
            binary
        )

        print(
            f"  masked {t + 1}/5",
            flush=True,
        )

    binary_movie = np.stack(
        binary_frames,
        axis=0,
    )

    print(
        "Building 6-connected "
        "components...",
        flush=True,
    )

    connectivity_6 = (
        ndi.generate_binary_structure(
            3,
            1,
        )
    )

    instance_frames = []
    component_counts = []

    for t in range(
        binary_movie.shape[0]
    ):
        labels, count = ndi.label(
            binary_movie[t],
            structure=connectivity_6,
        )

        instance_frames.append(
            labels.astype(
                np.int32,
                copy=False,
            )
        )

        component_counts.append(
            int(count)
        )

    instance_movie = np.stack(
        instance_frames,
        axis=0,
    )

    print(
        "Building one physical-EDT "
        "marker per component...",
        flush=True,
    )

    markers_movie = np.zeros_like(
        instance_movie,
        dtype=np.int32,
    )

    for t in range(
        instance_movie.shape[0]
    ):
        labels = instance_movie[t]

        objects = ndi.find_objects(
            labels
        )

        for instance_id, slc in enumerate(
            objects,
            start=1,
        ):
            if slc is None:
                continue

            # Match Notebook 01:
            # expand each CC bounding box
            # by one voxel.
            expanded = tuple(
                slice(
                    max(
                        0,
                        s.start - 1,
                    ),
                    min(
                        labels.shape[d],
                        s.stop + 1,
                    ),
                )
                for d, s
                in enumerate(slc)
            )

            local_labels = (
                labels[expanded]
            )

            component = (
                local_labels
                == instance_id
            )

            if not component.any():
                continue

            edt = (
                ndi.distance_transform_edt(
                    component,
                    sampling=(
                        SPACING_ZYX_UM
                    ),
                )
            )

            local_position = (
                np.unravel_index(
                    np.argmax(edt),
                    edt.shape,
                )
            )

            global_position = tuple(
                expanded[d].start
                + local_position[d]
                for d in range(3)
            )

            markers_movie[t][
                global_position
            ] = instance_id

    metadata = {
        "dataset": "BlastoSPIM1",
        "series": "F22",
        "frame_numbers": [
            int(x)
            for x in FRAME_NUMBERS
        ],
        "spacing_zyx_um": [
            float(x)
            for x in SPACING_ZYX_UM
        ],
        "pipeline": {
            "stage_1": (
                "canonical preprocessing"
            ),
            "stage_2": (
                "canonical Otsu masking"
            ),
            "stage_3": (
                "6-connected-component "
                "initialization only; "
                "probabilistic peak "
                "reasoning, geometric "
                "completion and watershed "
                "disabled"
            ),
        },
        "component_counts": (
            component_counts
        ),
    }

    print(
        "Saving training-required "
        "base artifacts locally...",
        flush=True,
    )

    np.save(
        output_local
        / "frame_numbers.npy",
        np.asarray(
            FRAME_NUMBERS,
            dtype=np.int32,
        ),
    )

    np.save(
        output_local
        / "raw_movie.npy",
        raw_movie,
    )

    np.save(
        output_local
        / "instance_movie.npy",
        instance_movie.astype(
            np.int32,
            copy=False,
        ),
    )

    np.save(
        output_local
        / "markers_movie.npy",
        markers_movie.astype(
            np.int32,
            copy=False,
        ),
    )

    np.save(
        output_local
        / "gt_movie.npy",
        gt_movie.astype(
            np.int32,
            copy=False,
        ),
    )

    (
        output_local
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # processed_movie.npy and
    # binary_movie.npy are intentionally
    # NOT persisted. They are required
    # only to derive instance_movie and
    # are not read by current
    # build_real_batch().
    del (
        processed_movie,
        binary_movie,
        processed_frames,
        binary_frames,
    )
    gc.collect()

    print(
        "Building target-frame "
        "STIR-Net spatial cache...",
        flush=True,
    )

    target_frame = (
        FRAME_NUMBERS[
            TARGET_LOCAL_T
        ]
    )

    target_raw = np.asarray(
        raw_movie[TARGET_LOCAL_T]
    )

    target_instances = np.asarray(
        instance_movie[
            TARGET_LOCAL_T
        ]
    )

    target_gt = np.asarray(
        gt_movie[TARGET_LOCAL_T]
    )

    target_markers = np.asarray(
        markers_movie[
            TARGET_LOCAL_T
        ]
    )

    dref_um = _estimate_dref_um(
        target_gt,
        SPACING_ZYX_UM,
    )

    source_dir = (
        output_local
        / "stirnet_source"
    )

    source_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_norm_path = (
        source_dir
        / "raw_norm_target.npy"
    )

    foreground_path = (
        source_dir
        / "foreground_target.npy"
    )

    edt_path = (
        source_dir
        / "edt_target.npy"
    )

    boundary_path = (
        source_dir
        / "boundary_target.npy"
    )

    marker_path = (
        source_dir
        / "marker_heatmap_target.npy"
    )

    dref_path = (
        source_dir
        / "dref_um.npy"
    )

    np.save(
        raw_norm_path,
        _robust_normalize(
            target_raw
        ),
    )

    np.save(
        foreground_path,
        target_instances > 0,
    )

    np.save(
        edt_path,
        _per_instance_edt_local(
            target_instances,
            SPACING_ZYX_UM,
            dref_um,
        ),
    )

    np.save(
        boundary_path,
        _make_instance_boundary(
            target_instances
        ),
    )

    np.save(
        marker_path,
        (
            target_markers > 0
        ).astype(np.float32),
    )

    np.save(
        dref_path,
        np.asarray(
            dref_um,
            dtype=np.float32,
        ),
    )

    source_metadata = {
        "target_local_t": int(
            TARGET_LOCAL_T
        ),
        "target_frame": int(
            target_frame
        ),
        "spacing_zyx_um": [
            float(x)
            for x in SPACING_ZYX_UM
        ],
        "dref_um": float(
            dref_um
        ),
        "channels": {
            "raw_norm": (
                raw_norm_path.name
            ),
            "foreground": (
                foreground_path.name
            ),
            "edt_normalized_by_dref": (
                edt_path.name
            ),
            "boundary": (
                boundary_path.name
            ),
            "marker_heatmap": (
                marker_path.name
            ),
        },
    }

    (
        source_dir
        / "metadata.json"
    ).write_text(
        json.dumps(
            source_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Copying final prepared data "
        "to Modal Volume...",
        flush=True,
    )

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copytree(
        output_local,
        output_dir,
    )

    data_volume.commit()

    sizes = {
        str(
            path.relative_to(
                output_dir
            )
        ): path.stat().st_size
        for path
        in output_dir.rglob("*")
        if path.is_file()
    }

    total_gib = (
        sum(sizes.values())
        / 1024**3
    )

    print(
        f"Prepared dataset: "
        f"{output_dir}",
        flush=True,
    )

    print(
        f"Persisted size: "
        f"{total_gib:.3f} GiB",
        flush=True,
    )

    print(
        f"dref_um: "
        f"{dref_um:.6f}",
        flush=True,
    )

    print(
        f"component counts: "
        f"{component_counts}",
        flush=True,
    )

    return {
        "output_dir": str(
            output_dir
        ),
        "total_gib": (
            total_gib
        ),
        "dref_um": dref_um,
        "component_counts": (
            component_counts
        ),
    }


@app.local_entrypoint()
def main():
    result = (
        prepare_spatial_data.remote()
    )
    print(result)
