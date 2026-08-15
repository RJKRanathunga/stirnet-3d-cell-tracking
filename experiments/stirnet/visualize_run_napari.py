from __future__ import annotations

"""
Reusable local Napari viewer for a trained STIR-Net V1 first-overfit run.

Normal use
----------
Change only MODEL_RUN_DIR below, then run this script locally:

    python experiments/stirnet/visualize_run_napari.py

The script:
- auto-selects the most useful checkpoint in that run directory;
- rebuilds the canonical BlastoSPIM first-overfit scene;
- applies the local_6gb runtime profile for local inference;
- runs the CURRENT full STIR-Net path, including temporal reasoning;
- finds proposal queries relevant to SOURCE_ID using source association OR
  spatial proximity to the source's GT cells;
- renders the current model-owned local native masks through model.render_masks;
- opens a 3-D Napari viewer.

This is a visualization/debugging tool, not competition post-processing.
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import napari
except ImportError as exc:
    raise ImportError(
        "Napari is not installed. Install it with:\n"
        "    pip install 'napari[all]'"
    ) from exc


# ---------------------------------------------------------------------------
# User-editable setting
# ---------------------------------------------------------------------------

# Change this to visualize another trained run.
MODEL_RUN_DIR = Path(
    r"runs/stirnet/experiments/30_overfit_oracle/20260815_163540"
)

# Normally leave this as None. The script searches the run directory using the
# priority list in choose_checkpoint().
CHECKPOINT_OVERRIDE: Path | None = None


# ---------------------------------------------------------------------------
# Visualization settings
# ---------------------------------------------------------------------------

SOURCE_ID = 9
SEED = 40266

# Candidate queries are included if they are explicitly associated with the
# source component OR their immutable proposal anchor is close to one of the
# source's GT centers. The proximity branch is important for off-mask proposals
# whose source_instance_id is -1.
CANDIDATE_RADIUS_DREF = 1.25

# Napari crop around the source component + its GT cells.
CROP_MARGIN_DREF = 2.0

# Visualization-only thresholds.
MASK_THRESHOLD = 0.50
EXIST_THRESHOLD = 0.50

# Only the most confident query probabilities get individual Napari layers.
# All candidate queries still contribute to the combined candidate labels.
MAX_INDIVIDUAL_MASK_LAYERS = 12

# Local inference profile. It changes execution/memory behavior, not model
# tensor shapes, so cloud-trained checkpoints remain compatible.
LOCAL_RUNTIME_PROFILE = "local_6gb"

AMP_DTYPE = torch.float16


# ---------------------------------------------------------------------------
# Resolve repository and imports
# ---------------------------------------------------------------------------

def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for path in (start, *start.parents):
        if (path / "learned").exists() and (path / "data").exists():
            return path
    raise RuntimeError(
        "Could not locate repository root. Run this script from inside the "
        "cell-tracking repository."
    )


REPO_ROOT = find_repo_root(Path.cwd())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learned.stirnet import StirNet
from learned.stirnet.debugging.acceptance.first_overfit import (
    _reduced_config,
    build_real_batch,
)
from learned.stirnet.model.matcher import target_ids
from learned.stirnet.model.query_builder import QUERY_SPATIAL_PROPOSAL
from learned.stirnet.model.runtime_profiles import (
    apply_runtime_profile,
    describe_runtime_profile,
)
from learned.stirnet.training.checkpoint import load_checkpoint
from learned.stirnet.training.trainer import (
    model_forward_from_batch,
    move_batch_to_device,
)


DATA_DIR = (
    REPO_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "first_overfit"
    / "BlastoSPIM1_F22_030_034"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_repo_relative(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def choose_checkpoint(run_dir: Path, override: Path | None) -> Path:
    if override is not None:
        checkpoint = resolve_repo_relative(override)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint override not found: {checkpoint}")
        return checkpoint

    priority = (
        "checkpoint_final.pt",
        "checkpoint_joint_best_native.pt",
        "checkpoint_joint_best_total.pt",
        "checkpoint_joint_best_or_last.pt",
        "checkpoint_joint_last.pt",
        "checkpoint_after_local_mask.pt",
        "checkpoint_after_query.pt",
        "checkpoint_best_spatial_query.pt",
    )

    for name in priority:
        candidate = run_dir / name
        if candidate.exists():
            return candidate

    candidates = sorted(
        run_dir.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No .pt checkpoint found in run directory: {run_dir}"
    )


def refs_cellscale_to_full_voxels(
    refs_cellscale: torch.Tensor | np.ndarray,
    *,
    dref_um: float,
    spacing_um: np.ndarray,
    full_shape: tuple[int, int, int],
) -> np.ndarray:
    refs = torch.as_tensor(refs_cellscale, dtype=torch.float32).cpu().numpy()
    refs_um = refs * float(dref_um)

    shape = np.asarray(full_shape, dtype=np.float64)
    spacing = np.asarray(spacing_um, dtype=np.float64)
    patch_center_um = 0.5 * (shape - 1.0) * spacing

    return (refs_um + patch_center_um[None]) / spacing[None]


def crop_for_source(
    current_labels: np.ndarray,
    gt_labels: np.ndarray,
    source_gt_ids: np.ndarray,
    *,
    spacing_um: np.ndarray,
    dref_um: float,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice]]:
    source_voxels = np.argwhere(current_labels == SOURCE_ID)
    gt_voxels = np.argwhere(np.isin(gt_labels, source_gt_ids))

    pieces = []
    if len(source_voxels):
        pieces.append(source_voxels)
    if len(gt_voxels):
        pieces.append(gt_voxels)
    if not pieces:
        raise RuntimeError(
            f"Neither current source {SOURCE_ID} nor its GT cells have voxels."
        )

    voxels = np.concatenate(pieces, axis=0)
    lo = voxels.min(axis=0)
    hi = voxels.max(axis=0) + 1

    margin_um = CROP_MARGIN_DREF * float(dref_um)
    margin_vox = np.ceil(
        margin_um / np.asarray(spacing_um, dtype=np.float64)
    ).astype(np.int64)

    shape = np.asarray(current_labels.shape, dtype=np.int64)
    lo = np.maximum(0, lo - margin_vox)
    hi = np.minimum(shape, hi + margin_vox)

    slices = tuple(
        slice(int(a), int(b))
        for a, b in zip(lo, hi)
    )
    return lo.astype(np.int64), hi.astype(np.int64), slices


def nearest_gt_information(
    refs: torch.Tensor,
    gt_centers: torch.Tensor,
    gt_ids: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray]:
    if refs.numel() == 0 or gt_centers.numel() == 0:
        return (
            torch.full((len(refs),), float("inf")),
            np.full((len(refs),), -1, dtype=np.int64),
        )

    distance = torch.cdist(
        refs.detach().float().cpu(),
        gt_centers.detach().float().cpu(),
    )
    nearest_distance, nearest_row = distance.min(dim=1)
    nearest_ids = np.asarray(gt_ids, dtype=np.int64)[nearest_row.numpy()]
    return nearest_distance, nearest_ids


def update_composition(
    best_probability: np.ndarray,
    labels: np.ndarray,
    probability: np.ndarray,
    *,
    visual_label_id: int,
) -> None:
    better = probability > best_probability
    best_probability[better] = probability[better]
    labels[better] = int(visual_label_id)


def apply_mask_threshold(
    labels: np.ndarray,
    best_probability: np.ndarray,
) -> np.ndarray:
    result = labels.copy()
    result[best_probability < MASK_THRESHOLD] = 0
    return result


def print_table(df: pd.DataFrame) -> None:
    with pd.option_context(
        "display.max_rows", 200,
        "display.max_columns", 50,
        "display.width", 240,
        "display.max_colwidth", 40,
    ):
        print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open a trained STIR-Net run in Napari."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=MODEL_RUN_DIR,
        help=(
            "Trained run directory. If omitted, MODEL_RUN_DIR at the top of "
            "the script is used."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_OVERRIDE,
        help="Optional explicit checkpoint path.",
    )
    args = parser.parse_args()

    run_dir = resolve_repo_relative(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    checkpoint = choose_checkpoint(run_dir, args.checkpoint)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {DATA_DIR}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This visualization script requires CUDA for the STIR-Net forward."
        )

    device = torch.device("cuda")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    print("=" * 88)
    print("STIR-Net V1 reusable Napari run viewer")
    print("=" * 88)
    print("Repository :", REPO_ROOT)
    print("Run        :", run_dir)
    print("Checkpoint :", checkpoint)
    print("Data       :", DATA_DIR)
    print("GPU        :", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    batch_cpu, sample_info = build_real_batch(DATA_DIR)
    target = batch_cpu["targets"][0]

    spatial_inputs_native = (
        batch_cpu["spatial_inputs"][0]
        .detach()
        .cpu()
        .float()
        .numpy()
    )
    raw_native = spatial_inputs_native[0]
    foreground_native = spatial_inputs_native[1]
    edt_native = spatial_inputs_native[2]
    boundary_native = spatial_inputs_native[3]
    marker_native = spatial_inputs_native[4]

    current_labels_native = (
        batch_cpu["instance_labels"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int32, copy=False)
    )
    gt_labels_native = (
        torch.as_tensor(target["label_map"])
        .detach()
        .cpu()
        .numpy()
        .astype(np.int32, copy=False)
    )
    spacing_native = (
        batch_cpu["spacing_um"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    dref_um = float(batch_cpu["dref_um"][0])

    gt_ids = target_ids(target).detach().cpu().numpy().astype(np.int64)
    gt_centers = (
        torch.as_tensor(target["centers_cellscale"])
        .detach()
        .cpu()
        .float()
    )

    source_gt_ids = np.unique(
        gt_labels_native[current_labels_native == SOURCE_ID]
    )
    source_gt_ids = source_gt_ids[source_gt_ids > 0].astype(np.int64)
    if len(source_gt_ids) == 0:
        raise RuntimeError(
            f"Current source component {SOURCE_ID} overlaps no GT cells."
        )

    source_gt_set = set(source_gt_ids.tolist())
    source_gt_rows = torch.as_tensor(
        [
            row
            for row, gt_id in enumerate(gt_ids.tolist())
            if int(gt_id) in source_gt_set
        ],
        dtype=torch.long,
    )
    source_gt_centers = gt_centers[source_gt_rows]

    print("Scene      :", sample_info)
    print(f"Source {SOURCE_ID} GT IDs:", source_gt_ids.tolist())

    # ------------------------------------------------------------------
    # Current model + local runtime profile
    # ------------------------------------------------------------------

    cfg = _reduced_config()
    cfg.proposals.enabled = True
    cfg.proposals.query_mode = "spatial_proposals"
    cfg.local_masks.enabled = True
    apply_runtime_profile(cfg, LOCAL_RUNTIME_PROFILE)

    model = StirNet(cfg).to(device)
    load_info = load_checkpoint(
        checkpoint,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        map_location="cpu",
        strict=True,
        migrate_history=True,
    )
    model.eval()

    print("Checkpoint step:", load_info.get("step"))
    print("Runtime profile :", describe_runtime_profile(cfg))

    # Full current-model forward, not the old Notebook-26 spatial-only path.
    b = move_batch_to_device(batch_cpu, device)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=AMP_DTYPE,
    ):
        outputs = model_forward_from_batch(
            model,
            b,
            return_debug=True,
            temporal_memory_ablation="full",
            temporal_routing_ablation="full",
            detection_graph_ablation="full",
        )

    torch.cuda.synchronize()
    print(
        "Forward peak CUDA memory:",
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
    )

    # ------------------------------------------------------------------
    # Candidate spatial-proposal queries around this merged component
    # ------------------------------------------------------------------

    valid = ~outputs.query_padding_mask[0].detach().cpu()
    qtypes = outputs.query_types[0].detach().cpu()
    source_ids = outputs.source_instance_ids[0].detach().cpu().long()

    proposal_query_rows = torch.nonzero(
        valid & (qtypes == QUERY_SPATIAL_PROPOSAL),
        as_tuple=False,
    ).flatten()

    proposal_initial_refs = (
        outputs.query_initial_references_cellscale[
            0, proposal_query_rows.to(device)
        ]
        .detach()
        .float()
        .cpu()
    )

    nearest_distance, nearest_gt_id = nearest_gt_information(
        proposal_initial_refs,
        source_gt_centers,
        source_gt_ids,
    )

    proposal_source_ids = source_ids[proposal_query_rows]

    relevant_local = (
        (proposal_source_ids == SOURCE_ID)
        | (nearest_distance <= CANDIDATE_RADIUS_DREF)
    )

    candidate_rows = proposal_query_rows[relevant_local]
    candidate_initial_refs = proposal_initial_refs[relevant_local]
    candidate_source_ids = proposal_source_ids[relevant_local]
    candidate_nearest_distance = nearest_distance[relevant_local]
    candidate_nearest_gt_id = nearest_gt_id[relevant_local.numpy()]

    if candidate_rows.numel() == 0:
        raise RuntimeError(
            f"No spatial-proposal queries found around source {SOURCE_ID}."
        )

    candidate_final_refs = (
        outputs.centers_cellscale[0, candidate_rows.to(device)]
        .detach()
        .float()
        .cpu()
    )
    candidate_exist = (
        outputs.exist_logits[0, candidate_rows.to(device)]
        .detach()
        .float()
        .sigmoid()
        .cpu()
    )
    candidate_selected = candidate_exist >= EXIST_THRESHOLD

    # Sort the query table by existence probability while preserving candidate
    # local IDs for the rendered label volumes.
    query_table = pd.DataFrame(
        {
            "visual_label_id": np.arange(1, len(candidate_rows) + 1),
            "query_index": candidate_rows.numpy(),
            "source_instance_id": candidate_source_ids.numpy(),
            "nearest_source9_gt_id": candidate_nearest_gt_id,
            "nearest_gt_dref": candidate_nearest_distance.numpy(),
            "exist_prob": candidate_exist.numpy(),
            "selected_at_0p5": candidate_selected.numpy(),
            "initial_z_dref": candidate_initial_refs[:, 0].numpy(),
            "initial_y_dref": candidate_initial_refs[:, 1].numpy(),
            "initial_x_dref": candidate_initial_refs[:, 2].numpy(),
            "final_z_dref": candidate_final_refs[:, 0].numpy(),
            "final_y_dref": candidate_final_refs[:, 1].numpy(),
            "final_x_dref": candidate_final_refs[:, 2].numpy(),
        }
    ).sort_values(
        ["selected_at_0p5", "exist_prob"],
        ascending=[False, False],
    )

    print()
    print(
        f"Candidate proposal queries for source {SOURCE_ID}: "
        f"{len(candidate_rows)}"
    )
    print(
        f"Selected at exist >= {EXIST_THRESHOLD:.2f}: "
        f"{int(candidate_selected.sum())}"
    )
    print_table(query_table)

    # ------------------------------------------------------------------
    # Crop
    # ------------------------------------------------------------------

    lo, hi, crop_slices = crop_for_source(
        current_labels_native,
        gt_labels_native,
        source_gt_ids,
        spacing_um=spacing_native,
        dref_um=dref_um,
    )

    crop_shape = tuple((hi - lo).tolist())

    raw_crop = raw_native[crop_slices]
    current_crop = current_labels_native[crop_slices]
    gt_crop = gt_labels_native[crop_slices]

    source_binary_crop = (
        current_crop == SOURCE_ID
    ).astype(np.uint8)

    source_gt_labels_crop = np.where(
        np.isin(gt_crop, source_gt_ids),
        gt_crop,
        0,
    ).astype(np.int32)

    foreground_crop = foreground_native[crop_slices]
    edt_crop = edt_native[crop_slices]
    boundary_crop = boundary_native[crop_slices]
    marker_crop = marker_native[crop_slices]

    proposal_score_crop = (
        outputs.dense_outputs["proposal_score_logits"][0, 0]
        .detach()
        .float()
        .cpu()
        .numpy()[crop_slices]
    )

    print()
    print("Crop start z/y/x:", lo.tolist())
    print("Crop end   z/y/x:", hi.tolist())
    print("Crop shape      :", crop_shape)

    # ------------------------------------------------------------------
    # Render current local native masks, one query at a time.
    #
    # Do not retain every candidate volume. We stream each rendered crop into
    # two compositions and retain only the top-N individual probability layers.
    # ------------------------------------------------------------------

    best_all = np.zeros(crop_shape, dtype=np.float32)
    labels_all = np.zeros(crop_shape, dtype=np.int32)

    best_selected = np.zeros(crop_shape, dtype=np.float32)
    labels_selected = np.zeros(crop_shape, dtype=np.int32)

    top_individual_local_ids = set(
        np.argsort(-candidate_exist.numpy())[
            :MAX_INDIVIDUAL_MASK_LAYERS
        ].tolist()
    )
    individual_probabilities: dict[int, np.ndarray] = {}

    print()
    print("Rendering proposal-local native masks...")

    for local_id, query_index in enumerate(candidate_rows.tolist()):
        selected_index = torch.tensor(
            [int(query_index)],
            device=device,
            dtype=torch.long,
        )

        # render_masks routes spatial proposals through the current model-owned
        # LocalMaskDecoder. It is deliberately called outside autocast; the
        # decoder's AMP-safe cached-feature path handles this case.
        with torch.no_grad():
            rendered = model.render_masks(
                outputs,
                [selected_index],
            )[0][0]

        probability_crop = (
            rendered[crop_slices]
            .float()
            .sigmoid()
            .detach()
            .cpu()
            .numpy()
        )

        visual_label_id = local_id + 1

        update_composition(
            best_all,
            labels_all,
            probability_crop,
            visual_label_id=visual_label_id,
        )

        if bool(candidate_selected[local_id]):
            update_composition(
                best_selected,
                labels_selected,
                probability_crop,
                visual_label_id=visual_label_id,
            )

        if local_id in top_individual_local_ids:
            individual_probabilities[local_id] = (
                probability_crop.astype(np.float16, copy=True)
            )

        del rendered, probability_crop, selected_index
        torch.cuda.empty_cache()

        print(
            f"  {local_id + 1:02d}/{len(candidate_rows):02d} "
            f"Q{query_index:03d} "
            f"exist={float(candidate_exist[local_id]):.3f} "
            f"nearest_GT={int(candidate_nearest_gt_id[local_id])} "
            f"d={float(candidate_nearest_distance[local_id]):.3f} dref",
            flush=True,
        )

    labels_all_thresholded = apply_mask_threshold(
        labels_all,
        best_all,
    )
    labels_selected_thresholded = apply_mask_threshold(
        labels_selected,
        best_selected,
    )

    print()
    print(
        "Visible candidate-mask IDs:",
        int(np.count_nonzero(np.unique(labels_all_thresholded) > 0)),
    )
    print(
        "Visible selected-mask IDs :",
        int(np.count_nonzero(np.unique(labels_selected_thresholded) > 0)),
    )

    # ------------------------------------------------------------------
    # Points: query anchors, final centers, GT centers, proposal-state anchors.
    # ------------------------------------------------------------------

    candidate_initial_points = refs_cellscale_to_full_voxels(
        candidate_initial_refs,
        dref_um=dref_um,
        spacing_um=spacing_native,
        full_shape=current_labels_native.shape,
    ) - lo[None]

    candidate_final_points = refs_cellscale_to_full_voxels(
        candidate_final_refs,
        dref_um=dref_um,
        spacing_um=spacing_native,
        full_shape=current_labels_native.shape,
    ) - lo[None]

    gt_points = refs_cellscale_to_full_voxels(
        source_gt_centers,
        dref_um=dref_um,
        spacing_um=spacing_native,
        full_shape=current_labels_native.shape,
    ) - lo[None]

    proposal_state = outputs.proposals
    if proposal_state is not None:
        proposal_valid = ~proposal_state.padding_mask[0].detach().cpu()
        proposal_refs_all = (
            proposal_state.references_cellscale[0, proposal_valid.to(device)]
            .detach()
            .float()
            .cpu()
        )
        proposal_sources_all = (
            proposal_state.source_instance_ids[0, proposal_valid.to(device)]
            .detach()
            .cpu()
            .long()
        )

        proposal_distance, _ = nearest_gt_information(
            proposal_refs_all,
            source_gt_centers,
            source_gt_ids,
        )
        proposal_relevant = (
            (proposal_sources_all == SOURCE_ID)
            | (proposal_distance <= CANDIDATE_RADIUS_DREF)
        )
        proposal_refs_relevant = proposal_refs_all[proposal_relevant]

        proposal_points = refs_cellscale_to_full_voxels(
            proposal_refs_relevant,
            dref_um=dref_um,
            spacing_um=spacing_native,
            full_shape=current_labels_native.shape,
        ) - lo[None]
    else:
        proposal_points = np.empty((0, 3), dtype=np.float32)

    selected_np = candidate_selected.numpy().astype(bool)
    selected_initial_points = candidate_initial_points[selected_np]
    rejected_initial_points = candidate_initial_points[~selected_np]
    selected_final_points = candidate_final_points[selected_np]
    rejected_final_points = candidate_final_points[~selected_np]

    # ------------------------------------------------------------------
    # Free unnecessary CUDA-heavy state before opening the GUI.
    # Napari uses CPU arrays prepared above.
    # ------------------------------------------------------------------

    del b
    del outputs
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Napari 3-D viewer
    # ------------------------------------------------------------------

    viewer = napari.Viewer(
        title=(
            f"STIR-Net V1 — {run_dir.name} — source {SOURCE_ID}"
        ),
        ndisplay=3,
    )

    scale = tuple(spacing_native.tolist())

    viewer.add_image(
        raw_crop,
        name="Raw normalized",
        scale=scale,
        blending="additive",
    )

    viewer.add_labels(
        current_crop,
        name="Current instance labels",
        scale=scale,
        visible=False,
    )

    viewer.add_labels(
        gt_crop,
        name="GT instance labels",
        scale=scale,
        visible=False,
    )

    viewer.add_labels(
        source_binary_crop,
        name=f"Current source {SOURCE_ID}",
        scale=scale,
        visible=True,
    )

    viewer.add_labels(
        source_gt_labels_crop,
        name=f"GT cells overlapping source {SOURCE_ID}",
        scale=scale,
        visible=False,
    )

    viewer.add_labels(
        labels_all_thresholded,
        name="All candidate proposal masks",
        scale=scale,
        visible=False,
    )

    viewer.add_labels(
        labels_selected_thresholded,
        name=f"Selected masks (exist >= {EXIST_THRESHOLD:.2f})",
        scale=scale,
        visible=True,
    )

    viewer.add_image(
        best_all,
        name="Max candidate mask probability",
        scale=scale,
        colormap="magma",
        opacity=0.55,
        visible=False,
    )

    viewer.add_image(
        best_selected,
        name="Max selected mask probability",
        scale=scale,
        colormap="magma",
        opacity=0.55,
        visible=False,
    )

    viewer.add_image(
        proposal_score_crop,
        name="Proposal score",
        scale=scale,
        colormap="viridis",
        opacity=0.60,
        visible=False,
    )

    viewer.add_image(
        edt_crop,
        name="Input EDT",
        scale=scale,
        colormap="turbo",
        visible=False,
    )

    viewer.add_image(
        marker_crop,
        name="Input marker heatmap",
        scale=scale,
        colormap="magma",
        visible=False,
    )

    viewer.add_image(
        boundary_crop,
        name="Input boundary",
        scale=scale,
        colormap="cyan",
        visible=False,
    )

    viewer.add_image(
        foreground_crop,
        name="Input foreground",
        scale=scale,
        colormap="gray",
        visible=False,
    )

    if len(gt_points):
        viewer.add_points(
            gt_points,
            name="GT cell centers",
            scale=scale,
            size=3.5,
            symbol="cross",
            face_color="white",
        )

    if len(proposal_points):
        viewer.add_points(
            proposal_points,
            name="Proposal-generator anchors",
            scale=scale,
            size=2.5,
            symbol="disc",
            face_color="yellow",
            visible=False,
        )

    if len(selected_initial_points):
        viewer.add_points(
            selected_initial_points,
            name="Selected immutable anchors",
            scale=scale,
            size=3.0,
            symbol="disc",
            face_color="lime",
        )

    if len(rejected_initial_points):
        viewer.add_points(
            rejected_initial_points,
            name="Rejected immutable anchors",
            scale=scale,
            size=2.5,
            symbol="disc",
            face_color="orange",
            visible=False,
        )

    if len(selected_final_points):
        viewer.add_points(
            selected_final_points,
            name="Selected final centers",
            scale=scale,
            size=3.5,
            symbol="cross",
            face_color="red",
        )

    if len(rejected_final_points):
        viewer.add_points(
            rejected_final_points,
            name="Rejected final centers",
            scale=scale,
            size=3.0,
            symbol="cross",
            face_color="magenta",
            visible=False,
        )

    # Individual mask probability layers in descending existence score.
    ranked_local_ids = np.argsort(-candidate_exist.numpy())[
        :MAX_INDIVIDUAL_MASK_LAYERS
    ]

    for local_id in ranked_local_ids:
        local_id = int(local_id)
        if local_id not in individual_probabilities:
            continue

        query_index = int(candidate_rows[local_id])
        exist = float(candidate_exist[local_id])
        nearest_gt = int(candidate_nearest_gt_id[local_id])
        nearest_dref = float(candidate_nearest_distance[local_id])
        selected_text = "SELECTED" if bool(candidate_selected[local_id]) else "rejected"

        viewer.add_image(
            individual_probabilities[local_id],
            name=(
                f"Q{query_index:03d} {selected_text} "
                f"exist={exist:.3f} GT={nearest_gt} d={nearest_dref:.3f}"
            ),
            scale=scale,
            colormap="magma",
            opacity=0.70,
            visible=False,
        )

    print()
    print("=" * 88)
    print("Napari viewer ready.")
    print()
    print("Recommended first comparison:")
    print(f"  1. Current source {SOURCE_ID}")
    print(f"  2. GT cells overlapping source {SOURCE_ID}")
    print(f"  3. Selected masks (exist >= {EXIST_THRESHOLD:.2f})")
    print("  4. Selected immutable anchors")
    print("  5. Selected final centers")
    print()
    print(
        "Then toggle 'All candidate proposal masks' to see whether a missing "
        "selected cell nevertheless has a usable rejected candidate."
    )
    print("=" * 88)

    napari.run()


if __name__ == "__main__":
    main()
