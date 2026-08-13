from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import time

import numpy as np
import torch

from .config import DebugConfig
from .hooks import StirNetHookRecorder
from .stats import flatten_stats_for_csv
from .trace import DebugTrace
from ..probes.matching import run_matching_probe
from ..probes.queries import build_query_table, select_queries_for_deep_probe
from ..probes.masks import probe_native_masks
from ..probes.spatial import capture_dense_arrays, capture_scene_arrays
from ..probes.temporal import build_temporal_table
from ...training.trainer import move_to_device


def _forward_with_debug(model, b: dict):
    return model(
        b["spatial_inputs"], b["instance_labels"], b["spacing_um"], b["dref_um"],
        b["instance_features"], b["instance_ids"], b["instance_batch"], b["instance_centroids_um"],
        b["graph_x"], b["graph_edge_index"], b["graph_edge_attr"], b["tracklet_id"],
        b["temporal_ref_um"], b["temporal_status"], b["hypothesis_edge_index"],
        b["hypothesis_edge_attr"], b["temporal_batch"], b.get("spatial_padding_mask"),
        return_debug=True,
        node_instance_grid=b.get("node_instance_grid"),
        node_history_valid=b.get("node_history_valid"),
        history_support=b.get("history_support"),
        history_support_valid=b.get("history_support_valid"),
        history_support_dt=b.get("history_support_dt"),
        history_support_center_um=b.get("history_support_center_um"),
        history_support_extent_um=b.get("history_support_extent_um"),
        best_current_component_id=b.get("best_current_component_id"),
        best_component_overlap=b.get("best_component_overlap"),
        second_best_component_overlap=b.get("second_best_component_overlap"),
    )


class StirNetInspector:
    """Run one standard, structured STIR-Net V1 diagnostic pass."""

    def __init__(self, model, config: DebugConfig | None = None):
        self.model = model
        self.config = config or DebugConfig.light()
        self.config.validate()

    @property
    def device(self):
        if self.config.device is not None:
            return torch.device(self.config.device)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_batch(self, batch):
        device = self.device
        b = {}
        for key, value in batch.items():
            if key == "targets":
                b[key] = value
            elif key == "spatial_inputs":
                if device.type == "cuda" and self.config.amp_dtype == "fp16":
                    dtype = torch.float16
                elif device.type == "cuda" and self.config.amp_dtype == "bf16":
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float32
                b[key] = value.to(device=device, dtype=dtype, non_blocking=True)
            elif key == "instance_labels":
                b[key] = value.to(device=device, dtype=torch.int32, non_blocking=True)
            else:
                b[key] = move_to_device(value, device)
        return b

    def _autocast(self):
        if self.device.type != "cuda" or self.config.amp_dtype == "none":
            return nullcontext()
        dtype = torch.float16 if self.config.amp_dtype == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    @torch.no_grad()
    def inspect(self, batch: dict) -> DebugTrace:
        if "targets" not in batch:
            raise KeyError("A debug batch must contain 'targets'.")

        targets = batch["targets"]
        b = self._prepare_batch(batch)
        self.model.to(self.device).eval()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        started = time.perf_counter()
        with StirNetHookRecorder(self.model, self.config) as recorder:
            with self._autocast():
                outputs = _forward_with_debug(self.model, b)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        trace = DebugTrace()
        spacing = outputs.spacing_um[0].detach().float().cpu().numpy()
        dref = float(outputs.dref_um[0].detach().float().cpu())
        scene_shape = [int(v) for v in outputs.instance_labels.shape[-3:]]
        trace.metadata = {
            "schema_version": 1,
            "debug_config": asdict(self.config),
            "forward_seconds": elapsed,
            "device": str(self.device),
            "batch_size": int(outputs.exist_logits.shape[0]),
            "query_count": int(outputs.exist_logits.shape[1]),
            "spacing_um": spacing.tolist(),
            "dref_um": dref,
            "scene_shape_zyx": scene_shape,
        }
        if self.device.type == "cuda":
            trace.metadata["peak_cuda_gib"] = torch.cuda.max_memory_allocated() / 1024**3

        if self.config.capture_module_stats:
            trace.tables["modules"] = [flatten_stats_for_csv(r) for r in recorder.capture.module_stats]

        matching = None
        if (
            self.config.capture_matching
            or self.config.capture_queries
            or self.config.capture_native_masks
        ):
            matching = run_matching_probe(outputs, targets)
            trace.tables["matching"] = matching.rows

        exist_threshold = self.config.final_exist_threshold
        if exist_threshold is None:
            exist_threshold = self.model.cfg.inference.final_exist_threshold
        mask_threshold = self.config.mask_threshold
        if mask_threshold is None:
            mask_threshold = self.model.cfg.inference.mask_threshold

        query_rows = []
        if self.config.capture_queries:
            query_rows = build_query_table(outputs, targets, matching, recorder.capture, float(exist_threshold))
            trace.tables["queries"] = query_rows
            counts = {}
            for row in query_rows:
                t = row["query_type"]
                counts[t] = counts.get(t, 0) + 1
                if row["survives_final_exist"]:
                    counts[f"{t}_survives"] = counts.get(f"{t}_survives", 0) + 1
            trace.metadata["query_counts"] = counts
            trace.metadata["matched_query_count"] = sum(bool(r["matched"]) for r in query_rows)
            trace.metadata["surviving_query_count"] = sum(bool(r["survives_final_exist"]) for r in query_rows)

        if self.config.capture_temporal:
            trace.tables["temporal"] = build_temporal_table(recorder.capture)

        if self.config.capture_dense_outputs:
            trace.arrays.update(capture_dense_arrays(outputs, dtype=self.config.dense_float_dtype))

        if self.config.capture_scene_arrays:
            trace.arrays.update(capture_scene_arrays(batch, targets, dtype=self.config.scene_float_dtype))

        if query_rows:
            shape = np.asarray(scene_shape, dtype=np.float32)
            extent = (shape - 1.0) * spacing
            final_points, initial_points = [], []
            for row in query_rows:
                if not row["valid_query"]:
                    continue
                if "layer3_center_z_um" in row:
                    rel = np.asarray([row["layer3_center_z_um"], row["layer3_center_y_um"], row["layer3_center_x_um"]], dtype=np.float32)
                    vox = (rel + 0.5 * extent) / spacing
                    final_points.append(np.concatenate([vox, np.asarray([row["query"], row["query_type_id"], float(row["survives_final_exist"])], dtype=np.float32)]))
                if "initial_z_um" in row:
                    rel = np.asarray([row["initial_z_um"], row["initial_y_um"], row["initial_x_um"]], dtype=np.float32)
                    vox = (rel + 0.5 * extent) / spacing
                    initial_points.append(np.concatenate([vox, np.asarray([row["query"], row["query_type_id"]], dtype=np.float32)]))
            if final_points:
                trace.arrays["queries/final_points"] = np.stack(final_points).astype(np.float32)
            if initial_points:
                trace.arrays["queries/initial_points"] = np.stack(initial_points).astype(np.float32)

        if self.config.capture_native_masks and query_rows:
            selected = select_queries_for_deep_probe(
                query_rows,
                explicit=self.config.selected_query_indices,
                max_queries=self.config.max_selected_queries,
                worst_center=self.config.select_worst_center,
                worst_coarse_dice=self.config.select_worst_coarse_dice,
                top_temporal=self.config.select_top_surviving_temporal,
                top_split=self.config.select_top_split,
            )
            trace.metadata["selected_queries"] = selected
            mask_rows, mask_arrays = probe_native_masks(
                outputs,
                targets[0],
                selected,
                matching.query_to_target,
                mask_threshold=float(mask_threshold),
                chunk_voxels=self.config.native_chunk_voxels,
                prior_inside_logit=self.model.cfg.queries.prior_inside_logit,
                prior_outside_logit=self.model.cfg.queries.prior_outside_logit,
                temporal_sigma_dref=self.model.cfg.queries.temporal_gaussian_sigma_dref,
                native_support_radius_dref=self.model.cfg.queries.native_support_radius_dref,
                native_source_dilation_dref=self.model.cfg.queries.native_source_dilation_dref,
                native_background_logit=self.model.cfg.queries.native_background_logit,
                crop_margin_dref=self.config.mask_crop_margin_dref,
                unmatched_crop_radius_dref=self.config.unmatched_mask_crop_radius_dref,
                out_dtype=self.config.mask_float_dtype,
            )
            lookup = {int(r["query"]): r for r in query_rows}
            for row in mask_rows:
                qrow = lookup.get(int(row["query"]), {})
                row["query_type"] = qrow.get("query_type", "")
                row["gt_id"] = qrow.get("gt_id", -1)
                row["exist_prob"] = qrow.get("layer3_exist_prob", float("nan"))
                row["center_error_um"] = qrow.get("center_error_um", float("nan"))
            trace.tables["masks"] = mask_rows
            trace.arrays.update(mask_arrays)

        return trace
