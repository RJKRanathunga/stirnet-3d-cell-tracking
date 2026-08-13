from __future__ import annotations

from contextlib import AbstractContextManager

import torch
from torch import Tensor

from .config import DebugConfig
from .stats import feature_norm_volume, tensor_stats
from .trace import HookCapture
from ...model.types import QueryState, SpatialPyramid, TemporalState


def _cpu_float(t: Tensor) -> Tensor:
    return t.detach().float().cpu()


def _copy_temporal(state: TemporalState | None):
    if state is None:
        return None
    return {
        "tokens": _cpu_float(state.tokens),
        "ref_um": _cpu_float(state.ref_um),
        "ref_cellscale": _cpu_float(state.ref_cellscale),
        "salience": _cpu_float(state.salience),
        "reliability": _cpu_float(state.reliability),
        "status": _cpu_float(state.status),
        "batch_index": state.batch_index.detach().long().cpu(),
        "history_support_valid": state.history_support_valid.detach().bool().cpu() if state.history_support_valid is not None else None,
        "history_support_dt": _cpu_float(state.history_support_dt) if state.history_support_dt is not None else None,
        "history_gate": _cpu_float(state.history_gate) if state.history_gate is not None else None,
        "node_history_valid": state.node_history_valid.detach().bool().cpu() if state.node_history_valid is not None else None,
        "best_component_overlap": _cpu_float(state.best_component_overlap) if state.best_component_overlap is not None else None,
    }


def _copy_query(state: QueryState | None):
    if state is None:
        return None
    return {
        "embeddings": _cpu_float(state.embeddings),
        "references_cellscale": _cpu_float(state.references_cellscale),
        "query_types": state.query_types.detach().long().cpu(),
        "padding_mask": state.padding_mask.detach().bool().cpu(),
        "source_instance_ids": state.source_instance_ids.detach().long().cpu(),
        "temporal_salience": _cpu_float(state.temporal_salience),
        "temporal_reliability": _cpu_float(state.temporal_reliability),
    }


def _copy_decoder_output(value):
    if not isinstance(value, tuple) or len(value) != 2:
        return None
    state, out = value
    if not isinstance(out, dict):
        return None
    return {
        "state": _copy_query(state),
        "exist_logits": _cpu_float(out["exist_logits"]),
        "centers_cellscale": _cpu_float(out["centers_cellscale"]),
        "coarse_mask_logits": _cpu_float(out["coarse_mask_logits"]),
        "query_embeddings": _cpu_float(out["query_embeddings"]),
    }


class StirNetHookRecorder(AbstractContextManager):
    """Observe current STIR-Net modules using forward hooks only.

    Large feature volumes are summarized on-device; only small structured
    temporal/query states are copied to CPU.
    """

    def __init__(self, model, config: DebugConfig):
        self.model = model
        self.config = config
        self.capture = HookCapture()
        self._handles = []

    def _add(self, module, fn, *, pre: bool = False):
        handle = (
            module.register_forward_pre_hook(fn)
            if pre
            else module.register_forward_hook(fn)
        )
        self._handles.append(handle)

    def _record_tensor(self, name: str, tensor: Tensor):
        self.capture.module_stats.append(
            tensor_stats(
                tensor,
                name=name,
                channel_stats=self.config.capture_channel_stats,
            )
        )

    def _encoder_hook(self, module, inputs, output):
        if not isinstance(output, SpatialPyramid):
            return
        info = {
            "spacings_um": [s.detach().float().cpu() for s in output.spacings_um],
            "strides": list(output.strides),
            "feature_shapes": [tuple(f.shape) for f in output.features],
        }
        for i, feature in enumerate(output.features):
            self._record_tensor(f"spatial.encoder.E{i}", feature)
            if self.config.capture_feature_norm_volumes:
                info[f"E{i}_norm"] = feature_norm_volume(feature)
        self.capture.encoder_pyramid = info

    def _graph_hook(self, module, inputs, output):
        self._record_tensor("temporal.graph_encoder", output)
        self.capture.graph_node_embeddings = _cpu_float(output)

    def _pool_hook(self, module, inputs, output):
        self._record_tensor("temporal.tracklet_pooler", output)
        self.capture.pooled_tracklets = _cpu_float(output)

    def _temporal_builder_hook(self, module, inputs, output):
        self.capture.temporal_initial = _copy_temporal(output)
        if isinstance(output, TemporalState):
            self._record_tensor("temporal.initial_tokens", output.tokens)

    def _cr_pre(self, which: str):
        def hook(module, inputs):
            if len(inputs) < 3:
                return
            spatial, temporal = inputs[0], inputs[2]
            self._record_tensor(f"coreasoning.{which}.spatial_in", spatial)
            if which == "cr1":
                self.capture.temporal_before_cr1 = _copy_temporal(temporal)
            else:
                self.capture.temporal_before_cr2 = _copy_temporal(temporal)
        return hook

    def _cr_post(self, which: str):
        def hook(module, inputs, output):
            if not isinstance(output, tuple) or len(output) != 2:
                return
            spatial, temporal = output
            self._record_tensor(f"coreasoning.{which}.spatial_out", spatial)
            if which == "cr1":
                self.capture.temporal_after_cr1 = _copy_temporal(temporal)
            else:
                self.capture.temporal_after_cr2 = _copy_temporal(temporal)
        return hook

    def _decoder_stage(self, name: str):
        def hook(module, inputs, output):
            self._record_tensor(name, output)
        return hook

    def _query_builder_hook(self, module, inputs, output):
        self.capture.query_initial = _copy_query(output)
        if isinstance(output, QueryState):
            self._record_tensor("queries.initial_embeddings", output.embeddings)

    def _query_layer_hook(self, index: int):
        def hook(module, inputs, output):
            copied = _copy_decoder_output(output)
            if copied is None:
                return
            while len(self.capture.decoder_layers) <= index:
                self.capture.decoder_layers.append(None)
            self.capture.decoder_layers[index] = copied
            self._record_tensor(
                f"queries.decoder{index + 1}.embeddings",
                output[0].embeddings,
            )
        return hook

    def _dense_hook(self, module, inputs, output):
        if isinstance(output, dict):
            for key, value in output.items():
                self._record_tensor(f"dense.{key}", value)

    def _native_head_hook(self, module, inputs, output):
        self._record_tensor("heads.native_mask_embeddings", output)

    def __enter__(self):
        if self.config.capture_spatial:
            self._add(self.model.encoder, self._encoder_hook)
            self._add(self.model.decoder.stage_e2, self._decoder_stage("spatial.decoder.E2"))
            self._add(self.model.decoder.stage_e1, self._decoder_stage("spatial.decoder.D1"))
            self._add(self.model.decoder.stage_e0, self._decoder_stage("spatial.decoder.D0"))

        if self.config.capture_temporal:
            self._add(self.model.graph_encoder, self._graph_hook)
            self._add(self.model.tracklet_pooler, self._pool_hook)
            self._add(self.model.temporal_builder, self._temporal_builder_hook)
            self._add(self.model.cr1, self._cr_pre("cr1"), pre=True)
            self._add(self.model.cr1, self._cr_post("cr1"))
            self._add(self.model.cr2, self._cr_pre("cr2"), pre=True)
            self._add(self.model.cr2, self._cr_post("cr2"))

        if self.config.capture_queries:
            self._add(self.model.query_builder, self._query_builder_hook)
            for index, layer in enumerate(self.model.query_decoder.layers):
                self._add(layer, self._query_layer_hook(index))

        if self.config.capture_module_stats:
            self._add(self.model.dense_heads, self._dense_hook)
            self._add(self.model.native_mask_head, self._native_head_hook)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False
