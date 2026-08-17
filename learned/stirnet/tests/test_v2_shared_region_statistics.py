from __future__ import annotations

from dataclasses import replace

import torch

from learned.stirnet.model.geometry.derived import build_geometry_derived_cache
from learned.stirnet.model.instances.tokenizer import InstanceTokenizer
from learned.stirnet.model.partition.rag import RAGBuilder
from learned.stirnet.model.partition.watershed import LearnedGeometryWatershed
from learned.stirnet.model.partition.statistics import (
    aggregate_supervoxel_statistics,
    build_supervoxel_statistics,
)
from learned.stirnet.model.types import (
    GeometryState,
    PartitionState,
    RefinedGeometryView,
    SparseGeometryDelta,
    SpatialDecodeState,
)
from learned.stirnet.model.utils.tensor_ops import reduce_labeled_voxels

from .conftest import small_model_config


def _fixture():
    torch.manual_seed(41)
    cfg = small_model_config()
    shape = (4, 8, 8)
    labels = torch.zeros(shape, dtype=torch.long)
    labels[:, :, :4] = 1
    labels[:, :, 4:] = 2

    def field(channels: int) -> torch.Tensor:
        return torch.randn((1, channels, *shape))

    geometry = GeometryState(
        foreground_logits=field(1),
        surface_logits=field(1),
        separator_logits=field(1),
        sdf=field(1),
        flow=field(3),
        centroid_offset=field(3),
        seed_logits=field(1),
        features=None,
    )
    decoded = SpatialDecodeState(
        d0=torch.randn((1, cfg.spatial.channels[0], *shape)),
        d1=torch.randn((1, cfg.spatial.channels[1], 2, 4, 4)),
        d2=torch.randn((1, cfg.spatial.channels[2], 1, 2, 2)),
    )
    inputs = torch.randn((1, cfg.spatial.in_channels, *shape))
    spacing = torch.tensor([[1.5, 0.4, 0.4]])
    dref = torch.tensor([3.0])
    cache = build_geometry_derived_cache(geometry, cfg.partition)
    statistics = build_supervoxel_statistics(
        [labels], inputs, geometry, spacing,
        (decoded.d0, decoded.d1, decoded.d2), derived=cache,
    )
    return cfg, labels, geometry, decoded, inputs, spacing, dref, cache, statistics


def test_supervoxel_and_aggregated_statistics_match_direct_voxel_reduction():
    _, labels, geometry, _, inputs, spacing, _, _, statistics = _fixture()
    stats = statistics[0]
    direct = reduce_labeled_voxels(
        labels,
        spacing[0],
        fields={"raw": inputs[0, 0], "sdf": geometry.sdf[0, 0]},
        argmax_field=geometry.sdf[0, 0],
        need_nearest_centroid=False,
    )
    torch.testing.assert_close(stats.counts, direct.counts)
    torch.testing.assert_close(stats.centroid_voxel, direct.centroid_voxel)
    torch.testing.assert_close(stats.variance_um2, direct.variance_um2)
    assert torch.equal(stats.min_voxel, direct.min_voxel)
    assert torch.equal(stats.max_voxel, direct.max_voxel)
    torch.testing.assert_close(stats.field_means("raw"), direct.field_means["raw"])
    torch.testing.assert_close(stats.field_maxima["sdf"], direct.field_maxima["sdf"])
    assert torch.equal(stats.sdf_argmax_flat_index, direct.argmax_flat_index)

    aggregated = aggregate_supervoxel_statistics(
        stats, torch.tensor([0, 0]), 1
    )
    instance_labels = (labels > 0).long()
    direct_instance = reduce_labeled_voxels(
        instance_labels,
        spacing[0],
        fields={"raw": inputs[0, 0], "sdf": geometry.sdf[0, 0]},
        argmax_field=geometry.sdf[0, 0],
        need_nearest_centroid=False,
    )
    torch.testing.assert_close(aggregated.counts, direct_instance.counts)
    torch.testing.assert_close(aggregated.centroid_voxel, direct_instance.centroid_voxel)
    torch.testing.assert_close(aggregated.variance_um2, direct_instance.variance_um2)
    torch.testing.assert_close(
        aggregated.field_means("raw"), direct_instance.field_means["raw"]
    )
    assert torch.equal(
        aggregated.sdf_argmax_flat_index, direct_instance.argmax_flat_index
    )


def test_cached_rag_and_compact_tokenizer_match_reference_paths(monkeypatch):
    cfg, labels, geometry, decoded, inputs, spacing, dref, cache, statistics = _fixture()
    builder = RAGBuilder(cfg.partition, cfg.spatial).eval()
    reference_rag = builder(
        [labels], decoded.d0, inputs, geometry, spacing, dref
    )
    cached_rag = builder(
        [labels], decoded.d0, inputs, geometry, spacing, dref,
        statistics_by_batch=statistics, derived_cache=cache,
    )
    torch.testing.assert_close(cached_rag.node_features, reference_rag.node_features)
    assert torch.equal(cached_rag.edge_index, reference_rag.edge_index)
    torch.testing.assert_close(cached_rag.edge_features, reference_rag.edge_features)

    partition = PartitionState(
        labels=[(labels > 0).long()],
        node_component=torch.tensor([0, 0]),
        node_component_global=torch.tensor([0, 0]),
        component_count_per_batch=torch.tensor([1]),
        edge_logits=torch.zeros(cached_rag.edge_index.shape[1]),
    )
    tokenizer = InstanceTokenizer(cfg.instances, cfg.spatial).eval()
    reference_instances = tokenizer(
        partition, replace(reference_rag, statistics=None), decoded,
        geometry, spacing, dref,
    )

    def fail_materialize(*_args, **_kwargs):
        raise AssertionError("compact tokenizer materialized refined geometry")

    monkeypatch.setattr(RefinedGeometryView, "materialize_field", fail_materialize)
    refined = RefinedGeometryView(geometry, SparseGeometryDelta())
    compact_instances = tokenizer(
        partition, cached_rag, decoded, refined, spacing, dref
    )
    torch.testing.assert_close(compact_instances.tokens, reference_instances.tokens)
    torch.testing.assert_close(compact_instances.ref_um, reference_instances.ref_um)
    torch.testing.assert_close(
        compact_instances.quality_logits, reference_instances.quality_logits
    )


def test_fast_component_bounded_watershed_is_deterministic_and_contiguous():
    cfg = small_model_config()
    cfg.partition.watershed_backend = "fast"
    cfg.partition.component_bounded_watershed = True
    cfg.partition.min_supervoxel_voxels = 1
    shape = (6, 12, 12)
    foreground = torch.full((1, 1, *shape), -10.0)
    foreground[:, :, 1:3, 1:5, 1:5] = 10.0
    foreground[:, :, 3:5, 7:11, 7:11] = 10.0
    seed = torch.full_like(foreground, -10.0)
    seed[0, 0, 2, 3, 3] = 10.0
    seed[0, 0, 4, 9, 9] = 10.0
    sdf = torch.zeros_like(foreground)
    sdf[0, 0, 2, 3, 3] = 1.0
    sdf[0, 0, 4, 9, 9] = 1.0
    zeros = torch.zeros_like(foreground)
    geometry = GeometryState(
        foreground_logits=foreground,
        surface_logits=zeros,
        separator_logits=zeros,
        sdf=sdf,
        flow=torch.zeros((1, 3, *shape)),
        centroid_offset=torch.zeros((1, 3, *shape)),
        seed_logits=seed,
        features=None,
    )
    module = LearnedGeometryWatershed(cfg.partition)
    args = (geometry, torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([2.0]))
    first = module(*args)[0]
    second = module(*args)[0]
    assert torch.equal(first, second)
    assert torch.equal(torch.unique(first), torch.tensor([0, 1, 2]))
    assert torch.all(first[foreground[0, 0] < 0] == 0)
