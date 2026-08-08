from __future__ import annotations

import numpy as np

from learned.instance_segmentation.datasets.core.models import AnnotatedVolume, InstanceGroup
from learned.instance_segmentation.datasets.core.sample_builder import SampleBuildError
from learned.instance_segmentation.datasets.core.sample_selection import select_valid_sample


class _FakeBuilder:
    def __init__(self, rejected_ids: set[tuple[int, ...]]) -> None:
        self.rejected_ids = rejected_ids

    def build(self, volume, group):
        if group.instance_ids in self.rejected_ids:
            raise SampleBuildError(f"rejected {group.instance_ids}")
        return {"group": group.instance_ids}


def _volume() -> AnnotatedVolume:
    labels = np.zeros((4, 4, 4), dtype=np.int32)
    return AnnotatedVolume(
        image=np.zeros_like(labels, dtype=np.float32),
        instance_labels=labels,
        spacing_zyx_um=(1.0, 1.0, 1.0),
        dataset_name="synthetic",
        sample_id="v0",
    )


def test_valid_index_skips_unbuildable_raw_pairs() -> None:
    groups = (
        InstanceGroup((1, 2), "pair"),
        InstanceGroup((3, 4), "pair"),
        InstanceGroup((5, 6), "pair"),
    )
    builder = _FakeBuilder({(1, 2), (3, 4)})

    selection = select_valid_sample(_volume(), groups, builder, valid_index=0)

    assert selection.raw_index == 2
    assert selection.valid_index == 0
    assert selection.group.instance_ids == (5, 6)
    assert selection.sample == {"group": (5, 6)}
    assert [r.raw_index for r in selection.rejections_before] == [0, 1]


def test_valid_index_counts_only_successful_samples() -> None:
    groups = (
        InstanceGroup((1, 2), "pair"),
        InstanceGroup((3, 4), "pair"),
        InstanceGroup((5, 6), "pair"),
        InstanceGroup((7, 8), "pair"),
    )
    builder = _FakeBuilder({(3, 4)})

    selection = select_valid_sample(_volume(), groups, builder, valid_index=2)

    assert selection.raw_index == 3
    assert selection.group.instance_ids == (7, 8)
    assert len(selection.rejections_before) == 1
