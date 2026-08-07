import numpy as np
import pandas as pd

from learned.instance_segmentation.datasets.merge_real.config import MergeRealConfig
from learned.instance_segmentation.datasets.merge_real.observations import ObservationIndex
from learned.instance_segmentation.datasets.merge_real.sources.source2_two_one_two import mine_source2
from learned.instance_segmentation.datasets.merge_real.sources.source3_disappearing_track import mine_source3


class FakeSample:
    sample_id = "sample"
    frame_count = 4
    spatial_shape_zyx = (9, 25, 25)

    def __init__(self):
        self._labels = []
        self._cells = []
        for t in range(self.frame_count):
            labels = np.zeros(self.spatial_shape_zyx, dtype=np.int32)
            rows = []
            if t == 1:
                labels[3:6, 8:17, 6:19] = 30
                rows = [self._row(30, 3 * 9 * 13, 4, 12, 12)]
            else:
                labels[3:6, 8:17, 4:11] = 10 + t
                labels[3:6, 8:17, 14:21] = 20 + t
                rows = [
                    self._row(10 + t, 3 * 9 * 7, 4, 12, 7),
                    self._row(20 + t, 3 * 9 * 7, 4, 12, 17),
                ]
            self._labels.append(labels)
            self._cells.append(pd.DataFrame(rows))

        # Track 1 has a one-frame gap at t=1 and reappears at t=2.
        # Track 2 continues through the merged component at t=1.
        self.tracks = pd.DataFrame([
            {"track_id": 1, "frame": 0, "cell_id": 10, "z": 4, "y": 12, "x": 7, "volume": 189},
            {"track_id": 1, "frame": 2, "cell_id": 12, "z": 4, "y": 12, "x": 7, "volume": 189},
            {"track_id": 2, "frame": 0, "cell_id": 20, "z": 4, "y": 12, "x": 17, "volume": 189},
            {"track_id": 2, "frame": 1, "cell_id": 30, "z": 4, "y": 12, "x": 12, "volume": 351},
            {"track_id": 2, "frame": 2, "cell_id": 22, "z": 4, "y": 12, "x": 17, "volume": 189},
            {"track_id": 2, "frame": 3, "cell_id": 23, "z": 4, "y": 12, "x": 17, "volume": 189},
            # Track 3 disappears at t=2 into track 2's component at t=3.
            {"track_id": 3, "frame": 1, "cell_id": 31, "z": 4, "y": 12, "x": 9, "volume": 189},
            {"track_id": 3, "frame": 2, "cell_id": 13, "z": 4, "y": 12, "x": 9, "volume": 189},
        ])
        # Make frame 3 component for track 2 large enough to explain track 2 + 3.
        self._labels[3][:] = 0
        self._labels[3][3:6, 8:17, 6:19] = 23
        self._cells[3] = pd.DataFrame([self._row(23, 351, 4, 12, 12)])

        self.track_id_remap = pd.DataFrame(columns=["original_segment_track_id", "predecessor_track_id", "canonical_track_id", "from_frame", "to_frame"])
        self.unresolved_endings = pd.DataFrame([{"source_track_id": 3, "source_end_frame": 2}])
        self.division_events = pd.DataFrame(columns=["parent_track_id", "parent_end_frame", "child_birth_frame", "child_track_a", "child_track_b", "decision"])
        self.endpoint_classifications = pd.DataFrame(columns=["track_id", "last_real_frame", "last_is_boundary", "source_exclusion_reason"])

    @staticmethod
    def _row(cell_id, volume, z, y, x):
        return {
            "cell_id": cell_id,
            "volume_voxels": volume,
            "centroid_z": z,
            "centroid_y": y,
            "centroid_x": x,
            "z_min": 1, "y_min": 1, "x_min": 1,
            "z_max": 8, "y_max": 24, "x_max": 24,
        }

    def cells(self, frame):
        return self._cells[frame]

    def labels(self, frame, mmap=False):
        return self._labels[frame]


def test_source2_detects_two_one_two():
    sample = FakeSample()
    config = MergeRealConfig(prediction_component_radius_um=8.0)
    index = ObservationIndex(sample, config)
    results = mine_source2(sample, index, config)
    assert any(r.frame == 1 and r.cell_id == 30 and {r.track_a, r.track_b} == {1, 2} for r in results)


def test_source3_detects_disappearing_track_into_continuing_component():
    sample = FakeSample()
    config = MergeRealConfig(prediction_component_radius_um=8.0, source3_require_continuing_next_frame=False)
    index = ObservationIndex(sample, config)
    results = mine_source3(sample, index, config)
    assert any(r.frame == 3 and r.cell_id == 23 and {r.track_a, r.track_b} == {2, 3} for r in results)
