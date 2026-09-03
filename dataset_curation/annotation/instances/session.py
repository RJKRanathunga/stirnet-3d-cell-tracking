from __future__ import annotations

"""Lazy per-frame spatial annotation state for split, merge, and hallucination edits."""

from collections import OrderedDict
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataset_curation.annotation.instances.split import (
    AnnotationError,
    SplitResult,
    _expand_seed_groups_by_contact_graph,
)
from dataset_curation.io.atomic import atomic_json, read_json


@dataclass(frozen=True)
class SpatialUndoResult:
    operation_type: str
    timepoint: int
    message: str


@dataclass(frozen=True)
class SpatialMergeResult:
    timepoint: int
    selected_supervoxel_ids: tuple[int, ...]
    source_instance_ids: tuple[int, ...]
    output_instance_id: int


def _dominant_parent_instance(
    sv_frame: np.ndarray,
    instance_frame: np.ndarray,
    sv_id: int,
) -> int:
    mask = np.asarray(sv_frame) == int(sv_id)
    values, counts = np.unique(
        np.asarray(instance_frame)[mask],
        return_counts=True,
    )
    nonzero = values > 0
    if not np.any(nonzero):
        return 0
    values = values[nonzero]
    counts = counts[nonzero]
    return int(values[np.argmax(counts)])


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class AnnotationSession:
    """
    Spatial correction state without a full corrected 4-D RAM copy.

    Base instance and supervoxel movies remain memory-mapped. Corrected frames
    are loaded on demand, cached in a small LRU, and persisted independently.
    """

    SCHEMA_VERSION = 3

    def __init__(
        self,
        *,
        sample_id: str,
        timepoints: tuple[int, ...],
        supervoxels: np.ndarray,
        base_instances: np.ndarray,
        output_dir: Path,
        resume: bool,
        cache_frames: int = 4,
    ) -> None:
        self.sample_id = str(sample_id)
        self.timepoints = tuple(int(v) for v in timepoints)
        self.supervoxels = supervoxels
        self.base_instances = base_instances
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_frames = max(int(cache_frames), 1)

        if self.supervoxels.ndim != 4 or self.base_instances.ndim != 4:
            raise AnnotationError(
                "Spatial annotation expects (T,Z,Y,X) supervoxel and instance movies."
            )
        if self.supervoxels.shape != self.base_instances.shape:
            raise AnnotationError(
                "Supervoxel and base-instance movies do not align: "
                f"{self.supervoxels.shape} vs {self.base_instances.shape}"
            )
        if len(self.timepoints) != int(self.base_instances.shape[0]):
            raise AnnotationError(
                "Unified annotation requires one dataset timepoint for every frame."
            )

        self.state_path = self.output_dir / "spatial_operations.json"
        self.hallucinations_path = self.output_dir / "hallucinations.csv"
        self.operations: list[dict[str, Any]] = []
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

        self._resume_existing = bool(resume)

        if self._resume_existing and self.state_path.is_file():
            self._load_state()
        elif not self._resume_existing:
            # Explicit fresh mode starts a new unified state and removes stale
            # corrected-frame files from the same annotation set.
            self.operations = []
            for stale in self.output_dir.glob("manual_instances_t*.npy"):
                stale.unlink(missing_ok=True)
            self._persist_state()

        max_base = int(np.max(self.base_instances)) if self.base_instances.size else 0
        max_logged = 0
        for op in self.operations:
            if op.get("type") == "split":
                for group in op.get("groups", []):
                    max_logged = max(
                        max_logged,
                        int(group.get("output_instance_id", 0)),
                    )
            elif op.get("type") == "merge":
                max_logged = max(
                    max_logged,
                    int(op.get("output_instance_id", 0)),
                )
        self.next_label = max(max_base, max_logged) + 1

    def _local_index_for_dataset_timepoint(self, dataset_t: int) -> int:
        try:
            return self.timepoints.index(int(dataset_t))
        except ValueError as exc:
            raise AnnotationError(
                f"Dataset timepoint {dataset_t} is not part of this session."
            ) from exc

    def output_path_for_timepoint(self, dataset_t: int) -> Path:
        return self.output_dir / f"manual_instances_t{int(dataset_t):03d}.npy"

    def _load_state(self) -> None:
        payload = read_json(self.state_path)
        if int(payload.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise AnnotationError(
                f"Unsupported spatial annotation schema in {self.state_path}: "
                f"{payload.get('schema_version')!r}. "
                "This unified annotator intentionally does not carry legacy "
                "annotation-state compatibility."
            )
        if str(payload.get("sample_id")) != self.sample_id:
            raise AnnotationError(
                f"Spatial state belongs to {payload.get('sample_id')!r}, "
                f"not {self.sample_id!r}."
            )

        existing_timepoints = tuple(
            int(v) for v in payload.get("timepoints", [])
        )
        if existing_timepoints != self.timepoints:
            raise AnnotationError(
                "Spatial annotation state was created for a different frame set."
            )

        self.operations = list(payload.get("operations", []))

        for local_t in self.changed_local_indices():
            dataset_t = self.timepoints[local_t]
            path = self.output_path_for_timepoint(dataset_t)
            if not path.is_file():
                raise AnnotationError(
                    f"Spatial state references t={dataset_t}, but corrected "
                    f"frame file is missing: {path}"
                )

    def changed_local_indices(self) -> tuple[int, ...]:
        changed: set[int] = set()
        for op in self.operations:
            if "timepoint" not in op:
                continue
            changed.add(
                self._local_index_for_dataset_timepoint(
                    int(op["timepoint"])
                )
            )
        return tuple(sorted(changed))

    def _cache_put(self, local_t: int, frame: np.ndarray) -> np.ndarray:
        self._cache.pop(int(local_t), None)
        self._cache[int(local_t)] = frame
        while len(self._cache) > self.cache_frames:
            self._cache.popitem(last=False)
        return frame

    def frame(self, local_t: int) -> np.ndarray:
        local_t = int(local_t)
        if not (0 <= local_t < len(self.timepoints)):
            raise AnnotationError(f"Invalid frame index: {local_t}")

        cached = self._cache.pop(local_t, None)
        if cached is not None:
            self._cache[local_t] = cached
            return cached

        dataset_t = self.timepoints[local_t]
        corrected_path = self.output_path_for_timepoint(dataset_t)
        changed_now = local_t in self.changed_local_indices()
        if corrected_path.is_file() and (
            self._resume_existing or changed_now
        ):
            frame = np.load(
                corrected_path,
                allow_pickle=False,
            ).astype(np.int32, copy=False)
        else:
            frame = np.asarray(
                self.base_instances[local_t],
                dtype=np.int32,
            ).copy()

        expected = tuple(int(v) for v in self.base_instances.shape[1:])
        if frame.shape != expected:
            raise AnnotationError(
                f"Corrected frame t={dataset_t} has shape {frame.shape}; "
                f"expected {expected}."
            )
        return self._cache_put(local_t, frame)

    def hallucinated_supervoxels(self, local_t: int) -> set[int]:
        dataset_t = int(self.timepoints[int(local_t)])
        return {
            int(op["supervoxel_id"])
            for op in self.operations
            if op.get("type") == "hallucination"
            and int(op.get("timepoint", -1)) == dataset_t
        }

    def split_corrections_in_frame(self, local_t: int) -> int:
        dataset_t = int(self.timepoints[int(local_t)])
        return sum(
            1
            for op in self.operations
            if op.get("type") == "split"
            and int(op.get("timepoint", -1)) == dataset_t
        )

    def hallucinations_in_frame(self, local_t: int) -> int:
        return len(self.hallucinated_supervoxels(local_t))

    def merge_corrections_in_frame(self, local_t: int) -> int:
        dataset_t = int(self.timepoints[int(local_t)])
        return sum(
            1
            for op in self.operations
            if op.get("type") == "merge"
            and int(op.get("timepoint", -1)) == dataset_t
        )

    def _parent_instance_for_sv(self, local_t: int, sv_id: int) -> int:
        return _dominant_parent_instance(
            np.asarray(self.supervoxels[int(local_t)]),
            self.frame(int(local_t)),
            int(sv_id),
        )

    def current_instance_for_supervoxel(
        self,
        local_t: int,
        sv_id: int,
    ) -> int:
        """Return the current corrected instance containing a supervoxel."""
        return self._parent_instance_for_sv(
            int(local_t),
            int(sv_id),
        )

    def _expected_supervoxels(
        self,
        local_t: int,
        original_instance_id: int,
    ) -> set[int]:
        frame = self.frame(local_t)
        sv_frame = np.asarray(self.supervoxels[local_t])
        mask = frame == int(original_instance_id)

        if np.any(mask & (sv_frame == 0)):
            missing_voxels = int(np.count_nonzero(mask & (sv_frame == 0)))
            raise AnnotationError(
                f"Instance {original_instance_id} contains {missing_voxels} "
                "voxels with supervoxel ID 0; it cannot be split losslessly."
            )

        ids = np.unique(sv_frame[mask])
        return {int(v) for v in ids.tolist() if int(v) > 0}

    def apply_split(
        self,
        local_t: int,
        groups: list[list[int]],
    ) -> SplitResult:
        local_t = int(local_t)
        frame = self.frame(local_t)
        sv_frame = np.asarray(self.supervoxels[local_t])

        seed_groups = tuple(
            tuple(int(v) for v in group)
            for group in groups
            if group
        )
        if len(seed_groups) < 2:
            raise AnnotationError(
                "At least two instance seed boxes must be filled before Save Split."
            )
        if len(seed_groups) > 4:
            raise AnnotationError("At most four split outputs are supported.")

        flat = [sv_id for group in seed_groups for sv_id in group]
        if len(flat) != len(set(flat)):
            raise AnnotationError(
                "A supervoxel cannot be used as a seed for two split outputs."
            )

        hallucinated = self.hallucinated_supervoxels(local_t)
        invalid_hallucinated = sorted(set(flat) & hallucinated)
        if invalid_hallucinated:
            raise AnnotationError(
                "Hallucinated supervoxels cannot be used as split seeds: "
                f"{invalid_hallucinated}"
            )

        frame_ids = {
            int(v)
            for v in np.unique(sv_frame).tolist()
            if int(v) > 0
        }
        missing = sorted(set(flat) - frame_ids)
        if missing:
            raise AnnotationError(
                f"These supervoxels are absent from the current frame: {missing}"
            )

        parent_ids = {
            self._parent_instance_for_sv(local_t, sv_id)
            for sv_id in flat
        }
        if 0 in parent_ids:
            raise AnnotationError(
                "At least one selected seed is currently background."
            )
        if len(parent_ids) != 1:
            raise AnnotationError(
                "Selected seed supervoxels already belong to different current "
                f"instances: {sorted(parent_ids)}."
            )

        original_instance_id = int(next(iter(parent_ids)))
        parent_supervoxels = self._expected_supervoxels(
            local_t,
            original_instance_id,
        )
        expanded_groups = _expand_seed_groups_by_contact_graph(
            sv_frame=sv_frame,
            parent_supervoxels=parent_supervoxels,
            seed_groups=seed_groups,
        )

        original_mask = frame == original_instance_id
        frame[original_mask] = 0

        output_ids: list[int] = []
        group_records: list[dict[str, Any]] = []

        for seed_group, expanded_group in zip(
            seed_groups,
            expanded_groups,
        ):
            new_instance_id = int(self.next_label)
            self.next_label += 1

            group_mask = np.isin(
                sv_frame,
                np.asarray(expanded_group, dtype=sv_frame.dtype),
            ) & original_mask
            frame[group_mask] = new_instance_id
            output_ids.append(new_instance_id)
            group_records.append(
                {
                    "output_instance_id": new_instance_id,
                    "seed_supervoxels": [int(v) for v in seed_group],
                    "assigned_supervoxels": [int(v) for v in expanded_group],
                }
            )

        if np.any(frame[original_mask] == 0):
            raise AnnotationError(
                "Internal error: split left part of the original instance unassigned."
            )

        dataset_t = int(self.timepoints[local_t])
        self.operations.append(
            {
                "type": "split",
                "timepoint": dataset_t,
                "original_instance_id": original_instance_id,
                "split_method": (
                    "multi_source_dijkstra_inverse_physical_contact_area"
                ),
                "groups": group_records,
            }
        )
        self._persist_changed_frame(local_t)

        return SplitResult(
            timepoint=dataset_t,
            original_instance_id=original_instance_id,
            output_instance_ids=tuple(output_ids),
            seed_groups=seed_groups,
            groups=expanded_groups,
        )

    def apply_merge(
        self,
        local_t: int,
        supervoxel_ids: list[int],
    ) -> SpatialMergeResult:
        """Merge the full corrected instances identified by selected SVs."""
        local_t = int(local_t)
        selected = tuple(
            int(value)
            for value in supervoxel_ids
            if int(value) > 0
        )
        if len(selected) < 2:
            raise AnnotationError(
                "Save Merge requires at least two selected supervoxels."
            )
        if len(selected) != len(set(selected)):
            raise AnnotationError(
                "The same supervoxel cannot be selected twice for Save Merge."
            )

        sv_frame = np.asarray(self.supervoxels[local_t])
        frame = self.frame(local_t)

        hallucinated = self.hallucinated_supervoxels(local_t)
        invalid_hallucinated = sorted(set(selected) & hallucinated)
        if invalid_hallucinated:
            raise AnnotationError(
                "Hallucinated supervoxels cannot be merged: "
                f"{invalid_hallucinated}"
            )

        frame_sv_ids = {
            int(value)
            for value in np.unique(sv_frame).tolist()
            if int(value) > 0
        }
        missing = sorted(set(selected) - frame_sv_ids)
        if missing:
            raise AnnotationError(
                "These supervoxels are absent from the current frame: "
                f"{missing}"
            )

        parent_ids: list[int] = []
        for sv_id in selected:
            parent = self._parent_instance_for_sv(
                local_t,
                sv_id,
            )
            if parent <= 0:
                raise AnnotationError(
                    f"Supervoxel {sv_id} is currently background."
                )
            parent_ids.append(int(parent))

        source_instance_ids = tuple(
            sorted(set(parent_ids))
        )
        if len(source_instance_ids) < 2:
            raise AnnotationError(
                "Save Merge needs supervoxels from at least two different "
                "current corrected instances. The selected supervoxels "
                f"all belong to instance {source_instance_ids[0]}."
            )

        source_groups: list[dict[str, Any]] = []
        for instance_id in source_instance_ids:
            source_supervoxels = sorted(
                self._expected_supervoxels(
                    local_t,
                    int(instance_id),
                )
            )
            if not source_supervoxels:
                raise AnnotationError(
                    f"Instance {instance_id} has no positive supervoxels."
                )
            source_groups.append(
                {
                    "instance_id": int(instance_id),
                    "supervoxels": [
                        int(value)
                        for value in source_supervoxels
                    ],
                }
            )

        merge_mask = np.isin(
            frame,
            np.asarray(
                source_instance_ids,
                dtype=frame.dtype,
            ),
        )
        if not np.any(merge_mask):
            raise AnnotationError(
                "Selected source instances are no longer present."
            )

        output_instance_id = int(self.next_label)
        self.next_label += 1
        frame[merge_mask] = output_instance_id

        dataset_t = int(self.timepoints[local_t])
        self.operations.append(
            {
                "type": "merge",
                "timepoint": dataset_t,
                "selected_supervoxels": [
                    int(value)
                    for value in selected
                ],
                "source_instance_ids": [
                    int(value)
                    for value in source_instance_ids
                ],
                "source_groups": source_groups,
                "output_instance_id": output_instance_id,
            }
        )
        self._persist_changed_frame(local_t)

        return SpatialMergeResult(
            timepoint=dataset_t,
            selected_supervoxel_ids=selected,
            source_instance_ids=source_instance_ids,
            output_instance_id=output_instance_id,
        )

    # DATASET_CURATION_MULTI_HALLUCINATION_V1
    def apply_hallucinations(
        self,
        local_t: int,
        sv_ids: list[int] | tuple[int, ...],
    ) -> tuple[dict[str, Any], ...]:
        """Mark all selected supervoxels as hallucinations atomically."""
        local_t = int(local_t)
        selected = tuple(
            int(value)
            for value in sv_ids
            if int(value) > 0
        )
        if not selected:
            raise AnnotationError(
                "Select at least one positive supervoxel first."
            )
        if len(selected) != len(set(selected)):
            raise AnnotationError(
                "The same supervoxel cannot be selected twice for Hallucination."
            )

        already_hallucinated = self.hallucinated_supervoxels(local_t)
        repeated = sorted(
            set(selected) & already_hallucinated
        )
        if repeated:
            raise AnnotationError(
                "These supervoxels are already marked as hallucinations: "
                f"{repeated}"
            )

        sv_frame = np.asarray(
            self.supervoxels[local_t]
        )
        frame = self.frame(local_t)
        dataset_t = int(
            self.timepoints[local_t]
        )

        pending: list[
            tuple[int, np.ndarray, int, int]
        ] = []

        # Validate the whole selection before mutating the corrected frame.
        for sv_id in selected:
            mask = sv_frame == int(sv_id)
            voxel_count = int(
                np.count_nonzero(mask)
            )
            if voxel_count == 0:
                raise AnnotationError(
                    f"Supervoxel {sv_id} does not exist in this frame."
                )

            parent_ids = np.unique(
                frame[mask]
            )
            parent_ids = parent_ids[
                parent_ids > 0
            ]
            if len(parent_ids) == 0:
                raise AnnotationError(
                    f"Supervoxel {sv_id} is already background in the "
                    "corrected labels."
                )
            if len(parent_ids) != 1:
                raise AnnotationError(
                    f"Supervoxel {sv_id} overlaps multiple corrected "
                    f"instances: {parent_ids.tolist()}."
                )

            pending.append(
                (
                    int(sv_id),
                    mask,
                    int(parent_ids[0]),
                    int(voxel_count),
                )
            )

        batch_id: str | None = None
        if len(pending) > 1:
            batch_id = (
                f"hallucination-{dataset_t}-"
                f"{len(self.operations)}"
            )

        records: list[
            dict[str, Any]
        ] = []

        for (
            sv_id,
            mask,
            previous_instance_id,
            voxel_count,
        ) in pending:
            frame[mask] = 0

            record: dict[str, Any] = {
                "type": "hallucination",
                "timepoint": dataset_t,
                "supervoxel_id": int(sv_id),
                "previous_instance_id": int(
                    previous_instance_id
                ),
                "voxel_count": int(
                    voxel_count
                ),
            }
            if batch_id is not None:
                record[
                    "batch_id"
                ] = batch_id

            records.append(
                record
            )

        self.operations.extend(
            records
        )
        self._persist_changed_frame(
            local_t
        )
        return tuple(records)

    def apply_hallucination(
        self,
        local_t: int,
        sv_id: int,
    ) -> dict[str, Any]:
        """Backward-compatible single-supervoxel Hallucination API."""
        return self.apply_hallucinations(
            int(local_t),
            [int(sv_id)],
        )[0]

    def can_undo(self) -> bool:
        return bool(self.operations)

    def undo(self) -> SpatialUndoResult:
        if not self.operations:
            raise AnnotationError("There is no spatial operation to undo.")

        op = self.operations[-1]
        op_type = str(op.get("type"))
        dataset_t = int(op["timepoint"])
        undo_count = 1
        local_t = self._local_index_for_dataset_timepoint(dataset_t)
        frame = self.frame(local_t)
        sv_frame = np.asarray(self.supervoxels[local_t])

        if op_type == "split":
            output_ids = tuple(
                int(group["output_instance_id"])
                for group in op.get("groups", [])
            )
            if len(output_ids) < 2:
                raise AnnotationError(
                    "Split history does not contain enough output IDs to undo."
                )
            mask = np.isin(
                frame,
                np.asarray(output_ids, dtype=frame.dtype),
            )
            if not np.any(mask):
                raise AnnotationError(
                    "Cannot undo split: its generated labels are no longer present."
                )
            original_id = int(op["original_instance_id"])
            frame[mask] = original_id
            message = (
                f"restored instance {original_id}; removed split labels "
                f"{output_ids}"
            )

        elif op_type == "merge":
            output_id = int(op["output_instance_id"])
            output_mask = frame == output_id
            if not np.any(output_mask):
                raise AnnotationError(
                    "Cannot undo merge: its output instance "
                    f"{output_id} is no longer present."
                )

            restored = np.zeros(
                frame.shape,
                dtype=bool,
            )
            source_groups = list(
                op.get("source_groups", [])
            )
            if len(source_groups) < 2:
                raise AnnotationError(
                    "Merge history does not contain enough source groups "
                    "to restore the original instances."
                )

            for group in source_groups:
                source_id = int(group["instance_id"])
                sv_ids = [
                    int(value)
                    for value in group.get(
                        "supervoxels",
                        [],
                    )
                ]
                if not sv_ids:
                    raise AnnotationError(
                        "Merge history contains an empty source "
                        f"supervoxel group for instance {source_id}."
                    )

                group_mask = (
                    output_mask
                    & np.isin(
                        sv_frame,
                        np.asarray(
                            sv_ids,
                            dtype=sv_frame.dtype,
                        ),
                    )
                )
                frame[group_mask] = source_id
                restored |= group_mask

            if np.any(output_mask & ~restored):
                raise AnnotationError(
                    "Cannot undo merge losslessly: some output voxels "
                    "cannot be assigned back to their source instances."
                )

            source_ids = tuple(
                int(value)
                for value in op.get(
                    "source_instance_ids",
                    [],
                )
            )
            message = (
                f"restored merged instances {source_ids}; "
                f"removed merged label {output_id}"
            )

        elif op_type == "hallucination":
            batch_id = op.get(
                "batch_id"
            )

            if batch_id is None:
                hallucination_ops = [
                    op
                ]
            else:
                hallucination_ops: list[
                    dict[str, Any]
                ] = []
                index = len(
                    self.operations
                ) - 1
                while index >= 0:
                    candidate = (
                        self.operations[
                            index
                        ]
                    )
                    if (
                        candidate.get(
                            "type"
                        )
                        != "hallucination"
                        or candidate.get(
                            "batch_id"
                        )
                        != batch_id
                    ):
                        break
                    hallucination_ops.append(
                        candidate
                    )
                    index -= 1

                hallucination_ops.reverse()

            # Validate the complete undo before changing the frame.
            restore_items: list[
                tuple[
                    int,
                    np.ndarray,
                    int,
                ]
            ] = []
            for hallucination_op in hallucination_ops:
                sv_id = int(
                    hallucination_op[
                        "supervoxel_id"
                    ]
                )
                mask = (
                    sv_frame
                    == sv_id
                )
                if not np.any(
                    mask
                ):
                    raise AnnotationError(
                        "Cannot undo hallucination: "
                        f"SV {sv_id} no longer exists."
                    )
                previous_id = int(
                    hallucination_op[
                        "previous_instance_id"
                    ]
                )
                restore_items.append(
                    (
                        sv_id,
                        mask,
                        previous_id,
                    )
                )

            for (
                sv_id,
                mask,
                previous_id,
            ) in restore_items:
                frame[
                    mask
                ] = previous_id

            undo_count = len(
                hallucination_ops
            )
            if undo_count == 1:
                sv_id, _, previous_id = (
                    restore_items[
                        0
                    ]
                )
                message = (
                    f"restored hallucinated SV {sv_id} "
                    f"to instance {previous_id}"
                )
            else:
                restored_ids = tuple(
                    int(item[0])
                    for item in restore_items
                )
                message = (
                    "restored hallucination batch "
                    f"{restored_ids}"
                )
        else:
            raise AnnotationError(
                f"Unknown spatial operation type: {op_type!r}"
            )

        del self.operations[
            -int(undo_count):
        ]
        self._persist_changed_frame(local_t)
        return SpatialUndoResult(
            operation_type=op_type,
            timepoint=dataset_t,
            message=message,
        )

    def _persist_changed_frame(self, local_t: int) -> None:
        dataset_t = int(self.timepoints[int(local_t)])
        path = self.output_path_for_timepoint(dataset_t)
        np.save(
            path,
            self.frame(local_t).astype(np.int32, copy=False),
            allow_pickle=False,
        )
        self._persist_state()

    def _persist_state(self) -> None:
        atomic_json(
            self.state_path,
            {
                "schema_version": self.SCHEMA_VERSION,
                "sample_id": self.sample_id,
                "timepoints": [int(t) for t in self.timepoints],
                "node_identity": ["frame", "spatial_instance_id"],
                "operations": self.operations,
            },
        )

        rows = [
            {
                "frame": int(op["timepoint"]),
                "supervoxel_id": int(op["supervoxel_id"]),
                "previous_instance_id": int(op["previous_instance_id"]),
                "voxel_count": int(op["voxel_count"]),
            }
            for op in self.operations
            if op.get("type") == "hallucination"
        ]
        _atomic_csv(
            self.hallucinations_path,
            pd.DataFrame(
                rows,
                columns=[
                    "frame",
                    "supervoxel_id",
                    "previous_instance_id",
                    "voxel_count",
                ],
            ),
        )


def parse_supervoxel_group(text: str) -> list[int]:
    text = str(text).strip()
    if not text:
        return []

    tokens = [part.strip() for part in text.split(",")]
    if any(token == "" for token in tokens):
        raise AnnotationError(
            f"Invalid comma-separated list: {text!r}. Example: 12, 15, 19"
        )

    values: list[int] = []
    for token in tokens:
        try:
            value = int(token)
        except ValueError as exc:
            raise AnnotationError(
                f"Supervoxel ID {token!r} is not an integer."
            ) from exc
        if value <= 0:
            raise AnnotationError(
                f"Supervoxel IDs must be positive; received {value}."
            )
        values.append(value)

    if len(values) != len(set(values)):
        raise AnnotationError(
            f"The same supervoxel appears twice in one box: {text!r}"
        )
    return values
