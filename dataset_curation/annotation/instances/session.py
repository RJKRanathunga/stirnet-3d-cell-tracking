from __future__ import annotations

"""Persistent instance-correction session state, resume and undo."""

from dataset_curation.annotation.instances.split import (
    AnnotationError,
    SplitResult,
    _expand_seed_groups_by_contact_graph,
)

import json

from dataclasses import dataclass

from pathlib import Path

import numpy as np

try:
    from magicgui.widgets import Container, Label, LineEdit, PushButton
except ImportError as exc:
    raise ImportError(
        "magicgui is required for the annotation panel. It normally comes with "
        "Napari. Install it with: pip install magicgui"
    ) from exc

@dataclass(frozen=True)
class UndoResult:
    timepoint: int
    original_instance_id: int
    removed_instance_ids: tuple[int, ...]

def _dominant_parent_instance(
    sv_frame: np.ndarray,
    instance_frame: np.ndarray,
    sv_id: int,
) -> int:
    mask = sv_frame == sv_id
    values, counts = np.unique(instance_frame[mask], return_counts=True)

    nonzero = values > 0
    if not np.any(nonzero):
        return 0

    values = values[nonzero]
    counts = counts[nonzero]
    return int(values[np.argmax(counts)])

class AnnotationSession:
    def __init__(
        self,
        *,
        sample_id: str,
        timepoints: tuple[int, ...],
        supervoxels: np.ndarray,
        base_instances: np.ndarray,
        output_dir: Path,
        resume: bool,
    ) -> None:
        self.sample_id = sample_id
        self.timepoints = timepoints
        self.supervoxels = np.asarray(supervoxels)
        self.base_instances = np.asarray(base_instances).astype(
            np.int32,
            copy=False,
        )
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.corrected = self.base_instances.copy()
        self.corrections: list[dict] = []
        self.corrected_original_ids: dict[int, set[int]] = {
            i: set() for i in range(len(timepoints))
        }

        self.log_path = self.output_dir / "supervoxel_split_corrections.json"

        if resume:
            self._resume_existing()

        max_label = int(self.corrected.max(initial=0))
        self.next_label = max_label + 1

    def output_path_for_timepoint(self, dataset_t: int) -> Path:
        return self.output_dir / f"manual_instances_t{dataset_t:03d}.npy"

    def _resume_existing(self) -> None:
        for local_t, dataset_t in enumerate(self.timepoints):
            path = self.output_path_for_timepoint(dataset_t)
            if not path.exists():
                continue

            existing = np.load(path)
            if existing.shape != self.corrected[local_t].shape:
                raise AnnotationError(
                    f"Cannot resume {path}: expected "
                    f"{self.corrected[local_t].shape}, got {existing.shape}."
                )
            self.corrected[local_t] = existing.astype(np.int32, copy=False)
            print(f"[resume] loaded {path}")

        if not self.log_path.exists():
            return

        with self.log_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if payload.get("sample_id") != self.sample_id:
            raise AnnotationError(
                f"Existing log belongs to sample {payload.get('sample_id')!r}, "
                f"not {self.sample_id!r}."
            )

        existing_timepoints = tuple(int(v) for v in payload.get("timepoints", []))
        if existing_timepoints and existing_timepoints != self.timepoints:
            print(
                "[resume] existing log uses a different timepoint selection; "
                "loading only corrections whose timepoints are in this session."
            )

        self.corrections = list(payload.get("corrections", []))

        time_to_local = {t: i for i, t in enumerate(self.timepoints)}
        for correction in self.corrections:
            dataset_t = int(correction["timepoint"])
            if dataset_t not in time_to_local:
                continue
            local_t = time_to_local[dataset_t]
            self.corrected_original_ids[local_t].add(
                int(correction["original_instance_id"])
            )

    def _parent_instance_for_sv(self, local_t: int, sv_id: int) -> int:
        # Use the CURRENT corrected partition. This permits another split of a
        # previously generated child instance later in the same session.
        return _dominant_parent_instance(
            self.supervoxels[local_t],
            self.corrected[local_t],
            sv_id,
        )

    def _expected_supervoxels(
        self,
        local_t: int,
        original_instance_id: int,
    ) -> set[int]:
        mask = self.corrected[local_t] == original_instance_id

        if np.any(mask & (self.supervoxels[local_t] == 0)):
            missing_voxels = int(
                np.count_nonzero(mask & (self.supervoxels[local_t] == 0))
            )
            raise AnnotationError(
                f"Original instance {original_instance_id} contains "
                f"{missing_voxels} voxels with supervoxel ID 0. The split cannot "
                "be made losslessly from supervoxels; inspect the spatial export."
            )

        ids = np.unique(self.supervoxels[local_t][mask])
        return {int(v) for v in ids.tolist() if int(v) > 0}

    def apply_split(
        self,
        local_t: int,
        groups: list[list[int]],
    ) -> SplitResult:
        if not (0 <= local_t < len(self.timepoints)):
            raise AnnotationError(f"Invalid local frame index: {local_t}")

        seed_groups = tuple(
            tuple(int(v) for v in group)
            for group in groups
            if group
        )

        if len(seed_groups) < 2:
            raise AnnotationError(
                "At least two instance boxes must be filled before Save."
            )

        if len(seed_groups) > 4:
            raise AnnotationError(
                "At most four output instances are supported."
            )

        flat = [
            sv_id
            for group in seed_groups
            for sv_id in group
        ]

        if len(flat) != len(set(flat)):
            duplicates = sorted(
                sv_id
                for sv_id in set(flat)
                if flat.count(sv_id) > 1
            )
            raise AnnotationError(
                "A supervoxel cannot be used as a seed for two cells. "
                f"Duplicates: {duplicates}"
            )

        frame_ids = {
            int(v)
            for v in np.unique(self.supervoxels[local_t]).tolist()
            if int(v) > 0
        }

        missing = sorted(set(flat) - frame_ids)
        if missing:
            raise AnnotationError(
                "These supervoxels do not exist in the current frame: "
                f"{missing}"
            )

        parent_ids = {
            self._parent_instance_for_sv(
                local_t,
                sv_id,
            )
            for sv_id in flat
        }

        if 0 in parent_ids:
            raise AnnotationError(
                "At least one selected seed supervoxel is currently background "
                "rather than part of a spatial instance."
            )

        if len(parent_ids) != 1:
            raise AnnotationError(
                "The selected seed supervoxels are already in DIFFERENT current "
                "spatial instances, so there is no single merged instance to "
                "split between them. Current instance IDs: "
                f"{sorted(parent_ids)}. "
                "Use the red leader lines to choose seed SVs from the same "
                "merged colored instance."
            )

        original_instance_id = next(iter(parent_ids))

        parent_supervoxels = self._expected_supervoxels(
            local_t,
            original_instance_id,
        )

        # No completeness requirement: the user's entries are ONLY split seeds.
        # Automatically assign every remaining supervoxel in the current merged
        # instance using the weighted contact graph.
        expanded_groups = _expand_seed_groups_by_contact_graph(
            sv_frame=self.supervoxels[local_t],
            parent_supervoxels=parent_supervoxels,
            seed_groups=seed_groups,
        )

        original_mask = (
            self.corrected[local_t] == original_instance_id
        )

        # Clear only this current merged component, then fill it from the
        # automatically expanded groups.
        self.corrected[local_t][original_mask] = 0

        output_ids: list[int] = []
        group_records: list[dict] = []

        for seed_group, expanded_group in zip(
            seed_groups,
            expanded_groups,
        ):
            new_instance_id = int(self.next_label)
            self.next_label += 1

            group_mask = np.isin(
                self.supervoxels[local_t],
                np.asarray(
                    expanded_group,
                    dtype=self.supervoxels.dtype,
                ),
            ) & original_mask

            self.corrected[local_t][group_mask] = new_instance_id
            output_ids.append(new_instance_id)

            group_records.append(
                {
                    "output_instance_id": new_instance_id,
                    "seed_supervoxels": [
                        int(v) for v in seed_group
                    ],
                    "assigned_supervoxels": [
                        int(v) for v in expanded_group
                    ],
                }
            )

        if np.any(
            self.corrected[local_t][original_mask] == 0
        ):
            raise AnnotationError(
                "Internal error: automatic graph split left part of the "
                "original instance unassigned."
            )

        dataset_t = int(self.timepoints[local_t])

        record = {
            "timepoint": dataset_t,
            "original_instance_id": int(original_instance_id),
            "split_method": (
                "multi_source_dijkstra_inverse_physical_contact_area"
            ),
            "groups": group_records,
        }

        self.corrections.append(record)
        self.corrected_original_ids[local_t].add(
            int(original_instance_id)
        )

        self.persist()

        return SplitResult(
            timepoint=dataset_t,
            original_instance_id=int(original_instance_id),
            output_instance_ids=tuple(output_ids),
            seed_groups=seed_groups,
            groups=expanded_groups,
        )

    def _local_index_for_dataset_timepoint(
        self,
        dataset_t: int,
    ) -> int | None:
        try:
            return self.timepoints.index(int(dataset_t))
        except ValueError:
            return None

    def can_undo(self) -> bool:
        """
        True when this session contains at least one persisted correction for
        one of the currently loaded timepoints.
        """
        for correction in reversed(self.corrections):
            dataset_t = int(correction["timepoint"])
            if self._local_index_for_dataset_timepoint(dataset_t) is not None:
                return True
        return False

    def undo_last_split(self) -> UndoResult:
        """
        Reverse the newest correction belonging to a loaded timepoint.

        This is safe for nested edits because undo is LIFO. If an output of an
        earlier split was itself split later, that later split must be undone
        first, after which the earlier output label exists again.
        """
        correction_index: int | None = None
        local_t: int | None = None

        for index in range(len(self.corrections) - 1, -1, -1):
            correction = self.corrections[index]
            candidate_local_t = self._local_index_for_dataset_timepoint(
                int(correction["timepoint"])
            )
            if candidate_local_t is not None:
                correction_index = index
                local_t = candidate_local_t
                break

        if correction_index is None or local_t is None:
            raise AnnotationError(
                "There is no saved split operation to undo."
            )

        correction = self.corrections[correction_index]
        dataset_t = int(correction["timepoint"])
        original_instance_id = int(
            correction["original_instance_id"]
        )

        groups = list(correction.get("groups", []))
        output_ids = tuple(
            int(group["output_instance_id"])
            for group in groups
            if "output_instance_id" in group
        )

        if len(output_ids) < 2:
            raise AnnotationError(
                "The newest correction log entry does not contain enough "
                "output instance IDs to undo safely."
            )

        frame = self.corrected[local_t]

        # Since this is the newest correction affecting the loaded data, these
        # labels should still exist. Restore their entire union back to the
        # pre-split instance ID.
        changed_mask = np.isin(
            frame,
            np.asarray(output_ids, dtype=frame.dtype),
        )

        changed_voxels = int(np.count_nonzero(changed_mask))
        if changed_voxels == 0:
            raise AnnotationError(
                "Cannot undo the newest correction because none of its output "
                f"instance IDs {output_ids} are present in t={dataset_t}. "
                "The annotation files may have been modified outside this tool."
            )

        frame[changed_mask] = original_instance_id

        # Remove exactly the operation we reversed.
        self.corrections.pop(correction_index)

        # This set is only a UI/count bookkeeping structure. IDs created by a
        # parent split are globally unique, so discarding the restored parent
        # operation is safe in the normal LIFO workflow.
        self.corrected_original_ids[local_t].discard(
            original_instance_id
        )

        # Persist both raster labels and the shortened correction history.
        self.persist()

        return UndoResult(
            timepoint=dataset_t,
            original_instance_id=original_instance_id,
            removed_instance_ids=output_ids,
        )

    def persist(self) -> None:
        # Save a complete pseudo-GT instance volume for every selected frame.
        # Frames not manually changed remain equal to the strong spatial output.
        for local_t, dataset_t in enumerate(self.timepoints):
            path = self.output_path_for_timepoint(dataset_t)
            np.save(path, self.corrected[local_t].astype(np.int32, copy=False))

        payload = {
            "format_version": 1,
            "sample_id": self.sample_id,
            "timepoints": [int(t) for t in self.timepoints],
            "description": (
                "Base spatial instance labels with manually corrected merged "
                "instances using atomic-supervoxel grouping."
            ),
            "corrections": self.corrections,
        }

        with self.log_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def corrections_in_frame(self, local_t: int) -> int:
        return len(self.corrected_original_ids[local_t])

def parse_supervoxel_group(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []

    tokens = [part.strip() for part in text.split(",")]
    if any(token == "" for token in tokens):
        raise AnnotationError(
            f"Invalid comma-separated list: {text!r}. "
            "Example: 12, 15, 19"
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
