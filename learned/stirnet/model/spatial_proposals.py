from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ProposalConfig
from .types import SpatialProposalState


class SpatialProposalGenerator(nn.Module):
    """Build sparse learned cell hypotheses from native spatial evidence.

    Dense work is deliberately restricted to four one-channel residual paths.
    Proposal representations are gathered only after NMS, so D0, the five
    inputs, and dense predictions are never concatenated at native resolution.
    """

    def __init__(
        self,
        cfg: ProposalConfig,
        *,
        d0_channels: int,
        e2_channels: int,
        spatial_input_channels: int = 5,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.from_d0 = nn.Conv3d(d0_channels, 1, 1)
        self.from_inputs = nn.Conv3d(spatial_input_channels, 1, 1)
        self.from_foreground = nn.Conv3d(1, 1, 1)
        self.from_boundary = nn.Conv3d(1, 1, 1)
        for residual in (
            self.from_d0,
            self.from_inputs,
            self.from_foreground,
            self.from_boundary,
        ):
            nn.init.zeros_(residual.weight)
            nn.init.zeros_(residual.bias)

        local_input_dim = (
            3 * d0_channels
            + 3 * spatial_input_channels
            + 3 * 3
            + e2_channels
            + 1
        )
        self.local_encoder = nn.Sequential(
            nn.Linear(local_input_dim, cfg.local_dim),
            nn.SiLU(),
            nn.Linear(cfg.local_dim, cfg.local_dim),
        )

    def proposal_score_logits(
        self,
        d0: Tensor,
        spatial_inputs: Tensor,
        dense: dict[str, Tensor],
    ) -> Tensor:
        return (
            dense["center_heatmap_logits"]
            + self.from_d0(d0)
            + self.from_inputs(spatial_inputs)
            + self.from_foreground(dense["foreground_logits"])
            + self.from_boundary(dense["boundary_logits"])
        )

    @staticmethod
    def _relative_voxel_centers_um(
        indices_zyx: Tensor, shape: tuple[int, int, int], spacing_um: Tensor
    ) -> Tensor:
        extent = indices_zyx.new_tensor(
            [shape[0] - 1, shape[1] - 1, shape[2] - 1],
            dtype=torch.float32,
        ) * spacing_um.float()
        return indices_zyx.float() * spacing_um.float()[None] - 0.5 * extent[None]

    @torch.no_grad()
    def _learned_centers(
        self,
        logits: Tensor,
        spacing_um: Tensor,
        dref_um: Tensor,
        padding_mask: Tensor | None,
        max_count: int,
    ) -> Tensor:
        """Anisotropic local maxima followed by exact physical greedy NMS."""
        if max_count <= 0:
            return torch.empty((0, 3), device=logits.device, dtype=torch.long)
        shape = tuple(int(value) for value in logits.shape)
        radius_um = float(self.cfg.nms_radius_dref) * float(dref_um)
        radii = torch.ceil(
            logits.new_tensor(radius_um) / spacing_um.float().clamp_min(1e-8)
        ).long().clamp_min(0)
        rz, ry, rx = (int(value) for value in radii.tolist())
        padded = F.pad(logits, (rx, rx, ry, ry, rz, rz), value=-torch.inf)
        pooled = torch.full_like(logits, -torch.inf)
        spacing_values = spacing_um.detach().float().cpu().tolist()
        for dz in range(-rz, rz + 1):
            for dy in range(-ry, ry + 1):
                for dx in range(-rx, rx + 1):
                    distance_squared = (
                        (dz * spacing_values[0]) ** 2
                        + (dy * spacing_values[1]) ** 2
                        + (dx * spacing_values[2]) ** 2
                    )
                    if distance_squared > (radius_um + 1e-6) ** 2:
                        continue
                    shifted = padded[
                        rz + dz : rz + dz + shape[0],
                        ry + dy : ry + dy + shape[1],
                        rx + dx : rx + dx + shape[2],
                    ]
                    pooled = torch.maximum(pooled, shifted)
        candidates = logits == pooled
        candidates &= torch.isfinite(logits)
        if padding_mask is not None:
            candidates &= ~padding_mask.bool()
        candidate_indices = torch.nonzero(candidates, as_tuple=False)
        if candidate_indices.numel() == 0:
            return torch.empty((0, 3), device=logits.device, dtype=torch.long)
        values = logits[
            candidate_indices[:, 0],
            candidate_indices[:, 1],
            candidate_indices[:, 2],
        ]
        pool_size = min(int(self.cfg.candidate_pool_size), len(values))
        order = torch.topk(values, pool_size, sorted=True).indices
        candidate_indices = candidate_indices[order]
        values = values[order]
        if not self.training:
            keep_score = values.sigmoid() >= float(self.cfg.inference_score_threshold)
            candidate_indices = candidate_indices[keep_score]
        if candidate_indices.numel() == 0:
            return candidate_indices.reshape(0, 3)

        centers_um = self._relative_voxel_centers_um(
            candidate_indices, shape, spacing_um
        )
        selected: list[int] = []
        for row in range(len(candidate_indices)):
            if selected:
                distance = torch.linalg.vector_norm(
                    centers_um[row] - centers_um[selected], dim=-1
                )
                if bool((distance <= radius_um).any()):
                    continue
            selected.append(row)
            if len(selected) >= int(max_count):
                break
        return candidate_indices[
            torch.as_tensor(selected, device=logits.device, dtype=torch.long)
        ]

    @staticmethod
    def _sampling_grid(
        references_um: Tensor,
        shape: tuple[int, int, int],
        spacing_um: Tensor,
        offsets_um: Tensor,
    ) -> Tensor:
        points = references_um[:, None].float() + offsets_um[None].float()
        extent = points.new_tensor(
            [shape[0] - 1, shape[1] - 1, shape[2] - 1]
        ) * spacing_um.float()
        normalized_zyx = points / (0.5 * extent[None, None]).clamp_min(1e-8)
        return normalized_zyx[..., [2, 1, 0]]

    @classmethod
    def _sample_points(
        cls,
        feature: Tensor,
        references_um: Tensor,
        spacing_um: Tensor,
        offsets_um: Tensor,
    ) -> Tensor:
        """Return [P,C,S] samples from one unbatched [C,Z,Y,X] tensor."""
        proposal_count = references_um.shape[0]
        if proposal_count == 0:
            return feature.new_zeros((0, feature.shape[0], offsets_um.shape[0]))
        grid = cls._sampling_grid(
            references_um, tuple(int(v) for v in feature.shape[-3:]), spacing_um, offsets_um
        )
        sampled = F.grid_sample(
            feature[None],
            grid.reshape(1, proposal_count, offsets_um.shape[0], 1, 3),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, :, :, :, 0]
        return sampled.permute(1, 0, 2)

    @classmethod
    def _sample_statistics(
        cls,
        feature: Tensor,
        references_um: Tensor,
        spacing_um: Tensor,
        offsets_um: Tensor,
        center_index: int,
    ) -> Tensor:
        sampled = cls._sample_points(feature, references_um, spacing_um, offsets_um)
        if sampled.shape[0] == 0:
            return sampled.new_zeros((0, 3 * feature.shape[0]))
        return torch.cat(
            [
                sampled[:, :, center_index],
                sampled.mean(dim=-1),
                sampled.max(dim=-1).values,
            ],
            dim=-1,
        )

    def _fallback_centers(
        self,
        learned_references_um: Tensor,
        learned_sources: Tensor,
        instance_ids: Tensor,
        instance_centroids_um: Tensor,
        dref_um: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not self.cfg.ensure_source_fallback or instance_ids.numel() == 0:
            return (
                instance_centroids_um.new_zeros((0, 3)),
                instance_ids.new_zeros((0,)),
            )
        missing = []
        radius_um = float(self.cfg.source_fallback_match_radius_dref) * dref_um.float()
        for row, source_id in enumerate(instance_ids):
            same_source = learned_sources == source_id
            covered = False
            if bool(same_source.any()):
                distances = torch.linalg.vector_norm(
                    learned_references_um[same_source]
                    - instance_centroids_um[row].float(),
                    dim=-1,
                )
                covered = bool((distances <= radius_um).any())
            if not covered:
                missing.append(row)
        if not missing:
            return (
                instance_centroids_um.new_zeros((0, 3)),
                instance_ids.new_zeros((0,)),
            )
        rows = torch.as_tensor(missing, device=instance_ids.device, dtype=torch.long)
        return instance_centroids_um[rows], instance_ids[rows]

    def _local_embeddings(
        self,
        batch_index: int,
        references_um: Tensor,
        d0: Tensor,
        e2: Tensor,
        spatial_inputs: Tensor,
        dense: dict[str, Tensor],
        spacing_um: Tensor,
        e2_spacing_um: Tensor,
        dref_um: Tensor,
        score_logits: Tensor,
    ) -> tuple[Tensor, Tensor]:
        grid_size = int(self.cfg.local_grid_size)
        axis = torch.linspace(
            -float(self.cfg.local_extent_dref),
            float(self.cfg.local_extent_dref),
            grid_size,
            device=d0.device,
            dtype=torch.float32,
        ) * dref_um.float()
        offsets = torch.stack(
            torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1
        ).reshape(-1, 3)
        center_index = offsets.shape[0] // 2
        parts = [
            self._sample_statistics(
                d0[batch_index], references_um, spacing_um, offsets, center_index
            ),
            self._sample_statistics(
                spatial_inputs[batch_index],
                references_um,
                spacing_um,
                offsets,
                center_index,
            ),
        ]
        for key in (
            "foreground_logits",
            "center_heatmap_logits",
            "boundary_logits",
        ):
            parts.append(
                self._sample_statistics(
                    dense[key][batch_index],
                    references_um,
                    spacing_um,
                    offsets,
                    center_index,
                )
            )
        zero_offset = offsets.new_zeros((1, 3))
        e2_center = self._sample_points(
            e2[batch_index], references_um, e2_spacing_um, zero_offset
        )[:, :, 0]
        score_sample = self._sample_points(
            score_logits[batch_index], references_um, spacing_um, zero_offset
        )[:, :, 0]
        parts.extend([e2_center, score_sample])
        local = torch.cat(parts, dim=-1)
        return self.local_encoder(local.to(d0.dtype)), score_sample[:, 0].sigmoid()

    def forward(
        self,
        d0: Tensor,
        e2: Tensor,
        spatial_inputs: Tensor,
        dense: dict[str, Tensor],
        instance_labels: Tensor,
        spacing_um: Tensor,
        e2_spacing_um: Tensor,
        dref_um: Tensor,
        instance_ids: Tensor,
        instance_batch: Tensor,
        instance_centroids_um: Tensor,
        spatial_padding_mask: Tensor | None = None,
    ) -> tuple[SpatialProposalState, Tensor]:
        score_logits = self.proposal_score_logits(d0, spatial_inputs, dense)
        per_batch = []
        for batch_index in range(d0.shape[0]):
            component_rows = torch.nonzero(
                instance_batch == batch_index, as_tuple=False
            ).flatten()
            padding = (
                None
                if spatial_padding_mask is None
                else spatial_padding_mask[batch_index]
            )
            learned_voxels = self._learned_centers(
                score_logits[batch_index, 0],
                spacing_um[batch_index],
                dref_um[batch_index],
                padding,
                int(self.cfg.max_proposals),
            )
            learned_refs_um = self._relative_voxel_centers_um(
                learned_voxels,
                tuple(int(value) for value in score_logits.shape[-3:]),
                spacing_um[batch_index],
            )
            learned_sources = (
                instance_labels[
                    batch_index,
                    learned_voxels[:, 0],
                    learned_voxels[:, 1],
                    learned_voxels[:, 2],
                ]
                if learned_voxels.numel()
                else instance_ids.new_zeros((0,))
            )
            learned_sources = torch.where(
                learned_sources > 0,
                learned_sources,
                torch.full_like(learned_sources, -1),
            )
            learned_count = len(learned_refs_um)
            while True:
                fallback_refs_um, fallback_sources = self._fallback_centers(
                    learned_refs_um[:learned_count],
                    learned_sources[:learned_count],
                    instance_ids[component_rows],
                    instance_centroids_um[component_rows],
                    dref_um[batch_index],
                )
                available_learned = int(self.cfg.max_proposals) - len(
                    fallback_refs_um
                )
                if available_learned < 0:
                    raise RuntimeError(
                        f"Batch item {batch_index} needs {len(fallback_refs_um)} "
                        f"source fallbacks, exceeding max_proposals="
                        f"{self.cfg.max_proposals}; raise the explicit proposal "
                        "limit so component coverage is not silently truncated."
                    )
                next_count = min(learned_count, available_learned)
                if next_count == learned_count:
                    break
                learned_count = next_count
            learned_refs_um = learned_refs_um[:learned_count]
            learned_sources = learned_sources[:learned_count]
            references_um = torch.cat([learned_refs_um, fallback_refs_um.float()])
            sources = torch.cat([learned_sources, fallback_sources])
            fallback = torch.zeros(
                references_um.shape[0], device=d0.device, dtype=torch.bool
            )
            fallback[len(learned_refs_um) :] = True
            embeddings, scores = self._local_embeddings(
                batch_index,
                references_um,
                d0,
                e2,
                spatial_inputs,
                dense,
                spacing_um[batch_index],
                e2_spacing_um[batch_index],
                dref_um[batch_index],
                score_logits,
            )
            per_batch.append(
                (
                    embeddings,
                    references_um / dref_um[batch_index].clamp_min(1e-8),
                    scores,
                    sources,
                    fallback,
                )
            )

        max_count = max((item[0].shape[0] for item in per_batch), default=0)
        embeddings = d0.new_zeros((d0.shape[0], max_count, self.cfg.local_dim))
        references = dref_um.new_zeros((d0.shape[0], max_count, 3))
        scores = d0.new_zeros((d0.shape[0], max_count))
        source_ids = instance_ids.new_full((d0.shape[0], max_count), -1)
        fallback_mask = torch.zeros(
            (d0.shape[0], max_count), device=d0.device, dtype=torch.bool
        )
        padding_mask = torch.ones_like(fallback_mask)
        for batch_index, (local, refs, score, source, fallback) in enumerate(per_batch):
            count = local.shape[0]
            embeddings[batch_index, :count] = local
            references[batch_index, :count] = refs
            scores[batch_index, :count] = score
            source_ids[batch_index, :count] = source
            fallback_mask[batch_index, :count] = fallback
            padding_mask[batch_index, :count] = False
        return (
            SpatialProposalState(
                embeddings=embeddings,
                references_cellscale=references,
                scores=scores,
                padding_mask=padding_mask,
                source_instance_ids=source_ids,
                fallback_mask=fallback_mask,
            ),
            score_logits,
        )
