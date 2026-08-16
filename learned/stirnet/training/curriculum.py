from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .config import CurriculumConfig


CURRICULUM_STAGES = (
    "geometry_bootstrap",
    "spatial_partition",
    "instance_temporal",
    "refinement_joint",
)

PARAMETER_GROUP_MODULES = {
    "geometry_spatial": (
        "acquisition",
        "evidence_stem",
        "spatial_backbone",
        "geometry_decoder",
    ),
    "partition": ("rag_builder", "rag_network"),
    "instances": ("instance_tokenizer",),
    "temporal": (
        "history_encoder",
        "temporal_encoder",
        "temporal_observer",
        "instance_temporal",
    ),
    "refinement": ("local_refiner",),
}


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    trainable_groups: frozenset[str]
    lr_scales: dict[str, float]
    use_temporal: bool
    run_refinement: bool
    execution_stage: str


def stage_name_for_step(config: CurriculumConfig, step: int) -> str:
    if config.fixed_stage is not None:
        if config.fixed_stage not in CURRICULUM_STAGES:
            raise ValueError(f"Unknown fixed V2 curriculum stage: {config.fixed_stage}")
        return config.fixed_stage
    if not config.enabled:
        return "refinement_joint"
    if step < 0:
        raise ValueError("curriculum step must be non-negative")
    boundaries = (
        ("geometry_bootstrap", config.geometry_bootstrap_steps),
        ("spatial_partition", config.spatial_partition_steps),
        ("instance_temporal", config.instance_temporal_steps),
    )
    remaining = step
    for name, duration in boundaries:
        if duration < 0:
            raise ValueError("curriculum stage durations cannot be negative")
        if remaining < duration:
            return name
        remaining -= duration
    return "refinement_joint"


def curriculum_stage(config: CurriculumConfig, step: int) -> CurriculumStage:
    name = stage_name_for_step(config, step)
    groups = frozenset(PARAMETER_GROUP_MODULES)
    if name == "geometry_bootstrap":
        trainable = frozenset({"geometry_spatial"})
        return CurriculumStage(
            name,
            trainable,
            {group: float(group in trainable) for group in groups},
            use_temporal=False,
            run_refinement=False,
            execution_stage="geometry",
        )
    if name == "spatial_partition":
        trainable = frozenset({"geometry_spatial", "partition"})
        return CurriculumStage(
            name,
            trainable,
            {group: float(group in trainable) for group in groups},
            use_temporal=False,
            run_refinement=False,
            execution_stage="spatial",
        )
    if name == "instance_temporal":
        trainable = frozenset(
            {"geometry_spatial", "partition", "instances", "temporal"}
        )
        scales = {group: float(group in trainable) for group in groups}
        scales["geometry_spatial"] = config.spatial_lr_scale_temporal
        return CurriculumStage(
            name,
            trainable,
            scales,
            use_temporal=True,
            run_refinement=False,
            execution_stage="temporal",
        )
    if name != "refinement_joint":
        raise ValueError(f"Unknown V2 curriculum stage: {name}")
    scales = {group: 1.0 for group in groups}
    scales["geometry_spatial"] = config.spatial_lr_scale_refinement
    return CurriculumStage(
        name,
        groups,
        scales,
        use_temporal=True,
        run_refinement=True,
        execution_stage="refinement",
    )


def model_parameter_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    grouped: dict[str, list[nn.Parameter]] = {}
    seen: set[int] = set()
    for group_name, module_names in PARAMETER_GROUP_MODULES.items():
        rows: list[nn.Parameter] = []
        for module_name in module_names:
            for parameter in model.get_submodule(module_name).parameters():
                identity = id(parameter)
                if identity in seen:
                    raise ValueError(
                        f"Parameter appears in multiple V2 groups: {module_name}"
                    )
                seen.add(identity)
                rows.append(parameter)
        grouped[group_name] = rows
    missing = [
        name for name, parameter in model.named_parameters() if id(parameter) not in seen
    ]
    if missing:
        raise ValueError(f"Unassigned STIR-Net V2 parameters: {missing}")
    return grouped


def optimizer_parameter_groups(model: nn.Module, base_lr: float) -> list[dict]:
    return [
        {"name": name, "params": parameters, "lr": float(base_lr)}
        for name, parameters in model_parameter_groups(model).items()
    ]


class CurriculumController:
    def __init__(self, model, optimizer, config: CurriculumConfig, base_lr: float):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.base_lr = float(base_lr)
        self.current: CurriculumStage | None = None

    def apply(self, step: int) -> CurriculumStage:
        stage = curriculum_stage(self.config, step)
        if self.current is not None and self.current.name == stage.name:
            return self.current
        groups = model_parameter_groups(self.model)
        for name, parameters in groups.items():
            trainable = name in stage.trainable_groups
            for parameter in parameters:
                parameter.requires_grad_(trainable)
        optimizer_groups = {
            group.get("name"): group for group in self.optimizer.param_groups
        }
        if set(optimizer_groups) != set(groups):
            raise ValueError("Optimizer groups do not match V2 curriculum groups")
        for name, group in optimizer_groups.items():
            group["lr"] = self.base_lr * stage.lr_scales[name]
            group["lr_scale"] = stage.lr_scales[name]
        self.current = stage
        return stage


__all__ = [
    "CURRICULUM_STAGES",
    "CurriculumController",
    "CurriculumStage",
    "curriculum_stage",
    "model_parameter_groups",
    "optimizer_parameter_groups",
    "stage_name_for_step",
]
