from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from ..model.config import CurriculumConfig


CURRICULUM_STAGES = (
    "spatial_dense",
    "temporal_dense",
    "query_bootstrap",
    "native_bootstrap",
    "joint",
)

PARAMETER_GROUP_MODULES = {
    "spatial": ("acquisition", "encoder", "decoder"),
    "dense": ("dense_heads",),
    "proposal": ("spatial_proposal_generator",),
    "temporal": (
        "graph_encoder",
        "history_encoder",
        "history_fusion",
        "tracklet_pooler",
        "temporal_builder",
        "cr1",
        "cr2",
    ),
    "query": ("query_builder", "query_decoder"),
    "native": ("native_mask_head",),
}


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    trainable_groups: frozenset[str]
    lr_scales: dict[str, float]
    loss_weight_overrides: dict[str, float]
    bypass_coreasoning: bool = False


def stage_name_for_step(config: CurriculumConfig, step: int) -> str:
    if not config.enabled:
        return "legacy"
    if step < 0:
        raise ValueError("curriculum step must be non-negative")
    remaining = int(step)
    durations = (
        ("spatial_dense", config.spatial_dense_steps),
        ("temporal_dense", config.temporal_dense_steps),
        ("query_bootstrap", config.query_bootstrap_steps),
        ("native_bootstrap", config.native_bootstrap_steps),
    )
    invalid = [name for name, duration in durations if duration < 0]
    if invalid:
        raise ValueError(
            "curriculum stage durations must be non-negative: "
            + ", ".join(invalid)
        )
    for name, duration in durations:
        if remaining < duration:
            return name
        remaining -= duration
    return "joint"


def curriculum_stage(config: CurriculumConfig, step: int) -> CurriculumStage:
    name = stage_name_for_step(config, step)
    all_groups = frozenset(PARAMETER_GROUP_MODULES)
    unit_lr = {group: 1.0 for group in PARAMETER_GROUP_MODULES}
    if name == "legacy":
        return CurriculumStage(name, all_groups, unit_lr, {})

    dense_only_overrides = {
        "exist": 0.0,
        "dice_hi": 0.0,
        "focal_hi": 0.0,
        "dice_coarse": 0.0,
        "focal_coarse": 0.0,
        "center": 0.0,
        "count": 0.0,
        "overlap": 0.0,
        "aux_layer": 0.0,
    }
    query_overrides = {
        "dice_hi": 0.0,
        "focal_hi": 0.0,
        "count": 0.0,
        "overlap": 0.0,
    }
    native_overrides = {"count": 0.0, "overlap": 0.0}
    if name == "spatial_dense":
        trainable = frozenset({"spatial", "dense", "proposal"})
        return CurriculumStage(
            name,
            trainable,
            {group: float(group in trainable) for group in PARAMETER_GROUP_MODULES},
            dense_only_overrides,
            bypass_coreasoning=True,
        )
    if name == "temporal_dense":
        trainable = frozenset({"spatial", "dense", "proposal", "temporal"})
        return CurriculumStage(
            name,
            trainable,
            {group: float(group in trainable) for group in PARAMETER_GROUP_MODULES},
            dense_only_overrides,
        )
    if name == "query_bootstrap":
        trainable = frozenset({"spatial", "dense", "proposal", "temporal", "query"})
        lr_scales = {group: float(group in trainable) for group in PARAMETER_GROUP_MODULES}
        lr_scales["spatial"] = 0.25
        lr_scales["dense"] = 0.50
        return CurriculumStage(
            name,
            trainable,
            lr_scales,
            query_overrides,
        )
    if name == "native_bootstrap":
        trainable = all_groups
        lr_scales = dict(unit_lr)
        lr_scales["spatial"] = 0.10
        lr_scales["dense"] = 0.50
        lr_scales["proposal"] = 0.50
        return CurriculumStage(
            name,
            trainable,
            lr_scales,
            native_overrides,
        )
    if name != "joint":
        raise ValueError(f"Unknown STIR-Net curriculum stage: {name}")
    joint_lr = dict(unit_lr)
    joint_lr["spatial"] = float(config.joint_spatial_lr_scale)
    joint_lr["dense"] = float(config.joint_dense_lr_scale)
    joint_lr["proposal"] = 0.50
    if any(scale < 0 for scale in joint_lr.values()):
        raise ValueError("curriculum LR scales must be non-negative")
    return CurriculumStage(name, all_groups, joint_lr, {"overlap": 0.0})


def model_parameter_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    groups: dict[str, list[nn.Parameter]] = {}
    seen: set[int] = set()
    for group_name, module_names in PARAMETER_GROUP_MODULES.items():
        parameters: list[nn.Parameter] = []
        for module_name in module_names:
            module = model.get_submodule(module_name)
            for parameter in module.parameters():
                identity = id(parameter)
                if identity in seen:
                    raise ValueError(
                        f"Parameter appears in multiple curriculum groups: {module_name}"
                    )
                seen.add(identity)
                parameters.append(parameter)
        groups[group_name] = parameters
    missing = [
        name for name, parameter in model.named_parameters() if id(parameter) not in seen
    ]
    if missing:
        raise ValueError(f"Unassigned STIR-Net parameters: {missing}")
    return groups


def optimizer_parameter_groups(
    model: nn.Module, base_lr: float
) -> list[dict]:
    return [
        {"name": name, "params": parameters, "lr": float(base_lr)}
        for name, parameters in model_parameter_groups(model).items()
    ]


class CurriculumController:
    """Apply stages without rebuilding the model, optimizer, or its state."""

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
        optimizer_groups = {group.get("name"): group for group in self.optimizer.param_groups}
        if set(optimizer_groups) != set(groups):
            raise ValueError("Optimizer parameter groups do not match curriculum groups")
        for name, group in optimizer_groups.items():
            group["lr"] = self.base_lr * stage.lr_scales[name]
            group["lr_scale"] = stage.lr_scales[name]
        self.current = stage
        return stage
