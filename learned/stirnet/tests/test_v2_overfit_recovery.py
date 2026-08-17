from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def _experiment_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "experiments" / "stirnet" / "31_spatial_first_overfit.py"
    spec = importlib.util.spec_from_file_location(
        "stirnet_overfit_recovery_test_target",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_history_uses_checkpoint_step_and_diagnostic_overlay(tmp_path):
    module = _experiment_module()
    recovery = tmp_path / "first_overfit"
    recovery.mkdir()

    history = recovery / "history.jsonl"
    history.write_text(
        "\n".join(
            [
                json.dumps({"record": "step", "step": 1, "loss": 5.0}),
                json.dumps({"record": "step", "step": 2, "loss": 4.0}),
                json.dumps({"record": "step", "step": 3, "loss": 3.0}),
                '{"record": "step", "step": 4,',
            ]
        ),
        encoding="utf-8",
    )
    diagnostics = recovery / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "step_000002.json").write_text(
        json.dumps({"step": 2, "loss": 4.0, "f1": 0.25}),
        encoding="utf-8",
    )

    rows = module._load_recovery_history(
        recovery,
        max_step=2,
    )

    assert [row["step"] for row in rows] == [1, 2]
    assert rows[1]["f1"] == 0.25


def test_atomic_recovery_checkpoint_contains_optimizer_progress(tmp_path):
    module = _experiment_module()

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        global_step=7,
        curriculum_stage=SimpleNamespace(name="refinement_joint"),
        checkpoint_metadata=lambda: {
            "curriculum_stage": "refinement_joint",
            "refinement_stage_step": 7,
        },
    )

    recovery = tmp_path / "recovery"
    checkpoint = module._atomic_save_recovery_checkpoint(
        recovery,
        trainer,
        cfg=None,
        train_cfg=None,
        run_dir=tmp_path / "attempt",
    )

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert payload["global_step"] == 7
    assert "optimizer" in payload
    assert payload["extra"]["refinement_stage_step"] == 7
    assert checkpoint.name == "checkpoint_latest.pt"
    assert not list(recovery.glob(".checkpoint_latest.*.tmp"))
