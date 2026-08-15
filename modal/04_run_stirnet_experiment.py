from __future__ import annotations

import os

"""
Generic Modal launcher for scripts in experiments/stirnet/.

Normal use
----------
1. Change EXPERIMENT_SCRIPT.
2. Adjust EXPERIMENT_ARGS only if the selected experiment needs different CLI
   settings.
3. Optionally set LOCAL_WARM_START_CHECKPOINT to package one local checkpoint
   into the container.
4. Run:

       modal run modal/04_run_stirnet_experiment.py

The selected experiment is executed as a normal Python subprocess, so argparse,
stdout/stderr, exit codes, and tracebacks behave exactly like terminal runs.
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal


# ===========================================================================
# User-editable experiment selection
# ===========================================================================

EXPERIMENT_SCRIPT = "30_overfit_oracle.py"

# Keep the experiment's internal cap below Modal's 3600-second hard timeout so
# it can flush scalar statistics and the wrapper can commit the runs Volume.
EXPERIMENT_ARGS = [
    "--runtime-profile",
    "cloud_48gb",
    "--hard-time-limit-seconds",
    "3480",
]

# Optional: point this at a useful local checkpoint for the first cloud run.
# It is mounted as /workspace/bootstrap/warm_start.pt and passed automatically
# as --warm-start. Leave as None if the stirnet-runs Volume already contains a
# checkpoint that the experiment can discover.
#
# Example:
# LOCAL_WARM_START_CHECKPOINT = Path(
#     "runs/stirnet/overnight/27_overnight_.../checkpoint_best_spatial_query.pt"
# )
LOCAL_WARM_START_CHECKPOINT: Path | None = None

GPU = "L40S"


# ===========================================================================
# Modal / repository layout
# ===========================================================================

app = modal.App("stirnet-experiment-runner")

LOCAL_REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_EXPERIMENTS_DIR = LOCAL_REPO_ROOT / "experiments" / "stirnet"

REMOTE_REPO_ROOT = "/workspace/cell-tracking"
REMOTE_EXPERIMENTS_DIR = f"{REMOTE_REPO_ROOT}/experiments/stirnet"
DATA_MOUNT = f"{REMOTE_REPO_ROOT}/data"
RUNS_MOUNT = f"{REMOTE_REPO_ROOT}/runs"

DATA_DIR = (
    f"{DATA_MOUNT}/learned/stirnet/first_overfit/"
    "BlastoSPIM1_F22_030_034"
)
REMOTE_WARM_START = "/workspace/bootstrap/warm_start.pt"

data_volume = modal.Volume.from_name("stirnet-data")
runs_volume = modal.Volume.from_name(
    "stirnet-runs",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.13.0",
        "numpy==2.4.6",
        "scipy==1.17.1",
        "scikit-image==0.26.0",
        "networkx>=3.0",
    )
    .workdir(REMOTE_REPO_ROOT)
    .add_local_dir(
        LOCAL_REPO_ROOT / "learned",
        remote_path=f"{REMOTE_REPO_ROOT}/learned",
    )
    .add_local_dir(
        LOCAL_EXPERIMENTS_DIR,
        remote_path=REMOTE_EXPERIMENTS_DIR,
    )
)

if LOCAL_WARM_START_CHECKPOINT is not None:
    warm = (LOCAL_REPO_ROOT / LOCAL_WARM_START_CHECKPOINT).resolve()
    if not warm.exists():
        raise FileNotFoundError(
            f"LOCAL_WARM_START_CHECKPOINT does not exist: {warm}"
        )
    image = image.add_local_file(
        warm,
        remote_path=REMOTE_WARM_START,
    )


def _validate_experiment_name(name: str) -> str:
    path = Path(name)
    if path.name != name or path.suffix != ".py":
        raise ValueError(
            "EXPERIMENT_SCRIPT must be one .py filename inside "
            "experiments/stirnet/, e.g. '30_overfit_oracle.py'."
        )
    return name


EXPERIMENT_SCRIPT = _validate_experiment_name(EXPERIMENT_SCRIPT)


@app.function(
    image=image,
    gpu=GPU,
    cpu=4.0,
    memory=8192,
    # True platform hard cap requested by the experiment workflow.
    timeout=60 * 60,
    volumes={
        DATA_MOUNT: data_volume,
        RUNS_MOUNT: runs_volume,
    },
)
def run_selected_experiment() -> dict:
    script_path = Path(REMOTE_EXPERIMENTS_DIR) / EXPERIMENT_SCRIPT
    if not script_path.exists():
        raise FileNotFoundError(
            f"Experiment script is missing in the container: {script_path}"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    experiment_stem = script_path.stem
    run_dir = (
        Path(RUNS_MOUNT)
        / "stirnet"
        / "experiments"
        / experiment_stem
        / timestamp
    )

    command = [
        sys.executable,
        "-u",
        str(script_path),
        "--data-dir",
        DATA_DIR,
        "--run-dir",
        str(run_dir),
        "--runs-root",
        RUNS_MOUNT,
        *EXPERIMENT_ARGS,
    ]

    if LOCAL_WARM_START_CHECKPOINT is not None:
        command.extend(["--warm-start", REMOTE_WARM_START])

    print("=" * 88, flush=True)
    print("STIR-Net generic Modal experiment runner", flush=True)
    print("=" * 88, flush=True)
    print(f"GPU               : {GPU}", flush=True)
    print(f"Experiment        : {EXPERIMENT_SCRIPT}", flush=True)
    print(f"Data              : {DATA_DIR}", flush=True)
    print(f"Run directory     : {run_dir}", flush=True)
    print(f"Modal hard timeout: 3600 s", flush=True)
    print("Command:", flush=True)
    print(" ".join(command), flush=True)
    print("=" * 88, flush=True)

    # The experiment is executed by absolute file path, so Python would normally
    # put experiments/stirnet on sys.path rather than the repository root.
    # Explicitly expose the repository root so imports such as
    # `from learned.stirnet import StirNet` work for every experiment script.

    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH")

    env["PYTHONPATH"] = (
        REMOTE_REPO_ROOT
        if not existing_pythonpath
        else REMOTE_REPO_ROOT + os.pathsep + existing_pythonpath
    )

    print(
        f"PYTHONPATH        : {env['PYTHONPATH']}",
        flush=True,
    )

    process = subprocess.Popen(
        command,
        cwd=REMOTE_REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)

    return_code = process.wait()

    # Explicitly persist results before returning. The selected experiment uses
    # an internal deadline shorter than the Modal timeout specifically to leave
    # room for this commit.
    runs_volume.commit()

    report = {
        "experiment": EXPERIMENT_SCRIPT,
        "run_dir": str(run_dir),
        "return_code": int(return_code),
        "gpu": GPU,
    }

    print("=" * 88, flush=True)
    print(f"Experiment return code: {return_code}", flush=True)
    print(f"Persistent run dir    : {run_dir}", flush=True)
    print("=" * 88, flush=True)

    if return_code != 0:
        raise RuntimeError(
            f"{EXPERIMENT_SCRIPT} failed with exit code {return_code}. "
            f"Scalar failure diagnostics, if written, are in {run_dir}."
        )

    return report


@app.local_entrypoint()
def main() -> None:
    report = run_selected_experiment.remote()
    print(report)

