"""
Run multiple Python scripts sequentially.

Usage:
    1. Put this file in the repository root.
    2. Edit SCRIPTS below.
    3. From the repository root, run:

        python scripts/run_scripts_sequentially/run_scripts_sequentially.py

Behavior:
    - Runs scripts one after another.
    - Uses the SAME Python interpreter / virtual environment as this launcher.
    - If one script fails, the next script still runs.
    - Prints a summary at the end.
    - Saves stdout/stderr for each script under ./sequential_run_logs/.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
import os

ROOT = Path(__file__).resolve().parents[2]

os.chdir(ROOT)

print("Current directory:", Path.cwd())
exit()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths are relative to the repository root.
#
# Simple script:
#     "investigations/stirnet/foo.py"
#
# Script with command-line arguments:
#     ["investigations/stirnet/foo.py", "--steps", "1000", "--batch-size", "8"]
#
#SCRIPTS = [
    # "investigations/stirnet/example_1.py",
    # "investigations/stirnet/example_2.py",
    # ["investigations/stirnet/example_3.py", "--steps", "1000"],
#]

SCRIPTS = [
    "investigations/stirnet/28_separator_reliability_full_volume_audit.py",
    "investigations/stirnet/27_edge_relational_group_reasoning_training.py",
]

# Continue with later scripts when one fails.
CONTINUE_ON_FAILURE = True

# Directory for stdout/stderr logs.
LOG_DIR = Path("sequential_run_logs")


# ============================================================================
# RUNNER
# ============================================================================

def normalize_job(job: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(job, str):
        return [job]

    if isinstance(job, (list, tuple)) and job:
        return [str(x) for x in job]

    raise ValueError(f"Invalid script entry: {job!r}")


def safe_log_name(script: str, index: int) -> str:
    script_path = Path(script)
    stem = script_path.stem

    parent = "_".join(script_path.parent.parts)
    if parent:
        name = f"{index:02d}_{parent}_{stem}.log"
    else:
        name = f"{index:02d}_{stem}.log"

    # Avoid characters that are problematic in Windows filenames.
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")

    return name


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def main() -> int:
    repo_root = Path.cwd()

    if not SCRIPTS:
        print("No scripts are configured.")
        print("Edit the SCRIPTS variable in this file and run it again.")
        return 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_log_dir = LOG_DIR / timestamp
    run_log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Sequential Python Runner")
    print("=" * 80)
    print(f"Repository root : {repo_root}")
    print(f"Python          : {sys.executable}")
    print(f"Jobs            : {len(SCRIPTS)}")
    print(f"Logs            : {run_log_dir.resolve()}")
    print("=" * 80)

    results: list[dict] = []

    overall_start = perf_counter()

    for index, raw_job in enumerate(SCRIPTS, start=1):
        job = normalize_job(raw_job)
        script = job[0]
        args = job[1:]

        script_path = Path(script)
        if not script_path.is_absolute():
            script_path = repo_root / script_path

        log_path = run_log_dir / safe_log_name(script, index)

        print()
        print("=" * 80)
        print(f"[{index}/{len(SCRIPTS)}] STARTING")
        print("=" * 80)
        print(f"Script : {script}")
        if args:
            print(f"Args   : {' '.join(args)}")
        print(f"Log    : {log_path}")
        print()

        start = perf_counter()
        return_code: int | None = None
        error_text: str | None = None

        if not script_path.exists():
            return_code = -1
            error_text = f"Script not found: {script_path}"
            elapsed = perf_counter() - start

            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(error_text + "\n")

            print(f"[FAILED] {error_text}")

        else:
            command = [sys.executable, str(script_path), *args]

            try:
                with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
                    log_file.write(f"Command: {command}\n")
                    log_file.write(f"Working directory: {repo_root}\n")
                    log_file.write("=" * 80 + "\n")
                    log_file.flush()

                    process = subprocess.run(
                        command,
                        cwd=repo_root,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )

                return_code = process.returncode
                elapsed = perf_counter() - start

                if return_code == 0:
                    print(
                        f"[SUCCESS] {script} finished in "
                        f"{format_duration(elapsed)}"
                    )
                else:
                    print(
                        f"[FAILED] {script} exited with code {return_code} "
                        f"after {format_duration(elapsed)}"
                    )
                    print(f"         See log: {log_path}")

            except Exception as exc:
                elapsed = perf_counter() - start
                return_code = -1
                error_text = f"{type(exc).__name__}: {exc}"

                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write("\nLauncher exception:\n")
                    log_file.write(error_text + "\n")

                print(f"[FAILED] Could not run {script}: {error_text}")

        results.append(
            {
                "script": script,
                "return_code": return_code,
                "elapsed": elapsed,
                "log": log_path,
            }
        )

        if return_code != 0 and not CONTINUE_ON_FAILURE:
            print("\nStopping because CONTINUE_ON_FAILURE=False.")
            break

    overall_elapsed = perf_counter() - overall_start

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    successes = 0
    failures = 0

    for result in results:
        ok = result["return_code"] == 0
        status = "OK" if ok else "FAILED"

        if ok:
            successes += 1
        else:
            failures += 1

        print(
            f"{status:7s} | "
            f"{format_duration(result['elapsed']):>12s} | "
            f"{result['script']}"
        )

    print("-" * 80)
    print(f"Successful : {successes}")
    print(f"Failed     : {failures}")
    print(f"Total time : {format_duration(overall_elapsed)}")
    print(f"Logs       : {run_log_dir.resolve()}")
    print("=" * 80)

    # The launcher itself exits successfully after completing the queue,
    # even if individual jobs failed. Their failure status is in the summary/logs.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
